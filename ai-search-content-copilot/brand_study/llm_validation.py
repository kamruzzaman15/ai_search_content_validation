"""Human validation of the LLM judges.

`make_sample` draws a blinded, stratified random sample (strata: the judges' majority category x whether they
agreed) and freezes the votes, weights and the bias-test specification in a snapshot. `evaluate` reads the human
labels plus that snapshot only, and writes accuracy, kappa, confusion matrices and a pre-specified, design-weighted
big-brand bias test to Markdown. Pure-Python statistics (math, random); no new dependencies.
"""
from __future__ import annotations
import csv
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

from brand_study import cache, config
from brand_study.analyze import wilson
from brand_study.config import ConfigError
from brand_study.labeling import _is_comment, _read_lines, _read_rows
from brand_study.llm_aggregate import classify, item_votes, majority_category
from brand_study.llm_evidence import atomic_write_text, canonical, clip, read_items, sha12
from brand_study.llm_judges import PROMPT_VERSION, VoteStore

NAN = float("nan")
CATS = ("big_brand", "small_local", "other")
AGREEMENT = ("unanimous", "non_unanimous")
STRATA = tuple(f"{c}|{a}" for c in CATS for a in AGREEMENT)
SAMPLE_COLUMNS = ["sample_id", "key_type", "key", "example_url", "example_title", "sectors", "surfaces",
                  "query_examples", "snippets", "local_pack", "homepage_title", "homepage_description",
                  "human_category", "human_note"]
LABELS = tuple(c for c in config.CATEGORIES)
SHA_RE = re.compile(r"snapshot_sha=([0-9a-f]{12})\b")
SAMPLE_FALLBACK_ENCODING = "cp1252"  # Excel's "CSV (comma delimited)" on Windows; UTF-8 (with or without BOM) first
MCNEMAR_HEADING = ("unweighted; conditional on the stratified sample, which over-samples judge-big and judge-small "
                   "items at different rates, so this p-value can show an asymmetry that does not exist in the "
                   "population, or hide one that does")


def sample_path() -> Path:
    return config.results_dir() / "llm_validation_sample.csv"


def snapshot_path() -> Path:
    return config.llm_dir() / "validation_snapshot.json"


def report_path() -> Path:
    return config.results_dir() / "llm_validation.md"


def snippet_path() -> Path:
    return config.results_dir() / "llm_validation_overrides_snippet.csv"


# --- Statistics ------------------------------------------------------------------------

def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b)]
    n = len(pairs)
    if n == 0:
        return NAN
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return NAN if pe >= 1 else (po - pe) / (1 - pe)


def fleiss_kappa_counts(table: list[list[int]]) -> float:
    """table[i][j] = raters putting item i in category j (same number of raters per item)."""
    table = [row for row in table if sum(row) > 0]
    if not table:
        return NAN
    n = sum(table[0])
    if n < 2 or any(sum(r) != n for r in table):
        return NAN
    N = len(table)
    k = len(table[0])
    p_j = [sum(r[j] for r in table) / (N * n) for j in range(k)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in table]
    p_bar = sum(P_i) / N
    pe = sum(p * p for p in p_j)
    return NAN if pe >= 1 else (p_bar - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """ratings[i] = the raters' labels for item i."""
    cats = sorted({c for r in ratings for c in r})
    return fleiss_kappa_counts([[r.count(c) for c in cats] for r in ratings])


def confusion(truth: list, pred: list, row_labels=None, col_labels=None) -> pd.DataFrame:
    rl = list(row_labels or sorted(set(truth)))
    cl = list(col_labels or sorted(set(pred)))
    counts = Counter(zip(truth, pred))
    rows = [{"human \\ judge": r, **{c: counts.get((r, c), 0) for c in cl}} for r in rl]
    return pd.DataFrame(rows, columns=["human \\ judge"] + cl)


def binom_upper_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n <= 0:
        return NAN
    return sum(math.comb(n, i) for i in range(max(0, k), n + 1)) / 2 ** n


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p for X ~ Binomial(n, 1/2), doubling the smaller tail."""
    if n <= 0:
        return NAN
    lo = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * lo)


def mcnemar_exact(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"two_sided": NAN, "one_sided": NAN}
    return {"two_sided": binom_two_sided(min(b, c), n), "one_sided": binom_upper_tail(b, n)}


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return NAN
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def discordant(rows: list[dict], col: str) -> tuple[int, int]:
    b = sum(1 for r in rows if r[col] == "big_brand" and r["human"] == "small_local")
    c = sum(1 for r in rows if r[col] == "small_local" and r["human"] == "big_brand")
    return b, c


def weighted_net_shift(rows: list[dict], col: str) -> float:
    """(sum w*[judge big, human small] - sum w*[judge small, human big]) / sum w."""
    tot = sum(r["w"] for r in rows)
    if tot <= 0:
        return NAN
    b = sum(r["w"] for r in rows if r[col] == "big_brand" and r["human"] == "small_local")
    c = sum(r["w"] for r in rows if r[col] == "small_local" and r["human"] == "big_brand")
    return (b - c) / tot


def weighted_rate(rows: list[dict], col: str, human: str, judge: str) -> float:
    den = sum(r["w"] for r in rows if r["human"] == human)
    return sum(r["w"] for r in rows if r["human"] == human and r[col] == judge) / den if den else NAN


def weighted_big_share_diff(rows: list[dict], col: str) -> float:
    tot = sum(r["w"] for r in rows)
    if tot <= 0:
        return NAN
    return (sum(r["w"] for r in rows if r[col] == "big_brand") -
            sum(r["w"] for r in rows if r["human"] == "big_brand")) / tot


def bootstrap_replicates(rows: list[dict], reps: int, seed: int) -> list[list[dict]]:
    """Resample with replacement within each stratum, keeping each stratum's n (weights N_h/n_h unchanged)."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["stratum"]].append(r)
    keys = sorted(groups)
    out = []
    for _ in range(reps):
        rep = []
        for k in keys:
            g = groups[k]
            rep += [g[rng.randrange(len(g))] for _ in range(len(g))]
        out.append(rep)
    return out


def stratified_bootstrap(fn: Callable, rows: list[dict], reps: int, seed: int) -> list[float]:
    return [fn(rep) for rep in bootstrap_replicates(rows, reps, seed)]


def bias_test(rows: list[dict], col: str, spec: dict, reps: list | None = None) -> dict:
    """Pre-specified one-sided test on the design-weighted net shift. `spec` comes from the snapshot.
    `reps` (optional) are precomputed bootstrap replicates of exactly these rows, shared between judges."""
    alpha, min_d = float(spec["alpha"]), int(spec["min_discordant"])
    b, c = discordant(rows, col)
    delta = weighted_net_shift(rows, col)
    if reps is None:
        reps = bootstrap_replicates(rows, int(spec["bootstrap_reps"]), int(spec["bootstrap_seed"])) if rows else []
    dist = [weighted_net_shift(r, col) for r in reps]
    lb = quantile(dist, alpha)
    lo2, hi2 = quantile(dist, alpha / 2), quantile(dist, 1 - alpha / 2)
    level1 = f"{100 * (1 - alpha):.0f}%"
    if b + c < min_d:
        verdict = f"inconclusive: only b+c={b + c} discordant big↔small items (minimum {min_d})"
        kind = "inconclusive"
    elif lb == lb and lb > 0:
        verdict = f"evidence of big-brand bias (pre-specified one-sided test, α = {alpha:g})"
        kind = "bias"
    elif hi2 == hi2 and hi2 < 0:
        verdict = "exploratory: evidence of the reverse (small-brand) shift"
        kind = "reverse"
    else:
        verdict = f"no evidence of big-brand bias (Δ = {delta:.3f}, one-sided {level1} lower bound {lb:.3f})"
        kind = "none"
        if hi2 == hi2 and lo2 == lo2 and (hi2 - lo2) / 2 > 0.05:
            verdict += "; the CI is wide, so a moderate bias cannot be ruled out (consider `llm-sample --extend`)"

    def rate_ci(h, j):
        d = [weighted_rate(r, col, h, j) for r in reps]
        return quantile(d, alpha / 2), quantile(d, 1 - alpha / 2)

    share = [weighted_big_share_diff(r, col) for r in reps]
    n_hs = sum(1 for r in rows if r["human"] == "small_local")
    n_hb = sum(1 for r in rows if r["human"] == "big_brand")
    k_bs = sum(1 for r in rows if r["human"] == "small_local" and r[col] == "big_brand")
    k_sb = sum(1 for r in rows if r["human"] == "big_brand" and r[col] == "small_local")
    return {
        "b": b, "c": c, "delta": delta, "lower_bound": lb, "ci": (lo2, hi2), "verdict": verdict, "kind": kind,
        "mcnemar": mcnemar_exact(b, c),
        "rate_big_given_small": (k_bs, n_hs, wilson(k_bs, n_hs)),
        "rate_small_given_big": (k_sb, n_hb, wilson(k_sb, n_hb)),
        "w_rate_big_given_small": (weighted_rate(rows, col, "small_local", "big_brand"),
                                   rate_ci("small_local", "big_brand")),
        "w_rate_small_given_big": (weighted_rate(rows, col, "big_brand", "small_local"),
                                   rate_ci("big_brand", "small_local")),
        "big_share_diff": (weighted_big_share_diff(rows, col), (quantile(share, alpha / 2),
                                                                 quantile(share, 1 - alpha / 2))),
    }


# --- Strata and allocation --------------------------------------------------------------

def stratum_of(votes: dict) -> str:
    m = majority_category(votes)
    cat = m if m in ("big_brand", "small_local") else "other"
    ok = [v["category"] for v in votes.values() if v and v.get("status") == "ok"]
    agr = "unanimous" if len(ok) == len(votes) and len(set(ok)) == 1 and ok[0] != "unknown" else "non_unanimous"
    return f"{cat}|{agr}"


def _largest_remainder(n: int, shares: dict) -> dict:
    raw = {k: n * v / sum(shares.values()) for k, v in shares.items()}
    out = {k: int(math.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), list(shares).index(k)))[: n - sum(out.values())]:
        out[k] += 1
    return out


def allocate(n: int, available: dict, shares: dict | None = None) -> dict:
    """Per-stratum sample sizes. Each category's share is split evenly between unanimous and non-unanimous.
    A shortfall goes first to the other agreement cell of the same category, then to the other categories in
    proportion to their shares (one item at a time, D'Hondt)."""
    shares = shares or config.SAMPLE_ALLOCATION
    per_cat = _largest_remainder(n, shares)
    target = {}
    for c in CATS:
        t = per_cat.get(c, 0)
        target[f"{c}|unanimous"] = t // 2
        target[f"{c}|non_unanimous"] = t - t // 2
    alloc = {s: min(target[s], available.get(s, 0)) for s in STRATA}
    for s in STRATA:
        deficit = target[s] - alloc[s]
        if deficit > 0:
            c, a = s.split("|")
            partner = f"{c}|{'non_unanimous' if a == 'unanimous' else 'unanimous'}"
            give = min(deficit, available.get(partner, 0) - alloc[partner])
            if give > 0:
                alloc[partner] += give
    short = n - sum(alloc.values())
    extra = Counter()
    while short > 0:
        spare = [c for c in CATS if any(available.get(f"{c}|{a}", 0) > alloc[f"{c}|{a}"] for a in AGREEMENT)]
        if not spare:
            break
        c = max(spare, key=lambda c: (shares.get(c, 0) / (extra[c] + 1), -CATS.index(c)))
        cells = [f"{c}|{a}" for a in AGREEMENT if available.get(f"{c}|{a}", 0) > alloc[f"{c}|{a}"]]
        cell = min(cells, key=lambda s: (alloc[s], STRATA.index(s)))
        alloc[cell] += 1
        extra[c] += 1
        short -= 1
    return alloc


# --- Sample and snapshot -------------------------------------------------------------------

def _vote_view(v):
    if not v:
        return None
    return {"category": v.get("category") if v.get("status") == "ok" else None,
            "confidence": v.get("confidence"), "status": v.get("status")}


def _has_local_pack(rec: dict) -> bool:
    return bool((rec.get("evidence") or {}).get("local_pack"))


def full_vote_table(records: list[dict], store: VoteStore, judges=config.LLM_JUDGE_ORDER,
                    policy: str = "unanimous") -> tuple[dict, int]:
    """Votes per complete item. `accepted` is what aggregation would accept under `policy` (None when the item
    goes to the human, e.g. a unanimous vote that conflicts with a candidate rule)."""
    table, pending = {}, 0
    for rec in records:
        votes, _ = item_votes(rec, store, judges)
        if any(v is None for v in votes.values()):
            pending += 1
            continue
        outcome, accepted, source = classify(votes, rec.get("rule_categories") or [], policy)
        table[rec["item_id"]] = {"key_type": rec["key_type"], "key": rec["key"],
                                 "evidence_sha": rec["evidence_sha"], "stratum": stratum_of(votes),
                                 "has_local_pack": _has_local_pack(rec), "outcome": outcome,
                                 "accepted": accepted, "accepted_source": source,
                                 "votes": {j: _vote_view(v) for j, v in votes.items()}}
    return table, pending


def _sample_item(sid: str, iid: str, stratum: str, t: dict) -> dict:
    return {"sample_id": sid, "item_id": iid, "key_type": t["key_type"], "key": t["key"],
            "evidence_sha": t["evidence_sha"], "stratum": stratum, "votes": t["votes"],
            "accepted": t.get("accepted"), "accepted_source": t.get("accepted_source", "")}


def _snapshot_sha(snap: dict) -> str:
    return sha12({k: v for k, v in snap.items() if k != "snapshot_sha"})


def _bias_spec() -> dict:
    return {"hypothesis": config.BIAS_HYPOTHESIS, "alpha": config.BIAS_ALPHA,
            "min_discordant": config.BIAS_MIN_DISCORDANT, "statistic": "weighted_net_shift",
            "bootstrap_reps": config.BOOTSTRAP_REPS, "bootstrap_seed": config.BOOTSTRAP_SEED}


def _judge_meta(store: VoteStore, judges) -> dict:
    out = {}
    for j in judges:
        recs = [r for r in store.records(j) if r.get("prompt_version") == PROMPT_VERSION]
        out[j] = {"model": config.LLM_JUDGES[j][1],
                  "digests": sorted({r.get("model_digest") for r in recs} - {"", None}),
                  "cli_versions": sorted({r.get("cli_version") for r in recs} - {"", None})}
    return out


def _sample_row(sid: str, rec: dict) -> dict:
    ev = rec.get("evidence") or {}
    qs = " | ".join(clip(q.get("text"), 80) for q in ev.get("queries", []))
    if rec["key_type"] == "domain":
        snips = " | ".join(clip(f"{s.get('title')}: {s.get('snippet')}", 300) for s in ev.get("snippets", []))
    else:
        snips = " | ".join(ev.get("aio_segments", []))
    lp = " | ".join(clip(f"{p.get('title')} ({p.get('type')}; {p.get('address')}; rating {p.get('rating')}, "
                         f"{p.get('reviews')} reviews; {p.get('city')})", 200) for p in ev.get("local_pack", []) if p)
    hp = ev.get("homepage") or {}
    return {"sample_id": sid, "key_type": rec["key_type"], "key": rec["key"],
            "example_url": clip(rec.get("example_url") or (ev.get("example_urls") or [""])[0], 300),
            "example_title": clip(rec.get("example_title"), 200), "sectors": clip(rec.get("sectors"), 60),
            "surfaces": clip(rec.get("surfaces"), 80), "query_examples": qs, "snippets": clip(snips, 900),
            "local_pack": lp, "homepage_title": clip(hp.get("title"), 200),
            "homepage_description": clip(hp.get("meta_description"), 300), "human_category": "", "human_note": ""}


def _sample_text(rows: list[dict], snapshot_sha: str) -> str:
    buf = io.StringIO()
    buf.write(f"# Blinded LLM-validation sample. snapshot_sha={snapshot_sha}\n")
    buf.write("# Fill human_category with one of: " + ", ".join(LABELS) + " (unknown = cannot tell; excluded "
              "from the metrics). Optional: human_note.\n")
    buf.write("# Judge from the evidence here and your own checks (open the website). Label this sample BEFORE "
              "opening llm_disagreements.csv, and never open validation_snapshot.json.\n")
    for c in LABELS:
        buf.write(f"# {c}: {config.CATEGORY_DEFINITIONS[c]}\n")
    w = csv.DictWriter(buf, fieldnames=SAMPLE_COLUMNS, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: " ".join(str(r.get(k) if r.get(k) is not None else "").split()) for k in SAMPLE_COLUMNS})
    return buf.getvalue()


def read_sample(path: Path) -> tuple[list[tuple[int, dict]], str]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"sample file not found: {path}")
    # Tolerates a spreadsheet round trip: BOM, CRLF, commas padded onto comment lines, quoted comment cells, and
    # Excel's legacy "CSV (comma delimited)" Windows-1252 encoding.
    sha = ""
    for ln in _read_lines(path, SAMPLE_FALLBACK_ENCODING):
        m = SHA_RE.search(ln) if _is_comment(ln) else None
        if m:
            sha = m.group(1)
            break
    rows = [(n, r) for n, r in _read_rows(path, SAMPLE_FALLBACK_ENCODING)
            if any((v or "").strip() for v in r.values() if isinstance(v, str))]  # drop blank spreadsheet rows
    return rows, sha


def load_snapshot(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"snapshot not found: {path}; run llm-sample first")
    snap = json.loads(path.read_text(encoding="utf-8"))
    if snap.get("snapshot_sha") != _snapshot_sha(snap):
        raise ConfigError(f"{path.name} was modified after it was written (snapshot_sha mismatch)")
    return snap


def _set_weights(snap: dict) -> None:
    n_by = Counter(it["stratum"] for it in snap["items"])
    for s, d in snap["strata"].items():
        d["n"] = n_by.get(s, 0)
    for it in snap["items"]:
        d = snap["strata"][it["stratum"]]
        it["weight"] = d["N"] / d["n"] if d["n"] else NAN


def _write_snapshot(snap: dict, path: Path) -> None:
    snap["snapshot_sha"] = _snapshot_sha(snap)
    atomic_write_text(path, json.dumps(snap, ensure_ascii=False, indent=1, sort_keys=True))


def _draw(rng: random.Random, pools: dict, alloc: dict) -> list[tuple[str, str]]:
    chosen = []
    for s in STRATA:
        ids = sorted(pools.get(s, []))
        chosen += [(s, i) for i in rng.sample(ids, alloc.get(s, 0))]
    rng.shuffle(chosen)
    return chosen


def make_sample(records: list[dict], store: VoteStore, *, n: int = config.DEFAULT_SAMPLE_N,
                seed: int = config.DEFAULT_SAMPLE_SEED, policy: str = "unanimous", runs=(), force: bool = False,
                allow_pending: bool = False, sample_out: Path | None = None, snapshot_out: Path | None = None,
                judges=config.LLM_JUDGE_ORDER, log: Callable = print) -> dict:
    sample_out = Path(sample_out or sample_path())
    snapshot_out = Path(snapshot_out or snapshot_path())
    if not force:
        if snapshot_out.exists():
            raise ConfigError(f"{snapshot_out} exists; use --extend to add items or --force to start over")
        if sample_out.exists():
            raise ConfigError(f"{sample_out} exists; use --force to overwrite it")
    table, pending = full_vote_table(records, store, judges, policy)
    if pending and not allow_pending:
        raise ConfigError(f"{pending} items do not have votes from all judges yet; finish llm-label first "
                          f"(or pass --allow-pending to sample from the complete items only)")
    pools = defaultdict(list)
    for iid, t in table.items():
        pools[t["stratum"]].append(iid)
    available = {s: len(pools[s]) for s in STRATA}
    alloc = allocate(min(n, len(table)), available)
    rng = random.Random(seed)
    chosen = _draw(rng, pools, alloc)
    by_id = {r["item_id"]: r for r in records}
    items, rows = [], []
    for k, (s, iid) in enumerate(chosen, start=1):
        sid = f"S{k:03d}"
        t = table[iid]
        items.append(_sample_item(sid, iid, s, t))
        rows.append(_sample_row(sid, by_id[iid]))
    state = rng.getstate()
    snap = {"created_at": cache.utc_now(), "seed": seed, "n": len(items), "prompt_version": PROMPT_VERSION,
            "policy": policy, "judges": _judge_meta(store, judges), "runs": list(runs),
            "strata": {s: {"N": available[s], "n": 0} for s in STRATA}, "items": items, "full_votes": table,
            "bias_test": _bias_spec(), "extensions": [], "rng_state": [state[0], list(state[1]), state[2]],
            "pending_excluded": pending}
    _set_weights(snap)
    _write_snapshot(snap, snapshot_out)
    atomic_write_text(sample_out, _sample_text(rows, snap["snapshot_sha"]))
    log(f"wrote {sample_out} ({len(items)} items; " + ", ".join(f"{s}={alloc[s]}/{available[s]}" for s in STRATA)
        + ")")
    log(f"wrote {snapshot_out} (do not open it before labeling)")
    log("Label the sample BEFORE opening llm_disagreements.csv, then run: python -m brand_study llm-agreement")
    return snap


def extend_sample(k: int, records: list[dict], *, sample_in: Path | None = None, snapshot_in: Path | None = None,
                  fallback_records: list[dict] | None = None, log: Callable = print) -> dict:
    """Add k items drawn from the snapshot's strata. Every not-yet-sampled item of the frozen population stays
    eligible, including items that have left the current item set since sampling (e.g. a human override was added
    for them): their evidence comes from `fallback_records` (default: items.jsonl). Otherwise the draw would be from
    a non-random remainder of each stratum while the weights still use N_h, which biases the weighted estimates."""
    sample_in = Path(sample_in or sample_path())
    snapshot_in = Path(snapshot_in or snapshot_path())
    snap = load_snapshot(snapshot_in)
    rows_in, sha = read_sample(sample_in)
    if sha != snap["snapshot_sha"]:
        raise ConfigError(f"{sample_in.name} does not belong to {snapshot_in.name} (snapshot_sha {sha or 'missing'} "
                          f"vs {snap['snapshot_sha']})")
    taken = {it["item_id"] for it in snap["items"]}
    known = {it["sample_id"] for it in snap["items"]}  # before the draw: new ids must not hide stray rows
    by_id = {r["item_id"]: r for r in records}
    eligible = [iid for iid in snap["full_votes"] if iid not in taken]
    gone = [iid for iid in eligible if iid not in by_id]
    if gone:
        fb = {r["item_id"]: r for r in (read_items() if fallback_records is None else fallback_records)}
        missing = [iid for iid in gone if iid not in fb]
        if missing:
            raise ConfigError(f"{len(missing)} items of the sampled population are no longer in the item set and "
                              f"have no record in items.jsonl (e.g. {', '.join(sorted(missing)[:3])}); extending "
                              f"without them would bias the weighted estimates. Restore items.jsonl, or draw a new "
                              f"sample with --force (this discards the current one).")
        for iid in gone:
            by_id[iid] = fb[iid]
        log(f"note: {len(gone)} eligible items have left the item set since sampling (e.g. human overrides); "
            f"they stay eligible, with their evidence from items.jsonl")
    pools = defaultdict(list)
    for iid in eligible:
        pools[snap["full_votes"][iid]["stratum"]].append(iid)
    available = {s: len(pools[s]) for s in STRATA}
    alloc = allocate(min(k, sum(available.values())), available, config.EXTEND_ALLOCATION)
    st = snap["rng_state"]
    rng = random.Random()
    rng.setstate((st[0], tuple(st[1]), st[2]))
    chosen = _draw(rng, pools, alloc)
    start = len(snap["items"])
    new_rows = []
    for j, (s, iid) in enumerate(chosen, start=start + 1):
        sid = f"S{j:03d}"
        t = snap["full_votes"][iid]
        snap["items"].append(_sample_item(sid, iid, s, t))
        new_rows.append(_sample_row(sid, by_id[iid]))
    state = rng.getstate()
    snap["rng_state"] = [state[0], list(state[1]), state[2]]
    snap["n"] = len(snap["items"])
    snap["extensions"].append({"k": k, "added": len(chosen), "at": cache.utc_now()})
    _set_weights(snap)
    _write_snapshot(snap, snapshot_in)
    stray = [f"line {n}: sample_id {(row.get('sample_id') or '').strip()!r}" for n, row in rows_in
             if (row.get("sample_id") or "").strip() not in known]
    if stray:
        log(f"WARNING: {len(stray)} rows of {sample_in.name} have a sample_id that is not in the snapshot; they are "
            f"kept as they are but llm-agreement will ignore them: " + "; ".join(stray[:5])
            + (" ..." if len(stray) > 5 else ""))
    old = [{c: row.get(c, "") for c in SAMPLE_COLUMNS} for _, row in rows_in]
    atomic_write_text(sample_in, _sample_text(old + new_rows, snap["snapshot_sha"]))
    log(f"added {len(chosen)} items to {sample_in} (now {snap['n']}); existing rows unchanged")
    return snap


# --- Evaluation ----------------------------------------------------------------------------

def eval_weights(snap: dict, labels: dict) -> dict:
    """Weight per stratum for evaluation: N_h / m_h, where m_h counts the stratum's sampled items that the human
    answered (a category or `unknown`). Blank rows are missing data, so they are treated as missing at random within
    their stratum instead of silently shrinking that stratum's weight (N_h / n_h would understate any stratum with
    blanks, e.g. right after `llm-sample --extend`). Human `unknown` rows stay in m_h, so the weighted estimates cover
    the population the human can decide. A stratum with m_h = 0 gets no weight (reported as dropped)."""
    m = Counter(it["stratum"] for it in snap["items"] if labels.get(it["sample_id"]))
    return {s: (d["N"] / m[s] if m.get(s) else NAN) for s, d in snap["strata"].items()}


def _rows_for_eval(snap: dict, labels: dict) -> list[dict]:
    judges = list(snap["judges"])
    weights = eval_weights(snap, labels)
    rows = []
    for it in snap["items"]:
        h = labels.get(it["sample_id"])
        if not h or h == "unknown":
            continue
        votes = {j: (v if v and v.get("status") == "ok" else None) for j, v in it["votes"].items()}
        row = {"sid": it["sample_id"], "item_id": it["item_id"], "key_type": it["key_type"], "key": it["key"],
               "stratum": it["stratum"], "w": float(weights[it["stratum"]]), "human": h}
        cats = []
        for j in judges:
            v = votes.get(j)
            row[j] = v["category"] if v else "abstain"
            if v:
                cats.append(v["category"])
        top = Counter(cats).most_common(1)
        row["majority"] = top[0][0] if top and top[0][1] >= 2 else "none"
        row["n_agree"] = top[0][1] if top else 0
        row["unanimous"] = row["majority"] if it["stratum"].endswith("|unanimous") else "none"
        # What aggregation actually accepted under the snapshot's policy (frozen at sampling time).
        row["accepted"] = it.get("accepted") or "none"
        rows.append(row)
    return rows


def _f(x, d=2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def _ci(t) -> str:
    return f"[{_f(t[0])}, {_f(t[1])}]"


def evaluate(sample_in: Path | None = None, snapshot_in: Path | None = None, out: Path | None = None,
             snippet_out: Path | None = None, log: Callable = print) -> dict:
    sample_in = Path(sample_in or sample_path())
    snapshot_in = Path(snapshot_in or snapshot_path())
    out = Path(out or report_path())
    snippet_out = Path(snippet_out or snippet_path())
    snap = load_snapshot(snapshot_in)
    rows_in, sha = read_sample(sample_in)
    if sha != snap["snapshot_sha"]:
        raise ConfigError(f"{sample_in.name} does not belong to {snapshot_in.name} (snapshot_sha {sha or 'missing'} "
                          f"vs {snap['snapshot_sha']})")
    bad, labels, notes, stray = [], {}, {}, []
    known = {it["sample_id"] for it in snap["items"]}
    for n, row in rows_in:
        sid = (row.get("sample_id") or "").strip()
        # lower(): spreadsheet autocorrect may capitalise the first letter ("Small_local")
        h = (row.get("human_category") or "").strip().lower()
        if sid not in known:
            stray.append(f"line {n}: sample_id {sid!r}")
            continue
        if h and h not in config.CATEGORIES:
            bad.append(f"line {n}: invalid human_category {h!r}")
        labels[sid] = h
        notes[sid] = (row.get("human_note") or "").strip()
    if bad:
        raise ConfigError(f"{sample_in.name}:\n  " + "\n  ".join(bad))
    if stray:
        log(f"WARNING: {len(stray)} rows of {sample_in.name} have a sample_id that is not in the snapshot and are "
            f"ignored (added or renamed rows?): " + "; ".join(stray[:5]) + (" ..." if len(stray) > 5 else ""))
    n_unknown = sum(1 for it in snap["items"] if labels.get(it["sample_id"]) == "unknown")
    n_empty = sum(1 for it in snap["items"] if not labels.get(it["sample_id"]))
    if n_empty:
        blank_by = Counter(it["stratum"] for it in snap["items"] if not labels.get(it["sample_id"]))
        log(f"WARNING: {n_empty} sampled rows have no human_category yet ("
            + ", ".join(f"{s}: {k}" for s, k in sorted(blank_by.items())) + "). They are treated as missing at random "
            f"within their stratum (weights N_h / answered_h). Label every row before relying on the results.")
    rows = _rows_for_eval(snap, labels)
    judges = list(snap["judges"])
    spec = snap["bias_test"]
    text, summary = _report(snap, rows, judges, spec, n_unknown, n_empty)
    atomic_write_text(out, text)
    n_contra = _write_snippet(snap, rows, notes, snippet_out)
    log(f"wrote {out} ({len(rows)} labelled items; {n_unknown} human unknown and {n_empty} empty excluded)")
    log(f"wrote {snippet_out} ({len(rows)} rows, {n_contra} contradict an accepted LLM label); append the rows you "
        f"agree with to brand_study/data/label_overrides.csv")
    log(summary)
    return {"n": len(rows), "summary": summary, "contradictions": n_contra}


def _write_snippet(snap, rows, notes, path) -> int:
    def contra(r):
        return r["accepted"] != "none" and r["accepted"] != r["human"]
    ordered = sorted(rows, key=lambda r: (not contra(r), r["sid"]))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["key_type", "key", "category", "note"])
    for r in ordered:
        note = f"human validation {r['sid']}" + (f"; contradicts accepted LLM label {r['accepted']}" if contra(r) else "")
        if notes.get(r["sid"]):
            note += f"; {notes[r['sid']]}"
        w.writerow([r["key_type"], r["key"], r["human"], note])
    atomic_write_text(path, buf.getvalue())
    return sum(1 for r in rows if contra(r))


def _bias_section(rows, col, spec, label, full=None, reps=None) -> tuple[list[str], dict]:
    t = bias_test(rows, col, spec, reps)
    alpha = float(spec["alpha"])
    lines = [f"#### {label}", "",
             f"- **Verdict: {t['verdict']}**",
             f"- Weighted net shift Δ = {_f(t['delta'], 3)}; one-sided {100 * (1 - alpha):.0f}% lower bound "
             f"{_f(t['lower_bound'], 3)}; two-sided {100 * (1 - alpha):.0f}% CI {_ci(t['ci'])} "
             f"(stratified bootstrap, {spec['bootstrap_reps']} reps)",
             f"- Descriptive ({MCNEMAR_HEADING}): b = {t['b']} (judge big, human small), c = {t['c']} (judge small, "
             f"human big); exact McNemar p two-sided {_f(t['mcnemar']['two_sided'], 4)}, one-sided (b > c) "
             f"{_f(t['mcnemar']['one_sided'], 4)}",
             f"- P(judge big | human small): {t['rate_big_given_small'][0]}/{t['rate_big_given_small'][1]} "
             f"(Wilson {_ci(t['rate_big_given_small'][2])}); population-weighted {_f(t['w_rate_big_given_small'][0])} "
             f"{_ci(t['w_rate_big_given_small'][1])}",
             f"- P(judge small | human big): {t['rate_small_given_big'][0]}/{t['rate_small_given_big'][1]} "
             f"(Wilson {_ci(t['rate_small_given_big'][2])}); population-weighted {_f(t['w_rate_small_given_big'][0])} "
             f"{_ci(t['w_rate_small_given_big'][1])}",
             f"- Weighted big-share difference (judge − human, includes moves from other categories into big): "
             f"{_f(t['big_share_diff'][0], 3)} {_ci(t['big_share_diff'][1])}"]
    if full is not None:
        lines.append(f"- Full-set proxy (no human labels): big_brand share {_f(full[0])} overall, {_f(full[1])} "
                     f"among items with local-pack evidence")
    return lines + [""], t


def _full_proxy(snap, judge) -> tuple[float, float]:
    vals = [(t["votes"].get(judge) or {}).get("category") for t in snap["full_votes"].values()]
    lp = [(t["votes"].get(judge) or {}).get("category") for t in snap["full_votes"].values() if t["has_local_pack"]]
    def share(xs):
        xs = [x for x in xs if x]
        return sum(1 for x in xs if x == "big_brand") / len(xs) if xs else NAN
    return share(vals), share(lp)


def _report(snap, rows, judges, spec, n_unknown, n_empty) -> tuple[str, str]:
    out = ["# LLM judge validation against human labels", ""]
    n = len(rows)
    dropped = [s for s, d in snap["strata"].items() if d["n"] > 0 and not any(r["stratum"] == s for r in rows)]
    out.append(f"- Labelled sample items used: {n} (human `unknown` excluded: {n_unknown}; empty: {n_empty})")
    if n_empty:
        out.append(f"- **{n_empty} sampled rows are still blank.** They are treated as missing at random within "
                   f"their stratum: weights are N_h / (answered items in stratum h) rather than N_h / n_h. Results "
                   f"are preliminary until every row is labelled.")
    if dropped:
        out.append(f"- Strata with no usable human labels (their population is left out, so weighted estimates "
                   f"cover the remaining population only): {', '.join(dropped)}")
    out.append("")
    # 1. per judge
    out += ["## 1. Per-judge accuracy", ""]
    per = []
    for j in judges:
        ans = [r for r in rows if r[j] != "abstain"]
        k = sum(1 for r in rows if r[j] == r["human"])
        lo, hi = wilson(k, n)
        tw = sum(r["w"] for r in rows)
        per.append({"judge": j, "model": snap["judges"][j]["model"], "n": n,
                    "accuracy (abstain = wrong)": k / n if n else NAN, "95% Wilson CI": f"{_f(lo)}–{_f(hi)}",
                    "answered": len(ans), "accuracy (answered)": k / len(ans) if ans else NAN,
                    "Cohen κ vs human": cohen_kappa([r["human"] for r in ans], [r[j] for r in ans]),
                    "population-weighted accuracy": sum(r["w"] for r in rows if r[j] == r["human"]) / tw
                    if tw else NAN})
    from brand_study.report import md_table
    out += [md_table(pd.DataFrame(per)), ""]
    # 2. ensemble
    out += ["## 2. Ensemble", ""]
    u = [r for r in rows if r["unanimous"] != "none"]
    acc_rows = [r for r in rows if r["accepted"] != "none"]
    maj = [r for r in rows if r["majority"] != "none"]
    two = [r for r in rows if r["n_agree"] == 2]
    acc_u = sum(1 for r in u if r["unanimous"] == r["human"]) / len(u) if u else NAN
    k_a = sum(1 for r in acc_rows if r["accepted"] == r["human"])
    acc_a = k_a / len(acc_rows) if acc_rows else NAN
    policy = snap.get("policy", "unanimous")
    ens = [{"set": f"accepted LLM labels (policy {policy}; excludes items sent to the human)",
            "n": len(acc_rows), "accuracy": acc_a, "95% Wilson CI": "–".join(_f(x) for x in wilson(k_a,
                                                                                               len(acc_rows)))},
           {"set": "unanimous-vote items (3 identical votes, accepted or not)", "n": len(u), "accuracy": acc_u,
            "95% Wilson CI": "–".join(_f(x) for x in wilson(sum(1 for r in u if r["unanimous"] == r["human"]),
                                                             len(u)))},
           {"set": "majority vote (items with a majority)", "n": len(maj),
            "accuracy": sum(1 for r in maj if r["majority"] == r["human"]) / len(maj) if maj else NAN,
            "95% Wilson CI": "–".join(_f(x) for x in wilson(sum(1 for r in maj if r["majority"] == r["human"]),
                                                             len(maj)))},
           {"set": "2-of-3 items (majority label)", "n": len(two),
            "accuracy": sum(1 for r in two if r["majority"] == r["human"]) / len(two) if two else NAN,
            "95% Wilson CI": "–".join(_f(x) for x in wilson(sum(1 for r in two if r["majority"] == r["human"]),
                                                             len(two)))}]
    out += [md_table(pd.DataFrame(ens)), ""]
    # 3. agreement between judges (full set)
    out += ["## 3. Agreement between judges (full item set, no human labels)", ""]
    fv = snap["full_votes"]
    pair_rows = []
    for a_i, a in enumerate(judges):
        for b in judges[a_i + 1:]:
            both = [(t["votes"][a]["category"], t["votes"][b]["category"]) for t in fv.values()
                    if (t["votes"].get(a) or {}).get("category") and (t["votes"].get(b) or {}).get("category")]
            pair_rows.append({"pair": f"{a} vs {b}", "n": len(both),
                              "agreement": sum(1 for x, y in both if x == y) / len(both) if both else NAN,
                              "Cohen κ": cohen_kappa([x for x, _ in both], [y for _, y in both])})
    out += [md_table(pd.DataFrame(pair_rows)), ""]
    allr = [[t["votes"][j]["category"] for j in judges] for t in fv.values()
            if all((t["votes"].get(j) or {}).get("category") for j in judges)]
    out += [f"- Fleiss' κ over {len(allr)} items with valid votes from all judges: {_f(fleiss_kappa(allr), 3)}", ""]
    # 4. confusion
    out += ["## 4. Confusion matrices (rows: human, columns: judge)", ""]
    hum = [r["human"] for r in rows]
    for j in judges:
        out += [f"### {j}", "", md_table(confusion(hum, [r[j] for r in rows], LABELS, LABELS + ("abstain",))), ""]
    out += [f"### accepted LLM labels (policy {policy})", "",
            md_table(confusion([r["human"] for r in acc_rows], [r["accepted"] for r in acc_rows], LABELS, LABELS)),
            ""]
    # 5. bias
    out += ["## 5. Big-brand bias check", "",
            f"Pre-specified when the sample was drawn (read from the snapshot): hypothesis `{spec['hypothesis']}` "
            f"(judges move human-small items to big more than the reverse), one-sided α = {spec['alpha']}, statistic "
            f"`{spec['statistic']}`, at least {spec['min_discordant']} unweighted discordant big↔small items needed "
            f"for a verdict. The reverse direction is exploratory and flagged only when the two-sided CI lies "
            f"entirely below 0.", ""]
    verdicts = {}
    reps = bootstrap_replicates(rows, int(spec["bootstrap_reps"]), int(spec["bootstrap_seed"])) if rows else []
    for j in judges:
        lines, t = _bias_section(rows, j, spec, f"{j} ({snap['judges'][j]['model']})", _full_proxy(snap, j), reps)
        out += lines
        verdicts[j] = t["verdict"]
    lines, t = _bias_section(rows, "majority", spec, "majority vote", reps=reps)
    out += lines
    verdicts["majority"] = t["verdict"]
    lines, t_a = _bias_section(acc_rows, "accepted", spec, f"accepted LLM labels (policy {policy}; domain "
                                                           f"estimate over the accepted items only)")
    out += lines
    verdicts["accepted"] = t_a["verdict"]
    out += ["Design note: weights are inverse sampling fractions (N_h / m_h, where m_h is the number of sampled "
            "items in the stratum that the human answered, including `unknown`) within the majority-category × "
            "agreement strata. The percentile bootstrap can undercover with small strata and discrete outcomes, so "
            "the verdict is a screening result, not a precise test.", ""]
    # 6. metadata
    out += ["## 6. Metadata", ""]
    for j in judges:
        m = snap["judges"][j]
        out.append(f"- {j}: model `{m['model']}`, digests {', '.join(m['digests']) or 'n/a'}, CLI "
                   f"{', '.join(m['cli_versions']) or 'n/a'}")
    out.append(f"- prompt {snap['prompt_version']}, policy {snap['policy']}, seed {snap['seed']}, snapshot_sha "
               f"{snap['snapshot_sha']}, extensions {len(snap['extensions'])}, runs {','.join(snap['runs']) or 'n/a'}, "
               f"sample drawn {snap['created_at']}, n = {snap['n']}")
    out.append(f"- bias-test specification (from the snapshot): `{canonical(spec)}`")
    out.append("- strata (N population / n sampled): " + ", ".join(f"{s}={d['N']}/{d['n']}"
                                                                     for s, d in snap["strata"].items()))
    summary = (f"**Summary:** accepted-label accuracy {_f(acc_a)} (n={len(acc_rows)}, policy {policy}); "
               f"big-brand bias (accepted LLM labels): {verdicts['accepted']}")
    out = out[:2] + [summary, ""] + out[2:]
    return "\n".join(out) + "\n", summary


def validation_summary_line(path: Path | None = None) -> str | None:
    p = Path(path or report_path())
    if not p.exists():
        return None
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("**Summary:**"):
            return ln[len("**Summary:**"):].strip()
    return None
