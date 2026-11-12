"""Turn the three judges' cached votes into accepted labels, a disagreement list and a pending count.

Always uses the full judge set. The outputs are rebuilt from the vote cache on every run (idempotent), and the
accepted file is only (re)written when nothing is pending and no `--limit` was given.
"""
from __future__ import annotations
import csv
import io
from collections import Counter
from pathlib import Path
from typing import Callable

from brand_study import config
from brand_study.llm_evidence import atomic_write_text
from brand_study.llm_judges import PROMPT_VERSION, VoteStore

POLICIES = ("unanimous", "majority")
DISAGREEMENT_COLUMNS = ["key_type", "key", "outcome", "majority_category", "rule_categories", "stale_evidence",
                        "example_url", "example_title", "sectors", "surfaces"]


def disagreements_path() -> Path:
    return config.results_dir() / "llm_disagreements.csv"


def item_votes(rec: dict, store: VoteStore, judges=config.LLM_JUDGE_ORDER, reuse_stale: bool = False,
               prompt_version: str = PROMPT_VERSION) -> tuple[dict, bool]:
    """judge -> vote (None if missing); True if any vote was reused from older evidence."""
    out, stale = {}, False
    for j in judges:
        model = config.LLM_JUDGES[j][1]
        v = store.get(j, model, prompt_version, rec["item_id"], rec["evidence_sha"])
        if v is None and reuse_stale:
            v = store.latest_any(j, model, prompt_version, rec["item_id"])
            stale = stale or v is not None
        out[j] = v
    return out, stale


def majority_category(votes: dict) -> str:
    cats = [v["category"] for v in votes.values() if v and v.get("status") == "ok"]
    if not cats:
        return ""
    cat, n = Counter(cats).most_common(1)[0]
    return cat if n >= 2 else ""


def classify(votes: dict, rule_categories, policy: str = "unanimous") -> tuple[str, str | None, str]:
    """-> (outcome, accepted category or None, source)."""
    if any(v is None for v in votes.values()):
        return "pending", None, ""
    ok = [v for v in votes.values() if v.get("status") == "ok"]
    cats = [v["category"] for v in ok]
    n = len(votes)
    accepted, source = None, ""
    if len(ok) == n and len(set(cats)) == 1 and cats[0] != "unknown":
        outcome, accepted, source = "unanimous", cats[0], "llm_unanimous"
    elif len(ok) < n:
        outcome = "abstention"
    elif all(c == "unknown" for c in cats):
        outcome = "all_unknown"
    else:
        outcome = "majority_2of3" if majority_category(votes) else "no_majority"
    if accepted is None and policy == "majority":
        m = majority_category(votes)
        if m and m != "unknown":
            accepted, source = m, "llm_majority"
    if accepted is not None and rule_categories:
        rc = set(rule_categories)
        if len(rc) > 1 or accepted not in rc:
            return "conflicts_rule", None, ""
        source += "_confirm"
    return outcome, accepted, source


def _vote_note(votes: dict) -> str:
    parts = []
    for j, v in votes.items():
        model = config.LLM_JUDGES[j][1]
        if v and v.get("status") == "ok":
            parts.append(f"{model}={v['category']}({float(v['confidence']):.2f})")
        else:
            parts.append(f"{model}=abstain")
    return "; ".join(parts)


def aggregate(records: list[dict], store: VoteStore, *, policy: str = "unanimous", reuse_stale: bool = False,
              limited: bool = False, out_path: Path | None = None, disagreements_out: Path | None = None,
              runs=(), judges=config.LLM_JUDGE_ORDER, log: Callable = print) -> dict:
    if policy not in POLICIES:
        raise ValueError(policy)
    out_path = Path(out_path) if out_path else config.llm_overrides_path()
    disagreements_out = Path(disagreements_out) if disagreements_out else disagreements_path()
    accepted_rows, dis_rows = [], []
    outcomes: Counter = Counter()
    pending_changed = 0
    stale_n = 0
    for rec in records:
        votes, stale = item_votes(rec, store, judges, reuse_stale)
        outcome, cat, source = classify(votes, rec.get("rule_categories") or [], policy)
        outcomes[outcome] += 1
        if outcome == "pending":
            if any(store.latest_any(j, config.LLM_JUDGES[j][1], PROMPT_VERSION, rec["item_id"]) for j in judges
                   if votes.get(j) is None):
                pending_changed += 1
            continue
        stale_n += int(stale)
        if cat is not None:
            note = f"{source}; {_vote_note(votes)}" + ("; stale_evidence" if stale else "")
            accepted_rows.append((rec["key_type"], rec["key"], cat, note))
            continue
        row = {"key_type": rec["key_type"], "key": rec["key"], "outcome": outcome,
               "majority_category": majority_category(votes),
               "rule_categories": "|".join(rec.get("rule_categories") or []), "stale_evidence": stale,
               "example_url": rec.get("example_url", ""), "example_title": rec.get("example_title", ""),
               "sectors": rec.get("sectors", ""), "surfaces": rec.get("surfaces", "")}
        for j in judges:
            v = votes.get(j) or {}
            row[f"{j}_category"] = v.get("category") if v.get("status") == "ok" else "abstain"
            row[f"{j}_confidence"] = v.get("confidence")
            row[f"{j}_reason"] = v.get("reason") or v.get("error") or ""
        dis_rows.append(row)

    n_pending = outcomes.get("pending", 0)
    summary = {"accepted": len(accepted_rows), "disagreements": len(dis_rows), "pending": n_pending,
               "pending_changed_evidence": pending_changed, "stale_reused": stale_n, "outcomes": dict(outcomes),
               "accepted_written": False, "disagreements_written": False}
    if not limited:
        cols = DISAGREEMENT_COLUMNS + [f"{j}_{c}" for j in judges for c in ("category", "confidence", "reason")]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in sorted(dis_rows, key=lambda r: (r["outcome"], r["key_type"], r["key"])):
            w.writerow(r)
        atomic_write_text(disagreements_out, buf.getvalue())
        summary["disagreements_written"] = True
    if n_pending == 0 and not limited:
        atomic_write_text(out_path, _accepted_text(accepted_rows, store, policy, runs, stale_n, judges))
        summary["accepted_written"] = True
    _print_summary(summary, records, store, judges, limited, out_path, disagreements_out, log)
    return summary


def _accepted_text(rows, store: VoteStore, policy, runs, stale_n, judges) -> str:
    models, stamps = [], []
    for j in judges:
        model = config.LLM_JUDGES[j][1]
        recs = [r for r in store.records(j) if r.get("prompt_version") == PROMPT_VERSION]
        digests = sorted({r.get("model_digest") or r.get("cli_version") or "" for r in recs} - {""})
        models.append(f"{j}={model}" + (f"@{'+'.join(d[:19] for d in digests)}" if digests else ""))
        stamps += [r.get("ts", "") for r in recs]
    date = max(stamps) if stamps else ""
    buf = io.StringIO()
    buf.write("# GENERATED by `python -m brand_study llm-label`; do not edit. Put corrections in "
              "label_overrides.csv (human overrides always win).\n")
    buf.write(f"# prompt_version={PROMPT_VERSION}\n# policy={policy}\n# models={'; '.join(models)}\n")
    buf.write(f"# runs={','.join(runs)}\n# stale_evidence={stale_n}\n# latest_vote={date}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["key_type", "key", "category", "note"])
    for r in sorted(rows, key=lambda r: (r[0], r[1])):
        w.writerow(r)
    return buf.getvalue()


def _print_summary(s, records, store, judges, limited, out_path, dis_path, log):
    log(f"aggregation: {s['accepted']} accepted, {s['disagreements']} disagreements, {s['pending']} pending"
        + (f", {s['stale_reused']} with reused (stale-evidence) votes" if s["stale_reused"] else ""))
    log("  outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(s["outcomes"].items())))
    for j in judges:
        model = config.LLM_JUDGES[j][1]
        dist = Counter()
        for rec in records:
            v = store.get(j, model, PROMPT_VERSION, rec["item_id"], rec["evidence_sha"])
            if v:
                dist[v.get("category") if v.get("status") == "ok" else "abstain"] += 1
        if dist:
            log(f"  {j}: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))
    if s["disagreements_written"]:
        log(f"  wrote {dis_path}; resolve these by adding rows to brand_study/data/label_overrides.csv")
    if s["accepted_written"]:
        log(f"  wrote {out_path}")
    elif limited:
        log("  --limit given: accepted and disagreement files not written (trial run)")
    else:
        why = f" ({s['pending_changed_evidence']} because evidence changed since they were judged)" \
            if s["pending_changed_evidence"] else ""
        log(f"aggregation incomplete: {s['pending']} pending{why}. The existing accepted file is kept and analyze "
            f"will still use it; finish llm-label, or use --reuse-stale-votes, to refresh it.")
