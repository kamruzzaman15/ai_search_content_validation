from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from brand_study import budget as bud
from brand_study import config, extract
from brand_study.labeling import Labeler, build_review_table, label_domain_frame, label_mentions

SURFACES = ("organic", "aio_reference", "aio_mention")
BOUND_CATEGORIES = ("big_brand", "small_local")
SHARE_CATEGORIES = ("big_brand", "small_local", "marketplace_retailer", "directory_media_other", "unknown")
PRESENT = ("present_inline", "present_followup")
SHOWN = PRESENT + ("followup_failed",)
NAN = float("nan")

SUMMARY_COLUMNS = ["run", "sector", "query_type", "surface", "category", "metric", "value", "bound_low", "bound_high",
                   "unresolved_share", "provisional", "n_queries", "n_queries_empty", "ci_low", "ci_high"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return NAN, NAN
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return NAN
    return len(a & b) / len(a | b)


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else NAN


# --- Building tidy frames ----------------------------------------------------------

def _ledger_search_errors(rows: list[dict]) -> dict[tuple[str, str], str]:
    out = {}
    for r in rows:
        if r.get("kind") == "search" and r.get("outcome") == "error":
            out[(r.get("run"), r.get("query_id"))] = r.get("error_kind") or "error"
    return out


def build_frames(queries, runs, labeler: Labeler, root: Path | None = None,
                 ledger_rows: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    ledger_rows = bud.read_ledger() if ledger_rows is None else ledger_rows
    errs = _ledger_search_errors(ledger_rows)
    status, organic, refs, local, mentions = [], [], [], [], []
    for run in runs:
        data = extract.load_run(run, root)
        for q in queries:
            d = data.get(q.id, {"search": None, "ai_overview": None})
            sw, aw = d["search"], d["ai_overview"]
            status.append(extract.query_status_row(q, run, sw, aw, errs.get((run, q.id))))
            if sw is None:
                continue
            body = sw.get("response") or {}
            organic += extract.organic_rows(run, q.id, body)
            lp = extract.local_pack_rows(run, q.id, body)
            local += lp
            ai = extract.aio_content(sw, aw)
            refs += extract.aio_reference_rows(run, q.id, ai)
            mentions += extract.aio_mention_rows(run, q.id, ai, labeler.alias_index_for(q.sector), lp)

    status_df = pd.DataFrame(status)
    sectors = {(r["run"], r["query_id"]): r["sector"] for r in status}
    organic_df = label_domain_frame(pd.DataFrame(organic, columns=[
        "run", "query_id", "rank", "url", "url_norm", "host", "root_domain", "title", "source"]), labeler, sectors)
    refs_df = label_domain_frame(pd.DataFrame(refs, columns=[
        "run", "query_id", "ref_index", "url", "url_norm", "host", "root_domain", "source", "title",
        "cited_in_blocks"]), labeler, sectors)
    local_df = label_domain_frame(pd.DataFrame(local, columns=[
        "run", "query_id", "position", "title", "entity_norm", "website", "root_domain", "rating", "reviews",
        "type"]), labeler, sectors)
    if not local_df.empty:
        for i, r in local_df[local_df["category"] == "unknown"].iterrows():
            cat, ent, src = labeler.label_entity(r["title"], sectors.get((r["run"], r["query_id"])))
            if cat != "unknown":
                local_df.loc[i, ["category", "entity", "label_source"]] = [cat, ent, src]
    mentions_df = label_mentions(pd.DataFrame(mentions, columns=[
        "run", "query_id", "entity", "entity_norm", "methods", "block_types", "local_pack_match_domain"]),
        labeler, sectors)
    review = build_review_table(organic_df, refs_df, local_df, mentions_df, status_df)
    return {"query_status": status_df, "organic_results": organic_df, "aio_references": refs_df,
            "aio_mentions": mentions_df, "local_pack": local_df, "labels_to_review": review}


# --- Per-query item sets --------------------------------------------------------------

def _domain_items(df: pd.DataFrame, order_col: str) -> dict:
    """(run, qid) -> {"items": {domain: category}, "order": [domain...], "slots": [category...]}."""
    out: dict = defaultdict(lambda: {"items": {}, "order": [], "slots": []})
    if df.empty:
        return out
    for r in df.sort_values(["run", "query_id", order_col]).itertuples():
        e = out[(r.run, r.query_id)]
        e["slots"].append(r.category)
        if r.root_domain and r.root_domain not in e["items"]:
            e["items"][r.root_domain] = r.category
            e["order"].append(r.root_domain)
    return out


def _mention_items(df: pd.DataFrame) -> tuple[dict, dict]:
    items: dict = defaultdict(lambda: {"items": {}, "order": [], "slots": []})
    unconfirmed: dict = defaultdict(int)
    if df.empty:
        return items, unconfirmed
    for r in df.itertuples():
        key = (r.run, r.query_id)
        if bool(r.confirmed):
            items[key]["items"].setdefault(r.entity_norm, r.category)
        else:
            unconfirmed[key] += 1
    return items, unconfirmed


def _per_query(entry: dict, n_unconfirmed: int, category: str) -> dict:
    eligible = {k: c for k, c in entry["items"].items() if c not in config.HEADLINE_EXCLUDED}
    e = len(eligible)
    n_c = sum(1 for c in eligible.values() if c == category)
    u = sum(1 for c in eligible.values() if c == "unknown")
    d = e + n_unconfirmed
    return {
        "E": e, "n_c": n_c, "present": n_c > 0,
        "share": n_c / e if e else NAN,
        "low": n_c / d if d else NAN,
        "high": (n_c + u + n_unconfirmed) / d if d else NAN,
        "unresolved": (u + n_unconfirmed) / d if d else NAN,
        "excluded": len(entry["items"]) - e,
    }


def _segment_keys(status: pd.DataFrame, run: str, sector: str, qtype: str, statuses=None, outcomes=None) -> list:
    s = status[status["run"] == run]
    if sector != "all":
        s = s[s["sector"] == sector]
    if qtype != "all":
        s = s[s["query_type"] == qtype]
    if statuses is not None:
        s = s[s["aio_status"].isin(statuses)]
    if outcomes is not None:
        s = s[s["search_outcome"].isin(outcomes)]
    return [(r.run, r.query_id) for r in s.itertuples()]


def _row(seg, surface, category, metric, value, n, **kw) -> dict:
    row = {"run": seg[0], "sector": seg[1], "query_type": seg[2], "surface": surface, "category": category,
           "metric": metric, "value": value, "bound_low": NAN, "bound_high": NAN, "unresolved_share": NAN,
           "provisional": False, "n_queries": n, "n_queries_empty": 0, "ci_low": NAN, "ci_high": NAN}
    row.update(kw)
    return row


def _surface_rows(seg, surface, keys, items, unconfirmed, threshold) -> list[dict]:
    rows = []
    n = len(keys)
    if n == 0:
        return rows
    unresolved = _mean([_per_query(items[k], unconfirmed.get(k, 0), "unknown")["unresolved"] for k in keys])
    provisional = bool(unresolved == unresolved and unresolved > threshold)
    for cat in SHARE_CATEGORIES:
        pq = [_per_query(items[k], unconfirmed.get(k, 0), cat) for k in keys]
        empty = sum(1 for p in pq if p["E"] == 0)
        bounds = {}
        if cat in BOUND_CATEGORIES:
            bounds = {"bound_low": _mean([p["low"] for p in pq]), "bound_high": _mean([p["high"] for p in pq])}
        common = {"unresolved_share": unresolved, "provisional": provisional, "n_queries_empty": empty}
        rows.append(_row(seg, surface, cat, "domain_share", _mean([p["share"] for p in pq]), n, **bounds, **common))
        tot_e = sum(p["E"] for p in pq)
        rows.append(_row(seg, surface, cat, "pooled_domain_share",
                         sum(p["n_c"] for p in pq) / tot_e if tot_e else NAN, n, **common))
        k = sum(1 for p in pq if p["present"])
        lo, hi = wilson(k, n)
        rows.append(_row(seg, surface, cat, "pct_queries_present", k / n, n, ci_low=lo, ci_high=hi, **common))
        if surface in ("organic", "aio_reference"):
            top = []
            for key in keys:
                order = [d for d in items[key]["order"][:3] if items[key]["items"][d] not in config.HEADLINE_EXCLUDED]
                top.append(sum(1 for d in order if items[key]["items"][d] == cat) / len(order) if order else NAN)
            rows.append(_row(seg, surface, cat, "top3_share", _mean(top), n,
                             n_queries_empty=sum(1 for t in top if t != t)))
        if surface == "organic":
            first = [items[key]["slots"][0] if items[key]["slots"] else None for key in keys]
            rows.append(_row(seg, surface, cat, "pct_rank1", sum(1 for f in first if f == cat) / n, n))
            slots = [items[key]["slots"] for key in keys]
            rows.append(_row(seg, surface, cat, "result_share",
                             _mean([s.count(cat) / len(s) if s else NAN for s in slots]), n))
            if cat in BOUND_CATEGORIES:
                best = []
                for key in keys:
                    ranks = [i + 1 for i, c in enumerate(items[key]["slots"]) if c == cat]
                    if ranks:
                        best.append(min(ranks))
                rows.append(_row(seg, surface, cat, "mean_best_rank", _mean(best), len(best)))
    excluded = sum(_per_query(items[k], 0, "unknown")["excluded"] for k in keys)
    rows.append(_row(seg, surface, "not_a_brand", "n_excluded_items", excluded, n))
    if surface == "aio_mention":
        rows.append(_row(seg, surface, "unknown", "n_unconfirmed_candidates",
                         sum(unconfirmed.get(k, 0) for k in keys), n))
    return rows


def _segments(status: pd.DataFrame, runs) -> list[tuple[str, str, str]]:
    segs = []
    for run in runs:
        if status.empty or not (status.loc[status["run"] == run, "search_outcome"] != "").any():
            continue  # run not collected yet
        for sector in ("all",) + config.SECTORS:
            for qtype in ("all",) + config.QUERY_TYPES:
                segs.append((run, sector, qtype))
    return segs


def overlap_table(status: pd.DataFrame, organic: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["run", "query_id", "sector", "query_type", "n_aio_domains", "n_organic_domains", "n_shared",
            "aio_in_organic", "jaccard", "url_level_shared", "shared_small_local", "shared_big"]
    if status.empty:
        return pd.DataFrame(columns=cols)
    for r in status[status["aio_status"].isin(PRESENT)].itertuples():
        o = organic[(organic["run"] == r.run) & (organic["query_id"] == r.query_id)] if not organic.empty else organic
        a = refs[(refs["run"] == r.run) & (refs["query_id"] == r.query_id)] if not refs.empty else refs
        od = set(o["root_domain"]) - {""} if not o.empty else set()
        ad = set(a["root_domain"]) - {""} if not a.empty else set()
        shared = od & ad
        cat = dict(zip(o["root_domain"], o["category"])) if not o.empty else {}
        ou = set(o["url_norm"]) if not o.empty else set()
        au = set(a["url_norm"]) if not a.empty else set()
        rows.append({
            "run": r.run, "query_id": r.query_id, "sector": r.sector, "query_type": r.query_type,
            "n_aio_domains": len(ad), "n_organic_domains": len(od), "n_shared": len(shared),
            "aio_in_organic": len(shared) / len(ad) if ad else NAN,
            "jaccard": jaccard(od, ad) if ad else NAN,
            "url_level_shared": len((ou & au) - {""}),
            "shared_small_local": sum(1 for d in shared if cat.get(d) == "small_local"),
            "shared_big": sum(1 for d in shared if cat.get(d) == "big_brand"),
        })
    return pd.DataFrame(rows, columns=cols)


def stability_table(status: pd.DataFrame, organic: pd.DataFrame, refs: pd.DataFrame, mentions: pd.DataFrame,
                    run_a: str = "run1", run_b: str = "run2") -> pd.DataFrame:
    cols = ["query_id", "sector", "query_type", "organic_jaccard", "aio_ref_jaccard", "aio_mention_jaccard",
            "aio_presence_agreement"]
    if status.empty or not {run_a, run_b} <= set(status["run"]):
        return pd.DataFrame(columns=cols)

    def sets(df, col, run, qid, extra=None):
        if df.empty:
            return set()
        s = df[(df["run"] == run) & (df["query_id"] == qid)]
        if extra is not None:
            s = s[extra(s)]
        return set(s[col]) - {""}

    def conf(s):
        return s["confirmed"].astype(bool) & (s["category"] != "not_a_brand")

    st = status.set_index(["run", "query_id"])
    rows = []
    for qid in status.loc[status["run"] == run_a, "query_id"]:
        sa, sb = st.loc[(run_a, qid)], st.loc[(run_b, qid)]
        if sa["search_outcome"] == "" or sb["search_outcome"] == "":
            continue
        agree = NAN
        if sa["aio_status"] not in ("search_failed", "not_collected") and \
                sb["aio_status"] not in ("search_failed", "not_collected"):
            agree = float((sa["aio_status"] in SHOWN) == (sb["aio_status"] in SHOWN))
        rows.append({
            "query_id": qid, "sector": sa["sector"], "query_type": sa["query_type"],
            "organic_jaccard": jaccard(sets(organic, "root_domain", run_a, qid), sets(organic, "root_domain", run_b, qid)),
            "aio_ref_jaccard": jaccard(sets(refs, "root_domain", run_a, qid), sets(refs, "root_domain", run_b, qid)),
            "aio_mention_jaccard": jaccard(sets(mentions, "entity_norm", run_a, qid, conf),
                                           sets(mentions, "entity_norm", run_b, qid, conf)),
            "aio_presence_agreement": agree,
        })
    return pd.DataFrame(rows, columns=cols)


def summarize(frames: dict, runs, threshold: float = config.DEFAULT_PROVISIONAL_THRESHOLD) -> pd.DataFrame:
    status = frames["query_status"]
    organic_items = _domain_items(frames["organic_results"], "rank")
    ref_items = _domain_items(frames["aio_references"], "ref_index")
    mention_items, unconfirmed = _mention_items(frames["aio_mentions"])
    rows = []
    for seg in _segments(status, runs):
        run, sector, qtype = seg
        ok_keys = _segment_keys(status, run, sector, qtype, outcomes=["ok", "no_results"])
        present_keys = _segment_keys(status, run, sector, qtype, statuses=PRESENT)
        rows += _surface_rows(seg, "organic", ok_keys, organic_items, {}, threshold)
        rows += _surface_rows(seg, "aio_reference", present_keys, ref_items, {}, threshold)
        rows += _surface_rows(seg, "aio_mention", present_keys, mention_items, unconfirmed, threshold)
        # AIO rate
        s = status[status["run"] == run]
        if sector != "all":
            s = s[s["sector"] == sector]
        if qtype != "all":
            s = s[s["query_type"] == qtype]
        counts = s["aio_status"].value_counts().to_dict()
        n_present = sum(counts.get(x, 0) for x in PRESENT)
        n_failed = counts.get("followup_failed", 0)
        denom = n_present + n_failed + counts.get("absent", 0) + counts.get("error", 0)
        lo, hi = wilson(n_present + n_failed, denom)
        rows.append(_row(seg, "aio", "", "aio_rate", (n_present + n_failed) / denom if denom else NAN, denom,
                         ci_low=lo, ci_high=hi))
        for st_name in ("present_inline", "present_followup", "followup_failed", "absent", "error",
                        "search_failed", "not_collected"):
            rows.append(_row(seg, "aio", "", f"n_{st_name}", counts.get(st_name, 0), len(s)))
        # Local pack (descriptive, local queries only)
        if qtype in ("local", "all"):
            loc = s[s["query_type"] == "local"]
            loc = loc[loc["search_outcome"].isin(["ok", "no_results"])]
            if len(loc):
                rows.append(_row(seg, "local_pack", "", "local_pack_rate", loc["has_local_pack"].mean(), len(loc)))
                rows.append(_row(seg, "local_pack", "", "mean_places", loc["n_local_pack"].mean(), len(loc)))
                lp = frames["local_pack"]
                if not lp.empty:
                    keys = set(zip(loc["run"], loc["query_id"]))
                    sel = lp[[k in keys for k in zip(lp["run"], lp["query_id"])]]
                    for cat in SHARE_CATEGORIES + ("not_a_brand",):
                        rows.append(_row(seg, "local_pack", cat, "place_share",
                                         (sel["category"] == cat).mean() if len(sel) else NAN, len(loc)))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def run_analysis(queries, runs, labeler: Labeler, root: Path | None = None, out_dir: Path | None = None,
                 threshold: float = config.DEFAULT_PROVISIONAL_THRESHOLD, ledger_rows=None) -> dict:
    frames = build_frames(queries, runs, labeler, root, ledger_rows)
    frames["summary_by_segment"] = summarize(frames, runs, threshold)
    frames["overlap_by_query"] = overlap_table(frames["query_status"], frames["organic_results"],
                                               frames["aio_references"])
    a, b = (list(runs) + [None, None])[:2]
    frames["stability_by_query"] = stability_table(frames["query_status"], frames["organic_results"],
                                                   frames["aio_references"], frames["aio_mentions"],
                                                   a or "run1", b or "run2")
    out_dir = out_dir or config.results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return frames
