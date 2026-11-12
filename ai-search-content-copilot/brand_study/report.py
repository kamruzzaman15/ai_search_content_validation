from __future__ import annotations
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from brand_study import config

SURFACE_TITLES = {"organic": "Organic top 10 (distinct root domains)",
                  "aio_reference": "AI Overview cited sources (distinct root domains)",
                  "aio_mention": "AI Overview brand/business mentions (confirmed entities)"}

DISCLAIMER = (
    "**These results are observational, not causal. They come from one geography (Florida, US) at specific "
    "point(s) in time, using SerpApi's simulated location and a desktop device, with no login or personalisation. "
    "Real users see results that vary with personalisation, exact location, device and time. AI Overviews are "
    "non-deterministic. Big/small labels come from transparent rules plus manual overrides and can be wrong. "
    "Cells are small (about 10-25 queries), so confidence intervals are wide.**"
)

LIMITATIONS = [
    "Domain-level labels undercount small businesses. Small businesses appear mostly *inside* directories and "
    "listicles (Yelp, Tripadvisor, local media) and in the local pack, and cited pages are not scraped.",
    "Small-business AIO mentions depend on candidates being confirmed (dictionary, local-pack match or override). "
    "Before review, the small share is biased down. The bounds show how far it could move.",
    "The upper bound for AIO mentions is loose, because unconfirmed candidates can include non-brands "
    "(headings, features).",
    "Social profiles (Instagram, Facebook, TikTok) are labeled as directory_media_other, even when the profile "
    "belongs to a small business.",
    "Mention detection is dictionary-based plus list-title candidates. Brands outside the rule file are found only "
    "through list titles or the local pack.",
    "Broad product queries use 'small/independent brand' and local queries use 'independent local business'. "
    "Results are split by query_type for this reason.",
    "Optional LLM labels (three judges, accepted only when all agree) fill gaps the rules leave. LLMs may favour "
    "global brands over local ones; the judges' accuracy and a pre-specified big-brand bias test come from a "
    "small blinded human sample (default 50 items), so their confidence intervals are wide.",
]

ORIGINS = ("rules", "human", "llm", "llm_confirm", "unlabeled")


def label_origin(src) -> str:
    s = str(src or "")
    if s.startswith("local_pack_website:"):
        s = s.split(":", 1)[1]
    if s in ("", "none", "unconfirmed", "nan"):
        return "unlabeled"
    if s == "override":
        return "human"
    if s.startswith("llm_"):
        return "llm_confirm" if s.endswith("_confirm") else "llm"
    return "rules"


def origin_table(frames: dict) -> tuple[pd.DataFrame, Counter]:
    rows, total = [], Counter()
    for surface, name in (("organic", "organic_results"), ("aio_reference", "aio_references"),
                          ("aio_mention", "aio_mentions"), ("local_pack", "local_pack")):
        df = frames.get(name)
        c = Counter()
        if df is not None and not df.empty and "label_source" in df:
            c.update(label_origin(x) for x in df["label_source"])
        total.update(c)
        rows.append({"surface": surface, **{o: c.get(o, 0) for o in ORIGINS}, "rows": sum(c.values())})
    rows.append({"surface": "total", **{o: total.get(o, 0) for o in ORIGINS}, "rows": sum(total.values())})
    return pd.DataFrame(rows), total


def llm_share_sentence(total: Counter, llm_meta: dict | None = None) -> str:
    llm_meta = llm_meta or {}
    labeled = sum(v for k, v in total.items() if k != "unlabeled")
    all_rows = sum(total.values())
    from_llm = total.get("llm", 0) + total.get("llm_confirm", 0)

    def pct(k, n):
        return f"{100 * k / n:.1f}%" if n else "n/a"
    return (f"{pct(from_llm, labeled)} of labeled rows ({pct(from_llm, all_rows)} of all rows) came from the LLM "
            f"judges: {pct(total.get('llm', 0), labeled)} labelled by the LLM and {pct(total.get('llm_confirm', 0), labeled)} "
            f"rule categories confirmed by the LLM (policy: {llm_meta.get('policy') or 'n/a'}, prompt "
            f"{llm_meta.get('prompt_version') or 'n/a'}).")

METHOD_NOTES = [
    "The headline unit is the distinct root domain per query (organic, AIO references), or the distinct confirmed "
    "entity per query (mentions). Shares are macro-averaged over queries. `not_a_brand` items are excluded from "
    "numerators and denominators, and `unknown` stays in the denominator.",
    "Bounds: the low bound counts unresolved items (unknown plus unconfirmed candidates) as 'not this category', and "
    "the high bound counts them as 'this category'. Rows are `provisional` when the unresolved share is above the "
    "threshold.",
    "Queries with no eligible items on a surface are left out of that surface's macro average (n_queries_empty). "
    "They still count as 'not present' in pct_queries_present.",
    "The local pack is descriptive only and is not part of the headline numbers. It is used to confirm AIO list "
    "candidates (`local_pack_name`) and to suggest labels.",
    "AIO rate = (present + follow-up failed) / (present + follow-up failed + absent + error). Follow-up failures "
    "(errors, expired 1-minute page_token, budget exhausted) are cached and never re-requested, except with "
    "`--retry-failed-aio`, and are excluded from content metrics.",
    "Retries: at most `--retries` extra attempts on 5xx/429 throughput/transport errors, each charged to the "
    "runtime budget before it is sent. The ledger has one line per HTTP attempt.",
    "Alias ambiguity modes: `no` (case-insensitive whole word), `case` (case-sensitive), `case_mid` "
    "(case-sensitive and not at the start of a sentence, list item, heading or cell), `candidate` (match "
    "creates an unconfirmed candidate only).",
]


def _fmt(v, floatfmt: str = ".2f") -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if math.isinf(v):
            return "∞"
        return format(v, floatfmt)
    return str(v).replace("|", "\\|")


def md_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df is None or df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def _pct(v) -> str:
    return "n/a" if v is None or v != v else f"{100 * v:.0f}%"


def _get(summary, seg, surface, category, metric, col="value"):
    s = summary
    sel = s[(s["run"] == seg[0]) & (s["sector"] == seg[1]) & (s["query_type"] == seg[2]) &
            (s["surface"] == surface) & (s["category"] == category) & (s["metric"] == metric)]
    return sel.iloc[0][col] if len(sel) else float("nan")


def headline_provisional(summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    h = summary[(summary["metric"] == "domain_share") & summary["category"].isin(["big_brand", "small_local"])]
    return bool(h["provisional"].astype(bool).any())


def _pooled_unresolved(frames: dict, surface: str) -> float:
    if surface == "organic":
        df = frames["organic_results"]
        if df.empty:
            return float("nan")
        d = df.drop_duplicates(["run", "query_id", "root_domain"])
        d = d[d["category"] != "not_a_brand"]
        return float((d["category"] == "unknown").mean()) if len(d) else float("nan")
    m = frames["aio_mentions"]
    if m.empty:
        return float("nan")
    m = m[m["category"] != "not_a_brand"]
    return float(((m["category"] == "unknown") | ~m["confirmed"].astype(bool)).mean()) if len(m) else float("nan")


def ratio_text(big: float, small: float, provisional: bool) -> str:
    if big != big or small != small:
        return "n/a"
    if small == 0:
        return "∞ (provisional)" if provisional or big > 0 else "n/a"
    text = f"{big / small:.1f}x"
    return text + " (provisional)" if provisional else text


def main_table(summary: pd.DataFrame, surface: str, segments) -> pd.DataFrame:
    rows = []
    for seg in segments:
        n = _get(summary, seg, surface, "big_brand", "domain_share", "n_queries")
        if n != n:
            continue
        cells = {"segment": f"{seg[0]} {seg[1]}/{seg[2]}", "n": int(n)}
        for cat, label in (("big_brand", "big"), ("small_local", "small_local")):
            v = _get(summary, seg, surface, cat, "domain_share")
            lo = _get(summary, seg, surface, cat, "domain_share", "bound_low")
            hi = _get(summary, seg, surface, cat, "domain_share", "bound_high")
            cells[f"{label} (low–high)"] = f"{_pct(v)} ({_pct(lo)}–{_pct(hi)})"
        cells["marketplace"] = _pct(_get(summary, seg, surface, "marketplace_retailer", "domain_share"))
        cells["directory/media"] = _pct(_get(summary, seg, surface, "directory_media_other", "domain_share"))
        cells["unknown+unconfirmed"] = _pct(_get(summary, seg, surface, "big_brand", "domain_share",
                                                 "unresolved_share"))
        prov = bool(_get(summary, seg, surface, "big_brand", "domain_share", "provisional"))
        cells["big:small"] = ratio_text(_get(summary, seg, surface, "big_brand", "domain_share"),
                                        _get(summary, seg, surface, "small_local", "domain_share"), prov)
        cells["provisional"] = "PROVISIONAL" if prov else ""
        rows.append(cells)
    return pd.DataFrame(rows)


def build_report(frames: dict, runs, *, queries_file: str = "queries_v1.csv", ledger_rows: list[dict] | None = None,
                 overrides_count: int = 0, threshold: float = config.DEFAULT_PROVISIONAL_THRESHOLD,
                 llm_overrides_count: int = 0, llm_meta: dict | None = None,
                 field_verification: str = "documentation (see brand_study/SERPAPI_NOTES.md) and live smoke-test "
                                           "responses") -> str:
    summary = frames["summary_by_segment"]
    status = frames["query_status"]
    present_runs = [r for r in runs if not status.empty and (status["run"] == r).any()
                    and (status[status["run"] == r]["search_outcome"] != "").any()]
    segs = [(run, sec, qt) for run in present_runs for sec in ("all",) + config.SECTORS
            for qt in ("all",) + config.QUERY_TYPES]
    out: list[str] = []
    if headline_provisional(summary):
        out.append(f"> **PROVISIONAL: {_pct(_pooled_unresolved(frames, 'organic'))} of organic domains and "
                   f"{_pct(_pooled_unresolved(frames, 'aio_mention'))} of AIO mentions are unlabeled or unconfirmed. "
                   f"Small-business shares are lower bounds until `labels_to_review.csv` is reviewed and `analyze` "
                   f"is re-run.**\n")
    out.append("# Brand visibility study, Part 1: Google organic vs AI Overviews (Florida)\n")
    collected = status.loc[status["collected_at"].astype(str) != "", "collected_at"] if not status.empty else []
    dates = f"{min(collected)} to {max(collected)}" if len(collected) else "nothing collected yet"
    out.append(f"- Collection window (UTC): {dates}")
    out.append(f"- Location: broad queries use `{config.STATE_LOCATION}`, and local queries use the city location. "
               f"gl={config.GL}, hl={config.HL}, device={config.DEVICE}, google_domain={config.GOOGLE_DOMAIN}")
    out.append(f"- Query file: `{queries_file}`. Runs analysed: {', '.join(runs)}. Runs with data: "
               f"{', '.join(present_runs) or 'none'}")
    for run in present_runs:
        s = status[status["run"] == run]
        out.append(f"  - {run}: {int((s['search_outcome'] != '').sum())} of {len(s)} queries collected")
    ledger_rows = ledger_rows or []
    kinds = Counter(r.get("kind") for r in ledger_rows)
    retries = sum(1 for r in ledger_rows if r.get("reason") == "retry")
    out.append(f"- SerpApi searches used (ledger attempts, whole study): {len(ledger_rows)} "
               f"(search {kinds.get('search', 0)}, ai_overview {kinds.get('ai_overview', 0)}, retries {retries}). "
               f"The Account API `this_month_usage` delta is printed by each `collect`.")
    sup = status[status["superseded"].astype(bool)] if not status.empty else status
    if len(sup):
        out.append("- Superseded (re-collected with `--retry-failed-aio`): " + ", ".join(
            f"{r.run}/{r.query_id} (old {r.supersedes}, new {r.collected_at})" for r in sup.itertuples()))
    out.append(f"- Field names verified against: {field_verification}\n")
    out.append(DISCLAIMER + "\n")
    llm_meta = llm_meta or {}
    if llm_overrides_count > 0:
        # Count what was actually applied, not the rows in the file: items that later got a human override (or a rule)
        # still sit in llm_label_overrides.csv until `llm-label --aggregate-only` is re-run, but they never win.
        _, tot = origin_table(frames)
        applied = tot.get("llm", 0) + tot.get("llm_confirm", 0)
        note = (f"**LLM labels: {llm_overrides_count} LLM labels for domains/entities the rules left unknown were "
                f"loaded from three LLM judges (accepted only under the `{llm_meta.get('policy') or 'unanimous'}` "
                f"policy; human overrides and rules always win), and they decided the label of {applied} rows in "
                f"this report. LLMs may favour global brands over local ones, so these labels can share a big-brand "
                f"bias.")
        if llm_meta.get("validation_summary"):
            note += f" Validation (`llm_validation.md`): {llm_meta['validation_summary']}"
        else:
            note += " See `llm_validation.md` (run `llm-sample` and `llm-agreement`) for their measured accuracy."
        if llm_meta.get("runs_note"):
            note += f" {llm_meta['runs_note']}"
        out.append(note + "**\n")

    out.append("## Category definitions\n")
    for cat in config.CATEGORIES:
        out.append(f"- **{cat}**: {config.CATEGORY_DEFINITIONS[cat]}")
    out.append("")

    out.append("## AI Overview rate\n")
    rate_rows = []
    for seg in segs:
        n = _get(summary, seg, "aio", "", "aio_rate", "n_queries")
        if n != n:
            continue
        rate_rows.append({
            "segment": f"{seg[0]} {seg[1]}/{seg[2]}", "aio_rate": _pct(_get(summary, seg, "aio", "", "aio_rate")),
            "95% CI": f"{_pct(_get(summary, seg, 'aio', '', 'aio_rate', 'ci_low'))}–"
                      f"{_pct(_get(summary, seg, 'aio', '', 'aio_rate', 'ci_high'))}",
            "inline": int(_get(summary, seg, "aio", "", "n_present_inline")),
            "followup": int(_get(summary, seg, "aio", "", "n_present_followup")),
            "followup_failed": int(_get(summary, seg, "aio", "", "n_followup_failed")),
            "absent": int(_get(summary, seg, "aio", "", "n_absent")),
            "error": int(_get(summary, seg, "aio", "", "n_error")),
            "not collected/failed": int(_get(summary, seg, "aio", "", "n_not_collected")
                                        + _get(summary, seg, "aio", "", "n_search_failed")),
        })
    out.append(md_table(pd.DataFrame(rate_rows)) + "\n")

    out.append("## Big vs small/local visibility\n")
    out.append(f"Each cell shows the point estimate (macro-averaged share of distinct items per query) and, in "
               f"brackets, its bounds with unresolved items counted as 'not this category' or 'this category'. "
               f"PROVISIONAL means the unresolved share is above {threshold:.0%}.\n")
    for surface in ("organic", "aio_reference", "aio_mention"):
        out.append(f"### {SURFACE_TITLES[surface]}\n")
        out.append(md_table(main_table(summary, surface, segs)) + "\n")
        all_segs = [s for s in segs if s[1] == "all" and s[2] == "all"]
        for seg in all_segs:
            excluded = _get(summary, seg, surface, "not_a_brand", "n_excluded_items")
            if excluded != excluded:
                continue  # no queries on this surface (e.g. no AI Overviews)
            extra = [f"not_a_brand items excluded: {int(excluded)}"]
            if surface == "organic":
                extra.append(f"result_share (share of the 10 result slots): big "
                             f"{_pct(_get(summary, seg, surface, 'big_brand', 'result_share'))}, small_local "
                             f"{_pct(_get(summary, seg, surface, 'small_local', 'result_share'))}, directory/media "
                             f"{_pct(_get(summary, seg, surface, 'directory_media_other', 'result_share'))}")
            if surface == "aio_mention":
                v = _get(summary, seg, surface, "unknown", "n_unconfirmed_candidates")
                extra.append(f"unconfirmed candidates: {int(v) if v == v else 0}")
            out.append(f"- {seg[0]}: " + "; ".join(extra))
        out.append("")

    out.append("## Overlap between AI Overview sources and the organic top 10\n")
    ov = frames["overlap_by_query"]
    if ov.empty:
        out.append("_No queries with AI Overview content._\n")
    else:
        g = ov.groupby(["run", "sector", "query_type"]).agg(
            n=("query_id", "count"), mean_aio_in_organic=("aio_in_organic", "mean"), mean_jaccard=("jaccard", "mean"),
            mean_shared=("n_shared", "mean"), shared_small_local=("shared_small_local", "sum"),
            shared_big=("shared_big", "sum"), zero_ref_queries=("n_aio_domains", lambda x: int((x == 0).sum()))
        ).reset_index()
        out.append(md_table(g) + "\n")

    out.append("## Run-to-run stability\n")
    stab = frames["stability_by_query"]
    if stab.empty:
        out.append("_run2 not collected yet (or only one run analysed): stability not computed._\n")
    else:
        g = stab.groupby(["sector", "query_type"]).agg(
            n=("query_id", "count"), organic_jaccard=("organic_jaccard", "mean"),
            aio_ref_jaccard=("aio_ref_jaccard", "mean"), aio_mention_jaccard=("aio_mention_jaccard", "mean"),
            aio_presence_agreement=("aio_presence_agreement", "mean")).reset_index()
        out.append(md_table(g) + "\n")

    out.append("## Local pack (descriptive, local queries only)\n")
    lp_rows = []
    for seg in segs:
        if seg[2] != "local":
            continue
        n = _get(summary, seg, "local_pack", "", "local_pack_rate", "n_queries")
        if n != n:
            continue
        row = {"segment": f"{seg[0]} {seg[1]}/{seg[2]}", "n": int(n),
               "local_pack_rate": _pct(_get(summary, seg, "local_pack", "", "local_pack_rate")),
               "mean_places": _get(summary, seg, "local_pack", "", "mean_places")}
        for cat in ("big_brand", "small_local", "unknown"):
            row[cat] = _pct(_get(summary, seg, "local_pack", cat, "place_share"))
        lp_rows.append(row)
    out.append(md_table(pd.DataFrame(lp_rows)) + "\n")

    out.append("## Labeling coverage\n")
    src = Counter()
    for name in ("organic_results", "aio_references", "aio_mentions", "local_pack"):
        df = frames[name]
        if not df.empty and "label_source" in df:
            src.update(df["label_source"].astype(str))
    review = frames["labels_to_review"]
    n_sug = int((review["suggested_category"] != "").sum()) if not review.empty else 0
    out.append("- Label sources: " + (", ".join(f"{k}={v}" for k, v in src.most_common()) or "none"))
    out.append(f"- Manual overrides loaded: {overrides_count}")
    out.append(f"- LLM labels loaded (`llm_label_overrides.csv`): {llm_overrides_count}")
    otab, ototal = origin_table(frames)
    out.append("- Label origin by surface (rows): rules, human overrides, LLM labels, rule categories confirmed by "
               "the LLM, unlabeled\n")
    out.append(md_table(otab) + "\n")
    out.append("- " + llm_share_sentence(ototal, llm_meta))
    if llm_meta.get("runs_note"):
        out.append(f"- WARNING: {llm_meta['runs_note']}")
    out.append(f"- Items in `labels_to_review.csv`: {len(review)} ({n_sug} with a suggestion, never applied "
               f"automatically)\n")

    out.append("## Limitations\n")
    out += [f"- {x}" for x in LIMITATIONS]
    out.append("\n## Method notes\n")
    out += [f"- {x}" for x in METHOD_NOTES]
    return "\n".join(out) + "\n"


def write_report(text: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
