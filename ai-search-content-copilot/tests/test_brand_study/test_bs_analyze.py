from __future__ import annotations
import math

import pandas as pd
import pytest

from bs_helpers import fx, q, write_wrap
from brand_study import analyze
from brand_study.labeling import Labeler, load_rules
from brand_study.report import ratio_text


def status_row(qid, run="run1", sector="coffee", qtype="local", aio="present_inline", outcome="ok", lp=False):
    return {"run": run, "query_id": qid, "sector": sector, "query_type": qtype, "aio_status": aio,
            "search_outcome": outcome, "has_local_pack": lp, "n_local_pack": 3 if lp else 0,
            "collected_at": "2026-11-01T00:00:00Z", "superseded": False, "supersedes": ""}


def dom_rows(run, qid, cats, col="rank"):
    """cats: list of (domain, category) in rank order."""
    return [{"run": run, "query_id": qid, col: i + 1, "root_domain": d, "category": c, "url_norm": f"https://{d}/{i}"}
            for i, (d, c) in enumerate(cats)]


def frames(status, organic=(), refs=(), mentions=(), local=()):
    return {
        "query_status": pd.DataFrame(status),
        "organic_results": pd.DataFrame(list(organic), columns=["run", "query_id", "rank", "root_domain", "category",
                                                                "url_norm"]),
        "aio_references": pd.DataFrame(list(refs), columns=["run", "query_id", "ref_index", "root_domain",
                                                            "category", "url_norm"]),
        "aio_mentions": pd.DataFrame(list(mentions), columns=["run", "query_id", "entity_norm", "confirmed",
                                                              "category"]),
        "local_pack": pd.DataFrame(list(local), columns=["run", "query_id", "category"]),
    }


def get(summary, surface, category, metric, col="value", seg=("run1", "all", "all")):
    s = summary
    sel = s[(s["run"] == seg[0]) & (s["sector"] == seg[1]) & (s["query_type"] == seg[2]) & (s["surface"] == surface)
            & (s["category"] == category) & (s["metric"] == metric)]
    assert len(sel) == 1, (surface, category, metric)
    return sel.iloc[0][col]


def test_wilson_and_jaccard():
    lo, hi = analyze.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert all(math.isnan(x) for x in analyze.wilson(0, 0))
    assert analyze.jaccard({1, 2, 3, 4}, {1, 2, 3, 5}) == pytest.approx(0.6)
    assert math.isnan(analyze.jaccard(set(), set()))


def test_domain_share_vs_result_share():
    others = [(f"d{i}.com", "big_brand") for i in range(8)]
    org = dom_rows("run1", "q1", [("yelp.com", "directory_media_other"), ("yelp.com", "directory_media_other")]
                   + others)
    s = analyze.summarize(frames([status_row("q1")], org), ["run1"])
    assert get(s, "organic", "directory_media_other", "domain_share") == pytest.approx(1 / 9)
    assert get(s, "organic", "directory_media_other", "result_share") == pytest.approx(2 / 10)
    assert get(s, "organic", "directory_media_other", "pct_rank1") == 1.0
    assert get(s, "organic", "big_brand", "mean_best_rank") == 3


def test_zero_denominator_dropped_but_counted():
    org = dom_rows("run1", "q1", [("a.com", "small_local"), ("b.com", "big_brand")]) + \
        dom_rows("run1", "q2", [("x.com", "not_a_brand"), ("y.com", "not_a_brand")])
    s = analyze.summarize(frames([status_row("q1"), status_row("q2")], org), ["run1"])
    assert get(s, "organic", "small_local", "domain_share") == pytest.approx(0.5)  # q2 dropped
    assert get(s, "organic", "small_local", "domain_share", "n_queries_empty") == 1
    assert get(s, "organic", "small_local", "pct_queries_present") == pytest.approx(0.5)  # q2 counts as absent
    assert get(s, "organic", "small_local", "top3_share", "n_queries_empty") == 1
    assert get(s, "organic", "not_a_brand", "n_excluded_items") == 2


def test_bounds_example_and_provisional():
    refs = dom_rows("run1", "q1", [("a.com", "big_brand")], "ref_index")
    mentions = [
        {"run": "run1", "query_id": "q1", "entity_norm": "bigco", "confirmed": True, "category": "big_brand"},
        {"run": "run1", "query_id": "q1", "entity_norm": "u1", "confirmed": True, "category": "unknown"},
        {"run": "run1", "query_id": "q1", "entity_norm": "u2", "confirmed": True, "category": "unknown"},
        {"run": "run1", "query_id": "q1", "entity_norm": "cand", "confirmed": False, "category": "unknown"},
        {"run": "run1", "query_id": "q1", "entity_norm": "ingredient", "confirmed": True, "category": "not_a_brand"},
    ]
    s = analyze.summarize(frames([status_row("q1")], refs=refs, mentions=mentions), ["run1"])
    assert get(s, "aio_mention", "small_local", "domain_share") == 0
    assert get(s, "aio_mention", "small_local", "domain_share", "bound_low") == 0
    assert get(s, "aio_mention", "small_local", "domain_share", "bound_high") == pytest.approx(3 / 4)
    assert get(s, "aio_mention", "big_brand", "domain_share") == pytest.approx(1 / 3)
    assert get(s, "aio_mention", "big_brand", "domain_share", "bound_low") == pytest.approx(1 / 4)
    assert get(s, "aio_mention", "big_brand", "domain_share", "unresolved_share") == pytest.approx(3 / 4)
    assert bool(get(s, "aio_mention", "big_brand", "domain_share", "provisional"))
    assert get(s, "aio_mention", "unknown", "n_unconfirmed_candidates") == 1
    assert not bool(get(s, "aio_reference", "big_brand", "domain_share", "provisional"))


@pytest.mark.parametrize("n_unknown,expected", [(1, False), (2, True)])
def test_provisional_threshold(n_unknown, expected):
    cats = [(f"b{i}.com", "big_brand") for i in range(10 - n_unknown)] + \
        [(f"u{i}.com", "unknown") for i in range(n_unknown)]
    s = analyze.summarize(frames([status_row("q1")], dom_rows("run1", "q1", cats)), ["run1"], threshold=0.10)
    assert bool(get(s, "organic", "big_brand", "domain_share", "provisional")) is expected


def test_ratio_text():
    assert ratio_text(0.5, 0.0, True) == "∞ (provisional)"
    assert ratio_text(0.5, 0.25, False) == "2.0x"
    assert ratio_text(0.5, 0.25, True) == "2.0x (provisional)"
    assert ratio_text(float("nan"), 0.2, False) == "n/a"


def test_aio_rate_and_segments():
    st = [status_row("q1"), status_row("q2", aio="absent"), status_row("q3", aio="followup_failed"),
          status_row("q4", aio="error"), status_row("q5", aio="not_collected", outcome=""),
          status_row("q6", sector="skincare", qtype="broad", aio="present_followup")]
    s = analyze.summarize(frames(st), ["run1"])
    assert get(s, "aio", "", "aio_rate") == pytest.approx(3 / 5)
    assert get(s, "aio", "", "aio_rate", "n_queries") == 5
    assert get(s, "aio", "", "aio_rate", seg=("run1", "skincare", "broad")) == 1.0
    assert get(s, "aio", "", "n_not_collected") == 1


def test_overlap_and_stability():
    st = [status_row("q1"), status_row("q1", run="run2"), status_row("q2", aio="absent"),
          status_row("q2", run="run2", aio="present_inline")]
    org = dom_rows("run1", "q1", [("a.com", "small_local"), ("b.com", "big_brand"), ("c.com", "unknown")]) + \
        dom_rows("run2", "q1", [("a.com", "small_local"), ("b.com", "big_brand"), ("d.com", "unknown")])
    refs = dom_rows("run1", "q1", [("a.com", "small_local"), ("z.com", "unknown")], "ref_index")
    ov = analyze.overlap_table(pd.DataFrame(st), frames(st, org)["organic_results"],
                               frames(st, refs=refs)["aio_references"])
    r = ov[(ov["run"] == "run1") & (ov["query_id"] == "q1")].iloc[0]
    assert (r["n_aio_domains"], r["n_shared"], r["shared_small_local"]) == (2, 1, 1)
    assert r["aio_in_organic"] == 0.5 and r["jaccard"] == pytest.approx(1 / 4)
    empty = ov[(ov["run"] == "run2") & (ov["query_id"] == "q2")].iloc[0]
    assert empty["n_aio_domains"] == 0 and math.isnan(empty["aio_in_organic"])

    f = frames(st, org, refs)
    stab = analyze.stability_table(f["query_status"], f["organic_results"], f["aio_references"], f["aio_mentions"])
    s1 = stab.set_index("query_id").loc["q1"]
    assert s1["organic_jaccard"] == pytest.approx(2 / 4)
    assert s1["aio_ref_jaccard"] == 0.0 and math.isnan(s1["aio_mention_jaccard"])
    assert stab.set_index("query_id").loc["q2", "aio_presence_agreement"] == 0.0


def test_stability_needs_both_runs():
    st = [status_row("q1")]
    f = frames(st)
    assert analyze.stability_table(f["query_status"], f["organic_results"], f["aio_references"],
                                   f["aio_mentions"]).empty


def test_end_to_end_from_cache(tmp_path):
    root = tmp_path / "raw"
    q1 = q()
    q2 = q("skn_b_01", "best moisturizer for oily skin", "skincare", "broad")
    write_wrap(root, "run1", q1, "search", fx("search_inline_aio.json"))
    write_wrap(root, "run1", q2, "search", fx("search_page_token_only.json"))
    write_wrap(root, "run1", q2, "ai_overview", fx("ai_overview_followup.json"))
    out = tmp_path / "results"
    fr = analyze.run_analysis([q1, q2], ["run1", "run2"], Labeler(load_rules()), root=root, out_dir=out,
                              ledger_rows=[])
    st = fr["query_status"].set_index(["run", "query_id"])
    assert st.loc[("run1", "cof_l_01"), "aio_status"] == "present_inline"
    assert st.loc[("run1", "skn_b_01"), "aio_status"] == "present_followup"
    assert st.loc[("run2", "cof_l_01"), "aio_status"] == "not_collected"
    for name in ("query_status", "organic_results", "aio_references", "aio_mentions", "local_pack",
                 "summary_by_segment", "overlap_by_query", "stability_by_query", "labels_to_review"):
        assert (out / f"{name}.csv").exists()
    s = fr["summary_by_segment"]
    assert set(s["run"]) == {"run1"}  # run2 has no data
    seg = ("run1", "coffee", "local")
    assert get(s, "organic", "big_brand", "domain_share", seg=seg) == pytest.approx(1 / 9)
    assert get(s, "organic", "directory_media_other", "result_share", seg=seg) == pytest.approx(6 / 10)
    review = fr["labels_to_review"].set_index(["key_type", "key"])
    assert review.loc[("domain", "buddybrew.com"), "suggested_category"] == "small_local"
