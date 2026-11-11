from __future__ import annotations

import pytest

from bs_helpers import fx, q, wrap
from brand_study import extract
from brand_study.labeling import Labeler, load_rules


@pytest.fixture(scope="module")
def labeler():
    return Labeler(load_rules())


def test_organic_rows_normalise_and_rank():
    rows = extract.organic_rows("run1", "cof_l_01", fx("search_inline_aio.json"))
    assert [r["rank"] for r in rows] == list(range(1, 11))
    roots = [r["root_domain"] for r in rows]
    assert roots == ["yelp.com", "yelp.com", "starbucks.com", "buddybrew.com", "tripadvisor.com", "reddit.com",
                     "blackandbrew.square.site", "facebook.com", "tampabay.com", "indiecafe.com"]
    assert "utm_source" not in rows[5]["url_norm"]
    assert "#" not in rows[8]["url_norm"]


def test_organic_keeps_top10_and_skips_garbage():
    rows = extract.organic_rows("run1", "x", fx("search_page_token_only.json"))
    assert len(rows) == 10 and "neutrogena.com" not in {r["root_domain"] for r in rows}
    odd = extract.organic_rows("run1", "x", fx("search_few_organic_local_pack.json"))
    assert [r["title"] for r in odd] == ["a", "b", "weird"]
    assert extract.organic_rows("run1", "x", {"organic_results": "nope"}) == []


def test_local_pack_both_shapes():
    rows = extract.local_pack_rows("run1", "x", fx("search_inline_aio.json"))
    assert [r["title"] for r in rows] == ["Buddy Brew Coffee", "Starbucks", "Black & Brew"]
    assert rows[0]["root_domain"] == "buddybrew.com" and rows[2]["website"] == ""
    lst = extract.local_pack_rows("run1", "x", fx("search_few_organic_local_pack.json"))
    assert [r["title"] for r in lst] == ["Odd Cafe", "No Site Cafe"]
    assert lst[0]["root_domain"] == "oddcafe.com"


def test_aio_references_and_citations():
    ai = fx("search_inline_aio.json")["ai_overview"]
    rows = extract.aio_reference_rows("run1", "x", ai)
    assert [r["root_domain"] for r in rows] == ["yelp.com", "buddybrew.com", "blackandbrew.square.site", "eater.com"]
    assert [r["cited_in_blocks"] for r in rows] == [1, 2, 1, 0]


def test_aio_mentions_inline(labeler):
    body = fx("search_inline_aio.json")
    lp = extract.local_pack_rows("run1", "x", body)
    rows = {r["entity_norm"]: r for r in extract.aio_mention_rows("run1", "x", body["ai_overview"],
                                                                     labeler.alias_index, lp)}
    assert {"starbucks", "dunkin", "target", "cvs"} <= set(rows)
    assert rows["starbucks"]["methods"] == "dictionary"
    assert rows["buddy brew coffee"]["methods"] == "list_title|local_pack_name"
    assert rows["buddy brew coffee"]["local_pack_match_domain"] == "buddybrew.com"
    assert "local_pack_name" in rows["black and brew"]["methods"]
    assert "pros" not in rows  # stoplist


def test_aio_mentions_ambiguous(labeler):
    body = fx("search_ambiguous_aliases.json")
    lp = extract.local_pack_rows("run1", "x", body)
    rows = {r["entity_norm"]: r for r in extract.aio_mention_rows("run1", "x", body["ai_overview"],
                                                                     labeler.alias_index, lp)}
    assert rows["the ordinary"]["methods"] == "dictionary"
    assert rows["illy"]["methods"] == "candidate"
    assert "dictionary" in rows["target"]["methods"]  # from "Sold at Target", not "Target dark spots"
    assert "local_pack_name" in rows["sunny skin studio"]["methods"]


def test_aio_text_walks_all_block_types():
    segs = extract.aio_segments(fx("ai_overview_followup.json")["ai_overview"])
    kinds = {(t, k) for t, k, _ in segs}
    assert ("Neutrogena Hydro Boost", "cell") in kinds
    assert ("Glow Lab Naturals", "list_item") in kinds
    nested = extract.aio_segments(fx("search_inline_aio.json")["ai_overview"])
    assert any("Dunkin" in t for t, _, _ in nested)  # expandable -> nested text_blocks


@pytest.mark.parametrize("search,aio,ledger,expected", [
    (None, None, "http_5xx", ("search_failed", "http_5xx")),
    (None, None, None, ("not_collected", "")),
    ("search_no_aio.json", None, None, ("absent", "")),
    ("search_inline_aio.json", None, None, ("present_inline", "")),
    ("search_page_token_only.json", "ok", None, ("present_followup", "")),
    ("search_page_token_only.json", "error", None, ("followup_failed", "api_error")),
    ("search_page_token_only.json", "budget", None, ("followup_failed", "budget_exhausted")),
    ("search_page_token_only.json", None, None, ("followup_failed", "stale_token")),
    ("search_aio_error.json", None, None, ("error", "An AI Overview is not available.")),
    ("no_results", None, None, ("absent", "no_results")),
])
def test_aio_status_table(search, aio, ledger, expected):
    sw = None
    if search == "no_results":
        sw = wrap(fx("search_no_results.json"), outcome="no_results")
    elif search:
        sw = wrap(fx(search))
    aw = None
    if aio == "ok":
        aw = wrap(fx("ai_overview_followup.json"), kind="ai_overview")
    elif aio == "error":
        aw = wrap({}, kind="ai_overview", outcome="error", error_kind="api_error")
    elif aio == "budget":
        aw = wrap({}, kind="ai_overview", outcome="error", error_kind="budget_exhausted")
    assert extract.aio_status(sw, aw, ledger) == expected


def test_aio_content_from_followup():
    sw = wrap(fx("search_page_token_only.json"))
    aw = wrap(fx("ai_overview_followup.json"), kind="ai_overview")
    ai = extract.aio_content(sw, aw)
    assert len(ai["references"]) == 3
    bare = wrap(fx("ai_overview_followup.json")["ai_overview"], kind="ai_overview")
    assert extract.aio_content(sw, bare)["references"]


def test_query_status_row():
    row = extract.query_status_row(q(), "run1", wrap(fx("search_inline_aio.json")), None)
    assert row["aio_status"] == "present_inline" and row["n_organic"] == 10
    assert row["n_organic_domains"] == 9 and row["has_local_pack"] and row["n_aio_refs"] == 4


def test_shopping_result_lists(labeler):
    """Live AIO shape: list items carry the product in shopping_result.title and cite Google Shopping."""
    ai = fx("ai_overview_followup_shopping.json")["ai_overview"]
    rows = {r["entity_norm"]: r for r in extract.aio_mention_rows("run1", "x", ai, labeler.alias_index, [])}
    assert rows["cerave"]["methods"] == "dictionary"
    assert "cerave oil control moisturizing gel cream" not in rows  # brand already counted, no extra candidate
    assert rows["glow lab gel moisturizer"]["methods"] == "list_title"
    assert rows["glow lab gel moisturizer"]["block_types"] == "shopping_result"
    refs = extract.aio_reference_rows("run1", "x", ai)
    assert [r["root_domain"] for r in refs] == ["youtube.com", "shopping.google.com", "shopping.google.com"]
    assert labeler.label_domain(refs[1]["host"])[0] == "marketplace_retailer"
