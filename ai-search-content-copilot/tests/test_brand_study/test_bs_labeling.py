from __future__ import annotations
import warnings

import pandas as pd
import pytest

from brand_study import config
from brand_study.config import ConfigError
from brand_study.labeling import (Labeler, build_review_table, label_mentions, load_overrides, load_rules)


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def test_seed_rules_valid(rules):
    assert len(rules) > 100
    big = [r for r in rules if r.category == "big_brand"]
    assert big and all(r.basis in ("a", "b", "c") for r in big)
    modes = {r.pattern: r.ambiguous for r in rules if r.match_type == "name"}
    assert modes["Target"] == "case_mid" and modes["The Ordinary"] == "case" and modes["illy"] == "candidate"
    assert "first aid" not in {p.lower() for p in modes}


def test_domain_precedence(rules):
    lab = Labeler(rules, {("domain", "yelp.com"): "small_local"})
    assert lab.label_domain("yelp.com") == ("small_local", "", "override")
    assert lab.label_domain("tripadvisor.com")[2] == "domain_rule"
    assert lab.label_domain("shop.starbucks.com", "coffee")[:1] == ("big_brand",)
    assert lab.label_domain("shop.starbucks.com", "coffee")[2] == "parent_domain_rule"
    assert lab.label_domain("cafe.square.site") == ("small_local", "", "platform_hosted")
    assert lab.label_domain("myflorida.gov") == ("directory_media_other", "", "tld")
    assert lab.label_domain("indiecafe.com") == ("unknown", "", "none")


def test_sector_specific_rules(rules):
    lab = Labeler(rules)
    assert lab.label_domain("starbucks.com", "coffee")[0] == "big_brand"
    assert lab.label_domain("starbucks.com", "skincare")[0] == "unknown"


def test_entity_labels(rules):
    lab = Labeler(rules, {("entity", "buddy brew coffee"): "small_local"})
    assert lab.label_entity("Dunkin’", "coffee")[0] == "big_brand"
    assert lab.label_entity("Buddy Brew Coffee Co.", "coffee")[:1] == ("small_local",)
    assert lab.label_entity("Hyaluronic Acid", "skincare")[0] == "not_a_brand"


def _write(tmp_path, text):
    p = tmp_path / "ov.csv"
    p.write_text(text)
    return p


def test_overrides_normalised_and_validated(tmp_path):
    p = _write(tmp_path, "key_type,key,category,note\n# comment\n"
                         "domain,https://www.BuddyBrew.com/x,small_local,\nentity,Café Bustelo,big_brand,\n")
    assert load_overrides(p) == {("domain", "buddybrew.com"): "small_local", ("entity", "cafe bustelo"): "big_brand"}
    with pytest.raises(ConfigError, match="line 3: invalid category"):
        load_overrides(_write(tmp_path, "key_type,key,category,note\n# c\ndomain,x.com,huge,\n"))
    with pytest.raises(ConfigError, match="line 2: invalid key_type"):
        load_overrides(_write(tmp_path, "key_type,key,category,note\nurl,x.com,small_local,\n"))
    with pytest.raises(ConfigError, match="line 2: empty key"):
        load_overrides(_write(tmp_path, "key_type,key,category,note\ndomain,,small_local,\n"))


def test_duplicate_override_last_wins(tmp_path):
    p = _write(tmp_path, "key_type,key,category,note\ndomain,a.com,small_local,\ndomain,www.a.com,big_brand,\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert load_overrides(p) == {("domain", "a.com"): "big_brand"}
    assert any("duplicate" in str(x.message) for x in w)


def test_shipped_overrides_file_loads():
    assert load_overrides(config.DEFAULT_OVERRIDES) == {}


def _mentions(rows):
    return pd.DataFrame(rows, columns=["run", "query_id", "entity", "entity_norm", "methods", "block_types",
                                       "local_pack_match_domain"])


def test_mention_confirmation(rules):
    sectors = {("run1", "q"): "coffee"}
    m = _mentions([
        ("run1", "q", "illy", "illy", "candidate", "paragraph", ""),
        ("run1", "q", "Squalane Oil", "squalane oil", "list_title", "list", ""),
        ("run1", "q", "Mandelic Acid", "mandelic acid", "list_title", "list", ""),
        ("run1", "q", "Starbucks", "starbucks", "dictionary", "paragraph", ""),
        ("run1", "q", "Dunkin", "dunkin", "list_title", "list", ""),
        ("run1", "q", "Sunny Cafe", "sunny cafe", "list_title|local_pack_name", "list", "sunnycafe.com"),
        ("run1", "q", "Kiki Roasters", "kiki roasters", "list_title|local_pack_name", "list", "kiki.square.site"),
    ])
    out = label_mentions(m, Labeler(rules), sectors).set_index("entity_norm")
    assert not out.loc["illy", "confirmed"] and out.loc["illy", "category"] == "unknown"  # candidate never self-confirms
    assert not out.loc["squalane oil", "confirmed"]  # ingredient heading not in the rules
    assert out.loc["starbucks", "confirmed"] and out.loc["starbucks", "category"] == "big_brand"
    assert out.loc["dunkin", "confirmed"] and out.loc["dunkin", "label_source"] == "name_rule"
    assert out.loc["sunny cafe", "confirmed"] and out.loc["sunny cafe", "category"] == "unknown"
    assert out.loc["kiki roasters", "category"] == "small_local"  # via the place's website domain

    ov = Labeler(rules, {("entity", "illy"): "big_brand", ("entity", "squalane oil"): "not_a_brand"})
    out2 = label_mentions(m, ov, sectors).set_index("entity_norm")
    assert out2.loc["illy", "confirmed"] and out2.loc["illy", "label_source"] == "override"
    assert out2.loc["squalane oil", "category"] == "not_a_brand"


def test_review_table_suggestions(rules):
    lab = Labeler(rules)
    status = pd.DataFrame([{"run": "run1", "query_id": "q", "sector": "coffee"}])
    organic = pd.DataFrame([
        {"run": "run1", "query_id": "q", "url": "https://sunnycafe.com/", "root_domain": "sunnycafe.com",
         "title": "Sunny", "category": "unknown"},
        {"run": "run1", "query_id": "q", "url": "https://other.com/", "root_domain": "other.com",
         "title": "Other", "category": "unknown"},
        {"run": "run1", "query_id": "q", "url": "https://yelp.com/", "root_domain": "yelp.com",
         "title": "Yelp", "category": "directory_media_other"},
    ])
    local = pd.DataFrame([{"run": "run1", "query_id": "q", "title": "Sunny Cafe", "website": "https://sunnycafe.com",
                           "root_domain": "sunnycafe.com", "category": "unknown"}])
    m = label_mentions(_mentions([
        ("run1", "q", "Sunny Cafe", "sunny cafe", "list_title|local_pack_name", "list", "sunnycafe.com"),
        ("run1", "q", "Mandelic Acid", "mandelic acid", "list_title", "list", ""),
        ("run1", "q", "Glow Lab Gel Moisturizer", "glow lab gel moisturizer", "list_title", "shopping_result", ""),
    ]), lab, {("run1", "q"): "coffee"})
    review = build_review_table(organic, pd.DataFrame(), local, m, status).set_index(["key_type", "key"])
    assert review.loc[("domain", "sunnycafe.com"), "suggested_category"] == "small_local"
    assert review.loc[("domain", "sunnycafe.com"), "suggest_reason"] == "local_pack_website"
    assert review.loc[("domain", "sunnycafe.com"), "count"] == 2  # organic + local pack
    assert review.loc[("entity", "sunny cafe"), "suggest_reason"] == "local_pack_name"
    assert review.loc[("entity", "mandelic acid"), "suggested_category"] == "not_a_brand"
    assert review.loc[("domain", "other.com"), "suggested_category"] == ""
    assert review.loc[("entity", "glow lab gel moisturizer"), "suggested_category"] == ""  # product, not ingredient
    assert ("domain", "yelp.com") not in review.index
    assert (review["category"] == "").all()  # never applied automatically


def test_invalid_rule_file(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("match_type,pattern,entity,category,sector,subtype,basis,ambiguous,notes\n"
                 "name,x,x,giant,any,,a,no,\nname,y,y,big_brand,any,,a,sometimes,\n")
    with pytest.raises(ConfigError) as e:
        load_rules(p)
    assert "line 2: invalid category" in str(e.value) and "line 3: invalid ambiguous" in str(e.value)


def test_alias_index_is_sector_filtered(rules):
    from brand_study.entities import find_aliases
    lab = Labeler(rules)
    text = [("Skip Starbucks; try CeraVe.", "sentence")]
    assert set(find_aliases(text, lab.alias_index)) >= {"Starbucks", "CeraVe"}
    assert set(find_aliases(text, lab.alias_index_for("skincare"))) == {"CeraVe"}
    assert set(find_aliases(text, lab.alias_index_for("coffee"))) == {"Starbucks"}
    # every dictionary match in a sector index has a rule label_entity() picks in that sector
    for e, _ in lab.alias_index_for("skincare").iter_compiled():
        assert lab.label_entity(e.entity, "skincare")[2] in ("name_rule", "override")
