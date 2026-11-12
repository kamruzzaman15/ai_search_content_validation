"""Precedence of the LLM labels: human override > rules > LLM fallback (LLM only fills rows still unknown)."""
from __future__ import annotations

import pandas as pd
import pytest

from bs_helpers import fx, q, write_wrap
from brand_study import analyze
from brand_study.config import ConfigError
from brand_study.labeling import Labeler, Rule, label_mentions, load_llm_overrides, load_rules

LLM = "llm_unanimous"


def rule(mt, pattern, cat, sector="any", ambiguous="no", entity=""):
    return Rule(mt, pattern, entity or pattern, cat, sector, "", "", ambiguous)


def _mentions(rows):
    return pd.DataFrame(rows, columns=["run", "query_id", "entity", "entity_norm", "methods", "block_types",
                                       "local_pack_match_domain"])


def test_label_domain_and_entity_ignore_llm():
    lab = Labeler([], {}, {("domain", "indiecafe.com"): ("small_local", LLM), ("entity", "acme"): ("big_brand", LLM)})
    assert lab.label_domain("indiecafe.com") == ("unknown", "", "none")
    assert lab.label_entity("Acme") == ("unknown", "Acme", "none")
    assert lab.llm_domain("www.indiecafe.com") == ("small_local", LLM)
    assert lab.llm_entity("ACME") == ("big_brand", LLM)


def test_sibling_subdomains_keep_their_rules():
    rules = [rule("domain", "maps.google.com", "directory_media_other"),
             rule("domain", "shopping.google.com", "marketplace_retailer")]
    lab = Labeler(rules, {}, {("domain", "google.com"): ("big_brand", LLM)})
    df = pd.DataFrame([{"run": "run1", "query_id": "q", "host": h, "root_domain": rd} for h, rd in
                       (("shopping.google.com", "shopping.google.com"), ("maps.google.com", "maps.google.com"),
                        ("google.com", "google.com"))])
    from brand_study.labeling import apply_llm_fallback_domains, label_domain_frame
    out = apply_llm_fallback_domains(label_domain_frame(df, lab, {}), lab).set_index("host")
    assert out.loc["shopping.google.com", "category"] == "marketplace_retailer"
    assert out.loc["maps.google.com", "category"] == "directory_media_other"
    assert tuple(out.loc["google.com", ["category", "label_source"]]) == ("big_brand", LLM)


def test_scenario1_human_entity_override_beats_llm_domain(tmp_path):
    root = tmp_path / "raw"
    query = q()
    write_wrap(root, "run1", query, "search", fx("search_few_organic_local_pack.json"))
    lab = Labeler(load_rules(), {("entity", "odd cafe"): "small_local"},
                  {("domain", "oddcafe.com"): ("big_brand", LLM)})
    fr = analyze.build_frames([query], ["run1"], lab, root, ledger_rows=[])
    lp = fr["local_pack"].set_index("title")
    assert tuple(lp.loc["Odd Cafe", ["category", "label_source"]]) == ("small_local", "override")
    org = fr["organic_results"].set_index("root_domain")
    assert tuple(org.loc["oddcafe.com", ["category", "label_source"]]) == ("big_brand", LLM)


def test_local_pack_llm_entity_hit_sets_entity(tmp_path):
    root = tmp_path / "raw"
    query = q()
    write_wrap(root, "run1", query, "search", fx("search_few_organic_local_pack.json"))
    lab = Labeler(load_rules(), {}, {("entity", "odd cafe"): ("small_local", LLM)})
    lp = analyze.build_frames([query], ["run1"], lab, root, ledger_rows=[])["local_pack"].set_index("title")
    assert tuple(lp.loc["Odd Cafe", ["category", "entity", "label_source"]]) == ("small_local", "Odd Cafe", LLM)


@pytest.mark.parametrize("rules,overrides,llm,expected", [
    ([], {("domain", "sunnycafe.com"): "small_local"}, {("entity", "sunny cafe"): ("big_brand", LLM)},
     ("small_local", "local_pack_website:override")),
    ([rule("domain", "sunnycafe.com", "small_local")], {}, {("entity", "sunny cafe"): ("big_brand", LLM)},
     ("small_local", "local_pack_website:domain_rule")),
    ([], {}, {("domain", "sunnycafe.com"): ("small_local", LLM)}, ("small_local", f"local_pack_website:{LLM}")),
])
def test_scenario2_local_pack_website_precedence(rules, overrides, llm, expected):
    m = _mentions([("run1", "q", "Sunny Cafe", "sunny cafe", "list_title|local_pack_name", "list", "sunnycafe.com")])
    out = label_mentions(m, Labeler(rules, overrides, llm), {("run1", "q"): "coffee"}).iloc[0]
    assert bool(out["confirmed"]) and (out["category"], out["label_source"]) == expected


def test_candidate_confirmation_scope():
    rules = [rule("name", "illy", "big_brand", "coffee", "candidate")]
    sectors = {("run1", "q"): "coffee"}
    m = _mentions([("run1", "q", "illy", "illy", "candidate", "paragraph", "")])
    ok = label_mentions(m, Labeler(rules, {}, {("entity", "illy"): ("big_brand", LLM)}), sectors).iloc[0]
    assert bool(ok["confirmed"]) and ok["label_source"] == f"{LLM}_confirm" and ok["category"] == "big_brand"
    bad = label_mentions(m, Labeler(rules, {}, {("entity", "illy"): ("small_local", LLM)}), sectors).iloc[0]
    assert not bool(bad["confirmed"]) and bad["label_source"] == "unconfirmed"
    m2 = _mentions([("run1", "q", "illy", "illy", "candidate|list_title", "list", "")])
    r2 = label_mentions(m2, Labeler(rules, {}, {("entity", "illy"): ("small_local", LLM)}), sectors).iloc[0]
    assert r2["label_source"] == "name_rule" and r2["category"] == "big_brand"


def test_local_pack_title_fallback_keeps_candidate_rule(tmp_path):
    root = tmp_path / "raw"
    query = q()
    write_wrap(root, "run1", query, "search", fx("search_few_organic_local_pack.json"))
    rules = [rule("name", "Odd Cafe", "small_local", "coffee", "candidate")]
    lab = Labeler(rules, {}, {("entity", "odd cafe"): ("big_brand", LLM)})
    lp = analyze.build_frames([query], ["run1"], lab, root, ledger_rows=[])["local_pack"].set_index("title")
    assert tuple(lp.loc["Odd Cafe", ["category", "label_source"]]) == ("small_local", "name_rule")


def test_sector_specific_candidate_confirmation():
    rules = [rule("name", "Acme", "big_brand", "coffee", "candidate"),
             rule("name", "Acme", "small_local", "skincare", "candidate")]
    sectors = {("run1", "a"): "coffee", ("run1", "b"): "skincare"}
    m = _mentions([("run1", "a", "Acme", "acme", "candidate", "paragraph", ""),
                   ("run1", "b", "Acme", "acme", "candidate", "paragraph", "")])
    out = label_mentions(m, Labeler(rules, {}, {("entity", "acme"): ("small_local", LLM)}), sectors)
    out = out.set_index("query_id")
    assert bool(out.loc["b", "confirmed"]) and out.loc["b", "label_source"] == f"{LLM}_confirm"
    assert not bool(out.loc["a", "confirmed"]) and out.loc["a", "label_source"] == "unconfirmed"


def test_list_title_only_and_rule_precedence():
    sectors = {("run1", "q"): "coffee"}
    m = _mentions([("run1", "q", "Glow Lab", "glow lab", "list_title", "list", "")])
    assert not bool(label_mentions(m, Labeler([]), sectors).iloc[0]["confirmed"])
    r = label_mentions(m, Labeler([], {}, {("entity", "glow lab"): ("small_local", LLM)}), sectors).iloc[0]
    assert bool(r["confirmed"]) and (r["category"], r["label_source"]) == ("small_local", LLM)
    rules = [rule("name", "Glow Lab", "big_brand")]
    r = label_mentions(m, Labeler(rules, {}, {("entity", "glow lab"): ("small_local", LLM)}), sectors).iloc[0]
    assert (r["category"], r["label_source"]) == ("big_brand", "name_rule")
    r = label_mentions(m, Labeler(rules, {("entity", "glow lab"): "not_a_brand"},
                                  {("entity", "glow lab"): ("small_local", LLM)}), sectors).iloc[0]
    assert (r["category"], r["label_source"]) == ("not_a_brand", "override")


def test_build_frames_with_llm_layer_shrinks_review(tmp_path):
    from brand_study.llm_evidence import build_items
    root = tmp_path / "raw"
    query = q()
    write_wrap(root, "run1", query, "search", fx("search_inline_aio.json"))
    rules = load_rules()
    base = analyze.build_frames([query], ["run1"], Labeler(rules), root, ledger_rows=[])["labels_to_review"]
    assert {"indiecafe.com", "buddybrew.com"} <= set(base["key"])
    fr = analyze.build_frames([query], ["run1"], Labeler(rules, {}, {("domain", "indiecafe.com"): ("small_local", LLM)}),
                              root, ledger_rows=[])
    assert "indiecafe.com" not in set(fr["labels_to_review"]["key"])
    assert len(fr["labels_to_review"]) == len(base) - 1
    # labelling a local-pack website also resolves the confirmed mention matched to that place
    fr = analyze.build_frames([query], ["run1"], Labeler(rules, {}, {("domain", "buddybrew.com"): ("small_local", LLM)}),
                              root, ledger_rows=[])
    m = fr["aio_mentions"].set_index("entity_norm")
    assert m.loc["buddy brew coffee", "label_source"] == f"local_pack_website:{LLM}"
    assert not ({"buddybrew.com", "buddy brew coffee"} & set(fr["labels_to_review"]["key"]))
    items, _ = build_items([query], ["run1"], Labeler(rules), root)
    assert sorted(i["key"] for i in items) == sorted(base["key"])


def test_load_llm_overrides(tmp_path):
    p = tmp_path / "llm.csv"
    assert load_llm_overrides(p) == ({}, {})
    p.write_text("# GENERATED\n# runs=run1,run2\n# prompt_version=v1-abc\nkey_type,key,category,note\n"
                 "domain,www.Indie.com,small_local,llm_unanimous; a=small_local(0.9)\n"
                 "entity,Illy,big_brand,llm_unanimous_confirm; x\nentity,Foo,big_brand,llm_majority; y\n")
    ov, meta = load_llm_overrides(p)
    assert ov == {("domain", "indie.com"): ("small_local", "llm_unanimous"),
                  ("entity", "illy"): ("big_brand", "llm_unanimous"), ("entity", "foo"): ("big_brand", "llm_majority")}
    assert meta["runs"] == "run1,run2" and meta["prompt_version"] == "v1-abc"
    p.write_text("key_type,key,category,note\ndomain,a.com,unknown,llm_unanimous\n")
    with pytest.raises(ConfigError, match="invalid category"):
        load_llm_overrides(p)
    p.write_text("key_type,key,category,note\ndomain,a.com,small_local,human note\n")
    with pytest.raises(ConfigError, match="llm_<policy>"):
        load_llm_overrides(p)
