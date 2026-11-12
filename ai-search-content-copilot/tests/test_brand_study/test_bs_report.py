from __future__ import annotations

import re

import pandas as pd

from bs_helpers import fx, q, write_wrap
from brand_study import analyze
from brand_study.labeling import Labeler, load_rules
from brand_study.report import build_report, headline_provisional, md_table, write_report


def _report(tmp_path, fixture, query, overrides=None, threshold=0.10):
    root = tmp_path / "raw"
    write_wrap(root, "run1", query, "search", fx(fixture))
    labeler = Labeler(load_rules(), overrides or {})
    fr = analyze.run_analysis([query], ["run1", "run2"], labeler, root=root, out_dir=tmp_path / "res",
                              threshold=threshold, ledger_rows=[{"kind": "search", "reason": "normal"}])
    return fr, build_report(fr, ["run1", "run2"], ledger_rows=[{"kind": "search", "reason": "normal"}],
                            threshold=threshold)


def test_md_table():
    t = md_table(pd.DataFrame({"a": [1.0, float("nan")], "b": ["x|y", True]}))
    assert t.splitlines()[0] == "| a | b |"
    assert "| 1.00 | x\\|y |" in t and "| n/a | yes |" in t
    assert md_table(pd.DataFrame()) == "_(no rows)_"


def test_report_provisional_banner_and_sections(tmp_path):
    fr, text = _report(tmp_path, "search_inline_aio.json", q())
    assert headline_provisional(fr["summary_by_segment"])
    assert text.startswith("> **PROVISIONAL:")
    assert "observational, not causal" in text
    assert "Operational test" in text and "**small_local**" in text
    assert "undercount small businesses" in text
    assert "run2 not collected yet" in text
    assert "SerpApi searches used (ledger attempts, whole study): 1" in text
    for section in ("## AI Overview rate", "## Big vs small/local visibility", "## Overlap", "## Local pack",
                    "## Labeling coverage", "## Limitations", "## Method notes"):
        assert section in text
    path = write_report(text, tmp_path / "res")
    assert path.read_text(encoding="utf-8") == text


def test_report_without_banner_when_resolved(tmp_path):
    query = q("cof_b_01", "best coffee beans for espresso", "coffee", "broad")
    fr, text = _report(tmp_path, "search_no_aio.json", query, {("domain", "smallroaster.com"): "small_local"})
    assert not headline_provisional(fr["summary_by_segment"])
    assert "PROVISIONAL" not in text.split("## Big vs small")[0]
    assert "observational, not causal" in text


def test_label_origin_grouping():
    from brand_study.report import label_origin
    assert label_origin("override") == "human" and label_origin("local_pack_website:override") == "human"
    assert label_origin("llm_unanimous") == "llm" and label_origin("llm_majority") == "llm"
    assert label_origin("local_pack_website:llm_unanimous") == "llm"
    assert label_origin("llm_unanimous_confirm") == "llm_confirm"
    for s in ("domain_rule", "parent_domain_rule", "name_rule", "platform_hosted", "tld",
              "local_pack_website:domain_rule"):
        assert label_origin(s) == "rules"
    assert label_origin("none") == "unlabeled" and label_origin("unconfirmed") == "unlabeled"


def test_llm_share_sentence():
    from collections import Counter
    from brand_study.report import llm_share_sentence
    t = llm_share_sentence(Counter({"rules": 6, "llm": 3, "llm_confirm": 1, "unlabeled": 10}),
                           {"policy": "unanimous", "prompt_version": "v1-abc"})
    assert t.startswith("40.0% of labeled rows (20.0% of all rows) came from the LLM judges: 30.0% labelled by the "
                        "LLM and 10.0% rule categories confirmed by the LLM (policy: unanimous, prompt v1-abc).")


def test_report_llm_disclaimer_and_coverage(tmp_path):
    root = tmp_path / "raw"
    query = q()
    write_wrap(root, "run1", query, "search", fx("search_inline_aio.json"))
    rules = load_rules()
    plain = analyze.run_analysis([query], ["run1"], Labeler(rules), root=root, out_dir=tmp_path / "a", ledger_rows=[])
    text0 = build_report(plain, ["run1"], ledger_rows=[])
    assert "**LLM labels:" not in text0 and "LLM labels loaded (`llm_label_overrides.csv`): 0" in text0
    assert "0.0% of labeled rows" in text0 and "| total |" in text0
    llm = {("domain", "indiecafe.com"): ("small_local", "llm_unanimous")}
    fr = analyze.run_analysis([query], ["run1"], Labeler(rules, {}, llm), root=root, out_dir=tmp_path / "b",
                              ledger_rows=[])
    text = build_report(fr, ["run1"], ledger_rows=[], llm_overrides_count=1,
                        llm_meta={"policy": "unanimous", "prompt_version": "v1-x",
                                  "validation_summary": "accepted-label accuracy 0.90 (n=18, policy unanimous); "
                                                        "big-brand bias (accepted LLM labels): no evidence of "
                                                        "big-brand bias",
                                  "runs_note": "the LLM labels were made from runs run1 but runs run1,run2 are being "
                                               "analysed"})
    head = text.split("## Category definitions")[0]
    assert "**LLM labels: 1 LLM labels" in head and "big-brand bias" in head
    m = re.search(r"decided the label of (\d+) rows", head)
    assert m and int(m.group(1)) > 0
    assert "accepted-label accuracy 0.90" in head and "made from runs run1" in head
    assert "came from the LLM judges" in text and "WARNING: the LLM labels were made" in text
    assert "small blinded human sample" in text
    # Review round 3: a loaded LLM label that a human override now beats is reported as loaded, not as applied.
    fr2 = analyze.run_analysis([query], ["run1"], Labeler(rules, {("domain", "indiecafe.com"): "small_local"}, llm),
                               root=root, out_dir=tmp_path / "c", ledger_rows=[])
    head2 = build_report(fr2, ["run1"], ledger_rows=[], llm_overrides_count=1).split("## Category definitions")[0]
    assert "**LLM labels: 1 LLM labels" in head2 and "decided the label of 0 rows" in head2
