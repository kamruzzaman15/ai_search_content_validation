from __future__ import annotations

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
