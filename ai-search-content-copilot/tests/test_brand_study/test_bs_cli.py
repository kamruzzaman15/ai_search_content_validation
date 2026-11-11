from __future__ import annotations

import pytest

from bs_helpers import FAKE_KEY, FakeSession, fx, search_key
from brand_study import cli, config
from brand_study.queries import load_queries


@pytest.fixture
def q01():
    return next(x for x in load_queries(warn=False) if x.id == "cof_l_01")


def call(argv, session=None, isatty=False):
    lines = []
    code = cli.main(argv, load_env=False, session_factory=(lambda: session) if session else None,
                    out=lines.append, isatty=lambda: isatty, sleep_fn=lambda s: None)
    return code, "\n".join(lines)


def test_check_queries():
    code, text = call(["check-queries"])
    assert code == 0 and "50 queries OK" in text and "coffee=25" in text and "skincare=25" in text


def test_account_prints_whitelisted_fields_only():
    code, text = call(["account"], FakeSession())
    assert code == 0 and "total_searches_left: 200" in text
    assert FAKE_KEY not in text and "someone@example.com" not in text


def test_dry_run_spends_nothing(q01):
    s = FakeSession()
    code, text = call(["collect", "--run", "run1", "--dry-run"], s)
    assert code == 0 and s.paid_calls() == []
    assert "ledger_total (study so far): 0" in text
    assert "DRY RUN: would spend <= 110 searches" in text


def test_dry_run_without_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY")
    code, text = call(["collect", "--run", "run1", "--dry-run", "--no-account"])
    assert code == 0 and "account searches left: unknown" in text


def test_missing_key_exit_3(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY")
    code, text = call(["collect", "--run", "run1", "--query-ids", "cof_l_01", "--yes"], FakeSession())
    assert code == 3 and "SERPAPI_API_KEY not set" in text


def test_non_tty_without_yes_exit_2(q01):
    s = FakeSession({search_key(q01): [(200, fx("search_no_aio.json"))]})
    code, text = call(["collect", "--run", "run1", "--query-ids", "cof_l_01"], s)
    assert code == 2 and "not a terminal" in text and s.paid_calls() == []


def test_study_cap_exit_2():
    code, text = call(["collect", "--run", "run1", "--study-cap", "50"], FakeSession())
    assert code == 2 and "exceeds study cap 50" in text


def test_collect_and_analyze(q01):
    s = FakeSession({search_key(q01): [(200, fx("search_inline_aio.json"))]})
    code, text = call(["collect", "--run", "smoke", "--query-ids", "cof_l_01", "--max-new", "4", "--retries", "0",
                       "--yes"], s)
    assert code == 0 and len(s.paid_calls()) == 1 and FAKE_KEY not in text
    code, text = call(["analyze", "--runs", "smoke"])
    assert code == 0 and "PROVISIONAL" in text
    report = (config.results_dir() / "report.md").read_text(encoding="utf-8")
    assert "observational, not causal" in report
    code, text = call(["scan-leaks"])
    assert code == 0 and text.startswith("clean (")


def test_auth_error_exit_3_redacted(q01):
    s = FakeSession({search_key(q01): [(401, {"error": f"Invalid API key {FAKE_KEY}"})]})
    code, text = call(["collect", "--run", "run1", "--query-ids", "cof_l_01", "--yes"], s)
    assert code == 3 and FAKE_KEY not in text and "401" in text


def test_scan_leaks_reports_names_only(capsys):
    out = config.output_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaky.json").write_text(f'{{"k": "{FAKE_KEY}"}}')
    code = cli.main(["scan-leaks"], load_env=False)
    captured = capsys.readouterr()
    assert code == 4 and "LEAK in 1 files:" in captured.out and "leaky.json" in captured.out
    assert FAKE_KEY not in captured.out + captured.err


def test_scan_leaks_refuses_short_key(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "short")
    code, text = call(["scan-leaks"])
    assert code == 3 and "refusing" in text


def test_no_account_requires_dry_run():
    code, text = call(["collect", "--run", "run1", "--no-account"])
    assert code == 3


def test_study_cap_help_text(capsys):
    with pytest.raises(SystemExit):
        cli.main(["collect", "--help"], load_env=False)
    help_text = " ".join(capsys.readouterr().out.split())
    assert "across the WHOLE study" in help_text and "Not per invocation" in help_text
    assert "THIS invocation" in help_text
