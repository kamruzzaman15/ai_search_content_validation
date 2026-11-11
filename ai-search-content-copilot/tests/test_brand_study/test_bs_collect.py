from __future__ import annotations
import copy
import json
import random
from datetime import timedelta

import pytest

from bs_helpers import FAKE_KEY, FakeSession, aio_key, fx, q, search_key, write_wrap
from brand_study import budget as bud
from brand_study import cache, collect, config, extract
from brand_study import serpapi_client as sc
from brand_study.config import ConfigError
from brand_study.leaks import scan_for_key

Q1 = q()  # cof_l_01, inline AIO
Q14 = q("cof_l_14", "coffee shop near me", city="Tampa")
Q15 = q("cof_l_15", "coffee shop near me", city="Miami")
QB = q("skn_b_01", "best moisturizer for oily skin", "skincare", "broad")


def run(plan, session, **kw):
    opts = dict(dry_run=False, cap=300, max_new=None, followup_rate=1.0, retries=1, retry_headroom=10, sleep_s=0,
                yes=True, no_account=False, session=session, log=kw.pop("log", lambda s: None),
                sleep_fn=lambda s: None)
    opts.update(kw)
    return collect.run_collection(plan, FAKE_KEY, **opts)


def token_body(token):
    body = fx("search_page_token_only.json")
    body["ai_overview"]["page_token"] = token
    return body


def three_query_session(followup=None):
    return FakeSession({
        search_key(Q1): [(200, fx("search_inline_aio.json"))],
        search_key(Q14): [(200, fx("search_page_token_only.json"))],
        search_key(Q15): [(200, fx("search_no_aio.json"))],
        aio_key(): followup or [(200, fx("ai_overview_followup.json"))],
    })


def ledger():
    return bud.read_ledger()


def test_end_to_end_and_never_rerequest():
    s = three_query_session()
    plan = collect.build_plan([Q1, Q14, Q15], "run1")
    summary = run(plan, s)
    engines = [c["engine"] for c in s.paid_calls()]
    assert engines == ["google", "google", "google_ai_overview", "google"]  # follow-up immediately after its search
    assert len(ledger()) == 4 and summary.attempts == {"search": 3, "ai_overview": 1}
    for qq in (Q1, Q14, Q15):
        assert cache.cache_path("run1", qq.id, "search").exists()
    assert cache.read_cached(cache.cache_path("run1", Q14.id, "ai_overview"))["_meta"]["outcome"] == "ok"
    assert summary.aio_states == {"present_inline": 1, "present_followup": 1, "absent": 1}
    assert summary.account_before == 50

    s2 = three_query_session()
    plan2 = collect.build_plan([Q1, Q14, Q15], "run1")
    assert [p.action for p in plan2] == ["skip"] * 3
    run(plan2, s2)
    assert s2.paid_calls() == [] and len(ledger()) == 4
    assert scan_for_key([config.output_dir()], FAKE_KEY)[0] == []


def test_ledger_has_one_line_per_attempt_and_retry_success():
    s = FakeSession({search_key(QB): [(500, {"error": "x"}), (200, fx("search_no_aio.json"))]})
    run(collect.build_plan([QB], "run1"), s)
    rows = ledger()
    assert len(rows) == 2 and rows[0]["request_key"] == rows[1]["request_key"] == "run1/skn_b_01/search/1"
    assert [r["attempt"] for r in rows] == [1, 2] and rows[1]["reason"] == "retry"
    assert rows[0]["charged_assumed"] is False and rows[1]["charged_assumed"] is True
    assert cache.read_cached(cache.cache_path("run1", QB.id, "search"))["_meta"]["attempts"] == 2


def test_failed_main_search_not_cached():
    s = FakeSession({search_key(QB): [(500, {}), (503, {})]})
    summary = run(collect.build_plan([QB], "run1"), s)
    assert not cache.cache_path("run1", QB.id, "search").exists()
    assert len(ledger()) == 2 and summary.failures == [("skn_b_01", "search", "http_5xx")]
    assert collect.build_plan([QB], "run1")[0].action == "search"  # can be retried later


def test_no_results_cached_without_followup():
    s = FakeSession({search_key(QB): [(200, fx("search_no_results.json"))]})
    run(collect.build_plan([QB], "run1"), s)
    assert cache.read_cached(cache.cache_path("run1", QB.id, "search"))["_meta"]["outcome"] == "no_results"
    assert len(s.paid_calls()) == 1


@pytest.mark.parametrize("script,kind", [
    ([(200, fx("ai_overview_followup_error.json"))], "api_error"),
    ([(200, {})], "empty"),
    (["transport", "transport"], "transport"),
    ([(500, {}), (500, {})], "http_5xx"),
])
def test_followup_failure_cached(script, kind):
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))], aio_key(): script})
    run(collect.build_plan([QB], "run1"), s)
    w = cache.read_cached(cache.cache_path("run1", QB.id, "ai_overview"))
    assert w["_meta"]["outcome"] == "error" and w["_meta"]["error_kind"] == kind
    assert w["_meta"]["params_without_key"]["page_token"] == "<redacted-token>"
    assert collect.cached_aio_state("run1", QB.id) == "followup_failed"
    assert collect.build_plan([QB], "run1")[0].action == "skip"


def test_main_5xx_with_two_left_stops_without_retry():
    s = FakeSession({search_key(QB): [(500, {}), (200, fx("search_no_aio.json"))],
                     search_key(Q1): [(200, fx("search_no_aio.json"))]})
    summary = run(collect.build_plan([QB, Q1], "run1"), s, max_new=2)
    assert len(ledger()) == 1 and len(s.paid_calls()) == 1
    assert not cache.cache_path("run1", QB.id, "search").exists()
    assert summary.stopped_reason == "budget exhausted" and summary.items_left == 2


def test_followup_retry_without_budget_is_cached_as_budget_exhausted():
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))], aio_key(): [(500, {})]})
    run(collect.build_plan([QB], "run1"), s, max_new=2)
    w = cache.read_cached(cache.cache_path("run1", QB.id, "ai_overview"))
    assert w["_meta"]["error_kind"] == "budget_exhausted" and len(ledger()) == 2


def test_max_new_reserve_stop():
    qs = [Q1, Q14, QB]
    s = FakeSession({search_key(x): [(200, fx("search_no_aio.json"))] for x in qs})
    summary = run(collect.build_plan(qs, "run1"), s, max_new=3)
    assert len(ledger()) == 2 and summary.items_left == 1  # third needs 2 (itself + reserved follow-up)


def test_property_ledger_never_exceeds_allowed(monkeypatch):
    rng = random.Random(1234)
    outcomes = [(500, {}), "transport", (429, {"error": "slow down"}), "ok"]
    hit_limit = 0
    for trial in range(60):
        out = config.output_dir() / f"trial{trial}"
        qs = [q(f"cof_b_{i:02d}", f"query {trial} {i}", qtype="broad") for i in range(rng.randint(1, 5))]
        scripts = {}
        for i, qq in enumerate(qs):
            tok = f"TOK{trial}_{i}"
            seq = []
            for _ in range(rng.randint(1, 3)):
                o = rng.choice(outcomes)
                seq.append((200, token_body(tok) if rng.random() < 0.6 else fx("search_no_aio.json")) if o == "ok"
                           else o)
            scripts[search_key(qq)] = seq
            scripts[("google_ai_overview", tok)] = [rng.choice(outcomes[:3] + [(200, fx("ai_overview_followup.json"))])
                                                    for _ in range(rng.randint(1, 3))]
        spent = rng.randint(0, 20)
        cap = spent + rng.randint(0, 40)
        max_new = rng.choice([None, 0, 1, 2, 3, 5, 8])
        s = FakeSession(scripts, account={"total_searches_left": rng.randint(0, 40), "this_month_usage": 1})
        allowed = min(cap - spent, s.account["total_searches_left"], max_new if max_new is not None else 10 ** 9)
        monkeypatch.setenv("BRAND_STUDY_OUTPUT_DIR", str(out))
        for _ in range(spent):
            bud.append_ledger({"kind": "search", "run": "old"})
        try:
            run(collect.build_plan(qs, "run1"), s, cap=cap, max_new=max_new, retries=rng.randint(0, 2),
                retry_headroom=0, followup_rate=0.0)
        except bud.BudgetError:
            pass
        new_lines = len(ledger()) - spent
        assert new_lines == len(s.paid_calls())
        assert new_lines <= max(0, allowed), (trial, new_lines, allowed)
        hit_limit += new_lines == allowed > 0
    assert hit_limit >= 5  # the limit was actually reached in several trials


def test_budget_refusals():
    for _ in range(295):
        bud.append_ledger({"kind": "search", "run": "smoke"})
    with pytest.raises(bud.BudgetError, match="exceeds study cap"):
        run(collect.build_plan([Q1], "run1"), FakeSession())
    with pytest.raises(bud.BudgetError, match="account searches left"):
        run(collect.build_plan([Q1], "run1"), FakeSession(account={"total_searches_left": 3}), cap=1000)


def test_account_unavailable_dry_run_ok_real_refuses():
    lines = []
    s = FakeSession(account="fail")
    run(collect.build_plan([Q1], "run1"), s, dry_run=True, log=lines.append)
    assert any("account searches left: unknown" in x for x in lines) and s.paid_calls() == []
    with pytest.raises(bud.BudgetError, match="Account API unavailable"):
        run(collect.build_plan([Q1], "run1"), s)


def test_dry_run_output_and_no_paid_calls():
    for _ in range(7):
        bud.append_ledger({"kind": "search", "run": "smoke"})
    lines = []
    s = three_query_session()
    run(collect.build_plan([Q1, Q14, Q15], "run1"), s, dry_run=True, log=lines.append)
    text = "\n".join(lines)
    assert s.paid_calls() == [] and len(s.calls) == 1  # only the free Account API
    assert "ledger_total (study so far): 7" in text and "study cap: 300" in text
    assert "DRY RUN: would spend <= 16 searches (study total after: <= 23 of cap 300)" in text
    assert not config.lock_path().exists() and not cache.cache_path("run1", Q1.id, "search").exists()


def test_non_tty_without_yes_refuses():
    with pytest.raises(bud.BudgetError, match="not a terminal"):
        run(collect.build_plan([Q1], "run1"), three_query_session(), yes=False, isatty=lambda: False)
    with pytest.raises(bud.BudgetError, match="cancelled"):
        run(collect.build_plan([Q1], "run1"), three_query_session(), yes=False, isatty=lambda: True,
            input_fn=lambda prompt: "n")


def test_lock_refuses_second_collect():
    config.lock_path().parent.mkdir(parents=True, exist_ok=True)
    config.lock_path().write_text("{}")
    with pytest.raises(cache.LockHeld):
        run(collect.build_plan([Q1], "run1"), three_query_session())


def test_ctrl_c_stops_cleanly_and_leaves_stale_token():
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))], aio_key(): ["interrupt"],
                     search_key(Q1): [(200, fx("search_no_aio.json"))]})
    summary = run(collect.build_plan([QB, Q1], "run1"), s)
    assert summary.stopped_reason == "interrupted (Ctrl+C)" and summary.items_left == 1
    assert not config.lock_path().exists()
    rows = ledger()
    assert len(rows) == 2 and rows[1]["error_kind"] == "interrupted"  # in-flight attempt still counted
    sw = cache.read_cached(cache.cache_path("run1", QB.id, "search"))
    assert extract.aio_status(sw, None) == ("followup_failed", "stale_token")


def test_auth_error_aborts_and_is_redacted():
    s = FakeSession({search_key(Q1): [(401, {"error": f"Invalid API key {FAKE_KEY}"})]})
    with pytest.raises(sc.AuthError) as e:
        run(collect.build_plan([Q1], "run1"), s)
    assert FAKE_KEY not in str(e.value) and len(ledger()) == 1
    assert FAKE_KEY not in config.ledger_path().read_text()


def test_quota_429_aborts():
    s = FakeSession({search_key(Q1): [(429, {"error": "Your account has run out of searches."})],
                     search_key(QB): [(200, fx("search_no_aio.json"))]})
    with pytest.raises(sc.QuotaError):
        run(collect.build_plan([Q1, QB], "run1"), s)
    assert len(s.paid_calls()) == 1


def test_run2_spacing():
    run(collect.build_plan([QB], "run1"), FakeSession({search_key(QB): [(200, fx("search_no_aio.json"))]}))
    s = FakeSession({search_key(QB): [(200, fx("search_no_aio.json"))]})
    with pytest.raises(bud.BudgetError, match="force-spacing"):
        run(collect.build_plan([QB], "run2"), s)
    newest = collect.newest_collected_at("run1")
    assert collect.spacing_problem("run2", 12, now=newest + timedelta(hours=13)) is None
    run(collect.build_plan([QB], "run2"), s, force_spacing=True)
    assert cache.cache_path("run2", QB.id, "search").exists()


def test_retry_failed_aio():
    with pytest.raises(ConfigError, match="--query-ids"):
        collect.build_plan([QB], "run1", retry_failed_aio=True)
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))],
                     aio_key(): [(200, fx("ai_overview_followup_error.json")), (200, fx("ai_overview_followup.json"))]})
    run(collect.build_plan([QB], "run1"), s, retries=0)
    old = cache.read_cached(cache.cache_path("run1", QB.id, "search"))["_meta"]["collected_at_utc"]
    assert collect.build_plan([QB], "run1", query_ids=[QB.id])[0].action == "skip"
    plan = collect.build_plan([QB], "run1", query_ids=[QB.id], retry_failed_aio=True)
    assert plan[0].action == "retry_search"
    assert "RETRY (re-requests cached search)" in collect.format_plan(plan)
    run(plan, s, retries=0)
    new = cache.read_cached(cache.cache_path("run1", QB.id, "search"))
    assert new["_meta"]["supersedes"] == old
    assert collect.cached_aio_state("run1", QB.id) == "present_followup"
    assert len(list((config.raw_dir() / "run1" / "_superseded").glob("*.json"))) == 2
    assert [r["reason"] for r in ledger()][-2:] == ["retry_failed_aio", "retry_failed_aio"]


def test_changed_query_text_refused():
    write_wrap(config.raw_dir(), "run1", QB, "search", fx("search_no_aio.json"))
    changed = q("skn_b_01", "best moisturiser for oily skin", "skincare", "broad")
    with pytest.raises(ConfigError, match="new id"):
        collect.build_plan([changed], "run1")


def test_build_plan_filters_and_validation():
    qs = [Q1, Q14, QB]
    assert [p.query.id for p in collect.build_plan(qs, "run1", sector="skincare")] == ["skn_b_01"]
    assert len(collect.build_plan(qs, "run1", limit=2)) == 2
    with pytest.raises(ConfigError, match="unknown query ids"):
        collect.build_plan(qs, "run1", query_ids=["nope"])
    with pytest.raises(ConfigError, match="invalid run label"):
        collect.build_plan(qs, "Run 1")


def test_cache_files_never_contain_key():
    s = FakeSession({search_key(QB): [(200, dict(fx("search_no_aio.json"),
                                                 search_metadata={"id": "x", "json_endpoint":
                                                                  f"https://serpapi.com?api_key={FAKE_KEY}"}))]})
    run(collect.build_plan([QB], "run1"), s)
    text = cache.cache_path("run1", QB.id, "search").read_text()
    assert FAKE_KEY not in text and "***" in text
    assert FAKE_KEY not in json.dumps(ledger())


def test_dry_run_respects_max_new_and_no_retries():
    lines = []
    run(collect.build_plan([Q1, Q14, Q15], "run1"), three_query_session(), dry_run=True, log=lines.append,
        retries=0, max_new=4)
    text = "\n".join(lines)
    assert "retry headroom 0 = 6" in text and "DRY RUN: would spend <= 4 searches" in text


def test_max_new_lets_late_run_pass_the_guard():
    for _ in range(200):
        bud.append_ledger({"kind": "search", "run": "run1"})
    lines = []
    s = FakeSession({search_key(QB): [(200, fx("search_no_aio.json"))]})
    plan = collect.build_plan([QB] + [q(f"skn_b_{i:02d}", f"query {i}", "skincare", "broad") for i in range(2, 60)],
                              "run2")
    run(plan, s, dry_run=True, max_new=20, log=lines.append)
    assert not any("WOULD REFUSE" in x for x in lines)
    with pytest.raises(bud.BudgetError, match="exceeds study cap"):
        run(plan, s, dry_run=False, max_new=None, force_spacing=True)


def test_quota_message_on_http_200_aborts():
    s = FakeSession({search_key(Q1): [(200, {"error": "Your account has run out of searches."})],
                     search_key(QB): [(200, fx("search_no_aio.json"))]})
    with pytest.raises(sc.QuotaError):
        run(collect.build_plan([Q1, QB], "run1"), s)
    assert len(s.paid_calls()) == 1 and not cache.cache_path("run1", Q1.id, "search").exists()


def test_followup_quota_aborts_after_caching():
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))],
                     aio_key(): [(200, {"error": "Your account has run out of searches."})],
                     search_key(Q1): [(200, fx("search_no_aio.json"))]})
    with pytest.raises(sc.QuotaError):
        run(collect.build_plan([QB, Q1], "run1"), s)
    w = cache.read_cached(cache.cache_path("run1", QB.id, "ai_overview"))
    assert w["_meta"]["error_kind"] == "quota" and len(s.paid_calls()) == 2


def test_followup_auth_on_retry_records_attempts():
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))],
                     aio_key(): [(500, {}), (401, {"error": "Invalid API key."})]})
    with pytest.raises(sc.AuthError):
        run(collect.build_plan([QB], "run1"), s)
    w = cache.read_cached(cache.cache_path("run1", QB.id, "ai_overview"))
    assert w["_meta"]["error_kind"] == "auth" and w["_meta"]["attempts"] == 2


def test_items_left_counts_recached_retry_search():
    s = FakeSession({search_key(QB): [(200, fx("search_page_token_only.json"))], aio_key(): [(500, {})]})
    run(collect.build_plan([QB], "run1"), s, retries=0)
    plan = collect.build_plan([QB], "run1", query_ids=[QB.id], retry_failed_aio=True)
    summary = run(plan, s, max_new=2)  # main + first follow-up attempt; the follow-up retry has no budget
    assert summary.stopped_reason == "budget exhausted" and summary.items_left == 0
