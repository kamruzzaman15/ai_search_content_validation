from __future__ import annotations

import pytest

from brand_study import budget as bud


class Item:
    def __init__(self, action):
        self.action = action


def test_estimate():
    plan = [Item("search")] * 50 + [Item("skip")] * 3
    e = bud.estimate(plan, 1.0, 10)
    assert (e.main, e.followup_max, e.retry_headroom, e.total) == (50, 50, 10, 110)
    assert bud.estimate([Item("skip")], 1.0, 10).total == 0
    assert bud.estimate(plan, 0.33, 0).followup_max == 17


def test_check_budget_cap_includes_ledger():
    e = bud.Estimate(50, 50, 10, 110)
    acct = bud.AccountInfo(1000, 0, {})
    bud.check_budget(e, 300, 190, acct)
    with pytest.raises(bud.BudgetError, match="exceeds study cap"):
        bud.check_budget(e, 300, 191, acct)


def test_check_budget_account():
    e = bud.Estimate(50, 50, 10, 110)
    with pytest.raises(bud.BudgetError, match="account searches left"):
        bud.check_budget(e, 300, 0, bud.AccountInfo(100, 0, {}))
    with pytest.raises(bud.BudgetError, match="Account API unavailable"):
        bud.check_budget(e, 300, 0, None)
    assert bud.check_budget(e, 300, 0, None, dry_run=True) == ["account searches left: unknown"]


def test_runtime_budget():
    b = bud.RuntimeBudget.from_limits(cap=300, spent=290, account_left=50, max_new=None)
    assert b.allowed == 10
    b = bud.RuntimeBudget.from_limits(cap=300, spent=0, account_left=50, max_new=4)
    assert b.allowed == 4
    assert b.can_spend(2)
    for _ in range(4):
        b.charge()
    assert not b.can_spend(1)
    with pytest.raises(bud.BudgetError):
        b.charge()


def test_ledger_roundtrip(tmp_path):
    p = tmp_path / "ledger.jsonl"
    bud.append_ledger({"kind": "search"}, p)
    bud.append_ledger({"kind": "ai_overview"}, p)
    rows = bud.read_ledger(p)
    assert bud.ledger_total(rows) == 2 and "ts" in rows[0]


def test_check_budget_clamped_to_max_new():
    e = bud.Estimate(50, 50, 10, 110)
    bud.check_budget(e, 300, 200, bud.AccountInfo(1000, 0, {}), max_new=20)  # 200 + 20 <= 300
    bud.check_budget(e, 300, 0, bud.AccountInfo(30, 0, {}), max_new=20)  # 20 <= 30 left
    with pytest.raises(bud.BudgetError, match="exceeds study cap"):
        bud.check_budget(e, 300, 290, bud.AccountInfo(1000, 0, {}), max_new=20)
