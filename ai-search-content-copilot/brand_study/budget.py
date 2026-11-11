from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

from brand_study import config
from brand_study.cache import utc_now


class BudgetError(Exception):
    """Refusal to spend searches (maps to exit code 2)."""


@dataclass
class Estimate:
    main: int
    followup_max: int
    retry_headroom: int
    total: int


@dataclass
class AccountInfo:
    total_searches_left: int | None
    this_month_usage: int | None
    raw: dict

    @classmethod
    def from_fields(cls, fields: dict | None) -> "AccountInfo | None":
        if fields is None:
            return None

        def _int(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        return cls(_int(fields.get("total_searches_left")), _int(fields.get("this_month_usage")), dict(fields))


# --- Ledger --------------------------------------------------------------------

def read_ledger(path: Path | None = None) -> list[dict]:
    path = path or config.ledger_path()
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ledger_total(rows: list[dict]) -> int:
    """Every ledger line is one HTTP attempt and is counted as spent."""
    return len(rows)


def append_ledger(entry: dict, path: Path | None = None) -> None:
    path = path or config.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": utc_now(), **entry}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# --- Estimate and guard ---------------------------------------------------------

def estimate(plan, followup_rate: float = config.DEFAULT_FOLLOWUP_RATE,
             retry_headroom: int = config.DEFAULT_RETRY_HEADROOM) -> Estimate:
    main = sum(1 for item in plan if item.action in ("search", "retry_search"))
    followup = int(math.ceil(main * max(0.0, followup_rate)))
    headroom = retry_headroom if main else 0
    return Estimate(main, followup, headroom, main + followup + headroom)


def check_budget(est: Estimate, cap: int, spent: int, account: AccountInfo | None, dry_run: bool = False,
                 max_new: int | None = None) -> list[str]:
    """Raise BudgetError on refusal; return informational notes otherwise.

    The estimate is clamped to max_new: RuntimeBudget never lets an invocation spend more than that."""
    notes = []
    total = est.total if max_new is None else min(est.total, max(0, max_new))
    if spent + total > cap:
        raise BudgetError(
            f"estimate {total} + study so far {spent} = {spent + total} exceeds study cap {cap}. "
            f"Use --limit/--sector/--query-ids, lower --retry-headroom, or raise --study-cap."
        )
    if account is None or account.total_searches_left is None:
        if not dry_run and est.main:
            raise BudgetError("Account API unavailable: cannot confirm remaining searches; refusing to collect.")
        notes.append("account searches left: unknown")
    elif total > account.total_searches_left:
        raise BudgetError(
            f"estimate {total} exceeds account searches left ({account.total_searches_left})."
        )
    return notes


class RuntimeBudget:
    """Charged once per HTTP attempt, before it is sent. Never lets charged exceed allowed."""

    def __init__(self, allowed: int):
        self.allowed = max(0, int(allowed))
        self.charged = 0

    @classmethod
    def from_limits(cls, cap: int, spent: int, account_left: int | None, max_new: int | None) -> "RuntimeBudget":
        limits = [cap - spent]
        if account_left is not None:
            limits.append(account_left)
        if max_new is not None:
            limits.append(max_new)
        return cls(min(limits))

    @property
    def remaining(self) -> int:
        return self.allowed - self.charged

    def can_spend(self, n_needed: int = 1) -> bool:
        return self.remaining >= n_needed

    def charge(self) -> None:
        if self.remaining < 1:
            raise BudgetError("runtime budget exhausted")
        self.charged += 1
