from __future__ import annotations
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from brand_study import budget as bud
from brand_study import cache, config, serpapi_client as sc
from brand_study.config import ConfigError
from brand_study.queries import Query

BACKOFF_S = (2.0, 5.0)
NO_RESULTS_MARKERS = ("hasn't returned any results", "has not returned any results", "no results")


@dataclass
class PlanItem:
    run: str
    query: Query
    main_cached: bool
    aio_cached: bool
    aio_state: str | None
    action: str  # skip | search | retry_search


@dataclass
class Summary:
    requests: Counter = field(default_factory=Counter)   # logical requests by kind
    attempts: Counter = field(default_factory=Counter)   # HTTP attempts by kind
    retries: int = 0
    aio_states: Counter = field(default_factory=Counter)
    failures: list = field(default_factory=list)
    stopped_reason: str | None = None
    items_left: int = 0
    account_before: int | None = None
    account_after: int | None = None


# --- Plan ----------------------------------------------------------------------

def _ai_overview_of(search_body: dict) -> dict | None:
    ai = (search_body or {}).get("ai_overview")
    return ai if isinstance(ai, dict) else None


def needs_followup(search_body: dict) -> str | None:
    ai = _ai_overview_of(search_body)
    if ai and ai.get("page_token") and not ai.get("text_blocks"):
        return ai["page_token"]
    return None


def cached_aio_state(run: str, qid: str, root: Path | None = None) -> str | None:
    main = cache.read_cached(cache.cache_path(run, qid, "search", root))
    if main is None:
        return None
    body = main.get("response") or {}
    ai = _ai_overview_of(body)
    if ai is None:
        return "absent"
    if ai.get("text_blocks"):
        return "present_inline"
    if ai.get("page_token"):
        aio = cache.read_cached(cache.cache_path(run, qid, "ai_overview", root))
        if aio and aio.get("_meta", {}).get("outcome") == "ok":
            return "present_followup"
        return "followup_failed"
    if ai.get("error"):
        return "error"
    return "absent"


def build_plan(queries: list[Query], run: str, *, sector: str | None = None, query_ids: list[str] | None = None,
               limit: int | None = None, retry_failed_aio: bool = False, root: Path | None = None) -> list[PlanItem]:
    if not re.match(config.RUN_LABEL_RE, run or ""):
        raise ConfigError(f"invalid run label {run!r} (expected {config.RUN_LABEL_RE})")
    if retry_failed_aio and not query_ids:
        raise ConfigError("--retry-failed-aio needs an explicit --query-ids list")
    selected = [q for q in queries if sector is None or q.sector == sector]
    if query_ids:
        known = {q.id for q in queries}
        unknown = [i for i in query_ids if i not in known]
        if unknown:
            raise ConfigError(f"unknown query ids: {unknown}")
        wanted = set(query_ids)
        selected = [q for q in selected if q.id in wanted]
    if limit is not None:
        selected = selected[:limit]

    plan = []
    for q in selected:
        main = cache.read_cached(cache.cache_path(run, q.id, "search", root))
        if main is not None:
            meta = main.get("_meta", {})
            if meta.get("query") != q.query or meta.get("location") != q.location:
                raise ConfigError(
                    f"{run}/{q.id}: cached query/location differs from the query file. "
                    f"Give the changed query a new id instead of reusing {q.id!r}."
                )
        aio_cached = cache.cache_path(run, q.id, "ai_overview", root).exists()
        state = cached_aio_state(run, q.id, root)
        if main is None:
            action = "search"
        elif retry_failed_aio and state == "followup_failed":
            action = "retry_search"
        else:
            action = "skip"
        plan.append(PlanItem(run, q, main is not None, aio_cached, state, action))
    return plan


def format_plan(plan: list[PlanItem]) -> str:
    lines = [f"{'query_id':<10} {'action':<36} {'location':<40} query"]
    for it in plan:
        label = {"skip": "skip (cached)", "search": "SEARCH",
                 "retry_search": "RETRY (re-requests cached search)"}[it.action]
        lines.append(f"{it.query.id:<10} {label:<36} {it.query.location:<40} {it.query.query}")
    return "\n".join(lines)


# --- Run spacing -----------------------------------------------------------------

def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def newest_collected_at(run: str, root: Path | None = None) -> datetime | None:
    newest = None
    for p in cache.iter_run_files(run, root):
        data = cache.read_cached(p) or {}
        ts = _parse_ts(data.get("_meta", {}).get("collected_at_utc"))
        if ts and (newest is None or ts > newest):
            newest = ts
    return newest


def spacing_problem(run: str, min_hours: float, now: datetime | None = None, root: Path | None = None) -> str | None:
    if run not in config.DEFAULT_RUNS or config.DEFAULT_RUNS.index(run) == 0:
        return None
    prev = config.DEFAULT_RUNS[config.DEFAULT_RUNS.index(run) - 1]
    newest = newest_collected_at(prev, root)
    if newest is None:
        return None
    now = now or datetime.now(timezone.utc)
    hours = (now - newest).total_seconds() / 3600
    if hours < min_hours:
        return (f"{run} would start {hours:.1f} h after the newest {prev} result "
                f"(minimum {min_hours:g} h). Wait, or pass --force-spacing.")
    return None


# --- Collection ------------------------------------------------------------------

def _classify_body(body: dict) -> tuple[str, str | None, str]:
    """For an HTTP-OK search body: (outcome, error_kind, message)."""
    err = body.get("error")
    status = (body.get("search_metadata") or {}).get("status")
    if isinstance(err, str) and err:
        if any(m in err.lower() for m in NO_RESULTS_MARKERS):
            return "no_results", None, err
        return "error", "api_error", err
    if status == "Error":
        return "error", "api_error", "search_metadata.status=Error"
    return "ok", None, ""


def _aio_payload(body: dict) -> dict | None:
    ai = body.get("ai_overview")
    if isinstance(ai, dict) and ai.get("text_blocks"):
        return ai
    if body.get("text_blocks"):
        return body
    return None


def _attempt_outcome(kind: str, resp) -> tuple[str, str | None]:
    if resp.error_kind is not None:
        return "error", resp.error_kind
    if kind == "search":
        outcome, ek, _ = _classify_body(resp.body)
        return outcome, ek
    if _aio_payload(resp.body):
        return "ok", None
    return "error", "api_error" if resp.body.get("error") else "empty"


class Collector:
    def __init__(self, api_key: str, *, budget: bud.RuntimeBudget, retries: int, sleep_s: float,
                 session=None, sleep_fn: Callable[[float], None] = time.sleep, log: Callable = print,
                 ledger_rows: list[dict] | None = None, root: Path | None = None):
        self.api_key = api_key
        self.budget = budget
        self.retries = max(0, retries)
        self.sleep_s = sleep_s
        self.session = session
        self.sleep_fn = sleep_fn
        self.log = log
        self.root = root
        self.summary = Summary()
        self._seq = Counter()
        for row in ledger_rows or []:
            parts = str(row.get("request_key", "")).split("/")
            if len(parts) == 4:
                key = "/".join(parts[:3])
                try:
                    self._seq[key] = max(self._seq[key], int(parts[3]))
                except ValueError:
                    pass
        self._sent_any = False
        self.main_cached = False

    def _next_request_key(self, run: str, qid: str, kind: str) -> str:
        base = f"{run}/{qid}/{kind}"
        self._seq[base] += 1
        return f"{base}/{self._seq[base]}"

    def _pace(self):
        if self._sent_any and self.sleep_s > 0:
            self.sleep_fn(self.sleep_s)
        self._sent_any = True

    def request_with_retries(self, kind: str, params: dict, item: PlanItem, reason: str):
        """Returns (Response | None, attempts, budget_stopped)."""
        url = config.SERPAPI_ENDPOINT
        need = 2 if kind == "search" else 1
        request_key = self._next_request_key(item.run, item.query.id, kind)
        self.summary.requests[kind] += 1
        resp = None
        attempts = 0
        for attempt in range(1, self.retries + 2):
            if not self.budget.can_spend(need):
                return resp, attempts, True
            self.budget.charge()
            attempts += 1
            self.summary.attempts[kind] += 1
            if attempt > 1:
                self.summary.retries += 1
            self._pace()
            entry = {"run": item.run, "query_id": item.query.id, "kind": kind, "request_key": request_key,
                     "attempt": attempt, "reason": reason if attempt == 1 else "retry"}
            auth_exc = None
            try:
                resp = sc.get_json_once(url, params, self.api_key, session=self.session)
            except sc.AuthError as exc:
                auth_exc = exc
                resp = sc.Response(None, {}, "auth", False, sc.redact(str(exc), self.api_key))
            except KeyboardInterrupt:
                # The request may have reached SerpApi: record it so the study total stays worst-case.
                bud.append_ledger({**entry, "http_status": None, "outcome": "error", "error_kind": "interrupted",
                                   "charged_assumed": True})
                raise
            outcome, ek = _attempt_outcome(kind, resp)
            charged = resp.status == 200 and outcome in ("ok", "no_results")
            bud.append_ledger({**entry, "http_status": resp.status, "outcome": outcome, "error_kind": ek,
                               "charged_assumed": charged})
            if auth_exc is not None:
                auth_exc.attempts = attempts
                raise auth_exc
            if not resp.retryable:
                break
            if attempt <= self.retries:
                self.sleep_fn(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        return resp, attempts, False

    def _meta(self, item: PlanItem, kind: str, params: dict, resp, attempts: int, outcome: str,
              error_kind: str | None, message: str, supersedes: str | None) -> dict:
        body = resp.body if resp is not None else {}
        return {
            "run": item.run, "query_id": item.query.id, "kind": kind, "query": item.query.query,
            "location": item.query.location, "params_without_key": params, "collected_at_utc": cache.utc_now(),
            "http_status": resp.status if resp is not None else None,
            "serpapi_search_id": (body.get("search_metadata") or {}).get("id"),
            "attempts": attempts, "outcome": outcome, "error_kind": error_kind,
            "error_message_redacted": sc.redact(message, self.api_key) or None, "supersedes": supersedes,
        }

    def collect_item(self, item: PlanItem) -> str:
        """Returns 'ok' or 'stop' (budget exhausted). Raises AuthError/QuotaError."""
        q = item.query
        self.main_cached = False
        reason = "retry_failed_aio" if item.action == "retry_search" else "normal"
        params = sc.build_search_params(q)
        resp, attempts, stopped = self.request_with_retries("search", params, item, reason)
        if resp is not None and resp.error_kind == "quota":
            raise sc.QuotaError(resp.message or "account has run out of searches")
        if stopped and (resp is None or resp.retryable):
            self.summary.failures.append((q.id, "search", "budget_exhausted"))
            return "stop"
        if resp is None or resp.error_kind is not None:
            self.summary.failures.append((q.id, "search", resp.error_kind if resp else "none"))
            return "ok"
        outcome, ek, msg = _classify_body(resp.body)
        if outcome == "error":
            self.summary.failures.append((q.id, "search", ek))
            return "ok"

        search_path = cache.cache_path(item.run, q.id, "search", self.root)
        aio_path = cache.cache_path(item.run, q.id, "ai_overview", self.root)
        supersedes = None
        if item.action == "retry_search":
            old = cache.read_cached(search_path) or {}
            supersedes = old.get("_meta", {}).get("collected_at_utc")
            cache.supersede(search_path)
            cache.supersede(aio_path)
        cache.write_cached(search_path, self._meta(item, "search", params, resp, attempts, outcome, None, msg,
                                                   supersedes), resp.body, self.api_key)
        self.main_cached = True
        token = needs_followup(resp.body) if outcome == "ok" else None
        if token:
            return self._followup(item, token, supersedes, reason)
        state = "absent" if outcome == "no_results" else (cached_aio_state(item.run, q.id, self.root) or "absent")
        self.summary.aio_states[state] += 1
        return "ok"

    def _followup(self, item: PlanItem, token: str, supersedes: str | None, reason: str) -> str:
        q = item.query
        params = sc.build_aio_params(token)
        path = cache.cache_path(item.run, q.id, "ai_overview", self.root)
        abort = None
        try:
            resp, attempts, stopped = self.request_with_retries("ai_overview", params, item, reason)
        except sc.AuthError as exc:
            resp, attempts, stopped, abort = None, getattr(exc, "attempts", 1), False, exc
        safe_params = {"engine": params["engine"], "page_token": "<redacted-token>"}
        if resp is not None and resp.error_kind is None and _aio_payload(resp.body):
            meta = self._meta(item, "ai_overview", safe_params, resp, attempts, "ok", None, "", supersedes)
            cache.write_cached(path, meta, resp.body, self.api_key)
            self.summary.aio_states["present_followup"] += 1
            return "ok"
        if abort is not None:
            ek = "auth"
        elif stopped and (resp is None or resp.retryable):
            ek = "budget_exhausted"
        elif resp is not None and resp.error_kind:
            ek = resp.error_kind
        elif resp is not None and resp.body.get("error"):
            ek = "api_error"
        else:
            ek = "empty"
        msg = resp.message if resp is not None else ""
        if resp is not None and isinstance(resp.body.get("error"), str):
            msg = resp.body["error"]
        meta = self._meta(item, "ai_overview", safe_params, resp, attempts, "error", ek, msg, supersedes)
        cache.write_cached(path, meta, {}, self.api_key)
        self.summary.aio_states["followup_failed"] += 1
        self.summary.failures.append((q.id, "ai_overview", ek))
        if abort is not None:
            raise abort
        if resp is not None and resp.error_kind == "quota":
            raise sc.QuotaError(resp.message or "account has run out of searches")
        return "stop" if ek == "budget_exhausted" else "ok"


def _account(api_key: str | None, session, no_account: bool) -> bud.AccountInfo | None:
    if no_account or not api_key:
        return None
    try:
        return bud.AccountInfo.from_fields(sc.fetch_account(api_key, session=session))
    except sc.AuthError:
        raise
    except Exception:
        return None


def run_collection(plan: list[PlanItem], api_key: str | None, *, dry_run: bool, cap: int, max_new: int | None,
                   followup_rate: float, retries: int, retry_headroom: int, sleep_s: float, yes: bool,
                   no_account: bool, min_hours_between_runs: float = config.DEFAULT_MIN_HOURS_BETWEEN_RUNS,
                   force_spacing: bool = False, session=None, log: Callable = print,
                   input_fn: Callable[[str], str] = input, isatty: Callable[[], bool] = lambda: sys.stdin.isatty(),
                   sleep_fn: Callable[[float], None] = time.sleep, now: datetime | None = None,
                   root: Path | None = None) -> Summary:
    log(format_plan(plan))
    ledger_rows = bud.read_ledger()
    spent = bud.ledger_total(ledger_rows)
    account = _account(api_key, session, no_account)
    est = bud.estimate(plan, followup_rate, retry_headroom if retries > 0 else 0)
    left_text = "unknown" if account is None or account.total_searches_left is None else str(account.total_searches_left)
    log(f"\nledger_total (study so far): {spent}")
    log(f"study cap: {cap}   left under cap: {cap - spent}   account searches left: {left_text}   "
        f"max-new: {max_new if max_new is not None else 'none'}")
    log(f"estimate: main {est.main} + follow-ups <= {est.followup_max} + retry headroom {est.retry_headroom} "
        f"= {est.total}")
    runs = {it.run for it in plan}
    spacing = None
    if est.main:
        for run in runs:
            spacing = spacing or spacing_problem(run, min_hours_between_runs, now, root)
    if dry_run:
        notes = []
        try:
            notes = bud.check_budget(est, cap, spent, account, dry_run=True, max_new=max_new)
        except bud.BudgetError as exc:
            log(f"WOULD REFUSE: {exc}")
        for n in notes:
            log(n)
        if spacing and not force_spacing:
            log(f"WOULD REFUSE: {spacing}")
        most = est.total if max_new is None else min(est.total, max(0, max_new))
        log(f"DRY RUN: would spend <= {most} searches (study total after: <= {spent + most} of cap {cap})")
        return Summary()
    if not est.main:
        log("nothing to collect: every planned item is cached")
        return Summary()
    if spacing and not force_spacing:
        raise bud.BudgetError(spacing)
    bud.check_budget(est, cap, spent, account, dry_run=False, max_new=max_new)
    if api_key is None:
        raise ConfigError("SERPAPI_API_KEY not set (env or .env)")
    if not yes:
        if not isatty():
            raise bud.BudgetError("not a terminal and --yes not given; refusing to spend searches")
        most = est.total if max_new is None else min(est.total, max(0, max_new))
        answer = input_fn(f"About to spend up to {most} searches (study so far {spent}, cap {cap}, "
                          f"account left {left_text}). Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            raise bud.BudgetError("cancelled by user")

    budget = bud.RuntimeBudget.from_limits(cap, spent, account.total_searches_left if account else None, max_new)
    collector = Collector(api_key, budget=budget, retries=retries, sleep_s=sleep_s, session=session,
                          sleep_fn=sleep_fn, log=log, ledger_rows=ledger_rows, root=root)
    summary = collector.summary
    summary.account_before = account.this_month_usage if account else None
    todo = [it for it in plan if it.action != "skip"]
    i = 0
    try:
        with cache.CollectLock():
            for i, item in enumerate(todo):
                if collector.collect_item(item) == "stop":
                    summary.stopped_reason = "budget exhausted"
                    summary.items_left = len(todo) - i - (1 if collector.main_cached else 0)
                    break
    except KeyboardInterrupt:
        summary.stopped_reason = "interrupted (Ctrl+C)"
        # Same rule as the budget stop: the current item counts as left unless its main search was saved this run.
        summary.items_left = len(todo) - i - (1 if collector.main_cached else 0)
    finally:
        after = _account(api_key, session, no_account) if summary.attempts else None
        summary.account_after = after.this_month_usage if after else None
        log(format_summary(summary))
    return summary


def format_summary(s: Summary) -> str:
    lines = ["", "collection summary"]
    for kind in ("search", "ai_overview"):
        lines.append(f"  {kind}: {s.requests[kind]} logical requests, {s.attempts[kind]} attempts")
    lines.append(f"  retries: {s.retries}   total attempts (ledger lines added): {sum(s.attempts.values())}")
    if s.aio_states:
        lines.append("  AIO states: " + ", ".join(f"{k}={v}" for k, v in sorted(s.aio_states.items())))
    if s.failures:
        lines.append("  failures: " + ", ".join(f"{q}/{k}:{e}" for q, k, e in s.failures))
    if s.stopped_reason:
        lines.append(f"  STOPPED: {s.stopped_reason}, {s.items_left} items left")
    if s.account_before is not None and s.account_after is not None:
        lines.append(f"  Account API this_month_usage delta: {s.account_after - s.account_before} "
                     f"(ground truth; may include other use of the same account)")
    return "\n".join(lines)
