"""Shared test helpers for brand_study tests (no network)."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from brand_study import config
from brand_study.queries import Query

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_KEY = "test_key_DO_NOT_LEAK_1234567890"


def fx(name: str) -> dict:
    with (FIXTURES / name).open() as f:
        return json.load(f)


def q(qid="cof_l_01", text="best coffee shop in Tampa", sector="coffee", qtype="local", city="Tampa") -> Query:
    loc = config.CITY_LOCATIONS[city] if qtype == "local" else config.STATE_LOCATION
    return Query(qid, sector, text, qtype, city if qtype == "local" else "", loc)


class FakeResp:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return copy.deepcopy(self._body)


class FakeSession:
    """Keyed by (engine, q, location) for searches, (engine, page_token) for follow-ups.

    Each script value is a list of attempts: (status, body) tuples, "transport" or "interrupt" (Ctrl+C).
    The last entry repeats once the list is exhausted.
    """

    def __init__(self, scripts: dict | None = None, account: dict | None = None):
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.account = account if account is not None else fx("account.json")
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append({"url": url, **params})
        if url == config.ACCOUNT_ENDPOINT:
            if self.account == "fail":
                raise ConnectionError("account down")
            return FakeResp(200, self.account)
        engine = params.get("engine")
        key = (engine, params.get("q"), params.get("location")) if engine == "google" else \
            (engine, params.get("page_token"))
        script = self.scripts.get(key)
        if not script:
            raise AssertionError(f"unexpected request {key}")
        item = script.pop(0) if len(script) > 1 else script[0]
        if item == "transport":
            raise ConnectionError(f"boom with key {params.get('api_key')}")
        if item == "interrupt":
            raise KeyboardInterrupt
        status, body = item
        return FakeResp(status, body)

    def paid_calls(self):
        return [c for c in self.calls if c["url"] != config.ACCOUNT_ENDPOINT]


def search_key(query: Query):
    return ("google", query.query, query.location)


def aio_key(token="FAKE_PAGE_TOKEN_abc"):
    return ("google_ai_overview", token)


def wrap(body: dict | None, query: Query | None = None, kind: str = "search", outcome: str = "ok",
         error_kind: str | None = None, collected_at: str = "2026-11-01T12:00:00Z", run: str = "run1") -> dict:
    """A cache wrapper as written by cache.write_cached."""
    query = query or q()
    return {"_meta": {"run": run, "query_id": query.id, "kind": kind, "query": query.query,
                      "location": query.location, "collected_at_utc": collected_at, "outcome": outcome,
                      "error_kind": error_kind, "supersedes": None},
            "response": copy.deepcopy(body) if body is not None else {}}


def write_wrap(root: Path, run: str, query: Query, kind: str, body: dict | None, **kw) -> Path:
    from brand_study import cache
    w = wrap(body, query, kind, run=run, **kw)
    return cache.write_cached(cache.cache_path(run, query.id, kind, root), w["_meta"], w["response"], FAKE_KEY)
