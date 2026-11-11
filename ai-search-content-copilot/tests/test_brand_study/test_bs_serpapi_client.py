from __future__ import annotations

import pytest

from bs_helpers import FAKE_KEY, FakeSession, fx, q, search_key
from brand_study import config
from brand_study import serpapi_client as sc


def test_search_params_have_no_key_and_no_num():
    p = sc.build_search_params(q())
    assert p["engine"] == "google" and p["location"] == "Tampa,Florida,United States"
    assert p["gl"] == "us" and p["hl"] == "en" and p["no_cache"] == "true"
    assert "num" not in p and "api_key" not in p


def test_aio_params():
    assert sc.build_aio_params("tok") == {"engine": "google_ai_overview", "page_token": "tok"}


def test_redact_and_scrub():
    assert sc.redact(f"x {FAKE_KEY} y", FAKE_KEY) == "x *** y"
    data = {"api_key": FAKE_KEY, "account_email": "a@b.c", "nested": [{"link": f"https://x?api_key={FAKE_KEY}"}]}
    out = sc.scrub(data, FAKE_KEY)
    assert "api_key" not in out and "account_email" not in out
    assert FAKE_KEY not in str(out)


def test_transport_error_redacted():
    s = FakeSession({search_key(q()): ["transport"]})
    r = sc.get_json_once(config.SERPAPI_ENDPOINT, sc.build_search_params(q()), FAKE_KEY, session=s)
    assert r.error_kind == "transport" and r.retryable and FAKE_KEY not in r.message
    assert s.calls[0]["api_key"] == FAKE_KEY  # added inside the function only


@pytest.mark.parametrize("status,body,kind,retry", [
    (500, {"error": "server"}, "http_5xx", True),
    (429, {"error": "Too many requests"}, "http_429", True),
    (429, {"error": "Your account has run out of searches."}, "quota", False),
    (200, {"error": "Your account has run out of searches."}, "quota", False),
    (400, {"error": "Your account has run out of searches."}, "quota", False),
    (400, {"error": "Missing query `q` parameter."}, "http_4xx", False),
    (200, None, "empty", False),
])
def test_classification(status, body, kind, retry):
    s = FakeSession({search_key(q()): [(status, body)]})
    r = sc.get_json_once(config.SERPAPI_ENDPOINT, sc.build_search_params(q()), FAKE_KEY, session=s)
    assert (r.error_kind, r.retryable) == (kind, retry)


def test_auth_error_raises_redacted():
    s = FakeSession({search_key(q()): [(401, {"error": f"Invalid API key {FAKE_KEY}"})]})
    with pytest.raises(sc.AuthError) as e:
        sc.get_json_once(config.SERPAPI_ENDPOINT, sc.build_search_params(q()), FAKE_KEY, session=s)
    assert FAKE_KEY not in str(e.value)


def test_error_body_returned_as_data():
    s = FakeSession({search_key(q()): [(200, fx("search_no_results.json"))]})
    r = sc.get_json_once(config.SERPAPI_ENDPOINT, sc.build_search_params(q()), FAKE_KEY, session=s)
    assert r.error_kind is None and "hasn't returned" in r.body["error"]


def test_fetch_account_whitelists_fields():
    fields = sc.fetch_account(FAKE_KEY, session=FakeSession())
    assert fields["total_searches_left"] == 200
    assert "api_key" not in fields and "account_email" not in fields and "account_id" not in fields
