from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus

import requests

from brand_study import config

SCRUB_KEYS = {"api_key", "account_email", "email"}
ACCOUNT_FIELDS = (
    "plan_name", "account_status", "searches_per_month", "plan_searches_left", "extra_credits",
    "total_searches_left", "this_month_usage", "this_hour_searches", "last_hour_searches",
    "account_rate_limit_per_hour",
)
QUOTA_MARKERS = ("run out of searches", "out of searches", "searches left", "plan limit")


class AuthError(Exception):
    """401/403 from SerpApi: bad key or account problem."""


class QuotaError(Exception):
    """The account has run out of searches."""


@dataclass
class Response:
    status: int | None
    body: dict
    error_kind: str | None = None
    retryable: bool = False
    message: str = ""


def build_search_params(query) -> dict:
    return {
        "engine": "google",
        "q": query.query,
        "location": query.location,
        "gl": config.GL,
        "hl": config.HL,
        "google_domain": config.GOOGLE_DOMAIN,
        "device": config.DEVICE,
        "no_cache": "true",
    }


def build_aio_params(page_token: str) -> dict:
    return {"engine": "google_ai_overview", "page_token": page_token}


def redact(text: Any, key: str | None) -> str:
    text = "" if text is None else str(text)
    if key:
        for form in {key, quote(key, safe=""), quote_plus(key)}:
            text = text.replace(form, "***")
    return text


def scrub(obj: Any, key: str | None) -> Any:
    if isinstance(obj, dict):
        return {k: scrub(v, key) for k, v in obj.items() if k not in SCRUB_KEYS}
    if isinstance(obj, list):
        return [scrub(v, key) for v in obj]
    if isinstance(obj, str) and key and key in obj:
        return redact(obj, key)
    return obj


def _is_quota_message(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in QUOTA_MARKERS)


def get_json_once(url: str, params: dict, api_key: str, session=None, timeout: float = 60) -> Response:
    """One HTTP attempt. Never raises for transport/HTTP errors except auth (AuthError)."""
    sess = session or requests
    full = dict(params)
    full["api_key"] = api_key
    try:
        r = sess.get(url, params=full, timeout=timeout)
    except Exception as exc:  # requests exceptions and anything raised by an injected session
        return Response(None, {}, "transport", True, redact(f"{type(exc).__name__}: {exc}", api_key))
    status = getattr(r, "status_code", None)
    try:
        body = r.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    err = redact(body.get("error", ""), api_key) if isinstance(body.get("error"), str) else ""
    if status in (401, 403):
        raise AuthError(f"SerpApi HTTP {status}: {err or 'unauthorised'}")
    if _is_quota_message(err):  # any status, incl. a 200 with an error body: abort, don't burn attempts
        return Response(status, body, "quota", False, err)
    if status == 429:
        return Response(status, body, "http_429", True, err or "throughput limit")
    if status is not None and status >= 500:
        return Response(status, body, "http_5xx", True, err or f"HTTP {status}")
    if status is not None and status >= 400:
        return Response(status, body, "http_4xx", False, err or f"HTTP {status}")
    if not body:
        return Response(status, body, "empty", False, "empty or non-JSON body")
    return Response(status, body, None, False, err)


def fetch_account(api_key: str, session=None, timeout: float = 30) -> dict:
    """Free Account API. Returns whitelisted numeric/plan fields only."""
    resp = get_json_once(config.ACCOUNT_ENDPOINT, {}, api_key, session=session, timeout=timeout)
    if resp.error_kind:
        raise RuntimeError(f"Account API failed: {resp.error_kind} {resp.message}")
    return {k: resp.body.get(k) for k in ACCOUNT_FIELDS if k in resp.body}
