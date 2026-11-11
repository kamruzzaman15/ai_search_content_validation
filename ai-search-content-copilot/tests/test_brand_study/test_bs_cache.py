from __future__ import annotations
import json

import pytest

from bs_helpers import FAKE_KEY
from brand_study import cache


def test_write_scrubs_and_is_atomic(tmp_path):
    p = cache.cache_path("run1", "q1", "search", tmp_path)
    cache.write_cached(p, {"run": "run1", "collected_at_utc": "2026-01-01T00:00:00Z"},
                       {"api_key": FAKE_KEY, "search_parameters": {"q": "x"}, "x": f"..{FAKE_KEY}.."}, FAKE_KEY)
    text = p.read_text()
    assert FAKE_KEY not in text
    assert json.loads(text)["response"]["search_parameters"] == {"q": "x"}
    assert not list(p.parent.glob(".tmp_*"))


def test_supersede_moves_not_deletes(tmp_path):
    p = cache.cache_path("run1", "q1", "search", tmp_path)
    cache.write_cached(p, {"collected_at_utc": "2026-01-01T00:00:00Z"}, {"a": 1}, FAKE_KEY)
    dest = cache.supersede(p)
    assert not p.exists() and dest.exists() and dest.parent.name == "_superseded"
    assert cache.supersede(p) is None


def test_lock(tmp_path):
    lock = tmp_path / ".lock"
    with cache.CollectLock(lock):
        assert lock.exists()
        with pytest.raises(cache.LockHeld, match="delete that file"):
            with cache.CollectLock(lock):
                pass
    assert not lock.exists()


def test_invalid_kind(tmp_path):
    with pytest.raises(ValueError):
        cache.cache_path("run1", "q1", "other", tmp_path)
