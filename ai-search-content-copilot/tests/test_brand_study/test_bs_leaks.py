from __future__ import annotations
from urllib.parse import quote

import pytest

from bs_helpers import FAKE_KEY
from brand_study.leaks import scan_for_key


def test_finds_raw_and_url_encoded_key(tmp_path):
    key = "abc/def+ghi=jkl_0123456789"
    (tmp_path / "clean.json").write_text('{"a": 1}')
    (tmp_path / "raw.json").write_text(f"x {key} y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "enc.log").write_text(f"https://serpapi.com/search?api_key={quote(key, safe='')}")
    hits, n = scan_for_key([tmp_path], key)
    assert n == 3 and sorted(h.name for h in hits) == ["enc.log", "raw.json"]


def test_clean_and_missing_paths(tmp_path):
    (tmp_path / "a.txt").write_text("nothing")
    assert scan_for_key([tmp_path, tmp_path / "missing"], FAKE_KEY) == ([], 1)


@pytest.mark.parametrize("key", ["", "   ", "short_key", None])
def test_refuses_empty_or_short_key(tmp_path, key):
    with pytest.raises(ValueError, match="refusing"):
        scan_for_key([tmp_path], key)
