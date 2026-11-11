from __future__ import annotations
import warnings

import pytest

from brand_study import config
from brand_study.config import ConfigError
from brand_study.queries import load_queries

HEADER = "id,sector,query,query_type,city,notes\n"


def write(tmp_path, body):
    p = tmp_path / "q.csv"
    p.write_text(HEADER + body)
    return p


def test_shipped_query_file_is_valid():
    qs = load_queries(config.DEFAULT_QUERIES)
    assert len(qs) == 50
    assert sum(q.sector == "coffee" for q in qs) == 25
    assert {q.city for q in qs if q.query_type == "local"} <= set(config.CITY_LOCATIONS)
    assert all(q.location == config.STATE_LOCATION for q in qs if q.query_type == "broad")


def test_duplicate_id(tmp_path):
    p = write(tmp_path, "a_1,coffee,x,broad,,\na_1,coffee,y,broad,,\n")
    with pytest.raises(ConfigError, match="line 3: duplicate id"):
        load_queries(p, warn=False)


def test_local_without_city(tmp_path):
    with pytest.raises(ConfigError, match="needs a city"):
        load_queries(write(tmp_path, "a_1,coffee,coffee in Tampa,local,,\n"), warn=False)


def test_unknown_city(tmp_path):
    with pytest.raises(ConfigError, match="not in CITY_LOCATIONS"):
        load_queries(write(tmp_path, "a_1,coffee,coffee in Naples,local,Naples,\n"), warn=False)


def test_broad_with_city_rejected(tmp_path):
    with pytest.raises(ConfigError, match="empty city"):
        load_queries(write(tmp_path, "a_1,coffee,coffee,broad,Tampa,\n"), warn=False)


def test_local_text_without_city_only_warns(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qs = load_queries(write(tmp_path, "a_1,coffee,best latte art,local,Tampa,\n"))
    assert len(qs) == 1
    assert any("names neither" in str(x.message) for x in w)


def test_same_text_different_location_ok_same_location_rejected(tmp_path):
    ok = write(tmp_path, "a_1,coffee,coffee shop near me,local,Tampa,\na_2,coffee,coffee shop near me,local,Miami,\n")
    assert len(load_queries(ok, warn=False)) == 2
    bad = write(tmp_path, "a_1,coffee,coffee shop near me,local,Tampa,\na_2,coffee,Coffee shop near me,local,Tampa,\n")
    with pytest.raises(ConfigError, match="same query text and location"):
        load_queries(bad, warn=False)


def test_invalid_sector_and_type(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_queries(write(tmp_path, "a_1,tea,x,broad,,\na_2,coffee,y,national,,\n"), warn=False)
    assert "line 2: invalid sector" in str(e.value) and "line 3: invalid query_type" in str(e.value)
