from __future__ import annotations
import csv
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from brand_study import config
from brand_study.config import ConfigError

ID_RE = re.compile(r"^[a-z0-9_]+$")
REQUIRED_COLUMNS = ("id", "sector", "query", "query_type", "city")


@dataclass(frozen=True)
class Query:
    id: str
    sector: str
    query: str
    query_type: str
    city: str
    location: str
    notes: str = ""


def _location_for(query_type: str, city: str) -> str:
    if query_type == "local":
        return config.CITY_LOCATIONS[city]
    return config.STATE_LOCATION


def _city_mentioned(text: str, city: str) -> bool:
    t = text.lower()
    variants = {city.lower(), city.lower().replace("st. ", "st "), city.lower().replace("st. ", "saint ")}
    return any(v in t for v in variants) or "near me" in t


def load_queries(path: Path | str = config.DEFAULT_QUERIES, warn: bool = True) -> list[Query]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ConfigError(f"{path.name}: missing columns {missing}")
        rows = list(reader)

    errors: list[str] = []
    out: list[Query] = []
    seen_ids: set[str] = set()
    seen_pairs: dict[tuple[str, str], str] = {}
    for i, row in enumerate(rows, start=2):  # header is line 1
        qid = (row.get("id") or "").strip()
        sector = (row.get("sector") or "").strip()
        text = (row.get("query") or "").strip()
        qtype = (row.get("query_type") or "").strip()
        city = (row.get("city") or "").strip()
        where = f"{path.name} line {i}"
        if not ID_RE.match(qid):
            errors.append(f"{where}: invalid id {qid!r}")
        elif qid in seen_ids:
            errors.append(f"{where}: duplicate id {qid!r}")
        seen_ids.add(qid)
        if sector not in config.SECTORS:
            errors.append(f"{where}: invalid sector {sector!r}")
        if qtype not in config.QUERY_TYPES:
            errors.append(f"{where}: invalid query_type {qtype!r}")
        if not text:
            errors.append(f"{where}: empty query")
        if qtype == "local":
            if not city:
                errors.append(f"{where}: local query needs a city")
                continue
            if city not in config.CITY_LOCATIONS:
                errors.append(f"{where}: city {city!r} not in CITY_LOCATIONS")
                continue
            if warn and text and not _city_mentioned(text, city):
                warnings.warn(f"{where}: local query text names neither {city!r} nor 'near me'")
        elif qtype == "broad" and city:
            errors.append(f"{where}: broad query must have an empty city")
        if qtype not in config.QUERY_TYPES:
            continue
        location = _location_for(qtype, city)
        pair = (text.lower(), location)
        if text and pair in seen_pairs:
            errors.append(f"{where}: same query text and location as {seen_pairs[pair]!r}")
        seen_pairs[pair] = qid
        out.append(Query(qid, sector, text, qtype, city, location, (row.get("notes") or "").strip()))

    if errors:
        raise ConfigError("query file invalid:\n  " + "\n  ".join(errors))
    if warn:
        for sector, n in Counter(q.sector for q in out).items():
            if abs(n - 25) > 5:
                warnings.warn(f"{path.name}: sector {sector!r} has {n} queries (expected about 25)")
    return out


def query_counts(queries: list[Query]) -> dict[str, Counter]:
    return {
        "sector": Counter(q.sector for q in queries),
        "sector_type": Counter(f"{q.sector}/{q.query_type}" for q in queries),
        "city": Counter(q.city for q in queries if q.city),
    }
