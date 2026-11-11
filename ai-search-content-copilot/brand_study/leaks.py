from __future__ import annotations
from pathlib import Path
from urllib.parse import quote, quote_plus

MIN_KEY_LEN = 16


def _iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    yield f


def scan_for_key(paths, key: str | None) -> tuple[list[Path], int]:
    """Return (files containing the key, files scanned). Never returns matched content."""
    if not key or len(key.strip()) < MIN_KEY_LEN:
        raise ValueError(f"key is empty or shorter than {MIN_KEY_LEN} characters; refusing to scan")
    key = key.strip()
    needles = {k.encode("utf-8") for k in (key, quote(key, safe=""), quote_plus(key))}
    hits, n = [], 0
    for f in _iter_files(paths):
        n += 1
        try:
            data = f.read_bytes()
        except OSError:
            continue
        if any(nd in data for nd in needles):
            hits.append(f)
    return hits, n
