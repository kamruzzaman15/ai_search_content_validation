from __future__ import annotations
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from brand_study import config
from brand_study.serpapi_client import scrub

KINDS = ("search", "ai_overview")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_path(run: str, query_id: str, kind: str, root: Path | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError(kind)
    return (root or config.raw_dir()) / run / f"{query_id}.{kind}.json"


def read_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_cached(path: Path, meta: dict, response: dict, api_key: str | None) -> Path:
    payload = {"_meta": scrub(meta, api_key), "response": scrub(response or {}, api_key)}
    text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False)
    if api_key and api_key in text:
        raise RuntimeError("refusing to cache: API key still present after scrubbing")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def supersede(path: Path) -> Path | None:
    data = read_cached(path)
    if data is None:
        return None
    stamp = (data.get("_meta", {}).get("collected_at_utc") or utc_now()).replace(":", "")
    query_id, kind = path.name.split(".")[0], path.name.split(".")[1]
    dest = path.parent / "_superseded" / f"{query_id}.{kind}.{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while dest.exists():
        dest = dest.with_name(f"{query_id}.{kind}.{stamp}_{n}.json")
        n += 1
    os.replace(path, dest)
    return dest


def iter_run_files(run: str, root: Path | None = None):
    d = (root or config.raw_dir()) / run
    if not d.exists():
        return
    for p in sorted(d.glob("*.json")):
        yield p


class LockHeld(Exception):
    pass


class CollectLock:
    def __init__(self, path: Path | None = None):
        self.path = path or config.lock_path()
        self._held = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise LockHeld(
                f"another collect seems to be running (lock {self.path}). "
                f"If no collect is running, delete that file and retry."
            )
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "started": utc_now(), "t": time.time()}))
        self._held = True
        return self

    def __exit__(self, *exc):
        if self._held:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._held = False
        return False
