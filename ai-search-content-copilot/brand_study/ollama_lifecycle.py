"""Pull an Ollama model, use it, then delete it (Part 2 helper).

    with ollama_model("gemma4:31b") as tag:
        ...  # call the model
    # the model is deleted here, also on errors, Ctrl+C and SIGTERM

Only a model that this helper pulled itself in this run is ever deleted.
"""
from __future__ import annotations
import json
import os
import shutil
import signal
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import requests

from brand_study import config

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OllamaError(Exception):
    pass


class OllamaPullError(OllamaError):
    pass


class InsufficientDiskError(OllamaError):
    pass


class InsufficientGpuError(OllamaError):
    pass


class DiskCheckUnavailable(OllamaError):
    pass


class OllamaAPI:
    def __init__(self, base_url: str | None = None, session=None, timeout: float = 30):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    # (connect, per-read) seconds: progress lines reset the read timer; a stalled stream raises instead of hanging
    PULL_TIMEOUT = (30, 600)

    def version(self) -> str:
        r = self.session.get(f"{self.base_url}/api/version", timeout=self.timeout)
        r.raise_for_status()
        return str(r.json().get("version", "?"))

    def list_tags(self) -> set[str]:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        r.raise_for_status()
        names = set()
        for m in r.json().get("models", []) or []:
            for k in ("name", "model"):
                if m.get(k):
                    names.add(normalize_tag(m[k]))
        return names

    def pull(self, tag: str, log: Callable = print) -> None:
        r = self.session.post(f"{self.base_url}/api/pull", json={"model": tag, "name": tag, "stream": True},
                              stream=True, timeout=self.PULL_TIMEOUT)
        if r.status_code >= 400:
            raise OllamaPullError(f"pull {tag}: HTTP {r.status_code} {getattr(r, 'text', '')[:200]}")
        last_status, last_pct = None, -5
        for line in r.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("error"):
                raise OllamaPullError(f"pull {tag}: {msg['error']}")
            last_status = msg.get("status", last_status)
            total, done = msg.get("total"), msg.get("completed")
            if total and done is not None:
                pct = int(100 * done / total)
                if pct >= last_pct + 5:
                    last_pct = pct
                    log(f"  pull {tag}: {last_status} {pct}%")
        if last_status != "success":
            raise OllamaPullError(f"pull {tag}: stream ended without success (last status {last_status!r})")

    def unload(self, tag: str) -> None:
        self.session.post(f"{self.base_url}/api/generate", json={"model": tag, "keep_alive": 0}, timeout=self.timeout)

    def delete(self, tag: str) -> bool:
        """True if deleted, False if not found (404). Raises on other errors."""
        r = self.session.delete(f"{self.base_url}/api/delete", json={"model": tag, "name": tag}, timeout=self.timeout)
        if r.status_code == 404:
            return False
        if r.status_code >= 300:
            raise OllamaError(f"delete {tag}: HTTP {r.status_code}")
        return True


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int


def normalize_tag(tag: str) -> str:
    tag = (tag or "").strip()
    return tag if ":" in tag.split("/")[-1] else f"{tag}:latest"


def query_gpus(run=subprocess.run) -> list[GpuInfo] | None:
    try:
        res = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if getattr(res, "returncode", 1) != 0:
        return None
    gpus = []
    for line in (res.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(GpuInfo(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), int(parts[4])))
        except ValueError:
            continue
    return gpus


def free_disk_gb(path) -> float:
    return shutil.disk_usage(str(path)).free / 1e9


def _exists(p: Path) -> bool:
    # os.path.exists maps every OSError (incl. EACCES on another user's 0750 home) to False;
    # Path.exists() on Python 3.9 raises PermissionError instead.
    return os.path.exists(str(p))


def _nearest_existing(p: Path) -> Path | None:
    p = p.expanduser()
    for cand in [p] + list(p.parents):
        if _exists(cand):
            return cand
    return None


DEFAULT_MODEL_DIRS = ("~/.ollama/models", "/usr/share/ollama/.ollama/models")


def disk_check_paths(explicit=None) -> list[Path]:
    explicit = explicit or os.environ.get("BRAND_STUDY_OLLAMA_MODELS_DIR")
    if explicit:
        p = _nearest_existing(Path(explicit))
        return [p] if p else []
    cands = ([os.environ["OLLAMA_MODELS"]] if os.environ.get("OLLAMA_MODELS") else []) + list(DEFAULT_MODEL_DIRS)
    out = []
    for c in cands:
        p = _nearest_existing(Path(c))  # unreadable or not-yet-created dirs fall back to their parent
        if p and p not in out:
            out.append(p)
    return out


def cli_host_from_base_url(base_url: str | None) -> str | None:
    try:
        parts = urlsplit(base_url or "")
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.username or parts.password or parts.query or parts.fragment:
        return None
    if parts.path not in ("", "/"):
        return None
    h = parts.hostname
    if ":" in h:
        h = f"[{h}]"
    return f"{parts.scheme}://{h}:{port or 11434}"


def cli_rm(tag: str, base_url: str, run=subprocess.run, which=shutil.which, log: Callable = print) -> bool:
    host = cli_host_from_base_url(base_url)
    if host is None:
        log(f"CLI fallback disabled: cannot derive OLLAMA_HOST from base URL {base_url!r}")
        return False
    exe = which("ollama")
    if not exe:
        log("CLI fallback disabled: `ollama` not on PATH")
        return False
    env = {**os.environ, "OLLAMA_HOST": host}
    log(f"CLI fallback: OLLAMA_HOST={host} ollama rm {tag}")
    try:
        res = run([exe, "rm", tag], env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"CLI fallback failed: {exc}")
        return False
    return getattr(res, "returncode", 1) == 0


def _is_local(base_url: str) -> bool:
    try:
        return (urlsplit(base_url).hostname or "") in LOCAL_HOSTS
    except ValueError:
        return False


@contextmanager
def _sigterm_as_exit():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _cleanup(tag, api, rm_fallback, disk_paths, disk_free, log, interrupted_pull: bool):
    try:
        api.unload(tag)
    except Exception:
        pass
    deleted = False
    try:
        found = api.delete(tag)
        deleted = True
        if not found:
            msg = "not found on server (nothing to delete)"
            if interrupted_pull:
                msg += "; the pull was interrupted, partial blobs may remain until Ollama prunes them"
            log(f"WARNING: delete {tag}: {msg}")
        else:
            log(f"deleted {tag}")
    except Exception as exc:
        log(f"WARNING: API delete of {tag} failed: {exc}")
        gone = False
        try:
            gone = tag not in api.list_tags()
        except Exception:
            pass
        if gone:
            log(f"{tag} no longer listed by the server; skipping CLI fallback")
            deleted = True
        else:
            try:
                deleted = bool(rm_fallback(tag, api.base_url))
            except Exception as exc2:
                log(f"WARNING: CLI fallback raised: {exc2}")
    if not deleted:
        host = cli_host_from_base_url(api.base_url) or api.base_url
        log(f"could not delete {tag}; run manually: OLLAMA_HOST={host} ollama rm {tag}")
    for p in disk_paths:
        try:
            log(f"free disk after cleanup at {p}: {disk_free(p):.1f} GB")
        except Exception:
            pass


@contextmanager
def ollama_model(tag: str, *, api: OllamaAPI | None = None, expected_size_gb: float = 25.0,
                 min_free_margin_gb: float = 10.0, required_gpu_gb: float = 24.0, strict_gpu: bool = False,
                 keep: bool = False, models_path=None, allow_unknown_disk: bool = False,
                 gpu_query: Callable = query_gpus, disk_free: Callable = free_disk_gb,
                 rm_fallback: Callable = cli_rm, log: Callable = print):
    if config.is_placeholder_tag(tag) or not (tag or "").strip():
        raise OllamaError(f"refusing placeholder model tag {tag!r}; set the real Ollama tag first")
    tag = normalize_tag(tag)
    api = api or OllamaAPI()
    remote = not _is_local(api.base_url)
    try:
        log(f"Ollama server {api.base_url} version {api.version()}")
    except Exception as exc:
        raise OllamaError(f"Ollama server at {api.base_url} is unreachable: {exc}") from exc
    if remote:
        log(f"WARNING: {api.base_url} is not local; the disk and GPU checks below describe THIS machine, "
            f"not the server")

    present_before = tag in api.list_tags()
    if present_before:
        log(f"{tag} is already present; using it and it will NOT be deleted afterwards")
        yield tag
        return

    paths = disk_check_paths(models_path)
    if not paths:
        if not allow_unknown_disk:
            raise DiskCheckUnavailable("no Ollama models directory to check; set BRAND_STUDY_OLLAMA_MODELS_DIR "
                                       "or pass allow_unknown_disk=True")
        log("WARNING: free disk space unknown (no checkable path)")
    else:
        frees = []
        for p in paths:
            f = disk_free(p)
            frees.append(f)
            log(f"free disk at {p} (models folder, or its nearest existing parent): {f:.1f} GB")
        left = min(frees) - expected_size_gb
        if left < min_free_margin_gb:
            raise InsufficientDiskError(
                f"pulling {tag} (~{expected_size_gb:.0f} GB) would leave {left:.1f} GB free, below the "
                f"{min_free_margin_gb:.0f} GB margin; not pulling")

    gpus = gpu_query()
    if gpus is None:
        log("WARNING: GPU memory unknown (nvidia-smi not available)")
    else:
        for g in gpus:
            log(f"GPU {g.index} {g.name}: {g.free_mb / 1024:.1f} GB free of {g.total_mb / 1024:.1f} GB")
        best = max((g.free_mb for g in gpus), default=0) / 1024
        if best < required_gpu_gb:
            msg = (f"WARNING: only {best:.1f} GB GPU memory free, {required_gpu_gb:.0f} GB wanted for {tag}. "
                   f"The shared GPU may be held by another job (~20 GB). The model will partly offload to CPU "
                   f"and run slowly.")
            if strict_gpu:
                raise InsufficientGpuError(msg)
            log("!" * 70 + "\n" + msg + "\n" + "!" * 70)
        if remote:
            log("note: GPU report is for this machine, not the remote Ollama server")

    pull_started = False
    pull_done = False
    with _sigterm_as_exit():
        try:
            pull_started = True
            log(f"pulling {tag} ...")
            api.pull(tag, log=log)
            pull_done = True
            yield tag
        finally:
            if pull_started and not present_before and not keep:
                try:
                    _cleanup(tag, api, rm_fallback, paths, disk_free, log, interrupted_pull=not pull_done)
                except Exception as exc:  # never raise from finally
                    log(f"WARNING: cleanup error: {exc}")
            elif keep:
                log(f"keep=True: leaving {tag} installed")
