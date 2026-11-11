from __future__ import annotations
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from brand_study import ollama_lifecycle as ol

TAG = "gemma4:31b"
REAL_DISK_CHECK_PATHS = ol.disk_check_paths


class FakeOllamaAPI:
    def __init__(self, tags=(), base_url="http://localhost:11434", pull_error=None, delete_error=None,
                 delete_removes=True, list_error_after_delete=False):
        self.tags = set(tags)
        self.base_url = base_url
        self.pull_error = pull_error
        self.delete_error = delete_error
        self.delete_removes = delete_removes
        self.list_error_after_delete = list_error_after_delete
        self.calls = []

    def version(self):
        self.calls.append("version")
        return "0.12.0"

    def list_tags(self):
        self.calls.append("list_tags")
        if self.list_error_after_delete and "delete" in self.calls:
            raise ConnectionError("down")
        return set(self.tags)

    def pull(self, tag, log=print):
        self.calls.append("pull")
        if self.pull_error:
            raise self.pull_error
        self.tags.add(tag)

    def unload(self, tag):
        self.calls.append("unload")

    def delete(self, tag):
        self.calls.append("delete")
        if self.delete_error:
            if self.delete_removes:
                self.tags.discard(tag)
            raise self.delete_error
        found = tag in self.tags
        self.tags.discard(tag)
        return found


def gpus(free_gb):
    return lambda: [ol.GpuInfo(0, "A100", 80 * 1024, int((80 - free_gb) * 1024), int(free_gb * 1024))]


def use(api, logs=None, **kw):
    opts = dict(api=api, models_path="/fake/models", gpu_query=gpus(60), disk_free=lambda p: 200.0,
                rm_fallback=lambda tag, base: pytest.fail("fallback not expected"),
                log=(logs.append if logs is not None else (lambda s: None)))
    opts.update(kw)
    return ol.ollama_model(TAG, **opts)


@pytest.fixture(autouse=True)
def _fake_paths(monkeypatch):
    monkeypatch.setattr(ol, "disk_check_paths", lambda explicit=None: [Path("/fake/models")] if explicit else [])


def test_pull_use_delete():
    api = FakeOllamaAPI()
    with use(api) as tag:
        assert tag == TAG and TAG in api.tags
    assert api.calls[:3] == ["version", "list_tags", "pull"] and "delete" in api.calls
    assert TAG not in api.tags


def test_already_present_is_not_deleted():
    api = FakeOllamaAPI(tags={TAG})
    logs = []
    with use(api, logs):
        pass
    assert "pull" not in api.calls and "delete" not in api.calls
    assert any("will NOT be deleted" in x for x in logs)


def test_tag_normalisation_and_placeholder():
    assert ol.normalize_tag("llama3") == "llama3:latest" and ol.normalize_tag("a/b:7b") == "a/b:7b"
    with pytest.raises(ol.OllamaError, match="placeholder"):
        with ol.ollama_model("REPLACE_ME_qwen3_27b_tag", api=FakeOllamaAPI()):
            pass


def test_low_disk_refuses_before_pull():
    api = FakeOllamaAPI()
    with pytest.raises(ol.InsufficientDiskError):
        with use(api, disk_free=lambda p: 30.0, expected_size_gb=25, min_free_margin_gb=10):
            pass
    assert "pull" not in api.calls and "delete" not in api.calls


def test_minimum_over_several_paths(monkeypatch):
    monkeypatch.setattr(ol, "disk_check_paths", lambda explicit=None: [Path("/a"), Path("/b")])
    free = {"/a": 500.0, "/b": 30.0}
    with pytest.raises(ol.InsufficientDiskError, match="leave 5.0 GB"):
        with use(FakeOllamaAPI(), models_path=None, disk_free=lambda p: free[str(p)]):
            pass


def test_no_checkable_path():
    with pytest.raises(ol.DiskCheckUnavailable):
        with use(FakeOllamaAPI(), models_path=None):
            pass
    api = FakeOllamaAPI()
    with use(api, models_path=None, allow_unknown_disk=True):
        pass
    assert "pull" in api.calls


def test_disk_check_paths_real(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "not" / "yet"))
    assert REAL_DISK_CHECK_PATHS()[0] == tmp_path  # nearest existing parent of a not-yet-created dir
    (tmp_path / "explicit").mkdir()
    assert REAL_DISK_CHECK_PATHS(str(tmp_path / "explicit" / "models")) == [tmp_path / "explicit"]


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores directory permissions")
def test_disk_check_paths_unreadable_dir(monkeypatch, tmp_path):
    # Typical shared install: /usr/share/ollama is the ollama user's 0750 home, so its models dir can't be stat'ed.
    home = tmp_path / "ollama_home"
    (home / ".ollama" / "models").mkdir(parents=True)
    home.chmod(0o000)
    try:
        models = home / ".ollama" / "models"
        if sys.version_info < (3, 12):
            with pytest.raises(PermissionError):
                models.exists()  # what Path.exists() does before Python 3.12 (newer versions return False)
        monkeypatch.delenv("OLLAMA_MODELS", raising=False)
        monkeypatch.delenv("BRAND_STUDY_OLLAMA_MODELS_DIR", raising=False)
        monkeypatch.setattr(ol, "DEFAULT_MODEL_DIRS", (str(models),))
        assert REAL_DISK_CHECK_PATHS() == [home]
        assert REAL_DISK_CHECK_PATHS(str(models)) == [home]  # the README's BRAND_STUDY_OLLAMA_MODELS_DIR route
        monkeypatch.setenv("BRAND_STUDY_OLLAMA_MODELS_DIR", str(models))
        assert REAL_DISK_CHECK_PATHS() == [home]
        assert ol.free_disk_gb(home) > 0
    finally:
        home.chmod(0o755)


def test_disk_check_paths_defaults_use_nearest_parent(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(ol, "DEFAULT_MODEL_DIRS", (str(tmp_path / "a" / "models"), str(tmp_path / "b" / "models")))
    assert REAL_DISK_CHECK_PATHS() == [tmp_path]  # both missing -> same existing parent, deduplicated


def test_remote_server_warns():
    logs = []
    with use(FakeOllamaAPI(base_url="http://gpu-box:11434"), logs):
        pass
    assert any("not local" in x for x in logs) and any("this machine" in x for x in logs)


def test_gpu_unknown_and_low(monkeypatch):
    assert ol.query_gpus(run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())) is None
    logs = []
    with use(FakeOllamaAPI(), logs, gpu_query=lambda: None):
        pass
    assert any("GPU memory unknown" in x for x in logs)
    logs = []
    with use(FakeOllamaAPI(), logs, gpu_query=gpus(18)):
        pass
    assert any("another job" in x and "offload to CPU" in x for x in logs)
    api = FakeOllamaAPI()
    with pytest.raises(ol.InsufficientGpuError):
        with use(api, gpu_query=gpus(18), strict_gpu=True):
            pass
    assert "pull" not in api.calls


def test_query_gpus_parses():
    res = SimpleNamespace(returncode=0, stdout="0, NVIDIA A100, 81920, 20480, 61440\nbad line\n")
    assert ol.query_gpus(run=lambda *a, **k: res) == [ol.GpuInfo(0, "NVIDIA A100", 81920, 20480, 61440)]


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_deletes_on_error_and_ctrl_c(exc):
    api = FakeOllamaAPI()
    with pytest.raises(type(exc)):
        with use(api):
            raise exc
    assert "delete" in api.calls and TAG not in api.tags


def test_pull_failure_still_cleans_up():
    api = FakeOllamaAPI(pull_error=ol.OllamaPullError("disk full"))
    logs = []
    with pytest.raises(ol.OllamaPullError):
        with use(api, logs):
            pass
    assert "delete" in api.calls and any("partial blobs" in x for x in logs)


def test_keep_true():
    api = FakeOllamaAPI()
    with use(api, keep=True):
        pass
    assert "delete" not in api.calls and TAG in api.tags


def test_cli_fallback_overrides_inherited_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://other:1234")
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen["cmd"], seen["host"] = cmd, env["OLLAMA_HOST"]
        return SimpleNamespace(returncode=0)

    api = FakeOllamaAPI(base_url="http://127.0.0.1:11500", delete_error=ol.OllamaError("HTTP 500"),
                        delete_removes=False)
    fallback = lambda tag, base: ol.cli_rm(tag, base, run=fake_run, which=lambda n: "/usr/bin/ollama",
                                           log=lambda s: None)
    with use(api, rm_fallback=fallback):
        pass
    assert seen == {"cmd": ["/usr/bin/ollama", "rm", TAG], "host": "http://127.0.0.1:11500"}


def test_path_prefixed_base_url_disables_fallback():
    logs = []
    ran = []
    api = FakeOllamaAPI(base_url="http://proxy.local/ollama", delete_error=ol.OllamaError("HTTP 502"),
                        delete_removes=False)
    fallback = lambda tag, base: ol.cli_rm(tag, base, run=lambda *a, **k: ran.append(a), which=lambda n: "/x",
                                           log=logs.append)
    with use(api, logs, rm_fallback=fallback):
        pass
    assert ran == [] and any("CLI fallback disabled" in x for x in logs)
    assert any(f"OLLAMA_HOST=http://proxy.local/ollama ollama rm {TAG}" in x for x in logs)


def test_fallback_skipped_when_tag_already_gone():
    api = FakeOllamaAPI(delete_error=ol.OllamaError("timeout"), delete_removes=True)
    logs = []
    with use(api, logs):
        pass
    assert any("skipping CLI fallback" in x for x in logs)


def test_both_deletes_fail_with_exception_in_flight():
    api = FakeOllamaAPI(delete_error=ol.OllamaError("HTTP 500"), delete_removes=False,
                        list_error_after_delete=True)
    logs = []
    with pytest.raises(ValueError, match="user code"):
        with use(api, logs, rm_fallback=lambda tag, base: False):
            raise ValueError("user code")
    assert any("run manually: OLLAMA_HOST=http://localhost:11434 ollama rm" in x for x in logs)


@pytest.mark.parametrize("url,host", [
    ("http://localhost:11434", "http://localhost:11434"),
    ("http://127.0.0.1", "http://127.0.0.1:11434"),
    ("https://gpu.example.org:8443/", "https://gpu.example.org:8443"),
    ("http://[::1]:11434", "http://[::1]:11434"),
    ("http://proxy/ollama", None),
    ("http://user:pw@host:11434", None),
    ("ftp://host", None),
    ("not a url", None),
    ("http://host:notaport", None),
])
def test_cli_host_from_base_url(url, host):
    assert ol.cli_host_from_base_url(url) == host


def test_cli_rm_without_binary():
    logs = []
    assert ol.cli_rm(TAG, "http://localhost:11434", which=lambda n: None, log=logs.append) is False
    assert "not on PATH" in logs[0]


def test_sigterm_handler_restored():
    before = signal.getsignal(signal.SIGTERM)
    seen = []
    with use(FakeOllamaAPI()):
        seen.append(signal.getsignal(signal.SIGTERM))
    assert seen[0] is not before and signal.getsignal(signal.SIGTERM) is before


# --- OllamaAPI over a fake HTTP session -------------------------------------------------

class FakeHTTP:
    def __init__(self, pull_lines=(), status=200, delete_status=200, tags=None):
        self.pull_lines, self.status, self.delete_status = list(pull_lines), status, delete_status
        self.tags = tags or {"models": [{"name": "a:1"}, {"model": "b"}]}
        self.requests = []

    def _resp(self, status, body=None, lines=()):
        return SimpleNamespace(status_code=status, json=lambda: body, text="",
                               raise_for_status=lambda: None, iter_lines=lambda: iter(lines))

    def get(self, url, timeout=None):
        self.requests.append(("GET", url, None))
        return self._resp(200, {"version": "0.12.0"} if url.endswith("version") else self.tags)

    def post(self, url, json=None, stream=False, timeout=None):
        self.requests.append(("POST", url, json))
        self.post_timeouts = getattr(self, "post_timeouts", []) + [timeout]
        return self._resp(self.status, {}, [l.encode() for l in self.pull_lines])

    def delete(self, url, json=None, timeout=None):
        self.requests.append(("DELETE", url, json))
        return self._resp(self.delete_status)


def test_api_pull_error_line_with_http_200():
    lines = [json.dumps({"status": "pulling manifest"}), json.dumps({"error": "no space left on device"})]
    api = ol.OllamaAPI("http://localhost:11434", session=FakeHTTP(lines))
    with pytest.raises(ol.OllamaPullError, match="no space"):
        api.pull(TAG, log=lambda s: None)


def test_api_pull_requires_success():
    ok = [json.dumps({"status": "downloading", "total": 100, "completed": 50}), json.dumps({"status": "success"})]
    http = FakeHTTP(ok)
    ol.OllamaAPI("http://localhost:11434", session=http).pull(TAG, log=lambda s: None)
    assert http.requests[0][2] == {"model": TAG, "name": TAG, "stream": True}
    connect, read = http.post_timeouts[0]  # a stalled stream must raise, not hang forever
    assert connect and read and read >= 60
    with pytest.raises(ol.OllamaPullError, match="without success"):
        ol.OllamaAPI(session=FakeHTTP([json.dumps({"status": "downloading"})])).pull(TAG, log=lambda s: None)


def test_api_list_and_delete():
    api = ol.OllamaAPI("http://localhost:11434/", session=FakeHTTP())
    assert api.base_url == "http://localhost:11434" and api.list_tags() == {"a:1", "b:latest"}
    assert api.delete(TAG) is True
    assert ol.OllamaAPI(session=FakeHTTP(delete_status=404)).delete(TAG) is False
    with pytest.raises(ol.OllamaError):
        ol.OllamaAPI(session=FakeHTTP(delete_status=500)).delete(TAG)
