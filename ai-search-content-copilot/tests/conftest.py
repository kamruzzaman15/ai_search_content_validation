from __future__ import annotations
import hashlib
import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAKE_KEY = "test_key_DO_NOT_LEAK_1234567890"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    import requests
    import dotenv
    import brand_study.cli

    def no_network(*a, **k):
        raise RuntimeError("network disabled in tests")

    def no_dotenv(*a, **k):
        pytest.fail("load_dotenv must not be called in tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(dotenv, "load_dotenv", no_dotenv)
    monkeypatch.setattr(brand_study.cli, "load_dotenv", no_dotenv)
    monkeypatch.setenv("SERPAPI_API_KEY", FAKE_KEY)
    for var in ("OLLAMA_HOST", "OLLAMA_BASE_URL", "OLLAMA_MODELS", "BRAND_STUDY_OLLAMA_MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)
    out = tmp_path / "out"
    monkeypatch.setenv("BRAND_STUDY_OUTPUT_DIR", str(out))
    yield out


def _file_state(p: Path):
    if not p.exists():
        return None
    data = p.read_bytes()
    st = p.stat()
    return (st.st_size, st.st_mtime_ns, hashlib.sha256(data).hexdigest())


@pytest.fixture(scope="session", autouse=True)
def _guard_real_llm_overrides():
    """Tests must never create or modify the real generated LLM labels file."""
    from brand_study import config
    before = _file_state(config.DEFAULT_LLM_OVERRIDES)
    yield
    assert _file_state(config.DEFAULT_LLM_OVERRIDES) == before, "tests touched brand_study/data/llm_label_overrides.csv"


@pytest.fixture(autouse=True)
def _isolated_llm(monkeypatch, tmp_path):
    import brand_study.llm_judges as lj

    def no_subprocess(*a, **k):
        raise RuntimeError("subprocess disabled in tests; inject a runner")

    monkeypatch.setenv("BRAND_STUDY_LLM_OVERRIDES", str(tmp_path / "llm_label_overrides.csv"))
    monkeypatch.setattr(lj, "_default_runner", no_subprocess)
    for var in ("CLAUDE_BIN", "BRAND_STUDY_GEMMA_TAG", "BRAND_STUDY_QWEN_TAG", "BRAND_STUDY_CLAUDE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path / "llm_label_overrides.csv"
