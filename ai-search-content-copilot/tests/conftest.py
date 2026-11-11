from __future__ import annotations
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
