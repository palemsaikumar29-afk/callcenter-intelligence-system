"""Live smoke test — skipped unless a real LLM key is present.

Run with a real key in the environment to verify the live LLM path:

    OPENAI_API_KEY=... pytest tests/test_live_openai.py -p no:cacheprovider

Intentionally cheap: a single tiny summarization-style call, no audio.
"""

import pytest

from src.services import llm_factory
from src.services.config import settings

needs_key = pytest.mark.skipif(
    not settings.openai_api_key, reason="OPENAI_API_KEY not set"
)


@needs_key
def test_live_llm_roundtrip(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    out = llm_factory.call_llm_with_retry("Reply with exactly: LIVE_OK", max_attempts=1)
    assert "LIVE_OK" in out


@needs_key
def test_live_provider_status(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    provider, _model, key_present, _ = llm_factory.provider_status()
    assert provider == "openai" and key_present is True
