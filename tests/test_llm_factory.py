"""LLM factory: provider switching via env, offline mode, retry/backoff."""
import pytest

from src.services import llm_factory


def test_provider_status_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    provider, model, key_present, _reason = llm_factory.provider_status()
    assert (provider, model, key_present) == ("openai", "gpt-4o", True)


def test_provider_status_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    provider, model, key_present, _ = llm_factory.provider_status()
    assert provider == "gemini" and key_present
    assert model == "gemini-2.0-flash"


def test_provider_status_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gq-test")
    provider, model, key_present, _ = llm_factory.provider_status()
    assert provider == "groq" and key_present
    assert model == "llama-3.3-70b-versatile"


def test_missing_key_reports_offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _provider, _model, key_present, reason = llm_factory.provider_status()
    assert key_present is False and "no API key" in reason


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="unsupported LLM_PROVIDER"):
        llm_factory.get_llm()


def test_get_llm_offline_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm_factory.LLMOffline):
        llm_factory.get_llm()


class _FakeLLM:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient 500")
        return type("R", (), {"content": "ok-result"})()


def test_retry_succeeds_after_transient_failures(monkeypatch):
    fake = _FakeLLM(failures=2)
    monkeypatch.setattr(llm_factory, "get_llm", lambda: fake)
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    assert llm_factory.call_llm_with_retry("hi") == "ok-result"
    assert fake.calls == 3


def test_retry_exhausted_reraises(monkeypatch):
    fake = _FakeLLM(failures=99)
    monkeypatch.setattr(llm_factory, "get_llm", lambda: fake)
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    with pytest.raises(RuntimeError, match="transient 500"):
        llm_factory.call_llm_with_retry("hi", max_attempts=3)
    assert fake.calls == 3


def test_offline_not_retried(monkeypatch):
    def boom():
        raise llm_factory.LLMOffline("no key")
    monkeypatch.setattr(llm_factory, "get_llm", boom)
    with pytest.raises(llm_factory.LLMOffline):
        llm_factory.call_llm_with_retry("hi")


def test_node_retry_decorator(monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    calls = []

    @llm_factory.with_node_retry
    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("boom")
        return "recovered"

    assert flaky() == "recovered"
    assert len(calls) == 2


def test_model_name_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    assert llm_factory.provider_status()[1] == "gpt-4o-mini"


class _StubLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, prompt):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_get_llm_openai_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory.get_llm()
    assert llm.__class__.__name__ == "ChatOpenAI"


def test_get_llm_gemini_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    llm = llm_factory.get_llm()
    assert llm.__class__.__name__ == "ChatGoogleGenerativeAI"


def test_get_llm_groq_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gq-test")
    llm = llm_factory.get_llm()
    assert llm.__class__.__name__ == "ChatGroq"


def test_call_llm_happy_path(monkeypatch):
    monkeypatch.setattr(llm_factory, "get_llm",
                        lambda: _StubLLM(["hello world"]))
    assert llm_factory.call_llm_with_retry("hi") == "hello world"


def test_call_llm_unexpected_error_not_retried(monkeypatch):
    class _AlwaysFails:
        def invoke(self, prompt):
            raise TypeError("boom")

    monkeypatch.setattr(llm_factory, "get_llm", lambda: _AlwaysFails())
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.0001")
    with pytest.raises(TypeError):
        llm_factory.call_llm_with_retry("hi", max_attempts=3)
