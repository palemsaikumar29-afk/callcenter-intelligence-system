"""Multi-provider LLM factory.

One environment variable selects the provider; provider, model name, API key
and timeout ALL come from the environment — nothing is hardcoded. When no
key is configured for the selected provider, the factory reports offline
mode and callers use clearly-labelled deterministic fallbacks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .config import settings

SUPPORTED_PROVIDERS = ("openai", "gemini", "groq")


class LLMOffline(RuntimeError):
    """Raised when no API key is configured for the selected provider."""


def provider_status() -> tuple[str, str, bool, str]:
    """Return (provider, model, key_present, offline_reason)."""
    provider = settings.llm_provider
    if provider == "openai":
        model, key = settings.openai_model, settings.openai_api_key
    elif provider == "gemini":
        model, key = settings.gemini_model, settings.gemini_api_key
    elif provider == "groq":
        model, key = settings.groq_model, settings.groq_api_key
    else:
        return provider, "", False, f"unsupported LLM_PROVIDER={provider!r}"
    if not key:
        return provider, model, False, f"no API key configured for {provider}"
    return provider, model, True, ""


def get_llm():
    """Build the chat model for the configured provider.

    Raises LLMOffline when the provider's API key is missing, ValueError for
    an unknown provider name.
    """
    provider = settings.llm_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMOffline("OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMOffline("GOOGLE_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            timeout=settings.gemini_timeout,
        )
    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMOffline("GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            timeout=settings.groq_timeout,
        )
    raise ValueError(
        f"unsupported LLM_PROVIDER={provider!r}; choose one of {SUPPORTED_PROVIDERS}"
    )


def call_llm_with_retry(prompt: str, max_attempts: int | None = None) -> str:
    """Call the LLM with exponential backoff (up to 3 attempts by default).

    Raises LLMOffline immediately (not retryable); other exceptions are
    retried with backoff 1s, 2s, 4s ... then re-raised.
    """
    attempts = max_attempts or settings.llm_max_attempts
    llm = get_llm()  # raises LLMOffline / ValueError without retrying
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = llm.invoke(prompt)
            content = getattr(resp, "content", resp)
            return content if isinstance(content, str) else str(content)
        except (LLMOffline, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001 - retry then surface
            last = exc
            if attempt < attempts - 1:
                time.sleep(settings.llm_backoff_base * (2**attempt))
    raise last  # type: ignore[misc]


def with_node_retry(fn: Callable[..., Any], *, attempts: int | None = None):
    """Decorator: retry a pipeline node function with exponential backoff."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tries = attempts or settings.llm_max_attempts
        last: Exception | None = None
        for attempt in range(tries):
            try:
                return fn(*args, **kwargs)
            except (LLMOffline, ValueError):
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < tries - 1:
                    time.sleep(settings.llm_backoff_base * (2**attempt))
        raise last  # type: ignore[misc]

    return wrapper
