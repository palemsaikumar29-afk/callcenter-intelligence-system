"""Central configuration — everything comes from environment variables.

No secrets, model names, timeouts, or provider choices are hardcoded here.
See .env.example for the full list with inline documentation.
"""

from __future__ import annotations

import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class Settings:
    """Read-on-access settings so tests can monkeypatch the environment."""

    # -- LLM provider selection -------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return _get("LLM_PROVIDER", "openai").strip().lower() or "openai"

    # -- OpenAI -----------------------------------------------------------------
    @property
    def openai_api_key(self) -> str:
        return _get("OPENAI_API_KEY")

    @property
    def openai_model(self) -> str:
        return _get("OPENAI_MODEL", "gpt-4o")

    @property
    def openai_timeout(self) -> float:
        return _get_float("OPENAI_TIMEOUT", 60.0)

    # -- Google Gemini ----------------------------------------------------------
    @property
    def gemini_api_key(self) -> str:
        return _get("GOOGLE_API_KEY")

    @property
    def gemini_model(self) -> str:
        return _get("GEMINI_MODEL", "gemini-2.0-flash")

    @property
    def gemini_timeout(self) -> float:
        return _get_float("GEMINI_TIMEOUT", 60.0)

    # -- Groq -------------------------------------------------------------------
    @property
    def groq_api_key(self) -> str:
        return _get("GROQ_API_KEY")

    @property
    def groq_model(self) -> str:
        return _get("GROQ_MODEL", "openai/gpt-oss-20b")

    @property
    def groq_timeout(self) -> float:
        return _get_float("GROQ_TIMEOUT", 60.0)

    # -- Retry policy -----------------------------------------------------------
    @property
    def llm_max_attempts(self) -> int:
        return _get_int("LLM_MAX_ATTEMPTS", 3)

    @property
    def llm_backoff_base(self) -> float:
        return _get_float("LLM_BACKOFF_BASE", 1.0)

    # -- Audio intake limits ----------------------------------------------------
    @property
    def max_audio_mb(self) -> float:
        return _get_float("CC_MAX_AUDIO_MB", 50.0)

    @property
    def max_audio_minutes(self) -> float:
        return _get_float("CC_MAX_AUDIO_MIN", 60.0)

    # -- Transcription ----------------------------------------------------------
    @property
    def whisper_model_name(self) -> str:
        return _get("CC_WHISPER_MODEL", "tiny")

    @property
    def whisper_device(self) -> str:
        return _get("CC_WHISPER_DEVICE", "auto")  # auto | cuda | cpu

    @property
    def preload_model(self) -> bool:
        return _get("CC_PRELOAD_MODEL", "1") != "0"

    # -- Persistence ------------------------------------------------------------
    @property
    def db_path(self) -> str:
        return _get("CC_DB_PATH", "data/calls.db")

    @property
    def reports_dir(self) -> str:
        return _get("CC_REPORTS_DIR", "reports")

    # -- App --------------------------------------------------------------------
    @property
    def app_host(self) -> str:
        return _get("CC_HOST", "0.0.0.0")

    @property
    def app_port(self) -> int:
        # Render/HF set PORT; local override via CC_PORT.
        return _get_int("PORT", _get_int("CC_PORT", 7860))

    # -- Observability ----------------------------------------------------------
    @property
    def langsmith_configured(self) -> bool:
        return bool(_get("LANGSMITH_API_KEY"))


settings = Settings()
