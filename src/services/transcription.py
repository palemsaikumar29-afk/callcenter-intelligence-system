"""Speech-to-text via faster-whisper.

The Whisper model is a module-level singleton: it is loaded ONCE (at app
startup via ``preload_model()``), never inside a request handler. Identical
audio (by SHA-256) returns the cached transcript instantly without running
inference again.

GPU auto-detection: CUDA -> float16, otherwise CPU int8 quantization.
VAD filtering is always on to drop silence.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from .audio import sha256_of_file
from .config import settings

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel

_MODEL: WhisperModel | None = None
_MODEL_LOCK = threading.Lock()
_MODEL_UNAVAILABLE: str | None = None


def _detect_device() -> str:
    override = settings.whisper_device
    if override in ("cuda", "cpu"):
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_model() -> WhisperModel:
    """Return the singleton model, loading it on first call (thread-safe)."""
    global _MODEL, _MODEL_UNAVAILABLE
    if _MODEL_UNAVAILABLE is not None:
        raise RuntimeError(f"whisper model unavailable: {_MODEL_UNAVAILABLE}")
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    from faster_whisper import WhisperModel

                    device = _detect_device()
                    compute = "float16" if device == "cuda" else "int8"
                    _MODEL = WhisperModel(
                        settings.whisper_model_name,
                        device=device,
                        compute_type=compute,
                    )
                except Exception as exc:  # pragma: no cover - env dependent
                    _MODEL_UNAVAILABLE = str(exc)
                    raise RuntimeError(
                        f"whisper model unavailable: {exc}"
                    ) from exc
    return _MODEL


def preload_model() -> None:
    """Load the model once at app startup. Safe to call repeatedly."""
    get_model()


def reset_model_cache() -> None:  # for tests
    global _MODEL, _MODEL_UNAVAILABLE
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_UNAVAILABLE = None


def diarize(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heuristic speaker diarization: Agent opens, speaker flips on long pauses.

    Rule: the first segment is the Agent (call-center greeting). Whenever the
    gap between consecutive segments exceeds 1.0s of silence, the speaker
    flips; otherwise the speaker continues. Deterministic and explainable —
    a documented heuristic, not true diarization.
    """
    out: list[dict[str, Any]] = []
    speaker = "Agent"
    prev_end: float | None = None
    for seg in segments:
        if prev_end is not None and seg["start"] - prev_end > 1.0:
            speaker = "Customer" if speaker == "Agent" else "Agent"
        out.append({**seg, "speaker": speaker})
        prev_end = seg["end"]
    return out


def transcribe_file(path: str, db=None) -> dict[str, Any]:
    """Transcribe an audio file; serve identical audio from the SHA-256 cache.

    Returns a Transcript-shaped dict with ``cached`` True/False.
    """
    digest = sha256_of_file(path)
    if db is not None:
        hit = db.cache_get(digest)
        if hit is not None:
            hit = dict(hit)
            hit["cached"] = True
            return hit

    model = get_model()
    raw_segments, info = model.transcribe(path, vad_filter=True)
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in raw_segments
    ]
    segments = diarize(segments)
    full_text = " ".join(
        f"{s['speaker']}: {s['text']}" for s in segments
    )
    transcript = {
        "segments": segments,
        "full_text": full_text,
        "language": getattr(info, "language", "en") or "en",
        "cached": False,
    }
    if db is not None:
        db.cache_put(digest, transcript)
    return transcript
