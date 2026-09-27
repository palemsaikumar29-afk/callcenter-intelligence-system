"""Typed Pydantic contracts shared across the pipeline.

Every stage of the pipeline speaks these contracts — nothing passes
unstructured dicts between nodes except the LangGraph TypedDict state,
which itself carries these models' serialized forms.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# 1. Audio file metadata ---------------------------------------------------------
class AudioMetadata(BaseModel):
    filename: str
    size_bytes: int = Field(ge=0)
    duration_sec: float = Field(ge=0)
    format: Literal["wav", "mp3", "flac", "m4a"]
    sha256: str = Field(min_length=64, max_length=64)


# 2. Intake validation result ----------------------------------------------------
class IntakeResult(BaseModel):
    valid: bool
    format: Literal["wav", "mp3", "flac", "m4a"] | None = None
    duration_sec: float = 0.0
    sha256: str = ""
    error: str | None = None
    pii_in_metadata: list[str] = Field(default_factory=list)


# 3. A single diarized transcript segment ----------------------------------------
class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: Literal["Agent", "Customer"]
    text: str

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        if v < info.data.get("start", 0):
            raise ValueError("segment end must be >= start")
        return v


# 4. Full transcript --------------------------------------------------------------
class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    full_text: str
    language: str = "en"
    cached: bool = False


# 5. Prompt-injection scan result --------------------------------------------------
class InjectionCheck(BaseModel):
    is_malicious: bool
    matched_patterns: list[str] = Field(default_factory=list)
    blocked: bool = False


# 6. PII redaction result ----------------------------------------------------------
class RedactionResult(BaseModel):
    redacted_text: str
    redacted_segments: list[TranscriptSegment]
    counts: dict[str, int] = Field(default_factory=dict)  # ssn/email/phone/card


# 7. Structured call summary -------------------------------------------------------
class CallSummary(BaseModel):
    purpose: str
    key_points: list[str]
    action_items: list[str]
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    entities: list[str] = Field(default_factory=list)
    offline: bool = False  # True when produced without an LLM key


# 8. One QA dimension ---------------------------------------------------------------
class QADimension(BaseModel):
    name: str
    score: int = Field(ge=1, le=5)
    weight: float = Field(gt=0, le=1)
    rationale: str


# 9. QA scorecard --------------------------------------------------------------------
class QAScore(BaseModel):
    dimensions: list[QADimension]
    overall_score: float = Field(ge=1, le=5)
    compliance_flags: list[str] = Field(default_factory=list)
    offline: bool = False

    @field_validator("dimensions")
    @classmethod
    def weights_sum_to_one(cls, v: list[QADimension]) -> list[QADimension]:
        total = sum(d.weight for d in v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"dimension weights must sum to 1.0, got {total}")
        return v


# 10. Final call report ---------------------------------------------------------------
class CallReport(BaseModel):
    call_id: str
    caller_id: str | None = None
    department: str | None = None
    audio_sha256: str
    duration_sec: float
    created_at: str
    summary: CallSummary
    qa: QAScore
    transcript: Transcript
    redaction_counts: dict[str, int] = Field(default_factory=dict)
    status: Literal["report", "supervisor_review"]


# 11. Audit log event ------------------------------------------------------------------
class AuditEvent(BaseModel):
    event_id: int | None = None
    timestamp: str
    call_id: str
    stage: str
    status: Literal["started", "ok", "error", "blocked", "flagged"]
    detail: str = ""


# 12. Pipeline error ---------------------------------------------------------------------
class PipelineError(BaseModel):
    stage: str
    message: str
    retryable: bool = False


# 13. Analyze request (UI input) ------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    audio_path: str
    caller_id: str | None = None
    department: str | None = None


# 14. Observability metrics --------------------------------------------------------------------
class ObservabilityMetrics(BaseModel):
    total_completed: int = 0
    total_failed: int = 0
    total_flagged: int = 0
    success_rate_pct: float = 0.0
    avg_qa_score: float | None = None
    total_compliance_flags: int = 0
    recent_events: list[AuditEvent] = Field(default_factory=list)
    langsmith_status: str = "not configured"


# Shared constants ---------------------------------------------------------------------------
QA_DIMENSIONS: list[dict[str, Any]] = [
    {"name": "Greeting & Professionalism", "weight": 0.15},
    {"name": "Problem Resolution", "weight": 0.30},
    {"name": "Communication Clarity", "weight": 0.20},
    {"name": "Compliance & Policy Adherence", "weight": 0.25},
    {"name": "Empathy & Customer Experience", "weight": 0.10},
]

SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac", "m4a")
