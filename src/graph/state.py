"""LangGraph pipeline state — one TypedDict shared by all seven stages.

Each node receives the full state and returns a *partial* update dict;
LangGraph merges it. Failures are isolated: a node records its error and
routes to the terminal error outcome instead of crashing the graph.
"""
from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    call_id: str
    audio_path: str
    caller_id: str | None
    department: str | None
    audio_format: str | None
    duration_sec: float
    audio_sha256: str
    pii_in_metadata: list[str]
    transcript: dict[str, Any] | None
    injection: dict[str, Any] | None
    redaction: dict[str, Any] | None
    summary: dict[str, Any] | None
    qa: dict[str, Any] | None
    report: dict[str, Any] | None
    report_paths: dict[str, str] | None
    status: Literal["processing", "report", "supervisor_review", "error"]
    error: dict[str, Any] | None


def new_state(call_id: str, audio_path: str,
              caller_id: str | None = None,
              department: str | None = None) -> PipelineState:
    return PipelineState(
        call_id=call_id,
        audio_path=audio_path,
        caller_id=caller_id,
        department=department,
        duration_sec=0.0,
        audio_sha256="",
        pii_in_metadata=[],
        status="processing",
        error=None,
    )
