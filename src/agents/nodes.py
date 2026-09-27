"""The seven pipeline stage nodes.

Each node takes the PipelineState TypedDict and returns a partial update.
LLM-backed nodes (summarize, qa) retry with exponential backoff; PII and
injection stages run BEFORE any LLM call so tainted text never reaches a
model. QA's overall score is ALWAYS recomputed deterministically in Python
after the LLM responds — the model's own overall_score is discarded.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from ..agents.contracts import (
    QA_DIMENSIONS,
    CallReport,
    CallSummary,
    IntakeResult,
    PipelineError,
    QADimension,
    QAScore,
    RedactionResult,
    Transcript,
    TranscriptSegment,
)
from ..database.db import CallCenterDB
from ..graph.state import PipelineState
from ..services import audio as audio_svc
from ..services import injection as injection_svc
from ..services import llm_factory, report_gen
from ..services import pii as pii_svc
from ..services import transcription as transcription_svc

_DB: CallCenterDB | None = None


def set_db(db: CallCenterDB) -> None:
    global _DB
    _DB = db


def _db() -> CallCenterDB:
    assert _DB is not None, "nodes DB not initialised — call set_db() first"
    return _DB


def _audit(call_id: str, stage: str, status: str, detail: str = "") -> None:
    _db().log_event(call_id, stage, status, detail)


def _fail(state: PipelineState, stage: str, message: str,
          retryable: bool = False) -> dict[str, Any]:
    err = PipelineError(
        stage=stage, message=message, retryable=retryable).model_dump()
    _audit(state["call_id"], stage, "error", message)
    return {"status": "error", "error": err}


# -- Stage 1: Intake ---------------------------------------------------------------
def intake_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "intake", "started", state["audio_path"])
    fmt, duration, digest, pii_meta, error = audio_svc.validate_audio(
        state["audio_path"])
    if error:
        return _fail(state, "intake", error)
    _audit(call_id, "intake", "ok",
           f"format={fmt} duration={duration:.1f}s sha256={digest[:12]}…")
    if pii_meta:
        _audit(call_id, "intake", "flagged",
               f"PII detected in file metadata: {', '.join(pii_meta)}")
    intake = IntakeResult(valid=True, format=fmt, duration_sec=duration,
                          sha256=digest, pii_in_metadata=pii_meta)
    return {
        "audio_format": fmt,
        "duration_sec": duration,
        "audio_sha256": digest,
        "pii_in_metadata": pii_meta,
        "intake": intake.model_dump(),
    }


# -- Stage 2: Transcription ----------------------------------------------------------
def transcribe_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "transcribe", "started", "")
    try:
        raw = transcription_svc.transcribe_file(state["audio_path"], db=_db())
    except RuntimeError as exc:
        return _fail(state, "transcribe", str(exc), retryable=True)
    transcript = Transcript(**raw)
    _audit(call_id, "transcribe", "ok",
           f"{len(transcript.segments)} segments cached={transcript.cached}")
    return {"transcript": transcript.model_dump()}


# -- Stage 3: Prompt-injection detection (BEFORE any LLM call) -------------------------
def injection_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    text = state["transcript"]["full_text"]
    _audit(call_id, "injection_check", "started", "")
    malicious, matched = injection_svc.scan_for_injection(text)
    result = {"is_malicious": malicious, "matched_patterns": matched,
              "blocked": malicious}
    if malicious:
        _audit(call_id, "injection_check", "blocked",
               f"patterns={','.join(matched)}")
        out = _fail(state, "injection_check",
                    f"prompt injection blocked: {', '.join(matched)}")
        out["injection"] = result
        return out
    _audit(call_id, "injection_check", "ok", "clean")
    return {"injection": result}


# -- Stage 4: PII redaction (BEFORE any LLM call) ---------------------------------------
def redact_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "redact", "started", "")
    t = state["transcript"]
    redacted_text, text_counts = pii_svc.redact_text(t["full_text"])
    redacted_segments, seg_counts = pii_svc.redact_segments(t["segments"])
    totals = {k: text_counts.get(k, 0) for k in text_counts}
    result = RedactionResult(redacted_text=redacted_text,
                             redacted_segments=[
                                 TranscriptSegment(**s).model_dump()
                                 for s in redacted_segments],
                             counts=totals)
    _audit(call_id, "redact", "ok",
           f"redacted={totals} segment_spans={seg_counts}")
    return {"redaction": result.model_dump()}


# -- Stage 5: Summarization --------------------------------------------------------------
_SUMMARY_PROMPT = """You are a call-center QA analyst. Read the redacted call
transcript below and respond with ONLY a JSON object (no markdown fences) with
exactly these keys:
- purpose: one sentence describing why the customer called
- key_points: list of 3-6 important facts from the call
- action_items: list of follow-up actions (empty list if none)
- sentiment: one of positive, neutral, negative, mixed
- entities: list of named entities (products, plans, amounts — PII already redacted)

Transcript:
{transcript}
"""


def _offline_summary(text: str) -> CallSummary:
    """Deterministic extractive summary, clearly labelled offline."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    customer_lines = [ln for ln in lines if ln.startswith("Customer:")]
    purpose = (customer_lines[0][len("Customer:"):].strip()
               if customer_lines else (lines[0] if lines else ""))
    return CallSummary(
        purpose="[OFFLINE] " + purpose[:200],
        key_points=["[OFFLINE] " + ln[:160] for ln in lines[1:5]],
        action_items=[],
        sentiment="neutral",
        entities=[],
        offline=True,
    )


def _parse_summary_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                     flags=re.IGNORECASE)
    return json.loads(cleaned)


@llm_factory.with_node_retry
def summarize_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "summarize", "started", "")
    text = state["redaction"]["redacted_text"]
    try:
        raw = llm_factory.call_llm_with_retry(_SUMMARY_PROMPT.format(transcript=text))
        summary = CallSummary(**_parse_summary_json(raw))
        _audit(call_id, "summarize", "ok",
               f"provider={llm_factory.provider_status()[0]}")
    except llm_factory.LLMOffline:
        summary = _offline_summary(text)
        _audit(call_id, "summarize", "ok", "offline fallback (no LLM key)")
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        return _fail(state, "summarize",
                     f"LLM returned invalid summary JSON: {exc}",
                     retryable=True)
    return {"summary": summary.model_dump()}


# -- Stage 6: QA scoring -------------------------------------------------------------------
_QA_PROMPT = """You are a call-center QA auditor. Score this redacted call on
five dimensions from 1 (poor) to 5 (excellent). Respond with ONLY a JSON object
(no markdown fences) with exactly these keys:
- dimensions: list of 5 objects, each with name, score (1-5 int), rationale.
  Use these exact dimension names: {dimensions}
- compliance_flags: list of strings describing any policy/compliance concerns
  (empty list if none). Prefix genuinely critical violations with "CRITICAL: ".

Transcript:
{transcript}
"""


def _offline_qa() -> QAScore:
    dims = [QADimension(name=d["name"], score=3, weight=d["weight"],
                        rationale="[OFFLINE] neutral placeholder — no LLM key")
            for d in QA_DIMENSIONS]
    return QAScore(dimensions=dims, overall_score=3.0,
                   compliance_flags=[], offline=True)


def recompute_overall(dimensions: list[dict[str, Any]]) -> float:
    """Deterministic weighted overall — the LLM's own overall is discarded."""
    return round(sum(d["score"] * d["weight"] for d in dimensions), 2)


@llm_factory.with_node_retry
def qa_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "qa_score", "started", "")
    text = state["redaction"]["redacted_text"]
    try:
        raw = llm_factory.call_llm_with_retry(
            _QA_PROMPT.format(
                transcript=text,
                dimensions=", ".join(d["name"] for d in QA_DIMENSIONS)))
        parsed = _parse_summary_json(raw)
        dims = []
        weights = {d["name"]: d["weight"] for d in QA_DIMENSIONS}
        for item in parsed["dimensions"]:
            name = item["name"]
            if name not in weights:
                raise ValueError(f"unexpected dimension {name!r}")
            dims.append(QADimension(name=name, score=int(item["score"]),
                                    weight=weights[name],
                                    rationale=str(item.get("rationale", ""))))
        overall = recompute_overall([d.model_dump() for d in dims])
        qa = QAScore(dimensions=dims, overall_score=overall,
                     compliance_flags=list(parsed.get("compliance_flags", [])))
        _audit(call_id, "qa_score", "ok",
               f"overall={overall} flags={len(qa.compliance_flags)}")
    except llm_factory.LLMOffline:
        qa = _offline_qa()
        _audit(call_id, "qa_score", "ok", "offline fallback (no LLM key)")
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        return _fail(state, "qa_score",
                     f"LLM returned invalid QA JSON: {exc}", retryable=True)

    critical = any("critical" in f.lower() for f in qa.compliance_flags)
    update: dict[str, Any] = {"qa": qa.model_dump()}
    if critical:
        _audit(call_id, "qa_score", "flagged",
               "critical compliance flag -> supervisor review")
        update["status"] = "supervisor_review"
    return update


# -- Stage 7: Report -------------------------------------------------------------------------
def report_node(state: PipelineState) -> dict[str, Any]:
    call_id = state["call_id"]
    _audit(call_id, "report", "started", "")
    created = datetime.now(timezone.utc).isoformat()
    status = state.get("status", "processing")
    final_status = "supervisor_review" if status == "supervisor_review" else "report"
    report = CallReport(
        call_id=call_id,
        caller_id=state.get("caller_id"),
        department=state.get("department"),
        audio_sha256=state["audio_sha256"],
        duration_sec=state["duration_sec"],
        created_at=created,
        summary=CallSummary(**state["summary"]),
        qa=QAScore(**state["qa"]),
        transcript=Transcript(**state["transcript"]),
        redaction_counts=state["redaction"]["counts"],
        status=final_status,
    )
    paths = report_gen.write_report_files(report.model_dump())
    _db().save_call_record({
        "call_id": call_id,
        "created_at": created,
        "caller_id": state.get("caller_id"),
        "department": state.get("department"),
        "audio_sha256": state["audio_sha256"],
        "duration_sec": state["duration_sec"],
        "status": final_status,
        "summary": report.summary.model_dump(),
        "qa": report.qa.model_dump(),
        "transcript": report.transcript.model_dump(),
        "report": report.model_dump(),
        "report_pdf_path": paths["pdf_path"],
    })
    _audit(call_id, "report", "ok",
           f"pdf={paths['pdf_path']} status={final_status}")
    return {"report": report.model_dump(), "report_paths": paths,
            "status": final_status}


def error_node(state: PipelineState) -> dict[str, Any]:
    """Terminal error node: persist the failed call record for observability."""
    call_id = state["call_id"]
    err = state.get("error") or {}
    try:
        _db().save_call_record({
            "call_id": call_id,
            "caller_id": state.get("caller_id"),
            "department": state.get("department"),
            "audio_sha256": state.get("audio_sha256", ""),
            "duration_sec": state.get("duration_sec", 0.0),
            "status": "error",
            "summary": None, "qa": None,
            "transcript": state.get("transcript"),
            "report": {"error": err},
            "report_pdf_path": None,
        })
    except Exception:  # noqa: BLE001,S110 - error path must never raise
        pass
    return {"status": "error"}


def new_call_id() -> str:
    return uuid.uuid4().hex[:12]
