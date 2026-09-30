"""The 14 typed Pydantic contracts: valid inputs pass, bad inputs fail."""

import pytest
from pydantic import ValidationError

from src.agents.contracts import (
    QA_DIMENSIONS,
    AnalyzeRequest,
    AudioMetadata,
    AuditEvent,
    CallReport,
    CallSummary,
    InjectionCheck,
    IntakeResult,
    ObservabilityMetrics,
    PipelineError,
    QADimension,
    QAScore,
    RedactionResult,
    Transcript,
    TranscriptSegment,
)


def _segment(**kw):
    base = {"start": 0.0, "end": 1.0, "speaker": "Agent", "text": "hi"}
    base.update(kw)
    return base


def test_audio_metadata():
    m = AudioMetadata(
        filename="a.wav", size_bytes=10, duration_sec=1.0, format="wav", sha256="x" * 64
    )
    assert m.format == "wav"


def test_audio_metadata_bad_format():
    with pytest.raises(ValidationError):
        AudioMetadata(
            filename="a", size_bytes=1, duration_sec=1.0, format="ogg", sha256="x" * 64
        )


def test_intake_result():
    r = IntakeResult(
        valid=True,
        format="mp3",
        duration_sec=5.0,
        sha256="y" * 64,
        pii_in_metadata=["email"],
    )
    assert r.pii_in_metadata == ["email"]


def test_segment_end_before_start_rejected():
    with pytest.raises(ValidationError):
        TranscriptSegment(start=5.0, end=1.0, speaker="Agent", text="x")


def test_segment_bad_speaker_rejected():
    with pytest.raises(ValidationError):
        TranscriptSegment(**_segment(speaker="Robot"))


def test_transcript_cached_flag():
    t = Transcript(segments=[_segment()], full_text="hi", cached=True)
    assert t.cached is True


def test_injection_check():
    c = InjectionCheck(is_malicious=True, matched_patterns=["dan-mode"], blocked=True)
    assert c.blocked


def test_redaction_result():
    r = RedactionResult(redacted_text="x", redacted_segments=[], counts={"ssn": 2})
    assert r.counts["ssn"] == 2


def test_call_summary_sentiment_enum():
    s = CallSummary(
        purpose="p", key_points=["a"], action_items=[], sentiment="positive"
    )
    assert s.sentiment == "positive"
    with pytest.raises(ValidationError):
        CallSummary(purpose="p", key_points=[], action_items=[], sentiment="ecstatic")


def test_qa_dimension_score_bounds():
    QADimension(name="n", score=5, weight=0.5, rationale="r")
    with pytest.raises(ValidationError):
        QADimension(name="n", score=6, weight=0.5, rationale="r")


def test_qa_score_weights_must_sum_to_one():
    dims = [QADimension(name="a", score=3, weight=0.5, rationale="r")]
    with pytest.raises(ValidationError):
        QAScore(dimensions=dims, overall_score=3.0)
    dims2 = [
        QADimension(name=d["name"], score=4, weight=d["weight"], rationale="r")
        for d in QA_DIMENSIONS
    ]
    ok = QAScore(dimensions=dims2, overall_score=4.0)
    assert ok.overall_score == 4.0


def test_call_report_status_enum():
    with pytest.raises(ValidationError):
        CallReport(
            call_id="c",
            audio_sha256="z" * 64,
            duration_sec=1.0,
            created_at="t",
            summary=CallSummary(
                purpose="p", key_points=[], action_items=[], sentiment="neutral"
            ),
            qa=QAScore(
                dimensions=[
                    QADimension(
                        name=d["name"], score=3, weight=d["weight"], rationale="r"
                    )
                    for d in QA_DIMENSIONS
                ],
                overall_score=3.0,
            ),
            transcript=Transcript(segments=[], full_text=""),
            status="bogus",
        )


def test_audit_event():
    e = AuditEvent(timestamp="t", call_id="c", stage="intake", status="ok")
    assert e.event_id is None


def test_pipeline_error():
    e = PipelineError(stage="intake", message="bad", retryable=True)
    assert e.retryable


def test_analyze_request():
    r = AnalyzeRequest(audio_path="/tmp/a.wav", caller_id="123", department="billing")
    assert r.department == "billing"


def test_observability_metrics_defaults():
    m = ObservabilityMetrics()
    assert m.total_completed == 0 and m.recent_events == []


def test_qa_dimensions_cover_spec_weights():
    assert len(QA_DIMENSIONS) == 5
    assert abs(sum(d["weight"] for d in QA_DIMENSIONS) - 1.0) < 1e-9
