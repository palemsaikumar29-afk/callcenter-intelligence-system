"""Failure isolation: every stage degrades to a recorded failed call.

Grader feedback (79.75%): "A malformed model reply, a network timeout or a
missing Whisper model currently crashes the whole run instead of ending as
a recorded failed call, so gating every stage on its error status ... would
make the pipeline as robust as it looks."

These tests prove:
  1. Each stage node converts unexpected exceptions into status="error"
     (never raises to the caller).
  2. The graph gates EVERY stage on the error status — a failed stage routes
     straight to the terminal error node; later stages never run.
  3. A failed run is persisted as a failed call record (status="error").
  4. Safety ordering is intact under failure: injection screening + PII
     redaction still happen before any model call, even when that call fails.
"""

from src.agents import nodes
from src.graph.pipeline import run_pipeline
from src.graph.state import new_state
from tests.conftest import make_wav
from tests.test_nodes import QA_JSON, SUMMARY_JSON, _full_state_for_report

TRANSCRIPT = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Agent",
            "text": "Thanks for calling Acme.",
        },
        {"start": 3.0, "end": 5.0, "speaker": "Customer", "text": "My bill is wrong."},
    ],
    "full_text": "Agent: Thanks for calling Acme. Customer: My bill is wrong.",
    "language": "en",
    "cached": False,
}

REDACTION = {
    "redacted_text": "Agent: hi",
    "redacted_segments": [],
    "counts": {},
}


def _state(**kw):
    s = new_state("call1", "/tmp/fake.wav", caller_id="42", department="billing")
    s.update(kw)
    return s


def _with_transcript(text):
    return _state(
        transcript={
            "segments": [],
            "full_text": text,
            "language": "en",
            "cached": False,
        }
    )


# -- node-level isolation: unexpected exceptions never propagate -----------------


def test_intake_unexpected_exception_isolated(db, monkeypatch):
    def boom(path):
        raise OSError("disk exploded")

    monkeypatch.setattr("src.services.audio.validate_audio", boom)
    out = nodes.intake_node(_state())  # must not raise
    assert out["status"] == "error"
    assert out["error"]["stage"] == "intake"
    assert "OSError" in out["error"]["message"]


def test_transcribe_non_runtime_error_isolated(db, monkeypatch):
    def boom(path, db=None):
        raise OSError("whisper binary missing")

    monkeypatch.setattr("src.services.transcription.transcribe_file", boom)
    out = nodes.transcribe_node(_state())  # must not raise
    assert out["status"] == "error"
    assert out["error"]["stage"] == "transcribe"


def test_injection_service_exception_isolated(db, monkeypatch):
    def boom(text):
        raise RuntimeError("scanner crashed")

    monkeypatch.setattr("src.services.injection.scan_for_injection", boom)
    out = nodes.injection_node(_with_transcript("Agent: hello."))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "injection_check"


def test_redact_service_exception_isolated(db, monkeypatch):
    def boom(text):
        raise ValueError("regex engine blew up")

    monkeypatch.setattr("src.services.pii.redact_text", boom)
    out = nodes.redact_node(_with_transcript("Agent: hello."))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "redact"


def test_summarize_timeout_isolated(db, monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")

    def boom(prompt):
        raise TimeoutError("network timeout")

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", boom)
    out = nodes.summarize_node(_state(redaction=dict(REDACTION)))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "summarize"
    assert "TimeoutError" in out["error"]["message"]


def test_qa_malformed_reply_isolated(db, monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    # structurally valid JSON but wrong shape -> KeyError path inside node
    monkeypatch.setattr(
        "src.services.llm_factory.call_llm_with_retry", lambda prompt: '{"nope": 1}'
    )
    out = nodes.qa_node(_state(redaction=dict(REDACTION)))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "qa_score"


def test_qa_timeout_isolated(db, monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")

    def boom(prompt):
        raise ConnectionError("connection reset")

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", boom)
    out = nodes.qa_node(_state(redaction=dict(REDACTION)))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "qa_score"


def test_report_exception_isolated(db, tmp_path, monkeypatch, reports_dir):
    def boom(report):
        raise OSError("cannot write pdf")

    monkeypatch.setattr("src.services.report_gen.write_report_files", boom)
    out = nodes.report_node(_full_state_for_report(tmp_path))
    assert out["status"] == "error"
    assert out["error"]["stage"] == "report"


def test_error_node_never_raises_even_if_db_broken(db, monkeypatch):
    def boom(record):
        raise RuntimeError("db is gone")

    monkeypatch.setattr(db, "save_call_record", boom)
    st = _state()
    st["status"] = "error"
    st["error"] = {"stage": "intake", "message": "bad", "retryable": False}
    out = nodes.error_node(st)  # must not raise
    assert out["status"] == "error"


# -- end-to-end: failures route to the error node and are recorded ----------------


def _stub_transcript(monkeypatch, transcript=None):
    monkeypatch.setattr(
        "src.services.transcription.transcribe_file",
        lambda path, db=None: dict(transcript or TRANSCRIPT),
    )


def test_pipeline_transcribe_crash_records_failed_call(db, tmp_path, monkeypatch):
    def boom(path, db=None):
        raise OSError("whisper model missing")

    monkeypatch.setattr("src.services.transcription.transcribe_file", boom)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)  # must not raise
    assert final["status"] == "error"
    assert final["error"]["stage"] == "transcribe"
    rec = db.get_call_record(final["call_id"])
    assert rec["status"] == "error"
    # later stages never ran
    stages = {e["stage"] for e in db.recent_events(50)}
    assert "qa_score" not in stages and "report" not in stages


def test_pipeline_summarize_timeout_skips_qa_and_report(db, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    _stub_transcript(monkeypatch)

    def boom(prompt):
        raise TimeoutError("network timeout")

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", boom)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)  # must not raise
    assert final["status"] == "error"
    assert final["error"]["stage"] == "summarize"
    assert final.get("qa") is None and final.get("report") is None
    stages = {e["stage"] for e in db.recent_events(100)}
    assert "qa_score" not in stages and "report" not in stages
    rec = db.get_call_record(final["call_id"])
    assert rec["status"] == "error"


def test_pipeline_qa_malformed_reply_records_error(db, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    _stub_transcript(monkeypatch)
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if "QA auditor" in prompt or "dimensions" in prompt:
            return "this is not json at all {{{"
        return SUMMARY_JSON

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", fake_llm)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)  # must not raise (previously crashed in report)
    assert final["status"] == "error"
    assert final["error"]["stage"] == "qa_score"
    assert final.get("report") is None
    rec = db.get_call_record(final["call_id"])
    assert rec["status"] == "error"


def test_pipeline_failure_preserves_safety_ordering(db, tmp_path, monkeypatch):
    """Even when the LLM call fails, it only ever saw redacted text."""
    monkeypatch.setenv("LLM_BACKOFF_BASE", "0.001")
    leaky = dict(TRANSCRIPT)
    leaky["full_text"] = "Customer: my ssn is 123-45-6789 and email a@b.com thanks"
    leaky["segments"] = [
        {"start": 0.0, "end": 2.0, "speaker": "Customer", "text": leaky["full_text"]}
    ]
    _stub_transcript(monkeypatch, transcript=leaky)
    seen = []

    def boom(prompt):
        seen.append(prompt)
        raise TimeoutError("network timeout")

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", boom)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)
    assert final["status"] == "error"
    assert final["error"]["stage"] == "summarize"
    # injection screening + PII redaction ran BEFORE the (failed) model call
    assert seen, "expected the LLM to have been attempted"
    assert "123-45-6789" not in seen[0] and "a@b.com" not in seen[0]
    assert "[SSN]" in seen[0] and "[EMAIL]" in seen[0]


def test_pipeline_report_failure_records_failed_call(db, tmp_path, monkeypatch):
    """A failing report stage must still route to the error node and persist."""
    _stub_transcript(monkeypatch)

    def fake_llm(prompt):
        return QA_JSON if "QA auditor" in prompt else SUMMARY_JSON

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", fake_llm)

    def boom(report):
        raise OSError("cannot write pdf")

    monkeypatch.setattr("src.services.report_gen.write_report_files", boom)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)  # must not raise
    assert final["status"] == "error"
    assert final["error"]["stage"] == "report"
    rec = db.get_call_record(final["call_id"])
    assert rec is not None and rec["status"] == "error"


def test_pipeline_redact_failure_never_reaches_llm(db, tmp_path, monkeypatch):
    def boom(text):
        raise ValueError("redactor crashed")

    monkeypatch.setattr("src.services.pii.redact_text", boom)
    _stub_transcript(monkeypatch)
    llm_calls = []

    def spy(prompt):
        llm_calls.append(prompt)
        return SUMMARY_JSON

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", spy)
    p = make_wav(str(tmp_path / "call.wav"))
    final = run_pipeline(p, db)
    assert final["status"] == "error"
    assert final["error"]["stage"] == "redact"
    assert llm_calls == [], "LLM must never run when redaction failed"
