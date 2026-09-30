"""Regression: report artifacts (JSON/PDF) and the UI transcript path must
show the REDACTED transcript — raw PII may never appear under a label
that says "PII redacted".

Covers the verified defect where a PII-heavy billing call logged
redacted={'phone': 1} in the audit trail, yet the UI transcript tab and
the JSON report still carried the raw phone number. All fixtures use a
synthetic 555-01xx number (reserved for fictional use).
"""

from __future__ import annotations

import re
import zlib

from src.graph.pipeline import run_pipeline
from src.ui import analyze_tab
from tests.conftest import make_wav
from tests.test_nodes import QA_JSON, SUMMARY_JSON

RAW_PHONE = "312-555-0147"  # synthetic, fictional-use range

PII_TRANSCRIPT = {
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Agent",
            "text": "Thanks for calling billing.",
        },
        {
            "start": 3.0,
            "end": 7.0,
            "speaker": "Customer",
            "text": "My number is 312-555-0147, please fix my bill.",
        },
    ],
    "full_text": (
        "Agent: Thanks for calling billing. "
        "Customer: My number is 312-555-0147, please fix my bill."
    ),
    "language": "en",
    "cached": False,
}


def _stub(monkeypatch):
    monkeypatch.setattr(
        "src.services.transcription.transcribe_file",
        lambda path, db=None: dict(PII_TRANSCRIPT),
    )

    def fake_llm(prompt):
        if "QA auditor" in prompt or "dimensions" in prompt:
            return QA_JSON
        return SUMMARY_JSON

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", fake_llm)


def _decompressed_pdf_streams(pdf_bytes: bytes) -> bytes:
    """Decode reportlab content streams (ASCII85 + Flate) to see real text."""
    import base64

    texts = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf_bytes, re.DOTALL):
        chunk = m.group(1).rstrip(b"\r\n")
        if chunk.endswith(b"~>"):  # adobe-style terminator, no <~ prefix
            chunk = chunk[:-2]
        try:
            chunk = zlib.decompress(base64.a85decode(chunk, adobe=False))
        except Exception:  # noqa: BLE001,S110 - best-effort decoding
            pass
        texts.append(chunk)
    return b"".join(texts)


def _run(db, tmp_path, monkeypatch):
    _stub(monkeypatch)
    p = make_wav(str(tmp_path / "pii-call.wav"))
    final = run_pipeline(p, db, caller_id="7", department="billing")
    assert final["status"] == "report"
    return final


def test_report_json_artifact_has_no_raw_pii(db, tmp_path, monkeypatch, reports_dir):
    final = _run(db, tmp_path, monkeypatch)
    report = final["report"]
    # audit/redaction accounting is unchanged
    assert report["redaction_counts"]["phone"] == 1
    assert "[PHONE]" in report["transcript"]["full_text"]
    assert RAW_PHONE not in report["transcript"]["full_text"]
    seg_texts = " ".join(s["text"] for s in report["transcript"]["segments"])
    assert "[PHONE]" in seg_texts and RAW_PHONE not in seg_texts
    # the JSON file on disk (downloadable from the UI)
    with open(final["report_paths"]["json_path"], encoding="utf-8") as f:
        raw = f.read()
    assert "[PHONE]" in raw
    assert RAW_PHONE not in raw


def test_report_pdf_artifact_has_no_raw_pii(db, tmp_path, monkeypatch, reports_dir):
    final = _run(db, tmp_path, monkeypatch)
    with open(final["report_paths"]["pdf_path"], "rb") as f:
        streams = _decompressed_pdf_streams(f.read())
    assert b"[PHONE]" in streams
    assert RAW_PHONE.encode() not in streams


def test_ui_transcript_tab_shows_redacted_text(db, tmp_path, monkeypatch, reports_dir):
    final = _run(db, tmp_path, monkeypatch)
    md = analyze_tab._fmt_transcript(final["report"]["transcript"])
    assert "[PHONE]" in md
    assert RAW_PHONE not in md
