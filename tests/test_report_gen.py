"""Report generation: JSON validity and PDF structure."""
import json

from src.services import report_gen


def _report():
    return {
        "call_id": "abc123",
        "caller_id": "42",
        "department": "billing",
        "audio_sha256": "x" * 64,
        "duration_sec": 95.5,
        "created_at": "2026-09-26T00:00:00+00:00",
        "summary": {
            "purpose": "Billing dispute",
            "key_points": ["Overcharged $40"],
            "action_items": ["Issue credit"],
            "sentiment": "neutral",
            "entities": [],
            "offline": True,
        },
        "qa": {
            "dimensions": [
                {"name": "Greeting & Professionalism", "score": 4,
                 "weight": 0.15, "rationale": "Good"},
                {"name": "Problem Resolution", "score": 4, "weight": 0.30,
                 "rationale": "Good"},
                {"name": "Communication Clarity", "score": 4, "weight": 0.20,
                 "rationale": "Good"},
                {"name": "Compliance & Policy Adherence", "score": 4,
                 "weight": 0.25, "rationale": "Good"},
                {"name": "Empathy & Customer Experience", "score": 4,
                 "weight": 0.10, "rationale": "Good"},
            ],
            "overall_score": 4.0,
            "compliance_flags": [],
            "offline": True,
        },
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Agent",
                 "text": "Hello"},
                {"start": 3.0, "end": 5.0, "speaker": "Customer",
                 "text": "My bill is wrong"},
            ],
            "full_text": "Agent: Hello Customer: My bill is wrong",
            "language": "en",
            "cached": False,
        },
        "redaction_counts": {"ssn": 0},
        "status": "report",
    }


def test_json_roundtrip():
    text = report_gen.build_report_json(_report())
    assert json.loads(text)["call_id"] == "abc123"


def test_pdf_magic_and_nonempty():
    pdf = report_gen.build_report_pdf(_report())
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1000


def test_write_report_files(tmp_path):
    paths = report_gen.write_report_files(_report(),
                                         reports_dir=str(tmp_path))
    assert paths["pdf_path"].endswith("abc123.pdf")
    assert paths["json_path"].endswith("abc123.json")
    with open(paths["json_path"]) as f:
        assert json.load(f)["status"] == "report"
