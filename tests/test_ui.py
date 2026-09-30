"""Gradio UI: app builds, both tabs exist, handlers behave."""

import gradio as gr
import pytest

from src.ui import analyze_tab, observability_tab
from src.ui import app as ui_app

gradio = pytest.importorskip("gradio")


def test_build_app_returns_blocks(db):
    demo = ui_app.build_app(db)
    assert isinstance(demo, gr.Blocks)


def _all_components(block):
    yield block
    for child in getattr(block, "children", []) or []:
        yield from _all_components(child)


def test_both_tabs_present(db):
    demo = ui_app.build_app(db)
    labels = [c.label for c in _all_components(demo) if isinstance(c, gr.Tab)]
    assert any("Analyze" in (l or "") for l in labels)
    assert any("Observability" in (l or "") for l in labels)


def test_analyze_tab_components(db):
    with gr.Blocks():
        audio, _caller_id, _department, btn = analyze_tab.build_analyze_tab(db)
    assert isinstance(audio, gr.Audio)
    assert isinstance(btn, gr.Button)


def test_analyze_rejects_missing_audio(db):
    header, *_ = analyze_tab.analyze_call(None, "", "", db)
    assert "upload" in header.lower()


def test_analyze_rejects_bad_audio(db, tmp_path):
    p = str(tmp_path / "bad.wav")
    with open(p, "wb") as f:
        f.write(b"junk")
    header, *_ = analyze_tab.analyze_call(p, "", "", db)
    assert header.startswith("❌")


def test_observability_renders(db):
    md, df, ls = observability_tab.render_metrics(db)
    assert "Calls completed" in md
    assert "LangSmith" in ls
    assert list(df.columns) == [
        "event_id",
        "timestamp",
        "call_id",
        "stage",
        "status",
        "detail",
    ]


def test_observability_reflects_db(db):
    db.log_event("c9", "intake", "ok", "fine")
    _md, df, _ = observability_tab.render_metrics(db)
    assert "c9" in df["call_id"].values


def test_format_helpers():
    assert analyze_tab._fmt_transcript(None).startswith("_No transcript")
    md = analyze_tab._fmt_transcript(
        {"segments": [{"speaker": "Agent", "start": 0.0, "text": "hi"}]}
    )
    assert "**Agent**" in md
    assert analyze_tab._fmt_summary(None).startswith("_No summary")
    assert analyze_tab._fmt_qa(None).startswith("_No QA")


def test_analyze_happy_path_end_to_end(db, tmp_path, monkeypatch, reports_dir):
    """UI handler with stubbed whisper + stubbed LLM: full happy path."""
    from tests.conftest import make_wav
    from tests.test_nodes import QA_JSON, SUMMARY_JSON

    monkeypatch.setattr(
        "src.services.transcription.transcribe_file",
        lambda path, db=None: {
            "segments": [
                {"start": 0.0, "end": 2.0, "speaker": "Agent", "text": "Hello"},
                {
                    "start": 3.0,
                    "end": 5.0,
                    "speaker": "Customer",
                    "text": "My bill is wrong",
                },
            ],
            "full_text": "Agent: Hello Customer: My bill is wrong",
            "language": "en",
            "cached": False,
        },
    )

    def fake_llm(prompt):
        return QA_JSON if "QA auditor" in prompt else SUMMARY_JSON

    monkeypatch.setattr("src.services.llm_factory.call_llm_with_retry", fake_llm)
    p = make_wav(str(tmp_path / "ui.wav"))
    header, transcript_md, summary_md, qa_md, pdf_path, json_path = (
        analyze_tab.analyze_call(p, "42", "billing", db)
    )
    assert header.startswith("✅")
    assert "**Customer**" in transcript_md
    assert "Billing dispute" in summary_md
    assert "4.4" in qa_md
    assert pdf_path.endswith(".pdf") and json_path.endswith(".json")
    rec = db.get_call_record(header.split("`")[1])
    assert rec["status"] == "report"
