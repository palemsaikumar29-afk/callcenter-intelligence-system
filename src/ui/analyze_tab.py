"""Analyze Call tab: upload audio (file or mic), run the 7-stage pipeline,
and review transcript, summary, QA scorecard, plus PDF/JSON downloads."""

from __future__ import annotations

import gradio as gr

from ..database.db import CallCenterDB
from ..graph.pipeline import run_pipeline


def _fmt_transcript(transcript: dict | None) -> str:
    if not transcript:
        return "_No transcript available._"
    lines = []
    for seg in transcript.get("segments", []):
        lines.append(f"**{seg['speaker']}** [{seg['start']:.1f}s]: {seg['text']}")
    return "\n\n".join(lines)


def _fmt_summary(summary: dict | None) -> str:
    if not summary:
        return "_No summary available._"
    md = [f"**Purpose:** {summary.get('purpose', '')}", "", "**Key points:**"]
    md += [f"- {kp}" for kp in summary.get("key_points", [])] or ["- —"]
    md += ["", "**Action items:**"]
    md += [f"- [ ] {a}" for a in summary.get("action_items", [])] or ["- —"]
    md += ["", f"**Sentiment:** {summary.get('sentiment', '')}"]
    ents = summary.get("entities", [])
    if ents:
        md += ["", f"**Entities:** {', '.join(ents)}"]
    if summary.get("offline"):
        md = ["_Summary generated in offline mode (no LLM key)._", ""] + md
    return "\n".join(md)


def _fmt_qa(qa: dict | None) -> str:
    if not qa:
        return "_No QA score available._"
    md = [f"## Overall: {qa.get('overall_score', 0):.2f} / 5", ""]
    for d in qa.get("dimensions", []):
        md.append(
            f"- **{d['name']}** ({d['weight']:.0%}): "
            f"{d['score']}/5 — {d.get('rationale', '')}"
        )
    flags = qa.get("compliance_flags", [])
    md += ["", f"**Compliance flags:** {', '.join(flags) if flags else 'none'}"]
    if qa.get("offline"):
        md = ["_QA scored in offline mode (no LLM key)._", ""] + md
    return "\n".join(md)


def analyze_call(
    audio_path: str | None, caller_id: str, department: str, db: CallCenterDB
):
    if not audio_path:
        return (
            "⚠️ Please upload an audio file or record with the microphone.",
            "",
            "",
            "",
            None,
            None,
        )
    try:
        final = run_pipeline(
            audio_path, db, caller_id=caller_id or None, department=department or None
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in UI
        return (f"❌ Pipeline failed: {exc}", "", "", "", None, None)

    status = final.get("status")
    if status == "error":
        err = (final.get("error") or {}).get("message", "unknown error")
        return (f"❌ Analysis failed: {err}", "", "", "", None, None)

    report = final.get("report", {})
    paths = final.get("report_paths", {})
    header = (
        f"✅ Analysis complete — `{report.get('call_id', '')}`  "
        f"({report.get('duration_sec', 0):.1f}s)"
    )
    if status == "supervisor_review":
        header += "  ⚠️ **Flagged for supervisor review** (compliance)."
    return (
        header,
        _fmt_transcript(report.get("transcript")),
        _fmt_summary(report.get("summary")),
        _fmt_qa(report.get("qa")),
        paths.get("pdf_path"),
        paths.get("json_path"),
    )


def build_analyze_tab(
    db: CallCenterDB,
) -> tuple[gr.Audio, gr.Textbox, gr.Textbox, gr.Button]:
    with gr.Tab("🔍 Analyze Call"):
        gr.Markdown(
            "Upload a call recording (WAV/MP3/FLAC/M4A, ≤50 MB, "
            "≤60 min) or record with your microphone."
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["upload", "microphone"], type="filepath", label="Call audio"
            )
        with gr.Row():
            caller_id = gr.Textbox(label="Caller ID (optional)")
            department = gr.Textbox(label="Department (optional)")
        btn = gr.Button("▶ Analyze Call", variant="primary")
        status_md = gr.Markdown()
        with gr.Tab("📝 Transcript"):
            transcript_md = gr.Markdown()
        with gr.Tab("🧾 Summary"):
            summary_md = gr.Markdown()
        with gr.Tab("⭐ QA Scorecard"):
            qa_md = gr.Markdown()
        with gr.Row():
            pdf_file = gr.File(label="Download PDF report")
            json_file = gr.File(label="Download JSON report")

        btn.click(
            fn=lambda a, c, d: analyze_call(a, c, d, db),
            inputs=[audio, caller_id, department],
            outputs=[status_md, transcript_md, summary_md, qa_md, pdf_file, json_file],
        )
    return audio, caller_id, department, btn
