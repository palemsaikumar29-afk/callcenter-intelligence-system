"""Gradio app assembly: two tabs — Analyze Call + Observability."""

from __future__ import annotations

import gradio as gr

from ..database.db import CallCenterDB
from .analyze_tab import build_analyze_tab
from .observability_tab import build_observability_tab


def build_app(db: CallCenterDB) -> gr.Blocks:
    with gr.Blocks(title="Call Center Intelligence") as demo:
        gr.Markdown("# 📞 Call Center Intelligence System")
        gr.Markdown(
            "Seven-stage LangGraph pipeline: intake → transcription → "
            "injection check → PII redaction → summarization → "
            "QA scoring → report."
        )
        build_analyze_tab(db)
        build_observability_tab(db)
    return demo
