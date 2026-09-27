"""Call Center Intelligence System — Gradio UI.

Expected interface (per instructor guidance): Gradio chat + observability
dashboard. This is the Phase 1 shell — agent logic lands in Phase 2 once the
official course specification is delivered.

Keys via environment (all optional — the app degrades gracefully):
    OPENAI_API_KEY   LLM agents
    TAVILY_API_KEY   web lookup fallback (if the spec requires it)
    SERPAPI_API_KEY  web lookup fallback (if the spec requires it)
"""
from __future__ import annotations

import gradio as gr

import callcenter


def phase1_notice() -> str:
    return (
        f"Call Center Intelligence System v{callcenter.__version__} — "
        "Phase 1 scaffold. Agent logic arrives in Phase 2 "
        "(waiting on the official course specification)."
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Call Center Intelligence") as demo:
        gr.Markdown("# 📞 Call Center Intelligence System")
        gr.Markdown(phase1_notice())
        gr.Textbox(label="Phase 1 status", value=phase1_notice(),
                   interactive=False)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
