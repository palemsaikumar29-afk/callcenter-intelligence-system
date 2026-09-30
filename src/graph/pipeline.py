"""LangGraph pipeline: 7 sequential stages with conditional routing.

Flow:
    intake -> (ok) transcribe -> injection_check -> (clean) redact
           -> summarize -> qa_score -> (clean|flagged) report -> END
    intake -> (invalid) error -> END
    injection_check -> (blocked) error -> END

Terminal outcomes: report | supervisor_review | error.
Failures are isolated to the error node — no stage can crash the graph.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from ..agents import nodes
from ..database.db import CallCenterDB
from .state import PipelineState, new_state


def _after_intake(state: PipelineState) -> Literal["transcribe", "error"]:
    return "error" if state.get("status") == "error" else "transcribe"


def _after_injection(state: PipelineState) -> Literal["redact", "error"]:
    return "error" if state.get("status") == "error" else "redact"


def _after_qa(state: PipelineState) -> Literal["report_clean", "report_flagged"]:
    # Critical compliance flag -> supervisor review; the report is still
    # generated so the supervisor has the full scorecard to review.
    return (
        "report_flagged"
        if state.get("status") == "supervisor_review"
        else "report_clean"
    )


def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("intake", nodes.intake_node)
    g.add_node("transcribe", nodes.transcribe_node)
    g.add_node("injection_check", nodes.injection_node)
    g.add_node("redact", nodes.redact_node)
    g.add_node("summarize", nodes.summarize_node)
    g.add_node("qa_score", nodes.qa_node)
    g.add_node("report", nodes.report_node)
    g.add_node("error", nodes.error_node)

    g.set_entry_point("intake")
    g.add_conditional_edges(
        "intake", _after_intake, {"transcribe": "transcribe", "error": "error"}
    )
    g.add_edge("transcribe", "injection_check")
    g.add_conditional_edges(
        "injection_check", _after_injection, {"redact": "redact", "error": "error"}
    )
    g.add_edge("redact", "summarize")
    g.add_edge("summarize", "qa_score")
    g.add_conditional_edges(
        "qa_score", _after_qa, {"report_clean": "report", "report_flagged": "report"}
    )
    g.add_edge("report", END)
    g.add_edge("error", END)
    return g.compile()


def run_pipeline(
    audio_path: str,
    db: CallCenterDB,
    caller_id: str | None = None,
    department: str | None = None,
) -> dict[str, Any]:
    """Run the full 7-stage pipeline for one call. Returns the final state."""
    nodes.set_db(db)
    call_id = nodes.new_call_id()
    db.log_event(call_id, "pipeline", "started", audio_path)
    graph = build_pipeline()
    final = graph.invoke(new_state(call_id, audio_path, caller_id, department))
    return dict(final)
