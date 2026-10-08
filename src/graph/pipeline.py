"""LangGraph pipeline: 7 sequential stages with conditional routing.

Flow:
    intake -> (ok) transcribe -> (ok) injection_check -> (clean) redact
           -> (ok) summarize -> (ok) qa_score -> (clean|flagged) report -> END
    any stage -> (error) error -> END

Terminal outcomes: report | supervisor_review | error.
Every stage is gated on the error status: a stage that records a failure
routes straight to the terminal error node, which persists the failed call
record — no stage can crash the graph, and a failed stage never lets later
stages run on its missing output.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from ..agents import nodes
from ..database.db import CallCenterDB
from .state import PipelineState, new_state


def _after_stage(ok_target: str):
    """Router factory: go to the error node if the stage failed, else continue.

    Applied after EVERY stage so a recorded failure can never flow into a
    later stage that would run on its missing output.
    """

    def _router(state: PipelineState) -> str:
        return "error" if state.get("status") == "error" else ok_target

    _router.__name__ = f"after_stage_to_{ok_target}"
    return _router


def _after_qa(
    state: PipelineState,
) -> Literal["error", "report_clean", "report_flagged"]:
    # A failed QA stage must not reach the report node (the summary/score it
    # needs is missing). Critical compliance flag -> supervisor review; the
    # report is still generated so the supervisor has the full scorecard.
    if state.get("status") == "error":
        return "error"
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
        "intake",
        _after_stage("transcribe"),
        {"transcribe": "transcribe", "error": "error"},
    )
    g.add_conditional_edges(
        "transcribe",
        _after_stage("injection_check"),
        {"injection_check": "injection_check", "error": "error"},
    )
    g.add_conditional_edges(
        "injection_check",
        _after_stage("redact"),
        {"redact": "redact", "error": "error"},
    )
    g.add_conditional_edges(
        "redact",
        _after_stage("summarize"),
        {"summarize": "summarize", "error": "error"},
    )
    g.add_conditional_edges(
        "summarize",
        _after_stage("qa_score"),
        {"qa_score": "qa_score", "error": "error"},
    )
    g.add_conditional_edges(
        "qa_score",
        _after_qa,
        {"report_clean": "report", "report_flagged": "report", "error": "error"},
    )
    g.add_conditional_edges(
        "report",
        _after_stage(END),
        {END: END, "error": "error"},
    )
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
