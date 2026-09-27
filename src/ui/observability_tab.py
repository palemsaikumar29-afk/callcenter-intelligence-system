"""Observability tab: pipeline metrics dashboard.

Auto-refreshes when the tab is selected, plus a manual refresh button.
Shows totals, success rate, average QA score, compliance flags, the 20 most
recent audit-log events, and LangSmith configuration status.
"""
from __future__ import annotations

import gradio as gr
import pandas as pd

from ..database.db import CallCenterDB
from ..services.config import settings


def render_metrics(db: CallCenterDB) -> tuple[str, pd.DataFrame, str]:
    m = db.metrics()
    avg = m["avg_qa_score"]
    md = (
        "## Pipeline metrics\n\n"
        f"- **Calls completed:** {m['total_completed']}\n"
        f"- **Calls failed:** {m['total_failed']}\n"
        f"- **Flagged for supervisor review:** {m['total_flagged']}\n"
        f"- **Success rate:** {m['success_rate_pct']}%\n"
        f"- **Average QA score:** {avg if avg is not None else 'n/a'}\n"
        f"- **Total compliance flags:** {m['total_compliance_flags']}\n"
    )
    events = db.recent_events(20)
    df = pd.DataFrame(events, columns=["event_id", "timestamp", "call_id",
                                       "stage", "status", "detail"])
    langsmith = ("✅ configured" if settings.langsmith_configured
                 else "⚠️ not configured (set LANGSMITH_API_KEY)")
    return md, df, f"**LangSmith:** {langsmith}"


def build_observability_tab(db: CallCenterDB) -> tuple[gr.Tab, gr.Button]:
    tab = gr.Tab("📊 Observability")
    with tab:
        gr.Markdown("Live view of pipeline health. Refreshes automatically "
                    "when you open this tab.")
        refresh_btn = gr.Button("🔄 Refresh metrics")
        metrics_md = gr.Markdown()
        langsmith_md = gr.Markdown()
        gr.Markdown("### Recent audit-log events (latest 20)")
        events_df = gr.Dataframe(label="Audit log", wrap=True)

        def _refresh():
            return render_metrics(db)

        refresh_btn.click(fn=_refresh,
                          outputs=[metrics_md, events_df, langsmith_md])
        tab.select(fn=_refresh,
                   outputs=[metrics_md, events_df, langsmith_md])

        # initial paint
        md0, df0, ls0 = render_metrics(db)
        metrics_md.value = md0
        events_df.value = df0
        langsmith_md.value = ls0
    return tab, refresh_btn
