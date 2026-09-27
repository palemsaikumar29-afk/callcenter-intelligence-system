"""Call report generation: structured JSON + PDF via ReportLab.

Reports are written under the configured reports directory and the paths
are persisted on the call record in SQLite.
"""
from __future__ import annotations

import io
import json
import os
from typing import Any

from .config import settings


def build_report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


def build_report_pdf(report: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    def p(text: str, style: str = "Normal"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 6))

    summary = report.get("summary", {})
    qa = report.get("qa", {})
    transcript = report.get("transcript", {})

    p(f"Call Intelligence Report — {report.get('call_id', '')}", "Title")
    meta = (
        f"Created: {report.get('created_at', '')}<br/>"
        f"Caller ID: {report.get('caller_id') or 'n/a'} &nbsp;|&nbsp; "
        f"Department: {report.get('department') or 'n/a'}<br/>"
        f"Duration: {report.get('duration_sec', 0):.1f}s &nbsp;|&nbsp; "
        f"Status: {report.get('status', '')}"
    )
    p(meta)

    p("Summary", "Heading2")
    p(f"<b>Purpose:</b> {summary.get('purpose', '')}")
    p("<b>Key points:</b><br/>" + "<br/>".join(
        f"• {kp}" for kp in summary.get("key_points", [])) or "—")
    p("<b>Action items:</b><br/>" + "<br/>".join(
        f"• {a}" for a in summary.get("action_items", [])) or "—")
    p(f"<b>Sentiment:</b> {summary.get('sentiment', '')}")

    p("QA Scorecard", "Heading2")
    rows = [["Dimension", "Score (1-5)", "Weight", "Rationale"]]
    for d in qa.get("dimensions", []):
        rows.append([d["name"], str(d["score"]),
                     f"{d['weight']:.0%}", d.get("rationale", "")])
    rows.append(["OVERALL", f"{qa.get('overall_score', 0):.2f}", "100%", ""])
    table = Table(rows, colWidths=[2.2 * inch, 1.0 * inch, 0.8 * inch, 3.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))
    flags = qa.get("compliance_flags", [])
    p(f"<b>Compliance flags:</b> {', '.join(flags) if flags else 'none'}")

    p("Transcript (PII redacted)", "Heading2")
    for seg in transcript.get("segments", []):
        p(f"<b>{seg.get('speaker', '?')} "
          f"[{seg.get('start', 0):.1f}s]:</b> {seg.get('text', '')}")

    doc.build(story)
    return buf.getvalue()


def write_report_files(report: dict[str, Any],
                       reports_dir: str = "") -> dict[str, str]:
    """Write <call_id>.json and <call_id>.pdf; return their paths."""
    target = reports_dir or settings.reports_dir
    os.makedirs(target, exist_ok=True)
    call_id = report["call_id"]
    json_path = os.path.join(target, f"{call_id}.json")
    pdf_path = os.path.join(target, f"{call_id}.pdf")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(build_report_json(report))
    with open(pdf_path, "wb") as f:
        f.write(build_report_pdf(report))
    return {"json_path": json_path, "pdf_path": pdf_path}
