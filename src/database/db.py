"""SQLite persistence: call_records, audit_log, transcription_cache.

The audit log is append-only — events are inserted, never updated or deleted.
All access is serialized through a module-level lock so the Gradio server's
threads share one connection safely.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS call_records (
    call_id        TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    caller_id      TEXT,
    department     TEXT,
    audio_sha256   TEXT NOT NULL,
    duration_sec   REAL NOT NULL,
    status         TEXT NOT NULL,          -- report | supervisor_review | error
    summary_json   TEXT,
    qa_json        TEXT,
    transcript_json TEXT,
    report_json    TEXT,
    report_pdf_path TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    event_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    call_id   TEXT NOT NULL,
    stage     TEXT NOT NULL,
    status    TEXT NOT NULL,               -- started|ok|error|blocked|flagged
    detail    TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_call ON audit_log(call_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
CREATE TABLE IF NOT EXISTS transcription_cache (
    sha256         TEXT PRIMARY KEY,
    transcript_json TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CallCenterDB:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- audit log (append-only) -------------------------------------------------
    def log_event(self, call_id: str, stage: str, status: str, detail: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO audit_log (timestamp, call_id, stage, status, detail)"
                " VALUES (?, ?, ?, ?, ?)",
                (_utcnow(), call_id, stage, status, detail),
            )
            self._conn.commit()
            return cur.lastrowid

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_id, timestamp, call_id, stage, status, detail"
                " FROM audit_log ORDER BY event_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- transcription cache -----------------------------------------------------
    def cache_get(self, sha256: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT transcript_json FROM transcription_cache WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
        return json.loads(row["transcript_json"]) if row else None

    def cache_put(self, sha256: str, transcript: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO transcription_cache"
                " (sha256, transcript_json, created_at) VALUES (?, ?, ?)",
                (sha256, json.dumps(transcript), _utcnow()),
            )
            self._conn.commit()

    # -- call records ------------------------------------------------------------
    def save_call_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO call_records
                   (call_id, created_at, caller_id, department, audio_sha256,
                    duration_sec, status, summary_json, qa_json,
                    transcript_json, report_json, report_pdf_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["call_id"],
                    record.get("created_at", _utcnow()),
                    record.get("caller_id"),
                    record.get("department"),
                    record["audio_sha256"],
                    record["duration_sec"],
                    record["status"],
                    json.dumps(record.get("summary"))
                    if record.get("summary")
                    else None,
                    json.dumps(record.get("qa")) if record.get("qa") else None,
                    json.dumps(record.get("transcript"))
                    if record.get("transcript")
                    else None,
                    json.dumps(record.get("report")) if record.get("report") else None,
                    record.get("report_pdf_path"),
                ),
            )
            self._conn.commit()

    def get_call_record(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM call_records WHERE call_id = ?", (call_id,)
            ).fetchone()
        if not row:
            return None
        rec = dict(row)
        for key in ("summary_json", "qa_json", "transcript_json", "report_json"):
            rec[key] = json.loads(rec[key]) if rec[key] else None
        return rec

    # -- metrics ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) c FROM call_records"
            ).fetchone()["c"]
            completed = self._conn.execute(
                "SELECT COUNT(*) c FROM call_records WHERE status='report'"
            ).fetchone()["c"]
            failed = self._conn.execute(
                "SELECT COUNT(*) c FROM call_records WHERE status='error'"
            ).fetchone()["c"]
            flagged = self._conn.execute(
                "SELECT COUNT(*) c FROM call_records WHERE status='supervisor_review'"
            ).fetchone()["c"]
            qa_rows = self._conn.execute(
                "SELECT qa_json FROM call_records WHERE qa_json IS NOT NULL"
            ).fetchall()
        scores = []
        flags = 0
        for r in qa_rows:
            try:
                qa = json.loads(r["qa_json"])
                scores.append(float(qa["overall_score"]))
                flags += len(qa.get("compliance_flags", []))
            except (ValueError, KeyError, TypeError):
                continue
        return {
            "total_completed": completed,
            "total_failed": failed,
            "total_flagged": flagged,
            "total_calls": total,
            "success_rate_pct": round(100.0 * completed / total, 1) if total else 0.0,
            "avg_qa_score": round(sum(scores) / len(scores), 2) if scores else None,
            "total_compliance_flags": flags,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
