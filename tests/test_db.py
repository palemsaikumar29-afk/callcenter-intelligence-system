"""SQLite layer: schema, append-only audit log, cache, records, metrics."""

import pytest


def test_tables_created(db):
    cur = db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in cur.fetchall()}
    assert {"call_records", "audit_log", "transcription_cache"} <= names


def test_audit_log_append_only(db):
    db.log_event("c1", "intake", "started")
    db.log_event("c1", "intake", "ok", "fine")
    events = db.recent_events(10)
    assert len(events) == 2
    assert events[0]["stage"] == "intake" and events[0]["status"] == "ok"
    assert events[0]["event_id"] > events[1]["event_id"]  # newest first


def test_recent_events_limit(db):
    for i in range(25):
        db.log_event("c1", "s", "ok", str(i))
    assert len(db.recent_events(20)) == 20


def test_cache_put_get_roundtrip(db):
    t = {"segments": [], "full_text": "hello", "language": "en", "cached": False}
    assert db.cache_get("abc") is None
    db.cache_put("abc", t)
    assert db.cache_get("abc")["full_text"] == "hello"


def test_cache_overwrite(db):
    db.cache_put("k", {"full_text": "one"})
    db.cache_put("k", {"full_text": "two"})
    assert db.cache_get("k")["full_text"] == "two"


def test_save_and_get_call_record(db):
    db.save_call_record(
        {
            "call_id": "r1",
            "caller_id": "42",
            "department": "billing",
            "audio_sha256": "s" * 64,
            "duration_sec": 12.5,
            "status": "report",
            "summary": {"purpose": "p"},
            "qa": {"overall_score": 4.0, "compliance_flags": ["x", "y"]},
            "transcript": {"full_text": "t"},
            "report": {"call_id": "r1"},
            "report_pdf_path": "/tmp/r1.pdf",
        }
    )
    rec = db.get_call_record("r1")
    assert rec["status"] == "report"
    assert rec["summary_json"]["purpose"] == "p"
    assert rec["report_pdf_path"] == "/tmp/r1.pdf"


def test_get_missing_record(db):
    assert db.get_call_record("nope") is None


def test_metrics_aggregation(db):
    def rec(cid, status, score=None, flags=()):
        db.save_call_record(
            {
                "call_id": cid,
                "audio_sha256": "s" * 64,
                "duration_sec": 1.0,
                "status": status,
                "qa": (
                    {"overall_score": score, "compliance_flags": list(flags)}
                    if score is not None
                    else None
                ),
            }
        )

    rec("a", "report", 4.0, ["late greeting"])
    rec("b", "report", 5.0, [])
    rec("c", "supervisor_review", 2.0, ["CRITICAL: data leak"])
    rec("d", "error")
    m = db.metrics()
    assert m["total_completed"] == 2
    assert m["total_failed"] == 1
    assert m["total_flagged"] == 1
    assert m["success_rate_pct"] == 50.0
    assert m["avg_qa_score"] == pytest.approx(3.67, abs=0.01)
    assert m["total_compliance_flags"] == 2


def test_metrics_empty_db(db):
    m = db.metrics()
    assert m["total_completed"] == 0 and m["avg_qa_score"] is None
    assert m["success_rate_pct"] == 0.0
