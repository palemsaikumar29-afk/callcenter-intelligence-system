# Expected behavior — Call Center Intelligence System

How each sample input should flow through the 7-stage pipeline.

## sample_transcript_1_billing.txt (PII: email)
- Intake: valid (once rendered to audio; as text it exercises stages 3+).
- Injection check: clean — no malicious patterns.
- Redaction: `maria.santos@example.com` → `[EMAIL]` in the full text AND in
  every segment. `counts["email"] == 1`.
- Summarization: purpose mentions evening internet drops; key points include
  the line test finding and the free technician visit; sentiment neutral or
  positive.
- QA: overall score in a sane 1–5 range; no compliance flags expected.

## sample_transcript_2_compliance.txt (PII: phone + SSN)
- Redaction: `312-555-0147` → `[PHONE]`, `078-05-1120` → `[SSN]`.
- QA: the agent refusing to confirm the SSN on a recorded line is a positive
  compliance signal — "Compliance & Policy Adherence" should score 4–5 and no
  CRITICAL flag should be raised.

## sample_transcript_3_injection.txt (prompt injection)
- Injection check: MUST match the `ignore-instructions` pattern.
- Pipeline outcome: `status == "error"`, error message contains
  "prompt injection blocked".
- The LLM must NEVER be called for this input (no summary, no QA).
- Audit log contains an `injection_check` event with status `blocked`.

## sample_call.wav (synthetic audio, generated — not committed)
Generate a small synthetic WAV for intake/transcription tests:
`python -c "from tests.conftest import make_wav; make_wav('test_data/sample_call.wav', seconds=5)"`
- Intake: format `wav`, duration ≈ 5s, SHA-256 stable across runs.
- Transcription of silence: faster-whisper tiny returns few/no segments with
  VAD filtering; the pipeline must not crash on empty transcripts.

## Global invariants
- PII (SSN/card/email/phone) NEVER appears in any prompt sent to the LLM —
  verified by `test_pipeline_pii_never_reaches_llm`.
- The QA `overall_score` is always recomputed in Python from the weighted
  dimensions — the LLM's own overall value is discarded.
- Every stage writes to the append-only audit log, including failures.
- No API keys or model weights are committed; `data/` and `reports/` are
  git-ignored.
