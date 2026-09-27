# Call Center Intelligence System

Capstone Agent 3 — a multi-stage LangGraph pipeline that turns raw call-center
audio into redacted transcripts, structured summaries, QA scorecards, and
PDF/JSON reports, with an append-only audit log and a Gradio observability
dashboard.

## Architecture

```
audio upload / mic
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ LangGraph pipeline (src/graph/pipeline.py)                  │
│  1. intake          magic-byte validation (WAV/MP3/FLAC/M4A),│
│                     ≤50 MB / ≤60 min, metadata PII scan     │
│  2. transcribe      faster-whisper (int8, VAD), heuristic   │
│                     diarization, SHA-256 cache in SQLite     │
│  3. injection_check 26 prompt-injection patterns — BEFORE   │
│                     any LLM call; blocks tainted input       │
│  4. redact          SSN/card/email/phone removed from text  │
│                     AND every segment — BEFORE any LLM call  │
│  5. summarize       structured LLM summary (Pydantic)        │
│  6. qa_score        5 weighted dimensions; overall score     │
│                     recomputed deterministically in Python   │
│  7. report          ReportLab PDF + JSON, persisted          │
└─────────────────────────────────────────────────────────────┘
      │                                    │
      ▼                                    ▼
 terminal: report ─────────► terminal: supervisor_review (critical
 terminal: error                        compliance flag)
```

Terminal outcomes: `report`, `supervisor_review`, `error`. Failures are
isolated to the error node — no stage can crash the graph.

### Five layers (`src/`)

| Layer | Contents |
|---|---|
| `ui/` | Gradio 5.x app: **Analyze Call** tab (upload/mic, transcript, summary, QA scorecard, PDF/JSON downloads) + **Observability** tab (metrics, audit-log table, LangSmith status) |
| `services/` | config (env-only), multi-provider LLM factory, faster-whisper singleton, PII redaction, injection detection, audio validation, ReportLab reports |
| `agents/` | 14 typed Pydantic contracts + the 7 pipeline stage nodes |
| `graph/` | `PipelineState` TypedDict + LangGraph pipeline with conditional routing |
| `database/` | SQLite: `call_records`, append-only `audit_log`, `transcription_cache` |

## LLM providers

One env var switches providers — OpenAI GPT-4o, Gemini 2.0 Flash, or Groq
Llama 3.3 70B. Provider, model name, API key, and timeout all come from the
environment (see `.env.example`); nothing is hardcoded. With no key set, the
pipeline runs in clearly-labelled offline mode (extractive summary, neutral
QA scores). LLM calls retry with exponential backoff (up to 3 attempts).

## Quickstart

```bash
make install          # create .venv and install pinned requirements
cp .env.example .env  # fill in at least one LLM API key
make run              # serves at http://localhost:7860
```

Or with Docker:

```bash
docker build -t callcenter-intel .
docker run -p 7860:7860 --env-file .env callcenter-intel
```

## Testing

```bash
make test      # fast suite (no network, no model download)
make test-all  # + live OpenAI smoke test (needs OPENAI_API_KEY) and app startup test
```

The suite covers magic-byte validation, all 26 injection patterns (plus
false-positive checks), PII redaction offsets, all 14 Pydantic contracts, the
SQLite layer, transcription caching, LLM provider switching and retries, every
pipeline node, full end-to-end runs (happy path, injection-blocked, PII
leak-proof, supervisor-review routing), PDF/JSON report generation, the Gradio
UI build, and a real `python app.py` startup check. Sample inputs live in
`test_data/` with `expected_behavior.md`.

## Key design decisions

- **Security ordering**: injection detection and PII redaction run strictly
  before any LLM call — verified by tests asserting the LLM is never invoked
  for blocked input and PII never appears in prompts.
- **Deterministic QA**: the LLM scores dimensions and rationales, but the
  overall score is recomputed in Python from the weights; the model's own
  overall value is discarded.
- **Whisper singleton**: the model loads once at app startup
  (`transcription.get_model()`), never inside a request handler; identical
  audio is served from the SHA-256 cache without re-running inference.
- **Heuristic diarization**: first speaker is the Agent; the speaker flips on
  >1s pauses. Documented as a heuristic — true diarization would use a
  dedicated model (e.g. pyannote) in production.
- **Append-only audit log**: every stage, retry, block, and flag is recorded;
  the Observability tab reads the 20 most recent events.

## Limitations

- Diarization is heuristic (see above).
- faster-whisper `tiny` model trades accuracy for download size/speed; set
  `CC_WHISPER_MODEL` for better quality.
- PDF export uses latin-1 core fonts (non-latin scripts are replaced).
- No real-time/streaming transcription — files are processed whole.
