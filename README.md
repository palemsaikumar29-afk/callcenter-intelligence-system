# Call Center Intelligence System

Interview Kickstart capstone, agent 3 — a multi-agent Call Center
Intelligence System (expected UI: Gradio chat + observability dashboard).

**Status: Phase 1 scaffold.** The multi-agent system is built in Phase 2
once the official course specification is delivered.

## Layout

- `callcenter/` — agent package (scaffold; logic in Phase 2)
- `app.py` — Gradio UI shell (Phase 1 placeholder)
- `tests/` — pytest suite (smoke test only in Phase 1)
- `test_data/` — sample transcripts + expected behavior (Phase 2)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -p no:cacheprovider
python app.py
```

## Secrets

All API keys come from environment variables only — never committed:

- `OPENAI_API_KEY` — LLM agents
- `TAVILY_API_KEY` — web lookup fallback (if the spec requires it)
- `SERPAPI_API_KEY` — web lookup fallback (if the spec requires it)
