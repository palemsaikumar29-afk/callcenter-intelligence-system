"""Prompt-injection detection.

Scans transcript text against 24+ malicious-instruction patterns BEFORE any
LLM call is made. A match blocks the input entirely — nothing tainted ever
reaches the language model.
"""

from __future__ import annotations

import re

# (pattern name, compiled regex) — all case-insensitive
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore-instructions",
        re.compile(
            r"ignor(e|ing)\s+(all\s+|previous\s+|prior\s+|above\s+)*instructions?",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard-instructions",
        re.compile(
            r"disregard\s+(all\s+|previous\s+|prior\s+|above\s+)*(instructions?|rules?|directives?)",
            re.IGNORECASE,
        ),
    ),
    (
        "forget-instructions",
        re.compile(
            r"forget\s+(all\s+|your\s+|previous\s+|prior\s+)*(instructions?|rules?|training)",
            re.IGNORECASE,
        ),
    ),
    (
        "override-instructions",
        re.compile(
            r"override\s+(your\s+|the\s+)?(instructions?|rules?|safety|guardrails?)",
            re.IGNORECASE,
        ),
    ),
    (
        "system-prompt-reveal",
        re.compile(
            r"(reveal|show|display|print|output|repeat|disclose)\s+"
            r"(me\s+)?(your\s+|the\s+)?(system\s+)?"
            r"(prompt|instructions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system-prompt-label",
        re.compile(r"(^|\n)\s*(###\s*system|\[system\]|<system>)\b", re.IGNORECASE),
    ),
    (
        "new-instructions",
        re.compile(
            r"\b(new|updated|revised|corrected)\s+instructions?\s*:", re.IGNORECASE
        ),
    ),
    ("you-are-now", re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE)),
    ("from-now-on", re.compile(r"\bfrom\s+now\s+on,?\s+you\b", re.IGNORECASE)),
    ("pretend-role", re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE)),
    ("roleplay-jailbreak", re.compile(r"\broleplay\s+as\b", re.IGNORECASE)),
    (
        "act-as-unrestricted",
        re.compile(
            r"\bact\s+as\s+(an?\s+)?(unrestricted|uncensored|unfiltered|evil)\b",
            re.IGNORECASE,
        ),
    ),
    ("dan-mode", re.compile(r"\bDAN\s+mode\b", re.IGNORECASE)),
    ("developer-mode", re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE)),
    ("godmode", re.compile(r"\bgod\s?-?\s?mode\b", re.IGNORECASE)),
    ("jailbreak", re.compile(r"\bjail\s?-?\s?break\b", re.IGNORECASE)),
    (
        "bypass-safety",
        re.compile(
            r"\bbypass\s+(the\s+)?(safety|filter|safeguard|guardrail|content\s+policy|moderation)\b",
            re.IGNORECASE,
        ),
    ),
    ("do-anything-now", re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE)),
    (
        "exfiltrate-data",
        re.compile(
            r"\b(exfiltrate|send|forward|email|upload)\s+(the\s+)?"
            r"(transcript|recording|PII|data|summary)\s+to\s+"
            r"(?!your\s+email\b)(?!me\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal-training",
        re.compile(
            r"\b(reveal|leak|expose)\s+(your\s+|the\s+)?(training|weights|confidential)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "repeat-above",
        re.compile(
            r"\brepeat\s+(everything|all)\s+(above|before|so\s+far)\b", re.IGNORECASE
        ),
    ),
    (
        "end-of-prompt",
        re.compile(r"\bend\s+of\s+(prompt|instructions?|transcript)\b", re.IGNORECASE),
    ),
    (
        "base64-decode",
        re.compile(
            r"\b(base64\s+decode|decode\s+this\s+base64|rot13\s+decode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction-hierarchy",
        re.compile(r"\binstruction\s+hierarchy\b", re.IGNORECASE),
    ),
    ("sudo-command", re.compile(r"\bsudo\s+(mode|execute|run)\b", re.IGNORECASE)),
    (
        "confidential-override",
        re.compile(
            r"\bthis\s+is\s+(a\s+)?(confidential|classified|secret)\s+override\b",
            re.IGNORECASE,
        ),
    ),
]

PATTERN_NAMES: list[str] = [name for name, _ in _PATTERNS]
assert len(PATTERN_NAMES) >= 22, "spec requires 22+ injection patterns"


def scan_for_injection(text: str) -> tuple[bool, list[str]]:
    """Return (is_malicious, matched_pattern_names)."""
    matched = [name for name, pat in _PATTERNS if pat.search(text)]
    return bool(matched), matched
