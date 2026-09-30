"""PII redaction — runs BEFORE any LLM call.

Removes SSNs, credit-card numbers, emails, and phone numbers from the full
transcript text AND from every diarized segment. Replacements are applied
right-to-left (descending span order) so earlier offsets are never shifted.
"""

from __future__ import annotations

import re

_PATTERNS = {
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    # SSN dictated digit-by-digit ("07-8-05-1-1-2-0"): 9 digits with optional
    # separators, as Whisper transcribes spoken digit sequences. The span
    # dedup drops it when contained in a longer card-number match.
    "ssn_dictated": re.compile(r"(?<!\d)(?:\d[-.\s]?){8}\d(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    # 13-19 digits with separators; Luhn-checked to avoid false positives
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
}

_TOKENS = {
    "ssn": "[SSN]",
    "ssn_dictated": "[SSN]",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CREDIT_CARD]",
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _find_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for kind, pat in _PATTERNS.items():
        for m in pat.finditer(text):
            if kind == "card" and not _luhn_ok(m.group(0)):
                continue
            # don't let a phone-shaped run inside a card number double-count
            spans.append((m.start(), m.end(), kind))
    # drop spans fully contained in a longer span (e.g. phone inside card)
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int, str]] = []
    for s in spans:
        if not any(k[0] <= s[0] and s[1] <= k[1] and k != s for k in kept):
            kept.append(s)
    return kept


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact PII in one string. Returns (redacted, counts_by_kind)."""
    spans = _find_spans(text)
    counts: dict[str, int] = {k: 0 for k in _PATTERNS}
    # right-to-left: descending start offset keeps earlier spans valid
    for start, end, kind in sorted(spans, key=lambda s: s[0], reverse=True):
        text = text[:start] + _TOKENS[kind] + text[end:]
        counts[kind] += 1
    return text, counts


def redact_segments(
    segments: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    """Redact PII inside every diarized segment. Returns (segments, totals)."""
    totals: dict[str, int] = {k: 0 for k in _PATTERNS}
    out = []
    for seg in segments:
        redacted, counts = redact_text(seg["text"])
        for kind, n in counts.items():
            totals[kind] += n
        out.append({**seg, "text": redacted})
    return out, totals
