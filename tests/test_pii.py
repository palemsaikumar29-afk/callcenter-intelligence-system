"""PII redaction: right-to-left replacement in text and segments."""

from src.services import pii as pii_svc

CARD = "4111 1111 1111 1111"  # Luhn-valid test Visa number


def test_redact_ssn():
    out, counts = pii_svc.redact_text("my ssn is 123-45-6789 ok")
    assert out == "my ssn is [SSN] ok"
    assert counts["ssn"] == 1


def test_redact_email():
    out, counts = pii_svc.redact_text("mail me at jane.doe@example.com today")
    assert "[EMAIL]" in out and "jane.doe@example.com" not in out
    assert counts["email"] == 1


def test_redact_phone_dashes():
    out, counts = pii_svc.redact_text("call 555-123-4567 now")
    assert "[PHONE]" in out and counts["phone"] == 1


def test_redact_phone_parens():
    out, _counts = pii_svc.redact_text("call (555) 123-4567 now")
    assert "[PHONE]" in out


def test_redact_credit_card():
    out, counts = pii_svc.redact_text(f"card number {CARD} please")
    assert "[CREDIT_CARD]" in out and CARD not in out
    assert counts["card"] == 1


def test_card_requires_luhn():
    _out, counts = pii_svc.redact_text("order 1234 5678 9012 3456 done")
    assert counts["card"] == 0  # fails Luhn -> not a real card


def test_multiple_kinds_right_to_left():
    text = f"ssn 123-45-6789, card {CARD}, mail a@b.com, phone 555-123-4567."
    out, counts = pii_svc.redact_text(text)
    assert out == ("ssn [SSN], card [CREDIT_CARD], mail [EMAIL], phone [PHONE].")
    assert sum(counts.values()) == 4


def test_no_pii_unchanged():
    text = "Thank you for calling support today."
    out, counts = pii_svc.redact_text(text)
    assert out == text and sum(counts.values()) == 0


def test_redact_segments():
    segs = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Agent",
            "text": "Your SSN 123-45-6789 please",
        },
        {
            "start": 3.0,
            "end": 5.0,
            "speaker": "Customer",
            "text": "It's fine, no PII here",
        },
    ]
    out, totals = pii_svc.redact_segments(segs)
    assert out[0]["text"] == "Your SSN [SSN] please"
    assert out[0]["speaker"] == "Agent"  # metadata preserved
    assert out[1]["text"] == "It's fine, no PII here"
    assert totals["ssn"] == 1


def test_empty_string():
    out, counts = pii_svc.redact_text("")
    assert out == "" and sum(counts.values()) == 0
