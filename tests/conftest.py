"""Shared fixtures for the Call Center Intelligence test suite."""

from __future__ import annotations

import os
import sys
import wave

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agents import nodes
from src.database.db import CallCenterDB


@pytest.fixture()
def db(tmp_path):
    database = CallCenterDB(str(tmp_path / "test.db"))
    nodes.set_db(database)
    yield database
    database.close()


@pytest.fixture()
def reports_dir(tmp_path, monkeypatch):
    d = str(tmp_path / "reports")
    monkeypatch.setenv("CC_REPORTS_DIR", d)
    return d


def make_wav(path: str, seconds: float = 2.0, framerate: int = 16000) -> str:
    """Write a tiny synthetic WAV (sine-free silence is fine for validation)."""
    nframes = int(seconds * framerate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * nframes)
    return path


@pytest.fixture()
def wav_file(tmp_path):
    return make_wav(str(tmp_path / "call.wav"), seconds=2.0)


SAMPLE_TRANSCRIPT = (
    "Agent: Thank you for calling Acme Support, this is Priya speaking. "
    "How can I help you today?\n"
    "Customer: Hi, my bill this month is wrong. I was charged $89.99 "
    "instead of $49.99.\n"
    "Agent: I'm sorry about that. Let me look into your account right away.\n"
    "Customer: My email is jane.doe@example.com and my number is 555-123-4567.\n"
    "Agent: I've found the error and issued a $40 credit. "
    "It will appear in 3 to 5 business days.\n"
    "Customer: That's great, thank you so much!\n"
)

SAMPLE_SEGMENTS = [
    {
        "start": 0.0,
        "end": 3.2,
        "speaker": "Agent",
        "text": "Thank you for calling Acme Support.",
    },
    {
        "start": 4.5,
        "end": 8.1,
        "speaker": "Customer",
        "text": "My bill is wrong, I was charged too much.",
    },
    {
        "start": 9.0,
        "end": 14.0,
        "speaker": "Agent",
        "text": "I've issued a $40 credit to your account.",
    },
]
