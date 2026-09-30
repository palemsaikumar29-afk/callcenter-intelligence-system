"""Audio intake validation.

Format is determined by magic bytes in the first 12 bytes of the file —
never by the file extension. Supported: WAV, MP3, FLAC, M4A.
Files over the size limit or duration limit are rejected.
File metadata (name + header text) is scanned for PII.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import wave

from .config import settings

# Magic-byte signatures: (format, matcher over first 12 bytes)
# WAV: "RIFF"...."WAVE" | MP3: "ID3" or frame sync 0xFF 0xFB/0xF3/0xF2
# FLAC: "fLaC" | M4A: ...."ftyp"
_M4A_BRANDS = (b"isom", b"iso2", b"mp41", b"mp42", b"M4A ", b"m4a ")


def detect_format(header: bytes) -> str | None:
    if len(header) < 12:
        return None
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:3] == b"ID3" or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "mp3"
    if header[:4] == b"fLaC":
        return "flac"
    if header[4:8] == b"ftyp" and header[8:12] in _M4A_BRANDS:
        return "m4a"
    return None


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _duration_via_ffprobe(path: str) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return float(out.stdout.strip()) if out.stdout.strip() else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _duration_via_wave(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError):
        return None


def audio_duration_sec(path: str, fmt: str) -> float | None:
    """Duration in seconds; ffprobe preferred, wave module fallback for WAV."""
    dur = _duration_via_ffprobe(path)
    if dur is not None:
        return dur
    if fmt == "wav":
        return _duration_via_wave(path)
    return None


# PII patterns reused for the metadata scan (lightweight subset)
_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
}


def scan_metadata_pii(filename: str, header: bytes) -> list[str]:
    """Return the PII kinds found in the file name / header text."""
    text = filename + " " + header.decode("latin-1", errors="ignore")
    return [kind for kind, pat in _PII_PATTERNS.items() if pat.search(text)]


def validate_audio(
    path: str,
) -> tuple[str | None, float | None, str, list[str], str | None]:
    """Validate an audio file.

    Returns (format, duration_sec, sha256, pii_in_metadata, error).
    error is None when the file is accepted.
    """
    if not os.path.isfile(path):
        return None, None, "", [], "file not found"
    size = os.path.getsize(path)
    if size > settings.max_audio_mb * 1024 * 1024:
        return (
            None,
            None,
            "",
            [],
            (
                f"file too large: {size / 1024 / 1024:.1f} MB "
                f"(limit {settings.max_audio_mb:.0f} MB)"
            ),
        )
    with open(path, "rb") as f:
        header = f.read(12)
    fmt = detect_format(header)
    if fmt is None:
        return (
            None,
            None,
            "",
            [],
            ("unsupported audio format: magic bytes do not match WAV/MP3/FLAC/M4A"),
        )
    duration = audio_duration_sec(path, fmt)
    if duration is None:
        return None, None, "", [], "could not determine audio duration"
    if duration > settings.max_audio_minutes * 60:
        return (
            None,
            None,
            "",
            [],
            (
                f"audio too long: {duration / 60:.1f} min "
                f"(limit {settings.max_audio_minutes:.0f} min)"
            ),
        )
    digest = sha256_of_file(path)
    pii = scan_metadata_pii(os.path.basename(path), header)
    return fmt, duration, digest, pii, None
