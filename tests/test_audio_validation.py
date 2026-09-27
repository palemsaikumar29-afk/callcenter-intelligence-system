"""Audio intake validation: magic bytes, limits, metadata PII scan."""


from src.services import audio as audio_svc
from tests.conftest import make_wav


def _wav_bytes(seconds=1.0):
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(seconds * 8000))
    return buf.getvalue()


def test_detect_wav():
    assert audio_svc.detect_format(_wav_bytes()[:12]) == "wav"


def test_detect_mp3_id3():
    assert audio_svc.detect_format(
        b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00") == "mp3"


def test_detect_mp3_frame_sync():
    assert audio_svc.detect_format(bytes([0xFF, 0xFB, 0x90, 0x00] + [0] * 8)) == "mp3"


def test_detect_flac():
    assert audio_svc.detect_format(b"fLaC\x00\x00\x00\x00\x00\x00\x00\x00") == "flac"


def test_detect_m4a():
    assert audio_svc.detect_format(b"\x00\x00\x00\x20ftypisom\x00\x00") == "m4a"


def test_detect_unknown():
    assert audio_svc.detect_format(b"\x00" * 12) is None


def test_detect_short_header():
    assert audio_svc.detect_format(b"RIFF") is None


def test_extension_ignored(tmp_path):
    # .txt extension but real WAV magic bytes -> accepted as wav
    p = str(tmp_path / "call.txt")
    with open(p, "wb") as f:
        f.write(_wav_bytes())
    fmt, _dur, _digest, _pii, err = audio_svc.validate_audio(p)
    assert err is None and fmt == "wav"


def test_reject_non_audio(tmp_path):
    p = str(tmp_path / "call.wav")
    with open(p, "wb") as f:
        f.write(b"definitely not audio data!!!!")
    _fmt, _dur, _digest, _pii, err = audio_svc.validate_audio(p)
    assert err is not None and "magic bytes" in err


def test_reject_missing_file(tmp_path):
    _fmt, _dur, _digest, _pii, err = audio_svc.validate_audio(
        str(tmp_path / "nope.wav"))
    assert err == "file not found"


def test_reject_too_large(tmp_path, monkeypatch):
    p = make_wav(str(tmp_path / "big.wav"), seconds=1.0)
    monkeypatch.setenv("CC_MAX_AUDIO_MB", "0.00001")  # ~10 bytes
    _fmt, _dur, _digest, _pii, err = audio_svc.validate_audio(p)
    assert err is not None and "too large" in err


def test_reject_too_long(tmp_path, monkeypatch):
    p = make_wav(str(tmp_path / "long.wav"), seconds=2.0)
    monkeypatch.setenv("CC_MAX_AUDIO_MIN", "0.0001")  # <1s of audio
    _fmt, _dur, _digest, _pii, err = audio_svc.validate_audio(p)
    assert err is not None and "too long" in err


def test_sha256_stable(tmp_path):
    p = make_wav(str(tmp_path / "a.wav"))
    assert audio_svc.sha256_of_file(p) == audio_svc.sha256_of_file(p)
    assert len(audio_svc.sha256_of_file(p)) == 64


def test_metadata_pii_email():
    hits = audio_svc.scan_metadata_pii("call_jane.doe@example.com.wav",
                                       b"RIFF\x00\x00\x00WAVE")
    assert "email" in hits


def test_metadata_pii_phone():
    hits = audio_svc.scan_metadata_pii("call.wav",
                                       b"RIFF 555-123-4567 WAVE")
    assert "phone" in hits


def test_metadata_pii_ssn():
    hits = audio_svc.scan_metadata_pii("123-45-6789.wav", b"RIFF\x00\x00\x00WAVE")
    assert "ssn" in hits


def test_metadata_clean():
    assert audio_svc.scan_metadata_pii("call_2026-09-26.wav",
                                       b"RIFF\x00\x00\x00WAVE") == []


def test_duration_wav(tmp_path):
    p = make_wav(str(tmp_path / "d.wav"), seconds=3.0)
    dur = audio_svc.audio_duration_sec(p, "wav")
    assert dur is not None and abs(dur - 3.0) < 0.5
