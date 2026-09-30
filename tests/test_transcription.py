"""Transcription service: singleton model, SHA-256 cache, diarization."""

from types import SimpleNamespace

import pytest

from src.services import transcription as tsvc
from tests.conftest import make_wav


class FakeModel:
    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.calls = 0

    def transcribe(self, path, vad_filter=True):
        self.calls += 1
        assert vad_filter is True
        segs = [
            SimpleNamespace(start=0.0, end=2.0, text="Hello, thanks for calling."),
            SimpleNamespace(start=3.5, end=5.0, text="My bill is wrong."),
            SimpleNamespace(start=5.5, end=8.0, text="Let me check that."),
        ]
        return segs, SimpleNamespace(language="en")


@pytest.fixture()
def fake_model(monkeypatch):
    tsvc.reset_model_cache()
    FakeModel.instances = 0
    model = FakeModel()
    monkeypatch.setattr(tsvc, "get_model", lambda: model)
    yield model
    tsvc.reset_model_cache()


def test_singleton_returns_cached_model(monkeypatch):
    tsvc.reset_model_cache()
    sentinel = object()
    monkeypatch.setattr(tsvc, "_MODEL", sentinel)
    assert tsvc.get_model() is sentinel  # no load attempted
    tsvc.reset_model_cache()
    assert tsvc._MODEL is None


def test_transcribe_cache_miss_runs_model(db, fake_model, tmp_path):
    p = make_wav(str(tmp_path / "c.wav"))
    out = tsvc.transcribe_file(p, db=db)
    assert fake_model.calls == 1
    assert out["cached"] is False
    assert len(out["segments"]) == 3
    assert out["language"] == "en"


def test_transcribe_cache_hit_skips_model(db, fake_model, tmp_path):
    p = make_wav(str(tmp_path / "c.wav"))
    tsvc.transcribe_file(p, db=db)
    out = tsvc.transcribe_file(p, db=db)  # identical audio
    assert fake_model.calls == 1  # model NOT called again
    assert out["cached"] is True


def test_diarization_heuristic():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.2, "end": 4.0, "text": "b"},  # 0.2s gap -> same speaker
        {"start": 6.5, "end": 8.0, "text": "c"},  # 2.5s gap -> flip
        {"start": 8.1, "end": 9.0, "text": "d"},  # 0.1s gap -> same
    ]
    out = tsvc.diarize(segs)
    assert [s["speaker"] for s in out] == ["Agent", "Agent", "Customer", "Customer"]


def test_diarization_first_is_agent():
    out = tsvc.diarize([{"start": 0.0, "end": 1.0, "text": "x"}])
    assert out[0]["speaker"] == "Agent"


def test_full_text_has_speaker_labels(db, fake_model, tmp_path):
    p = make_wav(str(tmp_path / "c.wav"))
    out = tsvc.transcribe_file(p, db=db)
    assert "Agent:" in out["full_text"] and "Customer:" in out["full_text"]


def test_detect_device_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv("CC_WHISPER_DEVICE", raising=False)
    assert tsvc._detect_device() == "cpu"


def test_detect_device_override(monkeypatch):
    monkeypatch.setenv("CC_WHISPER_DEVICE", "cuda")
    assert tsvc._detect_device() == "cuda"
    monkeypatch.setenv("CC_WHISPER_DEVICE", "cpu")
    assert tsvc._detect_device() == "cpu"
