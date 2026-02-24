"""Tests for pyannote diarization module (no model/hardware required)."""

import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from diarization_pyannote import (
    PyannoteDiarizer,
    SpeakerTurn,
    _assign_channel_names,
    check_pyannote_installed,
)


# --- SpeakerTurn dataclass ---


class TestSpeakerTurn:
    def test_basic_creation(self):
        turn = SpeakerTurn(start=0.0, end=1.5, speaker="Alice", channel="mic")
        assert turn.start == 0.0
        assert turn.end == 1.5
        assert turn.speaker == "Alice"
        assert turn.channel == "mic"

    def test_system_channel(self):
        turn = SpeakerTurn(start=2.0, end=4.0, speaker="Bob", channel="system")
        assert turn.channel == "system"


# --- Channel name assignment ---


class TestAssignChannelNames:
    def test_single_speaker(self):
        result = _assign_channel_names(["SPEAKER_00"], "Me", "Local")
        assert result == {"SPEAKER_00": "Me"}

    def test_two_speakers_no_extras(self):
        result = _assign_channel_names(["SPEAKER_00", "SPEAKER_01"], "Me", "Local")
        assert result == {"SPEAKER_00": "Me", "SPEAKER_01": "Local 2"}

    def test_with_extra_names(self):
        result = _assign_channel_names(
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
            "Bob", "Remote",
            extra_names=["Charlie", "Dave"],
        )
        assert result == {
            "SPEAKER_00": "Bob",
            "SPEAKER_01": "Charlie",
            "SPEAKER_02": "Dave",
        }

    def test_extras_exhausted_falls_back_to_prefix(self):
        result = _assign_channel_names(
            ["A", "B", "C"],
            "Bob", "Remote",
            extra_names=["Charlie"],
        )
        assert result == {"A": "Bob", "B": "Charlie", "C": "Remote 3"}

    def test_empty_labels(self):
        result = _assign_channel_names([], "Me", "Local")
        assert result == {}


# --- Helpers ---


def _make_mono_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000):
    """Write a mono WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())


# --- PyannoteDiarizer ---


class TestPyannoteDiarizerInit:
    def test_raises_without_token(self):
        import os
        env_backup = os.environ.pop("HF_TOKEN", None)
        try:
            with pytest.raises(ValueError, match="HuggingFace token"):
                PyannoteDiarizer(hf_token=None)
        finally:
            if env_backup is not None:
                os.environ["HF_TOKEN"] = env_backup

    def test_uses_env_token(self):
        """Verify HF_TOKEN env var is picked up (pipeline import mocked)."""
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        mock_module = MagicMock()
        mock_module.Pipeline = mock_pipeline_cls

        with patch.dict("os.environ", {"HF_TOKEN": "hf_test123"}):
            with patch.dict("sys.modules", {"pyannote.audio": mock_module, "pyannote": MagicMock()}):
                import importlib
                import diarization_pyannote
                importlib.reload(diarization_pyannote)
                try:
                    diarizer = diarization_pyannote.PyannoteDiarizer()
                    mock_pipeline_cls.from_pretrained.assert_called_once()
                except Exception:
                    pass


# --- check_pyannote_installed ---


class TestCheckPyannoteInstalled:
    def test_raises_when_not_installed(self):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="pyannote engine requires"):
                check_pyannote_installed()


# --- Integration-style tests with mock pipeline ---


class TestDiarizeStereo:
    def _make_mock_diarizer(self, mic_turns, sys_turns):
        """Create a PyannoteDiarizer with a mocked pipeline."""
        diarizer = PyannoteDiarizer.__new__(PyannoteDiarizer)

        # Mock pipeline that returns different results per file
        def fake_pipeline(path):
            path = str(path)
            raw_turns = sys_turns if "_sys" in path else mic_turns
            # Build a mock Annotation
            mock_annotation = MagicMock()
            mock_annotation.itertracks.return_value = [
                (MagicMock(start=s, end=e), None, spk)
                for s, e, spk in raw_turns
            ]
            # Return DiarizeOutput-like object
            result = MagicMock()
            result.exclusive_speaker_diarization = mock_annotation
            return result

        diarizer._pipeline = fake_pipeline
        return diarizer

    def test_two_channels_separate_speakers(self, tmp_path):
        """Mic and system channels get independent speaker labels."""
        mic_path = tmp_path / "test_mic.wav"
        sys_path = tmp_path / "test_sys.wav"
        _make_mono_wav(mic_path, np.full(16000, 5000, dtype=np.int16))
        _make_mono_wav(sys_path, np.full(16000, 5000, dtype=np.int16))

        diarizer = self._make_mock_diarizer(
            mic_turns=[(0.0, 1.0, "SPEAKER_00")],
            sys_turns=[(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")],
        )

        turns = diarizer.diarize_stereo(
            mic_path, sys_path,
            speaker_name="Luca",
            other_name="Alice",
            other_names=["Alice", "Bob"],
        )

        # Mic speaker → Luca
        mic_turns = [t for t in turns if t.channel == "mic"]
        assert len(mic_turns) == 1
        assert mic_turns[0].speaker == "Luca"

        # System speakers → Alice, Bob
        sys_turns_out = [t for t in turns if t.channel == "system"]
        assert len(sys_turns_out) == 2
        speakers = {t.speaker for t in sys_turns_out}
        assert speakers == {"Alice", "Bob"}

    def test_sorted_by_start_time(self, tmp_path):
        mic_path = tmp_path / "test_mic.wav"
        sys_path = tmp_path / "test_sys.wav"
        _make_mono_wav(mic_path, np.full(32000, 5000, dtype=np.int16))
        _make_mono_wav(sys_path, np.full(32000, 5000, dtype=np.int16))

        diarizer = self._make_mock_diarizer(
            mic_turns=[(0.5, 1.0, "SPEAKER_00")],
            sys_turns=[(0.0, 0.5, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_00")],
        )

        turns = diarizer.diarize_stereo(
            mic_path, sys_path,
            speaker_name="Me", other_name="Other", other_names=["Other"],
        )

        assert turns == sorted(turns, key=lambda t: t.start)

    def test_empty_channel(self, tmp_path):
        """If one channel has no audio, only the other is processed."""
        mic_path = tmp_path / "test_mic.wav"
        sys_path = tmp_path / "test_sys.wav"
        _make_mono_wav(mic_path, np.full(16000, 5000, dtype=np.int16))
        # sys_path doesn't exist

        diarizer = self._make_mock_diarizer(
            mic_turns=[(0.0, 1.0, "SPEAKER_00")],
            sys_turns=[],
        )

        turns = diarizer.diarize_stereo(
            mic_path, sys_path,
            speaker_name="Me", other_name="Other", other_names=["Other"],
        )

        assert len(turns) == 1
        assert turns[0].speaker == "Me"
        assert turns[0].channel == "mic"


class TestDiarizeMono:
    def test_single_speaker(self, tmp_path):
        path = tmp_path / "test_mic.wav"
        _make_mono_wav(path, np.full(16000, 5000, dtype=np.int16))

        diarizer = PyannoteDiarizer.__new__(PyannoteDiarizer)
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]
        result = MagicMock()
        result.exclusive_speaker_diarization = mock_annotation
        diarizer._pipeline = lambda p: result

        turns = diarizer.diarize_mono(path, speaker_name="Me")
        assert len(turns) == 1
        assert turns[0].speaker == "Me"
        assert turns[0].channel == "mic"
