"""
Speaker diarization using pyannote's full batch pipeline.

Runs pyannote/speaker-diarization-3.1 separately on each audio channel
(mic and system) after recording, then assigns real names to speakers.
"""

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000


@dataclass
class SpeakerTurn:
    """A single speaker turn from pyannote diarization."""
    start: float  # seconds from audio start
    end: float
    speaker: str  # final speaker name
    channel: str  # "mic" or "system"


class PyannoteDiarizer:
    """Batch speaker diarization using pyannote/speaker-diarization-3.1."""

    def __init__(self, hf_token: str | None = None):
        """Load pyannote pipeline.

        Args:
            hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
        """
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "pyannote requires a HuggingFace token. "
                "Set HF_TOKEN env var or add token to config [pyannote] section."
            )

        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )

    def _run_pipeline(self, audio_path: Path) -> list[tuple[float, float, str]]:
        """Run pyannote on a mono WAV file.

        Returns:
            List of (start, end, speaker_label) tuples.
        """
        result = self._pipeline(str(audio_path))

        # pyannote v4 returns DiarizeOutput; v3 returns Annotation directly
        if hasattr(result, "exclusive_speaker_diarization"):
            annotation = result.exclusive_speaker_diarization
        else:
            annotation = result

        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        return turns

    def diarize_stereo(
        self,
        mic_path: Path,
        sys_path: Path,
        speaker_name: str,
        other_name: str,
        other_names: list[str],
    ) -> list[SpeakerTurn]:
        """Run pyannote separately on each channel and assign names.

        Args:
            mic_path: Mono WAV of mic (left) channel.
            sys_path: Mono WAV of system (right) channel.
            speaker_name: Name for mic user.
            other_name: Primary name for system speaker.
            other_names: All remote speaker names.

        Returns:
            Combined speaker turns from both channels, sorted by start time.
        """
        all_turns = []

        # Mic channel — speaker_name is primary
        if mic_path.exists() and mic_path.stat().st_size > 44:  # >WAV header
            mic_raw = self._run_pipeline(mic_path)
            mic_labels = list(dict.fromkeys(label for _, _, label in mic_raw))
            mic_names = _assign_channel_names(mic_labels, speaker_name, "Local")
            for start, end, label in mic_raw:
                all_turns.append(SpeakerTurn(
                    start=start, end=end,
                    speaker=mic_names[label], channel="mic",
                ))

        # System channel — other_name is primary
        if sys_path.exists() and sys_path.stat().st_size > 44:
            sys_raw = self._run_pipeline(sys_path)
            sys_labels = list(dict.fromkeys(label for _, _, label in sys_raw))
            all_other = list(other_names) if other_names else [other_name]
            sys_names = _assign_channel_names(sys_labels, all_other[0], "Remote", all_other[1:])
            for start, end, label in sys_raw:
                all_turns.append(SpeakerTurn(
                    start=start, end=end,
                    speaker=sys_names[label], channel="system",
                ))

        all_turns.sort(key=lambda t: t.start)
        return all_turns

    def diarize_mono(
        self,
        audio_path: Path,
        speaker_name: str,
    ) -> list[SpeakerTurn]:
        """Run pyannote on a single mono file (mic-only mode).

        Args:
            audio_path: Mono WAV file.
            speaker_name: Name for primary speaker.

        Returns:
            Speaker turns sorted by start time.
        """
        raw = self._run_pipeline(audio_path)
        labels = list(dict.fromkeys(label for _, _, label in raw))
        names = _assign_channel_names(labels, speaker_name, "Speaker")

        turns = []
        for start, end, label in raw:
            turns.append(SpeakerTurn(
                start=start, end=end,
                speaker=names[label], channel="mic",
            ))

        turns.sort(key=lambda t: t.start)
        return turns


def _assign_channel_names(
    labels: list[str],
    primary_name: str,
    prefix: str,
    extra_names: list[str] | None = None,
) -> dict[str, str]:
    """Map pyannote labels to names for a single channel.

    First speaker gets primary_name, subsequent speakers get extra_names
    then fall back to "{prefix} N".
    """
    extra = list(extra_names) if extra_names else []
    name_map = {}

    for i, label in enumerate(labels):
        if i == 0:
            name_map[label] = primary_name
        elif extra:
            name_map[label] = extra.pop(0)
        else:
            name_map[label] = f"{prefix} {i + 1}"

    return name_map


def check_pyannote_installed() -> None:
    """Verify pyannote-audio is installed, raise with helpful message if not."""
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyannote engine requires pyannote-audio. "
            "Install with: uv sync --extra pyannote"
        )
