"""
Speaker diarization using MLX-native WeSpeaker embeddings.

Extracts 256-dim speaker embeddings from speech segments and matches
them against known speaker profiles via cosine similarity.
"""

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Embedding settings
MIN_AUDIO_SECONDS = 0.5
SIMILARITY_THRESHOLD = 0.5
MAX_SPEAKERS = 5
MEL_N_FFT = 400  # 25ms window at 16kHz (matches WeSpeaker/Kaldi)
MEL_HOP_LENGTH = 160  # 10ms hop
MEL_N_MELS = 80
MEL_FMIN = 20.0  # Kaldi default low freq
PREEMPHASIS = 0.97  # Kaldi default pre-emphasis coefficient
SAMPLE_RATE = 16000

# EMA settings
EMA_ALPHA_EARLY = 0.5  # First few segments — fast adaptation
EMA_ALPHA_STABLE = 0.1  # After settling — slow drift
EMA_EARLY_COUNT = 5  # Segments before switching to stable alpha


@dataclass
class SpeakerProfile:
    label: str
    centroid: np.ndarray
    count: int = 0
    last_seen: float = 0.0


class SpeakerEmbedder:
    """Loads WeSpeaker ResNet model and extracts speaker embeddings via MLX."""

    _REPO_ID = "mlx-community/wespeaker-voxceleb-resnet34-LM"
    _WEIGHTS_FILE = "weights.npz"
    _MODEL_FILE = "resnet_embedding.py"

    def __init__(self):
        import mlx.core as mx

        self._mx = mx

        # Download model files
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(self._REPO_ID, self._WEIGHTS_FILE)
        model_path = hf_hub_download(self._REPO_ID, self._MODEL_FILE)

        # Dynamically import the model module and use its loader
        spec = importlib.util.spec_from_file_location("resnet_embedding", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Suppress [INFO]/[WARN] prints from the model loader
        with redirect_stdout(io.StringIO()):
            self._model = module.load_resnet34_embedding(weights_path)

        # Precompute mel filterbank and window for fast feature extraction
        self._window = np.hanning(MEL_N_FFT).astype(np.float32)
        self._mel_fb = self._build_mel_filterbank()

    @staticmethod
    def _build_mel_filterbank() -> np.ndarray:
        """Build mel filterbank matrix (n_mels x n_fft//2+1)."""
        fmax = SAMPLE_RATE / 2.0
        n_fft_bins = MEL_N_FFT // 2 + 1

        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(MEL_FMIN), hz_to_mel(fmax), MEL_N_MELS + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((MEL_N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)

        fb = np.zeros((MEL_N_MELS, n_fft_bins), dtype=np.float32)
        for i in range(MEL_N_MELS):
            lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
            if mid > lo:
                fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
            if hi > mid:
                fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
        return fb

    def compute_mel(self, audio: np.ndarray) -> np.ndarray | None:
        """Compute Kaldi-compatible mel spectrogram from raw audio.

        Matches WeSpeaker training preprocessing: pre-emphasis, power spectrum,
        mel filterbank (20Hz+), log, and cepstral mean+variance normalization.

        Args:
            audio: float32 mono audio at 16kHz.

        Returns:
            Mel spectrogram as (time, n_mels) numpy array, or None if too short.
        """
        if len(audio) < int(MIN_AUDIO_SECONDS * SAMPLE_RATE):
            return None

        # Pre-emphasis filter: y[n] = x[n] - 0.97 * x[n-1]
        audio = np.append(audio[0], audio[1:] - PREEMPHASIS * audio[:-1])

        n_frames = 1 + (len(audio) - MEL_N_FFT) // MEL_HOP_LENGTH
        if n_frames < 1:
            return None

        # Frame the audio using stride tricks (zero-copy)
        audio_contiguous = np.ascontiguousarray(audio, dtype=np.float32)
        frames = np.lib.stride_tricks.as_strided(
            audio_contiguous,
            shape=(n_frames, MEL_N_FFT),
            strides=(audio_contiguous.strides[0] * MEL_HOP_LENGTH, audio_contiguous.strides[0]),
        )

        # Window + FFT + power spectrum
        windowed = frames * self._window
        spectrum = np.fft.rfft(windowed, n=MEL_N_FFT)
        power = np.abs(spectrum) ** 2

        # Apply mel filterbank: (n_frames, n_fft_bins) @ (n_fft_bins, n_mels) -> (n_frames, n_mels)
        mel = power @ self._mel_fb.T
        mel = np.maximum(mel, 1e-10)
        mel = np.log(mel)

        # Cepstral mean and variance normalization (per-frequency bin)
        mel -= mel.mean(axis=0, keepdims=True)
        std = mel.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-10)
        mel /= std

        return mel  # (time, n_mels) — ready for model input

    def extract_embedding_from_mel(self, mel: np.ndarray) -> np.ndarray | None:
        """Extract speaker embedding from precomputed mel spectrogram.

        Must be called under _model_lock (MLX is not thread-safe).

        Args:
            mel: Mel spectrogram from compute_mel().

        Returns:
            L2-normalized 256-dim embedding, or None on failure.
        """
        if mel is None:
            return None

        mx = self._mx

        try:
            # mel is (time, n_mels), model expects (batch, time, n_mels)
            mel_mx = mx.array(mel)[None, ...]
            embedding = self._model(mel_mx)
            embedding = np.array(embedding).flatten()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm < 1e-8:
                return None
            return embedding / norm
        except Exception as e:
            print(f"Embedding extraction failed: {e}", file=sys.stderr)
            return None

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        """Extract speaker embedding from raw audio (convenience method).

        Note: For threaded use, prefer compute_mel() outside lock +
        extract_embedding_from_mel() inside lock.

        Args:
            audio: float32 mono audio at 16kHz.

        Returns:
            L2-normalized 256-dim embedding, or None if audio too short.
        """
        mel = self.compute_mel(audio)
        if mel is None:
            return None
        return self.extract_embedding_from_mel(mel)


class SpeakerTracker:
    """Online speaker clustering for one audio channel.

    Maintains speaker profiles and assigns labels to new embeddings
    via cosine similarity matching.
    """

    def __init__(
        self,
        primary_name: str,
        secondary_prefix: str,
        threshold: float = SIMILARITY_THRESHOLD,
        max_speakers: int = MAX_SPEAKERS,
        secondary_names: list[str] | None = None,
    ):
        """
        Args:
            primary_name: Label for the first speaker on this channel
                (e.g. "Me" for mic, "Other" for system).
            secondary_prefix: Prefix for additional speakers
                (e.g. "Local" for mic, "Remote" for system).
            threshold: Cosine similarity threshold for matching.
            max_speakers: Maximum number of distinct speakers to track.
            secondary_names: Explicit names for additional speakers.
                Used before falling back to "{prefix} N" labels.
        """
        self.primary_name = primary_name
        self.secondary_prefix = secondary_prefix
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.secondary_names = list(secondary_names) if secondary_names else []
        self.profiles: list[SpeakerProfile] = []

    def identify(self, embedding: np.ndarray) -> str:
        """Identify speaker from embedding.

        Args:
            embedding: L2-normalized speaker embedding.

        Returns:
            Speaker label string.
        """
        import time

        now = time.monotonic()

        # First speaker on this channel gets the primary name
        if not self.profiles:
            self.profiles.append(SpeakerProfile(
                label=self.primary_name,
                centroid=embedding.copy(),
                count=1,
                last_seen=now,
            ))
            return self.primary_name

        # Compute cosine similarity against all profiles
        similarities = np.array([
            np.dot(embedding, p.centroid) for p in self.profiles
        ])
        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]

        if best_sim >= self.threshold:
            # Match — update centroid via EMA
            profile = self.profiles[best_idx]
            profile.count += 1
            profile.last_seen = now
            alpha = EMA_ALPHA_EARLY if profile.count <= EMA_EARLY_COUNT else EMA_ALPHA_STABLE
            profile.centroid = alpha * embedding + (1 - alpha) * profile.centroid
            # Re-normalize
            norm = np.linalg.norm(profile.centroid)
            if norm > 1e-8:
                profile.centroid /= norm
            return profile.label

        if len(self.profiles) < self.max_speakers:
            # New speaker
            speaker_num = len(self.profiles) + 1
            if speaker_num == 1:
                label = self.primary_name
            elif self.secondary_names:
                label = self.secondary_names.pop(0)
            else:
                label = f"{self.secondary_prefix} {speaker_num}"
            self.profiles.append(SpeakerProfile(
                label=label,
                centroid=embedding.copy(),
                count=1,
                last_seen=now,
            ))
            return label

        # At max speakers — assign to closest match
        profile = self.profiles[best_idx]
        profile.count += 1
        profile.last_seen = now
        return profile.label

    def reset(self) -> None:
        """Clear all speaker profiles."""
        self.profiles.clear()


def load_embedder() -> SpeakerEmbedder:
    """Load the speaker embedding model.

    Returns:
        Initialized SpeakerEmbedder ready for inference.
    """
    return SpeakerEmbedder()
