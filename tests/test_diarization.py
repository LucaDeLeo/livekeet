"""Tests for speaker diarization using synthetic embeddings (no model/hardware required)."""

import numpy as np
import pytest

from diarization import (
    EMA_ALPHA_EARLY,
    EMA_ALPHA_STABLE,
    EMA_EARLY_COUNT,
    SpeakerProfile,
    SpeakerTracker,
)


def _normalized(v: list[float]) -> np.ndarray:
    """Create an L2-normalized embedding from a list."""
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Reusable embeddings — orthogonal directions for distinct "speakers"
EMB_A = _normalized([1.0, 0.0, 0.0, 0.0])
EMB_B = _normalized([0.0, 1.0, 0.0, 0.0])
EMB_C = _normalized([0.0, 0.0, 1.0, 0.0])
EMB_D = _normalized([0.0, 0.0, 0.0, 1.0])

# Similar to A (cosine > 0.65)
EMB_A_SIMILAR = _normalized([0.9, 0.1, 0.05, 0.05])


class TestSpeakerTracker:
    def test_first_speaker_gets_primary_name(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        label = tracker.identify(EMB_A)
        assert label == "Me"
        assert len(tracker.profiles) == 1

    def test_similar_embedding_matches_same_speaker(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)
        label = tracker.identify(EMB_A_SIMILAR)
        assert label == "Me"
        assert len(tracker.profiles) == 1

    def test_dissimilar_embedding_creates_new_speaker(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)
        label = tracker.identify(EMB_B)
        assert label == "Local 2"
        assert len(tracker.profiles) == 2

    def test_third_speaker_numbered_correctly(self):
        tracker = SpeakerTracker(primary_name="Other", secondary_prefix="Remote")
        tracker.identify(EMB_A)
        tracker.identify(EMB_B)
        label = tracker.identify(EMB_C)
        assert label == "Remote 3"
        assert len(tracker.profiles) == 3

    def test_max_speakers_cap(self):
        tracker = SpeakerTracker(
            primary_name="Me", secondary_prefix="Local", max_speakers=2
        )
        tracker.identify(EMB_A)  # Me
        tracker.identify(EMB_B)  # Local 2
        # Third distinct embedding should be assigned to closest existing
        label = tracker.identify(EMB_C)
        assert label in ("Me", "Local 2")
        assert len(tracker.profiles) == 2

    def test_returning_speaker_recognized(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)  # Me
        tracker.identify(EMB_B)  # Local 2
        # Speaker A returns
        label = tracker.identify(EMB_A)
        assert label == "Me"
        assert len(tracker.profiles) == 2

    def test_ema_centroid_update_early(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)
        original_centroid = tracker.profiles[0].centroid.copy()

        # Feed a similar embedding — should update centroid with early alpha
        tracker.identify(EMB_A_SIMILAR)
        updated_centroid = tracker.profiles[0].centroid

        expected = EMA_ALPHA_EARLY * EMB_A_SIMILAR + (1 - EMA_ALPHA_EARLY) * original_centroid
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(updated_centroid, expected, atol=1e-6)

    def test_ema_switches_to_stable_alpha(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")

        # Feed enough segments to pass the early phase
        for _ in range(EMA_EARLY_COUNT + 1):
            tracker.identify(EMB_A)

        assert tracker.profiles[0].count == EMA_EARLY_COUNT + 1

        # Next update should use stable alpha
        centroid_before = tracker.profiles[0].centroid.copy()
        tracker.identify(EMB_A_SIMILAR)

        expected = EMA_ALPHA_STABLE * EMB_A_SIMILAR + (1 - EMA_ALPHA_STABLE) * centroid_before
        expected /= np.linalg.norm(expected)

        np.testing.assert_allclose(tracker.profiles[0].centroid, expected, atol=1e-6)

    def test_system_channel_labels(self):
        tracker = SpeakerTracker(primary_name="Other", secondary_prefix="Remote")
        assert tracker.identify(EMB_A) == "Other"
        assert tracker.identify(EMB_B) == "Remote 2"
        assert tracker.identify(EMB_C) == "Remote 3"

    def test_reset_clears_profiles(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)
        tracker.identify(EMB_B)
        assert len(tracker.profiles) == 2

        tracker.reset()
        assert len(tracker.profiles) == 0

        # After reset, first speaker gets primary name again
        label = tracker.identify(EMB_C)
        assert label == "Me"

    def test_custom_threshold(self):
        # Very high threshold — even similar embeddings create new speakers
        tracker = SpeakerTracker(
            primary_name="Me", secondary_prefix="Local", threshold=0.999
        )
        tracker.identify(EMB_A)
        label = tracker.identify(EMB_A_SIMILAR)
        assert label == "Local 2"

    def test_exact_same_embedding_always_matches(self):
        tracker = SpeakerTracker(primary_name="Me", secondary_prefix="Local")
        tracker.identify(EMB_A)
        for _ in range(10):
            assert tracker.identify(EMB_A) == "Me"
        assert len(tracker.profiles) == 1
        assert tracker.profiles[0].count == 11
