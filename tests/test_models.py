import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekeet_models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_ID,
    MULTILINGUAL_MODEL_ID,
    PARAKEET_V2,
    PARAKEET_V3,
    descriptor_for,
)


def test_default_is_v2_english():
    assert DEFAULT_MODEL_ID == PARAKEET_V2.id
    assert "English" in PARAKEET_V2.subtitle


def test_multilingual_is_v3():
    assert MULTILINGUAL_MODEL_ID == PARAKEET_V3.id
    assert "Multilingual" in PARAKEET_V3.subtitle


def test_catalog_ids_are_unique():
    ids = [m.id for m in AVAILABLE_MODELS]
    assert len(ids) == len(set(ids))


def test_descriptor_for_known_id():
    assert descriptor_for(PARAKEET_V2.id) is PARAKEET_V2


def test_descriptor_for_unknown_returns_none():
    assert descriptor_for("some/custom-model") is None


def test_descriptors_are_frozen():
    import dataclasses
    assert dataclasses.fields(PARAKEET_V2)  # is a dataclass
    # Mutating a frozen dataclass should raise
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        PARAKEET_V2.id = "hacked"  # type: ignore[misc]
