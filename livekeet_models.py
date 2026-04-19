"""Curated catalog of speech-to-text models livekeet knows about.

Mirrors the structure of livekeet-mlx/Sources/LivekeetCore/ModelCatalog.swift.
The underlying model id is still free-form — users may set any Hugging Face id
in config or via --model; the catalog is for display and CLI --help only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["parakeet"]


@dataclass(frozen=True)
class SpeechModelDescriptor:
    id: str              # Hugging Face model id (persisted value)
    display_name: str    # Short label
    subtitle: str        # One-line descriptor
    size_description: str
    backend: Backend


PARAKEET_V2 = SpeechModelDescriptor(
    id="mlx-community/parakeet-tdt-0.6b-v2",
    display_name="Parakeet TDT 0.6B v2",
    subtitle="English, highest accuracy",
    size_description="~600 MB",
    backend="parakeet",
)

PARAKEET_V3 = SpeechModelDescriptor(
    id="mlx-community/parakeet-tdt-0.6b-v3",
    display_name="Parakeet TDT 0.6B v3",
    subtitle="Multilingual (25 languages)",
    size_description="~600 MB",
    backend="parakeet",
)

AVAILABLE_MODELS: tuple[SpeechModelDescriptor, ...] = (PARAKEET_V2, PARAKEET_V3)

DEFAULT_MODEL_ID: str = PARAKEET_V2.id
MULTILINGUAL_MODEL_ID: str = PARAKEET_V3.id


def descriptor_for(model_id: str) -> SpeechModelDescriptor | None:
    """Return the catalog descriptor with matching id, or None for custom strings."""
    for m in AVAILABLE_MODELS:
        if m.id == model_id:
            return m
    return None
