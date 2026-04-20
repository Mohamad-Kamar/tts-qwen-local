from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    text: str
    profile: object
    language: str = "auto"
    voice: str | None = None
    instruct: str | None = None
    chunk_chars: int | None = None


@dataclass(frozen=True, slots=True)
class CloneRequest:
    text: str
    profile: object
    reference_audio: Path
    reference_text: str | None = None
    x_vector_only_mode: bool = False
    language: str = "auto"
    chunk_chars: int | None = None


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    chunks: list[str]
    elapsed_sec: float
    profile: str
    model_id: str
    device: str
    dtype: str
    trace: dict[str, Any] | None = None
