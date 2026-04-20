from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import ProfileSpec
from .types import CloneRequest, ProgressCallback, SynthesisRequest, SynthesisResult


class TTSBackend(Protocol):
    name: str
    device: str
    dtype_name: str

    def ensure_loaded(self, profile: ProfileSpec) -> None: ...

    def preload(self, profile: ProfileSpec, include_tokenizer: bool = True) -> list[Path]: ...

    def list_voices(self, profile: ProfileSpec) -> list[str]: ...

    def synthesize(
        self,
        request: SynthesisRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult: ...

    def clone(
        self,
        request: CloneRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult: ...

    def close(self) -> None: ...
