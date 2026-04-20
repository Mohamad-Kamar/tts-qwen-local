from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio import encode_audio_bytes, infer_audio_format, resolve_output_path, write_audio_file
from .backend.factory import create_backend
from .backend.types import CloneRequest, SynthesisRequest, SynthesisResult
from .config import (
    DEFAULT_LANGUAGE,
    DEFAULT_PROFILE_NAME,
    PROFILE_MAP,
    ProfileSpec,
    default_chunk_chars,
    get_profile,
    validate_clone_options,
    validate_synth_options,
)
from .text import normalize_text

ProgressCallback = Callable[["TTSProgress"], None]


@dataclass(frozen=True, slots=True)
class TTSProgress:
    chunk_index: int
    total_chunks: int
    chunk_text: str


@dataclass(frozen=True, slots=True)
class SynthesisOptions:
    text: str
    profile: str = DEFAULT_PROFILE_NAME
    voice: str | None = None
    language: str = DEFAULT_LANGUAGE
    instruct: str | None = None
    chunk_chars: int | None = None


@dataclass(frozen=True, slots=True)
class CloneOptions:
    text: str
    reference_audio: str | Path
    profile: str = "clone-fast"
    reference_text: str | None = None
    x_vector_only_mode: bool = False
    language: str = DEFAULT_LANGUAGE
    chunk_chars: int | None = None


class QwenTTSService:
    def __init__(
        self,
        *,
        backend: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        mlx_python: str | None = None,
        mlx_variant: str | None = None,
        mlx_model: str | None = None,
    ) -> None:
        self._backend = create_backend(
            backend,
            device=device,
            dtype=dtype,
            mlx_python=mlx_python,
            mlx_variant=mlx_variant,
            mlx_model=mlx_model,
        )
        self._lock = threading.RLock()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def device(self) -> str:
        return self._backend.device

    @property
    def dtype_name(self) -> str:
        return self._backend.dtype_name

    def close(self) -> None:
        with self._lock:
            self._backend.close()

    def __enter__(self) -> "QwenTTSService":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def list_profiles(self) -> list[ProfileSpec]:
        return list(PROFILE_MAP.values())

    def preload(self, profile: str = DEFAULT_PROFILE_NAME, include_tokenizer: bool = True) -> list[Path]:
        resolved = self._resolve_profile(profile)
        with self._lock:
            return self._backend.preload(resolved, include_tokenizer=include_tokenizer)

    def preload_all(self, include_tokenizer: bool = True) -> dict[str, list[Path]]:
        results: dict[str, list[Path]] = {}
        for profile in PROFILE_MAP:
            results[profile] = self.preload(profile, include_tokenizer=include_tokenizer)
        return results

    def list_voices(self, profile: str = DEFAULT_PROFILE_NAME) -> list[str]:
        resolved = self._resolve_profile(profile)
        with self._lock:
            self._backend.ensure_loaded(resolved)
            return self._backend.list_voices(resolved)

    def synthesize(
        self,
        options: SynthesisOptions,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        profile = self._resolve_profile(options.profile)
        validate_synth_options(profile, options.voice, options.instruct)
        text = _require_text(normalize_text(options.text))
        request = SynthesisRequest(
            text=text,
            profile=profile,
            language=options.language,
            voice=options.voice,
            instruct=options.instruct,
            chunk_chars=options.chunk_chars or default_chunk_chars(profile, self.device),
        )
        with self._lock:
            return self._backend.synthesize(
                request,
                on_progress=_wrap_progress(on_progress),
            )

    def clone(
        self,
        options: CloneOptions,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        profile = self._resolve_profile(options.profile)
        reference_audio = Path(options.reference_audio)
        validate_clone_options(
            profile,
            reference_audio,
            options.reference_text,
            options.x_vector_only_mode,
        )
        text = _require_text(normalize_text(options.text))
        reference_text = None if options.x_vector_only_mode else options.reference_text
        request = CloneRequest(
            text=text,
            profile=profile,
            reference_audio=reference_audio,
            reference_text=reference_text,
            x_vector_only_mode=options.x_vector_only_mode,
            language=options.language,
            chunk_chars=options.chunk_chars or default_chunk_chars(profile, self.device),
        )
        with self._lock:
            return self._backend.clone(
                request,
                on_progress=_wrap_progress(on_progress),
            )

    def synthesize_to_bytes(
        self,
        options: SynthesisOptions,
        *,
        audio_format: str = "wav",
    ) -> bytes:
        result = self.synthesize(options)
        return encode_audio_bytes(result.audio, result.sample_rate, audio_format=audio_format)

    def clone_to_bytes(
        self,
        options: CloneOptions,
        *,
        audio_format: str = "wav",
    ) -> bytes:
        result = self.clone(options)
        return encode_audio_bytes(result.audio, result.sample_rate, audio_format=audio_format)

    def synthesize_to_file(
        self,
        options: SynthesisOptions,
        output_path: str | Path,
        *,
        audio_format: str | None = None,
    ) -> Path:
        result = self.synthesize(options)
        format_name = infer_audio_format(output_path, audio_format)
        resolved = resolve_output_path(output_path, format_name)
        write_audio_file(resolved, result.audio, result.sample_rate, format_name)
        return resolved

    def clone_to_file(
        self,
        options: CloneOptions,
        output_path: str | Path,
        *,
        audio_format: str | None = None,
    ) -> Path:
        result = self.clone(options)
        format_name = infer_audio_format(output_path, audio_format)
        resolved = resolve_output_path(output_path, format_name)
        write_audio_file(resolved, result.audio, result.sample_rate, format_name)
        return resolved

    @staticmethod
    def _resolve_profile(profile: str | ProfileSpec) -> ProfileSpec:
        if isinstance(profile, ProfileSpec):
            return profile
        return get_profile(profile)


def create_service(
    *,
    backend: str = "auto",
    device: str = "auto",
    dtype: str = "auto",
    mlx_python: str | None = None,
    mlx_variant: str | None = None,
    mlx_model: str | None = None,
) -> QwenTTSService:
    return QwenTTSService(
        backend=backend,
        device=device,
        dtype=dtype,
        mlx_python=mlx_python,
        mlx_variant=mlx_variant,
        mlx_model=mlx_model,
    )


def _wrap_progress(callback: ProgressCallback | None):
    if callback is None:
        return None

    def _emit(index: int, total: int, chunk_text: str) -> None:
        callback(TTSProgress(chunk_index=index, total_chunks=total, chunk_text=chunk_text))

    return _emit


def _require_text(text: str) -> str:
    if not text:
        raise ValueError("Input text is empty.")
    return text
