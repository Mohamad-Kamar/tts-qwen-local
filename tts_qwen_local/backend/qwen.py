from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

from ..config import (
    CUSTOMVOICE_SPEAKERS,
    TOKENIZER_MODEL_ID,
    ProfileSpec,
    default_chunk_chars,
    normalize_language,
)
from ..text import chunk_text
from .types import CloneRequest, ProgressCallback, SynthesisRequest, SynthesisResult


class QwenBackend:
    name = "pytorch"

    def __init__(self, device: str = "auto", dtype: str = "auto"):
        self._configure_device_environment(device)
        self._device = self.detect_device(device)
        self._dtype_mode = dtype
        self._dtype = self.resolve_dtype(self._device, dtype)
        self._models: dict[str, Any] = {}
        self._clone_prompt_cache: dict[tuple[str, str, str, bool], Any] = {}

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_name(self) -> str:
        return str(self._dtype).replace("torch.", "")

    @staticmethod
    def _configure_device_environment(requested_device: str) -> None:
        if requested_device not in {"auto", "mps"}:
            return

        # These Apple-specific knobs gave a small but repeatable speedup in local tests.
        os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
        os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "1")

    @staticmethod
    def detect_device(device: str = "auto") -> str:
        if device != "auto":
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def resolve_dtype(device: str, dtype: str):
        import torch

        if dtype == "auto":
            if device in {"cuda", "mps"}:
                return torch.bfloat16
            return torch.float32

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            supported = ", ".join(sorted(["auto", *mapping]))
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {supported}") from exc

    def ensure_loaded(self, profile: ProfileSpec) -> None:
        self._run_with_model_retry(profile, lambda _model: None)

    def preload(self, profile: ProfileSpec, include_tokenizer: bool = True) -> list[Path]:
        downloads = [Path(snapshot_download(repo_id=profile.model_id))]
        if include_tokenizer:
            downloads.append(Path(snapshot_download(repo_id=TOKENIZER_MODEL_ID)))
        return downloads

    def list_voices(self, profile: ProfileSpec) -> list[str]:
        if profile.model_type != "CustomVoice":
            raise ValueError(f"Profile '{profile.name}' does not support preset voices.")

        model = self._get_model(profile)
        if model is not None and callable(getattr(model, "get_supported_speakers", None)):
            speakers = model.get_supported_speakers() or []
            if speakers:
                return [_canonicalize_voice_name(item) for item in speakers]
        return list(CUSTOMVOICE_SPEAKERS)

    def synthesize(
        self,
        request: SynthesisRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        start = time.perf_counter()
        voice = request.voice or request.profile.default_voice
        language_name = normalize_language(request.language)
        chunk_chars = request.chunk_chars or default_chunk_chars(request.profile, self._device)
        chunks = chunk_text(request.text, chunk_chars)
        if not chunks:
            raise ValueError("Input text is empty.")

        for index, chunk in enumerate(chunks):
            if on_progress is not None:
                on_progress(index, len(chunks), chunk)

        wavs, sample_rate = self._run_with_model_retry(
            profile=request.profile,
            operation=lambda model: self._generate_synthesis_batch(
                model=model,
                profile=request.profile,
                chunks=chunks,
                voice=voice,
                instruct=request.instruct,
                language_name=language_name,
            ),
        )

        final_audio = _concat_audio(_extract_audio_segments(wavs))
        return SynthesisResult(
            audio=final_audio,
            sample_rate=sample_rate,
            chunks=chunks,
            elapsed_sec=time.perf_counter() - start,
            profile=request.profile.name,
            model_id=request.profile.model_id,
            device=self._device,
            dtype=self.dtype_name,
        )

    def clone(
        self,
        request: CloneRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        start = time.perf_counter()
        language_name = normalize_language(request.language)
        chunk_chars = request.chunk_chars or default_chunk_chars(request.profile, self._device)
        chunks = chunk_text(request.text, chunk_chars)
        if not chunks:
            raise ValueError("Input text is empty.")

        for index, chunk in enumerate(chunks):
            if on_progress is not None:
                on_progress(index, len(chunks), chunk)

        wavs, sample_rate = self._run_with_model_retry(
            profile=request.profile,
            operation=lambda active_model: self._generate_clone_batch(
                model=active_model,
                request=request,
                chunks=chunks,
                language_name=language_name,
            ),
        )

        final_audio = _concat_audio(_extract_audio_segments(wavs))
        return SynthesisResult(
            audio=final_audio,
            sample_rate=sample_rate,
            chunks=chunks,
            elapsed_sec=time.perf_counter() - start,
            profile=request.profile.name,
            model_id=request.profile.model_id,
            device=self._device,
            dtype=self.dtype_name,
        )

    def _generate_synthesis_batch(
        self,
        model: Any,
        profile: ProfileSpec,
        chunks: list[str],
        voice: str | None,
        instruct: str | None,
        language_name: str,
    ) -> tuple[Any, int]:
        if profile.model_type == "CustomVoice":
            return model.generate_custom_voice(
                text=chunks,
                speaker=[voice] * len(chunks),
                language=[language_name] * len(chunks),
                instruct=[(instruct or "")] * len(chunks),
                non_streaming_mode=True,
            )
        if profile.model_type == "VoiceDesign":
            return model.generate_voice_design(
                text=chunks,
                instruct=[instruct] * len(chunks),
                language=[language_name] * len(chunks),
                non_streaming_mode=True,
            )
        raise ValueError("Clone profiles are not supported by synthesize().")

    def _generate_clone_batch(
        self,
        model: Any,
        request: CloneRequest,
        chunks: list[str],
        language_name: str,
    ) -> tuple[Any, int]:
        prompt = self._get_or_create_clone_prompt(
            profile=request.profile,
            model=model,
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
            x_vector_only_mode=request.x_vector_only_mode,
        )
        return model.generate_voice_clone(
            text=chunks,
            language=[language_name] * len(chunks),
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
        )

    def _run_with_model_retry(
        self,
        profile: ProfileSpec,
        operation: Callable[[Any], Any],
    ) -> Any:
        try:
            model = self._get_model(profile)
            return operation(model)
        except Exception as error:
            if not self._should_retry_with_float32(error):
                raise
            self._switch_dtype("float32")
            model = self._get_model(profile)
            return operation(model)

    def _should_retry_with_float32(self, error: Exception) -> bool:
        if self._device != "mps" or self._dtype_mode != "auto":
            return False

        import torch

        if self._dtype != torch.bfloat16:
            return False

        message = str(error).lower()
        retry_markers = (
            "probability tensor",
            "nan",
            "inf",
            "unsupported dtype",
            "bfloat16",
        )
        return any(marker in message for marker in retry_markers)

    def _switch_dtype(self, dtype: str) -> None:
        self._models.clear()
        self._clone_prompt_cache.clear()
        self.clear_unused_memory()
        self._dtype = self.resolve_dtype(self._device, dtype)

    def _get_or_create_clone_prompt(
        self,
        profile: ProfileSpec,
        model: Any,
        reference_audio: Path,
        reference_text: str | None,
        x_vector_only_mode: bool,
    ) -> Any:
        audio_stat = reference_audio.stat()
        cache_key = (
            profile.model_id,
            str(reference_audio.resolve()),
            audio_stat.st_mtime_ns,
            audio_stat.st_size,
            (reference_text or "").strip(),
            x_vector_only_mode,
            self.dtype_name,
        )
        if cache_key in self._clone_prompt_cache:
            return self._clone_prompt_cache[cache_key]

        prompt = model.create_voice_clone_prompt(
            ref_audio=str(reference_audio),
            ref_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        self._clone_prompt_cache[cache_key] = prompt
        return prompt

    def _get_model(self, profile: ProfileSpec) -> Any:
        if profile.model_id not in self._models:
            self._models[profile.model_id] = self._load_model(profile.model_id)
        return self._models[profile.model_id]

    def _load_model(self, model_id: str) -> Any:
        from qwen_tts import Qwen3TTSModel

        load_kwargs: dict[str, Any] = {
            "device_map": self._device,
            "dtype": self._dtype,
        }
        if self._device == "cuda":
            load_kwargs["attn_implementation"] = "flash_attention_2"

        return Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)

    def clear_unused_memory(self) -> None:
        import torch

        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def close(self) -> None:
        self._models.clear()
        self._clone_prompt_cache.clear()
        self.clear_unused_memory()


def _extract_audio_array(wavs: Any) -> np.ndarray:
    audio = wavs
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _extract_audio_segments(wavs: Any) -> list[np.ndarray]:
    if isinstance(wavs, list):
        return [_extract_audio_array(item) for item in wavs]
    return [_extract_audio_array(wavs)]


def _canonicalize_voice_name(voice: str) -> str:
    known = {name.lower(): name for name in CUSTOMVOICE_SPEAKERS}
    return known.get(voice.lower(), voice)


def _concat_audio(audio_segments: list[np.ndarray]) -> np.ndarray:
    if not audio_segments:
        return np.array([], dtype=np.float32)
    if len(audio_segments) == 1:
        return audio_segments[0]
    return np.concatenate(audio_segments)
