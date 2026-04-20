from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backend import CloneRequest, SynthesisRequest, create_backend
from .config import ProfileSpec, backend_model_id, default_chunk_chars


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    profile: str
    backend: str
    model_id: str
    device: str
    dtype: str
    seed: int | None
    chars: int
    chunk_chars: int
    chunk_count: int
    cold_audio_duration_sec: float
    warm_audio_duration_sec_avg: float
    warm_audio_duration_sec_runs: list[float]
    cold_elapsed_sec: float
    warm_elapsed_sec_avg: float
    warm_elapsed_sec_runs: list[float]
    cold_rtf: float
    warm_rtf_avg: float
    inputs: dict[str, Any]
    cold_trace: dict[str, Any]
    warm_traces: list[dict[str, Any]]


def run_benchmark(
    *,
    profile: ProfileSpec,
    text: str,
    voice: str | None,
    language: str,
    instruct: str | None,
    backend: str,
    device: str,
    dtype: str,
    mlx_python: str | None,
    mlx_variant: str | None,
    mlx_model: str | None,
    chunk_chars: int | None,
    warm_runs: int = 2,
    seed: int | None = None,
) -> BenchmarkResult:
    cold_backend = create_backend(
        backend,
        device=device,
        dtype=dtype,
        mlx_python=mlx_python,
        mlx_variant=mlx_variant,
        mlx_model=mlx_model,
    )
    try:
        effective_seed = _effective_seed(seed, cold_backend.name)
        effective_chunk_chars = chunk_chars or default_chunk_chars(profile, cold_backend.device)
        cold_request = SynthesisRequest(
            text=text,
            profile=profile,
            voice=voice,
            language=language,
            instruct=instruct,
            chunk_chars=effective_chunk_chars,
        )
        _apply_seed(effective_seed)
        cold_result = cold_backend.synthesize(cold_request)
        cold_audio_duration = len(cold_result.audio) / cold_result.sample_rate
    finally:
        cold_backend.close()

    warm_backend = create_backend(
        backend,
        device=device,
        dtype=dtype,
        mlx_python=mlx_python,
        mlx_variant=mlx_variant,
        mlx_model=mlx_model,
    )
    try:
        warm_backend.ensure_loaded(profile)
        warm_request = SynthesisRequest(
            text=text,
            profile=profile,
            voice=voice,
            language=language,
            instruct=instruct,
            chunk_chars=effective_chunk_chars,
        )

        warm_times: list[float] = []
        warm_audio_durations: list[float] = []
        warm_traces: list[dict[str, Any]] = []
        for run_index in range(max(warm_runs, 1)):
            _apply_seed(None if effective_seed is None else effective_seed + run_index + 1)
            result = warm_backend.synthesize(warm_request)
            warm_times.append(result.elapsed_sec)
            warm_audio_durations.append(len(result.audio) / result.sample_rate)
            warm_traces.append(dict(result.trace or {}))

        warm_avg = sum(warm_times) / len(warm_times)
        warm_audio_duration_avg = sum(warm_audio_durations) / len(warm_audio_durations)
        return BenchmarkResult(
            mode="synth",
            profile=profile.name,
            backend=warm_backend.name,
            model_id=backend_model_id(
                profile,
                warm_backend.name,
                mlx_variant=mlx_variant,
                mlx_model=mlx_model,
            ),
            device=warm_backend.device,
            dtype=cold_result.dtype,
            seed=effective_seed,
            chars=len(text),
            chunk_chars=effective_chunk_chars,
            chunk_count=len(cold_result.chunks),
            cold_audio_duration_sec=cold_audio_duration,
            warm_audio_duration_sec_avg=warm_audio_duration_avg,
            warm_audio_duration_sec_runs=warm_audio_durations,
            cold_elapsed_sec=cold_result.elapsed_sec,
            warm_elapsed_sec_avg=warm_avg,
            warm_elapsed_sec_runs=warm_times,
            cold_rtf=_rtf(cold_result.elapsed_sec, cold_audio_duration),
            warm_rtf_avg=_rtf(warm_avg, warm_audio_duration_avg),
            inputs={
                "language": language,
                "voice": voice,
                "instruct": instruct,
            },
            cold_trace=dict(cold_result.trace or {}),
            warm_traces=warm_traces,
        )
    finally:
        warm_backend.close()


def run_clone_benchmark(
    *,
    profile: ProfileSpec,
    text: str,
    reference_audio: Path,
    reference_text: str | None,
    x_vector_only_mode: bool,
    language: str,
    backend: str,
    device: str,
    dtype: str,
    mlx_python: str | None,
    mlx_variant: str | None,
    mlx_model: str | None,
    chunk_chars: int | None,
    warm_runs: int = 2,
) -> BenchmarkResult:
    cold_backend = create_backend(
        backend,
        device=device,
        dtype=dtype,
        mlx_python=mlx_python,
        mlx_variant=mlx_variant,
        mlx_model=mlx_model,
    )
    try:
        effective_chunk_chars = chunk_chars or default_chunk_chars(profile, cold_backend.device)
        cold_request = CloneRequest(
            text=text,
            profile=profile,
            reference_audio=reference_audio,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
            language=language,
            chunk_chars=effective_chunk_chars,
        )
        cold_result = cold_backend.clone(cold_request)
        cold_audio_duration = len(cold_result.audio) / cold_result.sample_rate
    finally:
        cold_backend.close()

    warm_backend = create_backend(
        backend,
        device=device,
        dtype=dtype,
        mlx_python=mlx_python,
        mlx_variant=mlx_variant,
        mlx_model=mlx_model,
    )
    try:
        warm_backend.ensure_loaded(profile)
        warm_request = CloneRequest(
            text=text,
            profile=profile,
            reference_audio=reference_audio,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
            language=language,
            chunk_chars=effective_chunk_chars,
        )

        warm_times: list[float] = []
        warm_audio_durations: list[float] = []
        warm_traces: list[dict[str, Any]] = []
        for _ in range(max(warm_runs, 1)):
            result = warm_backend.clone(warm_request)
            warm_times.append(result.elapsed_sec)
            warm_audio_durations.append(len(result.audio) / result.sample_rate)
            warm_traces.append(dict(result.trace or {}))

        warm_avg = sum(warm_times) / len(warm_times)
        warm_audio_duration_avg = sum(warm_audio_durations) / len(warm_audio_durations)
        return BenchmarkResult(
            mode="clone",
            profile=profile.name,
            backend=warm_backend.name,
            model_id=backend_model_id(
                profile,
                warm_backend.name,
                mlx_variant=mlx_variant,
                mlx_model=mlx_model,
            ),
            device=warm_backend.device,
            dtype=cold_result.dtype,
            seed=None,
            chars=len(text),
            chunk_chars=effective_chunk_chars,
            chunk_count=len(cold_result.chunks),
            cold_audio_duration_sec=cold_audio_duration,
            warm_audio_duration_sec_avg=warm_audio_duration_avg,
            warm_audio_duration_sec_runs=warm_audio_durations,
            cold_elapsed_sec=cold_result.elapsed_sec,
            warm_elapsed_sec_avg=warm_avg,
            warm_elapsed_sec_runs=warm_times,
            cold_rtf=_rtf(cold_result.elapsed_sec, cold_audio_duration),
            warm_rtf_avg=_rtf(warm_avg, warm_audio_duration_avg),
            inputs={
                "language": language,
                "reference_audio": str(reference_audio),
                "has_reference_text": bool(reference_text),
                "x_vector_only_mode": x_vector_only_mode,
            },
            cold_trace=dict(cold_result.trace or {}),
            warm_traces=warm_traces,
        )
    finally:
        warm_backend.close()


def write_benchmark_json(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def _rtf(elapsed_sec: float, audio_duration_sec: float) -> float:
    if audio_duration_sec <= 0:
        return float("inf")
    return elapsed_sec / audio_duration_sec


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return

    import torch

    torch.manual_seed(seed)


def _effective_seed(seed: int | None, backend_name: str) -> int | None:
    if backend_name != "pytorch":
        return None
    return seed
