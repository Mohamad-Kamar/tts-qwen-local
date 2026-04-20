from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import QwenBackend, SynthesisRequest
from .config import ProfileSpec, default_chunk_chars


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    profile: str
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


def run_benchmark(
    *,
    profile: ProfileSpec,
    text: str,
    voice: str | None,
    language: str,
    instruct: str | None,
    device: str,
    dtype: str,
    chunk_chars: int | None,
    warm_runs: int = 2,
    seed: int | None = None,
) -> BenchmarkResult:
    cold_backend = QwenBackend(device=device, dtype=dtype)
    effective_chunk_chars = chunk_chars or default_chunk_chars(profile, cold_backend.device)
    cold_request = SynthesisRequest(
        text=text,
        profile=profile,
        voice=voice,
        language=language,
        instruct=instruct,
        chunk_chars=effective_chunk_chars,
    )
    _apply_seed(seed)
    cold_result = cold_backend.synthesize(cold_request)
    cold_audio_duration = len(cold_result.audio) / cold_result.sample_rate

    warm_backend = QwenBackend(device=device, dtype=dtype)
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
    for run_index in range(max(warm_runs, 1)):
        _apply_seed(None if seed is None else seed + run_index + 1)
        result = warm_backend.synthesize(warm_request)
        warm_times.append(result.elapsed_sec)
        warm_audio_durations.append(len(result.audio) / result.sample_rate)

    warm_avg = sum(warm_times) / len(warm_times)
    warm_audio_duration_avg = sum(warm_audio_durations) / len(warm_audio_durations)
    return BenchmarkResult(
        profile=profile.name,
        model_id=profile.model_id,
        device=warm_backend.device,
        dtype=warm_backend.dtype_name,
        seed=seed,
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
    )


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
