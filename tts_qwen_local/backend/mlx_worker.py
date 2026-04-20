from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from mlx_audio.tts.utils import get_model_path, load
from mlx_audio.utils import load_audio

from ..audio import concat_audio_segments

class WorkerState:
    def __init__(self) -> None:
        self.model_id: str | None = None
        self.model: Any | None = None

    def ensure_loaded(self, model_id: str) -> tuple[Any, dict[str, Any]]:
        start = time.perf_counter()
        if self.model_id == model_id and self.model is not None:
            return self.model, {
                "model_reused": True,
                "model_load_sec": 0.0,
            }
        with contextlib.redirect_stdout(sys.stderr):
            self.model = load(model_id, lazy=False)
        self.model_id = model_id
        return self.model, {
            "model_reused": False,
            "model_load_sec": time.perf_counter() - start,
        }

    def preload(self, model_id: str, include_tokenizer: bool = True) -> list[str]:
        del include_tokenizer
        return [str(get_model_path(model_id))]


def main() -> int:
    state = WorkerState()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        try:
            if method == "shutdown":
                _write_response(request_id, {"status": "bye"})
                return 0
            result = dispatch(state, method, params)
            _write_response(request_id, result)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            details = traceback.format_exc(limit=8)
            _write_error(request_id, f"{message}\n{details}")
    return 0


def dispatch(state: WorkerState, method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "ensure_loaded":
        model, trace = state.ensure_loaded(params["model_id"])
        return {
            "model_id": params["model_id"],
            "sample_rate": int(model.sample_rate),
            "dtype": _dtype_name(model),
            "trace": trace,
        }
    if method == "preload":
        return {"paths": state.preload(params["model_id"], params.get("include_tokenizer", True))}
    if method == "list_voices":
        model, _trace = state.ensure_loaded(params["model_id"])
        voices = []
        if callable(getattr(model, "get_supported_speakers", None)):
            voices = list(model.get_supported_speakers() or [])
        return {"voices": voices}
    if method == "synthesize":
        return _synthesize(state, params)
    if method == "clone":
        return _clone(state, params)
    raise ValueError(f"Unknown method: {method}")


def _synthesize(state: WorkerState, params: dict[str, Any]) -> dict[str, Any]:
    model, load_trace = state.ensure_loaded(params["model_id"])
    chunks = list(params["chunks"])
    start = time.perf_counter()
    generation_start = time.perf_counter()

    if params["model_type"] == "CustomVoice":
        with contextlib.redirect_stdout(sys.stderr):
            results = list(
                model.batch_generate(
                    texts=chunks,
                    voices=[params["voice"]] * len(chunks),
                    instructs=[params.get("instruct")] * len(chunks),
                    lang_code=params["language"],
                    verbose=False,
                )
            )
    elif params["model_type"] == "VoiceDesign":
        with contextlib.redirect_stdout(sys.stderr):
            results = list(
                model.batch_generate(
                    texts=chunks,
                    voices=[None] * len(chunks),
                    instructs=[params.get("instruct")] * len(chunks),
                    lang_code=params["language"],
                    verbose=False,
                )
            )
    else:
        raise ValueError("Clone profiles are not supported by synthesize().")

    model_generate_elapsed = time.perf_counter() - generation_start
    concat_start = time.perf_counter()
    audio = _concat_results(results, sample_rate=int(model.sample_rate))
    concat_elapsed = time.perf_counter() - concat_start
    write_start = time.perf_counter()
    audio_path = _write_temp_array(audio)
    write_elapsed = time.perf_counter() - write_start
    total_elapsed = time.perf_counter() - start
    return {
        "audio_path": str(audio_path),
        "sample_rate": int(model.sample_rate),
        "elapsed_sec": total_elapsed,
        "dtype": _dtype_name(model),
        "trace": {
            **load_trace,
            "model_generate_sec": model_generate_elapsed,
            "concat_sec": concat_elapsed,
            "temp_write_sec": write_elapsed,
            "worker_total_sec": total_elapsed,
            "chunk_count": len(chunks),
            "worker_pid": os.getpid(),
        },
    }


def _clone(state: WorkerState, params: dict[str, Any]) -> dict[str, Any]:
    model, load_trace = state.ensure_loaded(params["model_id"])
    chunks = list(params["chunks"])
    start = time.perf_counter()
    ref_load_start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        reference_audio = load_audio(
            params["reference_audio"],
            sample_rate=model.sample_rate,
        )
    ref_load_elapsed = time.perf_counter() - ref_load_start
    generation_start = time.perf_counter()
    audio_parts: list[np.ndarray] = []

    for chunk in chunks:
        with contextlib.redirect_stdout(sys.stderr):
            results = list(
                model.generate(
                    text=chunk,
                    voice=None,
                    instruct=None,
                    lang_code=params["language"],
                    ref_audio=reference_audio,
                    ref_text=params.get("reference_text"),
                    verbose=False,
                )
            )
        chunk_audio = _concat_results(results)
        if chunk_audio.size:
            audio_parts.append(chunk_audio)

    model_generate_elapsed = time.perf_counter() - generation_start
    concat_start = time.perf_counter()
    audio = concat_audio_segments(audio_parts, int(model.sample_rate))
    concat_elapsed = time.perf_counter() - concat_start
    write_start = time.perf_counter()
    audio_path = _write_temp_array(audio)
    write_elapsed = time.perf_counter() - write_start
    total_elapsed = time.perf_counter() - start
    return {
        "audio_path": str(audio_path),
        "sample_rate": int(model.sample_rate),
        "elapsed_sec": total_elapsed,
        "dtype": _dtype_name(model),
        "trace": {
            **load_trace,
            "reference_load_sec": ref_load_elapsed,
            "model_generate_sec": model_generate_elapsed,
            "concat_sec": concat_elapsed,
            "temp_write_sec": write_elapsed,
            "worker_total_sec": total_elapsed,
            "chunk_count": len(chunks),
            "worker_pid": os.getpid(),
        },
    }


def _concat_results(results: list[Any], sample_rate: int) -> np.ndarray:
    audio_parts: list[np.ndarray] = []
    for result in results:
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            audio_parts.append(audio)
    return concat_audio_segments(audio_parts, sample_rate)


def _write_temp_array(audio: np.ndarray) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
        path = Path(handle.name)
    np.save(path, audio.astype(np.float32, copy=False))
    return path


def _dtype_name(model: Any) -> str:
    dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    if dtype is not None:
        return str(dtype)
    return "mlx"


def _write_response(request_id: Any, result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({"id": request_id, "ok": True, "result": result}) + "\n")
    sys.stdout.flush()


def _write_error(request_id: Any, error: str) -> None:
    sys.stdout.write(json.dumps({"id": request_id, "ok": False, "error": error}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
