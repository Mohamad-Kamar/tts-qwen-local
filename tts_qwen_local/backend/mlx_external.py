from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    CUSTOMVOICE_SPEAKERS,
    ProfileSpec,
    default_chunk_chars,
    default_mlx_python,
    backend_model_id,
    mlx_model_id,
    normalize_language,
)
from ..text import chunk_text
from .types import CloneRequest, ProgressCallback, SynthesisRequest, SynthesisResult


class MLXExternalBackend:
    name = "mlx"

    def __init__(
        self,
        device: str = "auto",
        dtype: str = "auto",
        mlx_python: str | None = None,
        mlx_variant: str | None = None,
        mlx_model: str | None = None,
    ):
        if device == "cuda":
            raise ValueError("The MLX backend does not support --device cuda.")
        self._device = "mlx"
        self._dtype_mode = dtype
        self._mlx_python = Path(mlx_python) if mlx_python else default_mlx_python()
        self._mlx_variant = mlx_variant
        self._mlx_model = mlx_model
        if self._mlx_python is None:
            raise ValueError(
                "MLX backend requested but no MLX Python was found. "
                "Set --mlx-python or TTS_QWEN_MLX_PYTHON."
            )
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._last_process_trace: dict[str, Any] = {
            "worker_process_reused": False,
            "worker_process_start_sec": 0.0,
        }

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype_name(self) -> str:
        return self._dtype_mode

    def ensure_loaded(self, profile: ProfileSpec) -> None:
        self._rpc("ensure_loaded", model_id=self.model_id_for_profile(profile))

    def preload(self, profile: ProfileSpec, include_tokenizer: bool = True) -> list[Path]:
        payload = self._rpc(
            "preload",
            model_id=self.model_id_for_profile(profile),
            include_tokenizer=include_tokenizer,
        )
        return [Path(item) for item in payload["paths"]]

    def list_voices(self, profile: ProfileSpec) -> list[str]:
        if profile.model_type != "CustomVoice":
            raise ValueError(f"Profile '{profile.name}' does not support preset voices.")

        payload = self._rpc("list_voices", model_id=self.model_id_for_profile(profile))
        voices = payload.get("voices") or []
        if voices:
            return [_canonicalize_voice_name(item) for item in voices]
        return list(CUSTOMVOICE_SPEAKERS)

    def synthesize(
        self,
        request: SynthesisRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        start = time.perf_counter()
        chunk_start = time.perf_counter()
        chunk_chars = request.chunk_chars or default_chunk_chars(request.profile, self._device)
        chunks = chunk_text(request.text, chunk_chars)
        chunk_elapsed = time.perf_counter() - chunk_start
        if not chunks:
            raise ValueError("Input text is empty.")

        for index, chunk in enumerate(chunks):
            if on_progress is not None:
                on_progress(index, len(chunks), chunk)

        rpc_start = time.perf_counter()
        payload = self._rpc(
            "synthesize",
            model_id=self.model_id_for_profile(request.profile),
            model_type=request.profile.model_type,
            chunks=chunks,
            voice=request.voice or request.profile.default_voice,
            instruct=request.instruct,
            language=normalize_language(request.language),
        )
        rpc_elapsed = time.perf_counter() - rpc_start

        audio = self._load_temp_array(payload["audio_path"])
        total_elapsed = time.perf_counter() - start
        return SynthesisResult(
            audio=audio,
            sample_rate=int(payload["sample_rate"]),
            chunks=chunks,
            elapsed_sec=total_elapsed,
            profile=request.profile.name,
            model_id=self.model_id_for_profile(request.profile),
            device=self.device,
            dtype=str(payload.get("dtype") or self.dtype_name),
            trace={
                "chunk_text_sec": chunk_elapsed,
                "rpc_sec": rpc_elapsed,
                "worker_process": dict(self._last_process_trace),
                "worker": payload.get("trace") or {},
            },
        )

    def clone(
        self,
        request: CloneRequest,
        on_progress: ProgressCallback | None = None,
    ) -> SynthesisResult:
        start = time.perf_counter()
        chunk_start = time.perf_counter()
        chunk_chars = request.chunk_chars or default_chunk_chars(request.profile, self._device)
        chunks = chunk_text(request.text, chunk_chars)
        chunk_elapsed = time.perf_counter() - chunk_start
        if not chunks:
            raise ValueError("Input text is empty.")

        for index, chunk in enumerate(chunks):
            if on_progress is not None:
                on_progress(index, len(chunks), chunk)

        rpc_start = time.perf_counter()
        payload = self._rpc(
            "clone",
            model_id=self.model_id_for_profile(request.profile),
            chunks=chunks,
            reference_audio=str(request.reference_audio),
            reference_text=request.reference_text,
            language=normalize_language(request.language),
        )
        rpc_elapsed = time.perf_counter() - rpc_start

        audio = self._load_temp_array(payload["audio_path"])
        total_elapsed = time.perf_counter() - start
        return SynthesisResult(
            audio=audio,
            sample_rate=int(payload["sample_rate"]),
            chunks=chunks,
            elapsed_sec=total_elapsed,
            profile=request.profile.name,
            model_id=self.model_id_for_profile(request.profile),
            device=self.device,
            dtype=str(payload.get("dtype") or self.dtype_name),
            trace={
                "chunk_text_sec": chunk_elapsed,
                "rpc_sec": rpc_elapsed,
                "worker_process": dict(self._last_process_trace),
                "worker": payload.get("trace") or {},
            },
        )

    def close(self) -> None:
        if self._process is None:
            return
        try:
            self._rpc("shutdown")
        except Exception:
            pass
        process = self._process
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def model_id_for_profile(self, profile: ProfileSpec) -> str:
        return backend_model_id(
            profile,
            self.name,
            mlx_variant=self._mlx_variant,
            mlx_model=self._mlx_model,
        )

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            self._last_process_trace = {
                "worker_process_reused": True,
                "worker_process_start_sec": 0.0,
            }
            return self._process

        process_start = time.perf_counter()
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        env.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[2]),
                env.get("PYTHONPATH", ""),
            ]
        ).strip(os.pathsep)

        self._process = subprocess.Popen(
            [
                str(self._mlx_python),
                "-m",
                "tts_qwen_local.backend.mlx_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self._last_process_trace = {
            "worker_process_reused": False,
            "worker_process_start_sec": time.perf_counter() - process_start,
        }
        return self._process

    def _rpc(self, method: str, **params: Any) -> dict[str, Any]:
        process = self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("MLX worker stdio is unavailable.")

        self._request_id += 1
        request = {
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        response = self._read_response(process)
        if response.get("ok"):
            return dict(response.get("result") or {})

        error = response.get("error") or "Unknown MLX worker error."
        raise RuntimeError(error)

    @staticmethod
    def _load_temp_array(path_str: str) -> np.ndarray:
        path = Path(path_str)
        try:
            return np.load(path).astype(np.float32, copy=False)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _read_response(process: subprocess.Popen[str]) -> dict[str, Any]:
        assert process.stdout is not None
        skipped_lines: list[str] = []
        while True:
            response_line = process.stdout.readline()
            if not response_line:
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()[:4000]
                skipped = "\n".join(skipped_lines).strip()
                details = "\n".join(part for part in (skipped, stderr) if part)
                raise RuntimeError(f"MLX worker exited unexpectedly. {details}".strip())
            try:
                return json.loads(response_line)
            except json.JSONDecodeError:
                skipped_lines.append(response_line.strip())


def _canonicalize_voice_name(voice: str) -> str:
    known = {name.lower(): name for name in CUSTOMVOICE_SPEAKERS}
    return known.get(voice.lower(), voice)
