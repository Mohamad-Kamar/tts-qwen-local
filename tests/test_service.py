from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tts_qwen_local.backend.types import SynthesisResult
from tts_qwen_local.service import CloneOptions, QwenTTSService, SynthesisOptions, TTSProgress


class _FakeBackend:
    name = "mlx"
    device = "mlx"
    dtype_name = "mlx"

    def __init__(self) -> None:
        self.preload_calls: list[str] = []
        self.ensure_loaded_calls: list[str] = []
        self.list_voices_calls: list[str] = []
        self.synth_requests = []
        self.clone_requests = []
        self.closed = False

    def ensure_loaded(self, profile):
        self.ensure_loaded_calls.append(profile.name)

    def preload(self, profile, include_tokenizer: bool = True):
        self.preload_calls.append(profile.name)
        return [Path(f"/tmp/{profile.name}")]

    def list_voices(self, profile):
        self.list_voices_calls.append(profile.name)
        return ["Ryan", "Vivian"]

    def synthesize(self, request, on_progress=None):
        self.synth_requests.append(request)
        if on_progress is not None:
            on_progress(0, 1, request.text)
        return SynthesisResult(
            audio=np.zeros(240, dtype=np.float32),
            sample_rate=24000,
            chunks=[request.text],
            elapsed_sec=1.0,
            profile=request.profile.name,
            model_id=request.profile.model_id,
            device=self.device,
            dtype=self.dtype_name,
        )

    def clone(self, request, on_progress=None):
        self.clone_requests.append(request)
        if on_progress is not None:
            on_progress(0, 1, request.text)
        return SynthesisResult(
            audio=np.zeros(240, dtype=np.float32),
            sample_rate=24000,
            chunks=[request.text],
            elapsed_sec=1.0,
            profile=request.profile.name,
            model_id=request.profile.model_id,
            device=self.device,
            dtype=self.dtype_name,
        )

    def close(self):
        self.closed = True


class ServiceTests(unittest.TestCase):
    @mock.patch("tts_qwen_local.service.create_backend")
    def test_service_synthesize_emits_progress_and_resolves_profile(self, create_backend: mock.Mock):
        backend = _FakeBackend()
        create_backend.return_value = backend
        service = QwenTTSService()
        progress_updates: list[TTSProgress] = []

        result = service.synthesize(
            SynthesisOptions(text="Hello world", profile="fast", voice="Ryan"),
            on_progress=progress_updates.append,
        )

        self.assertEqual(result.profile, "fast")
        self.assertEqual(backend.synth_requests[0].profile.name, "fast")
        self.assertEqual(len(progress_updates), 1)
        self.assertEqual(progress_updates[0].chunk_text, "Hello world")

    @mock.patch("tts_qwen_local.service.create_backend")
    def test_clone_x_vector_only_clears_reference_text(self, create_backend: mock.Mock):
        backend = _FakeBackend()
        create_backend.return_value = backend
        service = QwenTTSService()

        service.clone(
            CloneOptions(
                text="Clone this",
                profile="clone-fast",
                reference_audio="voice.wav",
                reference_text="Reference text",
                x_vector_only_mode=True,
            )
        )

        self.assertIsNone(backend.clone_requests[0].reference_text)
        self.assertTrue(backend.clone_requests[0].x_vector_only_mode)

    @mock.patch("tts_qwen_local.service.create_backend")
    def test_list_voices_ensures_loaded(self, create_backend: mock.Mock):
        backend = _FakeBackend()
        create_backend.return_value = backend
        service = QwenTTSService()

        voices = service.list_voices("fast")

        self.assertEqual(voices, ["Ryan", "Vivian"])
        self.assertEqual(backend.ensure_loaded_calls, ["fast"])
        self.assertEqual(backend.list_voices_calls, ["fast"])

    @mock.patch("tts_qwen_local.service.create_backend")
    def test_synthesize_to_file_adds_requested_suffix(self, create_backend: mock.Mock):
        backend = _FakeBackend()
        create_backend.return_value = backend
        service = QwenTTSService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = service.synthesize_to_file(
                SynthesisOptions(text="Hello world", profile="fast", voice="Ryan"),
                Path(tmp_dir) / "voice",
                audio_format="wav",
            )

            self.assertEqual(output.suffix, ".wav")
            self.assertTrue(output.exists())

    @mock.patch("tts_qwen_local.service.create_backend")
    def test_service_context_manager_closes_backend(self, create_backend: mock.Mock):
        backend = _FakeBackend()
        create_backend.return_value = backend

        with QwenTTSService() as service:
            self.assertEqual(service.backend_name, "mlx")

        self.assertTrue(backend.closed)
