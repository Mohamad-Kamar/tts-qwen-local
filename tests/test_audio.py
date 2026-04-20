from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tts_qwen_local.audio import encode_audio_bytes, infer_audio_format, resolve_output_path


class AudioTests(unittest.TestCase):
    def test_infer_format_from_request(self):
        self.assertEqual(infer_audio_format(None, "mp3"), "mp3")

    def test_infer_format_from_output_suffix(self):
        self.assertEqual(infer_audio_format("voice.flac", None), "flac")

    def test_resolve_output_path_defaults(self):
        path = resolve_output_path(None, "wav")
        self.assertEqual(path.name, "output.wav")

    def test_resolve_output_path_adds_suffix(self):
        path = resolve_output_path(Path("voice"), "mp3")
        self.assertEqual(path.name, "voice.mp3")

    def test_encode_audio_bytes_wav(self):
        audio = np.zeros(240, dtype=np.float32)
        payload = encode_audio_bytes(audio, 24000, audio_format="wav")
        self.assertTrue(payload.startswith(b"RIFF"))
