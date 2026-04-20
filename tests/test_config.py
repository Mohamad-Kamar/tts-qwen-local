from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tts_qwen_local.config import (
    DEFAULT_PROFILE_NAME,
    PROFILE_MAP,
    default_chunk_chars,
    get_profile,
    load_presets,
    normalize_language,
    validate_clone_options,
    validate_synth_options,
)


class ConfigTests(unittest.TestCase):
    def test_default_profile_exists(self):
        self.assertIn(DEFAULT_PROFILE_NAME, PROFILE_MAP)

    def test_get_profile(self):
        profile = get_profile("fast")
        self.assertEqual(profile.model_type, "CustomVoice")
        self.assertTrue(profile.supports_voice)
        self.assertFalse(profile.supports_clone)

    def test_normalize_language_alias(self):
        self.assertEqual(normalize_language("en"), "English")
        self.assertEqual(normalize_language("english"), "English")
        self.assertEqual(normalize_language("auto"), "Auto")

    def test_default_chunk_chars_prefers_device(self):
        profile = get_profile("fast")
        self.assertEqual(default_chunk_chars(profile, "mps"), profile.chunk_chars_mps)
        self.assertEqual(default_chunk_chars(profile, "cpu"), profile.chunk_chars_other)

    def test_validate_synth_rejects_voice_design_without_instruct(self):
        with self.assertRaisesRegex(ValueError, "requires --instruct"):
            validate_synth_options(get_profile("design"), None, None)

    def test_validate_clone_requires_ref_text_or_x_vector_mode(self):
        with self.assertRaisesRegex(ValueError, "requires --ref-text"):
            validate_clone_options(
                get_profile("clone-fast"),
                Path("voice.wav"),
                None,
                False,
            )

    def test_load_presets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "presets.yaml"
            path.write_text(
                "presets:\n  sample:\n    profile: fast\n    voice: Ryan\n",
                encoding="utf-8",
            )
            presets = load_presets(path)
            self.assertIn("sample", presets)
            self.assertEqual(presets["sample"]["voice"], "Ryan")
