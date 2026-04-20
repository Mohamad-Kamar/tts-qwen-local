from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tts_qwen_local.cli import _apply_preset_defaults, _load_input_text, build_parser


class CliHelperTests(unittest.TestCase):
    def test_load_input_text_from_inline(self):
        parser = build_parser()
        args = parser.parse_args(["synth", "--text", "Hello world"])
        self.assertEqual(_load_input_text(args), "Hello world")

    def test_load_input_text_from_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.txt"
            input_path.write_text("Hello from file", encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["synth", "--input", str(input_path)])
            self.assertEqual(_load_input_text(args), "Hello from file")

    def test_load_input_text_from_stdin(self):
        parser = build_parser()
        args = parser.parse_args(["synth", "--stdin"])
        with mock.patch("sys.stdin", io.StringIO("Hello from stdin")):
            self.assertEqual(_load_input_text(args), "Hello from stdin")

    def test_preset_defaults_fill_profile_when_unspecified(self):
        parser = build_parser()
        args = parser.parse_args(["synth", "--text", "Hello"])
        _apply_preset_defaults(
            args,
            {
                "profile": "quality",
                "voice": "Ryan",
                "backend": "mlx",
                "dtype": "bfloat16",
                "format": "mp3",
            },
        )
        self.assertEqual(args.profile, "quality")
        self.assertEqual(args.voice, "Ryan")
        self.assertEqual(args.backend, "mlx")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.format, "mp3")
