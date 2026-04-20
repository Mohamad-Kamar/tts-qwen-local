from __future__ import annotations

import unittest
from unittest import mock

from tts_qwen_local.backend.factory import resolve_backend_name


class BackendFactoryTests(unittest.TestCase):
    def test_explicit_backend_is_preserved(self):
        self.assertEqual(
            resolve_backend_name("pytorch", device="auto", mlx_python=None),
            "pytorch",
        )
        self.assertEqual(
            resolve_backend_name("mlx", device="auto", mlx_python=None),
            "mlx",
        )

    def test_auto_prefers_pytorch_for_cuda(self):
        self.assertEqual(
            resolve_backend_name("auto", device="cuda", mlx_python="/tmp/mlx-python"),
            "pytorch",
        )

    @mock.patch("tts_qwen_local.backend.factory.platform.system", return_value="Darwin")
    @mock.patch("tts_qwen_local.backend.factory.default_backend_name", return_value="auto")
    def test_auto_prefers_mlx_on_darwin_when_python_is_available(
        self,
        _default_backend_name: mock.Mock,
        _platform_system: mock.Mock,
    ):
        self.assertEqual(
            resolve_backend_name("auto", device="auto", mlx_python="/tmp/mlx-python"),
            "mlx",
        )

    @mock.patch("tts_qwen_local.backend.factory.platform.system", return_value="Darwin")
    @mock.patch("tts_qwen_local.backend.factory.default_backend_name", return_value="pytorch")
    def test_auto_respects_backend_env_override(
        self,
        _default_backend_name: mock.Mock,
        _platform_system: mock.Mock,
    ):
        self.assertEqual(
            resolve_backend_name("auto", device="auto", mlx_python="/tmp/mlx-python"),
            "pytorch",
        )
