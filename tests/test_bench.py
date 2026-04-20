from __future__ import annotations

import unittest

from tts_qwen_local.bench import _effective_seed


class BenchTests(unittest.TestCase):
    def test_effective_seed_keeps_pytorch_seed(self):
        self.assertEqual(_effective_seed(123, "pytorch"), 123)

    def test_effective_seed_omits_non_pytorch_seed(self):
        self.assertIsNone(_effective_seed(123, "mlx"))
        self.assertIsNone(_effective_seed(123, "auto"))
