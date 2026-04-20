from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from tts_qwen_local.progress import (
    EstimatedProgressTracker,
    estimate_generation_seconds,
    render_progress_line,
)


class _FakeStderr:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        with self._lock:
            self._parts.append(text)
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)


class ProgressTests(unittest.TestCase):
    def test_estimate_generation_seconds_increases_for_slower_profiles(self):
        fast = estimate_generation_seconds("mlx", "fast", 1200, 6)
        quality = estimate_generation_seconds("mlx", "quality", 1200, 6)
        clone = estimate_generation_seconds("mlx", "clone-fast", 1200, 6)

        self.assertGreater(quality, fast)
        self.assertGreater(clone, quality)

    def test_render_progress_line_includes_eta(self):
        line = render_progress_line("synth fast", elapsed=3.5, estimated_seconds=10.0)

        self.assertIn("ETA", line)
        self.assertIn("synth fast", line)

    def test_tracker_writes_eta_line_to_stderr(self):
        fake_stderr = _FakeStderr()
        with mock.patch("tts_qwen_local.progress.sys.stderr", fake_stderr):
            tracker = EstimatedProgressTracker("synth fast", 0.05, enabled=True, tick_seconds=0.001)
            tracker.start()
            time.sleep(0.01)
            tracker.stop()

        output = fake_stderr.getvalue()
        self.assertIn("ETA", output)
        self.assertIn("synth fast", output)
