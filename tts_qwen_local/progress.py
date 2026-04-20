from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass


PROGRESS_BAR_WIDTH = 28
PROGRESS_TICK_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class ETAProfile:
    base_sec: float
    per_char_sec: float
    per_chunk_sec: float


ETA_COEFFICIENTS: dict[str, dict[str, ETAProfile]] = {
    "mlx": {
        "fast": ETAProfile(base_sec=3.0, per_char_sec=0.006, per_chunk_sec=1.8),
        "quality": ETAProfile(base_sec=3.0, per_char_sec=0.0075, per_chunk_sec=2.4),
        "design": ETAProfile(base_sec=3.5, per_char_sec=0.0075, per_chunk_sec=2.5),
        "clone-fast": ETAProfile(base_sec=5.0, per_char_sec=0.0075, per_chunk_sec=4.5),
        "clone-quality": ETAProfile(base_sec=6.5, per_char_sec=0.0085, per_chunk_sec=5.5),
    },
    "pytorch": {
        "fast": ETAProfile(base_sec=55.0, per_char_sec=0.004, per_chunk_sec=3.5),
        "quality": ETAProfile(base_sec=60.0, per_char_sec=0.0045, per_chunk_sec=4.0),
        "design": ETAProfile(base_sec=60.0, per_char_sec=0.0045, per_chunk_sec=4.2),
        "clone-fast": ETAProfile(base_sec=70.0, per_char_sec=0.005, per_chunk_sec=5.2),
        "clone-quality": ETAProfile(base_sec=80.0, per_char_sec=0.005, per_chunk_sec=6.5),
    },
}


def estimate_generation_seconds(backend: str, profile: str, text_chars: int, chunk_count: int) -> float:
    backend_table = ETA_COEFFICIENTS.get(backend, ETA_COEFFICIENTS["mlx"])
    coeffs = backend_table.get(profile, backend_table["fast"])
    chars = max(text_chars, 1)
    chunks = max(chunk_count, 1)
    estimate = coeffs.base_sec + (coeffs.per_char_sec * chars) + (coeffs.per_chunk_sec * chunks)
    return max(1.0, estimate)


def render_progress_line(label: str, elapsed: float, estimated_seconds: float) -> str:
    estimated = max(estimated_seconds, 0.1)
    progress = min(elapsed / estimated, 0.999)
    filled = int(progress * PROGRESS_BAR_WIDTH)
    bar = "=" * filled
    if filled < PROGRESS_BAR_WIDTH:
        bar += ">"
        bar += " " * (PROGRESS_BAR_WIDTH - filled - 1)
    percent = progress * 100
    remaining = max(estimated - elapsed, 0.0)
    return (
        f"\r[{bar}] {percent:5.1f}% {label} "
        f"{elapsed:0.1f}s ETA {remaining:0.1f}s"
    )


def clear_progress_line() -> None:
    sys.stderr.write("\r" + " " * 120 + "\r")
    sys.stderr.flush()


class EstimatedProgressTracker:
    def __init__(
        self,
        label: str,
        estimated_seconds: float,
        enabled: bool | None = None,
        *,
        tick_seconds: float = PROGRESS_TICK_SECONDS,
    ) -> None:
        if enabled is None:
            enabled = sys.stderr.isatty()
        self._label = label
        self._estimated_seconds = estimated_seconds
        self._enabled = enabled
        self._tick_seconds = tick_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        if not self._enabled:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._enabled:
            clear_progress_line()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started_at

    def _run(self) -> None:
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - self._started_at
            sys.stderr.write(render_progress_line(self._label, elapsed, self._estimated_seconds))
            sys.stderr.flush()
            time.sleep(self._tick_seconds)
