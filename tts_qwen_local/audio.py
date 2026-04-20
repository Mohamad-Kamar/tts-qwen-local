from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DEFAULT_OUTPUT_STEM, SUPPORTED_AUDIO_FORMATS


def infer_audio_format(output: str | Path | None, requested_format: str | None) -> str:
    if requested_format:
        audio_format = requested_format.lower()
    elif output:
        suffix = Path(output).suffix.lower().lstrip(".")
        audio_format = suffix or "wav"
    else:
        audio_format = "wav"

    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
        raise ValueError(f"Unsupported audio format '{audio_format}'. Supported: {supported}")
    return audio_format


def resolve_output_path(output: str | Path | None, audio_format: str) -> Path:
    if output is None:
        return Path(f"{DEFAULT_OUTPUT_STEM}.{audio_format}")

    path = Path(output)
    if not path.suffix:
        return path.with_suffix(f".{audio_format}")
    return path


def write_audio_file(output_path: Path, audio: np.ndarray, sample_rate: int, audio_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_format == "wav":
        sf.write(output_path, audio, sample_rate, format="WAV")
        return

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"Requested output format '{audio_format}' requires ffmpeg, but ffmpeg was not found."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_wav = Path(handle.name)

    try:
        sf.write(temp_wav, audio, sample_rate, format="WAV")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_wav),
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
    finally:
        temp_wav.unlink(missing_ok=True)


def encode_audio_bytes(audio: np.ndarray, sample_rate: int, audio_format: str = "wav") -> bytes:
    if audio_format == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / f"audio.{audio_format}"
        write_audio_file(output_path, audio, sample_rate, audio_format)
        return output_path.read_bytes()
