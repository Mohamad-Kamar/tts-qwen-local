from __future__ import annotations

import platform

from ..config import default_backend_name, default_mlx_python
from .base import TTSBackend
from .mlx_external import MLXExternalBackend
from .qwen import QwenBackend


def create_backend(
    backend_name: str,
    *,
    device: str,
    dtype: str,
    mlx_python: str | None = None,
) -> TTSBackend:
    resolved = resolve_backend_name(backend_name, device=device, mlx_python=mlx_python)
    if resolved == "mlx":
        return MLXExternalBackend(
            device=device,
            dtype=dtype,
            mlx_python=mlx_python or _default_mlx_python_str(),
        )
    return QwenBackend(device=device, dtype=dtype)


def resolve_backend_name(
    backend_name: str,
    *,
    device: str,
    mlx_python: str | None = None,
) -> str:
    if backend_name != "auto":
        return backend_name

    configured = default_backend_name()
    if configured != "auto":
        return configured

    if device == "cuda":
        return "pytorch"
    if platform.system() != "Darwin":
        return "pytorch"
    if mlx_python or default_mlx_python():
        return "mlx"
    return "pytorch"


def _default_mlx_python_str() -> str | None:
    path = default_mlx_python()
    if path is None:
        return None
    return str(path)
