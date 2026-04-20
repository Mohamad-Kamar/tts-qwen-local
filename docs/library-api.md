# Library API

`tts-qwen-local` is not only a CLI. It also exposes a reusable Python service layer intended for app integration.

The main entrypoint is:

- `QwenTTSService`
- `create_service(...)`

The service keeps a backend alive across calls, which matters for performance and makes it suitable for long-lived app processes.

## Main Types

Import from the top-level package:

```python
from tts_qwen_local import (
    CloneOptions,
    QwenTTSService,
    SynthesisOptions,
    TTSProgress,
    create_service,
)
```

### `QwenTTSService`

Long-lived service object for:

- preloading models
- listing voices
- synthesizing speech
- cloning speech
- returning bytes
- writing files

Constructor:

```python
QwenTTSService(
    backend="auto",
    device="auto",
    dtype="auto",
    mlx_python=None,
    mlx_variant=None,
    mlx_model=None,
)
```

Notes:

- on Apple Silicon, `backend="auto"` prefers MLX when an MLX Python exists
- `dtype` mainly matters for the PyTorch backend
- MLX-specific tuning should use `mlx_variant` or `mlx_model`
- the service serializes backend access internally so one shared instance can be reused safely inside an app process

### `SynthesisOptions`

```python
SynthesisOptions(
    text,
    profile="fast",
    voice=None,
    language="auto",
    instruct=None,
    chunk_chars=None,
)
```

Use for:

- `fast`
- `quality`
- `design`

Rules:

- `design` requires `instruct`
- `fast` and `quality` can use `voice`

### `CloneOptions`

```python
CloneOptions(
    text,
    reference_audio,
    profile="clone-fast",
    reference_text=None,
    x_vector_only_mode=False,
    language="auto",
    chunk_chars=None,
)
```

Rules:

- `reference_text` is recommended
- `x_vector_only_mode=True` disables transcript-conditioned cloning and falls back to speaker-embedding-only clone

### `TTSProgress`

Progress callback payload:

```python
TTSProgress(
    chunk_index,
    total_chunks,
    chunk_text,
)
```

This is chunk-planning progress, not a hard real-time ETA contract.

## Common Examples

### Reuse one service instance

```python
from tts_qwen_local import QwenTTSService, SynthesisOptions

tts = QwenTTSService(backend="auto")
tts.preload("fast")

result = tts.synthesize(
    SynthesisOptions(
        text="Hello from a reused service.",
        profile="fast",
        voice="Uncle_Fu",
    )
)

tts.close()
```

### Use it as a context manager

```python
from tts_qwen_local import QwenTTSService, SynthesisOptions

with QwenTTSService(backend="auto") as tts:
    result = tts.synthesize(
        SynthesisOptions(
            text="Hello from a context-managed service.",
            profile="fast",
            voice="Uncle_Fu",
        )
    )
```

### Use the convenience factory

```python
from tts_qwen_local import SynthesisOptions, create_service

with create_service(backend="auto") as tts:
    result = tts.synthesize(
        SynthesisOptions(
            text="Create the service through the helper.",
            profile="fast",
            voice="Uncle_Fu",
        )
    )
```

### Return WAV bytes

```python
from tts_qwen_local import QwenTTSService, SynthesisOptions

with QwenTTSService(backend="auto") as tts:
    wav_bytes = tts.synthesize_to_bytes(
        SynthesisOptions(
            text="Return bytes for another application.",
            profile="fast",
            voice="Uncle_Fu",
        )
    )
```

### Write directly to a file

```python
from pathlib import Path

from tts_qwen_local import QwenTTSService, SynthesisOptions

with QwenTTSService(backend="auto") as tts:
    output_path = tts.synthesize_to_file(
        SynthesisOptions(
            text="Write this straight to disk.",
            profile="fast",
            voice="Uncle_Fu",
        ),
        Path("output.wav"),
    )
```

### Clone from a reference clip

```python
from tts_qwen_local import CloneOptions, QwenTTSService

with QwenTTSService(backend="auto") as tts:
    result = tts.clone(
        CloneOptions(
            text="Use the reference voice for this output.",
            profile="clone-fast",
            reference_audio="voice.wav",
            reference_text="This is the reference speech.",
        )
    )
```

### Progress callback

```python
from tts_qwen_local import QwenTTSService, SynthesisOptions, TTSProgress

def on_progress(update: TTSProgress) -> None:
    print(f"{update.chunk_index + 1}/{update.total_chunks}: {update.chunk_text[:40]}")

with QwenTTSService(backend="auto") as tts:
    tts.synthesize(
        SynthesisOptions(
            text="A longer text that will be chunked before synthesis.",
            profile="fast",
            voice="Uncle_Fu",
        ),
        on_progress=on_progress,
    )
```

## Methods

`QwenTTSService` exposes:

- `list_profiles()`
- `preload(profile, include_tokenizer=True)`
- `preload_all(include_tokenizer=True)`
- `list_voices(profile="fast")`
- `synthesize(options, on_progress=None)`
- `clone(options, on_progress=None)`
- `synthesize_to_bytes(options, audio_format="wav")`
- `clone_to_bytes(options, audio_format="wav")`
- `synthesize_to_file(options, output_path, audio_format=None)`
- `clone_to_file(options, output_path, audio_format=None)`
- `close()`

## Integration Guidance

If you want to port this into a `study-pipeline` style app:

- keep one `QwenTTSService` instance per process
- call `preload()` during startup for the main profile
- expose `list_voices()` for preset-speaker UI
- use `synthesize_to_bytes()` when your HTTP layer wants WAV bytes
- use `synthesize_to_file()` when your workflow writes assets directly to disk
- prefer `clone-fast` before `clone-quality`
- treat the service as the public integration boundary rather than reaching into backend modules directly

## See Also

- [README](../README.md)
- [CLI Reference](cli-reference.md)
- [Tuning Guide](tuning-guide.md)
