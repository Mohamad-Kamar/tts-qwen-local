# tts-qwen-local

Local `Qwen3-TTS` CLI and Python library.

This repo is for one job: take text, generate speech locally, and keep the workflow simple enough to use every day.

## What You Get

- local `Qwen3-TTS` generation
- straightforward CLI defaults
- preset voices, voice design, and voice cloning
- Apple Silicon MLX support
- original PyTorch `qwen-tts` support
- reusable library API for app integration
- benchmarking and tracing tools

## When To Use This Repo

Use `tts-qwen-local` when you want the local Qwen path.

- use this repo for local `Qwen3-TTS`, especially on Apple Silicon with MLX
- use `tts-local` when you want the existing Kokoro-based local path
- use `tts-cloud` when you want API-backed speech generation

## Who This Is For

Use this repo if you want:

- local text-to-speech with no required cloud dependency
- a CLI that stays close to the “input text, get audio file” workflow
- a Python service object another app can hold onto and reuse
- access to preset voices, design mode, and clone mode in one project

## Requirements

- Python `3.12` is recommended
- Apple Silicon is optional, but it is the best-supported path for this repo today because of MLX
- `ffmpeg` is optional and only needed for non-`wav` output such as `mp3`, `m4a`, `aac`, `flac`, or `opus`

## Current Recommendation

On Apple Silicon, start here:

- backend: `auto` or `mlx`
- profile: `fast`
- output: `wav`

If the machine is on battery or running hot, compare `quality`. In the later MLX measurements, `quality` was more stable than `fast` on longer text.

## Install

Main project environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional MLX environment for Apple Silicon:

```bash
python3.12 -m venv ../.venv-mlx-audio
source ../.venv-mlx-audio/bin/activate
python -m pip install -U mlx-audio
```

This environment is only the MLX worker runtime. It does not install the `tts-qwen-local`
CLI. Keep using the main project environment when you run `tts-qwen-local synth`.

If `../.venv-mlx-audio/bin/python` exists, `backend=auto` will prefer MLX on Apple Silicon.

## Quick Start

Default flow:

```bash
source .venv/bin/activate
tts-qwen-local synth
```

That reads `input.txt` and writes `output.wav`.

In an interactive terminal, `synth` and `clone` show a live approximate ETA bar on stderr.

Generate from inline text:

```bash
tts-qwen-local synth --profile fast --text "Hello from Qwen three TTS."
```

Generate from a file:

```bash
tts-qwen-local synth --profile fast --input input.txt --output output.wav
```

List voices:

```bash
tts-qwen-local voices --profile fast
```

Preload a model:

```bash
tts-qwen-local preload --profile fast
```

## Common Workflows

Use voice design:

```bash
tts-qwen-local synth \
  --profile design \
  --instruct "Calm, clear, steady educational narration with warm pacing." \
  --text "This is a design-mode example."
```

Clone from a reference clip:

```bash
tts-qwen-local clone \
  --profile clone-fast \
  --reference voice.wav \
  --ref-text "This is the reference speech." \
  --text "Now synthesize this in the same voice."
```

Clone without transcript conditioning:

```bash
tts-qwen-local clone \
  --profile clone-fast \
  --reference voice.wav \
  --x-vector-only-mode \
  --text "Clone this without transcript conditioning."
```

Benchmark a profile:

```bash
tts-qwen-local bench --profile fast --text "Benchmark this."
```

## Profiles

- `fast`: everyday preset-voice synthesis
- `quality`: slower preset-voice synthesis with stronger results
- `design`: voice design from a natural-language instruction
- `clone-fast`: smaller cloning profile
- `clone-quality`: larger cloning profile

Practical guidance:

- start with `fast`
- compare `quality` if you want better output or if `fast` is unstable under current power or thermal conditions
- use `design` when you want a described persona, not just a preset speaker
- use `clone-fast` before `clone-quality`

## Library Usage

`tts-qwen-local` can be embedded as a reusable library. The main entrypoint is `QwenTTSService`, which keeps a backend alive across calls instead of paying full startup cost on every request.

```python
from tts_qwen_local import QwenTTSService, SynthesisOptions

with QwenTTSService(backend="auto") as tts:
    tts.preload("fast")
    result = tts.synthesize(
        SynthesisOptions(
            text="Hello from the reusable Qwen TTS service.",
            profile="fast",
            voice="Uncle_Fu",
        )
    )
    wav_bytes = tts.synthesize_to_bytes(
        SynthesisOptions(
            text="Return WAV bytes for another app.",
            profile="fast",
            voice="Uncle_Fu",
        )
    )
```

For integration details, see [docs/library-api.md](docs/library-api.md).

## Presets

Use `presets.yaml` for named defaults when you want a stable study workflow without retyping the same flags every time.

```bash
tts-qwen-local synth --preset study-fast --input input.txt
```

## Output

Default output is `wav`.

With `ffmpeg`, the CLI and library can also write:

- `mp3`
- `m4a`
- `aac`
- `flac`
- `opus`

Without `ffmpeg`, non-WAV output fails with a clear error.

## Docs

- [CLI Reference](docs/cli-reference.md): commands and flags
- [Library API](docs/library-api.md): reusable service-layer usage
- [Tuning Guide](docs/tuning-guide.md): backend and model tuning
- [Performance Results](docs/performance-results.md): measured results and tradeoffs

## Testing

```bash
python -m unittest discover -s tests -v
```
