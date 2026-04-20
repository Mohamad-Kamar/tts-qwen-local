# tts-qwen-local

`tts-qwen-local` is an offline-first Python CLI and library for generating speech with the official `Qwen3-TTS` model family.

It is designed for the same simple flow as `tts-local`: give it text, get an audio file back. It adds Qwen-specific model profiles, voice presets, preload support, cloning support, and reproducible benchmarks.

## Status

This project now supports two local runtimes:

- `pytorch`: the original `qwen-tts` backend
- `mlx`: an Apple Silicon-native backend that runs through an isolated `mlx-audio` Python environment

On Apple Silicon, `auto` prefers `mlx` when an MLX Python is available. On this machine class, `fast` (`0.6B-CustomVoice`) is the intended daily-use profile.

## Features

- Input from file, inline text, or stdin
- Friendly model profiles:
  - `fast` -> `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
  - `quality` -> `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `design` -> `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
  - `clone-fast` -> `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  - `clone-quality` -> `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Preset voices for `CustomVoice`
- Optional style/design instructions
- Voice cloning with reusable prompt caching within a process
- Stable sentence-based chunking for long text
- `preload` command for downloading weights ahead of time
- `bench` command for cold/warm timing measurements
- Automatic backend selection on Apple Silicon
- Optional ffmpeg-based output conversion to `mp3`, `m4a`, `aac`, `flac`, or `opus`

## Requirements

- Python 3.12 recommended
- Apple Silicon, CUDA, or CPU
- Optional: `ffmpeg` for non-WAV output
- Optional but recommended on Apple Silicon: a separate MLX environment, for example at `../.venv-mlx-audio`

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional Apple Silicon MLX env:

```bash
python3.12 -m venv ../.venv-mlx-audio
source ../.venv-mlx-audio/bin/activate
python -m pip install -U mlx-audio
```

## Quick Start

List profiles:

```bash
tts-qwen-local models
```

Generate speech with the fast profile:

```bash
tts-qwen-local synth --profile fast --text "Hello from Qwen three TTS."
```

Force the Apple-native backend explicitly:

```bash
tts-qwen-local synth --profile fast --backend mlx --text "Hello from Qwen three TTS."
```

Generate from a file:

```bash
tts-qwen-local synth --profile fast --input input.txt --output output.wav
```

List voices for a `CustomVoice` profile:

```bash
tts-qwen-local voices --profile fast
```

Preload the daily-use model:

```bash
tts-qwen-local preload --profile fast
```

Clone a voice:

```bash
tts-qwen-local clone \
  --profile clone-fast \
  --reference voice.wav \
  --ref-text "This is the reference speech." \
  --text "Now synthesize this in the same voice."
```

Run a benchmark:

```bash
tts-qwen-local bench --profile fast --text "Benchmark this."
```

## Presets

The project reads `presets.yaml` from the repo root by default if it exists. You can also pass `--config path/to/presets.yaml`.

Example:

```yaml
presets:
  study-fast:
    profile: fast
    voice: Ryan
    language: en
    format: wav
    chunk_chars: 120
```

Use a preset:

```bash
tts-qwen-local synth --preset study-fast --input input.txt
```

## Apple Silicon Notes

- The default device is `auto`, which resolves to `mps` on Apple Silicon when available.
- The default dtype is `auto`, which prefers `bfloat16` on MPS and falls back to `float32` if generation becomes unstable.
- If `../.venv-mlx-audio/bin/python` or `TTS_QWEN_MLX_PYTHON` exists, `auto` resolves to the MLX backend on Apple Silicon.
- `fast` is the practical everyday option on this machine.
- The default MPS chunking is intentionally smaller than the original prototype because batched smaller chunks were measurably faster on this Mac.
- In local testing, the MLX fast path was materially faster than the PyTorch MPS path for the same `fast` profile.
- `quality` and `design` are slower and should be treated as deliberate higher-quality runs.
- Quantized CUDA-style acceleration is intentionally not part of this v1 project.

## Output Formats

Default output is `wav`. If `ffmpeg` is available, you can also request:

- `mp3`
- `m4a`
- `aac`
- `flac`
- `opus`

Without `ffmpeg`, non-WAV output fails with a clear error.

## Testing

```bash
python -m unittest discover -s tests -v
```
