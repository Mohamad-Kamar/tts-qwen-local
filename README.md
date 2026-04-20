# tts-qwen-local

`tts-qwen-local` is an offline-first CLI and library for local `Qwen3-TTS` speech generation.

It keeps the same simple contract as `tts-local`: give it text, get an audio file back. The difference is that this project is built around the `Qwen3-TTS` model family and supports both the original PyTorch runtime and an Apple Silicon-native MLX runtime.

## Best Default

On Apple Silicon, the practical default is:

- backend: `auto -> mlx` when an MLX Python is available
- profile: `fast`
- output: `wav`

That is the path this repo is optimized around.

## Default Flow

With no input flags:

```bash
tts-qwen-local synth
```

That reads `input.txt` and writes `output.wav`.

Input priority is:

- `--text`
- `--stdin`
- `--input input.txt`

Output rules are:

- default output path is `output.<format>` (`output.wav` unless you explicitly set `--format`)
- `--output name` adds the requested format suffix if needed
- `--format` defaults to the file suffix, otherwise `wav`

## Profiles

- `fast`: daily-use preset voice generation
- `quality`: slower but stronger preset voice generation
- `design`: voice design from a natural-language instruction
- `clone-fast`: smaller cloning profile
- `clone-quality`: larger cloning profile

Use these rules:

- use `fast` for routine study audio
- use `quality` when you want a better preset voice and can wait longer
- use `design` only when you want to describe the voice/persona
- use `clone-*` only when you have a reference clip

## Install

Python 3.12 is recommended.

Main project environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional Apple Silicon MLX environment:

```bash
python3.12 -m venv ../.venv-mlx-audio
source ../.venv-mlx-audio/bin/activate
python -m pip install -U mlx-audio
```

If `../.venv-mlx-audio/bin/python` exists, `backend=auto` will prefer MLX on Apple Silicon.

## Quick Start

List profiles:

```bash
tts-qwen-local models
```

List MLX variants for one profile:

```bash
tts-qwen-local variants --profile fast
```

List voices:

```bash
tts-qwen-local voices --profile fast
```

Generate speech from inline text:

```bash
tts-qwen-local synth --profile fast --text "Hello from Qwen three TTS."
```

Generate speech from a file:

```bash
tts-qwen-local synth --profile fast --input input.txt --output output.wav
```

Force the Apple-native runtime:

```bash
tts-qwen-local synth --profile fast --backend mlx --text "Hello from Qwen three TTS."
```

Choose an MLX variant directly:

```bash
tts-qwen-local synth \
  --profile fast \
  --backend mlx \
  --mlx-variant 8bit \
  --text "Compare this against the default MLX variant."
```

Point MLX at a specific local model directory or repo id:

```bash
tts-qwen-local synth \
  --profile fast \
  --backend mlx \
  --mlx-model mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit \
  --text "Use an explicit MLX model."
```

Use voice design:

```bash
tts-qwen-local synth \
  --profile design \
  --instruct "Calm, clear, steady educational narration with warm pacing." \
  --text "This is a design-mode example."
```

Clone a voice:

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

Preload a model:

```bash
tts-qwen-local preload --profile fast
```

Benchmark synthesis:

```bash
tts-qwen-local bench --profile fast --text "Benchmark this."
```

Benchmark cloning:

```bash
tts-qwen-local bench-clone \
  --profile clone-fast \
  --reference voice.wav \
  --ref-text "This is the reference speech." \
  --text "Benchmark this clone path."
```

Write a trace record:

```bash
tts-qwen-local synth \
  --profile fast \
  --text "Trace this." \
  --trace-json traces.jsonl
```

## Presets

The repo reads `presets.yaml` by default. You can also pass `--config path/to/presets.yaml`.

Use a preset like this:

```bash
tts-qwen-local synth --preset study-fast --input input.txt
```

The shipped presets are MLX-oriented on Apple Silicon and use chunk sizes that measured well on this Mac.

## Output Formats

Default output is `wav`.

With `ffmpeg`, the CLI can also write:

- `mp3`
- `m4a`
- `aac`
- `flac`
- `opus`

Without `ffmpeg`, non-WAV output fails with a clear error.

## Apple Silicon Notes

- `backend=auto` prefers `mlx` if an MLX Python exists
- `device` matters for the PyTorch backend; MLX always reports `device=mlx`
- `dtype` is mainly a PyTorch control; MLX tuning is done through `--mlx-variant` or `--mlx-model`
- `fast` is the practical everyday profile on this machine class
- if the machine is on battery or running hot, compare `quality`; the later battery-state measurements were more stable there
- `quality`, `design`, and clone modes are deliberate slower runs
- power state matters for measurements, so compare runs on the same charger/battery state when possible

## Docs

- [CLI Reference](docs/cli-reference.md)
- [Tuning Guide](docs/tuning-guide.md)
- [Performance Results](docs/performance-results.md)

## Testing

```bash
python -m unittest discover -s tests -v
```
