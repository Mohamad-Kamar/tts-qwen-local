# CLI Reference

This project is intentionally small. The shortest path is still:

```bash
tts-qwen-local synth
```

That reads `input.txt` and writes `output.wav`.

## Commands

### `models`

List supported profiles.

```bash
tts-qwen-local models
tts-qwen-local models --verbose
```

### `variants`

List supported MLX model variants for one profile or all profiles.

```bash
tts-qwen-local variants
tts-qwen-local variants --profile quality
```

### `voices`

List preset speakers for `CustomVoice` profiles.

```bash
tts-qwen-local voices --profile fast
tts-qwen-local voices --profile quality --backend mlx
```

### `synth`

Generate speech from text.

```bash
tts-qwen-local synth --profile fast --text "Hello."
tts-qwen-local synth --profile fast --input input.txt --output output.wav
tts-qwen-local synth --profile fast --stdin < input.txt
```

Important flags:

- `--profile`
- `--voice`
- `--instruct`
- `--language`
- `--backend`
- `--mlx-variant`
- `--mlx-model`
- `--chunk-chars`
- `--format`
- `--show-settings`
- `--trace-json`

Profile rules:

- `fast` and `quality` accept `--voice` and optional `--instruct`
- `design` requires `--instruct` and does not accept `--voice`
- clone profiles are not valid for `synth`
- in interactive terminals, `synth` shows a live approximate ETA bar instead of per-chunk log lines

### `clone`

Generate speech using a reference clip.

```bash
tts-qwen-local clone \
  --profile clone-fast \
  --reference voice.wav \
  --ref-text "This is the reference speech." \
  --text "Now say this in the same voice."
```

Important flags:

- `--reference`
- `--ref-text`
- `--x-vector-only-mode`
- `--language`
- `--backend`
- `--mlx-variant`
- `--mlx-model`
- `--chunk-chars`
- `--trace-json`

Rules:

- clone profiles are only valid with `clone` and `bench-clone`
- `--ref-text` is recommended
- `--x-vector-only-mode` is the fallback when you do not have the transcript
- when `--x-vector-only-mode` is set, transcript-conditioned cloning is disabled and `--ref-text` is ignored
- in interactive terminals, `clone` shows a live approximate ETA bar instead of per-chunk log lines

### `preload`

Download model weights ahead of time.

```bash
tts-qwen-local preload --profile fast
tts-qwen-local preload --all
tts-qwen-local preload --profile quality --backend mlx --mlx-variant 8bit
```

Important flags:

- `--profile`
- `--all`
- `--backend`
- `--mlx-python`
- `--mlx-variant`
- `--mlx-model`

### `bench`

Run cold and warm synthesis benchmarks.

```bash
tts-qwen-local bench --profile fast --text "Benchmark this."
tts-qwen-local bench --profile quality --backend mlx --runs 2 --output-json bench-quality.json
```

Important flags:

- all relevant `synth` generation flags
- `--runs`
- `--seed`
- `--output-json`

`--output-json` is the better artifact when you want to compare runs later. It includes the resolved inputs and backend traces.

### `bench-clone`

Run cold and warm clone benchmarks.

```bash
tts-qwen-local bench-clone \
  --profile clone-fast \
  --reference voice.wav \
  --ref-text "This is the reference speech." \
  --text "Benchmark this clone path." \
  --output-json bench-clone.json
```

Important flags:

- all relevant `clone` generation flags
- `--runs`
- `--output-json`

## Backend Selection

### `auto`

- on Apple Silicon: prefers `mlx` if an MLX Python exists
- otherwise: falls back to `pytorch`

### `pytorch`

Use the official `qwen-tts` runtime.

### `mlx`

Use the Apple Silicon-native MLX worker.

`--dtype` is mainly relevant to the PyTorch backend. For MLX, the meaningful tuning knobs are `--mlx-variant` and `--mlx-model`.

## MLX Selection

### `--mlx-variant`

Choose a named MLX variant for the selected profile.

Examples:

```bash
tts-qwen-local variants --profile fast
tts-qwen-local synth --profile fast --backend mlx --mlx-variant 8bit --text "Compare this."
```

### `--mlx-model`

Use an explicit MLX model repo id or local path.

Examples:

```bash
tts-qwen-local synth \
  --profile fast \
  --backend mlx \
  --mlx-model mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit \
  --text "Use an explicit repo id."
```

```bash
tts-qwen-local synth \
  --profile fast \
  --backend mlx \
  --mlx-model /path/to/local/model-dir \
  --text "Use a local model directory."
```

## Tracing

Use `--trace-json` on `synth` or `clone` when you want one record per generation.

Use `--output-json` on `bench` or `bench-clone` when you want a benchmark artifact with:

- resolved benchmark inputs
- cold trace
- warm per-run traces
- aggregate timings
