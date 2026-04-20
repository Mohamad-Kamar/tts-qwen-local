# Tuning Guide

This document captures the practical settings that came out of local measurement and debugging work.

It is written for two audiences:

- people using this repo directly on Apple Silicon
- people forking this repo and deciding which defaults to keep

## Start Here

If you are on Apple Silicon and just want a reliable local setup:

- use backend `auto` or `mlx`
- use profile `fast`
- keep output as `wav`
- start with the shipped presets in `presets.yaml`

That is the most practical path this repo currently supports.

If the machine is on battery or running hot, also compare `quality`. In the later battery-state reruns, `quality` was more stable than `fast` on longer text.

## Which Profile To Use

### `fast`

Use this when:

- you generate study audio regularly
- you want the best speed-to-quality tradeoff on Apple Silicon
- you care more about “good and usable” than absolute voice quality
- you are willing to verify it against `quality` on your own battery/thermal conditions if throughput starts looking worse than expected

### `quality`

Use this when:

- you want a stronger preset voice
- you are willing to wait longer
- the output is important enough that slower turnaround is acceptable

### `design`

Use this when:

- you want to describe the voice in natural language
- you are iterating on persona/style
- speed is not the first priority

### `clone-fast`

Use this when:

- you have a reference clip
- you want voice similarity without jumping to the biggest clone model first
- you want the most practical clone starting point
- if you do not have the reference transcript, use `--x-vector-only-mode` to force speaker-embedding-only clone

### `clone-quality`

Use this when:

- `clone-fast` is not good enough
- you specifically want to spend more latency on a higher-end clone attempt

## Backend Guidance

### Apple Silicon

Preferred order:

1. `mlx`
2. `auto`
3. `pytorch` only for comparison or fallback

The MLX backend is the serious path on this machine class. The PyTorch path is still supported, but it is not the daily-driver recommendation.

For MLX, `--dtype` is not the main tuning control. Use `--mlx-variant` or `--mlx-model` when you want to compare MLX model choices directly.

### Non-Apple Hardware

This repo still keeps the PyTorch backend, so CUDA users can stay on that path. The Apple-specific MLX guidance does not automatically apply there.

## MLX Variants

The CLI now exposes profile-specific MLX variants directly:

```bash
tts-qwen-local variants --profile fast
```

Use `--mlx-variant` when you want to compare quantized variants for the same profile.

Use `--mlx-model` when you want to point at:

- a specific Hugging Face repo id
- a manually downloaded local model directory

That is the clean escape hatch if you want to experiment without changing repo defaults.

## Chunking Guidance

Chunk size is not universally “bigger is better”.

Practical rules:

- start with the shipped preset values
- keep `fast` near the tested MLX defaults unless you have evidence to change it
- compare chunk sizes on the same power state and backend before deciding
- keep clone chunking fixed while comparing clone variants, otherwise the benchmark becomes noisy

## Tracing And Benchmarking

Use `--trace-json` when you want one detailed record for a single generation.

Use `bench` or `bench-clone` with `--output-json` when you want:

- cold and warm timings
- the actual benchmark inputs used
- backend phase traces for later comparison

Examples:

```bash
tts-qwen-local synth \
  --profile fast \
  --backend mlx \
  --text "Trace this." \
  --trace-json traces.jsonl
```

```bash
tts-qwen-local bench \
  --profile quality \
  --backend mlx \
  --text "Benchmark this." \
  --output-json bench-quality.json
```

## Power And Measurement Hygiene

Power state changed during the local optimization work on this machine, and the timings moved with it.

If you want numbers that are actually comparable:

- compare runs on the same charger/battery state
- avoid parallel large model downloads while benchmarking
- keep the same text, voice, instruction, and reference clip
- prefer JSON artifacts over terminal output when comparing runs later

## When To Fork And Change Defaults

You should keep the shipped defaults if:

- your main machine is Apple Silicon
- `fast` is your daily profile
- you want straightforward local study audio generation

You should change defaults if:

- your main machine is CUDA-first rather than Apple Silicon
- you care more about clone/design fidelity than turnaround time
- you want a different local model layout and prefer `--mlx-model` or repo-specific paths

## See Also

- [CLI Reference](cli-reference.md)
- [Performance Results](performance-results.md)
