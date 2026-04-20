# Optimization Plan

## Goal

Make `tts-qwen-local` materially faster on Apple Silicon while keeping the same
Qwen model family and preserving quality-oriented model choices.

## Ranked Paths

1. `mlx-audio` with `mlx-community` Qwen3-TTS models
   - Why first:
     - Apple-native MLX runtime.
     - Quantized model variants are available for the same Qwen3-TTS family.
     - The current PyTorch path is already optimized enough that backend/runtime
       is now the dominant constraint.
   - What to test:
     - `0.6B-CustomVoice-6bit`
     - `0.6B-CustomVoice-8bit`
     - `1.7B-CustomVoice-6bit`
     - `1.7B-VoiceDesign-6bit` if VoiceDesign is viable
   - Success criteria:
     - Better warm latency than the current PyTorch/MPS backend on the same text.
     - No quality collapse relative to the equivalent model class.

2. Official Qwen MLX runtime
   - Why second:
     - Closer to the official Qwen semantics than `mlx-audio`.
     - Useful fallback if `mlx-audio` is faster but has a feature gap or behavioral mismatch.
   - Known constraints:
     - Narrower feature surface than the full PyTorch runtime.
     - VoiceDesign and some instruct behavior may remain PyTorch-only.

3. Long-lived service/runtime reuse
   - Why still worth doing:
     - Eliminates cold-start penalties regardless of backend.
     - Useful if MLX works and we want the study pipeline to consume it later.
   - Scope:
     - Keep models resident in-process or behind a small local service.

4. Remaining PyTorch/MPS tuning
   - Why last:
     - High effort for diminishing returns.
     - Already improved with batching, chunk tuning, and MPS settings.
   - Only continue if MLX proves unusable.

## Execution Order

1. Add backend abstraction to support multiple local runtimes without changing the CLI UX.
2. Implement MLX backend through an isolated runtime path so it does not break the
   existing `qwen-tts` environment.
3. Benchmark MLX and PyTorch on the same texts and profiles.
4. Keep the faster backend as the Apple default if it is stable.
5. If MLX is faster but has feature gaps, keep PyTorch as a fallback backend for
   unsupported modes.

## Dead-End Checks

- If MLX model download reliability is the blocker, prefetch models explicitly
  rather than benchmarking through first-run network fetches.
- If `mlx-audio` cannot match required model features, test the official Qwen MLX runtime.
- If neither MLX path is materially faster, stop investing in Qwen3-TTS on this Mac
  and report that clearly.
