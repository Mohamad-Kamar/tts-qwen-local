# Performance Results

This document records the local optimization work on the MLX and PyTorch backends.

## Measurement Notes

- machine class: Apple Silicon Mac
- backend focus: MLX on Apple Silicon
- important caveat: power state changed during the broader optimization session, and later follow-up measurements were taken while the machine was on battery power
- practical rule: compare runs captured under the same power state before drawing close conclusions

Current power snapshot during the later follow-up benchmarking work:

- `Battery Power`
- `84%`
- `discharging`

## Main Conclusions

- `auto -> mlx` is the right default path on Apple Silicon
- the original PyTorch path is still useful for comparison and fallback, but it is not the daily-driver path on this machine
- `fast` remains the simplest starting profile, but not a guaranteed throughput winner in every battery/thermal state
- `quality` is now a real option, not just a theoretical higher-end mode
- `design` stayed usable in the measured MLX path
- `clone-fast` is practical in the MLX `6bit` path, especially with `--x-vector-only-mode`
- chunk size still matters; defaults should be set from measurement, not guesswork

## Commands Used

Short benchmark, `quality`, MLX:

```bash
tts-qwen-local bench \
  --profile quality \
  --backend mlx \
  --voice Ryan \
  --text "Hello. This is a short MLX benchmark sample with two sentences. The goal is to compare backend latency and not just audio output." \
  --runs 2 \
  --output-json output/bench-quality-6bit.json
```

Longer real-text slice, `quality`, MLX:

```bash
head -c 1200 notes.txt > /tmp/tts-qwen-notes-1200.txt

tts-qwen-local bench \
  --profile quality \
  --backend mlx \
  --voice Ryan \
  --input /tmp/tts-qwen-notes-1200.txt \
  --runs 2 \
  --output-json output/bench-quality-1200.json
```

Full-file `quality` synthesis:

```bash
tts-qwen-local synth \
  --profile quality \
  --backend mlx \
  --voice Ryan \
  --input notes.txt \
  --output output/quality-notes.wav \
  --trace-json output/quality-notes-trace.jsonl
```

## Baseline Results

Earlier `fast` measurements that established the MLX path:

- short sample, `fast`, MLX:
  - `cold_elapsed_sec=7.03`
  - `warm_elapsed_sec_avg=3.17`
  - `cold_audio_duration_sec=15.04`
  - `warm_rtf_avg=0.27`
- short sample, `fast`, PyTorch:
  - `cold_elapsed_sec=81.68`
  - `warm_elapsed_sec_avg=21.55`
  - `cold_audio_duration_sec=26.72`
  - `warm_rtf_avg=1.86`
- full `notes.txt`, `fast`, MLX:
  - `elapsed_sec=59.82`
  - `audio_duration_sec=455.76`
  - `chunk_count=21`

Those runs were enough to justify the Apple-native MLX backend.

## Follow-Up Results

### `quality` short sample

Artifact:

- `output/bench-quality-6bit.json`

Measured result:

- `model_id=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit`
- `chunk_chars=400`
- `cold_elapsed_sec=6.61`
- `cold_audio_duration_sec=12.16`
- `cold_rtf=0.54`
- `warm_elapsed_sec_avg=3.55`
- `warm_audio_duration_sec_avg=10.36`
- `warm_rtf_avg=0.34`

Important trace details from the cold run:

- `model_load_sec=2.94`
- `model_generate_sec=3.19`
- `worker_total_sec=3.20`

### `quality` 1200-character slice

Artifact:

- `output/bench-quality-1200.json`

Measured result:

- `chunk_chars=400`
- `chunk_count=6`
- `cold_elapsed_sec=78.73`
- `cold_audio_duration_sec=145.12`
- `cold_rtf=0.54`
- `warm_elapsed_sec_avg=24.84`
- `warm_audio_duration_sec_avg=94.88`
- `warm_rtf_avg=0.26`

### `quality` full `notes.txt`

Artifacts:

- `output/quality-notes.wav`
- `output/quality-notes-trace.jsonl`

Measured result:

- `chars=5497`
- `chunk_chars=400`
- `chunk_count=24`
- `elapsed_sec=75.60`
- `audio_duration_sec=416.24`
- `rtf=0.18`

Important trace details:

- `model_load_sec=3.11`
- `model_generate_sec=71.82`
- `worker_total_sec=71.85`

### `design` short sample

Artifacts:

- `output/bench-design-6bit.json`

Measured result:

- `model_id=mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit`
- `chunk_chars=400`
- `cold_elapsed_sec=8.10`
- `cold_audio_duration_sec=8.96`
- `cold_rtf=0.90`
- `warm_elapsed_sec_avg=3.38`
- `warm_audio_duration_sec_avg=9.68`
- `warm_rtf_avg=0.35`

Fixed instruction used:

- `Calm, clear, steady educational narration with warm pacing.`

### `design` 1200-character slice

Artifacts:

- `output/bench-design-1200.json`

Measured result:

- `chunk_chars=400`
- `chunk_count=6`
- `cold_elapsed_sec=17.42`
- `cold_audio_duration_sec=74.56`
- `cold_rtf=0.23`
- `warm_elapsed_sec_avg=19.67`
- `warm_audio_duration_sec_avg=77.64`
- `warm_rtf_avg=0.25`

### `clone-fast` short sample, transcript-conditioned

Artifacts:

- `output/bench-clone-fast-6bit.json`

Measured result:

- `model_id=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit`
- `chunk_chars=400`
- `cold_elapsed_sec=37.83`
- `cold_audio_duration_sec=4.24`
- `cold_rtf=8.92`
- `warm_elapsed_sec_avg=2.52`
- `warm_audio_duration_sec_avg=4.68`
- `warm_rtf_avg=0.54`

Important trace detail:

- the cold transcript-conditioned path paid a very large first-run `reference_load_sec=29.42`

### `clone-fast` short sample, x-vector-only

Artifacts:

- `output/bench-clone-fast-6bit-xvector.json`

Measured result:

- `chunk_chars=400`
- `cold_elapsed_sec=7.02`
- `cold_audio_duration_sec=4.88`
- `cold_rtf=1.44`
- `warm_elapsed_sec_avg=1.97`
- `warm_audio_duration_sec_avg=4.52`
- `warm_rtf_avg=0.44`

Practical interpretation:

- x-vector-only clone is the practical fast clone path on this backend
- transcript-conditioned clone may still be useful for fidelity, but it is not the speed-first default

### `clone-fast` 1200-character slice, x-vector-only

Artifacts:

- `output/bench-clone-fast-1200-xvector.json`

Measured result:

- `chunk_chars=400`
- `chunk_count=6`
- `cold_elapsed_sec=36.71`
- `cold_audio_duration_sec=70.80`
- `cold_rtf=0.52`
- `warm_elapsed_sec_avg=34.62`
- `warm_audio_duration_sec_avg=74.84`
- `warm_rtf_avg=0.46`

### Same-session battery-state reruns

Artifacts:

- `output/bench-fast-6bit-battery.json`
- `output/bench-fast-1200-battery.json`
- `output/bench-fast-1200-400.json`
- `output/bench-quality-1200-450.json`
- `output/fast-notes-battery-trace.jsonl`

Important results:

- `fast`, short sample, `chunk_chars=450`
  - `warm_elapsed_sec_avg=8.74`
  - `warm_audio_duration_sec_avg=11.04`
  - `warm_rtf_avg=0.79`
- `fast`, 1200-character slice, `chunk_chars=450`
  - `warm_elapsed_sec_avg=37.88`
  - `warm_audio_duration_sec_avg=108.08`
  - `warm_rtf_avg=0.35`
- `fast`, 1200-character slice, `chunk_chars=400`
  - `warm_elapsed_sec_avg=38.26`
  - `warm_audio_duration_sec_avg=102.84`
  - `warm_rtf_avg=0.37`
- `quality`, 1200-character slice, `chunk_chars=450`
  - `warm_elapsed_sec_avg=27.15`
  - `warm_audio_duration_sec_avg=97.00`
  - `warm_rtf_avg=0.28`
- `fast`, full `notes.txt`, `chunk_chars=450`
  - `elapsed_sec=254.93`
  - `audio_duration_sec=602.64`
  - `rtf=0.42`

Interpretation:

- under the later battery/thermal conditions, `fast` was not a stable “always fastest” winner
- `fast` also did not improve meaningfully from `450` to `400` chunk chars on the warm 1200-character slice
- `quality` remained stronger than `fast` on the 1200-character slice in that later same-session comparison
- the earlier plugged-in `fast` results were still much better than these battery-state reruns, so power and thermal state are not noise here; they are first-order variables

## Practical Interpretation

- MLX is the only backend that made this project clearly usable on this Mac
- `fast` is still the simplest starting profile, but its throughput was less stable than expected under battery/thermal pressure
- `quality` is now good enough to keep as a real second preset on this Mac
- `design` is no longer disqualified on performance grounds, but it is still a deliberate mode rather than the default study path
- `clone-fast` with `6bit` is worth keeping, and x-vector-only clone is the practical quick path
- `quality` is slower than `fast`, but not so slow that it should be hidden or treated as impractical
- cold-start overhead exists, but the real cost on longer texts is still model generation time, not CLI overhead

## Trace Guidance

Use `--trace-json` on `synth` or `clone` when you want one detailed record per generation.

Use `bench` or `bench-clone` with `--output-json` when you want a reusable benchmark artifact with:

- resolved inputs
- cold trace
- warm traces
- aggregate timings

## Next Results To Add

- optional future work:
  - compare `clone-fast 8bit` against `clone-fast 6bit`
  - compare `quality 8bit` against the current `quality 6bit` default
  - rerun the same matrix on charger power after cooldown for a cleaner thermal comparison
