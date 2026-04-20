# Performance Results

These results were captured locally while comparing the original PyTorch MPS path against the new Apple-native MLX path.

The machine power state changed during testing, so treat the exact numbers as approximate. The important signal is the size of the backend gap, not tiny differences between runs.

## Summary

- The MLX `fast` profile is the first backend that made this project practically usable on this Mac.
- The PyTorch `fast` profile remained too slow for long study files.
- Larger chunk sizes were not universally better on MLX; the `fast` default of `450` chars remained the safest practical setting in these tests.

## Commands

Short benchmark, MLX:

```bash
python3 -m tts_qwen_local.cli bench \
  --profile fast \
  --backend mlx \
  --text "Hello. This is a short MLX benchmark sample with two sentences. The goal is to compare backend latency and not just audio output." \
  --runs 2
```

Short benchmark, PyTorch:

```bash
.venv/bin/python -m tts_qwen_local.cli bench \
  --profile fast \
  --backend pytorch \
  --text "Hello. This is a short MLX benchmark sample with two sentences. The goal is to compare backend latency and not just audio output." \
  --runs 2
```

Long real-text run:

```bash
python3 -m tts_qwen_local.cli synth \
  --profile fast \
  --backend mlx \
  --input notes.txt \
  --output output/mlx-notes.wav
```

## Measured Results

Short sample, `fast`, MLX:

- `cold_elapsed_sec=7.03`
- `warm_elapsed_sec_avg=3.17`
- `cold_audio_duration_sec=15.04`
- `warm_rtf_avg=0.27`

Short sample, `fast`, PyTorch:

- `cold_elapsed_sec=81.68`
- `warm_elapsed_sec_avg=21.55`
- `cold_audio_duration_sec=26.72`
- `warm_rtf_avg=1.86`

Longer real-text slice, `1200` chars, MLX, `chunk_chars=450`:

- `cold_elapsed_sec=26.05`
- `warm_elapsed_sec_avg=22.31`
- `warm_audio_duration_sec_avg=110.96`
- `warm_rtf_avg=0.20`

Full `notes.txt`, MLX `fast`, default chunking:

- `elapsed_sec=59.82`
- `audio_duration_sec=455.76`
- `chunk_count=21`

Full `notes.txt`, MLX `fast`, `chunk_chars=900`:

- significantly worse than the default setting
- the run was still active after about `97` seconds and was stopped

## Practical Conclusion

- Default Apple path should be `auto -> mlx` when an MLX Python is available.
- Daily-use recommendation stays:
  - `fast` for routine study audio
  - larger profiles only when you explicitly want to trade speed for quality
- If more profiling is needed later, use `--trace-json` on `synth` or `clone` to keep per-run timing records.
