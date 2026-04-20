from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audio import infer_audio_format, resolve_output_path, write_audio_file
from .backend import CloneRequest, SynthesisRequest, create_backend
from .bench import run_benchmark, write_benchmark_json
from .config import (
    CUSTOMVOICE_SPEAKERS,
    DEFAULT_INPUT_PATH,
    DEFAULT_LANGUAGE,
    DEFAULT_PROFILE_NAME,
    PROFILE_MAP,
    SUPPORTED_BACKENDS,
    SUPPORTED_DEVICES,
    SUPPORTED_DTYPES,
    backend_model_id,
    default_chunk_chars,
    get_profile,
    load_presets,
    mlx_model_id,
    resolve_preset_path,
    validate_clone_options,
    validate_synth_options,
)
from .text import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Qwen3-TTS CLI.")
    parser.add_argument("--config", help="Path to presets YAML file.")
    parser.add_argument("--preset", help="Preset name from the YAML config.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="List available profiles and model ids.")
    models.add_argument("--verbose", action="store_true", help="Show extra profile details.")

    voices = subparsers.add_parser("voices", help="List voices for a CustomVoice profile.")
    _add_shared_generation_args(voices)

    preload = subparsers.add_parser("preload", help="Download model weights ahead of time.")
    preload.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        choices=PROFILE_MAP.keys(),
        help="Profile to preload.",
    )
    preload.add_argument("--all", action="store_true", help="Preload all profiles.")
    preload.add_argument(
        "--device",
        default="auto",
        choices=SUPPORTED_DEVICES,
        help="Device to use if the model is loaded later.",
    )
    preload.add_argument(
        "--dtype",
        default=None,
        choices=SUPPORTED_DTYPES,
        help="Computation dtype. Defaults to auto.",
    )
    preload.add_argument(
        "--backend",
        default=None,
        choices=SUPPORTED_BACKENDS,
        help="Backend runtime to use.",
    )
    preload.add_argument("--mlx-python", type=Path, help="Python executable for the MLX worker.")

    synth = subparsers.add_parser("synth", help="Generate speech from text.")
    _add_shared_generation_args(synth)
    _add_input_args(synth)
    synth.add_argument("--output", type=Path, help="Output audio path.")
    synth.add_argument(
        "--format",
        choices=("wav", "mp3", "m4a", "aac", "flac", "opus"),
        help="Output audio format. Defaults to the output file suffix or wav.",
    )
    synth.add_argument("--trace-json", type=Path, help="Append a JSON trace record for this run.")
    synth.add_argument("--show-settings", action="store_true", help="Print resolved settings.")

    clone = subparsers.add_parser("clone", help="Clone a voice from reference audio.")
    _add_shared_generation_args(clone)
    _add_input_args(clone)
    clone.add_argument("--reference", type=Path, required=True, help="Reference audio file.")
    clone.add_argument("--ref-text", help="Transcript of the reference audio.")
    clone.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="Use speaker embedding only when no transcript is available.",
    )
    clone.add_argument("--output", type=Path, help="Output audio path.")
    clone.add_argument(
        "--format",
        choices=("wav", "mp3", "m4a", "aac", "flac", "opus"),
        help="Output audio format. Defaults to the output file suffix or wav.",
    )
    clone.add_argument("--trace-json", type=Path, help="Append a JSON trace record for this run.")
    clone.add_argument("--show-settings", action="store_true", help="Print resolved settings.")

    bench = subparsers.add_parser("bench", help="Benchmark cold and warm synthesis times.")
    _add_shared_generation_args(bench)
    _add_input_args(bench)
    bench.add_argument("--runs", type=int, default=2, help="Warm benchmark runs.")
    bench.add_argument("--seed", type=int, help="Optional torch seed for reproducible benchmark runs.")
    bench.add_argument("--output-json", type=Path, help="Optional JSON output path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        presets = load_presets(resolve_preset_path(args.config))
        preset = presets.get(args.preset, {}) if getattr(args, "preset", None) else {}
        _apply_preset_defaults(args, preset)
        _finalize_common_defaults(args)

        if args.command == "models":
            return _cmd_models(args)
        if args.command == "voices":
            return _cmd_voices(args)
        if args.command == "preload":
            return _cmd_preload(args)
        if args.command == "synth":
            return _cmd_synth(args)
        if args.command == "clone":
            return _cmd_clone(args)
        if args.command == "bench":
            return _cmd_bench(args)
        parser.error(f"Unknown command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print("Canceled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _cmd_models(args: argparse.Namespace) -> int:
    for profile in PROFILE_MAP.values():
        print(f"{profile.name}: {profile.model_id}")
        if args.verbose:
            default_voice = profile.default_voice or "n/a"
            print(f"  type: {profile.model_type}")
            print(f"  default voice: {default_voice}")
            print(f"  description: {profile.description}")
            print(f"  chunk chars (mps): {profile.chunk_chars_mps}")
            print(f"  chunk chars (other): {profile.chunk_chars_other}")
            print(f"  pytorch model: {profile.model_id}")
            print(f"  mlx model: {mlx_model_id(profile)}")
    return 0


def _cmd_voices(args: argparse.Namespace) -> int:
    profile = get_profile(args.profile)
    if profile.model_type != "CustomVoice":
        raise ValueError(f"Profile '{profile.name}' does not support preset voices.")

    backend = create_backend(
        args.backend,
        device=args.device,
        dtype=args.dtype,
        mlx_python=str(args.mlx_python) if args.mlx_python else None,
    )
    try:
        backend.ensure_loaded(profile)
        voices = backend.list_voices(profile)
        for voice in voices:
            description = CUSTOMVOICE_SPEAKERS.get(voice, "")
            if description:
                print(f"{voice}: {description}")
            else:
                print(voice)
        return 0
    finally:
        backend.close()


def _cmd_preload(args: argparse.Namespace) -> int:
    backend = create_backend(
        args.backend,
        device=args.device,
        dtype=args.dtype,
        mlx_python=str(args.mlx_python) if args.mlx_python else None,
    )
    profiles = list(PROFILE_MAP.values()) if args.all else [get_profile(args.profile)]

    try:
        for profile in profiles:
            print(f"Preloading {profile.name}: {backend_model_id(profile, backend.name)}")
            paths = backend.preload(profile)
            for path in paths:
                print(f"  cached: {path}")
        return 0
    finally:
        backend.close()


def _cmd_synth(args: argparse.Namespace) -> int:
    total_start = time.perf_counter()
    profile = get_profile(args.profile)
    validate_synth_options(profile, args.voice, args.instruct)
    text = _load_input_text(args)
    backend = create_backend(
        args.backend,
        device=args.device,
        dtype=args.dtype,
        mlx_python=str(args.mlx_python) if args.mlx_python else None,
    )
    audio_format = infer_audio_format(args.output, args.format)
    output_path = resolve_output_path(args.output, audio_format)
    chunk_chars = args.chunk_chars or default_chunk_chars(profile, backend.device)

    try:
        if args.show_settings:
            _print_settings(
                {
                    "command": "synth",
                    "profile": profile.name,
                    "backend": backend.name,
                    "model_id": backend_model_id(profile, backend.name),
                    "device": backend.device,
                    "dtype": backend.dtype_name,
                    "voice": args.voice or profile.default_voice or "n/a",
                    "language": args.language or DEFAULT_LANGUAGE,
                    "chunk_chars": chunk_chars,
                    "output": str(output_path),
                    "format": audio_format,
                }
            )

        result = backend.synthesize(
            SynthesisRequest(
                text=text,
                profile=profile,
                language=args.language or DEFAULT_LANGUAGE,
                voice=args.voice,
                instruct=args.instruct,
                chunk_chars=chunk_chars,
            ),
            on_progress=_progress_printer,
        )
        write_start = time.perf_counter()
        write_audio_file(output_path, result.audio, result.sample_rate, audio_format)
        write_elapsed = time.perf_counter() - write_start

        duration = len(result.audio) / result.sample_rate
        if args.trace_json:
            _write_trace_json(
                args.trace_json,
                {
                    "command": "synth",
                    "backend": backend.name,
                    "profile": profile.name,
                    "model_id": result.model_id,
                    "device": backend.device,
                    "dtype": result.dtype,
                    "chars": len(text),
                    "chunk_chars": chunk_chars,
                    "chunk_count": len(result.chunks),
                    "audio_duration_sec": duration,
                    "elapsed_sec": result.elapsed_sec,
                    "rtf": _safe_rtf(result.elapsed_sec, duration),
                    "phases": {
                        "backend_call_sec": result.elapsed_sec,
                        "audio_write_sec": write_elapsed,
                        "total_cli_sec": time.perf_counter() - total_start,
                    },
                    "output": str(output_path),
                    "format": audio_format,
                },
            )
        print(
            f"Wrote {output_path} in {result.elapsed_sec:.2f}s "
            f"({duration:.2f}s audio, {len(result.chunks)} chunks, profile={profile.name}, backend={backend.name}, device={backend.device}, dtype={result.dtype})."
        )
        return 0
    finally:
        backend.close()


def _cmd_clone(args: argparse.Namespace) -> int:
    total_start = time.perf_counter()
    profile = get_profile(args.profile)
    validate_clone_options(profile, args.reference, args.ref_text, args.x_vector_only_mode)
    text = _load_input_text(args)
    backend = create_backend(
        args.backend,
        device=args.device,
        dtype=args.dtype,
        mlx_python=str(args.mlx_python) if args.mlx_python else None,
    )
    audio_format = infer_audio_format(args.output, args.format)
    output_path = resolve_output_path(args.output, audio_format)
    chunk_chars = args.chunk_chars or default_chunk_chars(profile, backend.device)

    try:
        if args.show_settings:
            _print_settings(
                {
                    "command": "clone",
                    "profile": profile.name,
                    "backend": backend.name,
                    "model_id": backend_model_id(profile, backend.name),
                    "device": backend.device,
                    "dtype": backend.dtype_name,
                    "reference": str(args.reference),
                    "ref_text": (args.ref_text or "").strip() or "n/a",
                    "x_vector_only_mode": args.x_vector_only_mode,
                    "language": args.language or DEFAULT_LANGUAGE,
                    "chunk_chars": chunk_chars,
                    "output": str(output_path),
                    "format": audio_format,
                }
            )

        result = backend.clone(
            CloneRequest(
                text=text,
                profile=profile,
                reference_audio=args.reference,
                reference_text=args.ref_text,
                x_vector_only_mode=args.x_vector_only_mode,
                language=args.language or DEFAULT_LANGUAGE,
                chunk_chars=chunk_chars,
            ),
            on_progress=_progress_printer,
        )
        write_start = time.perf_counter()
        write_audio_file(output_path, result.audio, result.sample_rate, audio_format)
        write_elapsed = time.perf_counter() - write_start

        duration = len(result.audio) / result.sample_rate
        if args.trace_json:
            _write_trace_json(
                args.trace_json,
                {
                    "command": "clone",
                    "backend": backend.name,
                    "profile": profile.name,
                    "model_id": result.model_id,
                    "device": backend.device,
                    "dtype": result.dtype,
                    "chars": len(text),
                    "chunk_chars": chunk_chars,
                    "chunk_count": len(result.chunks),
                    "audio_duration_sec": duration,
                    "elapsed_sec": result.elapsed_sec,
                    "rtf": _safe_rtf(result.elapsed_sec, duration),
                    "phases": {
                        "backend_call_sec": result.elapsed_sec,
                        "audio_write_sec": write_elapsed,
                        "total_cli_sec": time.perf_counter() - total_start,
                    },
                    "reference": str(args.reference),
                    "output": str(output_path),
                    "format": audio_format,
                },
            )
        print(
            f"Wrote {output_path} in {result.elapsed_sec:.2f}s "
            f"({duration:.2f}s audio, {len(result.chunks)} chunks, profile={profile.name}, backend={backend.name}, device={backend.device}, dtype={result.dtype})."
        )
        return 0
    finally:
        backend.close()


def _cmd_bench(args: argparse.Namespace) -> int:
    profile = get_profile(args.profile)
    validate_synth_options(profile, args.voice, args.instruct)
    text = _load_input_text(args)
    result = run_benchmark(
        profile=profile,
        text=text,
        voice=args.voice,
        language=args.language or DEFAULT_LANGUAGE,
        instruct=args.instruct,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        mlx_python=str(args.mlx_python) if args.mlx_python else None,
        chunk_chars=args.chunk_chars,
        warm_runs=args.runs,
        seed=args.seed,
    )

    print(f"profile={result.profile}")
    print(f"model_id={result.model_id}")
    print(f"backend={result.backend}")
    print(f"device={result.device}")
    print(f"dtype={result.dtype}")
    print(f"seed={result.seed if result.seed is not None else 'n/a'}")
    print(f"chars={result.chars}")
    print(f"chunk_chars={result.chunk_chars}")
    print(f"chunk_count={result.chunk_count}")
    print(f"cold_audio_duration_sec={result.cold_audio_duration_sec:.2f}")
    print(f"warm_audio_duration_sec_avg={result.warm_audio_duration_sec_avg:.2f}")
    print(
        "warm_audio_duration_sec_runs="
        + ", ".join(f"{item:.2f}" for item in result.warm_audio_duration_sec_runs)
    )
    print(f"cold_elapsed_sec={result.cold_elapsed_sec:.2f}")
    print(f"cold_rtf={result.cold_rtf:.2f}")
    print(f"warm_elapsed_sec_avg={result.warm_elapsed_sec_avg:.2f}")
    print(f"warm_rtf_avg={result.warm_rtf_avg:.2f}")
    print("warm_elapsed_sec_runs=" + ", ".join(f"{item:.2f}" for item in result.warm_elapsed_sec_runs))

    if args.output_json:
        write_benchmark_json(args.output_json, result)
        print(f"json={args.output_json}")
    return 0


def _add_shared_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=None,
        choices=PROFILE_MAP.keys(),
        help="Model profile to use.",
    )
    parser.add_argument("--voice", help="Preset speaker for CustomVoice models.")
    parser.add_argument("--instruct", help="Style or design instruction.")
    parser.add_argument(
        "--language",
        help="Language code or name. Defaults to auto.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=SUPPORTED_BACKENDS,
        help="Backend runtime to use.",
    )
    parser.add_argument("--mlx-python", type=Path, help="Python executable for the MLX worker.")
    parser.add_argument(
        "--device",
        default=None,
        choices=SUPPORTED_DEVICES,
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=SUPPORTED_DTYPES,
        help="Computation dtype. Defaults to auto.",
    )
    parser.add_argument("--chunk-chars", type=int, help="Chunk size for long text.")


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input text file.")
    parser.add_argument("--text", help="Inline text.")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin.")


def _load_input_text(args: argparse.Namespace) -> str:
    sources = sum(bool(source) for source in (args.text is not None, args.stdin))
    if sources > 1:
        raise ValueError("Use only one of --text or --stdin. --input is the file fallback.")

    if args.text is not None:
        return _require_text(normalize_text(args.text))

    if args.stdin:
        return _require_text(normalize_text(sys.stdin.read()))

    return _require_text(normalize_text(args.input.read_text(encoding="utf-8")))


def _require_text(text: str) -> str:
    if not text:
        raise ValueError("Input text is empty.")
    return text


def _apply_preset_defaults(args: argparse.Namespace, preset: dict[str, object]) -> None:
    if not preset:
        return

    fields = (
        "profile",
        "voice",
        "instruct",
        "language",
        "device",
        "dtype",
        "chunk_chars",
        "backend",
        "mlx_python",
        "format",
        "output",
    )
    for field in fields:
        current = getattr(args, field, None)
        if current in (None,):
            value = preset.get(field)
            if value is not None:
                setattr(args, field, Path(value) if field in {"output", "mlx_python"} else value)


def _finalize_common_defaults(args: argparse.Namespace) -> None:
    if hasattr(args, "profile") and args.profile is None:
        args.profile = DEFAULT_PROFILE_NAME
    if hasattr(args, "device") and args.device is None:
        args.device = "auto"
    if hasattr(args, "dtype") and args.dtype is None:
        args.dtype = "auto"
    if hasattr(args, "backend") and args.backend is None:
        args.backend = "auto"


def _progress_printer(index: int, total: int, _chunk: str) -> None:
    print(f"[chunk {index + 1}/{total}]", file=sys.stderr)


def _print_settings(values: dict[str, object]) -> None:
    print("Resolved settings:", file=sys.stderr)
    for key, value in values.items():
        print(f"  {key}: {value}", file=sys.stderr)


def _write_trace_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _safe_rtf(elapsed_sec: float, duration_sec: float) -> float | None:
    if duration_sec <= 0:
        return None
    return elapsed_sec / duration_sec


if __name__ == "__main__":
    raise SystemExit(main())
