from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = Path("input.txt")
DEFAULT_OUTPUT_STEM = "output"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "presets.yaml"
DEFAULT_PROFILE_NAME = "fast"
TOKENIZER_MODEL_ID = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_LANGUAGE = "auto"

SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "m4a", "aac", "flac", "opus")
SUPPORTED_DEVICES = ("auto", "mps", "cuda", "cpu")
SUPPORTED_DTYPES = ("auto", "bfloat16", "float32", "float16")

LANGUAGE_NAME_MAP = {
    "auto": "Auto",
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}

CUSTOMVOICE_SPEAKERS = {
    "Vivian": "Bright, slightly edgy young female voice.",
    "Serena": "Warm, gentle young female voice.",
    "Uncle_Fu": "Seasoned male voice with a low, mellow timbre.",
    "Dylan": "Youthful Beijing male voice with a clear, natural timbre.",
    "Eric": "Lively Chengdu male voice with a slightly husky brightness.",
    "Ryan": "Dynamic male voice with strong rhythmic drive.",
    "Aiden": "Sunny American male voice with a clear midrange.",
    "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre.",
    "Sohee": "Warm Korean female voice with rich emotion.",
}


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    name: str
    model_id: str
    model_type: str
    description: str
    default_voice: str | None
    chunk_chars_mps: int
    chunk_chars_other: int

    @property
    def supports_voice(self) -> bool:
        return self.model_type == "CustomVoice"

    @property
    def supports_instruct(self) -> bool:
        return self.model_type in {"CustomVoice", "VoiceDesign"}

    @property
    def supports_clone(self) -> bool:
        return self.model_type == "Base"


PROFILE_MAP: dict[str, ProfileSpec] = {
    "fast": ProfileSpec(
        name="fast",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        model_type="CustomVoice",
        description="Fastest daily-use preset on Apple Silicon.",
        default_voice="Ryan",
        chunk_chars_mps=120,
        chunk_chars_other=450,
    ),
    "quality": ProfileSpec(
        name="quality",
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_type="CustomVoice",
        description="Higher-quality preset speaker generation.",
        default_voice="Ryan",
        chunk_chars_mps=140,
        chunk_chars_other=400,
    ),
    "design": ProfileSpec(
        name="design",
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        model_type="VoiceDesign",
        description="Free-form voice design from natural-language instructions.",
        default_voice=None,
        chunk_chars_mps=140,
        chunk_chars_other=400,
    ),
    "clone-fast": ProfileSpec(
        name="clone-fast",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        model_type="Base",
        description="Smaller cloning profile for offline use.",
        default_voice=None,
        chunk_chars_mps=140,
        chunk_chars_other=400,
    ),
    "clone-quality": ProfileSpec(
        name="clone-quality",
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        model_type="Base",
        description="Higher-quality cloning profile.",
        default_voice=None,
        chunk_chars_mps=120,
        chunk_chars_other=360,
    ),
}


def get_profile(name: str) -> ProfileSpec:
    try:
        return PROFILE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile: {name}") from exc


def normalize_language(language: str | None) -> str:
    value = (language or DEFAULT_LANGUAGE).strip().lower()
    if value not in LANGUAGE_NAME_MAP:
        supported = ", ".join(sorted(set(LANGUAGE_NAME_MAP)))
        raise ValueError(f"Unsupported language '{language}'. Supported values: {supported}")
    return LANGUAGE_NAME_MAP[value]


def validate_synth_options(profile: ProfileSpec, voice: str | None, instruct: str | None) -> None:
    if profile.supports_clone:
        raise ValueError(
            f"Profile '{profile.name}' is a clone profile. Use the clone command instead of synth."
        )

    if profile.model_type == "VoiceDesign" and not (instruct or "").strip():
        raise ValueError("VoiceDesign requires --instruct.")

    if profile.model_type == "VoiceDesign" and voice:
        raise ValueError("VoiceDesign does not support --voice.")

    if profile.supports_voice and voice:
        supported = {item.lower() for item in CUSTOMVOICE_SPEAKERS}
        if voice.lower() in supported:
            return
        known = ", ".join(CUSTOMVOICE_SPEAKERS)
        raise ValueError(f"Unknown voice '{voice}'. Known voices: {known}")


def validate_clone_options(
    profile: ProfileSpec,
    reference: Path | None,
    ref_text: str | None,
    x_vector_only_mode: bool,
) -> None:
    if not profile.supports_clone:
        raise ValueError(
            f"Profile '{profile.name}' is not a clone profile. Use clone-fast or clone-quality."
        )

    if reference is None:
        raise ValueError("Voice cloning requires --reference.")

    if not x_vector_only_mode and not (ref_text or "").strip():
        raise ValueError(
            "Voice cloning requires --ref-text unless --x-vector-only-mode is enabled."
        )


def default_chunk_chars(profile: ProfileSpec, device: str) -> int:
    if device == "mps":
        return profile.chunk_chars_mps
    return profile.chunk_chars_other


def load_presets(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid presets file: {path}")
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in presets.items():
        if not isinstance(value, dict):
            raise ValueError(f"Preset '{name}' must be a mapping.")
        normalized[name] = dict(value)
    return normalized


def resolve_preset_path(path_arg: str | None) -> Path | None:
    if path_arg:
        return Path(path_arg)
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None
