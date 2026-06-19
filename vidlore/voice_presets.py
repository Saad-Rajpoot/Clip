"""Vidlore premium documentary narrator presets.

5 cinematic narrator voices, each mapped to per-engine config so the SAME preset
works on the premium backend (Chatterbox) and the fast fallback (Kokoro) and the
legacy backend (edge-tts). No celebrity voices; Chatterbox `ref` clips are either
None (built-in voice) or open-licensed/own recordings shipped under
``assets/voice_refs/``.

Chatterbox tuning: exaggeration LOWER = graver/calmer (documentary); cfg_weight
controls pacing/adherence. Kokoro: voice id + speed (≤1.0 = slower/graver).
"""
from __future__ import annotations

from .tts_backends.base import VoicePreset

PRESETS = {
    "deep_male_documentary": VoicePreset(
        key="deep_male_documentary",
        label="Deep Male Documentary",
        description="Serious, cinematic, investigative — spy / crime / history.",
        engine_cfg={
            "chatterbox": {"exaggeration": 0.30, "cfg_weight": 0.50, "ref": None},
            "kokoro": {"voice": "am_onyx", "speed": 0.90},
            "edge": {"voice": "en-US-GuyNeural", "rate": "+0%"},
        }),
    "warm_male_explainer": VoicePreset(
        key="warm_male_explainer",
        label="Warm Male Explainer",
        description="Clear, calm, educational — Vox-style / explainer docs.",
        engine_cfg={
            "chatterbox": {"exaggeration": 0.40, "cfg_weight": 0.55, "ref": None},
            "kokoro": {"voice": "am_fenrir", "speed": 0.95},
            "edge": {"voice": "en-US-AndrewNeural", "rate": "+2%"},
        }),
    "mature_female_documentary": VoicePreset(
        key="mature_female_documentary",
        label="Mature Female Documentary",
        description="Serious, premium, news / documentary tone.",
        engine_cfg={
            "chatterbox": {"exaggeration": 0.35, "cfg_weight": 0.50, "ref": None},
            "kokoro": {"voice": "bf_emma", "speed": 0.92},
            "edge": {"voice": "en-US-AriaNeural", "rate": "+0%"},
        }),
    "calm_female_storyteller": VoicePreset(
        key="calm_female_storyteller",
        label="Calm Female Storyteller",
        description="Emotional, soft — historical / health / biography.",
        engine_cfg={
            "chatterbox": {"exaggeration": 0.45, "cfg_weight": 0.55, "ref": None},
            "kokoro": {"voice": "af_heart", "speed": 0.95},
            "edge": {"voice": "en-US-JennyNeural", "rate": "+2%"},
        }),
    "dark_investigative_narrator": VoicePreset(
        key="dark_investigative_narrator",
        label="Dark Investigative Narrator",
        description="Tense, slow — true crime / espionage.",
        engine_cfg={
            "chatterbox": {"exaggeration": 0.30, "cfg_weight": 0.40, "ref": None},
            "kokoro": {"voice": "am_onyx", "speed": 0.84},
            "edge": {"voice": "en-US-GuyNeural", "rate": "-4%"},
        }),
}

DEFAULT_PRESET = "deep_male_documentary"


def get_preset(key):
    return PRESETS.get(key or DEFAULT_PRESET) or PRESETS[DEFAULT_PRESET]


def preset_list():
    """For the dashboard dropdown."""
    return [{"key": p.key, "label": p.label, "description": p.description}
            for p in PRESETS.values()]
