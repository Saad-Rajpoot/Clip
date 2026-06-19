"""Chatterbox Multilingual backend (premium default). MIT, zero-shot cloning,
all 7 target languages. Runs in the 3.11 sidecar via torch (MPS on Apple Silicon,
CUDA on Windows/Linux GPU, CPU otherwise)."""
from __future__ import annotations

from pathlib import Path

from . import sidecar
from .base import SynthResult, TTSBackend

# shipped reference clips for cloning presets (open-licensed / user-provided only)
_VOICE_REFS = Path(__file__).resolve().parent.parent / "assets" / "voice_refs"


class ChatterboxBackend(TTSBackend):
    name = "chatterbox"
    label = "Chatterbox — premium local (MIT)"
    engine = "chatterbox"

    def is_available(self) -> str:
        return sidecar.worker_available("chatterbox")

    def synth_batch(self, items, preset, *, device="auto", settings=None):
        cfg = preset.cfg_for("chatterbox")
        settings = settings or {}
        ref = cfg.get("ref")
        if ref and not Path(ref).is_absolute():
            ref = str(_VOICE_REFS / ref)
        use_ref = ref if (ref and Path(ref).exists()) else None
        lang = settings.get("language_id")
        job = {
            "device": device if device in ("mps", "cuda", "cpu") else "auto",
            "voice_ref": use_ref,
            "settings": {
                "exaggeration": float(cfg.get("exaggeration", 0.5)),
                "cfg_weight": float(cfg.get("cfg_weight", 0.5)),
                "multilingual": bool(lang) or bool(settings.get("multilingual")),
                "language_id": lang,
                # fixed seed => consistent speaker identity across all scenes
                "seed": int(settings.get("seed", cfg.get("seed", 12345))),
            },
            "items": [{"text": it.text, "out": str(it.out_wav)} for it in items],
        }
        res = sidecar.run_worker("chatterbox", job)
        out = []
        for r in res["results"]:
            sr = int(r["sr"]) or 24000
            out.append(SynthResult(Path(r["out"]), r["n"] / float(sr), sr))
        return out
