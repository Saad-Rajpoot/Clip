"""Kokoro-82M backend (fast fallback). Apache-2.0, deterministic, runs in the
3.11 sidecar via kokoro-onnx. Faster-than-real-time on Apple Silicon."""
from __future__ import annotations

from pathlib import Path

from . import sidecar
from .base import SynthResult, TTSBackend


class KokoroBackend(TTSBackend):
    name = "kokoro"
    label = "Kokoro — fast local (Apache-2.0)"
    engine = "kokoro"

    def is_available(self) -> str:
        return sidecar.worker_available("kokoro")

    def synth_batch(self, items, preset, *, device="auto", settings=None):
        cfg = preset.cfg_for("kokoro")
        settings = settings or {}
        job = {
            "device": "cpu",  # onnxruntime / CoreML; torch-MPS not used
            "voice": cfg.get("voice", "am_onyx"),
            "settings": {"speed": float(cfg.get("speed", 1.0)),
                         "lang": settings.get("lang", "en-us")},
            "models": sidecar.kokoro_models(),
            "items": [{"text": it.text, "out": str(it.out_wav)} for it in items],
        }
        res = sidecar.run_worker("kokoro", job)
        out = []
        for r in res["results"]:
            sr = int(r["sr"]) or 24000
            out.append(SynthResult(Path(r["out"]), r["n"] / float(sr), sr))
        return out
