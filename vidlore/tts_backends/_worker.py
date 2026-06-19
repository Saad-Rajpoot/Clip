#!/usr/bin/env python3
"""Vidlore TTS sidecar WORKER — runs INSIDE the Python 3.11 TTS venv.

The Vidlore app runs on Python 3.9, but the modern local-TTS stack (kokoro-onnx,
chatterbox, torch) needs Python 3.10+. So premium/fallback voices are synthesized
out-of-process: ``sidecar.py`` (in the 3.9 app) shells out to THIS script with the
3.11 interpreter, passing a JSON job; the worker loads the engine ONCE, synthesizes
every item (amortizing model load), writes wavs, and prints a JSON result.

NEVER imported by the 3.9 app — only executed by the TTS venv's python. No paid
APIs, no network calls beyond the engine's own (cached) model files.

Job JSON schema (argv[1] = path to job file):
  {
    "engine": "kokoro" | "chatterbox",
    "device": "cpu" | "mps" | "cuda",
    "voice":  "<engine voice id>",          # kokoro
    "voice_ref": "/path/ref.wav" | null,     # chatterbox cloning reference
    "settings": { "speed":.., "lang":.., "exaggeration":.., "cfg_weight":..,
                  "multilingual": bool, "language_id": "en" },
    "models": { "onnx": "...", "voices": "..." },   # kokoro file paths
    "items": [ { "text": "...", "out": "/abs/out.wav" }, ... ]
  }
Result JSON (stdout, last line): {"ok": true, "results":[{"out","sr","n"}...]}
"""
import json
import sys
import traceback
from pathlib import Path


def _kokoro_engine(models):
    from kokoro_onnx import Kokoro
    return Kokoro(models["onnx"], models["voices"])


def _kokoro_one(eng, text, voice, settings):
    samples, sr = eng.create(
        text, voice=voice,
        speed=float(settings.get("speed", 1.0)),
        lang=settings.get("lang", "en-us"))
    return samples, sr


def _chatterbox_engine(device, multilingual):
    if multilingual:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as C
    else:
        from chatterbox.tts import ChatterboxTTS as C
    return C.from_pretrained(device=device)


def _chatterbox_one(eng, text, voice_ref, settings):
    # Seed-lock for STABLE speaker identity across a long multi-scene script
    # (Chatterbox is stochastic; a fixed seed per preset keeps the voice
    # consistent scene-to-scene — the documented long-form fix).
    seed = settings.get("seed")
    if seed is not None:
        try:
            import torch
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:                                      # noqa: BLE001
            pass
    kw = {"exaggeration": float(settings.get("exaggeration", 0.5)),
          "cfg_weight": float(settings.get("cfg_weight", 0.5))}
    if voice_ref:
        kw["audio_prompt_path"] = voice_ref
    if settings.get("language_id"):
        kw["language_id"] = settings["language_id"]
    wav = eng.generate(text, **kw)
    a = wav.squeeze(0).detach().cpu().numpy().astype("float32")
    return a, eng.sr


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    engine = job["engine"]
    settings = job.get("settings", {})
    items = job["items"]
    import soundfile as sf

    device = job.get("device", "cpu")
    if device == "auto":
        try:
            import torch
            device = ("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:                                      # noqa: BLE001
            device = "cpu"

    if engine == "kokoro":
        eng = _kokoro_engine(job["models"])
        synth = lambda t: _kokoro_one(eng, t, job.get("voice"), settings)  # noqa: E731
    elif engine == "chatterbox":
        eng = _chatterbox_engine(device, bool(settings.get("multilingual")))
        ref = job.get("voice_ref")
        synth = lambda t: _chatterbox_one(eng, t, ref, settings)  # noqa: E731
    else:
        print(json.dumps({"ok": False, "error": f"unknown engine {engine}"}))
        return 2

    results = []
    for it in items:
        a, sr = synth(it["text"])
        out = it["out"]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, a, sr)
        results.append({"out": out, "sr": int(sr), "n": int(len(a))})
    print(json.dumps({"ok": True, "engine": engine, "results": results}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                                     # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e),
                          "trace": traceback.format_exc()[-1500:]}))
        sys.exit(1)
