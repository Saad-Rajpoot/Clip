"""One-command setup for Vidlore's premium LOCAL voice sidecar.

Creates a real Python-3.11 TTS venv under ~/.vidlore (NOT /tmp) + downloads the
Kokoro model, so the premium/fallback voices work on a fresh machine with no
paid APIs and no admin rights. Uses `uv` (auto-installed if missing) so it works
identically on macOS (Apple Silicon/Intel) and Windows.

Usage:
    python -m vidlore.tts_setup            # Kokoro fallback (light, ~0.4 GB)
    python -m vidlore.tts_setup --full     # + Chatterbox premium (torch, ~3 GB)
    python -m vidlore.tts_setup --status   # just print readiness

Then set (printed at the end, also written to ~/.vidlore/tts.env):
    VIDLORE_TTS_PYTHON, VIDLORE_KOKORO_ONNX, VIDLORE_KOKORO_VOICES
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
VF = HOME / ".vidlore"
VENV = VF / "tts-venv"
MODELS = VF / "models" / "kokoro"
IS_WIN = os.name == "nt"

_KOKORO_BASE = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                "model-files-v1.0/")
_KOKORO_FILES = {"kokoro-v1.0.onnx": _KOKORO_BASE + "kokoro-v1.0.onnx",
                 "voices-v1.0.bin": _KOKORO_BASE + "voices-v1.0.bin"}


def venv_python(venv: Path = VENV) -> Path:
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _find_uv() -> str:
    for c in (shutil.which("uv"), str(HOME / ".local" / "bin" / "uv"),
              str(HOME / ".cargo" / "bin" / "uv")):
        if c and Path(c).exists():
            return c
    # bootstrap uv via pip into the current interpreter
    print("[setup] installing uv ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                   check=True)
    return shutil.which("uv") or str(HOME / ".local" / "bin" / "uv")


def _run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def download_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, url in _KOKORO_FILES.items():
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[setup] {name} already present")
            continue
        print(f"[setup] downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"        -> {dest} ({dest.stat().st_size // (1024*1024)} MB)")


def write_env() -> Path:
    py = venv_python()
    env = VF / "tts.env"
    lines = [
        f"VIDLORE_TTS_PYTHON={py}",
        f"VIDLORE_KOKORO_ONNX={MODELS / 'kokoro-v1.0.onnx'}",
        f"VIDLORE_KOKORO_VOICES={MODELS / 'voices-v1.0.bin'}",
    ]
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env


def status() -> dict:
    py = venv_python()
    out = {"venv_python": str(py), "venv_ok": py.exists(),
           "kokoro_model": (MODELS / "kokoro-v1.0.onnx").exists()}
    if py.exists():
        for mod in ("kokoro_onnx", "chatterbox", "torch", "soundfile"):
            r = subprocess.run([str(py), "-c", f"import {mod}"],
                               capture_output=True)
            out[mod] = (r.returncode == 0)
    return out


def setup(full: bool) -> None:
    VF.mkdir(parents=True, exist_ok=True)
    uv = _find_uv()
    print(f"[setup] uv: {uv}")
    _run([uv, "python", "install", "3.11"])
    if not venv_python().exists():
        _run([uv, "venv", str(VENV), "--python", "3.11"])
    py = venv_python()
    # Kokoro fallback — light, always installed. setuptools<81 keeps
    # pkg_resources (needed by some deps); soundfile for wav IO.
    print("[setup] installing Kokoro (fast fallback) ...")
    _run([uv, "pip", "install", "--python", str(py),
          "kokoro-onnx", "soundfile", "setuptools<81"])
    if full:
        print("[setup] installing Chatterbox (premium, downloads torch ~2.5GB) ...")
        _run([uv, "pip", "install", "--python", str(py),
              "chatterbox-tts", "soundfile", "setuptools<81"])
    download_models()
    env = write_env()
    print("\n[setup] DONE. Add these to your environment (also in "
          f"{env}):")
    print(env.read_text().rstrip())
    print("\nThen enable premium voice with:")
    print("  VIDLORE_TTS_BACKEND=premium_local "
          "VIDLORE_TTS_MODEL=" + ("chatterbox" if full else "kokoro"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also install Chatterbox premium (torch, ~3 GB)")
    ap.add_argument("--status", action="store_true", help="print readiness only")
    a = ap.parse_args(argv)
    if a.status:
        import json
        print(json.dumps(status(), indent=2))
        return 0
    try:
        setup(full=a.full or os.environ.get("VIDLORE_TTS_FULL") == "1")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[setup] FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
