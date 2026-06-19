"""Bridge from the Python 3.9 app to the Python 3.11 TTS venv.

Resolves the sidecar interpreter + model files (env-overridable, with dev
fallbacks), then runs ``_worker.py`` out-of-process with a JSON job and parses
its JSON result. Pure stdlib — safe to import on 3.9.

Env (production sets these to a bundled venv under ~/.vidlore):
  VIDLORE_TTS_PYTHON            shared 3.11 venv python for all engines
  VIDLORE_TTS_PYTHON_KOKORO     per-engine override
  VIDLORE_TTS_PYTHON_CHATTERBOX per-engine override
  VIDLORE_KOKORO_ONNX           path to kokoro-v1.0.onnx
  VIDLORE_KOKORO_VOICES         path to voices-v1.0.bin
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

_WORKER = Path(__file__).with_name("_worker.py")
_HOME = Path(os.path.expanduser("~"))

# dev fallbacks (the venvs/models created during benchmarking on this machine)
_DEV_VENV = {"kokoro": "/tmp/tts311/bin/python",
             "chatterbox": "/tmp/ttscb/bin/python"}
_DEV_KOKORO = {"onnx": "/tmp/kokoro_models/kokoro-v1.0.onnx",
               "voices": "/tmp/kokoro_models/voices-v1.0.bin"}


def tts_venv_python(engine: str) -> str:
    """Best interpreter for `engine`, or "" if none is usable. Handles both the
    POSIX (bin/python) and Windows (Scripts/python.exe) venv layouts."""
    _vf = _HOME / ".vidlore" / "tts-venv"
    cands = [
        os.environ.get(f"VIDLORE_TTS_PYTHON_{engine.upper()}", "").strip(),
        os.environ.get("VIDLORE_TTS_PYTHON", "").strip(),
        str(_vf / "bin" / "python"),
        str(_vf / "Scripts" / "python.exe"),
        _DEV_VENV.get(engine, ""),
    ]
    for c in cands:
        if c and Path(c).exists():
            return c
    return ""


def kokoro_models() -> dict:
    onnx = (os.environ.get("VIDLORE_KOKORO_ONNX", "").strip()
            or str(_HOME / ".vidlore" / "models" / "kokoro" / "kokoro-v1.0.onnx"))
    voices = (os.environ.get("VIDLORE_KOKORO_VOICES", "").strip()
              or str(_HOME / ".vidlore" / "models" / "kokoro" / "voices-v1.0.bin"))
    if not Path(onnx).exists() and Path(_DEV_KOKORO["onnx"]).exists():
        onnx, voices = _DEV_KOKORO["onnx"], _DEV_KOKORO["voices"]
    return {"onnx": onnx, "voices": voices}


def worker_available(engine: str) -> str:
    """"" if the engine can run, else a human reason."""
    py = tts_venv_python(engine)
    if not py:
        return (f"no TTS sidecar venv for {engine} (set VIDLORE_TTS_PYTHON "
                "to a Python 3.11 venv with the engine installed)")
    if engine == "kokoro":
        m = kokoro_models()
        if not Path(m["onnx"]).exists():
            return f"kokoro model file missing: {m['onnx']}"
    if not _WORKER.exists():
        return f"worker script missing: {_WORKER}"
    return ""


def run_worker(engine: str, job: dict, *, timeout: int = 1800) -> dict:
    """Run the sidecar worker for one job; return parsed result. Raises on
    failure (non-zero exit, bad JSON, or {"ok": false})."""
    py = tts_venv_python(engine)
    if not py:
        raise RuntimeError(worker_available(engine) or f"no venv for {engine}")
    job = dict(job, engine=engine)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(job, fh)
        jobfile = fh.name
    try:
        proc = subprocess.run(
            [py, str(_WORKER), jobfile],
            capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1",
                     TOKENIZERS_PARALLELISM="false"))
        # worker prints a single JSON object as its LAST stdout line
        out = (proc.stdout or "").strip().splitlines()
        parsed = None
        for line in reversed(out):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    break
                except ValueError:
                    continue
        if parsed is None:
            raise RuntimeError(
                f"{engine} worker gave no JSON (exit {proc.returncode}): "
                f"{(proc.stderr or '')[-400:]}")
        if not parsed.get("ok"):
            raise RuntimeError(f"{engine} worker error: "
                               f"{parsed.get('error')} {parsed.get('trace','')[-300:]}")
        return parsed
    finally:
        try:
            os.unlink(jobfile)
        except OSError:
            pass
