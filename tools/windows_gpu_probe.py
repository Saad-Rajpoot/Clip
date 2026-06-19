#!/usr/bin/env python3
"""Vidlore Windows GPU-acceleration diagnostic (safe, no secrets).

Prints exactly what the engine will do on THIS machine:
  - detected GPU adapters + NVIDIA / nvidia-smi status
  - the ffmpeg binary Vidlore selected (path + source + version)
  - NVENC encoder availability + a real short NVENC encode probe (PASS/FAIL)
  - the encoder _pick_video_encoder() would choose, with its reason
  - ONNX Runtime providers + the CLIP visual-relevance provider actually selected
    (DirectML adapter id / gpu name, or CPU fallback + reason)
  - active ffmpeg processes (if any)

Run on the actual Windows machine via tools\\check_windows_gpu_acceleration.bat.
It also runs on macOS/Linux (showing that platform's values) as a sanity check.
NEVER prints API keys, tokens, or .env contents.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _line(c="─", n=70):
    print(c * n)


def _hdr(t):
    _line()
    print(t)
    _line()


def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return 127, "(not found)"
    except Exception as e:                                      # noqa: BLE001
        return 1, f"({type(e).__name__}: {e})"


def section_platform():
    _hdr("PLATFORM")
    print(f"  system   : {platform.system()} {platform.release()}")
    print(f"  machine  : {platform.machine()}")
    print(f"  python   : {sys.version.split()[0]}")
    print(f"  cpu_count: {os.cpu_count()}")


def section_gpus():
    _hdr("GPU ADAPTERS")
    # Best-effort adapter enumeration (same call the engine uses for DML auto).
    try:
        from vidlore.visual_relevance import _enumerate_video_adapters
        adapters = _enumerate_video_adapters()
    except Exception as e:                                      # noqa: BLE001
        adapters = []
        print(f"  (adapter enumeration unavailable: {type(e).__name__})")
    if adapters:
        for i, nm in enumerate(adapters):
            tag = " <- NVIDIA" if any(k in nm.lower() for k in
                                      ("nvidia", "geforce", "rtx", "quadro")) else ""
            print(f"  DXGI/WMI[{i}] {nm}{tag}")
        print("  (note: WMI index is a HINT; DirectML validates the real device by run)")
    else:
        print("  (no adapters reported here — on Windows this uses PowerShell WMI)")

    print()
    print("  nvidia-smi:")
    rc, out = _run(["nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,utilization.encoder,"
                    "memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader"])
    if rc == 0:
        for ln in out.strip().splitlines():
            print(f"    {ln.strip()}")
    else:
        print(f"    (nvidia-smi unavailable rc={rc}) {out.strip()[:80]}")


def section_ffmpeg():
    _hdr("FFMPEG (VIDEO ENCODE)")
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe, ffmpeg_source
        ff = ffmpeg_exe()
        src = ffmpeg_source()
    except Exception as e:                                      # noqa: BLE001
        print(f"  ERROR resolving ffmpeg: {type(e).__name__}: {e}")
        return
    print(f"  selected binary : {ff}")
    print(f"  source          : {src}   (bundled_nvenc = the shipped GPU build)")
    rc, out = _run([ff, "-hide_banner", "-version"])
    ver = out.strip().splitlines()[0] if out.strip() else "(no output)"
    print(f"  version         : {ver}")

    rc, enc = _run([ff, "-hide_banner", "-encoders"])
    hw = [ln.split()[1] for ln in enc.splitlines()
          if len(ln.split()) > 1 and any(k in ln for k in
          ("nvenc", "qsv", "amf", "videotoolbox"))]
    print(f"  hw encoders     : {', '.join(sorted(set(hw))) or 'NONE (CPU-only build)'}")

    print()
    print("  real encode probes (tiny throwaway encode on THIS ffmpeg):")
    try:
        from vidlore.assemble import _probe_encoder, _pick_video_encoder
        for name in ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox"):
            ok = _probe_encoder(name)
            print(f"    {name:20s} {'PASS' if ok else 'fail'}")
        print()
        # Clear the lru_cache so the log line prints fresh for this probe.
        try:
            _pick_video_encoder.cache_clear()
        except Exception:                                      # noqa: BLE001
            pass
        chosen = _pick_video_encoder()
        print(f"  => Vidlore will encode with: {chosen}")
    except Exception as e:                                      # noqa: BLE001
        print(f"    ERROR running encode probe: {type(e).__name__}: {e}")


def section_clip():
    _hdr("CLIP VISUAL-RELEVANCE (ONNX RUNTIME)")
    try:
        import onnxruntime as ort
        print(f"  onnxruntime ver : {ort.__version__}")
        print(f"  providers avail : {ort.get_available_providers()}")
        dml = "DmlExecutionProvider" in ort.get_available_providers()
        print(f"  DirectML present: {dml}  "
              f"({'onnxruntime-directml installed' if dml else 'install onnxruntime-directml for GPU CLIP'})")
    except Exception as e:                                      # noqa: BLE001
        print(f"  onnxruntime unavailable: {type(e).__name__}: {e}")
        return
    print()
    print(f"  VIDLORE_CLIP_PROVIDER = {os.environ.get('VIDLORE_CLIP_PROVIDER', 'auto')}")
    print(f"  VIDLORE_DML_DEVICE_ID = {os.environ.get('VIDLORE_DML_DEVICE_ID', '(auto)')}")
    os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
    try:
        import vidlore.visual_relevance as vr
        loaded = vr._try_load()
        st = vr.vr_status()
        print(f"  model loaded    : {loaded}")
        print(f"  active backend  : {st.get('backend')}")
        print(f"  is_directml     : {st.get('is_directml')}")
        print(f"  device_id       : {st.get('device_id')}")
        print(f"  gpu_name        : {st.get('gpu_name') or '(n/a)'}")
        print(f"  degraded        : {st.get('degraded')}")
        print("  (gate is fail-safe: if GPU inference fails it falls back to CPU, "
              "never disabled)")
    except Exception as e:                                      # noqa: BLE001
        print(f"  ERROR loading CLIP: {type(e).__name__}: {e}")


def section_processes():
    _hdr("ACTIVE FFMPEG PROCESSES")
    if platform.system() == "Windows":
        rc, out = _run(["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"])
    else:
        rc, out = _run(["pgrep", "-fl", "ffmpeg"])
    txt = out.strip()
    print("  " + ("\n  ".join(txt.splitlines()[:10]) if txt and "INFO:" not in txt
                  else "(none running)"))


def main():
    print()
    print("VIDLORE — WINDOWS GPU ACCELERATION DIAGNOSTIC")
    print("(safe diagnostics only — no API keys / secrets printed)")
    section_platform()
    section_gpus()
    section_ffmpeg()
    section_clip()
    section_processes()
    _line()
    print("Done. If 'h264_nvenc PASS' and CLIP backend=DmlExecutionProvider, the")
    print("RTX is fully wired. If either shows CPU/fail, that path safely fell back")
    print("(check the reason above) — the render still works, just on CPU.")
    _line()


if __name__ == "__main__":
    main()
