#!/usr/bin/env python3
"""Pre-render asset check — is this machine actually wired to produce a GOOD video?

Every item below has bitten a real render, and each one fails in its own way:

  music    missing -> build RAISES at the very end (the no-silent-music gate)
  CLIP     missing -> the render refuses to start at all (stage 0)
  Face-ID  missing -> renders fine, but wrong-character footage can no longer be rejected
  HD       missing -> renders fine, and EVERY source is silently ~360p
  ffprobe  missing -> the A/V-sync invariant aborts at the last step

So this prints a plain READY/PROBLEM block with the exact remedy for whatever is missing,
and the launcher runs it on every start. It is deliberately CHEAP: it checks that model
FILES are in place rather than loading them (loading CLIP means ~600 MB and several
seconds), because the question here is "did the assets get copied", not "does onnxruntime
work" — the render itself answers that.

Never raises, never blocks: exit code is always 0 unless --strict is passed.

    python tools/clipstudio_preflight.py [--strict]
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "OK", "!!", "XX"


def _music():
    try:
        from vidlore import musiclib
        cats = musiclib.scan()
        n = sum(len(v) for v in cats.values())
        if n >= 100 and len(cats) >= 8:
            return OK, f"music        {len(cats)} categories / {n} tracks", ""
        return BAD, f"music        only {len(cats)} categories / {n} tracks", (
            "vidlore\\assets\\music copy karein (118 mp3), ya VIDLORE_MUSIC_DIR set karein. "
            "Iske baghair build aakhir mein error dega.")
    except Exception as e:                                    # noqa: BLE001
        return BAD, f"music        check fail ({type(e).__name__})", (
            "vidlore\\assets\\music mojood nahi lagta.")


def _clip():
    try:
        from vidlore import visual_relevance as vr
        d = Path(getattr(vr, "_CLIP_DIR", "") or "")
        need = ["clip_vision.onnx", "clip_text.onnx", "tokenizer.json"]
        missing = [f for f in need if not (d / f).exists()]
        if d and not missing:
            return OK, f"CLIP models  {d}", ""
        return BAD, f"CLIP models  MISSING {', '.join(missing) or '(dir not set)'} in {d}", (
            "models\\clip\\ mein teen files rakhein (clip_vision.onnx, clip_text.onnx, "
            "tokenizer.json). Folder ka naam bilkul 'models\\clip' ho. INKE BAGHAIR RENDER "
            "SHURU HI NAHI HOGA.")
    except Exception as e:                                    # noqa: BLE001
        return BAD, f"CLIP models  check fail ({type(e).__name__})", ""


def _faceid():
    try:
        from vidlore.clipstudio import faceid as F
        d = Path(getattr(F, "MODELS_DIR", "") or "")
        missing = [f for f in ("yunet.onnx", "sface.onnx") if not (d / f).exists()]
        if not missing:
            return OK, f"Face-ID      {d}", ""
        return WARN, f"Face-ID      MISSING {', '.join(missing)} in {d}", (
            "models\\faceid\\ mein yunet.onnx + sface.onnx rakhein. Inke baghair render "
            "chalega, magar GALAT CHARACTER wali footage reject nahi hogi.")
    except Exception as e:                                    # noqa: BLE001
        return WARN, f"Face-ID      check fail ({type(e).__name__})", ""


def _hd():
    try:
        from vidlore.clipstudio import hd_download as H
        if H.available():
            return OK, "HD download  720-1080p ready (Deno + .hdvenv + PO server)", ""
        why = []
        if not getattr(H, "HD_ENABLED", True):
            why.append("VIDLORE_HD_DOWNLOAD=0")
        if not getattr(H, "HD_PY", ""):
            why.append(".hdvenv nahi")
        if not getattr(H, "DENO_BIN", ""):
            why.append("Deno nahi")
        if not getattr(H, "POT_SERVER_DIR", ""):
            why.append(".pot server nahi")
        return WARN, f"HD download  OFF ({', '.join(why) or 'unknown'})", (
            "Launcher ise khud set karta hai — dobara chalayein. Warna SAARI FOOTAGE ~360p "
            "aayegi (koi error nahi aayega). Node ki zaroorat NAHI hai.")
    except Exception as e:                                    # noqa: BLE001
        return WARN, f"HD download  check fail ({type(e).__name__})", ""


def _ffmpeg():
    try:
        from vidlore.clipstudio.config import ffmpeg_exe, ffprobe_exe
        fm, fp = ffmpeg_exe() or "", ffprobe_exe() or ""
        if fm and os.path.exists(fm) and fp and os.path.exists(fp):
            return OK, f"ffmpeg       + ffprobe found", ""
        miss = "ffmpeg" if not (fm and os.path.exists(fm)) else "ffprobe"
        return BAD, f"ffmpeg       {miss} NAHI mila", (
            "ffmpeg\\bin\\ffmpeg.exe aur ffmpeg\\bin\\ffprobe.exe rakhein "
            "(python tools\\fetch_windows_nvenc_ffmpeg.py --dest ffmpeg). ffprobe ASLI ho — "
            "ffmpeg ki copy ka naam badalna kaafi nahi.")
    except Exception as e:                                    # noqa: BLE001
        return BAD, f"ffmpeg       check fail ({type(e).__name__})", ""


def _keys():
    have = [k for k in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY") if (os.environ.get(k) or "").strip()]
    if len(have) == 2:
        return OK, "API keys     DeepSeek + Gemini set", ""
    missing = [k for k in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY") if k not in have]
    return BAD, f"API keys     MISSING {', '.join(missing)}", (
        ".env file folder ke root mein honi chahiye. .env.example se NA banayein — us mein "
        "DEEPSEEK_API_KEY nahi hai.")


def main() -> int:
    # the launcher already loads .env for the portal; do it here too so a direct run is honest
    try:
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:                                         # noqa: BLE001
        pass

    rows = [_keys(), _music(), _clip(), _faceid(), _hd(), _ffmpeg()]
    print("=" * 62)
    print("  CLIPSTUDIO PRE-FLIGHT")
    print("=" * 62)
    for state, line, _ in rows:
        print(f"  [{state}] {line}")
    fixes = [(s, r) for s, _l, r in rows if r and s in (WARN, BAD)]
    if fixes:
        print("-" * 62)
        for state, remedy in fixes:
            tag = "ZAROORI" if state == BAD else "behtar"
            print(f"  ({tag}) {remedy}")
    print("=" * 62)
    blockers = sum(1 for s, _l, _r in rows if s == BAD)
    warns = sum(1 for s, _l, _r in rows if s == WARN)
    if blockers:
        print(f"  {blockers} cheez ZAROORI hai — us ke baghair render theek nahi hoga.")
    elif warns:
        print(f"  Render ho jayega, magar {warns} cheez se quality girti hai (upar dekhein).")
    else:
        print("  Sab ready — video bana sakte hain.")
    print()
    return 1 if (blockers and "--strict" in sys.argv) else 0


if __name__ == "__main__":
    sys.exit(main())
