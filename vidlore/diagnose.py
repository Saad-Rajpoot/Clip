"""One-shot environment diagnostic.

Run:  python -m vidlore.diagnose   (or double-click windows-diagnose.bat)

Writes vidlore_diag.txt next to the folder AND prints it. It NEVER prints
secret values — only which KEY NAMES are present and which providers are
active. Send that file/screenshot back so the exact cause is known
instead of guessed.
"""
from __future__ import annotations

import os
import platform
import sys
import tempfile
from pathlib import Path


def _line(s: str = "") -> str:
    return s + "\n"


def report() -> str:
    out = ""
    out += _line("==== Vidlore diagnostic ====")
    out += _line(f"python   : {sys.version.split()[0]}  ({platform.system()} "
                 f"{platform.machine()})")
    out += _line(f"cwd      : {os.getcwd()}")
    out += _line(f"package  : {Path(__file__).resolve().parent}")
    out += _line()

    # --- .env presence + which KEY NAMES (NO values) ---
    env_path = Path(os.getcwd()) / ".env"
    out += _line(f".env at cwd: {'FOUND' if env_path.exists() else 'NOT FOUND'}"
                 f"  ({env_path})")
    if env_path.exists():
        try:
            names = []
            for ln in env_path.read_text(errors="ignore").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, _, v = ln.partition("=")
                    k = k.strip()
                    has = "set" if v.strip() else "EMPTY"
                    names.append(f"{k}={has}")
            out += _line("  keys: " + ", ".join(names) if names
                         else "  (no keys parsed!)")
        except Exception as e:  # noqa: BLE001
            out += _line(f"  (could not read .env: {e})")
    out += _line()

    # --- config: are the real providers ACTIVE? (the key signal) ---
    try:
        from .config import load_config
        cfg = load_config()
        out += _line("---- active providers (load_config) ----")
        out += _line(cfg.describe())
        out += _line()
        out += _line(f"  has_llm   : {cfg.has_llm}")
        out += _line(f"  has_stock : {cfg.has_stock}  (Pexels)")
        out += _line(f"  has_fal   : {cfg.has_fal}")
        out += _line(f"  PIXABAY   : {bool(os.environ.get('PIXABAY_API_KEY'))}")
        out += _line(f"  UNSPLASH  : "
                     f"{bool(os.environ.get('UNSPLASH_ACCESS_KEY'))}")
    except Exception as e:  # noqa: BLE001
        out += _line(f"!! load_config FAILED: {e!r}")
    out += _line()

    # --- fonts: is on-screen text/graphics able to render? ---
    out += _line("---- fonts (text/graphics ka linchpin) ----")
    try:
        import vidlore.assemble as A
        bundled = A._FONT_CANDIDATES[0]
        out += _line(f"  bundled font path : {bundled}")
        out += _line(f"  bundled exists    : {Path(bundled).exists()}")
        wd = Path(tempfile.mkdtemp(prefix="diagfont_"))
        picked = A._copy_font(wd)
        verdict = ("OK - text/graphics will render" if picked
                   else "None -> ALL text/graphics SKIPPED (this is the bug)")
        out += _line(f"  _copy_font picked : {picked}  ({verdict})")
    except Exception as e:  # noqa: BLE001
        out += _line(f"!! font check FAILED: {e!r}")
    try:
        import vidlore.footage as F
        fn = F._font(40)
        nm = fn.getname() if hasattr(fn, "getname") else str(fn)
        out += _line(f"  footage PIL font  : {nm}")
    except Exception as e:  # noqa: BLE001
        out += _line(f"!! footage font FAILED: {e!r}")
    out += _line()

    # --- imports / ffmpeg ---
    out += _line("---- imports / ffmpeg ----")
    try:
        import vidlore.pipeline  # noqa: F401
        import vidlore.web       # noqa: F401
        out += _line("  modules import : OK")
    except Exception as e:  # noqa: BLE001
        out += _line(f"!! import FAILED: {e!r}")
    try:
        from .ffmpeg_tool import ffmpeg_exe
        fe = ffmpeg_exe()
        out += _line(f"  ffmpeg         : {fe}  "
                     f"({'exists' if Path(fe).exists() else 'MISSING'})")
    except Exception as e:  # noqa: BLE001
        out += _line(f"!! ffmpeg FAILED: {e!r}")
    out += _line()
    out += _line("==== send this whole text back ====")
    return out


def main() -> None:
    txt = report()
    print(txt)
    try:
        Path(os.getcwd(), "vidlore_diag.txt").write_text(txt, encoding="utf-8")
        print(f"[saved] {Path(os.getcwd(), 'vidlore_diag.txt')}")
    except Exception as e:  # noqa: BLE001
        print(f"[could not save file: {e}]")


if __name__ == "__main__":
    main()
