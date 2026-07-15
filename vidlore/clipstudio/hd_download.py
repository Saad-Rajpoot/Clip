"""HD YouTube downloader — defeats YouTube's 2025+ SABR / PO-token gate.

Background: as of 2025 YouTube enforces SABR (Server-side Adaptive BitRate) and binds a GVS
PO-token to each request. The engine venv's yt-dlp (pinned to the last Python-3.9 build,
2025.10.14) can no longer see or fetch the HD (≥720p) formats — it silently degrades to the one
legacy muxed 360p stream (format 18). That is why portal/CLI clips came out at 360p.

The WORKING method (proven on real HBO clips) needs four things together; miss any one and HD
collapses back to 360p:
  1. a RECENT yt-dlp (≥2026.x) that implements SABR streaming  → an isolated Python-3.11 venv
     (.hdvenv) because the latest yt-dlp dropped Python 3.9, which the engine venv still uses;
  2. a bgutil PO-token provider HTTP server on :4416 (Deno)    → generates the GVS PO token that
     unlocks the adaptive HD formats;
  3. a JS runtime (node + deno) on PATH                        → solves YouTube's "n-challenge"
     (nsig); without it the HD format URLs come back empty;
  4. the browser's logged-in YouTube cookies                  → `--cookies-from-browser chrome`.

This module wraps all four behind one call. It NEVER bypasses access controls or DRM — it uses
YouTube's own public player clients + the user's own logged-in session, exactly as the browser
does. It only fetches what the user is already entitled to view.

Everything is path/ENV-overridable and degrades gracefully: if the .hdvenv or PO server can't be
brought up, `download_hd` returns None and the caller falls back to the engine's in-process
yt-dlp (legacy 360p) — so the tool keeps working, just without HD.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

# --- resolved locations (all overridable via env) ----------------------------
_ROOT = Path(__file__).resolve().parents[2]                  # vidlore-clipstudio/

def _firstpath(*cands: str) -> str:
    for c in cands:
        if c and Path(c).expanduser().exists():
            return str(Path(c).expanduser())
    return ""

HD_PY = os.environ.get("VIDLORE_HD_PYTHON", "") or _firstpath(
    str(_ROOT / ".hdvenv/bin/python"), str(_ROOT / ".hdvenv/bin/python3"),
    str(_ROOT / ".hdvenv/Scripts/python.exe"))          # Windows venv layout
NODE_BIN = os.environ.get("VIDLORE_HD_NODE", "") or _firstpath(
    str(Path.home() / ".local-node/bin/node"), shutil.which("node") or "")
DENO_BIN = os.environ.get("VIDLORE_HD_DENO", "") or _firstpath(
    str(Path.home() / ".deno/bin/deno"), shutil.which("deno") or "")
POT_SERVER_DIR = os.environ.get("VIDLORE_HD_POT_DIR", "") or _firstpath(
    str(_ROOT / ".pot/server"))
try:
    POT_PORT = int(os.environ.get("VIDLORE_HD_POT_PORT", "4416"))
except (TypeError, ValueError):
    POT_PORT = 4416
COOKIES_BROWSER = os.environ.get("VIDLORE_HD_COOKIES_BROWSER", "chrome").strip()
# turn the whole HD path off with VIDLORE_HD_DOWNLOAD=0 (case-insensitive; unset/empty = ON)
_HD_ENV = os.environ.get("VIDLORE_HD_DOWNLOAD", "").strip().lower()
HD_ENABLED = (_HD_ENV not in ("0", "false", "no", "off")) if _HD_ENV else True

import threading as _threading

_POT_PROC = None  # bgutil server we started (if any)
_POT_LOCK = _threading.Lock()


def is_youtube(url: str) -> bool:
    u = (url or "").lower()
    return ("youtube.com/" in u) or ("youtu.be/" in u) or ("youtube-nocookie.com/" in u)


def available() -> bool:
    """Is the HD path even possible on this machine?"""
    return bool(HD_ENABLED and HD_PY and NODE_BIN and DENO_BIN and POT_SERVER_DIR)


def _pot_alive() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{POT_PORT}/ping", timeout=4) as r:
            return r.status == 200 and b"version" in r.read()
    except Exception:
        return False


def ensure_pot_server(progress=None) -> bool:
    """Make sure the bgutil PO-token server is listening on :POT_PORT — start it if not.
    Lock-guarded: concurrent download workers must not race the check-then-act and spawn
    several servers fighting for the port."""
    global _POT_PROC
    with _POT_LOCK:
        if _pot_alive():
            return True
        if not (DENO_BIN and POT_SERVER_DIR):
            return False
        logf = None
        try:
            import tempfile as _tf
            logf = open(os.path.join(_tf.gettempdir(), f"clipstudio_pot_{POT_PORT}.log"), "ab")
            _POT_PROC = subprocess.Popen(
                [DENO_BIN, "run", "-A", "--node-modules-dir=auto", "src/main.ts", "--port", str(POT_PORT)],
                cwd=POT_SERVER_DIR, stdout=logf, stderr=logf,
                start_new_session=True,                      # detach; survive our process
            )
        except Exception as e:                               # noqa: BLE001
            if progress:
                progress(f"hd: PO server failed to start ({str(e)[:80]})")
            return False
        finally:
            if logf is not None:                             # child holds its own dup of the fd
                try:
                    logf.close()
                except Exception:
                    pass
        for _ in range(30):                                  # up to ~15s for first boot
            if _pot_alive():
                if progress:
                    progress(f"hd: PO-token server up on :{POT_PORT}")
                return True
            time.sleep(0.5)
        return False


def _env_with_runtimes() -> dict:
    env = dict(os.environ)
    extra = os.pathsep.join(p for p in (str(Path(NODE_BIN).parent) if NODE_BIN else "",
                                        str(Path(DENO_BIN).parent) if DENO_BIN else "") if p)
    env["PATH"] = extra + os.pathsep + env.get("PATH", "") if extra else env.get("PATH", "")
    return env


def _format_selector(max_h: int) -> str:
    # Grab the BEST video ≤max_h regardless of codec, paired with the best audio (then progressive
    # fallbacks). RESOLUTION FIRST — many YouTube uploads cap H.264 at 720p but offer 1080p only in
    # VP9/AV1, so an avc1-first selector silently downgraded to 720p. We let `-S res:max_h` (below)
    # pick the highest ≤max_h in ANY codec and use H.264 only as a same-resolution tie-breaker; the
    # engine re-encodes to H.264 on cut anyway, so VP9/AV1 sources are fine.
    return f"bv*[height<={max_h}]+ba/b[height<={max_h}]/bv*+ba/b"


def _format_sort(max_h: int) -> str:
    # RESOLUTION first, then BITRATE (the best available proxy for decoded quality), then codec
    # efficiency, then m4a audio. The old sort put `fps` and `vcodec:h264` AHEAD of bitrate, so among
    # 1080p renditions it chose a starved 60fps AVC1 stream (~0.6 Mbps — visibly soft/blocky) over a
    # higher-bitrate VP9/AV1 1080p; and it rewarded upconverted 60fps rips. We never prioritize 60fps
    # or H.264 over a cleaner same-resolution format — the engine re-encodes to H.264 on cut anyway,
    # so a VP9/AV1 source at higher bitrate is strictly better source quality.
    return f"res:{max_h},br,vcodec,acodec:m4a"


def probe_max_height(url: str, *, max_height: int = 1080, timeout: int = 60) -> int:
    """TRUE max video height available ≤max_height (any codec) via the SABR-capable setup — the
    legacy yt-dlp probe sees only the SABR-collapsed ~360p view, so discovery couldn't tell a 1080p
    upload from a 360p one. 0 on failure (caller keeps its legacy estimate)."""
    if not (available() and is_youtube(url)):
        return 0
    if not ensure_pot_server():
        return 0
    cmd = [
        HD_PY, "-m", "yt_dlp", "--cookies-from-browser", COOKIES_BROWSER,
        "--extractor-args", f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{POT_PORT}",
        "-f", f"bv*[height<={max_height}]/b[height<={max_height}]/bv*/b",
        "-S", _format_sort(max_height),
        "--no-warnings", "--quiet", "--skip-download", "--print", "%(height)s", url,
    ]
    try:
        p = subprocess.run(cmd, env=_env_with_runtimes(), capture_output=True, text=True, timeout=timeout)
        for line in (p.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    return 0


def _merge_ffmpeg_dir(hint: str = "") -> str:
    """A directory that contains an executable literally named `ffmpeg` — required for yt-dlp to MERGE
    the separate DASH video+audio streams into one file. CRITICAL: imageio-ffmpeg ships its binary as
    `ffmpeg-macos-aarch64-v7.1` (a VERSIONED name), so `--ffmpeg-location <imageio dir>` finds nothing
    named `ffmpeg` and yt-dlp silently SKIPS the merge → a video-only file with NO audio → ASR gets 0
    words → dialogue-lock (the strongest scene signal) is dead. We resolve a properly-named ffmpeg:
    a system one if present, else a one-time symlink of the imageio binary as `ffmpeg`."""
    import shutil
    # 1) the hint dir already has a real `ffmpeg`?
    for d in [hint]:
        if d and os.path.exists(os.path.join(d, "ffmpeg")):
            return d
    # 2) a system ffmpeg on PATH or in common install dirs
    cands = [shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg",
             os.path.expanduser("~/pinokio/bin/miniconda/bin/ffmpeg")]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.dirname(c)
    # 3) symlink the engine's (versioned) imageio binary as `ffmpeg` in a stable cache dir
    try:
        from .config import ffmpeg_exe as _fexe
        real = _fexe()
        if real and os.path.exists(real):
            cache = os.path.join(os.path.expanduser("~"), ".cache", "clipstudio_ffmpeg")
            os.makedirs(cache, exist_ok=True)
            link = os.path.join(cache, "ffmpeg")
            if not os.path.exists(link):
                try:
                    os.symlink(real, link)
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(real, link)
                    os.chmod(link, 0o755)
            if os.path.exists(link):
                return cache
    except Exception:                                       # noqa: BLE001
        pass
    return hint


def download_hd(url: str, out_stem: str, *, max_height: int = 1080, ffmpeg_dir: str = "",
                retries: int = 2, timeout: int = 600, progress=None) -> dict | None:
    """Download `url` at the best ≤max_height HD via the SABR-capable recipe.

    Writes to `<out_stem>.<ext>` (merged mp4). Returns a probe-style dict
    {path,height,width,duration,fps,title} on success, or None to signal the caller to fall back.
    """
    if not (available() and is_youtube(url)):
        return None
    if not ensure_pot_server(progress=progress):
        if progress:
            progress("hd: PO server unavailable — falling back to legacy downloader")
        return None

    # ABSOLUTE stem, always: yt-dlp resolves a relative -o template against the subprocess
    # cwd (set to the stem's dir below), so a relative stem nested the output into
    # <sources>/<stem-dir>/<stem>.mp4 and the produced-file lookup found nothing — every
    # download silently fell back to the 360p legacy path.
    out_stem = str(Path(out_stem).expanduser().resolve())
    out_tmpl = f"{out_stem}.%(ext)s"
    info_json = f"{out_stem}.info.json"
    cmd = [
        HD_PY, "-m", "yt_dlp",
        "--cookies-from-browser", COOKIES_BROWSER,
        "--extractor-args", f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{POT_PORT}",
        "-f", _format_selector(max_height),
        "-S", _format_sort(max_height),
        "--merge-output-format", "mp4",
        "--no-playlist", "--no-warnings", "--quiet", "--no-progress",
        "--retries", str(retries), "--fragment-retries", str(retries),
        "--write-info-json",
        "-o", out_tmpl,
    ]
    # ALWAYS give yt-dlp a properly-named ffmpeg so it MERGES video+audio (else ASR sees a silent
    # video-only file and dialogue-lock dies). _merge_ffmpeg_dir guarantees a usable `ffmpeg`.
    mdir = _merge_ffmpeg_dir(ffmpeg_dir)
    if mdir:
        cmd += ["--ffmpeg-location", mdir]
    cmd.append(url)

    env = _env_with_runtimes()
    last = ""
    ok = False
    for attempt in range(1, retries + 2):
        try:
            p = subprocess.run(cmd, cwd=os.path.dirname(out_stem) or ".", env=env,
                               capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0:
                ok = True
                break
            last = (p.stderr or p.stdout or "").strip()[-300:]
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
        except Exception as e:                               # noqa: BLE001
            last = str(e)[:300]
        time.sleep(min(2 ** attempt, 8))
    # locate produced file — only on a clean exit, and never a video-only .fNNN fragment
    # (an interrupted merge leaves <stem>.f616.mp4, which is silent → ASR/dialogue-lock dies)
    produced = None
    stem = Path(out_stem)
    if ok:
        for ext in (".mp4", ".mkv", ".webm", ".mov"):        # exact merge target first
            c = stem.parent / (stem.name + ext)
            if c.exists() and c.stat().st_size > 0:
                produced = c
                break
        if produced is None:
            for f in sorted(stem.parent.glob(stem.name + ".*")):
                if (f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")
                        and not re.search(r"\.f\d+$", f.stem)
                        and f.stat().st_size > 0):
                    produced = f
                    break
    if produced is None:
        # sweep the debris this gate just rejected — orphaned multi-GB .fNNN intermediates,
        # .part/.temp partials and the info.json would otherwise strand in sources/ forever
        # (the legacy fallback downloader writes to the same <stem>.* namespace next)
        for f in stem.parent.glob(stem.name + ".*"):
            try:
                inner = Path(f.stem)
                if re.search(r"\.f\d+$", inner.name) or \
                        f.suffix.lower() in (".part", ".temp", ".ytdl"):
                    f.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            os.remove(info_json)
        except OSError:
            pass
        if progress:
            progress(f"hd: no HD file ({last[:120]}) — fallback")
        return None
    meta = {}
    try:
        meta = json.load(open(info_json))
    except Exception:
        pass
    finally:
        try:
            os.remove(info_json)
        except Exception:
            pass
    # report the produced file's REAL dimensions — info.json describes the format yt-dlp
    # SELECTED, which mislabels a pre-existing lower-res file it skipped as "already downloaded"
    try:
        from .ingest import probe as _probe
        real = _probe(produced)
        if real.get("width"):
            meta["width"], meta["height"] = real["width"], real["height"]
            if real.get("duration"):
                meta["duration"] = real["duration"]
            if real.get("fps"):
                meta["fps"] = real["fps"]
    except Exception:
        pass
    return {
        "path": str(produced),
        "height": int(meta.get("height") or 0),
        "width": int(meta.get("width") or 0),
        "duration": float(meta.get("duration") or 0.0),
        "fps": float(meta.get("fps") or 0.0),
        "title": meta.get("title") or "",
        "uploader": meta.get("uploader") or meta.get("channel") or "",
    }
