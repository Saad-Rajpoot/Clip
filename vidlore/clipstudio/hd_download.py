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
import platform
import re
import shutil
import signal
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
#  Both runtimes are optional-to-locate here; only DENO is actually required (see available()).
#  The Windows candidates matter: the official Deno installer drops `%USERPROFILE%\.deno\bin\
#  deno.exe`, so a POSIX-only candidate list found nothing and HD silently degraded to 360p even
#  right after a successful install.
NODE_BIN = os.environ.get("VIDLORE_HD_NODE", "") or _firstpath(
    str(Path.home() / ".local-node/bin/node"),
    str(Path.home() / ".local-node/bin/node.exe"),
    shutil.which("node") or "")
DENO_BIN = os.environ.get("VIDLORE_HD_DENO", "") or _firstpath(
    str(Path.home() / ".deno/bin/deno"),
    str(Path.home() / ".deno/bin/deno.exe"),
    shutil.which("deno") or "")
POT_SERVER_DIR = os.environ.get("VIDLORE_HD_POT_DIR", "") or _firstpath(
    str(_ROOT / ".pot/server"))
try:
    POT_PORT = int(os.environ.get("VIDLORE_HD_POT_PORT", "4416"))
except (TypeError, ValueError):
    POT_PORT = 4416
COOKIES_BROWSER = os.environ.get("VIDLORE_HD_COOKIES_BROWSER", "chrome").strip()
# Cookies are an OPTIMISATION (logged-in access), never a requirement: the PO-token path fetches
# public videos perfectly well without them. Once the profile proves unreadable it will stay
# unreadable for the whole run — the browser does not release its lock mid-render — so the first
# failure disables them process-wide instead of re-failing once per source.
_COOKIES_OFF = False
# turn the whole HD path off with VIDLORE_HD_DOWNLOAD=0 (case-insensitive; unset/empty = ON)
_HD_ENV = os.environ.get("VIDLORE_HD_DOWNLOAD", "").strip().lower()
HD_ENABLED = (_HD_ENV not in ("0", "false", "no", "off")) if _HD_ENV else True

import threading as _threading

_POT_PROC = None  # bgutil server we started (if any)
_POT_LOCK = _threading.Lock()
_COOKIES_LOCK = _threading.Lock()


def _cookie_args() -> list:
    """The cookie flags for a yt-dlp call — one definition for every call site.

    A cookies FILE (exported once from a private window) is the wiki-endorsed stable method; live
    browser profiles get their cookies ROTATED by YouTube mid-run, and on Windows cannot usually be
    read at all. Returns [] once cookies have been disabled for this process."""
    if _COOKIES_OFF:
        return []
    cfile = os.environ.get("VIDLORE_HD_COOKIES_FILE", "").strip()
    if cfile and Path(cfile).expanduser().exists():
        return ["--cookies", str(Path(cfile).expanduser())]
    if COOKIES_BROWSER:
        return ["--cookies-from-browser", COOKIES_BROWSER]
    return []


def _disable_cookies(reason: str = "", progress=None) -> bool:
    """Drop cookies for the rest of the run. True if this call is the one that turned them off.

    yt-dlp aborts the whole download when it cannot read the browser profile, so without this a
    soft dependency takes the render down to the legacy 360p path — which is exactly what happened
    on the owner's Windows machine, 42 sources in a row."""
    global _COOKIES_OFF
    with _COOKIES_LOCK:
        if _COOKIES_OFF:
            return False
        _COOKIES_OFF = True
    if progress:
        progress("hd: browser cookies unreadable — continuing WITHOUT them (public videos do not "
                 f"need them; export a cookies.txt and set VIDLORE_HD_COOKIES_FILE if a source is "
                 f"age-restricted). Reason: {(reason or '').strip()[:160]}")
    return True

# Concurrency cap for ACTIVE HD yt-dlp subprocesses. A render fires download workers per core;
# dozens of simultaneous SABR/HD data fetches from one IP+session is exactly the burst profile
# that trips YouTube's 403 throttle (measured: a whole 86-source render mass-403'd to 360p while
# a 3-way test from the same machine minutes later succeeded). Legacy stays parallel — only the
# HD data fetch is paced. Backoff sleeps happen OUTSIDE the semaphore so waiters can proceed.
_HD_SEM = _threading.BoundedSemaphore(max(1, int(os.environ.get("VIDLORE_HD_MAX_CONCURRENCY", "3") or 3)))
_POT_403_RESTARTED = False  # one pot-server restart per process on first 403 (stale-token remedy)
_PACE_LOCK = _threading.Lock()
_PACE_LAST = [0.0]


def _pace_hd_start() -> None:
    """Global minimum gap between HD download STARTS (default 6s, env VIDLORE_HD_MIN_GAP).
    The Extractors wiki documents ~300 videos/hour as the guest ceiling with 5-10s delays
    recommended; a render fetching 100+ sources as fast as cores allow is exactly the burst
    profile that earns an hours-long IP block (measured twice)."""
    try:
        gap = float(os.environ.get("VIDLORE_HD_MIN_GAP", "6") or 6)
    except (TypeError, ValueError):
        gap = 6.0
    if gap <= 0:
        return
    with _PACE_LOCK:                                    # reserve the next start slot atomically
        now = time.monotonic()
        start_at = max(now, _PACE_LAST[0] + gap)
        _PACE_LAST[0] = start_at
        wait = start_at - now
    if wait > 0:
        time.sleep(wait)


def maybe_update_ytdlp(log=None, *, force: bool = False) -> bool:
    """Weekly (or forced) self-update of the HD venv's yt-dlp stack. The maintainers' standing
    fix for fleet-wide 403 waves is 'update to the latest release first' (verified: the
    Oct-2025 mass-403 event was closed by a stopgap RELEASE, not a config change) — so version
    rot IS a 403 root cause, and a pipeline that never updates re-earns it every few weeks.
    Env: VIDLORE_HD_AUTOUPDATE=0 disables. Fail-open always."""
    if os.environ.get("VIDLORE_HD_AUTOUPDATE", "1").strip().lower() in ("0", "false", "no"):
        return False
    if not HD_PY:
        return False
    marker = Path(HD_PY).parent.parent / ".last_ytdlp_check"
    try:
        if not force and marker.exists() and (time.time() - marker.stat().st_mtime) < 7 * 86400:
            return False
        p = subprocess.run([HD_PY, "-m", "pip", "install", "-q", "-U",
                            "yt-dlp", "yt-dlp-ejs", "bgutil-ytdlp-pot-provider"],
                           capture_output=True, text=True, timeout=240)
        marker.touch()
        if p.returncode == 0 and log:
            v = subprocess.run([HD_PY, "-m", "yt_dlp", "--version"],
                               capture_output=True, text=True, timeout=30)
            log(f"hd: yt-dlp stack updated (now {(v.stdout or '').strip()})")
        return p.returncode == 0
    except Exception:                                    # noqa: BLE001
        return False


#  YouTube's SABR/PO-token rejection. It arrives dressed as unavailability — "This video is
#  unavailable. Error code: 152" — but it is NOT: the very same ids download fine on the legacy
#  (non-SABR) path, which is exactly what happened on a measured render where 110/110 sources hit
#  this and the whole 22-minute video shipped at 360p upscaled to 1080p. It has to be treated like
#  a stale token (restart the PO server, sweep, retry), never as "this video does not exist".
_PO_REJECT_RX = re.compile(r"error code:\s*152\b", re.I)
# COOKIE-EXTRACTION failure — yt-dlp aborts the whole download when it cannot READ the browser
# profile, which has nothing to do with the video. Windows is where this bites: Chrome holds an
# exclusive lock on its cookie DB while running (yt-dlp issue 7271), and since Chrome 127 App-Bound
# Encryption puts the values out of reach anyway. Measured on the owner's Windows render: 42/42
# YouTube sources fell back to the legacy 360p downloader on this error alone.
_COOKIE_FAIL = (r"(?:could not|couldn'?t|cannot|can'?t|unable|fail(?:ed|s)?|denied|permission"
                r"|locked|no such|missing)")
_COOKIE_ERR_RX = re.compile(
    # the failure word sits on either side of the noun, so match both ways round
    rf"cookie\w*.{{0,60}}?{_COOKIE_FAIL}|{_COOKIE_FAIL}.{{0,60}}?cookie"
    # Chrome >=127 App-Bound Encryption: yt-dlp reports a DPAPI decrypt failure that never says
    # "cookie". Both Windows cookie failures cite the same tracking issue, so match that too.
    rf"|\bdpapi\b|yt-dlp/issues/7271", re.I | re.S)
# NOT a marker: the bare flag name. yt-dlp ADVISES "Use --cookies-from-browser" in its bot-check
# message, and reading that as a cookie failure would switch cookies OFF at the one moment they are
# the actual remedy. The sign-in/bot patterns below are also tested BEFORE this class for the same
# reason.


def _classify_dl_err(text: str) -> str:
    """'throttle_403' (transient: throttle, stale PO token, or a SABR/PO rejection), 'cookies'
    (the browser profile is unreadable — retry WITHOUT them), 'unavailable' (permanent — do not
    retry), or 'other'. Classification drives retry/backoff AND the recovery sweep; NEVER treat a
    token rejection as unavailability."""
    t = (text or "").lower()
    if "403" in t and "forbidden" in t:
        return "throttle_403"
    if _PO_REJECT_RX.search(t):
        return "throttle_403"                      # PO-token rejection, not a missing video
    for pat in ("video unavailable", "private video", "sign in to confirm",
                "members-only", "this video is not available", "account associated",
                "has been removed", "video is no longer available", "age-restricted",
                "geo restricted", "not available in your country"):
        if pat in t:
            return "unavailable"
    if _COOKIE_ERR_RX.search(t):
        return "cookies"                           # our own optional flag, not the video's fault
    return "other"


def is_po_token_failure(text: str) -> bool:
    """Did this fallback reason come from a PO-token/SABR rejection (recoverable) rather than the
    video genuinely being gone? Used by the download-stage recovery sweep, which previously keyed
    on the literal string '403' and so never fired for error 152."""
    t = (text or "").lower()
    return bool(("403" in t and "forbidden" in t) or _PO_REJECT_RX.search(t))


def is_cookie_failure(text: str) -> bool:
    """Did this fallback come from an unreadable browser profile rather than the video?"""
    return _classify_dl_err(text or "") == "cookies"


def is_recoverable_hd_failure(text: str) -> bool:
    """Should the download stage's sweep re-attempt HD for a source that fell back with this?

    Downloads run in PARALLEL, so on a machine where cookies are unreadable a burst of sources can
    fail together before the first one flips the process-wide switch. Those are stranded at 360p by
    a condition that no longer exists — exactly what the sweep is for. Only genuine unavailability
    may leave a source SD."""
    return is_po_token_failure(text) or is_cookie_failure(text)


def _restart_pot_server(progress=None) -> bool:
    """Kill + relaunch the bgutil PO-token server ONCE per process when HD data URLs 403.
    /ping proves the deno process answers, NOT that the tokens it mints are still valid — a
    long-lived server (observed: 6 days up) can hand out stale tokens that 403 on every
    adaptive format while health checks pass. Gated by VIDLORE_HD_POT_RESTART_ON_403."""
    global _POT_PROC, _POT_403_RESTARTED
    if os.environ.get("VIDLORE_HD_POT_RESTART_ON_403", "1").lower() in ("0", "false", "no"):
        return False
    with _POT_LOCK:
        if _POT_403_RESTARTED:
            return False
        _POT_403_RESTARTED = True
        try:
            if _POT_PROC is not None and _POT_PROC.poll() is None:
                _POT_PROC.terminate()
                try:
                    _POT_PROC.wait(timeout=8)
                except Exception:
                    _POT_PROC.kill()
                _POT_PROC = None
            else:
                # server started by a previous run/portal — find its pid on our port.
                # `lsof` does not exist on Windows; netstat+taskkill is the equivalent there
                # (the same idiom run-windows.bat already uses for the portal port).
                for pid in _pids_on_port(POT_PORT):
                    _kill_pid(pid)
                time.sleep(1.5)
        except Exception:                                    # noqa: BLE001
            pass
    if progress:
        progress("hd: 403 on data URLs — restarting PO-token server (stale-token remedy)")
    return ensure_pot_server(progress=progress)


def _pids_on_port(port: int) -> list:
    """PIDs listening on `port`. `lsof` is POSIX-only, so Windows uses netstat. [] on any failure —
    a failed reclaim just means the restart falls through to the liveness check."""
    out = []
    try:
        if platform.system() == "Windows":
            p = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                               capture_output=True, text=True, timeout=15)
            for line in (p.stdout or "").splitlines():
                parts = line.split()
                # PROTO  LOCAL            FOREIGN          STATE       PID
                if len(parts) >= 5 and parts[0].lower().startswith("tcp") \
                        and parts[1].endswith(f":{port}") and "LISTEN" in parts[3].upper():
                    out.append(parts[4])
        else:
            p = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                               capture_output=True, text=True, timeout=10)
            out = (p.stdout or "").split()
    except Exception:                                        # noqa: BLE001
        return []
    return [x for x in out if x.strip().isdigit()]


def _kill_pid(pid) -> None:
    """Terminate one pid. os.kill+SIGTERM has no useful meaning on Windows, where SIGTERM is
    emulated as an unconditional TerminateProcess for the CURRENT process only — taskkill is the
    portable way to stop another process there."""
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=15)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except Exception:                                        # noqa: BLE001
        pass


#  Win32 process-creation flags, spelled out rather than read off `subprocess`: those attributes
#  only EXIST on Windows, so `getattr(subprocess, ...)` collapses to 0 whenever this branch is
#  reviewed or unit-tested from macOS — i.e. the one place the value could be checked before
#  shipping is the one place it silently read as "no flags".
_WIN_DETACHED_PROCESS = 0x00000008
_WIN_CREATE_NEW_PROCESS_GROUP = 0x00000200


def _detach_kwargs() -> dict:
    """Popen kwargs that detach a long-lived helper so it survives this process.
    `start_new_session` is POSIX-only (setsid); Windows needs creation flags instead."""
    if platform.system() == "Windows":
        return {"creationflags": _WIN_DETACHED_PROCESS | _WIN_CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def is_youtube(url: str) -> bool:
    u = (url or "").lower()
    return ("youtube.com/" in u) or ("youtu.be/" in u) or ("youtube-nocookie.com/" in u)


def available() -> bool:
    """Is the HD path even possible on this machine?

    Node is deliberately NOT required. Nothing here runs it: the PO-token server is started with
    DENO (`deno run --node-modules-dir=auto`, which manages node_modules itself), yt-dlp is given
    only the `youtubepot-bgutilhttp` HTTP endpoint, and yt-dlp-ejs's JS challenges are served by
    the same Deno runtime. Node appeared only in this gate and in the PATH we hand the subprocess
    — so on a machine with Deno but no Node the whole HD path reported "unavailable" and EVERY
    source silently fell back to ~360p for a dependency that was never used. Its directory is
    still added to PATH when it happens to exist (see _env_with_runtimes)."""
    return bool(HD_ENABLED and HD_PY and DENO_BIN and POT_SERVER_DIR)


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
                **_detach_kwargs(),                          # detach; survive our process
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
    # the probe shares _cookie_args() with the downloader — it used to hardcode
    # --cookies-from-browser, so on Windows it died on the locked profile and returned 0, and
    # discovery then rated every 1080p upload as SD before a single byte was fetched.
    def _probe_cmd() -> list:
        return [
            HD_PY, "-m", "yt_dlp", *_cookie_args(),
            "--extractor-args", f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{POT_PORT}",
            "-f", f"bv*[height<={max_height}]/b[height<={max_height}]/bv*/b",
            "-S", _format_sort(max_height),
            "--no-warnings", "--quiet", "--skip-download", "--print", "%(height)s", url,
        ]
    try:
        p = subprocess.run(_probe_cmd(), env=_env_with_runtimes(), capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode != 0 and _classify_dl_err(p.stderr or "") == "cookies":
            _disable_cookies(p.stderr or "")
            p = subprocess.run(_probe_cmd(), env=_env_with_runtimes(), capture_output=True,
                               text=True, timeout=timeout)
        for line in (p.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    return 0


def _exe_suffix() -> str:
    """'.exe' on Windows, '' elsewhere. yt-dlp joins the ffmpeg_location directory with the LITERAL
    names 'ffmpeg'/'ffprobe' and appends '.exe' on Windows, so every name we synthesise for it must
    carry the platform suffix."""
    return ".exe" if platform.system() == "Windows" else ""


def _merge_ffmpeg_dir(hint: str = "") -> str:
    """A directory that contains an executable literally named `ffmpeg` (`ffmpeg.exe` on Windows) —
    required for yt-dlp to MERGE the separate DASH video+audio streams into one file.

    CRITICAL: imageio-ffmpeg ships its binary under a VERSIONED name
    (`ffmpeg-macos-aarch64-v7.1`, `ffmpeg-win-x86_64-v7.1.exe`), so `--ffmpeg-location <imageio
    dir>` matches NOTHING and yt-dlp silently SKIPS the merge → a video-only file with NO audio →
    ASR gets 0 words → dialogue-lock (the strongest scene signal) is dead.

    The engine already solves this correctly for BOTH platforms in `ffmpeg_tool.ytdlp_ffmpeg_dir()`
    (suffix-aware names, symlink with a copy fallback for Windows' privilege rules, and the PATH
    prepend yt-dlp's FFmpegFD.available() needs). This used to be a second, macOS-only
    implementation of the same idea — it hardcoded the bare name `ffmpeg` and POSIX install dirs,
    so on Windows it returned a directory with no `ffmpeg.exe` in it and every HD download arrived
    mute. Delegate instead of diverging."""
    import shutil
    _x = _exe_suffix()
    # 1) the hint dir already has a correctly-named ffmpeg?
    if hint and os.path.exists(os.path.join(hint, f"ffmpeg{_x}")):
        return hint
    # 2) the engine's shared, platform-correct resolver (also fixes ffprobe + PATH)
    try:
        from ..ffmpeg_tool import ytdlp_ffmpeg_dir as _yfd
        d = _yfd()
        if d and os.path.exists(os.path.join(str(d), f"ffmpeg{_x}")):
            return str(d)
    except Exception:                                       # noqa: BLE001 — fall through
        pass
    # 3) a system ffmpeg on PATH or in a common install dir (POSIX paths are skipped on Windows
    #    because they cannot exist there; `which` already covers the Windows install locations)
    cands = [shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg",
             os.path.expanduser("~/pinokio/bin/miniconda/bin/ffmpeg")]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.dirname(c)
    # 4) last resort: link/copy the engine's (versioned) binary under the exact name in a cache dir
    try:
        from .config import ffmpeg_exe as _fexe
        real = _fexe()
        if real and os.path.exists(real):
            cache = os.path.join(os.path.expanduser("~"), ".cache", "clipstudio_ffmpeg")
            os.makedirs(cache, exist_ok=True)
            link = os.path.join(cache, f"ffmpeg{_x}")
            if not os.path.exists(link):
                try:
                    os.symlink(real, link)
                except FileExistsError:
                    pass
                except OSError:                             # Windows without Developer Mode
                    shutil.copy2(real, link)
                    try:
                        os.chmod(link, 0o755)
                    except OSError:
                        pass
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

    def _base_cmd(client_args: str = "") -> list:
        # PERMANENT 403-resistance layer (each element maintainer-documented — see the
        # yt-dlp FAQ / Extractors wiki / PO-Token-Guide):
        #  -4                : googlevideo format URLs are BOUND to the requesting IP; macOS
        #                      rotating IPv6 privacy addresses make extraction and download
        #                      egress differ -> guaranteed 403. Pin the family.
        #  --sleep-requests  : the official heavy-use pacing (`-t sleep` preset value).
        #  --retry-sleep     : exponential backoff on http errors instead of hammering.
        c = [
            HD_PY, "-m", "yt_dlp", "-4",
            "--sleep-requests", "0.75",
            "--retry-sleep", "http:exp=1:120",
            "--extractor-args", f"youtubepot-bgutilhttp:base_url=http://127.0.0.1:{POT_PORT}",
            "-f", _format_selector(max_height),
            "-S", _format_sort(max_height),
            "--merge-output-format", "mp4",
            "--no-playlist", "--no-warnings", "--quiet", "--no-progress",
            "--retries", str(retries), "--fragment-retries", str(max(retries, 5)),
            "--write-info-json",
            "-o", out_tmpl,
        ]
        c += _cookie_args()
        if client_args:
            c += ["--extractor-args", f"youtube:player_client={client_args}"]
        return c

    # CLIENT LADDER on 403 (PO-Token-Guide): default clients first; then mweb (the guide's
    # recommended POT pairing); then android_vr/web_embedded, which need NO PO token at all —
    # the escape hatch when the token path itself is what YouTube is rejecting.
    _LADDER = ["", "mweb", "android_vr,web_embedded"]
    cmd = _base_cmd()
    # ALWAYS give yt-dlp a properly-named ffmpeg so it MERGES video+audio (else ASR sees a silent
    # video-only file and dialogue-lock dies). _merge_ffmpeg_dir guarantees a usable `ffmpeg`.
    mdir = _merge_ffmpeg_dir(ffmpeg_dir)

    def _cmd_for(rung: int) -> list:
        client = _LADDER[min(rung, len(_LADDER) - 1)]
        c = _base_cmd(client)
        if mdir:
            c += ["--ffmpeg-location", mdir]
        c.append(url)
        return c

    env = _env_with_runtimes()
    last = ""
    last_class = "other"
    ok = False
    # The client rung is tracked SEPARATELY from the attempt counter: the ladder exists to answer
    # 403s, and a retry forced by our own broken flag (unreadable cookies) must not consume a rung
    # that a real throttle will need.
    rung = 0
    for attempt in range(1, retries + 2):
        try:
            cmd = _cmd_for(rung)
            _pace_hd_start()   # global inter-start gap: stay far under the ~300 videos/hour
                               # guest ceiling the Extractors wiki documents
            with _HD_SEM:      # pace ACTIVE HD fetches — parallel bursts trip the 403 throttle
                p = subprocess.run(cmd, cwd=os.path.dirname(out_stem) or ".", env=env,
                                   capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0:
                ok = True
                break
            # keep enough stderr to actually diagnose (the old 300-char cut lost the 403 context)
            last = (p.stderr or p.stdout or "").strip()[-2000:]
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
        except Exception as e:                               # noqa: BLE001
            last = str(e)[:300]
        last_class = _classify_dl_err(last)
        if last_class == "cookies":
            # NOT the video's fault — yt-dlp could not read the browser profile and aborted before
            # it ever reached YouTube. Cookies are optional, so drop them and retry the SAME rung.
            # Windows is where this happens (Chrome locks its cookie DB; App-Bound Encryption since
            # v127) and it took a whole render to 360p, because every retry re-sent the dead flag
            # and the fallback then read a tooling failure as "no HD available".
            # only the call that actually FLIPS the switch earns a free retry; if another thread
            # already turned cookies off, the next attempt is cookie-free regardless, so let the
            # rung advance normally rather than spending every attempt on one rung.
            if _disable_cookies(last, progress=progress) and attempt <= retries:
                continue                           # rung unchanged — retry without cookies
        rung = min(rung + 1, len(_LADDER) - 1)
        if last_class == "unavailable":
            # permanent (private/removed/sign-in) — retrying is pure waste
            break
        if last_class == "throttle_403":
            # transient: stale PO token or burst throttle. Fresh tokens once, then long backoff —
            # the whole point is that a 403 must NOT quietly become a 360p render.
            if not _POT_403_RESTARTED:
                _restart_pot_server(progress=progress)
            if attempt <= retries:
                back = min(15 * attempt, 45)
                if progress:
                    progress(f"hd: 403 (throttle/token) — waiting {back}s before retry {attempt + 1}")
                time.sleep(back)
            continue
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
            # the [class] tag is machine-read by download.py's end-of-stage 403 sweep — keep it
            _tag = {"throttle_403": "[403]", "unavailable": "[unavailable]"}.get(last_class, "")
            _line = next((ln for ln in reversed(last.splitlines()) if "error" in ln.lower()),
                         last[-200:] if last else "")
            progress(f"hd: no HD file {_tag}({_line[:160]}) — fallback")
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
