"""Locate an ffmpeg binary. Resolution is PLATFORM-AWARE.

Windows (RTX acceleration pass, 2026-06-05):
  1. VIDLORE_FFMPEG       — explicit path override (testing / custom build)
  2. bundled NVENC build  — <dist_root>/ffmpeg/bin/ffmpeg.exe, a static
                            win64-gpl ffmpeg that ships h264_nvenc so the
                            NVIDIA GPU can hardware-encode. Kept OUTSIDE the
                            shared `vidlore/` package so the Mac and Windows
                            runtime packages stay byte-identical and the
                            ~200 MB Windows-only encoder never bloats the Mac
                            dist. The encoder PROBE in assemble.py still
                            decides whether nvenc actually works on this box;
                            this only makes an nvenc-capable binary available.
  3. system ffmpeg on PATH (allowed)
  4. bundled imageio-ffmpeg binary (CPU-only safe fallback)

macOS / Linux — UNCHANGED behaviour (env override → system PATH → imageio,
which is videotoolbox-capable on macOS). The bundled-NVENC lookup is skipped
entirely off Windows, so this file is a no-op change for Mac renders.

ffprobe is intentionally not required anywhere in the pipeline.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


def _bundled_windows_ffmpeg() -> str | None:
    """Path to a bundled NVENC-capable Windows ffmpeg.exe if one is shipped,
    else None. Searched ONLY on Windows so macOS / Linux resolution is
    byte-for-byte unchanged."""
    if platform.system() != "Windows":
        return None
    pkg_dir = Path(__file__).resolve().parent          # <dist_root>/vidlore
    dist_root = pkg_dir.parent                          # <dist_root>
    candidates = (
        dist_root / "ffmpeg" / "bin" / "ffmpeg.exe",    # primary documented location
        dist_root / "ffmpeg" / "ffmpeg.exe",
        pkg_dir / "bin" / "windows" / "ffmpeg.exe",     # alt in-package location
    )
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def _resolve_ffmpeg() -> tuple[str, str]:
    """(path, source) for the ffmpeg we will use. `source` is one of
    env | bundled_nvenc | system | imageio and is surfaced in logs/diagnostics
    so the encoder selection is never a silent mystery."""
    override = os.environ.get("VIDLORE_FFMPEG", "").strip()
    if override and Path(override).is_file():
        return override, "env"
    bundled = _bundled_windows_ffmpeg()
    if bundled:
        return bundled, "bundled_nvenc"
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff, "system"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe(), "imageio"


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    return _resolve_ffmpeg()[0]


@lru_cache(maxsize=1)
def ytdlp_ffmpeg_dir() -> str | None:
    """A directory holding ffmpeg/ffprobe under the EXACT names yt-dlp looks for, or None.

    Hand this to yt-dlp as ``ffmpeg_location`` — NEVER ``Path(ffmpeg_exe()).parent``.

    yt-dlp's FFmpegPostProcessor._determine_executables() joins the directory it is given with the
    LITERAL names 'ffmpeg'/'ffprobe' (Windows appends '.exe'). Our resolved binary is usually the
    imageio one, whose filename is VERSIONED — 'ffmpeg-win-x86_64-v7.1.exe' on Windows,
    'ffmpeg-macos-aarch64-v7.1' on macOS — so passing its own parent directory matches NOTHING:
    yt-dlp then reports the merger as unavailable and every 'bestvideo+bestaudio' download aborts
    ("You have requested merging of multiple formats but ffmpeg is not installed"), i.e. no footage
    at all. On a dev Mac this is masked whenever a real `ffmpeg` happens to be on PATH.

    So expose a small cached dir of correctly-named links (copies where symlinks are unavailable —
    Windows needs Developer Mode/admin for symlinks) pointing at the resolved binary.
    """
    try:
        import tempfile
        exe = Path(ffmpeg_exe())
        if not exe.exists():
            return None
        _win = platform.system() == "Windows"
        suffix = ".exe" if _win else ""
        d = Path(tempfile.gettempdir()) / "vidlore_ffbin"
        d.mkdir(parents=True, exist_ok=True)
        for nm in ("ffmpeg", "ffprobe"):
            link = d / f"{nm}{suffix}"
            if link.exists():
                continue
            try:
                link.symlink_to(exe)
            except Exception:                              # noqa: BLE001 — Windows / no privilege
                shutil.copy2(exe, link)
                try:
                    os.chmod(link, 0o755)
                except Exception:                          # noqa: BLE001
                    pass
        # yt-dlp's FFmpegFD.available() is a classmethod that scans PATH only (it ignores
        # ffmpeg_location), and partial/range downloads gate on it — so put this dir on PATH too.
        _ds = str(d)
        _pth = os.environ.get("PATH", "")
        if _ds not in _pth.split(os.pathsep):
            os.environ["PATH"] = _ds + os.pathsep + _pth
        return _ds
    except Exception:                                      # noqa: BLE001
        return None


def ffmpeg_source() -> str:
    """Where the active ffmpeg came from: env|bundled_nvenc|system|imageio."""
    return _resolve_ffmpeg()[1]


# ── Bump RLIMIT_NOFILE so big filter graphs don't hit "Too many open files"
# (USER-REPORTED BUG 2026-05-26: a 23-min doc died at 82% assembly with
#  "Error parsing global options: Too many open files" because each
#  movie='...' source in the final-mux filter graph opens its own fd, and
#  macOS' default soft limit is 256 — easily exceeded by a long video's
#  caption strips + drawtext boxes + motion graphics + overlay layers.)
# Raises the SOFT limit to the hard cap once at import time; idempotent.
def _bump_nofile() -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try the hard cap first, falling back through common safe values.
        for target in (hard, 65536, 16384, 4096):
            if target <= soft:
                return
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
                return
            except (ValueError, OSError):
                continue
    except Exception:                                          # noqa: BLE001
        pass


_bump_nofile()


def run(args: list[str], *, quiet: bool = True, cwd: str | None = None,
        timeout: float | None = None) -> None:
    """Run ffmpeg with the given args (without the leading 'ffmpeg').

    On failure, dump the FULL command (especially the often-huge
    -filter_complex value) to `<cwd>/last_ffmpeg_fail.txt` so we can
    diagnose syntax bugs after the fact — the RuntimeError message itself
    has to stay short because the user sees it in the web UI.

    `timeout` (seconds) guards against a pathological filter graph that
    spins forever without emitting frames (e.g. `fps` on a non-advancing
    looped still): the child is killed and a RuntimeError is raised so the
    caller's graceful-degradation fallback can take over instead of hanging
    the whole render."""
    cmd = [ffmpeg_exe(), "-y", "-hide_banner"]
    if quiet:
        cmd += ["-loglevel", "error"]
    cmd += args
    # ── PROFILING HOOK (VIDLORE_PROFILE=1) — appends per-call timing
    # to /tmp/vidlore_ffmpeg_profile.log so we can attribute wall-clock
    # to specific ffmpeg invocations.  Zero cost when env var unset.
    import os as _os
    _prof = _os.environ.get("VIDLORE_PROFILE") == "1"
    _t0 = None
    if _prof:
        import time as _time
        _t0 = _time.time()
    # ── SPAWN-FAILURE RETRY (encode-pool reliability, 2026-05-31).
    # Under a multi-worker encode pool (default 4), many ffmpeg children
    # plus their fds can exhaust the per-process resource caps, so the OS
    # rejects the *spawn itself* — `fork`/`posix_spawn` raises OSError
    # EAGAIN (Errno 35 / BlockingIOError), EMFILE (24) or ENFILE (23)
    # BEFORE ffmpeg ever runs. That is transient: other workers' children
    # reap within milliseconds. Retrying the spawn with a short backoff
    # turns a fatal, load-dependent crash into a momentary pause. A
    # non-zero ffmpeg *exit* is deterministic and is NOT retried here (the
    # caller's own fallback ladder handles that). TimeoutExpired keeps its
    # existing fast-fail → RuntimeError behaviour.
    import time as _t
    _TRANSIENT = {35, 11, 24, 23}        # EAGAIN, EWOULDBLOCK, EMFILE, ENFILE
    _spawn_tries = 0
    while True:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=cwd, timeout=timeout)
            break
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "ffmpeg timed out after %ss: %s ..."
                % (timeout, " ".join(cmd[:6]))
            )
        except OSError as _e:                              # spawn rejected
            if getattr(_e, "errno", None) in _TRANSIENT and _spawn_tries < 6:
                _t.sleep(0.25 * (2 ** _spawn_tries))       # 0.25→8s backoff
                _spawn_tries += 1
                continue
            raise
    if _prof and _t0 is not None:
        import time as _time
        _dt = _time.time() - _t0
        try:
            # Tag = the first non-flag positional arg so we can group calls
            tag = "ffmpeg"
            for i, a in enumerate(args):
                if a in ("-i", "-filter_complex", "-f"):
                    nxt = args[i+1] if i+1 < len(args) else ""
                    tag = f"{a}={nxt[:50]}"
                    break
            with open("/tmp/vidlore_ffmpeg_profile.log", "a") as _fp:
                _fp.write(f"{_dt:7.2f}s  rc={proc.returncode}  "
                          f"argc={len(args):3d}  {tag}\n")
        except Exception:                                  # noqa: BLE001
            pass
    if proc.returncode != 0:
        # Persist the failing cmd + stderr for offline inspection.
        try:
            import os
            from pathlib import Path
            dump_dir = Path(cwd) if cwd else Path.cwd()
            dump = dump_dir / "last_ffmpeg_fail.txt"
            lines = ["# FAILED FFMPEG COMMAND", ""]
            for i, a in enumerate(cmd):
                lines.append(f"arg[{i}]: {a}")
            lines += ["", "# STDERR (last 4000 chars):", proc.stderr[-4000:]]
            dump.write_text("\n".join(lines), encoding="utf-8")
        except Exception:                                  # noqa: BLE001
            pass
        raise RuntimeError(
            "ffmpeg failed:\n  cmd: %s\n  stderr: %s"
            % (" ".join(cmd[:6]) + " ...", proc.stderr[-2000:])
        )
