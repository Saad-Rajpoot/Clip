"""Step 6: assembly. Per-scene visual render (theme-graded, with motion)
-> concat -> mux narration + optional music + burned captions -> final MP4.
Mirrors Vidlore's cloud render stage; here it runs locally via ffmpeg.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import textwrap
import unicodedata as _u
from functools import lru_cache
from pathlib import Path

from .captions import write_ass
from .ffmpeg_tool import ffmpeg_exe, run
from .footage import FootageItem
from .scene_lineage_canary import (
    SceneLineageError,
    bind_encode_plan as _bind_scene_lineage,
    fail_audit as _fail_scene_lineage_audit,
    new_audit as _new_scene_lineage_audit,
    verify_encoded_plan as _verify_lineage_encoded_plan,
    verify_delivered_output as _verify_lineage_delivered_output,
    verify_timeline_order as _verify_lineage_timeline_order,
    write_audit as _write_scene_lineage_audit,
)


def _probe_encoder(name: str) -> bool:
    """True if this ffmpeg can actually encode with `name` right now (a
    tiny throwaway encode). This is how we stay safe cross-platform: a
    box without that GPU simply fails the probe and we move on."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1",
             "-c:v", name, "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:                      # noqa: BLE001
        return False


def _log_encoder_choice(selected: str, reason: str,
                        fallback: str | None = None) -> None:
    """Emit ONE clear line so the encoder path is never a silent mystery
    (RTX acceleration pass). Printed once because the caller is lru_cache'd.
    Format the Windows GPU diagnostics + report parse:
        [encoder] platform=windows selected=h264_nvenc reason=probe_pass ...
        [encoder] platform=windows selected=libx264 reason=nvenc_probe_failed fallback=safe_cpu ..."""
    try:
        from .ffmpeg_tool import ffmpeg_source
        src = ffmpeg_source()
    except Exception:                                          # noqa: BLE001
        src = "unknown"
    plat = (platform.system() or "unknown").lower()
    msg = (f"  [encoder] platform={plat} selected={selected} "
           f"reason={reason} ffmpeg={src}")
    if fallback:
        msg += f" fallback={fallback}"
    print(msg, flush=True)


@lru_cache(maxsize=1)
def _pick_video_encoder() -> str:
    """Choose the fastest WORKING H.264 encoder for this machine. Pure
    software libx264 on a long video is the multi-HOUR bottleneck; a
    hardware encoder does the same job many times faster. Cross-platform:
      macOS  -> h264_videotoolbox   (Apple Silicon / Intel VT)
      Windows-> h264_nvenc (NVIDIA) | h264_qsv (Intel) | h264_amf (AMD)
      Linux  -> h264_nvenc | h264_qsv
    Falls back to libx264 anywhere a probe fails (never breaks a render).
    EVERY path logs its selection + reason (never a silent downgrade).
    Override with VIDLORE_VENC = x264 | videotoolbox | nvenc | qsv | amf."""
    env = os.environ.get("VIDLORE_VENC", "auto").strip().lower()
    if env in ("x264", "libx264", "software", "sw"):
        _log_encoder_choice("libx264", "forced_env_x264")
        return "libx264"
    forced = {"vt": "h264_videotoolbox", "videotoolbox": "h264_videotoolbox",
              "nvenc": "h264_nvenc", "qsv": "h264_qsv", "amf": "h264_amf"}
    if env in forced:
        sel = forced[env]
        # A FORCED hw encoder is still probed so a bad/unsupported request can
        # never break the render — probe fail → loud log + safe CPU.
        if _probe_encoder(sel):
            _log_encoder_choice(sel, "forced_env_probe_pass")
            return sel
        _log_encoder_choice("libx264", f"forced_{env}_probe_failed",
                            fallback="safe_cpu")
        return "libx264"
    sysname = platform.system()
    if sysname == "Darwin":
        cands = ["h264_videotoolbox"]
    elif sysname == "Windows":
        cands = ["h264_nvenc", "h264_qsv", "h264_amf"]
    else:                                  # Linux / other
        cands = ["h264_nvenc", "h264_qsv"]
    for c in cands:
        if _probe_encoder(c):
            _log_encoder_choice(c, "probe_pass")
            return c
    # No hardware encoder worked → safe software path, ALWAYS logged.
    primary = cands[0] if cands else "none"
    _log_encoder_choice("libx264", f"{primary}_probe_failed", fallback="safe_cpu")
    return "libx264"


@lru_cache(maxsize=1)
def _ffthreads() -> list[str]:
    """ffmpeg parallelism flags — multi-thread the filter graph and
    non-complex filters across all cores.  ffmpeg defaults the filter
    pipeline to 1 thread, which is the actual bottleneck on documentary
    renders (the giant filter_complex with 25+ drawtexts and 40+ overlay
    layers is filter-bound, NOT encode-bound — hw videotoolbox encodes
    at 200 fps but a single-threaded filter graph can starve it down to
    5-10 fps).  Setting both knobs to the physical core count gives a
    3-5× speedup on long filter chains with zero quality cost (output is
    byte-identical, just faster).  Override with VIDLORE_FF_THREADS=N."""
    try:
        env = os.environ.get("VIDLORE_FF_THREADS", "").strip()
        if env and env.isdigit() and int(env) > 0:
            n = int(env)
        else:
            n = os.cpu_count() or 8
            # leave 1-2 cores for the OS / GUI so the machine stays usable
            n = max(2, n - 2)
    except Exception:                                          # noqa: BLE001
        n = 8
    s = str(n)
    return ["-filter_threads", s, "-filter_complex_threads", s,
            "-threads", "0"]   # threads=0 lets ffmpeg auto-pick for codecs


def _venc(crf: str = "20") -> list[str]:
    """Video-encoder args for the chosen encoder. Visual target is the
    same (~12 Mbps 1080p / crf 20); only the speed differs. libx264
    path is byte-for-byte the original behaviour."""
    enc = _pick_video_encoder()
    if enc == "h264_videotoolbox":
        return ["-c:v", enc, "-b:v", "12M", "-maxrate", "16M",
                "-bufsize", "24M", "-allow_sw", "1", "-pix_fmt", "yuv420p"]
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", "p4", "-rc", "vbr",
                "-cq", "21", "-b:v", "12M", "-maxrate", "18M",
                "-pix_fmt", "yuv420p"]
    if enc == "h264_qsv":
        return ["-c:v", enc, "-global_quality", "21",
                "-preset", "faster", "-pix_fmt", "yuv420p"]
    if enc == "h264_amf":
        return ["-c:v", enc, "-quality", "balanced", "-rc", "cqp",
                "-qp_i", "22", "-qp_p", "22", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast",
            "-crf", crf, "-pix_fmt", "yuv420p"]
from .music import (
    build_archival_bed, build_atmosphere_bed, build_number_sfx, build_sfx_bed,
    build_typewriter_sfx,
)
from . import sfx as _sfxlib
from . import motion as _mo
from .tts import Narration, WordTiming

# MULTILINGUAL: a glyph-complete font FIRST so on-screen text renders in
# ANY of the supported languages (Latin accents + Japanese + Korean +
# Chinese) — Arial/DejaVu have no CJK glyphs, so JA/KO would otherwise
# draw as tofu boxes even though the string now survives. "Arial
# Unicode" (macOS) covers Latin+CJK in one face and looks like Arial for
# English/European, so it's safe as the universal default; Noto CJK is
# the Linux fallback. Plain Arial/DejaVu remain last (Latin-only hosts).
# Bundled font shipped INSIDE the package (vidlore/assets/). This is the
# real cross-platform fix: text/graphics no longer depend on whatever
# fonts a customer's Windows happens to have — Mac, Windows and Linux all
# use the SAME font file, so the output is 100% identical everywhere.
# (A missing system font silently skipped EVERY overlay on Windows ->
# "no text / no graphics".) Noto Sans is OFL-licensed (free to ship).
_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_BUNDLED_FONT = str(_ASSET_DIR / "VidloreSans-Bold.ttf")
_BUNDLED_FONT_R = str(_ASSET_DIR / "VidloreSans.ttf")

_FONT_CANDIDATES = (
    _BUNDLED_FONT,
    _BUNDLED_FONT_R,
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    # WINDOWS (so it runs on Windows too — code is cross-platform; only
    # the font paths were *nix-only). ARIALUNI = true universal; the
    # CJK faces (Yu Gothic / Malgun / YaHei) cover JA/KO/ZH so the
    # multilingual feature works on Windows as well; Arial = Latin.
    "C:/Windows/Fonts/ARIALUNI.TTF",
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)

FPS = 30
XFADE = 0.4   # standard dissolve (s)
_CUT = 0.06   # near-hard "punch" cut on an impact/reveal
_DISS = 0.78  # slow reflective dissolve

# DOCUMENTARY TRANSITION VOCABULARY. Each motivated transition maps to a
# real ffmpeg xfade transition + a base duration. Clean hard CUT stays the
# default (not in this table) — these fire ONLY on a motivated boundary
# (emotion/location/timeline/evidence/act/map-flow), chosen by
# `_edit_plan`. All are pairwise (never chained -> no freeze). Subtle, not
# flashy: durations and picks are tuned to the MagnatesMedia/Johnny-Harris
# register, never TikTok.
class TimelineSyncError(RuntimeError):
    """The video clock and the composed-audio clock disagree by more than a frame.

    Not cosmetic. These are two independently-computed timelines (video = the sum of ENCODED
    segment durations; audio = the narration with breakouts spliced in at sample offsets), and
    nothing reconciled them. When they drifted, every breakout's picture landed late by the drift —
    measured at 1.13s and 1.18s, with the concat overrunning the audio by 1.200s end to end.

    Deliberately NOT recoverable by shifting a constant: the drift is variable, so 'subtract 1.1s'
    would be a second bug on top of the first. If this fires, the pairing/padding accounting is
    wrong and must be fixed at the source."""


def _probe_duration(p) -> float:
    """Container duration in seconds. Raises TimelineSyncError rather than returning a sentinel:
    an unreadable duration is a FAILED CHECK, not a passed one."""
    import subprocess as _sp
    from .clipstudio.config import ffprobe_exe as _ffprobe_exe
    try:
        out = _sp.check_output([str(_ffprobe_exe()), "-v", "error", "-show_entries",
                                "format=duration", "-of", "csv=p=0", str(p)], stderr=_sp.DEVNULL)
        d = float(out.decode().strip())
    except Exception as e:
        raise TimelineSyncError(f"cannot read duration of {p!r} ({type(e).__name__}: {e}) — the "
                                f"sync invariant cannot be checked, so it is not satisfied") from e
    if not (d > 0):
        raise TimelineSyncError(f"duration of {p!r} reads {d!r} — unusable; the sync invariant "
                                f"cannot be checked, so it is not satisfied")
    return d


def _audio_frame_anomalies(p) -> dict:
    """Does the delivered audio track tell the truth about its own timeline?

    The duration check below reads the container's declared `stream=duration`, which is exactly the
    number a broken timeline inflates. Job 957f56f925 shipped 106.688s of actual audio under a
    106.800s video — 124 ms truncated — and passed `delivered A/V sync OK` because ONE AAC frame
    declared 5488 samples instead of 1024, and that 93 ms phantom gap padded the declared duration
    back inside the 33 ms tolerance. The two defects hid each other. A macOS render carries the same
    class at 47 ms, so this is not a Windows quirk; it is simply invisible to a duration check.

    Every AAC frame holds a fixed number of samples, so ANY non-terminal frame whose declared
    duration differs from the modal one is a lie about the timeline. (The last frame is exempt: a
    short tail is normal padding.) Returns the evidence; the caller decides severity."""
    import collections as _c
    import json as _json
    import subprocess as _sp
    from .clipstudio.config import ffprobe_exe as _ffprobe_exe
    try:
        raw = _sp.check_output(
            [str(_ffprobe_exe()), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "packet=duration,pts_time", "-of", "json", str(p)],
            stderr=_sp.DEVNULL)
        pkts = _json.loads(raw.decode()).get("packets") or []
    except Exception:
        return {}                       # unreadable → say nothing; the duration gate still runs
    durs = []
    for k in pkts:
        try:
            durs.append((int(k.get("duration")), float(k.get("pts_time") or 0.0)))
        except (TypeError, ValueError):
            continue
    if len(durs) < 8:
        return {}
    modal = _c.Counter(d for d, _ in durs).most_common(1)[0][0]
    if modal <= 0:
        return {}
    odd = [(t, d) for i, (d, t) in enumerate(durs)
           if d != modal and i < len(durs) - 1]
    sr = 48000.0
    try:
        raw2 = _sp.check_output(
            [str(_ffprobe_exe()), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "json", str(p)], stderr=_sp.DEVNULL)
        sr = float((_json.loads(raw2.decode()).get("streams") or [{}])[0].get("sample_rate") or sr)
    except Exception:
        pass
    gap_s = sum(d - modal for _, d in odd) / sr
    return {"frames": len(durs), "modal_samples": modal, "anomalies": len(odd),
            "gap_s": round(gap_s, 4),
            "worst": ([{"at_s": round(t, 3), "samples": d,
                        "gap_ms": round((d - modal) / sr * 1000.0, 1)}
                       for t, d in sorted(odd, key=lambda x: -x[1])[:3]] if odd else []),
            "true_media_s": round(sum(d for d, _ in durs) / sr, 4)}


def _stream_times(p) -> dict:
    """{stream_type: (duration, first_pts, last_pts)} for the delivered file. Raises on anything
    unreadable — same reason as above."""
    import json as _json
    import subprocess as _sp
    from .clipstudio.config import ffprobe_exe as _ffprobe_exe
    out: dict = {}
    for kind in ("v:0", "a:0"):
        try:
            raw = _sp.check_output(
                [str(_ffprobe_exe()), "-v", "error", "-select_streams", kind,
                 "-show_entries", "stream=codec_type,duration,start_time",
                 "-of", "json", str(p)], stderr=_sp.DEVNULL)
            d = _json.loads(raw.decode()).get("streams") or []
        except Exception as e:
            raise TimelineSyncError(
                f"cannot probe {kind} of {p!r} ({type(e).__name__}: {e})") from e
        if not d:
            raise TimelineSyncError(f"{p!r} has no {kind} stream — a delivered render must carry "
                                    f"both a video and an audio stream")
        s = d[0]
        try:
            dur = float(s.get("duration"))
            st = float(s.get("start_time") or 0.0)
        except (TypeError, ValueError) as e:
            raise TimelineSyncError(f"{kind} of {p!r} has an unreadable duration/start_time "
                                    f"({s!r})") from e
        out[s.get("codec_type") or kind] = (dur, st, st + dur)
    return out


def _conform_video_to_audio(video_only, narration, workdir, *, max_fix_frames: int = 24) -> None:
    """Trim/pad the CONCATENATED video to the composed-audio length, at the file level.

    The video timeline is assembled from many per-scene frame counts plus fixed-length breakout
    mp4s inserted verbatim, and those accounting layers do not perfectly reconcile with the composed
    narration — a small, STABLE residual survives (measured +4 frames / +0.14s on the acceptance
    render, unmoved by the pad, carry, renderer-clock and beat-level conform fixes because it enters
    at the breakout-insertion boundary, not the beats). Rather than chase every accounting layer,
    conform the finished artifact to the authority the invariant checks: measure the concat, and if
    it is a few frames off the composed audio, retime it to match with a single tpad/trim.

    Bounded by max_fix_frames: this corrects sub-perceptible accumulation, never a gross error (a
    large gap means something is genuinely wrong and must reach the invariant, not be papered over).
    The bound scales with breakout count — each breakout insertion contributes a couple of frames of
    per-boundary rounding (video mp4 frame-count vs the composed-audio splice grid), so a video with
    5 breakouts accumulates ~13 frames (measured on a canary-trap render: +0.443s, video LONG). 24
    frames (0.8s at 30fps) covers realistic multi-breakout accumulation while still failing a truly
    gross error (>0.8s = seconds-scale = a genuinely misplaced beat/breakout, which must reach the
    invariant). The conform trims/pads to the COMPOSED AUDIO (the authority captions key to), so it
    never desyncs captions; the trimmed frames are the outro tail."""
    aud = getattr(narration, "audio", None)
    if not aud:
        return                                         # the invariant will fail closed on this
    try:
        vd = _probe_duration(video_only)
        ad = _probe_duration(aud)
    except TimelineSyncError:
        return                                         # let the invariant report it
    delta_f = round((ad - vd) * FPS)
    if delta_f == 0 or abs(delta_f) > max_fix_frames:
        return                                         # already frame-exact, or too large to touch
    target_frames = int(round(ad * FPS))
    tmp = Path(video_only).with_suffix(".conform.mp4")
    if delta_f > 0:
        # video SHORT — hold the last frame for the deficit (tpad), then hard-cap the frame count
        args = ["-i", str(video_only), "-vf",
                f"tpad=stop_mode=clone:stop_duration={(delta_f + 1) / FPS:.4f}",
                "-frames:v", str(target_frames)]
    else:
        # video LONG — keep exactly target_frames frames, dropping the tail
        args = ["-i", str(video_only), "-frames:v", str(target_frames)]
    try:
        run([*_ffthreads(), *args, "-r", str(FPS), *_venc(),
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-color_range", "tv", str(tmp)], cwd=str(workdir), timeout=600)
    except Exception:                                  # noqa: BLE001 — let the invariant judge
        tmp.unlink(missing_ok=True)
        return
    if tmp.exists():
        tmp.replace(Path(video_only))
        print(f"  [5/5] video conform: {delta_f:+d} frame(s) at the concat tail so the delivered "
              f"video == composed audio ({target_frames} frames)", flush=True)


def _assert_video_audio_sync(video_only, narration, workdir, *, tol_frames: float = 1.0) -> None:
    """PRE-MUX: final_video_duration == composed_audio_duration, within one frame.

    Compared against the COMPOSED audio (narration WITH breakouts spliced in), never the raw
    pre-breakout narration: the composed track is the clock the captions and the breakout splices
    are all keyed to, so it is the only meaningful reference.

    FAILS CLOSED throughout. The first cut of this returned early when the narration was missing or
    ffprobe failed, on the theory that it shouldn't "invent a fault" — but a check that cannot run
    has not passed, and silently skipping it is exactly the shape of the bug this whole branch
    exists to fix (a verifier that errored looked like a verifier that approved)."""
    aud = getattr(narration, "audio", None)
    if not aud:
        raise TimelineSyncError(
            "no composed narration audio to check the video against — the sync invariant cannot be "
            "verified, so it is not satisfied")
    va = _probe_duration(video_only)
    ad = _probe_duration(aud)
    tol = tol_frames / float(FPS)
    if abs(va - ad) > tol:
        raise TimelineSyncError(
            f"video/audio timeline drift {va - ad:+.3f}s exceeds {tol:.3f}s (1 frame): "
            f"video={va:.3f}s composed-audio={ad:.3f}s. The transition padding and the pairing "
            f"plan disagree — every breakout picture will lag its own audio by this much.")


def assert_delivered_av_sync(out_path, *, tol_frames: float = 1.0) -> dict:
    """POST-MUX: the FINAL delivered file, after overlays, captions, bars and muxing.

    The pre-mux check proves the concat matches the composed audio. It cannot prove the thing the
    viewer actually receives: everything after it — overlay bake, caption burn, letterboxing, the
    mux itself — re-encodes and re-times, and any of those can reintroduce drift or truncate a
    stream. So the delivered artifact is measured on its own terms: per-stream duration AND
    first/last PTS, because two streams can share a duration while starting at different offsets
    (a non-zero audio start_time is silent lip-sync error).

    Returns the measured facts so the caller can log them as evidence."""
    st = _stream_times(out_path)
    if "video" not in st or "audio" not in st:
        raise TimelineSyncError(f"delivered file {out_path!r} is missing a stream: {sorted(st)}")
    (vd, v0, v1), (ad, a0, a1) = st["video"], st["audio"]
    tol = tol_frames / float(FPS)
    if abs(vd - ad) > tol:
        raise TimelineSyncError(
            f"DELIVERED a/v duration drift {vd - ad:+.3f}s > {tol:.3f}s (1 frame): "
            f"video={vd:.3f}s audio={ad:.3f}s in {out_path}")
    if abs(v0 - a0) > tol:
        raise TimelineSyncError(
            f"DELIVERED first-PTS skew {v0 - a0:+.3f}s > {tol:.3f}s: video starts {v0:.3f}s, "
            f"audio starts {a0:.3f}s in {out_path} — the streams do not begin together")
    if abs(v1 - a1) > tol:
        raise TimelineSyncError(
            f"DELIVERED last-PTS skew {v1 - a1:+.3f}s > {tol:.3f}s: video ends {v1:.3f}s, "
            f"audio ends {a1:.3f}s in {out_path} — one stream was truncated")
    res = {"video": [vd, v0, v1], "audio": [ad, a0, a1], "tol_s": tol}
    # …and the check the three above cannot make: is that audio duration even real? See
    # _audio_frame_anomalies. Reported ALWAYS (it is the evidence), fatal only once the invented
    # gap is long enough to be heard, because renders on both platforms have been carrying small
    # ones for months and failing a four-hour render over 3 ms would be the wrong trade.
    try:
        anom = _audio_frame_anomalies(out_path)
    except Exception:
        anom = {}
    if anom:
        res["audio_frames"] = anom
        if anom.get("anomalies"):
            _gap_ms = abs(anom.get("gap_s") or 0.0) * 1000.0
            _msg = (f"DELIVERED audio timeline is INVENTED: {anom['anomalies']} non-terminal AAC "
                    f"frame(s) declare a duration other than {anom['modal_samples']} samples, "
                    f"adding {_gap_ms:.0f}ms of timeline that holds no audio "
                    f"(worst: {anom.get('worst')}). The container's duration is inflated by exactly "
                    f"this much, which is how it slipped past the A/V duration check.")
            try:
                _fatal_ms = float(os.environ.get(
                    "VIDLORE_AUDIO_GAP_FATAL_MS", "64") or 64)
            except (TypeError, ValueError):
                _fatal_ms = 64.0
            if _gap_ms >= _fatal_ms:
                raise TimelineSyncError(f"{_msg} — over the {_fatal_ms:.0f}ms audible floor.")
            res["audio_frames"]["warning"] = _msg
    return res


_TRANSITIONS = {
    # reflective / emotional exhale
    "dissolve":      ("fade",       0.55),
    "slow_dissolve": ("fade",       0.85),
    "film_dissolve": ("fadewhite",  0.45),   # historical film bloom
    "fadeblack":     ("fadeblack",  0.60),   # act break / chapter
    # evidence / archive / investigation
    "archive_flash": ("fadewhite",  0.22),   # fast white flash-frame
    "evidence_flash":("fadegrays",  0.30),
    # documents
    "page_wipe":     ("wipeleft",   0.42),
    "page_wipe_r":   ("wiperight",  0.42),
    # geography / maps
    "geo_push":      ("smoothleft", 0.70),   # map continuation
    "geo_push_r":    ("smoothright", 0.70),
    # fast hook / action — directional, quick
    "whip":          ("slideleft",  0.28),
    "whip_r":        ("slideright", 0.28),
    "blur_cut":      ("hblur",      0.30),   # directional motion blur
    # surveillance / tech (style-gated)
    "glitch":        ("pixelize",   0.34),
    # neutral motivated
    "soft_dissolve": ("dissolve",   0.45),
    "iris":          ("circleopen", 0.55),   # rare reflective open
}


def _motion_layer(name: str, label: str, st: float, fin_d: float,
                  fout_st: float, fout_d: float, win: str,
                  *, rise: float = 0.0, ease: str = "out_cubic") -> tuple:
    """Emit the (movie-stage, overlay-stage) for a full-frame RGBA layer
    that enters with the shared cinematic MOTION LANGUAGE: a soft alpha
    dissolve plus, when `rise` > 0, a subtle eased upward settle (the
    element rises a few px INTO place and decelerates — the MagnatesMedia
    / Netflix "settle", not a slide gimmick). Position easing carries the
    physical feel; alpha stays a clean dissolve. `label` is the unique
    filtergraph pad name. Returns two filter strings to append to stages.
    """
    movie = (
        f"movie='{name}',format=rgba,loop=loop=-1:size=1,"
        f"setpts=N/{FPS}/TB,"
        f"fade=t=in:st={st:.2f}:d={fin_d:.2f}:alpha=1,"
        f"fade=t=out:st={fout_st:.2f}:d={fout_d:.2f}:alpha=1[{label}]"
    )
    if rise and rise > 0.5:
        # rise from +rise px to 0, eased, over a touch longer than the
        # alpha so the motion finishes settling just after it's visible.
        yexpr = _mo.interp_ff(rise, 0.0, st, max(0.25, fin_d * 1.35), ease)
        ov = (f"[{{CUR}}][{label}]overlay=x=0:y='{yexpr}':eval=frame:"
              f"enable='{win}'[{{OUT}}]")
    else:
        ov = (f"[{{CUR}}][{label}]overlay=x=0:y=0:enable='{win}'"
              f"[{{OUT}}]")
    return movie, ov


def _rng01(seed: int) -> float:
    """Deterministic pseudo-random in [0,1). A real editor's timing is
    irregular but INTENTIONAL — not a metronome and not truly random.
    This gives that human unevenness reproducibly, so the footage plan
    and the assembler always agree and segment caches stay valid across
    re-renders (no global random state, no per-run drift)."""
    x = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    x ^= (x >> 15)
    x = (x * 1274126177) & 0xFFFFFFFF
    x ^= (x >> 13)
    return ((x >> 8) & 0xFFFFF) / float(0x100000)
# Loudness-normalize the NARRATION ALONE to YouTube-broadcast level
# (~-16 LUFS). loudnorm on pure speech is reliable and doesn't pump;
# running it across voice+music (old behaviour) rode the gain and either
# buried the bed or slammed the whole mix into a flat wall.
_VOXNORM = "loudnorm=I=-16:TP=-2:LRA=11"
# Fixed bed levels UNDER the normalized voice (predictable ratio: the
# music is always clearly present but soft, never competing).
def _music_vol_mult() -> float:
    """Review-Editor music-volume knob. Default 1.0 → byte-identical to before
    (no behavior change unless the editor sets VIDLORE_MUSIC_VOLUME)."""
    import os as _os
    try:
        return max(0.0, min(2.0, float(_os.environ.get("VIDLORE_MUSIC_VOLUME", "1") or "1")))
    except (TypeError, ValueError):
        return 1.0


_MUSIC_VOL = 0.20
# PHASE 3.1 — DYNAMIC DUCKING base level. When the music bed is
# sidechain-compressed under the voice we can run it HOTTER at rest
# (0.30) because the compressor automatically pulls it down ~8-10 dB
# the instant narration starts and lets it swell back up in the
# pauses. Net effect = the score "breathes" with the speech (the
# pro-documentary feel) instead of sitting as one timid flat bed.
_MUSIC_VOL_DUCK = 0.16   # v13.2 (2026-05-28): user said music still
#                          competes with the voice on the 25-min Mossad
#                          render. Dropped 0.22 -> 0.16 (another ~2.7 dB
#                          quieter at rest) and strengthened the
#                          sidechain (see _DUCK) so words pull it down
#                          harder. Music = subconscious bed; voice
#                          dominates absolutely.
# sidechaincompress params, tuned for music-under-VO: a low threshold
# + firm ratio = clear ducking the moment the voice keys it; a slow-ish
# 340 ms release = the bed rises back smoothly in gaps (no audible
# pumping); 12 ms attack catches the word onset without clipping it. A
# bigger ratio compensates for the hotter rest level so the voice still
# wins decisively.
_DUCK = ("sidechaincompress=threshold=0.030:ratio=12:attack=8:"
         "release=300:makeup=1:detection=rms")
#        v13.2: lower threshold (0.040->0.030) + firmer ratio (9->12) +
#        faster attack (12->8 ms) = the voice keys the duck sooner and
#        pulls the music down harder, so narration is never masked.
# v14 (2026-06-01): SFX were measured inaudible in the final MP4 (0/7 audible,
# eff peaks -24..-47 dB under a -16 LUFS voice). sfx.py now applies a per-event
# TIER gain (major/mid/minor) so a single bed gain makes major reveals punch
# while ticks stay subtle; these bed gains were calibrated with tools/sfx_calib.py
# (MAJOR 3/3, MID 8/8, MINOR 5/5 audible/subtle, none harsh).
_SFX_VOL = 1.15
# Number/stat reveal accent — a clear, confident "stat stinger". It is
# SPARSE (only on figure cards, ~1-3 a video) so it can sit at a real,
# audible level without ever becoming the disliked repeating SFX.
_NUMSFX_VOL = 0.90
# Text-moment SFX — the cinematic support the viewer FEELS on a graphic
# reveal: a soft directional whoosh as the card flies in + a low impact
# hit when a shocking word/name/number/warning LANDS. Sits clearly above
# the old (off-by-default) per-cut whoosh so text reveals are noticeable,
# but still restrained (documentary, not trailer).
_GFX_VOL = 1.35   # v14 (2026-06-01): calibrated WITH the new per-event tier
#                   gain (sfx.py). 0.50 left every reveal buried (-24..-47 dB
#                   eff). At 1.35 the tiered synth puts MAJOR reveals at a
#                   clearly-audible transient (~-11..-15 dB, still under the
#                   voice peaks) and ticks subtle (~-28). The anti-whoosh
#                   cadence throttle + per-primitive cooldown still prevent spam.
# Typewriter click track — tactile key clicks SYNCED to each character +
# a soft ENTER accent. Present and mechanical, but firmly under the voice.
_TYPE_VOL = 0.80
# Archival room-tone (hiss+hum) — sits FAR under everything; just makes
# recovered-footage scenes feel real, never noticeable as an effect.
# v9 (2026-05-26): user feedback "music/archbed too loud, drowns
# voice" — was 2.0 (drowning voiceover), now 0.30. The build_archival_bed
# generator already produces low-level hiss; the 2.0 multiplier was
# leftover calibration from an older source that was at -30 dBFS.
_ARCH_VOL = 0.30
# Environmental ATMOSPHERE bed (HE#8) — subconscious world texture; a touch
# above the raw synth so it's *felt* under the voice, never *heard*.
# v9: same calibration issue — was 4.0 (way too prominent), now 0.45.
_ATMOS_VOL = 0.45
# Transparent true-peak ceiling on the final sum (no auto make-up gain,
# so it only catches peaks — it never re-introduces pumping/loudness).
_LIMIT = "alimiter=limit=0.85:level=0"   # v14.2: 0.89->0.85 (~-1.4 dBFS sample ->
#   ~-1.0/-1.1 dBTP true-peak after AAC). v14.1 (0.89) measured a biography master
#   at -0.9 dBTP — just hot of the conservative -1.0 dBTP ceiling; 0.85 guarantees
#   <= -1.0 dBTP with margin. level=0 (no make-up gain) so it only trims the top
#   ~0.4 dB of rare peaks — integrated loudness stays at the -16 LUFS doc target.
#   Forensic vs Vidlore AI: their peaks sit -1.2/-1.7 dBTP; this keeps us in band.

# ====================================================================== #
# RC5.1 — GLOBAL OVERLAY-RESTRAINT POLICY                                  #
# ---------------------------------------------------------------------- #
# Forensic finding: real archival / stock FOOTAGE was over-treated. The
# cumulative per-scene + final-video stack layered TWO vignettes, MULTIPLE
# grain passes, a texture pack (grid / blur / chroma-shift) AND a grade —
# so faces / midtones / detail were lost (muddy, "damaged-looking"). This
# policy is ONE bounded place that scales every overlay layer and caps the
# cumulative darkening, while PRESERVING the intended premium cinematic
# grade. It is RESTRAINT, not removal: defaults pull the heaviest layers
# down a notch and DE-STACK the duplicated vignette/grain — they do not
# flatten the look. Every knob is env-overridable but CLAMPED to a SAFE
# MAXIMUM, so even a pushed value can never make footage muddy.
#
# Knobs (all 0..1 fractions of each layer's *safe-max* intensity):
#   VIDLORE_OVERLAY_STRENGTH  — master multiplier over all four below
#   VIDLORE_GRAIN_STRENGTH    — film-grain (noise=alls) amount
#   VIDLORE_VIGNETTE_STRENGTH — corner darkening / focus falloff
#   VIDLORE_TEXTURE_STRENGTH  — grid / blur / dust / chroma texture packs
#   VIDLORE_DARKEN_STRENGTH   — how much cumulative shadow crush is allowed
import os as _os_or  # noqa: E402  (local alias, restraint policy only)


def _ovr_knob(name: str, default: float) -> float:
    """Read a 0..1 overlay-restraint knob from the environment, clamped to
    [0, 1]. Bad / missing values fall back to the conservative default so a
    stray env never makes the look muddy (or disappears)."""
    try:
        v = _os_or.environ.get(name)
        if v is None or not str(v).strip():
            return default
        return max(0.0, min(1.0, float(v)))
    except Exception:                                              # noqa: BLE001
        return default


class _OverlayRestraint:
    """Bounded, env-overridable strengths for every footage/still overlay
    layer + the cumulative clarity gate. Constructed once at import; reads
    the knobs then. SAFE MAXIMUMS are encoded as the per-layer ``*_MAX``
    constants — a knob of 1.0 maps to the max, and nothing exceeds it.
    The maxima are deliberately set at-or-below today's values so turning
    this on can only REDUCE treatment, never intensify it."""

    # --- SAFE MAXIMUMS (a knob=1.0 reaches exactly these; never beyond) --- #
    GRAIN_MAX     = 9      # noise=alls cap for footage finish (was 7 base;
    #                        9 = the heaviest pack value, now the hard ceiling)
    GRAIN_ARCH_MAX = 12    # archival grain ceiling (was 16 — recovered film,
    #                        but 16 read as damage; cap at a filmic 12)
    GRAIN_FINAL_MAX = 4    # whole-video finishing grain ceiling (unchanged)
    # vignette: ffmpeg vignette angle — LARGER angle = SOFTER/less crush. We
    # store the *softest* (least darkening) and *hardest* (most) and let the
    # knob interpolate, so vignette_strength=0 → barely any corner falloff.
    VIGN_SOFT_ANGLE = 7.2  # PI/7.2 — almost no corner crush (knob 0.0)
    VIGN_HARD_ANGLE = 5.0  # PI/5.0 — today's footage vignette (knob 1.0 cap)
    VIGN_ARCH_SOFT  = 4.6  # archival focal vignette softest (knob 0.0)
    VIGN_ARCH_HARD  = 4.0  # archival focal vignette hardest (was PI/3.8 —
    #                        too tight, ate edge subjects; cap at PI/4.0)
    TEXTURE_MAX     = 1.0  # multiplier on pack texture alpha/sigma (knob cap)
    # DARKEN: minimum allowed luma-lift on the grade. The finish grade lifts
    # shadows (gamma>1); darken_strength controls how much we let the stack
    # pull back DOWN toward crush. floor stays a sane mid value.
    DARKEN_FLOOR_GAMMA = 1.06   # never let cumulative grade gamma fall below
    #                             this on footage (keeps a luma floor / no
    #                             crushed blacks)

    def __init__(self) -> None:
        # PHASE-1 CLARITY PASS (2026-06-05): the global overlay stack was too
        # heavy — footage read dark/muddy with crushed blacks, strong vignette
        # and visible grain (vs the brighter, clearer competitor). Defaults
        # softened so REAL FOOTAGE keeps its clarity + midtones; the cinematic
        # intent (mild grain/vignette/grade) is preserved, not removed. Every
        # value stays env-overridable, so a channel can dial the look back up.
        self.master   = _ovr_knob("VIDLORE_OVERLAY_STRENGTH", 0.78)
        # GRAIN CLARITY PASS (user 2026-06-06): film grain was blanketing EVERY
        # frame of EVERY video (modern stock + AI stills alike) at ~noise=4 per
        # clip + ~2 whole-video — it read as noise and softened the picture.
        # Grain should be INTELLIGENT: heavy only on genuinely OLD/archival
        # footage (that path uses arch_grain_amount, floor 6 — untouched) and a
        # whisper on clean modern footage just to stop gradient banding. So the
        # DEFAULT modern strength drops 0.52→0.16 (modern grain → ~1, near-
        # invisible; archival keeps its filmic 6). Dial back up any time with
        # VIDLORE_GRAIN_STRENGTH=0.5 (or 0 for perfectly clean).
        self.grain    = _ovr_knob("VIDLORE_GRAIN_STRENGTH",   0.16)
        self.vignette = _ovr_knob("VIDLORE_VIGNETTE_STRENGTH", 0.48)
        self.texture  = _ovr_knob("VIDLORE_TEXTURE_STRENGTH",  0.52)
        self.darken   = _ovr_knob("VIDLORE_DARKEN_STRENGTH",   0.38)

    # --- effective per-layer scalars (master folded in, clamped 0..1) ----- #
    def _eff(self, layer: float) -> float:
        return max(0.0, min(1.0, layer * self.master))

    def grain_amount(self, base_max: int | None = None) -> int:
        """Footage film-grain alls value: knob fraction of GRAIN_MAX. Floored at
        1 (was 2) so MODERN footage can read clean — a single noise unit only
        guards against gradient banding and is visually imperceptible. Archival
        keeps its own heavier floor via arch_grain_amount()."""
        cap = base_max if base_max is not None else self.GRAIN_MAX
        return max(1, int(round(cap * self._eff(self.grain))))

    def arch_grain_amount(self) -> int:
        """Archival grain: heavier floor (recovered film IS grainy) but capped
        so it never reads as damage."""
        return max(6, int(round(self.GRAIN_ARCH_MAX * self._eff(self.grain))))

    def final_grain_amount(self) -> int:
        return max(1, int(round(self.GRAIN_FINAL_MAX * self._eff(self.grain))))

    def vignette_angle(self, soft: float | None = None,
                       hard: float | None = None) -> float:
        """PI-divisor for a footage vignette. knob 0 → soft (no crush),
        knob 1 → hard cap. Returned as the *divisor* (e.g. 6.4 ⇒ PI/6.4)."""
        s = self.VIGN_SOFT_ANGLE if soft is None else soft
        h = self.VIGN_HARD_ANGLE if hard is None else hard
        # smaller divisor = harder; interpolate divisor between soft..hard.
        return s + (h - s) * self._eff(self.vignette)

    def arch_vignette_angle(self) -> float:
        return self.vignette_angle(self.VIGN_ARCH_SOFT, self.VIGN_ARCH_HARD)

    def texture_scale(self) -> float:
        """0..1 multiplier applied to a texture pack's alpha / sigma / shift
        so grids, blur and chroma-shift soften together with the knob."""
        return self.TEXTURE_MAX * self._eff(self.texture)

    def darken_gamma(self, proposed_gamma: float) -> float:
        """CLARITY GATE (luma floor): a finishing grade proposes a gamma. With
        darken_strength=0 we keep the full lift; as it rises we allow the
        grade toward neutral but NEVER below DARKEN_FLOOR_GAMMA, so footage
        keeps a reasonable luminance floor and blacks are never crushed."""
        floor = self.DARKEN_FLOOR_GAMMA
        # darken=0 → use proposed (max lift); darken=1 → pull toward 1.0 but
        # clamp at the floor. Blend then clamp.
        d = self._eff(self.darken)
        g = proposed_gamma + (1.0 - proposed_gamma) * d
        return max(floor, min(proposed_gamma, g)) if proposed_gamma >= floor \
            else max(floor, proposed_gamma)

    # --- THE AUTOMATIC CLARITY GATE -------------------------------------- #
    def clarity_gate(self, *, grain: int, vignette_div: float,
                     texture_layers: int, darken_gamma: float,
                     scene_kind: str = "footage") -> dict:
        """Pure, parameter-level gate run BEFORE the effect stack is committed
        for a scene. It estimates — from the chosen strengths + scene type, no
        pixel analysis needed — whether the projected result would crush
        luminance, drop subject-face visibility, reduce contrast, or STACK too
        many heavy overlays. If so it REDUCES (or skips) the heaviest layers
        first — vignette / darken / texture — keeping the grade. Returns a dict
        of the (possibly reduced) layer strengths + a list of applied
        reductions so callers/tests can see what fired.

        Heuristic "heaviness" budget (estimated, bounded — cheap):
          heavy = (grain>=8) + (vignette_div<=5.4) + (texture_layers>=2)
                  + (darken_gamma < FLOOR)
        A footage scene should carry AT MOST 2 heavy layers; >2 means the
        frame is being muddied, so we shed the heaviest until <=2 while always
        preserving the colour grade (gamma is only lifted, never crushed)."""
        reductions: list[str] = []
        g, vdiv, tex, gam = grain, vignette_div, texture_layers, darken_gamma

        # 1) hard luma floor — never allow a crushing gamma on footage/stills.
        if gam < self.DARKEN_FLOOR_GAMMA:
            gam = self.DARKEN_FLOOR_GAMMA
            reductions.append("darken→floor")

        # 2) count heavy layers (estimate).
        def _heavy(_g, _v, _t, _ga):
            return (int(_g >= 8) + int(_v <= 5.4) + int(_t >= 2)
                    + int(_ga < self.DARKEN_FLOOR_GAMMA))

        # MG card backgrounds are already designed — they should not carry a
        # cinematic overlay stack at all; collapse to the lightest treatment.
        if scene_kind == "card":
            return {"grain": min(g, 3), "vignette_div": max(vdiv, 6.8),
                    "texture_layers": 0, "darken_gamma": max(gam, 1.0),
                    "reductions": reductions + ["card→clean"]}

        budget = 1 if scene_kind == "archival" else 2
        # shed heaviest first: vignette → darken → texture → grain
        if _heavy(g, vdiv, tex, gam) > budget and vdiv <= 5.4:
            vdiv = max(vdiv, 6.4)             # soften the corner crush
            reductions.append("vignette↓")
        if _heavy(g, vdiv, tex, gam) > budget and gam < 1.10:
            gam = max(gam, 1.10)             # lift midtones back up
            reductions.append("darken↓")
        if _heavy(g, vdiv, tex, gam) > budget and tex >= 2:
            tex = 1                          # drop a texture layer
            reductions.append("texture↓")
        if _heavy(g, vdiv, tex, gam) > budget and g >= 8:
            g = 7                            # ease grain last (keep some)
            reductions.append("grain↓")
        return {"grain": g, "vignette_div": vdiv, "texture_layers": tex,
                "darken_gamma": gam, "reductions": reductions}


# Single import-time instance — read the knobs once.
_OVR = _OverlayRestraint()


# Faded-sepia archival film look (only for scenes the LLM marked
# "archival"): near-monochrome, warm age cast, punchy old contrast,
# real film grain. Tasteful and period — not a flashy filter.
_VINTAGE = (
    # Authentic recovered-archive treatment (only on LLM "archival"
    # scenes). Layered like real degraded film: gate-weave + handheld
    # instability, soft vintage optics, washed/low-contrast faded grade,
    # subtle analog chroma misregistration, heavy film grain, faint
    # scan-lines, archival vignette. Tuned subtle — recovered footage,
    # never a cheesy TikTok VHS filter.
    "scale=2012:1132,"
    "crop=1920:1080:"
    "x='(iw-1920)/2+5*sin(n/2.3)+3*sin(n/9.0)':"
    "y='(ih-1080)/2+4*sin(n/3.1)+3*sin(n/7.0)',"
    "gblur=sigma=0.6,"
    # heavily desaturated, low-contrast, lifted/washed blacks with a
    # WARM aged cast (real recovered film) — NOT a purple cross-process
    # 'vintage' preset, which reads as a cheesy phone filter.
    # RC5.1 — archival is REAL recovered film, but the old stack read as
    # *damaged*: gamma 0.95 darkened, alls=16 grain + a tight PI/3.8 focal
    # vignette ate faces / edge subjects. Lift the brightness floor (gamma
    # via the restraint policy never below DARKEN_FLOOR), cap the grain at a
    # filmic ~12 and soften the focal vignette so the recovered frame stays
    # readable. Still period, still grainy — just not muddy.
    f"eq=saturation=0.13:contrast=1.03:brightness=0.022:"
    f"gamma={_OVR.darken_gamma(0.95):.3f},"
    "colorbalance=rs=0.10:gs=0.03:bs=-0.10:rm=0.04:bm=-0.05,"
    "rgbashift=rh=1:bh=-1,"
    f"noise=alls={_OVR.arch_grain_amount()}:allf=t+u,"
    "drawgrid=width=iw:height=3:thickness=1:color=black@0.05,"
    # IMP_017 / RC5.1 — FOCAL vignette, now bounded by the restraint policy
    # (softest PI/4.6 .. hardest PI/4.0). Still guides the eye through busy
    # historical frames but never crushes the corners into a damaged-looking
    # 'binocular' mask; edge subjects stay clearly visible.
    f"vignette=angle=PI/{_OVR.arch_vignette_angle():.2f}"
)


# ISSUE #6 — LAYERED CINEMATIC FINISH. Between graphic moments the base
# footage was a single flat layer (scaled clip + theme grade + drift),
# which is exactly why it felt "flat / single-layer" vs Vidlore. This is
# a consistent, SUBTLE finishing stack composited over EVERY non-archival
# scene so the frame always has depth & texture (multiple things at
# once): micro-contrast clarity + fine temporal film grain + a soft
# cinematic vignette. It also unifies disparate stock clips AND AI
# stills into ONE filmic look (reduces the "AI montage" tell). Archival
# is left to _VINTAGE — never doubled (the user said keep it as-is).
# Leading comma: it is appended onto the theme `grade` substring.
# RC5.1: grain + vignette now flow through the bounded _OVR restraint
# policy (lighter grain, softer vignette) so footage stays clear.
_CINEMA_FINISH = (
    ",unsharp=5:5:0.32:5:5:0.0"        # gentle local clarity / depth
    f",noise=alls={_OVR.grain_amount()}:allf=t"   # fine grain (bounded)
    f",vignette=angle=PI/{_OVR.vignette_angle():.2f}"  # soft cinematic edges
)


# ---- OVERLAY PACK ENGINE (Phase 3) --------------------------------------
# Per-scene CINEMATIC FINISH chosen by the Variation Engine's pack.overlay
# (palette-driven, recency-aware). 6 subtle texture personalities — never
# TikTok, always documentary-grade. Replaces the single flat _CINEMA_FINISH
# so two adjacent scenes carry DIFFERENT tactile texture (intel scenes feel
# tactical with scan + flicker; sepia feels papery + grainy; broadcast feels
# analog) — viewer subconsciously reads "different camera / different stock"
# rather than "same automated stream of templates".
#
# Each finish always starts with the SHARED CLARITY + VIGNETTE base so the
# documentary identity is intact; only the texture additive varies.
# RC5.1 — the base vignette flows through the bounded restraint policy so it
# can never over-crush corners (PI/5.0 hard cap, softer by default).
_OVERLAY_BASE = (
    f",unsharp=5:5:0.32:5:5:0.0,vignette=angle=PI/{_OVR.vignette_angle():.2f}")


def _build_overlay_filters() -> "dict[str, str]":
    """RC5.1 — the 6 texture-personality packs, with their film-grain amount
    bounded by _OVR.grain_amount() and the heaviest texture elements (scanline
    grid alpha, paper blur sigma) softened by _OVR.texture_scale(). Restraint,
    not removal: each pack keeps its identity (scan / paper / dust / broadcast)
    but never stacks into the muddy, damaged look. The colour-balance casts are
    cheap and tasteful, so they are left intact (they tint, they don't crush)."""
    g_arch = _OVR.grain_amount(_OVR.GRAIN_ARCH_MAX)   # archival pack — heavier
    g_mid  = _OVR.grain_amount()                       # default footage grain
    g_fine = _OVR.grain_amount(6)                      # fine / broadcast grain
    tex    = _OVR.texture_scale()                      # 0..1 texture softener
    grid_a = 0.06 * tex                                # scanline darkness
    blur_s = 0.40 * tex                                # paper blur sigma
    g_dust = _OVR.grain_amount(_OVR.GRAIN_ARCH_MAX + 2)  # speckle floor (dust)
    return {
        # archival film grain — clean docs, period photos
        "archival_grain": (f",noise=alls={g_arch}:allf=t+u"
                           ",colorbalance=rs=0.03:gs=0.01:bs=-0.03"),
        # CRT / surveillance scanlines — tactical, intel, broadcast
        "crt_scan":       (f",noise=alls={_OVR.grain_amount(5)}:allf=t"
                           ",drawgrid=width=iw:height=4:thickness=1:"
                           f"color=black@{grid_a:.3f}"),
        # warm paper texture — newspaper, archive, period ledgers
        "paper_tex":      (f",noise=alls={g_arch}:allf=t+u"
                           f",gblur=sigma={blur_s:.3f}"
                           ",colorbalance=rs=0.04:gs=0.02:bs=-0.04"),
        # high-frequency speckle (analog film dust)
        "dust_speckle":   (f",noise=alls={g_dust}:allf=t+u"
                           ",unsharp=3:3:0.20:3:3:0.0"),
        # subtle time-varying brightness wobble — old-projector feel
        "projector_flicker": (f",noise=alls={g_fine}:allf=t"
                              ",eq=brightness='0.012*sin(t*9)':eval=frame"),
        # analog broadcast — fine grain + tiny chroma shift
        "broadcast_noise": (f",noise=alls={g_mid}:allf=t"
                            ",rgbashift=rh=1:bh=-1"
                            ",eq=contrast=1.02:eval=frame"),
    }


_OVERLAY_FILTERS: "dict[str, str]" = _build_overlay_filters()


def _grade_restraint(finish: str) -> str:
    """R4 (evidence-backed, 2026-06-04) — Vidlore renders ~2x darker than the
    Vidlore reference (measured mean luma 61 vs 116; 30% dark frames vs 4%)
    because the look grade + the UNIFORM per-scene finish vignette stack and
    crush footage. When VIDLORE_GRADE_RESTRAINT is on we (a) SOFTEN that
    uniform vignette (now handled by the bounded _OVERLAY_BASE itself) and
    (b) add a gentle midtone/shadow LIFT so the picture reads clear & premium
    rather than murky. Flag-gated; A/B-validated by re-measuring luma +
    inspecting frames. (Archival _VINTAGE is untouched.)

    RC5.1 — the lift gamma now passes through the OVERLAY-RESTRAINT clarity
    gate (_OVR.darken_gamma): the midtone lift is preserved but the policy
    guarantees the cumulative grade can never crush below the luma floor, so
    footage keeps detail / faces / readable shadows."""
    import os
    # Default-ON (2026-06-04): validated on the dark "Netflix Historical Epic"
    # look (mean luma 61->76, murky frames <40 luma 30%->11%, frames inspected:
    # clearer & still cinematic). The lift is gentle (can only brighten slightly),
    # so it is safe across looks; disable with VIDLORE_GRADE_RESTRAINT=0.
    if os.environ.get("VIDLORE_GRADE_RESTRAINT", "1").strip().lower() not in (
            "1", "true", "yes", "on"):
        return finish
    # The base vignette is already bounded by the restraint policy; only add
    # the midtone lift here (gamma routed through the luma-floor clarity gate).
    _g = _OVR.darken_gamma(1.14)
    return finish + f",eq=gamma={_g:.3f}:brightness=0.028:saturation=1.03"


def _overlay_finish_for_scene(scene_idx: int, sc) -> str:
    """Build the per-scene cinematic finish vf-snippet. Reads the Variation
    Engine's pack for this scene; falls back to canonical _CINEMA_FINISH if
    anything goes wrong (the doc must always render)."""
    try:
        from .templates._shared import set_scene_context, scene_pack
        set_scene_context(sc)
        ovname = scene_pack().overlay
        extra = _OVERLAY_FILTERS.get(ovname, "")
        if extra:
            return _grade_restraint(_OVERLAY_BASE + extra)
    except Exception:                                       # noqa: BLE001
        pass
    return _grade_restraint(_CINEMA_FINISH)


_ROLE_HOLD = {"reveal", "climax", "payoff", "turn"}
_ROLE_BUILD = {"escalation", "problem", "stakes"}
_ROLE_INFO = {"evidence", "proof", "context"}
_ROLE_CALM = {"resolution", "reaction"}

# DOCUMENTARY RHYTHM ENGINE (Human-Editor #4) — the impact beats and the
# beats that should breathe AROUND them. These drive cross-scene emotional
# flow (anticipation before a reveal, a held landing ON it, breathing room
# after), not just per-scene length.
_ROLE_PEAK = {"reveal", "climax"}                  # the moment it LANDS
_ROLE_AFTER = {"payoff", "resolution", "reaction"}  # the exhale after it

# CINEMATIC TASTE (HE#10) — graphic kinds that FILL the frame (data/text
# panels). When one of these owns a scene it IS the emphasis, so the camera
# shouldn't also push in behind it (let one element own the moment). Curated
# to the clearly full-frame cards; lower-thirds / corner insets are absent
# (a gentle drift behind those is fine).
_FULLFRAME_CARDS = {
    "timeline", "process_diagram", "comparison", "stat_dashboard",
    "classified", "quote_highlight", "bullet_list", "document",
    "map_route", "map_reveal", "map_region", "chapter_marker",
    "breaking_news", "did_you_know", "glossary", "define_the_term",
    "evidence", "cause_effect",
    "currency_stat", "era_banner", "network_graph", "title_card",
}


# ATMOSPHERE ENGINE (Human-Editor #8) — read each scene's ENVIRONMENT from
# its words so a subtle world-texture can sit underneath. Keyword lexicons,
# highest score wins; nothing clear -> no atmosphere (silence respected).
_ENV_LEX: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("desert", ("desert", "sand", "dune", "dunes", "arid", "wasteland",
                "sahara", "barren", "drought", "empty quarter", "nomad")),
    ("cave", ("cave", "cavern", "underground", "tunnel", "subterranean",
              "chamber", "mine", "grotto", "abyss", "depths", "limestone")),
    ("water", ("ocean", "sea", "water", "wave", "waves", "underwater",
               "lake", "flood", "coast", "marine", "tide", "river")),
    ("space", ("space", "satellite", "orbit", "cosmos", "cosmic", "galaxy",
               "planet", "lunar", "nasa", "telescope", "radar")),
    ("room", ("laboratory", "research", "evidence", "archive", "document",
              "records", "clinical", "facility", "office", "files",
              "interrogation", "study", "scientist")),
    ("tension", ("war", "military", "army", "soldier", "weapon", "battle",
                 "missile", "combat", "conspiracy", "classified", "threat",
                 "enemy", "spy", "surveillance", "secret", "attack")),
    ("urban", ("city", "street", "urban", "traffic", "downtown",
               "metropolis", "skyline", "building")),
    ("nature", ("forest", "jungle", "woods", "tree", "trees", "wildlife",
                "mountain", "valley", "meadow", "grass", "wilderness")),
)


def _atmos_kind(scene) -> str:
    """The environment texture for a scene (or '' for none) — scored from
    its narration + keywords + visual."""
    blob = " " + re.sub(
        r"[^a-z ]", " ",
        (" ".join([
            getattr(scene, "narration", "") or "",
            " ".join(getattr(scene, "keywords", None) or []),
            getattr(scene, "visual", "") or "",
        ])).lower()) + " "
    best, score = "", 0
    for kind, words in _ENV_LEX:
        s = sum(1 for w in words if (" " + w + " ") in blob or w in blob)
        if s > score:
            best, score = kind, s
    return best if score >= 1 else ""


# R1 — cinematic LOW-END WEIGHT gating (flag-gated in the assembler). Reserve the
# sustained sub floor for emotionally-weighted beats; most scenes get 0 (off),
# `strong` (1.0) is rare (climax/reveal-class or peak energy).
_CW_STRONG = {"climax", "reveal", "turn", "payoff"}
_CW_MED = {"hook", "tension", "problem", "stakes", "escalation",
           "investigation", "proof", "evidence"}


def _cinematic_weight_for(scene, energy: int) -> float:
    """Per-scene cinematic low-end weight 0..1 — FEWER BUT BETTER. Reserve the
    sustained sub for the genuinely heavy beats: the reveal/climax-class roles
    (strong), or a single high-energy tension/investigation beat (light). Every
    other scene is 0 — no broad energy-based trigger (that smeared weight across
    too many scenes and read muddy)."""
    role = (getattr(scene, "role", "") or "").strip().lower()
    if role in _CW_STRONG or energy >= 5:        # the genuine peak / reveal moment
        return 1.0
    if role in _CW_MED and energy >= 4:          # one heavy tension/hook beat (rare)
        return 0.5
    return 0.0


def _role_profile(role: str) -> str:
    """Issue #5 — map the narrative beat to a footage-DURATION profile:
      hold  : the impact/emotional/suspense beat — HOLD one long shot so
              it lands ('impact fully land nahi karta' was the complaint)
      build : escalation/problem — gradually tighten (cut a bit faster)
      info  : evidence/proof/context — brisk & even (informational)
      calm  : resolution/reaction — gentle, let it sit
      hook  : the OPENING — Issue #11: an editor doesn't start chopping;
              they hold an arresting first image to draw you in, then
              settle (front-loaded long first beat). This single beat is
              what separates "an editor opened the film" from "AI
              started arranging clips".
      ''    : unknown -> fall back to the energy-based rhythm (Issue #2)"""
    role = (role or "").strip().lower()
    if role == "hook":
        return "hook"
    if role in _ROLE_HOLD:
        return "hold"
    if role in _ROLE_BUILD:
        return "build"
    if role in _ROLE_INFO:
        return "info"
    if role in _ROLE_CALM:
        return "calm"
    return ""


def _blend_beat_target(sty_target: float, look_target) -> float:
    """REC pacing reconcile — let the per-video recipe beat_target NUDGE
    the StyleMode pacing without overriding it.

    The StyleMode is the niche ANCHOR (true_crime tense, history slow,
    explainer active).  The per-video recipe beat_target (carried on the
    resolved look) nudges that anchor so two same-niche videos pace
    DIFFERENTLY — but a bounded blend + clamp keeps every result inside the
    niche's safe, readable band: never TikTok-fast, never boring-slow.

      • no look target          → StyleMode unchanged (legacy)
      • STANDARD anchor (3.4)   → look fully owns pacing (legacy DOC_016)
      • non-default anchor      → blend 65% StyleMode / 35% recipe, clamped
                                  to ±~20% of the anchor and to an absolute
                                  [2.2, 6.2]s safety band
      • VIDLORE_PACING_BLEND=0 → legacy (StyleMode wins, recipe ignored)
    """
    import os
    if not isinstance(look_target, (int, float)):
        return sty_target
    lt = float(look_target)
    if abs(sty_target - 3.4) < 0.01:               # STANDARD sentinel → legacy
        return lt
    if os.environ.get("VIDLORE_PACING_BLEND", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return sty_target
    w = 0.35
    blended = sty_target * (1.0 - w) + lt * w
    lo, hi = sty_target * 0.82, sty_target * 1.22
    return round(max(2.2, min(6.2, max(lo, min(hi, blended)))), 3)


# REC_BIMODAL — map the recipe's per-niche rhythm_mode to a shot-length
# SPREAD factor.  >0 widens the spread (hero holds held longer against the
# quick density scenes → a bimodal cadence the genre benchmarks use); <0
# narrows it (flowing / uniform); 0 = even/legacy.
_RHYTHM_SPREAD = {
    "bimodal_burst": 1.0,             # business — punchy bursts + hard holds
    "bimodal_optional": 0.6,          # spy/crime/explainer/bio — bimodal allowed
    "slow_with_single_burst": 0.5,    # mystery — slow holds, one burst
    "slow_with_motivated_windows": 0.35,   # history — slow + faster windows
    "bimodal": 0.8, "punchy": 1.0, "staccato": 1.0, "dynamic": 0.7,
    "even": 0.0, "steady": 0.0,
    "flowing": -0.6, "smooth": -0.6, "legato": -0.6, "even_flow": -0.5,
}


def _rhythm_spread() -> float:
    """Per-video shot-length spread from the recipe rhythm_mode (carried on
    the look, so the footage fetcher and assembler — which both resolve the
    same look — read the SAME value).  Acts on beat LENGTHS only (beat COUNT
    untouched → cache-safe).  env VIDLORE_BIMODAL=0 disables (legacy)."""
    import os
    if os.environ.get("VIDLORE_BIMODAL", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return 0.0
    try:
        from .look_dna import look_get as _lg
        m = (_lg("rhythm_mode") or "").strip().lower()
    except Exception:                                              # noqa: BLE001
        return 0.0
    return _RHYTHM_SPREAD.get(m, 0.0)


def plan_beats(
    durs: list[float], target: float = 3.4, bmin: float = 2.4,
    cap: int = 140, energies: list[int] | None = None,
    roles: list[str] | None = None, hook_s: float = 42.0,
) -> list[tuple[int, float, bool, int]]:
    # Look DNA / per-video recipe nudges the beat target. STANDARD (3.4)
    # lets the look fully own pacing; a non-default StyleMode anchors the
    # niche rhythm and the recipe beat_target bounded-blends on top (so
    # same-niche videos pace differently, safely). Footage fetcher and
    # assembler BOTH route through plan_beats with the same StyleMode
    # target + active look → identical beat count (cache stays valid).
    try:
        from .look_dna import look_get as _lg_bt
        target = _blend_beat_target(target, _lg_bt("beat_target"))
        if abs(bmin - 2.4) < 0.01:
            _lb = _lg_bt("beat_min")
            if isinstance(_lb, (int, float)):
                bmin = float(_lb)
    except Exception:                                              # noqa: BLE001
        pass
    # REC_BIMODAL — per-video shot-length spread (LENGTHS only; the beat
    # COUNT below is unchanged so the footage fetcher and assembler agree).
    _bsp = _rhythm_spread()
    """Slice each scene's duration into ~`target`s beats. Shared by the
    footage fetcher (so it grabs one DISTINCT clip per beat) and the
    assembler (so each beat plays its own clip). Returns
    (sceneIndex, beatDuration, isLastBeatOfScene, beatIndexInScene).

    ISSUE #2 — the beat COUNT per scene is unchanged (so the footage
    fetcher and assembler always agree and caches stay valid), but the
    beat LENGTHS are no longer identical d/k clones. A real editor never
    cuts on a metronome: tense scenes accelerate (shots get shorter
    toward the climax), calm scenes let ONE shot breathe (a long hold),
    mid scenes drift organically. Per-scene lengths still sum EXACTLY to
    the scene duration, so the narration / caption / graphic timeline and
    A/V sync are untouched."""
    def _raw_role(idx: int) -> str:
        if roles and 0 <= idx < len(roles):
            return (roles[idx] or "").strip().lower()
        return ""

    def _beat_lengths(j: int, d: float, k: int) -> list[float]:
        if k <= 1:
            return [d]
        e = 3
        if energies and 0 <= j < len(energies):
            e = max(1, min(5, energies[j] or 3))
        prof = ""
        if roles and 0 <= j < len(roles):
            prof = _role_profile(roles[j])
        # ---- DOCUMENTARY RHYTHM ENGINE: cross-scene emotional flow ----
        # Read the NEIGHBOURS, not just this scene, so the cut breathes
        # with the story the way an editor shapes a sequence:
        cur, nxt, prv = _raw_role(j), _raw_role(j + 1), _raw_role(j - 1)
        is_peak = cur in _ROLE_PEAK                 # the reveal/climax lands
        pre_peak = (nxt in _ROLE_PEAK and not is_peak)   # anticipation beat
        after_peak = (prv in _ROLE_PEAK and cur in _ROLE_AFTER)  # the exhale
        tension_run = (prof == "build"
                       and _role_profile(prv) == "build")  # carry momentum
        # which beat carries the "held" shot. A REVEAL lands LAST (linger,
        # then the image holds as it lands / cuts away) — delayed-reveal
        # timing. Otherwise the hold sits on a SEEDED beat (HE#9 human
        # imperfection): `j % k` marched the long shot one beat further
        # each scene — a subtle metronome a viewer's eye learns to
        # predict. A real editor's hold doesn't land on a schedule.
        hold_m = (k - 1) if is_peak else int(_rng01(j * 71 + 13) * k) % k
        ws: list[float] = []
        for m in range(k):
            r = _rng01(j * 131 + m * 7 + 17)
            if prof == "hook":
                # Issue #11: the OPEN. An editor holds an arresting
                # FIRST image to pull you in, THEN settles into the
                # story — never starts chop-chop-chop. Front-loaded:
                # beat 0 is a long sit, the rest measured (not frantic).
                w = (2.30 if m == 0 else 0.80) * (0.95 + 0.12 * r)
            elif prof == "hold":
                # Issue #5: ONE long impact shot the moment lands on;
                # the rest brief connective. Bigger hold at higher
                # emotional intensity. Beats SLOW here, never frantic.
                # REC_BIMODAL widens the hold vs filler gap per video.
                big = (2.05 + 0.16 * e) * (1.0 + 0.30 * _bsp)
                fil = 0.55 * (1.0 - 0.22 * _bsp)
                w = (big if m == hold_m else fil) * (0.93 + 0.14 * r)
            elif prof == "build":
                # gradually tighten — accelerate, but a touch gentler
                # than a frantic action cut (suspense, not chaos). On a
                # CONTINUING tension run the scene starts already-tight
                # (momentum carries across scenes, not reset each time).
                hi = 1.10 if tension_run else 1.26
                lo = 0.62 if tension_run else 0.74
                ramp = hi - (hi - lo) * (m / (k - 1))
                w = ramp * (0.88 + 0.24 * r)
            elif prof == "info":
                # informational — brisk & EVEN (no shot drags, none
                # flashes); efficient delivery of facts
                w = 0.90 + 0.20 * r
            elif prof == "calm":
                # resolution/reaction — gentle, let it sit, soft hold
                w = (1.45 * (1.0 + 0.30 * _bsp) if m == hold_m
                     else 0.86 * (1.0 - 0.22 * _bsp)) * (0.92 + 0.18 * r)
            elif e >= 4:                       # no role -> energy logic
                ramp = 1.30 - 0.62 * (m / (k - 1))      # 1.30 -> 0.68
                w = ramp * (0.86 + 0.28 * r)
            elif e <= 2:
                w = (1.70 * (1.0 + 0.30 * _bsp) if m == hold_m
                     else 0.78 * (1.0 - 0.22 * _bsp)) * (0.92 + 0.16 * r)
            else:
                w = 0.78 + 0.46 * r
            ws.append(max(0.05, w))
        # ---- cross-scene shaping (relative; renormalized to d below) ----
        if pre_peak and k >= 2:
            # ANTICIPATION: a held breath on the last beat before we cut
            # INTO the reveal (the "wait for it…" the edit needs).
            ws[-1] *= 1.85
            for m in range(k - 1):
                ws[m] *= 0.9
        if after_peak and k >= 2:
            # AFTERMATH: let the first beat breathe after the shock lands,
            # then settle — the emotional exhale, not an instant restart.
            ws[0] *= 1.9
            for m in range(1, k):
                ws[m] *= 0.92
        s = sum(ws)
        bl = [d * w / s for w in ws]
        # keep beats sane, then renormalize so the sum is EXACTLY d
        # (sync is sacred). Only clamp when there's room to.
        if d / k >= 1.5:
            lo = 1.45
            bl = [max(lo, b) for b in bl]
            s2 = sum(bl)
            bl = [b * d / s2 for b in bl]
        return bl

    # OPENING-HOOK TIGHTENING. Scenes that START inside the first `hook_s`
    # seconds cut FASTER (more beats / shorter shots) so the hook builds
    # subconscious momentum like a pro YouTube/Netflix doc — then the film
    # naturally breathes once past the hook. This depends ONLY on `durs`
    # (+ target/bmin), so the footage fetcher and assembler — which both
    # call plan_beats with the same durs/target/bmin — compute the SAME
    # per-scene beat COUNT, and A/V sync stays exact. Beats never go below
    # ~1.4s, so it's tighter, not choppy/TikTok.
    # Pre-compute editorial mode per scene (cheap; role-driven).  Used
    # below to MULTIPLY the per-beat target so DENSITY scenes pack
    # more cuts (~14/min target like the genre benchmarks) while
    # RESTRAINT scenes hold longer (cinematic differentiator).
    _scene_modes: list[str] = []
    try:
        from .script_gen import _ROLE_DENSITY, _ROLE_RESTRAINT
        _hi_run = 0          # IMP_015 — consecutive intensity>=4 scenes
        for j in range(len(durs)):
            r = (roles[j] if roles and j < len(roles) else "") or ""
            r = r.lower().strip()
            e = (energies[j] if energies and j < len(energies) else 3) or 3
            # IMP_015 — BREATHING ROOM: after 3+ consecutive intense scenes,
            # force the next scene into a restraint 'exhale' (one slow
            # contemplative beat) so the doc breathes — unless that scene is
            # itself a hard climax (e>=5), which earns its own held shot.
            _breather = (_hi_run >= 3 and e < 5)
            if _breather:
                _scene_modes.append("restraint")
            elif e >= 5:
                _scene_modes.append("restraint")     # climax wins
            elif r in _ROLE_RESTRAINT:
                _scene_modes.append("restraint")
            elif r in _ROLE_DENSITY:
                _scene_modes.append("density")
            elif e >= 3:
                _scene_modes.append("density")
            else:
                _scene_modes.append("restraint")
            # cadence counter: a forced exhale RESETS it; intense scenes
            # extend it; a natural calm scene clears it.
            if _breather:
                _hi_run = 0
            elif e >= 4:
                _hi_run += 1
            else:
                _hi_run = 0
    except Exception:                                          # noqa: BLE001
        _scene_modes = ["density"] * len(durs)

    # IMP_014 — TENSION CADENCE. Across a RUN of 2+ consecutive 'build'
    # scenes (escalation/problem/stakes), progressively SHORTEN the per-scene
    # beat target so the cut rate visibly tightens toward the peak; the peak
    # scene the run leads into then RELEASES on its long hold (its
    # restraint/hold profile already provides that). Gentle and capped at
    # -15% so the footage fetcher's clip count is only mildly exceeded (the
    # proven safe base-clip-reuse fallback covers the +1-2 beats that a fast
    # tension montage wants).
    _tension_accel = [1.0] * len(durs)
    if roles:
        _run = 0
        for _tj in range(len(durs)):
            if _role_profile(roles[_tj] if _tj < len(roles) else "") == "build":
                _run += 1
                if _run >= 2:
                    # 2nd build 0.90x, 3rd+ 0.80x (floor) — enough to cross a
                    # beat-count boundary so the acceleration is perceptible.
                    _tension_accel[_tj] = max(0.80, 1.0 - 0.10 * (_run - 1))
            else:
                _run = 0

    while True:
        plan: list[tuple[int, float, bool, int]] = []
        acc = 0.0
        for j, d in enumerate(durs):
            if acc < hook_s:                       # this scene is in the hook
                eff_t = max(1.6, target * 0.62)
                eff_b = max(1.4, bmin * 0.70)
            else:
                eff_t, eff_b = target, bmin
            # ── MODE multiplier ──────────────────────────────────────
            # DENSITY: shorter target → more beats per scene (~+50 %
            # cuts), pushing cuts/min from our ~6 toward the genre
            # baseline of ~14.  RESTRAINT: longer target → fewer
            # beats, longer holds, more silence room.
            mode = (_scene_modes[j] if j < len(_scene_modes)
                    else "density")
            # IMP_002 — shot-length VARIANCE scaled by scene intensity.
            # Real docs vary shot length with emotional weight: intense
            # scenes cut FASTER, contemplative ones BREATHE.  Scaling the
            # density/restraint multiplier by this scene's energy WIDENS the
            # spread between fast and slow scenes (less robotic, more
            # cinematic) instead of one flat 0.70 / 1.30 for every scene.
            _e2 = (energies[j] if energies and j < len(energies) else 3) or 3
            if mode == "density":
                # Anchored at the legacy 0.70 for the NEUTRAL case (e=3) so the
                # footage fetcher — which calls beats_per_scene WITHOUT energies
                # and therefore sees every scene as e=3 density — fetches the
                # same clip count the assembler wants for ordinary scenes (zero
                # new divergence).  Only intense / calm scenes deviate:
                #   e=3 ->0.70x (legacy)   e=4 ->0.64x   e>=5 ->0.58x  (tighter)
                #   e=2 ->0.74x            e=1 ->0.78x          (a touch slower)
                _dens = (0.70
                         - 0.06 * max(0, min(2, _e2 - 3))
                         + 0.04 * max(0, min(2, 3 - _e2)))
                eff_t = max(1.4, eff_t * _dens)
                eff_b = max(1.2, eff_b * min(1.0, _dens + 0.10))
            else:
                # RESTRAINT only ever fires in the assembler (the fetcher's
                # e=3 default never reaches this branch), so the assembler
                # always wants FEWER beats here than were fetched — safe to
                # vary freely.  e>=2 ->1.30x (legacy), e<=1 ->1.45x (breathe).
                _rest = 1.30 + 0.15 * max(0, min(1, 2 - _e2))
                eff_t = eff_t * _rest
                eff_b = eff_b * (_rest - 0.10)
            # IMP_014 — accelerate the cut rate deeper into a build run.
            # Scale BOTH the target and the min-beat floor, else the
            # int(d/eff_b) cap below pins k and the acceleration is invisible.
            if _tension_accel[j] < 1.0:
                eff_t = max(1.4, eff_t * _tension_accel[j])
                eff_b = max(1.2, eff_b * _tension_accel[j])
            k = max(1, round(d / eff_t))
            k = max(1, min(k, int(d / eff_b) or 1))
            # IMP_002 cap — no single beat should hold longer than ~12s (a
            # dead, uniform hold reads as a frozen still); force enough beats
            # to keep every shot under the cap on long contemplative scenes.
            if d > 12.0:
                k = max(k, int(-(-d // 12)))
            for m, bd in enumerate(_beat_lengths(j, d, k)):
                plan.append((j, bd, m == k - 1, m))
            acc += d
        if len(plan) <= cap or target >= 12:
            return plan
        target += 1.0  # very long video -> coarser beats


def beats_per_scene(durs: list[float], target: float = 3.4,
                    bmin: float = 2.4) -> list[int]:
    """How many distinct clips each scene needs (one per beat). MUST be
    called with the SAME target/bmin the assembler will use (the Style
    Mode's pacing), or the footage fetcher and assembler disagree on the
    beat count."""
    out: list[int] = [0] * len(durs)
    for j, _bd, _last, _m in plan_beats(durs, target=target, bmin=bmin):
        out[j] += 1
    return out


_MAP_KINDS = {"map_reveal", "map_route", "map_region", "map_pin_cluster",
              "globe_highlight"}
_DOC_KINDS = {"document", "classified", "redacted", "case_file", "newspaper",
              "news_article", "press_release", "letter", "diary",
              "email_screenshot"}
_EVID_KINDS = {"evidence", "evidence_tag", "verdict_stamp",
               "conspiracy_board"}
_ACT_ROLES = {"reveal", "climax", "turn"}


# DOC_003 — transition vocabulary FOLLOWS the active Look DNA. We map the
# resolved look (preset name) → a transition "style key" that _edit_plan
# keys on, extended beyond the original true_crime/epic with cold-restrained
# (spy/mystery), archival-slow (history) and clean (explainer) families.
# When no look is active this returns the passed style_name unchanged, so
# legacy (no-channel, pre-auto-look) behaviour is byte-identical.
_LOOK_TRANSITION_STYLE = {
    "true_crime":       "true_crime",   # tension/investigative: flash, glitch, blur ok
    "midnight_pacific": "spy",          # cold, restrained: no whip/blur/glitch
    "amber_chronicles": "history",      # archival/film, slow
    "netflix_epic":     "epic",         # slow premium
    "atlas_explained":  "explainer",    # clean: content-motivated + clean cuts
    "homestead":        "history",
    "standard":         "standard",
}


def _active_transition_style(style_name: str) -> str:
    """Transition style key, preferring the active Look DNA identity over
    the passed StyleMode name → niche-appropriate transition vocabulary.
    No active look → exact legacy style_name (backward compatible)."""
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            key = _LOOK_TRANSITION_STYLE.get(
                (getattr(lk, "name", "") or "").lower())
            if key:
                return key
    except Exception:                                              # noqa: BLE001
        pass
    return (style_name or "").lower()


# REC transition weighting — flashy, EMOTION-driven transitions that a
# restrained recipe palette can veto (content-motivated geo_push / page_wipe
# / evidence_flash / glitch-on-surveillance are NEVER vetoed — they are
# justified by what the cut lands on).
_FLASHY_TRANS = frozenset(("whip", "whip_r", "blur_cut"))

# Scene-level anti-repetition — content-MOTIVATED joins are exempt from the
# "no identical transition back-to-back" rule (a run of maps SHOULD all
# geo_push; consecutive documents SHOULD page_wipe). Only the emotion/style
# transitions (dissolve / slow_dissolve / film_dissolve / fadeblack / whip /
# blur_cut / soft_dissolve) must not repeat consecutively.
_CONTENT_MOTIVATED_TRANS = frozenset((
    "geo_push", "geo_push_r", "page_wipe", "page_wipe_r",
    "evidence_flash", "archive_flash", "glitch"))


# PHASE-2 sound restraint — NICHE-AWARE whoosh cadence. The whoosh family
# (reveal/transition/text_slam/swoosh) fires at most once per this many
# seconds. Spy / mystery / history lean SILENCE-heavy (long gap → very few
# whooshes); explainer can be a touch more present. Keyed on the active look
# like the other DOC tables; no look → the neutral 22 s default.
_LOOK_WHOOSH_GAP = {
    "midnight_pacific": 34.0,   # spy / mystery — silence-heavy, very sparse
    "amber_chronicles": 30.0,   # history — archive texture, almost no whoosh
    "homestead":        30.0,
    "netflix_epic":     26.0,   # epic biography — restrained
    "true_crime":       26.0,   # tension + occasional restrained whoosh
    "atlas_explained":  18.0,   # explainer — clean, slightly more present
    "standard":         22.0,
}


def _look_whoosh_gap(default: float = 22.0) -> float:
    """Per-niche minimum seconds between whoosh-family SFX (silence as
    design).  No active look → `default` (legacy 22 s).  Env
    VIDLORE_WHOOSH_GAP_S still hard-overrides at the call site."""
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            return float(_LOOK_WHOOSH_GAP.get(
                (getattr(lk, "name", "") or "").lower(), default))
    except Exception:                                              # noqa: BLE001
        pass
    return default


# REC sfx_restraint — per-video SFX density lever; WIDENS the whoosh min-gap
# (fewer whooshes) on top of the niche default.  A multiplier so it composes
# with _look_whoosh_gap.  >1 = sparser.
_SFX_RESTRAINT_MULT = {
    "sparse":         1.30,   # history/mystery/explainer — silence-led
    "diegetic_light": 1.15,   # spy — light, diegetic-leaning
    "tension_bed":    1.10,   # true_crime — tension carries, fewer whooshes
    "light":          1.0,    # business — baseline
}


def _look_sfx_restraint_mult() -> float:
    """Whoosh-gap multiplier from the recipe sfx_restraint.  No look /
    VIDLORE_SFX_RESTRAINT=0 → 1.0 (legacy)."""
    import os
    if os.environ.get("VIDLORE_SFX_RESTRAINT", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return 1.0
    try:
        from .look_dna import look_get as _lg
        v = (_lg("sfx_restraint") or "").strip().lower()
        return _SFX_RESTRAINT_MULT.get(v, 1.0)
    except Exception:                                              # noqa: BLE001
        return 1.0


def _look_grade_mode() -> str:
    """Active per-video GRADE MODE (editorial recipe).  'gilded' = a warm
    firelight-on-near-black look for tycoon / industrialist business &
    biography arcs (crushed shadows + warm gold, saturation PRESERVED).
    Anything else → 'standard' (the muted documentary grade).  No active
    look → 'standard'.  Env VIDLORE_GILDED=0 forces standard."""
    import os
    if os.environ.get("VIDLORE_GILDED", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return "standard"
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            m = lk.get("grade_mode")
            if isinstance(m, str) and m:
                return m
    except Exception:                                              # noqa: BLE001
        pass
    return "standard"


# QUIET COLD-OPEN — (music_delay_s, music_fade_in_s) at the START of the
# film. Voice-led niches breathe in over footage + room tone (a brief
# silence then a long music fade); explainer opens ACTIVE. This is an AUDIO
# curve only — it never moves/delays the TITLE (DOC_006 stays removed). No
# active look → (0,0) = exact legacy (no fade). Env VIDLORE_COLD_OPEN=0 off.
_LOOK_COLD_OPEN = {
    "midnight_pacific": (1.2, 5.0),   # spy / mystery — quiet, atmospheric
    "amber_chronicles": (0.8, 4.5),   # history — let the world breathe in
    "netflix_epic":     (0.8, 4.0),   # biography — restrained build
    "homestead":        (0.8, 4.0),
    "true_crime":       (0.5, 3.0),   # tension, but not silent
    "atlas_explained":  (0.0, 1.2),   # explainer — active, music present early
    "standard":         (0.0, 2.0),
}


def _territory_flyin() -> bool:
    """True → map_region uses a cinematic fly-in: the basemap AND the glowing
    territory zoom together on an ease-out 'arrival' curve (the camera settles
    onto the region) instead of a weak linear push where only the bg moved.
    env VIDLORE_TERRITORY_FLYIN=0 → exact legacy (7% linear, glow static)."""
    import os
    return os.environ.get("VIDLORE_TERRITORY_FLYIN", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _look_cold_open() -> tuple:
    """(delay_s, fade_in_s) for the MUSIC bed at the film's open.  No active
    look → (0,0) = exact legacy (no change).  VIDLORE_COLD_OPEN=0 disables."""
    import os
    if os.environ.get("VIDLORE_COLD_OPEN", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return (0.0, 0.0)
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            return _LOOK_COLD_OPEN.get(
                (getattr(lk, "name", "") or "").lower(), (0.0, 0.0))
    except Exception:                                              # noqa: BLE001
        pass
    return (0.0, 0.0)


def _recipe_transition_palette():
    """The active per-video recipe's ALLOWED transition tokens, as a set,
    or None.  A flashy transition not in this set is downgraded to a clean
    cut in _edit_plan (premium restraint + per-video variation).  None →
    no constraint (legacy / no recipe).  VIDLORE_TRANS_WEIGHT=0 disables."""
    import os
    if os.environ.get("VIDLORE_TRANS_WEIGHT", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    try:
        from .look_dna import look_get as _lg
        p = _lg("transition_allowed")
        if isinstance(p, (list, tuple)) and p:
            return frozenset(str(x) for x in p)
    except Exception:                                              # noqa: BLE001
        pass
    return None


# DOC_004 — niche motion PERSONALITY (hold frequency). Vidlore's footage
# motion already varies per scene (scene_pack motion type + 4 KB modes +
# energy scaling + _breathe emotional holds + impact punches + per-look
# drift_scale). The one missing per-NICHE flavour is how OFTEN a shot locks
# off into a still hold: reverent history/biography should breathe & hold
# more (contemplative weight); explainer should stay active (fewer holds);
# spy holds for tension. This multiplies the seeded hold threshold. Look-
# gated — no active look → 1.0 (motion behaviour byte-identical to before).
_LOOK_HOLD_MULT = {
    "amber_chronicles": 1.45,   # reverent history — lingers, breathes
    "netflix_epic":     1.35,   # epic biography — held, weighty
    "midnight_pacific": 1.30,   # spy — restraint, tension holds
    "homestead":        1.30,
    "true_crime":       1.15,
    "atlas_explained":  0.70,   # explainer — active, fewer holds
    "standard":         1.0,
}


def _look_hold_mult() -> float:
    """Per-niche multiplier on the motivated-stillness probability.
    REC_WIRE_AXES: a per-video recipe value on the resolved look wins;
    else the per-preset table; else 1.0 (no active look)."""
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            v = lk.get("hold_mult")            # per-video editorial recipe
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
            return _LOOK_HOLD_MULT.get(
                (getattr(lk, "name", "") or "").lower(), 1.0)
    except Exception:                                              # noqa: BLE001
        pass
    return 1.0


# DOC_014 — niche GRADE saturation. Forensic finding: professional docs
# measure muted mean frame saturation (LEMMiNO 0.02, Johnny Harris 0.12,
# Fall of Civ 0.12) while our output measured 0.31 — the base cinematic
# finish was BOOSTING saturation to 1.06×, reading as "vivid AI/stock"
# rather than "graded documentary". Pull the finish saturation DOWN per
# niche toward a muted, filmic grade (mystery/spy coldest; explainer
# lightest). Look-gated — no active look → exact legacy 1.06.
_LOOK_GRADE_SAT = {
    "midnight_pacific": 0.80,   # spy / mystery — cold, desaturated
    "true_crime":       0.84,   # investigative — muted, slightly cold
    "amber_chronicles": 0.90,   # history — muted warm
    "netflix_epic":     0.90,   # epic biography — restrained, graded
    "homestead":        0.92,
    "atlas_explained":  0.94,   # explainer — clean, lightly muted
    "standard":         0.95,
}


def _look_grade_saturation() -> float:
    """Saturation multiplier for the global cinematic-finish eq. Pulls
    toward a muted documentary grade per niche; no active look → 1.06
    (exact legacy boost preserved)."""
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            v = lk.get("grade_sat")            # per-video editorial recipe
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
            return _LOOK_GRADE_SAT.get(
                (getattr(lk, "name", "") or "").lower(), 0.92)
    except Exception:                                              # noqa: BLE001
        pass
    return 1.06


# DOC_017 — niche MUSIC-BED level. Forensic finding: voice-led documentaries
# let the score RECEDE under narration (Fall of Civ runs music only ~87% of
# the time; LEMMiNO is sparse/dark) — the voice carries. Our music bed is a
# fixed level regardless of niche. Pull it DOWN for slow / voice-led niches
# (history, biography, mystery) so narration breathes; keep it present for
# explainer (whose density_floor is high — bed stays loud by design). This
# multiplies the (already sidechain-ducked) bed volume. Look-gated — no
# active look → 1.0 (exact legacy mix).
_LOOK_MUSIC_BED_MULT = {
    "amber_chronicles": 0.72,   # history — voice-led, score recedes
    "netflix_epic":     0.74,   # epic biography — restrained under VO
    "midnight_pacific": 0.74,   # mystery / spy — sparse, dark, recessive
    "homestead":        0.80,
    "true_crime":       0.85,   # tension bed, still sits under the voice
    "atlas_explained":  1.00,   # explainer — music-forward, bed stays present
    "standard":         1.00,
}


def _look_music_bed_mult() -> float:
    """Per-niche multiplier on the music-bed volume (voice-led niches
    recede the score). No active look → 1.0 (legacy mix unchanged)."""
    try:
        from .look_dna import current as _lc
        lk = _lc()
        if lk is not None:
            v = lk.get("music_bed")            # per-video editorial recipe
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
            return _LOOK_MUSIC_BED_MULT.get(
                (getattr(lk, "name", "") or "").lower(), 0.90)
    except Exception:                                              # noqa: BLE001
        pass
    return 1.0


# REC music_behavior — per-niche music PRESENCE character, a BOUNDED factor on
# the resting bed (distinct from the per-video music_bed LEVEL).  Momentum
# niches sustain a driving bed; mystery/history recede to leave room for the
# voice + silence (the "momentum vs silence-heavy" forensic signature).
_MUSIC_BEHAVIOR = {
    "momentum_continuous":      1.06,   # biz/geo/explainer — driving, present
    "tension_bed_with_pockets": 0.96,   # true_crime — tension bed → pockets
    "voice_led_recede":         0.90,   # history — gets out of the voice's way
    "silence_punctuated":       0.84,   # mystery — deepest recede, most silence
    "balanced":                 1.0,
}


def _look_music_behavior() -> float:
    """Bounded resting-bed factor from the recipe music_behavior.  No look /
    VIDLORE_MUSIC_BEHAVIOR=0 → 1.0 (legacy)."""
    import os
    if os.environ.get("VIDLORE_MUSIC_BEHAVIOR", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return 1.0
    try:
        from .look_dna import look_get as _lg
        b = (_lg("music_behavior") or "").strip().lower()
        return _MUSIC_BEHAVIOR.get(b, 1.0)
    except Exception:                                              # noqa: BLE001
        return 1.0


def _edit_plan(energies: list[int], n: int, *, gap: int = 3, drop: int = 2,
               durs: list[float] | None = None,
               roles: list[str] | None = None,
               gkinds: list[str] | None = None,
               style_name: str = "", hook_s: float = 45.0
               ) -> tuple[list[float], list[str]]:
    """THE DIRECTOR — a documentary transition decision engine. For every
    scene boundary it chooses a transition from emotion + pacing + story
    role + the graphic the cut lands on + style mode + position in the
    film. CLEAN HARD CUTS stay the default; a motivated transition fires
    only when the story earns it (exhale, act break, geography flow,
    evidence/archive reveal, document, tension). The OPENING HOOK (first
    ~`hook_s` s) runs tighter & faster (quick directional whips on energy
    rises) then settles, the way a real editor paces a YouTube doc.
    Returns (durations, styles) length n-1; style 'cut' = hard cut, else a
    key into `_TRANSITIONS`. All transitions are pairwise (no chains)."""
    def e(i):
        return energies[i] if 0 <= i < len(energies) else 2

    def role(i):
        return (roles[i] if roles and 0 <= i < len(roles) else "") or ""

    def gk(i):
        return (gkinds[i] if gkinds and 0 <= i < len(gkinds) else "") or ""

    starts = [0.0] * n
    if durs:
        acc = 0.0
        for i in range(n):
            starts[i] = acc
            acc += durs[i] if i < len(durs) else 0.0

    ts: list[float] = []
    styles: list[str] = []
    last_trans = -10
    DG = max(1, int(gap))                  # style: min scenes between visibles
    DROP = max(1, int(drop))               # style: exhale size for a dissolve
    sm = _active_transition_style(style_name)
    _palette = _recipe_transition_palette()    # REC transition weighting

    for i in range(n - 1):
        a, b = e(i), e(i + 1)
        d = b - a
        gj = gk(i + 1)                     # we cut INTO scene i+1
        opening = starts[i + 1] < hook_s   # inside the hook
        cut = _CUT * (0.75 + 0.55 * _rng01(i * 9 + 2))
        typ = "cut"

        # ---- CONTENT-MOTIVATED transitions: tied to the graphic the cut
        # lands on. These ALWAYS fire (the graphics are already capped &
        # spaced upstream), so a map always gets a geo-push, a document a
        # page-wipe, etc. — they are justified by content, not stylistic.
        if gj in _MAP_KINDS:                          # geography flow
            typ = "geo_push" if _rng01(i * 5 + 1) < 0.5 else "geo_push_r"
        elif gj in _EVID_KINDS:                       # evidence reveal
            typ = "evidence_flash" if sm == "true_crime" else "archive_flash"
        elif gj in _DOC_KINDS:                        # a document lands
            typ = "page_wipe" if _rng01(i * 5 + 3) < 0.6 else "page_wipe_r"
        elif gj == "surveillance" and sm == "true_crime":
            typ = "glitch"                            # style-gated tech cut
        else:
            # ---- EMOTION/INTENSITY-DRIVEN transitions (read the narration
            # arc, not just the graphic). Energy 1-5 and role are the
            # showrunner's reading of narration intensity, so these react to
            # emotional shifts / tension spikes / reveals / exhales the way a
            # real editor interprets the script. Spacing-gated so they stay
            # rare (opening hook denser at gap 1; else the style gap).
            spaced = (i - last_trans) >= (1 if opening else DG)
            if (b >= 5 and d >= 2) or (role(i + 1) in _ACT_ROLES and b >= 4):
                # a hard PUNCH into the reveal/climax reads strongest as a
                # clean cut + the impact-hit SFX (no visual transition).
                typ = "cut"
            elif d <= -DROP and spaced:               # reflective exhale
                # The EXHALE dissolve flavour follows the look's family:
                # history/epic bloom on film, spy/mystery breathe on a long
                # slow dissolve, explainer stays subtle, true_crime/standard
                # keep the neutral dissolve.
                if role(i + 1) in ("resolution", "payoff"):
                    typ = "slow_dissolve"
                elif sm in ("epic", "history"):
                    typ = "film_dissolve"
                elif sm in ("spy", "mystery"):
                    typ = "slow_dissolve"
                elif sm == "explainer":
                    typ = "soft_dissolve"
                else:
                    typ = "dissolve"
            elif d >= 2 and b >= 4 and spaced:
                # TENSION SPIKE / investigative discovery — a sharp intensity
                # rise gets a fast directional motion-blur cut. But that reads
                # as "flashy" for premium-slow & cold-restrained looks, so
                # those stay restrained (clean cut, or a slow dissolve if a
                # visible beat is due); explainer keeps it clean.
                if sm in ("spy", "mystery", "history", "epic"):
                    typ = "slow_dissolve" if (i - last_trans) >= DG else "cut"
                elif sm == "explainer":
                    typ = "cut"
                else:
                    typ = "blur_cut"
            elif opening and d >= 1 and spaced:       # tight, fast hook
                # A whip is energetic — right for true_crime/explainer/standard,
                # but premium-slow (history/epic) and cold (spy/mystery) looks
                # open with restraint: a gentle dissolve or a clean cut.
                if sm in ("history", "epic"):
                    typ = "dissolve"
                elif sm in ("spy", "mystery", "explainer"):
                    typ = "cut"          # clean / cold opens — no whip
                else:
                    typ = "whip" if _rng01(i * 5 + 7) < 0.5 else "whip_r"
            elif (sm in ("epic", "history") and role(i + 1) == "context"
                  and d <= -1 and spaced):
                # historical breathing room — a gentle film bloom on a calm
                # settle (EPIC mode only).
                typ = "film_dissolve"

        # REC transition weighting — veto a flashy transition the recipe's
        # palette doesn't allow (→ clean cut). Content-motivated joins and
        # restrained dissolves are never touched; spacing already prevents
        # spam. This gives same-niche videos different transition energy.
        if typ in _FLASHY_TRANS and _palette is not None and typ not in _palette:
            typ = "cut"
        # Scene-level anti-repetition — never the SAME non-cut transition
        # twice consecutively (a clean cut breaks the repeat). Content-
        # motivated joins are exempt (consecutive maps SHOULD geo_push).
        if (typ != "cut" and typ not in _CONTENT_MOTIVATED_TRANS
                and styles and styles[-1] == typ):
            typ = "cut"
        if typ != "cut" and typ in _TRANSITIONS:
            base = _TRANSITIONS[typ][1]
            xf = base * (0.85 + 0.3 * _rng01(i * 9 + 6))
            if opening:
                xf *= 0.82                            # quicker in the hook
            ts.append(round(xf, 3))
            styles.append(typ)
            last_trans = i
        else:
            ts.append(cut)
            styles.append("cut")
    return ts, styles


def _xfade_chain(durs: list[float], ts: list[float]) -> tuple[str, str]:
    """Crossfade N scenes with a PER-BOUNDARY duration ts[i] (so the cut
    rhythm varies with emotion). Seg i is rendered dur_i + ts[i] long
    (last seg unpadded); offsets accumulate so each scene's on-screen
    time stays == its narration and total == sum(durs) (audio stays in
    sync, no trailing trim needed). Returns (filter_complex, final_label).
    """
    n = len(durs)
    acc = durs[0] + (ts[0] if n > 1 else 0.0)
    prev = "[0:v]"
    parts: list[str] = []
    for i in range(1, n):
        ti = ts[i - 1]
        off = acc - ti
        out = f"[x{i}]"
        parts.append(
            f"{prev}[{i}:v]xfade=transition=fade:duration={ti:.3f}:"
            f"offset={off:.3f}{out}"
        )
        li = durs[i] + (ts[i] if i < n - 1 else 0.0)
        acc += li - ti
        prev = out
    return ";".join(parts), prev


def _copy_font(workdir: Path, theme: dict | None = None,
               sample_text: str = "") -> str | None:
    """Put a usable .ttf at workdir/font.ttf so ffmpeg drawtext can use
    a clean relative path (the final ffmpeg call runs with cwd=workdir).
    Returns the relative name, or None if no font is available
    (overlays skipped).

    Theme-aware: when `theme` is given, prefer that theme's display-font
    candidates first.

    SCRIPT-AWARE: when `sample_text` contains non-Latin codepoints
    (CJK / Arabic / Hebrew / Urdu), prepend the bundled Noto font for
    that script -- otherwise ffmpeg drawtext would render tofu boxes.
    The bundled VidloreSans family at the end of `_FONT_CANDIDATES`
    stays as the always-works safety net for Latin text.
    """
    candidates: list[str] = []
    # 1. Script-specific bundled Noto font FIRST when text needs it
    if sample_text:
        try:
            from . import lang as _lang
            script_paths = _lang.pick_font_paths(sample_text)
            for p in script_paths:
                if p not in candidates:
                    candidates.append(p)
        except Exception:                                  # noqa: BLE001
            pass
    # 2. Theme display font next (won't be picked if text is non-Latin)
    if theme:
        try:
            tf = (theme.get("fonts") or {}).get("display") or []
            candidates.extend([str(p) for p in tf])
        except Exception:                                  # noqa: BLE001
            pass
    # 3. Default fallback chain
    candidates.extend(_FONT_CANDIDATES)
    for p in candidates:
        if Path(p).exists():
            dest = workdir / "font.ttf"
            try:
                shutil.copyfile(p, dest)
                return "font.ttf"
            except Exception:                              # noqa: BLE001
                continue
    return None


def _sanitize_label(s: str) -> str:
    """Safe, short, drawtext-inline chapter label (no quotes/colons)."""
    s = re.sub(r"[^A-Za-z0-9 -]", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s[:30].strip()


def _overlay_filters(
    title: str | None,
    accent: tuple,
    duration: float,
    workdir: Path,
    chapters: list[tuple[float, str]] | None = None,
) -> list[str]:
    """Absolute-timeline polish. All burned-in corner overlays were
    removed at the user's request: the intro title card, the SUBSCRIBE
    lower-third, AND the top-corner chapter strip (it looked like a
    cheap floating tag over the footage). Nothing is drawn here now —
    on-screen meaning comes only from the in-context motion graphics
    (callouts, location cards, key-phrase stabs), never a corner label."""
    return []

    font = _copy_font(workdir)  # noqa: unreachable (kept for signature)
    if font is None:
        return []
    r, g, b = accent
    hexc = f"0x{r:02X}{g:02X}{b:02X}"
    out: list[str] = []

    for start, raw in chapters or []:
        label = _sanitize_label(raw)
        if not label:
            continue
        s = max(0.0, float(start))
        e = s + 2.6
        alpha = (
            f"if(lt(t,{s:.2f}),0,"
            f"if(lt(t,{s + 0.4:.2f}),(t-{s:.2f})/0.4,"
            f"if(lt(t,{e - 0.5:.2f}),1,"
            f"if(lt(t,{e:.2f}),({e:.2f}-t)/0.5,0))))"
        )
        win = f"between(t,{s:.2f},{e:.2f})"

        def slide_x(base: int) -> str:
            # EASED slide-in (shared motion language): off-left -> base
            # with out_cubic deceleration over 0.5s, hold, then a quick
            # exit. interp_ff holds `base` after settling, so the hold
            # phase is automatic. (single-quoted in the filter, so commas
            # stay unescaped.)
            x_in = _mo.interp_ff(base - 360, base, s, 0.5, "out_cubic")
            return (
                f"if(lt(t,{e - 0.5:.2f}),{x_in},"
                f"{base}-360*((t-{e - 0.5:.2f})/0.5))"
            )

        # No accent bar, no hard box — a clean, understated upper-third
        # tag: bold white with a soft drop-shadow + a hairline dark edge
        # for legibility over bright footage. Premium, not a cheap chip.
        out.append(
            f"drawtext=fontfile={font}:text='{label}':"
            "fontsize=40:fontcolor=white:borderw=2:bordercolor=black@0.5:"
            "shadowcolor=black@0.55:shadowx=0:shadowy=3:"
            f"x='{slide_x(70)}':y=72:alpha='{alpha}':enable='{win}'"
        )
    return out


def _u_clean(s: str, punct: str) -> str:
    """Category-based multilingual sanitizer: keep ALL Unicode letters,
    digits AND combining marks (Devanagari/Thai matras, Arabic harakat,
    decomposed accents — `\\w` silently DROPPED these, mangling Hindi
    'छिपी'->'छ प'), plus the allowed punctuation; replace only
    ffmpeg/filtergraph-dangerous & control chars with a space."""
    out = []
    for c in s:
        if c.isspace():
            out.append(" ")
        elif c in punct or c.isalnum() or _u.category(c)[0] == "M":
            out.append(c)
        else:
            out.append(" ")          # ' " \\ : % ; = [ ] { } ctrl …
    return "".join(out)


def _upper(s: str) -> str:
    """Locale-aware uppercase. Turkish/Azerbaijani have a special rule:
    dotted 'i' -> 'İ' and dotless 'ı' -> 'I'. Python's default
    str.upper() turns BOTH into a plain 'I' (wrong Turkish
    orthography: "GIZLI" instead of "GİZLİ"). We only switch to the
    Turkish rule when a Turkic-distinctive letter (ı İ ğ Ğ ş Ş) is
    present — those exact code points don't occur in any of the other
    supported Latin languages (Romanian's ș/Ș is a DIFFERENT, comma-
    below code point), so non-Turkish text is never affected."""
    if any(c in s for c in ("ı", "İ", "ğ", "Ğ", "ş", "Ş")):
        s = s.replace("i", "İ").replace("ı", "I")
    return s.upper()


def _dt(s: str) -> str:
    """Single source of truth for any string that's about to hit ffmpeg
    drawtext ``text=`` or be written to a ``textfile=``.  ffmpeg's
    drawtext doesn't run complex-script shaping, so RTL scripts must be
    pre-shaped (Arabic to presentation forms + BiDi, Hebrew BiDi only).
    Latin / CJK pass through with zero cost.

    Every site that previously did ``text='{user_str}'`` should call
    ``text='{_dt(user_str)}'`` instead — and every ``write_text(...)``
    feeding a textfile= should wrap content the same way.  This is the
    last line of defence against the multilingual-tofu class of bug.
    """
    if not s:
        return s
    try:
        from . import lang as _lang
        return _lang.shape_for_drawtext(s)
    except Exception:                                              # noqa: BLE001
        return s


def _card_text(s: str) -> str:
    """drawtext-safe premium card line: keep letters/digits/space and a
    few punctuation marks documentaries use, uppercase, short."""
    # MULTILINGUAL: keep ALL Unicode letters/digits/marks (é ñ ü ß,
    # 日本語, 한국어, Кириллица, छिपी, ภาษา …) — only strip
    # ffmpeg/filtergraph-dangerous & control chars. NFC first.
    s = _u_clean(_u.normalize("NFC", s or ""), " .,&·/-")
    s = _upper(re.sub(r"\s+", " ", s).strip())
    return s[:38].strip()


def _fig_text(s: str) -> str:
    """Like _card_text but for a STAT/NUMBER figure — keeps the symbols
    that carry the meaning ($, %, +, ×, etc.) instead of stripping them
    (so '$20' / '47%' / '3X' don't render as '20' / '47' / '3X')."""
    s = _u_clean(_u.normalize("NFC", s or ""), " .,&%$+#×/-")
    s = _upper(re.sub(r"\s+", " ", s).strip())
    return s[:14].strip()


def _parse_number(s: str):
    """Split a figure into (prefix, int_value, suffix, final_display) so
    a number can RAPIDLY COUNT UP and land on the formatted value.
    '$50'->('$',50,'','$50'); '47%'->('',47,'%','47%'); '4,000 YEARS'->
    ('',4000,' YEARS','4,000 YEARS'); '1950s'->('',1950,'S','1950S');
    '3X'->('',3,'X','3X'). Returns None if there's no countable int."""
    fig = _fig_text(s)
    m = re.search(r"\d[\d,]*", fig)
    if not m:
        return None
    raw = m.group(0)
    val = int(raw.replace(",", ""))
    prefix = fig[:m.start()]
    suffix = fig[m.end():]
    return prefix, val, suffix, fig


# IMP_025 — spelled-out magnitude detection for the floating_stat pill. Real
# TTS-targeted narration writes "four hundred and twenty million", not
# "420,000,000", so the comma-digit gate alone almost never fires. This parses
# a spelled quantity to an int ONLY when number-words scale a magnitude word
# (thousand/million/billion/trillion) — so years ("nineteen ninety-three"),
# ordinals ("forty-fourth") and bare counts ("three judges") never match.
_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_MAG_WORDS = {"thousand": 1_000, "million": 1_000_000,
              "billion": 1_000_000_000, "trillion": 1_000_000_000_000}


def _spelled_to_number(text: str):
    """Largest spelled-out magnitude quantity in `text` as an int, e.g.
    'four hundred and twenty million' -> 420000000, 'four thousand' -> 4000.
    Returns None unless a number-word actually scales a thousand/million/
    billion/trillion (high precision: years/ordinals/small counts are skipped)."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    best = 0
    result = 0         # accumulated thousands+ groups for current phrase
    current = 0        # current group being built (units + hundreds)
    scaled = False     # did a thousand+ magnitude apply with a real number?

    def _flush():
        nonlocal best
        total = result + current
        if scaled and total >= 1000:
            best = max(best, total)

    for w in toks:
        if w in _NUM_WORDS:
            current += _NUM_WORDS[w]
        elif w == "hundred":
            # "two hundred" -> 200; a BARE "hundred" (no leading number)
            # stays 0 so it can't seed a phantom magnitude.
            current = current * 100
        elif w in _MAG_WORDS:
            # close THIS group at its magnitude and add it to the running
            # total — a LOWER magnitude after a higher one (e.g. "two million
            # five hundred thousand") must ADD a new group, never re-multiply
            # the whole accumulator (the old bug: 2,500,000 -> 2,000,500,000).
            # REQUIRE a real preceding number: a BARE magnitude word
            # ("the million-dollar question", "a billion reasons") must NOT
            # conjure a phantom 1x figure — that put a wrong stat on screen.
            if current > 0:
                current *= _MAG_WORDS[w]
                result += current
                current = 0
                scaled = True
        elif w == "and":
            continue                          # "four hundred and twenty"
        else:
            _flush()                          # phrase boundary
            result = current = 0
            scaled = False
    _flush()
    return best if best >= 1000 else None


def _best_stat_figure(text: str) -> str:
    """Best notable quantity in `text` as a comma-grouped string (for the
    floating_stat count-up): the larger of any comma-grouped digits OR a
    spelled-out magnitude. '' when there is no notable figure."""
    cands = [int(s.replace(",", ""))
             for s in re.findall(r"\d{1,3}(?:,\d{3})+", text or "")]
    sp = _spelled_to_number(text)
    if sp:
        cands.append(sp)
    # digit + magnitude word ("50 million views", "1.5 billion users"):
    # a bare digit before a magword never reaches _spelled_to_number,
    # which reads number-WORDS only -- so "million" alone scored
    # 1,000,000 and the literal "50" was dropped. Mirrors the
    # _money_figure digit+magword loop; magnitude-gated, so bare
    # years / ordinals / counts still never match.
    for mnum, mword in re.findall(
            r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)",
            text or "", re.I):
        cands.append(int(float(mnum) * _MAG_WORDS[mword.lower()]))
    return f"{max(cands):,}" if cands else ""


# MONEY COUNT-UP — when a stat beat is about MONEY, count it up as money
# ($0B -> $420B) instead of a bare 12-digit number. Requires BOTH a real
# magnitude AND a currency cue, so "1,200,000 people" / "50 million views"
# never read as money. env VIDLORE_MONEY_COUNTUP=0 falls back to bare digits.
_CURRENCY_CUE = re.compile(
    r"\$|£|€|¥|₹|\bdollar|\beuro|\bpound\b|\byen\b|\brupee|\bcost\b|\bcosts\b|"
    r"\bworth\b|\brevenue|\bprofit|\bfortune|\bsalary|\bsalaries|\bbudget|"
    r"\bseized|\bstole\b|\bstolen\b|\bsmuggl|\blaunder|\bransom|\bbribe|"
    r"\bnet\s+worth|\bmarket\s+cap|\bGDP\b|\bbillion-dollar|\bmillion-dollar|"
    r"\bpaid\b|\bearn(?:ed|s|ings)?\b|\bdebt\b|\bloan\b|\bfunding|\bvaluation",
    re.I)
_CURRENCY_SYM = ("$", "£", "€", "¥", "₹")


def _compact_money(val: int, sym: str) -> str:
    """Compact currency with an INTEGER mantissa so the count-up ticks
    cleanly. Steps DOWN a magnitude when the top one needs a decimal
    (1,200,000,000 -> '$1,200M', never '$1.2B' which would tick 0->1)."""
    for thr, suf in ((1_000_000_000_000, "T"), (1_000_000_000, "B"),
                     (1_000_000, "M"), (1_000, "K")):
        if val >= thr and val % thr == 0:
            return f"{sym}{val // thr:,}{suf}"
    return f"{sym}{val:,}"


def _money_figure(text: str) -> str:
    """Compact currency string ('$420M', '$400B') when `text` names a money
    magnitude with a currency cue; '' otherwise (caller falls back to the
    bare figure). The returned string re-parses via `_parse_number` to a
    '$'-prefixed value the floating-stat roll ticks up as money."""
    import os
    if not text or os.environ.get(
            "VIDLORE_MONEY_COUNTUP", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return ""
    if not _CURRENCY_CUE.search(text):
        return ""
    cands = [int(s.replace(",", ""))
             for s in re.findall(r"\d{1,3}(?:,\d{3})+", text)]
    sp = _spelled_to_number(text)
    if sp:
        cands.append(sp)
    # digit + magnitude word ("$400 billion", "420 million dollars")
    for mnum, mword in re.findall(
            r"(\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion)",
            text, re.I):
        cands.append(int(float(mnum) * _MAG_WORDS[mword.lower()]))
    if not cands:
        return ""
    val = max(cands)
    if val < 1000:                 # too small to read as a money magnitude
        return ""
    sym = next((g for g in _CURRENCY_SYM if g in text), "$")
    return _compact_money(val, sym)


def _doc_clean(s: str) -> str:
    """Readable prose for the evidence card (kept for a textfile, so
    only strip control chars / collapse whitespace)."""
    s = re.sub(r"[\x00-\x1f]", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _pct_of(s: str) -> float:
    """Real 0..1 fraction for the stat bar. Handles '70%', '9 out of
    10', '3 in 5', '9/10', odds '1:4', and bare numbers. ('9 out of 10'
    must be 0.90 — the old code read just '9' and gave a 9% bar)."""
    s = (s or "").strip()

    def _c(x: float) -> float:
        return max(0.04, min(1.0, x))

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return _c(float(m.group(1)) / 100.0)
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:out\s*of|in|of|per|/)\s*(\d+(?:\.\d+)?)",
        s, re.I)
    if m and float(m.group(2)) != 0:
        return _c(float(m.group(1)) / float(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", s)  # odds a:b
    if m and (float(m.group(1)) + float(m.group(2))) != 0:
        a_, b_ = float(m.group(1)), float(m.group(2))
        return _c(a_ / (a_ + b_))
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return 0.6
    v = float(m.group(0))
    if 0.0 < v <= 1.0:
        return _c(v)
    if v <= 100.0:
        return _c(v / 100.0)
    return 0.85


def _graphic_card_filters(
    cues: list,
    font: str | None,
    accent: tuple,
    workdir: Path,
    type_events: list | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Premium documentary motion-graphics (Vidlore parity), used sparingly.
    Returns (comma_filters, overlay_stages, post_filters):
      * comma_filters  — drawbox/drawtext applied in the main [0:v] chain
      * overlay_stages — `movie`+`overlay` graph segments for image cards
                         (portrait photo, archive photo); each threads the
                         video label via the {CUR}/{OUT} placeholders
      * post_filters   — drawtext applied AFTER the image overlays (so a
                         portrait NAME sits on top of its full-screen photo)

    Devices: number/stat (slammed figure) · location/label (title bar) ·
    chart (animated bar) · callout (annotated leader) · document (archive
    page w/ B&W photo, SOURCE line, highlight sweep) · portrait (cinematic
    full-screen photo + giant name). Honest templates — no fake tracking."""
    if font is None:
        return [], [], []
    r, g, b = accent
    hexc = f"0x{r:02X}{g:02X}{b:02X}"
    out: list[str] = []
    stages: list[str] = []
    post: list[str] = []
    num_events: list[tuple[float, float]] = []  # (roll_start, land)
    gi = 0  # overlay-image counter (unique movie labels)
    for i, cue in enumerate(cues):
        start, d, kind, raw, body = cue[0], cue[1], cue[2], cue[3], cue[4]
        asset = cue[5] if len(cue) > 5 else ""
        rev_t = cue[6] if len(cue) > 6 else -1.0
        shot_type = (cue[7] if len(cue) > 7 else "") or ""   # IMP_005
        map_fig = (cue[8] if len(cue) > 8 else "") or ""      # IMP_016
        # Start the graphic DEEPER into the scene (~35% in), never at the
        # cut — so on-screen text doesn't appear before the narrator says
        # it (issue: text precedes narration). Always leave tail room.
        # Appear DEEP into the scene (~62% in) so the narrator has
        # already spoken the line before the card lands — never "text
        # first, voice later". Min 1.6s in; always keep a 1.3s tail.
        a = start + min(max(1.6, d * 0.62), max(1.6, d - 1.3))
        emax = start + d - 0.10

        if kind == "floating_stat":
            # IMP_023 — FLOATING STAT LOWER-THIRD. A footage-only scene named a
            # notable comma-grouped figure but carries no card; rather than
            # escalate to a full-frame number card (cheap) or drop it, tick the
            # figure up in a compact lower-LEFT pill OVER the footage — the one
            # Wendover "data-in-context" principle that fits a footage-first
            # doc. Reuses the proven IMP_016 count-up; the narrator's line is
            # the label, so the pill stays a single restrained number. Drawn in
            # the main chain (no overlay to ride on, unlike the map counter).
            _pnm = _parse_number(raw)
            if not _pnm:
                continue
            _pfx, _val, _psfx, _pfinal = _pnm
            _pfinal = _dt(_pfinal)
            fs_start = start + min(max(1.4, d * 0.45), max(1.4, d - 2.0))
            b_ = min(start + d - 0.50, emax)
            if b_ - fs_start < 1.6:
                continue
            _roll = min(0.95, max(0.55, (b_ - fs_start) * 0.45))
            _land = fs_start + _roll
            if _land >= b_ - 0.30:
                continue
            win = f"between(t,{fs_start:.2f},{b_:.2f})"
            # compact lower-LEFT pill (subtle over live footage; left-anchored
            # so it clears centred subtitles when captions are on)
            _PX, _PY, _PW, _PH = 90, 1080 - 250, 470, 120
            # IMP_025 — fit the font to the pill width so a large spelled-out
            # magnitude ("420,000,000" = 11 chars) never overflows the scrim.
            # VidloreSans bold digit ~0.56*fontsize; inner width ~= PW-56.
            _avail = _PW - 56
            _flen = max(len(_pfinal), 3)
            _fz = max(40, min(76, int(_avail / (_flen * 0.56))))
            _tx, _ty = _PX + 28, _PY + 22
            out.append(
                f"drawbox=x={_PX}:y={_PY}:w={_PW}:h={_PH}:"
                f"color=black@0.42:t=fill:enable='{win}'")
            out.append(
                f"drawbox=x={_PX}:y={_PY}:w=6:h={_PH}:"
                f"color={hexc}:t=fill:enable='{win}'")
            # MONEY COUNT-UP — carry the currency symbol (and a %-free magnitude
            # suffix B/M/K/T) THROUGH the roll so a money beat reads as money
            # from the first tick ($0B->$420B), not a bare number that gains a
            # '$' only at land. Non-currency stats unchanged (empty prefix).
            _ismoney = _pfx.strip() in _CURRENCY_SYM
            _rpre = _dt(_pfx) if _ismoney else ""
            _rsuf = (_dt(_psfx) if (_ismoney and "%" not in _psfx
                                    and "\\" not in _psfx) else "")
            _ceif = (
                f"%{{eif\\:floor({_val}*(1-pow(1-"
                f"clip((t-{fs_start:.2f})/{_roll:.2f}\\,0\\,1)\\,3)))\\:d}}")
            _cra = (f"if(lt(t,{fs_start:.2f}),0,"
                    f"if(lt(t,{fs_start + 0.10:.2f}),"
                    f"(t-{fs_start:.2f})/0.10,1))")
            out.append(
                f"drawtext=fontfile={font}:expansion=normal:"
                f"text='{_rpre}{_ceif}{_rsuf}':fontsize={_fz}:fontcolor=white:"
                "shadowcolor=black@0.65:shadowx=0:shadowy=4:"
                f"x={_tx}:y={_ty}:alpha='{_cra}':"
                f"enable='between(t,{fs_start:.2f},{_land:.2f})'")
            out.append(
                f"drawtext=fontfile={font}:text='{_pfinal}':"
                f"expansion=none:fontsize={_fz}:fontcolor=white:"
                "shadowcolor=black@0.65:shadowx=0:shadowy=4:"
                f"x={_tx}:y={_ty}:"
                f"enable='between(t,{_land:.2f},{b_:.2f})'")
            num_events.append((fs_start, _land))
            continue

        if kind == "typing_date":
            # CINEMATIC TYPEWRITER — the timestamp/intel line is REVEALED
            # character-by-character (per-char drawtext with gte() enable),
            # a caret tracks the cursor, with slight human timing variation
            # and a longer beat on punctuation. Char appearance times are
            # collected so the click track can be synced exactly.
            from PIL import ImageFont as _IF
            import random as _twr
            txt = re.sub(r"[^\w \-/.,:]", "", (raw or "")).strip().upper()[:40]
            if not txt:
                continue
            # CONTEXT-AWARE typing voice from the scene text + context line.
            _blob = f"{raw} {body}".lower()
            if any(w in _blob for w in (
                    "classif", "secret", "intel", "redact", "clearance",
                    "eyes only", "top secret", "encrypt", "secure")):
                tw_style = "secure"
            elif any(w in _blob for w in (
                    "terminal", "system", "server", "network", "cyber",
                    "grid", "satellite", "coordinate", "gps", "log",
                    "console", "mainframe", "data")):
                tw_style = "digital"
            elif (re.search(r"\b1[5-9]\d\d\b", _blob)        # pre-2000 year
                  or any(w in _blob for w in (
                      "archive", "telegram", "dispatch", "ministry",
                      "memo", "historical", "record"))):
                tw_style = "mech"
            else:
                tw_style = "digital"
            # The typing IS the reveal, so it starts EARLY (not 62% in) and
            # is paced to finish with a tail of breathing room inside the
            # scene — never overflowing the cut.
            a_tw = start + 0.50
            b_ = min(a_tw + max(2.6, d - 0.8), emax)
            if b_ - a_tw < 1.4:
                continue
            win = f"between(t,{a_tw:.2f},{b_:.2f})"
            BX, BY, BW, BH = 80, 1080 - 240, 1100, 160
            fs = 64
            try:
                fo = _IF.truetype(str(Path(workdir) / "font.ttf"), fs)
            except Exception:                              # noqa: BLE001
                fo = None
            out.append(f"drawbox=x={BX}:y={BY}:w={BW}:h={BH}:"
                       f"color=black@0.78:t=fill:enable='{win}'")
            # Accent bar — LEFT for LTR text, RIGHT for RTL text so it
            # marks the edge the eye enters the panel from.  Routed
            # through `lang.accent_bar_edge_for` so this single point
            # mirrors the bar automatically for Arabic/Urdu/Hebrew.
            try:
                from . import lang as _lang
                _ab_edge = _lang.accent_bar_edge_for(txt or body or "")
            except Exception:                                  # noqa: BLE001
                _ab_edge = "left"
            if _ab_edge == "right":
                out.append(f"drawbox=x={BX + BW - 6}:y={BY}:w=6:h={BH}:"
                           f"color={hexc}:t=fill:enable='{win}'")
            else:
                out.append(f"drawbox=x={BX}:y={BY}:w=6:h={BH}:"
                           f"color={hexc}:t=fill:enable='{win}'")
            # NOTE: 'ARCHIVE TIMESTAMP' kicker label REMOVED here -- user
            # feedback ("yeh acha text likn oper archieve timestamp ka word
            # nae ana chyia") was filed against this exact codepath; the
            # earlier removal only touched footage._render_typing_date_card
            # but assemble.py has its OWN typing_date renderer (this one).
            # The date + body text are now vertically centred inside the
            # 160px panel (shifted up by 20px from the original positions
            # that left empty space at the top for the kicker).
            ctx = _u_clean(_u.normalize("NFC", body or ""), " .,-/&·")
            ctx = _upper(re.sub(r"\s+", " ", ctx).strip())[:50]
            ctx = _dt(ctx)
            if ctx:
                out.append(f"drawtext=fontfile={font}:text='{ctx}':"
                           f"fontsize=30:fontcolor=white@0.70:x={BX+32}:"
                           f"y={BY+108}:enable='{win}'")

            def _esc(c):
                return (c.replace("\\", "\\\\").replace(":", r"\:")
                        .replace("'", r"\\\'").replace("%", r"\%")
                        .replace(",", r"\,"))

            # ── RTL fallback: per-char typewriter would destroy Arabic /
            # Hebrew contextual shaping (each glyph would render in
            # isolated form, no joins).  For RTL, draw the FULL shaped
            # string once with a soft fade-in + the same accent rule so
            # the chrome still reads "archival timestamp" without
            # mutilating the script.  Latin / CJK keep the typewriter.
            try:
                from . import lang as _lang
                _typing_is_rtl = _lang.is_rtl(txt)
            except Exception:                                  # noqa: BLE001
                _typing_is_rtl = False
            if _typing_is_rtl:
                txt_shaped = _dt(txt)
                tin = a_tw + 0.30
                tfade = (
                    f"if(lt(t,{a_tw:.2f}),0,"
                    f"if(lt(t,{tin:.2f}),(t-{a_tw:.2f})/0.30,"
                    f"if(lt(t,{b_ - 0.40:.2f}),1,"
                    f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.40,0))))"
                )
                # Right-anchored inside the panel for RTL
                out.append(
                    f"drawtext=fontfile={font}:text='{txt_shaped}':"
                    f"expansion=none:fontsize={fs}:fontcolor=white:"
                    f"shadowcolor=black@0.55:shadowx=0:shadowy=4:"
                    f"x={BX + BW - 32}-text_w:y={BY+34}:"
                    f"alpha='{tfade}':enable='{win}'")
                if type_events is not None:
                    # one "settle" event so the SFX still fires
                    type_events.append(([round(tin, 2)],
                                       round(tin + 0.2, 2), tw_style))
                continue
            rng = _twr.Random(int(a_tw * 1000) % 99991)
            # 1) per-char nominal step (human jitter + punctuation pause)
            steps = []
            for ch in txt:
                s = 0.085 * (0.72 + 0.56 * rng.random())
                if ch in ",.:/-":                          # natural pause
                    s *= 2.3
                steps.append(s)
            # 2) scale so the whole line types within the window + tail
            avail = max(0.8, (b_ - a_tw) - 0.7)
            tot = sum(steps) or 1.0
            if tot > avail:
                steps = [s * avail / tot for s in steps]
            # 3) place chars + a caret that tracks the cursor
            x = float(BX + 32)
            tcur = a_tw + 0.15
            char_times: list[float] = []
            for ch, step in zip(txt, steps):
                cw = fo.getlength(ch) if fo else fs * 0.55
                if ch != " ":
                    out.append(
                        f"drawtext=fontfile={font}:text='{_esc(ch)}':"
                        f"fontsize={fs}:fontcolor=white:x={int(x)}:"
                        f"y={BY+34}:enable='gte(t,{tcur:.2f})*{win}'")
                    char_times.append(round(tcur, 2))
                out.append(
                    f"drawbox=x={int(x+cw+3)}:y={BY+40}:w=6:h=64:"
                    f"color={hexc}:t=fill:"
                    f"enable='between(t,{tcur:.2f},{tcur+step:.2f})'")
                tcur += step
                x += cw
            # final blinking caret (holds to the end of the card)
            out.append(
                f"drawbox=x={int(x+3)}:y={BY+40}:w=6:h=64:color={hexc}:"
                f"t=fill:enable='gte(t,{tcur:.2f})*{win}*"
                f"lt(mod(t\\,1)\\,0.5)'")
            if type_events is not None and char_times:
                type_events.append((char_times, round(tcur, 2), tw_style))
            continue

        if kind == "portrait" and asset:
            # AI PORTRAIT HOLD: user spec is "thora zyada der screen per
            # rukhni chyia 4 to 6 sec" -- minimum 4s, ideally 4-6s, never
            # longer than the scene duration minus a tiny tail.  Env
            # overrides VIDLORE_AI_HOLD_MIN / VIDLORE_AI_HOLD_MAX let the
            # user fine-tune without code edits.
            try:
                _hold_min = float(os.environ.get(
                    "VIDLORE_AI_HOLD_MIN", "4.0"))
            except (TypeError, ValueError):
                _hold_min = 4.0
            try:
                _hold_max = float(os.environ.get(
                    "VIDLORE_AI_HOLD_MAX", "6.0"))
            except (TypeError, ValueError):
                _hold_max = 6.0
            _hold_max = max(_hold_min, _hold_max)
            # ideal hold = clamp(scene duration - 0.4 tail, hold_min, hold_max)
            _ideal_hold = max(_hold_min, min(d - 0.4, _hold_max))
            b_ = min(a + _ideal_hold, emax)
            if b_ - a < 1.0:
                continue
            win = f"between(t,{a:.2f},{b_:.2f})"
            fade_out_dur = 0.45
            fade_out_start = b_ - fade_out_dur

            # NEW (post-redesign): if the renderer produced a LAYERED card
            # via _render_name_reveal_card -> portrait_NNN_bg.png / _pic /
            # _name with a .manifest.txt, use the cinematic name_reveal
            # animation path (left feathered portrait, dark scrim, medium
            # typography on the right, upward-rise motion).
            # Fall back to the OLD full-screen overlay only when the
            # manifest is missing (rare: layered render failed).
            manifest_path = workdir / asset.replace(".png", ".manifest.txt")
            if asset.endswith(".png") and manifest_path.is_file():
                try:
                    layer_names = [ln.strip() for ln in
                                   manifest_path.read_text(
                                       encoding="utf-8").splitlines()
                                   if ln.strip()]
                except Exception:                           # noqa: BLE001
                    layer_names = []
                if len(layer_names) >= 3:
                    bg_name, pic_name, name_name = layer_names[:3]
                    # bg scrim -- gentle vignette
                    stages.append(
                        f"movie='{bg_name}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={a:.2f}:d=0.55:alpha=1,"
                        f"fade=t=out:st={fade_out_start:.2f}:"
                        f"d={fade_out_dur:.2f}:alpha=1[prbg{gi}]")
                    stages.append(f"[{{CUR}}][prbg{gi}]overlay=x=0:y=0:"
                                  f"enable='{win}'[{{OUT}}]")
                    # PORTRAIT rises upward + fades in
                    p_t = a + 0.18
                    py = _mo.interp_ff(44.0, 0.0, p_t, 0.70, "out_cubic")
                    stages.append(
                        f"movie='{pic_name}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={p_t:.2f}:d=0.65:alpha=1,"
                        f"fade=t=out:st={fade_out_start:.2f}:"
                        f"d={fade_out_dur:.2f}:alpha=1[prpic{gi}]")
                    stages.append(f"[{{CUR}}][prpic{gi}]overlay=x=0:y='{py}':"
                                  f"eval=frame:enable='{win}'[{{OUT}}]")
                    # NAME staggered, smaller upward rise
                    n_t = a + 0.55
                    nwin = f"between(t,{n_t:.2f},{b_:.2f})"
                    ny = _mo.interp_ff(24.0, 0.0, n_t, 0.55, "out_cubic")
                    stages.append(
                        f"movie='{name_name}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={n_t:.2f}:d=0.55:alpha=1,"
                        f"fade=t=out:st={fade_out_start:.2f}:"
                        f"d={fade_out_dur:.2f}:alpha=1[prnm{gi}]")
                    stages.append(f"[{{CUR}}][prnm{gi}]overlay=x=0:y='{ny}':"
                                  f"eval=frame:enable='{nwin}'[{{OUT}}]")
                    gi += 1
                    continue

            # ---- FALLBACK: legacy full-screen overlay --------------- #
            # Only reached when the layered render failed (no manifest).
            stages.append(
                f"movie='{asset}',scale=1920:1080:force_original_"
                f"aspect_ratio=increase,crop=1920:1080,setsar=1,"
                f"format=rgba,loop=loop=-1:size=1,setpts=N/{FPS}/TB,"
                f"fade=t=in:st={a:.2f}:d=0.4:alpha=1,"
                f"fade=t=out:st={b_ - 0.4:.2f}:d=0.4:alpha=1[gp{gi}]")
            stages.append(
                f"[{{CUR}}][gp{gi}]overlay=0:0:enable='{win}'[{{OUT}}]")
            gi += 1
            nm = _dt(_card_text(raw)[:22])
            pwin = f"between(t,{a:.2f},{b_:.2f})"
            pa = (
                f"if(lt(t,{a:.2f}),0,"
                f"if(lt(t,{a + 0.30:.2f}),(t-{a:.2f})/0.30,"
                f"if(lt(t,{b_ - 0.35:.2f}),1,"
                f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.35,0))))"
            )
            post.append(
                f"drawbox=x=0:y=0:w=iw:h=240:color=black@0.45:t=fill:"
                f"enable='{pwin}'")
            post.append(
                f"drawtext=fontfile={font}:text='{nm}':fontsize=132:"
                "fontcolor=white:borderw=6:bordercolor=black@0.9:"
                "shadowx=4:shadowy=5:x=(w-text_w)/2:y=46:"
                f"alpha='{pa}':enable='{pwin}'")
            continue

        if kind == "chart" and (_card_text(raw) or body):
            # A "chart" scene == CUT TO THE DATA: hold it for ~the whole
            # scene (appear early, small tail) so there's always room to
            # fill + count + land — not a brief 62%-in card that short
            # scenes skip entirely.
            ac = start + 0.40
            b_ = min(start + d - 0.20, emax)
            if b_ - ac < 1.6:
                continue
            a = ac
            win = f"between(t,{a:.2f},{b_:.2f})"
            frac = _pct_of(raw)
            pctN = max(1, int(round(frac * 100)))
            FILL = min(1.6, max(0.9, (b_ - a) * 0.5))
            L = a + FILL                       # bar full + number lands
            num_events.append((a, L))          # ticks while filling +
            #                                    impact when it completes
            # SHORT headline — clean by WORD, never by char count.
            # (_card_text hard-caps at 38 chars which sliced "1994"->"199"
            # mid-number; build it from whole words instead so it can
            # never garble.)
            _raw_h = re.sub(r"[^\w .,&%/\-]", " ",
                            _u.normalize("NFC", body or raw),
                            flags=re.UNICODE)
            head = " ".join(_raw_h.upper().split()[:6])
            hlines = textwrap.wrap(head, 22)[:2] if head else []
            (workdir / f"chart{i}_h.txt").write_text(
                _dt("\n".join(hlines)), encoding="utf-8")
            hfs = 50 if (hlines and max(len(x)
                         for x in hlines) <= 20) else 40
            BX, BY, BW, BH = 905, 250, 132, 560
            # ease-out cubic fill (momentum, then a smooth settle)
            pe = (f"(1-pow(1-min(1\\,max(0\\,"
                  f"(t-{a:.2f})/{FILL:.2f}))\\,3))")
            fh = f"({BH}*{frac:.4f}*{pe})"
            fy = f"({BY}+{BH}-{fh})"
            nx = BX + BW + 70
            out.append(  # CINEMATIC data scrim — footage stays ghosted
                # behind (a deliberate "cut to the data" beat), NOT a
                # dead-black PowerPoint slate. This was the single biggest
                # "weak motion-graphics" tell.
                f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.66:t=fill:"
                f"enable='{win}'")
            out.append(
                f"drawgrid=width=160:height=160:thickness=1:"
                f"color=white@0.035:enable='{win}'")
            if hlines:
                out.append(  # premium headline — soft shadow, accent rule
                    f"drawtext=fontfile={font}:textfile=chart{i}_h.txt:"
                    f"expansion=none:fontsize={hfs}:fontcolor=white:"
                    "shadowcolor=black@0.55:shadowx=0:shadowy=4:"
                    f"line_spacing=12:x=(w-text_w)/2:y=92:enable='{win}'")
                out.append(  # short accent underline under the headline
                    f"drawbox=x=(iw-150)/2:y={92 + hfs * len(hlines) + 26}:"
                    f"w=150:h=4:color=0xF5C518:t=fill:enable='{win}'")
            out.append(  # recessed track (the empty bar) — soft, premium
                f"drawbox=x={BX}:y={BY}:w={BW}:h={BH}:"
                f"color=white@0.10:t=fill:enable='{win}'")
            out.append(  # soft accent glow halo behind the rising fill
                f"drawbox=x={BX - 12}:y={fy}:w={BW + 24}:h={fh}:"
                f"color=0xF5C518@0.20:t=fill:enable='{win}'")
            out.append(  # the accent fill, rising with momentum
                f"drawbox=x={BX}:y={fy}:w={BW}:h={fh}:"
                f"color=0xF5C518:t=fill:enable='{win}'")
            out.append(  # bright leading cap on the fill (premium edge)
                f"drawbox=x={BX}:y={fy}:w={BW}:h=6:"
                f"color=white@0.85:t=fill:"
                f"enable='between(t,{a:.2f},{L:.2f})'")
            out.append(  # hairline track outline + accent baseline
                f"drawbox=x={BX}:y={BY}:w={BW}:h={BH}:"
                f"color=white@0.30:t=2:enable='{win}'")
            out.append(
                f"drawbox=x={BX - 14}:y={BY + BH}:w={BW + 28}:h=3:"
                f"color=0xF5C518@0.8:t=fill:enable='{win}'")
            # big % counts up IN SYNC with the fill, then lands
            eif = (
                f"%{{eif\\:floor({pctN}*(1-pow(1-"
                f"clip((t-{a:.2f})/{FILL:.2f}\\,0\\,1)\\,3)))\\:d}}"
            )
            ra = (f"if(lt(t,{a:.2f}),0,"
                  f"if(lt(t,{a + 0.12:.2f}),(t-{a:.2f})/0.12,1))")
            out.append(  # premium clean type — bold white, soft shadow
                f"drawtext=fontfile={font}:expansion=normal:text='{eif}':"
                "fontsize=140:fontcolor=white:"
                "shadowcolor=black@0.55:shadowx=0:shadowy=5:"
                f"x={nx}:y=(h-text_h)/2:alpha='{ra}':"
                f"enable='between(t,{a:.2f},{L:.2f})'")
            lwin = f"between(t,{L:.2f},{b_:.2f})"
            la = (
                f"if(lt(t,{L:.2f}),0,"
                f"if(lt(t,{L + 0.10:.2f}),(t-{L:.2f})/0.10,"
                f"if(lt(t,{b_ - 0.35:.2f}),1,"
                f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.35,0))))"
            )
            yb = (
                f"(h-text_h)/2 + if(lt(t,{L:.2f}),0,"
                f"if(lt(t,{L + 0.10:.2f}),-16*((t-{L:.2f})/0.10),"
                f"if(lt(t,{L + 0.30:.2f}),"
                f"-16*(1-((t-{L + 0.10:.2f})/0.20)),0)))"
            )
            out.append(  # soft accent glow bloom behind the landed value
                f"drawtext=fontfile={font}:text='{pctN}%':"
                "expansion=none:fontsize=150:fontcolor=0xF5C518@0.0:"
                "borderw=22:bordercolor=0xF5C518@0.40:"
                f"x={nx}:y='{yb}':alpha='{la}':enable='{lwin}'")
            out.append(  # final % value — big, clean white, accent shadow
                f"drawtext=fontfile={font}:text='{pctN}%':"
                "expansion=none:fontsize=150:fontcolor=white:"
                "shadowcolor=black@0.6:shadowx=0:shadowy=6:"
                f"x={nx}:y='{yb}':alpha='{la}':enable='{lwin}'")
            continue

        if kind == "callout":
            b_ = min(a + max(2.0, min(d - 0.4, 3.4)), emax)
            if b_ - a < 1.0:
                continue
            win = f"between(t,{a:.2f},{b_:.2f})"
            # VISION-TARGETED ANIMATED ARROW. build_graphic_images asked a
            # vision model WHERE the subject is in the footage and rendered
            # a manifest: [label-chip bg, arrow-segment a0..aN]. We fade the
            # chip in, then reveal the segments in sequence so the curved
            # arrow DRAWS ON and lands exactly on the real object (the last
            # layer carries the arrowhead + the accent pulse on the target).
            manifest_path = (
                workdir / asset.replace(".png", ".manifest.txt")
                if asset.endswith(".png") else None)
            layers = []
            if manifest_path and manifest_path.is_file():
                try:
                    layers = [ln.strip() for ln in
                              manifest_path.read_text(encoding="utf-8")
                              .splitlines() if ln.strip()]
                except Exception:                              # noqa: BLE001
                    layers = []
            if len(layers) >= 3:
                fout = b_ - 0.40
                bg_name = layers[0]
                seg = layers[1:]
                stages.append(                      # label chip lands first
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={a:.2f}:d=0.35:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.40:alpha=1[cobg{gi}]")
                stages.append(f"[{{CUR}}][cobg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                t0 = a + 0.45                       # arrow starts after chip
                draw_dur = max(0.35, min(0.70, (b_ - 0.6 - t0)))
                step = draw_dur / max(1, len(seg))
                for k, sl in enumerate(seg):
                    ks = t0 + k * step
                    # one segment shows at a time (clean growth); the FINAL
                    # layer (arrowhead + target pulse) holds to the end.
                    ke = b_ if k == len(seg) - 1 else (t0 + (k + 1) * step)
                    lwin = f"between(t,{ks:.2f},{ke:.2f})"
                    fo = (f",fade=t=out:st={fout:.2f}:d=0.40:alpha=1"
                          if k == len(seg) - 1 else "")
                    stages.append(
                        f"movie='{sl}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={ks:.2f}:d=0.04:alpha=1{fo}"
                        f"[coa{gi}_{k}]")
                    stages.append(f"[{{CUR}}][coa{gi}_{k}]overlay=0:0:"
                                  f"enable='{lwin}'[{{OUT}}]")
                gi += 1
                continue
            # FALLBACK: vision found nothing -> a clean label chip only,
            # never a random arrow pointing at nothing.
            if _card_text(raw):
                lbl = _dt(_card_text(raw)[:24])
                # IMP_005 — SAFE-ZONE placement. On a portrait/reaction shot
                # the subject's face owns the upper-center frame, so the two
                # UPPER fallback slots (1150,320) / (130,250) would land a
                # label across the face — amateur. For those shot types pin
                # the chip to the bottom-left quadrant (documentary lower-
                # third convention). Other shots keep the rotating variety.
                if shot_type in ("portrait", "reaction"):
                    LX, LY = (110, 760)              # bottom-left, clear of face
                else:
                    LX, LY = [(1150, 320), (110, 760),
                              (130, 250), (1200, 760)][i % 4]
                out.append(
                    f"drawtext=fontfile={font}:text='{lbl}':fontsize=46:"
                    "fontcolor=black:box=1:boxcolor=white@0.96:boxborderw=24:"
                    f"x={LX}:y={LY}:enable='{win}'")
            continue

        if kind == "progress_bar" and asset.startswith("pb_") \
                and asset.endswith(".png"):
            # VERTICAL % BAR — bg fades in, then the gold fill RISES while
            # the % counts up (manifest fill-step layers, one at a time).
            pb_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - pb_start < 1.6:
                continue
            win = f"between(t,{pb_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            if len(ly) >= 2:
                stages.append(
                    f"movie='{ly[0]}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={pb_start:.2f}:d=0.45:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[pbbg{gi}]")
                stages.append(f"[{{CUR}}][pbbg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                seg = ly[1:]
                t0 = pb_start + 0.40
                dur = max(0.6, min(1.1, b_ - 0.7 - t0))
                step = dur / max(1, len(seg))
                for k, nm in enumerate(seg):
                    ks = t0 + k * step
                    ke = b_ if k == len(seg) - 1 else (t0 + (k + 1) * step)
                    fo = (f",fade=t=out:st={fout:.2f}:d=0.45:alpha=1"
                          if k == len(seg) - 1 else "")
                    stages.append(
                        f"movie='{nm}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={ks:.2f}:d=0.03:alpha=1{fo}"
                        f"[pb{gi}_{k}]")
                    stages.append(f"[{{CUR}}][pb{gi}_{k}]overlay=0:0:"
                                  f"enable='between(t,{ks:.2f},{ke:.2f})'"
                                  f"[{{OUT}}]")
                gi += 1
                continue

        if kind == "line_chart" and asset.startswith("lc_") \
                and asset.endswith(".png"):
            # LINE CHART — opaque navy data card; gentle fade + slow push.
            lc_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - lc_start < 1.6:
                continue
            win = f"between(t,{lc_start:.2f},{b_:.2f})"
            zexpr = (f"1920*(1.0+0.04*min(1\\,max(0\\,"
                     f"(t-{lc_start:.2f})/{b_-lc_start:.2f})))")
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,scale=w='{zexpr}':h=-2:eval=frame,"
                f"crop=1920:1080,setsar=1,"
                f"fade=t=in:st={lc_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1[lc{gi}]")
            stages.append(f"[{{CUR}}][lc{gi}]overlay=0:0:"
                          f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "section_title" and asset.startswith("sec_") \
                and asset.endswith(".png"):
            # SECTION TITLE over footage — scrim fades in, title rises in.
            se_start = start + 0.35
            b_ = min(start + d - 0.35, emax)
            if b_ - se_start < 1.4:
                continue
            win = f"between(t,{se_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            if len(ly) >= 2:
                stages.append(
                    f"movie='{ly[0]}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={se_start:.2f}:d=0.50:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[sebg{gi}]")
                stages.append(f"[{{CUR}}][sebg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                tt = se_start + 0.20
                lwin = f"between(t,{tt:.2f},{b_:.2f})"
                mv, ov = _motion_layer(
                    ly[1], f"sect{gi}", tt, 0.45, fout, 0.45, lwin, rise=22)
                stages.append(mv)
                stages.append(ov)
                gi += 1
                continue

        if kind == "diagram_labels" and asset.startswith("dl_") \
                and asset.endswith(".png"):
            # LABELED DIAGRAM — footage bg, then each leader-line label
            # pops in one-by-one (staggered draw-on).
            dl_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - dl_start < 1.6:
                continue
            win = f"between(t,{dl_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            if len(ly) >= 2:
                stages.append(
                    f"movie='{ly[0]}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={dl_start:.2f}:d=0.45:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[dlbg{gi}]")
                stages.append(f"[{{CUR}}][dlbg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                t_cur = dl_start + 0.45
                step = min(0.45, max(0.25,
                           (b_ - 0.8 - t_cur) / max(1, len(ly) - 1)))
                for k, nm in enumerate(ly[1:]):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    stages.append(
                        f"movie='{nm}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={t_cur:.2f}:d=0.28:alpha=1,"
                        f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1"
                        f"[dl{gi}_{k}]")
                    stages.append(f"[{{CUR}}][dl{gi}_{k}]overlay=0:0:"
                                  f"enable='{lwin}'[{{OUT}}]")
                    t_cur += step
                gi += 1
                continue

        if kind == "statement" and asset.startswith("st_") \
                and asset.endswith(".png"):
            # LIGHT BIG-STATEMENT — opaque light card; bg fades in, then the
            # narration reveals line-by-line (calm kinetic build).
            st_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - st_start < 1.6:
                continue
            win = f"between(t,{st_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            if len(ly) >= 2:
                bgn = ly[0]
                line_layers = ly[1:]
                stages.append(
                    f"movie='{bgn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={st_start:.2f}:d=0.50:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[stbg{gi}]")
                stages.append(f"[{{CUR}}][stbg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                t_cur = st_start + 0.45
                budget = max(0.9, (b_ - 0.8 - t_cur))
                # PHASE-1 LEGIBILITY (2026-06-05): the old reveal rose each text
                # line/word 16 px with only ~0.34 s stagger, so during the
                # cascade adjacent lines/words crossed into each other and a
                # mid-animation frame read as scrambled, illegible text
                # ("SIXDECA D / ZER ORPO"). Fix: reveal each line as a NEAR-STATIC
                # fade in its FINAL position (rise 16->3) with a gentler fade, so
                # every frame shows correctly-placed letters (translucent at
                # most, never displaced) and the block then HOLDS fully readable
                # until fade-out. "Simpler animation that preserves legibility."
                step = min(0.34, budget / max(1, len(line_layers)))
                for k, ll in enumerate(line_layers):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        ll, f"stl{gi}_{k}", t_cur, 0.42,
                        fout, 0.45, lwin, rise=3)
                    stages.append(mv)
                    stages.append(ov)
                    t_cur += step
                gi += 1
                continue

        if kind == "framed_insert" and asset.startswith("fi_") \
                and asset.endswith(".png"):
            # ANNOTATED FRAMED INSERT — dark bg, then the tilted gold-framed
            # footage card SCALES + fades in, the angular gold leaders draw
            # on, and the caption chip drops in last (manifest layers).
            fi_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - fi_start < 1.6:
                continue
            win = f"between(t,{fi_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            # ly = [bg, anchor-frame, insight_1, insight_2, ...]
            if len(ly) >= 3:
                bgn, cardn = ly[0], ly[1]
                insights = ly[2:]
                stages.append(
                    f"movie='{bgn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={fi_start:.2f}:d=0.45:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[fibg{gi}]")
                stages.append(f"[{{CUR}}][fibg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                # anchor image settles: scales in (1.06 -> 1.00) with a fade
                cs = fi_start + 0.15
                zc = (f"1920*(1.06-0.06*min(1\\,max(0\\,"
                      f"(t-{cs:.2f})/0.45)))")
                stages.append(
                    f"movie='{cardn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,scale=w='{zc}':h=-2:eval=frame,"
                    f"crop=1920:1080,setsar=1,"
                    f"fade=t=in:st={cs:.2f}:d=0.40:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[ficd{gi}]")
                stages.append(f"[{{CUR}}][ficd{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                # each insight (chip + its connector) reveals one-by-one with
                # a small upward rise — guided, directed cinematic build.
                t_cur = cs + 0.50
                budget = max(0.9, (b_ - 0.8 - t_cur))
                step = min(0.46, budget / max(1, len(insights)))
                for k, nm in enumerate(insights):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        nm, f"fi{gi}_{k}", t_cur, 0.34,
                        fout, 0.45, lwin, rise=14)
                    stages.append(mv)
                    stages.append(ov)
                    t_cur += step
                gi += 1
                continue

        if kind == "newspaper" and asset.startswith("np_") \
                and asset.endswith(".png"):
            # NEWSPAPER — opaque cream paper PNG, no footage bleed.
            np_start = start + 0.45
            b_ = min(start + d - 0.45, emax)
            if b_ - np_start < 2.0:
                continue
            win = f"between(t,{np_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={np_start:.2f}:d=0.60:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1[np{gi}]")
            stages.append(
                f"[{{CUR}}][np{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "terminal" and asset.startswith("tm_") \
                and asset.endswith(".png"):
            # TERMINAL — opaque CRT phosphor green-on-black.
            tm_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - tm_start < 1.8:
                continue
            win = f"between(t,{tm_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={tm_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1[tm{gi}]")
            stages.append(
                f"[{{CUR}}][tm{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "military_hud" and asset.startswith("mh_") \
                and asset.endswith(".png"):
            # MILITARY HUD — TRANSPARENT frame overlay (footage shows
            # through the centre reticle area).
            mh_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - mh_start < 1.5:
                continue
            win = f"between(t,{mh_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mh_start:.2f}:d=0.45:alpha=1,"
                f"fade=t=out:st={b_-0.40:.2f}:d=0.40:alpha=1[mh{gi}]")
            stages.append(
                f"[{{CUR}}][mh{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "sms_text" \
                and asset.startswith("sms_") \
                and asset.endswith(".png"):
            ss_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - ss_start < 1.8:
                continue
            win = f"between(t,{ss_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ss_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[sms{gi}]")
            stages.append(
                f"[{{CUR}}][sms{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "postmark" \
                and asset.startswith("pm_") \
                and asset.endswith(".png"):
            pm_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - pm_start < 1.8:
                continue
            win = f"between(t,{pm_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={pm_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[pm{gi}]")
            stages.append(
                f"[{{CUR}}][pm{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "quote_stream" \
                and asset.startswith("qs_") \
                and asset.endswith(".png"):
            qs_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - qs_start < 2.0:
                continue
            win = f"between(t,{qs_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={qs_start:.2f}:d=0.60:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[qs{gi}]")
            stages.append(
                f"[{{CUR}}][qs{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "mini_timeline" \
                and asset.startswith("mt_") \
                and asset.endswith(".png"):
            mt_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - mt_start < 1.8:
                continue
            win = f"between(t,{mt_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mt_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[mt{gi}]")
            stages.append(
                f"[{{CUR}}][mt{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "clock_face" \
                and asset.startswith("cf2_") \
                and asset.endswith(".png"):
            cs_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cs_start < 1.8:
                continue
            win = f"between(t,{cs_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cs_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[cf2{gi}]")
            stages.append(
                f"[{{CUR}}][cf2{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "network_graph" \
                and asset.startswith("ng_") \
                and asset.endswith(".png"):
            # SYMBOLIC MOTION — if layered, the network FORMS: nodes pop
            # in one-by-one (eased), then the connections DRAW IN in
            # sequence (a pulse spreading through the network). Otherwise
            # fall back to the single static composite.
            ng_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - ng_start < 2.2:
                continue
            win = f"between(t,{ng_start:.2f},{b_:.2f})"
            fade_out_dur = 0.55
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(".png", ".manifest.txt")
            layer_names = []
            if manifest_path.is_file():
                try:
                    layer_names = [ln.strip() for ln in
                                   manifest_path.read_text(
                                       encoding="utf-8").splitlines()
                                   if ln.strip()]
                except Exception:
                    layer_names = []

            if len(layer_names) >= 3:
                bg_name = layer_names[0]
                node_layers = [ln for ln in layer_names[1:]
                               if re.search(r"_n\d+\.png$", ln)]
                edge_layers = [ln for ln in layer_names[1:]
                               if re.search(r"_e\d+\.png$", ln)]
                # bg
                stages.append(
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={ng_start:.2f}:d=0.65:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[ngbg{gi}]")
                stages.append(f"[{{CUR}}][ngbg{gi}]overlay=x=0:y=0:"
                              f"enable='{win}'[{{OUT}}]")
                # nodes pop in one-by-one (eased rise)
                t_cur = ng_start + 0.50
                # keep the whole build inside ~55% of the window
                budget = max(1.2, (b_ - 0.6 - t_cur))
                n_steps = max(1, len(node_layers) + len(edge_layers))
                step = min(0.30, budget / n_steps)
                for k, nl in enumerate(node_layers):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        nl, f"ngn{gi}_{k}", t_cur, 0.34,
                        fade_out_start, fade_out_dur, lwin, rise=18)
                    stages.append(mv)
                    stages.append(ov)
                    t_cur += step
                # connections draw in (quick sequential fades = spread)
                for k, el in enumerate(edge_layers):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    stages.append(
                        f"movie='{el}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={t_cur:.2f}:d=0.20:alpha=1,"
                        f"fade=t=out:st={fade_out_start:.2f}:"
                        f"d={fade_out_dur:.2f}:alpha=1[nge{gi}_{k}]")
                    stages.append(f"[{{CUR}}][nge{gi}_{k}]overlay=x=0:y=0:"
                                  f"enable='{lwin}'[{{OUT}}]")
                    t_cur += min(step, 0.18)
                gi += 1
                continue

            # fallback: single static composite
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ng_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[ng{gi}]")
            stages.append(
                f"[{{CUR}}][ng{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "status_indicator" \
                and asset.startswith("si2_") \
                and asset.endswith(".png"):
            stat_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - stat_start < 1.6:
                continue
            win = f"between(t,{stat_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={stat_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[si2{gi}]")
            stages.append(
                f"[{{CUR}}][si2{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "speedometer" \
                and asset.startswith("sp2_") \
                and asset.endswith(".png"):
            sp_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - sp_start < 1.8:
                continue
            win = f"between(t,{sp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sp_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[sp2{gi}]")
            stages.append(
                f"[{{CUR}}][sp2{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "film_slate" \
                and asset.startswith("fs_") \
                and asset.endswith(".png"):
            # FILM SLATE — TRANSPARENT centred slate.
            fs_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - fs_start < 1.8:
                continue
            win = f"between(t,{fs_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={fs_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[fs{gi}]")
            stages.append(
                f"[{{CUR}}][fs{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "speech_bubble" \
                and asset.startswith("spb_") \
                and asset.endswith(".png"):
            # SPEECH BUBBLE — TRANSPARENT comic-style.
            sb_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - sb_start < 1.6:
                continue
            win = f"between(t,{sb_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sb_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[spb{gi}]")
            stages.append(
                f"[{{CUR}}][spb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "compass_bearing" \
                and asset.startswith("cb_") \
                and asset.endswith(".png"):
            # COMPASS BEARING — TRANSPARENT centred.
            cb_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - cb_start < 1.6:
                continue
            win = f"between(t,{cb_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cb_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[cb{gi}]")
            stages.append(
                f"[{{CUR}}][cb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "stat_insight" \
                and asset.startswith("sti_") \
                and asset.endswith(".png"):
            # STAT INSIGHT — opaque data overlay.
            si_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - si_start < 1.8:
                continue
            win = f"between(t,{si_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={si_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[sti{gi}]")
            stages.append(
                f"[{{CUR}}][sti{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "id_card" \
                and asset.startswith("idc_") \
                and asset.endswith(".png"):
            # ID CARD — TRANSPARENT centred card. 0.70s fade-in.
            ic_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - ic_start < 2.0:
                continue
            win = f"between(t,{ic_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ic_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[idc{gi}]")
            stages.append(
                f"[{{CUR}}][idc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "email_screenshot" \
                and asset.startswith("em_") \
                and asset.endswith(".png"):
            # EMAIL SCREENSHOT — TRANSPARENT centred window.
            em_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - em_start < 2.0:
                continue
            win = f"between(t,{em_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={em_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[em{gi}]")
            stages.append(
                f"[{{CUR}}][em{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "call_log" \
                and asset.startswith("cll_") \
                and asset.endswith(".png"):
            # CALL LOG — TRANSPARENT centred card.
            cl_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - cl_start < 1.8:
                continue
            win = f"between(t,{cl_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cl_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[cll{gi}]")
            stages.append(
                f"[{{CUR}}][cll{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "receipt" \
                and asset.startswith("rcp_") \
                and asset.endswith(".png"):
            # RECEIPT — TRANSPARENT centred receipt. Slow 0.70s
            # fade-in (receipt "settles" onto the table).
            rc_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - rc_start < 2.0:
                continue
            win = f"between(t,{rc_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={rc_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[rcp{gi}]")
            stages.append(
                f"[{{CUR}}][rcp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "sticky_note" \
                and asset.startswith("sn_") \
                and asset.endswith(".png"):
            # STICKY NOTE — TRANSPARENT centred pinned card.
            sn_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - sn_start < 1.6:
                continue
            win = f"between(t,{sn_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sn_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[sn{gi}]")
            stages.append(
                f"[{{CUR}}][sn{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "press_release" \
                and asset.startswith("prl_") \
                and asset.endswith(".png"):
            # PRESS RELEASE — TRANSPARENT centred letterhead.
            # Slow 0.85s fade-in (formal documents land slow).
            pr_start = start + 0.40
            b_ = min(start + d - 0.55, emax)
            if b_ - pr_start < 2.2:
                continue
            win = f"between(t,{pr_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={pr_start:.2f}:d=0.85:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[prl{gi}]")
            stages.append(
                f"[{{CUR}}][prl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "radio_dial" \
                and asset.startswith("rd_") \
                and asset.endswith(".png"):
            # RADIO DIAL — TRANSPARENT bottom card. 0.55s fade-in.
            rd_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - rd_start < 1.8:
                continue
            win = f"between(t,{rd_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={rd_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[rd{gi}]")
            stages.append(
                f"[{{CUR}}][rd{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "vertical_bar_chart" \
                and asset.startswith("vbc_") \
                and asset.endswith(".png"):
            # VERTICAL BAR CHART — opaque data overlay.
            vbc_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - vbc_start < 2.0:
                continue
            win = f"between(t,{vbc_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={vbc_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[vbc{gi}]")
            stages.append(
                f"[{{CUR}}][vbc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "heatmap_grid" \
                and asset.startswith("hm_") \
                and asset.endswith(".png"):
            # HEATMAP — opaque data overlay. 0.65s fade-in.
            hm_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - hm_start < 1.8:
                continue
            win = f"between(t,{hm_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={hm_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[hm{gi}]")
            stages.append(
                f"[{{CUR}}][hm{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "map_pin_cluster" \
                and asset.startswith("mpc_") \
                and asset.endswith(".png"):
            # MAP PIN CLUSTER — opaque map overlay.
            mp_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - mp_start < 2.0:
                continue
            win = f"between(t,{mp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mp_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[mpc{gi}]")
            stages.append(
                f"[{{CUR}}][mpc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "numerical_ratio" \
                and asset.startswith("nr_") \
                and asset.endswith(".png"):
            # NUMERICAL RATIO — opaque data overlay.
            nr_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - nr_start < 1.6:
                continue
            win = f"between(t,{nr_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={nr_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[nr{gi}]")
            stages.append(
                f"[{{CUR}}][nr{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "document_stack" \
                and asset.startswith("ds_") \
                and asset.endswith(".png"):
            # DOCUMENT STACK — TRANSPARENT pinned-stack overlay.
            # Slow 0.80s fade-in (the documents "settle" on the
            # desk).
            ds_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - ds_start < 2.0:
                continue
            win = f"between(t,{ds_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ds_start:.2f}:d=0.80:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[ds{gi}]")
            stages.append(
                f"[{{CUR}}][ds{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "currency_stat" \
                and asset.startswith("cs_") \
                and asset.endswith(".png"):
            # CURRENCY STAT — opaque data overlay. 0.65s fade-in.
            cs_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - cs_start < 1.6:
                continue
            win = f"between(t,{cs_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cs_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[cs{gi}]")
            stages.append(
                f"[{{CUR}}][cs{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "era_banner" \
                and asset.startswith("eb_") \
                and asset.endswith(".png"):
            # ERA BANNER — opaque full-screen overlay. A beat, not a
            # backdrop: flashes ~3s for the era shift, then dissolves to
            # the footage (used to hold the whole scene).
            er_start = start + 0.40
            b_ = min(er_start + 3.0, start + d - 0.40, emax)
            if b_ - er_start < 1.8:
                continue
            win = f"between(t,{er_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={er_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[eb{gi}]")
            stages.append(
                f"[{{CUR}}][eb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "headline_crawl" \
                and asset.startswith("hc_") \
                and asset.endswith(".png"):
            # HEADLINE CRAWL — TRANSPARENT bottom ticker. Slide
            # the long crawl line LEFT during the window to give
            # the illusion of motion (uses overlay x= expression).
            hc_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - hc_start < 2.0:
                continue
            win = f"between(t,{hc_start:.2f},{b_:.2f})"
            # the crawl strip is rendered statically; we slide the
            # whole overlay LEFT 200px over the window so it looks
            # like the ticker is moving.
            sd_ = (b_ - hc_start) - 0.30
            x_expr = (f"-200*min(1\\,max(0\\,"
                      f"(t-{hc_start + 0.30:.2f})/{sd_:.2f}))")
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={hc_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[hc{gi}]")
            stages.append(
                f"[{{CUR}}][hc{gi}]overlay=x='{x_expr}':y=0:"
                f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "studio_two_shot" \
                and asset.startswith("ts_") \
                and asset.endswith(".png"):
            # STUDIO TWO-SHOT — TRANSPARENT split-screen interview
            # frame. 0.65s fade-in.
            ts_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - ts_start < 2.0:
                continue
            win = f"between(t,{ts_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ts_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[ts{gi}]")
            stages.append(
                f"[{{CUR}}][ts{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "score_display" \
                and asset.startswith("sd_") \
                and asset.endswith(".png"):
            # SCORE DISPLAY — opaque data overlay. 0.65s fade-in.
            sd_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - sd_start < 1.8:
                continue
            win = f"between(t,{sd_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sd_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[sd{gi}]")
            stages.append(
                f"[{{CUR}}][sd{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "polaroid_stack" \
                and asset.startswith("ps_") \
                and asset.endswith(".png"):
            # POLAROID STACK — TRANSPARENT centred pinned cards.
            # Slow 0.75s fade-in (intimate "settles in").
            ps_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - ps_start < 2.0:
                continue
            win = f"between(t,{ps_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ps_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[ps{gi}]")
            stages.append(
                f"[{{CUR}}][ps{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "crosshair_lock" \
                and asset.startswith("cl_") \
                and asset.endswith(".png"):
            # CROSSHAIR LOCK — TRANSPARENT tactical overlay. Quick
            # 0.40s fade-in (target-lock is fast).
            cl_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - cl_start < 1.5:
                continue
            win = f"between(t,{cl_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cl_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[cl{gi}]")
            stages.append(
                f"[{{CUR}}][cl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "hashtag_trend" \
                and asset.startswith("ht_") \
                and asset.endswith(".png"):
            # HASHTAG TREND — TRANSPARENT centred card. Quick
            # 0.45s fade-in.
            ht_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - ht_start < 1.6:
                continue
            win = f"between(t,{ht_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ht_start:.2f}:d=0.45:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[ht{gi}]")
            stages.append(
                f"[{{CUR}}][ht{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "globe_map" \
                and asset.startswith("glb_") \
                and asset.endswith(".png"):
            # GLOBE — opaque world-map overlay. 0.70s fade-in.
            gl_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - gl_start < 2.0:
                continue
            win = f"between(t,{gl_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={gl_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[glb{gi}]")
            stages.append(
                f"[{{CUR}}][glb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "family_tree_3gen" \
                and asset.startswith("ft3_") \
                and asset.endswith(".png"):
            # FAMILY TREE 3-GEN — opaque data overlay, slow
            # 0.75s fade-in (the eye needs time to scan 3 levels).
            ft_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - ft_start < 2.4:
                continue
            win = f"between(t,{ft_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ft_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[ft{gi}]")
            stages.append(
                f"[{{CUR}}][ft{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "verdict_stamp" \
                and asset.startswith("vd_") \
                and asset.endswith(".png"):
            # VERDICT STAMP — opaque data overlay. SLOW 0.85s
            # fade-in (the stamp "comes down hard"). Long hold.
            vd_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - vd_start < 2.0:
                continue
            win = f"between(t,{vd_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={vd_start:.2f}:d=0.85:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[vd{gi}]")
            stages.append(
                f"[{{CUR}}][vd{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "mini_bio" \
                and asset.startswith("mb_") \
                and asset.endswith(".png"):
            # MINI BIO — TRANSPARENT bottom-right small card.
            # Quick 0.40s fade-in.
            mb_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - mb_start < 1.6:
                continue
            win = f"between(t,{mb_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mb_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[mb{gi}]")
            stages.append(
                f"[{{CUR}}][mb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "pull_quote_portrait" \
                and asset.startswith("pqp_") \
                and asset.endswith(".png"):
            # PULL QUOTE WITH PORTRAIT — TRANSPARENT vignette
            # overlay. Slow 0.70s fade-in (the quote "lands").
            pq_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - pq_start < 2.0:
                continue
            win = f"between(t,{pq_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={pq_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[pqp{gi}]")
            stages.append(
                f"[{{CUR}}][pqp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "spotlight" \
                and asset.startswith("spt_") \
                and asset.endswith(".png"):
            # SPOTLIGHT — TRANSPARENT darken-mask. 0.55s fade-in
            # (the dim "settles"), long hold.
            sp_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - sp_start < 1.6:
                continue
            win = f"between(t,{sp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sp_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[spt{gi}]")
            stages.append(
                f"[{{CUR}}][spt{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "calendar_grid" \
                and asset.startswith("cal_") \
                and asset.endswith(".png"):
            # CALENDAR GRID — opaque data overlay. 0.65s fade-in.
            cg_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cg_start < 2.0:
                continue
            win = f"between(t,{cg_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cg_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[cal{gi}]")
            stages.append(
                f"[{{CUR}}][cal{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "disclaimer" \
                and asset.startswith("dcl_") \
                and asset.endswith(".png"):
            # DISCLAIMER — TRANSPARENT bottom card. Slow 0.65s
            # fade-in, long hold (viewer must read).
            dc_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - dc_start < 1.8:
                continue
            win = f"between(t,{dc_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={dc_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[dcl{gi}]")
            stages.append(
                f"[{{CUR}}][dcl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "caution_tape" \
                and asset.startswith("ct_") \
                and asset.endswith(".png"):
            # CAUTION TAPE — TRANSPARENT centre band. Quick 0.40s
            # fade-in (warnings should LAND).
            ct_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - ct_start < 1.6:
                continue
            win = f"between(t,{ct_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ct_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[ct{gi}]")
            stages.append(
                f"[{{CUR}}][ct{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "award_ribbon" \
                and asset.startswith("awd_") \
                and asset.endswith(".png"):
            # AWARD RIBBON — opaque data overlay. SLOW 0.85s
            # fade-in (the medal "settles" into frame). Long hold.
            aw_start = start + 0.40
            b_ = min(start + d - 0.55, emax)
            if b_ - aw_start < 2.2:
                continue
            win = f"between(t,{aw_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={aw_start:.2f}:d=0.85:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[awd{gi}]")
            stages.append(
                f"[{{CUR}}][awd{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "sfx_cue" \
                and asset.startswith("sfx_") \
                and asset.endswith(".png"):
            # SFX CUE — TRANSPARENT bottom-centre small tag.
            # Fast 0.30s fade-in (cues are quick).
            sf_start = start + 0.20
            b_ = min(start + d - 0.35, emax)
            if b_ - sf_start < 1.0:
                continue
            win = f"between(t,{sf_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sf_start:.2f}:d=0.30:alpha=1,"
                f"fade=t=out:st={b_-0.40:.2f}:d=0.40:alpha=1"
                f"[sfx{gi}]")
            stages.append(
                f"[{{CUR}}][sfx{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "footnote" \
                and asset.startswith("fn_") \
                and asset.endswith(".png"):
            # FOOTNOTE — TRANSPARENT bottom-right small tag.
            # Subtle 0.30s fade-in.
            ft_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - ft_start < 1.0:
                continue
            win = f"between(t,{ft_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ft_start:.2f}:d=0.30:alpha=1,"
                f"fade=t=out:st={b_-0.40:.2f}:d=0.40:alpha=1"
                f"[fn{gi}]")
            stages.append(
                f"[{{CUR}}][fn{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "tally_counter" \
                and asset.startswith("tc_") \
                and asset.endswith(".png"):
            # TALLY COUNTER — opaque data overlay. 0.55s fade-in
            # and long hold so the figure lands.
            tc_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - tc_start < 1.6:
                continue
            win = f"between(t,{tc_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={tc_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[tc{gi}]")
            stages.append(
                f"[{{CUR}}][tc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "gps_stamp" \
                and asset.startswith("gps_") \
                and asset.endswith(".png"):
            # GPS STAMP — TRANSPARENT bottom-left small card. Quick
            # 0.35s fade-in (location stamps should land fast).
            gp_start = start + 0.25
            b_ = min(start + d - 0.40, emax)
            if b_ - gp_start < 1.4:
                continue
            win = f"between(t,{gp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={gp_start:.2f}:d=0.35:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[gps{gi}]")
            stages.append(
                f"[{{CUR}}][gps{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "microscope_inset" \
                and asset.startswith("mi_") \
                and asset.endswith(".png"):
            # MICROSCOPE INSET — TRANSPARENT top-right magnified
            # lens. 0.55s fade-in (the lens "rises" into view).
            mi_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - mi_start < 1.6:
                continue
            win = f"between(t,{mi_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mi_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[mi{gi}]")
            stages.append(
                f"[{{CUR}}][mi{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "relationship_tree" \
                and asset.startswith("rt_") \
                and asset.endswith(".png"):
            # RELATIONSHIP TREE — opaque data overlay. 0.65s
            # fade-in matching the data family.
            rt_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - rt_start < 2.0:
                continue
            win = f"between(t,{rt_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={rt_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[rt{gi}]")
            stages.append(
                f"[{{CUR}}][rt{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "step_indicator" \
                and asset.startswith("si_") \
                and asset.endswith(".png"):
            # STEP INDICATOR — TRANSPARENT top strip. Quick 0.40s
            # fade-in (it's a wayfinding signal — should land fast).
            si_start = start + 0.25
            b_ = min(start + d - 0.40, emax)
            if b_ - si_start < 1.4:
                continue
            win = f"between(t,{si_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={si_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[si{gi}]")
            stages.append(
                f"[{{CUR}}][si{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "did_you_know" \
                and asset.startswith("dyk_") \
                and asset.endswith(".png"):
            # DID YOU KNOW — TRANSPARENT centred trivia card.
            # 0.55s fade-in (the card "pops" but not too fast),
            # long hold so the fact lands.
            dy_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - dy_start < 1.8:
                continue
            win = f"between(t,{dy_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={dy_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[dyk{gi}]")
            stages.append(
                f"[{{CUR}}][dyk{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "voice_memo" \
                and asset.startswith("vm_") \
                and asset.endswith(".png"):
            # VOICE MEMO — TRANSPARENT bottom-third tape deck.
            # 0.50s fade-in, long hold for transcript reading.
            vm_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - vm_start < 1.8:
                continue
            win = f"between(t,{vm_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={vm_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[vm{gi}]")
            stages.append(
                f"[{{CUR}}][vm{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "inscription" \
                and asset.startswith("ins_") \
                and asset.endswith(".png"):
            # INSCRIPTION — TRANSPARENT centred bronze plaque.
            # SLOW 0.85s fade-in (the camera SETTLES on the
            # plaque, doesn't snap to it). Long hold.
            in_start = start + 0.40
            b_ = min(start + d - 0.55, emax)
            if b_ - in_start < 2.2:
                continue
            win = f"between(t,{in_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={in_start:.2f}:d=0.85:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[ins{gi}]")
            stages.append(
                f"[{{CUR}}][ins{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "subtitle_translation" \
                and asset.startswith("sub_") \
                and asset.endswith(".png"):
            # SUBTITLE TRANSLATION — TRANSPARENT bottom-third
            # strip. Quick 0.40s fade-in; longer hold so the
            # reader can read both lines.
            sub_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - sub_start < 1.6:
                continue
            win = f"between(t,{sub_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sub_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[sub{gi}]")
            stages.append(
                f"[{{CUR}}][sub{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "reaction_insert" \
                and asset.startswith("ri_") \
                and asset.endswith(".png"):
            # REACTION INSERT — TRANSPARENT top-right PIP frame.
            # Slides in from the right edge by 30px during the
            # 0.50s reveal window so it feels like a hard-cut
            # to inset.
            ri_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - ri_start < 1.6:
                continue
            win = f"between(t,{ri_start:.2f},{b_:.2f})"
            sd_ = 0.50
            p = f"min(1\\,max(0\\,(t-{ri_start:.2f})/{sd_}))"
            ease = f"(1-pow(1-{p}\\,3))"
            x_expr = f"30*(1-{ease})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ri_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[ri{gi}]")
            stages.append(
                f"[{{CUR}}][ri{gi}]overlay=x='{x_expr}':y=0:"
                f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "evidence_tag" \
                and asset.startswith("et_") \
                and asset.endswith(".png"):
            # EVIDENCE TAG — TRANSPARENT bottom-left card. Quick
            # 0.40s fade-in (forensic tags should LAND fast like a
            # stamp), long hold for reading.
            et_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - et_start < 1.4:
                continue
            win = f"between(t,{et_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={et_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[et{gi}]")
            stages.append(
                f"[{{CUR}}][et{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "diary_entry" \
                and asset.startswith("de_") \
                and asset.endswith(".png"):
            # DIARY ENTRY — TRANSPARENT pinned-card overlay (same
            # family as letter/social_post but slower fade-in to
            # feel "settling onto" the page).
            de_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - de_start < 2.0:
                continue
            win = f"between(t,{de_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={de_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[de{gi}]")
            stages.append(
                f"[{{CUR}}][de{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "case_file" and asset.startswith("cf_") \
                and asset.endswith(".png"):
            # CASE FILE — opaque investigator card. 0.70s fade-in
            # (the file is being placed on the desk in front of the
            # camera). Long hold for reading the info rows.
            cf_start = start + 0.40
            b_ = min(start + d - 0.50, emax)
            if b_ - cf_start < 2.0:
                continue
            win = f"between(t,{cf_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cf_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[cf{gi}]")
            stages.append(
                f"[{{CUR}}][cf{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "demographic_split" and asset.startswith("dem_") \
                and asset.endswith(".png"):
            # DEMOGRAPHIC SPLIT — opaque data overlay. Standard
            # 0.65s fade-in matching the data-template family.
            dm_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - dm_start < 1.8:
                continue
            win = f"between(t,{dm_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={dm_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[dem{gi}]")
            stages.append(
                f"[{{CUR}}][dem{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind in ("glossary", "define_the_term") and asset.startswith("glo_") \
                and asset.endswith(".png"):
            # GLOSSARY / DEFINE_THE_TERM — TRANSPARENT bottom-third card.
            # 0.50s fade-in (it's an educational beat, snappy but
            # not snap). Long hold so the definition can be read.
            gl_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - gl_start < 1.8:
                continue
            win = f"between(t,{gl_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={gl_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[glo{gi}]")
            stages.append(
                f"[{{CUR}}][glo{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "letter" and asset.startswith("ltr_") \
                and asset.endswith(".png"):
            # LETTER — TRANSPARENT pinned correspondence card.
            # Slow 0.75s fade-in (the paper "settles onto" the lens),
            # long hold for reading.
            lt_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - lt_start < 2.0:
                continue
            win = f"between(t,{lt_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={lt_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[ltr{gi}]")
            stages.append(
                f"[{{CUR}}][ltr{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "stats_bar" and asset.startswith("sb_") \
                and asset.endswith(".png"):
            # STATS BAR — opaque data overlay (same family as the
            # other dark-navy data templates). 0.65s fade-in,
            # standard 0.50s tail.
            sb_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - sb_start < 1.8:
                continue
            win = f"between(t,{sb_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sb_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[sb{gi}]")
            stages.append(
                f"[{{CUR}}][sb{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "social_post" and asset.startswith("sp_") \
                and asset.endswith(".png"):
            # SOCIAL POST — TRANSPARENT pinned-card overlay. Quick
            # 0.40s fade-in, long hold for reading. Card stays
            # centred (no slide-in — social posts feel pinned).
            sp_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - sp_start < 1.6:
                continue
            win = f"between(t,{sp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sp_start:.2f}:d=0.45:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[sp{gi}]")
            stages.append(
                f"[{{CUR}}][sp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "news_article" and asset.startswith("na_") \
                and asset.endswith(".png"):
            # NEWS ARTICLE — TRANSPARENT centred excerpt card.
            # 0.55s fade-in (a page settling onto the lens), long
            # hold so the headline + excerpt land.
            na_start = start + 0.35
            b_ = min(start + d - 0.45, emax)
            if b_ - na_start < 1.8:
                continue
            win = f"between(t,{na_start:.2f},{b_:.2f})"
            _fade = (f"fade=t=in:st={na_start:.2f}:d=0.55:alpha=1,"
                     f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1")
            # MNT_5 — ANIMATED highlighter. The renderer now writes a second
            # pixel-identical page `na_*_hl.png` (marker behind text). We mask
            # the hl page's ALPHA to X < W*progress(t) via geq, then overlay it
            # over the plain page — so the marker DRAWS IN left->right like a
            # real felt-tip, timed to the spoken emphasis word, never a static
            # bake. (xfade=wiperight mangled the card's translucent alpha; the
            # geq alpha-wipe is alpha-safe so the transparent card margins stay
            # clear and footage shows around the card.) Falls back to the plain
            # static fade for cached cards lacking the _hl page.
            na_hl = asset[:-4] + "_hl.png"
            if (workdir / na_hl).is_file():
                ws = (rev_t if (na_start + 0.5 <= rev_t <= b_ - 0.9)
                      else na_start + 0.7)            # marker reveal start
                mt = 0.65                             # stroke duration
                if ws + mt > b_ - 0.4:                # keep inside the card
                    ws = max(na_start + 0.5, b_ - 0.4 - mt)
                _prog = f"clip((T-{ws:.2f})/{mt:.2f}\\,0\\,1)"
                stages.append(
                    f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,{_fade}[naP{gi}]")
                stages.append(
                    f"movie='{na_hl}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,{_fade}[naH{gi}]")
                stages.append(f"[naH{gi}]split[naHa{gi}][naHb{gi}]")
                stages.append(
                    f"[naHb{gi}]alphaextract,"
                    f"geq=lum='if(lt(X\\,W*{_prog})\\,lum(X\\,Y)\\,0)'"
                    f"[naAW{gi}]")
                stages.append(f"[naHa{gi}][naAW{gi}]alphamerge[naHW{gi}]")
                # NOTE: the two-overlay composite (plain page, then the
                # animated marker on top) MUST be a SINGLE stage entry.
                # The overlay threaders (single-stage `vchain` build and
                # the per-scene bake) advance the running `{CUR}` label on
                # *any* entry containing `{CUR}` and only substitute `{OUT}`
                # within that same entry. Splitting this into two entries
                # left the 2nd overlay's `{OUT}` unsubstituted → ffmpeg
                # "overlay has an unconnected output". Keeping `{CUR}` and
                # `{OUT}` in one entry (intermediate `naM` internal) makes
                # it one well-formed threadable unit. Other multi-overlay
                # cards already follow this contract.
                stages.append(
                    f"[{{CUR}}][naP{gi}]overlay=x=0:y=0:enable='{win}'"
                    f"[naM{gi}];"
                    f"[naM{gi}][naHW{gi}]overlay=x=0:y=0:enable='{win}'"
                    f"[{{OUT}}]")
            else:
                stages.append(
                    f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,{_fade}[na{gi}]")
                stages.append(
                    f"[{{CUR}}][na{gi}]overlay=x=0:y=0:enable='{win}'"
                    f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "cause_effect" and asset.startswith("ce_") \
                and asset.endswith(".png"):
            # CAUSE/EFFECT — opaque data overlay. Standard 0.65s
            # fade-in (matches comparison/process_diagram), long hold.
            ce_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - ce_start < 1.8:
                continue
            win = f"between(t,{ce_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ce_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[ce{gi}]")
            stages.append(
                f"[{{CUR}}][ce{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "map_route" and asset.startswith("mrt_") \
                and asset.endswith(".png"):
            # MAP ROUTE — DYNAMIC map storytelling. If layered, reveal:
            # chart bg → FROM pin (location pulse) → route DRAWS across
            # the map (segments reveal in sequence) → TO pin lands. The
            # journey animates instead of appearing as one static chart.
            mr_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - mr_start < 1.8:
                continue
            win = f"between(t,{mr_start:.2f},{b_:.2f})"
            fade_out_dur = 0.50
            fade_out_start = b_ - fade_out_dur

            # REAL ROUTE (JSON flipbook): bg zoom, START pin drops, the route
            # DRAWS via a cumulative frame flipbook (glowing dot rides the
            # tip), END pin lands, label slides in.
            jpath = workdir / asset.replace(".png", ".json")
            if jpath.is_file():
                import json as _json
                try:
                    rj = _json.loads(jpath.read_text(encoding="utf-8"))
                except Exception:                             # noqa: BLE001
                    rj = None
                if rj and rj.get("frames"):
                    fout = b_ - 0.45
                    span = b_ - mr_start
                    prog = f"min(1\\,max(0\\,(t-{mr_start:.2f})/{span:.2f}))"
                    Z = f"1920*(1.0+0.05*{prog})"
                    stages.append(
                        f"movie='{rj['bg']}',format=rgba,loop=loop=-1:"
                        f"size=1,setpts=N/{FPS}/TB,scale=w='{Z}':h=-2:"
                        f"eval=frame,crop=1920:1080,setsar=1,"
                        f"fade=t=in:st={mr_start:.2f}:d=0.60:alpha=1,"
                        f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[mrbg{gi}]")
                    stages.append(f"[{{CUR}}][mrbg{gi}]overlay=0:0:"
                                  f"enable='{win}'[{{OUT}}]")
                    # start pin
                    t_sp = mr_start + 0.35
                    mv, ov = _motion_layer(rj["sp"], f"mrsp{gi}", t_sp, 0.30,
                                           fout, 0.45,
                                           f"between(t,{t_sp:.2f},{b_:.2f})",
                                           rise=20)
                    stages.append(mv)
                    stages.append(ov)
                    # flipbook draw (cumulative frames, hard-swap per slice)
                    # FILTER-GRAPH BUDGET (USER BUG 2026-05-26): each frame
                    # adds 2 stages (movie load + overlay) to the giant
                    # final-mux filter graph. A long doc with many
                    # map_route scenes × 10 frames each pushed the graph
                    # past ~600 chains, where ffmpeg's auto_scale
                    # negotiation collapses ("Failed to configure output
                    # pad"). Cap to MAX_FLIPBOOK_FRAMES — sample evenly
                    # so the route still appears to draw in, with the
                    # complete last frame guaranteed at the end.
                    fr = rj["frames"]
                    MAX_FLIPBOOK_FRAMES = 4
                    if len(fr) > MAX_FLIPBOOK_FRAMES:
                        _n = MAX_FLIPBOOK_FRAMES
                        _step = (len(fr) - 1) / max(1, _n - 1)
                        fr = [fr[int(round(i * _step))]
                              for i in range(_n)]
                    t_draw = t_sp + 0.45
                    ddur = min(2.6, max(1.4, span * 0.55))
                    ddur = min(ddur, max(0.8, fout - 0.8 - t_draw))
                    stepf = ddur / max(1, len(fr))
                    for k, fn in enumerate(fr):
                        ts = t_draw + k * stepf
                        te = (t_draw + (k + 1) * stepf) if k < len(fr) - 1 \
                            else fout
                        stages.append(
                            f"movie='{fn}',format=rgba,loop=loop=-1:size=1,"
                            f"setpts=N/{FPS}/TB[mrf{gi}_{k}]")
                        stages.append(
                            f"[{{CUR}}][mrf{gi}_{k}]overlay=0:0:"
                            f"enable='between(t,{ts:.2f},{te:.2f})'"
                            f"[{{OUT}}]")
                    # end pin lands when the draw arrives
                    t_ep = min(t_draw + ddur, b_ - 0.7)
                    mv, ov = _motion_layer(rj["ep"], f"mrep{gi}", t_ep, 0.32,
                                           fout, 0.45,
                                           f"between(t,{t_ep:.2f},{b_:.2f})",
                                           rise=22)
                    stages.append(mv)
                    stages.append(ov)
                    # route label slides up
                    t_lb = min(t_ep + 0.20, b_ - 0.5)
                    mv, ov = _motion_layer(rj["lb"], f"mrlb{gi}", t_lb, 0.32,
                                           fout, 0.45,
                                           f"between(t,{t_lb:.2f},{b_:.2f})",
                                           rise=16)
                    stages.append(mv)
                    stages.append(ov)
                    gi += 1
                    continue

            manifest_path = workdir / asset.replace(".png", ".manifest.txt")
            layer_names = []
            if manifest_path.is_file():
                try:
                    layer_names = [ln.strip() for ln in
                                   manifest_path.read_text(
                                       encoding="utf-8").splitlines()
                                   if ln.strip()]
                except Exception:
                    layer_names = []

            # need bg + from-pin + >=1 route seg + to-pin
            if len(layer_names) >= 4:
                bg_name, p0_name = layer_names[0], layer_names[1]
                to_name = layer_names[-1]
                route_segs = layer_names[2:-1]
                # bg
                stages.append(
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={mr_start:.2f}:d=0.65:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[mrbg{gi}]")
                stages.append(f"[{{CUR}}][mrbg{gi}]overlay=x=0:y=0:"
                              f"enable='{win}'[{{OUT}}]")
                # FROM pin lands (rises a touch into place)
                p0_t = mr_start + 0.45
                p0win = f"between(t,{p0_t:.2f},{b_:.2f})"
                mv, ov = _motion_layer(p0_name, f"mrp0{gi}", p0_t, 0.35,
                                       fade_out_start, fade_out_dur,
                                       p0win, rise=18)
                stages.append(mv)
                stages.append(ov)
                # route DRAWS across — each segment fades in in sequence
                t_cur = p0_t + 0.35
                for k, rn in enumerate(route_segs):
                    rwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    stages.append(
                        f"movie='{rn}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"fade=t=in:st={t_cur:.2f}:d=0.18:alpha=1,"
                        f"fade=t=out:st={fade_out_start:.2f}:"
                        f"d={fade_out_dur:.2f}:alpha=1[mrr{gi}_{k}]")
                    stages.append(f"[{{CUR}}][mrr{gi}_{k}]overlay=x=0:y=0:"
                                  f"enable='{rwin}'[{{OUT}}]")
                    t_cur += 0.16
                # TO pin lands when the route arrives
                p1_t = min(t_cur + 0.05, b_ - 0.6)
                p1win = f"between(t,{p1_t:.2f},{b_:.2f})"
                mv, ov = _motion_layer(to_name, f"mrp1{gi}", p1_t, 0.35,
                                       fade_out_start, fade_out_dur,
                                       p1win, rise=18)
                stages.append(mv)
                stages.append(ov)
                gi += 1
                continue

            # fallback: single composite
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mr_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1[mrt{gi}]")
            stages.append(
                f"[{{CUR}}][mrt{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "map_region" and asset.startswith("mrg_") \
                and asset.endswith(".png"):
            # REAL REGION HIGHLIGHT — bg zooms in, the DIM+GLOW territory
            # layer fades up, the label slides in, then a pulse ring expands
            # from the region centre.
            rg_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - rg_start < 1.6:
                continue
            win = f"between(t,{rg_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly, cxr, cyr = [], 960, 540
            if mpath.is_file():
                for x in mpath.read_text(encoding="utf-8").splitlines():
                    x = x.strip()
                    if not x:
                        continue
                    if x.startswith("center="):
                        try:
                            cxr, cyr = (int(v) for v in
                                        x.split("=", 1)[1].split(","))
                        except Exception:                     # noqa: BLE001
                            pass
                    else:
                        ly.append(x)
            if len(ly) >= 3:
                span = b_ - rg_start
                prog = f"min(1\\,max(0\\,(t-{rg_start:.2f})/{span:.2f}))"
                # TERRITORY FLY-IN: ease-out 'arrival' (decelerates onto the
                # region) + a touch more travel; legacy = weak 7% linear push.
                if _territory_flyin():
                    Z = (f"1920*(1.0+0.11*(1-pow(1-{prog}\\,3)))")
                else:
                    Z = f"1920*(1.0+0.07*{prog})"
                stages.append(
                    f"movie='{ly[0]}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,scale=w='{Z}':h=-2:eval=frame,"
                    f"crop=1920:1080,setsar=1,"
                    f"fade=t=in:st={rg_start:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[rgbg{gi}]")
                stages.append(f"[{{CUR}}][rgbg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                # dim + glow region. IMP_019 — VO-timed territory fill: land
                # the highlight on the instant the narrator NAMES the region
                # (rev_t = the emphasis word's spoken time) instead of a fixed
                # 0.45s offset, so the colour "arrives" on the word. Clamp
                # inside the window (leave room for the label + pulse after).
                t_hi = rg_start + 0.45
                if rev_t and rg_start + 0.30 <= rev_t <= b_ - 1.10:
                    t_hi = float(rev_t)
                # FLY-IN: zoom the glow territory with the SAME transform as the
                # basemap so the fill stays locked to the land (legacy overlaid
                # the glow un-zoomed, so it drifted off the moving map).
                _hi_zoom = (f"scale=w='{Z}':h=-2:eval=frame,crop=1920:1080,"
                            f"setsar=1," if _territory_flyin() else "")
                stages.append(
                    f"movie='{ly[1]}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,{_hi_zoom}"
                    f"fade=t=in:st={t_hi:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[rghi{gi}]")
                stages.append(f"[{{CUR}}][rghi{gi}]overlay=0:0:"
                              f"enable='between(t,{t_hi:.2f},{b_:.2f})'"
                              f"[{{OUT}}]")
                # label slides up
                t_lb = t_hi + 0.45
                mv, ov = _motion_layer(ly[2], f"rglb{gi}", t_lb, 0.34,
                                       fout, 0.45,
                                       f"between(t,{t_lb:.2f},{b_:.2f})",
                                       rise=16)
                stages.append(mv)
                stages.append(ov)
                # pulse ring from region centre (if sprite present)
                if len(ly) >= 4:
                    tp = t_lb + 0.20
                    if tp < b_ - 0.6:
                        sp = f"0.30+2.20*min(1\\,max(0\\,(t-{tp:.2f})/1.10))"
                        stages.append(
                            f"movie='{ly[3]}',format=rgba,loop=loop=-1:"
                            f"size=1,setpts=N/{FPS}/TB,"
                            f"scale=w='260*({sp})':h='260*({sp})':"
                            f"eval=frame,setsar=1,"
                            f"fade=t=in:st={tp:.2f}:d=0.10:alpha=1,"
                            f"fade=t=out:st={tp + 0.7:.2f}:d=0.40:alpha=1"
                            f"[rgp{gi}]")
                        stages.append(
                            f"[{{CUR}}][rgp{gi}]overlay="
                            f"x='{cxr}-130*({sp})':y='{cyr}-130*({sp})':"
                            f"eval=frame:enable='between(t,{tp:.2f},"
                            f"{tp + 1.15:.2f})'[{{OUT}}]")
                gi += 1
                continue
            # fallback: single composite fade-in
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={rg_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[rg{gi}]")
            stages.append(f"[{{CUR}}][rg{gi}]overlay=0:0:enable='{win}'"
                          f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "donut_chart" and asset.startswith("dnt_") \
                and asset.endswith(".png"):
            # DONUT CHART — opaque data-template overlay. Standard
            # 0.65s fade-in (matches comparison/stat_dashboard) and
            # a longer hold so the percentage lands.
            dn_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - dn_start < 1.8:
                continue
            win = f"between(t,{dn_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={dn_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[dnt{gi}]")
            stages.append(
                f"[{{CUR}}][dnt{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "title_card" and asset.startswith("ttl_") \
                and asset.endswith(".png"):
            # TITLE CARD — opaque cinematic cold-open overlay. A title
            # card is a BEAT, not a backdrop: it flashes for ~3s and then
            # dissolves to reveal the footage (it used to hold the whole
            # scene, ~18s, which read as a static slide). Quick dramatic
            # fade-in, brief hold, soft tail into the cut-to-footage.
            tt_start = start + 0.30
            b_ = min(tt_start + 3.0, start + d - 0.40, emax)
            if b_ - tt_start < 1.8:
                continue
            win = f"between(t,{tt_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={tt_start:.2f}:d=0.70:alpha=1,"
                f"fade=t=out:st={b_-0.65:.2f}:d=0.65:alpha=1"
                f"[ttl{gi}]")
            stages.append(
                f"[{{CUR}}][ttl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "redacted" and asset.startswith("rdc_") \
                and asset.endswith(".png"):
            # REDACTED — opaque cream-paper full-frame overlay.
            # Cinematic 0.75s fade-in (the page settles in front
            # of the camera). Long hold so the viewer can read the
            # non-redacted parts. 0.55s gentle tail.
            rd_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - rd_start < 2.0:
                continue
            win = f"between(t,{rd_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={rd_start:.2f}:d=0.75:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[rdc{gi}]")
            stages.append(
                f"[{{CUR}}][rdc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "text_on_black" and asset.startswith("tob_") \
                and asset.endswith(".png"):
            # TEXT ON BLACK — opaque near-black full-frame card whose
            # monospace text IS the subject. Slightly slower fade-in
            # (0.60s, the message resolves), LONG hold so the viewer can
            # READ it (this is a reading beat, not a flash), 0.55s tail.
            tb_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - tb_start < 2.2:
                continue
            win = f"between(t,{tb_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={tb_start:.2f}:d=0.60:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[tob{gi}]")
            stages.append(
                f"[{{CUR}}][tob{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "audio_waveform" and asset.startswith("aw_") \
                and asset.endswith(".png"):
            # AUDIO WAVEFORM — TRANSPARENT bottom-third overlay.
            # Quick fade-in (0.40s) — the audio strip should
            # ANNOUNCE itself like a broadcast cut to tape. Slow
            # tail (0.50s) so it dissolves naturally.
            aw_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - aw_start < 1.6:
                continue
            win = f"between(t,{aw_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={aw_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[aw{gi}]")
            stages.append(
                f"[{{CUR}}][aw{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "chapter_marker" and asset.startswith("chap_") \
                and asset.endswith(".png"):
            # CHAPTER MARKER — opaque full-screen act break. A beat, not
            # a backdrop: flashes ~2.8s to mark the section change, then
            # dissolves into the scene's footage (it used to hold the
            # whole scene). Cinematic fade-in, brief hold, soft tail.
            ch_start = start + 0.35
            b_ = min(ch_start + 2.8, start + d - 0.40, emax)
            if b_ - ch_start < 1.8:
                continue
            win = f"between(t,{ch_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ch_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.60:.2f}:d=0.60:alpha=1"
                f"[chap{gi}]")
            stages.append(
                f"[{{CUR}}][chap{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "breaking_news" and asset.startswith("brk_") \
                and asset.endswith(".png"):
            # BREAKING NEWS BANNER — TRANSPARENT bottom-third
            # overlay. Snappy 0.35s slide-up/fade-in (network alerts
            # SHOULD feel like they punch on), long hold, gentle
            # 0.45s tail. Banner uses a slight y-lift via overlay
            # y-expression so it slides into place from below.
            bk_start = start + 0.30
            b_ = min(bk_start + 4.0, start + d - 0.35, emax)
            if b_ - bk_start < 1.6:
                continue
            win = f"between(t,{bk_start:.2f},{b_:.2f})"
            sd = 0.45
            p = f"min(1\\,max(0\\,(t-{bk_start:.2f})/{sd}))"
            ease = f"(1-pow(1-{p}\\,3))"
            # Banner sits in bottom third; slide it up 40px during
            # the reveal window so it feels mechanical/punchy.
            y_expr = f"40*(1-{ease})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={bk_start:.2f}:d=0.35:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[brk{gi}]")
            stages.append(
                f"[{{CUR}}][brk{gi}]overlay=x=0:y='{y_expr}':"
                f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "countdown" and asset.startswith("ctd_") \
                and asset.endswith(".png"):
            # COUNTDOWN / URGENCY — TRANSPARENT bottom-right deadline
            # readout. Composites OVER footage. Snappy fade-in (0.40s)
            # because urgency should LAND fast, not drift in. Long
            # hold; gentle 0.45s fade-out (urgency doesn't snap off).
            ct_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - ct_start < 1.4:
                continue
            win = f"between(t,{ct_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={ct_start:.2f}:d=0.40:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[ctd{gi}]")
            stages.append(
                f"[{{CUR}}][ctd{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind in ("quote_highlight", "long_quote") and asset.startswith("qh_") \
                and asset.endswith(".png"):
            # KINETIC TYPOGRAPHY — if layered, the scrim/quote-mark fade
            # in, then the quote reveals PHRASE-BY-PHRASE (line by line,
            # each rising into place) and the attribution lands last. The
            # emphasis keyword is already punched accent in its line.
            qh_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - qh_start < 1.8:
                continue
            win = f"between(t,{qh_start:.2f},{b_:.2f})"
            fade_out_dur = 0.50
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(".png", ".manifest.txt")
            layer_names = []
            if manifest_path.is_file():
                try:
                    layer_names = [ln.strip() for ln in
                                   manifest_path.read_text(
                                       encoding="utf-8").splitlines()
                                   if ln.strip()]
                except Exception:
                    layer_names = []

            if len(layer_names) >= 2:
                bg_name, *rest = layer_names
                # bg (scrim + quote mark) — soft dissolve, no rise
                mv, ov = _motion_layer(bg_name, f"qhbg{gi}", qh_start, 0.55,
                                       fade_out_start, fade_out_dur,
                                       win, rise=0)
                stages.append(mv)
                stages.append(ov)
                # lines + attribution reveal in sequence, each rising in
                t_cur = qh_start + 0.45
                n_rest = max(1, len(rest))
                step = min(0.42, max(0.26,
                                     (b_ - 0.7 - t_cur) / n_rest))
                for k, ln in enumerate(rest):
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        ln, f"qhl{gi}_{k}", t_cur, 0.42,
                        fade_out_start, fade_out_dur, lwin, rise=22)
                    stages.append(mv)
                    stages.append(ov)
                    t_cur += step
                gi += 1
                continue

            # fallback: single composite (legacy) — dissolve + rise
            mv, ov = _motion_layer(
                asset, f"qh{gi}", qh_start, 0.70,
                b_ - 0.50, 0.50, win, rise=20)
            stages.append(mv)
            stages.append(ov)
            gi += 1
            continue

        if kind == "bullet_list" and asset.startswith("bl_") \
                and asset.endswith(".png"):
            # BULLET LIST — LAYERED staggered reveal.
            #
            # If the renderer produced a manifest, fade in bg first
            # (carries vignette + headline) then each bullet b0..bK
            # at 0.30s intervals so the list "builds" rather than
            # appearing as a wall of text. Fall back to the single
            # composite PNG if no manifest is present.
            bl_start = start + 0.35
            b_ = min(start + d - 0.40, emax)
            if b_ - bl_start < 1.8:
                continue
            win = f"between(t,{bl_start:.2f},{b_:.2f})"
            fade_out_dur = 0.45
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(
                ".png", ".manifest.txt")
            layer_names: list[str] = []
            if manifest_path.is_file():
                try:
                    layer_names = [
                        ln.strip() for ln in
                        manifest_path.read_text(
                            encoding="utf-8").splitlines()
                        if ln.strip()]
                except Exception:
                    layer_names = []

            # Need bg + ≥1 bullet for the layered path.
            if len(layer_names) >= 2:
                bg_name, *bullet_names = layer_names
                prof = _mo.profile("standard")
                # bg = backdrop + headline: clean dissolve, no rise.
                mv, ov = _motion_layer(
                    bg_name, f"blbg{gi}", bl_start, 0.55,
                    fade_out_start, fade_out_dur, win, rise=0)
                stages.append(mv)
                stages.append(ov)

                # bullets cascade in (motion.stagger) and each RISES a
                # few px into place with the shared cinematic settle, so
                # the list BUILDS like a motion-graphics editor timed it.
                b_ts = _mo.stagger(len(bullet_names), bl_start + 0.50,
                                   prof, step=0.26)
                for k, bn in enumerate(bullet_names):
                    bt = min(b_ts[k], b_ - 0.6)
                    bwin = f"between(t,{bt:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        bn, f"blb{gi}_{k}", bt, 0.38,
                        fade_out_start, fade_out_dur, bwin, rise=26)
                    stages.append(mv)
                    stages.append(ov)
                gi += 1
                continue

            # Fallback: single composite (legacy renders)
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={bl_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1[bl{gi}]")
            stages.append(
                f"[{{CUR}}][bl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "typing_date" and asset.startswith("td_") \
                and asset.endswith(".png"):
            td_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - td_start < 1.5:
                continue
            win = f"between(t,{td_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={td_start:.2f}:d=0.45:alpha=1,"
                f"fade=t=out:st={b_-0.40:.2f}:d=0.40:alpha=1[td{gi}]")
            stages.append(
                f"[{{CUR}}][td{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "lower_third" and asset.startswith("lt_") \
                and asset.endswith(".png"):
            # DYNAMIC LOWER THIRD — transparent PNG (one of 4 layout
            # variants). Cinematic entrance: soft dissolve + a subtle
            # eased RISE into place (the classic broadcast lower-third
            # settle), understated so the variant placement still leads.
            lt_start = start + 0.30
            b_ = min(start + d - 0.35, emax)
            if b_ - lt_start < 1.2:
                continue
            win = f"between(t,{lt_start:.2f},{b_:.2f})"
            mv, ov = _motion_layer(
                asset, f"lt{gi}", lt_start, 0.55,
                b_ - 0.45, 0.45, win, rise=22)
            stages.append(mv)
            stages.append(ov)
            gi += 1
            continue

        if kind == "name_reveal" and asset.startswith("nmr_") \
                and asset.endswith(".png"):
            # CHARACTER REVEAL. Layered (photo card): scrim fades, the
            # framed PORTRAIT slides in from the LEFT, the NAME slides in
            # from the RIGHT — they converge into the 'meet the character'
            # card. Name-only fallback: the card slides in from the left.
            nr_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - nr_start < 1.2:
                continue
            win = f"between(t,{nr_start:.2f},{b_:.2f})"
            fade_out_dur = 0.45
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(".png", ".manifest.txt")
            layer_names = []
            if manifest_path.is_file():
                try:
                    layer_names = [ln.strip() for ln in
                                   manifest_path.read_text(
                                       encoding="utf-8").splitlines()
                                   if ln.strip()]
                except Exception:
                    layer_names = []

            if len(layer_names) >= 3:
                bg_name, pic_name, name_name = layer_names[:3]
                # CINEMATIC REVEAL MOTION (Netflix doc feel):
                #   * bg vignette fades in first (sets the depth)
                #   * portrait RISES UP from +44 px with opacity ramp
                #     -- not a hard slide; gives it weight
                #   * name then rises +24 px slightly delayed
                #   * everything decelerates (out_cubic ease)
                # The portrait NEVER slides horizontally now -- horizontal
                # slides read as a streamer/gaming overlay.

                # bg scrim
                stages.append(
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={nr_start:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[nrbg{gi}]")
                stages.append(f"[{{CUR}}][nrbg{gi}]overlay=x=0:y=0:"
                              f"enable='{win}'[{{OUT}}]")

                # PORTRAIT — rises upward, longer fade-in for cinematic feel
                p_t = nr_start + 0.18
                py = _mo.interp_ff(44.0, 0.0, p_t, 0.70, "out_cubic")
                stages.append(
                    f"movie='{pic_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={p_t:.2f}:d=0.65:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[nrpic{gi}]")
                stages.append(f"[{{CUR}}][nrpic{gi}]overlay=x=0:y='{py}':"
                              f"eval=frame:enable='{win}'[{{OUT}}]")

                # NAME — staggered, rises a touch less (typography breathes)
                n_t = nr_start + 0.55
                nwin = f"between(t,{n_t:.2f},{b_:.2f})"
                ny = _mo.interp_ff(24.0, 0.0, n_t, 0.55, "out_cubic")
                stages.append(
                    f"movie='{name_name}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={n_t:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[nrnm{gi}]")
                stages.append(f"[{{CUR}}][nrnm{gi}]overlay=x=0:y='{ny}':"
                              f"eval=frame:enable='{nwin}'[{{OUT}}]")
                gi += 1
                continue

            # name-only fallback: whole card slides in from the left
            xexpr = _mo.interp_ff(-180.0, 0.0, nr_start, 0.55, "out_cubic")
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={nr_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1[nmr{gi}]")
            stages.append(
                f"[{{CUR}}][nmr{gi}]overlay=x='{xexpr}':y=0:eval=frame:"
                f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "surveillance" and asset.startswith("sv_") \
                and asset.endswith(".png"):
            # SURVEILLANCE / CCTV — TRANSPARENT viewfinder overlay
            # framing the live footage. Same fade pattern as news.
            sv_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - sv_start < 1.5:
                continue
            win = f"between(t,{sv_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={sv_start:.2f}:d=0.45:alpha=1,"
                f"fade=t=out:st={b_-0.40:.2f}:d=0.40:alpha=1"
                f"[sv{gi}]")
            stages.append(
                f"[{{CUR}}][sv{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "news_broadcast" and asset.startswith("nws_") \
                and asset.endswith(".png"):
            # NEWS BROADCAST — TRANSPARENT PNG so footage shows
            # through (only the bars + ticker are opaque). Slide-up
            # animation feels broadcast-natural.
            nw_start = start + 0.30
            b_ = min(start + d - 0.40, emax)
            if b_ - nw_start < 1.5:
                continue
            win = f"between(t,{nw_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={nw_start:.2f}:d=0.50:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1"
                f"[nws{gi}]")
            stages.append(
                f"[{{CUR}}][nws{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "classified" and asset.startswith("clf_") \
                and asset.endswith(".png"):
            # CLASSIFIED dossier overlay — opaque PNG fade-in/out.
            cf_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cf_start < 1.8:
                continue
            win = f"between(t,{cf_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cf_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[clf{gi}]")
            stages.append(
                f"[{{CUR}}][clf{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "photo_collage" and asset.startswith("clg_") \
                and asset.endswith(".png"):
            # PHOTO COLLAGE — pinned-photos evidence wall. Same
            # opaque PNG overlay pattern as the other data templates.
            cl_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cl_start < 1.8:
                continue
            win = f"between(t,{cl_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cl_start:.2f}:d=0.60:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[clg{gi}]")
            stages.append(
                f"[{{CUR}}][clg{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "process_diagram" and asset.startswith("prc_") \
                and asset.endswith(".png"):
            # PROCESS DIAGRAM — LAYERED staggered reveal.
            #
            # If the renderer produced a manifest, fade in bg first
            # then each step+arrow at 0.40s intervals so the diagram
            # BUILDS on screen instead of arriving as a wall. The
            # 'how it works' beat reads like cinema, not a slide.
            #
            # Fallback to single PNG if no manifest is present.
            pr_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - pr_start < 2.0:
                continue
            win = f"between(t,{pr_start:.2f},{b_:.2f})"
            fade_out_dur = 0.50
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(
                ".png", ".manifest.txt")
            layer_names: list[str] = []
            if manifest_path.is_file():
                try:
                    layer_names = [
                        ln.strip() for ln in
                        manifest_path.read_text(
                            encoding="utf-8").splitlines()
                        if ln.strip()]
                except Exception:
                    layer_names = []

            if len(layer_names) >= 2:
                bg_name, *seq_names = layer_names
                # bg fade-in
                stages.append(
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:"
                    f"size=1,setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={pr_start:.2f}:d=0.65:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[prcbg{gi}]")
                stages.append(
                    f"[{{CUR}}][prcbg{gi}]overlay=x=0:y=0:"
                    f"enable='{win}'[{{OUT}}]")
                # Sequential build: a STEP rises into place, then its
                # ARROW DRAWS toward the next step (slides L->R), then the
                # next step — the flow animates like a real process build.
                t_cur = pr_start + 0.55
                for k, sn in enumerate(seq_names):
                    is_arrow = bool(re.search(r"_a\d+\.png$", sn))
                    lwin = f"between(t,{t_cur:.2f},{b_:.2f})"
                    if is_arrow:
                        # arrow draws on: slide in from the left + quick fade
                        xexpr = _mo.interp_ff(-70.0, 0.0, t_cur, 0.34,
                                              "out_cubic")
                        stages.append(
                            f"movie='{sn}',format=rgba,loop=loop=-1:"
                            f"size=1,setpts=N/{FPS}/TB,"
                            f"fade=t=in:st={t_cur:.2f}:d=0.28:alpha=1,"
                            f"fade=t=out:st={fade_out_start:.2f}:"
                            f"d={fade_out_dur:.2f}:alpha=1[prca{gi}_{k}]")
                        stages.append(
                            f"[{{CUR}}][prca{gi}_{k}]overlay=x='{xexpr}':"
                            f"y=0:eval=frame:enable='{lwin}'[{{OUT}}]")
                        t_cur += 0.32
                    else:
                        mv, ov = _motion_layer(
                            sn, f"prcs{gi}_{k}", t_cur, 0.40,
                            fade_out_start, fade_out_dur, lwin, rise=24)
                        stages.append(mv)
                        stages.append(ov)
                        t_cur += 0.46
                gi += 1
                continue

            # Fallback: single composite PNG (legacy)
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={pr_start:.2f}:d=0.65:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[prc{gi}]")
            stages.append(
                f"[{{CUR}}][prc{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "conspiracy_board" and asset.startswith("cnsp_") \
                and asset.endswith(".png"):
            # CONSPIRACY INVESTIGATION BOARD — opaque corkboard PNG
            # with pinned cards + red strings. Slower fade-in (0.80s)
            # than the data-template family on purpose — investigation
            # boards SHOULD feel like the camera is *settling on* the
            # wall, not snapping a graphic up. Hold long; gentle tail.
            cp_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cp_start < 2.0:
                continue
            win = f"between(t,{cp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cp_start:.2f}:d=0.80:alpha=1,"
                f"fade=t=out:st={b_-0.55:.2f}:d=0.55:alpha=1"
                f"[cnsp{gi}]")
            stages.append(
                f"[{{CUR}}][cnsp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "stat_dashboard" and asset.startswith("stsh_") \
                and asset.endswith(".png"):
            # STAT DASHBOARD — pre-rendered multi-stat infographic.
            # Same overlay pattern as timeline/map/comparison: opaque
            # PNG fades in, holds, fades out.
            st_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - st_start < 1.8:
                continue
            win = f"between(t,{st_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={st_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1"
                f"[stsh{gi}]")
            stages.append(
                f"[{{CUR}}][stsh{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "comparison" and asset.startswith("cmp_") \
                and asset.endswith(".png"):
            # COMPARISON LAYOUT — full-frame overlay of pre-rendered
            # PIL composition (two-column split + central VS badge).
            # Opaque PNG → no stock-footage bleed during the window.
            cmp_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - cmp_start < 1.8:
                continue
            win = f"between(t,{cmp_start:.2f},{b_:.2f})"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={cmp_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1[cmp{gi}]")
            stages.append(
                f"[{{CUR}}][cmp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "timeline" and asset.startswith("tl_") \
                and asset.endswith(".png"):
            # TIMELINE SEQUENCE — LAYERED documentary reveal.
            #
            # If the renderer produced a manifest sidecar, stage each
            # layer separately so the timeline "draws" on screen:
            #   bg    → fade-in at tl_start            (the canvas)
            #   axis  → fade-in 0.50s later            (the line draws)
            #   eN    → fade-in staggered every 0.40s  (events punch in)
            # All layers share one alpha fade-OUT at the end.
            #
            # If the manifest is missing (legacy renders), fall back to
            # the original single-PNG overlay so old work_dirs still
            # render. This keeps the polish backwards-compatible.
            tl_start = start + 0.40
            b_ = min(start + d - 0.45, emax)
            if b_ - tl_start < 1.8:
                continue
            win = f"between(t,{tl_start:.2f},{b_:.2f})"
            fade_out_dur = 0.50
            fade_out_start = b_ - fade_out_dur

            manifest_path = workdir / asset.replace(
                ".png", ".manifest.txt")
            layer_names: list[str] = []
            if manifest_path.is_file():
                try:
                    layer_names = [
                        ln.strip() for ln in
                        manifest_path.read_text(
                            encoding="utf-8").splitlines()
                        if ln.strip()]
                except Exception:
                    layer_names = []
            # Sanity: need at least bg + axis + 1 event for the
            # layered path to be meaningful.
            if len(layer_names) >= 3:
                # ---- layered reveal ---------------------------- #
                # Reveal schedule (relative to tl_start):
                #   bg    : 0.00s  (fade-in 0.55s)
                #   axis  : 0.50s  (fade-in 0.55s) — the line draws
                #   e0..  : 0.95s + k*0.40s (fade-in 0.35s each)
                # Each layer fades OUT together at fade_out_start.
                bg_name, ax_name, *ev_names = layer_names
                bg_t = tl_start
                ax_t = tl_start + 0.50
                ev_t = _mo.stagger(len(ev_names), tl_start + 0.95,
                                   _mo.profile("standard"), step=0.36)

                # bg (opaque, kills stock footage bleed)
                stages.append(
                    f"movie='{bg_name}',format=rgba,loop=loop=-1:"
                    f"size=1,setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={bg_t:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[tlbg{gi}]")
                stages.append(
                    f"[{{CUR}}][tlbg{gi}]overlay=x=0:y=0:"
                    f"enable='{win}'[{{OUT}}]")

                # axis (transparent line + glow)
                stages.append(
                    f"movie='{ax_name}',format=rgba,loop=loop=-1:"
                    f"size=1,setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={ax_t:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fade_out_start:.2f}:"
                    f"d={fade_out_dur:.2f}:alpha=1[tlax{gi}]")
                ax_win = f"between(t,{ax_t:.2f},{b_:.2f})"
                stages.append(
                    f"[{{CUR}}][tlax{gi}]overlay=x=0:y=0:"
                    f"enable='{ax_win}'[{{OUT}}]")

                # per-event layers — staggered cascade, each event
                # RISES into place with the shared cinematic settle
                # (events "punch in" along the line like a real
                # motion-graphics timeline build).
                for k, en in enumerate(ev_names):
                    et = min(ev_t[k], b_ - 0.6)
                    ev_win = f"between(t,{et:.2f},{b_:.2f})"
                    mv, ov = _motion_layer(
                        en, f"tlev{gi}_{k}", et, 0.35,
                        fade_out_start, fade_out_dur, ev_win, rise=22)
                    stages.append(mv)
                    stages.append(ov)
                gi += 1
                continue

            # ---- fallback: single composite PNG (legacy path) -- #
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={tl_start:.2f}:d=0.60:alpha=1,"
                f"fade=t=out:st={b_-0.50:.2f}:d=0.50:alpha=1[tl{gi}]")
            stages.append(
                f"[{{CUR}}][tl{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "figure_locator" and asset.startswith("figloc_") \
                and asset.endswith(".png"):
            # IMP_021 — single composite PNG (map + portrait badge + leader +
            # marker, baked). Full-frame like a map_reveal: soft fade in/out.
            fl_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - fl_start < 1.4:
                continue
            win = f"between(t,{fl_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={fl_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[figl{gi}]")
            stages.append(
                f"[{{CUR}}][figl{gi}]overlay=0:0:enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "map_reveal" and asset.startswith("map_") \
                and asset.endswith(".png"):
            mp_start = start + 0.40
            b_ = min(start + d - 0.40, emax)
            if b_ - mp_start < 1.6:
                continue
            win = f"between(t,{mp_start:.2f},{b_:.2f})"
            fout = b_ - 0.45
            # IMP_016 — STATS ON THE MAP. If this map scene's narration named
            # a real quantity (comma-grouped figure threaded in via the cue),
            # tick it UP on the map in a clean upper-left data readout — the
            # number stays anchored to its geographic context (Vox style),
            # never a disconnected full-screen stat card. Drawn into `post`
            # so it rides ON TOP of the map overlay; independent of the map
            # layer rendering, so both the layered and fallback paths get it.
            if map_fig:
                _pnm = _parse_number(map_fig)
                if _pnm:
                    _mpfx, _mval, _msfx, _mfinal = _pnm
                    _mfinal = _dt(_mfinal)
                    _cst = mp_start + 0.60          # start counting after settle
                    _croll = min(0.95, max(0.50, (b_ - _cst) * 0.40))
                    _cl = _cst + _croll             # lands here
                    if _cl < b_ - 0.40:
                        _cwin = f"between(t,{_cst:.2f},{b_:.2f})"
                        _PX, _PY, _PW, _PH = 80, 196, 540, 138
                        _fz = 84
                        _tx, _ty = _PX + 30, _PY + 30
                        # scrim pill (legibility on a busy map) + accent rule
                        post.append(
                            f"drawbox=x={_PX}:y={_PY}:w={_PW}:h={_PH}:"
                            f"color=black@0.45:t=fill:enable='{_cwin}'")
                        post.append(
                            f"drawbox=x={_PX}:y={_PY}:w=6:h={_PH}:"
                            f"color={hexc}:t=fill:enable='{_cwin}'")
                        # rolling digits: ease-out 0 -> val (the proven count-up)
                        _ceif = (
                            f"%{{eif\\:floor({_mval}*(1-pow(1-"
                            f"clip((t-{_cst:.2f})/{_croll:.2f}\\,0\\,1)\\,3)))\\:d}}"
                        )
                        _cra = (f"if(lt(t,{_cst:.2f}),0,"
                                f"if(lt(t,{_cst + 0.10:.2f}),"
                                f"(t-{_cst:.2f})/0.10,1))")
                        post.append(
                            f"drawtext=fontfile={font}:expansion=normal:"
                            f"text='{_ceif}':fontsize={_fz}:fontcolor=white:"
                            "shadowcolor=black@0.65:shadowx=0:shadowy=4:"
                            f"x={_tx}:y={_ty}:alpha='{_cra}':"
                            f"enable='between(t,{_cst:.2f},{_cl:.2f})'")
                        # landed final value, held to the end of the map beat
                        post.append(
                            f"drawtext=fontfile={font}:text='{_mfinal}':"
                            f"expansion=none:fontsize={_fz}:fontcolor=white:"
                            "shadowcolor=black@0.65:shadowx=0:shadowy=4:"
                            f"x={_tx}:y={_ty}:"
                            f"enable='between(t,{_cl:.2f},{b_:.2f})'")
                        num_events.append((_cst, _cl))
            mpath = workdir / asset.replace(".png", ".manifest.txt")
            ly = []
            if mpath.is_file():
                try:
                    ly = [x.strip() for x in mpath.read_text(
                        encoding="utf-8").splitlines() if x.strip()]
                except Exception:                             # noqa: BLE001
                    ly = []
            # REAL MAP REVEAL (layered): bg slow-zoom INTO the location,
            # pin DROPS with a bounce, a double PULSE ring expands, then the
            # label SLIDES in — the documentary "geo-locate" beat.
            if len(ly) >= 4:
                PI = 3.14159
                bgn, pinn, ringn, labn = ly[0], ly[1], ly[2], ly[3]
                span = b_ - mp_start
                # 1) map background: slow cinematic zoom-in about centre
                prog = f"min(1\\,max(0\\,(t-{mp_start:.2f})/{span:.2f}))"
                Z = f"1920*(1.0+0.10*{prog})"
                stages.append(
                    f"movie='{bgn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,scale=w='{Z}':h=-2:eval=frame,"
                    f"crop=1920:1080,setsar=1,"
                    f"fade=t=in:st={mp_start:.2f}:d=0.55:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[mpbg{gi}]")
                stages.append(f"[{{CUR}}][mpbg{gi}]overlay=0:0:"
                              f"enable='{win}'[{{OUT}}]")
                # timing for pin/pulse/label (clamped inside the window)
                t_pin = min(mp_start + 0.65, b_ - 1.2)
                pd = 0.55
                t_land = t_pin + pd
                tp0 = t_land + 0.05
                tp1 = t_land + 0.55
                t_lab = t_land + 0.30
                # 2) pin drops in (ease-out fall + a small landing bounce)
                P = f"min(1\\,max(0\\,(t-{t_pin:.2f})/{pd}))"
                ydrop = f"-170*(1-{P})*(1-{P})"
                ypin = (f"if(lt(t,{t_land:.2f}),{ydrop},"
                        f"if(lt(t,{t_land + 0.18:.2f}),"
                        f"-11*sin((t-{t_land:.2f})/0.18*{PI}),0))")
                stages.append(
                    f"movie='{pinn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={t_pin:.2f}:d=0.12:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[mppin{gi}]")
                stages.append(
                    f"[{{CUR}}][mppin{gi}]overlay=x=0:y='{ypin}':"
                    f"eval=frame:enable='between(t,{t_pin:.2f},{b_:.2f})'"
                    f"[{{OUT}}]")
                # 3) double pulse ring (expands + fades), centred on the pin
                for kk, tp in enumerate((tp0, tp1)):
                    if tp > b_ - 0.6:
                        continue
                    sp = f"0.30+2.30*min(1\\,max(0\\,(t-{tp:.2f})/1.00))"
                    stages.append(
                        f"movie='{ringn}',format=rgba,loop=loop=-1:size=1,"
                        f"setpts=N/{FPS}/TB,"
                        f"scale=w='240*({sp})':h='240*({sp})':eval=frame,"
                        f"setsar=1,"
                        f"fade=t=in:st={tp:.2f}:d=0.10:alpha=1,"
                        f"fade=t=out:st={tp + 0.62:.2f}:d=0.40:alpha=1"
                        f"[mpr{gi}_{kk}]")
                    stages.append(
                        f"[{{CUR}}][mpr{gi}_{kk}]"
                        f"overlay=x='960-120*({sp})':y='540-120*({sp})':"
                        f"eval=frame:"
                        f"enable='between(t,{tp:.2f},{tp + 1.05:.2f})'"
                        f"[{{OUT}}]")
                # 4) label slides in from the pin
                slx = (f"44*pow(1-min(1\\,max(0\\,"
                       f"(t-{t_lab:.2f})/0.40))\\,2)")
                stages.append(
                    f"movie='{labn}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={t_lab:.2f}:d=0.30:alpha=1,"
                    f"fade=t=out:st={fout:.2f}:d=0.45:alpha=1[mplab{gi}]")
                stages.append(
                    f"[{{CUR}}][mplab{gi}]overlay=x='{slx}':y=0:"
                    f"eval=frame:enable='between(t,{t_lab:.2f},{b_:.2f})'"
                    f"[{{OUT}}]")
                gi += 1
                continue
            # FALLBACK: vector chart (single opaque PNG) — soft fade in.
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=in:st={mp_start:.2f}:d=0.55:alpha=1,"
                f"fade=t=out:st={b_-0.45:.2f}:d=0.45:alpha=1[mp{gi}]")
            stages.append(
                f"[{{CUR}}][mp{gi}]overlay=x=0:y=0:enable='{win}'"
                f"[{{OUT}}]")
            gi += 1
            continue

        if kind == "ranking" and raw and asset.startswith("rcard_") \
                and asset.endswith(".png"):
            # VIDLORE TOP-LIST CAROUSEL v5 — multi-card perspective PNG.
            # The PNG is FULL FRAME (1920x1080) with opaque dark-grid
            # background, so once it's on screen the underlying stock
            # footage is fully blocked (user explicit requirement: no
            # stock video bleed during the ranking sequence). To honour
            # that, we DO NOT alpha-fade the overlay — the carousel
            # slides in fully opaque from the right edge, so the moment
            # it covers a pixel that pixel is pure carousel design.
            # Tail still gets a 0.40s alpha fade-out so it doesn't snap.
            rk_start = start + 0.30
            b_ = min(start + d - 0.30, emax)
            if b_ - rk_start < 1.5:
                continue
            win = f"between(t,{rk_start:.2f},{b_:.2f})"
            fade_out = 0.40
            # Slide-in from RIGHT (off-canvas by 1920 → 0). 0.85 s cubic
            # ease-out feels like a confident documentary push, not a
            # snap. After landing, ±4 px x-drift on a slow 5.2 s period
            # + ±2 px y-drift on 3.9 s period → real layered parallax.
            sd = 0.85
            p = f"min(1\\,max(0\\,(t-{rk_start:.2f})/{sd}))"
            ease = f"(1-pow(1-{p}\\,3))"
            x_expr = (f"1920*(1-{ease})+4*sin((t-{rk_start:.2f})/5.2)"
                      f"*{ease}")
            y_expr = f"2*sin((t-{rk_start:.2f})/3.9)*{ease}"
            stages.append(
                f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                f"setpts=N/{FPS}/TB,"
                f"fade=t=out:st={b_-fade_out:.2f}:d={fade_out:.2f}:alpha=1"
                f"[rcd{gi}]")
            stages.append(
                f"[{{CUR}}][rcd{gi}]overlay=x='{x_expr}':y='{y_expr}':"
                f"enable='{win}'[{{OUT}}]")
            gi += 1
            continue

        if kind == "ranking" and raw:
            # Fallback (carousel PNG missing) — keep the v3.1 drawbox
            # composition as a safety net so a render never loses the
            # ranking signal even if PIL rendering failed upstream.
            # CINEMATIC RANKING REVEAL v3 — "Netflix documentary"
            # ASYMMETRIC LAYOUT + SLOWER DOCUMENTARY PACING.
            #
            # v2 problem: motion was correct but composition was still
            # bottom-CENTER (title + badge stacked) → symmetric/template.
            # v3 directs the scene like a human editor: badge is a
            # discreet TOP-RIGHT tag (the "exhibit no." stamp), title
            # lives in the BOTTOM-LEFT as a documentary lower-third
            # — your eye runs a natural diagonal Z (badge → footage
            # → title) instead of locking dead-centre.
            #
            # Pacing (was 0.0-1.0s tight; now 0.0-2.1s, breathes):
            #   0.00s  FOOTAGE ALONE — no overlays. Viewer absorbs the
            #          image before the design system steps in.
            #   0.60s  frame edges start a SLOW 5-stage soft fade-in
            #          (0.60-1.10s) — feels drawn, not snapped.
            #   1.00s  TITLE rises from below at the bottom-LEFT,
            #          eased cubic over 0.90s (longer than v2's 0.55).
            #   1.40s  TOP-RIGHT badge backing lands.
            #   1.40s  "#N" digit settles into the badge with a SOFT
            #          overshoot (amplitude halved from v2 — no snap,
            #          just a gentle land).
            #   tail   everything fades out cleanly over 0.40s.
            #
            # On top of that: independent sin-drift frequencies on the
            # title (±3 px x) and badge (±2 px y) so they breathe at
            # different rates — the eye reads "depth", not "pasted".
            # Used only for ranked content; capped at 5/video.
            m = re.search(r"\d+", raw or "")
            if not m:
                continue
            badge = f"#{m.group(0)[:2]}"
            title = _card_text(body or "")[:36]
            # v3.1 PACING — slower confidence. Hold extended (0.60→0.80),
            # frame fade extended (0.50→0.65), title slide stretched
            # (0.90→1.10), tail fade-out 0.55s so nothing snaps off.
            hold = 0.80
            rk_start = start + hold
            b_ = min(start + d - 0.55, emax)
            if b_ - rk_start < 2.4:
                continue
            win = f"between(t,{rk_start:.2f},{b_:.2f})"
            fade_out_t = max(rk_start + 1.0, b_ - 0.55)

            # ---- ATMOSPHERIC EDGE VIGNETTE (cinematic darkening) ----
            # Four thin dark drawboxes along the inner edges of the
            # frame, very low alpha. Reads as "cinematic edge falloff"
            # not as a graphic. Rides the same fade ramp as the frame.
            VG_T = 56                          # vignette thickness
            # ---- FRAME 5-stage SOFTER fade-in (final 0.85, not 0.96)
            # Less "graphic UI white" — softer integration with footage.
            # Stretched to 0.65s vs v3's 0.50s for documentary calm.
            FX, FY, FW, FH = 80, 60, 1760, 960
            FT = 4                              # was 5 — slimmer = less "PNG"
            _ramps = (                       # (rel_t_a, rel_t_b, alpha)
                (0.00, 0.13, 0.14),
                (0.13, 0.28, 0.34),
                (0.28, 0.43, 0.55),
                (0.43, 0.58, 0.72),
                (0.58, None, 0.85),          # was 0.96 — softer final
            )

            def _staged(x, y, w, h, color):
                for (ra, rb, a) in _ramps:
                    ta = rk_start + ra
                    tb = b_ if rb is None else rk_start + rb
                    out.append(
                        f"drawbox=x={x}:y={y}:w={w}:h={h}:"
                        f"color={color}@{a:.2f}:t=fill:"
                        f"enable='between(t,{ta:.2f},{tb:.2f})'")

            # Atmospheric corner vignettes — fade with the frame, very
            # subtle (max 0.18 alpha) so they READ as light falloff,
            # not as drawn boxes. Top + bottom only (most visible).
            for (ra, rb, a) in _ramps:
                ta = rk_start + ra
                tb = b_ if rb is None else rk_start + rb
                a_vg = min(0.18, a * 0.22)
                out.append(  # top vignette
                    f"drawbox=x=0:y=0:w=iw:h={VG_T}:"
                    f"color=black@{a_vg:.2f}:t=fill:"
                    f"enable='between(t,{ta:.2f},{tb:.2f})'")
                out.append(  # bottom vignette
                    f"drawbox=x=0:y=ih-{VG_T}:w=iw:h={VG_T}:"
                    f"color=black@{a_vg:.2f}:t=fill:"
                    f"enable='between(t,{ta:.2f},{tb:.2f})'")

            # Layered drop shadow: TWO offset boxes for soft depth
            # (not a single hard drop). Outer = wider+darker, inner =
            # tighter+softer. Both ride the fade ramp.
            for (ra, rb, a) in _ramps:
                ta = rk_start + ra
                tb = b_ if rb is None else rk_start + rb
                out.append(  # outer soft shadow
                    f"drawbox=x={FX+10}:y={FY+14}:w={FW}:h={FH}:"
                    f"color=black@{a*0.22:.2f}:t={FT+4}:"
                    f"enable='between(t,{ta:.2f},{tb:.2f})'")
                out.append(  # inner tighter shadow
                    f"drawbox=x={FX+4}:y={FY+6}:w={FW}:h={FH}:"
                    f"color=black@{a*0.18:.2f}:t={FT+2}:"
                    f"enable='between(t,{ta:.2f},{tb:.2f})'")
            _staged(FX, FY, FW, FT, "white")             # top
            _staged(FX, FY+FH-FT, FW, FT, "white")       # bottom
            _staged(FX, FY, FT, FH, "white")             # left
            _staged(FX+FW-FT, FY, FT, FH, "white")       # right

            # ---- TOP-RIGHT badge (v3.1 — softer cinematic integration)
            # Layered soft shadow underneath (2 offset boxes) for real
            # depth instead of a hard rectangle. Backing alpha dropped
            # 0.97 → 0.86 so it READS through into the footage —
            # documentary classifier feel, not "UI box on top".
            BW, BH = 200, 80
            BX = FX + FW - FT - 36 - BW       # 36px inset from right edge
            BY = FY + FT + 36                  # 36px inset from top edge
            b_appear = rk_start + 1.05         # lands AFTER title (was 0.80)
            b_win = f"between(t,{b_appear:.2f},{b_:.2f})"
            # Two-layer soft shadow under the badge
            out.append(  # outer wide shadow (low alpha, big offset)
                f"drawbox=x={BX+6}:y={BY+10}:w={BW}:h={BH}:"
                f"color=black@0.16:t=fill:enable='{b_win}'")
            out.append(  # inner tighter shadow
                f"drawbox=x={BX+2}:y={BY+4}:w={BW}:h={BH}:"
                f"color=black@0.12:t=fill:enable='{b_win}'")
            out.append(  # theme-accent strip — softer (0.92 → 0.78)
                f"drawbox=x={BX}:y={BY}:w={BW}:h=5:color={hexc}@0.78:"
                f"t=fill:enable='{b_win}'")
            out.append(  # badge background — softer translucent white
                f"drawbox=x={BX}:y={BY+5}:w={BW}:h={BH-5}:"
                f"color=white@0.86:t=fill:enable='{b_win}'")
            out.append(  # very subtle outer hairline border
                f"drawbox=x={BX}:y={BY}:w={BW}:h={BH}:color=black@0.06:"
                f"t=1:enable='{b_win}'")

            # "#N" SETTLES in — gentle drop with HALVED overshoot vs v2.
            # Slower ease (quartic) + small sin bump that decays fast,
            # so the digit "lands" without snapping. Plus a tiny
            # continuous y-drift (±2 px / 3.4 s) for ambient life.
            # BUGFIX vs the v3 first cut: this used to be `BY+BH-22`
            # (= 159) which puts the text BELOW the badge — the "#1"
            # was hanging out the bottom. drawtext's y is the TOP of
            # the glyph, so to vertically centre a 58-px font inside
            # a BH-tall badge we need `BY + (BH - 58)/2` (~112) so
            # the digit sits CENTRED INSIDE the white pill.
            _NFONT = 58
            n_end_y = BY + (BH - _NFONT) // 2 + 4   # +4 = optical lift
            n_dur = 0.85                        # slower settle (was 0.70)
            n_p = f"min(1\\,max(0\\,(t-{b_appear:.2f})/{n_dur}))"
            n_ease = f"(1-pow(1-{n_p}\\,4))"
            # Even gentler bounce — amplitude down from 2.5 to 1.6 so
            # it READS as a soft land, not an animation cue.
            n_bump = f"(1.6*sin({n_p}*PI*1.5)*pow(1-{n_p}\\,3))"
            # Multi-frequency parallax: y-drift on 3.4s, x-drift on
            # 5.1s — independent periods so the badge breathes
            # asynchronously from the title (real layered depth).
            n_drift_y = f"+2.5*sin((t-{rk_start:.2f})/3.4)"
            n_drift_x = f"+1*sin((t-{rk_start:.2f})/5.1)"
            n_y = f"{n_end_y}-22*(1-{n_ease})+{n_bump}{n_drift_y}"
            n_alpha = (
                f"if(lt(t,{b_appear:.2f})\\,0\\,"
                f"if(lt(t,{b_appear+0.45:.2f})\\,"  # slower 0.45 (was 0.35)
                f"(t-{b_appear:.2f})/0.45\\,"
                f"if(lt(t,{fade_out_t:.2f})\\,1\\,"
                f"max(0\\,({b_:.2f}-t)/0.55))))"
            )
            out.append(
                f"drawtext=fontfile={font}:text='{badge}':fontsize=58:"
                f"fontcolor=black:x={BX}+({BW}-text_w)/2{n_drift_x}:"
                f"y='{n_y}':enable='{b_win}':alpha='{n_alpha}'")

            # ---- LOCATION TITLE: BOTTOM-LEFT lower-third (asymmetric)
            # Anchored 52px from the left edge so it doesn't centre
            # under the frame. Slow eased rise (0.90s — was 0.55) +
            # subtle ±3 px x-drift (different period from badge → real
            # parallax). Backing pill stays attached to the text.
            if title:
                tt_start = rk_start + 0.50     # arrives BEFORE badge
                tt_dur = 1.10                  # slower documentary slide
                tt_end_y = FY + FH - 96        # bottom area, above frame
                tt_p = f"min(1\\,max(0\\,(t-{tt_start:.2f})/{tt_dur}))"
                tt_ease = f"(1-pow(1-{tt_p}\\,3))"
                # Tiny y-drift on a slow period (4.2s) — independent
                # from badge's drift rates → real layered parallax.
                tt_drift_y = f"+1*sin((t-{rk_start:.2f})/4.2)"
                tt_y = (f"{tt_end_y}+34*(1-{tt_ease}){tt_drift_y}")
                tt_alpha = (
                    f"if(lt(t,{tt_start:.2f})\\,0\\,"
                    f"if(lt(t,{tt_start+0.70:.2f})\\,"
                    f"(t-{tt_start:.2f})/0.70\\,"  # slower 0.70 (was 0.55)
                    f"if(lt(t,{fade_out_t:.2f})\\,1\\,"
                    f"max(0\\,({b_:.2f}-t)/0.55))))"
                )
                # ±4px x-drift on a 2.6s period (was 3px). Now the
                # title moves x@2.6s + y@4.2s and the badge moves
                # x@5.1s + y@3.4s — four independent breathing rates,
                # the eye reads "depth" even though footage is still.
                tt_x = f"{FX+52}+4*sin((t-{rk_start:.2f})/2.6)"
                tt_win = f"between(t,{tt_start:.2f},{b_:.2f})"
                rk_t = workdir / f"rank_t_{i}.txt"
                rk_t.write_text(_dt(title), encoding="utf-8")
                out.append(
                    f"drawtext=fontfile={font}:textfile={rk_t.name}:"
                    f"fontsize=40:fontcolor=white:box=1:"
                    f"boxcolor=black@0.42:boxborderw=18:"  # softer 0.42
                    f"x='{tt_x}':y='{tt_y}':"
                    f"enable='{tt_win}':alpha='{tt_alpha}'")
            continue

        if kind == "evidence" and (raw or body):
            # SPLIT-SCREEN EVIDENCE / SUMMARY PANEL (Vidlore-style).
            # Footage stays full-frame on the left; a clean white panel
            # covers the right ~48% with a headline (appears first) +
            # 3-5 short bullets that fade in ONE BY ONE so the viewer
            # absorbs each as the narrator says it. Documentary, calm,
            # premium — not flashy. Hard-capped to 3 per video upstream
            # so it stays the rare high-information moment.
            panel_start = start + 0.50
            b_ = min(start + d - 0.40, emax)
            if b_ - panel_start < 2.2:
                continue   # scene too short for sequential reveals
            win = f"between(t,{panel_start:.2f},{b_:.2f})"
            # Parse bullets from body — split on '|' or newline, trim,
            # cap to 5 short lines.
            bullets = [b.strip() for b in re.split(r"[|\n]", body or "")
                       if b.strip()][:5]
            # BUGFIX (v3.2): headline + bullets used to spill OFF the
            # right edge of the screen because drawtext does NOT wrap
            # and the fonts were too large for the panel's inner width.
            # Three changes hold the layout inside the panel + frame:
            #   1) panel is narrower with a real 60 px right margin
            #      (was 30 px → text + serif kerning overshot)
            #   2) headline 72 → 54 pt, capped at 26 chars (always fits)
            #   3) bullets 44 → 34 pt with a simple word-wrap helper
            #      that breaks each bullet into <=38-char lines and
            #      uses drawtext's line_spacing so multi-line bullets
            #      stack correctly inside the panel.
            headline = _doc_clean(raw or "")[:26]
            # Panel geometry — narrower (was 910) with 60px right margin.
            # 1920 - 60 (right margin) - 880 (PW) = 980 (PX) → right
            # edge at 1860, safely inside frame.
            PX, PY, PW, PH = 980, 30, 880, 1020
            INNER = PW - 70                # text-area width (35 px padding)
            # Panel base — near-opaque white with a hairline border.
            out.append(
                f"drawbox=x={PX}:y={PY}:w={PW}:h={PH}:color=white@0.97:"
                f"t=fill:enable='{win}'")
            out.append(
                f"drawbox=x={PX}:y={PY}:w={PW}:h={PH}:"
                f"color=black@0.08:t=2:enable='{win}'")
            # Headline fade-in (0.45 s ramp after panel lands).
            h_start = panel_start + 0.35
            h_alpha = (f"if(lt(t,{h_start:.2f}),0,"
                       f"if(lt(t,{h_start+0.45:.2f}),"
                       f"(t-{h_start:.2f})/0.45,1))")
            if headline:
                ev_h = workdir / f"evid_h_{i}.txt"
                ev_h.write_text(_dt(headline), encoding="utf-8")
                out.append(
                    f"drawtext=fontfile={font}:textfile={ev_h.name}:"
                    f"fontsize=54:fontcolor=black:"     # was 72
                    f"x={PX+35}:y={PY+90}:enable='{win}':"
                    f"alpha='{h_alpha}'")
                _u_in = h_start + 0.45
                _u_win = f"between(t,{_u_in:.2f},{b_:.2f})"
                out.append(
                    f"drawbox=x={PX+35}:y={PY+170}:w=140:h=5:"
                    f"color={hexc}@0.85:t=fill:enable='{_u_win}'")

            # Simple word-wrap: break each bullet into <=`max_chars`
            # chunks at the nearest word boundary. textfile= reads the
            # newlines and drawtext renders them as separate lines
            # when line_spacing is set.
            def _wrap_bullet(text: str, max_chars: int = 38) -> str:
                words = (text or "").split()
                lines, cur = [], ""
                for w in words:
                    if len(cur) + len(w) + (1 if cur else 0) <= max_chars:
                        cur = (cur + " " + w) if cur else w
                    else:
                        if cur:
                            lines.append(cur)
                        # word longer than max → hard cut so it still fits
                        cur = w if len(w) <= max_chars else w[:max_chars-1] + "…"
                if cur:
                    lines.append(cur)
                return "\n".join(lines[:2])     # cap at 2 lines per bullet

            if bullets:
                window = max(0.6, b_ - (panel_start + 1.2))
                gap = min(0.85, max(0.55, window / max(1, len(bullets))))
                # Each bullet may now be 1 OR 2 lines — give every bullet
                # row enough vertical room so a 2-line bullet doesn't
                # crowd into the next one. 2 lines × ~40 px + 18 px
                # breathing room = 100 px per row.
                row_h = 104                     # was 88 (1-line spacing)
                for bi, btext in enumerate(bullets):
                    b_appear = panel_start + 1.20 + bi * gap
                    if b_appear > b_ - 0.5:
                        break
                    yi = PY + 240 + bi * row_h
                    b_alpha = (f"if(lt(t,{b_appear:.2f}),0,"
                               f"if(lt(t,{b_appear+0.40:.2f}),"
                               f"(t-{b_appear:.2f})/0.40,1))")
                    b_dot_win = f"between(t,{b_appear:.2f},{b_:.2f})"
                    out.append(
                        f"drawbox=x={PX+38}:y={yi+18}:w=10:h=10:"
                        f"color={hexc}@0.95:t=fill:enable='{b_dot_win}'")
                    ev_b = workdir / f"evid_b_{i}_{bi}.txt"
                    ev_b.write_text(
                        _dt(_wrap_bullet(_doc_clean(btext))),
                        encoding="utf-8")
                    out.append(
                        f"drawtext=fontfile={font}:textfile={ev_b.name}:"
                        f"fontsize=34:fontcolor=0x111111:"        # was 44
                        f"line_spacing=6:"                          # multi-line
                        f"x={PX+62}:y={yi}:enable='{win}':"
                        f"alpha='{b_alpha}'")
            continue

        if kind == "explainer" and asset.startswith("expl_") \
                and asset.endswith(".png"):
            # PREMIUM EXPLAINER CARD — FOUR layers, each with its own
            # cinematic entrance (bg fades, image panel slides in from
            # the right, title drops from the top, caption rises) plus a
            # shared slow float so the board is alive, never a static
            # slide. Subtle / intentional, never flashy.
            ad = start + 0.35
            b_ = min(start + d - 0.20, emax)
            if b_ - ad < 1.8:
                continue
            base4 = asset[:-4]
            fo = f"fade=t=out:st={b_ - 0.4:.2f}:d=0.4:alpha=1"
            win = f"between(t,{ad:.2f},{b_:.2f})"
            # shared gentle float (content layers breathe together)
            dx = f"4*sin((t-{ad:.2f})/2.4)"
            dy = f"3*sin((t-{ad:.2f})/3.2)"

            def _ease(st, dr):     # 1 -> 0, ease-out (decelerating)
                # plain commas — these go inside single-quoted overlay
                # x/y expressions (quotes already protect the commas)
                return f"pow(1-clip((t-{st:.2f})/{dr:.2f},0,1),3)"

            si, stt, stc = ad + 0.28, ad + 0.52, ad + 0.86
            # (layer, x_expr, y_expr, fade_in_start, fade_in_dur)
            specs = [
                (f"{base4}_bg.png", "0", "0", ad, 0.40),
                (f"{base4}_img.png",
                 f"{dx}+150*{_ease(si, 0.70)}", dy, si, 0.55),
                (f"{base4}_ttl.png",
                 dx, f"{dy}-80*{_ease(stt, 0.55)}", stt, 0.50),
                (f"{base4}_cap.png",
                 dx, f"{dy}+46*{_ease(stc, 0.55)}", stc, 0.55),
            ]
            for lf, xe, ye, fst, fdur in specs:
                stages.append(
                    f"movie='{lf}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"fade=t=in:st={fst:.2f}:d={fdur:.2f}:alpha=1,"
                    f"{fo}[gp{gi}]")
                stages.append(
                    f"[{{CUR}}][gp{gi}]overlay=x='{xe}':y='{ye}':"
                    f"enable='{win}'[{{OUT}}]")
                gi += 1
            out.append(  # whisper-faint grain so it isn't sterile
                f"drawgrid=width=iw:height=3:thickness=1:"
                f"color=black@0.035:enable='{win}'")
            continue

        if kind == "document" and asset.endswith("_plain.png"):
            # INVESTIGATIVE EVIDENCE REVEAL. Two pixel-identical PIL
            # pages (plain + focus-with-glowing-highlight). Both get the
            # SAME slow cinematic push-in; the focus page is then
            # cross-revealed EXACTLY when the narrator hits the point —
            # so the highlight is perfectly registered with zero
            # geometry math. A faint scan-line keeps it "alive".
            #
            # A "document" scene == we CUT TO THE EVIDENCE: hold it for
            # ~the whole scene (appear early, leave a tail) so there is
            # room for the push-in + the synced reveal — not a brief
            # 62%-in card.
            ad = start + 0.40
            b_ = min(start + d - 0.20, emax)
            if b_ - ad < 1.6:
                continue
            dur = b_ - ad
            focus = asset.replace("_plain.png", "_focus.png")
            # reveal: when the emphasis word is really spoken (Whisper);
            # else a touch after the page settles.
            if ad + 0.5 <= rev_t <= b_ - 0.9:
                rev = rev_t
            else:
                rev = ad + min(1.4, dur * 0.34)
            rdur = 0.7
            win = f"between(t,{ad:.2f},{b_:.2f})"

            # =========================================================== #
            #  EDITORIAL PUSH-IN  (per-document personality)
            # =========================================================== #
            # Linear 8% scale was robotic and barely felt alive.  Now:
            #   * stronger push (8-16% depending on document personality)
            #   * EASE-OUT curve (front-loaded -> decelerates)
            #     -- feels like an editor leaning in, not a slideshow
            #   * highlight-aware crop drift: the framing subtly slides
            #     toward the highlighted line so the viewer's eye lands
            #     on the important phrase automatically.
            raw_u = (raw or "").upper()
            if any(k in raw_u for k in (
                    "URGENT", "BREAKING", "WARNING", "ALERT", "DANGER",
                    "CRITICAL")):
                # SHOCKING EVIDENCE -- stronger, faster front-loaded push
                zoom_amt, ease_pow = 0.150, 1.6
            elif any(k in raw_u for k in (
                    "CLASSIFIED", "TOP SECRET", "COVERT", "CONFIDENTIAL",
                    "RESTRICTED", "EYES ONLY", "CIA", "KGB", "FBI",
                    "MI6", "INTERCEPT", "DECLASSIFIED", "REDACTED")):
                # INVESTIGATIVE PUSH -- strong, smooth, sustained drive
                zoom_amt, ease_pow = 0.130, 2.0
            elif any(k in raw_u for k in (
                    "LETTER", "DIARY", "DEAR ", "SINCERELY", "BELOVED",
                    "MEMORIAL", "OBITUARY", "PASSED AWAY")):
                # EMOTIONAL QUOTE -- softer, slower, sustained breath
                zoom_amt, ease_pow = 0.085, 2.8
            elif any(k in raw_u for k in (
                    "FIELD", "REPORT", "OBSERV", "STUDY", "RESEARCH",
                    "EXTENSION", "CORRESPONDENCE", "ARCHIVE")):
                # DOCUMENTARY PUSH -- mid-strong, smooth doc feel
                zoom_amt, ease_pow = 0.120, 2.2
            else:
                # DEFAULT cinematic push (still ~50% stronger than the
                # old flat 0.08, with ease-out curve).
                zoom_amt, ease_pow = 0.115, 2.0
            # Ease-out: ease(s) = 1 - (1-s)^ease_pow.  In ffmpeg's eval
            # language pow(x,y) is supported.  Output 0..1, monotonic,
            # front-loaded (slope max at s=0, slope=0 at s=1).
            _s = f"min(1\\,max(0\\,(t-{ad:.2f})/{dur:.2f}))"
            ease_expr = f"(1-pow(1-{_s}\\,{ease_pow:.2f}))"
            zexpr = f"1920*(1.0+{zoom_amt:.3f}*{ease_expr})"
            # ---- TEXT-LOCKED HIGHLIGHTER (xfade wiperight on baked focus) -
            # The marker is BAKED into `focus` by the renderer, so plain and
            # focus differ ONLY in the marker region. xfade=wiperight between
            # them produces a left->right reveal — the only visibly-changing
            # pixels are the marker columns. The xfade RESULT then goes
            # through the SAME push-in scale as a single image, so the marker
            # is GEOMETRICALLY GLUED to its phrase (no per-frame coordinate
            # math drift, no float-above-text bug). Outside the marker
            # x-range the wipe transitions plain->plain, invisible.
            try:
                import json as _json
                meta = workdir / asset.replace("_plain.png", ".json")
                hm = _json.loads(meta.read_text(encoding="utf-8")) \
                    if meta.is_file() else {}
            except Exception:                                  # noqa: BLE001
                hm = {}
            if hm.get("hl_w"):
                BW = int(hm["hl_w"])
                HX = int(hm["hl_x"])
                HY = int(hm.get("hl_y", 540))
                HH = int(hm.get("hl_h", 47))
                # ---- Highlight-aware framing via OVERLAY OFFSET ------ #
                # ffmpeg's positional crop can't take complex t-driven
                # expressions in the X/Y slots (it tries to parse them
                # as named-option keys and bails).  Instead, push the
                # FINAL overlay position by a few negative pixels in the
                # direction of the highlight as the ease curve runs --
                # same visual outcome ("camera drifts toward the line"),
                # no parser fragility.  The values stay small (<= 22%
                # of the native delta) so it's subtle, not lurching.
                _DRIFT_F = 0.22
                h_cx_native = HX + BW / 2.0
                h_cy_native = HY + HH / 2.0
                drift_ox = -(h_cx_native - 960.0) * _DRIFT_F
                drift_oy = -(h_cy_native - 540.0) * _DRIFT_F
                # NO crop -- keep the scaled image at its full (>=1920)
                # size and use overlay '(W-w)/2 + drift' to centre + drift.
                # If we crop first then drift via overlay, the cropped
                # 1920px image leaves footage exposed on one side.  By
                # leaving the scaled image oversize, ffmpeg's overlay
                # naturally clips the excess and the frame stays covered.
                crop_str = "null"
                # marker_time = perceived stroke duration (longer phrase
                # = slightly slower drag); wd_full is the FULL-FRAME wipe
                # span chosen so the marker reveal occupies marker_time.
                marker_time = max(0.55, min(0.95, BW / 700.0))
                wd_full = max(1.2, marker_time * 1920.0 / max(BW, 1))
                # ws = ideal marker-reveal-start (just after spoken word);
                # offset = global time the wipe must START so the wipe's
                # x-line reaches the marker's left edge at exactly ws.
                ws = rev + 0.10
                offset = ws - (HX / 1920.0) * wd_full
                # clamp: wipe can't start before the page has fully faded in
                _min_off = ad + 0.50
                if offset < _min_off:
                    # tighten the wipe so it still aligns with ws but starts
                    # at _min_off — keeps the marker reveal correctly timed.
                    avail = max(0.30, ws - _min_off)
                    wd_full = avail * 1920.0 / max(HX, 1)
                    offset = _min_off
                # clamp wd_full so the wipe finishes before card exits
                if offset + wd_full > b_ - 0.30:
                    wd_full = max(0.50, b_ - 0.30 - offset)
                focus = asset.replace("_plain.png", "_focus.png")
                # xfade REQUIRES CFR — force fps + yuv420p on both inputs
                # before the wipe, then re-format for compositing later.
                stages.append(
                    f"movie='{asset}',format=yuv420p,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,fps={FPS}[dp{gi}]")
                stages.append(
                    f"movie='{focus}',format=yuv420p,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,fps={FPS}[df{gi}]")
                stages.append(
                    f"[dp{gi}][df{gi}]xfade=transition=wiperight:"
                    f"duration={wd_full:.2f}:offset={offset:.2f}[dxf{gi}]")
                stages.append(
                    f"[dxf{gi}]format=rgba,scale=w='{zexpr}':h=-2:eval=frame,"
                    f"{crop_str},setsar=1,"
                    f"fade=t=in:st={ad:.2f}:d=0.45:alpha=1,"
                    f"fade=t=out:st={b_ - 0.4:.2f}:d=0.4:alpha=1[gp{gi}]")
                # Centre the oversized scaled image in the frame AND
                # add the highlight drift (ease-driven).  W/H = main canvas
                # (1920x1080), w/h = overlay's actual size after scale.
                ox_expr = f"(W-w)/2 + ({drift_ox:.1f})*{ease_expr}"
                oy_expr = f"(H-h)/2 + ({drift_oy:.1f})*{ease_expr}"
                stages.append(
                    f"[{{CUR}}][gp{gi}]overlay=x='{ox_expr}':y='{oy_expr}':"
                    f"eval=frame:enable='{win}'[{{OUT}}]")
                gi += 1
            else:
                # no marker -> plain page with eased push-in, no drift.
                # Skip crop here too -- centre the oversized scaled image
                # via overlay '(W-w)/2' so the frame stays fully covered.
                stages.append(
                    f"movie='{asset}',format=rgba,loop=loop=-1:size=1,"
                    f"setpts=N/{FPS}/TB,"
                    f"scale=w='{zexpr}':h=-2:eval=frame,"
                    f"setsar=1,"
                    f"fade=t=in:st={ad:.2f}:d=0.45:alpha=1,"
                    f"fade=t=out:st={b_ - 0.4:.2f}:d=0.4:alpha=1[gp{gi}]")
                stages.append(
                    f"[{{CUR}}][gp{gi}]overlay=x='(W-w)/2':y='(H-h)/2':"
                    f"eval=frame:enable='{win}'[{{OUT}}]")
                gi += 1

            out.append(  # faint archival scan-line over the page
                f"drawgrid=width=iw:height=3:thickness=1:"
                f"color=black@0.05:enable='{win}'")
            continue

        if kind in ("number", "stat") and _fig_text(raw):
            txt = _dt(_fig_text(raw))
            b_ = min(a + max(2.4, min(d - 0.35, 3.6)), emax)
            if b_ - a < 1.0:
                continue
            win = f"between(t,{a:.2f},{b_:.2f})"
            fs = 168 if len(txt) <= 6 else 120 if len(txt) <= 12 else 84
            pn = _parse_number(raw)
            if pn is None or pn[1] < 10:
                # too small / not countable -> clean slam (old behaviour)
                alpha = (
                    f"if(lt(t,{a:.2f}),0,"
                    f"if(lt(t,{a + 0.22:.2f}),(t-{a:.2f})/0.22,"
                    f"if(lt(t,{b_ - 0.35:.2f}),1,"
                    f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.35,0))))"
                )
                out.append(  # designed hero stat — clean heavy white,
                    # soft cinematic shadow (no chunky meme outline)
                    f"drawtext=fontfile={font}:text='{txt}':"
                    f"expansion=none:fontsize={fs}:fontcolor=white:"
                    "shadowcolor=black@0.6:shadowx=0:shadowy=6:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:"
                    f"alpha='{alpha}':enable='{win}'")
                out.append(  # accent kicker rule under it (Vidlore look)
                    f"drawbox=x=(iw-240)/2:y=(ih+{fs})/2+22:w=240:h=5:"
                    f"color={hexc}:t=fill:enable='{win}'")
                num_events.append((a, a))      # single impact stinger
                continue

            # ---- CINEMATIC COUNT-UP ------------------------------- #
            _pfx, val, _sfx, final = pn
            final = _dt(final)                # shape Arabic units e.g. "سنة"
            ROLL = min(0.85, max(0.45, (b_ - a) * 0.42))
            L = a + ROLL                       # the number LANDS here
            cy = "(h-text_h)/2"
            # rolling digits: ease-out 0 -> val, quick fade-in
            eif = (
                f"%{{eif\\:floor({val}*(1-pow(1-"
                f"clip((t-{a:.2f})/{ROLL:.2f}\\,0\\,1)\\,3)))\\:d}}"
            )
            ra = (f"if(lt(t,{a:.2f}),0,"
                  f"if(lt(t,{a + 0.10:.2f}),(t-{a:.2f})/0.10,1))")
            out.append(  # rolling digits — premium clean type
                f"drawtext=fontfile={font}:expansion=normal:text='{eif}':"
                f"fontsize={fs}:fontcolor=white:"
                "shadowcolor=black@0.55:shadowx=0:shadowy=5:"
                f"x=(w-text_w)/2:y={cy}:alpha='{ra}':"
                f"enable='between(t,{a:.2f},{L:.2f})'")
            # landing: full formatted value, accent GLOW + settle bounce,
            # held for a dramatic pause
            lwin = f"between(t,{L:.2f},{b_:.2f})"
            la = (
                f"if(lt(t,{L:.2f}),0,"
                f"if(lt(t,{L + 0.10:.2f}),(t-{L:.2f})/0.10,"
                f"if(lt(t,{b_ - 0.35:.2f}),1,"
                f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.35,0))))"
            )
            yb = (
                f"{cy} + if(lt(t,{L:.2f}),0,"
                f"if(lt(t,{L + 0.10:.2f}),-18*((t-{L:.2f})/0.10),"
                f"if(lt(t,{L + 0.30:.2f}),"
                f"-18*(1-((t-{L + 0.10:.2f})/0.20)),0)))"
            )
            out.append(  # soft accent glow behind the landed value
                f"drawtext=fontfile={font}:text='{final}':"
                f"expansion=none:fontsize={fs}:fontcolor={hexc}@0.0:"
                f"borderw={max(10, fs // 7)}:bordercolor={hexc}@0.5:"
                f"x=(w-text_w)/2:y='{yb}':alpha='{la}':enable='{lwin}'")
            out.append(  # final value — clean heavy white, soft shadow
                f"drawtext=fontfile={font}:text='{final}':"
                f"expansion=none:fontsize={fs}:fontcolor=white:"
                "shadowcolor=black@0.6:shadowx=0:shadowy=6:"
                f"x=(w-text_w)/2:y='{yb}':alpha='{la}':enable='{lwin}'")
            out.append(  # accent kicker rule under the landed value
                f"drawbox=x=(iw-240)/2:y=(ih+{fs})/2+22:w=240:h=5:"
                f"color={hexc}:t=fill:enable='{lwin}'")
            num_events.append((a, L))          # ticks [a,L] + impact @L
            continue

        if kind == "document" and _doc_clean(raw):
            title = _doc_clean(raw)[:80]
            bd = _doc_clean(body)[:300]
            (workdir / f"doc{i}_t.txt").write_text(
                _dt(title), encoding="utf-8")
            wrapped = "\n".join(textwrap.wrap(bd, 50)[:6]) if bd else ""
            (workdir / f"doc{i}_b.txt").write_text(
                _dt(wrapped or title), encoding="utf-8")
            b_ = min(a + max(3.0, min(d - 0.4, 8.0)), emax)
            if b_ - a < 1.4:
                continue
            pa = (
                f"if(lt(t,{a:.2f}),0,"
                f"if(lt(t,{a + 0.40:.2f}),(t-{a:.2f})/0.40,"
                f"if(lt(t,{b_ - 0.45:.2f}),1,"
                f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.45,0))))"
            )
            win = f"between(t,{a:.2f},{b_:.2f})"
            # slow downward drift on the text = subtle camera move
            dy = f"+(min(28,(t-{a:.2f})*4))"
            # 1) drop shadow  2) aged paper  3) accent header bar
            out.append(
                f"drawbox=x=156:y=120:w=1620:h=850:color=black@0.55:"
                f"t=fill:enable='{win}'")
            out.append(
                f"drawbox=x=140:y=104:w=1620:h=850:color=0xEDE7D8@0.96:"
                f"t=fill:enable='{win}'")
            out.append(
                f"drawbox=x=140:y=104:w=1620:h=92:color={hexc}@0.92:"
                f"t=fill:enable='{win}'")
            out.append(  # thin divider under header
                f"drawbox=x=140:y=196:w=1620:h=3:color=0x33312B@0.9:"
                f"t=fill:enable='{win}'")
            out.append(  # ARCHIVE kicker (small, on the bar)
                f"drawtext=fontfile={font}:text='ARCHIVE':fontsize=24:"
                f"fontcolor=white@0.85:x=1640:y=138:enable='{win}'")
            out.append(  # title on the accent header
                f"drawtext=fontfile={font}:textfile=doc{i}_t.txt:"
                "fontsize=46:fontcolor=white:"
                f"x=180:y=124:alpha='{pa}':enable='{win}'")
            out.append(  # body, serif-clean, slow drift
                f"drawtext=fontfile={font}:textfile=doc{i}_b.txt:"
                "fontsize=37:fontcolor=0x1b1b1b:line_spacing=18:"
                f"x=190:y='250{dy}':alpha='{pa}':enable='{win}'")
            continue

        # default: clean cinematic title (location / label / term).
        txt = _dt(_card_text(raw))
        if not txt:
            continue
        b_ = min(a + max(2.0, min(d - 0.6, 3.2)), emax)
        if b_ - a < 0.8:
            continue
        fs = 58
        alpha = (
            f"if(lt(t,{a:.2f}),0,"
            f"if(lt(t,{a + 0.30:.2f}),(t-{a:.2f})/0.30,"
            f"if(lt(t,{b_ - 0.40:.2f}),1,"
            f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.40,0))))"
        )
        win = f"between(t,{a:.2f},{b_:.2f})"
        # VARY placement like a human editor — never the same spot. Each
        # preset has its own x/y and a matching subtle slide-in.
        sl = f"40*(1-min(1,(t-{a:.2f})/0.30))"      # ease-in offset
        presets = [
            (f"(w-text_w)/2", f"(h*0.17)-{sl}"),       # upper-centre
            (f"96+{sl}", "h*0.80"),                    # lower-left
            (f"w-text_w-96-{sl}", "h*0.15"),           # upper-right
            (f"(w-text_w)/2", f"h*0.80+{sl}"),         # lower-centre
            (f"110+{sl}", "h*0.16"),                   # upper-left
        ]
        px, py = presets[i % len(presets)]
        base_dt = (
            f"drawtext=fontfile={font}:text='{{T}}':"
            f"fontsize={fs}:fontcolor=white:borderw=2:"
            "bordercolor=black@0.45:shadowcolor=black@0.55:"
            f"shadowx=0:shadowy=4:x='{px}':y='{py}':"
            f"alpha='{alpha}':enable='{{EN}}'"
        )
        if kind == "location":
            # location / date cards get a subtle TYPED terminal cursor
            # (the 'SINCE 1960S|' look). Exact placement with NO width
            # guessing: draw two MUTUALLY-EXCLUSIVE variants — plain text
            # vs text+caret — and let ffmpeg lay the caret out itself.
            # The 0.30s after appear has no caret (it just "typed in"),
            # then a 0.9s blink. Used only here so it never repeats.
            on = (f"between(t,{a:.2f},{b_:.2f})*"
                  f"gte(t,{a + 0.30:.2f})*lt(mod(t-{a:.2f},0.9),0.45)")
            off = (f"between(t,{a:.2f},{b_:.2f})*"
                   f"(lt(t,{a + 0.30:.2f})+gte(mod(t-{a:.2f},0.9),0.45))")
            out.append(base_dt.replace("{T}", txt + "|").replace("{EN}", on))
            out.append(base_dt.replace("{T}", txt).replace("{EN}", off))
        else:  # clean cinematic title — no box, no accent line
            out.append(base_dt.replace("{T}", txt).replace("{EN}", win))
    return out, stages, post, num_events


# Every CARD KIND that already paints its own text on-screen suppresses
# the per-scene keyphrase stab below.  The earlier list (chart, explainer,
# document, portrait) missed every PERSON / TITLE / LOWER-THIRD card,
# letting the same word land twice: once big on the card AND once
# stabbed at the bottom.  User feedback: "when a text element is already
# on screen, the bottom caption shouldn't come -- it looks unprofessional."
_FULLSCREEN_G = frozenset((
    # graphic visual cards
    "chart", "explainer",
    # document family
    "document", "case_file", "classified", "classified_dossier",
    "redacted", "newspaper", "article", "diary", "letter", "press",
    "letterhead", "telegram", "text_on_black",
    # person / identity cards (the big offenders)
    "portrait", "name_reveal", "lower_third", "mini_bio", "id_card",
    "pull_quote", "pull_quote_portrait", "studio_two_shot",
    "police_case_file", "mugshot", "family_tree", "mini_bio_card",
    # title / chapter / era banners
    "title_card", "chapter_marker", "era_banner", "act_break",
    # news / broadcast
    "news_broadcast", "breaking_news", "live_news", "ticker",
    "headline_crawl",
    # quote / monument / inscription
    "quote_highlight", "long_quote", "inscription", "monument", "plaque",
    # data-as-text cards
    "stat_dashboard", "big_stat", "score_display", "currency",
    "verdict", "did_you_know", "trivia",
    # surveillance / dossier text overlays
    "surveillance", "evidence_tag", "footnote_citation", "disclaimer",
))


def _actbreak_filters(
    durs: list[float], roles: list[str], energies: list[int],
) -> list[str]:
    """ISSUE #12 — the edit is already 100% hard cuts with smart uneven
    timing (the old repetitive crossfades are gone). The one thing a
    great documentary editor adds on top: a RARE, motivated punctuation
    at a genuine act break — a quick light-flash into the reveal/climax.
    NOT a transition on every cut (that's the cheap/repetitive tell).

    Implemented as a brief full-frame white pulse on the FINAL
    concatenated stream at the absolute boundary time — the same proven
    overlay mechanism as the key-phrase stabs, so there is NO xfade
    chain (that caused the freeze) and zero effect on timing/sync.
    Capped at 2 per video so it stays special, never a habit."""
    cand: list[tuple[int, float]] = []   # (energy_at, boundary_time)
    t = 0.0
    for i, d in enumerate(durs):
        if i > 0:
            r = (roles[i] if i < len(roles) else "") or ""
            e = energies[i] if i < len(energies) else 2
            ep = energies[i - 1] if i - 1 < len(energies) else 2
            motivated = (r in ("reveal", "climax", "turn") and e >= 4) \
                or (e - ep >= 2 and e >= 5)
            if motivated:
                cand.append((e, t))
        t += d
    # keep only the 1-2 strongest (rarest = most powerful)
    cand.sort(key=lambda c: -c[0])
    picks = sorted(bt for _e, bt in cand[:2])
    out: list[str] = []
    for bt in picks:
        # soft tail (dim, ~5 frames) + bright core (~2 frames): reads as
        # a real camera/light hit, not a crude on/off block.
        out.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.16:t=fill:"
            f"enable='between(t,{max(0.0, bt - 0.04):.2f},{bt + 0.16:.2f})'")
        out.append(
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.42:t=fill:"
            f"enable='between(t,{bt:.2f},{bt + 0.07:.2f})'")
    return out


def _keyphrase_filters(
    narr_scenes, emphasis: list[str], font: str | None, accent: tuple,
    shot_types: list[str] | None = None,
    graphic_kinds: list[str] | None = None,
    roles: list[str] | None = None,
    intensities: list[int] | None = None,
) -> list[str]:
    """Engaging-but-restrained on-screen text. NOT full captions (those
    read busy/cheap). The ONE charged word the editor-LLM picked per
    scene is stabbed on screen at the EXACT second it's spoken.

    ISSUE #10 — SCENE-AWARE typography (was a blind i%5 cycle that could
    slap text over a face or on top of a full-screen card):
      * a scene already showing a full-screen graphic card (chart /
        explainer / document / portrait) SUPPRESSES the stab — the card
        already owns the screen; overlapping = clutter.
      * placement is chosen from the shot's COMPOSITION: portrait /
        reaction / tracking -> safe lower-third (never over the face);
        macro / detail (subject fills frame) -> bottom band + a soft
        readability scrim; wide / establishing / aerial (negative
        space) -> free varied placement.
      * a subtle dark scrim sits behind the text ONLY on busy shots so
        it reads as an integrated lower-third, not text slapped on
        (kept off clean wide shots — no chunky box, per the user)."""
    if font is None or not emphasis:
        return []
    shot_types = shot_types or []
    graphic_kinds = graphic_kinds or []
    r, g, b = accent
    hexc = f"0x{r:02X}{g:02X}{b:02X}"
    out: list[str] = []
    # Pre-compute per-scene editorial mode from roles/intensities so
    # DENSITY scenes always paint the stab on screen (genre baseline
    # = ~65 % type-on-frame) while RESTRAINT scenes only stab on the
    # true climax/punch line.  Restraint stabs are randomised below
    # via emphasis_fire_prob ≈ 0.55 from script_gen.mode_weights.
    try:
        from .script_gen import _ROLE_DENSITY, _ROLE_RESTRAINT
        _scene_mode_l: list[str] = []
        for j in range(len(narr_scenes)):
            r_ = (roles[j] if roles and j < len(roles) else "") or ""
            r_ = r_.lower().strip()
            it_ = (intensities[j] if intensities and j < len(intensities)
                   else 3) or 3
            if it_ >= 5:
                _scene_mode_l.append("restraint")
            elif r_ in _ROLE_RESTRAINT:
                _scene_mode_l.append("restraint")
            elif r_ in _ROLE_DENSITY:
                _scene_mode_l.append("density")
            else:
                _scene_mode_l.append("density" if it_ >= 3 else "restraint")
    except Exception:                                          # noqa: BLE001
        _scene_mode_l = ["density"] * len(narr_scenes)

    for i, ns in enumerate(narr_scenes):
        gk = (graphic_kinds[i] if i < len(graphic_kinds) else "") or ""
        if gk in _FULLSCREEN_G:           # card owns the screen — no stab
            continue
        # Mode-aware restraint: skip ~45 % of restraint-mode stabs
        # (only the truly punched words land); always fire on density
        # scenes (the genre demands type-on-screen).
        _sm = _scene_mode_l[i] if i < len(_scene_mode_l) else "density"
        # Look DNA per-channel fire probability — slow historical
        # channels stab less often; density-heavy channels stab on
        # every charged word.  Density mode default = 1.0 (always
        # fire), restraint default = 0.55.  Both overridable per
        # channel via ``look.text.emphasis_fire_prob_{density,restraint}``.
        try:
            from .look_dna import look_get as _lg_ef
            if _sm == "restraint":
                _fire_p = float(_lg_ef("text.emphasis_fire_prob_restraint",
                                       default=0.55))
            else:
                _fire_p = float(_lg_ef("text.emphasis_fire_prob_density",
                                       default=1.0))
        except Exception:                                          # noqa: BLE001
            _fire_p = 0.55 if _sm == "restraint" else 1.0
        if _fire_p < 1.0:
            # Deterministic skip pattern (seeded by scene index) so
            # the same render always produces the same result.
            if _rng01(i * 311 + 7) > _fire_p:
                continue
        shot = (shot_types[i] if i < len(shot_types) else "") or ""
        e = (emphasis[i] if i < len(emphasis) else "") or ""
        etoks = [t for t in re.findall(r"[\w']+", e.lower(),
                                       re.UNICODE) if t]
        words = getattr(ns, "words", None) or []
        if not etoks or not words:
            continue
        # find the contiguous run of this scene's words == the phrase
        norm = [re.sub(r"[^\w]", "", w.word.lower(), flags=re.UNICODE)
                for w in words]
        et = [re.sub(r"[^\w]", "", x, flags=re.UNICODE) for x in etoks if x]
        hit = None
        for s in range(len(norm) - len(et) + 1):
            if norm[s:s + len(et)] == et:
                hit = (words[s].start, words[s + len(et) - 1].end)
                break
        if hit is None:                       # fallback: first token only
            for s, nw in enumerate(norm):
                if nw and nw == et[0]:
                    hit = (words[s].start, words[s].end)
                    break
        if hit is None:
            continue
        txt = _dt(_card_text(e))
        if not txt:
            continue
        a = max(0.0, hit[0] - 0.06)
        b_ = max(a + 1.25, hit[1] + 0.85)
        b_ = min(b_, a + 2.6)
        # ISSUE #8 — is THIS the charged line? (the picked word itself,
        # or any spoken word in the scene). Impact words POP: bigger,
        # accent-coloured, with a quick accent bloom — vs the clean
        # white stab for a normal key word.
        imp = _is_impact_text(e) or _is_impact_text(
            " ".join(w.word for w in words))
        fs = 78 if len(txt) <= 10 else 60 if len(txt) <= 18 else 46
        if imp:
            fs = int(fs * 1.24)
        alpha = (
            f"if(lt(t,{a:.2f}),0,"
            f"if(lt(t,{a + 0.16:.2f}),(t-{a:.2f})/0.16,"
            f"if(lt(t,{b_ - 0.30:.2f}),1,"
            f"if(lt(t,{b_:.2f}),({b_:.2f}-t)/0.30,0))))"
        )
        win = f"between(t,{a:.2f},{b_:.2f})"
        # quick 0.18s settle so it still reads as a punch, not a title.
        sl = f"26*(1-min(1,(t-{a:.2f})/0.18))"
        shot_l = shot.lower()
        if shot_l in ("portrait", "reaction", "tracking"):
            # subject is a person / moving subject -> SAFE lower-third,
            # never over the face; busy frame -> readability scrim.
            px, py, scrim = f"(w-text_w)/2", f"h*0.80+{sl}", True
        elif shot_l in ("macro", "detail"):
            # subject fills the frame -> bottom band + scrim (any
            # placement covers something here)
            px, py, scrim = f"(w-text_w)/2", f"h*0.83+{sl}", True
        else:
            # wide / establishing / aerial / unknown -> negative space,
            # free varied placement, no scrim (clean bg reads fine)
            free = [
                (f"(w-text_w)/2", f"h*0.74+{sl}"),       # lower-centre
                (f"96+{sl}", "h*0.30"),                  # upper-left 3rd
                (f"w-text_w-96-{sl}", "h*0.75"),         # lower-right
                (f"96+{sl}", "h*0.75"),                  # lower-left
                (f"w-text_w-96-{sl}", "h*0.27"),         # upper-right
            ]
            px, py = free[i % len(free)]
            scrim = False
        if scrim:
            # subtle integrated lower-third darkening behind the word
            # (centred band, low alpha — NOT a chunky box; only on busy
            # shots so most scenes stay clean per the user's preference)
            # NOTE: in drawbox, `h`/`w` in y/x exprs are the BOX dims, not
            # the frame — must use ih/iw for the frame (drawtext differs).
            by = "ih*0.80" if shot_l in ("macro", "detail") else "ih*0.775"
            bh = int(fs * 1.7)
            out.append(
                f"drawbox=x=(iw-1000)/2:y={by}:w=1000:h={bh}:"
                f"color=black@0.17:t=fill:enable='{win}'")
        if imp:
            # high-impact word: a quick ACCENT BLOOM punch (blooms in
            # ~0.16s then settles) behind the word, then the word itself
            # in the accent colour — a real "pop", not a plain fade.
            bloom = (
                f"if(lt(t,{a:.2f}),0,"
                f"if(lt(t,{a + 0.16:.2f}),0.55*((t-{a:.2f})/0.16),"
                f"if(lt(t,{a + 0.42:.2f}),"
                f"0.55*(1-((t-{a + 0.16:.2f})/0.26)),0)))"
            )
            out.append(
                f"drawtext=fontfile={font}:text='{txt}':expansion=none:"
                f"fontsize={fs}:fontcolor={hexc}@0.0:"
                f"borderw=22:bordercolor={hexc}:"
                f"x='{px}':y='{py}':alpha='{bloom}':enable='{win}'"
            )
            out.append(
                f"drawtext=fontfile={font}:text='{txt}':expansion=none:"
                f"fontsize={fs}:fontcolor={hexc}:borderw=2:"
                "bordercolor=black@0.55:shadowcolor=black@0.65:"
                f"shadowx=0:shadowy=5:"
                f"x='{px}':y='{py}':alpha='{alpha}':enable='{win}'"
            )
            continue
        # Premium documentary "word stab": clean bold type, a soft
        # cinematic drop-shadow + hairline edge for legibility — NO
        # chunky meme outline, NO accent underline (read as designed,
        # not "plain text slapped on").
        out.append(
            f"drawtext=fontfile={font}:text='{txt}':expansion=none:"
            f"fontsize={fs}:fontcolor=white:borderw=2:"
            "bordercolor=black@0.45:shadowcolor=black@0.6:"
            f"shadowx=0:shadowy=4:"
            f"x='{px}':y='{py}':alpha='{alpha}':enable='{win}'"
        )
    return out


# ISSUE #8 — high-impact line detection. When the narration hits a
# charged line, a human editor PUNCHES it (zoom-in + a popped word)
# instead of treating it like any other sentence. These are the trigger
# words; a scene is also "impact" if the showrunner marked it a
# reveal/climax/turn or pinned it at peak intensity.
_IMPACT_WORDS = frozenset((
    "deadly", "death", "die", "died", "dead", "kill", "killed", "killing",
    "fatal", "lethal", "poison", "poisoned", "toxic", "danger",
    "dangerous", "warning", "warned", "shocking", "shock", "shocked",
    "never", "hidden", "hide", "secret", "secretly", "buried", "bury",
    "banned", "forbidden", "destroyed", "destroy", "survive", "survival",
    "survived", "disaster", "catastrophe", "collapse", "exposed",
    "truth", "forever", "vanished", "erased", "covered", "conspiracy",
    "threat", "terrifying", "horror", "nightmare", "irreversible",
    "permanent", "epidemic", "outbreak", "weapon", "attack", "trapped",
))


def _is_impact_text(text: str) -> bool:
    """True if a line carries a charged, high-impact word."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return any(t in _IMPACT_WORDS for t in toks)


def _kenburns(mode: int, nf: int, energy: int = 2,
              seed: int = 0, impact: bool = False,
              hold: bool = False, drift: float = 1.0) -> tuple[str, str, str]:
    """Per-still motion, varied per scene (zoom-in / zoom-out / pan L-R /
    pan T-B) AND scaled by emotional energy: calm scenes drift slowly,
    intense scenes push harder/faster — visual pacing that builds
    tension. Returns (z, x, y) zoompan exprs (commas escaped for -vf).

    ISSUE #2 — every calm scene used to get the exact same drift speed
    and zoom amount (a mechanical tell). Now each shot's rate / zoom /
    direction get a small SEEDED jitter so no two moves are identical,
    and on a calm beat the camera sometimes simply HOLDS (a near
    locked-off shot) — a real editor lets an emotional image breathe
    instead of always pushing in."""
    e = max(1, min(5, energy))
    r1, r2, r3 = (_rng01(seed * 3 + 1), _rng01(seed * 3 + 2),
                  _rng01(seed * 7 + 5))
    # GENTLE, barely-noticeable drift — premium docs use a soft slow
    # move, not an obvious zoom. Calm scenes ~6%, climax ~12%, plus a
    # +/-18% human jitter so the move is never twice the same.
    # Style mode scales the drift: an epic uses slower, grander moves
    # (drift<1) so the picture glides; a tense mode could push harder.
    dr = max(0.3, min(1.6, float(drift)))
    rate = (0.00035 + 0.00022 * e) * (0.82 + 0.36 * r1) * dr
    zmax = 1.0 + (((1.055 + 0.012 * e) * (0.986 + 0.028 * r2)) - 1.0) * dr
    # ~1 in 4 CALM beats: let it breathe — near locked-off (intentional
    # held shot, the human "don't move the camera here" choice).
    if e <= 2 and r3 < 0.26:
        rate, zmax = 0.00009, 1.012
    # VISUAL RESTRAINT (Human-Editor #3): an emotionally-motivated HOLD.
    # The editor chose to leave this beat still (an emotional reaction, a
    # quiet resolution) so the image and the words carry it — no drift to
    # distract. Forced regardless of energy (an emotional beat isn't
    # always "low energy"); the handheld micro-float below still keeps it
    # alive so it reads as a held camera, not a frozen JPEG. Skipped when
    # the beat is a deliberate impact push-in (that wins).
    if hold and not impact:
        rate, zmax = 0.00007, 1.010
    # ISSUE #9 — organic HANDHELD micro-float. A perfectly linear digital
    # zoom on a static still is THE "slideshow / AI montage" tell. Real
    # documentary footage breathes: the operator's hand drifts a few
    # pixels on two slow, out-of-phase sines (per-scene seeded phase, x
    # and y on different periods so it floats, never orbits). Tiny vs
    # _VINTAGE's deliberate archival gate-weave — barely perceptible,
    # just "alive". Applies to every still (incl. a 'held' shot, so even
    # that reads as a person holding the camera, not a frozen JPEG).
    phx = _rng01(seed * 5 + 2) * 6.283
    phy = _rng01(seed * 5 + 9) * 6.283
    swx = f"+9*sin(on/34+{phx:.2f})+5*sin(on/79)"
    swy = f"+8*sin(on/41+{phy:.2f})+4*sin(on/97)"
    cx = f"iw/2-(iw/zoom/2){swx}"
    cy = f"ih/2-(ih/zoom/2){swy}"
    # ── EASE-OUT-CUBIC zoom envelope ─────────────────────────────────
    # A linear zoom (z = 1 + rate*on, capped at zmax) is the single
    # most "automated slideshow" tell in motion-graphics.  Real dops
    # decelerate INTO the final framing — fast off the start, soft
    # arrival.  Ease-out-cubic: ease = 1 - (1-p)^3 where p = on/nf.
    # The visible motion is identical in TOTAL travel, but it lands
    # gracefully instead of hitting the cap.  The drift case still
    # uses ease (just over a longer window), and the impact push-in
    # uses an even more pronounced ease so the "stop on the word"
    # feels intentional.
    # IMP_024 (slow_motivated_push_in, DNA cinematography gap) — a gentle
    # drift used to reach its small zoom cap in ~3s (nf_c) and then FREEZE for
    # the rest of the shot: the "push then hold a still JPEG" slideshow tell.
    # Premium docs keep the push CREEPING across the whole shot (a slow 5-8s
    # move). So for the non-impact moves we stretch the ease window to land
    # near the END of the beat (`span_nf`) — this only ever SLOWS an
    # early-finishing push (never speeds one up) and keeps total travel
    # (zmax) identical. The impact punch is left on its tight nf_c timing so
    # it still "lands on the word".
    def _ease_in_z(rate_v: float, zmax_v: float,
                   span_nf: float | None = None) -> str:
        # frames-to-cap at linear rate
        nf_c = max(1.0, (zmax_v - 1.0) / max(rate_v, 1e-6))
        win = nf_c if span_nf is None else max(nf_c, span_nf)
        return (f"1+({zmax_v:.4f}-1)*"
                f"(1-pow(1-min(1\\,on/{win:.1f})\\,3))")
    def _ease_out_z(rate_v: float, zmax_v: float,
                    span_nf: float | None = None) -> str:
        # zoom OUT mirrors: start at zmax, decelerate into 1.0
        nf_c = max(1.0, (zmax_v - 1.0) / max(rate_v, 1e-6))
        win = nf_c if span_nf is None else max(nf_c, span_nf)
        return (f"1+({zmax_v:.4f}-1)*"
                f"pow(1-min(1\\,on/{win:.1f})\\,3)")
    # finish the gentle move near the end of the shot (continuous creep)
    _span = max(1.0, nf * 0.92)

    if impact:
        # ISSUE #8 — a charged line: deliberate, noticeably stronger
        # PUSH-IN ("zoom in on the important moment"), centered, never a
        # locked-off hold. Still smooth/cinematic, not a snap.
        rate = 0.00075 + 0.00010 * e        # markedly faster than drift
        zmax = 1.15 + 0.012 * e             # 1.16 .. 1.21
        z_in = _ease_in_z(rate, zmax)
        return (z_in, cx, cy)
    z_in = _ease_in_z(rate, zmax, _span)
    z_out = _ease_out_z(rate, zmax, _span)
    panx = f"(iw-iw/zoom)*on/{nf}{swx}"
    pany = f"(ih-ih/zoom)*on/{nf}{swy}"
    return [
        (z_in, cx, cy),     # 0 zoom in, centered
        (z_out, cx, cy),    # 1 zoom out, centered
        (z_in, panx, cy),   # 2 zoom in + pan left->right
        (z_in, cx, pany),   # 3 zoom in + pan top->bottom
    ][mode % 4]


# ISSUE #7 — visual-fatigue prevention. The old map sent shot_type to a
# SINGLE motion, so detail/macro/portrait/reaction ALL collapsed to
# push-in: ~53% of scenes used the identical move and many ran
# back-to-back ("repeated motion / same visual energy"). Now each
# shot_type carries an ORDERED PALETTE of semantically-fitting moves
# (>=2 each), and the assembler picks the first that differs from the
# previous scene — guaranteeing the camera language never repeats
# back-to-back and the full move vocabulary is used.
#   modes: 0 push-in · 1 pull-out · 2 pan L-R · 3 pan T-B
_SHOT_KB = {
    "establishing": [1, 2, 3],   # reveal — pull-out / drift, never push
    "aerial":        [1, 2, 3],
    "wide":          [1, 3, 2],
    "detail":        [0, 2, 3],  # intimate push, with pan variety
    "macro":         [0, 3, 2],
    "portrait":      [0, 3],     # push-in or a gentle vertical
    "reaction":      [0, 1],     # push-in or a slight pull
    "tracking":      [2, 3, 1],  # lateral drift primary
    "archival":      [3, 2],     # vertical drift (also has _VINTAGE)
}


# ═══════════════════════════════════════════════════════════════════════
# ENCODE-POOL RELIABILITY (2026-05-31) — file-readiness validation +
# a slate safety net that CANNOT raise.
#
# Root cause fixed: under the default multi-worker encode pool a scene
# whose footage clip is partially-written / zero-byte / corrupt made all
# four ffmpeg fallback tiers fail; the last tier (the "guaranteed" graded
# slate) was the ONE unguarded `run()` in `_scene_video`, so when its
# subprocess spawn was rejected under fd/process pressure (OSError EAGAIN)
# the exception propagated and `pool.map` abandoned the whole render.
# These helpers (a) validate a clip BEFORE the doomed encode tiers run,
# and (b) make the slate genuinely unable to raise.
# ═══════════════════════════════════════════════════════════════════════

_SLATE_VENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-colorspace", "bt709",
               "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv"]


def _clip_seek_ok(path, dur_s: float, *, ff: str | None = None,
                  fracs=(0.25, 0.5, 0.75, 0.97)) -> tuple[bool, str]:
    """STRONG seek-table validation (V3.2.2). A clip can decode from the START
    yet fail on a later SEEK when its stsc/stco/stss index tables are corrupt or
    the mdat is truncated — exactly the archive.org failure that crashed assembly
    past the lightweight 3-frame check. For each fraction of the duration, fast
    input-seek (`-ss` before `-i`, as xfade/trim do) + decode one frame at
    `-v error`; ANY non-zero rc or stderr error → reject. Returns (ok, reason).
    Never raises. Bounded by short per-seek timeouts so a pathological clip fails
    fast instead of hanging the validator."""
    ff = ff or ffmpeg_exe()
    if dur_s <= 0:
        return False, "no-duration-for-seek"
    for fr in fracs:
        t = max(0.0, min(dur_s - 0.05, dur_s * fr))
        try:
            r = subprocess.run(
                [ff, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                 "-frames:v", "1", "-f", "null", "-"],
                capture_output=True, text=True, timeout=30)
        except Exception as e:                                  # noqa: BLE001
            return False, f"seek-raised@{fr:.2f}({type(e).__name__})"
        err = (r.stderr or "").strip()
        if r.returncode != 0:
            return False, f"seek-rc@{fr:.2f}({r.returncode})"
        # valid clips emit 0 error lines at -v error; a corrupt seek table
        # ("partial file", "Invalid NAL", "moov atom", "STSC/STCO") emits ≥1.
        if err:
            return False, f"seek-stderr@{fr:.2f}({err.splitlines()[0][:40]})"
    return True, "seek-ok"


def _clip_ready(path, *, want_video: bool = True, min_bytes: int = 20_000,
                decode_frames: int = 3, strong: bool = False,
                min_dur_s: float = 0.4) -> tuple[bool, str]:
    """Validate that a footage file is safe to hand to the encoder.

    FAST path (default): exists · size STABLE across two stats (not still being
    written) · size ≥ min_bytes · ffmpeg reads a video stream + a real (non-N/A)
    duration · a short decode of the first `decode_frames` frames returns 0.

    STRONG path (`strong=True`, used right before a clip joins the FINAL timeline /
    assembly — V3.2.2): additionally requires duration ≥ `min_dur_s` · a FULL
    decode (`-f null -`) with rc 0 AND no stderr errors (catches truncation / mid-
    stream corruption a 3-frame decode misses) · and a multi-point SEEK test
    (`_clip_seek_ok`) catching broken stsc/stco seek tables. Returns ``(ok,
    reason)`` — never raises. Pure read-only; safe to call from any worker."""
    import time as _t
    p = Path(path)
    try:
        if not p.exists():
            return False, "missing"
        s1 = p.stat().st_size
        if s1 == 0:
            return False, "zero-byte"
        _t.sleep(0.04)
        s2 = p.stat().st_size
        if s1 != s2:
            return False, f"size-unstable({s1}->{s2}, still writing)"
        if s1 < min_bytes:
            return False, f"too-small({s1}<{min_bytes})"
    except OSError as e:
        return False, f"stat-error({e})"
    ff = ffmpeg_exe()
    # probe: does ffmpeg see a usable stream?
    try:
        pr = subprocess.run([ff, "-hide_banner", "-i", str(p)],
                            capture_output=True, text=True, timeout=25)
        info = pr.stderr or ""
    except Exception as e:                                  # noqa: BLE001
        return False, f"probe-raised({type(e).__name__})"
    if want_video and "Video:" not in info:
        return False, "no-video-stream"
    dur_s = 0.0
    if want_video:
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
        dur_s = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                 + float(m.group(3))) if m else 0.0
        if not m or dur_s <= 0.0:
            return False, "no-duration"
    # decode test: actually pull the first few frames (catches truncated /
    # corrupt streams that probe but won't decode).
    try:
        dec = subprocess.run(
            [ff, "-v", "error", "-i", str(p), "-frames:v", str(decode_frames),
             "-f", "null", "-"], capture_output=True, text=True, timeout=40)
        if dec.returncode != 0:
            return False, f"decode-failed(rc={dec.returncode})"
    except Exception as e:                                  # noqa: BLE001
        return False, f"decode-raised({type(e).__name__})"
    # ── STRONG path (V3.2.2): only before a clip joins the FINAL timeline.
    if strong and want_video:
        if dur_s < min_dur_s:
            return False, f"too-short({dur_s:.2f}<{min_dur_s}s)"
        # full decode — catches truncation / mid-stream NAL corruption that a
        # 3-frame head decode passes (valid clips emit 0 stderr at -v error).
        try:
            full = subprocess.run([ff, "-v", "error", "-i", str(p), "-f",
                                   "null", "-"], capture_output=True, text=True,
                                  timeout=90)
            if full.returncode != 0:
                return False, f"fulldecode-rc({full.returncode})"
            ferr = (full.stderr or "").strip()
            if ferr and len(ferr.splitlines()) >= 2:
                return False, f"fulldecode-stderr({ferr.splitlines()[0][:40]})"
        except Exception as e:                              # noqa: BLE001
            return False, f"fulldecode-raised({type(e).__name__})"
        ok_seek, why_seek = _clip_seek_ok(p, dur_s, ff=ff)
        if not ok_seek:
            return False, why_seek
    return True, "ok"


def _safe_slate(out: Path, nframes: int, grade: str, cs_tail: str) -> bool:
    """Write a frame-exact graded dark slate to `out`. The render's last-
    resort visual — it MUST keep the timeline frame-exact and it MUST NOT
    raise, no matter what. Tries, in order: the graded lavfi slate, a plain
    lavfi slate, a PIL-generated solid frame looped by ffmpeg, and finally
    a bare black lavfi source. Returns True if `out` was written."""
    def _ok() -> bool:
        try:
            return out.exists() and out.stat().st_size > 1_000
        except OSError:
            return False
    attempts = [
        # 1) CLEAN cinematic dark bed — a deep look-dark colour with only a soft
        # vignette + a gentle vertical gradient. NO grain / paper-texture: the
        # old slate applied the full look GRADE, whose grain+tiled texture turned
        # a flat dark frame into the ugly grey "concrete wall" the user flagged.
        # Keep the safety net; lose the concrete (reads as an intentional breath).
        ["-f", "lavfi",
         "-i", f"gradients=s=1920x1080:c0=0x141a26:c1=0x090c12:x0=960:y0=120:"
               f"x1=960:y1=1080:r={FPS}",
         "-vf", f"vignette=angle=PI/4.2,format=yuv420p{cs_tail}",
         "-frames:v", str(nframes), "-an", *_SLATE_VENC, str(out)],
        # 2) plain dark slate (no grade graph — removes any bad-filter risk)
        ["-f", "lavfi", "-i", f"color=c=0x0d1016:s=1920x1080:r={FPS}",
         "-vf", "format=yuv420p", "-frames:v", str(nframes), "-an",
         *_SLATE_VENC, str(out)],
        # 4) bare black (simplest possible graph)
        ["-f", "lavfi", "-i", f"color=c=black:s=1920x1080:r={FPS}",
         "-frames:v", str(nframes), "-an", *_SLATE_VENC, str(out)],
    ]
    for i, args in enumerate(attempts):
        try:
            run(args, timeout=120)
            if _ok():
                return True
        except Exception:                                  # noqa: BLE001
            pass
        # 3) between the lavfi attempts, try a PIL still looped (no lavfi)
        if i == 1:
            try:
                from PIL import Image as _Img
                png = out.with_name(out.stem + "_slate.png")
                _Img.new("RGB", (1920, 1080), (13, 16, 22)).save(png)
                run(["-loop", "1", "-i", str(png), "-vf",
                     f"format=yuv420p{cs_tail}", "-frames:v", str(nframes),
                     "-an", *_SLATE_VENC, str(out)], timeout=120)
                png.unlink(missing_ok=True)
                if _ok():
                    return True
            except Exception:                              # noqa: BLE001
                pass
    return _ok()


def _scene_video(
    item: FootageItem, dur: float, grade: str, out: Path, energy: int = 2,
    kb_mode: int | None = None, kb_seed: int = 0, kb_impact: bool = False,
    kb_hold: bool = False, kb_drift: float = 1.0, archival: bool = False,
) -> bool:
    """Render one scene; return ``True`` only when a fallback slate aired."""
    # Frame count is the single source of truth so the visual length matches
    # the narration exactly (avoids audio being cut by -shortest).
    nframes = max(1, int(round(max(0.2, dur) * FPS)))
    # Stock clips arrive in mixed color spaces (bt601/bt709/unspecified).
    # Without pinning, xfade/concat then the final overlay graph hits a
    # colorspace CHANGE at a scene boundary -> "Error reinitializing
    # filters! Invalid color space" and the whole render dies. Coerce
    # every segment to one canonical space (bt709, tv range) + stamp the
    # metadata so the stream is uniform end to end.
    _CS = (
        ",scale=in_range=tv:out_range=tv,format=yuv420p,"
        "setparams=color_primaries=bt709:color_trc=bt709:"
        "colorspace=bt709:range=tv"
    )
    _CTAG = [
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
    ]
    # FRONT GUARD: a stock clip can arrive with an UNSPECIFIED / reserved
    # colorspace; the FIRST scale then dies with "Invalid color space /
    # Error reinitializing filters" (esp. on h264_videotoolbox) before the
    # tail `_CS` can fix it. Stamp a valid matrix on the decoded frames up
    # front so every filter inits cleanly and uniformly.
    _CSIN = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,"

    _LX = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p"]

    def _enc(pre: list[str], vf: str) -> bool:
        """Encode a segment with graceful degradation so a SINGLE broken
        clip never kills the whole render:
          1) the chosen (hardware) encoder,
          2) libx264 retry (covers videotoolbox -22 flakiness),
          3) a clip-sanitising re-decode (drop bad frames, ignore errors)
             feeding a clean linear transcode, then re-run the filter,
          4) a last-resort GRADED DARK SLATE of the exact length — keeps
             the timeline frame-exact and the render alive."""
        base = pre + ["-vf", vf, "-r", str(FPS), "-frames:v", str(nframes),
                      "-an"]
        # A single segment is only a few seconds of 1080p; 150s is plenty even
        # for the software encoder. The timeout turns a pathological/hung
        # encode into a fast failure that drops to the slate fallback instead
        # of stalling the entire render.
        try:
            run(base + _venc() + _CTAG + [str(out)], timeout=150)
            return False
        except Exception:                                  # noqa: BLE001
            pass
        try:
            run(base + _LX + _CTAG + [str(out)], timeout=150)
            return False
        except Exception:                                  # noqa: BLE001
            pass
        # 3) sanitise the source clip: tolerant decode -> clean intermediate
        if item.is_video and Path(item.path).exists():
            try:
                clean = out.with_name(out.stem + "_clean.mp4")
                run(["-err_detect", "ignore_err", "-fflags", "+genpts",
                     "-i", str(item.path),
                     "-vf", "format=yuv420p,setparams=colorspace=bt709:"
                            "color_primaries=bt709:color_trc=bt709",
                     "-an", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "18", *_CTAG, str(clean)])
                run(["-stream_loop", "-1", "-i", str(clean), "-vf", vf,
                     "-r", str(FPS), "-frames:v", str(nframes), "-an"]
                    + _LX + _CTAG + [str(out)])
                clean.unlink(missing_ok=True)
                return False
            except Exception:                              # noqa: BLE001
                pass
        # 4) guaranteed graded slate (never touches the bad clip). This is
        # the render's safety net and it must NEVER raise — `_safe_slate`
        # owns its own try/except ladder (graded lavfi -> plain lavfi ->
        # PIL still -> bare black). Only if EVERY method fails to write the
        # file do we raise, so the encode-pool layer can record it and
        # retry/fallback this one beat instead of the timeline going short.
        print(f"  [5/5] clip unreadable ({Path(item.path).name}); "
              f"using graded slate for this beat", flush=True)
        if not _safe_slate(out, nframes, grade, _CS):
            raise RuntimeError(
                f"slate write failed for {Path(out).name} "
                f"(src={Path(item.path).name})")
        return True

    # ── READINESS GATE (encode-pool reliability). A partially-written /
    # zero-byte / corrupt clip would otherwise fail all three encode tiers
    # and only THEN hit the slate — three doomed ffmpeg spawns per bad
    # beat, which is exactly the fd/process pressure that crashed the pool.
    # Validate the source FIRST; on a transient miss (a download a few ms
    # from flushing) wait briefly and re-check; if it is genuinely bad, go
    # straight to the slate and skip the doomed tiers. Logs the exact
    # reason + scene index so a bad clip is never silent.
    _want_v = bool(item.is_video)
    _minb = 20_000 if _want_v else 1_000
    # V3.2.2: a clip already quarantined this/last render is slated immediately
    # (never re-opened by any ffmpeg stage).
    try:
        from . import render_quarantine as _q
        _qbad = _q.is_quarantined(str(item.path),
                                  getattr(item, "source_url", "") or "")
    except Exception:                                          # noqa: BLE001
        _q, _qbad = None, False
    if _qbad:
        _ready, _why = False, "quarantined"
    else:
        _ready, _why = _clip_ready(item.path, want_video=_want_v, min_bytes=_minb)
        if not _ready:
            import time as _t
            for _ in range(3):
                _t.sleep(0.3)
                _ready, _why = _clip_ready(item.path, want_video=_want_v,
                                           min_bytes=_minb)
                if _ready:
                    break
        # STRONG gate (V3.2.2): a clip can pass the fast head-decode yet fail on
        # SEEK / full-decode (corrupt stsc/stco, truncation) and crash a later
        # ffmpeg stage. Validate strong before it becomes a timeline segment.
        if _ready and _want_v:
            _sok, _swhy = _clip_ready(item.path, want_video=True,
                                      min_bytes=_minb, strong=True)
            if not _sok:
                _ready, _why = False, f"strong:{_swhy}"
    if not _ready:
        if _q is not None and not _qbad:                       # record once
            try:
                _q.quarantine(str(item.path),
                              source_url=getattr(item, "source_url", "") or "",
                              reason=_why, replacement_source_type="graded_slate",
                              replacement_path=str(out),
                              replacement_reason="clip failed validation → slate",
                              timestamp=os.environ.get("VIDLORE_RUN_TS", ""))
            except Exception:                                  # noqa: BLE001
                pass
        print(f"  [5/5] footage not ready (scene {getattr(item,'index','?')}, "
              f"{Path(item.path).name}, {_why}); graded slate", flush=True)
        if not _safe_slate(out, nframes, grade, _CS):
            raise RuntimeError(f"slate write failed: {Path(out).name} ({_why})")
        return True

    if item.is_video:
        if archival:
            # IMP_011 — preserve archival aspect ratio with textured pillars.
            # Premium docs NEVER stretch/crop 4:3 archival footage to fill
            # 16:9; they letter/pillar-box it against a blurred copy of the
            # frame, which signals authentic period material. This graph is
            # SELF-GATING: the foreground scales with force_original_aspect_
            # ratio=decrease, so a true 16:9 archival clip fills the frame and
            # the blurred bg is fully hidden (no visible change) — only
            # narrower (4:3 / academy) sources actually get the pillars.
            vf = (
                "%ssplit=2[a][b];"
                "[a]scale=1920:1080:force_original_aspect_ratio=decrease,"
                "setsar=1[fg];"
                "[b]scale=1920:1080:force_original_aspect_ratio=increase,"
                "crop=1920:1080,boxblur=24:2,eq=brightness=-0.05[bg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,fps=%d,%s,format=yuv420p%s"
                % (_CSIN, FPS, grade, _CS)
            )
        else:
            vf = (
                "%sscale=1920:1080:force_original_aspect_ratio=increase,"
                "crop=1920:1080,setsar=1,fps=%d,%s,format=yuv420p%s"
                % (_CSIN, FPS, grade, _CS)
            )
        return _enc(["-stream_loop", "-1", "-i", str(item.path)], vf)
    else:
        mode = item.index if kb_mode is None else kb_mode
        z, x, y = _kenburns(mode, max(1, nframes - 1), energy, kb_seed,
                            impact=kb_impact, hold=kb_hold, drift=kb_drift)
        # Ken-Burns supersample: 2560x1440 keeps plenty of zoom headroom
        # for a 1080p output while being ~40% cheaper per frame than the
        # old 4K (3840x2160) — across hundreds of scenes that is the
        # difference between a fast render and an hours-long one.
        vf = (
            "%sscale=2560:1440:force_original_aspect_ratio=increase,"
            "crop=2560:1440,"
            "zoompan=z='%s':x='%s':y='%s':"
            "d=1:s=1920x1080:fps=%d,%s,setsar=1,format=yuv420p%s"
            % (_CSIN, z, x, y, FPS, grade, _CS)
        )
        return _enc(["-loop", "1", "-i", str(item.path)], vf)


def _srt(words: list[WordTiming], path: Path, *, protected_windows=None,
         schedule: list[dict] | None = None) -> list[dict]:
    from .captions import assert_caption_schedule, caption_schedule_problems

    def ts(t: float) -> str:
        total_ms = max(0, int(round(float(t) * 1000.0)))
        h, rem_ms = divmod(total_ms, 3_600_000)
        m, rem_ms = divmod(rem_ms, 60_000)
        s, ms = divmod(rem_ms, 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # SRT and burned ASS must consume the same schedule.  Persist the viewer-facing metrics before
    # serialising anything and fail the render on zero/backwards/overlapping or >20-CPS cues
    # (the official adult timed-text publish ceiling, with spaces/punctuation counted).
    if schedule is None:
        schedule = assert_caption_schedule(
            words, path.with_name("caption_readability_audit.json"),
            protected_windows=protected_windows)
    else:
        flattened = [word for cue in schedule for word in (cue.get("words") or [])]
        if (len(flattened) != len(words)
                or any(a is not b for a, b in zip(flattened, words))):
            raise RuntimeError("approved SRT schedule does not own this word stream")
        problems = caption_schedule_problems(schedule)
        if problems:
            raise RuntimeError("approved SRT schedule is unsafe: " + problems[0]["reason"])
    lines = []
    for i, cue in enumerate(schedule, 1):
        lines += [str(i), f"{ts(cue['start'])} --> {ts(cue['end'])}", cue["text"], ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return schedule


# ═══════════════════════════════════════════════════════════════════════
# PER-SCENE OVERLAY BAKE (v11, 2026-05-27) — eliminates the 80+ stage
# `movie=` filter graph that dominated the final-mux wall clock.
#
# Background. Each graphic card (quote_highlight, number_reveal, timeline,
# document, process_diagram, network_graph, map_route, …) compiles into
# 3-5 `movie='<png>',...` source declarations plus their `[base][src]
# overlay=...:enable='between(t,X,Y)'` statements. With 24 cards in a
# 5-min render that's ~82 stages stacked in ONE ffmpeg call's filter
# graph — each PNG decoder + scale + fade alpha + overlay composite
# layered on top of every other one. Even after splitting V/A into two
# stages, Stage V ran for 2.5+ hours on a 5-min sample (5/27 evening
# debug session) because the single filter graph cannot parallelise the
# sequential overlay chain even with -filter_threads=N.
#
# Fix. Move the heavy PNG-card stages OUT of the final mux into a new
# per-scene bake pass. For each scene:
#   1. Decode the relevant window of video_only.mp4 (trim filter, frame-
#      accurate).
#   2. Apply ONLY that scene's overlays, with timestamps shifted from
#      global → scene-local.
#   3. Re-encode to a small per-scene baked mp4.
# Per-scene bakes run in parallel (one ffmpeg per scene, capped at 4
# concurrent so we don't OOM on AI cards' 4K PNGs). Concat the baked
# mp4s → video_baked.mp4. The final mux then uses video_baked.mp4 and
# its vchain is just vignette+grade+noise+subtitles+keyphrase-stabs+
# actbreak — fast even on a 25-min render.
#
# NOTHING is removed. Every card, every PNG, every drawtext, every
# drawbox is preserved bit-for-bit — they just bake earlier, in
# parallel, instead of stacking in one giant final-mux graph.
#
# Failure path. If the parser cannot extract `enable='between(t,X,Y)'`
# from a stage, that stage assigns to the LAST scene (safe default).
# If the bake fails for any reason, we fall back to the legacy single-
# stage path (gates by env `VIDLORE_SKIP_SCENE_BAKE=1` for debug).
# ═══════════════════════════════════════════════════════════════════════

_TS_WINDOW_RE = re.compile(r"enable='between\(t,([\d.]+),([\d.]+)\)'")
_TS_ST_RE = re.compile(r"\bst=([\d.]+)")
_TS_TREF_RE = re.compile(r"\(t-([\d.]+)\)")
_TS_OFFSET_RE = re.compile(r"\boffset=([\d.]+)")


def _shift_filter_timestamps(stage: str, shift: float) -> str:
    """Re-time every time-bearing parameter inside a filter graph fragment.

    Handles four forms used by `_graphic_card_filters` output:
      • `enable='between(t,X,Y)'`  — overlay visibility window
      • `st=X` (fade start time on `movie=` sources)
      • `(t-X)` (scene-anchored animation easing inside x/y expressions)
      • `offset=X` (xfade transition timing on `movie=`+`movie=` pairs)

    `shift` is added to every X/Y. Use a negative shift to convert global
    timestamps to scene-local (subtract scene start).
    """
    if shift == 0.0:
        return stage
    s = _TS_WINDOW_RE.sub(
        lambda m: (
            f"enable='between(t,"
            f"{float(m.group(1)) + shift:.4f},"
            f"{float(m.group(2)) + shift:.4f})'"
        ),
        stage,
    )
    s = _TS_ST_RE.sub(lambda m: f"st={float(m.group(1)) + shift:.4f}", s)
    s = _TS_TREF_RE.sub(lambda m: f"(t-{float(m.group(1)) + shift:.4f})", s)
    s = _TS_OFFSET_RE.sub(
        lambda m: f"offset={float(m.group(1)) + shift:.4f}", s
    )
    return s


def _group_stages_by_scene(
    g_stages: list[str],
    scene_starts: list[float],
    scene_ends: list[float],
) -> list[list[str]]:
    """Partition `g_stages` (the raw output of `_graphic_card_filters`)
    into per-scene buckets.

    Each "card cluster" is a contiguous run of source decls (`movie=...
    [label]`, no `{CUR}`) followed by exactly one overlay statement
    (`[{CUR}][label]overlay=...:enable='between(t,X,Y)'[{OUT}]`). The
    cluster is assigned to whichever scene window contains the
    overlay's midpoint. This is robust to multi-source cards (a quote
    card with bg + 2 text layers + attribution = 4 source decls + 4
    overlay stmts = 4 clusters that all land in the same scene by
    midpoint).
    """
    n = len(scene_starts)
    buckets: list[list[str]] = [[] for _ in range(n)]
    i = 0
    pending: list[str] = []
    while i < len(g_stages):
        stg = g_stages[i]
        if "{CUR}" not in stg:
            # source decl — pend until its overlay arrives
            pending.append(stg)
            i += 1
            continue
        # overlay stmt — close out the cluster
        cluster = pending + [stg]
        pending = []
        m = _TS_WINDOW_RE.search(stg)
        if m:
            mid = (float(m.group(1)) + float(m.group(2))) / 2.0
        else:
            mid = 0.0
        idx = n - 1
        for s in range(n):
            if scene_starts[s] <= mid < scene_ends[s]:
                idx = s
                break
        buckets[idx].extend(cluster)
        i += 1
    # any trailing pending sources without a matching overlay → last scene
    if pending:
        buckets[-1].extend(pending)
    return buckets


def _bake_per_scene_overlays(
    *,
    video_in: Path,
    workdir: Path,
    g_stages: list[str],
    scene_starts: list[float],
    scene_ends: list[float],
    venc_args: list[str],
    fps: int,
    max_parallel: int = 4,
) -> tuple[Path, dict]:
    """Bake `g_stages` overlays into per-scene segments and concat.

    Returns (`video_baked.mp4` path, timings dict). Timings keys:
      • `n_scenes_with_overlays`
      • `parallel_bake_wall_s`
      • `per_scene_max_s`
      • `concat_wall_s`
      • `total_s`

    On any bake failure the caller can catch the exception and fall back
    to the single-stage path.
    """
    import threading as _th
    import time as _time

    t_total = _time.time()
    n = len(scene_starts)

    buckets = _group_stages_by_scene(g_stages, scene_starts, scene_ends)
    n_with = sum(1 for b in buckets if b)

    baked: list[Path | None] = [None] * n
    errors: list[BaseException | None] = [None] * n
    per_scene_times: list[float] = [0.0] * n

    def _bake_one(s_idx: int) -> None:
        try:
            t0 = _time.time()
            ss = scene_starts[s_idx]
            se = scene_ends[s_idx]
            dur = max(0.04, se - ss)
            stages = buckets[s_idx]
            out = workdir / f"baked_scene_{s_idx:03d}.mp4"

            # Fast keyframe-based input seek (`-ss` BEFORE `-i`) + hard
            # output cap (`-t DUR`). The trim filter from v11.0 was
            # decoding the full video_only.mp4 even though each scene
            # used only its own slice — and h264_videotoolbox hung
            # waiting for the infinite-loop movie sources to EOF. By
            # using input-side seek the decoder reads only the needed
            # GOP, and `-t` guarantees the output stops at the scene
            # boundary regardless of how long the looped overlay
            # sources keep emitting.
            #
            # Encoder: libx264 software, not h264_videotoolbox.
            # videotoolbox does NOT play nicely with movie+loop+overlay
            # chains (it consistently took 50+ minutes of CPU per scene
            # before bailing into its own fallback). libx264 -preset
            # veryfast handles the per-scene graph in 1-3s per scene
            # because each scene's graph is small (3-5 overlay stages,
            # not 80).
            #
            # Threads: 2 per bake. With 4 parallel bakes that's 8
            # cores — matches typical Mac/Linux workstations. Setting
            # `-threads 0` (auto) made every ffmpeg try to grab all
            # cores, multiplying contention by parallelism.
            segs: list[str] = ["[0:v]setpts=PTS-STARTPTS[vt]"]
            cur = "vt"
            if stages:
                local = [_shift_filter_timestamps(stg, -ss) for stg in stages]
                oi = 0
                for stg in local:
                    if "{CUR}" in stg:
                        nxt = f"vov{oi}"
                        segs.append(
                            stg.replace("{CUR}", cur).replace("{OUT}", nxt)
                        )
                        cur, oi = nxt, oi + 1
                    else:
                        segs.append(stg)
            segs.append(f"[{cur}]null[vout]")
            fc = ";".join(segs)

            args = [
                "-ss", f"{ss:.6f}",
                "-i", video_in.name,
                "-t", f"{dur:.6f}",
                "-filter_complex", fc,
                "-map", "[vout]", "-an",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-threads", "2",
                "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv",
                out.name,
            ]
            run(args, cwd=str(workdir))
            baked[s_idx] = out
            per_scene_times[s_idx] = _time.time() - t0
        except BaseException as e:                            # noqa: BLE001
            errors[s_idx] = e

    # Chunked parallel launch — at most max_parallel ffmpegs at once.
    t_parallel = _time.time()
    pending: list[_th.Thread] = []
    next_idx = 0
    while next_idx < n or pending:
        while next_idx < n and len(pending) < max_parallel:
            th = _th.Thread(
                target=_bake_one, args=(next_idx,), name=f"bake-{next_idx}"
            )
            th.start()
            pending.append(th)
            next_idx += 1
        if pending:
            done = pending.pop(0)
            done.join()
    parallel_wall = _time.time() - t_parallel

    for i, e in enumerate(errors):
        if e is not None:
            raise RuntimeError(
                f"per-scene bake failed on scene {i}: {e}"
            ) from e

    # Concat all baked scenes (re-encode to guarantee clean joins; the
    # baked scenes already share the venc args, so this is fast).
    t_concat = _time.time()
    concat_txt = workdir / "_baked_concat.txt"
    concat_txt.write_text(
        "\n".join(f"file '{p.name}'" for p in baked if p) + "\n",
        encoding="utf-8",
    )
    baked_full = workdir / "video_baked.mp4"
    run([
        "-f", "concat", "-safe", "0", "-i", concat_txt.name,
        "-c", "copy",
        baked_full.name,
    ], cwd=str(workdir))
    concat_wall = _time.time() - t_concat

    # v13.1 (2026-05-28): drop the per-scene baked mp4s now that they're
    # concatenated. On a 25-min / 217-scene render these eat 6-10 GB of
    # disk for the rest of the pipeline — already caused one full-disk
    # crash during the audio-sfx synth step. Keep `video_baked.mp4` for
    # the final mux; everything else here is throwaway intermediate.
    for _b in baked:
        try:
            if _b and _b.exists():
                _b.unlink()
        except Exception:                                  # noqa: BLE001
            pass
    try:
        concat_txt.unlink(missing_ok=True)
    except Exception:                                       # noqa: BLE001
        pass

    timings = {
        "n_scenes_with_overlays": n_with,
        "n_scenes_total": n,
        "parallel_bake_wall_s": parallel_wall,
        "per_scene_max_s": max(per_scene_times) if per_scene_times else 0.0,
        "concat_wall_s": concat_wall,
        "total_s": _time.time() - t_total,
    }
    return baked_full, timings


def _detect_black_spans(video_in: Path, *, min_d: float = 0.30,
                        pix_th: float = 0.10) -> list[tuple[float, float]]:
    """Return [(start, end), …] black spans in `video_in` via ffmpeg
    blackdetect. Empty list = clean."""
    import subprocess as _sp
    try:
        proc = _sp.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(video_in),
             "-vf", f"blackdetect=d={min_d}:pix_th={pix_th}",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=900,
        )
    except Exception:                                      # noqa: BLE001
        return []
    spans: list[tuple[float, float]] = []
    for ln in proc.stderr.splitlines():
        if "black_start" not in ln:
            continue
        ms = re.search(r"black_start:([\d.]+)", ln)
        me = re.search(r"black_end:([\d.]+)", ln)
        if ms and me:
            spans.append((float(ms.group(1)), float(me.group(1))))
    return spans


def _detect_dark_spans(video_in: Path, *, min_d: float = 0.40,
                       yavg_floor: float = 26.0,
                       ymax_ceiling: float = 130.0) -> list[tuple[float, float]]:
    """Complement to `_detect_black_spans`: catch SUSTAINED near-black *empty*
    spans that blackdetect's pic_th=0.98 misses, WITHOUT touching legitimately
    dark-but-detailed footage.

    Forensic root-cause (2026-06-01): a failed / empty source clip can render
    at YAVG ≈ 18 (raw Y; black=16) — essentially black, but with a handful of
    non-black pixels so fewer than 98 % of pixels clear the per-pixel floor.
    blackdetect therefore reports only the few frames that ARE 98 %-pure (the
    gap's lead-in), the freeze-fill covers just that slice, the dark TAIL
    survives, and the look grade crushes it to pure black in the final mux
    (the 6.8–8.8 s gap that read 'clean' in metadata but was visibly black).

    Discriminator (measured): a failed clip is near-black AND has NO bright
    pixel anywhere — seg YAVG ~18, YMAX ≤ 82. A genuine low-key documentary
    shot (a dark room with a lit subject) has the SAME low YAVG (~24) but a
    full bright range — YMAX ≈ 250. So we flag a frame only when BOTH the mean
    is near-black (YAVG < yavg_floor) AND the brightest pixel is itself dark
    (YMAX < ymax_ceiling): truly empty. Emit runs of such frames lasting
    >= min_d. One signalstats pass. This never sweeps up real moody footage
    (its YMAX clears the ceiling), and the repair's own neighbour-luma test
    still protects intentional fade-to-black."""
    import subprocess as _sp
    try:
        proc = _sp.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(video_in),
             "-vf", "signalstats,metadata=print", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=900,
        )
    except Exception:                                          # noqa: BLE001
        return []
    times: list[float] = []
    dark: list[bool] = []
    cur_t: float | None = None
    cur_yavg: float | None = None
    cur_ymax: float | None = None

    def _flush() -> None:
        if cur_t is not None and cur_yavg is not None:
            ymax_ok = (cur_ymax is None) or (cur_ymax < ymax_ceiling)
            times.append(cur_t)
            dark.append(cur_yavg < yavg_floor and ymax_ok)

    for ln in proc.stderr.splitlines():
        mt = re.search(r"pts_time:([\d.]+)", ln)
        if mt:
            _flush()
            cur_t = float(mt.group(1))
            cur_yavg = None
            cur_ymax = None
            continue
        ma = re.search(r"signalstats\.YAVG=([\d.]+)", ln)
        if ma:
            cur_yavg = float(ma.group(1))
            continue
        mx = re.search(r"signalstats\.YMAX=([\d.]+)", ln)
        if mx:
            cur_ymax = float(mx.group(1))
    _flush()
    spans: list[tuple[float, float]] = []
    run_s: float | None = None
    run_e: float = 0.0
    for t, is_dark in zip(times, dark):
        if is_dark:
            if run_s is None:
                run_s = t
            run_e = t
        else:
            if run_s is not None and (run_e - run_s) >= min_d:
                spans.append((run_s, run_e))
            run_s = None
    if run_s is not None and (run_e - run_s) >= min_d:
        spans.append((run_s, run_e))
    return spans


def _frame_mean_luma(video_in: Path, t: float) -> float:
    """Mean luma (0–255) of the frame at time `t`, via a tiny PNG extract.
    Returns **-1.0 if the frame could not be read** (e.g. a seek at/past the
    last decodable frame, or a transient ffmpeg error). Callers MUST treat a
    negative result as 'unreadable / NOT a valid anchor' — never as bright —
    so a failed extract can never (a) misclassify a dark gap as bright nor
    (b) anchor a freeze to a timestamp that has no extractable frame (which
    would crash the freeze step and abort the whole repair)."""
    import subprocess as _sp, tempfile as _tf, os as _os
    fd, png = _tf.mkstemp(suffix=".png")
    _os.close(fd)
    try:
        _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{max(0.0, t):.4f}", "-i", str(video_in),
                 "-frames:v", "1", "-vf", "scale=128:72", png],
                check=True, timeout=30)
        from PIL import Image
        with Image.open(png) as im:
            px = list(im.convert("L").getdata())
        return (sum(px) / len(px)) if px else -1.0
    except Exception:                                          # noqa: BLE001
        return -1.0
    finally:
        try:
            _os.unlink(png)
        except Exception:                                      # noqa: BLE001
            pass


def _frame_luma_stats(video_in: Path, t: float):
    """(mean_luma, spatial_std) of the frame at `t`, on a 0–255 scale, via the
    same tiny 128×72 PNG extract `_frame_mean_luma` uses — plus the spatial
    luma standard deviation, so a caller can tell a frame with real content
    (high std) from a near-uniform 'blank' one (std ≈ 0).

    Returns **(-1.0, -1.0) if the frame could not be read** (a seek at/past the
    last decodable frame, or a transient ffmpeg error); callers MUST treat a
    negative mean as 'unreadable / NOT a valid anchor', never as bright."""
    import subprocess as _sp, tempfile as _tf, os as _os
    fd, png = _tf.mkstemp(suffix=".png")
    _os.close(fd)
    try:
        _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{max(0.0, t):.4f}", "-i", str(video_in),
                 "-frames:v", "1", "-vf", "scale=128:72", png],
                check=True, timeout=30)
        from PIL import Image
        import numpy as _np
        with Image.open(png) as im:
            arr = _np.asarray(im.convert("L"), dtype="float32")
        if arr.size == 0:
            return -1.0, -1.0
        return float(arr.mean()), float(arr.std())
    except Exception:                                          # noqa: BLE001
        return -1.0, -1.0
    finally:
        try:
            _os.unlink(png)
        except Exception:                                      # noqa: BLE001
            pass


def _is_blank_bright_frame(mean_luma: float, luma_std: float) -> bool:
    """A frame is 'blank-bright' — a near-white, near-uniform flash carrying no
    real content (a transition white-flash, a blown-out gradient, a blank
    clip) — when it is BOTH very bright AND low-variance. The freeze-hold
    repair must never anchor to such a frame: holding it across a gap paints a
    pale 'blank flash', the symmetric twin of holding a black frame.

    Gate on LOW VARIANCE, not brightness alone, so a legitimately bright frame
    with real detail (snow with a subject, a bright lab, a sky with a figure)
    — which always carries edges / contrast — is NOT rejected. Mirrors the
    footage-acceptance guard footage._is_blank_bright_clip so both layers agree
    on what 'blank' means.

    PHASE-1 (2026-06-05): thresholds loosened (mean 170->158, std 25->28) so a
    slightly-less-blown or faintly-textured white flash is also caught and never
    frozen across a gap — the QA tool independently COUNTS any surviving
    near-white frame in the final MP4 (target = 0)."""
    return mean_luma > 158.0 and luma_std < 28.0


def _find_valid_anchor(video_in: Path, span_start: float, span_end: float,
                       fps: int, total: float, *, lum_floor: float = 28.0,
                       window: float = 1.2):
    """Pick a NON-dark, ACTUALLY-READABLE frame timestamp to freeze across a
    black gap.

    Search BACKWARD from just before the span (bounded `window` seconds),
    then FORWARD past the span end. This is the fix for the deleted-scene
    reflow edge case where the frame immediately before a black span is
    itself dark (a fade/transition boundary), so the old 'freeze the frame
    at s-1/fps' held black.

    A candidate qualifies ONLY if it is a real, readable frame with luma
    >= `lum_floor` that is NOT near-white-and-uniform (`_is_blank_bright_frame`
    — the symmetric twin of the dark floor, so a transition white-flash or a
    blown-out blank frame is skipped, never frozen across the gap, the way a
    too-dark frame already is); a failed/empty read (negative) is never treated
    as valid, so the freeze that follows is GUARANTEED an extractable frame. The
    backward start equals the caller's `before_lum` probe timestamp, so an
    'unintended_empty_gap' (bright on both sides) always resolves to that
    bright, in-bounds frame. Returns (anchor_t, direction, mean_luma);
    `mean_luma < lum_floor` (incl. the all-unreadable fallback, reported as
    luma 0.0) signals the caller to PRESERVE the span (intentional fade /
    genuinely dark passage) rather than freeze a phantom or dark frame."""
    step = max(2.0 / fps, 0.06)                    # ≈ every other frame, ≥60 ms
    safe_back = max(0.0, span_start - 1.0 / fps)
    best = (safe_back, 0.0)            # readable-but-dark fallback -> preserve
    # backward — prefer the OUTGOING scene's last good frame
    t = safe_back
    floor = max(0.0, span_start - window)
    while t >= floor:
        lum, std = _frame_luma_stats(video_in, t)
        if lum >= lum_floor and not _is_blank_bright_frame(lum, std):
            return (t, "backward", lum)
        if lum > best[1] and not _is_blank_bright_frame(lum, std):
            best = (t, lum)                        # brightest *readable* frame
        t -= step
    # forward — the incoming scene's first good frame; stay safely IN-BOUNDS
    # (a seek at/after the last frame returns -1 and must never be chosen).
    safe_ceil = max(0.0, total - 1.5 / fps)
    t = min(safe_ceil, span_end + 1.0 / fps)
    ceil = min(safe_ceil, span_end + window)
    while t <= ceil:
        lum, std = _frame_luma_stats(video_in, t)
        if lum >= lum_floor and not _is_blank_bright_frame(lum, std):
            return (t, "forward", lum)
        if lum > best[1] and not _is_blank_bright_frame(lum, std):
            best = (t, lum)
        t += step
    return (best[0], "fallback_brightest", best[1])


def _probe_duration_s(path: Path) -> float:
    """Probe a media file's duration (seconds) using the SAME ffmpeg
    `Duration:` stderr regex the black-frame writer relies on. Returns 0.0
    on any failure — never raises."""
    try:
        import subprocess as _sp
        info = _sp.run([ffmpeg_exe(), "-i", str(path)],
                       capture_output=True, text=True)
        dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info.stderr)
        if dm:
            return (int(dm.group(1)) * 3600 + int(dm.group(2)) * 60
                    + float(dm.group(3)))
    except Exception:                                          # noqa: BLE001
        pass
    return 0.0


def _write_export_metrics(final_mp4: Path, fps: int) -> dict:
    """Write a `render_export_metrics.json` sidecar next to the FINAL
    delivered MP4 (post-mux), recording its filename, absolute path,
    streamed sha256, probed duration and fps. This is the authoritative
    final-output fingerprint — distinct from the PRE-mux
    render_black_frame_metrics.json (which describes the shorter
    intermediate the repair pass scanned). Returns the metrics dict (also
    useful for tests/logging). Best-effort: never raises — a hash/probe
    failure degrades the affected field to None rather than breaking the
    render."""
    metrics: dict = {
        "schema": "render_export_metrics/1",
        "final_video": final_mp4.name,
        "final_path": str(final_mp4.resolve()),
        "sha256": None,
        "duration_s": None,
        "fps": int(fps),
    }
    try:
        import hashlib as _hashlib
        h = _hashlib.sha256()
        with open(final_mp4, "rb") as _fh:
            for _chunk in iter(lambda: _fh.read(1024 * 1024), b""):
                h.update(_chunk)
        metrics["sha256"] = h.hexdigest()
    except Exception:                                          # noqa: BLE001
        pass
    try:
        _d = _probe_duration_s(final_mp4)
        metrics["duration_s"] = round(_d, 3) if _d > 0 else None
    except Exception:                                          # noqa: BLE001
        pass
    try:
        import json as _json
        (final_mp4.parent / "render_export_metrics.json").write_text(
            _json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass
    return metrics


def _repair_black_frames(video_in: Path, workdir: Path, fps: int,
                         breakout_windows: list | None = None,
                         lineage_windows: list | None = None) -> Path:
    """Iterate-until-clean driver around `_repair_black_frames_once`.

    `breakout_windows` = [(start_s, end_s), ...] real-audio breakout windows. A black span
    overlapping one of these is NEVER a legitimate 'intentional fade' (a breakout is validated
    real footage that must be on screen while its audio plays) — the once-pass repairs it and
    flags it `breakout_window_black` so the failure is visible, never silently preserved.

    User requirement (2026-06-01): "re-run post-render black detection AFTER
    the repair — never trust repair metadata alone — and iterate until clean
    so accidental pure-black gaps = 0." A single freeze-fill pass can leave a
    residual accidental gap (e.g. two gaps that merge across the join, or a
    span the first anchor only partially covered). So we run the freeze-fill,
    then RE-DETECT on the actual output; if any ACCIDENTAL (non-fade) black
    survives we repair the repaired video again, up to 3 passes. Intentional
    fades are classified `preserved_intentional_fade` inside the once-pass and
    are EXCLUDED from the re-pass trigger (they are supposed to stay dark), so
    this converges instead of looping forever on a real fade-to-black.

    Re-reading the same `video_repaired.mp4` path across passes is safe: every
    read of the input (detection + segment slicing + freeze-PNG grab) completes
    BEFORE the concat re-encode writes the output, so the overwrite never races
    its own source.
    """
    cur = video_in
    sidecar = workdir.parent / "render_black_frame_metrics.json"
    for _pass in range(3):
        out = _repair_black_frames_once(
            cur, workdir, fps, breakout_windows=breakout_windows,
            lineage_windows=lineage_windows)
        if out is cur:
            break                       # nothing detected this pass → clean
        cur = out
        unresolved = 0
        try:
            import json as _j
            meta = _j.loads(sidecar.read_text(encoding="utf-8"))
            unresolved = int(meta.get("unresolved_repair_count", 0))
        except Exception:               # noqa: BLE001
            unresolved = 0
        if unresolved <= 0:
            break                       # only intentional fades remain → done
        print(f"  [5/5] black-frame repair: pass {_pass + 1} left "
              f"{unresolved} accidental black span(s) → re-pass", flush=True)
    return cur


def _assert_lineage_repair_owner(span_start: float, span_end: float, anchor_t: float,
                                 lineage_windows: list, fps: int) -> None:
    """Block a freeze whose donor is outside the black span's own aired beat."""
    eps = 1.0 / max(1, int(fps)) + 1e-4
    owners = []
    for rec in lineage_windows or []:
        try:
            a, b = float(rec[0]), float(rec[1])
        except (TypeError, ValueError, IndexError):
            continue
        if span_start >= a - eps and span_end <= b + eps:
            owners.append((a, b, rec[2] if len(rec) > 2 else None))
    if len(owners) != 1:
        raise SceneLineageError(
            f"black-frame repair span {span_start:.3f}-{span_end:.3f}s crosses or lacks an "
            f"owned beat; a neighbour freeze is forbidden")
    a, b, owner = owners[0]
    if anchor_t < a - eps or anchor_t >= b + eps:
        raise SceneLineageError(
            f"black-frame repair for beat {owner!r} would use neighbour donor "
            f"{anchor_t:.3f}s outside {a:.3f}-{b:.3f}s")


def _repair_black_frames_once(video_in: Path, workdir: Path, fps: int,
                              breakout_windows: list | None = None,
                              lineage_windows: list | None = None) -> Path:
    """Eliminate black/blank spans from a video-only stream by FREEZE-
    HOLDING the nearest VALID (non-dark) frame across each gap (total
    duration + frame count preserved, so downstream audio stays in sync).

    v13.3 (2026-05-30): the freeze frame is now chosen by `_find_valid_anchor`
    (backward-then-forward search for a non-dark frame) instead of blindly
    grabbing s-1/fps — which was itself dark at deleted-scene reflow / fade
    boundaries, so the black span survived the 'repair'. Also writes a
    `render_black_frame_metrics.json` sidecar to the run dir.

    User requirement (2026-05-28): "never output black/blank frames".
    The footage tier ladder already falls through to a themed slide, but
    a corrupt/short clip or a degenerate bake slice can still leave a
    black scene (the 25-min Mossad render had one 6.7s black span at
    12:03). This is the last-line guarantee: scan the assembled video,
    and for every black span splice in a freeze of the frame immediately
    before it. Returns the repaired path (or the original if clean).
    """
    # Pure-black (blackdetect) UNION sustained-dark (mean-luma) spans — the
    # latter catches a failed/empty clip that is near-black but not 98 %-pure,
    # whose dark tail otherwise survives the freeze-fill and grades to black.
    spans = (_detect_black_spans(video_in, min_d=0.30, pix_th=0.10)
             + _detect_dark_spans(video_in))
    if not spans:
        return video_in

    print(f"  [5/5] black-frame repair: {len(spans)} span(s) detected — "
          f"freeze-holding previous frame", flush=True)

    # Probe total duration so the final tail segment is exact.
    import subprocess as _sp
    try:
        info = _sp.run([ffmpeg_exe(), "-i", str(video_in)],
                       capture_output=True, text=True)
        dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info.stderr)
        total = (int(dm.group(1)) * 3600 + int(dm.group(2)) * 60
                 + float(dm.group(3))) if dm else 0.0
    except Exception:                                      # noqa: BLE001
        total = 0.0
    if total <= 0:
        return video_in

    # Merge overlapping/adjacent spans, clamp into the timeline.
    spans = sorted(spans)
    merged: list[list[float]] = []
    for s, e in spans:
        s = max(0.0, s)
        e = min(total, e)
        if e - s < 0.10:
            continue
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if not merged:
        return video_in

    venc = _venc("20")
    _CTAG = ["-colorspace", "bt709", "-color_primaries", "bt709",
             "-color_trc", "bt709", "-color_range", "tv"]
    parts: list[Path] = []
    cursor = 0.0
    span_metrics: list[dict] = []
    repaired_count = 0
    try:
        for i, (s, e) in enumerate(merged):
            gap = e - s
            # classify the span from its immediate neighbours
            before_lum = _frame_mean_luma(video_in, max(0.0, s - 1.0 / fps))
            after_lum = _frame_mean_luma(
                video_in, min(max(0.0, total - 1.5 / fps), e + 1.0 / fps))
            anchor_t, direction, anchor_lum = _find_valid_anchor(
                video_in, s, e, fps, total)
            # a black span inside a real-audio breakout window is NEVER an intentional fade:
            # a breakout is validated real footage that must be visible while its audio plays.
            # Force a repairable classification so it is freeze-filled, not preserved-as-fade.
            _in_breakout = any(bs < e - 0.05 and be > s + 0.05
                               for (bs, be) in (breakout_windows or []))
            if before_lum >= 28.0 and after_lum >= 28.0:
                klass = "unintended_empty_gap"      # bright→black→bright
            elif anchor_lum >= 28.0:
                klass = "dark_boundary_repairable"   # fade/dark edge, valid frame nearby
            elif _in_breakout:
                klass = "breakout_window_black"      # black over breakout audio — must repair
            else:
                klass = "dark_cinematic_or_fade"     # genuinely dark surroundings

            # PRESERVE intentional fades / genuinely-dark cinematic spans.
            # If NO valid (non-dark) frame exists anywhere in the bounded
            # search window on EITHER side, the whole region is dark by
            # design — a fade-to-black hold or a dark cinematic shot. Forcing
            # a bright freeze here would flash and destroy the fade, so we
            # leave the original footage untouched (it stays inside the next
            # keep/tail segment because `cursor` is NOT advanced past it).
            # This honours BOTH "never output unintended blank frames" AND
            # "don't break intentional cinematic fades". An unintended footage
            # gap is, by contrast, bordered by real (bright) content, so it
            # classifies A/B and IS repaired below.
            if klass == "dark_cinematic_or_fade":
                span_metrics.append({
                    "start_s": round(s, 3), "end_s": round(e, 3),
                    "duration_s": round(gap, 3),
                    "before_luma": round(before_lum, 1),
                    "after_luma": round(after_lum, 1),
                    "classification": klass,
                    "repair_method": "preserved_intentional_fade",
                    "anchor_timestamp_s": round(anchor_t, 3),
                    "anchor_direction": direction,
                    "anchor_luma": round(anchor_lum, 1),
                    "anchor_is_dark": True,
                })
                continue

            if lineage_windows is not None:
                _assert_lineage_repair_owner(s, e, anchor_t, lineage_windows, fps)

            # REPAIR (unintended_empty_gap | dark_boundary_repairable):
            # 1) good footage [cursor .. s]
            if s - cursor > 0.04:
                seg = workdir / f"_blk_keep_{i:03d}.mp4"
                _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel",
                         "error", "-ss", f"{cursor:.4f}", "-to", f"{s:.4f}",
                         "-i", str(video_in), "-an", "-r", str(fps),
                         *venc, *_CTAG, str(seg)], check=True)
                parts.append(seg)
            # 2) freeze the nearest VALID (non-dark) frame across the gap
            frz_png = workdir / f"_blk_frame_{i:03d}.png"
            _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel",
                     "error", "-ss", f"{anchor_t:.4f}", "-i", str(video_in),
                     "-frames:v", "1", str(frz_png)], check=True)
            frz_mp4 = workdir / f"_blk_freeze_{i:03d}.mp4"
            _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel",
                     "error", "-loop", "1", "-i", str(frz_png),
                     "-t", f"{gap:.4f}", "-r", str(fps),
                     "-vf", "format=yuv420p,scale=1920:1080",
                     *venc, *_CTAG, str(frz_mp4)], check=True)
            parts.append(frz_mp4)
            span_metrics.append({
                "start_s": round(s, 3), "end_s": round(e, 3),
                "duration_s": round(gap, 3),
                "before_luma": round(before_lum, 1),
                "after_luma": round(after_lum, 1),
                "classification": klass,
                "repair_method": "freeze_hold",
                "anchor_timestamp_s": round(anchor_t, 3),
                "anchor_direction": direction,
                "anchor_luma": round(anchor_lum, 1),
                "anchor_is_dark": bool(anchor_lum < 28.0),
            })
            cursor = e
            repaired_count += 1

        # All detected spans were intentional fades / dark cinematic holds —
        # nothing to splice; preserve the original video byte-for-byte.
        if repaired_count == 0:
            print(f"  [5/5] black-frame repair: all {len(merged)} span(s) are "
                  "intentional fades / dark cinematic — preserved, no freeze "
                  "applied", flush=True)
            for m in span_metrics:
                m["resolved"] = True          # intentionally preserved
                m["still_black"] = True        # expected (it is a real fade)
                m["confidence"] = 0.8
            try:
                import json as _json
                (workdir.parent
                 / "render_black_frame_metrics.json").write_text(
                    _json.dumps({
                        # SCOPE: this sidecar describes the PRE-MUX intermediate
                        # video that black-frame repair scanned — NOT the final
                        # delivered MP4 (which is longer once audio is muxed and
                        # is fingerprinted separately in render_export_metrics.json).
                        "scope": "intermediate_pre_mux",
                        "scanned_file": video_in.name,
                        "video": video_in.name, "fps": fps,
                        "total_s": round(total, 3),
                        "detector":
                            "blackdetect d=0.30 pix_th=0.10 (pic_th=0.98)",
                        "luma_floor": 28.0,
                        "before_scan_span_count": len(merged),
                        "after_scan_span_count": len(merged),
                        "unresolved_repair_count": 0,
                        "preserved_count": len(span_metrics),
                        "result": "preserved",
                        "spans": span_metrics,
                    }, indent=2), encoding="utf-8")
            except Exception:                                  # noqa: BLE001
                pass
            return video_in
        # 3) tail [cursor .. end]
        if total - cursor > 0.04:
            seg = workdir / "_blk_keep_tail.mp4"
            _sp.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel",
                     "error", "-ss", f"{cursor:.4f}", "-i", str(video_in),
                     "-an", "-r", str(fps), *venc, *_CTAG, str(seg)],
                    check=True)
            parts.append(seg)

        concat_txt = workdir / "_blk_concat.txt"
        concat_txt.write_text(
            "\n".join(f"file '{p.name}'" for p in parts) + "\n",
            encoding="utf-8")
        repaired = workdir / "video_repaired.mp4"
        # Forensic v2 — RE-ENCODE the concat (NOT -c copy). The freeze/keep/tail
        # parts are each encoded separately; stream-COPYING them through the
        # concat demuxer left GOP / timebase boundary glitches where the freeze
        # frame failed to decode — so the "repaired" gap stayed BLACK in the
        # final MP4 even though post-detect reported still_black=False (it probes
        # the span start, not the glitchy join). A single clean CFR re-encode
        # makes every segment (incl. the freeze) land frame-exact + continuous,
        # which is what actually removes the visible black gap.
        run(["-f", "concat", "-safe", "0", "-i", concat_txt.name,
             "-an", "-r", str(fps), "-vsync", "cfr", *venc, *_CTAG,
             repaired.name], cwd=str(workdir))

        # Verify each REPAIRED span actually cleared. Preserved intentional
        # fades are expected to remain dark — that is correct, not a failure,
        # so they never count against the result.
        leftover = (_detect_black_spans(repaired, min_d=0.30, pix_th=0.10)
                    + _detect_dark_spans(repaired))
        unresolved_repairs = 0
        for m in span_metrics:
            still = any(abs(ls - m["start_s"]) < 0.5 for ls, _ in leftover)
            m["still_black"] = bool(still)
            if m["repair_method"] == "preserved_intentional_fade":
                m["resolved"] = True            # intentionally kept
                m["confidence"] = 0.8
            else:
                m["resolved"] = (not still) and (not m["anchor_is_dark"])
                m["confidence"] = 0.95 if m["resolved"] else (
                    0.4 if m["anchor_is_dark"] else 0.7)
                if not m["resolved"]:
                    unresolved_repairs += 1
        preserved_n = sum(1 for m in span_metrics
                          if m["repair_method"] == "preserved_intentional_fade")
        if unresolved_repairs:
            print(f"  [5/5] black-frame repair: {unresolved_repairs} repaired "
                  "span(s) still dark after freeze (using repaired anyway)",
                  flush=True)
        else:
            print(f"  [5/5] black-frame repair: clean — {repaired_count} "
                  f"gap(s) freeze-filled, {preserved_n} intentional fade(s) "
                  "preserved", flush=True)
        # metrics sidecar in the run dir (workdir is run_dir/work_PID)
        try:
            import json as _json
            (workdir.parent / "render_black_frame_metrics.json").write_text(
                _json.dumps({
                    # SCOPE: this sidecar describes the PRE-MUX intermediate
                    # video that black-frame repair scanned — NOT the final
                    # delivered MP4 (which is longer once audio is muxed and
                    # is fingerprinted separately in render_export_metrics.json).
                    "scope": "intermediate_pre_mux",
                    "scanned_file": video_in.name,
                    "video": video_in.name, "fps": fps,
                    "total_s": round(total, 3),
                    "detector": "blackdetect d=0.30 pix_th=0.10 (pic_th=0.98)",
                    "luma_floor": 28.0,
                    "before_scan_span_count": len(merged),
                    "after_scan_span_count": len(leftover),
                    "unresolved_repair_count": unresolved_repairs,
                    "preserved_count": preserved_n,
                    "result": "clean" if unresolved_repairs == 0 else "partial",
                    "spans": span_metrics,
                }, indent=2), encoding="utf-8")
        except Exception:                                      # noqa: BLE001
            pass
        # cleanup intermediates
        for p in parts:
            p.unlink(missing_ok=True)
        concat_txt.unlink(missing_ok=True)
        for i in range(len(merged)):
            (workdir / f"_blk_frame_{i:03d}.png").unlink(missing_ok=True)
        return repaired
    except Exception as _e:                                # noqa: BLE001
        print(f"  [5/5] black-frame repair FAILED ({str(_e)[:80]}); "
              "using unrepaired video", flush=True)
        return video_in


# Graphic kinds whose reveal is synced to the spoken-word time (VO-timed):
#   document/news_article — highlighter marker draws as the line is spoken;
#   maps — territory fill/label lands when the narrator names the place.
# news_article was previously MISSING here, so MNT_5's animated highlighter
# silently fell back to a fixed offset instead of the documented VO sync.
_VO_REVEAL_KINDS = {"document", "news_article"}


def _reveal_time(emphasis_phrase: str, words: list | None) -> float:
    """Spoken start-time (s) of the first narration word matching the scene's
    emphasis phrase, else -1.0 (caller uses a safe fixed offset). Pure — no
    API/render. Mirrors the inline logic document/maps already relied on."""
    et = re.findall(r"[\w']+", (emphasis_phrase or "").lower(), re.UNICODE)
    et = [re.sub(r"[^\w]", "", x, flags=re.UNICODE) for x in et if x]
    if not et:
        return -1.0
    for wt in (words or []):
        w = re.sub(r"[^\w]", "", getattr(wt, "word", "").lower(),
                   flags=re.UNICODE)
        if w and w in et:
            return float(wt.start)
    return -1.0


def assemble(
    footage: list[FootageItem],
    narration: Narration,
    theme: dict,
    workdir: Path,
    out_path: Path,
    *,
    captions: bool = True,
    music: str | None = None,
    transitions: bool = True,
    title: str | None = None,
    overlays: bool = True,
    chapters: list[str] | None = None,
    energies: list[int] | None = None,
    emphasis: list[str] | None = None,
    graphics: list[tuple[str, str, str]] | None = None,
    graphic_assets: dict | None = None,
    shot_types: list[str] | None = None,
    roles: list[str] | None = None,
    beat_clips: dict | None = None,
    sfx: bool = False,
    style: object | None = None,
    motion_graphics: dict | None = None,
    motion_graphics_windows: dict | None = None,
    motion_graphics_primitives: dict | None = None,
    caption_suppress_windows: list | None = None,
    breakout_windows: list | None = None,
    scene_lineage: object | None = None,
    lineage_expectations: object | None = None,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    # Caption feasibility depends only on the final timed narration and trusted
    # breakout windows.  Fail here, before hundreds of scene encodes, and retain
    # this exact approved schedule for both SRT and burned ASS.
    words = narration.all_words()
    _approved_caption_schedule = _srt(
        words, out_path.with_suffix(".srt"),
        protected_windows=breakout_windows)
    # ASSEMBLY LINEAGE CONTRACT.  Generic engine callers remain unchanged
    # (None = disabled), but a caller that supplies provenance gets a strict,
    # fail-closed contract.  ``lineage_expectations`` is a compatibility alias
    # for integrations that use the more explicit name; accepting both at once
    # would make ownership ambiguous and is therefore rejected.
    if scene_lineage is not None and lineage_expectations is not None:
        raise SceneLineageError(
            "pass either scene_lineage or lineage_expectations, not both")
    _lineage_contract = (scene_lineage if scene_lineage is not None
                         else lineage_expectations)
    _lineage_enabled = _lineage_contract is not None
    _lineage_audit_path = Path(out_path).parent / "scene_lineage_audit.json"
    _lineage_audit = _new_scene_lineage_audit(Path(out_path)) if _lineage_enabled else None
    _lineage_encoded_banks: dict = {}
    if _lineage_enabled:
        # Persistence is itself part of the invariant: a render may not claim
        # lineage protection when its evidence sidecar cannot be written.
        _write_scene_lineage_audit(_lineage_audit_path, _lineage_audit)
    # STYLE MODE (cinematic personality) — biases pacing, transitions,
    # camera restraint and atmosphere. Defaults to the neutral baseline.
    if style is None:
        from .style_modes import STANDARD as style
    _sty = style
    # Theme grade + theme's overlay-effect chain (film_grain / vignette /
    # paper_grain / edge_glow). Effects are baked into every regular
    # footage clip's encode so the chosen theme delivers a distinct
    # MOOD, not just colour. Vintage/archival clips still skip this
    # (the _VINTAGE branch below) because they already have their look.
    from .themes import effects_filters as _theme_effects
    grade = theme["grade"] + _theme_effects(theme)
    by_idx = {f.index: f for f in footage}
    scenes = narration.scenes
    durs = [ns.duration for ns in scenes]
    energies = energies or []
    # MOTION GRAPHICS (VIDLORE_MOTION_GRAPHICS): a {scene_pos: clip_path} map of
    # premium director-selected graphics that REPLACE a scene's footage visual.
    # When a scene has one we suppress its flat card (the graphic is the visual)
    # — the clip is sliced across the scene's beats in the encode pre-pass so the
    # beat / transition / concat machinery is untouched. Empty map = legacy.
    _mg = motion_graphics or {}
    # V1.1 — per-scene USEFUL window (seconds). The MG owns the scene only for
    # this long; later beats of the scene return to footage. None/absent = the
    # MG fills the whole scene (legacy behaviour).
    _mgw = motion_graphics_windows or {}
    if _mg and graphics:
        graphics = [("", "", "") if i in _mg else g
                    for i, g in enumerate(graphics)]

    shot_types = shot_types or []

    def _en(i: int) -> int:
        return energies[i] if 0 <= i < len(energies) else 2

    def _kb(i: int) -> list | None:
        st = shot_types[i] if 0 <= i < len(shot_types) else ""
        return _SHOT_KB.get(st)  # list palette, or None -> fallback

    # Vintage treatment for scenes the editor-LLM deliberately marked
    # "archival" in its storyboard — a tasteful faded-sepia film look +
    # grain, applied only there (natural, not a flashy global filter).
    def _is_vintage(i: int) -> bool:
        return 0 <= i < len(shot_types) and shot_types[i] == "archival"

    # Per-scene chapter strip timing on the absolute timeline. Scene 0 is
    # skipped (the big title card already owns the intro); only scenes
    # long enough to show a 2.6s strip get one.
    chapter_cues: list[tuple[float, str]] = []
    # (start, dur, kind, text, body)
    graphic_cues: list[tuple] = []
    # IMP_023 — floating-stat restraint state. These pills bypass script_gen's
    # graphic density cap (footage-only scenes carry no graphic_kind), so the
    # cap lives HERE: at most 2 per video, >=4 scenes apart, never adjacent to
    # a real card — a stat pill is a rare spice, not a staple.
    _FSTAT_MAX, _FSTAT_GAP = 2, 4
    _fstat_placed, _fstat_last = 0, -99
    boundaries: list[float] = []
    arch_windows: list[tuple[float, float]] = []   # archival scene spans
    graphics = graphics or []
    gassets = graphic_assets or {}
    t = 0.0
    for i, d in enumerate(durs):
        if 0 <= i < len(shot_types) and shot_types[i] == "archival":
            arch_windows.append((t, t + d))
        if i > 0:
            boundaries.append(t)  # scene-cut timestamp (for transition SFX)
            if chapters and d >= 3.4 and i < len(chapters) and chapters[i]:
                chapter_cues.append((t, chapters[i]))
        if i < len(graphics) and d >= 2.4:
            gk, gt, gb = graphics[i]
            if gk and gt:
                sidx = scenes[i].index if i < len(scenes) else i
                # When this scene's emphasis word is actually spoken
                # (real Whisper timing) — the document highlight reveal
                # is synced to THIS instant. -1 => use a safe default.
                etime = -1.0
                # IMP_019 — also compute it for MAP scenes so a region's
                # highlight/fill can land on the instant the narrator NAMES
                # it (VO-timed territory fill, like the document highlight).
                # MNT_9 — news_article added so MNT_5's animated highlighter
                # actually syncs to the spoken word (was silently fixed-offset).
                if gk in _VO_REVEAL_KINDS or gk in _MAP_KINDS:
                    _el = emphasis or []
                    etime = _reveal_time(
                        _el[i] if i < len(_el) else "",
                        getattr(scenes[i], "words", None))
                # IMP_005 — carry this scene's shot_type so the callout
                # renderer can keep its label out of the face zone on
                # portrait / reaction shots (8th tuple slot, optional).
                _cue_st = (shot_types[i] if 0 <= i < len(shot_types) else "")
                # IMP_016 — for a MAP scene, carry a prominent COMMA-GROUPED
                # figure from the narration (e.g. "130,000") so the assembler
                # can tick it ON the map as a dynamic counter instead of a
                # standalone stat card. Comma-grouping is a high-precision
                # signal for a real quantity (vs a bare year like 1949).
                _cue_fig = ""
                if gk in _MAP_KINDS:
                    _narr_i = " ".join(
                        getattr(w, "word", "") for w in
                        (getattr(scenes[i], "words", None) or []))
                    _figs = re.findall(r"\d{1,3}(?:,\d{3})+", _narr_i)
                    if _figs:
                        _cue_fig = max(_figs,
                                       key=lambda s: int(s.replace(",", "")))
                graphic_cues.append((t, d, gk, gt, gb,
                                     gassets.get(sidx, ""), etime,
                                     _cue_st, _cue_fig))
            elif (not gk and d >= 3.2 and i >= 2
                  and _fstat_placed < _FSTAT_MAX
                  and (i - _fstat_last) >= _FSTAT_GAP):
                # IMP_023 — footage-only scene with a notable comma-grouped
                # figure and NO card: float it as a lower-third stat pill, but
                # only if the neighbours are also card-free (never card+pill
                # back-to-back) and the restraint caps allow it.
                _pg = (graphics[i - 1][0] if 0 <= i - 1 < len(graphics)
                       else "")
                _ng = (graphics[i + 1][0] if i + 1 < len(graphics) else "")
                if not _pg and not _ng:
                    _narr_i = " ".join(
                        getattr(w, "word", "") for w in
                        (getattr(scenes[i], "words", None) or []))
                    # IMP_025 — accept comma-grouped digits OR a spelled-out
                    # magnitude ("four hundred and twenty million"), since real
                    # narration spells numbers out. Magnitude-gated so years /
                    # ordinals / bare counts never trip it.
                    # MONEY COUNT-UP: a money beat counts up as money
                    # ($0B->$420B); everything else stays a bare figure.
                    _fig = _money_figure(_narr_i) or _best_stat_figure(_narr_i)
                    if _fig:
                        _cue_st = (shot_types[i] if 0 <= i < len(shot_types)
                                   else "")
                        graphic_cues.append((t, d, "floating_stat", _fig, "",
                                             "", -1.0, _cue_st, _fig))
                        _fstat_placed += 1
                        _fstat_last = i
        t += d

    # Compute graphic-card filters EARLY so the number-reveal sound can
    # be synced to the exact roll/landing times the cards use.
    g_comma: list[str] = []
    g_stages: list[str] = []
    g_post: list[str] = []
    num_events: list[tuple[float, float]] = []
    type_events: list = []                 # typewriter char-time schedules
    if overlays:
        # Script-aware font pick — sample narration so CJK / RTL render
        # in their bundled Noto face instead of tofu via the default
        # Latin VidloreSans.  NarratedScene has no `.narration` field —
        # the text lives in its WordTiming list, so reconstruct from
        # there.  (Also pull from `graphic_cues` titles/bodies as a
        # second source so cards in a different script still trigger
        # the bundled Noto pick.)
        try:
            _sample = " ".join(
                " ".join(getattr(w, "word", "") for w in
                         (getattr(s, "words", []) or []))[:200]
                for s in narration.scenes[:3])
            if not _sample.strip() and graphic_cues:
                _sample = " ".join(
                    " ".join(str(x) for x in (g[1:] if g else []) if x)[:200]
                    for g in graphic_cues[:3])
        except Exception:                                  # noqa: BLE001
            _sample = ""
        g_comma, g_stages, g_post, num_events = _graphic_card_filters(
            graphic_cues, _copy_font(workdir, theme, sample_text=_sample),
            theme.get("accent", (255, 210, 90)), workdir,
            type_events=type_events,
        )

    # TEXT-MOMENT SFX events — the cinematic support the viewer FEELS on a
    # graphic reveal: a soft directional WHOOSH as each card flies in, plus
    # a low IMPACT hit when a shocking name / warning / claim LANDS. Numbers
    # & stats are excluded from the impact (they already get the dedicated
    # count-up stinger) but still get the reveal whoosh.
    # PROFESSIONAL SFX MATCHING (sfx.py library). Each template emits a
    # SEQUENCE of category-matched micro-sounds timed to its real animation
    # (pin drop, pulse, route draw, stamp hit, highlighter swipe, timeline
    # ticks, process step pops, node connects, chart reveal, …) — not one
    # generic whoosh per card. The library's matcher rotates variants +
    # jitters pitch/volume/timing so nothing is ever spammed.
    def _gfx_sfx_seq(gk: str) -> list:
        """[(offset_from_reveal, event_kind, intensity), ...] per template."""
        if gk == "map_reveal":
            return [(0.0, "reveal", .55), (1.15, "map_pin", .7),
                    (1.5, "location_lock", .45), (1.7, "map_pulse", .5)]
        if gk == "map_route":
            return [(0.0, "transition", .55), (0.45, "map_pin", .6),
                    (0.7, "map_route", .5), (2.6, "map_pin", .6)]
        if gk == "map_region":
            return [(0.0, "reveal", .5), (0.95, "map_region", .6),
                    (1.6, "map_pulse", .45)]
        if gk == "text_on_black":
            # terminal / cipher aesthetic — the text TYPES on, it NEVER
            # whooshes. A short sparse run of keyboard clicks (soft texture,
            # not in the throttled whoosh family), then silence to read.
            return [(0.05, "keyboard", .32), (0.45, "keyboard", .30),
                    (0.95, "keyboard", .32), (1.6, "keyboard", .28)]
        if gk in ("classified", "case_file", "redacted", "conspiracy_board"):
            return [(0.0, "doc_slide", .5), (0.55, "stamp", .8)]
        if gk in ("verdict_stamp", "postmark"):
            return [(0.0, "doc_slide", .4), (0.5, "stamp", .85)]
        if gk == "document":
            return [(0.0, "doc_slide", .5), (1.7, "doc_highlight", .55)]
        if gk in ("newspaper", "news_article", "press_release",
                  "letter", "diary", "email_screenshot"):
            return [(0.0, "doc_slide", .5), (0.4, "page_flip", .4)]
        if gk in ("timeline", "mini_timeline"):
            return [(0.0, "timeline_draw", .5)] + [
                (0.6 + k * 0.42, "timeline_tick", .45) for k in range(4)]
        if gk == "process_diagram":
            return [(0.0, "reveal", .5)] + [
                (0.45 + k * 0.5, "process_step", .55) for k in range(4)]
        if gk in ("network_graph", "relationship_tree", "family_tree"):
            return [(0.0, "reveal", .5)] + [
                (0.5 + k * 0.45, "node_connect", .5) for k in range(5)]
        if gk == "cause_effect":
            return [(0.0, "reveal", .5), (0.6, "process_step", .55),
                    (1.4, "arrow_draw", .4), (2.0, "process_step", .55)]
        if gk in ("stat_dashboard", "comparison", "heatmap",
                  "vertical_bar", "donut_chart", "progress_bar",
                  "population_split", "ratio"):
            return [(0.0, "reveal", .5), (0.5, "chart_reveal", .5),
                    (1.5, "stat_settle", .6)]
        if gk == "currency_stat":
            # diegetic foley — a BUSINESS MONEY stat lands on soft metallic
            # register ticks + a quiet settle; never a generic whoosh.
            return [(0.0, "money_tick", .42), (0.5, "money_tick", .36),
                    (1.5, "stat_settle", .45)]
        if gk in ("surveillance", "military_hud", "status_indicator",
                  "radio_dial"):
            return [(0.0, "surveillance", .5), (0.5, "rec", .4)]
        if gk in ("breaking_news", "headline_crawl"):
            return [(0.0, "transition", .6), (0.1, "impact", .5)]
        if gk in ("name_reveal", "portrait", "mini_bio", "id_card"):
            return [(0.0, "reveal", .6), (0.5, "emphasis", .5)]
        if gk in ("title_card", "era_banner", "section_title",
                  "chapter_marker"):
            return [(0.0, "reveal", .6), (0.15, "text_slam", .55)]
        if gk == "statement":
            return [(0.0, "reveal", .5), (0.6, "word_pop", .4),
                    (1.2, "word_pop", .4)]
        if gk in ("quote_highlight", "long_quote", "footnote", "glossary",
                  "define_the_term", "did_you_know", "speech_bubble",
                  "sticky_note", "pull_quote"):
            return [(0.0, "reveal", .5), (0.6, "word_pop", .4)]
        if gk in ("gps_stamp", "compass", "calendar", "analog_clock",
                  "speedometer"):
            return [(0.0, "blip", .4), (0.4, "beep", .4)]
        if gk in ("evidence", "evidence_tag"):
            # diegetic foley — a subtle EVIDENCE HIT (soft low thud) + a quiet
            # ui tick when the tag snaps in; never a generic whoosh.
            return [(0.0, "evidence", .5), (0.55, "ui", .32)]
        if gk in ("framed_insert", "diagram_labels", "spotlight",
                  "microscope"):
            return [(0.0, "reveal", .55), (0.5, "ui", .4)]
        return [(0.0, "reveal", .55)]              # default: a soft reveal

    gfx_events: list[tuple] = []
    if overlays:
        _seen_rev: set = set()
        # v13.2 WHOOSH-CADENCE THROTTLE — user feedback: the 25-min
        # render had 365 reveal SFX (≈1 whoosh / 4s) which read as
        # repetitive and artificial. Apply a min-gap so the dominant
        # "whoosh" family (reveal / transition / text_slam / text
        # swooshes) fires at most once per WHOOSH_GAP seconds. The
        # quiet/synced sub-events (ui, word_pop, blip, beep, rec,
        # node_connect, typewriter clicks, stat_settle) are NOT
        # throttled — they're soft texture, not the distracting whoosh.
        # `VIDLORE_WHOOSH_GAP_S` overrides the default 22 s.
        try:                                   # env hard-override > niche*restraint
            _wenv = os.environ.get("VIDLORE_WHOOSH_GAP_S")
            WHOOSH_GAP = (float(_wenv) if _wenv
                          else _look_whoosh_gap() * _look_sfx_restraint_mult())
        except (TypeError, ValueError):
            WHOOSH_GAP = _look_whoosh_gap() * _look_sfx_restraint_mult()
        _WHOOSH_KINDS = {
            "reveal", "transition", "text_slam", "whoosh", "swoosh",
        }
        _last_whoosh = [-999.0]
        # SFX DIRECTOR (Phase 4) — per-primitive restraint: cap each cue's
        # intensity at its verified max, and keep silence-default cards
        # (statement / quote / definition …) SILENT so the line lands. All
        # defensive: any failure leaves the proven SFX path unchanged.
        _sd = None
        _sd_niche = ""
        _sd_events: list = []
        try:
            from .audio_director import sfx_director as _sd
            try:
                from .look_dna import look_get as _lg
                _sd_niche = str(_lg("niche") or "")
            except Exception:                                  # noqa: BLE001
                _sd_niche = ""
        except Exception:                                      # noqa: BLE001
            _sd = None

        for cue in graphic_cues:
            ct, gk = float(cue[0]), cue[2]
            rev = round(ct + 0.40, 2)          # ~when the card lands
            if rev <= 0 or rev >= narration.total or rev in _seen_rev:
                continue
            _seen_rev.add(rev)
            for off, kind, q in _gfx_sfx_seq(gk):
                tt = round(rev + off, 2)
                if not (0.0 < tt < narration.total):
                    continue
                if kind in _WHOOSH_KINDS:
                    # throttle the whoosh family by min-gap
                    if tt - _last_whoosh[0] < WHOOSH_GAP:
                        continue
                    _last_whoosh[0] = tt
                # SFX-director restraint: silence-default cards stay silent;
                # everything else is capped at its verified per-primitive max.
                qv = q
                if _sd is not None:
                    try:
                        if _sd.should_silence(gk=gk, kind=kind, niche=_sd_niche):
                            continue
                        qv = _sd.cap_intensity(q, gk=gk, kind=kind, niche=_sd_niche)
                    except Exception:                          # noqa: BLE001
                        qv = q
                    _sd_events.append({"time": tt, "kind": kind, "gk": gk,
                                       "intensity": qv})
                gfx_events.append((tt, kind, qv))

        # ── P5 (2026-06-01) — voice director-INJECTED stat / number cards ──
        # gold_number_callout / statistic_bar_reveal / … are chosen by the MG
        # DIRECTOR and carry NO script graphic_kind, so the graphic_cues loop
        # above never emitted their reveal SFX — the number beat (e.g. the
        # Cornell "96%" gold_number_callout at ~2:23) landed SILENT. Honour
        # each primitive's declared audio_cue with a soft reveal→settle:
        # de-duped against the cue reveals, whoosh-throttle-respecting, and
        # cooldown-gated (≥6 s between stat cards) so a cluster never machine-
        # guns; capped via the same SFX-director per-primitive restraint.
        # Director-injected scenes have their script card cleared upstream, so
        # there is never a double-trigger. Fully defensive — any failure
        # leaves the proven SFX path untouched.
        _MG_STAT_SEQ = {
            # the hero number COUNTS UP — give it a real, RAPID count rhythm:
            # a soft reveal, then a fast run of louder money ticks WHILE the
            # digits roll, then a confident settle when it LANDS. (Was a single
            # faint settle = inaudible; then only 3 ticks = too sparse/quiet.)
            "gold_number_callout":  [(0.0, "reveal", .5),
                                     (0.30, "money_tick", .74), (0.48, "money_tick", .74),
                                     (0.66, "money_tick", .74), (0.84, "money_tick", .74),
                                     (1.02, "money_tick", .74), (1.20, "money_tick", .74),
                                     (1.38, "money_tick", .74), (1.56, "money_tick", .74),
                                     (1.74, "money_tick", .74), (2.05, "stat_settle", .82)],
            "statistic_bar_reveal": [(0.0, "reveal", .45), (0.5, "chart_reveal", .5),
                                     (1.5, "stat_settle", .55)],
            "growth_curve_chart":   [(0.0, "reveal", .45), (0.6, "chart_reveal", .5),
                                     (1.6, "stat_settle", .5)],
            "proportion_ring":      [(0.0, "reveal", .45), (0.5, "chart_reveal", .5),
                                     (1.4, "stat_settle", .5)],
            "pictograph_scale":     [(0.0, "reveal", .45), (0.5, "chart_reveal", .5),
                                     (1.3, "stat_settle", .5)],
            "composition_stack":    [(0.0, "reveal", .45), (0.5, "chart_reveal", .5),
                                     (1.3, "stat_settle", .5)],
            "money_flow_empire":    [(0.0, "money_tick", .42), (0.6, "money_tick", .36),
                                     (1.5, "stat_settle", .45)],
        }
        if motion_graphics_primitives:
            _mg_starts, _acc = [], 0.0
            for _sd_d in durs:
                _mg_starts.append(_acc)
                _acc += _sd_d
            _last_mg_sfx = -999.0
            _mg_sfx_n = 0
            for _si in sorted(motion_graphics_primitives):
                _prim = str(motion_graphics_primitives.get(_si) or "")
                if not (0 <= _si < len(_mg_starts)):
                    continue
                if _prim == "gold_number_callout":
                    # The hero number COUNTS UP and LANDS at dur*0.55 (the renderer's
                    # count_frac in numbers/gold_number_callout.py). Its ticks must roll
                    # WITH the rolling digits and STOP the instant the number lands —
                    # never run on after (the old fixed offsets settled at +2.6s even on
                    # a 3.4s card whose count finished at 1.87s, so the ticking kept
                    # going after the digits froze). Build the tick run from the REAL
                    # count window, anchored at the card start (counting begins the
                    # moment the card appears, not +0.55s later).
                    _cdur = max(2.0, float(durs[_si]) if _si < len(durs) else 3.4)
                    _cend = round(_cdur * 0.55, 2)           # MUST match renderer count_frac
                    _ntick = max(5, min(14, int(round(_cend / 0.26))))  # ~rapid, steady cadence
                    _seq = [(0.10, "reveal", .5)]
                    for _k in range(_ntick):
                        # mild easeOut: ticks slightly denser early (digits roll fast,
                        # then settle); the last tick + the settle land at _cend, so no
                        # tick ever fires after the number has stopped counting.
                        _fr = 1.0 - (1.0 - (_k + 1) / (_ntick + 1)) ** 1.5
                        _seq.append((round(0.18 + (_cend - 0.18) * _fr, 2),
                                     "money_tick", .74))
                    _seq.append((_cend, "stat_settle", .82))
                    _rev = round(_mg_starts[_si], 2)         # count begins at card start
                else:
                    _seq = _MG_STAT_SEQ.get(_prim)
                    if not _seq:
                        continue
                    _rev = round(_mg_starts[_si] + 0.55, 2)  # ~when the number lands
                # `<= 0.0` (not `< 0.0`): a gold_number_callout on the OPENING scene
                # anchors at card start = 0.0s (the count begins the instant the card
                # appears). The old `0.0 < _rev` dropped its entire tick run; its events
                # are still clamped to t>0 individually below, so a 0.0 anchor is safe.
                if not (0.0 <= _rev < narration.total) or (_rev - _last_mg_sfx) < 6.0:
                    continue
                _any = False
                for _off, _kind, _q in _seq:
                    _tt = round(_rev + _off, 2)
                    if not (0.0 < _tt < narration.total) or _tt in _seen_rev:
                        continue
                    if _kind in _WHOOSH_KINDS:
                        if _tt - _last_whoosh[0] < WHOOSH_GAP:
                            continue
                        _last_whoosh[0] = _tt
                    _qv = _q
                    if _sd is not None:
                        try:
                            _qv = _sd.cap_intensity(_q, key=_prim, gk=_prim,
                                                    kind=_kind, niche=_sd_niche)
                        except Exception:                  # noqa: BLE001
                            _qv = _q
                        _sd_events.append({"time": _tt, "kind": _kind,
                                           "gk": _prim, "intensity": _qv})
                    gfx_events.append((_tt, _kind, _qv))
                    _seen_rev.add(_tt)
                    _any = True
                if _any:
                    _last_mg_sfx = _rev
                    _mg_sfx_n += 1
            if _mg_sfx_n:
                print(f"  [sfx] voiced {_mg_sfx_n} director-injected stat/number "
                      f"card(s) that were previously silent", flush=True)

    # Crossfades need every scene long enough to absorb the overlap;
    # otherwise fall back to safe hard-cut concat (never breaks a render).
    n_sc = len(scenes)
    ts, styles = _edit_plan(
        [_en(i) for i in range(n_sc)], n_sc,
        gap=getattr(_sty, "dissolve_gap", 3),
        drop=getattr(_sty, "dissolve_drop", 2),
        durs=durs,
        roles=[(roles[i] if i < len(roles) else "") for i in range(n_sc)],
        gkinds=[(graphics[i][0] if i < len(graphics) and graphics[i]
                 else "") for i in range(n_sc)],
        style_name=getattr(_sty, "name", ""))
    use_x = (
        transitions
        and n_sc >= 2
        and min(durs) > (max(ts) if ts else XFADE) + 0.15
    )

    # ---- BEAT CUTTING (the 3-second rule) -----------------------------
    # A real editor changes the visual every ~3s. Each scene's duration
    # is sliced into ~3.4s BEATS, EACH BEAT GETS ITS OWN DISTINCT CLIP
    # (footage.beat_clips) so the screen genuinely changes — not the same
    # shot re-framed. Narration + absolute caption/graphic timeline are
    # untouched, so total == narration (sync preserved).
    # Intra-scene B-roll changes are CLEAN HARD CUTS (real docs don't
    # crossfade every shot inside a topic) — _CUT is ~2 frames, visually
    # a cut. Crossfades/dissolves only happen at motivated scene
    # boundaries via the director plan above.
    # energy- AND role-aware uneven beat plan: Issue #2 (human rhythm,
    # not a metronome) + Issue #5 (footage-duration optimization — hold
    # the impact shots, cut faster when building). Same per-scene COUNT
    # as before so the footage fetcher stays valid.
    _roles = roles or []

    def _impact(i: int) -> bool:
        """Issue #8 — a high-impact line gets the zoom-in punch: a
        charged word in the narration OR a showrunner reveal/climax/turn
        beat. Kept RARE so the punch stays meaningful.

        CINEMATIC TASTE (HE#10) — let ONE element own a moment. When a
        full-frame data/text CARD is on this scene, the card IS the
        emphasis; an aggressive camera push-in behind it just fights for
        attention (the over-stacked, 'everything at once' tell). So a
        scene that carries a full-frame card never also earns the punch —
        the footage drifts gently under the card instead."""
        if 0 <= i < len(graphics) and graphics[i]:
            gk = (graphics[i][0] or "")
            if gk in _FULLFRAME_CARDS:
                return False
        if 0 <= i < len(_roles) and _roles[i] in (
                "reveal", "climax", "turn"):
            return True
        if 0 <= i < len(scenes):
            ws = getattr(scenes[i], "words", None) or []
            return _is_impact_text(" ".join(w.word for w in ws))
        return False

    def _breathe(i: int, m: int = 0) -> bool:
        """VISUAL RESTRAINT (HE#3) + AVOID CONSTANT MOTION (HE#7 camera).
        Should THIS beat HOLD (near-still) instead of drifting? An editor
        lets an emotional reaction or quiet resolution breathe, never
        fights a deliberate impact push-in — AND doesn't drift on every
        single shot (constant motion is the camera-tell of automation).
        So: always hold reaction/resolution & very-quiet beats; and on
        calm/info beats lock off SOME beats (seeded, ~1 in 3, per beat) so
        movement reads as a chosen accent, not a default. A build/peak or
        charged-impact beat never holds (those earn their motion)."""
        if _impact(i):
            return False
        role = _roles[i] if 0 <= i < len(_roles) else ""
        en = _en(i)
        if role in ("reaction", "resolution") and en <= 3:
            return True
        if en <= 1:
            return True
        # motivated stillness: not on a tension/build or peak beat. The
        # style mode scales how often we lock off (an epic lingers more).
        if en <= 3 and role not in (
                "escalation", "problem", "stakes",
                "reveal", "climax", "turn"):
            thr = min(0.78, 0.34 * getattr(_sty, "hold_bias", 1.0)
                      * _look_hold_mult())   # DOC_004 niche hold personality
            if _rng01(i * 53 + m * 17 + 5) < thr:
                return True
        return False

    plan = plan_beats(
        durs,
        target=getattr(_sty, "beat_target", 3.4),
        bmin=getattr(_sty, "beat_min", 2.4),
        energies=[_en(i) for i in range(n_sc)],
        roles=[_roles[i] if 0 <= i < len(_roles) else "" for i in range(n_sc)],
    )
    nb = len(plan)
    # FRAME-SNAP every beat so each segment is an EXACT integer-frame
    # length and a plain frame-accurate concat stays perfectly in sync.
    # (A 69-deep chained ffmpeg xfade with non-frame float offsets
    # accumulated drift and FROZE the tail ~100s — caught in deep
    # testing.) Each SCENE's beats are snapped to sum to that scene's
    # exact frame count, so the audio / caption / graphic timelines —
    # which key off scene durations — never drift.
    # CARRY THE ROUNDING. Each scene used to snap to its OWN frame count —
    # int(round(scene_seconds * FPS)) — which absorbs rounding WITHIN a scene but lets every
    # scene contribute up to half a frame of independent error. Across ~43 scenes that random
    # walk reached +0.227s in the acceptance render (~7 frames), which the sync invariant caught:
    # video 188.467s vs composed audio 188.240s. Fixing the transition pairing was necessary but
    # not sufficient — this was the other half.
    #
    # So allocate against the RUNNING clock instead of per scene: a scene's frame count is
    # whatever makes the cumulative total land on the exact cumulative narration time. Per-scene
    # error stays under a frame and the TOTAL is exact by construction, because each scene absorbs
    # its predecessor's remainder rather than starting a fresh one.
    from itertools import groupby as _gb
    beat_durs: list[float] = []
    _emitted_f = 0                       # frames committed so far
    _cum_t = 0.0                         # exact narration time up to and including this scene
    for _j, _grp in _gb(plan, key=lambda p: p[0]):
        _g = list(_grp)
        _cum_t += sum(x[1] for x in _g)
        _tot_f = max(len(_g), int(round(_cum_t * FPS)) - _emitted_f)
        _fr = [max(1, int(round(x[1] * FPS))) for x in _g]
        _k = _fr.index(max(_fr))         # absorb rounding on longest beat
        _fr[_k] = max(1, _fr[_k] + (_tot_f - sum(_fr)))
        _emitted_f += sum(_fr)
        beat_durs.extend(f / FPS for f in _fr)
    # CONFORM THE VIDEO TO THE COMPOSED-AUDIO CLOCK. The carry above snaps beat_durs to the SCENE
    # durations (ns.duration), but those are the engine's per-scene bookkeeping and diverge from the
    # audio the viewer actually hears by a sub-frame per scene — accumulating to ~0.13s over ~40
    # scenes, which the sync invariant refuses to publish.
    #
    # Target the DURATION OF narration.audio — the exact file the invariant probes — NOT
    # narration.total. They are different objects: for an uploaded VO with breakouts spliced,
    # narration.total is the intended scene sum (measured 188.367s) while the composed wav is
    # 188.227s. An earlier cut targeted narration.total and was a silent no-op because it equals the
    # scene sum the video already had. Probe the wav so the conform and the invariant share one
    # authority; fall back to narration.total only if the file cannot be read.
    _audio_f = getattr(narration, "audio", None)
    _audio_dur = None
    if _audio_f:
        try:
            _audio_dur = _probe_duration(_audio_f)
        except Exception:
            _audio_dur = None
    _tgt_f = int(round(float(_audio_dur if _audio_dur else
                             getattr(narration, "total", sum(beat_durs))) * FPS))
    _cur_f = int(round(sum(beat_durs) * FPS))
    if beat_durs and _tgt_f != _cur_f:
        # Absorb the delta into the longest NON-BREAKOUT beat. A breakout is a pre-rendered fixed
        # mp4 inserted directly (bmap), NOT encoded from its beat_dur — and it is usually the
        # LONGEST beat (8-10s vs 1-6s narration), so a naive argmax picks it and adjusting its
        # beat_dur changes nothing about the file that actually plays. That was the silent no-op
        # that survived four render attempts. A breakout / cold-open pseudo-scene is a NarratedScene
        # with NO word timings (`words=[]`); normal narration scenes always carry words, so exclude
        # the word-less ones and absorb into a real beat that the renderer actually encodes.
        def _is_breakout_beat(_bi):
            _j = plan[_bi][0] if _bi < len(plan) else -1
            _sc = scenes[_j] if 0 <= _j < len(scenes) else None
            return bool(_sc is not None and not getattr(_sc, "words", None))
        _cands = [i for i in range(len(beat_durs)) if not _is_breakout_beat(i)]
        if _cands:
            _k = max(_cands, key=lambda i: beat_durs[i])
            _adj = (_tgt_f - _cur_f) / FPS
            if beat_durs[_k] + _adj >= 0.2:      # never starve a beat below the floor
                beat_durs[_k] = round((beat_durs[_k] + _adj) * FPS) / FPS
                print(f"  [5/5] timeline conform: {_adj*1000:+.0f}ms into beat {_k} (non-breakout) "
                      f"so video == composed audio ({_tgt_f} frames)", flush=True)
    # Hard cuts by default — frame-exact concat, NO deep xfade chain (that
    # chain was the freeze source). The uneven RHYTHM lives in beat_durs,
    # independent of the join mechanism.
    beat_ts = [0.0] * max(0, nb - 1)
    use_x = False
    # DOCUMENTARY TRANSITION ENGINE. Clean cuts are the premium-doc default;
    # `_edit_plan` marked each motivated scene boundary with a transition
    # type (dissolve / archive_flash / geo_push / page_wipe / whip / glitch
    # / film_dissolve / fadeblack ...). We apply each as an ISOLATED PAIRWISE
    # xfade between the two adjacent beats — never a chain, so the old freeze
    # can't return. Frame-exact & sync-preserving: the tail beat renders +xf
    # longer and the xfade consumes that pad, so merged == the two beats'
    # sum. A boundary whose beats are too short to absorb the overlap simply
    # falls back to a clean cut (restraint over reflex).
    _tail_beat: dict[int, int] = {}                # scene -> last beat index
    for _bi, (_j, _bd, _last, _m) in enumerate(plan):
        if _last:
            _tail_beat[_j] = _bi
    beat_pad = [0.0] * nb
    # tail beat index -> (xf seconds, ffmpeg xfade transition name)
    trans_tails: dict[int, tuple] = {}
    _tcount: dict[str, int] = {}
    # PASS 1 — collect CANDIDATE tails. No padding is committed here.
    _cand_tails: dict[int, tuple] = {}
    for _j in range(n_sc - 1):
        styj = styles[_j] if 0 <= _j < len(styles) else "cut"
        if styj == "cut" or styj not in _TRANSITIONS:
            continue
        bi = _tail_beat.get(_j, -1)
        if bi < 0 or bi >= nb - 1:
            continue
        name, _base = _TRANSITIONS[styj]
        xf = float(ts[_j]) if 0 <= _j < len(ts) else _base
        xf = max(0.16, min(0.95, xf))
        # QUANTISE THE PAD TO WHOLE FRAMES. It is added to an already frame-aligned beat_durs and
        # the sum is then re-rounded at int(round((bd + pad) * FPS)) — so a fractional-frame pad
        # reintroduces up to half a frame of error on every PADDED segment. Measured: after the
        # pairing and carry fixes the acceptance render still drifted +0.127s ~= 3.8 frames, with 8
        # motivated transitions. Frame-aligned, (bd + xf) * FPS is an integer, so round() is
        # identity and the xfade offset arithmetic (off = seg0_fr - xf_fr) becomes exact instead of
        # approximately right.
        xf = max(1, int(round(xf * FPS))) / FPS
        # only if BOTH adjacent beats can absorb the overlap cleanly
        if (beat_durs[bi] >= xf + 0.4 and beat_durs[bi + 1] >= xf + 0.4):
            _cand_tails[bi] = (xf, name, styj)
    # PASS 2 — resolve a NON-OVERLAPPING pairing BEFORE any padding exists, then derive the pads
    # from the pairs that were actually selected.
    #
    # The pad used to be committed here, at plan time, for every candidate; the merge loop below
    # then re-decided pairing GREEDILY and independently. When two ADJACENT beats were both tails,
    # beat bi+1 got padded and was then consumed as the SECOND input of trans_{bi} — and ffmpeg's
    # xfade copies input 2 in full, so that pad rode through untouched. trans_{bi+1} was never
    # emitted, so nothing ever consumed it. Every later frame was permanently late by xf, the
    # transition was silently dropped, and nothing checked.
    #
    # Measured on the render that exposed this: 46 motivated transitions planned, 44 trans_*.mp4 on
    # disk. Two collisions × ~0.6s = the 1.200s by which the concat (925.100s) overran the audio
    # (923.900s) — and the 1.13s/1.18s by which both breakouts' pictures lagged their own audio.
    # Breakouts were merely where it became VISIBLE: a breakout is the only shot whose audio is
    # locked to its own picture.
    _taken: set = set()
    for bi in sorted(_cand_tails):
        if bi in _taken or (bi + 1) in _taken:
            continue                       # consumed by an earlier pair → clean cut, and NO pad
        xf, name, styj = _cand_tails[bi]
        trans_tails[bi] = (xf, name)
        _taken.update((bi, bi + 1))
        _tcount[styj] = _tcount.get(styj, 0) + 1
    for bi, (xf, _n) in trans_tails.items():
        beat_pad[bi] = xf
    if trans_tails:
        _summ = ", ".join(f"{k}×{v}" for k, v in sorted(_tcount.items()))
        _dropped = len(_cand_tails) - len(trans_tails)
        print(f"  [5/5] transitions: {len(trans_tails)} motivated "
              f"({_summ}) · rest clean cuts"
              + (f" · {_dropped} candidate(s) yielded to an adjacent pair (no pad)"
                 if _dropped else ""), flush=True)

    beat_clips = beat_clips or {}
    segs: list[Path] = []

    # ── PARALLEL SEGMENT ENCODING (USER RENDER-SPEED FIX 2026-05-25) ──
    # Old loop did `_scene_video()` once per beat serially. Each call is
    # a CPU-bound ffmpeg subprocess (1920x1080, libx264 / videotoolbox)
    # that pegged ~1.5-2s. For a 14-scene doc with 31 beats that adds up
    # to ~60s of pure encode wall time.
    #
    # Approach:
    #   1) PRE-PASS (serial, microseconds) — build per-beat encode args
    #      including the seeded camera mode (which used to depend on
    #      `prev_mode` from the previous loop iteration).
    #   2) PARALLEL ENCODE — ThreadPoolExecutor fires N ffmpeg
    #      subprocesses concurrently. Default 4, env VIDLORE_ENCODE_WORKERS.
    #      Threads (not processes) is fine: each ffmpeg is its own OS
    #      process so the GIL doesn't matter.
    #
    # Thread safety: the only shared state inside _scene_video is the
    # SCENE_CTX dict used by templates/_shared.scene_pack(). We resolve
    # the motion factor PER BEAT in the pre-pass (still on the main
    # thread) and pass the resulting kb_drift/kb_impact in — so the
    # parallel encode body touches no module-level mutable state.

    # PRE-PASS: build encode plan
    encode_plan: list[dict] = []
    prev_mode = -1
    _mg_off: dict = {}                 # scene pos -> cumulative MG slice offset
    for bi, (j, _bd_raw, _last, m) in enumerate(plan):
        # RENDER THE CARRIED TIMELINE, not the raw plan float. beat_durs is the frame-snapped,
        # carry-corrected clock built above; the encoder used to take `bd` straight from `plan` and
        # re-round it per segment at `int(round((bd + pad) * FPS))`, so beat_durs was computed and
        # then ignored by the very code it exists to constrain. Every segment then contributed its
        # own independent rounding error: measured +0.193s (~6 frames) across 43 beats in the
        # acceptance render, which the sync invariant refused to publish.
        bd = beat_durs[bi] if bi < len(beat_durs) else _bd_raw
        pad = beat_pad[bi] if bi < nb else 0.0      # dissolve-tail overlap
        ns = scenes[j]
        base = by_idx[ns.index]
        # use this beat's OWN distinct clip when available
        clips = beat_clips.get(ns.index) or []
        if m < len(clips):
            # IMPORTANT: derive is_video from THIS beat's actual file, not the
            # scene base. Beats within one scene now mix stock VIDEO with
            # archival STILLS (Wikimedia/LoC/Internet Archive/YouTube frames),
            # so inheriting base.is_video would tag a .jpg as video and send it
            # down the `-stream_loop -1 + fps` path — which deadlocks on a
            # still (the fps filter never gets an advancing PTS). Detect by
            # extension so each beat takes the correct (loop-still vs
            # stream-video) encode path.
            _cp = clips[m]
            _isv = str(_cp).lower().endswith(
                (".mp4", ".mkv", ".webm", ".mov", ".m4v"))
            item = FootageItem(ns.index, _cp, _isv)
        else:
            item = base
        kb = _kb(j)                         # palette for this shot_type
        if kb:
            # Issue #7: rotate the (seeded) palette and take the first
            # move that ISN'T the previous scene's — keeps motion
            # semantically right for the shot_type yet never repeats
            # back-to-back, so the camera language stays varied.
            s = int(_rng01(j * 17 + m * 5 + 11) * len(kb)) % len(kb)
            order = kb[s:] + kb[:s]
            mode = next((o for o in order if o != prev_mode), order[0])
        else:
            # no shot_type — seeded pick, also never repeating the
            # previous move (Issue #2 human rhythm).
            cand = [0, 1, 2, 3]
            if prev_mode in cand and len(cand) > 1:
                cand.remove(prev_mode)
            mode = cand[int(_rng01(j * 17 + m * 5 + 11) * len(cand))
                        % len(cand)]
        prev_mode = mode
        seg = workdir / f"seg_{bi:04d}.mp4"
        # Issue #6: layered cinematic finish on every non-archival scene;
        # archival keeps its own _VINTAGE stack (never doubled).
        # Per-scene OVERLAY PACK (palette-driven, recency-aware) — replaces
        # the single _CINEMA_FINISH so each scene carries its own subtle
        # texture personality (crt_scan vs paper_tex vs broadcast_noise...).
        sgrade = (_VINTAGE if _is_vintage(j)
                  else (grade + _overlay_finish_for_scene(j, ns)))
        # MOTION PACK MODULATION — resolved here on the main thread so the
        # parallel encode body never touches SCENE_CTX (which is not
        # thread-safe).
        _mfact = {"drift": 1.0, "impact": 1.0}
        try:
            from .templates._shared import set_scene_context, scene_pack
            set_scene_context(ns)
            _mv = scene_pack().motion
            _mfact = {
                "slow_push":     {"drift": 0.75, "impact": 0.60},
                "snap_hook":     {"drift": 1.20, "impact": 1.45},
                "parallax":      {"drift": 1.30, "impact": 1.00},
                "masked_reveal": {"drift": 0.85, "impact": 0.95},
                "wipe":          {"drift": 1.00, "impact": 1.05},
                "fade_up":       {"drift": 0.90, "impact": 0.85},
                "slide":         {"drift": 1.10, "impact": 1.10},
                "stagger":       {"drift": 1.00, "impact": 1.00},
            }.get(_mv, _mfact)
        except Exception:                                  # noqa: BLE001
            pass
        # CAMERA MOTION (P-camera): Look DNA drift_scale wins over
        # the style mode's drift when a channel is active.  Atlas
        # pushes 1.2× harder; Amber drifts at 0.55× (very slow
        # contemplative); Midnight runs at 0.85× (controlled
        # investigative).  Falls back to style mode when no channel.
        _ld_drift = None
        try:
            from .look_dna import current as _ld_current, look_get
            if _ld_current() is not None:
                _ld_drift = look_get("drift_scale", None)
                # drift_scale knob is a Range {min, max} resolved to
                # a point value by sample(); take it as-is when float.
                if isinstance(_ld_drift, dict):
                    _ld_drift = (
                        (_ld_drift.get("min", 1.0)
                         + _ld_drift.get("max", 1.0)) / 2.0)
        except Exception:                                  # noqa: BLE001
            _ld_drift = None
        _base_drift = (float(_ld_drift)
                       if _ld_drift is not None
                       else getattr(_sty, "drift_scale", 1.0))
        # IMP_003 — archival material gets a SLOWER, respectful drift (~0.5x).
        # Old photos / film should glide gently, not glide-zoom like modern
        # stock — a fast Ken-Burns on a 1920s portrait reads as cheap. The
        # sepia LUT + film grain are already applied to these scenes via
        # _VINTAGE (see `sgrade` above); halving the drift completes the
        # archival treatment. Detection reuses the LLM's explicit archival
        # shot_type signal (_is_vintage), so it never mis-fires on modern B-roll.
        _arch_drift = 0.5 if _is_vintage(j) else 1.0
        # motion-graphic beat: slice the scene's MG clip by cumulative offset so
        # it plays continuously across the scene's beats (no Ken-Burns, no card).
        _mgc = _mg.get(j)
        _mgoff = 0.0
        _win = None
        if _mgc:
            _mgoff = _mg_off.get(j, 0.0)
            _mg_off[j] = _mgoff + bd + pad
            # V1.1 DURATION WINDOW — the graphic owns the scene only for its
            # useful window; later beats return to FOOTAGE. For a beat that the
            # window ends INSIDE (common when a graphic scene is one long beat),
            # `_encode_one` builds a segment-internal composite: MG for the
            # window, then footage for the remainder (so footage truly returns
            # mid-scene). Whole beats past the window are cleared to footage.
            # No window (legacy) → the MG fills the scene as before.
            _win = _mgw.get(j)
            if _win is not None and _mgoff >= float(_win) - 0.05:
                _mgc = None
        encode_plan.append({
            "bi": bi, "j": j, "m": m, "bd": bd, "pad": pad,
            # ``j`` is the positional scene slot; ``ns.index`` is the stable
            # owner used by beat_clips and by ClipStudio's aired manifest.
            "scene_index": ns.index,
            "item": item, "seg": seg, "sgrade": sgrade, "mode": mode,
            "kb_seed":   j * 97 + m * 13 + 7,
            "kb_impact": _impact(j) * _mfact["impact"],
            "kb_hold":   _breathe(j, m),
            "kb_drift":  _base_drift * _mfact["drift"] * _arch_drift,
            "archival":  _is_vintage(j),          # IMP_011 — 4:3 pillarbox
            "mg_clip":   _mgc, "mg_off": _mgoff, "mg_window": _win,
        })
        segs.append(seg)

    # Bind BEFORE any ffmpeg work.  The decoded checks below are independent
    # canaries, but path/kind ownership is primary: a visually similar clip is
    # still wrong when it belongs to another selection.
    if _lineage_enabled:
        try:
            _bind_rows, _bind_failures = _bind_scene_lineage(
                encode_plan, _lineage_contract)
            _lineage_audit["binding"] = _bind_rows
            if _bind_failures:
                _fail_scene_lineage_audit(
                    _lineage_audit_path, _lineage_audit, "binding", _bind_failures)
            _lineage_audit["stage"] = "binding"
            _write_scene_lineage_audit(_lineage_audit_path, _lineage_audit)
        except SceneLineageError:
            raise
        except Exception as _lineage_exc:                       # noqa: BLE001
            _fail_scene_lineage_audit(
                _lineage_audit_path, _lineage_audit, "binding", [{
                    "stage": "binding",
                    "reason": f"lineage binding could not be checked: {_lineage_exc}",
                }])

    # PARALLEL ENCODE: ffmpeg subprocesses fire concurrently. Default 4
    # workers (good for an 8-core Mac); env override for tuning. Each
    # ffmpeg already uses internal threads; >4 outer workers tends to
    # thrash CPU/disk on consumer hardware so we cap modestly.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor as _Pool
    try:
        # V1 safe-tuning: documented VIDLORE_FFMPEG_WORKERS alias → falls back to
        # legacy VIDLORE_ENCODE_WORKERS → unchanged default 4 (LEVEL-A: default
        # byte-identical; knob only raises concurrency on a multi-core box).
        _enc_workers = max(1, int(
            _os.environ.get("VIDLORE_FFMPEG_WORKERS")
            or _os.environ.get("VIDLORE_ENCODE_WORKERS") or "4"))
    except ValueError:
        _enc_workers = 4
    _enc_workers = min(_enc_workers, len(encode_plan))

    _mg_dur_cache: dict = {}              # MG clip path -> probed duration (s)

    def _encode_one(p):
        if p.get("mg_clip"):
            # MOTION-GRAPHIC beat. The clip is sliced frame-exact (no Ken-Burns,
            # no overlays); tpad clones the last frame so a slice can never
            # under-run its beat (keeps the concat frame-exact).
            _mgp = str(Path(p["mg_clip"]).resolve())   # absolute: ffmpeg cwd=workdir
            # probe the clip duration once (cached) — the seek is clamped inside
            # the clip so -ss never lands past EOF (which would yield a 0-frame
            # 261-byte segment that crashes the xfade concat).
            _cd = _mg_dur_cache.get(_mgp)
            if _cd is None:
                import subprocess as _sp2
                try:
                    _pi = _sp2.run([ffmpeg_exe(), "-i", _mgp],
                                   capture_output=True, text=True)
                    _dm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", _pi.stderr)
                    _cd = (int(_dm.group(1)) * 3600 + int(_dm.group(2)) * 60
                           + float(_dm.group(3))) if _dm else 0.0
                except Exception:                              # noqa: BLE001
                    _cd = 0.0
                _mg_dur_cache[_mgp] = _cd

            def _mg_slice(out_name, dur_s, off):
                _nf = max(1, int(round(max(0.2, dur_s) * FPS)))
                _o = off
                if _cd > 1.0:
                    _o = min(_o, max(0.0, _cd - 0.6))   # bright pre-fade hold frame
                run(["-ss", f"{_o:.3f}", "-i", _mgp,
                     # RC5 — CINEMATIC FULLSCREEN MAP. A motion-graphic clip whose
                     # aspect is NOT 16:9 (e.g. a map/location primitive that leaked
                     # a portrait canvas) was previously FIT + black-padded
                     # (`decrease` + `pad`), which crushed it into a tiny centred
                     # vertical strip drowning in dead space — the "portrait
                     # primitive in a landscape video" look. FILL the frame instead
                     # (`increase` + centre-`crop`): a true 16:9 clip is unchanged
                     # (increase→1920x1080, crop is a no-op — SELF-GATING, zero
                     # effect on the 71 registry primitives' normal output), while a
                     # non-16:9 clip now expands edge-to-edge with the subject kept
                     # centred. No dead space, no speed cost (one scale, same as before).
                     "-vf", (f"tpad=stop_mode=clone:stop_duration=4,fps={FPS},"
                             "scale=1920:1080:force_original_aspect_ratio=increase,"
                             "crop=1920:1080,setsar=1,"
                             "format=yuv420p"),
                     "-frames:v", str(_nf), "-an", *_venc("18"),
                     "-colorspace", "bt709", "-color_primaries", "bt709",
                     "-color_trc", "bt709", "-color_range", "tv",
                     out_name], cwd=str(workdir))

            # The per-primitive USEFUL DURATION is enforced at the CLIP level
            # (the dispatched clip is window-length, the seek clamp holds a
            # bright pre-fade frame) and via the per-beat windowing above, which
            # returns later beats of a multi-beat scene to footage. A
            # segment-internal MG→footage composite for single long beats was
            # trialled but introduced a boundary artefact for no clear gain, so
            # the safe per-beat slice is used (footage-return for single-beat
            # graphic scenes is a beat-planning change, tracked separately).
            _beat = max(0.2, p["bd"] + p["pad"])
            _mg_slice(p["seg"].name, _beat, p["mg_off"])
            return p["bi"]
        _slated = _scene_video(
            p["item"], p["bd"] + p["pad"], p["sgrade"], p["seg"],
            _en(p["j"]), p["mode"],
            kb_seed=p["kb_seed"], kb_impact=p["kb_impact"],
            kb_hold=p["kb_hold"], kb_drift=p["kb_drift"],
            archival=p.get("archival", False))   # IMP_011
        if _slated:
            # `_scene_video` owns its own final slate fallback and historically
            # returned silently, so the outer pool classified it as a normal
            # success.  Surface it on the plan row: strict lineage rejects it,
            # and reliability accounting finally tells the truth.
            p["_lineage_emergency_slate"] = True
        return p["bi"]

    # ── HARDENED EXECUTION (encode-pool reliability, 2026-05-31). A single
    # malformed beat must NEVER abort the whole render. Each beat is wrapped:
    # on any exception we capture the FULL leaf traceback + context to a per-
    # render log, retry the beat once, then write a guaranteed slate so the
    # timeline stays frame-exact. `pool.map` re-raised the first failure and
    # abandoned every remaining beat — `submit`/`as_completed` + a per-worker
    # guard drains all of them instead.
    import threading as _th
    import traceback as _tb
    _CS_TAIL = (",scale=in_range=tv:out_range=tv,format=yuv420p,"
                "setparams=color_primaries=bt709:color_trc=bt709:"
                "colorspace=bt709:range=tv")
    _fail_log = workdir / "encode_failures.log"
    _enc_lock = _th.Lock()
    _enc_stats = {"ok": 0, "retried": 0, "slated": 0, "failed": 0}

    def _emergency_slate(p) -> bool:
        _nf = max(1, int(round(max(0.2, p["bd"] + p["pad"]) * FPS)))
        try:
            return _safe_slate(Path(p["seg"]), _nf,
                               p.get("sgrade") or "null", _CS_TAIL)
        except Exception:                                  # noqa: BLE001
            return False

    def _log_fail(p, stage):
        try:
            ipath = getattr(p.get("item"), "path", "?")
            try:
                isz = Path(str(ipath)).stat().st_size
            except Exception:                              # noqa: BLE001
                isz = -1
            with _enc_lock:
                with open(_fail_log, "a") as fh:
                    fh.write(
                        f"[{stage}] bi={p.get('bi')} scene_j={p.get('j')} "
                        f"mg={'Y' if p.get('mg_clip') else 'N'} "
                        f"seg={Path(p['seg']).name} "
                        f"src={Path(str(ipath)).name} src_bytes={isz}\n"
                        f"{_tb.format_exc()}\n\n")
        except Exception:                                  # noqa: BLE001
            pass

    def _encode_safe(p):
        """Run one beat; never let a single failure abort the pool."""
        try:
            r = _encode_one(p)
            with _enc_lock:
                if p.get("_lineage_emergency_slate"):
                    _enc_stats["slated"] += 1
                else:
                    _enc_stats["ok"] += 1
            return r
        except Exception:                                  # noqa: BLE001
            _log_fail(p, "first")
        try:                                               # retry once
            import time as _t
            _t.sleep(0.5)
            r = _encode_one(p)
            with _enc_lock:
                _enc_stats["retried"] += 1
            print(f"  [5/5] beat {p.get('bi')} (scene {p.get('j')}) "
                  f"recovered on retry", flush=True)
            return r
        except Exception:                                  # noqa: BLE001
            _log_fail(p, "retry")
        if _emergency_slate(p):                            # guaranteed frame
            p["_lineage_emergency_slate"] = True
            with _enc_lock:
                _enc_stats["slated"] += 1
            print(f"  [5/5] beat {p.get('bi')} (scene {p.get('j')}) "
                  f"unrecoverable → emergency slate "
                  f"(see {_fail_log.name})", flush=True)
            return p["bi"]
        with _enc_lock:
            _enc_stats["failed"] += 1
        print(f"  [5/5] beat {p.get('bi')} FAILED — even the slate could "
              f"not be written (see {_fail_log.name})", flush=True)
        return None

    if _enc_workers > 1 and len(encode_plan) > 1:
        from concurrent.futures import as_completed as _ac
        with _Pool(max_workers=_enc_workers) as pool:
            _futs = [pool.submit(_encode_safe, p) for p in encode_plan]
            for _f in _ac(_futs):
                _f.result()      # _encode_safe never raises; just drains
    else:
        # Serial path — same per-beat safety net.
        for p in encode_plan:
            _encode_safe(p)
    if _enc_stats["retried"] or _enc_stats["slated"] or _enc_stats["failed"]:
        print(f"  [5/5] encode reliability: {_enc_stats['ok']} ok · "
              f"{_enc_stats['retried']} retried · {_enc_stats['slated']} "
              f"emergency-slate · {_enc_stats['failed']} failed", flush=True)
    if _lineage_enabled:
        try:
            _enc_rows, _enc_failures, _lineage_encoded_banks = \
                _verify_lineage_encoded_plan(encode_plan)
            _lineage_audit["encoded_segments"] = _enc_rows
            if _enc_failures:
                _fail_scene_lineage_audit(
                    _lineage_audit_path, _lineage_audit,
                    "encoded_segments", _enc_failures)
            _lineage_audit["stage"] = "encoded_segments"
            _write_scene_lineage_audit(_lineage_audit_path, _lineage_audit)
        except SceneLineageError:
            raise
        except Exception as _lineage_exc:                       # noqa: BLE001
            _fail_scene_lineage_audit(
                _lineage_audit_path, _lineage_audit, "encoded_segments", [{
                    "stage": "encoded_segments",
                    "reason": f"encoded lineage could not be checked: {_lineage_exc}",
                }])
    if not use_x:
        bmin, bmax = min(beat_durs), max(beat_durs)
        print(f"  [5/5] {nb} beats from {n_sc} scenes · hard cuts, "
              f"frame-exact · shot len {bmin:.1f}-{bmax:.1f}s "
              f"(uneven human rhythm)", flush=True)

    video_only = workdir / "video_only.mp4"
    if use_x:
        fc, final = _xfade_chain(beat_durs, beat_ts)
        # Parallelism flags (multi-threaded filter graph) + route the
        # encoder through the smart picker so this path uses hw
        # videotoolbox on Mac / nvenc on Win-NVIDIA instead of the
        # hardcoded libx264 it had before.  Quality target identical.
        args: list[str] = list(_ffthreads())
        for s in segs:
            args += ["-i", s.name]
        args += [
            "-filter_complex", fc,
            "-map", final, "-r", str(FPS),
            *_venc("20"),
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
            video_only.name,
        ]
        run(args, cwd=str(workdir))
    else:
        # Pre-merge the motivated TRANSITION pairs (isolated pairwise
        # xfades — each its own transition type), then frame-exact
        # hard-concat everything. With no transitions this is byte-identical
        # to the plain concat.
        units: list[Path] = []
        skip = -1
        for bi, s in enumerate(segs):
            if bi == skip:
                continue
            if bi in trans_tails and bi + 1 < len(segs):
                xf, tname = trans_tails[bi]
                xf_fr = max(1, int(round(xf * FPS)))
                # seg[bi] was rendered (plan dur + xf) long; the transition
                # begins after its nominal tail so merged == the two beats'
                # sum (frame-exact, sync preserved).
                # the CARRIED duration, matching what was actually encoded — reading plan[bi][1]
                # here re-introduced the raw float the renderer no longer uses, so the xfade offset
                # described a segment length that does not exist
                _bd0 = beat_durs[bi] if bi < len(beat_durs) else plan[bi][1]
                seg0_fr = max(1, int(round((_bd0 + xf) * FPS)))
                off = max(0.0, (seg0_fr - xf_fr) / FPS)
                merged = workdir / f"trans_{bi:04d}.mp4"
                run([
                    *_ffthreads(),
                    "-i", segs[bi].name, "-i", segs[bi + 1].name,
                    "-filter_complex",
                    f"[0:v][1:v]xfade=transition={tname}:duration={xf:.3f}:"
                    f"offset={off:.3f},format=yuv420p[v]",
                    "-map", "[v]", "-r", str(FPS), *_venc(),
                    "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv",
                    merged.name,
                ], cwd=str(workdir))
                units.append(merged)
                skip = bi + 1
            else:
                units.append(s)
        concat = workdir / "scenes.txt"
        concat.write_text(
            "\n".join(f"file '{u.name}'" for u in units) + "\n"
        , encoding="utf-8")
        # RE-ENCODE the concatenation (not -c copy): independently
        # encoded segments have separate GOPs/PTS, and a stream-copy
        # concat can glitch/stall at the joins. A single clean re-encode
        # of the frame-exact segments guarantees a continuous, perfectly
        # synced stream end to end.
        run([
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-r", str(FPS), *_venc(),
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
            str(video_only),
        ])
        _conform_video_to_audio(video_only, narration, workdir)
        _assert_video_audio_sync(video_only, narration, workdir)

    if _lineage_enabled:
        try:
            _timeline_rows, _timeline_failures = _verify_lineage_timeline_order(
                video_only, encode_plan, beat_durs, trans_tails,
                _lineage_encoded_banks, FPS)
            _lineage_audit["timeline_order"] = _timeline_rows
            if _timeline_failures:
                _fail_scene_lineage_audit(
                    _lineage_audit_path, _lineage_audit,
                    "timeline_order", _timeline_failures)
            _lineage_audit["status"] = "passed"
            _lineage_audit["stage"] = "timeline_order"
            _write_scene_lineage_audit(_lineage_audit_path, _lineage_audit)
            print(f"  [5/5] scene-lineage canary: {len(encode_plan)} beat(s) "
                  f"bound + decoded in order → {_lineage_audit_path.name}", flush=True)
        except SceneLineageError:
            raise
        except Exception as _lineage_exc:                       # noqa: BLE001
            _fail_scene_lineage_audit(
                _lineage_audit_path, _lineage_audit, "timeline_order", [{
                    "stage": "timeline_order",
                    "reason": f"timeline lineage could not be checked: {_lineage_exc}",
                }])

    _srt(words, out_path.with_suffix(".srt"),
         protected_windows=breakout_windows,
         schedule=_approved_caption_schedule)

    # ═══════════════════════════════════════════════════════════════════
    # PRE-BAKE per-scene overlays (v11). The 80+ `movie=` overlay stages
    # used to be applied at the final mux on top of video_only.mp4;
    # that was the single biggest wall-clock cost (2.5+ hr on 5-min
    # samples) because the giant linear filter graph couldn't parallelise.
    # Now we bake them per-scene in parallel BEFORE the final mux.
    # ═══════════════════════════════════════════════════════════════════
    import time as _time
    _video_for_final: Path = video_only
    _bake_timings: dict = {}
    if g_stages and os.environ.get("VIDLORE_SKIP_SCENE_BAKE") != "1":
        # Compute scene windows from `durs` (the per-scene duration list
        # the rest of `assemble` already uses).
        _scene_starts: list[float] = []
        _t_acc = 0.0
        for _d in durs:
            _scene_starts.append(_t_acc)
            _t_acc += _d
        _scene_ends = [s + d for s, d in zip(_scene_starts, durs)]
        try:
            _t_bake = _time.time()
            print(
                f"  [5/5] per-scene overlay bake: {len(g_stages)} stages "
                f"across {len(durs)} scenes (parallel x4) …",
                flush=True,
            )
            _video_for_final, _bake_timings = _bake_per_scene_overlays(
                video_in=video_only,
                workdir=workdir,
                g_stages=g_stages,
                scene_starts=_scene_starts,
                scene_ends=_scene_ends,
                venc_args=_venc("20"),
                fps=FPS,
            )
            print(
                f"  [5/5] bake done: "
                f"{_bake_timings['n_scenes_with_overlays']}/"
                f"{_bake_timings['n_scenes_total']} scenes had overlays · "
                f"parallel_wall={_bake_timings['parallel_bake_wall_s']:.1f}s · "
                f"slowest_scene={_bake_timings['per_scene_max_s']:.1f}s · "
                f"concat={_bake_timings['concat_wall_s']:.1f}s · "
                f"total={_bake_timings['total_s']:.1f}s",
                flush=True,
            )
            # The card stages have already been baked → drop them from
            # the final-mux video filter chain so we don't apply them
            # twice (which would also re-introduce the bottleneck).
            g_stages = []
        except Exception as _e:                                # noqa: BLE001
            print(
                f"  [5/5] per-scene bake FAILED ({type(_e).__name__}: "
                f"{str(_e)[:120]}); falling back to single-stage final mux",
                flush=True,
            )
            _video_for_final = video_only
            # leave g_stages intact for the legacy path

    # BLACK-FRAME REPAIR (v13.2) — last-line guarantee against any black
    # span that slipped through the footage ladder / bake. Freeze-holds
    # the previous frame across the gap; no-op when the video is clean.
    _lineage_repair_windows = None
    if _lineage_enabled:
        _lineage_repair_windows = []
        _lrw_start = 0.0
        for _lrw_pos, _lrw_plan in enumerate(encode_plan):
            _lrw_dur = (beat_durs[_lrw_pos] if _lrw_pos < len(beat_durs)
                        else float(_lrw_plan.get("bd") or 0.0))
            _lineage_repair_windows.append((
                _lrw_start, _lrw_start + _lrw_dur, _lrw_plan.get("bi")))
            _lrw_start += _lrw_dur
    try:
        _video_for_final = _repair_black_frames(
            _video_for_final, workdir, FPS, breakout_windows=breakout_windows,
            lineage_windows=_lineage_repair_windows)
    except SceneLineageError:
        raise
    except Exception as _e:                                # noqa: BLE001
        print(f"  [5/5] black-frame repair skipped ({str(_e)[:60]})",
              flush=True)

    inputs = ["-i", _video_for_final.name,
              "-i", str(narration.audio.resolve())]
    idx = 2
    music_i = sfx_i = None
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        music_i = idx
        idx += 1

    # Transition whoosh — OFF by default (a whoosh on every cut sounds
    # unnatural); the ambient music bed carries the cinematic sound.
    sfx_path = None
    if sfx and len(scenes) >= 2 and boundaries:
        # Motivated layer (PHASE 3.3): a boom when a high-energy scene
        # LANDS (depth scaled by how high the energy is) + a riser into
        # it; a soft whoosh ONLY on a motivated emotional DISSOLVE (a
        # story exhale). Ordinary clean cuts get nothing.
        sfx_events: list[tuple] = []
        # ── SFX RESTRAINT (anti-automation) ─────────────────────────
        # Old policy: every motivated dissolve = whoosh.  After 2-3 in
        # a row the viewer registers "automated transition library."
        # New rules:
        #   * Max 1 whoosh per 12s window (cinematic restraint cadence)
        #   * Boom/riser also throttled — max 1 deep climax hit per 30s
        #   * Skip the whoosh entirely when the music is already
        #     swelling INTO this cut (the swell IS the transition).
        # Caller tracks the last-fire timestamps in a small dict.
        _last_fire: dict[str, float] = {"whoosh": -99.0, "boom": -99.0}
        # SFX rhythm — Look DNA may override the dedup cadence so a
        # tense investigative channel can fire whooshes more often than
        # a slow historical one.  Falls back to the cinematic-pass
        # defaults (12s / 30s) when no channel is active.
        try:
            from .look_dna import look_get as _lg
            _MIN_GAP = {
                "whoosh": float(_lg("sfx.whoosh_min_gap_s", default=12.0)),
                "boom":   float(_lg("sfx.boom_min_gap_s",   default=30.0)),
            }
        except Exception:                                          # noqa: BLE001
            _MIN_GAP = {"whoosh": 12.0, "boom": 30.0}
        for i in range(1, len(scenes)):
            st = boundaries[i - 1]
            en = _en(scenes[i].index)
            style = styles[i - 1] if 0 <= i - 1 < len(styles) else "cut"
            if en >= 4:
                # energy 4 -> moderate impact, 5+ -> deep climax hit
                if st - _last_fire["boom"] >= _MIN_GAP["boom"]:
                    q = 0.45 if en == 4 else 1.0
                    sfx_events.append((st, "boom", q))
                    sfx_events.append((st, "riser", q))
                    _last_fire["boom"] = st
                    _last_fire["whoosh"] = st       # boom suppresses whoosh
            elif style == "dissolve":
                if st - _last_fire["whoosh"] >= _MIN_GAP["whoosh"]:
                    # vary the amplitude per fire so the whoosh doesn't
                    # sound identical each time (real foley editors pick
                    # different files; we vary level + a 5-10ms timing
                    # micro-jitter via the 'q' field which feeds into
                    # build_sfx_bed's per-event gain).
                    q = 0.50 + 0.18 * ((i * 7 + 3) % 5) / 4.0
                    sfx_events.append((st, "whoosh", round(q, 3)))
                    _last_fire["whoosh"] = st
        if sfx_events:
            sfx_path = build_sfx_bed(
                sfx_events, narration.total, workdir / "sfx.wav"
            )
            inputs += ["-i", str(sfx_path.resolve())]
            sfx_i = idx
            idx += 1

    # Number/stat/chart reveal accent — independent of the (off) boom
    # layer; this is the soft data-hit the user asked for, low + sparse.
    gsfx_i = None
    if overlays and num_events:
        gpath = build_number_sfx(
            num_events, narration.total, workdir / "numsfx.wav")
        inputs += ["-i", str(gpath.resolve())]
        gsfx_i = idx
        idx += 1

    # INTRO ACCENT (v14) — the hook must have sound design. The baseline render
    # had ZERO SFX in the first 37 s (the open felt flat). A soft riser into the
    # open + one low impact as the hook lands gives the first seconds a cinematic
    # lift even before any motion-graphic card. Single + tasteful (never a
    # trailer stinger); only when there is a real intro to support.
    if overlays and float(getattr(narration, "total", 0) or 0) >= 12:
        _intro_acc = [(0.45, "riser", 0.6), (2.5, "impact", 0.55)]
        gfx_events = _intro_acc + list(gfx_events)
        # also log it on the cue-sheet timeline (_sd_events) so QA/probe see it
        _sde0 = locals().get("_sd_events")
        if isinstance(_sde0, list):
            _sde0[:0] = [{"time": _t, "kind": _k, "gk": "intro", "intensity": _q}
                         for (_t, _k, _q) in _intro_acc]

    # TEXT-MOMENT SFX bed (whoosh-on-reveal + impact-on-landing) — ON
    # whenever overlays are on, INDEPENDENT of the old per-cut sfx flag, so
    # every text/graphic reveal has matching cinematic sound support.
    gfxrev_i = None
    if overlays and gfx_events:
        grpath = _sfxlib.build_event_bed(
            gfx_events, narration.total, workdir / "gfxsfx.wav")
        inputs += ["-i", str(grpath.resolve())]
        gfxrev_i = idx
        idx += 1

    # TYPEWRITER click track — synced to the per-character reveal times the
    # typing_date typewriter emitted (always on with overlays).
    type_i = None
    if overlays and type_events:
        tpath = build_typewriter_sfx(
            type_events, narration.total, workdir / "typesfx.wav")
        inputs += ["-i", str(tpath.resolve())]
        type_i = idx
        idx += 1

    # Faint analog room-tone only over 'archival' scenes (hiss + hum) —
    # makes recovered footage feel real; sits far under everything.
    arch_i = None
    if arch_windows:
        apath = build_archival_bed(
            arch_windows, narration.total, workdir / "archbed.wav")
        inputs += ["-i", str(apath.resolve())]
        arch_i = idx
        idx += 1

    # ATMOSPHERE (HE#8 + IMP_012) — read each scene's environment and lay a
    # faint, evolving world texture under it (desert wind, cave rumble, room
    # hum …). Adjacent same-environment scenes MERGE into one continuous bed
    # (no re-triggering) and intensity rises a touch with scene energy.
    # IMP_012 — ROOM TONE UNDER EVERYTHING: scenes with no clear environment
    # no longer fall to dead digital silence; they get the faintest neutral
    # "room" hum (mains hum + soft hiss, the quietest texture) at the
    # intensity floor. A real documentary always carries a subconscious
    # ambient layer — and this gives the IMP_006 music-silence reveals an
    # "ambient only" floor (per the DNA) instead of a sterile dropout. The
    # level (~0.019 amplitude after _ATMOS_VOL) stays felt, not heard.
    atmos_i = None
    atmos_windows: list[tuple] = []
    # R1 cinematic weight (flag-gated, default OFF → byte-identical mix). A
    # per-scene sustained low-end level on emotionally-weighted beats, carried
    # as window[4] and synthesized INSIDE the atmosphere bed (build_atmosphere_bed).
    # Default-ON (2026-06-04): validated on Edison (low-end lifted only on the
    # climax/payoff beats; voice band, LUFS, true-peak, dynamics all unchanged;
    # tuned gentle after listening review). Disable with VIDLORE_CINEMATIC_WEIGHT=0.
    _cw_on = os.environ.get("VIDLORE_CINEMATIC_WEIGHT", "1").strip().lower() in (
        "1", "true", "yes", "on")
    _cw_mult = {"light": 0.62, "balanced": 1.0, "strong": 1.45}.get(
        os.environ.get("VIDLORE_CINEMATIC_WEIGHT_LEVEL", "balanced").strip().lower(), 1.0)
    # FEWER BUT BETTER: pre-rank scenes by "heat" (energy + role) and weight ONLY
    # the top few heavy moments (cap 3) — the single heaviest gets strong (1.0),
    # the next 1-2 get light (0.6). Avoids the broad-energy smear (muddy) and a
    # single lonely window. Capped + flag-gated.
    _scene_wt = [0.0] * len(durs)
    if _cw_on:
        _cand = []
        for _ci in range(min(len(durs), len(scenes))):
            _e = max(1, min(5, _en(_ci)))
            _role = (getattr(scenes[_ci], "role", "") or "").strip().lower()
            if _e >= 4 or _role in _CW_STRONG:
                _heat = _e + (1.5 if _role in _CW_STRONG
                              else 0.6 if _role in _CW_MED else 0.0)
                _cand.append((_heat, _e, _ci, _role))
        _cand.sort(reverse=True)
        for _rank, (_heat, _e, _ci, _role) in enumerate(_cand[:3]):
            _base = 1.0 if (_rank == 0 and (_e >= 5 or _role in _CW_STRONG)) else 0.6
            _scene_wt[_ci] = round(_base * _cw_mult, 3)
    _at = 0.0
    _cur = ""
    _cs = 0.0
    _cq = 0.0
    _cw = 0.0
    for _i, _d in enumerate(durs):
        kind = _atmos_kind(scenes[_i]) if _i < len(scenes) else ""
        _energy = max(1, min(5, _en(_i)))
        q = (_energy - 1) / 4.0
        wt = _scene_wt[_i] if (_cw_on and _i < len(durs)) else 0.0
        if not kind:
            kind, q = "room", 0.0          # IMP_012 — neutral floor room tone
        # Merge adjacent scenes only when BOTH the environment kind AND the
        # weight bucket match — so a weighted reveal/climax beat is its own
        # window (scene-aware), never smeared across calm scenes into a constant
        # floor. Flag OFF => wt==_cw==0 for all => merges by kind exactly as before.
        if kind and kind == _cur and wt == _cw:
            _cq = max(_cq, q)
        else:
            if _cur:
                atmos_windows.append((_cs, _at, _cur, _cq, _cw))
            _cur, _cs, _cq, _cw = kind, _at, q, wt
        _at += _d
    if _cur:
        atmos_windows.append((_cs, _at, _cur, _cq, _cw))
    if _cw_on:
        _wtw = [w for w in atmos_windows if len(w) > 4 and w[4] > 0]
        print(f"  [cinematic-weight] level x{_cw_mult:.2f} · "
              f"{len(_wtw)}/{len(atmos_windows)} weighted window(s): "
              + ", ".join(f"{w[0]:.0f}-{w[1]:.0f}s w={w[4]:.2f}" for w in _wtw[:8]),
              flush=True)
    if atmos_windows:
        atpath = build_atmosphere_bed(
            atmos_windows, narration.total, workdir / "atmosbed.wav")
        inputs += ["-i", str(atpath.resolve())]
        atmos_i = idx
        idx += 1

    vfilters: list[str] = []
    if overlays:
        # Cinematic finish, but LIGHT-handed: a SOFT vignette edge plus a
        # midtone/shadow LIFT (gamma>1, small brightness bump) so the
        # picture stays clearly visible — the earlier deep vignette +
        # medium_contrast curve + gamma<1 stacked up and crushed footage
        # into near-black on phones. Subtle grain keeps it filmic.
        #
        # RC5.1 — this whole-video pass is a SECOND vignette/grade/grain on
        # top of the per-scene baked finish. The OVERLAY-RESTRAINT clarity
        # gate runs here to de-stack: it bounds this final vignette (softer
        # so two vignettes don't compound), caps the grade gamma at the luma
        # floor (the GILDED gamma=1.05 used to crush shadows toward black),
        # and trims the finishing grain — keeping the premium grade, losing
        # the mud. Pure parameter-level (no pixel pass), so it's cheap.
        _grade_sat = _look_grade_saturation()   # DOC_014 niche muted grade
        _gilded = _look_grade_mode() == "gilded"
        _final_gamma = 1.05 if _gilded else 1.08
        _gate = _OVR.clarity_gate(
            grain=_OVR.final_grain_amount(),
            vignette_div=_OVR.vignette_angle(),   # bounded final vignette
            texture_layers=1,                     # grade counts as one layer
            darken_gamma=_OVR.darken_gamma(_final_gamma),
            scene_kind="footage",
        )
        if _gate["reductions"]:
            print("  [overlay-restraint] final stack de-staged: "
                  + ", ".join(_gate["reductions"]), flush=True)
        vfilters.append(
            f"vignette=angle=PI/{_gate['vignette_div']:.2f}:eval=init")
        if _gilded:
            # GILDED sub-grade (P2) — warm firelight-on-near-black for
            # tycoon business / biography arcs: shadows crushed (lower
            # brightness, gamma~1 so they don't lift), warm GOLD push
            # across shadows/mids/highlights, saturation PRESERVED (the
            # gold IS the point — not the muted documentary floor).
            # RC5.1: gamma comes from the clarity gate (luma floor) so even
            # the deliberately moody gilded look never crushes to pure black.
            vfilters.append(
                f"eq=contrast=1.07:brightness=0.012:saturation=0.94:"
                f"gamma={_gate['darken_gamma']:.3f}")
            vfilters.append(
                "colorbalance=rs=0.055:gs=0.02:bs=-0.06:"
                "rm=0.045:bm=-0.05:rh=0.035:bh=-0.045")
        else:
            vfilters.append(
                f"eq=contrast=1.04:brightness=0.030:"
                f"saturation={_grade_sat:.3f}:gamma={_gate['darken_gamma']:.3f}"
            )
        vfilters.append(f"noise=alls={_gate['grain']}:allf=t")
    if captions and words:
        emph: set[str] = set()
        for e in (emphasis or []):
            for tok in re.findall(r"[a-z']{3,}", (e or "").lower()):
                emph.add(tok)
        # EDITORIAL RESTRAINT (full-doc QA): when a full-screen graphic
        # CARD is on screen, drop the burned caption for that window —
        # the card already carries the on-screen text, and stacking a
        # caption over it (seen ghosting under the timeline card) reads
        # as cluttered, not premium. A real editor lets ONE text element
        # own the frame. Cards are sparse/capped, so this only quiets the
        # caption for a few seconds at a time.
        cap_words = words
        # drop the burned caption over (a) a full-screen graphic card, and (b) any window where
        # the FOOTAGE itself already carries on-screen text (a ripped clip's burned-in dialogue
        # subtitle / logo) — stacking our caption on top reads as a messy double-text. One text
        # element owns the frame. `caption_suppress_windows` = [(start,end), ...] in seconds.
        _drop_wins = []
        if graphic_cues:
            _drop_wins += [(float(gc[0]), float(gc[0]) + float(gc[1])) for gc in graphic_cues]
        if caption_suppress_windows:
            for w_ in caption_suppress_windows:
                _drop_wins.append(w_)
        if _drop_wins:
            from .captions import _normalize_caption_windows, assert_caption_schedule
            try:
                _drop_wins = _normalize_caption_windows(
                    _drop_wins, label="blocked", merge_overlaps=True)
            except ValueError:
                # Persist the malformed contract before failing; otherwise an
                # all-covering bad window could remove every word and bypass
                # the burn-caption gate entirely.
                assert_caption_schedule(
                    words, workdir / "caption_burn_readability_audit.json",
                    protected_windows=breakout_windows,
                    blocked_windows=_drop_wins)
                raise
        if _drop_wins:
            def _under_card(w) -> bool:
                word_start = float(getattr(w, "start", 0.0))
                word_end = float(getattr(w, "end", word_start))
                return any(word_start < end and word_end > start
                           for start, end in _drop_wins)

            cap_words = [w for w in words if not _under_card(w)]
        if _drop_wins:
            from .captions import assert_caption_schedule
            _burn_schedule = assert_caption_schedule(
                cap_words, workdir / "caption_burn_readability_audit.json",
                protected_windows=breakout_windows,
                blocked_windows=_drop_wins)
        else:
            _burn_schedule = _approved_caption_schedule
        if cap_words:
            ass = write_ass(
                cap_words, workdir / "captions.ass",
                style=theme["caption"],
                # captions use `caption_accent` when set (a per-caption active-word colour that
                # must NOT recolour title/graphic overlays or key-phrase stabs), else the accent.
                accent=theme.get("caption_accent", theme.get("accent", (255, 210, 90))),
                emphasis_words=emph,
                schedule=_burn_schedule,
            )
            vfilters.append("subtitles=filename=%s" % ass.name)
    if overlays:
        vfilters += _overlay_filters(
            title, theme["accent"], narration.total, workdir,
            chapters=chapter_cues,
        )
        # (graphic-card filters already computed above)
        # Synced key-phrase stabs (engaging text, exact word timing).
        if not captions:                 # skip if full captions are on
            try:
                # NarratedScene exposes its text via WordTiming list, not a
                # `.narration` attribute — pull words instead so the sample
                # actually contains the scene's RTL / CJK characters.
                _sample_kp = " ".join(
                    " ".join(getattr(w, "word", "") for w in
                             (getattr(s, "words", []) or []))[:200]
                    for s in narration.scenes[:3])
                if not _sample_kp.strip():
                    # last-ditch: pull from emphasis itself
                    _sample_kp = " ".join(str(x) for x in (emphasis or [])[:3] if x)
            except Exception:                              # noqa: BLE001
                _sample_kp = ""
            vfilters += _keyphrase_filters(
                narration.scenes, emphasis or [],
                _copy_font(workdir, theme, sample_text=_sample_kp),
                theme.get("accent", (255, 210, 90)),
                shot_types=shot_types,
                graphic_kinds=[
                    (gg[0] if gg else "") for gg in (graphics or [])],
                roles=_roles,
                intensities=[_en(i) for i in range(len(durs))],
            )
        # Issue #12: rare, motivated act-break light-flash (into the
        # reveal/climax) — everywhere else stays a pure hard cut.
        vfilters += _actbreak_filters(
            durs, _roles, [_en(i) for i in range(len(durs))])
    # Pin the base stream's pixel format + colorspace BEFORE anything
    # else so the graph can never "reinitialize filters" on a colorspace
    # change (and so the movie-overlay cards composite in a defined
    # space). This is the safety net for the -22 / Invalid color space
    # crash even if an upstream file slips through untagged.
    norm = [
        "format=yuv420p",
        "setparams=color_primaries=bt709:color_trc=bt709:"
        "colorspace=bt709:range=tv",
    ]
    # Build the video graph as ordered segments so timed image cards
    # (portrait / archive photo) can be `movie`+`overlay`'d between the
    # base look and the on-top text, without extra -i inputs. Factored
    # into a closure so the SAME graph can be composed twice: once WITH the
    # burned-subtitle filter (the captioned export — byte-identical to
    # before) and — for the Review Editor's instant caption toggle — once
    # WITHOUT it, yielding a footage-matched no-caption base (LIVE_CAPTION).
    def _compose_vchain(_vfilters: list[str]) -> tuple[str, str]:
        comma = norm + _vfilters + g_comma
        segs: list[str] = []
        cur = "0:v"
        if comma:
            segs.append(f"[{cur}]" + ",".join(comma) + "[vbf]")
            cur = "vbf"
        oi = 0
        for seg in g_stages:
            if "{CUR}" in seg:
                nxt = f"vov{oi}"
                segs.append(seg.replace("{CUR}", cur).replace("{OUT}", nxt))
                cur, oi = nxt, oi + 1
            else:
                segs.append(seg)
        if g_post:
            segs.append(f"[{cur}]" + ",".join(g_post) + "[v]")
            cur = "v"
        elif cur != "0:v":
            segs.append(f"[{cur}]null[v]")
            cur = "v"
        if cur == "0:v":
            return "", "0:v"
        return ";".join(segs), "[v]"

    vchain, vmap = _compose_vchain(vfilters)

    # LIVE_CAPTION (2026-06-03) — emit a footage-matched NO-CAPTION base
    # alongside the captioned MP4 *in this same render*, so the editor can
    # toggle captions on/off instantly (an HTML overlay over a clean base)
    # WITHOUT the old non-deterministic whole-pipeline re-render (which
    # re-selected footage). The only difference from the captioned graph is
    # the dropped `subtitles=` filter — same baked `_video_for_final`, same
    # grade / cards / overlays / act-breaks — so the two outputs differ
    # ONLY by the burned captions. (When captions are ON, key-phrase stabs
    # are already skipped, so nothing else diverges.) Gated by
    # VIDLORE_EMIT_NOCAP (default on) and only when captions were burned.
    _emit_nocap = (
        bool(captions) and bool(words)
        and os.environ.get("VIDLORE_EMIT_NOCAP", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    vchain_nocap = vmap_nocap = None
    if _emit_nocap:
        _vfilters_nocap = [
            f for f in vfilters if not str(f).startswith("subtitles=")
        ]
        if len(_vfilters_nocap) != len(vfilters):     # a subtitle burn existed
            vchain_nocap, vmap_nocap = _compose_vchain(_vfilters_nocap)
        else:
            _emit_nocap = False

    # PRO DOC MIX. The voice is loudness-normalized ON ITS OWN (loudnorm's
    # ideal use — consistent spoken word, so it never pumps), THEN a soft
    # music bed + subtle SFX are mixed UNDER it at FIXED levels, and a
    # final true-peak limiter guards against clipping. Crucially there is
    # NO loudnorm across the whole mix: a single-pass loudnorm on
    # voice+bed rides the gain — it buried the music to silence on one
    # render and pumped a quiet mix into a flat, fatiguing wall on
    # another. Fixed ratios = the bed is always clearly there but soft,
    # and the voice is always broadcast-loud and dominant.
    # Generic mix: voice (always) + whatever optional beds exist, each at
    # its own fixed low level. Built dynamically so any combination of
    # music / boom-sfx / number-accent works without special-casing.
    # PHASE 3.1 — when there's a music bed, the voice is split into the
    # audible leg [n] AND a sidechain KEY [nkey] that ducks the music.
    # (asplit duplicates the SAME normalized voice; one copy is heard,
    # the other only drives the compressor.) Only the MUSIC ducks — the
    # boom / number / archival accents are meant to punch through, so
    # they stay un-ducked. If there's no music we keep the plain voice.
    duck_music = music_i is not None
    if duck_music:
        legs = [f"[1:a]{_VOXNORM},asplit=2[n][nkey]"]
    else:
        legs = [f"[1:a]{_VOXNORM}[n]"]
    mix = "[n]"
    if music_i is not None:
        # Music runs hotter (0.30) then ducks ~8-10 dB under the voice
        # and swells back in the pauses — the "score breathes" move.
        _co_d, _co_f = _look_cold_open()       # quiet niche cold-open
        _mco = ""
        if _co_f > 0:
            if _co_d > 0:
                _mco += f"adelay={int(_co_d * 1000)}:all=1,"
            _mco += f"afade=t=in:st={_co_d:.2f}:d={_co_f:.2f},"
        legs.append(
            f"[{music_i}:a]{_mco}volume="
            f"{_MUSIC_VOL_DUCK * _music_vol_mult() * _look_music_bed_mult() * _look_music_behavior():.4f}[mraw];"
            f"[mraw][nkey]{_DUCK}[m]")
        mix += "[m]"
    if sfx_i is not None:
        legs.append(f"[{sfx_i}:a]volume={_SFX_VOL}[s]")
        mix += "[s]"
    if gsfx_i is not None:
        legs.append(f"[{gsfx_i}:a]volume={_NUMSFX_VOL}[g]")
        mix += "[g]"
    if gfxrev_i is not None:
        legs.append(f"[{gfxrev_i}:a]volume={_GFX_VOL}[gr]")
        mix += "[gr]"
    if type_i is not None:
        legs.append(f"[{type_i}:a]volume={_TYPE_VOL}[tw]")
        mix += "[tw]"
    if arch_i is not None:
        legs.append(f"[{arch_i}:a]volume={_ARCH_VOL}[av]")
        mix += "[av]"
    if atmos_i is not None:
        # environmental world texture — un-ducked (already far under the
        # voice); just present enough to feel the world. Style mode scales
        # its presence (an epic carries a touch more world underneath).
        _atvol = _ATMOS_VOL * getattr(_sty, "atmos_scale", 1.0)
        legs.append(f"[{atmos_i}:a]volume={_atvol:.3f}[atm]")
        mix += "[atm]"
    # TIMESTAMPS FOLLOW SAMPLES. `asetpts=N/SR/TB` rebuilds the audio PTS from the running sample
    # count just before the encoder, so every AAC frame lands exactly one frame-length after the
    # last one. Without it the graph can hand the encoder a PTS jump, and the muxer faithfully
    # records it: measured on two delivered renders, one non-terminal frame declared 5488 samples
    # (Windows, 93 ms, mid-speech) and 3280 (macOS, 47 ms) instead of 1024. No samples are missing
    # — the container simply lies about its own timeline — but any consumer that honours timestamps
    # materialises the gap as silence, and it survives a stream copy all the way to an upload.
    # PROVEN on the pipeline, not in a lab: the exact trigger resisted synthetic reproduction (an
    # amix with a short bed and the same dropout_transition emits a clean timeline), so this was
    # first written as prevention aimed at the class. A/B on the SAME job and beats then settled
    # it — 1 anomaly of 47 ms at 66.347s before, 0 after. `_audio_frame_anomalies` remains the
    # backstop: it catches the defect whatever introduces it next.
    _APTS = "asetpts=N/SR/TB"
    if mix == "[n]":                       # voice only
        achain = f"[1:a]{_VOXNORM},{_LIMIT},{_APTS}[a]"
    else:
        k = mix.count("[")
        achain = ";".join(legs) + (
            f";{mix}amix=inputs={k}:duration=first:"
            f"normalize=0:dropout_transition=3,{_LIMIT},{_APTS}[a]"
        )
    amap = "[a]"

    # ═══════════════════════════════════════════════════════════════════
    # TWO-STAGE MUX (v10, 2026-05-27) — replaces the single-call ffmpeg
    # that combined a 200-stage video filter graph WITH a 5-leg audio
    # sidechain duck into ONE process. On long renders (25-min Mossad)
    # that single call stalled for 70+ minutes because ffmpeg coordinates
    # the whole filter graph through a single "graph_thread" — even with
    # filter_threads=N, the audio chain blocked the video chain and
    # vice-versa, leaving 9 of 10 cores idle.
    #
    # New flow:
    #   Stage V (video-only): full vchain (vignette+eq+noise+subtitles+
    #     drawtext stabs+PNG card overlays+act-break flashes) → encoded
    #     with the hw videotoolbox encoder (or libx264 fallback).
    #     Inputs: just the concat-segment mp4. No -i for any audio bed.
    #   Stage A (audio-only): full achain (loudnorm voice + sidechain-
    #     ducked music + atmos/archbed/sfx/numsfx/gfxsfx/typewriter beds
    #     + true-peak limiter) → encoded to AAC.
    #     Inputs: every audio file the achain references.
    #   Stage V + A run IN PARALLEL via two threads — the slower one
    #     (always Stage V for documentary work) sets wall-clock; Stage A
    #     finishes in seconds while Stage V is still encoding.
    #   Stage M (mux): -c:v copy -c:a copy container join, microseconds.
    #
    # NOTHING is skipped. Every overlay, every card, every drawtext,
    # every audio bed, the sidechain duck — all preserved verbatim.
    # The only behaviour change is concurrency.
    # ═══════════════════════════════════════════════════════════════════

    # All inputs after index 0 are AUDIO inputs (the voiceover plus any
    # music/sfx/atmos/archbed beds appended in fixed order above). The
    # video stage needs only the first input; the audio stage needs only
    # the rest. Audio-input indices in achain therefore shift down by 1.
    v_inputs: list[str] = inputs[:2]               # ["-i", "video_only.mp4"]
    a_inputs: list[str] = inputs[2:]               # voiceover + all beds

    def _shift_audio_idx(chain: str, delta: int) -> str:
        """Re-number every `[N:a]` reference in `chain` by `delta`."""
        if not chain or delta == 0:
            return chain
        return re.sub(
            r"\[(\d+):a\]",
            lambda m: f"[{int(m.group(1)) + delta}:a]",
            chain,
        )

    # Voiceover was input index 1 in the combined graph; becomes index 0
    # when we drop the video input. Shift every audio ref accordingly.
    a_achain = _shift_audio_idx(achain, -1)

    _tmp_video = workdir / "_stage_v.mp4"
    _tmp_video_nocap = workdir / "_stage_v_nocap.mp4"   # LIVE_CAPTION base
    _tmp_audio = workdir / "_stage_a.m4a"
    _final_tmp = workdir / f"final_{out_path.stem[:40]}.mp4"
    for _t in (_tmp_video, _tmp_video_nocap, _tmp_audio, _final_tmp):
        try:
            _t.unlink()
        except FileNotFoundError:
            pass

    # ── Stage V: video-only pass ─────────────────────────────────────
    # The video graph is the bulk of the wall-clock cost. It runs with
    # the parallel filter flags and the chosen video encoder.
    def _build_v_args(venc_args: list[str], _vchain: str = vchain,
                      _vmap: str = vmap, _out: Path = _tmp_video) -> list[str]:
        a = _ffthreads() + list(v_inputs)
        if _vchain:
            a += ["-filter_complex", _vchain]
        a += [
            "-map", _vmap, "-an",
            "-r", str(FPS),
            *venc_args,
            "-movflags", "+faststart",
            _out.name,
        ]
        return a

    # ── Stage A: audio-only pass ─────────────────────────────────────
    # Audio is cheap (a handful of resamples + a sidechain compressor +
    # an amix + a limiter), but doing it independently lets the video
    # stage saturate the encoder without sharing the graph thread.
    def _build_a_args() -> list[str]:
        a = list(a_inputs)
        if a_achain:
            a += ["-filter_complex", a_achain]
            a += ["-map", "[a]"]
        else:
            # Voice-only edge case (no beds + no loudnorm chain). Just
            # take input 0's audio stream straight.
            a += ["-map", "0:a"]
        a += [
            "-vn",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000",
            _tmp_audio.name,
        ]
        return a

    # Run V and A in parallel. We catch each thread's exception so the
    # caller sees the real error (not a meaningless Thread death). The
    # video stage retains the hw→libx264 fallback ladder from the
    # single-call version (h264_videotoolbox sometimes -22's on deep
    # filter graphs even after the split — software encode always works).
    import threading as _th
    import time as _time_mux

    _err: dict[str, BaseException | None] = {"v": None, "a": None}
    _stage_times: dict[str, float] = {"v": 0.0, "a": 0.0, "m": 0.0}

    def _run_v() -> None:
        _t0 = _time_mux.time()
        try:
            run(_build_v_args([*_venc("20")]), cwd=str(workdir))
        except Exception as _e:                            # noqa: BLE001
            print(
                f"  [5/5] stage-V hw encoder failed ({str(_e)[:80]}); "
                "falling back to libx264 software encoder…",
                flush=True,
            )
            _tmp_video.unlink(missing_ok=True)
            try:
                run(_build_v_args([
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                ]), cwd=str(workdir))
            except BaseException as _e2:                   # noqa: BLE001
                _err["v"] = _e2
        _stage_times["v"] = _time_mux.time() - _t0

    def _run_a() -> None:
        _t0 = _time_mux.time()
        try:
            run(_build_a_args(), cwd=str(workdir))
        except BaseException as _e:                        # noqa: BLE001
            _err["a"] = _e
        _stage_times["a"] = _time_mux.time() - _t0

    _t_parallel = _time_mux.time()
    t_v = _th.Thread(target=_run_v, name="stage-V")
    t_a = _th.Thread(target=_run_a, name="stage-A")
    t_v.start()
    t_a.start()
    t_v.join()
    t_a.join()
    _parallel_wall = _time_mux.time() - _t_parallel

    if _err["v"] is not None:
        raise _err["v"]                                    # type: ignore[misc]
    if _err["a"] is not None:
        raise _err["a"]                                    # type: ignore[misc]

    # ── Stage V (no-caption base, LIVE_CAPTION) ──────────────────────
    # Runs AFTER the captioned V/A so the main encode path is byte-for-byte
    # unperturbed (same reliability, same hw encoder, no contention). It is
    # the SAME graph minus the burned `subtitles=` filter, over the SAME
    # baked video → footage-matched. Best-effort: a failure here NEVER
    # affects the captioned export (we just skip the editor base).
    _err_nocap: BaseException | None = None
    if _emit_nocap and vchain_nocap is not None:
        try:
            run(_build_v_args([*_venc("20")], _vchain=vchain_nocap,
                              _vmap=vmap_nocap or "0:v",
                              _out=_tmp_video_nocap), cwd=str(workdir))
        except Exception as _e:                            # noqa: BLE001
            print(f"  [5/5] stage-V-nocap hw encoder failed "
                  f"({str(_e)[:70]}); libx264 retry…", flush=True)
            _tmp_video_nocap.unlink(missing_ok=True)
            try:
                run(_build_v_args([
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p",
                ], _vchain=vchain_nocap, _vmap=vmap_nocap or "0:v",
                    _out=_tmp_video_nocap), cwd=str(workdir))
            except BaseException as _e2:                   # noqa: BLE001
                _err_nocap = _e2

    # ── Stage M: container-only mux (microseconds) ───────────────────
    # No re-encode. Just join the two finished streams + faststart so the
    # player can start before the file fully downloads.
    _t_m = _time_mux.time()
    run([
        "-i", _tmp_video.name,
        "-i", _tmp_audio.name,
        "-c:v", "copy", "-c:a", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        _final_tmp.name,
    ], cwd=str(workdir))
    _stage_times["m"] = _time_mux.time() - _t_m

    # Per-stage timing breakdown — exposes whether the bake/V/A split
    # actually helped on this render.
    print(
        f"  [5/5] mux done: "
        f"stage-V={_stage_times['v']:.1f}s · "
        f"stage-A={_stage_times['a']:.1f}s · "
        f"stage-M={_stage_times['m']:.2f}s · "
        f"parallel_wall={_parallel_wall:.1f}s "
        f"(V+A; bottleneck = {'V' if _stage_times['v'] >= _stage_times['a'] else 'A'})",
        flush=True,
    )

    # Drop the captioned video temp now; KEEP _tmp_audio until the
    # no-caption base mux below has reused it.
    _tmp_video.unlink(missing_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(str(_final_tmp), str(out_path.resolve()))
    except OSError:
        # cross-device or a locked destination: fall back to copy+swap
        shutil.copyfile(str(_final_tmp), str(out_path.resolve()) + ".tmp")
        os.replace(str(out_path.resolve()) + ".tmp", str(out_path.resolve()))
        _final_tmp.unlink(missing_ok=True)

    # ── No-caption editor base (LIVE_CAPTION) — mux the caption-free Stage-V
    # with the SAME audio, written AFTER the captioned MP4 so its mtime is
    # >= the project MP4 (the editor's `_nocap_fresh` rule). Best-effort:
    # the captioned export is already finalized on disk above, so any
    # failure here only skips the editor's instant-toggle base.
    if (_emit_nocap and vchain_nocap is not None and _err_nocap is None
            and _tmp_video_nocap.exists() and _tmp_audio.exists()):
        try:
            _nocap_tmp = workdir / "_nocap_mux.mp4"
            _nocap_tmp.unlink(missing_ok=True)
            run([
                "-i", _tmp_video_nocap.name,
                "-i", _tmp_audio.name,
                "-c:v", "copy", "-c:a", "copy",
                "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-movflags", "+faststart",
                _nocap_tmp.name,
            ], cwd=str(workdir))
            _nocap_dst = out_path.parent / "editor_cache" / "preview_nocap.mp4"
            _nocap_dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(_nocap_tmp), str(_nocap_dst.resolve()))
            print("  [5/5] no-caption editor base → editor_cache/"
                  f"{_nocap_dst.name} (footage-matched; live caption toggle)",
                  flush=True)
        except Exception as _e:                                # noqa: BLE001
            print(f"  [5/5] no-caption base skipped ({str(_e)[:90]})",
                  flush=True)
    elif _emit_nocap and _err_nocap is not None:
        print("  [5/5] no-caption base skipped (stage-V-nocap encode "
              f"failed: {str(_err_nocap)[:80]})", flush=True)

    # Drop the shared audio + the no-caption video temp now that both
    # muxes have consumed them.
    _tmp_audio.unlink(missing_ok=True)
    _tmp_video_nocap.unlink(missing_ok=True)

    # The earlier timeline check runs before overlay bake, black-frame repair,
    # caption burn and mux.  Recheck the actual assembled artifact after every
    # one of those mutating passes; only this stage may call the assembly audit
    # passed.  ClipStudio performs one more identical check after its own
    # letterbox/breakout/final-QA post-passes.
    if _lineage_enabled:
        _verify_lineage_delivered_output(
            out_path, _lineage_audit_path, stage="assembled_output")
        print(f"  [5/5] scene-lineage delivered canary PASS → {out_path.name}",
              flush=True)

    # MNT_1 — render_meta.json sidecar: the pipeline's OWN authoritative
    # pacing/beat numbers, written beside the final MP4 so the benchmark scores
    # pacing from GROUND TRUTH (real beats) instead of guessing via ffmpeg
    # scene-detection (which under-counts camera-move beats and over-softens
    # crossfades). Best-effort — a sidecar failure never breaks the render.
    try:
        import json as _json
        import statistics as _stats
        _bd = list(beat_durs)
        try:
            _trans = int(len(trans_tails))
        except Exception:                                      # noqa: BLE001
            _trans = 0
        try:                                       # DOC_012 editor signature
            from .look_dna import last_look_decision as _lld
            _ed_sig = _lld()
        except Exception:                                      # noqa: BLE001
            _ed_sig = {}
        try:                                       # editorial RECIPE (Layer 2)
            from .look_dna import last_look_recipe as _llr
            _recipe = _llr() or {}
        except Exception:                                      # noqa: BLE001
            _recipe = {}
        _recipe_summary = ""
        if _recipe:
            _tp = _recipe.get("transition_palette") or []
            _recipe_summary = " · ".join(str(x) for x in [
                _recipe.get("niche"),
                "accent " + ",".join(str(c) for c in (_recipe.get("accent") or [])),
                "map " + str(_recipe.get("map_style")),
                "beat " + str(_recipe.get("beat_target")),
                (_tp[0] if _tp else "") + " transitions",
                _recipe.get("lower_third_family") or "",
            ] if x)
        try:                                   # P2 — accurate per-scene timing
            _sd = [round(float(d), 3) for d in durs]   # real per-scene durations
            _ss, _acc = [], 0.0
            for _d in _sd:
                _ss.append(round(_acc, 3))
                _acc += _d
        except Exception:                                      # noqa: BLE001
            _sd, _ss = [], []
        _meta = {
            "schema": "render_meta/1",
            "editor_signature": _ed_sig,
            "editorial_recipe": _recipe,
            "editorial_recipe_summary": _recipe_summary,
            "scenes": int(n_sc),
            "beats": int(len(_bd)),
            "cuts": int(max(0, len(_bd) - 1)),
            "transitions_motivated": _trans,
            "fps": int(FPS),
            "video_seconds": round(sum(_bd), 2),
            "shot_len_s": {
                "min": round(min(_bd), 2) if _bd else 0.0,
                "max": round(max(_bd), 2) if _bd else 0.0,
                "avg": round(sum(_bd) / len(_bd), 2) if _bd else 0.0,
                "median": round(_stats.median(_bd), 2) if _bd else 0.0,
            },
            # P2 — exact per-scene start/duration (ground truth for editor sync).
            "scene_starts": _ss,
            "scene_durations": _sd,
        }
        (out_path.parent / "render_meta.json").write_text(
            _json.dumps(_meta, indent=2), encoding="utf-8")
        print(f"  [meta] render_meta.json: {_meta['beats']} beats / "
              f"{_meta['scenes']} scenes / {_meta['video_seconds']:.0f}s "
              f"(avg shot {_meta['shot_len_s']['avg']}s)", flush=True)
    except Exception as _e:                                    # noqa: BLE001
        print(f"  [meta] render_meta.json skipped ({_e})", flush=True)

    # FINAL-EXPORT METRICS — authoritative fingerprint of the DELIVERED MP4
    # (post-mux: video + all audio). Records sha256 + probed duration + fps so
    # the final output can never be confused with the shorter pre-mux
    # intermediate measured by render_black_frame_metrics.json. Best-effort.
    try:
        _xm = _write_export_metrics(out_path, FPS)
        print(f"  [meta] render_export_metrics.json: {out_path.name} · "
              f"dur={_xm.get('duration_s')}s · "
              f"sha256={(_xm.get('sha256') or '')[:12]}…", flush=True)
    except Exception as _e:                                    # noqa: BLE001
        print(f"  [meta] render_export_metrics.json skipped ({_e})", flush=True)

    # SFX CUE SHEET (Phase 4) — write the restraint-applied SFX timeline beside
    # the MP4 for QA (audio_quality_audit) + post-production review. Best-effort.
    try:
        _sde = locals().get("_sd_events") or []
        if _sde:
            from .audio_director import sfx_director as _sdir
            _total_s = round(sum(beat_durs), 2) if beat_durs else 0.0
            _sheet = _sdir.build_cue_sheet(
                _sde, _total_s, niche=locals().get("_sd_niche", ""))
            _sdir.write_cue_sheet(_sheet, out_path.parent / "sfx_cue_sheet.json")
            print(f"  [meta] sfx_cue_sheet.json: {_sheet['total_events']} cues "
                  f"({_sheet['whoosh_per_min']}/min whoosh)", flush=True)
    except Exception as _e:                                    # noqa: BLE001
        print(f"  [meta] sfx_cue_sheet.json skipped ({_e})", flush=True)

    # v13.1 (2026-05-28): drop large intermediates now that the final
    # mp4 is on disk. video_only (1-3 GB on long renders) +
    # video_baked (1-3 GB) + _stage_v + _stage_a are all throwaway
    # once the final exists. Kept work_*/footage and the small per-card
    # PNGs so a re-render can reuse them via cache.
    for _drop_name in ("video_only.mp4", "video_baked.mp4",
                       "video_repaired.mp4",
                       "_stage_v.mp4", "_stage_v_nocap.mp4",
                       "_nocap_mux.mp4", "_stage_a.m4a",
                       "_baked_concat.txt", "_blk_concat.txt"):
        try:
            (workdir / _drop_name).unlink(missing_ok=True)
        except Exception:                                  # noqa: BLE001
            pass

    return out_path


if __name__ == "__main__":
    # ---- local unit tests for the floating_stat figure extraction ----
    # Run from the repo root with:  python -m vidlore.assemble
    def _t(got, want, label):
        mark = "ok " if got == want else "FAIL"
        print("  [" + mark + "] " + label + ": got " + repr(got)
              + " want " + repr(want))
        assert got == want, label + ": got " + repr(got) + ", want " + repr(want)

    print("floating_stat figure tests:")
    # the bug: a literal digit before a magnitude word must scale the digit,
    # not drop it and let the bare magword read as 1,000,000.
    _t(_best_stat_figure("the video reached 50 million views"),
       "50,000,000", "digit plus magword 50 million")
    _t(_best_stat_figure("50 million views"), "50,000,000", "50 million views")
    _t(_best_stat_figure("1.5 billion users"),
       "1,500,000,000", "decimal digit plus magword 1.5 billion")
    # magnitude-gated: bare years / ordinals / small counts never match.
    _t(_best_stat_figure("in 1949"), "", "bare year in 1949")
    _t(_best_stat_figure("three judges"), "", "spelled small count three judges")
    _t(_best_stat_figure("the 50 states"), "", "bare digit no magword")
    # spelled magnitudes still work (no regression to _spelled_to_number).
    _t(_best_stat_figure("four hundred and twenty million"),
       "420,000,000", "spelled 420 million")
    _t(_best_stat_figure("1,200,000 people"), "1,200,000", "comma-grouped digits")
    # money count-up path must NOT regress (_money_figure / _compact_money).
    _t(_money_figure("$420 million"), "$420M", "money 420 million")
    _t(_money_figure("$1.2 billion budget"), "$1,200M", "money 1.2 billion")
    _t(_compact_money(420_000_000, "$"), "$420M", "compact_money 420M")
    print("ALL FLOATING_STAT TESTS PASSED")
