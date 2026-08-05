"""Stage 5 — cut.

Extract each selected [in,out] to its own trimmed .mp4 under <project>/clips/. The clips are
VIDEO-ONLY (the renderer lays continuous TTS narration over the timeline, matching the sample's
wall-to-wall VO; the source movie audio would only conflict). Because each clip starts at its
own frame 0, the existing renderer plays it as-is with no in/out seek — so assemble.py needs
no modification (DESIGN.md §3).
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .models import ClipProject, ClipSelection
from .config import ClipConfig, ffmpeg_exe

# ---------------------------------------------------------------- legibility grade
# Half of this show is shot at night. Measured on job 6a26707939, over the 501 frames the renderer
# would actually have shown: median beat mean-luma 31.2, a QUARTER of all beats under 20, and the
# p90 luma (the brightest tenth of pixels) sitting at 63 in the median beat. A frame-by-frame vision
# audit of the same render flagged 44 beats too_dark and put 22 of them in the <=4/10 bucket with
# notes like "an essentially empty dark rectangle with a speck in it".
#
# REJECTING dark footage has already been tried and measured and does not work: the pool median is
# what it is, so a luma floor evicts correct, canonical, perfectly-intentional night scenes and the
# beat falls back to generic filler that scores WORSE. What the pipeline does today is worse still —
# a near-black cut clip is freeze-REPLACED with a neighbouring beat's frame, so the viewer sees the
# wrong content rather than dim correct content.
#
# So: grade it, do not judge it. This is presentation-only. It runs after the window is chosen, it
# changes no ranking, no verdict and no selection, and it cannot move a beat to different footage.
# Because build's own near-black sweep probes the CUT CLIP, a beat that becomes legible here also
# stops being freeze-replaced — which is a relevance gain paid for with no relevance risk.
#
# The trigger reuses build's measured legibility score, max(YAVG, spread): brightness alone rejects
# candle-lit scenes wholesale, while genuine dead frames fail BOTH mean and contrast. Gamma is the
# right instrument because its slope is steepest in the shadows, so it opens up crushed blacks
# instead of flattening the whole picture, and it cannot clip highlights.
_GRADE_FLOOR = 40.0        # max(YAVG, spread) at or above this is already legible — leave it alone
_GRADE_TARGET = 46.0       # what a graded frame's mean luma aims for; ~ the low end of "readable"
_GRADE_MAX_GAMMA = 1.75    # hard cap: beyond this, night footage turns to grey noise


def _luma_stats(src_path, start: float, dur: float) -> Optional[tuple]:
    """(mean luma, 10th->90th percentile spread) over the window, or None if unprobeable."""
    try:
        p = subprocess.run(
            [ffmpeg_exe(), "-y", "-hide_banner", "-nostats",
             "-ss", f"{max(0.0, float(start)):.3f}", "-i", str(src_path),
             "-t", f"{max(0.5, float(dur)):.3f}", "-an",
             "-vf", "scale=128:72,signalstats,"
                    "metadata=print:key=lavfi.signalstats.YAVG:file=-:direct=1,"
                    "metadata=print:key=lavfi.signalstats.YLOW:file=-:direct=1,"
                    "metadata=print:key=lavfi.signalstats.YHIGH:file=-:direct=1",
             "-f", "null", "-"],
            capture_output=True, timeout=60)
        blob = (p.stdout or b"").decode("utf-8", "ignore") \
            + (p.stderr or b"").decode("utf-8", "ignore")
        yavg = [float(m) for m in re.findall(r"YAVG=([0-9.]+)", blob)]
        if not yavg:
            return None
        ylow = [float(m) for m in re.findall(r"YLOW=([0-9.]+)", blob)]
        yhigh = [float(m) for m in re.findall(r"YHIGH=([0-9.]+)", blob)]
        spread = 0.0
        if ylow and yhigh and len(ylow) == len(yhigh):
            spread = sum(h - l for h, l in zip(yhigh, ylow)) / len(yhigh)
        return sum(yavg) / len(yavg), spread
    except Exception:                                  # noqa: BLE001 — a grade must never fail a cut
        return None


def legibility_gamma(yavg: float, spread: float) -> float:
    """The gamma this window needs to become readable, or 1.0 for 'leave it alone'.

    Pure arithmetic so it can be tested without ffmpeg. Gamma g maps a normalised level v to
    v**(1/g), so to lift a mean of `yavg` to `_GRADE_TARGET` we want
    (yavg/255)**(1/g) == target/255, i.e. g == ln(yavg/255) / ln(target/255)."""
    if max(yavg, spread) >= _GRADE_FLOOR:
        return 1.0
    v = max(1.0, float(yavg)) / 255.0
    t = _GRADE_TARGET / 255.0
    if v >= t:                                         # dark by contrast, not by level — gamma is
        return 1.0                                     # the wrong tool, and lifting would grey it
    g = math.log(v) / math.log(t)
    return round(min(_GRADE_MAX_GAMMA, max(1.0, g)), 3)


def legibility_filter(src_path, start: float, dur: float) -> tuple:
    """(ffmpeg filter string or "", note for the log). Empty string = no grade applied."""
    if os.environ.get("VIDLORE_CLIPSTUDIO_LEGIBILITY_GRADE", "1").strip() in ("0", "false", "no"):
        return "", ""
    st = _luma_stats(src_path, start, dur)
    if st is None:
        return "", ""                                  # unknown is never "dark"
    yavg, spread = st
    g = legibility_gamma(yavg, spread)
    if g <= 1.0:
        return "", ""
    # a touch of contrast with the gamma: gamma alone on a collapsed histogram reads flat/milky
    return (f"eq=gamma={g}:contrast=1.06",
            f"legibility grade gamma={g} (YAVG {yavg:.1f}, spread {spread:.1f})")


# FRAME-EXACT HALF-OPEN CUTTING — the selection owns [in, out), never the frame AT `out`.
#
# `-t <dur>` is a DURATION bound, and ffmpeg resolves it against decoded frame timestamps with
# rounding, so a cut can legitimately end up holding one frame from the shot that starts at `out`.
# Measured on job ee93371e41 scene 170: the cut carried 49 frames — 48 Varys frames plus the first
# frame of the following Oberyn shot. Downstream, `_fit_verified_selection_clip` clones the final
# decoded frame to fill a longer narration beat, so that single foreign frame became 126 of 174
# output frames (72.4%, ~4.2s) of the wrong character.
#
# The blanket answer was to stop cloning at 88% of the clip and hold that frame instead. It stops
# the leak and costs real motion on every padded beat: 48.30s of verified footage on that job.
#
# The right answer is to make the contaminating frame not exist. `select='lt(t,dur)'` filters on
# each frame's own decoded timestamp, which is exactly the half-open test, and it is indifferent to
# frame rate, VFR, container time base and seek behaviour — no fps arithmetic to round wrong.
# Clips carrying this marker are certified frame-exact, so the fitter may hold their true final
# frame; anything without it keeps the conservative behaviour.
_CUT_CONTRACT = "halfopen_v1"


def cut_selection(proj: ClipProject, sel: ClipSelection, cfg: ClipConfig,
                  *, resume: bool = False) -> Optional[Path]:
    """Trim one selection to clips/seg_NNN.mp4. Returns the path or None."""
    if not sel.source_id:
        return None
    src = proj.source(sel.source_id)
    if src is None or not src.local_path:
        return None
    # RESUME: this selection was already cut on a prior run — reuse the clip instead of re-encoding.
    # The output path is deterministic (keyed on segment_index) and the selection is unchanged (the
    # resume checkpoint only skips cut when match's signature still matches), so the existing clip IS
    # this selection's clip. A zero-byte/absent file falls through to a normal re-encode.
    # A clip cut before the half-open contract may still hold the frame at `out`, and the fitter
    # trusts the marker to decide whether it may clone the final frame. Re-cut rather than inherit
    # an uncertified clip — the alternative is a silent 72%-of-the-beat freeze on the wrong shot.
    if resume:
        _existing = proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        if _existing.exists() and _existing.stat().st_size > 0 \
                and getattr(sel, "cut_contract", "") == _CUT_CONTRACT:
            return _existing
    in_p = max(0.0, sel.in_point)
    dur = max(cfg.min_clip_sec, sel.out_point - sel.in_point)
    if src.duration:
        in_p = min(in_p, max(0.0, src.duration - cfg.min_clip_sec))
        dur = min(dur, max(cfg.min_clip_sec, src.duration - in_p))
    # CUT-WINDOW FLAG VALIDATION (safety net): padding a short window to min_clip_sec can bleed
    # into an adjacent shot carrying burned subs / a logo / murk (observed: a 1.4s clean shot
    # padded straight into the next shot's Turkish subtitle). If the PADDED window is dirty but
    # the selection's own window is clean and long enough, don't pad into the dirt.
    # EXACT-MOMENT protection is inherent here: this path only ever REFUSES the pad and keeps
    # the selection's own [in, out] — it never relocates the aired moment, so no policy split.
    import os as _os_w
    if _os_w.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() not in ("0", "false", "no") \
            and dur > (sel.out_point - sel.in_point) + 1e-3:
        try:
            from . import index as _index
            from .match import clean_cut_window
            _shots = _index.load_shots(proj, sel.source_id)
            _own = max(0.0, sel.out_point - sel.in_point)
            _, _, _act, _why = clean_cut_window(_shots, in_p, in_p + dur, cfg.min_clip_sec,
                                                anchor=(in_p, in_p + max(0.1, _own)))
            if _act != "ok" and _own >= 1.0:
                _, _, _act2, _ = clean_cut_window(_shots, in_p, in_p + _own, 0.0,
                                                  anchor=(in_p, in_p + _own))
                if _act2 == "ok":
                    dur = _own                     # own window is clean — don't pad into dirt
        except Exception:
            pass                                   # QC must never break the cut itself
    out = proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    # LEGIBILITY GRADE — presentation only, applied to the window already chosen (see the note at
    # the top of this module). Recorded on the selection so the audit can see which beats were lit.
    _vf, _note = legibility_filter(src.local_path, in_p, dur)
    try:
        sel.legibility_grade = _note
    except Exception:                                  # noqa: BLE001 — an older model has no slot
        pass
    # `-ss` before `-i` makes the first output frame the one at `in_p`, so a frame's own `t` is its
    # offset INTO the selection — `lt(t, dur)` is therefore the exact half-open [in, out) test.
    # It runs ahead of any grade so the dropped frame is never graded, encoded or measured.
    # `-t` stays as a coarse decode bound; `fps_mode=passthrough` stops vsync re-manufacturing the
    # frame the select just removed.
    # THE THRESHOLD MUST BE IN SOURCE TIME. `-ss` lands on the first frame at or after the request,
    # so a seek-relative `t` is measured from the LANDING, not from `in_p`. Measured on scene 170:
    # in_p 87.879 is not a frame boundary, the decoder lands ~21ms late, and a seek-relative bound
    # therefore reaches ~21ms PAST `out` — straight back onto the first frame of the next shot.
    # `-copyts` keeps each frame's own SOURCE timestamp, so `lt(t, out)` is the authorized window
    # stated in the only clock that cannot drift. `setpts=PTS-STARTPTS` then rebases for output.
    #
    # TRUNCATE, never round, when writing it: "%.6f" % 1.4666666 is 1.466667 — a threshold LARGER
    # than the authorized out-time, which re-admits the very frame this filter exists to remove.
    # Flooring to nanoseconds keeps the bound at or below `out`, and 1ns sits orders of magnitude
    # below any real frame spacing, so no in-window frame is ever lost to it.
    _out_abs = math.floor(max(0.0, in_p + dur) * 1e9) / 1e9
    _chain = [f"select='lt(t\\,{_out_abs:.9f})'", "setpts=PTS-STARTPTS"]
    if _vf:
        _chain.append(_vf)
    cmd = [
        ffmpeg_exe(), "-y",
        "-copyts", "-ss", f"{in_p:.3f}", "-i", str(src.local_path),
        "-t", f"{dur:.3f}", "-an", "-fps_mode", "passthrough",
        "-vf", ",".join(_chain),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return None
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0:
        try:
            sel.cut_contract = _CUT_CONTRACT     # certifies [in, out) — the fitter reads this
        except Exception:                        # noqa: BLE001 — an older model has no slot
            pass
        return out
    return None


def cut_all(proj: ClipProject, cfg: ClipConfig, *, workers: Optional[int] = None,
            resume: bool = False, progress=None) -> int:
    """Cut every selection that has a source. Sets sel.clip_path in place. Returns count.

    `workers` defaults to cfg.cut_workers (auto-scaled to the machine's cores) — each trim is an
    independent libx264 subprocess, so more cores = proportionally faster cutting.

    resume=True reuses any already-cut clips/seg_NNN.mp4 (a partial cut stage that died halfway
    finishes the remaining clips instead of re-encoding all of them).
    """
    proj.clips_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in proj.selections if s.source_id]
    done = 0
    w = workers if (workers and workers > 0) else int(getattr(cfg, "cut_workers", 4) or 4)

    def _one(sel: ClipSelection):
        p = cut_selection(proj, sel, cfg, resume=resume)
        return sel, p

    with cf.ThreadPoolExecutor(max_workers=max(1, w)) as ex:
        for sel, p in ex.map(_one, todo):
            if p is not None:
                sel.clip_path = str(p)
                done += 1
            if progress and done % 15 == 0:
                progress(f"cut: {done}/{len(todo)}")
    proj.save()
    if progress:
        progress(f"cut: done {done}/{len(todo)} clips")
    return done
