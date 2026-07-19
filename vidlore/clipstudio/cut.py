"""Stage 5 — cut.

Extract each selected [in,out] to its own trimmed .mp4 under <project>/clips/. The clips are
VIDEO-ONLY (the renderer lays continuous TTS narration over the timeline, matching the sample's
wall-to-wall VO; the source movie audio would only conflict). Because each clip starts at its
own frame 0, the existing renderer plays it as-is with no in/out seek — so assemble.py needs
no modification (DESIGN.md §3).
"""
from __future__ import annotations

import concurrent.futures as cf
import subprocess
from pathlib import Path
from typing import Optional

from .models import ClipProject, ClipSelection
from .config import ClipConfig, ffmpeg_exe


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
    if resume:
        _existing = proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        if _existing.exists() and _existing.stat().st_size > 0:
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
    cmd = [
        ffmpeg_exe(), "-y",
        "-ss", f"{in_p:.3f}", "-i", str(src.local_path),
        "-t", f"{dur:.3f}", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return None
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0:
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
