#!/usr/bin/env python3
"""Commenter-avatar badge gate + caption-dodge repair — regression proof.

The 'Jacquelyn Sutphen' leak: a comment-overlay badge (avatar + personal name, cyan border,
bottom-left) aired on 2 beats and forced a 3.3s caption dropout. It defeated every text rule at
once — the name matches no _OCR_JUNK keyword, 2 mixed-case words stay under _ocr_text_heavy's
floors, and the source-level pixel corner detector needs ≥25% shot presence (intermittent
overlays never qualify). These tests pin the three-layer fix:
  1. per-shot `overlay-badge` dirtiness (name-like OCR + dense corner mask) in window-QC + pool
  2. caption-dodge REPAIR-first (corner-localized text → crop, keep caption; suppress = last
     resort and logged per-beat)
  3. duration-scaled flag sampling (a 51s shot no longer has ~17s OCR-blind stretches)
"""
import subprocess
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.match import (                          # noqa: E402
    _shot_overlay_badge, _shot_dirty_reason, clean_cut_window)
from vidlore.clipstudio.build import (                          # noqa: E402
    _clip_text_corner, _crop_clip_corner, _clip_has_burned_text)
from vidlore.clipstudio.index import _sample_times              # noqa: E402
from vidlore.clipstudio.config import ffmpeg_exe                # noqa: E402

FF = ffmpeg_exe()
PASS = FAIL = 0

# REAL persisted flags from the leaking source's index (why_did_they_cut_off_j_bdee6ae9 shot 8):
# the badge's border/avatar/text edges make a DENSE bottom-left mask.
DENSE_BL = "300c0381801ff00c000000600000037fffff9bfffffcdeffbfe60000003ffffff9ffffffc0000000"
SPARSE_BR = "00000000000000000000000000000000000000000000000000000000200000008000000600000010"


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def _sh(idx, start, end, ocr="", masks=None, subs=0, luma_avg=120.0, luma_hi=200.0):
    return types.SimpleNamespace(index=idx, start=start, end=end, ocr_text=ocr,
                                 corner_masks=masks or {}, subs_flag=subs,
                                 luma_avg=luma_avg, luma_hi=luma_hi, text_conf=0.0,
                                 keyframe_path="", quality=0.8)


class PixelOCR:
    """Deterministic OCR stand-in: 'reads' the badge name iff the cyan badge pixels exist in
    the frame — so a cropped clip genuinely reads clean (a text-always mock could not prove
    the repair removed anything)."""
    def __call__(self, img_path):
        from PIL import Image
        import numpy as np
        a = np.asarray(Image.open(img_path).convert("RGB"), dtype=int)
        mask = (a[:, :, 0] < 80) & (a[:, :, 1] > 170) & (a[:, :, 2] > 170)
        if mask.sum() < 200:
            return ([], 0.0)
        ys, xs = mask.nonzero()
        box = [[xs.min(), ys.min()], [xs.max(), ys.min()],
               [xs.max(), ys.max()], [xs.min(), ys.max()]]
        return ([(box, "Jacquelyn Sutphen", 0.92)], 0.0)


class SubsOCR:
    """Frame-wide bottom strip (dialogue subtitle) — must NOT be treated as corner-croppable."""
    def __call__(self, img_path):
        from PIL import Image
        W, H = Image.open(img_path).size
        box = [[int(W*0.1), int(H*0.86)], [int(W*0.9), int(H*0.86)],
               [int(W*0.9), int(H*0.95)], [int(W*0.1), int(H*0.95)]]
        return ([(box, "you know nothing of debts", 0.9)], 0.0)


def main():
    # (1) _shot_overlay_badge — calibrated rule (4/1957 shots on the real project, 0 FP)
    badge = _sh(8, 19.9, 21.9, ocr="Jacquelyn Sutphen", masks={"bl": DENSE_BL, "br": SPARSE_BR})
    _say(_shot_overlay_badge(badge), "name-like OCR + dense corner mask = badge")
    _say(_shot_dirty_reason(badge) == "overlay-badge", "dirty reason is 'overlay-badge'")
    _say(not _shot_overlay_badge(_sh(1, 0, 2, ocr="Jacquelyn Sutphen")),
         "no corner masks (old index) → fail-open, not flagged")
    _say(not _shot_overlay_badge(_sh(2, 0, 2, ocr="Jacquelyn Sutphen",
                                     masks={"br": SPARSE_BR})),
         "sparse corner mask only → not a badge")
    _say(not _shot_overlay_badge(_sh(3, 0, 2, ocr="winterfell gate", masks={"bl": DENSE_BL})),
         "non-name-like text → not a badge (lowercase scene text)")
    _say(not _shot_overlay_badge(_sh(4, 0, 2, ocr="Stannis Tywin Robb Stark",
                                     masks={"bl": DENSE_BL})),
         "3+ words stays _ocr_text_heavy's call, not badge")
    _say(_shot_dirty_reason(_sh(5, 0, 2, ocr="Jacquelyn Sutphen",
                                masks={"bl": DENSE_BL}, subs=1)) == "subs",
         "subs precedence unchanged (subs_flag beats badge)")

    # (2) window-QC avoids the badge span — the REAL leaked window shape: badge on the first
    #     ~4s of a 9.2s window, clean footage after (this is beats 171/174's exact geometry)
    shots = [_sh(8, 19.9, 21.9, ocr="Jacquelyn Sutphen", masks={"bl": DENSE_BL}),
             _sh(9, 21.9, 24.7, ocr="Jacquelyn Sutphen", masks={"bl": DENSE_BL}),
             _sh(10, 24.7, 26.0), _sh(11, 26.0, 31.1)]
    nt0, nt1, act, why = clean_cut_window(shots, 20.7, 29.9, 4.0)
    _say(act == "shortened" and nt0 >= 24.7 and "overlay-badge" in why,
         f"window over the badge is SHORTENED past it ([{nt0:.1f},{nt1:.1f}] {why})")

    # (3) caption-dodge repair chain on a real synthetic clip: cyan-border badge bottom-left
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "badge.mp4"
        vf = ("drawbox=x=20:y=ih-160:w=170:h=140:color=cyan:t=6,"
              "drawbox=x=26:y=ih-154:w=158:h=110:color=white:t=fill,"
              "drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:text='Jacquelyn Sutphen':"
              "fontsize=16:fontcolor=black:x=28:y=h-40")
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3", "-vf", vf,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True,
                       capture_output=True)
        ocr = PixelOCR()
        _say(_clip_has_burned_text(clip, ocr), "badge clip reads as burned text pre-repair")
        corner = _clip_text_corner(clip, ocr)
        _say(corner == "bl", f"text corner located ({corner!r})")
        _say(_crop_clip_corner(clip, corner) and not _clip_has_burned_text(clip, ocr),
             "corner-crop REPAIR removes the badge (post-crop clip reads clean)")
        # duration neutrality: repair must not change clip length (timeline conform depends on it)
        p = subprocess.run([FF, "-i", str(clip)], capture_output=True, text=True)
        _say("00:00:0" in (p.stderr or "") and "Duration: 00:00:02.9" in (p.stderr or "")
             or "Duration: 00:00:03" in (p.stderr or ""),
             "cropped clip keeps its duration (crop is time-neutral)")

        # a frame-wide dialogue subtitle must NOT vote a corner (suppress path stays)
        clip2 = Path(td) / "subs.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "mandelbrot=s=1280x720:rate=30", "-t", "3",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip2)], check=True,
                       capture_output=True)
        _say(_clip_text_corner(clip2, SubsOCR()) == "",
             "frame-wide subtitle strip is NOT corner-croppable (falls back to suppress)")

    # (4) dodge loop wiring: repair before suppress, and suppression is logged per-beat
    bsrc = (Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text()
    _say("caption-dodge REPAIR" in bsrc and "_clip_text_corner(Path(cp)" in bsrc
         and "caption-dodge SUPPRESS" in bsrc,
         "dodge loop: crop-repair attempted first; suppression logged per-beat")

    # (5) duration-scaled flag sampling — a 51s shot gets ~6s coverage, not 17s blind gaps
    _say(len(_sample_times(0, 1.4)) == 5, "short shot keeps 5 samples")
    _say(len(_sample_times(0, 6.0)) == 3, "6s shot keeps 3 samples")
    ts = _sample_times(100.0, 151.0)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    _say(len(ts) == 9 and max(gaps) < 7.0 and ts[0] > 100.0 and ts[-1] < 151.0,
         f"51s shot gets {len(ts)} spread samples (max gap {max(gaps):.1f}s < 7s)")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
