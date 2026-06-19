"""Regression tests for the dark-stat-card black-dip fix (mg-0.2.1).

The gold_number_callout card was a near-black graded stage (steady YAVG ~28,
YMIN 7) that also faded IN/OUT from pure black — so a dissolve/cut landing on it
read as a black frame (blackdetect span at 2:18 in the 1860s-Secret render).

Fix: look.graded_background gained a luminance `floor` (lifts the dark stage to a
deep charcoal, gradient/hue preserved), and the primitive now fades the
FOREGROUND (numeral/label) over an always-lit stage instead of multiplying the
whole frame toward black. These tests lock both behaviours.

Run:  .venv/bin/python tools/test_dark_card_floor.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
from vidlore.motion_graphics import look                        # noqa: E402
from vidlore.motion_graphics.numbers import gold_number_callout as G  # noqa: E402
from vidlore.motion_graphics import render_dispatch              # noqa: E402
from vidlore.ffmpeg_tool import ffmpeg_exe                       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _img_min_luma(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return float(lum.min()), float(lum.mean())


def test_lift_color():
    print("\n_lift_color:")
    dark = look._lift_color((12, 9, 6), 40.0)          # parchment bg_b-ish
    check("dark color lifted to target luma", abs(look._luma(dark) - 40.0) < 1.5,
          f"luma={look._luma(dark):.1f}")
    check("lifted color keeps warm hue (R>G>B)", dark[0] > dark[1] > dark[2])
    bright = look._lift_color((200, 200, 200), 40.0)
    check("already-bright color unchanged", look._luma(bright) > 190)
    blackish = look._lift_color((0, 0, 0), 40.0)
    check("pure black → neutral floor", abs(look._luma(blackish) - 40.0) < 1.0)


def test_graded_background_floor():
    print("\ngraded_background floor:")
    pal = look.palette("parchment_sepia")
    base0 = look.graded_background(320, 180, pal, seed=7, floor=0.0)
    mn0, me0 = _img_min_luma(base0)
    baseF = look.graded_background(320, 180, pal, seed=7,
                                   floor=look.CARD_STAGE_FLOOR)
    mnF, meF = _img_min_luma(baseF)
    check("legacy (floor=0) stage is near-black", mn0 < 15, f"min={mn0:.1f}")
    check("floored stage lifts the dark edge well above black", mnF > 28,
          f"min={mnF:.1f} (floor={look.CARD_STAGE_FLOOR})")
    check("floored stage mean clears blackdetect pix_th (25.5)", meF > 30,
          f"mean={meF:.1f}")
    check("floor only lifts (never darkens)", mnF >= mn0 and meF >= me0)


def test_card_has_no_black_frames():
    print("\ngold_number_callout renders with NO black frames (fast 480p):")
    td = Path(tempfile.mkdtemp())
    out = td / "card.mp4"
    res = G.render(str(out), value=96.0, suffix="%",
                   palette_name="parchment_sepia", seed=11737,
                   dur=2.0, fps=30, w=854, h=480)
    check("card render ok", bool(res.get("ok")) and out.exists())
    if not out.exists():
        return
    ff = ffmpeg_exe()
    r = subprocess.run([ff, "-hide_banner", "-i", str(out),
                        "-vf", "blackdetect=d=0.03:pic_th=0.98", "-an",
                        "-f", "null", "-"], capture_output=True, text=True)
    spans = re.findall(r"black_start:", r.stderr)
    check("zero blackdetect spans over the whole card (incl. fade in/out)",
          len(spans) == 0, f"{len(spans)} span(s)")
    # first & last frame (the fade moments) must be well above the black floor
    r2 = subprocess.run([ff, "-hide_banner", "-i", str(out),
                         "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                         "-an", "-f", "null", "-"], capture_output=True, text=True)
    yv = [float(m) for m in re.findall(r"YAVG=(\S+)", r2.stderr)]
    if yv:
        check("first frame (fade-in) YAVG well above 25", yv[0] > 30, f"{yv[0]:.1f}")
        check("last frame (fade-out) YAVG well above 25", yv[-1] > 30, f"{yv[-1]:.1f}")
        check("global min YAVG never dips below 25", min(yv) >= 25,
              f"min={min(yv):.1f}")
    import shutil
    shutil.rmtree(td, ignore_errors=True)


def test_version_bumped():
    print("\nMG cache version:")
    check("render_dispatch.VERSION bumped (invalidates cached dark card)",
          render_dispatch.VERSION != "mg-0.2.0",
          render_dispatch.VERSION)


if __name__ == "__main__":
    test_lift_color()
    test_graded_background_floor()
    test_card_has_no_black_frames()
    test_version_bumped()
    print(f"\n{'='*48}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
    print("ALL GREEN")
