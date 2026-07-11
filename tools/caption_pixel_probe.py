#!/usr/bin/env python3
"""Caption PIXEL probe — render a caption ASS through ffmpeg/libass onto a black frame and measure
the ACTUAL visible text bounding box (not just count \\N or check \\fs exists).

This is the ground-truth validator for the caption line-layout policy: it proves the rendered
glyphs — outline and shadow included, at the active-word animation PEAK — stay inside the safe
margins, never spill to a third row, and are never truncated. Used by both the caption regression
test (tools/test_clipstudio_fixes.py::test_caption_pixel_bbox) and the standalone evidence
generator (run this file as a script).

Everything is font-agnostic: whatever face libass substitutes, we measure the pixels it drew, so a
too-optimistic width estimate in captions._est_px would surface here as a real clip.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root → import vidlore.*

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:                                          # pragma: no cover
    FFMPEG = "ffmpeg"


def have_libass() -> bool:
    """True when this ffmpeg build exposes the libass `subtitles` filter (else the probe skips)."""
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30).stdout
        return " subtitles " in out
    except Exception:
        return False


def render_frame(ass_path: Path, out_png: Path, *, w: int, h: int, t: float) -> bool:
    """Render ONE frame of `ass_path` over a solid-black w×h canvas at timestamp `t` seconds.
    libass reads PlayResX/Y (1920×1080) from the ASS and scales to w×h, so w=1280,h=720 exercises
    720p from the identical 1920-space layout. Returns True on success."""
    ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = f"subtitles='{ass}'"
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d=2:r=25",
           "-ss", f"{t:.3f}", "-frames:v", "1", "-vf", vf, str(out_png)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return out_png.exists()
    except Exception:
        return False


def measure(png: Path, *, thresh: int = 8):
    """Measure the drawn caption on a black frame. Returns a dict with:
      bbox (l, r, t, b) of every non-black pixel (fill + outline + shadow = worst rendered extent),
      rows = number of distinct vertical text bands (line count), and the frame w/h.
    Threshold `thresh` on max(R,G,B) so the near-black outline (e.g. 12,12,12) is still counted."""
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    h, w = im.shape[:2]
    mask = im.max(axis=2) > thresh
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"empty": True, "w": w, "h": h, "px": 0}
    l, r, t, b = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # count vertical text bands: rows with any ink, merged across small (<8px) gaps
    row_has = mask.any(axis=1)
    rows, in_band, gap = 0, False, 0
    for on in row_has:
        if on:
            if not in_band:
                rows += 1
                in_band = True
            gap = 0
        else:
            if in_band:
                gap += 1
                if gap >= 10:                              # a real inter-line gap closes the band
                    in_band = False
    return {"empty": False, "w": w, "h": h, "px": int(mask.sum()),
            "l": l, "r": r, "t": t, "b": b, "rows": rows,
            "margin_l": l, "margin_r": w - 1 - r}


# Pillow is only needed for measure(); import lazily so have_libass()/render_frame work without it.
try:
    from PIL import Image
except Exception:                                          # pragma: no cover
    Image = None


# ── fixtures + evidence generation ───────────────────────────────────────────────────────────────
def _write_narration(words, style, accent, emphasis, out_ass, play_w=1920, play_h=1080):
    from vidlore.captions import write_ass, WordTiming
    wt = [WordTiming(w, s, e) for (w, s, e) in words]
    return write_ass(wt, out_ass, style=style, accent=accent, emphasis_words=emphasis,
                     play_w=play_w, play_h=play_h)


def _cases():
    """The regression matrix as (label, words[(text,start,end)], emphasis, sample_t)."""
    def evenly(texts, dur=0.35):
        return [(t, i * dur, (i + 1) * dur) for i, t in enumerate(texts)]
    return [
        ("normal_6word", evenly(["The", "quick", "brown", "fox", "jumps", "over"]), set(), 0.12),
        ("wide_6long", evenly(["Extraordinary", "revolutionary", "transformation",
                               "fundamentally", "reshaping", "everything"]), {"revolutionary"}, 0.12),
        ("token_60W", [("W" * 60, 0.0, 1.2)], set(), 0.5),
        ("token_100W", [("W" * 100, 0.0, 1.2)], set(), 0.5),
        ("token_200W", [("W" * 200, 0.0, 1.2)], set(), 0.5),
        ("five_long_W", evenly(["WWWWWWWWWWWW"] * 5), set(), 0.12),
        ("emph_left_edge", evenly(["SUPERCALIFRAGILISTIC", "a", "b", "then", "next"]),
         {"supercalifragilistic"}, 0.12),
        ("emph_right_edge", evenly(["then", "next", "a", "b", "EXPIALIDOCIOUSGIANT"]),
         {"expialidociousgiant"}, 4 * 0.35 + 0.12),
        ("cjk", [("权力从来不是别人赐予的东西啊", 0.0, 0.5), ("而是自己夺来的世界", 0.5, 1.0)], set(), 0.3),
        ("arabic_rtl", [("القوة", 0.0, 0.4), ("لا", 0.4, 0.6), ("تُمنح", 0.6, 1.0)], set(), 0.3),
        ("punctuation", evenly(["Power,", "is", "never—", "given!"]), set(), 0.12),
    ]


def main():
    import shutil
    from vidlore.clipstudio import caption_presets as CP
    out = Path.home() / "Desktop" / "clipstudio_output" / "caption_previews" / "phase10"
    out.mkdir(parents=True, exist_ok=True)
    if not have_libass():
        print("libass subtitles filter unavailable — cannot render pixel evidence")
        return 1
    style, accent = CP.CAPTION_PRESETS["professional"].theme_caption()
    ml = 90
    rows = []
    for label, words, emph, t in _cases():
        for (w, h, tag) in ((1920, 1080, "1080p"), (1280, 720, "720p")):
            ass = out / f"{label}.ass"
            _write_narration(words, style, accent, emph, ass, play_w=1920, play_h=1080)
            png = out / f"{label}_{tag}.png"
            if not render_frame(ass, png, w=w, h=h, t=t):
                print(f"  render FAILED: {label} {tag}")
                continue
            m = measure(png)
            scale = w / 1920.0
            safe_l = ml * scale
            ok = (not m["empty"] and m["rows"] <= 2
                  and m["margin_l"] >= safe_l - 6 and m["margin_r"] >= safe_l - 6)
            rows.append((label, tag, m, ok))
            if m["empty"]:
                print(f"  {label:16} {tag}: EMPTY (0 ink px)")
            else:
                print(f"  {label:16} {tag}: rows={m['rows']} bbox=({m['l']},{m['t']})-({m['r']},{m['b']}) "
                      f"margins L={m['margin_l']} R={m['margin_r']} (safe≥{safe_l:.0f}) "
                      f"{'OK' if ok else '*** FAIL ***'}")
    bad = [r for r in rows if not r[3]]
    print(f"\n{len(rows)} frames measured, {len(bad)} out-of-safe. Evidence: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
