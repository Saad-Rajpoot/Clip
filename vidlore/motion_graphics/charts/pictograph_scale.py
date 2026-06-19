"""Primitive: pictograph_scale.

A scale beat — a grid of small figure icons where the first N are lit in gold and
the rest stay muted, making a ratio physically countable: "3 IN 10". The lit
figures fill in one after another while an "N in M" line reads beneath.

Distinct from `proportion_ring` (one share as an arc) and `statistic_bar_reveal`
(magnitudes as bars): this is a COUNT pictograph — it shows the ratio as objects
you can literally count, the way MagnatesMedia makes "1 in 8 businesses fail"
land in the gut.

Forensic principle (NOT asset copy): a proportion rendered as countable figures
reads more viscerally than a percentage; the premium is the clean figure glyph +
gold-vs-muted contrast + staggered fill + restraint. Pure-local. No paid API.

    render("x.mp4", count=3, total=10, label="businesses that fail in year one")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "pictograph_scale", "family": "charts",
    "roles": ["scale", "ratio", "share", "proportion", "stat"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_count_tick",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["figure_grid"],
    "review_override": ["count", "total", "label", "palette"],
    "fallback": "proportion_ring if the figure grid would be too dense",
}


def _person(d, cx, cy, s, *, fill=None, outline=None, width=2):
    hr = s * 0.22
    d.ellipse([cx - hr, cy - s * 0.52, cx + hr, cy - s * 0.52 + 2 * hr],
              fill=fill, outline=outline, width=width)
    bw = s * 0.52
    d.rounded_rectangle([cx - bw / 2, cy - s * 0.04, cx + bw / 2, cy + s * 0.46],
                        radius=s * 0.16, fill=fill, outline=outline, width=width)


def render(out_path, *, count=3, total=10, label: str = "", dur: float = 5.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    try:
        total = max(1, min(100, int(round(float(total)))))
        count = max(0, min(total, int(round(float(count)))))
    except (TypeError, ValueError):
        total, count = 10, 3

    cols = total if total <= 10 else 10
    rows = math.ceil(total / cols)
    gw = w * 0.62
    cell = min(gw / cols, h * 0.16)
    s = cell * 0.74
    grid_w = cell * cols
    grid_h = cell * rows
    gx0 = w * 0.5 - grid_w / 2 + cell / 2
    gy0 = h * 0.44 - grid_h / 2 + cell / 2

    title_font = look.font("label", int(h * 0.030))
    num_font = look.font("numeral", int(h * 0.060))
    lab_font = look.font("label", int(h * 0.030))
    accent_hi = tuple(pal["accent_hi"]); muted = tuple(pal["muted"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        # all figures fade in muted first
        ga = look.ease_out_cubic(min(1.0, t / 0.5))
        for k in range(total):
            r, c = divmod(k, cols)
            cx = gx0 + c * cell
            cy = gy0 + r * cell
            _person(d, cx, cy, s, outline=(*muted, int(150 * ga)), width=2)
        # lit figures fill gold, staggered
        for k in range(count):
            ka = look.ease_out_cubic(max(0.0, (t - (0.5 + k * (1.6 / max(1, count))))
                                         / 0.4))
            if ka <= 0.01:
                continue
            r, c = divmod(k, cols)
            cx = gx0 + c * cell
            cy = gy0 + r * cell
            _person(d, cx, cy, s, fill=(*accent_hi, int(255 * ka)),
                    outline=(*accent_hi, int(255 * ka)), width=2)
        # "N in M" line beneath
        la = look.ease_out_cubic(max(0.0, (t - 0.4) / 0.6))
        if la > 0.01:
            lit = sum(1 for k in range(count)
                      if (t - (0.5 + k * (1.6 / max(1, count)))) > 0)
            ny = int(gy0 + grid_h - cell / 2 + h * 0.10)
            ni = look.gold_fill(str(lit), num_font, pal,
                                glow_radius=int(h * 0.010), glow_alpha=0.4)
            inw = look.text_with_glow(f"  in {total}", num_font, fill=pal["text"],
                                      glow=pal["bg_b"], glow_radius=4,
                                      glow_alpha=0.0, pad=6)
            gap = inw.width // 2 + 6
            frame = look.paste_center(frame, ni, cx=w * 0.5 - gap, cy=ny, opacity=la)
            frame = look.paste_center(frame, inw, cx=w * 0.5 + ni.width // 2,
                                      cy=ny, opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")
        # title above the grid
        if label:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(label.upper(), title_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5,
                                      cy=int(gy0 - cell / 2 - h * 0.06), opacity=ta)

        frame = look.vignette(frame, strength=0.58)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        ff = "ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
           "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and Path(out_path).exists()
    look.cleanup_frames(td)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
