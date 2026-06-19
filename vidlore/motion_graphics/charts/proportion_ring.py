"""Primitive: proportion_ring.

A share beat — a single PARTS-OF-A-WHOLE figure: a gold arc sweeps clockwise from
12 o'clock around a faint full ring, filling to the share percentage, while the
numeral counts up in the centre and a label names what the share is OF.

Distinct from `statistic_bar_reveal` (magnitudes of several categories),
`gold_number_callout` (one hero figure with no whole to compare against) and
`growth_curve_chart` (a trend over time): this is ONE proportion of a whole —
"controlled 90% of the market", "two-thirds of all output", "a 4% margin".

Forensic principle (NOT asset copy): MagnatesMedia shows dominance as a ring
filling up, not a flat pie; the premium is the grade + smooth sweep + serif
count-up + a rounded glowing arc head + restraint. Pure-local. No paid API.

    render("x.mp4", share=90, label="OF ALL U.S. OIL", center_sub="STANDARD OIL")
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
    "id": "proportion_ring", "family": "charts",
    "roles": ["share", "proportion", "percent", "dominance", "stat"],
    # Forensic v2: a percent/share ring is universal — un-gate so explainer/
    # science/_default percentage beats can use it. Director cue (share/pct)
    # + threshold + per_video_cap=2 still gate WHEN it fires.
    "niches_ok": ["all"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_sweep_rise",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["center_ring"],
    "review_override": ["share", "label", "center_sub", "suffix", "palette"],
    "fallback": "gold_number_callout if no clear share/percent is available",
}


def _coerce_share(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 50.0
    if 0 < f <= 1.0:                       # accept 0.9 as 90 %
        f *= 100.0
    return max(0.0, min(100.0, f))


def _arc(d, box, a0, a1, fill, width):
    d.arc(box, a0, a1, fill=fill, width=width)


def render(out_path, *, share=50, label: str = "", center_sub: str = "",
           suffix: str = "%", dur: float = 5.5, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "amber_gold", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    pct = _coerce_share(share)

    cx, cy = w * 0.5, h * 0.46
    R = int(h * 0.23)
    thick = int(h * 0.034)
    box = [int(cx - R), int(cy - R), int(cx + R), int(cy + R)]
    num_font = look.font("numeral", int(h * 0.13))
    sub_font = look.font("label", int(h * 0.026))
    lab_font = look.font("label", int(h * 0.030))
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        # faint full track
        _arc(d, box, 0, 360, (*pal["muted"], 90), thick)
        # filled sweep from 12 o'clock (=-90°) clockwise
        sweep = look.ease_out_cubic(min(1.0, t / max(0.1, dur * 0.6)))
        ang = 360.0 * (pct / 100.0) * sweep
        if ang > 0.4:
            _arc(d, box, -90, -90 + ang, (*accent_hi, 255), thick)
            # rounded cap + glowing head at the leading edge
            hx = cx + R * math.cos(math.radians(-90 + ang))
            hy = cy + R * math.sin(math.radians(-90 + ang))
            sx = cx + R * math.cos(math.radians(-90))
            sy = cy + R * math.sin(math.radians(-90))
            rr = thick // 2
            d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], fill=(*accent_hi, 255))
            for gr, ga in ((rr + 8, 70), (rr + 3, 140), (rr, 255)):
                d.ellipse([hx - gr, hy - gr, hx + gr, hy + gr],
                          fill=(*accent_hi, ga) if gr == rr else None,
                          outline=(*accent_hi, ga) if gr != rr else None, width=2)
        # centre count-up numeral
        cur = pct * sweep
        body = f"{int(round(cur))}{suffix}"
        ni = look.gold_fill(body, num_font, pal, glow_radius=int(h * 0.016),
                            glow_alpha=0.45)
        frame = look.paste_center(frame, ni, cx=cx, cy=cy - int(h * 0.01),
                                  opacity=1.0)
        d = ImageDraw.Draw(frame, "RGBA")
        # centre sub-label (inside the ring, under the numeral)
        if center_sub:
            ca = look.ease_out_cubic(min(1.0, t / 0.5))
            si = look.text_with_glow(center_sub.upper(), sub_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=10)
            frame = look.paste_center(frame, si, cx=cx, cy=cy + int(h * 0.085),
                                      opacity=ca)
            d = ImageDraw.Draw(frame, "RGBA")
        # label below the ring
        if label:
            la = look.ease_out_cubic(max(0.0, (t - 0.5) / 0.6))
            cyl = int(cy + R + h * 0.10)
            look.hairline(d, int(cx), cyl - int(h * 0.045), int(w * 0.10 * la), pal)
            li = look.text_with_glow(label.upper(), lab_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=14)
            frame = look.paste_center(frame, li, cx=cx, cy=cyl, opacity=la)

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
