"""Primitive: countdown_clock.

A tension beat — a circular dial whose depleting gold arc and ticking serif count
run DOWN toward zero, the ring warming to ember as time runs out. The "with N
hours left…" / "the clock was ticking" moment made visible.

Distinct from `gold_number_callout` (a single figure counting UP to a fact): this
is TIME PRESSURE — a value falling toward a deadline inside a clock face, with a
sweep hand and a pulse near zero.

Forensic principle (NOT asset copy): MagnatesMedia dramatises deadlines with a
literal depleting clock; the premium is the graded dial + serif numerals + a clean
depleting arc + the ember warning as it empties, not a busy timer UI. Pure-local
(PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", value=72, to=0, unit="HOURS", label="UNTIL THE DEADLINE")
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
    "id": "countdown_clock", "family": "clocks",
    "roles": ["countdown", "deadline", "timer", "urgency", "tension"],
    "niches_ok": ["crime", "geopolitics", "history", "business", "tech", "biography"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeInOutCubic", "audio_cue": "soft_clock_tick",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["dial"],
    "review_override": ["value", "to", "unit", "label", "palette"],
    "fallback": "gold_number_callout if the beat is a fact, not a deadline",
}


def _num(v):
    return f"{int(round(v)):,}"


def render(out_path, *, value=10, to=0, unit: str = "", label: str = "",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 10.0
    try:
        to = float(to)
    except (TypeError, ValueError):
        to = 0.0
    if value <= to:
        value, to = to + max(1.0, abs(to)), to
    unit = (unit or "").strip().upper()
    label = (label or "").strip()

    cx, cy = w / 2, int(h * 0.46)
    R = int(h * 0.27)
    ember = (188, 44, 30)
    num_font = look.font("numeral", int(h * 0.12))
    unit_font = look.font("label", int(h * 0.030))
    label_font = look.font("label", int(h * 0.032))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # count down over the first ~78%, then hold near `to`
        p = look.ease_in_out_cubic(min(1.0, t / (dur * 0.78)))
        cur = value - (value - to) * p
        frac = max(0.0, min(1.0, (cur - to) / (value - to)))   # ring remaining
        warn = 1.0 - frac                                      # warming as it empties
        ring = tuple(int(a + (b - a) * warn) for a, b in zip(pal["accent_hi"], ember))
        pulse = 1.0 + (0.035 * math.sin(t * 9) if frac < 0.25 else 0.0)

        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        bbox = [cx - R, cy - R, cx + R, cy + R]
        # dim full track
        d.arc(bbox, 0, 360, fill=(*pal["muted"], 90), width=int(h * 0.012))
        # 60 ticks (majors every 5)
        for k in range(60):
            ang = math.radians(k * 6 - 90)
            r_in = R - (int(h * 0.028) if k % 5 == 0 else int(h * 0.016))
            x0, y0 = cx + math.cos(ang) * r_in, cy + math.sin(ang) * r_in
            x1, y1 = cx + math.cos(ang) * (R - int(h * 0.004)), cy + math.sin(ang) * (R - int(h * 0.004))
            col = pal["text"] if k % 5 == 0 else pal["muted"]
            d.line([(x0, y0), (x1, y1)], fill=(*col, 160), width=2 if k % 5 == 0 else 1)
        # depleting arc from 12 o'clock clockwise + a bright head marker riding
        # its leading edge (no centre pivot, so the numeral stays clean)
        if frac > 0.001:
            d.arc(bbox, -90, -90 + 360 * frac, fill=(*ring, 255), width=int(h * 0.014))
        ha = math.radians(-90 + 360 * frac)
        ex, ey = cx + math.cos(ha) * R, cy + math.sin(ha) * R
        mr = int(h * 0.012) * (1.0 + (0.25 * math.sin(t * 9) if frac < 0.25 else 0.0))
        d.ellipse([ex - mr - 4, ey - mr - 4, ex + mr + 4, ey + mr + 4],
                  fill=(*ring, 70))
        d.ellipse([ex - mr, ey - mr, ex + mr, ey + mr], fill=(*pal["accent_hi"], 255))
        # central count (gold, pulses near zero) + unit
        ni = look.gold_fill(_num(cur), num_font, pal,
                            glow_radius=int(h * 0.02), glow_alpha=0.45)
        frame = look.paste_center(frame, ni, cx=cx, cy=cy - int(h * 0.01),
                                  scale=pulse, opacity=1.0)
        if unit:
            ui = look.text_with_glow(unit, unit_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ui, cx=cx, cy=cy + int(h * 0.075),
                                      opacity=1.0)
        d = ImageDraw.Draw(frame, "RGBA")
        if label:
            la = look.ease_out_cubic(min(1.0, t / 0.5))
            look.hairline(d, int(cx), int(cy + R + h * 0.07), int(w * 0.09 * la), pal)
            li = look.text_with_glow(label.upper(), label_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=14)
            frame = look.paste_center(frame, li, cx=cx, cy=cy + R + int(h * 0.11),
                                      opacity=la)

        frame = look.vignette(frame, strength=0.6)
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
