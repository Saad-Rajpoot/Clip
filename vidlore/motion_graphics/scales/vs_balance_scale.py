"""Primitive: vs_balance_scale.

A tension beat — two labelled forces hang from a balance beam that tips toward the
heavier side and settles with a small wobble. Power, leverage, who outweighs whom.

Distinct from `comparison_split` (two static side-by-side values with a VS
medallion): this one is DYNAMIC — the scale physically tips, so the imbalance is
felt, not just read. "Capital vs Labour", "Risk vs Reward", "the press vs the
palace".

Forensic principle (NOT asset copy): a tipping scale turns an abstract imbalance
into a physical one; the premium is the clean fulcrum + beam + hanging pans + a
single gold accent + a damped settle, not a clip-art seesaw. Pure-local. No paid
API. Deterministic.

    render("x.mp4", left="Capital", right="Labour", leftval=70, rightval=30,
           title="WHO HELD THE POWER")
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
    "id": "vs_balance_scale", "family": "scales",
    "roles": ["balance", "tension", "versus", "power", "tradeoff"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_scale_settle",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["balance_beam"],
    "review_override": ["left", "right", "leftval", "rightval", "title", "palette"],
    "fallback": "comparison_split if a tipping scale would be unclear",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 50.0


def render(out_path, *, left: str = "", right: str = "", leftval=50, rightval=50,
           title: str = "", dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "amber_gold", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    left = (left or "").strip() or "A"
    right = (right or "").strip() or "B"
    lv, rv = _num(leftval), _num(rightval)
    diff = rv - lv
    denom = max(1.0, abs(rv) + abs(lv))
    theta_f = max(-0.26, min(0.26, (diff / denom) * 0.5))      # final tip (rad), ≤15°

    cx = w * 0.5
    apex_y = int(h * 0.40)
    BL = int(w * 0.26)                                          # beam half-length
    pan_drop = int(h * 0.16)
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    title_font = look.font("label", int(h * 0.030))
    lab_font = look.font("title", int(h * 0.044))
    val_font = look.font("numeral", int(h * 0.040))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=cx, cy=h * 0.16, opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # damped tip: 0 → theta_f with a small settle
        tau = min(1.0, t / max(0.1, dur * 0.5))
        damp = 1.0 - math.exp(-3.2 * tau) * math.cos(5.0 * tau)
        theta = theta_f * damp
        # fulcrum (triangle)
        fa = look.ease_out_cubic(min(1.0, t / 0.4))
        d.polygon([(cx, apex_y), (cx - int(w * 0.03), apex_y + int(h * 0.16)),
                   (cx + int(w * 0.03), apex_y + int(h * 0.16))],
                  fill=(*pal["bg_b"], int(235 * fa)),
                  outline=(*accent, int(230 * fa)), width=2)
        d.line([(cx - int(w * 0.05), apex_y + int(h * 0.16)),
                (cx + int(w * 0.05), apex_y + int(h * 0.16))],
               fill=(*pal["muted"], int(200 * fa)), width=3)
        # beam
        lx = cx - BL * math.cos(theta); ly = apex_y - BL * math.sin(theta)
        rx = cx + BL * math.cos(theta); ry = apex_y + BL * math.sin(theta)
        d.line([(int(lx), int(ly)), (int(rx), int(ry))],
               fill=(*accent, int(255 * fa)), width=6)
        d.ellipse([cx - 7, apex_y - 7, cx + 7, apex_y + 7], fill=(*accent_hi, 255))
        # pans hang straight down from each beam end
        lpy, rpy = ly + pan_drop, ry + pan_drop
        pw = int(w * 0.072)
        for (ex, ey, py, lab, val) in ((lx, ly, lpy, left, lv),
                                       (rx, ry, rpy, right, rv)):
            d.line([(int(ex), int(ey)), (int(ex), int(py))],
                   fill=(*pal["muted"], int(200 * fa)), width=2)
            d.arc([int(ex - pw), int(py - pw * 0.5), int(ex + pw), int(py + pw * 0.6)],
                  200, 340, fill=(*accent_hi, int(255 * fa)), width=4)
            d.line([(int(ex - pw * 0.92), int(py + pw * 0.05)),
                    (int(ex + pw * 0.92), int(py + pw * 0.05))],
                   fill=(*accent_hi, int(255 * fa)), width=4)
        # labels + values, fade in after the beam settles a bit
        la = look.ease_out_cubic(max(0.0, (t - 0.5) / 0.6))
        if la > 0.01:
            for (ex, lab, val) in ((lx, left, lv), (rx, right, rv)):
                li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=12)
                frame = look.paste_center(frame, li, cx=ex, cy=apex_y - int(h * 0.16),
                                          opacity=la)
                if rv != lv or leftval != 50:
                    vi = look.gold_fill(str(int(round(val))), val_font, pal,
                                        glow_radius=int(h * 0.006), glow_alpha=0.3 * la)
                    frame = look.paste_center(frame, vi, cx=ex,
                                              cy=apex_y - int(h * 0.105), opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.58)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fade = look.fade_alpha(t, dur, fps)
        if fade < 1.0:
            frame = look.fade_frame(frame, fade, pal)
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
