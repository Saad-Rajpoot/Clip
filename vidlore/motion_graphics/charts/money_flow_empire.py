"""Primitive: money_flow_empire.

A business empire / ownership / money-flow network: a central entity node with a
small number of branch nodes, connecting lines that draw on (staggered), and
limited value labels — a clean hierarchy, NOT a SaaS dashboard. For business /
crime-money / influence scenes only.

Forensic evidence: MagnatesMedia 1058s (Standard-Oil empire / octopus reach),
1300s (cash). Pure-local (PIL + numpy → ffmpeg). No paid API. Original layout
(we do NOT reproduce their octopus art).

    render(out, center="STANDARD OIL",
           branches=[("Pipelines","$40M"),("Railroads","$25M"),
                     ("Refineries","$90M"),("Banks","$60M")])
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
    "id": "money_flow_empire", "family": "charts",
    "roles": ["empire", "ownership", "money_flow", "influence", "network"],
    "niches_ok": ["business", "crime", "geopolitics", "history", "biography"],
    "intensity_range": [3, 5], "duration_range": [4.0, 8.0],
    "easing": "easeOutCubic", "audio_cue": "line_draw_ticks_low_rise",
    "repeat_cooldown_s": 90, "per_video_cap": 2, "cost": "med",
    "review_override": ["center", "branches", "palette"],
    "fallback": "gold_number_callout (single figure) if <2 branches",
    "max_branches": 6,
}


def _node(draw, x, y, r, pal, fill_dark=True):
    col = pal["bg_b"] if fill_dark else pal["accent"]
    draw.ellipse([x - r, y - r, x + r, y + r], fill=(*col, 235),
                 outline=(*pal["accent_hi"], 255), width=3)


def render(out_path, *, center: str, branches, dur: float = 5.6, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    branches = list(branches)[:6]
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    cx, cy = w * 0.30, h * 0.52
    cr = int(h * 0.085)
    # branch node positions on the right arc
    m = len(branches)
    nodes = []
    for k, (lbl, val) in enumerate(branches):
        frac = (k + 0.5) / m
        ang = math.radians(-58 + 116 * frac)
        nx = cx + math.cos(ang) * w * 0.42
        ny = cy + math.sin(ang) * h * 0.34
        nodes.append((nx, ny, lbl, val))
    cen_font = look.font("title", int(h * 0.034))
    nl_font = look.font("label", int(h * 0.026))
    nv_font = look.font("title", int(h * 0.034))
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps; pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        # lines draw on, staggered per branch after the centre appears
        for k, (nx, ny, lbl, val) in enumerate(nodes):
            ld = look.ease_out_cubic(max(0.0, (t - 0.6 - 0.22 * k) / 0.6))
            ex = cx + (nx - cx) * ld
            ey = cy + (ny - cy) * ld
            if ld > 0.01:
                d.line([(cx, cy), (ex, ey)], fill=(*pal["accent"], 200), width=3)
        # centre node + label
        cen = look.ease_out_cubic(min(1.0, t / 0.45))
        _node(d, cx, cy, int(cr * cen), pal, fill_dark=False)
        if cen > 0.6:
            cl = look.text_with_glow(center.upper(), cen_font, fill=pal["bg_b"],
                                     glow=pal["accent_hi"], glow_radius=3,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, cl, cx=cx, cy=cy, opacity=(cen - 0.6) / 0.4)
            d = ImageDraw.Draw(frame, "RGBA")
        # branch nodes pop after their line lands + labels/values fade
        for k, (nx, ny, lbl, val) in enumerate(nodes):
            ld = look.ease_out_cubic(max(0.0, (t - 0.6 - 0.22 * k) / 0.6))
            if ld < 0.9:
                continue
            pop = look.ease_out_cubic(min(1.0, (t - (0.6 + 0.22 * k) - 0.5) / 0.4))
            _node(d, nx, ny, int(h * 0.045 * pop), pal, fill_dark=True)
            if pop > 0.5:
                vop = (pop - 0.5) / 0.5
                vl = look.text_with_glow(val, nv_font, fill=pal["accent_hi"],
                                         glow=pal["glow"], glow_radius=8,
                                         glow_alpha=0.5, pad=18)
                frame = look.paste_center(frame, vl, cx=nx, cy=ny - h * 0.085, opacity=vop)
                ll = look.text_with_glow(lbl.upper(), nl_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=14)
                frame = look.paste_center(frame, ll, cx=nx, cy=ny + h * 0.072, opacity=vop)
                d = ImageDraw.Draw(frame, "RGBA")
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
    import shutil as _sh
    _sh.rmtree(td, ignore_errors=True)        # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2),
            "err": (r.stderr[-200:] if not ok else "")}
