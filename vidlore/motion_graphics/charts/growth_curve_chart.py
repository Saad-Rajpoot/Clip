"""Primitive: growth_curve_chart.

A trajectory beat — a smooth line/area curve that DRAWS IN left→right over time
while a glowing plot-head rides the leading edge and a serif value counts up to
the value under the head. Faint y-gridlines + x-axis year labels that light as the
head passes them. Over a graded, vignetted bg with grain.

Distinct from `statistic_bar_reveal` (discrete categorical columns) and
`money_flow_empire` (a node/branch graph): this is a CONTINUOUS time-series trend
— "revenue / users / output grew from X to Y over the years".

Forensic principle (NOT asset copy): MagnatesMedia draws the growth curve into the
frame and lets the number climb with it; the premium is the grade + smooth
Catmull-Rom curve + soft area + glowing head + serif count-up + restraint.
Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", points=[["2010", 4], ["2014", 30], ["2018", 90],
           ["2022", 240]], title="USERS (MILLIONS)", suffix="M")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "growth_curve_chart", "family": "charts",
    "roles": ["trend", "growth", "trajectory", "data", "stat"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.5],
    "easing": "easeInOutCubic", "audio_cue": "soft_rise_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["area_curve"],
    "review_override": ["points", "title", "prefix", "suffix", "palette"],
    "fallback": "statistic_bar_reveal if fewer than 2 points are available",
}


def _coerce(points):
    out = []
    for p in (points or []):
        if isinstance(p, dict):
            lab, val = str(p.get("label", "")), p.get("value", 0)
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            lab, val = str(p[0]), p[1]
        else:
            continue
        try:
            out.append((lab.strip(), float(val)))
        except (TypeError, ValueError):
            continue
    return out[:6]


def _num(v, prefix, suffix, decimals):
    body = f"{v:,.{decimals}f}" if decimals > 0 else f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _densify(xs, ys, vs, steps=22):
    """Catmull-Rom through the control points → aligned dense (x, y, val) arrays
    for a smooth curve with a value carried under every pixel."""
    m = len(xs)
    if m < 2:
        return list(xs), list(ys), list(vs)
    dx, dy, dv = [], [], []
    for k in range(m - 1):
        x0, x1, x2, x3 = (xs[max(0, k - 1)], xs[k], xs[k + 1], xs[min(m - 1, k + 2)])
        y0, y1, y2, y3 = (ys[max(0, k - 1)], ys[k], ys[k + 1], ys[min(m - 1, k + 2)])
        v1, v2 = vs[k], vs[k + 1]
        last = (k == m - 2)
        for s in range(steps + (1 if last else 0)):
            tt = s / steps
            dx.append(_catmull(x0, x1, x2, x3, tt))
            dy.append(_catmull(y0, y1, y2, y3, tt))
            dv.append(v1 + (v2 - v1) * tt)         # value lerps linearly
    return dx, dy, dv


def render(out_path, *, points=None, title: str = "", prefix: str = "",
           suffix: str = "", decimals: int = 0, dur: float = 6.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(points)
    if len(data) < 2:                              # degenerate → flat 2-point line
        v = data[0][1] if data else 1.0
        lab = data[0][0] if data else ""
        data = [(lab, v), (lab, v)]
    m = len(data)
    vmax = max(v for _, v in data) or 1.0

    base_y, top_y = int(h * 0.74), int(h * 0.30)
    max_bh = base_y - top_y
    span = w * 0.62
    left = w * 0.5 - span / 2
    xs = [left + span * (k / (m - 1)) for k in range(m)]
    ys = [base_y - max_bh * (v / vmax) for _, v in data]
    vs = [v for _, v in data]
    dx, dy, dv = _densify(xs, ys, vs)
    nd = len(dx)

    title_font = look.font("label", int(h * 0.030))
    num_font = look.font("numeral", int(h * 0.078))
    lab_font = look.font("label", int(h * 0.026))
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        # title
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.18,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # faint y-gridlines (3) + baseline
        ga = look.ease_out_cubic(min(1.0, t / 0.5))
        for gy in (top_y, (top_y + base_y) // 2):
            d.line([(int(left), gy), (int(left + span), gy)],
                   fill=(*pal["muted"], int(40 * ga)), width=1)
        bl = look.ease_out_cubic(min(1.0, t / 0.55))
        d.line([(int(left), base_y), (int(left + span * bl), base_y)],
               fill=(*pal["muted"], 200), width=2)

        # curve reveal — head rides the leading edge (eased so it settles)
        hp = look.ease_in_out_cubic(min(1.0, t / max(0.1, dur * 0.62)))
        hj = max(1, int(hp * (nd - 1)))
        rx, ry = dx[:hj + 1], dy[:hj + 1]
        if len(rx) >= 2:
            # area fill under the revealed curve (soft, on its own layer)
            poly = [(int(left), base_y)] + [(int(a), int(b)) for a, b in zip(rx, ry)] \
                   + [(int(rx[-1]), base_y)]
            area = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(area).polygon(poly, fill=(*accent, 70))
            area = area.filter(ImageFilter.GaussianBlur(1.2))
            frame = Image.alpha_composite(frame.convert("RGBA"), area).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            # the curve line (glow underlay + bright stroke)
            line = [(int(a), int(b)) for a, b in zip(rx, ry)]
            glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(glow).line(line, fill=(*accent_hi, 150), width=10,
                                      joint="curve")
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                glow.filter(ImageFilter.GaussianBlur(6))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.line(line, fill=(*accent_hi, 255), width=4, joint="curve")
            # faint tracer from head down to baseline
            hx, hy = int(rx[-1]), int(ry[-1])
            d.line([(hx, hy), (hx, base_y)], fill=(*pal["muted"], 110), width=1)
            # glowing plot-head
            for rr, al in ((16, 60), (10, 120), (6, 255)):
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr],
                          fill=(*accent_hi, al) if rr == 6 else None,
                          outline=(*accent_hi, al) if rr != 6 else None,
                          width=2)
            # value count-up under the head
            ni = look.gold_fill(_num(dv[hj], prefix, suffix, decimals), num_font,
                                 pal, glow_radius=int(h * 0.013), glow_alpha=0.45)
            frame = look.paste_center(frame, ni, cx=min(max(hx, int(left) + 70),
                                      int(left + span) - 70),
                                      cy=max(hy - int(h * 0.07), top_y - int(h * 0.02)),
                                      opacity=1.0)
            d = ImageDraw.Draw(frame, "RGBA")
        # x-axis labels light as the head passes their control x
        head_x = dx[hj]
        for k, (lab, _v) in enumerate(data):
            if not lab:
                continue
            # light each label as the head reaches it (full AT its x, so the
            # final label — where the head stops — also reaches full opacity)
            la = look.ease_out_cubic(
                max(0.0, min(1.0, (head_x - (xs[k] - span * 0.05)) / (span * 0.05))))
            if la <= 0.02:
                continue
            li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=10)
            frame = look.paste_center(frame, li, cx=int(xs[k]),
                                      cy=base_y + int(h * 0.045), opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")

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
