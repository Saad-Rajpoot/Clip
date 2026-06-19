"""Primitive: sankey_flow.

A "where it goes" beat — a single source on the left splits into proportional-
width ribbons that flow right into labelled branches. The WIDTH of each ribbon is
the story: you see at a glance how a whole divides (revenue → costs, a budget →
departments, a vote → factions).

Distinct from `money_flow_empire` (a centre node with thin radiating connector
lines to branch cards) and `composition_stack` (a single stacked bar): here the
flow itself has mass — fat gold ribbons peel off the source and fan to each
destination, proportional end to end.

Forensic principle (NOT asset copy): MagnatesMedia shows money MOVING with weight
rather than listing it; the premium is the graded source column + translucent
gold bezier ribbons + clean branch nodes + restraint. Pure-local (PIL + numpy →
ffmpeg). No paid API. Deterministic.

    render("x.mp4", source="REVENUE", branches=[["Refining", 60], ["Pipelines", 25],
           ["Railroads", 15]], title="WHERE THE DOLLAR WENT", prefix="$", suffix="M")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "sankey_flow", "family": "charts",
    "roles": ["flow", "allocation", "split", "breakdown", "money"],
    "niches_ok": ["business", "history", "geopolitics", "tech", "crime", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "soft_flow_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["source_split"],
    "review_override": ["source", "branches", "title", "prefix", "suffix", "palette"],
    "fallback": "composition_stack if the split reads better as one stacked bar",
}


def _coerce(branches):
    out = []
    for b in (branches or []):
        if isinstance(b, dict):
            lab, val = str(b.get("label") or b.get("name") or ""), b.get("value", 0)
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            lab, val = str(b[0]), b[1]
        else:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append((lab.strip(), v))
    return out[:5]


def _num(v, prefix, suffix, decimals):
    body = f"{v:,.{decimals}f}" if decimals > 0 else f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def _bez(p0, p1, p2, p3, t):
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def render(out_path, *, source: str = "", branches=None, total=None,
           title: str = "", prefix: str = "", suffix: str = "", decimals: int = 0,
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(branches)
    if not data:
        data = [("", 1.0)]
    S = sum(v for _, v in data) or 1.0
    tot = float(total) if total not in (None, "") else S
    m = len(data)
    source = (source or "TOTAL").strip()

    title_font = look.font("label", int(h * 0.030))
    src_font = look.font("label", int(h * 0.030))
    srcnum_font = look.font("numeral", int(h * 0.044))
    lab_font = look.font("label", int(h * 0.026))
    val_font = look.font("numeral", int(h * 0.030))

    # source column geometry
    SRC_H = int(h * 0.50)
    cy = int(h * 0.54)
    src_top = cy - SRC_H // 2
    src_x0, src_x1 = int(w * 0.16), int(w * 0.215)
    # branch nodes geometry (spread with gaps, height ∝ value)
    br_x0, br_x1 = int(w * 0.70), int(w * 0.745)
    gap = int(h * 0.024)
    span = int(h * 0.56)
    avail = span - gap * (m - 1)
    node_h = [max(6, int(avail * (v / S))) for _, v in data]
    by_start = cy - span // 2
    br_y = []
    yy = by_start
    for k in range(m):
        br_y.append((yy, yy + node_h[k]))
        yy += node_h[k] + gap
    # source segments stacked in same order (height ∝ value)
    sy = []
    yy = src_top
    for k in range(m):
        sh = int(SRC_H * (data[k][1] / S))
        sy.append((yy, yy + sh))
        yy += sh

    NS = 24
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
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.13,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # ribbons (drawn first, behind nodes) — translucent gold beziers
        for k in range(m):
            ent = look.ease_in_out_cubic(max(0.0, (t - (0.3 + k * 0.12)) / 0.7))
            if ent <= 0.01:
                continue
            ys0, ys1 = sy[k]
            yb0, yb1 = br_y[k]
            dx = br_x0 - src_x1
            top = ((src_x1, ys0), (src_x1 + 0.45 * dx, ys0),
                   (br_x0 - 0.45 * dx, yb0), (br_x0, yb0))
            bot = ((src_x1, ys1), (src_x1 + 0.45 * dx, ys1),
                   (br_x0 - 0.45 * dx, yb1), (br_x0, yb1))
            ts = [ent * j / (NS - 1) for j in range(NS)]
            pts = [_bez(*top, tt) for tt in ts] + \
                  [_bez(*bot, tt) for tt in reversed(ts)]
            col = pal["accent_hi"] if k == 0 else pal["accent"]
            # thin dark outline separates adjacent ribbons even where they touch
            d.polygon(pts, fill=(*col, 150), outline=(*pal["bg_b"], 210))
        # source column (solid gold, gradient cap)
        sa = look.ease_out_cubic(min(1.0, t / 0.5))
        if sa > 0.01:
            d.rectangle([src_x0, src_top, src_x1, src_top + int(SRC_H * sa)],
                        fill=(*pal["accent"], 255))
            d.rectangle([src_x0, src_top, src_x0 + 5, src_top + int(SRC_H * sa)],
                        fill=(*pal["accent_hi"], 255))
        # source label + total above the column
        if sa > 0.2:
            si = look.text_with_glow(source.upper(), src_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=10)
            frame = look.paste_center(frame, si, cx=(src_x0 + src_x1) // 2,
                                      cy=src_top - int(h * 0.075), opacity=sa)
            ni = look.gold_fill(_num(tot, prefix, suffix, decimals), srcnum_font,
                                pal, glow_radius=int(h * 0.012), glow_alpha=0.4 * sa)
            frame = look.paste_center(frame, ni, cx=(src_x0 + src_x1) // 2,
                                      cy=src_top - int(h * 0.035), opacity=sa)
            d = ImageDraw.Draw(frame, "RGBA")
        # branch nodes + labels
        for k, (lab, val) in enumerate(data):
            ent = look.ease_out_cubic(max(0.0, (t - (0.55 + k * 0.12)) / 0.5))
            if ent <= 0.01:
                continue
            yb0, yb1 = br_y[k]
            col = pal["accent_hi"] if k == 0 else pal["accent"]
            d.rounded_rectangle([br_x0, yb0, br_x1, yb1], radius=4,
                                fill=(*col, int(255 * ent)))
            # label + value to the right of the node
            lab_im = look.text_with_glow(
                (lab.upper() if lab else f"PART {k+1}"), lab_font,
                fill=(pal["text"] if k == 0 else pal["muted"]),
                glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0, pad=6)
            frame.paste(lab_im, (br_x1 + int(w * 0.012),
                                 (yb0 + yb1) // 2 - lab_im.height + 2), lab_im)
            vtxt = _num(val, prefix, suffix, decimals) + f"  ·  {val/S*100:.0f}%"
            if k == 0:
                vi = look.gold_fill(vtxt, val_font, pal,
                                    glow_radius=int(h * 0.008), glow_alpha=0.35 * ent)
            else:
                vi = look.text_with_glow(vtxt, val_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=2,
                                         glow_alpha=0.0, pad=6)
            frame.paste(vi, (br_x1 + int(w * 0.012), (yb0 + yb1) // 2 + 2), vi)
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
