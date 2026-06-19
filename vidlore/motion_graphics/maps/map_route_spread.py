"""Primitive: map_route_spread.

A movement beat — a route DRAWS across a graded antique map: a faint ghost path
first, then a bright line traces from the origin pin to the destination while a
glowing comet-head rides the leading edge; waypoint ticks + names light as the
head passes them, and the destination pin pulses once it arrives.

Distinct from `location_establish_card` (a single static place) and
`portrait_name_over_map` (a person tethered to a place): this is GEOGRAPHY IN
MOTION — an expansion, a journey, a trade route, an invasion, a supply line
spreading from A to B (to C…).

Forensic principle (NOT asset copy): MagnatesMedia animates reach/expansion as a
line crawling across a period map; the premium is the grade + smooth route +
comet-head + restrained pins/labels. Reuses the maps-family antique synth (or a
supplied map image). Pure-local. No paid API. Deterministic.

    render("x.mp4", stops=[["London", "0.20,0.34"], ["Suez", "0.55,0.46"],
           ["Bombay", "0.82,0.55"]], title="THE EMPIRE ROUTE · 1869")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look
from . import geo
from .location_establish_card import _antique_map

SPEC = {
    "id": "map_route_spread", "family": "maps",
    "roles": ["route", "journey", "expansion", "spread", "movement"],
    "niches_ok": ["history", "geopolitics", "biography", "business", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.5],
    "easing": "easeInOutCubic", "audio_cue": "soft_route_draw",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    # V3.4 — route-line TREATMENT variants (skin only; coordinates, labels and
    # stop ORDER are never changed by the variant — only the line texture).
    "layout_variants": ["route_trace", "dashed_march"],
    "review_override": ["stops", "title", "map_image", "palette"],
    "fallback": "location_establish_card if fewer than 2 stops are available",
}


def _parse_stops(stops):
    out = []
    for s in (stops or []):
        name, pos = "", None
        if isinstance(s, dict):
            name = str(s.get("name") or s.get("label") or "")
            pos = s.get("pos") or s.get("xy")
        elif isinstance(s, (list, tuple)) and len(s) >= 2:
            name, pos = str(s[0]), s[1]
        else:
            name = str(s)
        fxy = None
        try:
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                fxy = (float(pos[0]), float(pos[1]))
            elif isinstance(pos, str) and "," in pos:
                a, b = pos.replace(" ", "").split(",")[:2]
                fxy = (float(a), float(b))
        except Exception:                                      # noqa: BLE001
            fxy = None
        out.append((name.strip(), fxy))
    return out[:5]


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _densify(xs, ys, steps=26):
    m = len(xs)
    if m < 2:
        return list(xs), list(ys), [0]
    dx, dy, ctrl = [], [], []
    for k in range(m - 1):
        x0, x1, x2, x3 = (xs[max(0, k - 1)], xs[k], xs[k + 1], xs[min(m - 1, k + 2)])
        y0, y1, y2, y3 = (ys[max(0, k - 1)], ys[k], ys[k + 1], ys[min(m - 1, k + 2)])
        last = (k == m - 2)
        for s in range(steps + (1 if last else 0)):
            tt = s / steps
            dx.append(_catmull(x0, x1, x2, x3, tt))
            dy.append(_catmull(y0, y1, y2, y3, tt))
            if s == 0:
                ctrl.append(len(dx) - 1)                        # dense idx of stop k
    ctrl.append(len(dx) - 1)                                   # last stop
    return dx, dy, ctrl


def render(out_path, *, stops=None, title: str = "", map_image=None, bg_image=None,
           dur: float = 6.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", layout: str = "", seed: int = 0,
           reference: bool = False, borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    _dashed = (layout or "").strip().lower() == "dashed_march"  # V3.4 skin variant
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    sp = _parse_stops(stops)
    if len(sp) < 2:                                            # degenerate guard
        sp = (sp + [("", None), ("", None)])[:2]
    m = len(sp)
    # REAL geography: project resolvable stop names at TRUE positions on a real
    # region-framed bed; else auto-place along a gentle arc (synthetic fallback).
    _geo_bed = None
    _res = [(nm, geo.resolve(nm)) for nm, _ in sp if nm]
    _res = [(nm, ll) for nm, ll in _res if ll]
    if (map_image or bg_image) is None and len(_res) >= 2:
        _bbox = geo.region_bbox([ll for _, ll in _res], pad_frac=0.4)
        _gp = geo.make_projector(_bbox, w, h)
        _gb = geo.draw_basemap(w, h, _bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=_gp, graticule=True,
                               reference=reference, borders=borders)
        _geo_bed = Image.fromarray((np.asarray(_gb).astype(np.float32) * 0.62)
                                   .astype("uint8"), "RGB")
        pts = [(nm, _gp(*ll)[0], _gp(*ll)[1]) for nm, ll in _res]
    else:
        pts = []
        for k, (name, fxy) in enumerate(sp):
            if fxy is None:
                fx = 0.18 + 0.64 * (k / max(1, m - 1))
                fy = 0.40 + 0.12 * math.sin(math.pi * (k / max(1, m - 1)))
                fxy = (fx, fy)
            fx = min(0.9, max(0.1, fxy[0]))
            fy = min(0.62, max(0.22, fxy[1]))
            pts.append((name, fx * w, fy * h))
    m = len(pts)                  # geo may resolve fewer stops than requested
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    dx, dy, ctrl = _densify(xs, ys)
    nd = len(dx)

    # map bed (real or synth), dimmed for legibility
    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.45)
            bed = Image.fromarray((np.asarray(bed).astype(np.float32) * 0.6)
                                  .astype("uint8"), "RGB")
        except Exception:                                      # noqa: BLE001
            bed = None
    if bed is None and _geo_bed is not None:
        bed = _geo_bed
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    title_font = look.font("label", int(h * 0.030))
    stop_font = look.font("label", int(h * 0.026))
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    glow_c = tuple(pal["glow"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.04 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # ghost of the full route (faint), draws in early
        ga = look.ease_out_cubic(min(1.0, t / 0.5))
        if ga > 0.01:
            ghost = [(int(a), int(b)) for a, b in zip(dx, dy)]
            d.line(ghost, fill=(*accent, int(60 * ga)), width=2, joint="curve")

        # traced portion up to the comet head
        hp = look.ease_in_out_cubic(min(1.0, t / max(0.1, dur * 0.66)))
        hj = max(1, int(hp * (nd - 1)))
        traced = [(int(a), int(b)) for a, b in zip(dx[:hj + 1], dy[:hj + 1])]
        if len(traced) >= 2:
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line(traced, fill=(*accent_hi, 150), width=9,
                                    joint="curve")
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                gl.filter(ImageFilter.GaussianBlur(5))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            if _dashed:
                # V3.4 dashed_march: SAME path (same coords/order), but a segmented
                # dash texture (supply/advance feel) instead of a solid trace.
                _step = 11
                for _s in range(0, len(traced) - 1, _step):
                    _seg = traced[_s:_s + max(2, _step - 5)]
                    if len(_seg) >= 2:
                        d.line(_seg, fill=(*accent_hi, 255), width=5, joint="curve")
            else:
                d.line(traced, fill=(*accent_hi, 255), width=4, joint="curve")

        # origin pin (filled), drawn early
        oa = look.ease_out_cubic(min(1.0, t / 0.4))
        ox, oy = int(xs[0]), int(ys[0])
        d.ellipse([ox - 9, oy - 9, ox + 9, oy + 9], fill=(*accent_hi, int(255 * oa)))
        d.ellipse([ox - 15, oy - 15, ox + 15, oy + 15],
                  outline=(*accent_hi, int(150 * oa)), width=2)

        # waypoint ticks + names that light as the head passes their control idx
        head_idx = hj
        for k in range(m):
            cx, cy = int(xs[k]), int(ys[k])
            reached = head_idx >= ctrl[k] - 2
            la = 1.0 if reached else 0.0
            name = pts[k][0]
            if 0 < k < m - 1 and la > 0.01:                    # intermediate tick
                d.line([(cx - 7, cy), (cx + 7, cy)], fill=(*accent, 220), width=2)
                d.line([(cx, cy - 7), (cx, cy + 7)], fill=(*accent, 220), width=2)
            if name and la > 0.01:
                ni = look.text_with_glow(name.upper(), stop_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=10)
                frame = look.paste_center(frame, ni, cx=cx,
                                          cy=cy - int(h * 0.045), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")

        # destination pin pulses once the head arrives
        ex, ey = int(xs[-1]), int(ys[-1])
        if head_idx >= nd - 2:
            pls = math.sin(t * 5) * 0.5 + 0.5
            rr = int(h * (0.015 + 0.012 * pls))
            d.ellipse([ex - rr, ey - rr, ex + rr, ey + rr],
                      outline=(*glow_c, int(140 + 90 * pls)), width=3)
            d.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=(*accent_hi, 255))
        # comet head while travelling
        elif len(traced) >= 1:
            hx, hy = traced[-1]
            for rr, al in ((14, 70), (9, 140), (5, 255)):
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr],
                          fill=(*accent_hi, al) if rr == 5 else None,
                          outline=(*accent_hi, al) if rr != 5 else None, width=2)

        # title lower-third
        if title:
            ta = look.ease_out_cubic(max(0.0, (t - 0.4) / 0.6))
            if ta > 0.01:
                look.hairline(d, int(w * 0.5), int(h * 0.80), int(w * 0.1 * ta), pal)
                ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=5,
                                         glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, ti, cx=w * 0.5, cy=int(h * 0.85),
                                          opacity=ta)

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
