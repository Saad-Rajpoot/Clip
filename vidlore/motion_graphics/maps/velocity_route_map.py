"""Primitive: velocity_route_map.

A velocity / journey beat — a cool SATELLITE-style base (deliberately NOT the
parchment look, so map sequences vary) while a thick route line DRAWS across a
chain of stops, led by a glowing marker that travels fast then DECELERATES into
the destination. Stop labels pop as the marker passes. The grammar of "it moved
from A through B to C — fast" — shipping lanes, expansion, a chase, a launch
trajectory.

Distinct from `map_route_spread` (parchment comet route) and `supply_route_dashes`
(thin dashed one-to-many flow): this is a satellite-bed single hero journey with
a velocity feel and decelerate-on-arrival.

Forensic principle (NOT an asset copy): premium systems/logistics storytelling
draws a thick route on a satellite base with a decelerating head + node labels.
Rebuilt from that PRINCIPLE with Vidlore's own look system — no reference asset,
colour or layout reproduced. Pure-local, deterministic.

    render("x.mp4", stops=["SHANGHAI","SUEZ","ROTTERDAM"])
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

SPEC = {
    "id": "velocity_route_map", "family": "maps",
    "roles": ["journey", "velocity", "route", "shipping", "trajectory", "expansion"],
    "niches_ok": ["business", "tech", "geopolitics", "history", "science"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "whoosh_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["ltr", "rtl"],
    "review_override": ["stops", "title", "map_image", "palette"],
    "fallback": "map_route_spread for a parchment route",
    "required_inputs": ["stops"],
}


def _satellite_bed(w, h, seed, pal):
    """A cool, dark satellite-style base: terrain mottling + faint graticule."""
    rng = np.random.RandomState((seed * 22695477 + 1) & 0xFFFFFFFF)
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 14; base[..., 1] = 22; base[..., 2] = 30      # deep blue-green
    # multi-octave value noise → terrain
    acc = np.zeros((h, w), np.float32)
    for oct_ in (6, 14, 30):
        small = rng.rand(oct_, oct_).astype(np.float32)
        up = np.asarray(Image.fromarray((small * 255).astype("uint8")).resize(
            (w, h), Image.BILINEAR), np.float32) / 255.0
        acc += up / oct_
    acc = (acc - acc.min()) / ((acc.max() - acc.min()) + 1e-6)
    land = acc > 0.52
    base[land] += np.array([18, 22, 14], np.float32)            # landmass lift (olive)
    base += (acc[..., None] - 0.5) * 22.0
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "RGB")
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    d = ImageDraw.Draw(img, "RGBA")
    for gx in range(0, w, w // 12):
        d.line([(gx, 0), (gx, h)], fill=(120, 150, 170, 26), width=1)
    for gy in range(0, h, h // 7):
        d.line([(0, gy), (w, gy)], fill=(120, 150, 170, 26), width=1)
    return img


def _catmull(pts, samples=24):
    if len(pts) < 3:
        return pts
    p = [pts[0]] + list(pts) + [pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for s in range(samples):
            tt = s / samples
            t2, t3 = tt * tt, tt * tt * tt
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * tt +
                       (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                       (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * tt +
                       (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                       (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-1])
    return out


def render(out_path, *, stops=None, title: str = "", map_image=None, bg_image=None,
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           reference: bool = False, borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    stops = [str(x).strip() for x in (stops or []) if str(x).strip()][:5] or ["A", "B", "C"]
    rtl = layout == "rtl"

    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.45)
        except Exception:                                          # noqa: BLE001
            bed = None
    # REAL geography: project the journey stops (in order) onto a satellite bed
    # framed to the region. Keep only resolvable stops; ≥2 needed for a route.
    geo_stops = [(s, geo.resolve(s)) for s in stops] if bed is None else []
    geo_stops = [(s, ll) for s, ll in geo_stops if ll]
    nodes = None
    if bed is None and len(geo_stops) >= 2:
        bbox = geo.region_bbox([ll for _, ll in geo_stops], pad_frac=0.45)
        gproj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name, "satellite"),
                               seed=seed, proj=gproj, graticule=True,
                               reference=reference, borders=borders)
        nodes = [gproj(*ll) for _, ll in geo_stops]
        stops = [s for s, _ in geo_stops]
    if bed is None:
        bed = _satellite_bed(w, h, seed, pal)
    if nodes is None:
        rng = np.random.RandomState((seed * 2654435761 + 11) & 0xFFFFFFFF)
        k = len(stops)
        nodes = []
        for i in range(k):
            fx = 0.12 + 0.76 * (i / max(1, k - 1))
            if rtl:
                fx = 1.0 - fx
            fy = 0.34 + 0.34 * rng.rand()
            nodes.append((fx * w, fy * h))
    k = len(nodes)
    path = _catmull(nodes, samples=26)
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    label_font = look.font("label", int(h * 0.027))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.04 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # route head: fast then decelerate (ease_out) over 0.5–end-0.6
        draw_p = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.5) / (dur - 1.1))))
        nseg = max(1, int(len(path) * draw_p))
        seg = [(int(x), int(y)) for (x, y) in path[:nseg + 1]]
        if len(seg) >= 2:
            # glow underlay
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line(seg, fill=(*accent_hi, 90), width=12, joint="curve")
            gl = gl.filter(ImageFilter.GaussianBlur(7))
            frame = Image.alpha_composite(frame.convert("RGBA"), gl).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.line(seg, fill=(*accent_hi, 240), width=5, joint="curve")
            # marker head
            hx, hy = seg[-1]
            d.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(*accent_hi, 255))
            d.ellipse([hx - 14, hy - 14, hx + 14, hy + 14],
                      outline=(*accent_hi, 150), width=2)

        # node dots + labels pop as the head passes each
        for j, nd in enumerate(nodes):
            node_frac = j / max(1, k - 1)
            passed = draw_p >= node_frac - 0.01
            op = 1.0 if passed else 0.0
            if op:
                nx, ny = int(nd[0]), int(nd[1])
                d.ellipse([nx - 6, ny - 6, nx + 6, ny + 6], fill=(*accent, 240),
                          outline=(*pal["bg_b"], 200), width=2)
                if j < len(stops):
                    frame = geo.place_label(frame, stops[j], nx, ny, pal, dy=-26)
                d = ImageDraw.Draw(frame, "RGBA")

        if title:
            ti = max(0.0, min(1.0, (t - 0.1) / 0.5))
            chip = look.text_with_glow(title.upper(), label_font, fill=pal["text"],
                                       glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=12)
            cw, ch = chip.width, chip.height
            bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(bar).rounded_rectangle(
                [(w - cw) // 2 - 24, int(h * 0.07) - 8, (w + cw) // 2 + 24,
                 int(h * 0.07) + ch + 8], radius=10, fill=(*pal["bg_b"], int(208 * ti)))
            frame = Image.alpha_composite(frame.convert("RGBA"), bar).convert("RGB")
            frame = look.paste_center(frame, chip, cx=w // 2,
                                      cy=int(h * 0.07) + ch // 2, opacity=ti)

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
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
