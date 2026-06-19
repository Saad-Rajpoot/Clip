"""Primitive: world_map_arc.

An "across the world" beat — a single graceful great-circle ARC sweeps from an
ORIGIN city to a distant DESTINATION city over a graded period world map. The arc
is a bezier that BOWS upward (control lifted above the chord) so it reads as a
curved global path, not a straight line. It DRAWS IN progressively
(easeInOutCubic) with a bright comet head riding the leading edge and a soft glow
trail; pulsing pin dots sit at both endpoints with gold city-name labels placed
clear of the arc, and an optional title rides a centred hairline near the bottom.

Distinct from `map_route_spread` (a flat multi-stop polyline through 3-5 pins)
and `location_establish_card` (a single static place): this is exactly TWO
distant points joined by ONE bowed great-circle link — a transatlantic cable, a
flight, a wire, a connection across the globe. Restraint: one arc, two pins, two
labels.

Forensic principle (NOT asset copy): MagnatesMedia draws a connection across a
period map as a single luminous curve, not a straight rule; the premium is the
grade + smooth bowed arc + comet head + restrained pins/labels. Reuses the
maps-family antique synth (or a supplied map image). Pure-local (PIL + numpy →
ffmpeg). No paid API. Deterministic given seed.

    render("x.mp4", from_place="London", from_pos="0.49,0.33",
           to_place="New York", to_pos="0.27,0.42",
           title="THE TRANSATLANTIC CABLE")
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
    "id": "world_map_arc", "family": "maps",
    "roles": ["connection", "arc", "route", "link", "global"],
    "niches_ok": ["history", "geopolitics", "business", "crime", "biography", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "soft_arc_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["great_circle"],
    "review_override": ["from_place", "to_place", "from_pos", "to_pos",
                        "map_image", "palette"],
    "fallback": "map_route_spread if there are more than two waypoints",
}


def _parse_pos(pos, default):
    """Defensive 'fx,fy' fraction string or [fx,fy] → (fx, fy), clamped to the
    on-frame safe band 0.06-0.94. Falls back to `default` on any bad input."""
    fx, fy = default
    try:
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            fx, fy = float(pos[0]), float(pos[1])
        elif isinstance(pos, dict):
            fx = float(pos.get("x", pos.get("fx", fx)))
            fy = float(pos.get("y", pos.get("fy", fy)))
        elif isinstance(pos, str) and "," in pos:
            a, b = pos.replace(" ", "").split(",")[:2]
            fx, fy = float(a), float(b)
    except Exception:                                          # noqa: BLE001
        fx, fy = default
    if not (math.isfinite(fx) and math.isfinite(fy)):
        fx, fy = default
    fx = min(0.94, max(0.06, fx))
    fy = min(0.94, max(0.06, fy))
    return fx, fy


def _bezier(p0, p1, p2, steps=120):
    """Quadratic bezier p0→(ctrl p1)→p2, returned as dense (xs, ys) arrays."""
    t = np.linspace(0.0, 1.0, steps)
    mt = 1.0 - t
    xs = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
    ys = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
    return xs, ys


def _world_bed(w, h, seed, pal):
    """Purpose-built antique WORLD-map paper for the great-circle arc.

    A clean lat/long graticule over softly + evenly mottled aged parchment, with
    the equator and prime meridian *barely* emphasised so the bed reads as a world
    chart rather than graph paper. Deliberately omits the distinct 'landmass blobs'
    the establishing-card synth paints: at full-frame world scale those blobs read
    as ink stains and fight the arc. Here we want quiet, even paper so the single
    luminous curve owns the frame. Kept local (no coupling to stable primitives).
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (198, 180, 145)
    # gentle low-frequency mottling — upscaled blurred noise, very low contrast,
    # so the paper looks aged and uneven without any readable 'shapes'
    small = rng.random((max(2, h // 20), max(2, w // 20))).astype(np.float32)
    mott = np.asarray(Image.fromarray((small * 255).astype("uint8"))
                      .resize((w, h), Image.BICUBIC), np.float32) / 255.0
    base += (mott[:, :, None] - 0.5) * 13.0                     # ±~6.5 luma
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "RGB")
    d = ImageDraw.Draw(img, "RGBA")
    grid = (168, 148, 112)
    for gx in range(0, w + 1, max(1, w // 16)):                 # meridians
        d.line([(gx, 0), (gx, h)], fill=(*grid, 115), width=1)
    for gy in range(0, h + 1, max(1, h // 9)):                  # parallels
        d.line([(0, gy), (w, gy)], fill=(*grid, 115), width=1)
    d.line([(0, h // 2), (w, h // 2)], fill=(150, 128, 92, 140), width=2)  # equator
    d.line([(w // 2, 0), (w // 2, h)], fill=(150, 128, 92, 120), width=2)  # meridian
    img = img.filter(ImageFilter.GaussianBlur(0.5))            # knock off aliasing
    img = look.grade_media(img, pal, strength=0.34)
    arr = (np.asarray(img).astype(np.float32) * 0.6).astype("uint8")
    return Image.fromarray(arr, "RGB")


def render(out_path, *, from_place: str = "", to_place: str = "",
           from_pos=None, to_pos=None, map_image=None, bg_image=None,
           title: str = "", dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, reference: bool = False,
           borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    from_place = (str(from_place or "").strip()) or "Origin"
    to_place = (str(to_place or "").strip()) or "Destination"
    fx, fy = _parse_pos(from_pos, (0.24, 0.40))
    tx, ty = _parse_pos(to_pos, (0.76, 0.44))
    ox, oy = fx * w, fy * h                                     # origin px
    dx_, dy_ = tx * w, ty * h                                   # destination px

    # REAL geography: project both endpoints at TRUE positions on a real bed
    # framed to span them (else the normalized from_pos/to_pos fallback above).
    _geo_bed = None
    _a = geo.resolve(from_place); _b = geo.resolve(to_place)
    if (map_image or bg_image) is None and _a and _b:
        _bbox = geo.region_bbox([_a, _b], pad_frac=0.55, min_span=20.0)
        _gp = geo.make_projector(_bbox, w, h)
        _gb = geo.draw_basemap(w, h, _bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=_gp, graticule=True,
                               reference=reference, borders=borders)
        _geo_bed = Image.fromarray((np.asarray(_gb).astype(np.float32) * 0.58)
                                   .astype("uint8"), "RGB")
        ox, oy = _gp(*_a); dx_, dy_ = _gp(*_b)

    # ── great-circle bow: lift a control point above the chord midpoint by a
    # fraction of the chord length (perpendicular, biased upward) so the arc
    # bows up into the northern part of the frame like a globe path.
    mx, my = (ox + dx_) * 0.5, (oy + dy_) * 0.5
    chord = math.hypot(dx_ - ox, dy_ - oy) or 1.0
    # perpendicular unit vector, forced to point UP the frame (negative y)
    nx, ny = -(dy_ - oy) / chord, (dx_ - ox) / chord
    if ny > 0:
        nx, ny = -nx, -ny
    bow = chord * 0.22                                          # ~22% of chord
    cx_ctrl = mx + nx * bow
    cy_ctrl = my + ny * bow
    # keep the apex on-frame (a near-horizontal long chord could push it off top)
    cy_ctrl = max(h * 0.07, cy_ctrl)
    cx_ctrl = min(w * 0.93, max(w * 0.07, cx_ctrl))

    xs, ys = _bezier((ox, oy), (cx_ctrl, cy_ctrl), (dx_, dy_), steps=140)
    nd = len(xs)

    # ── map bed (real or synth), dimmed + vignetted for legibility
    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.45)
            bed = Image.fromarray((np.asarray(bed).astype(np.float32) * 0.58)
                                  .astype("uint8"), "RGB")
        except Exception:                                      # noqa: BLE001
            bed = None
    if bed is None and _geo_bed is not None:
        bed = _geo_bed
    if bed is None:
        bed = _world_bed(w, h, seed, pal)

    title_font = look.font("label", int(h * 0.030))
    name_font = look.font("title", int(h * 0.040))
    accent = tuple(int(v) for v in pal["accent"])
    accent_hi = tuple(int(v) for v in pal["accent_hi"])
    glow_c = tuple(int(v) for v in pal["glow"])

    # label placement: push each name vertically AWAY from the arc apex so it
    # never sits on the curve (apex bows up → labels sit below their pins).
    def _label_xy(px, py):
        ly = py + int(h * 0.052)
        if ly > h * 0.90:                                      # too low → flip up
            ly = py - int(h * 0.052)
        lx = min(w * 0.86, max(w * 0.14, px))
        return lx, ly

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # slow push-in on the bed
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # faint ghost of the whole arc, draws in early (sets the destination)
        ga = look.ease_out_cubic(min(1.0, t / 0.5))
        if ga > 0.01:
            ghost = [(int(a), int(b)) for a, b in zip(xs, ys)]
            d.line(ghost, fill=(*accent, int(52 * ga)), width=2, joint="curve")

        # arc draws in progressively up to the comet head (easeInOutCubic)
        hp = look.ease_in_out_cubic(min(1.0, t / max(0.1, dur * 0.62)))
        hj = max(1, int(hp * (nd - 1)))
        traced = [(int(a), int(b)) for a, b in zip(xs[:hj + 1], ys[:hj + 1])]
        if len(traced) >= 2:
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line(traced, fill=(*accent_hi, 150), width=10,
                                    joint="curve")
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                gl.filter(ImageFilter.GaussianBlur(6))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.line(traced, fill=(*accent_hi, 255), width=4, joint="curve")

        # origin pin (filled), drawn early with a soft pulse
        oa = look.ease_out_cubic(min(1.0, t / 0.4))
        ip = math.sin(t * 4.0) * 0.5 + 0.5
        orr = int(h * (0.012 + 0.006 * ip))
        d.ellipse([int(ox) - orr, int(oy) - orr, int(ox) + orr, int(oy) + orr],
                  outline=(*glow_c, int((110 + 90 * ip) * oa)), width=3)
        d.ellipse([int(ox) - 8, int(oy) - 8, int(ox) + 8, int(oy) + 8],
                  fill=(*accent_hi, int(255 * oa)))

        # destination pin pulses once the head arrives; comet head while travelling
        arrived = hj >= nd - 2
        if arrived:
            pls = math.sin(t * 5.0) * 0.5 + 0.5
            rr = int(h * (0.015 + 0.012 * pls))
            d.ellipse([int(dx_) - rr, int(dy_) - rr, int(dx_) + rr, int(dy_) + rr],
                      outline=(*glow_c, int(140 + 90 * pls)), width=3)
            d.ellipse([int(dx_) - 8, int(dy_) - 8, int(dx_) + 8, int(dy_) + 8],
                      fill=(*accent_hi, 255))
        elif len(traced) >= 1:
            hx, hy = traced[-1]
            # soft comet glow then the bright core
            cg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(cg).ellipse(
                [hx - 18, hy - 18, hx + 18, hy + 18], fill=(*accent_hi, 150))
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                cg.filter(ImageFilter.GaussianBlur(8))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.ellipse([hx - 9, hy - 9, hx + 9, hy + 9], outline=(*accent_hi, 150),
                      width=2)
            d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=(*accent_hi, 255))

        # city-name labels (gold, with glow) placed clear of the arc
        oa_lab = look.ease_out_cubic(min(1.0, t / 0.5))
        if oa_lab > 0.01:
            lx, ly = _label_xy(ox, oy)
            ni = look.text_with_glow(from_place.upper(), name_font,
                                     fill=pal["accent_hi"], glow=pal["glow"],
                                     glow_radius=int(h * 0.010), glow_alpha=0.45,
                                     pad=22)
            frame = look.paste_center(frame, ni, cx=lx, cy=ly, opacity=oa_lab)
            d = ImageDraw.Draw(frame, "RGBA")
        # destination label appears as the head nears / arrives
        da_lab = look.ease_out_cubic(min(1.0, max(0.0, (hp - 0.72) / 0.22)))
        if da_lab > 0.01:
            lx, ly = _label_xy(dx_, dy_)
            ni = look.text_with_glow(to_place.upper(), name_font,
                                     fill=pal["accent_hi"], glow=pal["glow"],
                                     glow_radius=int(h * 0.010), glow_alpha=0.45,
                                     pad=22)
            frame = look.paste_center(frame, ni, cx=lx, cy=ly, opacity=da_lab)
            d = ImageDraw.Draw(frame, "RGBA")

        # title on a centred hairline near the bottom
        if title:
            ta = look.ease_out_cubic(max(0.0, (t - 0.4) / 0.6))
            if ta > 0.01:
                look.hairline(d, int(w * 0.5), int(h * 0.855),
                              int(w * 0.11 * ta), pal)
                ti = look.text_with_glow(str(title).upper(), title_font,
                                         fill=pal["text"], glow=pal["bg_b"],
                                         glow_radius=5, glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, ti, cx=w * 0.5,
                                          cy=int(h * 0.905), opacity=ta)

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
