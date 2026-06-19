"""Primitive: map_region_highlight.

A "this place on the map" beat — a graded antique/period map fills the frame and
ONE region is singled out: a soft warm gold glow over the area, ringed by a
slightly irregular hand-drawn boundary (a closed ~16-vertex polygon on a radius
with seeded jitter, never a perfect circle) that draws in then gently pulses, a
small pin at the centre, and the region NAME (serif, gold-glow) above with an
optional sub-label beneath. The rest of the map is dimmed/vignetted so the
region reads as the single focus. Distinct from `location_establish_card` (a
point lower-third) and `map_heat_spread` (multiple spreading hotspots): this is
ONE region, ONE glow, ONE boundary — the editorial "right here".

Forensic principle (NOT asset copy): MagnatesMedia frames a place with a graded
map + restrained gold + refined serif type, not a flat coloured blob. Reuses the
shared antique-map bed (or a supplied `map_image`, graded). Pure-local (PIL +
numpy → ffmpeg). No paid API. Deterministic given seed.

    render("x.mp4", region="The Rhineland", pos="0.52,0.42",
           sub="Demilitarized Zone · 1936", title="THE FLASHPOINT OF EUROPE")
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
    "id": "map_region_highlight", "family": "maps",
    "roles": ["region", "territory", "place", "geography", "highlight"],
    "niches_ok": ["history", "geopolitics", "biography", "business", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_region_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["region_glow"],
    "review_override": ["region", "pos", "sub", "map_image", "palette"],
    "fallback": "location_establish_card if a single point reads better than an area",
}


def _parse_pos(pos):
    """Defensively read a region centre as (fx, fy) fractions in [0,1].
    Accepts 'fx,fy' (or 'fx fy' / 'fx;fy'), a [fx, fy] / (fx, fy) sequence, or
    None → frame centre. Out-of-range / garbage values clamp to a safe default."""
    fx, fy = 0.5, 0.45
    try:
        if pos is None:
            pass
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            fx, fy = float(pos[0]), float(pos[1])
        elif isinstance(pos, str):
            s = pos.replace(";", ",").replace("/", ",").replace(" ", ",")
            parts = [p for p in s.split(",") if p.strip() != ""]
            if len(parts) >= 2:
                fx, fy = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        fx, fy = 0.5, 0.45
    # keep the region (and its name above it) comfortably inside the frame
    fx = min(0.86, max(0.14, fx))
    fy = min(0.80, max(0.22, fy))
    return fx, fy


def _boundary_pts(cx, cy, radius, *, seed, n=16, jitter=0.16, squash=0.82):
    """A closed hand-drawn-looking region boundary: `n` vertices on `radius`
    with small seeded per-vertex jitter (NOT a perfect circle), slightly
    ellipse-squashed so it reads like a territory, not a coin."""
    rng = np.random.default_rng(seed * 131 + 7)
    # a fixed seeded radial profile so the SHAPE is stable across all frames
    rfac = 1.0 + jitter * (rng.random(n) * 2 - 1)
    a0 = rng.random() * math.tau                      # seeded rotation
    pts = []
    for k in range(n):
        a = a0 + math.tau * (k / n)
        rr = radius * rfac[k]
        pts.append((cx + rr * math.cos(a),
                    cy + rr * squash * math.sin(a)))
    return pts


def _region_glow(w, h, cx, cy, radius, pal, *, strength=1.0):
    """A soft warm radial bloom (palette glow) centred on the region — an RGBA
    layer to alpha-composite over the dimmed map so the area lifts gently."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / (radius * 1.55)) ** 2 +
                ((yy - cy) / (radius * 1.55)) ** 2)
    fall = np.clip(1.0 - r, 0.0, 1.0) ** 2            # smooth core → 0 at edge
    a = (fall * 168.0 * look.clamp01(strength)).astype(np.float32)
    glow = pal["glow"]
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[..., 0] = glow[0]
    rgba[..., 1] = glow[1]
    rgba[..., 2] = glow[2]
    rgba[..., 3] = a
    img = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    return img.filter(ImageFilter.GaussianBlur(max(6, int(radius * 0.10))))


def _focus_dim(frame, cx, cy, radius, *, keep=0.30):
    """Dim everything EXCEPT a soft pool around the region so the highlight reads
    as the focus. `keep` is the residual brightness far from the region."""
    w, h = frame.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / (radius * 2.3)) ** 2 +
                ((yy - cy) / (radius * 2.3)) ** 2)
    near = np.clip(1.0 - r, 0.0, 1.0)
    m = keep + (1.0 - keep) * near                    # 1.0 at region → keep far
    arr = np.asarray(frame).astype(np.float32) * m[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def render(out_path, *, region: str = "", pos=None, sub: str = "",
           map_image=None, bg_image=None, title: str = "", dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", layout: str = "",
           seed: int = 0, reference: bool = False, borders: bool = True,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    region = (region or "").strip()

    # ── period-map bed: supplied image (graded + darkened) or antique synth ──
    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            mp = Image.open(str(src)).convert("RGB").resize((w, h), Image.LANCZOS)
            bed = look.grade_media(mp, pal, strength=0.45)
            bed = Image.fromarray((np.asarray(bed).astype(np.float32) * 0.6)
                                  .astype("uint8"), "RGB")
        except Exception:                                      # noqa: BLE001
            bed = None
    # REAL geography: when the region resolves to a country, frame + highlight the
    # actual country and centre the focus on its true centroid (else antique synth).
    geo_cxy = None
    if bed is None and region:
        ent = geo.country_entry(region); gll = geo.resolve(region)
        if ent or gll:
            if ent:
                bw_, bs_, be_, bn_ = ent["bbox"]
                bbox = geo.region_bbox([(bw_, bs_), (be_, bn_)], pad_frac=0.28)
                hl = [ent["name"]]
            else:
                bbox = geo.region_bbox([gll], pad_frac=0.5, min_span=12.0); hl = []
            gproj = geo.make_projector(bbox, w, h)
            gbed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                                    seed=seed, proj=gproj, highlight=hl,
                                    reference=reference, borders=borders)
            bed = Image.fromarray((np.asarray(gbed).astype(np.float32) * 0.62)
                                  .astype("uint8"), "RGB")
            geo_cxy = tuple(int(v) for v in gproj(*(gll or ent["centroid"])))
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    if geo_cxy:
        cx, cy = geo_cxy
    else:
        fx, fy = _parse_pos(pos)
        cx, cy = int(w * fx), int(h * fy)
    radius = int(min(w, h) * 0.155)
    # the seeded shape, computed ONCE so it is stable across the whole clip
    shape = _boundary_pts(cx, cy, radius, seed=seed)

    nm_font = look.font("title", int(h * 0.058))
    sb_font = look.font("label", int(h * 0.030))
    ti_font = look.font("label", int(h * 0.026))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # very slow push for 2.5-D life on the map bed
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))

        # gentle breathing pulse used by both glow + boundary once revealed
        pulse = math.sin(t * 2.0) * 0.5 + 0.5

        # focus-dim the rest of the map toward the region
        dim = look.ease_out_cubic(min(1.0, t / 0.7))
        frame = _focus_dim(frame, cx, cy, radius, keep=1.0 - 0.50 * dim)

        # warm region glow (fades in, then breathes)
        ga = look.ease_out_cubic(max(0.0, (t - 0.25) / 0.7))
        if ga > 0.01:
            gstr = ga * (0.80 + 0.20 * pulse)
            glow = _region_glow(w, h, cx, cy, radius, pal, strength=gstr)
            frame = Image.alpha_composite(frame.convert("RGBA"), glow).convert("RGB")

        d = ImageDraw.Draw(frame, "RGBA")

        # hand-drawn boundary: draws in (partial polyline) → closes → pulses
        ba = look.ease_out_cubic(max(0.0, (t - 0.45) / 0.85))
        if ba > 0.01:
            closed = ba >= 0.999
            draw_n = max(2, int(round(len(shape) * min(1.0, ba))))
            seq = shape[:draw_n] + ([shape[0]] if closed else [])
            line_a = int(150 + 90 * (pulse if closed else 1.0))
            # soft outer halo stroke (thicker, low alpha) then the crisp line
            d.line([(int(x), int(y)) for (x, y) in seq],
                   fill=(*pal["glow"], int(line_a * 0.35)), width=8, joint="curve")
            d.line([(int(x), int(y)) for (x, y) in seq],
                   fill=(*pal["accent_hi"], min(255, line_a)), width=3,
                   joint="curve")

        # centre pin (small dot with a breathing ring)
        pa = look.ease_out_cubic(max(0.0, (t - 0.55) / 0.5))
        if pa > 0.01:
            rr = int(h * (0.012 + 0.010 * pulse))
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                      outline=(*pal["glow"], int(120 + 80 * pulse)), width=3)
            dot = int(h * 0.008)
            d.ellipse([cx - dot, cy - dot, cx + dot, cy + dot],
                      fill=(*pal["accent_hi"], int(255 * pa)))

        # region NAME above the area (serif, gold glow) + optional sub beneath
        la = look.ease_out_cubic(max(0.0, (t - 0.65) / 0.6))
        if la > 0.01 and region:
            name_y = cy - radius - int(h * 0.075)
            name_y = max(int(h * 0.085), name_y)       # never clip off the top
            ni = look.text_with_glow(region, nm_font, fill=pal["text"],
                                     glow=pal["glow"], glow_radius=int(h * 0.012),
                                     glow_alpha=0.5 * la, pad=28)
            frame = look.paste_center(frame, ni, cx=cx, cy=name_y, opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")
            if sub:
                si = look.text_with_glow(sub.upper(), sb_font, fill=pal["accent_hi"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, si, cx=cx,
                                          cy=cy + radius + int(h * 0.062),
                                          opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")

        # optional title riding a hairline near the bottom
        if title:
            tia = look.ease_out_cubic(min(1.0, t / 0.6))
            ty = int(h * 0.90)
            look.hairline(d, int(w * 0.5), ty - int(h * 0.046),
                          int(w * 0.16 * tia), pal)
            tii = look.text_with_glow(title.upper(), ti_font, fill=pal["muted"],
                                      glow=pal["bg_b"], glow_radius=3,
                                      glow_alpha=0.0, pad=14)
            frame = look.paste_center(frame, tii, cx=w * 0.5, cy=ty, opacity=tia)

        frame = look.vignette(frame, strength=0.62)
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
