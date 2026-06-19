"""Primitive: footage_route_trace  (Section C · hybrid).

Grounded in a REAL frame audit — Wendover "Cities at Sea: How Aircraft Carriers
Work" 04:08–04:16 (`reference_videos/frame_sequences/wendover/
02_map_route_singapore/`): aerial/satellite TERRAIN footage stays the foundation
while a thick accent route is PROGRESSIVELY DRAWN between two places — a single
travelling head crawls along the polyline, a junction dot and endpoint PINS land,
and place labels pop on scrim plates as the head reaches them (the boxes-then-map
dissolve at frame 12→16, the Louisville→New York draw at 19→21). The footage is
visible underneath the whole time; the route is an OVERLAY, not a map slide.

It is NOT a standalone map card with a legend / north-arrow / mini-dashboard
(that is the infographic-template language every premium reference avoids). The
footage is the subject; the route is the annotation traced over it.

Original Vidlore design — Wendover's PRINCIPLE only (trace a path over footage
with a travelling head + sequential pins + labels), rebuilt on the charcoal
`look` system with our palette, easing, glow and grain. No asset/layout/colour
copied. Pure-local, deterministic, footage-first, no paid API.

    render("x.mp4", points=[{"x":0.2,"y":0.7,"label":"Start"},
                            {"x":0.5,"y":0.45},
                            {"x":0.8,"y":0.3,"label":"End"}],
           bg_image="aerial.jpg")          # bg_image optional
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "footage_route_trace", "family": "hybrid",
    "roles": ["route", "path", "journey", "trace", "supply_line", "corridor",
              "voyage", "flight_path", "connection", "from_to", "trail"],
    "niches_ok": ["geopolitics", "history", "business", "tech", "science"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeInOutCubic", "audio_cue": "draw_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "full_screen": "overlay",                  # drawn OVER footage, not a card
    "layout_variants": ["overlay"],
    "review_override": ["points", "bg_image", "palette"],
    "fallback": "system_planview_flow for an abstract from→to path with no footage",
    "required_inputs": ["points"],
    "accepts_media": ["bg_image"],
    "grounded_in": "wendover/02_map_route_singapore (04:08-04:16)",
}


# ───────────────────────── inputs ─────────────────────────
def _norm_points(points):
    """Normalise points to [{x,y,label}] with x,y in [0,1]. Accepts dicts
    ({x,y,label?} or {x,y,name?}) and [x,y[,label]] pairs. Keeps 2–5 in order."""
    out = []
    for p in (points or []):
        x = y = None
        lbl = ""
        if isinstance(p, dict):
            x = p.get("x"); y = p.get("y")
            lbl = str(p.get("label") or p.get("name") or "").strip()
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
            if len(p) >= 3 and p[2] is not None:
                lbl = str(p[2]).strip()
        try:
            x = look.clamp01(float(x)); y = look.clamp01(float(y))
        except (TypeError, ValueError):
            continue
        out.append({"x": x, "y": y, "label": lbl})
    return out[:5]


def _catmull_rom(pts, *, samples_per_seg: int = 28):
    """Smooth the waypoint polyline through every control point with a
    centripetal-ish Catmull-Rom spline → a gentle curve (the route 'follows'
    rather than zig-zags). `pts` are (x,y) pixel tuples; returns a dense list of
    (x,y) the head crawls along. With 2 points this is a straight line."""
    if len(pts) < 2:
        return list(pts)
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [(x0 + (x1 - x0) * (i / samples_per_seg),
                 y0 + (y1 - y0) * (i / samples_per_seg))
                for i in range(samples_per_seg + 1)]
    ext = [pts[0]] + list(pts) + [pts[-1]]            # phantom endpoints
    dense = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(samples_per_seg + 1):
            t = s / samples_per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                       + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                       + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            if dense and s == 0:                       # avoid duplicate joints
                continue
            dense.append((x, y))
    return dense


def _cumlen(path):
    """Cumulative arc length (px) along the dense path; returns (cum, total)."""
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + math.dist(path[i - 1], path[i]))
    return cum, (cum[-1] if cum else 0.0)


def _point_at(path, cum, total, frac):
    """Interpolated (x,y) at arc-length fraction `frac` in [0,1] along the path,
    plus the path index just consumed (so callers can know which leg we're on)."""
    if total <= 0 or len(path) < 2:
        return path[0] if path else (0.0, 0.0), 0
    target = look.clamp01(frac) * total
    lo, hi = 0, len(cum) - 1
    while lo < hi:                                     # binary search the segment
        mid = (lo + hi) // 2
        if cum[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    j = max(1, lo)
    seg = cum[j] - cum[j - 1]
    u = 0.0 if seg <= 1e-6 else (target - cum[j - 1]) / seg
    x = path[j - 1][0] + (path[j][0] - path[j - 1][0]) * u
    y = path[j - 1][1] + (path[j][1] - path[j - 1][1]) * u
    return (x, y), j


# ───────────────────────── simulated terrain bed ─────────────────────────
def _terrain_bed(w, h, pal, *, seed):
    """A neutral aerial/landscape-ish bed to simulate footage when no bg_image
    is given — a graded charcoal base with a few soft tonal land-masses and a
    faint coastal edge, so the route has somewhere to live. NOT a map (no
    borders, grid, labels) — just a textured foundation that reads as terrain."""
    if hasattr(look, "graded_background"):
        bed = look.graded_background(w, h, pal, seed=seed, drift=0.2,
                                     floor=look.CARD_STAGE_FLOOR
                                     if hasattr(look, "CARD_STAGE_FLOOR") else 0.0)
    else:
        bed = Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    # soft tonal "land" blobs (slightly warmer/lighter than the cold bed)
    blob = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blob)
    import random as _r
    rng = _r.Random(seed * 7 + 11)
    land = tuple(int(c) for c in (
        min(255, pal["bg_a"][0] + 16), min(255, pal["bg_a"][1] + 22),
        min(255, pal["bg_a"][2] + 14)))
    for _ in range(7):
        cx = rng.uniform(0.08, 0.92) * w
        cy = rng.uniform(0.10, 0.90) * h
        rw = rng.uniform(0.14, 0.34) * w
        rh = rng.uniform(0.12, 0.30) * h
        a = rng.randint(26, 54)
        bd.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(*land, a))
    blob = blob.filter(ImageFilter.GaussianBlur(int(h * 0.05)))
    bed = bed.convert("RGBA")
    bed.alpha_composite(blob)
    bed = bed.convert("RGB")
    return bed


def _footage_bed(bg_image, w, h, pal):
    """Compose a real aerial/landscape frame as the foundation: cover-fit + graded
    toward the palette so the overlay route reads cleanly on top. Returns None on
    any failure so the caller falls back to the simulated bed."""
    try:
        p = Path(str(bg_image))
        if not p.exists():
            return None
        im = Image.open(p).convert("RGB")
        sw, sh = im.size
        s = max(w / sw, h / sh)
        im = im.resize((max(w, int(sw * s)), max(h, int(sh * s))), Image.LANCZOS)
        x = (im.width - w) // 2
        y = (im.height - h) // 2
        im = im.crop((x, y, x + w, y + h))
        if hasattr(look, "grade_media"):
            im = look.grade_media(im, pal, strength=0.55)
        # very gentle darken only — the gold route self-glows and is opaque, so it
        # pops without crushing the bed. Keep footage clearly visible (it is the
        # subject; the route is an annotation traced OVER it, not a scrim card).
        from PIL import ImageEnhance
        im = ImageEnhance.Brightness(im).enhance(0.90)
        return im
    except Exception:                                  # noqa: BLE001
        return None


# ───────────────────────── pins / labels ─────────────────────────
def _draw_pin(d, x, y, *, r, ring, fill, ring_col, alpha):
    a = int(255 * look.clamp01(alpha))
    if a <= 1:
        return
    d.ellipse([x - ring, y - ring, x + ring, y + ring],
              outline=(*ring_col, int(a * 0.55)), width=2)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(*fill, a))
    d.ellipse([x - r, y - r, x + r, y + r], outline=(*ring_col, a), width=2)


def _label_plate(frame, text, font, x, y, *, pal, accent, alpha, above):
    """A small scrim plate + label near a waypoint (the Wendover pin-plate),
    offset above or below the point so it never sits on the pin. Pushed inside
    the frame so it never clips an edge."""
    if not text or alpha <= 0.02:
        return frame
    ink = tuple(pal["text"])
    lay = look.text_with_glow(text, font, fill=ink, glow=pal["bg_b"],
                              glow_radius=3, glow_alpha=0.0, pad=6)
    W, H = frame.size
    tw, th = lay.width, lay.height
    padx, pady = int(th * 0.5), int(th * 0.28)
    pw, ph = tw + padx * 2, th + pady * 2
    cx = int(x)
    cy = int(y - th * 1.7) if above else int(y + th * 1.7)
    # keep plate fully on-screen
    cx = max(pw // 2 + 8, min(W - pw // 2 - 8, cx))
    cy = max(ph // 2 + 8, min(H - ph // 2 - 8, cy))
    plate = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    sc = tuple(int(c * 0.45) for c in pal["bg_b"])
    pa = int(190 * look.clamp01(alpha))
    pd.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(ph * 0.32),
                         fill=(*sc, pa))
    pd.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(ph * 0.32),
                         outline=(*accent, int(150 * look.clamp01(alpha))), width=2)
    frame = frame.convert("RGBA")
    frame.alpha_composite(plate, (cx - pw // 2, cy - ph // 2))
    frame = frame.convert("RGB")
    frame = look.paste_center(frame, lay, cx=cx, cy=cy, opacity=look.clamp01(alpha))
    # tick from plate to the point
    d = ImageDraw.Draw(frame, "RGBA")
    ty = cy + (ph // 2 if above else -ph // 2)
    d.line([(cx, ty), (int(x), int(y))], fill=(*accent, int(150 * look.clamp01(alpha))),
           width=2)
    return frame


# ───────────────────────── glowing progressive polyline ─────────────────────
def _draw_route(frame, path, cum, total, reach, *, accent, accent_hi, glow):
    """Draw the polyline up to arc-length fraction `reach`, with a soft outer
    glow under a crisp accent core (rounded joints), plus a glowing travelling
    HEAD at the frontier. Returns (frame, head_xy)."""
    if total <= 0 or len(path) < 2:
        return frame, (path[0] if path else (0, 0))
    head, _ = _point_at(path, cum, total, reach)
    # build the visible sub-path (all full segments below reach + the partial)
    target = look.clamp01(reach) * total
    pl = [path[0]]
    for i in range(1, len(path)):
        if cum[i] <= target:
            pl.append(path[i])
        else:
            break
    if pl[-1] != head:
        pl.append(head)

    W, H = frame.size
    # outer glow on its own layer, blurred — a confident bloom under the band
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_layer)
    gd.line(pl, fill=(*glow, 175), width=22, joint="curve")
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(9))
    frame = frame.convert("RGBA")
    frame.alpha_composite(glow_layer)
    frame = frame.convert("RGB")

    d = ImageDraw.Draw(frame, "RGBA")
    # Solid bold band: step uniformly along the visible arc (px) and stamp
    # overlapping discs → a TRUE continuous stroke independent of PIL's curve
    # joints (which bead a dense control list). Bright thin core on top.
    body_r, core_r = 6.0, 2.4
    sub_cum, sub_total = _cumlen(pl)
    step = 2.0                                          # px between stamps (overlap)
    s = 0.0
    while s <= sub_total:
        (sx, sy), _ = _point_at(pl, sub_cum, sub_total, s / sub_total if sub_total else 0)
        d.ellipse([sx - body_r, sy - body_r, sx + body_r, sy + body_r],
                  fill=(*accent, 245))
        s += step
        if sub_total <= 0:
            break
    d.line(pl, fill=(*accent_hi, 250), width=int(core_r * 2), joint="curve")
    # crisp end-cap on the body at the head so the band meets the head cleanly
    d.ellipse([head[0] - body_r, head[1] - body_r, head[0] + body_r, head[1] + body_r],
              fill=(*accent, 245))

    # travelling head: bright dot + soft halo (the moving cap in the reference)
    hx, hy = head
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    hd.ellipse([hx - 28, hy - 28, hx + 28, hy + 28], fill=(*glow, 185))
    halo = halo.filter(ImageFilter.GaussianBlur(11))
    frame = frame.convert("RGBA")
    frame.alpha_composite(halo)
    frame = frame.convert("RGB")
    d = ImageDraw.Draw(frame, "RGBA")
    d.ellipse([hx - 15, hy - 15, hx + 15, hy + 15], outline=(*accent_hi, 220), width=3)
    d.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(*accent_hi, 255))
    return frame, head


# ───────────────────────── render ─────────────────────────
def render(out_path, *, points=None, bg_image: str | None = None, title: str = "",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    pts = _norm_points(points)
    if len(pts) < 2:                                   # safe default journey
        pts = [{"x": 0.22, "y": 0.70, "label": "Start"},
               {"x": 0.52, "y": 0.46, "label": ""},
               {"x": 0.80, "y": 0.30, "label": "End"}]

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    glow = tuple(pal["glow"])

    # foundation: real footage frame if provided + valid, else simulated terrain
    bed = _footage_bed(bg_image, w, h, pal) if bg_image else None
    simulated = bed is None
    if bed is None:
        bed = _terrain_bed(w, h, pal, seed=seed)

    # waypoints → pixels (kept inside a safe margin so pins/labels never clip)
    mx, my = int(w * 0.07), int(h * 0.10)
    px = [(int(mx + p["x"] * (w - 2 * mx)), int(my + p["y"] * (h - 2 * my)))
          for p in pts]
    path = _catmull_rom(px)
    cum, total = _cumlen(path)
    # arc-length fraction of each control point (for sequential pin/label reveal)
    ctrl_frac = []
    for (cx, cy) in px:
        best_i = min(range(len(path)), key=lambda i: math.dist(path[i], (cx, cy)))
        ctrl_frac.append((cum[best_i] / total) if total > 0 else 0.0)
    ctrl_frac[0], ctrl_frac[-1] = 0.0, 1.0

    label_font = look.font("label", int(h * 0.026))
    title_font = look.font("title", int(h * 0.034))

    # timeline: brief settle, draw over the MIDDLE, then hold (then global fade)
    draw_t0 = max(0.6, dur * 0.16)
    draw_t1 = min(dur - 0.9, dur * 0.66)
    if draw_t1 <= draw_t0:
        draw_t1 = draw_t0 + 1.0

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()

        # optional small title chip top-left (kept minimal — footage is the star).
        # A soft DARK glow behind the bright title acts as a legibility shadow so
        # it pops off the footage as cleanly as the point labels read on their dark
        # chips — no plate/HUD, just a restrained bloom (the glow was dormant at
        # alpha 0, leaving the title low-contrast dim over a mid-bright bed).
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            if ta > 0.01:
                ti = look.text_with_glow(title.upper(), title_font,
                                         fill=tuple(pal["text"]), glow=pal["bg_b"],
                                         glow_radius=7, glow_alpha=0.85, pad=14)
                frame = look.paste_center(frame, ti, cx=int(mx + ti.width * 0.5),
                                          cy=int(h * 0.10), opacity=ta)

        # how far the route has drawn (eased over the draw window, holds at 1)
        if t < draw_t0:
            reach = 0.0
        elif t >= draw_t1:
            reach = 1.0
        else:
            reach = look.ease_in_out_cubic((t - draw_t0) / (draw_t1 - draw_t0))

        # start pin drops in just before the draw begins
        sa = look.ease_out_cubic(min(1.0, max(0.0, (t - (draw_t0 - 0.35)) / 0.35)))
        d = ImageDraw.Draw(frame, "RGBA")
        _draw_pin(d, px[0][0], px[0][1], r=10, ring=18, fill=accent_hi,
                  ring_col=accent, alpha=sa)

        # the progressive route + travelling head
        head_xy = px[0]
        if reach > 0.0:
            frame, head_xy = _draw_route(frame, path, cum, total, reach,
                                         accent=accent, accent_hi=accent_hi, glow=glow)
            d = ImageDraw.Draw(frame, "RGBA")

        # intermediate waypoint dots + end pin reveal as the head passes them
        for ci in range(1, len(px)):
            passed = reach >= ctrl_frac[ci] - 1e-3
            pa = 1.0 if passed else 0.0
            if pa <= 0.01:
                continue
            if ci == len(px) - 1:                      # END pin (bigger)
                # pop scale as it lands
                arrive = look.ease_out_cubic(
                    min(1.0, max(0.0, (reach - (ctrl_frac[ci] - 0.06)) / 0.06)))
                _draw_pin(d, px[ci][0], px[ci][1], r=10, ring=int(14 + 7 * arrive),
                          fill=accent_hi, ring_col=accent, alpha=pa)
            else:                                      # waypoint dot
                d.ellipse([px[ci][0] - 8, px[ci][1] - 8, px[ci][0] + 8, px[ci][1] + 8],
                          fill=(*accent_hi, 245))
                d.ellipse([px[ci][0] - 13, px[ci][1] - 13, px[ci][0] + 13, px[ci][1] + 13],
                          outline=(*accent, 180), width=2)

        # labels pop on scrim plates as the head reaches each labeled waypoint.
        # The reveal trigger is slightly BEFORE the control fraction so a label
        # at the very end (ctrl_frac==1.0) still fully appears when the head lands.
        for ci, p in enumerate(pts):
            if not p["label"]:
                continue
            if ci == 0:                                # start label rides the settle
                la = look.ease_out_cubic(min(1.0, max(0.0, (t - draw_t0) / 0.3)))
            else:
                trig = max(0.0, ctrl_frac[ci] - 0.10)  # begin just before arrival
                la = look.ease_out_cubic(min(1.0, max(0.0, (reach - trig) / 0.10)))
            if la <= 0.02:
                continue
            above = px[ci][1] > h * 0.5                # plate away from frame edge
            frame = _label_plate(frame, p["label"], label_font, px[ci][0], px[ci][1],
                                 pal=pal, accent=accent, alpha=la, above=above)

        # premium finish: vignette + grain + global in/out dissolve. A lighter
        # vignette over REAL footage keeps the bed clearly visible (footage is the
        # subject); the simulated terrain bed keeps the full cinematic frame.
        frame = look.vignette(frame, strength=0.40 if not simulated else 0.55)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                  # noqa: BLE001
        ff = "ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
           "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and Path(out_path).exists()
    if hasattr(look, "cleanup_frames"):
        look.cleanup_frames(td)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "simulated_bed": simulated, "points": len(pts),
            "err": (r.stderr[-200:] if not ok else "")}
