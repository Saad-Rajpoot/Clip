"""Primitive: map_badge_node.

An actor-placement beat — a hexagon BADGE (a portrait, a flag-stripe fill, or
initials) drops onto a textured map, tethered by a thin leader line + pin to an
exact location, with a name label. The grammar of "this player sits HERE" —
placing a leader, an agency, a flag, a faction onto geography before the story
moves. Pins read instantly and never clutter the map.

Distinct from `portrait_name_over_map` (a large cinematic portrait beside a map):
this is a COMPACT pinned node you can place several of in a sequence to populate
a theatre with actors.

Forensic principle (NOT an asset copy): premium mapped storytelling pins actors
with hexagon portrait/flag badges + leader lines. Rebuilt from that PRINCIPLE
with Vidlore's own look system — no reference asset, colour or layout
reproduced. Pure-local, deterministic.

    render("x.mp4", place="TEHRAN", label="KHOMEINI", at=[0.62,0.40])
    render("x.mp4", place="USA", flag=[[60,90,150],[235,235,235],[200,60,60]])
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
from . import geo
from .location_establish_card import _antique_map

SPEC = {
    "id": "map_badge_node", "family": "maps",
    "roles": ["actor", "placement", "leader", "faction", "flag", "agency"],
    "niches_ok": ["history", "geopolitics", "crime", "business", "biography", "tech"],
    "intensity_range": [2, 4], "duration_range": [3.0, 5.5],
    "easing": "easeOutCubic", "audio_cue": "pin_soft",
    "repeat_cooldown_s": 30, "per_video_cap": 5, "cost": "low",
    "layout_variants": ["badge_above", "badge_left"],
    "review_override": ["place", "label", "at", "portrait", "flag", "map_image", "palette"],
    "fallback": "location_establish_card if no actor is named",
    "required_inputs": ["place"],
}


def _hexagon(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * k - 90)),
             cy + r * math.sin(math.radians(60 * k - 90))) for k in range(6)]


def _badge(r, pal, *, portrait=None, flag=None, label=""):
    """RGBA hexagon badge: portrait (masked), flag stripes, or initials."""
    size = r * 2 + 8
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hexpts = _hexagon(size / 2, size / 2, r)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(hexpts, fill=255)
    inner = Image.new("RGBA", (size, size), (*pal["bg_a"], 255))
    if portrait is not None and Path(str(portrait)).exists():
        try:
            p = look.grade_media(Image.open(portrait).convert("RGB"), pal, strength=0.4)
            p = p.resize((size, size), Image.LANCZOS)
            inner = p.convert("RGBA")
        except Exception:                                          # noqa: BLE001
            pass
    elif flag:
        dd = ImageDraw.Draw(inner)
        k = len(flag)
        for j, col in enumerate(flag):
            dd.rectangle([0, int(size * j / k), size, int(size * (j + 1) / k)],
                         fill=(int(col[0]), int(col[1]), int(col[2]), 255))
    else:
        ini = "".join([s[0] for s in label.split()[:2]]).upper() or label[:2].upper()
        f = look.font("numeral", int(r * 0.9))
        ti = look.text_with_glow(ini, f, fill=pal["accent_hi"], glow=pal["bg_b"],
                                 glow_radius=4, glow_alpha=0.0, pad=6)
        inner.alpha_composite(ti, ((size - ti.width) // 2, (size - ti.height) // 2))
    img.paste(inner, (0, 0), mask)
    # hex border
    ImageDraw.Draw(img).line(hexpts + [hexpts[0]], fill=(*pal["accent_hi"], 255), width=4)
    return img


def render(out_path, *, place: str = "", label: str = "", at=None, portrait=None,
           flag=None, map_image=None, bg_image=None, dur: float = 4.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, reference: bool = False,
           borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    place = str(place or "").strip()
    label = str(label or "").strip() or place
    left = layout == "badge_left"
    try:
        ax, ay = float(at[0]), float(at[1])
    except Exception:                                              # noqa: BLE001
        rng = np.random.RandomState((seed * 2654435761) & 0xFFFFFFFF)
        ax, ay = 0.42 + 0.22 * rng.rand(), 0.46 + 0.16 * rng.rand()
    ax = min(0.86, max(0.14, ax)); ay = min(0.84, max(0.20, ay))

    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            bed = None
    # REAL geography: frame to the place's country (or a region around a city),
    # highlight it, and pin the badge's anchor at the true location.
    px = py = None
    gll = geo.resolve(place) if bed is None else None
    if bed is None and gll:
        ent = geo.country_entry(place)
        if ent:
            bw, bs, be, bn = ent["bbox"]
            bbox = geo.region_bbox([(bw, bs), (be, bn)], pad_frac=0.25)
        else:
            bbox = geo.region_bbox([gll], pad_frac=0.4, min_span=10.0)
        gproj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=gproj, highlight=[place],
                               reference=reference, borders=borders)
        px, py = (int(v) for v in gproj(*gll))
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    r = int(h * 0.085)
    badge = _badge(r, pal, portrait=portrait, flag=flag, label=label)
    if px is None:
        px, py = int(ax * w), int(ay * h)
    bx = px + (-int(w * 0.13) if left else 0)
    by = py - int(h * 0.20)
    label_font = look.font("label", int(h * 0.030))
    accent_hi = tuple(pal["accent_hi"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.035 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # pin + pulse ring at the location
        pin_op = look.ease_out_cubic(min(1.0, t / 0.4))
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(*accent_hi, int(240 * pin_op)),
                  outline=(*pal["bg_b"], int(200 * pin_op)), width=2)
        ring = (t * 0.9) % 1.0
        if pin_op > 0.5:
            rr = int(6 + 30 * ring)
            d.ellipse([px - rr, py - rr, px + rr, py + rr],
                      outline=(*accent_hi, int(150 * (1 - ring))), width=2)

        # leader line from pin to badge (draws 0.3–0.8s)
        lp = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.3) / 0.5)))
        if lp > 0.01:
            ex, ey = bx + (px - bx) * (1 - lp), by + (py - by) * (1 - lp)
            d.line([(px, py), (ex, ey)], fill=(*accent_hi, 210), width=2)

        # badge scales in (0.45–1.0s)
        bop = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.45) / 0.55)))
        if bop > 0.01:
            sc = 0.7 + 0.3 * bop
            frame = look.paste_center(frame, badge, cx=bx, cy=by, scale=sc, opacity=bop)
            d = ImageDraw.Draw(frame, "RGBA")
            # label under badge
            lab = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.7) / 0.5)))
            if lab > 0.01:
                li = look.text_with_glow(label.upper(), label_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0, pad=8)
                frame = look.paste_center(frame, li, cx=bx, cy=by + int(r * sc) + 22,
                                          opacity=lab)

        frame = look.vignette(frame, strength=0.58)
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
