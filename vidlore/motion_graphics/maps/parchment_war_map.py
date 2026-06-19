"""Primitive: parchment_war_map.

A tactical-status map beat — a textured parchment/topographic base holds with a
slow push-in while TWO territory fills (side A vs side B) settle in and a jagged
CONTESTED FRONT LINE draws down the middle and then pulses. Side labels fade in
over each territory. The grammar of "here is who holds what, and where they meet"
— the spine of any mapped war / border / empire sequence.

Distinct from `map_region_highlight` (one highlighted area) and `map_heat_spread`
(a diffusing intensity): this is the two-combatant front, the single most-used
shot in mapped-conflict storytelling.

Forensic principle (NOT an asset copy): premium mapped-history channels show a
war as two solid territory fills meeting at an animated front, on a textured (not
flat) base, muted palette, all-caps labels. Rebuilt from that PRINCIPLE with
Vidlore's own look system, palettes, typography, easing — no reference asset,
colour scheme or layout reproduced. Pure-local, deterministic, no paid API.

    render("x.mp4", title="THE FRONT, 1916",
           side_a="ENTENTE", side_b="CENTRAL POWERS", front_x=0.5)
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
    "id": "parchment_war_map", "family": "maps",
    "roles": ["status", "front", "territory", "conflict", "border", "advance"],
    "niches_ok": ["history", "geopolitics", "war", "crime"],
    "intensity_range": [3, 5], "duration_range": [4.0, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "map_swell_low",
    "repeat_cooldown_s": 55, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["front_vertical", "front_diagonal"],
    "review_override": ["title", "side_a", "side_b", "front_x", "map_image", "palette"],
    "fallback": "map_region_highlight if only one region is named",
    "required_inputs": ["side_a"],
}


def _front_path(w, h, front_x, seed, diagonal=False):
    """Seeded jagged front line from top→bottom (list of (x,y))."""
    rng = np.random.RandomState((seed * 2654435761) & 0xFFFFFFFF)
    n = 11
    pts = []
    base = front_x * w
    drift = (0.12 * w) if diagonal else 0.0
    for i in range(n + 1):
        fy = i / n
        y = fy * h
        jitter = (rng.rand() - 0.5) * 0.10 * w
        x = base + jitter + drift * (fy - 0.5) * 2.0
        pts.append((float(np.clip(x, 0.12 * w, 0.88 * w)), float(y)))
    return pts


def render(out_path, *, title: str = "", side_a: str = "", side_b: str = "",
           front_x: float = 0.5, map_image=None, bg_image=None, dur: float = 5.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", layout: str = "", seed: int = 0,
           reference: bool = False, borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    title = str(title or "").strip()
    side_a = str(side_a or "").strip()
    side_b = str(side_b or "").strip()
    diagonal = layout == "front_diagonal"

    # base map (real or synth antique), dimmed a touch for fill legibility
    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            bed = None

    # territory colours: A = warm accent, B = a cool slate (clear A/B contrast)
    col_a = tuple(pal["accent"])
    col_b = (96, 116, 132) if palette_name != "ember_red" else (110, 96, 120)
    accent_hi = tuple(pal["accent_hi"])
    label_font = look.font("label", int(h * 0.030))
    title_font = look.font("label", int(h * 0.034))

    # REAL geography: fill the two combatants' ACTUAL territory polygons on a real
    # regional coastline bed; the shared boundary reads as the contested front.
    geo_mode = False
    polys_a = polys_b = la_pt = lb_pt = None
    a_ent = geo.country_entry(side_a) if bed is None else None
    b_ent = geo.country_entry(side_b) if bed is None else None
    if bed is None and a_ent and b_ent:
        aw, asth, ae, an = a_ent["bbox"]; bw_, bs_, be_, bn_ = b_ent["bbox"]
        bbox = geo.region_bbox([(aw, asth), (ae, an), (bw_, bs_), (be_, bn_)],
                               pad_frac=0.28)
        gproj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=gproj, graticule=True,
                               reference=reference, borders=borders)
        polys_a = [[gproj(lo, la) for lo, la in r] for r in a_ent["rings"]]
        polys_b = [[gproj(lo, la) for lo, la in r] for r in b_ent["rings"]]
        la_pt = gproj(*a_ent["centroid"]); lb_pt = gproj(*b_ent["centroid"])
        geo_mode = True
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    fx = float(min(0.82, max(0.18, front_x)))
    fpts = _front_path(w, h, fx, seed, diagonal)
    poly_a = [(0, 0)] + fpts + [(0, h)]
    poly_b = [(w, 0)] + list(reversed(fpts)) + [(w, h)]

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.05 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))

        # territory fills + contested front
        fill_a = look.ease_out_cubic(min(1.0, t / 0.9))
        pulse = 0.5 + 0.5 * math.sin(t * 3.2)
        lab = max(0.0, min(1.0, (t - 0.8) / 0.8))
        if geo_mode:
            def _zf(p):                      # match the bed's center push-in
                return (w / 2 + (p[0] - w / 2) * z, h / 2 + (p[1] - h / 2) * z)
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            for r in polys_a:
                od.polygon([_zf(p) for p in r], fill=(*col_a, int(140 * fill_a)))
            for r in polys_b:
                od.polygon([_zf(p) for p in r], fill=(*col_b, int(132 * fill_a)))
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gd = ImageDraw.Draw(gl)
            ew = 3 + int(2 * pulse)
            for r in polys_a:
                pp = [_zf(p) for p in r]; gd.line(pp + [pp[0]], fill=(*accent_hi, 235), width=ew)
            for r in polys_b:
                pp = [_zf(p) for p in r]; gd.line(pp + [pp[0]], fill=(180, 208, 228, 220), width=ew)
            frame = Image.alpha_composite(frame.convert("RGBA"),
                                          gl.filter(ImageFilter.GaussianBlur(5))).convert("RGB")
            frame = Image.alpha_composite(frame.convert("RGBA"), gl).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            if lab > 0.01:
                if side_a:
                    ai = look.text_with_glow(side_a.upper(), label_font, fill=accent_hi,
                                             glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=10)
                    zp = _zf(la_pt)
                    frame = look.paste_center(frame, ai, cx=int(zp[0]), cy=int(zp[1]), opacity=lab)
                if side_b:
                    bi = look.text_with_glow(side_b.upper(), label_font, fill=(206, 220, 230),
                                             glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=10)
                    zp = _zf(lb_pt)
                    frame = look.paste_center(frame, bi, cx=int(zp[0]), cy=int(zp[1]), opacity=lab)
                d = ImageDraw.Draw(frame, "RGBA")
        else:
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            od.polygon(poly_a, fill=(*col_a, int(70 * fill_a)))
            od.polygon(poly_b, fill=(*col_b, int(66 * fill_a)))
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            draw_p = look.ease_in_out_cubic(min(1.0, max(0.0, (t - 0.25) / 1.35)))
            seg = fpts[:max(1, int(len(fpts) * draw_p)) + 1]
            if len(seg) >= 2:
                gw = 3 + int(2 * pulse)
                d.line([(int(x), int(y)) for (x, y) in seg], fill=(*accent_hi, 235),
                       width=gw, joint="curve")
                gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                ImageDraw.Draw(gl).line([(int(x), int(y)) for (x, y) in seg],
                                        fill=(*accent_hi, 90), width=gw + 8, joint="curve")
                frame = Image.alpha_composite(frame.convert("RGBA"),
                                              gl.filter(ImageFilter.GaussianBlur(6))).convert("RGB")
                d = ImageDraw.Draw(frame, "RGBA")
            if lab > 0.01:
                if side_a:
                    ai = look.text_with_glow(side_a.upper(), label_font, fill=accent_hi,
                                             glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0, pad=10)
                    frame = look.paste_center(frame, ai, cx=int(fx * w * 0.46),
                                              cy=int(h * 0.30), opacity=lab)
                    d = ImageDraw.Draw(frame, "RGBA")
                if side_b:
                    bi = look.text_with_glow(side_b.upper(), label_font, fill=(206, 220, 230),
                                             glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0, pad=10)
                    frame = look.paste_center(frame, bi, cx=int(fx * w + (w - fx * w) * 0.56),
                                              cy=int(h * 0.70), opacity=lab)
                    d = ImageDraw.Draw(frame, "RGBA")

        # optional title chip, top-center
        if title:
            ti = max(0.0, min(1.0, (t - 0.1) / 0.5))
            chip = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                       glow=pal["bg_b"], glow_radius=6,
                                       glow_alpha=0.0, pad=12)
            cw, ch = chip.width, chip.height
            bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(bar).rounded_rectangle(
                [(w - cw) // 2 - 24, int(h * 0.06) - 8,
                 (w + cw) // 2 + 24, int(h * 0.06) + ch + 8],
                radius=10, fill=(*pal["bg_b"], int(210 * ti)))
            frame = Image.alpha_composite(frame.convert("RGBA"), bar).convert("RGB")
            frame = look.paste_center(frame, chip, cx=w // 2,
                                      cy=int(h * 0.06) + ch // 2, opacity=ti)

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
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
