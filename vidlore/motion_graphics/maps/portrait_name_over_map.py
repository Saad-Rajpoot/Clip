"""Primitive: portrait_name_over_map.

A public-figure + geography composite: a framed portrait, a synthesised antique
map bed, a pulsing location marker, a connecting line that draws from the
portrait to the marker, and a lower-third name + place — over a slow controlled
zoom. The premium version of a name-over-place card.

Forensic evidence: MagnatesMedia 135.7s (framed portrait + typewriter name over
a parchment map). Pure-local (PIL + numpy → ffmpeg). No paid API. The map bed is
ORIGINAL parchment texture (the pipeline can pass a real Natural-Earth map image
as `map_image` for true geography).
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
    "id": "portrait_name_over_map", "family": "maps",
    "roles": ["person_place", "intro", "origin", "operation"],
    "niches_ok": ["history", "biography", "geopolitics", "crime", "business"],
    "intensity_range": [2, 4], "duration_range": [3.5, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_pulse_bed",
    "repeat_cooldown_s": 35, "per_video_cap": 4, "cost": "med",
    "layout_variants": ["portrait_left_pin_right", "portrait_right_pin_left"],
    "review_override": ["name", "place", "side", "map_image", "palette"],
    "fallback": "cinematic_portrait_hold if no place/map",
}


def _antique_map(w, h, seed, pal):
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32); base[:] = (206, 188, 152)
    img = Image.fromarray(base.astype("uint8"), "RGB")
    d = ImageDraw.Draw(img)
    for gx in range(0, w, max(1, w // 16)):
        d.line([(gx, 0), (gx, h)], fill=(170, 150, 114), width=1)
    for gy in range(0, h, max(1, h // 10)):
        d.line([(0, gy), (w, gy)], fill=(170, 150, 114), width=1)
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    for _ in range(6):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rr = int(rng.integers(90, 240)); pts = []
        for a in range(0, 360, 24):
            ra = rr * (0.55 + 0.6 * rng.random())
            pts.append((cx + ra * math.cos(math.radians(a)),
                        cy + ra * math.sin(math.radians(a))))
        od.polygon(pts, fill=(150, 126, 88, 64))
    ov = ov.filter(ImageFilter.GaussianBlur(9))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    return look.grade_media(img, pal, strength=0.30)


def _framed(img, fw, fh, pal):
    # FACE-AWARE crop — keep the whole head (no forehead clip); grayscale-tolerant.
    crop = look.grade_media(look.face_safe_crop(img, fw, fh).convert("RGB"),
                            pal, strength=0.4)
    bd = 6
    tile = Image.new("RGBA", (fw + bd * 2 + 56, fh + bd * 2 + 56), (0, 0, 0, 0))
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([30, 34, 30 + fw + bd * 2, 34 + fh + bd * 2], fill=(0, 0, 0, 165))
    tile.alpha_composite(sh.filter(ImageFilter.GaussianBlur(18)))
    fr = Image.new("RGBA", (fw + bd * 2, fh + bd * 2), (*[int(c * 0.82) for c in pal["accent"]], 255))
    fr.paste(crop, (bd, bd))
    tile.alpha_composite(fr, (24, 24))
    return tile


def render(out_path, *, portrait_path=None, name: str = "", place: str = "",
           map_image=None, side: str = "left", dur: float = 5.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           seed: int = 0, reference: bool = False, borders: bool = True,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    # map bed (real map if supplied, else antique synth)
    if map_image and Path(map_image).exists():
        try:
            mp = look.grade_media(Image.open(map_image).convert("RGB").resize((w, h)), pal, 0.4)
        except Exception:                                      # noqa: BLE001
            mp = _antique_map(w, h, seed, pal)
    else:
        # REAL geography: a real country/region bed behind the portrait when the
        # place resolves (else antique synth) — the person is shown over true land.
        _gll = geo.resolve(place) if place else None
        if _gll:
            _ent = geo.country_entry(place)
            if _ent:
                _bw, _bs, _be, _bn = _ent["bbox"]
                _bbox = geo.region_bbox([(_bw, _bs), (_be, _bn)], pad_frac=0.3)
                _hl = [_ent["name"]]
            else:
                _bbox = geo.region_bbox([_gll], pad_frac=0.5, min_span=14.0); _hl = []
            _gp = geo.make_projector(_bbox, w, h)
            mp = geo.draw_basemap(w, h, _bbox, style=geo.style_for(palette_name),
                                  seed=seed, proj=_gp, highlight=_hl,
                                  reference=reference, borders=borders)
        else:
            mp = _antique_map(w, h, seed, pal)
    mp = Image.fromarray((np.asarray(mp).astype(np.float32) * 0.66).astype("uint8"), "RGB")
    # portrait tile
    tile = None
    if portrait_path and Path(portrait_path).exists():
        try:
            fw, fh = int(h * 0.46 * 0.78), int(h * 0.46)
            tile = _framed(Image.open(portrait_path).convert("RGB"), fw, fh, pal)
        except Exception:                                      # noqa: BLE001
            tile = None
    pcx = w * (0.30 if side == "left" else 0.70)
    pin = (w * (0.66 if side == "left" else 0.34), h * 0.40)
    nm_font = look.font("title", int(h * 0.050))
    pl_font = look.font("label", int(h * 0.030))
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps; pr = i / max(1, n - 1)
        z = 1.0 + 0.05 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = mp.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        dctx = ImageDraw.Draw(frame, "RGBA")
        # connecting line draws from portrait to pin after portrait settles
        ln = look.ease_out_cubic(max(0.0, (t - 0.9) / 0.7))
        ex = pcx + (pin[0] - pcx) * ln
        ey = h * 0.5 + (pin[1] - h * 0.5) * ln
        if ln > 0.02:
            dctx.line([(pcx, h * 0.5), (ex, ey)], fill=(*pal["accent_hi"], 220), width=3)
        # pin pulse appears at end of line
        if ln > 0.96:
            pr2 = (math.sin(t * 4) * 0.5 + 0.5)
            rr = int(h * (0.012 + 0.012 * pr2))
            dctx.ellipse([pin[0] - rr, pin[1] - rr, pin[0] + rr, pin[1] + rr],
                         outline=(*pal["glow"], int(120 + 80 * pr2)), width=3)
            dctx.ellipse([pin[0] - 7, pin[1] - 7, pin[0] + 7, pin[1] + 7], fill=(*pal["accent_hi"], 255))
        # portrait
        if tile is not None:
            ent = look.ease_out_cubic(min(1.0, t / 0.6))
            frame = look.paste_center(frame, tile, cx=pcx, cy=h * 0.5,
                                      scale=(0.96 + 0.04 * ent) * z, opacity=ent)
        # name lower-third near the pin
        if name:
            la = look.ease_out_cubic(max(0.0, (t - 1.0) / 0.7))
            ty = h * 0.62 if side == "left" else h * 0.62
            tx = pin[0]
            dctx2 = ImageDraw.Draw(frame)
            look.hairline(dctx2, int(tx), int(ty - h * 0.04), int(w * 0.10 * la), pal)
            nm = look.text_with_glow(name, nm_font, fill=pal["text"], glow=pal["bg_b"],
                                     glow_radius=6, glow_alpha=0.0, pad=26)
            frame = look.paste_center(frame, nm, cx=tx, cy=ty, opacity=la)
            if place:
                pl = look.text_with_glow(place.upper(), pl_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=4, glow_alpha=0.0, pad=20)
                frame = look.paste_center(frame, pl, cx=tx, cy=ty + int(h * 0.052), opacity=la)
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
