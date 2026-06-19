"""Primitive: location_establish_card.

An establishing beat — a place name + era over a graded map / landscape, with
corner coordinate ticks, a pulsing location pin, and a slow push. The cinematic
version of a "WHERE: …" lower-third. Distinct from `portrait_name_over_map`
(which pairs a person with a place): this is geography ALONE — the cut-away that
orients the viewer ("Cleveland, Ohio · 1863").

Forensic principle (NOT asset copy): MagnatesMedia grounds a story in place with
a graded map + refined type, not a flat caption. Uses a real `map_image` /
`bg_image` if the pipeline supplies one, else an original antique-map synth.
Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", place="Cleveland, Ohio", sub="1863", coords="41.5°N 81.7°W")
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
    "id": "location_establish_card", "family": "maps",
    "roles": ["location", "origin", "establish", "place", "setting"],
    "niches_ok": ["history", "biography", "geopolitics", "crime", "business", "tech"],
    "intensity_range": [2, 4], "duration_range": [3.5, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_locator_ping",
    "repeat_cooldown_s": 40, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["map_lower_third"],
    "review_override": ["place", "sub", "coords", "map_image", "palette"],
    "fallback": "name-over-graded-bg if no map asset is available",
}


def _antique_map(w, h, seed, pal):
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (200, 182, 146)
    img = Image.fromarray(base.astype("uint8"), "RGB")
    d = ImageDraw.Draw(img)
    for gx in range(0, w, max(1, w // 18)):
        d.line([(gx, 0), (gx, h)], fill=(168, 148, 112), width=1)
    for gy in range(0, h, max(1, h // 11)):
        d.line([(0, gy), (w, gy)], fill=(168, 148, 112), width=1)
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
    for _ in range(7):                                # soft landmass blobs
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rr = int(rng.integers(110, 280)); pts = []
        for a in range(0, 360, 22):
            ra = rr * (0.55 + 0.6 * rng.random())
            pts.append((cx + ra * math.cos(math.radians(a)),
                        cy + ra * math.sin(math.radians(a))))
        od.polygon(pts, fill=(150, 126, 88, 60))
    ov = ov.filter(ImageFilter.GaussianBlur(10))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    img = look.grade_media(img, pal, strength=0.34)
    arr = (np.asarray(img).astype(np.float32) * 0.62).astype("uint8")
    return Image.fromarray(arr, "RGB")


def render(out_path, *, place: str = "", sub: str = "", coords: str = "",
           map_image=None, bg_image=None, dur: float = 5.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, reference: bool = False,
           borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    place = (place or "").strip() or "Unknown"

    src = map_image or bg_image
    bed = None
    pin = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.45)
            bed = Image.fromarray((np.asarray(bed).astype(np.float32) * 0.6)
                                  .astype("uint8"), "RGB")
        except Exception:                                      # noqa: BLE001
            bed = None
    # REAL geography: frame to the place's country (or a region around a city) +
    # pin at the TRUE location. Falls back to the antique synth for unknown places.
    if bed is None:
        _pname = place.split(",")[0].strip() if "," in place else place
        gll = geo.resolve(place) or geo.resolve(_pname)
        if gll:
            ent = geo.country_entry(place) or geo.country_entry(_pname)
            if ent:
                bw_, bs_, be_, bn_ = ent["bbox"]
                bbox = geo.region_bbox([(bw_, bs_), (be_, bn_)], pad_frac=0.3)
                hl = [ent["name"]]
            else:
                bbox = geo.region_bbox([gll], pad_frac=0.5, min_span=12.0)
                hl = []
            gproj = geo.make_projector(bbox, w, h)
            gbed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                                    seed=seed, proj=gproj, highlight=hl,
                                    reference=reference, borders=borders)
            bed = Image.fromarray((np.asarray(gbed).astype(np.float32) * 0.66)
                                  .astype("uint8"), "RGB")
            pin = tuple(int(v) for v in gproj(*gll))
    if bed is None:
        bed = _antique_map(w, h, seed, pal)
    if pin is None:
        pin = (int(w * 0.5), int(h * 0.42))
    nm_font = look.font("title", int(h * 0.062))
    sb_font = look.font("label", int(h * 0.032))
    co_font = look.font("label", int(h * 0.024))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.05 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")
        # corner coordinate ticks (locator framing) draw in
        ca = look.ease_out_cubic(min(1.0, t / 0.6))
        mlen = int(w * 0.05 * ca)
        for (ox, oy, dx, dy) in ((0.08, 0.12, 1, 1), (0.92, 0.12, -1, 1),
                                 (0.08, 0.88, 1, -1), (0.92, 0.88, -1, -1)):
            px, py = int(w * ox), int(h * oy)
            d.line([(px, py), (px + dx * mlen, py)], fill=(*pal["accent"], 200), width=2)
            d.line([(px, py), (px, py + dy * mlen)], fill=(*pal["accent"], 200), width=2)
        # pulsing pin
        pa = look.ease_out_cubic(max(0.0, (t - 0.4) / 0.5))
        if pa > 0.01:
            pr2 = (math.sin(t * 4) * 0.5 + 0.5)
            rr = int(h * (0.014 + 0.012 * pr2))
            d.ellipse([pin[0] - rr, pin[1] - rr, pin[0] + rr, pin[1] + rr],
                      outline=(*pal["glow"], int(120 + 90 * pr2)), width=3)
            d.ellipse([pin[0] - 7, pin[1] - 7, pin[0] + 7, pin[1] + 7],
                      fill=(*pal["accent_hi"], int(255 * pa)))
        # place name lower-third + era + coords
        la = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.6))
        if la > 0.01:
            ny = int(h * 0.66)
            look.hairline(d, int(w * 0.5), ny - int(h * 0.05),
                          int(w * 0.12 * la), pal)
            ni = look.text_with_glow(place, nm_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=6,
                                     glow_alpha=0.0, pad=26)
            frame = look.paste_center(frame, ni, cx=w * 0.5, cy=ny, opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")
            if sub:
                si = look.text_with_glow(sub.upper(), sb_font, fill=pal["accent_hi"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, si, cx=w * 0.5,
                                          cy=ny + int(h * 0.058), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")
            if coords:
                ci = look.text_with_glow(coords.upper(), co_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=12)
                frame = look.paste_center(frame, ci, cx=w * 0.5,
                                          cy=ny + int(h * (0.10 if sub else 0.058)),
                                          opacity=la)

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
