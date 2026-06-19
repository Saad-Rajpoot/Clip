"""Primitive: map_heat_spread.

A spread beat — heat blooms across a graded antique map from one or more origin
points and intensifies, showing something taking hold across geography: control,
unrest, an epidemic, an idea catching fire.

Distinct from `map_route_spread` (a single LINE traced from A to B) and
`location_establish_card` (one static place): this is AREA INTENSITY — the glow
grows and bleeds outward, so the viewer feels the spread rather than tracing it.

Forensic principle (NOT asset copy): a heat bloom over a period map turns an
abstract "it spread" into something seen; the premium is the graded map + a warm
ember→amber field + soft pins + restraint, not a garish heatmap overlay.
Pure-local. No paid API. Deterministic.

    render("x.mp4", hotspots=[["London", "0.30,0.32"], ["Paris", "0.42,0.46"],
           ["Berlin", "0.55,0.38"]], title="THE REVOLUTION SPREADS · 1848")
"""
from __future__ import annotations

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
    "id": "map_heat_spread", "family": "maps",
    "roles": ["spread", "heat", "contagion", "influence", "movement"],
    "niches_ok": ["history", "geopolitics", "crime", "biography", "business", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.5, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "soft_spread_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["heat_bloom"],
    "review_override": ["hotspots", "title", "map_image", "palette"],
    "fallback": "map_route_spread if spread reads better as a line",
}


def _parse(hotspots):
    out = []
    for hs in (hotspots or []):
        name, pos = "", None
        if isinstance(hs, dict):
            name, pos = str(hs.get("name") or hs.get("label") or ""), hs.get("pos")
        elif isinstance(hs, (list, tuple)) and len(hs) >= 2:
            name, pos = str(hs[0]), hs[1]
        else:
            name = str(hs)
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
    return out[:6]


# ember→amber heat ramp (0..1 → rgb)
_RAMP = np.array([[0, 0, 0], [120, 24, 10], [200, 70, 24],
                  [236, 140, 40], [250, 210, 120]], np.float32)


def _heat_rgb(field):
    f = np.clip(field, 0, 1) * (len(_RAMP) - 1)
    lo = np.floor(f).astype(int); hi = np.minimum(lo + 1, len(_RAMP) - 1)
    frac = (f - lo)[..., None]
    return _RAMP[lo] * (1 - frac) + _RAMP[hi] * frac


def render(out_path, *, hotspots=None, title: str = "", map_image=None,
           bg_image=None, dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "parchment_sepia", layout: str = "",
           seed: int = 0, reference: bool = False, borders: bool = True,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    hs = _parse(hotspots)
    if not hs:
        hs = [("", (0.5, 0.45))]
    # REAL geography: ignite hotspots at TRUE positions on a real region-framed bed
    _geo_bed = None
    _res = [(nm, geo.resolve(nm)) for nm, _ in hs if nm]
    _res = [(nm, ll) for nm, ll in _res if ll]
    if (map_image or bg_image) is None and len(_res) >= 1:
        _bbox = geo.region_bbox([ll for _, ll in _res], pad_frac=0.45, min_span=14.0)
        _gp = geo.make_projector(_bbox, w, h)
        _gb = geo.draw_basemap(w, h, _bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=_gp, graticule=True,
                               reference=reference, borders=borders)
        _geo_bed = Image.fromarray((np.asarray(_gb).astype(np.float32) * 0.6)
                                   .astype("uint8"), "RGB")
        pts = [(nm, _gp(*ll)[0], _gp(*ll)[1]) for nm, ll in _res]
    else:
        pts = []
        for k, (name, fxy) in enumerate(hs):
            if fxy is None:
                fxy = (0.28 + 0.46 * (k / max(1, len(hs) - 1)), 0.42)
            pts.append((name, min(0.92, max(0.08, fxy[0])) * w,
                        min(0.86, max(0.14, fxy[1])) * h))

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
    bed_np = np.asarray(bed).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # precompute each hotspot's gaussian (kept tight so blooms stay distinct
    # instead of merging into one uniform wash)
    rad = w * 0.10
    gauss = [np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * rad * rad)))
             for (_, px, py) in pts]
    lab_font = look.font("label", int(h * 0.026))
    title_font = look.font("label", int(h * 0.030))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # heat field grows: hotspots ignite in sequence, then all intensify
        field = np.zeros((h, w), np.float32)
        for k, g in enumerate(gauss):
            ig = look.ease_in_out_cubic(max(0.0, (t - k * 0.45) / (dur * 0.5)))
            pulse = 0.92 + 0.08 * np.sin(t * 3 + k)
            field += g * ig * 0.8 * pulse
        field = np.clip(field, 0, 1.0)
        heat = _heat_rgb(field)
        # heat GLOW that the map shows through: dim the bed slightly under the
        # hottest zones + add an amber field capped so overlaps never blow to
        # white (alpha-replace washed the whole frame flat; pure-add blew out).
        a = (field ** 1.2)[..., None] * 0.5
        comp = np.clip(bed_np * (1 - 0.32 * a) + heat * a, 0, 248)
        frame = Image.fromarray(comp.astype("uint8"), "RGB")
        d = ImageDraw.Draw(frame, "RGBA")
        # soft pins + labels at each ignited hotspot
        for k, (name, px, py) in enumerate(pts):
            ka = look.ease_out_cubic(max(0.0, (t - k * 0.45 - 0.1) / 0.4))
            if ka <= 0.02:
                continue
            d.ellipse([px - 6, py - 6, px + 6, py + 6],
                      fill=(*pal["accent_hi"], int(255 * ka)))
            d.ellipse([px - 12, py - 12, px + 12, py + 12],
                      outline=(*pal["accent_hi"], int(150 * ka)), width=2)
            if name:
                frame = geo.place_label(frame, name, int(px), int(py), pal,
                                        opacity=ka, dy=-int(h * 0.045))
                d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.6))
            look.hairline(d, int(w * 0.5), int(h * 0.83), int(w * 0.1 * ta), pal)
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=5,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=int(h * 0.88),
                                      opacity=ta)

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fade = look.fade_alpha(t, dur, fps)
        if fade < 1.0:
            frame = look.fade_frame(frame, fade, pal)
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
