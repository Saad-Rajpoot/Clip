"""Primitive: map_status_banner.

A chronological anchor beat — a graded antique/topographic map holds with a slow
push-in while a clean STATUS BANNER slides down from the top: a large YEAR and an
all-caps EVENT line ("1982 · IRAN INVADES IRAQ"), under a thin accent rule. A few
optional position ticks can mark a front/border on the map. The banner parks in
the upper third; the map stays the bed so a sequence of these reads as a moving
timeline over geography.

Distinct from `era_band_timeline` (a horizontal multi-event spine) and
`location_establish_card` (a single place): this stamps ONE dated moment onto a
map — the cut-to-the-next-phase device in mapped history/geopolitics/crime.

Forensic principle (NOT an asset copy): premium mapped-history channels punctuate
a war/expansion with a sliding year+event banner so each map phase is time-stamped
without a separate card. Rendered with Vidlore's own look system, palettes,
typography and easing. Pure-local, deterministic, no paid API.

    render("x.mp4", year="1982", event="Iran invades Iraq",
           ticks=["0.46,0.50", "0.52,0.44"])
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
    "id": "map_status_banner", "family": "maps",
    "roles": ["status", "phase", "turning_point", "date", "chapter", "advance"],
    "niches_ok": ["history", "geopolitics", "crime", "biography", "war", "tech"],
    "intensity_range": [2, 4], "duration_range": [3.5, 6.0],
    "easing": "easeOutCubic", "audio_cue": "stamp_soft",
    "repeat_cooldown_s": 40, "per_video_cap": 4, "cost": "low",
    "layout_variants": ["banner_top", "banner_lower"],
    "review_override": ["year", "event", "ticks", "map_image", "palette"],
    "fallback": "location_establish_card if no year/event is available",
    "required_inputs": ["event"],
}


def _parse_ticks(ticks):
    out = []
    for s in (ticks or []):
        try:
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                out.append((float(s[0]), float(s[1])))
            elif isinstance(s, str) and "," in s:
                a, b = s.replace(" ", "").split(",")[:2]
                out.append((float(a), float(b)))
        except Exception:                                          # noqa: BLE001
            pass
    return out[:6]


def render(out_path, *, year: str = "", event: str = "", ticks=None, place: str = "",
           map_image=None, bg_image=None, dur: float = 5.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, reference: bool = False,
           borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    year = str(year or "").strip()
    event = str(event or "").strip()
    tk = _parse_ticks(ticks)
    lower = layout == "banner_lower"

    # map bed (real or synth), dimmed for legibility
    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.45)
            bed = Image.fromarray((np.asarray(bed).astype(np.float32) * 0.62)
                                  .astype("uint8"), "RGB")
        except Exception:                                          # noqa: BLE001
            bed = None
    # REAL geography: zoom to the relevant country, highlight it, dim for banner
    # legibility — so the frame is never visually empty under the banner.
    if bed is None and place:
        ent = geo.country_entry(place)
        gll = geo.resolve(place)
        if ent or gll:
            if ent:
                bw, bs, be, bn = ent["bbox"]
                bbox = geo.region_bbox([(bw, bs), (be, bn)], pad_frac=0.3)
            else:
                bbox = geo.region_bbox([gll], pad_frac=0.4, min_span=10.0)
            gbed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                                    seed=seed, highlight=[place],
                                    reference=reference, borders=borders)
            bed = Image.fromarray((np.asarray(gbed).astype(np.float32) * 0.66)
                                  .astype("uint8"), "RGB")
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    year_font = look.font("numeral", int(h * 0.072))
    event_font = look.font("label", int(h * 0.034))
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    ink = tuple(pal["text"])
    # banner stage colour: a deep charcoal panel (matches look.fade_frame floor)
    panel = tuple(pal.get("bg_b", (16, 18, 24)))

    by = int(h * (0.66 if lower else 0.16))                        # banner anchor y
    bw_box = int(w * 0.56)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # slow push-in on the map bed
        z = 1.0 + 0.05 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # position ticks (small crosshairs) fade in early
        ta_t = look.ease_out_cubic(min(1.0, t / 0.5))
        for (fx, fy) in tk:
            cx, cy = int(min(0.94, max(0.06, fx)) * w), int(min(0.9, max(0.1, fy)) * h)
            a = int(210 * ta_t)
            d.line([(cx - 8, cy), (cx + 8, cy)], fill=(*accent_hi, a), width=2)
            d.line([(cx, cy - 8), (cx, cy + 8)], fill=(*accent_hi, a), width=2)
            d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(*accent_hi, a))

        # banner slides down (or up) into place with ease-out, then holds
        intro = look.ease_out_cubic(min(1.0, t / 0.55))
        slide = (1.0 - intro) * (h * 0.10) * (-1 if lower else 1)
        cy = by - int(slide)
        # panel
        ph = int(h * 0.155)
        x0, x1 = (w - bw_box) // 2, (w + bw_box) // 2
        y0, y1 = cy - ph // 2, cy + ph // 2
        pa = int(225 * intro)
        panel_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(panel_layer).rounded_rectangle(
            [x0, y0, x1, y1], radius=int(h * 0.012), fill=(*panel, pa))
        frame = Image.alpha_composite(frame.convert("RGBA"), panel_layer).convert("RGB")
        d = ImageDraw.Draw(frame, "RGBA")
        # accent rule under the panel
        if intro > 0.05:
            rw = int((x1 - x0) * 0.82 * intro)
            ry = y1 - int(ph * 0.16)
            d.line([((w - rw) // 2, ry), ((w + rw) // 2, ry)],
                   fill=(*accent_hi, int(230 * intro)), width=3)

        # YEAR (left, large) + EVENT (right or below) — adaptive ink on the panel
        tio = max(0.0, min(1.0, (t - 0.25) / 0.5))
        if tio > 0.01:
            label_ink = look.pick_text_ink(panel) if hasattr(look, "pick_text_ink") else ink
            if year:
                yi = look.text_with_glow(year, year_font, fill=label_ink,
                                         glow=panel, glow_radius=4, glow_alpha=0.0,
                                         pad=8)
                frame = look.paste_center(frame, yi, cx=x0 + int((x1 - x0) * 0.20),
                                          cy=cy - int(ph * 0.06), opacity=tio)
                d = ImageDraw.Draw(frame, "RGBA")
            if event:
                ev = event.upper()
                ex_cx = (x0 + int((x1 - x0) * 0.62)) if year else int(w * 0.5)
                ei = look.text_with_glow(ev, event_font, fill=accent_hi,
                                         glow=panel, glow_radius=4, glow_alpha=0.0,
                                         pad=8)
                # shrink if it would overflow the panel
                maxw = int((x1 - x0) * (0.66 if year else 0.9))
                if ei.width > maxw:
                    ei = ei.resize((maxw, int(ei.height * maxw / ei.width)),
                                   Image.LANCZOS)
                frame = look.paste_center(frame, ei, cx=ex_cx,
                                          cy=cy - int(ph * 0.04), opacity=tio)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.55)
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
