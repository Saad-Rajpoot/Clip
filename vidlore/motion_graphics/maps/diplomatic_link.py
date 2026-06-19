"""Primitive: diplomatic_link.

A relationship-outcome beat — two capitals/actors pinned on a textured map, a
connector LINE DRAWS between them, then RESOLVES: an accepted tie settles into a
calm steady link (gentle pulse); a rejected/broken tie snaps — a jagged break
opens in the middle and the line flushes alert-red. The grammar of "they made a
deal" / "the alliance collapsed" — pacts, treaties, betrayals, mergers.

Distinct from `supply_route_dashes` (continuous flow) and `connection_web` (a
many-node board): this is ONE bilateral tie with a binary outcome.

Forensic principle (NOT an asset copy): premium mapped storytelling signals a
failed link by flipping a connector's colour (calm → alert). Rebuilt from that
PRINCIPLE with Vidlore's own look system — no reference asset, colour or layout
reproduced. Pure-local, deterministic.

    render("x.mp4", a="MOSCOW", b="BAGHDAD", outcome="accept")
    render("x.mp4", a="LONDON", b="BERLIN", outcome="reject")
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
    "id": "diplomatic_link", "family": "maps",
    "roles": ["relationship", "treaty", "alliance", "deal", "rupture", "pact"],
    "niches_ok": ["geopolitics", "history", "business", "crime"],
    "intensity_range": [3, 5], "duration_range": [3.5, 6.0],
    "easing": "easeOutCubic", "audio_cue": "connect_or_reject",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["horizontal", "diagonal"],
    "review_override": ["a", "b", "outcome", "title", "map_image", "palette"],
    "fallback": "supply_route_dashes for a flow rather than an outcome",
    "required_inputs": ["a", "b"],
}

_REJECT = {"reject", "rejected", "fail", "failed", "broken", "break", "collapse",
           "betray", "betrayal", "denied", "no", "war"}


def render(out_path, *, a: str = "", b: str = "", outcome: str = "accept",
           title: str = "", map_image=None, bg_image=None, dur: float = 5.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           reference: bool = False, borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    a = str(a or "").strip(); b = str(b or "").strip()
    title = str(title or "").strip()
    reject = str(outcome or "").strip().lower() in _REJECT
    diag = layout == "diagonal"

    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            bed = None
    # REAL geography: when both actors resolve to places, pin them at their true
    # positions on a real bed framed to span both, both countries highlighted.
    pa = pb = None
    ga = geo.resolve(a) if bed is None else None
    gb = geo.resolve(b) if bed is None else None
    if bed is None and ga and gb:
        bbox = geo.region_bbox([ga, gb], pad_frac=0.55)
        gproj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=gproj, highlight=[a, b],
                               reference=reference, borders=borders)
        pa, pb = gproj(*ga), gproj(*gb)
    if bed is None:
        bed = _antique_map(w, h, seed, pal)
    if pa is None:
        pa = (0.26 * w, (0.40 if diag else 0.50) * h)
        pb = (0.74 * w, (0.62 if diag else 0.50) * h)
    ok_col = (120, 196, 150)          # calm green-steel (accepted)
    bad_col = (224, 96, 74)           # alert red (rejected)
    accent_hi = tuple(pal["accent_hi"])
    label_font = look.font("label", int(h * 0.030))
    title_font = look.font("label", int(h * 0.032))

    def pin(frame, pt, label, op):
        d = ImageDraw.Draw(frame, "RGBA")
        cx, cy = int(pt[0]), int(pt[1]); rr = 8
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(*accent_hi, int(235 * op)),
                  outline=(*pal["bg_b"], int(200 * op)), width=2)
        if label:
            frame = geo.place_label(frame, label, cx, cy, pal, opacity=op, dy=-28)
        return frame

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.04 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # connector draws 0.4–1.3s; resolves at 1.5s
        draw_p = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.4) / 0.9)))
        resolved = t >= 1.5
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        end = (pa[0] + (pb[0] - pa[0]) * draw_p, pa[1] + (pb[1] - pa[1]) * draw_p)
        if draw_p > 0.01:
            col = accent_hi
            if resolved:
                col = bad_col if reject else ok_col
            pulse = 0.5 + 0.5 * math.sin(t * (5.0 if reject and resolved else 2.6))
            gw = 4 + int(2 * pulse)
            if resolved and reject:
                # broken link: gap in the middle + jagged spark
                gap = 0.10
                d.line([pa, (mid[0] - (pb[0] - pa[0]) * gap,
                             mid[1] - (pb[1] - pa[1]) * gap)], fill=(*col, 240), width=gw)
                d.line([(mid[0] + (pb[0] - pa[0]) * gap,
                         mid[1] + (pb[1] - pa[1]) * gap), pb], fill=(*col, 240), width=gw)
                # spark marks at the break
                for s in (-1, 1):
                    d.line([mid, (mid[0] + s * 18, mid[1] - 22)], fill=(*col, 220), width=3)
            else:
                d.line([pa, end], fill=(*col, 240), width=gw)
            # glow
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line([pa, pb if resolved else end],
                                    fill=(*col, 80), width=gw + 8)
            gl = gl.filter(ImageFilter.GaussianBlur(6))
            frame = Image.alpha_composite(frame.convert("RGBA"), gl).convert("RGB")

        sop = look.ease_out_cubic(min(1.0, t / 0.5))
        frame = pin(frame, pa, a, sop)
        frame = pin(frame, pb, b, look.ease_out_cubic(min(1.0, max(0.0, (t - 0.15) / 0.5))))

        # outcome word at the midpoint after resolve
        if resolved:
            ro = look.ease_out_cubic(min(1.0, (t - 1.5) / 0.4))
            word = "BROKEN" if reject else "AGREED"
            wi = look.text_with_glow(word, label_font,
                                     fill=(bad_col if reject else ok_col),
                                     glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=10)
            frame = look.paste_center(frame, wi, cx=int(mid[0]),
                                      cy=int(mid[1]) + (34 if diag else -30), opacity=ro)

        if title:
            ti = max(0.0, min(1.0, (t - 0.1) / 0.5))
            chip = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                       glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=12)
            cw, ch = chip.width, chip.height
            bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(bar).rounded_rectangle(
                [(w - cw) // 2 - 24, int(h * 0.07) - 8, (w + cw) // 2 + 24,
                 int(h * 0.07) + ch + 8], radius=10, fill=(*pal["bg_b"], int(210 * ti)))
            frame = Image.alpha_composite(frame.convert("RGBA"), bar).convert("RGB")
            frame = look.paste_center(frame, chip, cx=w // 2,
                                      cy=int(h * 0.07) + ch // 2, opacity=ti)

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
