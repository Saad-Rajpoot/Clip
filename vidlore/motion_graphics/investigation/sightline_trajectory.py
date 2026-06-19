"""Primitive: sightline_trajectory.

A forensic-reconstruction beat — on a charcoal schematic stage, an ORIGIN marker
(shooter / observer / vantage) anchors concentric range rings and a bearing arc,
and a dashed SIGHTLINE sweeps out along a compass bearing to a TARGET marker, with
a distance read-out at the mid-point. The grammar of "from here, along this line,
to there" — ballistics, line-of-sight, escape vectors, approach paths in spy /
true-crime / military history.

Diagrammatic (not a real map): the bearing, range rings and angle arc are an
honest schematic, not survey-grade geography. Distinct from `route_comparison`
(two routes weighed against each other) and `map_route_spread` (a multi-stop
journey on real geography): this is a SINGLE reconstructed line with bearing +
range instrumentation.

Forensic principle (NOT an asset copy): premium docs reconstruct a line of
sight/fire with an origin, a bearing arc, range rings and a dashed vector to a
target. Rebuilt from that PRINCIPLE with Vidlore's own look system. Pure-local,
deterministic, no paid API.

    render("x.mp4", title="LINE OF FIRE", origin="Sniper's nest",
           target="Motorcade", bearing=58, distance="265 m")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "sightline_trajectory", "family": "investigation",
    "roles": ["sightline", "trajectory", "line_of_sight", "line_of_fire",
              "vector", "approach", "bearing", "reconstruction"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "war"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "scan_soft",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["diagram"],
    "review_override": ["title", "origin", "target", "bearing", "distance",
                        "elevation", "palette"],
    "fallback": "annotated_detail_callout if no spatial relationship is implied",
    "required_inputs": ["origin", "target"],
}


def _dashed(d, p1, p2, prog, dash, gap, fill, width):
    """Draw a dashed segment p1->p2 revealed up to `prog` (0..1)."""
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1:
        return
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    reach = L * max(0.0, min(1.0, prog))
    s = 0.0
    while s < reach:
        e = min(s + dash, reach)
        d.line([(x1 + ux * s, y1 + uy * s), (x1 + ux * e, y1 + uy * e)],
               fill=fill, width=width)
        s += dash + gap


def render(out_path, *, title: str = "LINE OF SIGHT", origin: str = "", target: str = "",
           bearing: float = 52.0, distance: str = "", elevation: str = "",
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    origin = str(origin or "Origin").strip()
    target = str(target or "Target").strip()
    try:
        bearing = float(bearing)
    except Exception:                                              # noqa: BLE001
        bearing = 52.0
    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"]); muted = tuple(pal["muted"])
    ember = (206, 96, 80)

    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    # faint schematic graph paper
    gd = ImageDraw.Draw(bed, "RGBA")
    step = int(h * 0.06)
    for x in range(0, w, step):
        gd.line([(x, 0), (x, h)], fill=(*muted, 16), width=1)
    for y in range(0, h, step):
        gd.line([(0, y), (w, y)], fill=(*muted, 16), width=1)

    # geometry: origin lower-left, target along the bearing
    ox, oy = int(w * 0.27), int(h * 0.70)
    rad = math.radians(bearing)
    dirx, diry = math.sin(rad), -math.cos(rad)                  # 0deg = up/north, CW
    reach = min(w * 0.52, h * 0.62)
    tx = int(ox + dirx * reach); ty = int(oy + diry * reach)
    tx = max(int(w * 0.40), min(int(w * 0.86), tx))
    ty = max(int(h * 0.14), min(int(h * 0.62), ty))
    seg_len = math.hypot(tx - ox, ty - oy)
    ring_rs = [int(seg_len * f) for f in (0.34, 0.62, 0.9)]

    title_font = look.font("title", int(h * 0.034))
    lab_font = look.font("label", int(h * 0.024))
    val_font = look.font("numeral", int(h * 0.030))
    deg_font = look.font("numeral", int(h * 0.026))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # title chip
        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink,
                                     glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0, pad=8)
            d.rectangle([int(w * 0.06), int(h * 0.10), int(w * 0.06) + int(ti.width * 0.5),
                         int(h * 0.105)], fill=(*accent, int(200 * ta)))
            frame = look.paste_center(frame, ti, cx=int(w * 0.06) + ti.width // 2,
                                      cy=int(h * 0.145), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # range rings (fade in 0.3-1.0s)
        ra = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.3) / 0.7)))
        if ra > 0.01:
            for rr in ring_rs:
                d.ellipse([ox - rr, oy - rr, ox + rr, oy + rr],
                          outline=(*muted, int(60 * ra)), width=1)
            # north reference tick
            d.line([(ox, oy), (ox, oy - ring_rs[-1])], fill=(*muted, int(80 * ra)), width=1)
            ni = look.text_with_glow("N", deg_font, fill=muted, glow=pal["bg_b"],
                                     glow_radius=2, glow_alpha=0.0, pad=3)
            frame = look.paste_center(frame, ni, cx=ox, cy=oy - ring_rs[-1] - int(h * 0.02),
                                      opacity=ra)
            d = ImageDraw.Draw(frame, "RGBA")

        # bearing arc (north -> sightline) 0.6-1.2s
        ba = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.6) / 0.6)))
        if ba > 0.01:
            arc_r = int(ring_rs[0] * 0.7)
            sweep = bearing * ba
            d.arc([ox - arc_r, oy - arc_r, ox + arc_r, oy + arc_r],
                  start=-90, end=-90 + sweep, fill=(*accent, int(220 * ba)), width=3)
            if ba > 0.7:
                bi = look.text_with_glow(f"{int(round(bearing))}°", deg_font, fill=accent,
                                         glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0, pad=3)
                mid = math.radians(bearing * 0.5)
                lx = ox + math.sin(mid) * (arc_r + int(h * 0.03))
                ly = oy - math.cos(mid) * (arc_r + int(h * 0.03))
                frame = look.paste_center(frame, bi, cx=int(lx), cy=int(ly), opacity=ba)
                d = ImageDraw.Draw(frame, "RGBA")

        # dashed sightline vector (0.9 -> 1.9s)
        va = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.9) / 1.0)))
        if va > 0.01:
            _dashed(d, (ox, oy), (tx, ty), va, dash=18, gap=12,
                    fill=(*accent, 235), width=3)
            # travelling head marker
            hx = int(ox + (tx - ox) * va); hy = int(oy + (ty - oy) * va)
            d.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], fill=(*accent, 235))
            # distance read-out at mid once mostly drawn
            if distance and va > 0.55:
                da = min(1.0, (va - 0.55) / 0.3)
                di = look.text_with_glow(distance, val_font, fill=ink, glow=pal["bg_b"],
                                         glow_radius=3, glow_alpha=0.0, pad=4)
                mx, my = (ox + tx) // 2, (oy + ty) // 2
                d.rounded_rectangle([mx - di.width // 2 - 10, my - di.height // 2 - 6,
                                     mx + di.width // 2 + 10, my + di.height // 2 + 6],
                                    radius=6, fill=(*pal["bg_b"], int(190 * da)))
                frame = look.paste_center(frame, di, cx=mx, cy=my, opacity=da)
                d = ImageDraw.Draw(frame, "RGBA")

        # origin marker + label
        oa = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.15) / 0.4)))
        if oa > 0.01:
            d.ellipse([ox - 9, oy - 9, ox + 9, oy + 9], outline=(*accent, int(240 * oa)), width=3)
            d.ellipse([ox - 3, oy - 3, ox + 3, oy + 3], fill=(*accent, int(240 * oa)))
            oi = look.text_with_glow(origin.upper(), lab_font, fill=accent, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=5)
            frame = look.paste_center(frame, oi, cx=ox + oi.width // 2,
                                      cy=oy + int(h * 0.045), opacity=oa)
            d = ImageDraw.Draw(frame, "RGBA")

        # target marker (crosshair) + label, once vector arrives
        if va > 0.92:
            ca = min(1.0, (va - 0.92) / 0.08)
            cr = int(h * 0.022)
            d.ellipse([tx - cr, ty - cr, tx + cr, ty + cr], outline=(*ember, int(240 * ca)), width=3)
            d.line([(tx - cr - 6, ty), (tx + cr + 6, ty)], fill=(*ember, int(220 * ca)), width=2)
            d.line([(tx, ty - cr - 6), (tx, ty + cr + 6)], fill=(*ember, int(220 * ca)), width=2)
            tl = look.text_with_glow(target.upper(), lab_font, fill=ember,
                                     glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0, pad=5)
            frame = look.paste_center(frame, tl, cx=tx, cy=ty - int(h * 0.05), opacity=ca)
            d = ImageDraw.Draw(frame, "RGBA")
            if elevation:
                ei = look.text_with_glow(elevation, lab_font, fill=muted, glow=pal["bg_b"],
                                         glow_radius=2, glow_alpha=0.0, pad=4)
                frame = look.paste_center(frame, ei, cx=tx, cy=ty + int(h * 0.05), opacity=ca)
                d = ImageDraw.Draw(frame, "RGBA")

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
