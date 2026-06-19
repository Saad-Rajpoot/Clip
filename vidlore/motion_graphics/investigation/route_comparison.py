"""Primitive: route_comparison.

A "two stories, one journey" beat — between a shared START and END node, two paths
are drawn: the muted reference route (the official account) and the highlighted
route (what actually happened), each carrying a label and a metric chip (time /
distance / claim). The discrepancy is the point. The grammar of "they said this —
the evidence shows that" in spy / true-crime / investigative history.

Schematic by design (not survey geography — the map family handles real routes):
two contrasting arcs between the same endpoints make the *difference* legible.
Distinct from `sightline_trajectory` (one reconstructed line) and `comparison_split`
(two static panels): this weighs two ROUTES against each other.

Forensic principle (NOT an asset copy): premium docs contrast an official route
with an actual route between the same two points, one de-emphasised, one stressed,
each with a metric. Rebuilt from that PRINCIPLE with Vidlore's own look system.
Pure-local, deterministic, no paid API.

    render("x.mp4", title="THE MISSING HOUR", start="Hotel", end="Embassy",
           route_a={"label":"Official account","metric":"12 min"},
           route_b={"label":"CCTV timeline","metric":"47 min","highlight":True})
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
    "id": "route_comparison", "family": "investigation",
    "roles": ["route_comparison", "two_routes", "official_vs_actual",
              "discrepancy", "timeline_conflict", "alibi"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "business"],
    "intensity_range": [2, 4], "duration_range": [5.0, 7.5],
    "easing": "easeOutCubic", "audio_cue": "draw_soft",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["arc"],
    "review_override": ["title", "start", "end", "route_a", "route_b", "palette"],
    "fallback": "comparison_split for non-spatial either/or contrasts",
    "required_inputs": ["route_a", "route_b"],
}


def _bezier(p0, p1, p2, steps=72):
    pts = []
    for s in range(steps + 1):
        t = s / steps
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        pts.append((x, y))
    return pts


def _draw_path(d, pts, prog, fill, width, dashed=False):
    m = max(1, int(len(pts) * max(0.0, min(1.0, prog))))
    seg = pts[:m]
    if len(seg) < 2:
        return seg[-1] if seg else pts[0]
    if dashed:
        on = True; acc = 0.0
        for a, b in zip(seg[:-1], seg[1:]):
            if on:
                d.line([a, b], fill=fill, width=width)
            acc += math.hypot(b[0] - a[0], b[1] - a[1])
            if acc >= (16 if on else 11):
                on = not on; acc = 0.0
    else:
        d.line(seg, fill=fill, width=width, joint="curve")
    return seg[-1]


def _norm_route(r, default_label):
    if isinstance(r, dict):
        return (str(r.get("label") or default_label).strip(),
                str(r.get("metric") or "").strip(),
                bool(r.get("highlight")))
    if isinstance(r, (list, tuple)):
        return (str(r[0]) if r else default_label, str(r[1]) if len(r) > 1 else "", False)
    return (str(r or default_label), "", False)


def render(out_path, *, title: str = "", start: str = "Start", end: str = "End",
           route_a=None, route_b=None, dur: float = 6.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    la, ma, hi_a = _norm_route(route_a, "Route A")
    lb, mb, hi_b = _norm_route(route_b, "Route B")
    if not (hi_a or hi_b):                                         # default: stress route B
        hi_b = True
    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"]); muted = tuple(pal["muted"])
    start = str(start or "Start").strip(); end = str(end or "End").strip()

    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    gd = ImageDraw.Draw(bed, "RGBA")
    step = int(h * 0.06)
    for x in range(0, w, step):
        gd.line([(x, 0), (x, h)], fill=(*muted, 14), width=1)
    for y in range(0, h, step):
        gd.line([(0, y), (w, y)], fill=(*muted, 14), width=1)

    ax, ay = int(w * 0.16), int(h * 0.70)                          # START node
    bx, by = int(w * 0.84), int(h * 0.34)                          # END node
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1.0
    perp = (-dy / L, dx / L)                                       # unit perpendicular
    # route A bows one way (reference), route B bows further the other (detour)
    ctrlA = (mx + perp[0] * L * 0.20, my + perp[1] * L * 0.20)
    ctrlB = (mx - perp[0] * L * 0.42, my - perp[1] * L * 0.42)
    pathA = _bezier((ax, ay), ctrlA, (bx, by))
    pathB = _bezier((ax, ay), ctrlB, (bx, by))

    title_font = look.font("title", int(h * 0.036))
    node_font = look.font("label", int(h * 0.025))
    rlab_font = look.font("label", int(h * 0.023))
    met_font = look.font("numeral", int(h * 0.030))

    def _chip(text, fg, big=False):
        f = met_font if big else rlab_font
        ti = look.text_with_glow(text, f, fill=fg, glow=pal["bg_b"],
                                 glow_radius=3, glow_alpha=0.0, pad=5)
        return ti

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=8)
            d.rectangle([int(w * 0.06), int(h * 0.10), int(w * 0.06) + int(ti.width * 0.5),
                         int(h * 0.105)], fill=(*accent, int(200 * ta)))
            frame = look.paste_center(frame, ti, cx=int(w * 0.06) + ti.width // 2,
                                      cy=int(h * 0.145), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # reference route (muted) draws first, 0.5-1.6s
        pa = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.5) / 1.1)))
        if pa > 0.01:
            _draw_path(d, pathA, pa, (*muted, 210), 3, dashed=hi_b and not hi_a)
        # stressed route (accent) draws second, 1.1-2.3s
        pb = look.ease_out_cubic(min(1.0, max(0.0, (t - 1.1) / 1.2)))
        if pb > 0.01:
            head = _draw_path(d, pathB, pb, (*accent, 240), 5, dashed=False)
            hx, hy = int(head[0]), int(head[1])
            d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=(*accent, 240))

        # route labels + metric chips near each apex
        if pa > 0.6:
            aa = min(1.0, (pa - 0.6) / 0.3)
            apexA = pathA[len(pathA) // 2]
            txt = la + (f"  ·  {ma}" if ma else "")
            ci = _chip(txt, muted)
            ox = int(apexA[0] + perp[0] * 36); oy = int(apexA[1] + perp[1] * 36)
            frame = look.paste_center(frame, ci, cx=ox, cy=oy, opacity=aa)
            d = ImageDraw.Draw(frame, "RGBA")
        if pb > 0.7:
            ab = min(1.0, (pb - 0.7) / 0.3)
            apexB = pathB[len(pathB) // 2]
            li = _chip(lb.upper(), accent)
            mi = _chip(mb, ink, big=True) if mb else None
            ox = int(apexB[0] - perp[0] * 46); oy = int(apexB[1] - perp[1] * 46)
            # scrim behind the stressed label/metric
            blockw = max(li.width, mi.width if mi else 0)
            blockh = li.height + (mi.height + 6 if mi else 0)
            d.rounded_rectangle([ox - blockw // 2 - 12, oy - blockh // 2 - 10,
                                 ox + blockw // 2 + 12, oy + blockh // 2 + 10],
                                radius=8, fill=(*pal["bg_b"], int(195 * ab)))
            frame = look.paste_center(frame, li, cx=ox,
                                      cy=oy - (mi.height // 2 + 3 if mi else 0), opacity=ab)
            if mi:
                frame = look.paste_center(frame, mi, cx=ox, cy=oy + li.height // 2 + 3, opacity=ab)
            d = ImageDraw.Draw(frame, "RGBA")

        # endpoint nodes
        na = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.25) / 0.4)))
        if na > 0.01:
            for (nx, ny, lab, al) in ((ax, ay, start, "below"), (bx, by, end, "above")):
                d.ellipse([nx - 10, ny - 10, nx + 10, ny + 10], outline=(*ink, int(240 * na)), width=3)
                d.ellipse([nx - 4, ny - 4, nx + 4, ny + 4], fill=(*ink, int(240 * na)))
                ni = look.text_with_glow(lab.upper(), node_font, fill=ink, glow=pal["bg_b"],
                                         glow_radius=3, glow_alpha=0.0, pad=5)
                dyl = int(h * 0.045) if al == "below" else -int(h * 0.045)
                frame = look.paste_center(frame, ni, cx=nx, cy=ny + dyl, opacity=na)
            d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.58)
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
