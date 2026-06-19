"""Primitive: supply_route_dashes.

An external-influence / logistics beat — a textured map base holds with a slow
push-in while DASHED ROUTES draw from a source node out to one or more
destinations, and small dots TRAVEL along each route (source → destination) to
show flow/direction. Node pins drop in with labels. The grammar of "X is
supplying / funding / shipping to Y" — used constantly in mapped geopolitics,
war (arms supply) and business (trade lanes).

Distinct from `map_route_spread` (one hero route with a comet head over many
stops) and `world_map_arc` (a great-circle arc): this is a one-to-many SUPPLY
fan of dashed, dotted, directional lanes.

Forensic principle (NOT an asset copy): premium mapped storytelling shows supply
/ influence as dashed lines with traveling dots between pinned nodes on a muted
textured base. Rebuilt from that PRINCIPLE with Vidlore's own look system —
no reference asset, colour or layout reproduced. Pure-local, deterministic.

    render("x.mp4", source="USSR", dests=["IRAQ", "SYRIA"])
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
    "id": "supply_route_dashes", "family": "maps",
    "roles": ["supply", "influence", "logistics", "trade", "funding", "flow"],
    "niches_ok": ["geopolitics", "business", "history", "tech", "war", "crime"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_tick",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["fan_right", "fan_out"],
    "review_override": ["source", "dests", "map_image", "palette"],
    "fallback": "map_route_spread for a single ordered route",
    "required_inputs": ["source"],
}


def _node_layout(w, h, k, seed, fan_out):
    """Source node + k destination nodes (seeded, readable spread)."""
    rng = np.random.RandomState((seed * 40503 + 7) & 0xFFFFFFFF)
    src = (0.24 * w, 0.62 * h)
    dests = []
    if fan_out:
        for i in range(k):
            ang = math.radians(-58 + (116 * (i + 0.5) / max(1, k)))
            r = (0.34 + 0.10 * rng.rand()) * w
            dests.append((src[0] + r * math.cos(ang), src[1] + r * math.sin(ang)))
    else:
        for i in range(k):
            fy = 0.26 + 0.5 * (i / max(1, k - 1)) if k > 1 else 0.4
            dx = (0.60 + 0.16 * rng.rand()) * w
            dests.append((dx, fy * h + (rng.rand() - 0.5) * 0.08 * h))
    return src, [(float(np.clip(x, 0.08 * w, 0.93 * w)),
                  float(np.clip(y, 0.14 * h, 0.9 * h))) for (x, y) in dests]


def _dashed(d, p0, p1, prog, col, dash=22, gap=16, width=3):
    """Draw a dashed segment from p0→p1 revealed to `prog` (0..1)."""
    x0, y0 = p0; x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    drawn = 0.0
    end = L * prog
    while drawn < end:
        a = drawn
        b = min(drawn + dash, end)
        d.line([(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)],
               fill=col, width=width)
        drawn += dash + gap


def _drop_coincident(src_pt, dest_pts, dests_lbl, *, min_px=1.0):
    """Drop destinations that project onto (or within `min_px` of) the SOURCE
    pixel. The gazetteer deliberately aliases some sources to a city coordinate
    (e.g. USSR / Soviet Union → Moscow's lon,lat — see maps/geo.py), so a derived
    destination such as "Moscow" can resolve to the EXACT source pixel. That route
    has zero length: it cannot be drawn, divides-by-zero in the traveling-dots
    block, and would stack a redundant pin + label on the source node. Such
    destinations are removed before drawing. Order is preserved; length-mismatched
    inputs are returned unchanged (defensive). Returns (dest_pts, dests_lbl)."""
    if len(dest_pts) != len(dests_lbl):
        return dest_pts, dests_lbl
    keep = [k for k, dp in enumerate(dest_pts)
            if math.hypot(dp[0] - src_pt[0], dp[1] - src_pt[1]) >= min_px]
    if len(keep) == len(dest_pts):
        return dest_pts, dests_lbl
    return [dest_pts[k] for k in keep], [dests_lbl[k] for k in keep]


def render(out_path, *, source: str = "", dests=None, map_image=None,
           bg_image=None, dur: float = 5.5, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "cold_steel", layout: str = "",
           seed: int = 0, reference: bool = False, borders: bool = True,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    source = str(source or "").strip()
    dests = [str(x).strip() for x in (dests or []) if str(x).strip()][:4] or ["DEST"]
    fan_out = layout == "fan_out"

    src_img = map_image or bg_image
    bed = None
    proj = None
    dests_lbl = dests
    if src_img and Path(str(src_img)).exists():
        try:
            bed = look.grade_media(Image.open(src_img).convert("RGB").resize((w, h)),
                                   pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            bed = None
    # REAL geography: project source + resolvable destinations onto a real
    # coastline/border bed framed to the region (else synthetic fallback below).
    geo_src = geo.resolve(source) if bed is None else None
    geo_dests = [(dn, geo.resolve(dn)) for dn in dests] if bed is None else []
    geo_dests = [(dn, ll) for dn, ll in geo_dests if ll]
    if bed is None and geo_src and geo_dests:
        bbox = geo.region_bbox([geo_src] + [ll for _, ll in geo_dests], pad_frac=0.5)
        proj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=proj, highlight=[source],
                               reference=reference, borders=borders)
        src_pt = proj(*geo_src)
        dest_pts = [proj(*ll) for _, ll in geo_dests]
        dests_lbl = [dn for dn, _ in geo_dests]
    if bed is None:
        bed = _antique_map(w, h, seed, pal)
    if proj is None:
        src_pt, dest_pts = _node_layout(w, h, len(dests), seed, fan_out)
    # A destination can resolve to the SOURCE pixel (e.g. the gazetteer aliases
    # USSR / Soviet Union to Moscow's coordinate): drop those zero-length routes so
    # they neither divide-by-zero in the dots block nor stack a redundant node on
    # the source. The dots block below also guards L >= 1 as defence-in-depth.
    dest_pts, dests_lbl = _drop_coincident(src_pt, dest_pts, dests_lbl)
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    label_font = look.font("label", int(h * 0.026))

    def pin(frame, pt, label, ink, op):
        d = ImageDraw.Draw(frame, "RGBA")
        cx, cy = int(pt[0]), int(pt[1])
        rr = 7
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                  fill=(*ink, int(235 * op)), outline=(*pal["bg_b"], int(200 * op)),
                  width=2)
        return frame

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        d = ImageDraw.Draw(frame, "RGBA")

        # each route draws (staggered), then traveling dots run along it
        for j, dp in enumerate(dest_pts):
            t_start = 0.4 + j * 0.45
            prog = look.ease_out_cubic(min(1.0, max(0.0, (t - t_start) / 1.0)))
            if prog <= 0:
                continue
            _dashed(d, src_pt, dp, prog, (*accent, 200), width=3)
            # traveling dots once the line is ~half drawn
            L = math.hypot(dp[0] - src_pt[0], dp[1] - src_pt[1])
            if prog > 0.25 and L >= 1.0:      # guard zero-length (coincident places)
                ux, uy = (dp[0] - src_pt[0]) / L, (dp[1] - src_pt[1]) / L
                phase = (t * 0.33) % 1.0
                for kdot in range(4):
                    fpos = (phase + kdot / 4.0) % 1.0
                    if fpos > prog:
                        continue
                    dx, dy = src_pt[0] + ux * L * fpos, src_pt[1] + uy * L * fpos
                    d.ellipse([dx - 4, dy - 4, dx + 4, dy + 4],
                              fill=(*accent_hi, 235))

        # destination + source pins (dots only); labels laid out collision-free
        items = []
        for j, dp in enumerate(dest_pts):
            op = look.ease_out_cubic(min(1.0, max(0.0, (t - (0.4 + j * 0.45)) / 0.5)))
            if op > 0.01:
                frame = pin(frame, dp, "", accent_hi, op)
                items.append({"text": dests_lbl[j], "x": dp[0], "y": dp[1],
                              "priority": 2, "opacity": op, "size_frac": 0.028})
        sop = look.ease_out_cubic(min(1.0, t / 0.5))
        frame = pin(frame, src_pt, "", (236, 224, 198), sop)
        items.append({"text": source, "x": src_pt[0], "y": src_pt[1],
                      "priority": 3, "opacity": sop, "size_frac": 0.033})
        frame = geo.layout_labels(frame, items, pal)

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
