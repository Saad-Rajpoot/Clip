"""Primitive: org_hierarchy_tree.

A structure beat — a single ROOT node at the top with 2-4 CHILD nodes below it,
joined by clean elbow connectors that draw down from the root to a horizontal bus
and into each child, revealed top-down. A power chart: who sits above whom.

Distinct from `money_flow_empire` (a RADIAL center→branches graph showing reach /
flow) and `bullet_list` (a LEFT→RIGHT ordered sequence): this is a vertical
TOP-DOWN hierarchy — the holding company over its subsidiaries, the boss over the
lieutenants, the ministry over its agencies.

Forensic principle (NOT asset copy): MagnatesMedia builds an org/power chart node
by node so the structure lands; the premium is the grade + serif type + clean
elbow connectors + staggered restraint, not a clip-art org chart. Pure-local.

    render("x.mp4", root="Standard Oil Trust",
           children=["Standard of Ohio", "Standard of New Jersey",
                     "Standard of New York"], title="THE TRUST")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "org_hierarchy_tree", "family": "diagrams",
    "roles": ["hierarchy", "structure", "org", "chain", "control"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_build_tick",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["top_down"],
    "review_override": ["root", "children", "title", "palette"],
    "fallback": "kinetic_keyword on the root if no children are available",
}


def _coerce(children):
    out = []
    for c in (children or []):
        if isinstance(c, dict):
            out.append(str(c.get("label") or c.get("name") or "").strip())
        elif isinstance(c, (list, tuple)) and c:
            out.append(str(c[0]).strip())
        elif c:
            out.append(str(c).strip())
    return [c for c in out if c][:4]


def _chip(d, cx, cy, text, font, pal, *, alpha, accent, pad_x, pad_y):
    """Draw a rounded chip centred at (cx,cy) sized to the text; returns its box."""
    tw = d.textlength(text, font=font)
    bx0, by0 = int(cx - tw / 2 - pad_x), int(cy - pad_y)
    bx1, by1 = int(cx + tw / 2 + pad_x), int(cy + pad_y)
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=10,
                        fill=(*pal["bg_b"], int(235 * alpha)),
                        outline=(*accent, int(255 * alpha)), width=3)
    return bx0, by0, bx1, by1


def render(out_path, *, root: str = "", children=None, title: str = "",
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    root = (root or "").strip() or "Root"
    kids = _coerce(children)
    if not kids:
        kids = ["—"]
    m = len(kids)

    title_font = look.font("label", int(h * 0.030))
    root_font = look.font("title", int(h * 0.040))
    kid_font = look.font("label", int(h * 0.028))
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])

    rcx, rcy = w * 0.5, h * 0.30
    kcy = int(h * 0.66)
    bus_y = int((rcy + kcy) / 2)
    span = w * (0.66 if m > 1 else 0.0)
    left = w * 0.5 - span / 2
    kxs = [left + (span * (k / (m - 1)) if m > 1 else 0) for k in range(m)]

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.13,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # connectors (draw after the root appears, before children land)
        ca = look.ease_out_cubic(max(0.0, (t - 0.5) / 0.5))
        if ca > 0.01:
            col = (*accent, int(220 * ca))
            ry = int(rcy + h * 0.035)                       # root bottom
            yb = int(ry + (bus_y - ry) * ca)
            d.line([(int(rcx), ry), (int(rcx), yb)], fill=col, width=3)
            if ca > 0.5:                                    # horizontal bus
                ba = (ca - 0.5) / 0.5
                bx0 = int(rcx + (kxs[0] - rcx) * ba)
                bx1 = int(rcx + (kxs[-1] - rcx) * ba)
                d.line([(bx0, bus_y), (bx1, bus_y)], fill=col, width=3)
                for k in range(m):                          # drops into each child
                    da = look.ease_out_cubic(max(0.0, (t - (0.9 + k * 0.12)) / 0.3))
                    if da > 0.01:
                        ye = int(bus_y + (kcy - int(h * 0.045) - bus_y) * da)
                        d.line([(int(kxs[k]), bus_y), (int(kxs[k]), ye)],
                               fill=(*accent, int(220 * da)), width=3)
        # root chip
        ra = look.ease_out_cubic(min(1.0, t / 0.45))
        _chip(d, rcx, rcy, root, root_font, pal, alpha=ra, accent=accent_hi,
              pad_x=int(w * 0.018), pad_y=int(h * 0.035))
        ri = look.text_with_glow(root, root_font, fill=pal["text"],
                                 glow=pal["bg_b"], glow_radius=4,
                                 glow_alpha=0.0, pad=10)
        frame = look.paste_center(frame, ri, cx=rcx, cy=rcy, opacity=ra)
        d = ImageDraw.Draw(frame, "RGBA")
        # child chips, staggered
        for k, kid in enumerate(kids):
            ka = look.ease_out_cubic(max(0.0, (t - (1.05 + k * 0.12)) / 0.4))
            if ka <= 0.01:
                continue
            _chip(d, kxs[k], kcy, kid, kid_font, pal, alpha=ka, accent=accent,
                  pad_x=int(w * 0.012), pad_y=int(h * 0.030))
            ki = look.text_with_glow(kid, kid_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ki, cx=kxs[k], cy=kcy, opacity=ka)
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
