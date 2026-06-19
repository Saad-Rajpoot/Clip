"""Primitive: comparison_split.

A two-sided contrast — A versus B — given a cinematic split. Each side gets a
refined label, an optional big numeral with a proportional bar that grows + counts
up, and an optional sub-line; a thin gold divider with a circular "VS" medallion
separates them. Each side reveals from its own edge. Over a graded, vignetted,
grained background with a slow parallax drift.

Forensic principle (NOT asset copy): MagnatesMedia stages two entities side by
side on contrast beats; the premium is the grade + serif type + the framed
divider, not a UI widget. If only labels are given it stays a clean two-panel
contrast; with values it adds proportional bars.

Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", left="Standard Oil", right="The Independents",
           leftval=90, rightval=10, prefix="", suffix="%", palette_name="amber_gold")
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
    "id": "comparison_split", "family": "comparison",
    "roles": ["comparison", "contrast", "versus", "ratio"],
    "niches_ok": ["business", "geopolitics", "history", "tech", "crime", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.0, 7.5],
    "easing": "easeOutCubic", "audio_cue": "soft_double_impact",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    # V3.4 — composition variants (seeded/anti-repeat). Values + labels identical
    # across both; only the layout differs (side-by-side vs over/under rows).
    "layout_variants": ["vs_center", "stacked_rows"],
    "review_override": ["left", "right", "leftval", "rightval", "vs", "palette"],
    "fallback": "two-panel label contrast if no values are supplied",
}


def _wrap(draw, text, font, max_w, max_lines=2):
    words, lines, cur = (text or "").split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _num(v, prefix, suffix, decimals):
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    body = f"{fv:,.{decimals}f}" if decimals > 0 else f"{int(round(fv)):,}"
    return f"{prefix}{body}{suffix}"


def _stacked_frame(frame, pal, left, right, lv, rv, has_vals, vmax, t, dur, w, h,
                   lab_font, num_font, sub_font, prefix, suffix, decimals, vs):
    """V3.4 stacked_rows: A over B — each a horizontal row (label left, numeral,
    proportional bar). Same values/labels as vs_center, a different silhouette."""
    d = ImageDraw.Draw(frame, "RGBA")
    rows = (("t", int(h * 0.34), left, lv), ("b", int(h * 0.64), right, rv))
    x_lab = int(w * 0.50)
    for k, (_pos, cy, lab, val) in enumerate(rows):
        ent = look.ease_out_cubic(max(0.0, (t - (0.2 + k * 0.22)) / 0.6))
        if ent <= 0.01:
            continue
        dxoff = int((1 - ent) * (-50 if k == 0 else 50))
        li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                 glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0,
                                 pad=16)
        frame = look.paste_center(frame, li, cx=x_lab + dxoff, cy=cy - int(h * 0.055),
                                  opacity=ent)
        d = ImageDraw.Draw(frame, "RGBA")
        if has_vals:
            cf = look.ease_out_expo(min(1.0, max(0.0, (t - 0.4)) / (dur * 0.45)))
            ni = look.gold_fill(_num(val * cf, prefix, suffix, decimals), num_font,
                                pal, glow_radius=int(h * 0.012), glow_alpha=0.4 * cf)
            frame = look.paste_center(frame, ni, cx=int(w * 0.30), cy=cy, opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
            bx0 = int(w * 0.42)
            bw_full = int(w * 0.40 * (val / vmax))
            bw = int(bw_full * cf)
            d.rounded_rectangle([bx0, cy - 7, bx0 + int(w * 0.40), cy + 7], radius=7,
                                fill=(*pal["bg_b"], 180))
            if bw > 3:
                d.rounded_rectangle([bx0, cy - 7, bx0 + bw, cy + 7], radius=7,
                                    fill=(*pal["accent_hi"], 255))
    # thin divider + small VS between the two rows
    md = int(h * 0.49)
    dv = look.ease_out_cubic(min(1.0, t / 0.6))
    d.line([(int(w * 0.22), md), (int(w * (0.22 + 0.56 * dv)), md)],
           fill=(*pal["accent"], 180), width=2)
    va = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.4))
    if va > 0.01:
        vimg = look.text_with_glow(vs, sub_font, fill=pal["accent_hi"],
                                   glow=pal["glow"], glow_radius=6, glow_alpha=0.4 * va,
                                   pad=10)
        frame = look.paste_center(frame, vimg, cx=w * 0.5, cy=md, opacity=va)
    return frame


def render(out_path, *, left: str = "", right: str = "", leftval=None,
           rightval=None, leftsub: str = "", rightsub: str = "", prefix: str = "",
           suffix: str = "", decimals: int = 0, vs: str = "VS", dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    lay = (layout or "vs_center").strip().lower()        # V3.4 composition variant
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    left = (left or "A").strip()
    right = (right or "B").strip()
    has_vals = leftval is not None and rightval is not None
    try:
        lv = float(leftval) if leftval is not None else 0.0
        rv = float(rightval) if rightval is not None else 0.0
    except (TypeError, ValueError):
        has_vals, lv, rv = False, 0.0, 0.0
    vmax = max(lv, rv, 1e-6)

    lab_font = look.font("title", int(h * 0.052))
    num_font = look.font("numeral", int(h * 0.11))
    sub_font = look.font("label", int(h * 0.028))
    vs_font = look.font("title", int(h * 0.040))
    cxl, cxr = int(w * 0.275), int(w * 0.725)
    half_w = int(w * 0.34)
    midy = int(h * 0.47)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")

        if lay == "stacked_rows":                         # V3.4 over/under variant
            frame = _stacked_frame(frame, pal, left, right, lv, rv, has_vals, vmax,
                                   t, dur, w, h, lab_font, num_font, sub_font,
                                   prefix, suffix, decimals, vs)
            frame = look.vignette(frame, strength=0.58)
            frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
            fa = look.fade_alpha(t, dur, fps)
            if fa < 1.0:
                frame = look.fade_frame(frame, fa, pal)
            frame.save(td / f"f{i:05d}.png")
            continue

        # central divider draws top→bottom, then the VS medallion pops
        dv = look.ease_out_cubic(min(1.0, t / 0.6))
        dy0, dy1 = int(h * 0.24), int(h * 0.76)
        ye = int(dy0 + (dy1 - dy0) * dv)
        d.line([(w // 2, dy0), (w // 2, ye)], fill=(*pal["accent"], 220), width=2)

        for side, cx, lab, val, sub in (
                ("l", cxl, left, lv, leftsub), ("r", cxr, right, rv, rightsub)):
            ent = look.ease_out_cubic(max(0.0, (t - 0.2) / 0.6))
            if ent <= 0.01:
                continue
            dxoff = int((1 - ent) * (-60 if side == "l" else 60))
            # label (top)
            llines = _wrap(d, lab.upper(), lab_font, half_w * 0.96)
            for k, ln in enumerate(llines):
                limg = look.text_with_glow(ln, lab_font, fill=pal["text"],
                                           glow=pal["bg_b"], glow_radius=5,
                                           glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, limg, cx=cx + dxoff,
                                          cy=int(h * 0.30) + k * int(h * 0.060),
                                          opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
            if has_vals:
                cf = look.ease_out_expo(min(1.0, max(0.0, (t - 0.5)) / (dur * 0.45)))
                cur = val * cf
                nimg = look.gold_fill(_num(cur, prefix, suffix, decimals),
                                      num_font, pal, glow_radius=int(h * 0.016),
                                      glow_alpha=0.4 + 0.5 * cf)
                frame = look.paste_center(frame, nimg, cx=cx + dxoff, cy=midy,
                                          opacity=ent)
                d = ImageDraw.Draw(frame, "RGBA")
                # proportional bar under the numeral
                bw_full = int(half_w * 0.82 * (val / vmax))
                bw = int(bw_full * cf)
                by = int(h * 0.60)
                bx0 = cx - bw_full // 2
                d.rounded_rectangle([bx0, by, bx0 + half_w * 0.82, by + 10],
                                    radius=5, fill=(*pal["bg_b"], 180))
                if bw > 2:
                    d.rounded_rectangle([bx0, by, bx0 + bw, by + 10], radius=5,
                                        fill=(*pal["accent_hi"], 255))
            if sub:
                sa = look.ease_out_cubic(max(0.0, (t - 0.8) / 0.6))
                simg = look.text_with_glow(sub.upper(), sub_font, fill=pal["muted"],
                                           glow=pal["bg_b"], glow_radius=3,
                                           glow_alpha=0.0, pad=12)
                frame = look.paste_center(frame, simg, cx=cx,
                                          cy=int(h * 0.66 if has_vals else 0.46 * h),
                                          opacity=sa)
                d = ImageDraw.Draw(frame, "RGBA")

        # VS medallion at centre, after the divider
        va = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.4))
        if va > 0.01:
            mr = int(h * 0.052)
            mc = (w // 2, midy)
            d.ellipse([mc[0] - mr, mc[1] - mr, mc[0] + mr, mc[1] + mr],
                      fill=(*pal["bg_b"], int(235 * va)),
                      outline=(*pal["accent"], int(240 * va)), width=3)
            vimg = look.text_with_glow(vs, vs_font, fill=pal["accent_hi"],
                                       glow=pal["glow"], glow_radius=8,
                                       glow_alpha=0.4 * va, pad=14)
            frame = look.paste_center(frame, vimg, cx=mc[0], cy=mc[1], opacity=va)

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
    import shutil as _sh
    _sh.rmtree(td, ignore_errors=True)        # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
