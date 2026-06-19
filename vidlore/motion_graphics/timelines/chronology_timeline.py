"""Primitive: chronology_timeline.

An EDITORIAL time anchor — not a boxed UI widget. Two modes, chosen by input:

  • ERA BAND (single year/era): a large serif year centred over a thin gold rule
    with a condensed label — the "By 1882…" beat.
  • SPINE (2–4 events): a thin gold spine draws across the lower third; year+label
    nodes pop in sequence while a travelling accent marker rides the spine and
    lights each node as it passes.

Both sit over a graded, vignetted, grained background with a slow parallax drift,
so a chronology reads instantly premium and orients the viewer. Forensic
principle (NOT asset copy): MagnatesMedia leans on year/era anchors constantly;
the premium comes from the grade + refined serif type + restraint, not a chart.

Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", events=[["1865","Refinery #1"],["1870","Standard Oil"],
           ["1882","The Trust"]], title="THE RISE", palette_name="amber_gold")
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
    "id": "chronology_timeline", "family": "timelines",
    "roles": ["chapter", "turn", "origin", "timeline", "era"],
    "niches_ok": ["history", "biography", "business", "crime", "geopolitics", "tech"],
    "intensity_range": [2, 4], "duration_range": [3.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_tick_bed",
    "repeat_cooldown_s": 40, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["spine_bottom", "era_band"],
    "review_override": ["events", "year", "label", "title", "palette"],
    "fallback": "era band (single year) if only one date is available",
}


def _coerce_events(events, year, label):
    """Normalise to a list of (year_str, label_str). Accepts list of [y,l] /
    (y,l) / {'year','label'}, or a single year+label."""
    out = []
    if events:
        for e in events:
            if isinstance(e, dict):
                y = str(e.get("year", "")).strip()
                lb = str(e.get("label", "")).strip()
            elif isinstance(e, (list, tuple)) and len(e) >= 1:
                y = str(e[0]).strip()
                lb = str(e[1]).strip() if len(e) > 1 else ""
            else:
                y = str(e).strip()
                lb = ""
            if y:
                out.append((y, lb))
    elif year is not None:
        out.append((str(year).strip(), str(label or "").strip()))
    return out[:4]


def _wrap(draw, text, font, max_w):
    if not text:
        return []
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines[:2]


def render(out_path, *, events=None, year=None, label: str = "", title: str = "",
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    evs = _coerce_events(events, year, label)
    if not evs:                                    # nothing to show → safe slate
        evs = [(str(year or ""), label or "")]
    band = (len(evs) == 1) or (layout == "era_band")

    yr_font = look.font("numeral", int(h * (0.135 if band else 0.060)))
    lb_font = look.font("label", int(h * (0.034 if band else 0.026)))
    kick_font = look.font("label", int(h * 0.026))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")

        # optional kicker title (top, condensed, letter-spaced feel via spaces)
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            kick = look.text_with_glow(title.upper(), kick_font, fill=pal["muted"],
                                       glow=pal["bg_b"], glow_radius=4,
                                       glow_alpha=0.0, pad=18)
            frame = look.paste_center(frame, kick, cx=w / 2, cy=h * 0.20,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        if band:
            yr, lb = evs[0]
            ent = look.ease_out_cubic(min(1.0, t / 0.6))
            sc = 1.05 - 0.05 * ent
            yimg = look.gold_fill(yr, yr_font, pal,
                                  glow_radius=int(h * 0.02),
                                  glow_alpha=0.4 + 0.5 * ent)
            frame = look.paste_center(frame, yimg, cx=w / 2, cy=h * 0.46,
                                      scale=sc, opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
            rule = look.ease_out_cubic(max(0.0, (t - 0.4) / 0.6))
            look.hairline(d, int(w / 2), int(h * 0.585),
                          int(w * 0.13 * rule), pal)
            if lb:
                la = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.6))
                limg = look.text_with_glow(lb.upper(), lb_font, fill=pal["text"],
                                           glow=pal["bg_b"], glow_radius=5,
                                           glow_alpha=0.0, pad=22)
                frame = look.paste_center(frame, limg, cx=w / 2, cy=h * 0.64,
                                          opacity=la)
        else:
            # horizontal spine across the lower third
            sy = int(h * 0.62)
            x0, x1 = int(w * 0.16), int(w * 0.84)
            draw_p = look.ease_out_cubic(min(1.0, t / 0.7))
            xe = int(x0 + (x1 - x0) * draw_p)
            d.line([(x0, sy), (xe, sy)], fill=(*pal["accent"], 235), width=3)
            m = len(evs)
            xs = [int(x0 + (x1 - x0) * (k / (m - 1))) for k in range(m)] if m > 1 \
                else [int((x0 + x1) / 2)]
            # travelling marker rides the spine after it draws
            mk = look.ease_in_out_cubic(max(0.0, (t - 0.5) / 0.8))
            mx = int(x0 + (x1 - x0) * mk)
            for k, (yr, lb) in enumerate(evs):
                nx = xs[k]
                lit = mx >= nx - 6                       # marker has reached node
                na = look.ease_out_cubic(
                    max(0.0, (t - (0.6 + k * 0.45)) / 0.5))
                if na <= 0.01:
                    continue
                rr = int(h * 0.012)
                ring = pal["accent_hi"] if lit else pal["muted"]
                d.ellipse([nx - rr - 4, sy - rr - 4, nx + rr + 4, sy + rr + 4],
                          outline=(*ring, int(220 * na)), width=2)
                d.ellipse([nx - rr, sy - rr, nx + rr, sy + rr],
                          fill=(*(pal["accent_hi"] if lit else pal["accent"]),
                                int(255 * na)))
                # year above, label below
                yimg = look.text_with_glow(
                    yr, yr_font, fill=pal["accent_hi"], glow=pal["glow"],
                    glow_radius=int(h * 0.012), glow_alpha=0.5 if lit else 0.0,
                    pad=16)
                frame = look.paste_center(frame, yimg, cx=nx,
                                          cy=sy - int(h * 0.085), opacity=na)
                d = ImageDraw.Draw(frame, "RGBA")
                if lb:
                    lines = _wrap(d, lb.upper(), lb_font, int((x1 - x0) / m * 0.92))
                    for li, ln in enumerate(lines):
                        limg = look.text_with_glow(
                            ln, lb_font, fill=pal["text"], glow=pal["bg_b"],
                            glow_radius=4, glow_alpha=0.0, pad=14)
                        frame = look.paste_center(
                            frame, limg, cx=nx,
                            cy=sy + int(h * 0.055) + li * int(h * 0.040),
                            opacity=na)
                    d = ImageDraw.Draw(frame, "RGBA")
            # marker glow
            if 0.0 < mk < 1.0:
                d.ellipse([mx - 6, sy - 6, mx + 6, sy + 6],
                          fill=(*pal["glow"], 255))

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
