"""Primitive: statistic_bar_reveal.

A clean breakdown beat — labelled values animate into frame. Distinct from
`gold_number_callout` (one hero figure) and `comparison_split` (two contrasted
sides): this is a small multi-bar proportion / ranking chart.

Forensic principle (NOT asset copy): MagnatesMedia animates stats into the frame
rather than dropping a flat chart; the premium is the grade + serif numerals +
refined bars + restraint. Pure-local (PIL + numpy → ffmpeg). No paid API.

V3.4 — three internal LAYOUT variants (seeded, anti-repeat; the director picks
`layout`, the values/labels are UNCHANGED across all three):
  • `columns`         — grouped vertical bars rising from a baseline (default).
  • `hero_bar`        — ONE dominant value as a big horizontal bar + giant numeral
                        (the "single-hero" stat beat; auto-picked when there is one
                        value and no explicit layout).
  • `horizontal_bars` — ranked horizontal rows: label left, bar grows right, value
                        at the end (a ranked-list silhouette, distinct from columns).

    render("x.mp4", bars=[["1870", 4], ["1880", 30], ["1890", 90]],
           title="STANDARD OIL PROFIT", suffix="M", prefix="$", layout="hero_bar")
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
    "id": "statistic_bar_reveal", "family": "charts",
    "roles": ["stat", "data", "breakdown", "scale", "ranking"],
    "niches_ok": ["all"],
    "intensity_range": [2, 4], "duration_range": [4.0, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_tick_rise",
    "repeat_cooldown_s": 50, "per_video_cap": 3, "cost": "low",
    # V3.4 — internal visual variants (seeded/anti-repeat). Values are identical
    # across variants; only the composition/animation differs.
    "layout_variants": ["columns", "hero_bar", "horizontal_bars"],
    "review_override": ["bars", "title", "prefix", "suffix", "palette"],
    "fallback": "single hero number if only one value is available",
}


def _coerce(bars):
    out = []
    for b in (bars or []):
        if isinstance(b, dict):
            lab, val = str(b.get("label", "")), b.get("value", 0)
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            lab, val = str(b[0]), b[1]
        else:
            continue
        try:
            out.append((lab.strip(), float(val)))
        except (TypeError, ValueError):
            continue
    return out[:4]


def _num(v, prefix, suffix, decimals):
    body = f"{v:,.{decimals}f}" if decimals > 0 else f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def _draw_title(frame, pal, title, font, t, *, cy):
    if not title:
        return frame
    ta = look.ease_out_cubic(min(1.0, t / 0.5))
    ti = look.text_with_glow(title.upper(), font, fill=pal["muted"],
                             glow=pal["bg_b"], glow_radius=4, glow_alpha=0.0, pad=16)
    return look.paste_center(frame, ti, cx=frame.width * 0.5, cy=cy, opacity=ta)


def _draw_columns(frame, pal, data, vmax, t, w, h, num_font, lab_font, prefix,
                  suffix, decimals):
    """Grouped vertical bars (the original composition)."""
    d = ImageDraw.Draw(frame, "RGBA")
    m = len(data)
    base_y, top_y = int(h * 0.74), int(h * 0.30)
    max_bh = base_y - top_y
    span = w * 0.62
    bw = int(min(span / m * 0.52, w * 0.12))
    xs = [int(w * 0.5 - span / 2 + span * ((k + 0.5) / m)) for k in range(m)]
    bl = look.ease_out_cubic(min(1.0, t / 0.6))
    d.line([(int(w * 0.5 - span / 2 * bl), base_y),
            (int(w * 0.5 + span / 2 * bl), base_y)],
           fill=(*pal["muted"], 200), width=2)
    for k, (lab, val) in enumerate(data):
        ent = look.ease_out_cubic(max(0.0, (t - (0.35 + k * 0.18)) / 0.7))
        if ent <= 0.01:
            continue
        full = int(max_bh * (val / vmax))
        bh = int(full * ent)
        x = xs[k]
        x0, x1 = x - bw // 2, x + bw // 2
        d.rounded_rectangle([x0, base_y - bh, x1, base_y], radius=4,
                            fill=(*pal["accent"], 255))
        if bh > 6:
            d.rounded_rectangle([x0, base_y - bh, x1, base_y - bh + 6], radius=3,
                                fill=(*pal["accent_hi"], 255))
        cur = val * ent
        ni = look.gold_fill(_num(cur, prefix, suffix, decimals), num_font, pal,
                            glow_radius=int(h * 0.012), glow_alpha=0.4 * ent)
        frame = look.paste_center(frame, ni, cx=x, cy=base_y - bh - int(h * 0.045),
                                  opacity=ent)
        d = ImageDraw.Draw(frame, "RGBA")
        if lab:
            li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0,
                                     pad=12)
            frame = look.paste_center(frame, li, cx=x, cy=base_y + int(h * 0.045),
                                      opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
    return frame


def _draw_hero(frame, pal, data, vmax, t, w, h, prefix, suffix, decimals):
    """ONE dominant value: a big horizontal bar grows L→R under a giant numeral."""
    lab, val = max(data, key=lambda kv: kv[1]) if data else ("", 1.0)
    d = ImageDraw.Draw(frame, "RGBA")
    ent = look.ease_out_cubic(min(1.0, t / 0.8))
    bar_y = int(h * 0.62)
    x0 = int(w * 0.19)
    full_w = int(w * 0.62)
    cur_w = int(full_w * ent)
    bh = int(h * 0.052)
    # track
    d.rounded_rectangle([x0, bar_y, x0 + full_w, bar_y + bh], radius=bh // 2,
                        outline=(*pal["muted"], 120), width=2)
    if cur_w > bh:
        d.rounded_rectangle([x0, bar_y, x0 + cur_w, bar_y + bh], radius=bh // 2,
                            fill=(*pal["accent"], 255))
        d.rounded_rectangle([x0, bar_y, x0 + cur_w, bar_y + max(3, bh // 5)],
                            radius=bh // 4, fill=(*pal["accent_hi"], 255))
    # giant numeral above the bar
    hero_font = look.font("numeral", int(h * 0.20))
    ni = look.gold_fill(_num(val * ent, prefix, suffix, decimals), hero_font, pal,
                        glow_radius=int(h * 0.02), glow_alpha=0.5 * ent)
    frame = look.paste_center(frame, ni, cx=w * 0.5, cy=int(h * 0.40), opacity=ent)
    if lab:
        lab_font = look.font("label", int(h * 0.034))
        li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                 glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0,
                                 pad=12)
        frame = look.paste_center(frame, li, cx=w * 0.5, cy=bar_y + int(h * 0.075),
                                  opacity=ent)
    return frame


def _draw_hbars(frame, pal, data, vmax, t, w, h, lab_font, num_font, prefix,
                suffix, decimals):
    """Ranked horizontal rows: label left, bar grows right, value at the end."""
    d = ImageDraw.Draw(frame, "RGBA")
    m = len(data)
    top = int(h * 0.34)
    row_h = int(h * 0.40 / max(1, m))
    bar_h = int(min(row_h * 0.5, h * 0.05))
    x_lab = int(w * 0.20)
    x0 = int(w * 0.40)
    full_w = int(w * 0.40)
    for k, (lab, val) in enumerate(data):
        ent = look.ease_out_cubic(max(0.0, (t - (0.3 + k * 0.16)) / 0.7))
        if ent <= 0.01:
            continue
        cy = top + int((k + 0.5) * row_h)
        cur_w = int(full_w * (val / vmax) * ent)
        # label (right-aligned to the bar start)
        if lab:
            li = look.text_with_glow(lab.upper(), lab_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0,
                                     pad=8)
            frame = look.paste_center(frame, li, cx=x_lab, cy=cy, opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
        # bar
        if cur_w > 3:
            d.rounded_rectangle([x0, cy - bar_h // 2, x0 + cur_w, cy + bar_h // 2],
                                radius=bar_h // 2, fill=(*pal["accent"], 255))
            d.rounded_rectangle([x0, cy - bar_h // 2, x0 + cur_w,
                                 cy - bar_h // 2 + max(2, bar_h // 5)],
                                radius=bar_h // 4, fill=(*pal["accent_hi"], 255))
        # value at the bar end
        ni = look.gold_fill(_num(val * ent, prefix, suffix, decimals), num_font, pal,
                            glow_radius=int(h * 0.008), glow_alpha=0.3 * ent)
        frame = look.paste_center(frame, ni, cx=x0 + cur_w + int(w * 0.045), cy=cy,
                                  opacity=ent)
        d = ImageDraw.Draw(frame, "RGBA")
    return frame


def render(out_path, *, bars=None, title: str = "", prefix: str = "",
           suffix: str = "", decimals: int = 0, dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(bars) or [("", 1.0)]
    vmax = max((v for _, v in data), default=1.0) or 1.0
    m = len(data)
    # variant: explicit layout wins; otherwise a lone value auto-uses the hero bar
    lay = (layout or "").strip().lower()
    if lay not in ("columns", "hero_bar", "horizontal_bars"):
        lay = "hero_bar" if m == 1 else "columns"

    title_font = look.font("label", int(h * 0.030))
    num_font = look.font("numeral", int(h * (0.060 if m > 2 else 0.072)))
    hbar_num = look.font("numeral", int(h * 0.046))
    lab_font = look.font("label", int(h * 0.028))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        frame = _draw_title(frame, pal, title, title_font, t,
                            cy=h * (0.16 if lay == "hero_bar" else 0.20))
        if lay == "hero_bar":
            frame = _draw_hero(frame, pal, data, vmax, t, w, h, prefix, suffix,
                               decimals)
        elif lay == "horizontal_bars":
            frame = _draw_hbars(frame, pal, data, vmax, t, w, h, lab_font, hbar_num,
                                prefix, suffix, decimals)
        else:
            frame = _draw_columns(frame, pal, data, vmax, t, w, h, num_font,
                                  lab_font, prefix, suffix, decimals)
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
