"""Primitive: ranked_list_countdown.

A leaderboard beat — a Top-N ranked list where rows drop in from the bottom rank
UP to number one, each with a rank numeral, a label, a proportional bar and a
value; the #1 row lands last and is crowned in gold. The classic "number five…
number four… and the number-one…" reveal.

Distinct from `statistic_bar_reveal` (unranked vertical columns) and
`composition_stack` (parts of ONE whole): this is an ORDERED ranking with rank
numerals and a climactic top slot.

Forensic principle (NOT asset copy): MagnatesMedia builds a ranking one slot at a
time so the viewer anticipates the top; the premium is the grade + serif rank
numerals + refined bars + the gold crowning of #1, not a flat table. Pure-local
(PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", items=[["Standard Oil", 90], ["Carnegie Steel", 60],
           ["Vanderbilt Rail", 45]], title="THE GILDED-AGE GIANTS", prefix="$",
           suffix="M")
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
    "id": "ranked_list_countdown", "family": "charts",
    "roles": ["ranking", "leaderboard", "countdown", "top_list", "data"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_rank_tick",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["leaderboard"],
    "review_override": ["items", "title", "prefix", "suffix", "palette"],
    "fallback": "statistic_bar_reveal if fewer than 2 ranked items resolve",
}


def _coerce(items):
    out = []
    for it in (items or []):
        if isinstance(it, dict):
            lab, val = str(it.get("label") or it.get("name") or ""), it.get("value", 0)
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            # allow [label, value] or [rank, label, value]
            if len(it) >= 3 and not _isnum(it[1]):
                lab, val = str(it[1]), it[2]
            else:
                lab, val = str(it[0]), it[1]
        else:
            continue
        try:
            out.append((lab.strip(), float(val)))
        except (TypeError, ValueError):
            continue
    # rank descending by value (a leaderboard); keep at most 6 slots
    out.sort(key=lambda r: -r[1])
    return out[:6]


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _num(v, prefix, suffix, decimals):
    body = f"{v:,.{decimals}f}" if decimals > 0 else f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def render(out_path, *, items=None, title: str = "", prefix: str = "",
           suffix: str = "", decimals: int = 0, dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(items)
    if not data:
        data = [("", 1.0)]
    vmax = max((v for _, v in data), default=1.0) or 1.0
    m = len(data)

    title_font = look.font("label", int(h * 0.030))
    rank_font = look.font("numeral", int(h * 0.052))
    rank_font_top = look.font("numeral", int(h * 0.072))
    lab_font = look.font("label", int(h * 0.030))
    val_font = look.font("numeral", int(h * 0.034))

    top_y, bot_y = int(h * 0.30), int(h * 0.84)
    rowh = (bot_y - top_y) / m
    rank_cx = int(w * 0.20)
    lab_x = int(w * 0.27)
    bar_x0 = int(w * 0.50)
    bar_max = int(w * 0.30)
    bar_h = int(min(rowh * 0.34, h * 0.05))

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
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.19,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        for k, (lab, val) in enumerate(data):
            # reveal bottom rank first, #1 (k==0) last → countdown climax
            ent = look.ease_out_cubic(max(0.0, (t - (0.35 + (m - 1 - k) * 0.28)) / 0.6))
            if ent <= 0.01:
                continue
            cy = int(top_y + (k + 0.5) * rowh)
            is_top = (k == 0)
            slide = (1 - ent) * w * 0.05               # slide in from the right
            # faint row separator
            d.line([(int(w * 0.16), int(cy + rowh * 0.46)),
                    (int(w * 0.84), int(cy + rowh * 0.46))],
                   fill=(*pal["muted"], int(60 * ent)), width=1)
            # rank numeral (#1 crowned gold + larger)
            rnum = f"{k + 1}"
            if is_top:
                ri = look.gold_fill(rnum, rank_font_top, pal,
                                    glow_radius=int(h * 0.014), glow_alpha=0.5 * ent)
            else:
                ri = look.text_with_glow(rnum, rank_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=10)
            frame = look.paste_center(frame, ri, cx=rank_cx + int(slide), cy=cy,
                                      opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
            # label (left-aligned, vertically centred)
            if lab:
                li = look.text_with_glow(
                    lab.upper(), lab_font,
                    fill=(pal["text"] if is_top else pal["muted"]),
                    glow=pal["bg_b"], glow_radius=3, glow_alpha=0.0, pad=8)
                ly = cy - li.height // 2
                frame.paste(li, (lab_x + int(slide), ly), li)
                d = ImageDraw.Draw(frame, "RGBA")
            # proportional bar
            full = int(bar_max * (val / vmax))
            bw = max(2, int(full * ent))
            col = pal["accent_hi"] if is_top else pal["accent"]
            by0, by1 = cy - bar_h // 2, cy + bar_h // 2
            d.rounded_rectangle([bar_x0, by0, bar_x0 + bw, by1], radius=bar_h // 2,
                                fill=(*col, 255))
            # value at the bar's leading edge, counting up
            cur = val * ent
            if is_top:
                vi = look.gold_fill(_num(cur, prefix, suffix, decimals), val_font,
                                    pal, glow_radius=int(h * 0.010),
                                    glow_alpha=0.4 * ent)
            else:
                vi = look.text_with_glow(_num(cur, prefix, suffix, decimals),
                                         val_font, fill=pal["text"], glow=pal["bg_b"],
                                         glow_radius=3, glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, vi,
                                      cx=bar_x0 + bw + int(w * 0.035),
                                      cy=cy, opacity=ent)
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
