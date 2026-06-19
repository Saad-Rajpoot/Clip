"""Primitive: quote_stream.

A chorus beat — two to four short quotations cascade in and stack, each with its
attribution, building the sense of many voices saying the same thing: a pile-on
of critics, a wave of praise, a crowd of witnesses.

Distinct from `pull_quote_portrait` (ONE quotation beside ONE face) and
`statement_card` (a single authored claim): this is MULTIPLICITY — the weight is
in the accumulation of voices, not any single one.

Forensic principle (NOT asset copy): MagnatesMedia stacks reactions to show
consensus or condemnation; the premium is the graded field + gold quote marks +
serif lines + staggered rise + muted attributions, not a cluttered wall. Pure-
local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", quotes=[["A menace to free trade.", "The Times"],
           ["The octopus of its age.", "Harper's"], ["Greed made monstrous.",
           "The Tribune"]], title="THE VERDICT OF THE PRESS")
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
    "id": "quote_stream", "family": "quotes",
    "roles": ["quotes", "chorus", "reactions", "voices", "verdict"],
    "niches_ok": ["history", "biography", "crime", "geopolitics", "business", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_voice_layer",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["stacked_quotes"],
    "review_override": ["quotes", "title", "palette"],
    "fallback": "pull_quote_portrait if only one quotation is available",
}


def _coerce(quotes):
    out = []
    for q in (quotes or []):
        txt, by = "", ""
        if isinstance(q, dict):
            txt = str(q.get("text") or q.get("quote") or "")
            by = str(q.get("by") or q.get("attribution") or q.get("name") or "")
        elif isinstance(q, (list, tuple)) and q:
            txt = str(q[0])
            by = str(q[1]) if len(q) > 1 else ""
        elif q:
            txt = str(q)
        txt = txt.strip().strip('"“”')
        if txt:
            out.append((txt, by.strip()))
    return out[:4]


def _wrap(d, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines[:2]


def render(out_path, *, quotes=None, title: str = "", dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(quotes)
    if not data:
        data = [("…", "")]
    m = len(data)

    title_font = look.font("label", int(h * 0.030))
    q_font = look.font("title", int(h * (0.044 if m > 3 else 0.052)))
    by_font = look.font("caption", int(h * 0.026))
    mark_font = look.font("title", int(h * 0.11))

    top, bot = int(h * 0.26), int(h * 0.88)
    slot = (bot - top) / m

    td = Path(tempfile.mkdtemp())
    _dd = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = [(_wrap(_dd, t, q_font, w * 0.66), by) for t, by in data]

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
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.15,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        for k, (lines, by) in enumerate(wrapped):
            ent = look.ease_out_cubic(max(0.0, (t - (0.3 + k * 0.45)) / 0.55))
            if ent <= 0.01:
                continue
            cyk = int(top + (k + 0.5) * slot)
            dy = int((1 - ent) * h * 0.03)                     # rise in
            # opening quote mark (gold), left of the block
            mi = look.gold_fill("“", mark_font, pal, glow_radius=int(h * 0.01),
                                glow_alpha=0.3 * ent)
            frame = look.paste_center(frame, mi, cx=w * 0.20,
                                      cy=cyk - int(h * 0.012) - dy, opacity=ent)
            d = ImageDraw.Draw(frame, "RGBA")
            # quote lines (serif), centred
            ly = cyk - dy - int(len(lines) * h * 0.030 / 2)
            for ln in lines:
                qi = look.text_with_glow(ln, q_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=12)
                frame = look.paste_center(frame, qi, cx=w * 0.52, cy=ly, opacity=ent)
                d = ImageDraw.Draw(frame, "RGBA")
                ly += int(h * 0.052)
            # attribution (muted, right side under the quote)
            if by:
                bi = look.text_with_glow("— " + by.upper(), by_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=2,
                                         glow_alpha=0.0, pad=8)
                frame.paste(bi, (int(w * 0.78) - bi.width, ly - int(h * 0.006)), bi)
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
