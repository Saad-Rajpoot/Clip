"""Primitive: definition_card.

A define-the-term beat — a single word or phrase set like a dictionary entry: the
TERM large in gold serif, an italic part-of-speech tag, a thin rule, then the
definition in restrained serif beneath, fading up line by line.

Distinct from `kinetic_keyword` (a single stabbed word with no meaning given),
`statement_card` (an editorial claim, not a defined term) and
`pull_quote_portrait` (an attributed quotation): this is "here is what this word
actually means" — the explainer's anchor.

Forensic principle (NOT asset copy): a clean dictionary-style term + definition
reads as authoritative and lets a key concept land; the premium is the serif
hierarchy + one gold accent + a hairline + restraint, not a busy lower-third.
Pure-local. No paid API. Deterministic.

    render("x.mp4", term="Monopoly", pos="noun",
           definition="exclusive control of a commodity or service in a market.")
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
    "id": "definition_card", "family": "statements",
    "roles": ["definition", "term", "explainer", "concept", "reveal"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_term_chime",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["dictionary"],
    "review_override": ["term", "definition", "pos", "palette"],
    "fallback": "kinetic_keyword on the term if no definition is available",
}


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
    return lines


def render(out_path, *, term: str = "", definition: str = "", pos: str = "",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    term = (term or "").strip() or "—"
    definition = (definition or "").strip()

    term_font = look.font("title", int(h * (0.105 if len(term) < 16 else 0.078)))
    pos_font = look.font("caption", int(h * 0.030))
    def_font = look.font("title", int(h * 0.040))
    tag_font = look.font("label", int(h * 0.022))

    _probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    def_lines = _wrap(_probe, definition, def_font, w * 0.62)[:4]
    term_cy = int(h * 0.40)
    rule_y = int(h * 0.50)
    def_y0 = int(h * 0.56)
    dlh = int(h * 0.058)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        # small DEFINITION tag chip at top
        ta = look.ease_out_cubic(min(1.0, t / 0.45))
        if ta > 0.01:
            tag = "DEFINITION"
            tw = d.textlength(tag, font=tag_font)
            bx = int(w * 0.5 - tw / 2 - 16)
            by = int(h * 0.245)
            d.rounded_rectangle([bx, by, bx + tw + 32, by + int(h * 0.040)],
                                radius=6, fill=(*pal["bg_b"], int(210 * ta)),
                                outline=(*pal["accent"], int(220 * ta)), width=2)
            ci = look.text_with_glow(tag, tag_font, fill=pal["accent_hi"],
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ci, cx=w * 0.5,
                                      cy=by + int(h * 0.020), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # TERM (gold serif) + part-of-speech italic
        ea = look.ease_out_cubic(max(0.0, (t - 0.25) / 0.5))
        if ea > 0.01:
            ti = look.gold_fill(term, term_font, pal, glow_radius=int(h * 0.014),
                                glow_alpha=0.4 * ea)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=term_cy, opacity=ea)
            d = ImageDraw.Draw(frame, "RGBA")
            if pos:
                pi = look.text_with_glow(pos.lower(), pos_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=8)
                frame = look.paste_center(frame, pi, cx=w * 0.5,
                                          cy=term_cy + int(h * 0.072), opacity=ea)
                d = ImageDraw.Draw(frame, "RGBA")
        # hairline rule
        ra = look.ease_out_cubic(max(0.0, (t - 0.5) / 0.4))
        if ra > 0.01:
            look.hairline(d, int(w * 0.5), rule_y, int(w * 0.10 * ra), pal)
        # definition lines, staggered fade-up
        for li, ln in enumerate(def_lines):
            la = look.ease_out_cubic(max(0.0, (t - (0.6 + li * 0.22)) / 0.5))
            if la <= 0.01:
                continue
            di = look.text_with_glow(ln, def_font, fill=pal["text"], glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=18)
            frame = look.paste_center(frame, di, cx=w * 0.5,
                                      cy=def_y0 + li * dlh + dlh // 2, opacity=la)
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
