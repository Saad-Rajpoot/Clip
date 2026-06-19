"""Primitive: statement_card.

A thesis beat — one bold editorial CLAIM held full-screen in large serif, the
lines rising in one after another, a key phrase underscored in gold, an optional
small source tag beneath. The "here is the point" card.

Distinct from `pull_quote_portrait` (an attributed QUOTATION beside a face),
`kinetic_keyword` (a single stabbed WORD) and `headline_document_reveal` (a
newspaper headline on newsprint): this is the narrator's own unattributed claim,
typeset to land — "He didn't break the law. He rewrote it."

Forensic principle (NOT asset copy): MagnatesMedia drops a clean typographic
thesis on a graded ground and lets it breathe, one gold accent carrying the
emphasis; the premium is the serif + restraint + a single gold underscore, not a
busy lower-third. Pure-local. No paid API. Deterministic.

    render("x.mp4", text="He didn't break the law — he rewrote it.",
           emphasis="rewrote it", source="The Gilded Age")
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
    "id": "statement_card", "family": "statements",
    "roles": ["statement", "thesis", "claim", "takeaway", "reveal"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 5], "duration_range": [3.5, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_thesis_swell",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["center_serif"],
    "review_override": ["text", "emphasis", "source", "palette"],
    "fallback": "kinetic_keyword on the first strong word if no statement text",
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


def render(out_path, *, text: str = "", emphasis: str = "", source: str = "",
           dur: float = 5.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    text = (text or "").strip() or "…"

    # size the serif to the statement length (shorter → bigger)
    base = 0.092 if len(text) < 48 else (0.072 if len(text) < 90 else 0.058)
    fnt = look.font("title", int(h * base))
    src_font = look.font("label", int(h * 0.026))

    _probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lines = _wrap(_probe, text, fnt, w * 0.78)[:5]
    lh = int(h * (base + 0.028))
    total_h = lh * len(lines)
    y0 = int(h * 0.5 - total_h / 2)

    # locate the emphasis phrase within a line (for a gold underscore)
    emph = (emphasis or "").strip()
    emph_line, emph_x0, emph_x1 = -1, 0, 0
    if emph:
        for li, ln in enumerate(lines):
            pos = ln.lower().find(emph.lower())
            if pos >= 0:
                pre, mid = ln[:pos], ln[pos:pos + len(emph)]
                emph_line = li
                wln = _probe.textlength(ln, font=fnt)
                lx = w * 0.5 - wln / 2
                emph_x0 = lx + _probe.textlength(pre, font=fnt)
                emph_x1 = emph_x0 + _probe.textlength(mid, font=fnt)
                break

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        # top + bottom gold hairlines framing the block (draw in)
        d = ImageDraw.Draw(frame, "RGBA")
        ha = look.ease_out_cubic(min(1.0, t / 0.5))
        look.hairline(d, int(w * 0.5), y0 - int(h * 0.06), int(w * 0.07 * ha), pal)
        # lines rise + fade in, staggered
        for li, ln in enumerate(lines):
            la = look.ease_out_cubic(max(0.0, (t - (0.25 + li * 0.28)) / 0.55))
            if la <= 0.01:
                continue
            yy = y0 + li * lh + int((1 - la) * h * 0.03)         # slight rise
            ti = look.text_with_glow(ln, fnt, fill=pal["text"], glow=pal["bg_b"],
                                     glow_radius=6, glow_alpha=0.0, pad=20)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=yy + lh // 2,
                                      opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")
            # gold underscore under the emphasis phrase, after its line lands
            if li == emph_line and la > 0.6:
                ua = look.ease_out_cubic((la - 0.6) / 0.4)
                uy = yy + lh - int(h * 0.012)
                ux1 = int(emph_x0 + (emph_x1 - emph_x0) * ua)
                d.line([(int(emph_x0), uy), (ux1, uy)],
                       fill=(*pal["accent_hi"], 255), width=4)
        # source tag beneath the block
        if source:
            sa = look.ease_out_cubic(max(0.0, (t - (0.3 + len(lines) * 0.28)) / 0.5))
            if sa > 0.01:
                si = look.text_with_glow(source.upper(), src_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=10)
                frame = look.paste_center(frame, si, cx=w * 0.5,
                                          cy=y0 + total_h + int(h * 0.07),
                                          opacity=sa)

        frame = look.vignette(frame, strength=0.62)
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
