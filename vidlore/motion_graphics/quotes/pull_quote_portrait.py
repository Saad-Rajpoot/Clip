"""Primitive: pull_quote_portrait.

A sourced quotation given a human voice + a face. A large refined-serif quote
with an oversized gold quotation mark, an attribution hairline + name (condensed),
and — when a real portrait is available — a small face-safe FRAMED portrait to one
side. Over a graded, vignetted, grained background with a slow parallax drift.

Pairs naturally with the V1.1 real-portrait pipeline (face_safe_crop keeps the
whole head). Forensic principle (NOT asset copy): MagnatesMedia sets a figure's
words beside their photo; the premium is the grade + serif type + the framed
subject, not a template card. Falls back to a centred quote-only card if no
portrait resolves.

Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", quote="The day of combination is here to stay.",
           name="John D. Rockefeller", sub="1899", portrait_path="jdr.jpg")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "pull_quote_portrait", "family": "quotes",
    "roles": ["quote", "quote_subject", "reveal", "statement"],
    "niches_ok": ["history", "biography", "business", "crime", "geopolitics", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 8.0],
    "easing": "easeOutCubic", "audio_cue": "silence_then_soft_bed",
    "repeat_cooldown_s": 70, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["portrait_left", "portrait_right", "quote_only"],
    "review_override": ["quote", "name", "sub", "side", "portrait_path", "palette"],
    "fallback": "centred quote-only card if no portrait is available",
}


def _wrap(draw, text, font, max_w, max_lines=5):
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


def _frame_portrait(img, fw, fh, pal):
    crop = look.grade_media(look.face_safe_crop(img, fw, fh).convert("RGB"),
                            pal, strength=0.45)
    bd = 5
    tile = Image.new("RGBA", (fw + bd * 2 + 44, fh + bd * 2 + 44), (0, 0, 0, 0))
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([26, 30, 26 + fw + bd * 2, 30 + fh + bd * 2],
                                 fill=(0, 0, 0, 160))
    tile.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
    fr = Image.new("RGBA", (fw + bd * 2, fh + bd * 2),
                   (*[int(c * 0.82) for c in pal["accent"]], 255))
    fr.paste(crop, (bd, bd))
    tile.alpha_composite(fr, (22, 22))
    return tile


def render(out_path, *, quote: str = "", name: str = "", sub: str = "",
           portrait_path=None, side: str = "left", dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    quote = (quote or "").strip().strip('"').strip("'").strip()
    if not quote:
        quote = name or "…"

    # portrait tile (optional)
    tile = None
    has_port = False
    if portrait_path and Path(str(portrait_path)).exists() and layout != "quote_only":
        try:
            fw, fh = int(h * 0.40 * 0.80), int(h * 0.40)
            tile = _frame_portrait(Image.open(portrait_path).convert("RGB"),
                                   fw, fh, pal)
            has_port = True
        except Exception:                                      # noqa: BLE001
            tile = None
    if layout == "portrait_right":
        side = "right"
    elif layout == "portrait_left":
        side = "left"

    # geometry: portrait on `side`, quote on the other; or centred quote-only
    if has_port:
        pcx = w * (0.26 if side == "left" else 0.74)
        qcx = w * (0.60 if side == "left" else 0.40)
        qmax = int(w * 0.52)
    else:
        pcx = 0
        qcx = w * 0.5
        qmax = int(w * 0.74)

    # size the quote to fit (shrink if long)
    qsize = int(h * 0.058)
    probe = Image.new("RGBA", (8, 8))
    pd = ImageDraw.Draw(probe)
    for _ in range(6):
        qf = look.font("title", qsize)
        lines = _wrap(pd, quote, qf, qmax)
        if len(lines) <= 4:
            break
        qsize = int(qsize * 0.9)
    qf = look.font("title", qsize)
    lines = _wrap(pd, quote, qf, qmax)
    nm_font = look.font("label", int(h * 0.032))
    sb_font = look.font("label", int(h * 0.026))
    qm_font = look.font("title", int(h * 0.22))
    line_h = int(qsize * 1.30)
    block_h = line_h * len(lines)
    qtop = h * 0.5 - block_h / 2 - h * 0.02

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)

        # portrait enters
        if tile is not None:
            ent = look.ease_out_cubic(min(1.0, t / 0.6))
            frame = look.paste_center(frame, tile, cx=pcx, cy=h * 0.50,
                                      scale=0.96 + 0.04 * ent, opacity=ent)

        # oversized opening quote mark (behind the text, soft)
        qa = look.ease_out_cubic(min(1.0, t / 0.5))
        qmk = look.text_with_glow("“", qm_font, fill=pal["accent"],
                                  glow=pal["glow"], glow_radius=10,
                                  glow_alpha=0.25, pad=20)
        frame = look.paste_center(frame, qmk,
                                  cx=qcx - qmax * 0.46, cy=qtop - h * 0.01,
                                  opacity=0.5 * qa)

        # quote lines fade in sequence
        for li, ln in enumerate(lines):
            la = look.ease_out_cubic(max(0.0, (t - (0.35 + li * 0.28)) / 0.5))
            if la <= 0.01:
                continue
            limg = look.text_with_glow(ln, qf, fill=pal["text"], glow=pal["bg_b"],
                                       glow_radius=6, glow_alpha=0.0, pad=20)
            frame = look.paste_center(frame, limg, cx=qcx,
                                      cy=qtop + li * line_h + line_h / 2,
                                      opacity=la)

        # attribution: hairline + name + sub, after the quote
        if name:
            aa = look.ease_out_cubic(max(0.0, (t - (0.4 + len(lines) * 0.28)) / 0.6))
            if aa > 0.01:
                ay = qtop + block_h + h * 0.055
                dctx = ImageDraw.Draw(frame, "RGBA")
                hx = int(qcx - w * 0.16)
                dctx.line([(hx, int(ay)), (int(hx + w * 0.05 * aa), int(ay))],
                          fill=(*pal["accent"], int(230 * aa)), width=2)
                nimg = look.text_with_glow("  " + name.upper(), nm_font,
                                           fill=pal["accent_hi"], glow=pal["bg_b"],
                                           glow_radius=4, glow_alpha=0.0, pad=14)
                frame = look.paste_center(
                    frame, nimg, cx=hx + w * 0.05 + nimg.width / 2 - 14,
                    cy=ay, opacity=aa)
                if sub:
                    simg = look.text_with_glow(sub.upper(), sb_font,
                                               fill=pal["muted"], glow=pal["bg_b"],
                                               glow_radius=3, glow_alpha=0.0, pad=12)
                    frame = look.paste_center(frame, simg, cx=qcx,
                                              cy=ay + h * 0.045, opacity=aa)

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
    import shutil as _sh
    _sh.rmtree(td, ignore_errors=True)        # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
