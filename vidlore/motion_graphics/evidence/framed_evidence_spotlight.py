"""Primitive: framed_evidence_spotlight.

An exhibit beat — a single artifact (document, photograph, object, note) held in
a gold frame under a warm spotlight on a textured, vignetted bg, with a slow
push-in, an optional EVIDENCE/EXHIBIT tag chip, and a caption that names it.
Forensic principle (NOT asset copy): MagnatesMedia frames a piece of evidence and
lets it breathe (e.g., a framed $100 note under a warm key light @1598s); the
premium is the grade + frame + spotlight + restraint, not a UI card.

If no image resolves, falls back to a framed typographic evidence card so the
beat still lands. Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic.

    render("x.mp4", image_path="note.jpg", caption="The $100,000 cheque · 1872",
           tag="EXHIBIT A", palette_name="amber_gold")
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
    "id": "framed_evidence_spotlight", "family": "evidence",
    "roles": ["evidence", "exhibit", "artifact", "object", "reveal"],
    "niches_ok": ["history", "biography", "crime", "business", "geopolitics", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_key_light",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["center_spotlight"],
    "review_override": ["image_path", "caption", "tag", "palette"],
    "fallback": "framed typographic evidence card if no image is available",
}


def _spotlight(w, h, pal, *, cx_frac=0.5, cy_frac=0.44, t=0.0):
    """A warm radial key light over the graded bg — bright at the artifact,
    crushed at the edges."""
    base = np.asarray(look.graded_background(w, h, pal, seed=int(t * 7))
                      ).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * cx_frac, h * cy_frac
    r = np.sqrt(((xx - cx) / (w * 0.42)) ** 2 + ((yy - cy) / (h * 0.42)) ** 2)
    glow = np.clip(1.0 - r, 0.0, 1.0) ** 1.6
    warm = np.array(pal["glow"], np.float32)
    out = base + glow[..., None] * (warm[None, None] * 0.22)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _fit_contain(img, bw, bh):
    """Fit the whole artifact inside bw×bh (letterbox on a dark mat) — evidence
    must be shown in full, not cropped."""
    iw, ih = img.size
    s = min(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    mat = Image.new("RGB", (bw, bh), (10, 8, 6))
    mat.paste(img.resize((nw, nh), Image.LANCZOS), ((bw - nw) // 2, (bh - nh) // 2))
    return mat


def _exhibit_paper(fw, fh, pal, *, caption="", tag="", seed=0):
    """Premium aged-paper EXHIBIT document — used when no artifact photo is
    available. Warm parchment with foxing, a printed keyline, the caption
    typeset as the document title, faded typed body lines, and an ink seal.
    Reads as a real framed exhibit, never a hollow card. Deterministic."""
    rng = np.random.default_rng(int(seed or 7) & 0xFFFFFFFF)
    paper = np.empty((fh, fw, 3), np.float32)
    paper[:] = np.array([226, 212, 182], np.float32)           # warm aged paper
    paper += rng.normal(0, 6.5, (fh, fw, 1))                   # fiber grain
    # broad tonal mottling (age) — low-freq noise upscaled
    lf = rng.normal(0, 30, (max(2, fh // 36), max(2, fw // 36))).astype(np.float32)
    lf_img = Image.fromarray(np.clip(128 + lf, 0, 255).astype(np.uint8)
                             ).resize((fw, fh), Image.BILINEAR)
    paper += (np.asarray(lf_img, np.float32)[..., None] - 128) * 0.5
    # corner foxing / darkening
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    r = np.sqrt(((xx - fw / 2) / (fw * 0.60)) ** 2 + ((yy - fh / 2) / (fh * 0.60)) ** 2)
    paper -= np.clip(r - 0.5, 0, 1)[..., None] * 46
    card = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(card)
    ink, ink2, faint = (58, 42, 26), (110, 88, 56), (150, 128, 96)
    d.rectangle([int(fw * 0.055), int(fh * 0.055),
                 int(fw * 0.945), int(fh * 0.945)], outline=ink2, width=2)
    # title = caption, serif ink, wrapped, upper area
    tf = look.font("title", int(fh * 0.082))
    words = (caption or "EXHIBIT").split()
    lines, cur = [], ""
    for wd in words:
        tt = (cur + " " + wd).strip()
        if d.textlength(tt, font=tf) <= fw * 0.80 or not cur:
            cur = tt
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    ty = fh * (0.30 - 0.05 * min(3, len(lines)))
    for ln in lines[:3]:
        wln = d.textlength(ln, font=tf)
        d.text(((fw - wln) / 2, ty), ln, font=tf, fill=ink)
        ty += fh * 0.105
    d.line([(fw * 0.30, ty + fh * 0.02), (fw * 0.70, ty + fh * 0.02)],
           fill=ink2, width=2)
    # faded typed body lines (suggest a document body, never gibberish text)
    by, bw = ty + fh * 0.085, fw * 0.66
    for _ in range(5):
        if by > fh * 0.84:
            break
        ll = bw * (0.60 + 0.36 * float(rng.random()))
        lx = (fw - bw) / 2
        d.line([(lx, by), (lx + ll, by)], fill=faint, width=3)
        by += fh * 0.060
    # faint ink seal ring, lower-right
    sx, sy, sr = int(fw * 0.77), int(fh * 0.80), int(fh * 0.085)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], outline=(150, 60, 44), width=4)
    d.ellipse([sx - sr + 7, sy - sr + 7, sx + sr - 7, sy + sr - 7],
              outline=(150, 60, 44), width=2)
    return card


def _frame_tile(media, pal):
    fw, fh = media.size
    bd = 7
    tile = Image.new("RGBA", (fw + bd * 2 + 60, fh + bd * 2 + 60), (0, 0, 0, 0))
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([34, 40, 34 + fw + bd * 2, 40 + fh + bd * 2],
                                 fill=(0, 0, 0, 180))
    tile.alpha_composite(sh.filter(ImageFilter.GaussianBlur(22)))
    fr = Image.new("RGBA", (fw + bd * 2, fh + bd * 2),
                   (*[int(c * 0.85) for c in pal["accent"]], 255))
    fr.paste(media.convert("RGB"), (bd, bd))
    tile.alpha_composite(fr, (28, 28))
    return tile


def render(out_path, *, image_path=None, caption: str = "", tag: str = "",
           dur: float = 6.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    # the artifact (framed) — image if available, else a typographic card
    fw, fh = int(w * 0.40), int(h * 0.52)
    media = None
    if image_path and Path(str(image_path)).exists():
        try:
            im = look.grade_media(Image.open(image_path).convert("RGB"),
                                  pal, strength=0.5)
            media = _fit_contain(im, fw, fh)
        except Exception:                                      # noqa: BLE001
            media = None
    # When no artifact photo resolves, render a premium aged-paper exhibit
    # (the caption becomes the document title) instead of a hollow card — and
    # suppress the duplicate lower-third caption so the beat reads clean.
    _is_textcard = media is None
    if _is_textcard:
        media = _exhibit_paper(fw, fh, pal, caption=caption, tag=tag, seed=seed)
    tile = _frame_tile(media, pal)
    cap_font = look.font("title", int(h * 0.040))
    tag_font = look.font("label", int(h * 0.026))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = _spotlight(w, h, pal, t=t)
        ent = look.ease_out_cubic(min(1.0, t / 0.7))
        z = (0.97 + 0.03 * ent) * (1.0 + 0.02 * pr)            # settle + slow push
        frame = look.paste_center(frame, tile, cx=w * 0.5, cy=h * 0.42,
                                  scale=z, opacity=ent)
        d = ImageDraw.Draw(frame, "RGBA")
        # tag chip (EXHIBIT A / EVIDENCE) above the frame
        if tag:
            ta = look.ease_out_cubic(max(0.0, (t - 0.3) / 0.5))
            if ta > 0.01:
                tw = d.textlength(tag.upper(), font=tag_font)
                bx, by = int(w * 0.5 - tw / 2 - 18), int(h * 0.085)
                d.rounded_rectangle([bx, by, bx + tw + 36, by + int(h * 0.046)],
                                    radius=6, fill=(*pal["bg_b"], int(220 * ta)),
                                    outline=(*pal["accent"], int(230 * ta)), width=2)
                ti = look.text_with_glow(tag.upper(), tag_font, fill=pal["accent_hi"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=10)
                frame = look.paste_center(frame, ti, cx=w * 0.5,
                                          cy=by + int(h * 0.023), opacity=ta)
                d = ImageDraw.Draw(frame, "RGBA")
        # caption lower-third (skipped for the paper exhibit — caption is the
        # document title there, so a second copy would duplicate it)
        if caption and not _is_textcard:
            ca = look.ease_out_cubic(max(0.0, (t - 0.7) / 0.6))
            cy = int(h * 0.84)
            look.hairline(d, int(w * 0.5), cy - int(h * 0.038),
                          int(w * 0.12 * ca), pal)
            ci = look.text_with_glow(caption, cap_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=5,
                                     glow_alpha=0.0, pad=24)
            frame = look.paste_center(frame, ci, cx=w * 0.5, cy=cy, opacity=ca)

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
    look.cleanup_frames(td)                                    # never leak frames
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
