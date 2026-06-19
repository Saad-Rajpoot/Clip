"""Primitive: headline_document_reveal.

A premium aged-paper newspaper / document clipping: masthead/source label, a
serif headline, faux justified body lines (no lorem, no screenshot junk), and an
animated HIGHLIGHTER sweep (or underline) on the key phrase, with a subtle
push-in over a dark graded bed. The document is fully SYNTHESISED (original) —
not a scraped web screenshot.

Forensic evidence: MagnatesMedia 2630s ("...Trade Conspiracy" newspaper, legible
headline + highlight), 1598s (framed archival note). Pure-local. No paid API.

    render(out, headline="HELD GUILTY OF TRADE CONSPIRACY",
           source="THE EVENING WORLD · 1911", highlight="TRADE CONSPIRACY",
           layout="angled", palette_name="parchment_sepia")
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
    "id": "headline_document_reveal", "family": "documents",
    "roles": ["evidence", "proof", "reveal", "quote_source", "legal"],
    "niches_ok": ["crime", "history", "business", "geopolitics", "investigation"],
    "intensity_range": [2, 5], "duration_range": [3.0, 7.0],
    "easing": "easeOutCubic", "audio_cue": "paper_move_soft_impact",
    "repeat_cooldown_s": 40, "per_video_cap": 4, "cost": "med",
    "layout_variants": ["center", "angled"],
    "review_override": ["headline", "source", "highlight", "body", "layout", "palette"],
    "fallback": "static document (no sweep) if frame budget tight",
}


def _aged_paper(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (226, 212, 182)                       # warm cream
    # uneven tone (low-freq blotches)
    lf = rng.normal(0, 1, (h // 24, w // 24, 1)).astype(np.float32)
    lf = np.asarray(Image.fromarray(
        np.clip(lf * 18 + 128, 0, 255).astype(np.uint8)[..., 0]).resize(
        (w, h))).astype(np.float32)[..., None] - 128
    base += lf * np.array([1.0, 0.9, 0.7])
    # fibre grain
    base += rng.normal(0, 5, (h, w, 1)).astype(np.float32)
    # a few faint stains
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    st = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(st)
    for _ in range(int(rng.integers(3, 6))):
        cxp, cyp = int(rng.integers(0, w)), int(rng.integers(0, h))
        rr = int(rng.integers(40, 130))
        sd.ellipse([cxp - rr, cyp - rr, cxp + rr, cyp + rr],
                   fill=(120, 96, 54, int(rng.integers(10, 26))))
    st = st.filter(ImageFilter.GaussianBlur(30))
    img = Image.alpha_composite(img.convert("RGBA"), st).convert("RGB")
    # edge darkening (aging)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx-w/2)/(w/2))**2 + ((yy-h/2)/(h/2))**2)
    dk = np.clip(1 - 0.32*np.clip(r-0.7, 0, 1)/0.3, 0, 1)
    a = np.asarray(img).astype(np.float32) * dk[..., None]
    return Image.fromarray(a.astype(np.uint8), "RGB")


def _wrap(draw, text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def render(out_path, *, headline: str, source: str = "", highlight: str = "",
           body: str = "", layout: str = "angled", dur: float = 4.8, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    ink = (38, 30, 22)
    pw, ph = int(w * 0.60), int(h * 0.72)
    paper = _aged_paper(pw, ph, seed)
    d = ImageDraw.Draw(paper)
    pad = int(pw * 0.08)
    src_f = look.font("label", int(ph * 0.040))
    hd_f = look.font("title", int(ph * 0.092))
    bd_f = look.font("label", int(ph * 0.036))
    y = pad
    if source:
        d.text((pad, y), source.upper(), font=src_f, fill=(110, 86, 50))
        y += int(ph * 0.058)
        d.line([(pad, y), (pw - pad, y)], fill=(90, 70, 44), width=3)
        y += int(ph * 0.045)
    hl_lines = _wrap(d, headline.upper(), hd_f, pw - pad * 2)
    head_box = []                                   # (x,y,x2,y2) of each line
    for ln in hl_lines[:3]:
        d.text((pad, y), ln, font=hd_f, fill=ink)
        wln = d.textlength(ln, font=hd_f)
        head_box.append((pad, y, pad + wln, y + ph * 0.10))
        y += int(ph * 0.105)
    y += int(ph * 0.02)
    # body: real excerpt if given (wrapped), else faux justified bars
    if body:
        for ln in _wrap(d, body, bd_f, pw - pad * 2)[:6]:
            d.text((pad, y), ln, font=bd_f, fill=(70, 58, 44))
            y += int(ph * 0.05)
    else:
        rng = np.random.default_rng(seed + 1)
        for _ in range(6):
            if y > ph - pad:
                break
            ww = int((pw - pad * 2) * rng.uniform(0.72, 0.99))
            d.rectangle([pad, y, pad + ww, y + int(ph * 0.013)], fill=(96, 82, 64))
            y += int(ph * 0.043)
    # highlight target box = the headline line containing the phrase (or 1st line)
    htarget = head_box[0] if head_box else (pad, pad, pw - pad, pad + 40)
    if highlight and hl_lines:
        for i, ln in enumerate(hl_lines[:len(head_box)]):
            if highlight.upper() in ln:
                htarget = head_box[i]
                break

    # depth bg
    bg = look.graded_background(w, h, pal, seed=seed)
    ang = -3.0 if layout == "angled" else 0.0
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = bg.copy()
        # paper push-in + settle + enter fade
        ent = look.ease_out_cubic(min(1.0, t / 0.55))
        z = (0.97 + 0.03 * ent) * (1.0 + 0.035 * look.ease_in_out_cubic(pr))
        tile = paper
        # highlighter sweep on a copy (so it composites under nothing)
        sweep = look.ease_out_cubic(max(0.0, (t - 0.7) / 0.9))
        if sweep > 0:
            hl = paper.copy()
            hd = ImageDraw.Draw(hl, "RGBA")
            x0, yb, x1, y1 = htarget
            cur_x1 = x0 + (x1 - x0) * sweep
            hd.rectangle([x0 - 8, yb + (y1 - yb) * 0.05, cur_x1 + 8, y1 + (y1 - yb) * 0.08],
                         fill=(*pal["glow"], 96))
            tile = Image.blend(paper, hl, 0.9)
        # rotate + shadow + place
        rt = tile.rotate(ang, expand=True, resample=Image.BICUBIC,
                         fillcolor=(0, 0, 0))
        # build an RGBA tile w/ shadow
        rgba = Image.new("RGBA", (rt.width + 80, rt.height + 80), (0, 0, 0, 0))
        sh = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([46, 50, 46 + rt.width, 50 + rt.height],
                                     fill=(0, 0, 0, 150))
        sh = sh.filter(ImageFilter.GaussianBlur(22))
        rgba.alpha_composite(sh)
        rgba.alpha_composite(rt.convert("RGBA"), (34, 30))
        frame = look.paste_center(frame, rgba, cx=w/2, cy=h/2,
                                  scale=z * (w * 0.62 / rgba.width), opacity=ent)
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
    import shutil as _sh
    _sh.rmtree(td, ignore_errors=True)        # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "layout": layout,
            "err": (r.stderr[-200:] if not ok else "")}
