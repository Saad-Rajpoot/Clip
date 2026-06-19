"""Primitive: headline_montage.

A press-frenzy beat — three to five period headlines cascade onto the frame as
aged-newsprint chips, each slightly rotated and overlapping, the latest landing
on top. The story has broken; the papers are piling up.

Distinct from `headline_document_reveal` (ONE headline read on a single newsprint
page): this is the MEDIA STORM — several outlets at once, the sense of a scandal
or moment sweeping the press.

Forensic principle (NOT asset copy): a stack of clippings conveys momentum and
consensus that a single headline can't; the premium is the aged paper + serif
type + a restrained scatter + staggered drops, not a busy collage. Pure-local.
No paid API. Deterministic.

    render("x.mp4", headlines=["MONOPOLY BROKEN UP", "COURT RULES AGAINST TRUST",
           "THE OCTOPUS FALLS"], title="THE PRESS ERUPTS · 1911")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "headline_montage", "family": "media",
    "roles": ["press", "headlines", "scandal", "coverage", "reveal"],
    "niches_ok": ["history", "biography", "crime", "geopolitics", "business", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_paper_drops",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["clipping_stack"],
    "review_override": ["headlines", "title", "palette"],
    "fallback": "headline_document_reveal on the first headline if only one given",
}


def _coerce(headlines):
    out = []
    for hd in (headlines or []):
        if isinstance(hd, dict):
            out.append(str(hd.get("text") or hd.get("headline") or "").strip())
        elif isinstance(hd, (list, tuple)) and hd:
            out.append(str(hd[0]).strip())
        elif hd:
            out.append(str(hd).strip())
    return [h for h in out if h][:5]


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
    return lines[:3]


def _clipping(text, w, h, pal, seed):
    """An aged-newsprint clipping sized to its headline."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    cw, ch = int(w * 0.46), int(h * 0.20)
    paper = np.empty((ch, cw, 3), np.float32)
    paper[:] = np.array([222, 214, 196], np.float32)           # aged newsprint
    paper += rng.normal(0, 6.0, (ch, cw, 1))
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    r = np.sqrt(((xx - cw / 2) / (cw * 0.62)) ** 2 + ((yy - ch / 2) / (ch * 0.62)) ** 2)
    paper -= np.clip(r - 0.5, 0, 1)[..., None] * 34            # foxed edges
    chip = Image.fromarray(np.clip(paper, 0, 255).astype("uint8"), "RGB")
    d = ImageDraw.Draw(chip)
    ink = (38, 30, 22)
    d.rectangle([3, 3, cw - 4, ch - 4], outline=(120, 100, 74), width=2)
    # masthead rule + tiny "newspaper" dashes
    d.line([(int(cw * 0.08), int(ch * 0.18)), (int(cw * 0.92), int(ch * 0.18))],
           fill=(110, 92, 66), width=1)
    hf = look.font("title", int(ch * 0.26))
    lines = _wrap(d, text.upper(), hf, cw * 0.86)
    ty = ch * 0.30
    for ln in lines:
        wln = d.textlength(ln, font=hf)
        d.text(((cw - wln) / 2, ty), ln, font=hf, fill=ink)
        ty += ch * 0.30
    # body-text suggestion lines
    by = ty + ch * 0.04
    for _ in range(2):
        if by > ch * 0.9:
            break
        d.line([(int(cw * 0.1), by), (int(cw * 0.9), by)], fill=(150, 130, 102), width=2)
        by += ch * 0.09
    return chip.convert("RGBA")


def render(out_path, *, headlines=None, title: str = "", dur: float = 5.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(headlines)
    if not data:
        data = ["BREAKING"]
    m = len(data)

    rng = np.random.default_rng((seed or 5) & 0xFFFFFFFF)
    chips = [_clipping(t, w, h, pal, (seed or 5) + k) for k, t in enumerate(data)]
    # scatter targets — earlier clippings fan out wider (each stays partly
    # readable); the latest sits centred + upright + on top.
    plac = []
    for k in range(m):
        if k == m - 1:
            cx, cy, rot = w * 0.5, h * 0.45, float(rng.uniform(-2, 2))
        else:
            ang = (k / max(1, m - 1)) * 2 * math.pi + 0.6
            rad = 0.20 + 0.045 * (k % 2)
            cx = w * (0.5 + rad * math.cos(ang))
            cy = h * (0.45 + rad * 0.72 * math.sin(ang))
            rot = float(rng.uniform(-8, 8))
        plac.append((cx, cy, rot))
    title_font = look.font("label", int(h * 0.030))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        # drop each clipping in sequence (oldest first, latest last/on top)
        for k in range(m):
            ka = look.ease_out_cubic(max(0.0, (t - k * 0.5) / 0.45))
            if ka <= 0.01:
                continue
            cx, cy, rot = plac[k]
            chip = chips[k].rotate(rot, expand=True, resample=Image.BICUBIC)
            # drop from slightly above + settle
            dy = (1 - ka) * h * 0.06
            sc = 0.9 + 0.1 * ka
            frame = look.paste_center(frame, chip, cx=cx, cy=cy - dy, scale=sc,
                                      opacity=min(1.0, ka * 1.1))
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(max(0.0, (t - (0.2 + m * 0.5)) / 0.5))
            if ta > 0.01:
                look.hairline(d, int(w * 0.5), int(h * 0.83), int(w * 0.1 * ta), pal)
                ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=5,
                                         glow_alpha=0.0, pad=16)
                frame = look.paste_center(frame, ti, cx=w * 0.5, cy=int(h * 0.88),
                                          opacity=ta)

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fade = look.fade_alpha(t, dur, fps)
        if fade < 1.0:
            frame = look.fade_frame(frame, fade, pal)
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
