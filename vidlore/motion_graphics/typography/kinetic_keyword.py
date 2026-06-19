"""Primitive: kinetic_keyword.

ONE concept word emphasised with restraint — a serif keyword revealed by a soft
mask-up + settle, with up to three faint context words that fade-stagger behind
it, over a graded/textured bed. Deliberately quiet: no TikTok word-slam, no
per-word spam. For MAJOR reveals only (strict cap + long cooldown in the SPEC).

Forensic evidence: MagnatesMedia 366s ("crisis" + faint context words over a
money texture). Pure-local (PIL + numpy → ffmpeg). No paid API.

    render(out, keyword="MONOPOLY", context=["control","leverage","power"],
           palette_name="ember_red")
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
    "id": "kinetic_keyword", "family": "typography",
    "roles": ["reveal", "thesis", "turn", "emphasis"],
    "niches_ok": ["all"], "intensity_range": [3, 5], "duration_range": [1.6, 3.2],
    "easing": "easeOutExpo", "audio_cue": "soft_impact_on_word",
    "repeat_cooldown_s": 70, "per_video_cap": 2, "cost": "low",
    "review_override": ["keyword", "context", "palette"],
    "fallback": "skip (footage holds) if cap/cooldown exceeded",
    "restraint": "MAJOR reveals only — strict cap 2, cooldown 70s",
}


def render(out_path, *, keyword: str, context=None, bg_image=None,
           dur: float = 2.4, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    context = [c for c in (context or [])][:3]
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    kw_font = look.font("title", int(h * 0.135))
    cx_font = look.font("label", int(h * 0.044))
    media = None
    if bg_image and Path(bg_image).exists():
        try:
            media = look.grade_media(Image.open(bg_image).convert("RGB").resize((w, h)), pal, 0.62)
        except Exception:                                      # noqa: BLE001
            media = None
    # pre-render the keyword layer once (gold, with glow)
    kw_layer = look.text_with_glow(keyword.upper(), kw_font, fill=pal["text"],
                                   glow=pal["glow"], glow_radius=int(h * 0.02),
                                   glow_alpha=0.5, pad=70)
    # context word positions (seeded, around the keyword, kept faint + sparse)
    rng = np.random.default_rng(seed)
    cpos = []
    for k, word in enumerate(context):
        ang = rng.uniform(0, 6.28)
        rad = h * (0.20 + 0.06 * k)
        cpos.append((word, w / 2 + np.cos(ang) * rad * 1.4, h / 2 + np.sin(ang) * rad))
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps; pr = i / max(1, n - 1)
        drift = pr
        if media is not None:
            z = 1.0 + 0.04 * drift
            bw, bh = int(w * z), int(h * z)
            frame = media.resize((bw, bh), Image.LANCZOS).crop(
                ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        else:
            frame = look.graded_background(w, h, pal, seed=seed, drift=drift)
        # faint context words fade in first, stay subdued
        for k, (word, x, y) in enumerate(cpos):
            ca = look.ease_out_cubic(max(0.0, (t - 0.1 - 0.12 * k) / 0.5)) * 0.34
            if ca > 0.01:
                cl = look.text_with_glow(word.upper(), cx_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=18)
                frame = look.paste_center(frame, cl, cx=x, cy=y, opacity=ca)
        # keyword: soft mask-up reveal + settle (no slam)
        rev = look.ease_out_expo(min(1.0, t / (dur * 0.45)))
        kl = kw_layer
        if rev < 1.0:                                    # clip from the bottom up
            kh = kl.height
            cut = int(kh * (1 - rev))
            arr = np.asarray(kl).copy()
            arr[:cut, :, 3] = 0
            # soft edge on the wipe
            band = max(2, int(kh * 0.06))
            for b in range(band):
                row = cut + b
                if 0 <= row < kh:
                    arr[row, :, 3] = (arr[row, :, 3].astype(np.float32) * (b / band)).astype(np.uint8)
            kl = Image.fromarray(arr, "RGBA")
        scale = 1.0 + 0.05 * (1 - rev)
        frame = look.paste_center(frame, kl, cx=w / 2, cy=h / 2, scale=scale,
                                  opacity=look.ease_out_cubic(min(1.0, t / 0.3)))
        frame = look.vignette(frame, strength=0.62)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
        fa = look.fade_alpha(t, dur, fps, fin=0.18, fout=0.4)
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
            "render_s": round(time.time() - t0, 2),
            "err": (r.stderr[-200:] if not ok else "")}
