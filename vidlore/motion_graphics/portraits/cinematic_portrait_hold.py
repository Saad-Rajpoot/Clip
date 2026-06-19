"""Primitive: cinematic_portrait_hold.

A framed archival/real portrait, cinematically composed over a graded depth
background (a blurred, darkened copy of the portrait), with a slow push-in,
vignette, grain, and an optional lower-third (name + role/dates) that reveals
after the portrait settles. The premium version of a flat photo card.

Forensic evidence: MagnatesMedia 2332s (Rockefeller, library, warm, vignette),
135.7s (framed portrait + name). Pure-local (PIL + numpy → ffmpeg). No paid API.

    render("rockefeller.jpg", out, name="John D. Rockefeller",
           sub="1839 – 1937 · Standard Oil", palette_name="parchment_sepia")
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
    "id": "cinematic_portrait_hold", "family": "portraits",
    "roles": ["person", "intro", "reveal", "quote_subject"],
    "niches_ok": ["history", "biography", "crime", "business", "geopolitics"],
    "intensity_range": [2, 4], "duration_range": [3.0, 7.0],
    "easing": "easeInOutCubic", "audio_cue": "bed_soft_swell",
    "repeat_cooldown_s": 25, "per_video_cap": 8, "cost": "low",
    "review_override": ["name", "sub", "side", "frame", "palette", "motion"],
    "fallback": "static framed portrait if frame budget is tight",
}


def _frame_portrait(img: Image.Image, fw: int, fh: int, pal: dict) -> Image.Image:
    """Crop to fw:fh, add inset border + drop shadow → RGBA tile."""
    # FACE-AWARE crop — keep the WHOLE head (top margin + shoulders), never clip
    # the forehead. Grayscale-tolerant; falls back to a top-biased crop if no
    # face is found. Replaces the old centre cover-crop that cut foreheads.
    crop = look.face_safe_crop(img, fw, fh).convert("RGB")
    crop = look.grade_media(crop, pal, strength=0.45)
    bd = 6
    tile = Image.new("RGBA", (fw + bd * 2 + 48, fh + bd * 2 + 48), (0, 0, 0, 0))
    # drop shadow
    sh = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle(
        [28, 30, 28 + fw + bd * 2, 30 + fh + bd * 2], fill=(0, 0, 0, 170))
    sh = sh.filter(ImageFilter.GaussianBlur(18))
    tile.alpha_composite(sh)
    # border + portrait
    frame = Image.new("RGBA", (fw + bd * 2, fh + bd * 2),
                      (*[int(c * 0.8) for c in pal["accent"]], 255))
    frame.paste(crop, (bd, bd))
    tile.alpha_composite(frame, (24, 24))
    return tile


def render(out_path, *, portrait_path=None, name: str = "", sub: str = "",
           dur: float = 5.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", side: str = "left",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    try:
        src = Image.open(portrait_path).convert("RGB")
    except Exception as e:                                      # noqa: BLE001
        return {"ok": False, "err": f"portrait load: {e}", "path": str(out_path)}
    fw, fh = int(h * 0.52 * 0.78), int(h * 0.52)               # ~3:4 frame
    tile = _frame_portrait(src, fw, fh, pal)
    # depth bg = blurred, darkened, graded portrait fills the frame
    bg_src = look.grade_media(src.resize((w, h), Image.LANCZOS), pal, strength=0.7)
    bg_src = bg_src.filter(ImageFilter.GaussianBlur(34))
    bg_arr = (np.asarray(bg_src).astype(np.float32) * 0.42).astype(np.uint8)
    bg_src = Image.fromarray(bg_arr, "RGB")
    cx = w * (0.34 if side == "left" else 0.66)
    cy = h * 0.50
    nm_font = look.font("title", int(h * 0.052))
    sb_font = look.font("label", int(h * 0.030))
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # slow push-in on the whole comp
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bg_src.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        # portrait enters: settle + fade
        ent = look.ease_out_cubic(min(1.0, t / 0.6))
        psc = (0.96 + 0.04 * ent) * z
        frame = look.paste_center(frame, tile, cx=cx, cy=cy, scale=psc, opacity=ent)
        # lower-third text on the opposite side, reveals after settle
        if name:
            la = look.ease_out_cubic(max(0.0, (t - 0.7) / 0.7))
            tx = w * (0.66 if side == "left" else 0.34)
            dctx = ImageDraw.Draw(frame)
            ty = int(h * 0.52)
            look.hairline(dctx, int(tx), ty - int(h * 0.045),
                          int(w * 0.12 * la), pal)
            nm = look.text_with_glow(name, nm_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=6,
                                     glow_alpha=0.0, pad=28)
            frame = look.paste_center(frame, nm, cx=tx, cy=ty, opacity=la)
            if sub:
                sbl = look.text_with_glow(sub.upper(), sb_font, fill=pal["muted"],
                                          glow=pal["bg_b"], glow_radius=4,
                                          glow_alpha=0.0, pad=22)
                frame = look.paste_center(frame, sbl, cx=tx,
                                          cy=ty + int(h * 0.055), opacity=la)
        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
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
