"""Primitive: before_after_slider.

A transformation beat — a bright vertical divider sweeps across the frame,
wiping an "AFTER" state over a "BEFORE" state of the same image: then → now,
ruin → restoration, plan → reality. BEFORE / AFTER tags ride the two sides.

Distinct from `comparison_split` (two DIFFERENT subjects side by side) and
`annotated_detail_callout` (one image with a detail ringed): this is the SAME
frame in two states, the change revealed by a moving seam.

Forensic principle (NOT asset copy): a wipe between two states of one image makes
change tangible; the premium is the matched grade + a clean lit seam + restrained
tags, not a cheesy slider UI. When only one image is available the "before" is a
degraded (cold, desaturated, grain-heavy) version of the same frame, so the beat
always lands. Pure-local. No paid API. Deterministic.

    render("x.mp4", image_path="city.jpg", before_label="1900",
           after_label="TODAY", caption="The same street, a century apart")
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .. import look

SPEC = {
    "id": "before_after_slider", "family": "reveals",
    "roles": ["transformation", "before_after", "change", "then_now", "reveal"],
    "niches_ok": ["history", "biography", "geopolitics", "business", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeInOutCubic", "audio_cue": "soft_wipe_swish",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["vertical_wipe"],
    "review_override": ["image_path", "before_label", "after_label", "caption",
                        "palette"],
    "fallback": "two graded then/now panels if no image is available",
}


def _cover(img, bw, bh):
    iw, ih = img.size
    s = max(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    im = img.resize((nw, nh), Image.LANCZOS)
    return im.crop(((nw - bw) // 2, (nh - bh) // 2,
                    (nw - bw) // 2 + bw, (nh - bh) // 2 + bh))


def _degrade(img, pal):
    """A believable 'before': desaturate, cool/sepia, darken, heavy grain — the
    same frame, visibly older."""
    im = ImageEnhance.Color(img).enhance(0.32)
    im = ImageEnhance.Brightness(im).enhance(0.82)
    im = ImageEnhance.Contrast(im).enhance(0.92)
    arr = np.asarray(im).astype(np.float32)
    arr[..., 0] *= 1.04; arr[..., 2] *= 0.9                    # warm-sepia bias
    rng = np.random.default_rng(7)
    arr += rng.normal(0, 9.0, arr.shape[:2] + (1,))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def _panel(w, h, pal, *, warm, seed):
    """A self-contained then/now panel with guaranteed brightness + hue contrast
    (does NOT derive from the near-black graded bg, so it reads even on the very
    dark cold-steel palette). Cold = dim steel-blue with a faint plan grid; warm =
    brighter amber ground with a soft centre glow."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if warm:
        arr = np.empty((h, w, 3), np.float32)
        arr[:] = np.array([54, 40, 24], np.float32)            # warm mid-tone
        r = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2
                    + ((yy - h * 0.42) / (h * 0.62)) ** 2)
        glow = np.clip(1.0 - r, 0.0, 1.0) ** 1.5
        arr += glow[..., None] * np.array(pal["glow"], np.float32) * 0.55
    else:
        arr = np.empty((h, w, 3), np.float32)
        arr[:] = np.array([16, 22, 32], np.float32)            # cool steel-blue
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        gd = ImageDraw.Draw(img)
        step = max(40, w // 26)
        for gx in range(0, w, step):
            gd.line([(gx, 0), (gx, h)], fill=(28, 36, 50), width=1)
        for gy in range(0, h, step):
            gd.line([(0, gy), (w, gy)], fill=(28, 36, 50), width=1)
        arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 5.0, (h, w, 1))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def render(out_path, *, image_path=None, before_label: str = "BEFORE",
           after_label: str = "AFTER", caption: str = "", dur: float = 5.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    after_img = before_img = None
    if image_path and Path(str(image_path)).exists():
        try:
            src = _cover(Image.open(image_path).convert("RGB"), w, h)
            after_img = look.grade_media(src, pal, strength=0.4)
            before_img = _degrade(src, pal)
        except Exception:                                      # noqa: BLE001
            after_img = before_img = None
    if after_img is None:
        before_img = _panel(w, h, pal, warm=False, seed=seed)
        after_img = _panel(w, h, pal, warm=True, seed=seed + 1)
    before_np = np.asarray(before_img).astype(np.uint8)
    after_np = np.asarray(after_img).astype(np.uint8)

    tag_font = look.font("label", int(h * 0.030))
    cap_font = look.font("title", int(h * 0.038))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        # seam sweeps left→right; AFTER on the left of the seam, BEFORE on right
        wipe = look.ease_in_out_cubic(min(1.0, (t - 0.3) / max(0.1, dur * 0.55)))
        wipe = max(0.0, wipe)
        sx = int(w * (0.04 + 0.92 * wipe))
        comp = before_np.copy()
        if sx > 0:
            comp[:, :sx] = after_np[:, :sx]
        frame = Image.fromarray(comp, "RGB")
        d = ImageDraw.Draw(frame, "RGBA")
        # lit seam line + soft glow
        if 2 < sx < w - 2:
            gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(gl).line([(sx, 0), (sx, h)], fill=(*pal["accent_hi"], 180),
                                    width=8)
            frame = Image.alpha_composite(
                frame.convert("RGBA"), gl.filter(ImageFilter.GaussianBlur(6))
            ).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            d.line([(sx, 0), (sx, h)], fill=(*pal["accent_hi"], 255), width=3)
            # a small handle on the seam
            d.ellipse([sx - 11, int(h * 0.5) - 11, sx + 11, int(h * 0.5) + 11],
                      outline=(*pal["accent_hi"], 255), width=3,
                      fill=(*pal["bg_b"], 200))
        # corner tags: AFTER (left, lit once revealed) / BEFORE (right)
        def _tag(txt, cx, lit):
            tw = d.textlength(txt.upper(), font=tag_font)
            bx, by = int(cx - tw / 2 - 16), int(h * 0.08)
            col = pal["accent_hi"] if lit else pal["muted"]
            d.rounded_rectangle([bx, by, bx + tw + 32, by + int(h * 0.044)], radius=6,
                                fill=(*pal["bg_b"], 210), outline=(*col, 230), width=2)
            ti = look.text_with_glow(txt.upper(), tag_font, fill=col, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=8)
            return ti, cx, by + int(h * 0.022)
        for txt, cx, lit in ((after_label, w * 0.16, sx > w * 0.18),
                             (before_label, w * 0.84, sx < w * 0.82)):
            ti, tcx, tcy = _tag(txt, cx, lit)
            frame = look.paste_center(frame, ti, cx=tcx, cy=tcy, opacity=1.0)
            d = ImageDraw.Draw(frame, "RGBA")
        # caption lower-third
        if caption:
            ca = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.6))
            if ca > 0.01:
                ci = look.text_with_glow(caption, cap_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=6,
                                         glow_alpha=0.0, pad=22)
                frame = look.paste_center(frame, ci, cx=w * 0.5, cy=int(h * 0.9),
                                          opacity=ca)

        frame = look.vignette(frame, strength=0.5)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
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
