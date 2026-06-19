"""Primitive: logo_link_beam.

Two icons/logos fly in from the sides, a glowing beam travels from one to the
other and lands with a pulse, and an optional title/caption settles below — the
"these two connect" beat (integration, partnership, A→B, tool↔tool, alliance,
cause→effect handoff). The visual cousin of `diplomatic_link` (which plots two
actors on a MAP); this one is map-free and logo-first, so it works for tech /
business / product / biography beats, not just geopolitics.

Icons are OPTIONAL: pass `a_icon` / `b_icon` as image paths (PNG logos look
best on the auto white plate) and they're fitted into rounded tiles; with no
images it falls back to clean lettered tiles built from `a_label` / `b_label`,
so the beat always lands. Pure-local PIL — no paid API, same render vocabulary
as every other primitive.

    render("x.mp4", a_label="AI", b_label="MG", title="Script to motion",
           caption="one prompt, full animation", palette_name="amber_gold")
    render("x.mp4", a_icon="codex.png", b_icon="remotion.png",
           title="Codex × Remotion")
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "logo_link_beam", "family": "diagrams",
    "roles": ["connect", "link", "integration", "partnership", "relationship",
              "network", "handoff"],
    "niches_ok": ["tech", "business", "geopolitics", "history", "biography", "all"],
    "intensity_range": [2, 4], "duration_range": [3.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_connect_pulse",
    "repeat_cooldown_s": 45, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["beam_link"],
    "review_override": ["a_label", "b_label", "a_icon", "b_icon", "title",
                        "caption", "palette"],
    "fallback": "two lettered icon tiles linked by a travelling beam if no "
                "logos are supplied",
}


def _icon_tile(src, label, size, pal, accent):
    """A rounded icon tile — a fitted logo on a light plate if `src` resolves,
    else a clean lettered tile in the palette accent. Always returns RGBA."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rad = int(size * 0.18)
    logo = None
    if src:
        try:
            if Path(str(src)).exists():
                lg = Image.open(src).convert("RGBA")
                box = int(size * 0.72)
                lg.thumbnail((box, box), Image.LANCZOS)
                logo = lg
        except Exception:                                      # noqa: BLE001
            logo = None
    if logo is not None:
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=rad,
                            fill=(244, 244, 246, 255))
        img.alpha_composite(logo, ((size - logo.width) // 2,
                                   (size - logo.height) // 2))
        ImageDraw.Draw(img).rounded_rectangle(
            [0, 0, size - 1, size - 1], radius=rad,
            outline=(*accent, 210), width=4)
    else:
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=rad,
                            fill=(accent[0], accent[1], accent[2], 255))
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=rad,
                            outline=(255, 255, 255, 90), width=3)
        txt = ((label or "").strip()[:3].upper()) or "•"
        f = look.font("title", int(size * 0.40))
        tw = d.textlength(txt, font=f)
        d.text(((size - tw) / 2, size * 0.28), txt, font=f, fill=(22, 17, 10, 255))
    return img


def render(out_path, *, a_label: str = "", b_label: str = "",
           a_icon=None, b_icon=None, title: str = "", caption: str = "",
           outcome: str = "connect", dur: float = 4.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    size = int(h * 0.22)
    tile_a = _icon_tile(a_icon, a_label, size, pal, pal["accent"])
    tile_b = _icon_tile(b_icon, b_label, size, pal, pal["accent_hi"])

    cy = h * 0.46
    axf, bxf = w * 0.345, w * 0.655          # final centres
    a_start, b_start = w * 0.09, w * 0.91    # fly-in from near the edges
    lab_font = look.font("label", int(h * 0.030))

    def E(x):
        return look.ease_out_cubic(max(0.0, min(1.0, x)))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)

        fin = E(t / 0.6)                                       # fly-in
        ax = a_start + (axf - a_start) * fin
        bx = b_start - (b_start - bxf) * fin
        frame = look.paste_center(frame, tile_a, cx=ax, cy=cy, opacity=fin)
        frame = look.paste_center(frame, tile_b, cx=bx, cy=cy, opacity=fin)
        d = ImageDraw.Draw(frame, "RGBA")

        # the travelling beam (only once both tiles have landed)
        beam = E((t - 0.72) / 1.15)
        if beam > 0.01 and fin >= 0.99:
            edge_a = axf + size * 0.5
            edge_b = bxf - size * 0.5
            d.line([(edge_a, cy), (edge_b, cy)],
                   fill=(*pal["accent_hi"], 105), width=3)
            bxp = edge_a + (edge_b - edge_a) * beam
            for r, a in ((28, 55), (17, 135), (8, 255)):       # glowing ball
                d.ellipse([bxp - r, cy - r, bxp + r, cy + r],
                          fill=(*pal["accent_hi"], a))
            if beam > 0.95:                                    # arrival pulse on B
                p2 = E((t - (0.72 + 1.15 * 0.95)) / 0.5)
                rr = int(size * 0.95 * p2)
                d.ellipse([bxf - rr, cy - rr, bxf + rr, cy + rr],
                          outline=(*pal["accent_hi"], int(170 * (1 - p2))), width=4)

        # title (above) + caption (below) settle in last
        if title:
            ta = E((t - 0.4) / 0.6)
            if ta > 0.01:
                ti = look.text_with_glow(title, look.font("title", int(h * 0.045)),
                                         fill=pal["text"], glow=pal["bg_b"],
                                         glow_radius=6, glow_alpha=0.0, pad=16)
                if ti.width > w * 0.8:
                    s = (w * 0.8) / ti.width
                    ti = ti.resize((int(ti.width * s), int(ti.height * s)), Image.LANCZOS)
                frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.20, opacity=ta)
        if caption:
            ca = E((t - 2.1) / 0.6)
            if ca > 0.01:
                ci = look.text_with_glow(caption, lab_font, fill=pal["text_dim"]
                                         if "text_dim" in pal else pal["text"],
                                         glow=pal["bg_b"], glow_radius=5,
                                         glow_alpha=0.0, pad=14)
                frame = look.paste_center(frame, ci, cx=w * 0.5, cy=h * 0.72, opacity=ca)

        frame = look.vignette(frame, strength=0.5)
        frame = look.film_grain(frame, seed=seed, amount=1.6, t=t)
        fa = look.fade_alpha(t, dur, fps) if hasattr(look, "fade_alpha") else 1.0
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
