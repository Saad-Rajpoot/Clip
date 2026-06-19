"""Primitive: footage_fact_overlay  (Section C · hybrid).

Grounded in a REAL frame audit — ColdFusion "Apple's AI Disaster - A Rare
Failure" 00:57–01:01 (`reference_videos/frame_sequences/coldfusion/
seq2_june2025_title/`): a single hard fact ("June 2025") sits SCREEN-LOCKED in
the lower-third, dead-still in screen space, while the LIVE SHOT behind it
(a presenter on the Apple Park stage) keeps moving / slowly parallaxes. There is
NO box, NO panel, NO dashboard — just thin white type that stays put over the
footage, readable purely because the type sits over a mid-tone region (and, when
it doesn't, a whisper of darkening). A tiny kicker rides top-left. Then it
dissolves. This is the ANTI-SLIDESHOW WORKHORSE: it turns a plain footage frame
into a "they're telling me a fact" beat without ever leaving the shot.

This is a HYBRID OVERLAY — it composites OVER a footage frame. Given a real
`bg_image` it grades + slow-zooms that frame (so the footage parallaxes under the
locked text); with no frame it falls back to a NEUTRAL graded charcoal bed that
SIMULATES footage so the micro-render still reads.

Original Vidlore design — ColdFusion's PRINCIPLE only (one screen-locked fact,
tiny rule, optional kicker, dark-plate legibility only, no box, ≤~4s, dissolve),
rebuilt on the charcoal `look` system with our palette, typography, easing and
grain. No asset / layout / colour / font copied. Pure-local, deterministic,
footage-first, no paid API.

    render("x.mp4", fact="OVER 2 BILLION DEVICES AFFECTED",
           kicker="JUNE 2025", bg_image="frame.png")
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
    "id": "footage_fact_overlay", "family": "hybrid",
    "roles": ["fact", "stat", "number", "overlay", "footage_fact", "lower_third",
              "screen_locked", "callout", "context", "date"],
    "niches_ok": ["tech", "business", "science", "history", "geopolitics", "crime"],
    "intensity_range": [1, 3], "duration_range": [3.0, 5.0],
    "easing": "easeOutCubic", "audio_cue": "tick_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 4, "cost": "low",
    "full_screen": "overlay",
    "layout_variants": ["lower_left", "lower_third"],
    "review_override": ["fact", "kicker", "palette"],
    "fallback": "gold_number_callout for a standalone hero number on a dark card",
    "required_inputs": ["fact"],
    "grounded_in": "coldfusion/seq2_june2025_title (00:57-01:01)",
}


# ───────────────────────── footage bed ─────────────────────────
def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Cover-crop `img` to exactly w×h (fill the frame, centre, no letterbox)."""
    im = img.convert("RGB")
    W, H = im.size
    s = max(w / W, h / H)
    rw, rh = max(w, int(round(W * s))), max(h, int(round(H * s)))
    im = im.resize((rw, rh), Image.LANCZOS)
    x, y = (rw - w) // 2, (rh - h) // 2
    return im.crop((x, y, x + w, y + h))


def _footage_bed(bg_image, w: int, h: int, pal: dict, seed: int) -> tuple[Image.Image, bool]:
    """Return (bed, is_real). Real footage frame -> cover-crop + gentle palette
    grade (it stays the live shot, only nudged toward the palette). No frame ->
    a NEUTRAL graded charcoal bed that SIMULATES footage so the overlay reads in
    a standalone micro-render. The bed is what PARALLAXES under the locked text."""
    if bg_image and Path(str(bg_image)).exists():
        try:
            bed = _cover(Image.open(str(bg_image)), w, h)
            if hasattr(look, "grade_media"):
                bed = look.grade_media(bed, pal, strength=0.4)
            return bed, True
        except Exception:                                          # noqa: BLE001
            pass
    # Simulated-footage bed: a graded gradient, then a soft diagonal "sheen" band
    # + a faint horizon so the neutral charcoal reads as a real-ish plate (not a
    # flat slide) and the lower-third has something to lock over.
    if hasattr(look, "graded_background"):
        bed = look.graded_background(w, h, pal, seed=seed,
                                     floor=look.CARD_STAGE_FLOOR * 0.8)
    else:
        bed = Image.new("RGB", (w, h), tuple(pal["bg_a"]))
    arr = np.asarray(bed).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sheen = np.exp(-(((xx - w * 0.34) + (yy - h * 0.30)) / (w * 0.55)) ** 2)
    arr += sheen[..., None] * np.array(pal["accent"], np.float32) * 0.05
    horizon = np.exp(-(((yy - h * 0.62) / (h * 0.10)) ** 2))
    arr -= horizon[..., None] * 10.0
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB"), False


def _adaptive_scrim(frame: Image.Image, box, *, base: float) -> float:
    """Sample mean luma under the text `box` and return a scrim strength in
    [base, ~0.62]: bright/busy footage behind the type -> a stronger whisper of
    darkening; an already-dark region -> almost none. This is dark-plate
    legibility ONLY — it never becomes a solid box."""
    x0, y0, x1, y1 = (max(0, int(v)) for v in box)
    x1 = min(frame.width, x1); y1 = min(frame.height, y1)
    if x1 <= x0 or y1 <= y0:
        return base
    crop = np.asarray(frame.crop((x0, y0, x1, y1))).astype(np.float32)
    lum = float((0.299 * crop[..., 0] + 0.587 * crop[..., 1]
                 + 0.114 * crop[..., 2]).mean())
    # luma 0 -> base ; luma 200 -> ~0.62
    return float(min(0.62, base + max(0.0, (lum - 40.0) / 200.0) * 0.42))


def _scrim(frame: Image.Image, cx: int, cy: int, rw: int, rh: int,
           strength: float) -> Image.Image:
    """Composite a soft elliptical darkening centred on the text — feathered to
    nothing at its edge (NO hard rectangle, NO box). Lifts contrast just enough."""
    if strength <= 0.01:
        return frame
    yy, xx = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
    r = np.sqrt(((xx - cx) / max(1.0, rw)) ** 2 + ((yy - cy) / max(1.0, rh)) ** 2)
    m = np.clip(1.0 - r, 0.0, 1.0) ** 1.6            # feathered ellipse, 1 at core
    arr = np.asarray(frame).astype(np.float32)
    arr *= (1.0 - strength * m[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")


def render(out_path, *, fact: str = "", kicker: str = "", bg_image=None,
           dur: float = 4.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    dur = float(max(3.0, min(5.0, dur)))
    n = max(2, int(round(dur * fps)))
    fact = (fact or "").strip()
    if not fact:
        fact = "FACT"
    kicker = (kicker or "").strip()

    accent = tuple(pal["accent_hi"])
    # ColdFusion's fact is near-WHITE, not gold — keep that restraint. Use the
    # palette's brightest text and push it toward white for the locked fact.
    ink = tuple(int(min(255, v + 14)) for v in pal["text"])
    muted = tuple(pal["muted"])

    bed, is_real = _footage_bed(bg_image, w, h, pal, seed)

    # screen-LOCKED layout: lower-left fact (ColdFusion sits it lower-third; we
    # default lower-left for a longer fact line, lower-third-centre for a short
    # one). Everything below is fixed in screen space — it NEVER tracks footage.
    lower_third = (layout or "").startswith("lower_third") or len(fact) <= 14
    margin = int(w * 0.072)
    # auto-fit the fact to one line within the safe width. ColdFusion's locked
    # fact is a THIN CLEAN SANS (not a serif) — use the condensed/sans face so
    # the over-footage fact reads as a lower-third, not a book title.
    fact_w_max = int(w * (0.62 if lower_third else 0.82))
    fsize = int(h * (0.062 if lower_third else 0.055))
    fact_font = look.font("label", fsize, weight=620)
    fact_layer = look.text_with_glow(fact, fact_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=6, glow_alpha=0.0, pad=10)
    if fact_layer.width > fact_w_max:
        sc = fact_w_max / fact_layer.width
        fact_layer = fact_layer.resize(
            (int(fact_layer.width * sc), int(fact_layer.height * sc)), Image.LANCZOS)
    fw, fh = fact_layer.width, fact_layer.height

    if lower_third:
        fact_cx = w // 2
        fact_cy = int(h * 0.74)
        rule_x0 = fact_cx - int(w * 0.05)
        rule_x1 = fact_cx + int(w * 0.05)
        rule_y = int(fact_cy - fh * 0.5 - h * 0.030)
        kick_cx = fact_cx
        kick_cy = rule_y - int(h * 0.034)
        kick_align = "center"
    else:
        fact_cx = margin + fw // 2
        fact_cy = int(h * 0.80)
        rule_x0 = margin
        rule_x1 = margin + int(w * 0.052)
        rule_y = int(fact_cy - fh * 0.5 - h * 0.034)
        kick_cx = margin
        kick_cy = rule_y - int(h * 0.030)
        kick_align = "left"

    kick_font = look.font("label", int(h * 0.022))
    kick_layer = None
    if kicker:
        kick_layer = look.text_with_glow(kicker.upper(), kick_font, fill=muted,
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=6)

    # scrim ellipse geometry (covers fact + rule + kicker region), fixed in screen
    scrim_cx = fact_cx if lower_third else (margin + max(fw, int(w * 0.18)) // 2)
    scrim_cy = int((fact_cy + (kick_cy if kicker else rule_y)) / 2)
    scrim_rw = int(max(fw, (rule_x1 - rule_x0)) * 0.62 + w * 0.055)
    scrim_rh = int((fact_cy - (kick_cy if kicker else rule_y)) * 0.62 + h * 0.075)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)

        # FOOTAGE parallaxes: a slow Ken-Burns zoom-crop of the bed (the live
        # shot keeps moving) — this is the only thing that moves.
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))

        # entrance: fact fades in + rises a hair (0->0.55s), then DEAD STILL.
        ea = look.ease_out_cubic(min(1.0, t / 0.55))
        rise = int((1.0 - ea) * h * 0.018)             # ≤~19px, settles to 0

        # adaptive scrim (sampled under where the fact will sit) — legibility only
        scr_base = 0.16 if is_real else 0.10
        box = (scrim_cx - scrim_rw, scrim_cy - scrim_rh + rise,
               scrim_cx + scrim_rw, scrim_cy + scrim_rh + rise)
        scr = _adaptive_scrim(frame, box, base=scr_base) * ea
        frame = _scrim(frame, scrim_cx, scrim_cy + rise, scrim_rw, scrim_rh, scr)

        d = ImageDraw.Draw(frame, "RGBA")

        # kicker (small, above the rule) — fades first
        if kick_layer is not None:
            ka = look.ease_out_cubic(min(1.0, t / 0.45))
            kx = kick_cx + (kick_layer.width // 2 if kick_align == "left" else 0)
            frame = look.paste_center(frame, kick_layer, cx=kx,
                                      cy=kick_cy + rise, opacity=ka)
            d = ImageDraw.Draw(frame, "RGBA")

        # tiny accent rule (draws in just before the fact)
        ra = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.10) / 0.40)))
        if ra > 0.01:
            rx1 = int(rule_x0 + (rule_x1 - rule_x0) * ra)
            d.line([(rule_x0, rule_y + rise), (rx1, rule_y + rise)],
                   fill=(*accent, int(235 * ra)), width=3)

        # the screen-LOCKED fact (the whole point) — still, white, no box
        frame = look.paste_center(frame, fact_layer, cx=fact_cx,
                                  cy=fact_cy + rise, opacity=ea)

        # finishing: vignette + grain on the COMPOSITE (footage + text together,
        # like a real over-footage lower-third), then the global dissolve.
        frame = look.vignette(frame, strength=0.5)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
        fa = look.fade_alpha(t, dur, fps, fin=0.30, fout=0.45)   # dissolve in/out
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                              # noqa: BLE001
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
            "real_footage": is_real,
            "err": (r.stderr[-200:] if not ok else "")}
