"""Primitive: measurement_callout  (Section C · mechanism).

Grounded in a REAL frame audit — Wendover "Cities at Sea (How Aircraft Carriers
Work)" 05:24-05:33 (`reference_videos/frame_sequences/wendover/04_runway_measure/`).
What Wendover ACTUALLY does there: it states a real dimension HONESTLY as a SPAN
laid over the subject — an aircraft-carrier deck, then a top-down satellite shot
of London City Airport — with a single dual-unit invariant label ("~1100ft
(330m)", "~5000ft (1500m)", "~12001ft (3658m)"): imperial primary, metric in
parentheses, sitting in the lower third over the footage, with a second comparison
caption that re-anchors ("London City Airport" / "London Heathrow"). No dashboard,
no gauges — one measured fact, drawn over the real thing, that then cuts away.

This is a HYBRID OVERLAY. It accepts an optional `bg_image` (a footage frame); if
absent it renders over the charcoal `graded_background`. The measure itself is an
ORIGINAL Vidlore composition: a thin span line GROWS between two end-caps with a
travelling leader, then a readable dual-unit label fades up on a scrim over the
span — built entirely on the charcoal `look` palette / typography / easing / grain.
Wendover's PRINCIPLE only (honest dimension as a span + dual-unit invariant label
over the subject, then dissolve) — NONE of its colours, fonts, layout, satellite
look or comparison-caption stack is copied. Pure-local (PIL + numpy -> ffmpeg),
deterministic, footage-first, no paid API.

    render("x.mp4", value="120 m (394 ft)", label="Flight deck",
           p0=(0.16, 0.58), p1=(0.84, 0.58), palette_name="cold_steel")
    render("x.mp4", value={"value": "120 m", "unit2": "394 ft"},
           bg_image="frame.jpg")
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
    "id": "measurement_callout", "family": "mechanism",
    "roles": ["measurement", "dimension", "scale", "length", "span", "size",
              "distance", "how_big", "to_scale", "callout"],
    "niches_ok": ["science", "tech", "history", "geopolitics"],
    "niches_unsuitable": ["comedy", "lifestyle", "reaction"],
    "intensity_range": [2, 3], "duration_range": [3.5, 6.0],
    "easing": "easeInOutCubic", "audio_cue": "measure_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["span_overlay"],
    "overlay": True,                      # hybrid: drawn OVER footage/bg or charcoal
    "review_override": ["value", "label", "p0", "p1", "palette"],
    "fallback": "stat_emphasis_card for a dimension with no spatial span to draw",
    "required_inputs": ["value"],
    "grounded_in": "wendover/04_runway_measure (05:24-05:33)",
}


def _split_value(value):
    """Normalise `value` -> (primary, secondary) for a dual-unit label.

    Accepts:
      "120 m (394 ft)"            -> ("120 m", "394 ft")
      {"value": "120 m", "unit2": "394 ft"} -> ("120 m", "394 ft")
      "120 m"                     -> ("120 m", "")   (single-unit is allowed)
    """
    if isinstance(value, dict):
        p = str(value.get("value") or value.get("primary") or "").strip()
        s = str(value.get("unit2") or value.get("secondary") or
                value.get("alt") or "").strip()
        return p, s
    s = str(value or "").strip()
    if "(" in s and s.endswith(")"):
        head, _, tail = s.partition("(")
        return head.strip(), tail[:-1].strip()
    return s, ""


def _norm_pt(p, default):
    try:
        x, y = float(p[0]), float(p[1])
        # accept either normalised (0..1) or pixel-ish; clamp to a sane band
        return min(0.97, max(0.03, x)), min(0.92, max(0.08, y))
    except Exception:                                              # noqa: BLE001
        return default


def _bed(bg_image, w, h, pal, seed):
    """The stage the span is drawn over: a graded footage frame if one is
    supplied, otherwise the charcoal graded_background (lifted off black)."""
    if bg_image is not None:
        try:
            src = bg_image if isinstance(bg_image, Image.Image) \
                else Image.open(str(bg_image))
            im = look.grade_media(src.convert("RGB"), pal, strength=0.55)
            # cover-fit to the canvas
            iw, ih = im.size
            s = max(w / iw, h / ih)
            im = im.resize((max(w, int(iw * s)), max(h, int(ih * s))), Image.LANCZOS)
            x = (im.width - w) // 2
            y = (im.height - h) // 2
            im = im.crop((x, y, x + w, y + h))
            # gently darken footage so the white span + label stay legible
            arr = np.asarray(im).astype(np.float32) * 0.82
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        except Exception:                                          # noqa: BLE001
            pass
    return look.graded_background(w, h, pal, seed=seed, floor=look.CARD_STAGE_FLOOR)


def _scrim(w, h, cy, half_h, alpha):
    """A horizontal readability scrim (soft dark band) behind the label so the
    dual-unit text reads over busy footage. RGBA layer the size of the frame."""
    band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if alpha <= 0.01:
        return band
    yy = np.mgrid[0:h, 0:1][0].astype(np.float32).reshape(h, 1)
    fall = np.clip(1.0 - (np.abs(yy - cy) / max(1.0, half_h)) ** 2, 0.0, 1.0)
    a = (fall * (170.0 * look.clamp01(alpha))).astype(np.uint8)
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 3] = np.repeat(a, w, axis=1)
    return Image.fromarray(arr, "RGBA")


def render(out_path, *, value="", label: str = "", p0=None, p1=None,
           bg_image=None, dur: float = 5.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "cold_steel", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    dur = float(min(6.0, max(3.5, dur)))
    n = max(2, int(round(dur * fps)))

    primary, secondary = _split_value(value)
    if not primary:
        primary = "—"
    label_txt = str(label or "").strip()

    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"])
    muted = tuple(pal["muted"]); accent_mid = tuple(pal["accent"])

    # span endpoints (normalised) — honest horizontal-ish dimension
    ax, ay = _norm_pt(p0, (0.16, 0.60)) if p0 else (0.16, 0.60)
    bx, by = _norm_pt(p1, (0.84, 0.60)) if p1 else (0.84, 0.60)
    x0, y0 = int(ax * w), int(ay * h)
    x1, y1 = int(bx * w), int(by * h)
    span_cy = (y0 + y1) // 2
    cap_h = int(h * 0.055)                       # end-cap half-height

    bed = _bed(bg_image, w, h, pal, seed)

    num_font = look.font("numeral", int(h * 0.064))
    alt_font = look.font("label", int(h * 0.030))
    lab_font = look.font("label", int(h * 0.026))

    # animation phases (fractions of dur)
    grow_t0, grow_t1 = 0.45, max(1.4, dur * 0.55)     # span grows
    label_t0 = grow_t1 - 0.25                          # label fades up near full span

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # span growth 0->1 (eased), leader travels with the growing tip
        gp = look.ease_in_out_cubic((t - grow_t0) / max(0.1, grow_t1 - grow_t0))
        # appear envelope for the end-caps (cap0 from the start, cap1 once reached)
        c0a = look.ease_out_cubic((t - grow_t0) / 0.35)
        if gp > 0.0 or c0a > 0.0:
            tipx = int(x0 + (x1 - x0) * gp)
            tipy = int(y0 + (y1 - y0) * gp)

            # ELEMENT 1 — start end-cap (a perpendicular tick) drawn first
            if c0a > 0.01:
                d.line([(x0, y0 - cap_h), (x0, y0 + cap_h)],
                       fill=(*accent, int(235 * look.clamp01(c0a))), width=4)

            # ELEMENT 2 — the measurement line GROWS from cap0 to the tip
            if gp > 0.001:
                d.line([(x0, y0), (tipx, tipy)],
                       fill=(*ink, 235), width=3)
                # a soft accent under-stroke gives it presence over footage
                d.line([(x0, y0 + 2), (tipx, tipy + 2)],
                       fill=(*accent_mid, 90), width=2)
                # travelling leader knot at the growing tip (until it lands)
                if gp < 0.999:
                    d.ellipse([tipx - 6, tipy - 6, tipx + 6, tipy + 6],
                              fill=(*accent, 240))
                    d.ellipse([tipx - 11, tipy - 11, tipx + 11, tipy + 11],
                              outline=(*accent, 120), width=2)

            # end end-cap snaps in once the span has (nearly) reached it
            c1a = look.ease_out_cubic((gp - 0.92) / 0.08)
            if c1a > 0.01:
                d.line([(x1, y1 - cap_h), (x1, y1 + cap_h)],
                       fill=(*accent, int(235 * look.clamp01(c1a))), width=4)

        # ELEMENT 3 — dual-unit label on a scrim, centred over the span midpoint
        la = look.ease_out_cubic((t - label_t0) / 0.5)
        if la > 0.01:
            mid_x = (x0 + x1) // 2
            # keep the label block on-canvas
            mid_x = max(int(w * 0.22), min(int(w * 0.78), mid_x))
            # lift the block above the span line so the line stays visible
            base_y = span_cy - cap_h - int(h * 0.10)
            base_y = max(int(h * 0.12), base_y)
            scrim_h = int(h * 0.16)
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                _scrim(w, h, base_y + int(h * 0.045), scrim_h, la)).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")

            # primary numeral (serif, blooms in the accent glow)
            num = look.text_with_glow(primary, num_font, fill=accent,
                                      glow=pal["glow"], glow_radius=12,
                                      glow_alpha=0.0, pad=10)
            if num.width > w * 0.62:                  # never overflow the frame
                sc = (w * 0.62) / num.width
                num = num.resize((int(num.width * sc), int(num.height * sc)),
                                 Image.LANCZOS)
            frame = look.paste_center(frame, num, cx=mid_x, cy=base_y, opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")

            # secondary unit in parentheses (the dual-unit invariant)
            if secondary:
                alt = look.text_with_glow("(" + secondary + ")", alt_font,
                                          fill=ink, glow=pal["bg_b"],
                                          glow_radius=4, glow_alpha=0.0, pad=6)
                frame = look.paste_center(frame, alt, cx=mid_x,
                                          cy=base_y + int(h * 0.052), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")

            # a tick connector from the label block down to the span line
            d.line([(mid_x, base_y + int(h * 0.078)),
                    (mid_x, span_cy - cap_h)],
                   fill=(*accent, int(150 * la)), width=2)

        # optional subject label, lower-third, muted (re-anchors the dimension)
        if label_txt:
            lba = look.ease_out_cubic((t - (label_t0 + 0.2)) / 0.5)
            if lba > 0.01:
                lab = look.text_with_glow(label_txt.upper(), lab_font,
                                          fill=muted, glow=pal["bg_b"],
                                          glow_radius=3, glow_alpha=0.0, pad=4)
                frame = look.paste_center(frame, lab, cx=(x0 + x1) // 2,
                                          cy=span_cy + cap_h + int(h * 0.055),
                                          opacity=min(1.0, lba) * 0.95)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.55)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
        fa = look.fade_alpha(t, dur, fps)
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
            "err": (r.stderr[-200:] if not ok else "")}
