"""Primitive: spectrum_meter.

A qualitative gauge beat — a clean horizontal track spanning the mid-frame with a
restrained cool→warm gradient (steel → accent → ember), divided into N labelled
bands ("LOW · MODERATE · HIGH · SEVERE") by thin tick dividers. A slim gold
marker (vertical pointer + diamond head) sweeps from the left, EASES to the
value position (0-100) and settles with a tiny overshoot; the band it lands in
brightens (label → text colour + soft glow) while the others stay muted. A big
serif READOUT word (the reached band, or a supplied word) sits above the marker,
with the measured label + an optional title above that.

This answers "THREAT LEVEL: SEVERE" / "CONFIDENCE: LOW" — a *level*, not a
precise figure. Distinct from `gold_number_callout` (one exact hero number) and
`statistic_bar_reveal` (multi-bar proportions): here the meaning is the position
on a qualitative spectrum, felt as one confident needle settling into a band.

Forensic principle (NOT asset copy): a documentary turns an abstract judgement
into a single instrument reading; the premium is the graded gradient + serif
readout + restrained gold needle + damped settle, never a cheap dashboard dial.
Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic given seed.

    render("x.mp4", value=88, label="THREAT LEVEL",
           bands=["LOW", "GUARDED", "ELEVATED", "HIGH", "SEVERE"],
           title="THE ASSESSMENT")
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
    "id": "spectrum_meter", "family": "meters",
    "roles": ["gauge", "meter", "rating", "level", "threat"],
    "niches_ok": ["crime", "geopolitics", "business", "tech", "history", "biography"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeOutExpo", "audio_cue": "soft_meter_sweep",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["band_gauge"],
    "review_override": ["value", "label", "bands", "readout", "palette"],
    "fallback": "gold_number_callout if the figure is precise rather than a qualitative level",
}

_DEFAULT_BANDS = ["LOW", "MODERATE", "HIGH", "SEVERE"]


def _coerce_value(value) -> float:
    """Clamp the marker position to 0-100."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 50.0
    if math.isnan(v) or math.isinf(v):
        v = 50.0
    return max(0.0, min(100.0, v))


def _coerce_bands(bands):
    out = []
    for b in (bands or []):
        s = str(b).strip()
        if s:
            out.append(s)
    if len(out) < 2:
        out = list(_DEFAULT_BANDS)
    return out[:6]


def _band_index(value: float, n: int) -> int:
    """Which band a 0-100 value lands in (n equal segments)."""
    idx = int(value / 100.0 * n)
    return max(0, min(n - 1, idx))


def _gradient_track(width: int, height: int, pal: dict, radius: int) -> Image.Image:
    """Build the cool→warm track as an RGBA layer with numpy: a horizontal sweep
    steel/muted → accent → accent_hi/ember, rounded ends, faint inner top sheen."""
    steel = np.array(pal["muted"], np.float32)
    mid = np.array(pal["accent"], np.float32)
    warm = np.array(pal["accent_hi"], np.float32)
    # cool down the left end a touch so the contrast reads cool→warm
    cool = np.clip(steel * 0.82 + np.array([0, 8, 18], np.float32), 0, 255)

    t = np.linspace(0.0, 1.0, width, dtype=np.float32)
    # two-segment ramp: cool→mid over [0,.5], mid→warm over [.5,1]
    lo = np.clip(t / 0.5, 0, 1)[:, None]
    hi = np.clip((t - 0.5) / 0.5, 0, 1)[:, None]
    seg1 = cool[None] * (1 - lo) + mid[None] * lo
    rampcol = seg1 * (1 - hi) + warm[None] * hi          # (width,3)
    grad = np.repeat(rampcol[None, :, :], height, axis=0)  # (h,w,3)

    # vertical sheen — slightly brighter near the top, darker at the bottom edge
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    shade = 1.0 - 0.16 * np.clip(yy, 0, 1) + 0.10 * np.clip(-yy, 0, 1)
    grad = np.clip(grad * shade, 0, 255)

    rgb = Image.fromarray(grad.astype(np.uint8), "RGB")
    # rounded-rect alpha mask
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, width - 1, height - 1],
                                           radius=radius, fill=255)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer.paste(rgb, (0, 0))
    layer.putalpha(mask)
    return layer


def render(out_path, *, value=50, label: str = "", bands=None, readout: str = "",
           title: str = "", dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "ember_red", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    val = _coerce_value(value)
    band_list = _coerce_bands(bands)
    nb = len(band_list)
    land = _band_index(val, nb)
    label = (label or "").strip()
    title = (title or "").strip()
    word = (readout or "").strip() or band_list[land]

    accent = tuple(pal["accent"])
    accent_hi = tuple(pal["accent_hi"])
    glow = tuple(pal["glow"])
    text = tuple(pal["text"])
    muted = tuple(pal["muted"])

    # ── geometry ──────────────────────────────────────────────
    cx = w * 0.5
    track_w = int(w * 0.62)
    track_h = int(h * 0.052)
    x0 = int(cx - track_w / 2)
    x1 = x0 + track_w
    track_cy = int(h * 0.585)
    y_top = track_cy - track_h // 2
    y_bot = track_cy + track_h // 2
    radius = track_h // 2

    # band boundary x positions (nb segments → nb+1 edges)
    edges = [x0 + int(track_w * k / nb) for k in range(nb + 1)]
    band_centres = [(edges[k] + edges[k + 1]) // 2 for k in range(nb)]
    # marker target x for the value
    marker_x_final = x0 + (val / 100.0) * track_w

    # ── fonts ─────────────────────────────────────────────────
    title_font = look.font("label", int(h * 0.028))
    label_font = look.font("label", int(h * 0.034))
    read_font = look.font("numeral", int(h * 0.108))
    band_font = look.font("label", int(h * 0.0235))

    # pre-build the gradient track once (static) — reused every frame
    track_layer = _gradient_track(track_w, track_h, pal, radius)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")

        # title (small, muted, top)
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=muted,
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=14)
            frame = look.paste_center(frame, ti, cx=cx, cy=h * 0.205, opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # measured label (what is being gauged), just under the title
        if label:
            la = look.ease_out_cubic(min(1.0, (t - 0.12) / 0.5))
            if la > 0.01:
                li = look.text_with_glow(label.upper(), label_font, fill=text,
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=14)
                frame = look.paste_center(frame, li, cx=cx, cy=h * 0.275,
                                          opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")

        # ── track: a muted trough wipes in, then the gradient reveals L→R ──
        tw = look.ease_out_cubic(min(1.0, (t - 0.2) / 0.55))   # trough grow
        if tw > 0.01:
            half = (track_w / 2) * tw
            d.rounded_rectangle([int(cx - half), y_top, int(cx + half), y_bot],
                                radius=radius, fill=(*pal["bg_b"], 230),
                                outline=(*muted, int(120 * tw)), width=2)
        gw = look.ease_out_cubic(min(1.0, (t - 0.45) / 0.85))  # gradient fill
        if gw > 0.01:
            reveal_w = max(1, int(track_w * gw))
            crop = track_layer.crop((0, 0, reveal_w, track_h))
            frame = frame.convert("RGBA")
            frame.alpha_composite(crop, (x0, y_top))
            frame = frame.convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")

        # ── band tick dividers + labels beneath ──
        ba = look.ease_out_cubic(min(1.0, (t - 0.6) / 0.7))
        if ba > 0.01:
            settle_t = max(0.0, min(1.0, (t - 0.55) / max(0.1, dur * 0.42)))
            for e in edges[1:-1]:                              # interior ticks
                d.line([(e, y_top - int(h * 0.012)), (e, y_bot + int(h * 0.012))],
                       fill=(*pal["bg_b"], int(220 * ba)), width=2)
            # end caps (subtle)
            for e in (edges[0], edges[-1]):
                d.line([(e, y_top - int(h * 0.006)), (e, y_bot + int(h * 0.006))],
                       fill=(*muted, int(150 * ba)), width=2)
            for k, name in enumerate(band_list):
                lit = (k == land) and settle_t > 0.78
                fill = text if lit else muted
                galpha = 0.55 if lit else 0.0
                gr = int(h * 0.012) if lit else 4
                bi = look.text_with_glow(name.upper(), band_font, fill=fill,
                                         glow=glow, glow_radius=gr,
                                         glow_alpha=galpha, pad=12)
                frame = look.paste_center(frame, bi, cx=band_centres[k],
                                          cy=y_bot + int(h * 0.052),
                                          opacity=ba * (1.0 if lit else 0.78))
            d = ImageDraw.Draw(frame, "RGBA")

        # ── needle sweep: ease from left, settle with a tiny overshoot ──
        sweep_dur = max(0.1, dur * 0.46)
        s = max(0.0, min(1.0, (t - 0.55) / sweep_dur))
        # eased approach + damped overshoot toward the final position
        base = look.ease_out_expo(s)
        overshoot = math.exp(-5.5 * s) * math.sin(6.2 * s) * 0.05 * (1 - base)
        frac = max(0.0, min(1.06, base + overshoot)) if s > 0 else 0.0
        marker_x = x0 + frac * (marker_x_final - x0)
        marker_x = max(x0, min(x1, marker_x))

        na = look.ease_out_cubic(min(1.0, (t - 0.55) / 0.3))
        if na > 0.01:
            mx = int(marker_x)
            head_h = int(h * 0.030)
            head_w = int(h * 0.020)
            tip_y = y_top - int(h * 0.006)
            top_y = tip_y - head_h
            # faint vertical guide through the track (behind head)
            d.line([(mx, y_top + 2), (mx, y_bot - 2)],
                   fill=(*pal["bg_b"], int(180 * na)), width=3)
            d.line([(mx, y_top + 2), (mx, y_bot - 2)],
                   fill=(*accent_hi, int(200 * na)), width=1)
            # slim pointer stem above the track
            d.line([(mx, top_y), (mx, tip_y)], fill=(*accent_hi, int(235 * na)),
                   width=3)
            # diamond head (gold) with a darker rim for definition
            diamond = [(mx, top_y), (mx + head_w, top_y + head_h // 2),
                       (mx, top_y + head_h), (mx - head_w, top_y + head_h // 2)]
            d.polygon(diamond, fill=(*accent_hi, int(255 * na)),
                      outline=(*pal["bg_b"], int(220 * na)))
            # tiny seat tick below the track at the marker
            d.line([(mx, y_bot + 2), (mx, y_bot + int(h * 0.012))],
                   fill=(*accent_hi, int(220 * na)), width=3)
            # soft glow dot riding the tip
            gd = look.text_with_glow(".", band_font, fill=(0, 0, 0),
                                     glow=glow, glow_radius=int(h * 0.018),
                                     glow_alpha=0.5 * na, pad=20)
            frame = look.paste_center(frame, gd, cx=mx, cy=top_y + head_h // 2,
                                      opacity=na)
            d = ImageDraw.Draw(frame, "RGBA")

        # ── READOUT word above the marker (serif, gold), reveals as it settles ──
        ra = look.ease_out_cubic(max(0.0, (t - (0.55 + sweep_dur * 0.62)) / 0.5))
        if ra > 0.01:
            rise = int((1 - ra) * h * 0.018)
            ri = look.gold_fill(word.upper(), read_font, pal,
                                glow_radius=int(h * 0.020), glow_alpha=0.55 * ra)
            frame = look.paste_center(frame, ri, cx=cx,
                                      cy=h * 0.415 + rise, opacity=ra)
            d = ImageDraw.Draw(frame, "RGBA")
            # a thin gold hairline under the readout for finish
            hw = int(track_w * 0.16 * ra)
            d.line([(int(cx - hw), int(h * 0.470)), (int(cx + hw), int(h * 0.470))],
                   fill=(*accent, int(190 * ra)), width=2)

        # ── finish: vignette + grain + global fade ──
        frame = look.vignette(frame, strength=0.58)
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
    look.cleanup_frames(td)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
