"""Primitive: wealth_arc_counter.

A single live metric that tells its whole arc through MOTION + SEMANTIC COLOUR.
A big serif figure counts up while a small line draws beneath it; if the metric
later falls (``end`` < ``peak``), a second phase recedes and the line + number
morph from rising-green toward falling-red. When only a ``value`` is given it
simply climbs in personal-wealth GOLD. An optional portrait is kept half-in-frame
at the lower edge so the data visibly "has an author".

The colour is the argument: green = rising, red = falling, gold = personal
wealth — the verdict lands before a word is spoken.

Distinct from its siblings (original Vidlore composition, no asset copied):
  * ``gold_number_callout`` is a single STATIC count-up with no trajectory/verdict.
  * ``growth_curve_chart`` draws a multi-point trend but never FALLS or shifts
    colour to carry an editorial turn.
This one fuses a hero count-up + a rise→(optional)fall curve + a verdict hue.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("x.mp4", value=251_000_000_000, peak=340_000_000_000,
           end=180_000_000_000, label="NET WORTH", prefix="$",
           portrait_path="musk.jpg", palette_name="cold_steel")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, ...}.

Grounding (forensic principle, NOT pixels): M 0:10-0:22 seq01 — a price line
counts up green then turns red and crashes, a portrait dropping in over the
wreckage; M 17:28 seq10 — a proportion rises then recedes as its hue morphs
positive->warning; R seq05 — the subject kept half-in-frame so the number has an
owner. We reproduce the GRAMMAR (rise-green / fall-red / gold-wealth, count-up,
half-frame author), never the Market-Summary chrome, ticker styling or portraits.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "wealth_arc_counter", "family": "charts",
    "roles": ["scale", "wealth", "metric", "proof", "stakes"],
    "niches_ok": ["business", "biography"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.0],
    "easing": "easeInOutCubic", "audio_cue": "soft_riser_then_low_thud_on_fall",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "review_override": ["value", "peak", "end", "label", "prefix", "suffix",
                        "portrait_path", "palette"],
    "fallback": False,
}

# Verdict colours (semantic, not branded). Rising-green / falling-red / wealth-gold
# are tuned to sit on the dark charcoal stage and read at a glance.
_RISE = (74, 214, 132)        # gain green
_FALL = (224, 86, 72)         # crash red
_GOLD = (240, 196, 110)       # personal-wealth gold


def _to_num(v, default=None):
    """Coerce anything to a float; tolerate strings like '$1.2B', '3,400', None."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else default
    s = str(v).strip().lower().replace(",", "").replace("$", "").replace("%", "")
    mult = 1.0
    for suf, m in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6),
                   ("t", 1e12), ("b", 1e9), ("m", 1e6), ("k", 1e3)):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            mult = m
            break
    try:
        return float(s) * mult
    except (TypeError, ValueError):
        return default


def _abbrev(v: float) -> tuple[float, str]:
    """Human magnitude suffix so a 12-digit net worth stays readable (251B)."""
    a = abs(v)
    for thresh, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= thresh:
            return v / thresh, suf
    return v, ""


def _fmt(v: float, prefix: str, suffix: str, decimals: int,
         auto_abbrev: bool) -> str:
    if auto_abbrev:
        sv, mag = _abbrev(v)
        dec = 1 if (mag and abs(sv) < 100) else 0
        body = f"{sv:,.{dec}f}{mag}"
    elif decimals > 0:
        body = f"{v:,.{decimals}f}"
    else:
        body = f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def _lerp_rgb(a, b, t):
    t = look.clamp01(t)
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def _arc_shape(value: float, peak: float, end: float, n_pts: int, seed: int):
    """Build the curve's y-profile in DATA space (0..1 of peak) as a believable
    price line: ease up to the peak, then — if end < peak — ease back down to
    end, with a little seeded micro-jitter so it reads like a real series rather
    than a clean parabola. Returns (ys[0..1], rise_end_idx, has_fall)."""
    peak = max(peak, 1e-9)
    # a "fall" arc exists when the metric ends measurably below its peak.
    has_fall = end < peak - 1e-9 and end < value + 1e-6
    # split the curve: ~62% rising to the peak, the rest receding to `end`.
    rise_end = int(round(n_pts * (0.62 if has_fall else 1.0)))
    rise_end = min(max(rise_end, 2), n_pts)
    start = min(value, peak) * 0.10        # start low so the climb has runway
    ys = np.zeros(n_pts, np.float32)
    for i in range(rise_end):
        f = look.ease_out_cubic(i / max(1, rise_end - 1))
        ys[i] = start + (peak - start) * f
    for i in range(rise_end, n_pts):
        f = look.ease_in_out_cubic((i - rise_end + 1) / max(1, n_pts - rise_end))
        ys[i] = peak + (end - peak) * f
    # seeded organic jitter (relative to peak), damped at the very ends
    rng = np.random.default_rng(seed + 7)
    jit = rng.normal(0.0, 0.018, n_pts).astype(np.float32) * peak
    win = np.sin(np.linspace(0, math.pi, n_pts)) ** 0.5
    ys = ys + jit * win
    ys = np.clip(ys, peak * 0.02, peak * 1.06)
    return ys / peak, rise_end, has_fall


def _smooth(vals, k=5):
    if len(vals) < 3:
        return vals
    ker = np.ones(k, np.float32) / k
    pad = k // 2
    return np.convolve(np.pad(vals, pad, mode="edge"), ker, "valid")[: len(vals)]


def render(out_path, *, value, peak=None, end=None, label: str = "",
           prefix: str = "", suffix: str = "", portrait_path=None,
           dur: float = 5.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "cold_steel", seed: int = 0,
           decimals: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    # ---- coerce numbers safely; degrade gracefully on garbage --------------
    value = _to_num(value, 0.0) or 0.0
    peak = _to_num(peak, None)
    end = _to_num(end, None)
    if peak is None or peak <= 0:
        peak = abs(value) if value else 1.0
    peak = max(peak, abs(value), 1e-6)
    has_end = end is not None
    if not has_end:
        end = value                                  # no fall → settle at value
    # wealth-gold when it's a bare personal figure (no explicit fall verdict);
    # otherwise the arc carries a rise/fall verdict in green↔red.
    wealth_mode = not has_end or abs(end - peak) < 1e-9
    final_val = value if not has_end else end
    auto_abbrev = decimals == 0 and abs(peak) >= 1e4   # big money → 251B not 251,000,000,000

    # ---- curve geometry ----------------------------------------------------
    n_pts = 96
    ys01, rise_end, has_fall = _arc_shape(value, peak, end, n_pts, seed)
    ys01 = _smooth(ys01, 5)
    base_y, top_y = int(h * 0.80), int(h * 0.50)     # curve lives lower third
    plot_h = base_y - top_y
    span = w * 0.56
    left = w * 0.5 - span / 2
    if portrait_path:                                # shift plot left of portrait
        left = w * 0.40 - span / 2
    px = np.array([left + span * (i / (n_pts - 1)) for i in range(n_pts)])
    py = np.array([base_y - plot_h * v for v in ys01])
    # per-point data value (for the count-up under the head) in real units
    pv = ys01 * peak

    # ---- portrait (kept HALF in frame, lower-right) ------------------------
    port = None
    if portrait_path and Path(str(portrait_path)).exists():
        try:
            pw, ph = int(w * 0.30), int(h * 0.78)
            src = Image.open(str(portrait_path))
            port = look.face_safe_crop(src, pw, ph, face_top_frac=0.30)
            port = look.grade_media(port, pal, strength=0.55)
            # soft feather on the LEFT edge so it melts into the stage
            mask = Image.new("L", (pw, ph), 255)
            md = ImageDraw.Draw(mask)
            for x in range(int(pw * 0.34)):
                md.line([(x, 0), (x, ph)], fill=int(255 * (x / (pw * 0.34))))
            port = (port, mask, pw, ph)
        except Exception:                            # noqa: BLE001 — never crash
            port = None

    num_font = look.font("numeral", int(h * 0.155))  # hero number ≥ h*0.12
    lab_font = look.font("label", int(h * 0.034))
    tick_font = look.font("label", int(h * 0.022))
    accent = tuple(pal["accent"])
    muted = tuple(pal["muted"])

    # axis tick values (0 .. peak rounded) for the financial-UI restraint
    n_ticks = 4
    tick_ys = [base_y - plot_h * (k / n_ticks) for k in range(n_ticks + 1)]
    tick_vals = [peak * (k / n_ticks) for k in range(n_ticks + 1)]

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr,
                                       floor=look.CARD_STAGE_FLOOR)

        # portrait first (behind the data), rising into half-frame + settling
        if port is not None:
            pim, pmask, pw, ph = port
            ent = look.ease_out_cubic(min(1.0, t / 0.9))
            slide = int((1 - ent) * h * 0.12)
            x0 = w - int(pw * 0.62)                   # ~62% width on-screen → half-ish
            y0 = int(h - ph * 0.92) + slide
            pa = look.fade_alpha(t, dur, fps) * (0.30 + 0.55 * ent)
            m = pmask.point(lambda v: int(v * look.clamp01(pa)))
            frame = frame.convert("RGBA")
            frame.paste(pim.convert("RGB"), (x0, y0), m)
            frame = frame.convert("RGB")

        d = ImageDraw.Draw(frame, "RGBA")
        ga = look.ease_out_cubic(min(1.0, t / 0.55))
        # faint gridlines + lit tick labels
        for ty, tv in zip(tick_ys, tick_vals):
            d.line([(int(left), int(ty)), (int(left + span), int(ty))],
                   fill=(*muted, int(34 * ga)), width=1)
        bl = look.ease_out_cubic(min(1.0, t / 0.6))
        d.line([(int(left), base_y), (int(left + span * bl), base_y)],
               fill=(*muted, 210), width=2)
        for ty, tv in zip(tick_ys, tick_vals):
            tl = look.text_with_glow(_fmt(tv, prefix, "", 0, auto_abbrev),
                                     tick_font, fill=muted, glow=pal["bg_b"],
                                     glow_radius=2, glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, tl, cx=int(left) - int(w * 0.025),
                                      cy=int(ty), opacity=ga * 0.85)
        d = ImageDraw.Draw(frame, "RGBA")

        # ---- curve reveal: head rides the leading edge (eased) -------------
        hp = look.ease_in_out_cubic(min(1.0, t / max(0.1, dur * 0.70)))
        hj = max(1, int(hp * (n_pts - 1)))
        # colour at the head: gold in wealth-mode; else green while at/above the
        # peak-ward rise, morphing to red as the head descends below the peak.
        if wealth_mode:
            head_col = _GOLD
        else:
            # how far the head has receded from the peak toward `end`
            if hj <= rise_end or peak <= end:
                fall_f = 0.0
            else:
                fall_f = (peak - pv[hj]) / max(1e-6, (peak - end))
            head_col = _lerp_rgb(_RISE, _FALL, look.clamp01(fall_f))

        rx, ry = px[: hj + 1], py[: hj + 1]
        if len(rx) >= 2:
            line = [(int(a), int(b)) for a, b in zip(rx, ry)]
            # area fill under the revealed curve (verdict-tinted, soft)
            poly = [(int(left), base_y)] + line + [(int(rx[-1]), base_y)]
            area = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(area).polygon(poly, fill=(*head_col, 60))
            area = area.filter(ImageFilter.GaussianBlur(2.0))
            frame = Image.alpha_composite(frame.convert("RGBA"), area).convert("RGB")
            # the curve: per-segment colour so the rising part stays green and
            # the falling tail reddens (the seq01 green→red flip along ONE line).
            glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            for s in range(len(line) - 1):
                if wealth_mode:
                    cseg = _GOLD
                else:
                    seg_fall = (0.0 if (s <= rise_end or peak <= end)
                                else (peak - pv[s]) / max(1e-6, (peak - end)))
                    cseg = _lerp_rgb(_RISE, _FALL, look.clamp01(seg_fall))
                gd.line([line[s], line[s + 1]], fill=(*cseg, 150), width=9)
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                glow.filter(ImageFilter.GaussianBlur(5))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            for s in range(len(line) - 1):
                if wealth_mode:
                    cseg = _GOLD
                else:
                    seg_fall = (0.0 if (s <= rise_end or peak <= end)
                                else (peak - pv[s]) / max(1e-6, (peak - end)))
                    cseg = _lerp_rgb(_RISE, _FALL, look.clamp01(seg_fall))
                d.line([line[s], line[s + 1]], fill=(*cseg, 255), width=4)
            # tracer from head to baseline + glowing plot-head
            hx, hy = int(rx[-1]), int(ry[-1])
            d.line([(hx, hy), (hx, base_y)], fill=(*head_col, 110), width=1)
            for rr, al, fill in ((18, 55, False), (11, 120, False), (6, 255, True)):
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr],
                          fill=(*head_col, al) if fill else None,
                          outline=None if fill else (*head_col, al), width=2)

        # ---- hero count-up number (top), semantic colour + glow -----------
        # value tracks the head while drawing, then snaps to the true final value
        cur = float(pv[hj]) if hj < n_pts - 1 else final_val
        if hp >= 0.999:
            cur = final_val
        num_col = head_col
        settle = 1.0 + 0.05 * (1 - look.ease_out_cubic(min(1.0, t / (dur * 0.45))))
        numeral = look.text_with_glow(
            _fmt(cur, prefix, suffix, decimals, auto_abbrev), num_font,
            fill=num_col, glow=num_col, glow_radius=int(h * 0.020),
            glow_alpha=0.42, pad=int(h * 0.05))
        nfa = look.fade_alpha(t, dur, fps)
        ncx = left + span * 0.5 if not portrait_path else w * 0.40
        frame = look.paste_center(frame, numeral, cx=ncx, cy=h * 0.255,
                                  scale=settle, opacity=nfa)

        # label under the number (fades in after a beat)
        if label:
            la = look.ease_out_cubic(max(0.0, (pr - 0.10) / 0.30)) * nfa
            if la > 0.02:
                lab = look.text_with_glow(label.upper(), lab_font,
                                          fill=pal["text"], glow=pal["bg_b"],
                                          glow_radius=4, glow_alpha=0.0, pad=24)
                frame = look.paste_center(frame, lab, cx=ncx, cy=h * 0.39,
                                          opacity=la)

        # finish: vignette + grain; fade the WHOLE composite toward the charcoal
        # STAGE floor (never black) on the in/out envelope.
        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = look.fade_frame(frame, fa, pal)
        frame.save(td / f"f{i:05d}.png")

    return _encode(td, out_path, fps, crf, n, dur, w, h, t0)


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                           # noqa: BLE001
        ff = "ffmpeg"
    out_path = str(out_path)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
           "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = (r.returncode == 0 and Path(out_path).exists())
    look.cleanup_frames(td)
    return {"ok": ok, "path": out_path, "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
