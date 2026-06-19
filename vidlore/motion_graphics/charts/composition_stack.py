"""Primitive: composition_stack.

A breakdown beat — ONE horizontal bar that equals the whole (100 %) divided into
labelled segments sized by share, the bar wiping in left→right while each
segment's name + percentage settle above it on a tone-ramped fill. "Where every
dollar went."

Distinct from `proportion_ring` (a single share of a whole), `statistic_bar_reveal`
(separate categories at different magnitudes) and `pictograph_scale` (a countable
ratio): this is the FULL composition of one whole — every part, summing to 100 %.

Forensic principle (NOT asset copy): a single 100 % bar split into shares reads as
"this is all of it, here's how it divides"; the premium is the grade + a gold→muted
tone ramp + clean per-segment labels + a left-to-right wipe + restraint.
Pure-local. No paid API. Deterministic.

    render("x.mp4", segments=[["Profit", 60], ["Costs", 30], ["Tax", 10]],
           title="OF EVERY DOLLAR", suffix="%")
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
    "id": "composition_stack", "family": "charts",
    "roles": ["composition", "breakdown", "share", "split", "stat"],
    "niches_ok": ["business", "history", "biography", "geopolitics", "crime", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeOutCubic", "audio_cue": "soft_wipe_rise",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["stacked_bar"],
    "review_override": ["segments", "title", "suffix", "palette"],
    "fallback": "proportion_ring on the largest segment if only one share is clear",
}


def _coerce(segments):
    out = []
    for sgmt in (segments or []):
        if isinstance(sgmt, dict):
            lab, val = str(sgmt.get("label", "")), sgmt.get("value", 0)
        elif isinstance(sgmt, (list, tuple)) and len(sgmt) >= 2:
            lab, val = str(sgmt[0]), sgmt[1]
        else:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append((lab.strip(), v))
    return out[:5]


def _ramp(pal, k, m):
    """Tone from accent_hi (first) toward muted (last) so adjacent segments read
    apart while staying in one gold family."""
    a = np.array(pal["accent_hi"], float)
    b = np.array(pal["muted"], float)
    f = 0.0 if m <= 1 else (k / (m - 1)) * 0.78
    return tuple(int(x) for x in (a * (1 - f) + b * f))


def render(out_path, *, segments=None, title: str = "", suffix: str = "%",
           dur: float = 5.5, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(segments)
    if not data:
        data = [("", 1.0)]
    tot = sum(v for _, v in data) or 1.0
    fracs = [v / tot for _, v in data]
    m = len(data)

    bw, bh = int(w * 0.70), int(h * 0.085)
    bx0 = int(w * 0.5 - bw / 2)
    by0 = int(h * 0.52)
    # precompute segment x ranges
    xs = [bx0]
    for fr in fracs:
        xs.append(xs[-1] + fr * bw)

    title_font = look.font("label", int(h * 0.030))
    name_font = look.font("label", int(h * 0.026))
    pct_font = look.font("numeral", int(h * 0.040))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["muted"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=int(by0 - h * 0.16),
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # bar track
        tr = look.ease_out_cubic(min(1.0, t / 0.5))
        d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=6,
                            fill=(*pal["bg_b"], int(180 * tr)),
                            outline=(*pal["muted"], int(120 * tr)), width=1)
        # segments wipe in left→right
        wipe = look.ease_in_out_cubic(min(1.0, t / max(0.1, dur * 0.5)))
        wipe_x = bx0 + bw * wipe
        for k in range(m):
            x0, x1 = xs[k], xs[k + 1]
            vx1 = min(x1, wipe_x)
            if vx1 <= x0 + 0.5:
                continue
            d.rectangle([int(x0), by0, int(vx1), by0 + bh], fill=(*_ramp(pal, k, m), 255))
            # thin divider
            if vx1 >= x1 - 0.5 and k < m - 1:
                d.line([(int(x1), by0), (int(x1), by0 + bh)],
                       fill=(*pal["bg_b"], 200), width=2)
        # per-segment labels (name + pct) above, once that segment has filled
        for k, (lab, _v) in enumerate(data):
            x0, x1 = xs[k], xs[k + 1]
            cx = (x0 + x1) / 2
            la = look.ease_out_cubic(max(0.0, (wipe_x - x1) / (bw * 0.06) + 0.2))
            la = min(1.0, la)
            if la <= 0.02:
                continue
            seg_w = x1 - x0
            d.line([(int(cx), by0 - int(h * 0.018)), (int(cx), by0)],
                   fill=(*pal["muted"], int(160 * la)), width=1)
            pct = f"{int(round(fracs[k] * 100))}{suffix}"
            pi = look.gold_fill(pct, pct_font, pal, glow_radius=int(h * 0.006),
                                glow_alpha=0.3 * la)
            frame = look.paste_center(frame, pi, cx=cx, cy=by0 - int(h * 0.045),
                                      opacity=la)
            d = ImageDraw.Draw(frame, "RGBA")
            if lab and seg_w > w * 0.06:                       # name below the bar
                ni = look.text_with_glow(lab.upper(), name_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=8)
                frame = look.paste_center(frame, ni, cx=cx,
                                          cy=by0 + bh + int(h * 0.05), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")

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
