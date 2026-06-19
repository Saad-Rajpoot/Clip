"""Primitive: era_band_timeline.

An "ages of…" beat — a horizontal time axis divided into 2-5 labelled era bands,
each a period block whose WIDTH is its span, wiping in left→right with the era
name above and its year range below. Older eras sit dim on the left; the march of
time brightens toward the present.

Distinct from `chronology_timeline` (discrete dated EVENTS pinned to a spine):
this shows continuous PERIODS — "the three ages of the republic", "from steam to
silicon" — as proportional bands, not point events.

Forensic principle (NOT asset copy): MagnatesMedia frames history as eras you can
see the length of; the premium is the graded track + a tonal ramp across the
bands + serif year markers + restraint. Pure-local (PIL + numpy → ffmpeg). No
paid API. Deterministic.

    render("x.mp4", eras=[["Steam", 1780, 1840], ["Rail", 1840, 1900],
           ["Oil", 1900, 1945], ["Silicon", 1945, 2000]], title="THE AGES OF POWER")
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
    "id": "era_band_timeline", "family": "timelines",
    "roles": ["eras", "ages", "periods", "timeline", "chronology"],
    "niches_ok": ["history", "business", "biography", "geopolitics", "tech", "crime"],
    "intensity_range": [2, 4], "duration_range": [4.5, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_era_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["era_bands"],
    "review_override": ["eras", "title", "palette"],
    "fallback": "chronology_timeline if the periods read better as point events",
}


def _intish(x):
    try:
        return int(round(float(str(x).strip())))
    except (TypeError, ValueError):
        return None


def _coerce(eras):
    out = []
    for e in (eras or []):
        name, start, end = "", None, None
        if isinstance(e, dict):
            name = str(e.get("name") or e.get("label") or e.get("era") or "")
            start, end = _intish(e.get("start")), _intish(e.get("end"))
            if end is None and e.get("range"):
                rg = str(e["range"]).replace("—", "-").replace("–", "-")
                parts = [p for p in rg.split("-") if p.strip()]
                if len(parts) >= 2:
                    start, end = _intish(parts[0]), _intish(parts[1])
        elif isinstance(e, (list, tuple)) and e:
            name = str(e[0])
            if len(e) >= 3:
                start, end = _intish(e[1]), _intish(e[2])
            elif len(e) == 2:
                rg = str(e[1]).replace("—", "-").replace("–", "-")
                parts = [p for p in rg.split("-") if p.strip()]
                if len(parts) >= 2:
                    start, end = _intish(parts[0]), _intish(parts[1])
        out.append((name.strip(), start, end))
    return out[:5]


def render(out_path, *, eras=None, title: str = "", dur: float = 6.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _coerce(eras)
    if not data:
        data = [("", None, None)]
    m = len(data)
    # band weights ∝ span when years are usable, else equal
    spans = []
    for _, s, e in data:
        spans.append(max(1, e - s) if (s is not None and e is not None and e > s) else 0)
    if not any(spans):
        spans = [1] * m
    else:
        avg = sum(v for v in spans if v) / max(1, sum(1 for v in spans if v))
        spans = [v if v else int(avg) for v in spans]
    tot = float(sum(spans)) or 1.0

    track_x0, track_x1 = int(w * 0.10), int(w * 0.90)
    track_w = track_x1 - track_x0
    gap = int(w * 0.006)
    xs = [track_x0]
    for k in range(m):
        xs.append(xs[-1] + int(track_w * spans[k] / tot))
    xs[-1] = track_x1
    band_cy = int(h * 0.52)
    band_h = int(h * 0.15)
    band_top, band_bot = band_cy - band_h // 2, band_cy + band_h // 2

    title_font = look.font("label", int(h * 0.032))
    name_font = look.font("label", int(h * 0.030))
    year_font = look.font("numeral", int(h * 0.030))

    def _band_col(k):
        f = 0.5 + 0.5 * (k / max(1, m - 1))         # dim (old) → bright (recent)
        base = np.array(pal["accent"], np.float32)
        hi = np.array(pal["accent_hi"], np.float32)
        c = base * (1 - f) + hi * f
        return tuple(int(x) for x in c)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        d = ImageDraw.Draw(frame, "RGBA")
        if title:
            ta = look.ease_out_cubic(min(1.0, t / 0.5))
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=4,
                                     glow_alpha=0.0, pad=16)
            frame = look.paste_center(frame, ti, cx=w * 0.5, cy=h * 0.20,
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
        # baseline rule under the whole track
        bl = look.ease_out_cubic(min(1.0, t / 0.5))
        d.line([(track_x0, band_bot + int(h * 0.012)),
                (track_x0 + int(track_w * bl), band_bot + int(h * 0.012))],
               fill=(*pal["muted"], 180), width=2)
        for k, (name, s, e) in enumerate(data):
            ent = look.ease_out_cubic(max(0.0, (t - (0.3 + k * 0.32)) / 0.55))
            if ent <= 0.01:
                continue
            x0, x1 = xs[k], xs[k + 1]
            bw = int((x1 - x0 - gap) * ent)
            col = _band_col(k)
            d.rounded_rectangle([x0, band_top, x0 + bw, band_bot], radius=5,
                                fill=(*col, 255), outline=(*pal["bg_b"], 200))
            # bright leading edge while wiping
            if ent < 0.99 and bw > 4:
                d.rectangle([x0 + bw - 4, band_top, x0 + bw, band_bot],
                            fill=(*pal["accent_hi"], 255))
            # era name above the band (fades in once mostly wiped)
            la = look.ease_out_cubic(max(0.0, (ent - 0.5) / 0.5))
            if name and la > 0.02:
                ni = look.text_with_glow(name.upper(), name_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=10)
                frame = look.paste_center(frame, ni, cx=(x0 + x1) // 2,
                                          cy=band_top - int(h * 0.05), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")
            # year range below the band
            if la > 0.02 and (s is not None or e is not None):
                yr = (f"{s}–{e}" if (s is not None and e is not None)
                      else (str(s) if s is not None else str(e)))
                yi = look.text_with_glow(yr, year_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=2,
                                         glow_alpha=0.0, pad=8)
                frame = look.paste_center(frame, yi, cx=(x0 + x1) // 2,
                                          cy=band_bot + int(h * 0.055), opacity=la)
                d = ImageDraw.Draw(frame, "RGBA")
            # boundary tick at each band start
            d.line([(x0, band_bot + int(h * 0.012)),
                    (x0, band_bot + int(h * 0.030))],
                   fill=(*pal["muted"], int(200 * ent)), width=2)

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
