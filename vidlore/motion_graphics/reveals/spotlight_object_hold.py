"""Primitive: spotlight_object_hold.

A dramatic "behold" reveal — TEXT-LED, no photo. A near-black stage; a soft
circular spotlight pool (a warm-gold radial gradient, bright centre falling to
black) SWEEPS in from an off-centre/edge start and EASES (easeOutExpo) to settle
dead-centre over the first ~45% of the clip. As the light arrives it lifts the
SUBJECT — a bold serif word/short phrase (gold_fill) — out of the darkness,
fading up only as the beam reaches it. A condensed-caps KICKER rides just above
the subject, an optional sub line just below, and an optional title on a hairline
near the bottom. Everything outside the moving pool is crushed to near-black with
a heavy vignette; a slow 3% push-in and faint grain give a cinematic stage feel.

Distinct from `framed_evidence_spotlight` (a STATIC gold-framed PHOTO card under a
fixed key light): there is NO frame and NO image here — the SIGNATURE is the
MOVING light that sweeps then holds, and the payload is type, not an artifact.
Pure-local (PIL + numpy → ffmpeg). No paid API. Deterministic given seed.

    render("x.mp4", subject="THE BUTLER", kicker="THE ONE WHO LIED",
           sub="Seen in the library at midnight", title="CASE CLOSED")
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
    "id": "spotlight_object_hold", "family": "reveals",
    "roles": ["reveal", "spotlight", "subject", "unveil", "focus"],
    "niches_ok": ["crime", "history", "biography", "geopolitics", "business", "tech"],
    "intensity_range": [3, 5], "duration_range": [4.0, 6.5],
    "easing": "easeOutExpo", "audio_cue": "soft_spot_swell",
    "repeat_cooldown_s": 55, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["moving_spotlight"],
    "review_override": ["subject", "kicker", "sub", "title", "palette"],
    "fallback": "framed_evidence_spotlight if there is a real artifact image to show",
}


def _clean(v) -> str:
    """Defensively coerce None/str/list/dict/number to a single clean string."""
    if v is None:
        return ""
    if isinstance(v, dict):
        v = v.get("text", v.get("label", v.get("value", "")))
    if isinstance(v, (list, tuple)):
        v = " ".join(_clean(x) for x in v if x is not None)
    s = str(v).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(s.split())


def _spot_pool(w: int, h: int, pal: dict, *, cx: float, cy: float,
               radius: float, intensity: float) -> np.ndarray:
    """Additive warm-gold light pool (float32 HxWx3): bright at (cx,cy), soft
    falloff to black at `radius`. Built entirely in numpy so it is the genuine
    MOVING signature of this primitive (cheap to re-evaluate each frame)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(1.0, radius)
    # smooth bright core + soft skirt → reads as a real lamp, not a hard disc
    pool = np.clip(1.0 - r, 0.0, 1.0) ** 1.7
    pool = pool + 0.32 * np.clip(1.0 - r * 0.72, 0.0, 1.0) ** 3.0   # hot centre
    warm = np.array(pal["glow"], np.float32)
    return pool[..., None] * (warm[None, None] * (0.62 * intensity))


def _stage_floor(arr: np.ndarray, w: int, h: int, cx: float, cy: float,
                 radius: float) -> np.ndarray:
    """A faint elliptical ground-glow under the pool so the light feels like it
    lands on a stage floor, not floating in a void. Subtle, warm, low."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fy = cy + radius * 0.46                       # pool of light on the 'floor'
    e = np.sqrt(((xx - cx) / (radius * 1.05)) ** 2
                + ((yy - fy) / (radius * 0.26)) ** 2)
    floor = np.clip(1.0 - e, 0.0, 1.0) ** 2.4
    return arr + floor[..., None] * np.array([34.0, 24.0, 12.0], np.float32)


def render(out_path, *, subject: str = "", kicker: str = "", sub: str = "",
           title: str = "", dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "amber_gold", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))

    subject = _clean(subject) or "REVEALED"
    kicker = _clean(kicker)
    sub = _clean(sub)
    title = _clean(title)

    # ── fonts (serif subject scaled down for long phrases so it stays on-frame)
    sl = len(subject)
    s_frac = 0.150 if sl <= 9 else 0.118 if sl <= 16 else 0.092 if sl <= 24 else 0.072
    subj_font = look.font("title", int(h * s_frac))
    kick_font = look.font("label", int(h * 0.030))
    sub_font = look.font("caption", int(h * 0.032))
    title_font = look.font("label", int(h * 0.026))

    # subject pre-rendered once (gold fill + bloom) → reused, then faded/scaled
    subj_layer = look.gold_fill(subject, subj_font, pal,
                                glow_radius=int(h * 0.022), glow_alpha=0.85,
                                pad=int(h * 0.05))
    # shrink if it would overrun the frame width (keep content on-frame)
    max_w = w * 0.84
    if subj_layer.width > max_w:
        sc = max_w / subj_layer.width
        subj_layer = subj_layer.resize(
            (max(1, int(subj_layer.width * sc)),
             max(1, int(subj_layer.height * sc))), Image.LANCZOS)

    cxc, cyc = w * 0.5, h * 0.485                 # settle point (subject centre)
    # spotlight START: off-centre/edge, seeded so repeats vary
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    side = -1.0 if (int(seed) % 2 == 0) else 1.0
    start_x = w * (0.5 + side * (0.46 + 0.10 * rng.random()))
    start_y = h * (0.30 + 0.40 * rng.random())
    r_settle = h * 0.50                           # held pool radius
    r_start = h * 0.30                            # tighter while sweeping in

    sweep_dur = max(0.1, dur * 0.45)              # arrives by ~45%

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)

        # ── slow 3% push-in over the whole clip
        push = 1.0 + 0.03 * look.ease_out_cubic(pr)

        # ── moving spotlight: easeOutExpo sweep from start → centre, then HOLD
        s = look.ease_out_expo(min(1.0, t / sweep_dur))
        scx = start_x + (cxc - start_x) * s
        scy = start_y + (cyc - start_y) * s
        srad = (r_start + (r_settle - r_start) * s) * push
        # tiny living breath once settled (lamp never goes dead-still)
        if s >= 0.999:
            settle_t = t - sweep_dur
            srad *= 1.0 + 0.012 * math.sin(settle_t * 1.6)
            scy += h * 0.004 * math.sin(settle_t * 1.1)

        # near-black stage from the graded bg, crushed hard so only the pool lights
        base = np.asarray(look.graded_background(w, h, pal, seed=seed, drift=pr)
                          ).astype(np.float32) * 0.16
        base = _stage_floor(base, w, h, scx, scy, srad)
        base = base + _spot_pool(w, h, pal, cx=scx, cy=scy, radius=srad,
                                 intensity=1.0)
        frame = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

        # how far the beam has reached the subject (drives the lift-out fade)
        reach = look.ease_out_cubic(
            look.clamp01((s - 0.45) / 0.5)) * look.fade_alpha(t, dur, fps)

        # ── KICKER (condensed caps, just above the subject) — tracks in with light
        if kicker and reach > 0.01:
            ki = look.text_with_glow(kicker.upper(), kick_font, fill=pal["accent_hi"],
                                     glow=pal["glow"], glow_radius=int(h * 0.010),
                                     glow_alpha=0.5 * reach, pad=int(h * 0.018))
            frame = look.paste_center(frame, ki, cx=cxc,
                                      cy=cyc - subj_layer.height * 0.5 * push
                                      - int(h * 0.052),
                                      scale=push, opacity=reach)

        # ── SUBJECT (gold serif) — rises a touch as it brightens
        rise = (1.0 - reach) * h * 0.018
        frame = look.paste_center(frame, subj_layer, cx=cxc,
                                  cy=cyc + rise, scale=push,
                                  opacity=reach)

        d = ImageDraw.Draw(frame, "RGBA")

        # ── SUB line (just below the subject)
        if sub:
            sa = look.ease_out_cubic(look.clamp01((s - 0.6) / 0.4)) \
                * look.fade_alpha(t, dur, fps)
            if sa > 0.01:
                si = look.text_with_glow(sub, sub_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=4,
                                         glow_alpha=0.0, pad=int(h * 0.02))
                frame = look.paste_center(
                    frame, si, cx=cxc,
                    cy=cyc + subj_layer.height * 0.5 * push + int(h * 0.052),
                    scale=push, opacity=sa)
                d = ImageDraw.Draw(frame, "RGBA")

        # ── TITLE on a hairline near the bottom (settles last)
        if title:
            ta = look.ease_out_cubic(look.clamp01((t - sweep_dur - 0.2) / 0.6)) \
                * look.fade_alpha(t, dur, fps)
            if ta > 0.01:
                ty = int(h * 0.885)
                ti = look.text_with_glow(title.upper(), title_font, fill=pal["muted"],
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=int(h * 0.016))
                hw = int(min(w * 0.20, ti.width * 0.5 + w * 0.06))
                look.hairline(d, int(w * 0.5 - hw - ti.width * 0.5 - w * 0.012),
                              ty, int(hw * ta), pal)
                look.hairline(d, int(w * 0.5 + hw + ti.width * 0.5 + w * 0.012),
                              ty, int(hw * ta), pal)
                frame = look.paste_center(frame, ti, cx=w * 0.5, cy=ty, opacity=ta)

        # ── stage treatment: heavy vignette so off-pool is truly crushed
        frame = look.vignette(frame, strength=0.72)
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
