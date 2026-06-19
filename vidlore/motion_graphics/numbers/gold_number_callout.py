"""Primitive: gold_number_callout.

A marquee statistic / currency figure that counts up in glowing gold over a
graded, vignetted, grained background — the cinematic version of a flat stat
card. Forensic evidence: MagnatesMedia 689s ("$72,500"), 2125s ("$ $ $").

Pure-local (PIL + numpy frames → ffmpeg encode). No paid API. Deterministic.

    render(value=72500, label="STOLEN PER YEAR", prefix="$",
           out_path="x.mp4", palette_name="amber_gold")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h}.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from .. import look

SPEC = {
    "id": "gold_number_callout", "family": "numbers",
    "roles": ["stat", "reveal", "money", "scale"],
    # Forensic v2 (vs Vidlore b13 "Cornell ... 96%"): a hero numeric stat is
    # universal to every documentary — gating to a few niches starved
    # science/explainer/_default beats of any animated stat, forcing a static
    # card. "all" lets the director's cue+threshold+cooldown scoring decide WHEN.
    "niches_ok": ["all"],
    "intensity_range": [2, 5], "duration_range": [2.0, 4.5],
    "easing": "easeOutExpo", "audio_cue": "silence_then_soft_impact",
    "repeat_cooldown_s": 45, "per_video_cap": 4, "cost": "low",
    "review_override": ["value", "label", "prefix", "suffix", "palette"],
    "fallback": "static gold number (no count-up) if frame budget is tight",
}


def _fmt(v: float, prefix: str, suffix: str, decimals: int) -> str:
    if decimals > 0:
        body = f"{v:,.{decimals}f}"
    else:
        body = f"{int(round(v)):,}"
    return f"{prefix}{body}{suffix}"


def render(out_path, *, value: float, label: str = "", prefix: str = "",
           suffix: str = "", decimals: int = 0, dur: float = 3.4, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "amber_gold",
           seed: int = 0, bg_image: str | None = None,
           media_floor: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    count_frac = 0.55          # count-up finishes at 55% of the clip, then holds
    num_font = look.font("numeral", int(h * 0.20))
    lab_font = look.font("label", int(h * 0.040))
    cx, cyn = w / 2, h * 0.45
    # static graded media background (if footage/image supplied) — graded once
    media = None
    if bg_image and Path(bg_image).exists():
        try:
            media = look.grade_media(Image.open(bg_image), pal, strength=0.6)
            media = media.resize((w, h), Image.LANCZOS)
        except Exception:                                       # noqa: BLE001
            media = None

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        drift = i / max(1, n - 1)
        # background — a graded footage still (bg_image), else a CHARCOAL stat
        # stage. The floor lifts the near-black graded background to a deep
        # charcoal so a dissolve/cut landing on this sparse-bright card never
        # reads as a black frame (the 2:18 "96%" dip); the gold still pops.
        if media is not None:
            zb = 1.0 + 0.05 * drift                       # slow push on footage
            bw, bh = int(w * zb), int(h * zb)
            frame = media.resize((bw, bh), Image.LANCZOS).crop(
                ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
            # luma floor on the FOOTAGE path too: a near-black bg_image (e.g. an
            # ember/flame still behind "64 BYTES") must never read as a dead-black
            # frame. Lift shadows to the same charcoal stage floor used for the
            # media-less branch — flame/gold stay bright, the black bed becomes a
            # deep charcoal stage. (Fixes the 0:37 gold_number-over-ember dip.)
            _fl = int(look.CARD_STAGE_FLOOR)
            if media_floor:
                frame = ImageChops.lighter(
                    frame, Image.new("RGB", frame.size, (_fl, _fl, _fl)))
        else:
            frame = look.graded_background(w, h, pal, seed=seed, drift=drift,
                                           floor=look.CARD_STAGE_FLOOR)
        # Clip-level in/out envelope applied to the FOREGROUND ONLY. The lit stage
        # is always present; the numeral + label fade in/out without ever
        # multiplying the whole frame toward black (the old whole-frame fade is
        # what dropped the card's first/last frames to near-black under a
        # dissolve). The assembler supplies the visual cross-blend at the boundary.
        fa = look.fade_alpha(t, dur, fps)
        # count-up value + reveal envelope
        cf = look.ease_out_expo(min(1.0, t / (dur * count_frac)))
        cur = value * cf
        scale = 1.0 + 0.06 * (1 - cf)                      # settle 1.06 -> 1.0
        glow_a = 0.45 + 0.55 * cf                          # bloom swells in
        numeral = look.gold_fill(_fmt(cur, prefix, suffix, decimals), num_font,
                                 pal, glow_radius=int(h * 0.022),
                                 glow_alpha=glow_a)
        frame = look.paste_center(frame, numeral, cx=cx, cy=cyn, scale=scale,
                                  opacity=fa)
        # hairline + label (fade in after the count settles)
        if label:
            la = look.ease_out_cubic(max(0.0, (cf - 0.5)) / 0.5) * fa
            dctx = ImageDraw.Draw(frame)
            ly = int(cyn + h * 0.135)
            look.hairline(dctx, int(cx), ly - int(h * 0.03),
                          int(w * 0.10 * la), pal)
            lab = look.text_with_glow(label.upper(), lab_font,
                                      fill=pal["text"], glow=pal["bg_b"],
                                      glow_radius=6, glow_alpha=0.0, pad=30)
            frame = look.paste_center(frame, lab, cx=cx, cy=ly, opacity=la)
        # finish: vignette + grain on the composite (stage + faded foreground)
        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
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
    import shutil as _sh
    _sh.rmtree(td, ignore_errors=True)        # never leak PNG frames to /tmp
    return {"ok": ok, "path": out_path, "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
