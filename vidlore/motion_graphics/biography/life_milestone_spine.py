"""Primitive: life_milestone_spine.

The SHAPE OF A LIFE. A horizontal spine is drawn across a dark charcoal stage and
a travelling accent head rides it left-to-right, IGNITING one milestone dot at a
time — childhood, early career, the turning point, the achievement, the setback,
the legacy. As each dot lights, its BOLD YEAR resolves beside it and a SINGLE
re-anchoring caption (in the upper third) swaps in for that beat, so the viewer is
always reading exactly one milestone — never a crowded, all-at-once timeline and
never a slideshow. An optional face-safe portrait sits at the head of the spine,
making the arc about a person.

Forensic principle (NOT an asset copy): premium biography docs anchor a life to an
ORDERED chain of year-stamped beats revealed progressively (an era stamp lands,
holds, then the next), so the audience absorbs one milestone before the next
arrives. Rebuilt from that PRINCIPLE with Vidlore's own dark look system
(deep-charcoal CARD_STAGE_FLOOR stage, warm gold accent, serif numerals, eased
motion, vignette + grain) — never a competitor's art, wordmark, fonts or layout.

Distinct from `chronology_timeline` (generic dated events, all nodes pop in a
short window) — this is a portrait-anchorable LIFE arc whose dots light strictly
one-at-a-time with a single foregrounded caption per beat.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("x.mp4", milestones=[
        {"year": 1839, "label": "Born in poverty"},
        {"year": 1859, "label": "First refinery"},
        {"year": 1870, "label": "Standard Oil founded"},
        {"year": 1911, "label": "Empire broken up"},
        {"year": 1937, "label": "A lasting legacy"},
    ], portrait_path="rockefeller.jpg", palette_name="parchment_sepia")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h}.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "life_milestone_spine", "family": "timelines",
    "roles": ["context", "chronology", "arc", "reveal", "legacy"],
    "niches_ok": ["biography", "history"],
    "intensity_range": [2, 4], "duration_range": [5.0, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_per_dot_tick",
    "repeat_cooldown_s": 999, "per_video_cap": 1, "cost": "low",
    "review_override": ["milestones", "portrait_path", "palette"],
    # a portrait-anchored life arc (dots light one-at-a-time); not a generic
    # multi-event timeline, not a slideshow.
    "fallback": False,
    "required_inputs": ["milestones"],
}

_MAX_DOTS = 6


# ───────────────────────── input normalisation ─────────────────────────
def _normalise_milestones(milestones) -> list[dict]:
    """Accept a list of dicts ({year, label} — plus year/date/when and
    label/caption/text/event aliases) OR (year, label) pairs/tuples OR plain
    strings (treated as a year with no caption). Skip fully-empty entries, clamp
    to 6, and NEVER crash on malformed input."""
    out: list[dict] = []
    if not milestones:
        return out
    if isinstance(milestones, (str, bytes, dict)):     # a single bare item
        milestones = [milestones]
    try:
        seq = list(milestones)
    except TypeError:
        return out
    for it in seq:
        year, label = "", ""
        if isinstance(it, dict):
            year = str(it.get("year") or it.get("date") or it.get("when")
                       or it.get("age") or "").strip()
            label = str(it.get("label") or it.get("caption") or it.get("text")
                        or it.get("event") or it.get("note") or "").strip()
        elif isinstance(it, (list, tuple)):
            if len(it) >= 1 and it[0] is not None:
                year = str(it[0]).strip()
            if len(it) >= 2 and it[1] is not None:
                label = str(it[1]).strip()
        elif it is not None:
            year = str(it).strip()
        if not (year or label):                        # fully empty → skip
            continue
        out.append({"year": year, "label": label})
        if len(out) >= _MAX_DOTS:
            break
    return out


def _wrap(draw, text, fnt, max_w, max_lines=2):
    """Greedy word-wrap to <= max_lines lines that each fit max_w px."""
    if not text:
        return []
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        cand = (cur + " " + wd).strip()
        if draw.textlength(cand, font=fnt) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = wd
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


# ───────────────────────── portrait head ─────────────────────────
def _portrait_head(portrait_path, size: int, pal: dict) -> Image.Image | None:
    """A round-cornered, gold-keylined, face-safe portrait chip for the head of
    the spine. Returns an RGBA canvas (with shadow padding) or None when no usable
    image is supplied — the spine then simply opens flush-left."""
    if not portrait_path:
        return None
    p = str(portrait_path)
    if not Path(p).exists():
        return None
    try:
        im = Image.open(p).convert("RGB")
    except Exception:                                          # noqa: BLE001
        return None
    iw = ih = int(size)
    crop = (look.face_safe_crop(im, iw, ih) if hasattr(look, "face_safe_crop")
            else im.resize((iw, ih), Image.LANCZOS))
    inner = look.grade_media(crop, pal, strength=0.5).convert("RGB")

    bd = max(3, int(size * 0.035))                     # gold border thickness
    pad = max(14, int(size * 0.16))                    # room for the soft shadow
    canvas = Image.new("RGBA", (iw + 2 * bd + 2 * pad, ih + 2 * bd + 2 * pad),
                       (0, 0, 0, 0))
    # drop shadow
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    so = max(5, int(size * 0.05))
    ImageDraw.Draw(sh).rounded_rectangle(
        [pad + so, pad + so, pad + so + iw + 2 * bd, pad + so + ih + 2 * bd],
        radius=int(size * 0.10), fill=(0, 0, 0, 170))
    sh = sh.filter(ImageFilter.GaussianBlur(max(7, int(size * 0.06))))
    canvas.alpha_composite(sh)
    # gold frame plate + photo + bright keyline
    frame = Image.new("RGBA", (iw + 2 * bd, ih + 2 * bd),
                      (*[int(c * 0.82) for c in pal["accent"]], 255))
    frame.paste(inner, (bd, bd))
    ImageDraw.Draw(frame).rectangle(
        [bd - 1, bd - 1, iw + bd, ih + bd], outline=(*pal["accent_hi"], 150), width=1)
    canvas.alpha_composite(frame, (pad, pad))
    return canvas


# ───────────────────────── render ─────────────────────────
def render(out_path, *, milestones, portrait_path=None, dur: float = 6.0,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "parchment_sepia", seed: int = 0,
           crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    arc = _normalise_milestones(milestones)
    if not arc:                                        # never crash on empty input
        return {"ok": False, "err": "life_milestone_spine: no valid milestones",
                "path": str(out_path)}
    m = len(arc)
    n = max(2, int(round(dur * fps)))

    # ── layout ──────────────────────────────────────────────────
    # An optional portrait sits at the head (left); the spine occupies the band to
    # its right. With no portrait the spine simply uses the full width.
    port_sz = int(h * 0.30)
    port = _portrait_head(portrait_path, port_sz, pal)
    spine_cy = int(h * 0.56)
    if port is not None:
        port_cx = int(w * 0.135)
        port_cy = spine_cy
        x0 = int(port_cx + port_sz * 0.62 + w * 0.035)
    else:
        x0 = int(w * 0.115)
    x1 = int(w * 0.90)
    # single dot → centre it; otherwise spread evenly.
    if m == 1:
        dot_x = [int((x0 + x1) / 2)]
    else:
        dot_x = [int(x0 + (x1 - x0) * (k / (m - 1))) for k in range(m)]

    dot_r = max(7, int(h * 0.0125))
    # year alternates above / below the spine so adjacent stamps never collide;
    # the active caption owns the upper third (one milestone read at a time).
    yr_dy = int(h * 0.085)
    yr_font = look.font("numeral", int(h * 0.052))
    cap_font = look.font("label", int(h * 0.040))
    kick_font = look.font("label", int(h * 0.024))

    # ── progressive reveal schedule ─────────────────────────────
    # The spine draws in first, then a travelling head rides it and lights each
    # dot one-at-a-time. Each dot's reveal is centred on the moment the head
    # passes it, so the years arrive strictly in order (year-by-year).
    intro = min(0.9, dur * 0.16)                       # spine + portrait settle
    outro = min(0.9, dur * 0.16)                       # final hold before exit
    travel0 = intro * 0.85                             # head starts riding
    travel1 = dur - outro                              # head reaches the end
    # fraction of the spine the head has covered at time t (eased)
    def _head_frac(t):
        return look.ease_in_out_cubic(
            min(1.0, max(0.0, (t - travel0) / max(1e-3, travel1 - travel0))))
    # the t at which the head reaches dot k (so its reveal can be timed to it)
    dot_frac = [(0.5 if m == 1 else k / (m - 1)) for k in range(m)]

    # pre-render the static portrait head once
    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        drift = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=drift,
                                       floor=look.CARD_STAGE_FLOOR)

        # small kicker eyebrow, top-centre (anchors the device as a life arc)
        ka = look.ease_out_cubic(min(1.0, t / 0.5))
        if ka > 0.01:
            ki = look.text_with_glow("A LIFE IN MILESTONES", kick_font,
                                     fill=pal["muted"], glow=pal["bg_b"],
                                     glow_radius=4, glow_alpha=0.0, pad=14)
            frame = look.paste_center(frame, ki, cx=w / 2, cy=int(h * 0.135),
                                      opacity=ka * 0.9)

        d = ImageDraw.Draw(frame, "RGBA")

        # ── the spine (draws in left→right during the intro) ────
        draw_p = look.ease_out_cubic(min(1.0, max(0.0, (t - intro * 0.15)
                                                  / max(1e-3, intro * 0.9))))
        sx_end = int(x0 + (x1 - x0) * draw_p)
        # base unlit rail (muted), then the lit portion up to the head
        d.line([(x0, spine_cy), (sx_end, spine_cy)],
               fill=(*pal["muted"], 150), width=3)
        hf = _head_frac(t)
        head_x = int(x0 + (x1 - x0) * hf)
        if hf > 0.0:
            lit_end = min(head_x, sx_end)
            d.line([(x0, spine_cy), (lit_end, spine_cy)],
                   fill=(*pal["accent"], 235), width=4)

        # ── milestone dots: light ONE AT A TIME as the head passes ──
        for k in range(m):
            nx = dot_x[k]
            # this dot "ignites" right as the travelling head reaches it
            ignite = look.ease_out_cubic(
                min(1.0, max(0.0, (hf - dot_frac[k]) / 0.06 + 1.0))
            ) if hf >= dot_frac[k] - 0.001 else 0.0
            # also require the spine to have physically reached the dot
            if draw_p < dot_frac[k] - 0.02 and m > 1:
                appear = look.ease_out_cubic(min(1.0, max(0.0, (draw_p - (dot_frac[k] - 0.08)) / 0.08)))
            else:
                appear = 1.0
            lit = ignite
            base_a = max(appear * 0.5, lit)            # unlit dots are faint, not absent
            if base_a <= 0.01:
                continue
            # outer ring
            ring = pal["accent_hi"] if lit > 0.4 else pal["muted"]
            rr = dot_r + int(4 + 3 * lit)
            d.ellipse([nx - rr, spine_cy - rr, nx + rr, spine_cy + rr],
                      outline=(*ring, int(210 * base_a)), width=2)
            # core fill (gold when lit, muted-amber when still ahead of the head)
            core = pal["accent_hi"] if lit > 0.4 else pal["accent"]
            cr = dot_r
            d.ellipse([nx - cr, spine_cy - cr, nx + cr, spine_cy + cr],
                      fill=(*core, int(255 * base_a)))
            # a soft gold bloom on the freshly-lit dot
            if lit > 0.05:
                glow = Image.new("RGBA", (rr * 6, rr * 6), (0, 0, 0, 0))
                ImageDraw.Draw(glow).ellipse(
                    [rr * 2, rr * 2, rr * 4, rr * 4],
                    fill=(*pal["glow"], int(150 * lit)))
                glow = glow.filter(ImageFilter.GaussianBlur(rr))
                frame = look.paste_center(frame, glow, cx=nx, cy=spine_cy)
                d = ImageDraw.Draw(frame, "RGBA")

            # ── bold year beside the dot (alternates above / below) ──
            if arc[k]["year"] and lit > 0.04:
                above = (k % 2 == 0)
                yimg = look.text_with_glow(
                    arc[k]["year"], yr_font, fill=pal["accent_hi"],
                    glow=pal["glow"], glow_radius=int(h * 0.012),
                    glow_alpha=0.45 * lit, pad=14)
                ycy = spine_cy - yr_dy if above else spine_cy + yr_dy
                # tiny tick connecting the year to the dot
                tick_a = int(170 * lit)
                ty0 = spine_cy - dot_r - 3 if above else spine_cy + dot_r + 3
                ty1 = ycy + (yimg.height // 4 if above else -yimg.height // 4)
                d.line([(nx, ty0), (nx, ty1)], fill=(*pal["accent"], tick_a), width=2)
                frame = look.paste_center(frame, yimg, cx=nx, cy=ycy,
                                          opacity=min(1.0, lit * 1.05))
                d = ImageDraw.Draw(frame, "RGBA")

        # ── ONE re-anchoring caption (upper third) for the ACTIVE milestone ──
        # the active milestone = the last dot the head has passed; its caption
        # cross-fades in as that dot lights and out as the next takes over, so
        # exactly one caption is foregrounded — never a wall of text.
        active = -1
        for k in range(m):
            if hf >= dot_frac[k] - 0.001:
                active = k
        if active >= 0 and arc[active]["label"]:
            # local progress of the active dot (0 at ignite → 1 well after)
            since = hf - dot_frac[active]
            nxt = dot_frac[active + 1] if active + 1 < m else 1.05
            span = max(0.04, nxt - dot_frac[active])
            cap_in = look.ease_out_cubic(min(1.0, max(0.0, since / (span * 0.45 + 1e-3))))
            cap_out = 1.0
            if active + 1 < m:                          # fade out as next ignites
                cap_out = 1.0 - look.ease_in_out_cubic(
                    min(1.0, max(0.0, (since - span * 0.72) / (span * 0.28 + 1e-3))))
            cap_a = cap_in * cap_out
            if cap_a > 0.01:
                lines = _wrap(d, arc[active]["label"].upper(), cap_font,
                              int((x1 - x0) * 0.82), max_lines=2)
                cap_cy0 = int(h * 0.245)
                for li, ln in enumerate(lines):
                    cimg = look.text_with_glow(ln, cap_font, fill=pal["text"],
                                               glow=pal["bg_b"], glow_radius=5,
                                               glow_alpha=0.0, pad=16)
                    frame = look.paste_center(
                        frame, cimg, cx=w / 2,
                        cy=cap_cy0 + li * int(h * 0.058) + (1.0 - cap_in) * h * 0.02,
                        opacity=cap_a)
                d = ImageDraw.Draw(frame, "RGBA")

        # ── travelling head (a small gold comet riding the spine) ──
        if 0.0 < hf < 0.999:
            hb = Image.new("RGBA", (dot_r * 8, dot_r * 8), (0, 0, 0, 0))
            ImageDraw.Draw(hb).ellipse(
                [dot_r * 3, dot_r * 3, dot_r * 5, dot_r * 5],
                fill=(*pal["glow"], 230))
            hb = hb.filter(ImageFilter.GaussianBlur(dot_r))
            frame = look.paste_center(frame, hb, cx=head_x, cy=spine_cy)
            d2 = ImageDraw.Draw(frame, "RGBA")
            d2.ellipse([head_x - 5, spine_cy - 5, head_x + 5, spine_cy + 5],
                       fill=(*pal["accent_hi"], 255))

        # ── portrait head at the spine origin (settles during intro) ──
        if port is not None:
            pa = look.ease_out_cubic(min(1.0, t / max(1e-3, intro)))
            psc = 0.94 + 0.06 * pa
            frame = look.paste_center(frame, port, cx=port_cx,
                                      cy=port_cy, scale=psc, opacity=pa)

        frame = look.vignette(frame, strength=0.58)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
        fa = look.fade_alpha(t, dur, fps)
        if fa < 1.0:
            frame = (look.fade_frame(frame, fa, pal) if hasattr(look, "fade_frame")
                     else frame)
        frame.save(td / f"f{i:05d}.png")

    return _encode(td, out_path, fps, crf, n, dur, w, h, t0)


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
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
    if hasattr(look, "cleanup_frames"):
        look.cleanup_frames(td)
    else:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)
    return {"ok": ok, "path": str(out_path), "frames": n, "dur_s": round(dur, 2),
            "render_s": round(time.time() - t0, 2), "w": w, "h": h,
            "err": (r.stderr[-200:] if not ok else "")}
