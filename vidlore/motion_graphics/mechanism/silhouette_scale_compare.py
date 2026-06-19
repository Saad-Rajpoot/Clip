"""Primitive: silhouette_scale_compare  (Section C · mechanism).

Grounded in a REAL frame audit — Wendover "Cities at Sea: How Aircraft Carriers
Work" 05:06–05:16 (`reference_videos/frame_sequences/wendover/03_silhouette_-
compare/`): two flat-vector PLAN-VIEW silhouettes are placed on a single dark
textured stage at HONEST TRUE SCALE (the two shapes sized to their real relative
proportion so the eye reads the size difference directly), each carrying a label
and a measurement. The two forms ENTER BY SLIDING APART from the centre to make
room for one another — it is NOT a crossfade and NOT a bar chart — and ONE of
them (the hero) wears a single accent so the comparison has a subject.

This is the PRINCIPLE only — original Vidlore composition on the charcoal `look`
system (our palette, typography, easing, grain, scrim). Nothing of Wendover's
asset / layout / colour / subject is copied: their grey aircraft top-views on
navy become our flat charcoal silhouette blocks with an accent hero, a shared
ground baseline + scale tick that make the true-scale relationship explicit, and
our dissolve envelope. Pure-local, deterministic, footage-first, no paid API.

    render("x.mp4", title="TRUE SCALE",
           items=[{"label": "Blue whale", "size": 30, "note": "30 m"},
                  {"label": "School bus", "size": 12, "note": "12 m"}])
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "silhouette_scale_compare", "family": "mechanism",
    "roles": ["scale", "compare", "size_compare", "true_scale", "proportion",
              "silhouette", "versus", "how_big", "dimension", "side_by_side"],
    "niches_ok": ["science", "tech", "history", "geopolitics", "business"],
    "intensity_range": [2, 4], "duration_range": [4.0, 6.5],
    "easing": "easeInOutCubic", "audio_cue": "slide_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["fullscreen"],
    "review_override": ["title", "items", "palette"],
    "fallback": "stat_compare_bars for an abstract two-value comparison",
    "required_inputs": ["items"],
    "grounded_in": "wendover/03_silhouette_compare (05:06-05:16)",
}


def _norm(items):
    """-> list of {label, size, note} with at least a positive size; keep 2."""
    out = []
    for it in (items or []):
        if isinstance(it, dict):
            lbl = str(it.get("label") or it.get("name") or "").strip()
            try:
                sz = float(it.get("size") if it.get("size") is not None
                           else it.get("value"))
            except (TypeError, ValueError):
                sz = 0.0
            if lbl and sz > 0:
                out.append({"label": lbl, "size": sz,
                            "note": str(it.get("note") or it.get("metric") or "").strip()})
    return out[:2]


def _silhouette(w: int, h: int, fill, *, kind: int = 0, alpha: int = 255):
    """A simple flat-vector silhouette (an opaque RGBA block) of footprint w×h.

    `kind` selects one of a few CLEAN primitive forms so two compared things do
    not read as identical bars — but each is a single smooth body (no awkward
    notches that look like glitches). 0 = rounded capsule, 1 = tapered tower
    (narrows toward the top — a tall slender thing), 2 = broad rounded hull
    (a wide low body). The form is decorative; the SIZE (footprint) carries the
    honest-scale meaning, so every form fills the same w×h bounding box."""
    w = max(2, int(w)); h = max(2, int(h))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    a = max(0, min(255, int(alpha)))
    c = (*tuple(int(v) for v in fill), a)
    r = max(3, int(min(w, h) * 0.18))
    if kind == 1:                                   # tapered tower (tall, slender)
        tw = max(2, int(w * 0.30))                  # narrow top, wider base
        d.polygon([(w // 2 - tw // 2, 0), (w // 2 + tw // 2, 0),
                   (w, h), (0, h)], fill=c)
        d.rounded_rectangle([0, int(h * 0.55), w, h], radius=r, fill=c)
    elif kind == 2:                                 # broad rounded hull (wide, low)
        d.rounded_rectangle([0, int(h * 0.10), w, h], radius=r, fill=c)
        d.ellipse([0, 0, w, int(h * 0.42)], fill=c)  # smooth domed top, no notch
    else:                                           # rounded capsule (default)
        d.rounded_rectangle([0, 0, w, h], radius=r, fill=c)
    return img


def render(out_path, *, title: str = "", items=None, dur: float = 5.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    data = _norm(items)
    if len(data) < 2:
        data = [{"label": "Larger", "size": 2.0, "note": ""},
                {"label": "Smaller", "size": 1.0, "note": ""}]

    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    ink = tuple(pal["text"]); muted = tuple(pal["muted"])

    # hero = the larger thing (gets the single accent). Order [left, right] but
    # keep the original pairing; the bigger size is the hero whichever side.
    hero_idx = 0 if data[0]["size"] >= data[1]["size"] else 1
    sizes = [data[0]["size"], data[1]["size"]]
    smax = max(sizes)

    # ---- TRUE-SCALE geometry -------------------------------------------------
    # Both silhouettes share ONE pixels-per-unit so the on-screen heights are in
    # exactly the real ratio. The LARGEST is sized to a target footprint; the
    # other follows honestly (it can end up much smaller — that is the point).
    ground_y = int(h * 0.80)                         # shared baseline both stand on
    hero_max_h = h * 0.52                             # tallest allowed pixel height
    ppu = hero_max_h / smax                           # pixels per scale-unit (shared)
    px_h = [max(8, sizes[i] * ppu) for i in (0, 1)]
    # honest aspect: a thing's footprint width tracks its height but each form has
    # its own slenderness so they don't read as twin bars.
    aspect = [0.74, 0.50]                             # left a touch wider than right
    forms = [2, 1]                                    # left = slab, right = tower
    if hero_idx == 1:                                 # keep the wider form on hero
        aspect = [0.50, 0.74]; forms = [1, 2]
    px_w = [max(8, px_h[i] * aspect[i]) for i in (0, 1)]

    # final resting x-centres (side by side, generous gutter)
    final_cx = [int(w * 0.30), int(w * 0.70)]
    centre_x = w // 2

    # pre-render the two silhouette layers. The hero is the bright accent at full
    # opacity (the subject); the other is a dim slate at reduced opacity so the
    # eye is led to the hero WITHOUT distorting the honest size comparison.
    slate = tuple(int(v * 0.72 + 18) for v in muted)   # cool dim grey-blue
    sil = []
    for i in (0, 1):
        if i == hero_idx:
            sil.append(_silhouette(int(px_w[i]), int(px_h[i]), accent,
                                   kind=forms[i], alpha=255))
        else:
            sil.append(_silhouette(int(px_w[i]), int(px_h[i]), slate,
                                   kind=forms[i], alpha=210))

    title_font = look.font("title", int(h * 0.040))
    label_font = look.font("label", int(h * 0.028))
    note_font = look.font("numeral", int(h * 0.034))

    # charcoal textured stage (never a flat fill)
    bed = look.graded_background(w, h, pal, seed=seed)
    gd = ImageDraw.Draw(bed, "RGBA")
    step = int(h * 0.045)                             # faint plan dot-grid (subtle)
    for gy in range(step, h, step):
        for gx in range(step, w, step):
            gd.point((gx, gy), fill=(*muted, 16))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # title fades in top-centre with an accent underscore
        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink,
                                     glow=pal["bg_b"], glow_radius=6,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ti, cx=w // 2, cy=int(h * 0.13),
                                      opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
            uw = int(ti.width * 0.30)
            d.rectangle([w // 2 - uw, int(h * 0.175), w // 2 + uw, int(h * 0.178)],
                        fill=(*accent, int(210 * ta)))

        # shared ground baseline draws in (the "true-scale" floor both stand on)
        ga = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.25) / 0.5)))
        if ga > 0.01:
            gx0 = int(w * 0.12); gx1 = int(w * 0.88)
            d.line([(gx0, ground_y), (int(gx0 + (gx1 - gx0) * ga), ground_y)],
                   fill=(*muted, 150), width=2)

        # SLIDE APART: both forms start overlapped at centre, ease to their sides
        sa = look.ease_in_out_cubic(min(1.0, max(0.0, (t - 0.35) / 1.1)))
        appear = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.35) / 0.45)))
        for s in (0, 1):
            cx = int(centre_x + (final_cx[s] - centre_x) * sa)
            sy = ground_y - sil[s].height // 2          # sit ON the baseline
            frame = look.paste_center(frame, sil[s], cx=cx, cy=sy, opacity=appear)
            d = ImageDraw.Draw(frame, "RGBA")

        # once mostly apart, dress each: scrim label, measurement, hero accents
        dress = look.ease_out_cubic(min(1.0, max(0.0, (t - 1.25) / 0.5)))
        if dress > 0.01:
            for s in (0, 1):
                cx = final_cx[s]
                top_y = ground_y - int(px_h[s])
                is_hero = (s == hero_idx)

                # hero gets a single accent: a bright cap stroke + a top tick
                if is_hero:
                    hw = int(px_w[s] * 0.5)
                    d.line([(cx - hw, top_y), (cx + hw, top_y)],
                           fill=(*accent_hi, int(230 * dress)), width=3)
                    d.line([(cx, top_y - int(h * 0.03)), (cx, top_y)],
                           fill=(*accent_hi, int(180 * dress)), width=2)
                    d.ellipse([cx - 5, top_y - int(h * 0.03) - 5,
                               cx + 5, top_y - int(h * 0.03) + 5],
                              fill=(*accent_hi, int(235 * dress)))

                # measurement (note) above the form — accent on hero, ink else
                note = data[s]["note"]
                ny = top_y - int(h * 0.075)
                if note:
                    col = accent_hi if is_hero else ink
                    ni = look.text_with_glow(note, note_font, fill=col,
                                             glow=pal["glow"] if is_hero else pal["bg_b"],
                                             glow_radius=7 if is_hero else 3,
                                             glow_alpha=0.5 if is_hero else 0.0, pad=6)
                    frame = look.paste_center(frame, ni, cx=cx, cy=ny, opacity=dress)
                    d = ImageDraw.Draw(frame, "RGBA")

                # label on a dark scrim BELOW the baseline (always readable)
                lbl = data[s]["label"].upper()
                li = look.text_with_glow(lbl, label_font, fill=ink,
                                         glow=pal["bg_b"], glow_radius=3,
                                         glow_alpha=0.0, pad=6)
                maxw = int(w * 0.34)
                if li.width > maxw:
                    li = li.resize((maxw, int(li.height * maxw / li.width)),
                                   Image.LANCZOS)
                ly = ground_y + int(h * 0.055)
                pad_x, pad_y = int(w * 0.012), int(h * 0.012)
                sx0 = cx - li.width // 2 - pad_x; sx1 = cx + li.width // 2 + pad_x
                sy0 = ly - li.height // 2 - pad_y; sy1 = ly + li.height // 2 + pad_y
                d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=int(h * 0.012),
                                    fill=(8, 10, 14, int(190 * dress)))
                if is_hero:                              # accent edge on hero scrim
                    d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=int(h * 0.012),
                                        outline=(*accent, int(170 * dress)), width=2)
                frame = look.paste_center(frame, li, cx=cx, cy=ly, opacity=dress)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.6)
        frame = look.film_grain(frame, seed=seed, amount=4.5, t=t)
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
            "hero": data[hero_idx]["label"],
            "err": (r.stderr[-200:] if not ok else "")}
