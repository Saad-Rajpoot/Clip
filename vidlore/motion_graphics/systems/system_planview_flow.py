"""Primitive: system_planview_flow  (Section C · systems).

Grounded in a REAL frame audit — Wendover "Cities at Sea" 05:36–05:48
(`reference_videos/frame_sequences/wendover/05_deck_schematic/`): a documentary
system diagram is a PERSISTENT TOP-DOWN PLAN-VIEW of the real thing's shape on a
textured stage, explained over time by a translucent highlight that SWEEPS to the
active region while ONE caption RE-ANCHORS — then it dissolves to matched footage.
It is NOT a vertical box-and-arrows tier stack (that is the SaaS-slide / PowerPoint
language every premium reference avoids — see EXISTING_PRIMITIVE_VISUAL_REAUDIT.md).

This REPLACES the earlier self-invented `system_architecture_flow` (a tier stack).
Original Vidlore design — Wendover's PRINCIPLE only (persistent plan + sweep +
re-anchor + path token), rebuilt on the charcoal `look` system with our palette,
typography, easing and grain. No asset/layout/colour copied. Pure-local,
deterministic, footage-first, no paid API.

    render("x.mp4", title="THE REQUEST PATH",
           regions=[{"label":"Client","note":"the browser"},
                    {"label":"Edge / CDN"},{"label":"API"},{"label":"Database"}])
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look

SPEC = {
    "id": "system_planview_flow", "family": "systems",
    "roles": ["system", "architecture", "plan_view", "pipeline", "flow",
              "data_path", "request_path", "stages", "layers"],
    "niches_ok": ["tech", "business"],
    "intensity_range": [2, 4], "duration_range": [5.5, 8.0],
    "easing": "easeInOutCubic", "audio_cue": "sweep_soft",
    "repeat_cooldown_s": 60, "per_video_cap": 1, "cost": "low",
    "layout_variants": ["planview"],
    "review_override": ["title", "regions", "palette"],
    "fallback": "bullet_list for a simple ordered list of steps",
    "required_inputs": ["regions"],
    "grounded_in": "wendover/05_deck_schematic (05:36-05:48)",
}


def _norm(regions):
    out = []
    for r in (regions or []):
        if isinstance(r, str) and r.strip():
            out.append({"label": r.strip(), "note": ""})
        elif isinstance(r, dict):
            lbl = str(r.get("label") or r.get("name") or "").strip()
            if lbl:
                out.append({"label": lbl, "note": str(r.get("note") or r.get("metric") or "").strip()})
    return out[:5]


def render(out_path, *, title: str = "", regions=None, dur: float = 6.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    items = _norm(regions)
    if len(items) < 2:
        items = [{"label": "Input", "note": ""}, {"label": "System", "note": ""},
                 {"label": "Output", "note": ""}]
    k = len(items)
    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"]); muted = tuple(pal["muted"])
    accent_mid = tuple(pal["accent"])

    # charcoal textured plan-view stage (NOT a slide bg)
    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    gd = ImageDraw.Draw(bed, "RGBA")
    for gx in range(0, w, int(h * 0.05)):                      # faint plan graticule
        gd.line([(gx, 0), (gx, h)], fill=(*muted, 12), width=1)
    for gy in range(0, h, int(h * 0.05)):
        gd.line([(0, gy), (w, gy)], fill=(*muted, 12), width=1)

    # the persistent "system body": one elongated horizontal plan shape, divided
    # into spatial ZONES along its length (the real-object plan-view, not tiers).
    bx0, bx1 = int(w * 0.10), int(w * 0.90)
    bcy = int(h * 0.52); bhh = int(h * 0.15)                    # body half-height
    by0, by1 = bcy - bhh, bcy + bhh
    zone_w = (bx1 - bx0) / k
    zone_cx = [int(bx0 + (i + 0.5) * zone_w) for i in range(k)]
    spine_y = bcy

    title_font = look.font("title", int(h * 0.038))
    zone_font = look.font("label", int(h * 0.026))
    note_font = look.font("label", int(h * 0.020))
    cap_font = look.font("title", int(h * 0.034))

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=8)
            d.rectangle([bx0, int(h * 0.16), bx0 + int(ti.width * 0.5), int(h * 0.165)],
                        fill=(*accent, int(200 * ta)))
            frame = look.paste_center(frame, ti, cx=bx0 + ti.width // 2,
                                      cy=int(h * 0.20), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # body outline draws in (0.3-1.0s) — the persistent plan
        ba = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.3) / 0.7)))
        if ba > 0.01:
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=int(bhh * 0.5),
                                outline=(*muted, int(200 * ba)), width=2)
            # zone separators + faint zone fills
            for zi in range(1, k):
                zx = int(bx0 + zi * zone_w)
                d.line([(zx, by0 + 6), (zx, by1 - 6)], fill=(*muted, int(110 * ba)), width=1)

        # which zone is "active": a highlight sweeps zone->zone (1.2 -> end-0.8)
        sweep_t0, sweep_t1 = 1.2, max(1.6, dur - 1.2)
        sp = (t - sweep_t0) / max(0.1, (sweep_t1 - sweep_t0))
        active = -1
        if 0.0 <= sp <= 1.0:
            active = min(k - 1, int(sp * k))
        elif sp > 1.0:
            active = k - 1

        # zones: labels reveal (staggered), active zone gets translucent accent fill
        for zi, it in enumerate(items):
            za = look.ease_out_cubic(min(1.0, max(0.0, (t - (0.7 + zi * 0.18)) / 0.4)))
            if za <= 0.01:
                continue
            zx0 = int(bx0 + zi * zone_w); zx1 = int(bx0 + (zi + 1) * zone_w)
            if zi == active:                                    # sweeping highlight
                d.rounded_rectangle([zx0 + 4, by0 + 4, zx1 - 4, by1 - 4], radius=8,
                                    fill=(*accent, 46))
            col = ink if zi == active else muted
            zl = look.text_with_glow(it["label"], zone_font, fill=col, glow=pal["bg_b"],
                                     glow_radius=3, glow_alpha=0.0, pad=4)
            if zl.width > zone_w - 16:
                zl = zl.resize((int(zone_w - 16), int(zl.height * (zone_w - 16) / zl.width)), Image.LANCZOS)
            frame = look.paste_center(frame, zl, cx=zone_cx[zi], cy=by0 - int(h * 0.04), opacity=za)
            d = ImageDraw.Draw(frame, "RGBA")

        # flow path along the spine + travelling token (draws with the sweep)
        if sp > 0.0:
            reach = max(0.0, min(1.0, sp))
            hx = int(bx0 + (bx1 - bx0) * reach)
            # dashed drawn path up to the token
            x = bx0
            while x < hx:
                d.line([(x, spine_y), (min(x + 16, hx), spine_y)], fill=(*accent_mid, 200), width=3)
                x += 26
            # token
            d.ellipse([hx - 7, spine_y - 7, hx + 7, spine_y + 7], fill=(*accent, 240))
            d.ellipse([hx - 13, spine_y - 13, hx + 13, spine_y + 13],
                      outline=(*accent, 110), width=2)

        # ONE caption re-anchors below the active zone (the Wendover move)
        if active >= 0:
            cap = items[active]["label"].upper()
            note = items[active]["note"]
            ci = look.text_with_glow(cap, cap_font, fill=accent, glow=pal["bg_b"],
                                     glow_radius=5, glow_alpha=0.0, pad=6)
            cx = max(int(w * 0.18), min(int(w * 0.82), zone_cx[active]))
            frame = look.paste_center(frame, ci, cx=cx, cy=by1 + int(h * 0.08), opacity=1.0)
            d = ImageDraw.Draw(frame, "RGBA")
            if note:
                ni = look.text_with_glow(note, note_font, fill=muted, glow=pal["bg_b"],
                                         glow_radius=2, glow_alpha=0.0, pad=4)
                frame = look.paste_center(frame, ni, cx=cx, cy=by1 + int(h * 0.13), opacity=0.9)
                d = ImageDraw.Draw(frame, "RGBA")
            # a tick mark from active zone down to the caption
            d.line([(zone_cx[active], by1 + 4), (zone_cx[active], by1 + int(h * 0.05))],
                   fill=(*accent, 180), width=2)

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
            "err": (r.stderr[-200:] if not ok else "")}
