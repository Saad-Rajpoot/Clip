"""Primitive: territory_advance_arrows.

A kinetic-momentum beat over REAL geography — the theatre is framed to the origin
+ target countries, the attacker's territory is tinted warm and the defender's
cool (so the shared BORDER reads clearly), then bold ADVANCE ARROWS push from the
origin across the border into the target along historically-sensible CORRIDORS
(a thick PRIMARY thrust + lighter SECONDARY thrusts), a starting frontline and an
advancing ending frontline bracket the gains, a soft fill sweeps behind, and an
IMPACT PULSE blooms where the primary arrow arrives. The grammar of "they pushed
from HERE into HERE" — invasions, offensives, expansion.

Real corridors (city→city) are drawn at true positions when supplied/derivable;
otherwise an honest **APPROXIMATE ADVANCE** treatment is used (origin→target
centroid axis) and logged. Distinct from `supply_route_dashes` (thin dashed
logistics) and `parchment_war_map` (a static front): this is the front MOVING.

Forensic principle (NOT an asset copy): premium mapped-conflict storytelling
shows momentum as directional arrows crossing a border between two tinted
territories. Rebuilt with Vidlore's own look system — no reference asset, colour
or layout reproduced. Pure-local, deterministic.

    render("x.mp4", origin_region="Iraq", target_region="Iran",
           event="Iraqi offensive", year="1980", target_subregion="Khuzestan",
           advance_corridors=[["Basra","Ahvaz"], ["Basra","Khorramshahr"]])
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
from . import geo
from .location_establish_card import _antique_map

SPEC = {
    "id": "territory_advance_arrows", "family": "maps",
    "roles": ["advance", "momentum", "offensive", "expansion", "takeover", "push",
              "invasion"],
    "niches_ok": ["history", "geopolitics", "war", "business"],
    "intensity_range": [3, 5], "duration_range": [3.5, 6.5],
    "easing": "easeOutCubic", "audio_cue": "push_low",
    "repeat_cooldown_s": 50, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["advance_right", "advance_left"],
    "review_override": ["origin_region", "target_region", "event", "year",
                        "advance_corridors", "target_subregion", "frontline_start",
                        "frontline_end", "label", "map_image", "palette"],
    "fallback": "parchment_war_map for a static front",
    "required_inputs": [],
}


def _arrow(d, base, tip, col, width, head):
    """Thick line base→tip with a filled triangular head at tip."""
    bx, by = base; tx, ty = tip
    if math.hypot(tx - bx, ty - by) < 2:
        return
    d.line([(bx, by), (tx, ty)], fill=col, width=width)
    ang = math.atan2(ty - by, tx - bx)
    a1 = ang + math.radians(150); a2 = ang - math.radians(150)
    d.polygon([(tx, ty), (tx + head * math.cos(a1), ty + head * math.sin(a1)),
               (tx + head * math.cos(a2), ty + head * math.sin(a2))], fill=col)


def _parse_corridors(corr):
    out = []
    for c in (corr or []):
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append((str(c[0]).strip(), str(c[1]).strip()))
        elif isinstance(c, str) and "," in c:
            a, b = c.split(",")[:2]
            out.append((a.strip(), b.strip()))
    return out


def render(out_path, *, label: str = "", event: str = "", year: str = "",
           origin_region: str = "", target_region: str = "", region: str = "",
           advance_corridors=None, frontline_start: str = "", frontline_end: str = "",
           target_subregion: str = "", count: int = 3, angle: float = 6.0,
           map_image=None, bg_image=None, dur: float = 5.0, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "parchment_sepia",
           layout: str = "", seed: int = 0, reference: bool = False,
           borders: bool = True, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    target = (target_region or region).strip()
    origin = origin_region.strip()
    title = (f"{event.upper()} · {year}" if event and year
             else (event or label or "").strip()).strip()
    count = int(max(1, min(5, count)))
    left = layout == "advance_left"
    sgn = -1.0 if left else 1.0
    notes = []

    col = tuple(pal["accent_hi"]); fill_col = tuple(pal["accent"])
    col_o = tuple(pal["accent"])                       # attacker tint (warm)
    col_t = (96, 116, 132) if palette_name != "ember_red" else (110, 96, 120)
    label_font = look.font("label", int(h * 0.032))

    src = map_image or bg_image
    bed = None
    if src and Path(str(src)).exists():
        try:
            bed = look.grade_media(Image.open(src).convert("RGB").resize((w, h)),
                                   pal, strength=0.5)
        except Exception:                                          # noqa: BLE001
            bed = None

    # ── REAL geography: frame origin + target, tint each, derive/real corridors ──
    geo_mode = False
    o_polys = t_polys = o_pt = t_pt = sub_pt = None
    corridors = []                                     # [(base_pt, tip_pt, primary)]
    o_ent = geo.country_entry(origin) if (bed is None and origin) else None
    t_ent = geo.country_entry(target) if (bed is None and target) else None
    sub_ll = geo.resolve(target_subregion) if (bed is None and target_subregion) else None
    if bed is None and (o_ent or t_ent):
        pts = []
        for e in (o_ent, t_ent):
            if e:
                bw_, bs_, be_, bn_ = e["bbox"]; pts += [(bw_, bs_), (be_, bn_)]
        if sub_ll:
            pts.append(sub_ll)
        bbox = geo.region_bbox(pts, pad_frac=0.18)
        gproj = geo.make_projector(bbox, w, h)
        bed = geo.draw_basemap(w, h, bbox, style=geo.style_for(palette_name),
                               seed=seed, proj=gproj, graticule=True,
                               reference=reference, borders=borders)
        if o_ent:
            o_polys = [[gproj(lo, la) for lo, la in r] for r in o_ent["rings"]]
            o_pt = gproj(*o_ent["centroid"])
        if t_ent:
            t_polys = [[gproj(lo, la) for lo, la in r] for r in t_ent["rings"]]
            t_pt = gproj(*t_ent["centroid"])
        if sub_ll:
            sub_pt = gproj(*sub_ll)
        # real corridors (city→city) at true positions
        for a_pl, b_pl in _parse_corridors(advance_corridors):
            a, b = geo.resolve(a_pl), geo.resolve(b_pl)
            if a and b:
                corridors.append((gproj(*a), gproj(*b), len(corridors) == 0))
        # derive an honest approximate advance if no real corridors
        if not corridors and o_pt and t_pt:
            notes.append("approximate advance (origin→target axis; no corridor data)")
            ang_v = math.atan2(t_pt[1] - o_pt[1], t_pt[0] - o_pt[0])
            px, py = -math.sin(ang_v), math.cos(ang_v)        # perpendicular
            aim = sub_pt or t_pt
            base0 = (o_pt[0] * 0.45 + aim[0] * 0.55 - (aim[0] - o_pt[0]) * 0.30,
                     o_pt[1] * 0.45 + aim[1] * 0.55 - (aim[1] - o_pt[1]) * 0.30)
            spread = 0.16 * math.hypot(w, h)
            for k in range(count):
                off = (k - (count - 1) / 2.0) * spread / max(1, count - 1)
                b = (base0[0] + px * off, base0[1] + py * off)
                tp = (aim[0] + px * off * 0.5, aim[1] + py * off * 0.5)
                corridors.append((b, tp, k == count // 2))
        geo_mode = True
    if bed is None:
        bed = _antique_map(w, h, seed, pal)

    # synthetic fallback corridors (no geography)
    if not corridors:
        notes.append("synthetic advance (no resolvable region)")
        rng = np.random.RandomState((seed * 2246822519) & 0xFFFFFFFF)
        base_x = (0.30 if not left else 0.70) * w
        for i in range(count):
            by = (0.24 + 0.52 * ((i + 0.5) / count)) * h
            length = (0.30 + 0.10 * rng.rand()) * w
            tip = (base_x + sgn * length, by + (rng.rand() - 0.5) * 0.05 * h)
            corridors.append(((base_x, by), tip, i == count // 2))

    # start/end frontline x (perp lines bracket the gains) — mean of bases/tips
    bx_mean = sum(c[0][0] for c in corridors) / len(corridors)
    tx_mean = sum(c[1][0] for c in corridors) / len(corridors)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        z = 1.0 + 0.045 * look.ease_in_out_cubic(pr)
        bw, bh = int(w * z), int(h * z)
        frame = bed.resize((bw, bh), Image.LANCZOS).crop(
            ((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))

        def _zf(p):
            return (w / 2 + (p[0] - w / 2) * z, h / 2 + (p[1] - h / 2) * z)

        # territory tints: attacker (warm) + defender (cool) → border reads clearly
        if geo_mode and (o_polys or t_polys):
            ti_fill = look.ease_out_cubic(min(1.0, t / 0.8))
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            for r in (o_polys or []):
                od.polygon([_zf(p) for p in r], fill=(*col_o, int(60 * ti_fill)))
            for r in (t_polys or []):
                od.polygon([_zf(p) for p in r], fill=(*col_t, int(72 * ti_fill)))
            frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")
            for r in (o_polys or []):
                pp = [_zf(p) for p in r]; d.line(pp + [pp[0]], fill=(*col_o, 210), width=2)
            for r in (t_polys or []):
                pp = [_zf(p) for p in r]; d.line(pp + [pp[0]], fill=(180, 208, 228, 210), width=2)
        d = ImageDraw.Draw(frame, "RGBA")

        # advancing fill between start + end frontlines (grows with the push)
        adv = look.ease_in_out_cubic(min(1.0, max(0.0, (t - 0.3) / 1.7)))
        zbx, ztx = _zf((bx_mean, 0))[0], _zf((tx_mean, 0))[0]
        reach = zbx + (ztx - zbx) * adv
        if adv > 0.02:
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(ov).rectangle([min(zbx, reach), 0, max(zbx, reach), h],
                                         fill=(*fill_col, int(46 * adv)))
            frame = Image.alpha_composite(frame.convert("RGBA"),
                                          ov.filter(ImageFilter.GaussianBlur(20))).convert("RGB")
            d = ImageDraw.Draw(frame, "RGBA")

        def _front(xc, rgb, a, wob=0.013):
            pts = [(int(xc + math.sin(yy / h * 6.0 + seed * 0.7) * (wob * w)), yy)
                   for yy in range(0, h + 1, 22)]
            d.line(pts, fill=(int(rgb[0]), int(rgb[1]), int(rgb[2]), a), width=4, joint="curve")
        if adv > 0.05:
            _front(zbx, pal["muted"], 150)                            # starting front
            ep = 0.5 + 0.5 * math.sin(t * 4.2)
            _front(reach, col, int(180 + 60 * ep), wob=0.016)         # advancing front

        # arrows: primary thick, secondary lighter; staggered; decelerate at tip
        sec = [c for c in corridors if not c[2]]
        for k, (base, tip, primary) in enumerate(corridors):
            t_start = 0.4 + k * 0.28
            gp = look.ease_out_cubic(min(1.0, max(0.0, (t - t_start) / 1.1)))
            if gp <= 0.02:
                continue
            zb, zt = _zf(base), _zf(tip)
            cur = (zb[0] + (zt[0] - zb[0]) * gp, zb[1] + (zt[1] - zb[1]) * gp)
            if primary:
                width = max(8, int(h * 0.015)); head = max(22, int(h * 0.032)); a = 248
            else:
                width = max(4, int(h * 0.008)); head = max(13, int(h * 0.020)); a = 180
            _arrow(d, zb, cur, (*col, a), width, int(head * min(1.0, gp * 1.4)))
            # impact pulse where the PRIMARY arrow arrives
            if primary and gp > 0.9:
                ep = (math.sin(t * 5.0) * 0.5 + 0.5)
                rr = int(14 + 26 * ep)
                d.ellipse([cur[0] - rr, cur[1] - rr, cur[0] + rr, cur[1] + rr],
                          outline=(*col, int(150 * (1 - ep) + 60)), width=3)

        # country + subregion labels (collision-free), then the title chip
        items = []
        lab = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.7) / 0.6)))
        if geo_mode and lab > 0.01:
            if o_ent:
                items.append({"text": origin, "x": _zf(o_pt)[0], "y": _zf(o_pt)[1],
                              "priority": 3, "accent": True, "opacity": lab, "size_frac": 0.032})
            if t_ent:
                items.append({"text": target, "x": _zf(t_pt)[0], "y": _zf(t_pt)[1],
                              "priority": 3, "opacity": lab, "size_frac": 0.032})
            if target_subregion and sub_pt:
                items.append({"text": target_subregion, "x": _zf(sub_pt)[0],
                              "y": _zf(sub_pt)[1], "priority": 2, "opacity": lab,
                              "size_frac": 0.026})
            frame = geo.layout_labels(frame, items, pal)
            d = ImageDraw.Draw(frame, "RGBA")

        if title:
            tti = max(0.0, min(1.0, (t - 0.1) / 0.5))
            chip = look.text_with_glow(title, label_font, fill=pal["text"],
                                       glow=pal["bg_b"], glow_radius=6, glow_alpha=0.0, pad=12)
            cw, ch = chip.width, chip.height
            bar = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(bar).rounded_rectangle(
                [(w - cw) // 2 - 24, int(h * 0.07) - 8, (w + cw) // 2 + 24,
                 int(h * 0.07) + ch + 8], radius=10, fill=(*pal["bg_b"], int(212 * tti)))
            frame = Image.alpha_composite(frame.convert("RGBA"), bar).convert("RGB")
            frame = look.paste_center(frame, chip, cx=w // 2,
                                      cy=int(h * 0.07) + ch // 2, opacity=tti)

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
            "notes": "; ".join(notes), "err": (r.stderr[-200:] if not ok else "")}
