"""Primitive: investigation_location_map.

A "here is where it happened — and here is what was there" beat. On a REAL
geographic bed (Natural Earth backbone, region-focused to the places in play),
1–3 coordinates are pinned with target markers, each tethered by a thin leader
line to a framed PHOTO POP-OUT card (graded still if supplied, else a captioned
location plate). The grammar of an investigative location board — spy / crime /
history / geopolitics.

Real geography when the places resolve; an honest synthetic dark-grid fallback
(logged via the dispatch map_mode) otherwise. Distinct from `map_badge_node`
(a single actor badge) and `location_establish_card` (one establishing pin): this
binds MULTIPLE real coordinates to evidence imagery.

Forensic principle (NOT an asset copy): premium docs anchor evidence stills to a
real map with leader-lined pop-outs. Rebuilt from that PRINCIPLE with Vidlore's
own look + geo system. Pure-local, deterministic, no paid API.

    render("x.mp4", title="THE NETWORK",
           pins=[{"place":"Vienna","label":"Safe house","photo":"a.jpg"},
                 {"place":"Berlin","label":"Dead drop"}])
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

from .. import look
from ..maps import geo

SPEC = {
    "id": "investigation_location_map", "family": "investigation",
    "roles": ["location_map", "evidence_map", "where", "sites", "pop_outs",
              "scene_map", "network_map"],
    "niches_ok": ["crime", "spy", "history", "geopolitics", "war"],
    "intensity_range": [2, 4], "duration_range": [5.0, 8.0],
    "easing": "easeOutCubic", "audio_cue": "map_soft",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["geo", "synthetic"],
    "review_override": ["title", "pins", "style", "palette", "reference"],
    "fallback": "location_establish_card if only one place is known",
    "required_inputs": ["pins"],
    "map_place_keys": ["pins"],
}

_SLOTS = [(0.82, 0.27), (0.82, 0.66), (0.18, 0.72)]      # card anchors in side margins


def _norm_pins(pins, place, label):
    out = []
    if pins:
        for p in pins:
            if isinstance(p, str):
                out.append({"place": p, "label": "", "photo": None})
            elif isinstance(p, dict):
                out.append({"place": str(p.get("place") or p.get("name") or "").strip(),
                            "label": str(p.get("label") or p.get("caption") or "").strip(),
                            "photo": p.get("photo")})
    elif place:
        out.append({"place": str(place).strip(), "label": str(label or "").strip(),
                    "photo": None})
    return [p for p in out if p["place"]][:3]


def _popout(photo, label, cw, ch, pal, name):
    """A framed evidence pop-out (graded still or captioned plate)."""
    card = Image.new("RGB", (cw, ch), tuple(pal["bg_a"]))
    inner_h = int(ch * (0.72 if label else 0.92))
    iw, ih = cw - 16, inner_h - 12
    used_photo = False
    if photo and Path(str(photo)).exists():
        try:
            im = Image.open(str(photo)).convert("RGB")
            im = look.face_safe_crop(im, iw, ih) if hasattr(look, "face_safe_crop") \
                else im.resize((iw, ih), Image.LANCZOS)
            card.paste(look.grade_media(im, pal, strength=0.5).convert("RGB"), (8, 8))
            used_photo = True
        except Exception:                                          # noqa: BLE001
            used_photo = False
    if not used_photo:
        sub = Image.new("RGB", (iw, ih), tuple(pal["bg_b"]))
        sd = ImageDraw.Draw(sub)
        for y in range(ih):
            tt = y / ih
            sd.line([(0, y), (iw, y)],
                    fill=tuple(int(c * (0.7 + 0.5 * tt)) for c in pal["bg_a"]))
        gf = look.font("numeral", int(ih * 0.5))
        ini = "".join(wd[0] for wd in (name or "•").split()[:2]).upper() or "•"
        gi = look.text_with_glow(ini, gf, fill=pal["muted"], glow=pal["bg_b"],
                                 glow_radius=3, glow_alpha=0.0, pad=4)
        sub.paste(gi.convert("RGB"), ((iw - gi.width) // 2, (ih - gi.height) // 2), gi)
        card.paste(sub, (8, 8))
    cd = ImageDraw.Draw(card)
    cd.rectangle([0, 0, cw - 1, ch - 1], outline=tuple(pal["accent_hi"]), width=2)
    cd.rectangle([8, 8, 8 + iw - 1, 8 + ih - 1], outline=tuple(pal["bg_b"]), width=1)
    if label:
        lf = look.font("label", int(ch * 0.10))
        li = look.text_with_glow(label.upper(), lf, fill=pal["text"], glow=pal["bg_b"],
                                 glow_radius=3, glow_alpha=0.0, pad=4)
        if li.width > cw - 14:
            li = li.resize((cw - 14, int(li.height * (cw - 14) / li.width)), Image.LANCZOS)
        card.paste(li.convert("RGB"),
                   ((cw - li.width) // 2, inner_h + (ch - inner_h - li.height) // 2), li)
    return card


def _synthetic_bed(w, h, pal, seed):
    bed = look.graded_background(w, h, pal, seed=seed) if hasattr(look, "graded_background") \
        else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    d = ImageDraw.Draw(bed, "RGBA")
    step = int(h * 0.075)
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(*pal["muted"], 22), width=1)
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(*pal["muted"], 22), width=1)
    return bed


def render(out_path, *, title: str = "", pins=None, place: str = "", label: str = "",
           style: str = "", reference=False, dur: float = 6.5, fps: int = 30,
           w: int = 1920, h: int = 1080, palette_name: str = "cold_steel",
           layout: str = "", seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    items = _norm_pins(pins, place, label)

    # real-geo bed when places resolve, else synthetic grid
    proj = None; hits = 0
    bed = None
    if items and geo.available():
        try:
            names = [p["place"] for p in items]
            bed, proj, hits = geo.basemap_for(
                names, w, h, palette_name=palette_name,
                style=(style or ("tactical" if palette_name == "cold_steel" else "")),
                seed=seed, highlight=names[0] if names else None,
                graticule=True, pad_frac=0.55, min_hits=1,
                borders=True, reference=bool(reference))
        except Exception:                                          # noqa: BLE001
            bed, proj, hits = None, None, 0
    if bed is None or proj is None:
        bed = _synthetic_bed(w, h, pal, seed)
        # synthetic anchored node positions for unresolved places
        sx = [int(w * fx) for fx in (0.40, 0.55, 0.46)]
        sy = [int(h * fy) for fy in (0.42, 0.55, 0.66)]
        pts = [(sx[k % 3], sy[k % 3]) for k in range(len(items))]
    else:
        pts = []
        for k, p in enumerate(items):
            c = geo.resolve(p["place"])
            if c:
                px, py = proj(c[0], c[1])
                pts.append((int(px), int(py)))
            else:
                pts.append((int(w * (0.4 + 0.1 * k)), int(h * (0.45 + 0.08 * k))))

    # pre-render pop-out cards
    cw, ch = int(w * 0.20), int(h * 0.28)
    cards = [_popout(p["photo"], p["label"], cw, ch, pal, p["place"]) for p in items]
    # assign card slots by pin geography to minimise leader crossings:
    # leftmost pin -> left slot, the rest -> right slots ordered top->bottom by py.
    anchors = [None] * len(items)
    if items:
        order = sorted(range(len(pts)), key=lambda k: pts[k][0])
        left_slot = (int(w * 0.18), int(h * 0.72))
        right_slots = [(int(w * 0.82), int(h * 0.27)), (int(w * 0.82), int(h * 0.66))]
        if len(order) == 1:
            anchors[order[0]] = (int(w * 0.80), int(h * 0.30))
        else:
            anchors[order[0]] = left_slot
            rest = sorted(order[1:], key=lambda k: pts[k][1])
            for j, k in enumerate(rest[:2]):
                anchors[k] = right_slots[j]
            for k in rest[2:]:                       # >3 guarded elsewhere, safe default
                anchors[k] = right_slots[-1]

    title_font = look.font("title", int(h * 0.040))
    accent = tuple(pal["accent_hi"]); muted = tuple(pal["muted"])

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # title chip top-left
        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=pal["text"],
                                     glow=pal["bg_b"], glow_radius=5, glow_alpha=0.0, pad=8)
            d.rectangle([int(w * 0.05), int(h * 0.085),
                         int(w * 0.05) + int(ti.width * 0.5), int(h * 0.09)],
                        fill=(*accent, int(200 * ta)))
            frame = look.paste_center(frame, ti, cx=int(w * 0.05) + ti.width // 2,
                                      cy=int(h * 0.13), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")

        # each pin: target marker drops, leader draws, card fades
        for k, (px, py) in enumerate(pts):
            start = 0.5 + k * 0.55
            pa = look.ease_out_cubic(min(1.0, max(0.0, (t - start) / 0.4)))
            if pa <= 0.01:
                continue
            ax, ay = anchors[k]
            # leader meets the card on its facing edge (not buried under centre)
            conn_x = ax - cw // 2 if px < ax else ax + cw // 2
            conn_y = ay
            # leader line pin -> card edge
            la = look.ease_out_cubic(min(1.0, max(0.0, (t - (start + 0.15)) / 0.35)))
            if la > 0.01:
                ex = int(px + (conn_x - px) * la); ey = int(py + (conn_y - py) * la)
                d.line([(px, py), (ex, ey)], fill=(*accent, int(180 * pa)), width=2)
                if la > 0.95:                         # connector node at the card edge
                    d.ellipse([conn_x - 3, conn_y - 3, conn_x + 3, conn_y + 3],
                              fill=(*accent, int(220 * pa)))
            # target pin marker (concentric + pulse)
            pulse = 0.5 + 0.5 * math.sin(t * 3.0 + k)
            r0 = int(h * 0.012)
            d.ellipse([px - r0, py - r0, px + r0, py + r0], outline=(*accent, int(230 * pa)), width=2)
            rr = int(r0 * (1.6 + pulse * 0.9))
            d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=(*accent, int(70 * pa * (1 - pulse * 0.5))), width=2)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(*accent, int(240 * pa)))
            # card
            ca = look.ease_out_cubic(min(1.0, max(0.0, (t - (start + 0.3)) / 0.4)))
            if ca > 0.01:
                frame = look.paste_center(frame, cards[k].convert("RGBA"),
                                          cx=ax, cy=ay, opacity=ca)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.55)
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
            "geo_hits": hits, "err": (r.stderr[-200:] if not ok else "")}
