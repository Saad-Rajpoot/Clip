"""Primitive: supply_chain_network  (Section C · business).

Grounded in a REAL frame audit — MagnatesMedia "J.P. Morgan: The Man Who Owned
America" 04:02 (`reference_videos/frame_sequences/magnatesmedia/
01_moneyflow_profiteering/`, frames 4m2s–4m15s): a DIRECTIONAL FLOW is drawn as
ONE connector spine running left→right across a dark textured stage; real things
(a wooden crate, paper-card icons, a photo) DOCK as NODES centred ON that single
line; ONE accent (an orange streak) is the flow itself and travels through the
nodes; the stage reveals PROGRESSIVELY — one node is featured at a time, with a
small label tag beneath it ("Government" → "Consortium"). It is a process you read
from one end to the other, NOT a proportion split (that is sankey) and NOT a
box-and-arrows dashboard (the SaaS-slide language every premium reference avoids).

This is an ORIGINAL Vidlore composition: MagnatesMedia's PRINCIPLE only — one
flow line · footage docked as nodes · one accent for flow · progressive lighting ·
restraint — rebuilt on the charcoal `look` system with OUR palette (amber_gold /
cold_steel), typography, easing and grain. No asset/layout/colour/texture copied
(no torn paper, no asphalt, no orange-on-black). Pure-local, deterministic,
footage-first, no paid API.

A directional supply / value chain: raw → make → move → sell, as 4–5 stage nodes
on ONE horizontal flow line. An amber pulse travels L→R; each node LIGHTS as the
pulse passes (progressive, one at a time); if a stage has a photo it is graded
into the node tile; a readable label sits on a scrim beneath each lit node.

    render("x.mp4", palette_name="amber_gold",
           stages=[{"label":"Raw oil"}, {"label":"Refinery"},
                   {"label":"Pipeline"}, {"label":"Market"}])
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
    "id": "supply_chain_network", "family": "business",
    "roles": ["supply_chain", "value_chain", "pipeline", "process", "flow",
              "stages", "production_chain", "directional_process",
              "raw_to_market", "logistics"],
    "niches_ok": ["business", "tech", "geopolitics"],
    "intensity_range": [2, 4], "duration_range": [5.5, 8.0],
    "easing": "easeInOutCubic", "audio_cue": "flow_soft",
    "repeat_cooldown_s": 90, "per_video_cap": 1, "cost": "low",
    "full_screen": True,
    "layout_variants": ["flowline"],
    "review_override": ["title", "stages", "palette"],
    "fallback": "bullet_list for a plain ordered list without docked media; "
                "sankey_flow for a PROPORTION split (not a directional process)",
    "required_inputs": ["stages"],
    "grounded_in": "magnatesmedia/01_moneyflow_profiteering (04:02)",
    "anti": "NOT sankey ribbons (proportion), NOT a box-and-arrows dashboard, "
            "NOT all stages lit at once — one connector spine, progressive.",
}


def _norm(stages):
    out = []
    for s in (stages or []):
        if isinstance(s, str) and s.strip():
            out.append({"label": s.strip(), "note": "", "photo": None})
        elif isinstance(s, dict):
            lbl = str(s.get("label") or s.get("name") or "").strip()
            if lbl:
                out.append({
                    "label": lbl,
                    "note": str(s.get("note") or s.get("metric") or "").strip(),
                    "photo": s.get("photo") or s.get("image") or s.get("media"),
                })
    return out[:5]


def _load_photo(path):
    try:
        p = Path(str(path))
        if p.exists():
            return Image.open(p).convert("RGB")
    except Exception:                                          # noqa: BLE001
        pass
    return None


def _node_tile(stage, side, pal, *, lit: float):
    """An RGB node tile (square) for the flow line. If the stage has a photo it is
    graded + face-safe-cropped into the tile; otherwise a charcoal placeholder with
    a stage-order glyph. `lit` in [0,1] drives a muted→graded reveal so an
    un-reached node sits darker than the featured one (progressive)."""
    accent = tuple(pal["accent"]); muted = tuple(pal["muted"])
    bg_b = tuple(pal["bg_b"]); ink = tuple(pal["text"])
    img = _load_photo(stage.get("photo"))
    if img is not None:
        if hasattr(look, "face_safe_crop"):
            tile = look.face_safe_crop(img, side, side)
        else:                                                  # cover-crop fallback
            W, H = img.size
            s = max(side / W, side / H)
            rw, rh = max(side, int(W * s)), max(side, int(H * s))
            im2 = img.resize((rw, rh), Image.LANCZOS)
            tile = im2.crop(((rw - side) // 2, int((rh - side) * 0.12),
                             (rw - side) // 2 + side, int((rh - side) * 0.12) + side))
        tile = look.grade_media(tile, pal, strength=0.6)
    else:                                                      # placeholder glyph
        bed = look.graded_background(side, side, pal, seed=side) \
            if hasattr(look, "graded_background") else Image.new("RGB", (side, side), bg_b)
        d = ImageDraw.Draw(bed, "RGBA")
        # a simple "package / stage" mark: stacked bar packets centred
        cx, cy = side // 2, side // 2
        bw, bh = int(side * 0.30), int(side * 0.12)
        for r in range(3):
            yb = cy - bh - 4 + r * (bh + 6)
            col = accent if r == side % 3 else muted          # one packet warm
            d.rounded_rectangle([cx - bw, yb, cx + bw, yb + bh], radius=4,
                                outline=(*col, 210), width=2)
        bed = bed.convert("RGB")
        tile = bed
    # darken an un-reached node so the lit one clearly leads the eye (progressive)
    import numpy as np
    dim = 0.40 + 0.60 * look.clamp01(lit)
    arr = np.asarray(tile).astype(np.float32) * dim
    tile = Image.fromarray(arr.astype("uint8"), "RGB")
    # gentle bottom scrim so a label below the tile stays readable over any photo
    sc = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sc)
    for yy in range(int(side * 0.62), side):
        a = int(150 * (yy - side * 0.62) / (side * 0.38))
        sd.line([(0, yy), (side, yy)], fill=(*bg_b, a))
    tile = Image.alpha_composite(tile.convert("RGBA"), sc).convert("RGB")
    return tile


def render(out_path, *, title: str = "", stages=None, dur: float = 6.5,
           fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", layout: str = "", seed: int = 0,
           crf: int = 18) -> dict:
    import numpy as np
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    items = _norm(stages)
    if len(items) < 2:
        items = [{"label": "Raw", "note": "", "photo": None},
                 {"label": "Make", "note": "", "photo": None},
                 {"label": "Move", "note": "", "photo": None},
                 {"label": "Sell", "note": "", "photo": None}]
    k = len(items)
    accent = tuple(pal["accent"]); accent_hi = tuple(pal["accent_hi"])
    ink = tuple(pal["text"]); muted = tuple(pal["muted"]); glow = tuple(pal["glow"])

    floor = getattr(look, "CARD_STAGE_FLOOR", 52.0)
    bed = look.graded_background(w, h, pal, seed=seed, floor=floor) \
        if hasattr(look, "graded_background") else Image.new("RGB", (w, h), tuple(pal["bg_b"]))
    gd = ImageDraw.Draw(bed, "RGBA")
    # one faint baseline graticule along the flow (subtle, not a slide grid)
    spine_y = int(h * 0.55)
    for gx in range(0, w, int(h * 0.06)):
        gd.line([(gx, spine_y - 1), (gx + int(h * 0.03), spine_y - 1)],
                fill=(*muted, 9), width=1)

    # node geometry: evenly spaced tiles centred ON the single spine line
    side = int(h * 0.205)                                       # node tile edge
    margin = int(w * 0.095)
    span = w - 2 * margin
    if k > 1:
        node_cx = [int(margin + i * span / (k - 1)) for i in range(k)]
    else:
        node_cx = [w // 2]
    sx0, sx1 = node_cx[0], node_cx[-1]                          # spine extent

    # pre-render the node tiles at two states (muted / lit) once — cheap reuse
    tiles_lit = [_node_tile(it, side, pal, lit=1.0) for it in items]
    tiles_dim = [_node_tile(it, side, pal, lit=0.0) for it in items]

    title_font = look.font("title", int(h * 0.040))
    label_font = look.font("label", int(h * 0.026))
    note_font = look.font("label", int(h * 0.0195))

    # timeline: spine draws (0.3–1.0s); the flow pulse sweeps the rail and lights
    # each node in turn from t_flow0 to t_flow1; each node "snaps" lit as the
    # pulse front crosses its x (progressive — one at a time).
    t_flow0, t_flow1 = 1.05, max(1.6, dur - 1.25)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()
        d = ImageDraw.Draw(frame, "RGBA")

        # title (condensed, top-left, with an accent underline tick)
        ta = look.ease_out_cubic(min(1.0, t / 0.5))
        if title and ta > 0.01:
            ti = look.text_with_glow(title.upper(), title_font, fill=ink,
                                     glow=pal["bg_b"], glow_radius=5,
                                     glow_alpha=0.0, pad=8)
            frame = look.paste_center(frame, ti, cx=margin + ti.width // 2,
                                      cy=int(h * 0.135), opacity=ta)
            d = ImageDraw.Draw(frame, "RGBA")
            d.rectangle([margin, int(h * 0.175),
                         margin + int((ti.width - 16) * ta), int(h * 0.179)],
                        fill=(*accent, int(220 * ta)))

        # the persistent flow SPINE draws in L→R (0.3–1.0s) — ONE connector line
        ba = look.ease_out_cubic(min(1.0, max(0.0, (t - 0.3) / 0.7)))
        if ba > 0.01:
            xr = int(sx0 + (sx1 - sx0) * ba)
            d.line([(sx0, spine_y), (xr, spine_y)], fill=(*muted, 150), width=3)
            # directional chevrons between node slots (faint, reinforce L→R)
            for ci in range(k - 1):
                mx = (node_cx[ci] + node_cx[ci + 1]) // 2
                if mx <= xr:
                    ch = int(h * 0.012)
                    d.line([(mx - ch, spine_y - ch), (mx + ch, spine_y)],
                           fill=(*muted, 130), width=2)
                    d.line([(mx + ch, spine_y), (mx - ch, spine_y + ch)],
                           fill=(*muted, 130), width=2)

        # flow progress 0..1 along the spine; the pulse FRONT x lights nodes
        fp = look.clamp01((t - t_flow0) / max(0.1, (t_flow1 - t_flow0)))
        front_x = sx0 + (sx1 - sx0) * fp
        # amber FILL of the spine up to the front (one accent = the flow itself)
        if fp > 0.0 and ba > 0.99:
            d.line([(sx0, spine_y), (int(front_x), spine_y)],
                   fill=(*accent, 230), width=4)
            # a brighter travelling pulse head
            d.ellipse([int(front_x) - 8, spine_y - 8,
                       int(front_x) + 8, spine_y + 8], fill=(*accent_hi, 245))
            d.ellipse([int(front_x) - 16, spine_y - 16,
                       int(front_x) + 16, spine_y + 16],
                      outline=(*accent_hi, 110), width=2)

        # nodes: each docks ON the spine; reveals (staggered) then LIGHTS when the
        # pulse front crosses it (progressive). A lit node = un-dimmed tile + accent
        # ring + label scrim tag below.
        for ni in range(k):
            ncx = node_cx[ni]
            # appear slightly before the pulse can reach it
            appear = look.ease_out_cubic(
                min(1.0, max(0.0, (t - (0.55 + ni * 0.12)) / 0.4)))
            if appear <= 0.01:
                continue
            # lit factor: 0 until the front reaches this node x, then eases to 1
            reached = front_x >= ncx - side * 0.10
            lit_t = 0.0
            if reached:
                # how long since reached → smooth pop-in of the light
                # approximate by distance the front has travelled past the node
                over = (front_x - ncx) / max(1.0, side * 0.9)
                lit_t = look.ease_out_cubic(look.clamp01(0.15 + over))
            ny = spine_y                                        # tile centred ON spine
            tile = tiles_lit[ni] if lit_t > 0.45 else tiles_dim[ni]
            # blend dim→lit for a smooth light-up
            if 0.0 < lit_t <= 0.45:
                a = np.asarray(tiles_dim[ni]).astype(np.float32)
                b = np.asarray(tiles_lit[ni]).astype(np.float32)
                tile = Image.fromarray(
                    (a + (b - a) * (lit_t / 0.45)).astype("uint8"), "RGB")
            # paste_center applies opacity on the alpha channel → need RGBA
            frame = look.paste_center(frame, tile.convert("RGBA"), cx=ncx, cy=ny,
                                      opacity=appear)
            d = ImageDraw.Draw(frame, "RGBA")
            # node frame ring: muted when un-lit, accent when lit
            ring = tuple(int(m + (a - m) * lit_t)
                         for m, a in zip(muted, accent_hi))
            rw = 2 + int(2 * lit_t)
            d.rounded_rectangle(
                [ncx - side // 2, ny - side // 2, ncx + side // 2, ny + side // 2],
                radius=int(side * 0.07), outline=(*ring, int(235 * appear)),
                width=rw)
            if lit_t > 0.25:                                    # soft accent halo
                d.rounded_rectangle(
                    [ncx - side // 2 - 4, ny - side // 2 - 4,
                     ncx + side // 2 + 4, ny + side // 2 + 4],
                    radius=int(side * 0.08), outline=(*accent, int(70 * lit_t)),
                    width=2)

            # label tag on a charcoal scrim beneath the node (reveals with light)
            lbl_a = appear if lit_t < 0.05 else 1.0
            col = ink if lit_t > 0.4 else muted
            ll = look.text_with_glow(items[ni]["label"], label_font, fill=col,
                                     glow=pal["bg_b"], glow_radius=3,
                                     glow_alpha=0.0, pad=5)
            maxw = int(side * 1.18)
            if ll.width > maxw:
                ll = ll.resize((maxw, int(ll.height * maxw / ll.width)),
                               Image.LANCZOS)
            tag_cy = ny + side // 2 + int(h * 0.052)
            # scrim plate behind the label
            pad_x, pad_y = int(h * 0.014), int(h * 0.010)
            plate = [ncx - ll.width // 2 - pad_x, tag_cy - ll.height // 2 - pad_y,
                     ncx + ll.width // 2 + pad_x, tag_cy + ll.height // 2 + pad_y]
            d.rounded_rectangle(plate, radius=int(h * 0.010),
                                fill=(*pal["bg_b"], int(180 * lbl_a)))
            if lit_t > 0.4:                                     # accent left-edge tick
                d.rectangle([plate[0], plate[1], plate[0] + 4, plate[3]],
                            fill=(*accent, 220))
            frame = look.paste_center(frame, ll, cx=ncx, cy=tag_cy, opacity=lbl_a)
            d = ImageDraw.Draw(frame, "RGBA")
            # optional note under the label (only on the currently-featured node)
            note = items[ni]["note"]
            featured = reached and (ni == k - 1 or front_x < node_cx[ni + 1] - side * 0.10)
            if note and featured and lit_t > 0.5:
                ni_img = look.text_with_glow(note, note_font, fill=muted,
                                             glow=pal["bg_b"], glow_radius=2,
                                             glow_alpha=0.0, pad=4)
                if ni_img.width > maxw:
                    ni_img = ni_img.resize(
                        (maxw, int(ni_img.height * maxw / ni_img.width)),
                        Image.LANCZOS)
                frame = look.paste_center(frame, ni_img, cx=ncx,
                                          cy=tag_cy + int(h * 0.036), opacity=0.92)
                d = ImageDraw.Draw(frame, "RGBA")

        frame = look.vignette(frame, strength=0.62)
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
            "stages": k, "err": (r.stderr[-200:] if not ok else "")}
