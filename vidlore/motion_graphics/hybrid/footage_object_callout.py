"""Primitive: footage_object_callout  (Section C · hybrid).

Grounded in TWO real frame audits:

  • Wendover "Cities at Sea (How Aircraft Carriers Work)" 04:08
    (`reference_videos/frame_sequences/wendover/02_map_route_singapore/`) — a busy,
    crowded LIVE documentary shot (a mail-sort floor packed with parcels, pallets,
    people) where ONE real object has to be made the subject. Wendover's move when a
    real thing in a real shot is the point: a QUIET pointer lands on it — never a
    heavy HUD box that buries the footage.
  • ColdFusion "Apple's AI Disaster - A Rare Failure" 04:42
    (`reference_videos/frame_sequences/coldfusion/seq4_product_grid/`) — the hero
    (an iPhone held in-hand) sits in front of a deliberately LOW-CONTRAST, near-white
    blown-out backdrop (the rig behind it desaturated to a whisper). The lesson:
    keep everything around the hero quiet so the SUBJECT stays the hero. A callout
    graphic must obey the same restraint — whisper-quiet, low-contrast, never loud.

PRINCIPLE (only): point at a REAL object IN the live shot. A low-contrast animated
ring (or a subtle corner box) scales in and LOCKS onto one normalised point, a thin
leader line runs to a small label chip parked in nearby NEGATIVE SPACE (so it never
covers the object), the label reads on a soft scrim — then it dissolves. The graphic
stays whisper-quiet so the subject stays the hero. ~3s default, holds, fades out.

This is a HYBRID OVERLAY. It accepts an optional `bg_image` (a footage frame); if
absent it renders over a neutral graded bed (charcoal, lifted off black) to simulate
the live shot. The callout itself is an ORIGINAL Vidlore composition built entirely
on the charcoal `look` palette / typography / easing / grain — NONE of Wendover's or
ColdFusion's colours, fonts, layout, HUD, product-grid or on-screen text is copied.
We take the PRINCIPLE only (quiet pointer onto a real object + leader + label chip in
negative space, then dissolve). Pure-local (PIL + numpy -> ffmpeg), deterministic,
footage-first, no paid API.

    render("x.mp4", label="The cooling vent", point=(0.62, 0.45),
           note="passively drawn air", palette_name="cold_steel")
    render("x.mp4", label="Serial number", point=(0.4, 0.7),
           bg_image="frame.jpg", look="box")
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
    "id": "footage_object_callout", "family": "hybrid",
    "roles": ["callout", "point_out", "highlight_object", "this_one", "locate",
              "annotate", "label_object", "indicate", "spotlight", "circle_this"],
    "niches_ok": ["tech", "science", "business", "history", "geopolitics", "crime"],
    "niches_unsuitable": ["comedy", "lifestyle", "reaction"],
    "intensity_range": [1, 3], "duration_range": [3.0, 5.0],
    "easing": "easeOutCubic", "audio_cue": "locate_soft",
    "repeat_cooldown_s": 45, "per_video_cap": 3, "cost": "low",
    "layout_variants": ["ring", "box"],
    "overlay": True,                      # hybrid: drawn OVER footage/bg or sim bed
    "review_override": ["label", "point", "note", "palette", "look"],
    "fallback": "measurement_callout when the point has a spannable dimension to show",
    "required_inputs": ["label"],
    "grounded_in": "wendover/02_map_route_singapore (04:08) + "
                   "coldfusion/seq4_product_grid (04:42)",
}


def _norm_pt(p, default=(0.5, 0.5)):
    """Normalise the target point to (x, y) in [0.06, 0.94] / [0.08, 0.92] so the
    ring + its label chip always have room on-canvas. Accepts (x, y), [x, y] or a
    {'x':, 'y':} dict; falls back to centre on anything malformed."""
    try:
        if isinstance(p, dict):
            x, y = float(p.get("x", default[0])), float(p.get("y", default[1]))
        else:
            x, y = float(p[0]), float(p[1])
        return min(0.94, max(0.06, x)), min(0.92, max(0.08, y))
    except Exception:                                              # noqa: BLE001
        return default


def _bed(bg_image, w, h, pal, seed):
    """The live shot the callout is drawn over: a graded footage frame if one is
    supplied AND exists, otherwise a neutral graded bed (charcoal, lifted off
    black) to SIMULATE footage. Footage is gently darkened so the quiet pointer +
    label stay legible without the graphic having to get loud."""
    if bg_image is not None:
        try:
            ok = isinstance(bg_image, Image.Image) or Path(str(bg_image)).exists()
            if ok:
                src = bg_image if isinstance(bg_image, Image.Image) \
                    else Image.open(str(bg_image))
                im = look.grade_media(src.convert("RGB"), pal, strength=0.5)
                iw, ih = im.size                       # cover-fit to the canvas
                s = max(w / iw, h / ih)
                im = im.resize((max(w, int(iw * s)), max(h, int(ih * s))),
                               Image.LANCZOS)
                x = (im.width - w) // 2
                y = (im.height - h) // 2
                im = im.crop((x, y, x + w, y + h))
                arr = np.asarray(im).astype(np.float32) * 0.84
                return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        except Exception:                                          # noqa: BLE001
            pass
    # simulated footage bed: charcoal graded background, lifted off black so the
    # frame never reads as black on a dissolve in/out (CARD_STAGE_FLOOR contract).
    return look.graded_background(w, h, pal, seed=seed, drift=0.25,
                                  floor=look.CARD_STAGE_FLOOR)


def _chip_corner(px, py, w, h, chip_w, chip_h, pad):
    """Pick which of the four diagonal corners (relative to the point) the label
    chip should sit in: the one whose chip box fits fully on-canvas AND is most
    into NEGATIVE SPACE (i.e. furthest from the point, so it never covers the
    object). Returns (chip_cx, chip_cy, corner_sign_x, corner_sign_y)."""
    best = None
    for sx in (1, -1):                       # right / left of the point
        for sy in (-1, 1):                   # above / below the point
            # chip centre offset diagonally away from the point
            cx = px + sx * (pad + chip_w / 2)
            cy = py + sy * (pad + chip_h / 2)
            # how far the chip box would spill off the frame (0 = fully inside)
            spill = 0.0
            spill += max(0.0, (chip_w / 2 + 8) - cx)               # left edge
            spill += max(0.0, cx - (w - chip_w / 2 - 8))           # right edge
            spill += max(0.0, (chip_h / 2 + 8) - cy)               # top edge
            spill += max(0.0, cy - (h - chip_h / 2 - 8))           # bottom edge
            # prefer fully-inside corners; among those, the one furthest from
            # the point (deepest into negative space) and biased to upper area.
            dist = math.hypot(cx - px, cy - py)
            score = spill * 1000.0 - dist - (40.0 if sy < 0 else 0.0)
            if best is None or score < best[0]:
                best = (score, cx, cy, sx, sy)
    _, cx, cy, sx, sy = best
    # final clamp so the chip is always fully on-canvas
    cx = max(chip_w / 2 + 8, min(w - chip_w / 2 - 8, cx))
    cy = max(chip_h / 2 + 8, min(h - chip_h / 2 - 8, cy))
    return int(cx), int(cy), sx, sy


def _chip_scrim(w, h, cx, cy, cw, ch, alpha):
    """A soft rounded dark scrim behind the label chip so the text reads over busy
    footage WITHOUT a hard opaque box (whisper-quiet). RGBA, frame-sized."""
    band = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if alpha <= 0.01:
        return band
    d = ImageDraw.Draw(band)
    a = int(120 * look.clamp01(alpha))                # soft, never a solid plate
    d.rounded_rectangle([cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2],
                        radius=int(ch * 0.32), fill=(8, 10, 14, a))
    band = band.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"])
                       .GaussianBlur(3))
    return band


def render(out_path, *, label: str = "", point=None, note: str = "",
           bg_image=None, dur: float = 4.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "cold_steel", look_variant: str = "",
           layout: str = "", seed: int = 0, crf: int = 18, **_ignored) -> dict:
    """`look_variant`/`layout`: 'ring' (default) or 'box' (a subtle corner-bracket
    box instead of a ring). `point`: normalised (x, y) of the object, default
    centre. `note`: optional sub-line under the label."""
    pal = look.palette(palette_name)
    t0 = time.time()
    dur = float(min(5.0, max(3.0, dur)))
    n = max(2, int(round(dur * fps)))

    label_txt = (str(label or "").strip() or "Here").upper()
    note_txt = str(note or "").strip()
    variant = (look_variant or layout or "ring").strip().lower()
    if variant not in ("ring", "box"):
        variant = "ring"

    accent = tuple(pal["accent_hi"]); ink = tuple(pal["text"])
    muted = tuple(pal["muted"]); accent_mid = tuple(pal["accent"])

    bed = _bed(bg_image, w, h, pal, seed)
    # RC4 anchor precision: when no explicit point is supplied, lock onto the
    # frame's SUBJECT via fast spectral-residual saliency computed on the ACTUAL
    # displayed bed (coords then align with what's shown) — instead of the
    # geometric-centre default that landed on film sprocket holes / flat texture.
    # An explicit `point` (e.g. a Review-Editor override) still wins; any failure
    # falls back to centre so a render is never blocked.
    if point is None and bg_image is not None:
        try:
            from ..saliency import salient_point
            point = salient_point(bed)
        except Exception:                                          # noqa: BLE001
            pass
    nx, ny = _norm_pt(point, (0.5, 0.5))
    px, py = int(nx * w), int(ny * h)
    ring_r = int(h * 0.072)                       # locked ring radius (quiet, small)

    lab_font = look.font("label", int(h * 0.030))
    note_font = look.font("label", int(h * 0.022))

    # measure the chip so we can park it in negative space away from the point
    _tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lb = _tmp.textbbox((0, 0), label_txt, font=lab_font)
    lw, lh = lb[2] - lb[0], lb[3] - lb[1]
    nbw = 0
    if note_txt:
        nb = _tmp.textbbox((0, 0), note_txt, font=note_font)
        nbw = nb[2] - nb[0]
    chip_w = int(max(lw, nbw) + h * 0.05)
    chip_h = int(lh + (h * 0.035 if note_txt else 0) + h * 0.040)
    lead_pad = int(ring_r + h * 0.06)             # leader length from ring -> chip
    chip_cx, chip_cy, sgx, sgy = _chip_corner(px, py, w, h, chip_w, chip_h, lead_pad)

    # animation phases (fractions of dur):
    #   lock_t0..lock_t1  ring scales from large -> locks onto the point
    #   lead_t0           leader line draws out toward the chip
    #   chip_t0           label chip fades up
    lock_t0, lock_t1 = 0.15, 0.85
    lead_t0 = 0.75
    chip_t0 = 1.05

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        frame = bed.copy()

        # ── the LOCK: a ring (or corner box) scales in from ~1.9x and settles ──
        lp = look.ease_out_cubic((t - lock_t0) / max(0.1, lock_t1 - lock_t0))
        if lp > 0.001:
            cur_r = ring_r * (1.9 - 0.9 * lp)         # 1.9x -> 1.0x
            la = look.clamp01(lp * 1.2)
            d = ImageDraw.Draw(frame, "RGBA")
            if variant == "box":
                # subtle corner-bracket box (four short L's) — no full heavy box
                rr = int(cur_r)
                arm = max(8, int(rr * 0.45))
                bx0, by0, bx1, by1 = px - rr, py - rr, px + rr, py + rr
                col = (*accent, int(150 * la))
                wdt = 2
                for (cxn, cyn, dx, dy) in [(bx0, by0, 1, 1), (bx1, by0, -1, 1),
                                           (bx0, by1, 1, -1), (bx1, by1, -1, -1)]:
                    d.line([(cxn, cyn), (cxn + dx * arm, cyn)], fill=col, width=wdt)
                    d.line([(cxn, cyn), (cxn, cyn + dy * arm)], fill=col, width=wdt)
            else:
                # a thin low-contrast ring + a soft outer halo (whisper-quiet):
                # present enough to lock with authority, never loud enough to
                # out-shout the subject it points at.
                d.ellipse([px - cur_r, py - cur_r, px + cur_r, py + cur_r],
                          outline=(*accent, int(200 * la)), width=3)
                d.ellipse([px - cur_r - 5, py - cur_r - 5,
                           px + cur_r + 5, py + cur_r + 5],
                          outline=(*accent_mid, int(70 * la)), width=2)
            # tiny centre tick on the object once nearly locked
            if lp > 0.6:
                ta = look.ease_out_cubic((lp - 0.6) / 0.4)
                tk = max(3, int(h * 0.006))
                d.line([(px - tk, py), (px + tk, py)],
                       fill=(*accent, int(150 * ta)), width=2)
                d.line([(px, py - tk), (px, py + tk)],
                       fill=(*accent, int(150 * ta)), width=2)

        # ── the LEADER: a thin line draws from the ring edge toward the chip ──
        leadp = look.ease_out_cubic((t - lead_t0) / 0.45)
        # leader anchor = ring edge in the chip's direction
        ang = math.atan2(chip_cy - py, chip_cx - px)
        ex = px + math.cos(ang) * (ring_r + 3)
        ey = py + math.sin(ang) * (ring_r + 3)
        # leader target = the chip edge nearest the point
        tx = chip_cx - sgx * (chip_w / 2)
        ty = chip_cy - sgy * (chip_h / 2)
        if leadp > 0.001:
            cxp = ex + (tx - ex) * leadp
            cyp = ey + (ty - ey) * leadp
            d = ImageDraw.Draw(frame, "RGBA")
            d.line([(ex, ey), (cxp, cyp)], fill=(*ink, int(195 * leadp)), width=2)
            d.line([(ex, ey + 1), (cxp, cyp + 1)],
                   fill=(*accent_mid, int(80 * leadp)), width=1)
            # small knot where the leader meets the ring (anchors the eye-path
            # ring -> leader -> label so the trace reads cleanly)
            d.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=(*accent, int(225 * leadp)))

        # ── the LABEL CHIP: soft scrim + readable label (+ optional note) ──
        cp = look.ease_out_cubic((t - chip_t0) / 0.5)
        if cp > 0.01:
            frame = Image.alpha_composite(
                frame.convert("RGBA"),
                _chip_scrim(w, h, chip_cx, chip_cy, chip_w, chip_h, cp)).convert("RGB")
            ty_lab = chip_cy - (int(h * 0.016) if note_txt else 0)
            lab = look.text_with_glow(label_txt, lab_font, fill=ink,
                                      glow=pal["bg_b"], glow_radius=4,
                                      glow_alpha=0.0, pad=6)
            if lab.width > w * 0.42:
                sc = (w * 0.42) / lab.width
                lab = lab.resize((int(lab.width * sc), int(lab.height * sc)),
                                 Image.LANCZOS)
            frame = look.paste_center(frame, lab, cx=chip_cx, cy=ty_lab, opacity=cp)
            if note_txt:
                nt = look.text_with_glow(note_txt, note_font, fill=muted,
                                         glow=pal["bg_b"], glow_radius=2,
                                         glow_alpha=0.0, pad=4)
                if nt.width > w * 0.42:
                    sc = (w * 0.42) / nt.width
                    nt = nt.resize((int(nt.width * sc), int(nt.height * sc)),
                                   Image.LANCZOS)
                frame = look.paste_center(frame, nt, cx=chip_cx,
                                          cy=chip_cy + int(h * 0.022),
                                          opacity=cp * 0.95)

        frame = look.vignette(frame, strength=0.5)
        frame = look.film_grain(frame, seed=seed, amount=4.0, t=t)
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
            "variant": variant, "point": [round(nx, 3), round(ny, 3)],
            "err": (r.stderr[-200:] if not ok else "")}
