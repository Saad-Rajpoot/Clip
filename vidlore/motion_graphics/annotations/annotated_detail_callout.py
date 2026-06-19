"""Primitive: annotated_detail_callout.

An "examine this" beat — an archival photo with ONE detail spotlit: the area
around a focus point stays bright while the rest of the frame is gently crushed,
a bright ring is drawn around the detail, a leader line runs out to a label chip
that names what you're looking at, and the frame pushes slowly toward the detail.

Distinct from `framed_evidence_spotlight` (frames the WHOLE artifact under a
spotlight) and `cinematic_portrait_hold` (a clean held portrait): this one points
INTO an image and says "look HERE" — the face in the crowd, the ship on the
horizon, the signature on the page.

Forensic principle (NOT asset copy): MagnatesMedia rings a detail in a period
photo and tethers a label to it; the premium is the local spotlight + clean ring
+ leader + restraint, not a clip-art arrow. If no photo resolves, a tasteful aged
archival plate stands in so the beat still lands. Pure-local. No paid API.

    render("x.mp4", image_path="crowd.jpg", label="The only known photograph",
           focus="0.62,0.40", tag="DETAIL", palette_name="parchment_sepia")
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
    "id": "annotated_detail_callout", "family": "annotations",
    "roles": ["annotate", "detail", "highlight", "evidence", "reveal"],
    "niches_ok": ["history", "biography", "crime", "geopolitics", "business", "tech"],
    "intensity_range": [2, 4], "duration_range": [4.0, 7.0],
    "easing": "easeOutCubic", "audio_cue": "soft_focus_pulse",
    "repeat_cooldown_s": 50, "per_video_cap": 2, "cost": "low",
    "layout_variants": ["spotlight_ring"],
    "review_override": ["image_path", "label", "focus", "tag", "palette"],
    "fallback": "aged archival plate with the ring + label if no photo is available",
}


def _parse_focus(focus, default=(0.5, 0.44)):
    try:
        if isinstance(focus, (list, tuple)) and len(focus) >= 2:
            fx, fy = float(focus[0]), float(focus[1])
        else:
            a, b = str(focus).replace(" ", "").split(",")[:2]
            fx, fy = float(a), float(b)
        return min(0.88, max(0.12, fx)), min(0.86, max(0.14, fy))
    except Exception:                                          # noqa: BLE001
        return default


def _cover(img, bw, bh):
    """Crop-to-fill bw×bh (the photo should fill its frame, not letterbox)."""
    iw, ih = img.size
    s = max(bw / iw, bh / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    im = img.resize((nw, nh), Image.LANCZOS)
    return im.crop(((nw - bw) // 2, (nh - bh) // 2,
                    (nw - bw) // 2 + bw, (nh - bh) // 2 + bh))


def _archival_plate(bw, bh, pal, seed):
    """A tasteful aged photographic plate for the no-photo path — sepia/grey
    grain, soft tonal drift, dust, heavy vignette. Reads as 'a photograph we are
    examining' rather than a hollow card."""
    rng = np.random.default_rng(int(seed or 7) & 0xFFFFFFFF)
    base = np.full((bh, bw, 3), np.array([150, 140, 124], np.float32))
    # low-freq tonal drift (suggests a developed image)
    lf = rng.normal(0, 34, (max(2, bh // 30), max(2, bw // 30))).astype(np.float32)
    lf = np.asarray(Image.fromarray(np.clip(128 + lf, 0, 255).astype(np.uint8)
                    ).resize((bw, bh), Image.BILINEAR), np.float32)
    base += (lf[..., None] - 128) * 0.7
    base += rng.normal(0, 9, (bh, bw, 1))                      # film grain
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    return look.grade_media(img, pal, strength=0.6)


def render(out_path, *, image_path=None, label: str = "", focus="0.5,0.44",
           tag: str = "", dur: float = 6.0, fps: int = 30, w: int = 1920,
           h: int = 1080, palette_name: str = "amber_gold", layout: str = "",
           seed: int = 0, crf: int = 18) -> dict:
    pal = look.palette(palette_name)
    t0 = time.time()
    n = max(2, int(round(dur * fps)))
    fx, fy = _parse_focus(focus)

    # the photo (cover-filled into a framed plate)
    pw, ph = int(w * 0.74), int(h * 0.80)
    try:
        if image_path and Path(str(image_path)).exists():
            photo = _cover(look.grade_media(
                Image.open(image_path).convert("RGB"), pal, strength=0.45), pw, ph)
        else:
            photo = _archival_plate(pw, ph, pal, seed)
    except Exception:                                          # noqa: BLE001
        photo = _archival_plate(pw, ph, pal, seed)
    # thin frame around the photo
    plate = Image.new("RGB", (pw + 12, ph + 12),
                      tuple(int(c * 0.9) for c in pal["accent"]))
    plate.paste(photo, (6, 6))
    plate = plate.convert("RGBA")                               # paste_center needs alpha
    pcx, pcy = w * 0.5, h * 0.48                                # plate centre on screen

    cap_font = look.font("title", int(h * 0.040))
    tag_font = look.font("label", int(h * 0.024))
    # detail position relative to plate centre (constant; the push is tiny)
    det_dx = (fx - 0.5) * (pw + 12)
    det_dy = (fy - 0.5) * (ph + 12)
    ring_r = int(min(pw, ph) * 0.16)

    td = Path(tempfile.mkdtemp())
    for i in range(n):
        t = i / fps
        pr = i / max(1, n - 1)
        frame = look.graded_background(w, h, pal, seed=seed, drift=pr)
        push = 1.0 + 0.045 * look.ease_out_cubic(min(1.0, t / (dur)))   # slow push-in
        ent = look.ease_out_cubic(min(1.0, t / 0.6))
        frame = look.paste_center(frame, plate, cx=pcx, cy=pcy, scale=push,
                                  opacity=ent)
        dcx = pcx + det_dx * push
        dcy = pcy + det_dy * push

        # local spotlight — crush everything except a soft disc around the detail
        sp = look.ease_out_cubic(max(0.0, (t - 0.25) / 0.7))
        if sp > 0.01:
            arr = np.asarray(frame).astype(np.float32)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            rr = np.sqrt(((xx - dcx) / (ring_r * 2.4)) ** 2
                         + ((yy - dcy) / (ring_r * 2.4)) ** 2)
            keep = np.clip(1.0 - (rr - 0.6) / 0.9, 0.30, 1.0)      # bright disc → dim
            mask = 1.0 - sp * (1.0 - keep)
            frame = Image.fromarray(
                np.clip(arr * mask[..., None], 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(frame, "RGBA")

        # the ring around the detail (draws/settles in)
        rg = look.ease_out_cubic(max(0.0, (t - 0.35) / 0.55))
        if rg > 0.01:
            r = int(ring_r * (1.18 - 0.18 * rg))                  # settle inward
            al = int(255 * rg)
            for off, a in ((4, int(70 * rg)), (0, al)):
                d.ellipse([dcx - r - off, dcy - r - off, dcx + r + off, dcy + r + off],
                          outline=(*pal["accent_hi"], a if off else al),
                          width=4 if off == 0 else 2)
            # a couple of tick marks on the ring (cardinal), restrained
            for ang in (0, 90, 180, 270):
                import math
                ax, ay = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                d.line([(dcx + ax * r, dcy + ay * r),
                        (dcx + ax * (r + 12), dcy + ay * (r + 12))],
                       fill=(*pal["accent_hi"], al), width=2)

        # leader line from the ring to a label BLOCK in the lower third. The label
        # (big) and the optional tag chip (small) are STACKED with a MEASURED gap so
        # they can never collide — they previously sat at fixed 0.80h / 0.88h and the
        # label's glow + padding bled up into the tag chip (user-reported text
        # overlap). Now the label is built first (so its true height is known), the
        # tag chip is parked a fixed gap ABOVE the label's top edge, and the leader
        # lands just above whichever element is on top. Collision-proof for any font
        # size / label length.
        if label:
            la = look.ease_out_cubic(max(0.0, (t - 0.6) / 0.5))
            if la > 0.01:
                import math
                lx = int(w * 0.5)
                ci = look.text_with_glow(label, cap_font, fill=pal["text"],
                                         glow=pal["bg_b"], glow_radius=5,
                                         glow_alpha=0.0, pad=18)
                if ci.width > w * 0.60:                  # keep the label on-canvas
                    _s = (w * 0.60) / ci.width
                    ci = ci.resize((max(1, int(ci.width * _s)),
                                    max(1, int(ci.height * _s))), Image.LANCZOS)
                ly = int(h * 0.905)                       # label centre (low third)
                lab_top = ly - ci.height // 2
                top_y = lab_top
                # optional tag chip, stacked a clean measured gap ABOVE the label
                if tag:
                    th = int(h * 0.044)                   # chip height
                    gap = int(h * 0.026)                  # gap chip-bottom -> label-top
                    by = lab_top - gap - th               # chip top
                    tw = d.textlength(tag.upper(), font=tag_font)
                    bx = int(lx - tw / 2 - 16)
                    d.rounded_rectangle([bx, by, bx + tw + 32, by + th],
                                        radius=6, fill=(*pal["bg_b"], int(210 * la)),
                                        outline=(*pal["accent"], int(220 * la)), width=2)
                    ti = look.text_with_glow(tag.upper(), tag_font, fill=pal["accent_hi"],
                                             glow=pal["bg_b"], glow_radius=3,
                                             glow_alpha=0.0, pad=8)
                    frame = look.paste_center(frame, ti, cx=lx, cy=by + th // 2,
                                              opacity=la)
                    d = ImageDraw.Draw(frame, "RGBA")
                    top_y = by
                # leader line: ring edge -> just above the text block
                tgt_y = top_y - int(h * 0.02)
                ang = math.atan2(tgt_y - dcy, lx - dcx)
                sxp = dcx + math.cos(ang) * ring_r
                syp = dcy + math.sin(ang) * ring_r
                exp = sxp + (lx - sxp) * la
                eyp = syp + (tgt_y - syp) * la
                d.line([(sxp, syp), (exp, eyp)], fill=(*pal["accent_hi"], int(210 * la)),
                       width=2)
                d.ellipse([sxp - 4, syp - 4, sxp + 4, syp + 4],
                          fill=(*pal["accent_hi"], int(230 * la)))
                frame = look.paste_center(frame, ci, cx=lx, cy=ly, opacity=la)

        frame = look.vignette(frame, strength=0.6)
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
