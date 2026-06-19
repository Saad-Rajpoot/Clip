"""Primitive: portrait_legend_reveal.

A biography COLD-OPEN that frames the subject as a legend on first contact. A
hero portrait (face-safe, slow reverent push-in) sits centred on a graded
charcoal stage; the NAME settles in two-line serif with a soft glow and a single
accent underline; an optional small kicker (rank / era) sits above. Everything
ENTERS and EXITS with eased motion — nothing pops. The stage rides on
look.CARD_STAGE_FLOOR so a cut/dissolve into the open never reads as a dead-black
frame.

Grounded in the Rockefeller cold-open beat (research .../seq01_cold_open_myth/,
audit pattern P1): a giant figure dominating centre with a glowing serif name in
the lower third, held slow and reverent. This is an ORIGINAL Vidlore composition
— a clean charcoal stage with one disciplined warm glow and a single underline —
NOT the reference's burning-refinery skyline, peer-pantheon busts, letter-scramble
glyphs or any of its specific art/branding.

Pure-local (PIL + numpy frames -> ffmpeg encode). No paid API. Deterministic.

    render("legend.mp4", name="John D. Rockefeller",
           kicker="THE FIRST BILLIONAIRE", portrait_path="jdr.jpg",
           palette_name="amber_gold")

Returns a dict: {ok, path, frames, dur_s, render_s, w, h, err}.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .. import look

SPEC = {
    "id": "portrait_legend_reveal", "family": "portraits",
    "roles": ["hook", "intro", "reveal", "legend"],
    # The opening "who this is." A named-legend cold open — distinct from
    # cinematic_portrait_hold (a static lower-third hold with no name reveal /
    # rank framing). Reserved for the very first beat of a character piece.
    "niches_ok": ["biography", "history", "business", "crime", "spy"],
    "intensity_range": [2, 4], "duration_range": [3.5, 6.0],
    "easing": "easeInOutCubic", "audio_cue": "bed_soft_swell",
    "repeat_cooldown_s": 999, "per_video_cap": 1, "cost": "low",
    "required_inputs": ["name"],
    "review_override": ["name", "kicker", "portrait", "palette"],
    "fallback": False,
}


# ───────────────────────── text helpers ─────────────────────────
def _clean(s) -> str:
    """Coerce to a safe single-line string: drop control chars, collapse
    whitespace, normalise unicode. Never raises."""
    try:
        s = "" if s is None else str(s)
    except Exception:                                          # noqa: BLE001
        return ""
    s = unicodedata.normalize("NFC", s)
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] == "C" and ch not in ("\n", "\t", " "):     # control glyphs
            continue
        out.append(" " if ch in ("\n", "\t") else ch)
    return " ".join("".join(out).split()).strip()


def _split_two_lines(name: str) -> list[str]:
    """Split a name into at most two balanced lines for the serif lock-up.
    A single word stays one line; multi-word names break near the middle so the
    surname tends to land on line two (the reference's two-line treatment)."""
    words = name.split()
    if len(words) <= 1:
        return [name] if name else []
    if len(words) == 2:
        return words
    # balance by character length around the midpoint, biased so line two is
    # not longer than line one (keeps the surname grounded at the baseline)
    best_i, best_d = 1, 10 ** 9
    total = len(name)
    acc = 0
    for i in range(1, len(words)):
        acc += len(words[i - 1]) + 1
        d = abs(acc - total / 2)
        if d <= best_d:
            best_d, best_i = d, i
    return [" ".join(words[:best_i]), " ".join(words[best_i:])]


def _fit_font(role: str, text: str, max_w: int, start_px: int,
              min_px: int) -> "object":
    """Largest serif/condensed font (role) whose `text` fits in `max_w`.
    Shrinks from start_px down to min_px. Always returns a usable font."""
    tmp = Image.new("L", (8, 8))
    d = ImageDraw.Draw(tmp)
    px = max(min_px, int(start_px))
    while px > min_px:
        f = look.font(role, px)
        try:
            box = d.textbbox((0, 0), text, font=f)
            if (box[2] - box[0]) <= max_w:
                return f
        except Exception:                                      # noqa: BLE001
            return f
        px -= max(2, int(px * 0.06))
    return look.font(role, min_px)


# ───────────────────────── hero subject ─────────────────────────
def _feathered_portrait(img: Image.Image, pw: int, ph: int,
                        pal: dict) -> Image.Image:
    """Face-safe hero crop, palette-graded, with a soft feathered alpha so the
    figure sits INSIDE the charcoal stage (a vignetted column of light) rather
    than as a hard-edged pasted rectangle. Returns an RGBA tile."""
    crop = look.face_safe_crop(img, pw, ph).convert("RGB")
    crop = look.grade_media(crop, pal, strength=0.5)
    tile = crop.convert("RGBA")
    # radial-ish feather: full opacity in the centre, soft falloff toward the
    # vertical edges + bottom so the subject melts into the stage.
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    nx = (xx - pw / 2) / (pw / 2)
    ny = (yy - ph * 0.42) / (ph * 0.60)
    r = np.sqrt(nx * nx + ny * ny)
    a = np.clip(1.0 - np.clip(r - 0.62, 0, 1) / 0.42, 0.0, 1.0)
    # extra fade on the very bottom rows so feet/shoulders dissolve into stage
    btm = np.clip((yy - ph * 0.80) / (ph * 0.20), 0, 1)
    a = np.clip(a * (1.0 - 0.85 * btm), 0.0, 1.0)
    arr = np.asarray(tile).astype(np.float32)
    arr[..., 3] = a * 255.0
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def _initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _monogram_pedestal(name: str, pw: int, ph: int, pal: dict) -> Image.Image:
    """No-portrait fallback: a graded charcoal medallion bearing the subject's
    monogram (initials) over a soft pedestal of light. Reads as a deliberate
    'silhouette stand-in', never an empty frame. Returns an RGBA tile."""
    tile = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    cx, cy = pw / 2, ph * 0.44
    rad = int(min(pw, ph) * 0.34)
    # soft warm pedestal glow behind the medallion
    glow = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([cx - rad * 1.5, cy - rad * 1.5, cx + rad * 1.5, cy + rad * 1.5],
               fill=(*pal["glow"], 70))
    glow = glow.filter(ImageFilter.GaussianBlur(rad // 2))
    tile.alpha_composite(glow)
    # charcoal medallion disc with a thin accent ring
    disc = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    dd = ImageDraw.Draw(disc)
    fl = int(look.CARD_STAGE_FLOOR)
    dd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
               fill=(fl + 8, fl + 6, fl + 2, 255),
               outline=(*pal["accent"], 235), width=max(2, rad // 28))
    tile.alpha_composite(disc)
    # monogram initials in serif, gold, glowing
    mono = _initials(name)
    mf = _fit_font("title", mono, int(rad * 1.5), int(rad * 1.2), 12)
    mg = look.text_with_glow(mono, mf, fill=pal["accent_hi"], glow=pal["glow"],
                             glow_radius=max(6, rad // 8), glow_alpha=0.85,
                             pad=int(rad * 0.5))
    mx = int(cx - mg.width / 2)
    my = int(cy - mg.height / 2)
    tile.alpha_composite(mg, (mx, my))
    return tile


# ───────────────────────── render ─────────────────────────
def render(out_path, *, name: str, portrait_path=None, kicker: str = "",
           dur: float = 5.0, fps: int = 30, w: int = 1920, h: int = 1080,
           palette_name: str = "amber_gold", seed: int = 0,
           crf: int = 18) -> dict:
    t0 = time.time()
    out_path = str(out_path)
    try:
        pal = look.palette(palette_name)
        name = _clean(name)
        kicker = _clean(kicker)
        if not name:
            return {"ok": False, "path": out_path, "frames": 0,
                    "dur_s": 0.0, "render_s": round(time.time() - t0, 2),
                    "w": w, "h": h, "err": "empty name"}

        dur = max(2.0, float(dur))
        fps = max(1, int(fps))
        n = max(2, int(round(dur * fps)))

        # hero subject tile (portrait if available, else monogram pedestal)
        pw, ph = int(h * 0.50 * 0.80), int(h * 0.62)          # tall ~4:5 hero
        subj = None
        if portrait_path:
            try:
                if Path(str(portrait_path)).exists():
                    src = Image.open(str(portrait_path)).convert("RGB")
                    subj = _feathered_portrait(src, pw, ph, pal)
            except Exception:                                  # noqa: BLE001
                subj = None
        if subj is None:
            subj = _monogram_pedestal(name, pw, ph, pal)

        # name lock-up (two balanced serif lines, fit to a generous safe width)
        lines = _split_two_lines(name)
        max_tw = int(w * 0.82)
        # base size ~ h*0.085 for the name; clamp so it is always >= h*0.05
        name_px = int(h * 0.085)
        min_name_px = max(int(h * 0.052), 14)
        longest = max(lines, key=len) if lines else name
        nm_font = _fit_font("title", longest, max_tw, name_px, min_name_px)
        line_imgs = [
            look.text_with_glow(ln, nm_font, fill=pal["text"], glow=pal["glow"],
                                glow_radius=max(8, int(h * 0.016)),
                                glow_alpha=0.55, pad=int(h * 0.05))
            for ln in lines
        ]
        # measured glyph heights (drive the line spacing + underline placement)
        tmpd = ImageDraw.Draw(Image.new("L", (8, 8)))
        try:
            gb = tmpd.textbbox((0, 0), longest, font=nm_font)
            glyph_h = gb[3] - gb[1]
        except Exception:                                      # noqa: BLE001
            glyph_h = int(h * 0.07)
        line_gap = int(glyph_h * 1.18)

        kick_font = None
        kick_img = None
        if kicker:
            kick_font = _fit_font("label", kicker.upper(), int(w * 0.62),
                                  int(h * 0.030), max(int(h * 0.020), 10))
            kick_img = look.text_with_glow(
                kicker.upper(), kick_font, fill=pal["accent_hi"],
                glow=pal["glow"], glow_radius=5, glow_alpha=0.0,
                pad=int(h * 0.03))

        # layout anchors
        subj_cx, subj_cy = w * 0.5, h * 0.40
        # name block sits in the lower third, never over the face
        name_cy = h * 0.74
        td = Path(tempfile.mkdtemp())

        for i in range(n):
            t = i / fps
            pr = i / max(1, n - 1)

            # graded charcoal stage with a slow parallax drift + dark floor
            frame = look.graded_background(w, h, pal, seed=seed, drift=pr,
                                           floor=look.CARD_STAGE_FLOOR)

            # reverent slow push-in on the whole composite
            z = 1.0 + 0.05 * look.ease_in_out_cubic(pr)

            # foreground in/out envelope (NEVER multiplies the whole frame to
            # black — the lit stage always stays; only overlays fade)
            fa = look.fade_alpha(t, dur, fps, fin=0.42, fout=0.45)

            # subject: eased entrance settle (scale up + rise a touch) + a small
            # eased EXIT drift so it leaves with motion, not a pop.
            ent = look.ease_out_cubic(min(1.0, t / 0.85))
            ext = look.ease_in_out_cubic(
                max(0.0, (t - (dur - 0.55)) / 0.55))           # 0..1 over exit
            srise = (1.0 - ent) * h * 0.035 + ext * h * 0.025  # in from below/out up
            ssc = (0.965 + 0.035 * ent) * z
            frame = look.paste_center(frame, subj, cx=subj_cx,
                                      cy=subj_cy + srise, scale=ssc, opacity=fa)

            # gentle scrim under the lower third so the name always reads
            scrim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            sd = ImageDraw.Draw(scrim)
            sy = int(h * 0.60)
            sd.rectangle([0, sy, w, h], fill=(0, 0, 0, int(120 * fa)))
            scrim = scrim.filter(ImageFilter.GaussianBlur(int(h * 0.05)))
            frame = Image.alpha_composite(frame.convert("RGBA"),
                                          scrim).convert("RGB")

            # kicker (small, above the name) — fades in slightly after the
            # subject settles, fades with the global out-envelope
            ka = look.ease_out_cubic(max(0.0, (t - 0.45) / 0.6)) * fa
            ky = name_cy - line_gap * 0.5 - int(h * 0.072)
            if kick_img is not None and ka > 0.01:
                frame = look.paste_center(frame, kick_img, cx=w * 0.5,
                                          cy=ky, opacity=ka)

            # name: two lines settle in (rise + fade) after the subject lands,
            # leave with the same eased out-envelope. Tiny per-line stagger.
            n_lines = len(line_imgs)
            top_y = name_cy - (line_gap * (n_lines - 1)) / 2.0
            for li, limg in enumerate(line_imgs):
                lead = 0.55 + li * 0.12
                la_in = look.ease_out_cubic(max(0.0, (t - lead) / 0.7))
                la = la_in * fa
                if la <= 0.01:
                    continue
                rise = (1.0 - la_in) * h * 0.028               # settle up into place
                ly = top_y + li * line_gap + rise
                frame = look.paste_center(frame, limg, cx=w * 0.5, cy=ly,
                                          opacity=la)

            # one accent underline beneath the name block — wipes out from
            # centre after the name settles, recedes on exit
            uw = look.ease_out_cubic(max(0.0, (t - (0.55 + (n_lines - 1) * 0.12
                                                    + 0.45)) / 0.6))
            uw = min(uw, 1.0) * fa
            if uw > 0.01:
                uy = int(top_y + (n_lines - 1) * line_gap + glyph_h * 0.72)
                half = int(w * 0.20 * uw)
                ud = ImageDraw.Draw(frame)
                acc = tuple(int(c) for c in pal["accent"])
                ud.line([(w // 2 - half, uy), (w // 2 + half, uy)],
                        fill=acc, width=max(2, int(h * 0.004)))
                # a soft gold node at the centre of the rule
                nd = max(3, int(h * 0.006))
                ud.ellipse([w // 2 - nd, uy - nd, w // 2 + nd, uy + nd],
                           fill=tuple(int(c) for c in pal["accent_hi"]))

            # finish: vignette + grain on the full composite (stage + overlays)
            frame = look.vignette(frame, strength=0.62)
            frame = look.film_grain(frame, seed=seed, amount=5.0, t=t)
            frame.save(td / f"f{i:05d}.png")

        return _encode(td, out_path, fps, crf, n, dur, w, h, t0)
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "path": out_path, "frames": 0, "dur_s": 0.0,
                "render_s": round(time.time() - t0, 2), "w": w, "h": h,
                "err": f"render: {type(e).__name__}: {e}"}


def _encode(td, out_path, fps, crf, n, dur, w, h, t0) -> dict:
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        ff = "ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate",
           str(fps), "-i", f"{td}/f%05d.png", "-c:v", "libx264", "-crf",
           str(crf), "-pix_fmt", "yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags",
           "+faststart", str(out_path)]
    err = ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = (r.returncode == 0 and Path(out_path).exists())
        if not ok:
            err = (r.stderr or "")[-200:]
    except Exception as e:                                     # noqa: BLE001
        ok, err = False, f"encode: {e}"
    look.cleanup_frames(td)                  # never leak PNG frames to /tmp
    return {"ok": ok, "path": str(out_path), "frames": n,
            "dur_s": round(dur, 2), "render_s": round(time.time() - t0, 2),
            "w": w, "h": h, "err": err}
