"""The shared 'cinematic composite' look layer.

This is the premium treatment EVERY motion-graphics primitive wears — the thing
that separates a flat template card from a cinematic documentary graphic. Pure
PIL + numpy (no paid API, deterministic, Mac/Win portable).

Forensic basis (MagnatesMedia): warm amber/sepia/gold grade · crushed blacks ·
textured background (never a flat fill) · framed subject with drop shadow ·
vignette on everything · gold bloom on key numerals · slow 2.5-D parallax ·
refined serif/condensed typography. See research/forensic_refs/
magnatesmedia_motion_graphics/MAGNATESMEDIA_FORENSIC_REPORT.md.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

# ───────────────────────── palettes ─────────────────────────
# Each palette is the per-video colour identity. The director picks one (seeded
# + niche-biased) so two videos don't look identical. RGB tuples.
PALETTES = {
    "amber_gold": {       # the MagnatesMedia-family default (money / business)
        "bg_a": (26, 18, 10), "bg_b": (6, 4, 3),
        "accent": (214, 158, 74), "accent_hi": (247, 206, 122),
        "glow": (255, 176, 64), "text": (240, 232, 218), "muted": (150, 132, 104),
    },
    "ember_red": {        # crime / power / conspiracy
        "bg_a": (30, 12, 10), "bg_b": (8, 4, 4),
        "accent": (196, 84, 60), "accent_hi": (240, 130, 96),
        "glow": (240, 96, 64), "text": (238, 226, 220), "muted": (150, 110, 100),
    },
    "cold_steel": {       # geopolitics / tech / intelligence
        "bg_a": (14, 20, 28), "bg_b": (4, 6, 9),
        "accent": (120, 168, 200), "accent_hi": (176, 212, 236),
        "glow": (120, 184, 232), "text": (224, 232, 240), "muted": (110, 130, 148),
    },
    "parchment_sepia": {  # history / archival
        "bg_a": (38, 28, 16), "bg_b": (12, 9, 6),
        "accent": (198, 160, 96), "accent_hi": (232, 200, 142),
        "glow": (236, 196, 120), "text": (236, 224, 198), "muted": (158, 138, 104),
    },
}


def palette(name: str = "amber_gold") -> dict:
    return PALETTES.get(name, PALETTES["amber_gold"])


# Luminance floor for a DARK stat-card stage (gold_number_callout & kin). The
# palettes above are near-black by design (bg_a luma ~16-30, bg_b ~5-10) — a
# beautiful gold-on-black still, but a sparse-bright card (one hero numeral) is
# ~98% sub-25-luma, so a dissolve/cut LANDING on it reads as a black frame to
# `blackdetect`/signalstats (the 2:18 "96%" dip). Raising the graded-background
# floor to a deep CHARCOAL (not near-black) keeps the cinematic dark identity —
# the gold still pops — while guaranteeing the stage never reads as black. Tuned
# (forensic) so the lifted stage clears blackdetect (pix_th 25.5) with margin and
# YAVG never dips below ~25 at any transition.
CARD_STAGE_FLOOR = 52.0


def _luma(c) -> float:
    """Rec.601 luma of an RGB triple (matches ffmpeg signalstats/blackdetect)."""
    return 0.299 * float(c[0]) + 0.587 * float(c[1]) + 0.114 * float(c[2])


def _lift_color(color, target_luma: float):
    """Return `color` (RGB float array) brightened — hue preserved — so its luma
    is at least `target_luma`. A color already brighter is returned unchanged, so
    this only ever LIFTS a too-dark point (never darkens a palette)."""
    c = np.asarray(color, np.float32)
    lum = _luma(c)
    if lum >= target_luma:
        return c
    if lum <= 1.0:                                   # ~pure black → neutral floor
        return np.full(3, float(target_luma), np.float32)
    return np.clip(c * (target_luma / lum), 0.0, 255.0)


# ───────────────────────── easing ─────────────────────────
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_expo(t: float) -> float:
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - pow(2, -10 * t)


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - pow(1 - t, 3)


# ───────────────────────── fonts ─────────────────────────
# Prefer a bundled premium serif (added for Win portability); fall back to a
# system serif (great on Mac); finally the bundled brand sans (always present).
_SERIF_CANDIDATES = [
    _ASSETS / "MGSerif.ttf", _ASSETS / "PlayfairDisplay-Bold.ttf",
    Path("/System/Library/Fonts/Supplemental/Didot.ttc"),
    Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Baskerville.ttc"),
    _ASSETS / "VidloreSans-Bold.ttf",
]
_COND_CANDIDATES = [
    _ASSETS / "MGCondensed.ttf",
    Path("/System/Library/Fonts/Avenir Next Condensed.ttc"),
    _ASSETS / "VidloreSans-Bold.ttf", _ASSETS / "VidloreSans.ttf",
]


def _first_font(cands, size: int) -> ImageFont.FreeTypeFont:
    for p in cands:
        try:
            if Path(p).exists():
                return ImageFont.truetype(str(p), size)
        except Exception:                                       # noqa: BLE001
            continue
    return ImageFont.load_default()


def font(role: str, size: int, *, weight: int | None = None) -> ImageFont.FreeTypeFont:
    """role: 'numeral'|'title' -> serif; 'label'|'caption' -> condensed.
    For variable fonts (e.g. bundled Playfair Display) set a heavy weight so big
    numerals read bold; harmless on static fonts (the call is caught)."""
    if role in ("label", "caption"):
        f = _first_font(_COND_CANDIDATES, size)
        w = weight or 700
    else:
        f = _first_font(_SERIF_CANDIDATES, size)
        w = weight or (820 if role == "numeral" else 720)
    try:
        f.set_variation_by_axes([w])
    except Exception:                                          # noqa: BLE001
        pass
    return f


# ───────────────────────── background ─────────────────────────
def graded_background(w: int, h: int, pal: dict, *, seed: int = 0,
                      drift: float = 0.0, floor: float = 0.0) -> Image.Image:
    """Warm radial vignette-gradient base with a faint paper texture and a slow
    parallax drift. `drift` in [0,1] shifts the hot-spot for 2.5-D motion.

    `floor` (luma 0-255, default 0 = legacy/byte-identical) lifts the gradient's
    hot-spot to at least `floor` and its dark edge proportionally, turning a
    near-black stage into a deep charcoal one WITHOUT flattening it (the radial
    gradient + warm hue are preserved). Used by dark sparse-bright stat cards so a
    dissolve/cut never lands on a frame that reads as black. See CARD_STAGE_FLOOR."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * (0.5 + 0.06 * math.sin(drift * math.pi))
    cy = h * (0.46 + 0.04 * math.cos(drift * math.pi))
    r = np.sqrt(((xx - cx) / (w * 0.72)) ** 2 + ((yy - cy) / (h * 0.72)) ** 2)
    r = np.clip(r, 0, 1)
    a = np.array(pal["bg_a"], np.float32)
    b = np.array(pal["bg_b"], np.float32)
    if floor > 0:
        a = _lift_color(a, floor)            # centre hot-spot ≥ floor luma
        b = _lift_color(b, floor * 0.72)     # edge lifted too, gradient preserved
    base = (a[None, None] * (1 - r[..., None]) + b[None, None] * r[..., None])
    # faint procedural paper grain (seeded, static per video)
    rng = np.random.default_rng(seed)
    tex = rng.normal(0, 5.0, (h // 2, w // 2, 1)).astype(np.float32)
    tex = np.asarray(Image.fromarray(
        np.clip(tex + 128, 0, 255).astype(np.uint8)[..., 0]).resize(
        (w, h))).astype(np.float32)[..., None] - 128
    base = np.clip(base + tex * 0.25, 0, 255)
    return Image.fromarray(base.astype(np.uint8), "RGB")


_FACE_CASCADE = None


def _detect_primary_face(img: Image.Image):
    """Return (x, y, w, h) of the largest face, or None. Grayscale-tolerant
    (runs on luma, so B&W / sepia historical photos work). Tries the frontal
    cascade then the alt + profile cascades."""
    global _FACE_CASCADE
    try:
        import cv2
        g = np.asarray(img.convert("L"))
        names = ["haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"]
        best = None
        for nm in names:
            cas = cv2.CascadeClassifier(cv2.data.haarcascades + nm)
            if cas.empty():
                continue
            mn = max(40, min(img.size) // 12)
            faces = cas.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(mn, mn))
            for f in faces:
                if best is None or f[2] * f[3] > best[2] * best[3]:
                    best = tuple(int(v) for v in f)
            if best is not None:
                break
        return best
    except Exception:                                          # noqa: BLE001
        return None


def face_safe_crop(img: Image.Image, out_w: int, out_h: int, *,
                   face_top_frac: float = 0.34) -> Image.Image:
    """Crop `img` to exactly out_w×out_h keeping the WHOLE head of the primary
    face — top margin above the hair, plus shoulder/upper-body context below.

    Face-aware (cv2 Haar on luma → works on colour, B&W and sepia). If no face
    is found, falls back to a TOP-BIASED cover crop (keeps the head, never the
    bottom-centre torso). Never returns a head-clipped frame for a detectable
    face. Returns an RGB image.
    """
    im = img.convert("RGB")
    W, H = im.size
    tgt = out_w / out_h
    box = _detect_primary_face(im)
    if box:
        fx, fy, fw, fh = box
        fcx = fx + fw / 2.0
        # A face box is ~brow→chin; the head extends ~0.75·fh above it (hair)
        # and the torso ~2.6·fh below. Build a crop tall enough for the full
        # head + shoulders, with the face sitting `face_top_frac` down it.
        crop_h = fh * 4.4
        crop_w = crop_h * tgt
        # if the source isn't tall enough, fit to height and shrink the crop
        if crop_h > H:
            crop_h = float(H)
            crop_w = crop_h * tgt
        if crop_w > W:                       # not wide enough → fit width
            crop_w = float(W)
            crop_h = crop_w / tgt
        crop_top = fy - face_top_frac * crop_h
        crop_left = fcx - crop_w / 2.0
        # clamp inside the image (prefer keeping the HEAD: bias the clamp so the
        # top is never below the hairline when possible)
        crop_top = max(0.0, min(crop_top, H - crop_h))
        crop_left = max(0.0, min(crop_left, W - crop_w))
        c = im.crop((int(round(crop_left)), int(round(crop_top)),
                     int(round(crop_left + crop_w)),
                     int(round(crop_top + crop_h))))
    else:
        s = max(out_w / W, out_h / H)
        rw, rh = max(out_w, int(W * s)), max(out_h, int(H * s))
        im2 = im.resize((rw, rh), Image.LANCZOS)
        x = (rw - out_w) // 2
        y = int((rh - out_h) * 0.12)         # bias UP → keep the head, drop legs
        c = im2.crop((x, y, x + out_w, y + out_h))
    if c.size != (out_w, out_h):
        c = c.resize((out_w, out_h), Image.LANCZOS)
    return c


def portrait_blurfill_canvas(img: Image.Image, W: int, H: int, *,
                             port_aspect: float = 0.80) -> Image.Image:
    """Premium portrait-on-canvas: a face-safe FULL-HEAD portrait (port_aspect,
    fit to the canvas height) centred over a blurred, darkened fill of the same
    image — so a tall portrait never has to be head-cropped into a wide canvas
    (the fix for `_cover_to_canvas` clipping foreheads). Returns RGB W×H."""
    im = img.convert("RGB")
    ph = H
    pw = int(round(ph * port_aspect))
    port = face_safe_crop(im, pw, ph)
    bg = im.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(40))
    bg = Image.fromarray((np.asarray(bg).astype(np.float32) * 0.45)
                         .astype(np.uint8), "RGB")
    bg.paste(port, ((W - pw) // 2, (H - ph) // 2))
    return bg


def grade_media(img: Image.Image, pal: dict, *, strength: float = 0.5) -> Image.Image:
    """Warm-grade an arbitrary image/footage frame toward the palette: crush
    blacks, lift warm midtones, gentle desaturate."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    g = arr.mean(axis=2, keepdims=True)
    arr = arr * (1 - 0.25 * strength) + g * (0.25 * strength)       # desat
    warm = np.array([1.0 + 0.10 * strength, 1.0, 1.0 - 0.12 * strength])
    arr = np.clip(arr * warm[None, None], 0, 1)
    arr = np.clip((arr - 0.04 * strength) / (1 - 0.04 * strength), 0, 1)  # crush
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


# ───────────────────────── overlays ─────────────────────────
def vignette(img: Image.Image, *, strength: float = 0.55) -> Image.Image:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - strength * np.clip(r - 0.55, 0, 1) / 0.45, 0, 1)
    arr = np.asarray(img).astype(np.float32) * m[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def film_grain(img: Image.Image, *, seed: int = 0, amount: float = 6.0,
               t: float = 0.0) -> Image.Image:
    w, h = img.size
    rng = np.random.default_rng(seed + int(t * 1000))
    g = rng.normal(0, amount, (h, w, 1)).astype(np.float32)
    arr = np.clip(np.asarray(img).astype(np.float32) + g, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


# ──────────────── RC5.1: central text-readability adoption ────────────────
# The MG primitives draw essentially ALL their card text through the two shared
# helpers below (`text_with_glow`, used directly + via `gold_fill`, is the single
# most-called text helper in the whole `motion_graphics/` tree — ~80+ call sites).
# RC5.1 routes *those* helpers through the proven readability gate in
# `motion_graphics/_shared.py` (WCAG contrast + heavier-sans body font + restrained
# scrim) so every primitive that uses them benefits CENTRALLY, with no per-primitive
# patching and — critically — NO visual change to cards that are already readable.
#
# CONSERVATIVE CONTRACT (so we never shift the ~100 already-good cards):
#   * The gate only ever engages when a caller HANDS US the background the text will
#     sit on (`bg=` on text_with_glow / `bg_under=` on paste_center). Every existing
#     call site passes neither → the byte-identical legacy path runs untouched. A
#     primitive opts a specific card in by supplying the bed; it is never forced.
#   * Even when a bed IS supplied, if contrast already PASSES (body 4.5:1, large
#     3:1) we draw EXACTLY as before — same glyph, same fill, same glow, no stroke,
#     no scrim. Repairs (recolour → heavier weight → subtle stroke on the glyph
#     layer; restrained scrim on the bed at paste time) fire ONLY on a fail.
#   * Everything is wrapped: if `_shared` can't import or a helper raises, we fall
#     straight back to the existing drawing — a readability assist must never crash
#     or alter a render.

_SHARED = None
_SHARED_TRIED = False


def _shared_mod():
    """Lazily + defensively import the sibling readability gate. Returns the module
    or None; cached. A failure here just means callers get the legacy path."""
    global _SHARED, _SHARED_TRIED
    if _SHARED_TRIED:
        return _SHARED
    _SHARED_TRIED = True
    try:                                                        # noqa: SIM105
        from . import _shared as _s
        _SHARED = _s
    except Exception:                                           # noqa: BLE001
        _SHARED = None
    return _SHARED


def _readability_plan(text: str, fnt: "ImageFont.FreeTypeFont", fill, bg, *,
                      large: bool, body: bool):
    """Run the shared contrast gate for one text run against a known background.

    `bg` is either an RGB triple (the mean bed colour under the text) or an
    ``(img, box)`` tuple to sample. Returns ``(fill, font, plan)`` where ``fill`` /
    ``font`` may be repaired (recolour / heavier-sans body swap) and ``plan`` is the
    :class:`_shared.ContrastPlan` (or ``None`` if the gate was unavailable / passed
    untouched). NEVER raises — on any problem it returns the inputs unchanged so the
    caller draws exactly as before."""
    S = _shared_mod()
    if S is None or bg is None:
        return fill, fnt, None
    try:
        # Resolve the background to a mean RGB the gate can score.
        img = None
        if (isinstance(bg, (tuple, list)) and len(bg) == 2
                and isinstance(bg[0], Image.Image)):
            img, box = bg[0], bg[1]
            bg_rgb = S.box_bg_stats(img, box)["mean_rgb"]
            sampler = S.box_bg_sampler(img)
        else:
            bg_rgb = tuple(float(v) for v in bg)
            sampler = lambda _box, _c=bg_rgb: _c       # noqa: E731
            box = (0, 0, 1, 1)

        in_fill = tuple(int(v) for v in fill)
        plan = S.ensure_contrast(None, box, in_fill, sampler,
                                 large=large, img=img,
                                 allow_scrim=False)      # scrim is a bed op, not glyph
        # If the gate proposed NO change to the glyph (same colour, no weight bump,
        # no stroke), the card was already readable → draw it byte-identically. This
        # is what keeps the ~100 already-good cards untouched.
        if (tuple(int(v) for v in plan.text_color) == in_fill
                and not plan.weight_boost and not plan.stroke_w and not body):
            return fill, fnt, None

        out_fill = tuple(int(v) for v in plan.text_color)
        out_font = fnt
        # Body/supporting copy that came in on a thin serif and still struggles:
        # swap to the readable heavier-sans at a matching size (the RC5 fix). Only
        # for `body=True` runs — headings keep their premium serif.
        if body:
            try:
                size = int(getattr(fnt, "size", 0) or 0)
                if size > 0:
                    out_font = font("caption", size,
                                    weight=max(620, 600 + int(plan.weight_boost)))
            except Exception:                                   # noqa: BLE001
                out_font = fnt
        return out_fill, out_font, plan
    except Exception:                                           # noqa: BLE001
        return fill, fnt, None


def text_with_glow(text: str, fnt: ImageFont.FreeTypeFont, *,
                   fill=(240, 232, 218), glow=(255, 176, 64),
                   glow_radius: int = 18, glow_alpha: float = 1.0,
                   pad: int = 60, bg=None, large: bool = False,
                   body: bool = False) -> Image.Image:
    """RGBA layer: text with a soft gold bloom behind it (gaussian blur).

    RC5.1 readability assist (opt-in, off by default → byte-identical legacy path):
    pass ``bg`` — either the mean RGB of the bed the text will land on, or an
    ``(img, box)`` to sample — and the run is routed through the shared contrast
    gate. If contrast already passes it draws exactly as before; if it FAILS the
    glyph is repaired in-place (recolour toward max contrast, a heavier sans for
    ``body=True`` runs, and a thin halo stroke as a last resort) before rasterising.
    ``large=True`` selects the 3:1 large-text target (titles / hero numerals).
    Any failure in the gate silently falls back to the unmodified drawing."""
    # RC5.1: resolve any contrast repair BEFORE measuring/rasterising (so a heavier
    # body font is measured at its real width). No-op unless `bg` was supplied.
    plan = None
    if bg is not None:
        fill, fnt, plan = _readability_plan(text, fnt, fill, bg,
                                            large=large, body=body)
    stroke_w = int(getattr(plan, "stroke_w", 0) or 0) if plan is not None else 0
    stroke_fill = getattr(plan, "stroke_color", None) if plan is not None else None

    tmp = Image.new("RGBA", (16, 16))
    d = ImageDraw.Draw(tmp)
    box = d.textbbox((0, 0), text, font=fnt, stroke_width=stroke_w)
    tw, thh = box[2] - box[0], box[3] - box[1]
    W, H = tw + pad * 2, thh + pad * 2
    ox, oy = pad - box[0], pad - box[1]
    glow_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow_img)
    gd.text((ox, oy), text, font=fnt, fill=(*glow, int(235 * clamp01(glow_alpha))))
    glow_img = glow_img.filter(ImageFilter.GaussianBlur(glow_radius))
    # intensify the bloom
    ga = np.asarray(glow_img).astype(np.float32)
    ga[..., 3] = np.clip(ga[..., 3] * 1.8, 0, 255)
    glow_img = Image.fromarray(ga.astype(np.uint8), "RGBA")
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt)
    if stroke_w and stroke_fill is not None:
        td.text((ox, oy), text, font=fnt, fill=(*fill, 255),
                stroke_width=stroke_w, stroke_fill=(*tuple(stroke_fill), 255))
    else:
        td.text((ox, oy), text, font=fnt, fill=(*fill, 255))
    out = Image.alpha_composite(glow_img, txt)
    return out


def gold_fill(text: str, fnt: ImageFont.FreeTypeFont, pal: dict, **kw) -> Image.Image:
    """Convenience: palette-gold numeral with bloom."""
    return text_with_glow(text, fnt, fill=pal["accent_hi"], glow=pal["glow"], **kw)


def paste_center(base: Image.Image, layer: Image.Image, *, cx: float, cy: float,
                 scale: float = 1.0, opacity: float = 1.0,
                 scrim_for=None, large: bool = False) -> Image.Image:
    """Composite ``layer`` centred on ``base``.

    RC5.1 readability assist (opt-in, off by default → byte-identical legacy path):
    pass ``scrim_for`` = the text colour carried by ``layer`` and, BEFORE compositing,
    the bed where the layer lands is contrast-tested. ONLY if it fails (or is busy)
    does a restrained, feathered, low-opacity scrim get laid under the text — the one
    repair that genuinely needs the destination image (a transparent glyph layer can't
    carry it). A passing/calm bed is left untouched. Never raises — any gate failure
    falls back to a plain composite."""
    if scale != 1.0:
        nw = max(1, int(layer.width * scale))
        nh = max(1, int(layer.height * scale))
        layer = layer.resize((nw, nh), Image.LANCZOS)
    if opacity < 1.0:
        a = np.asarray(layer).astype(np.float32)
        a[..., 3] *= clamp01(opacity)
        layer = Image.fromarray(a.astype(np.uint8), "RGBA")
    x = int(cx - layer.width / 2)
    y = int(cy - layer.height / 2)
    # RC5.1: restrained, contrast-gated scrim under the text — only on a fail.
    if scrim_for is not None:
        try:
            base = _scrim_under_layer(base, layer, (x, y), scrim_for, large=large)
        except Exception:                                       # noqa: BLE001
            pass
    base = base.convert("RGBA")
    base.alpha_composite(layer, (x, y))
    return base.convert("RGB")


def _scrim_under_layer(base: Image.Image, layer: Image.Image, xy, text_color, *,
                       large: bool) -> Image.Image:
    """Lay the shared restrained scrim under ``layer``'s opaque-text bounds on
    ``base`` IFF contrast against the real bed fails or the bed is busy. Returns the
    (possibly) plated base. No-op + no raise when the gate is unavailable or passes."""
    S = _shared_mod()
    if S is None:
        return base
    bw, bh = base.size
    lx, ly = int(xy[0]), int(xy[1])
    # Tighten the box to the layer's actually-marked pixels (so the scrim hugs the
    # text, not the transparent glow padding), clamped to the canvas.
    bbox = layer.split()[-1].getbbox() if layer.mode == "RGBA" else None
    if bbox:
        x0, y0, x1, y1 = (lx + bbox[0], ly + bbox[1], lx + bbox[2], ly + bbox[3])
    else:
        x0, y0, x1, y1 = lx, ly, lx + layer.width, ly + layer.height
    box = (max(0, x0), max(0, y0), min(bw, x1), min(bh, y1))
    if box[2] <= box[0] or box[3] <= box[1]:
        return base
    plan = S.ensure_contrast(None, box, tuple(int(v) for v in text_color),
                             S.box_bg_sampler(base), large=large, img=base,
                             allow_scrim=True, allow_weight=False,
                             allow_stroke=False)
    if plan.scrim is None:
        return base                                     # passing/calm bed → untouched
    return S.apply_plan(base, plan, box)


def hairline(draw: ImageDraw.ImageDraw, cx: int, y: int, half_w: int, pal: dict,
             *, alpha: float = 1.0):
    c = tuple(int(v) for v in pal["muted"])
    draw.line([(cx - half_w, y), (cx + half_w, y)], fill=c,
              width=max(1, 2))


def fade_alpha(t: float, dur: float, fps: int, *, fin: float = 0.25,
               fout: float = 0.35) -> float:
    """Global in/out fade envelope for a clip of `dur` seconds at time t (s)."""
    a = 1.0
    if t < fin:
        a = ease_out_cubic(t / fin)
    if t > dur - fout:
        a = min(a, ease_out_cubic((dur - t) / fout))
    return clamp01(a)


# Deep-charcoal STAGE floor every dark MG card fades toward — NOT black. All MG
# palettes are near-black by design (bg_b luma ~5-10), so the old whole-frame
# `frame * fa` multiply faded the entire card to true black on its in/out
# envelope → a 1-frame black flash on a CUT-in and a near-black trough when a
# dissolve lands on the card (the 2:18 gold_number_callout dip). gold_number_-
# callout was fixed to a foreground-only fade; `fade_frame` brings every other
# dark primitive in line centrally: it fades the composited frame toward this
# charcoal floor instead of black, so a cut/dissolve into the card never dips
# below ~luma 22. Luma-matched to the lifted graded_background stage.
_STAGE_FLOOR = (20, 22, 27)


def fade_frame(frame, fa: float, pal: dict | None = None):
    """Fade a composited dark-card `frame` by `fa` toward the charcoal STAGE
    floor (NOT black). fa=1 → unchanged; fa=0 → deep charcoal. Replaces the old
    `Image.fromarray((np.asarray(frame)*fa)...)` whole-frame multiply so a card
    never flashes/troughs to black on its fade envelope."""
    fa = clamp01(float(fa))
    if fa >= 0.999:
        return frame
    arr = np.asarray(frame).astype(np.float32)
    floor = np.array(_STAGE_FLOOR, dtype=np.float32)
    out = floor + (arr - floor) * fa
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")


# ───────────────────────── temp-frame cleanup (V1.2.1) ─────────────────────
# Every primitive stages a `f%05d.png` sequence in a tempfile.mkdtemp() dir for
# the ffmpeg encode. These helpers guarantee those dirs are removed (so a long /
# repeated render never leaks gigabytes of PNGs to the temp dir) — once per
# render via cleanup_frames(td), and as a safety net for any orphan left by a
# crash via sweep_stale_frames(). Both are SAFE: they only ever touch a
# mkdtemp() frame dir (one holding the `marker` frame), never the output mp4
# (which lives in run_dir/motion_graphics) or any cached asset, and never an
# in-progress render (sweep honours an age cutoff). Neither raises.

_FRAME_MARKER = "f00000.png"


def cleanup_frames(td) -> bool:
    """Remove ONE primitive's temp frame dir after its encode. Returns True if
    a dir was removed. Idempotent; never raises. Refuses to remove anything
    that does not look like a frame dir (must contain the marker frame), so it
    can never delete an output mp4 or an unrelated directory."""
    import shutil
    try:
        p = Path(str(td))
        if not p.is_dir():
            return False
        if not (p / _FRAME_MARKER).exists():
            # Not a frame dir — only delete if it is genuinely empty (a fresh
            # mkdtemp that errored before the first frame), never otherwise.
            try:
                if any(p.iterdir()):
                    return False
            except OSError:
                return False
        shutil.rmtree(p, ignore_errors=True)
        return not p.exists()
    except Exception:                                          # noqa: BLE001
        return False


def sweep_stale_frames(root=None, *, max_age_s: float = 3600.0,
                       marker: str = _FRAME_MARKER) -> dict:
    """SAFE stale-temp cleanup utility. Removes orphaned mkdtemp frame dirs
    (those holding `marker`) older than `max_age_s` under the temp `root`
    (defaults to the system temp dir). The age cutoff means an in-progress
    render — whose frames were written seconds ago — is NEVER deleted; only
    abandoned leftovers from a crashed/old run are. Returns
    {"removed": n, "freed_mb": x}. Never raises."""
    import shutil
    import tempfile
    import time as _t
    removed, freed = 0, 0
    try:
        base = Path(root) if root else Path(tempfile.gettempdir())
        now = _t.time()
        for d in base.glob("tmp*"):
            try:
                if not d.is_dir():
                    continue
                mk = d / marker
                if not mk.exists():
                    continue                       # not a frame dir → skip
                age = now - max(d.stat().st_mtime, mk.stat().st_mtime)
                if age < max_age_s:
                    continue                       # too fresh → could be live
                try:
                    sz = sum(f.stat().st_size for f in d.glob("*.png"))
                except OSError:
                    sz = 0
                shutil.rmtree(d, ignore_errors=True)
                if not d.exists():
                    removed += 1
                    freed += sz
            except Exception:                      # noqa: BLE001
                continue
    except Exception:                              # noqa: BLE001
        pass
    return {"removed": removed, "freed_mb": round(freed / 1e6, 1)}
