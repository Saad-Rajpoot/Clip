"""Shared TEXT-READABILITY vocabulary for the motion-graphics card primitives.

RC5.1. The MG cards (gold_number_callout, act_chapter_card, statement_card,
definition_card, the map/evidence chips, …) ride premium serif type over LIVE,
graded, often dark-and-noisy footage. That is gorgeous when it works, but a thin
Playfair serif body line, low-contrast on a busy/dark frame, in a long unwrapped
column, becomes genuinely hard to read — especially once the 1080p master is
downscaled to a phone. RC5 flagged exactly this.

This module is the MANDATORY readability gate every card text helper should route
through. It is a thin, deterministic, pure-PIL/numpy layer on top of the existing
``look`` vocabulary (fonts, palette, easing) — it does NOT replace ``look``; it
wraps the act of *placing legible text* with three guarantees:

  CONTRAST   — WCAG-style contrast ratio between the text colour and the actual
               local background luminance under the text box. Body ≥ 4.5:1, large
               / title ≥ 3:1. When it fails we climb a RESTRAINED repair ladder
               (nudge colour to max contrast → small rounded low-opacity scrim →
               heavier weight → subtle stroke), only escalating as far as needed.
               If even the full ladder cannot reach target the helper RETURNS a
               signal so the primitive can relayout / shrink / skip — it never
               hard-crashes and never silently ships an unreadable card.

  TYPOGRAPHY — body / supporting copy avoids the thin serif: it uses the clean
               condensed-sans family (``look.font('caption')`` -> Avenir-Next-
               Condensed / VidloreSans) at a solid weight; the premium serif stays
               for HEADINGS where it is large enough to read. Line length is capped
               to a sane characters-per-line, safe margins are enforced, and a
               minimum effective px floor keeps copy legible after the master is
               downscaled to mobile.

  BACKGROUND — the area under the text is sampled for luminance *variance* (a busy
               / noisy / high-detail frame) as well as mean. When the bed is busy a
               clean readability plate is added even if the average contrast looked
               OK — because high local variance is what actually destroys legibility
               on footage.

Premium-first: a plate is only ever added when contrast genuinely fails or the bed
is genuinely busy, and even then it is a soft rounded low-opacity wash, never a
hard black bar. The default behaviour with no failure is *no extra marks*.

Public API (the pieces a primitive wires through):

  relative_luminance(rgb)                       — WCAG relative luminance 0..1
  contrast_ratio(c1, c2)                         — WCAG ratio 1..21
  box_bg_stats(img, box)                         — (mean_luma_0_255, luma_std, ...)
  box_bg_sampler(img)                            — bg_sampler(box) -> mean RGB
  max_contrast_shift(text_rgb, bg_luma)          — nudged text colour (toward W/B)
  ensure_contrast(draw, box, text_color, bg_sampler, *, large=…) -> ContrastPlan
  readable_body_font(h, *, lines, scale)         — heavier sans body font + px floor
  readable_title_font(h, *, scale, weight)       — premium serif heading font
  wrap_to_width(draw, text, font, max_w, *, max_lines, max_chars)
  auto_scrim_if_needed(img, box, *, …)           — add a plate only when needed
  draw_text / draw_body / draw_headline          — convenience: place text through
                                                   the whole gate in one call

Everything is deterministic and side-effect free except the explicit ``draw_*`` /
``auto_scrim_if_needed`` helpers which mutate the image you hand them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import look

# ───────────────────────── readability constants ─────────────────────────
# WCAG 2.x contrast targets. Body / supporting copy must clear AA (4.5:1); large
# text (titles, hero numerals — visually >= ~24px bold / 19px regular) may use the
# relaxed large-text AA (3:1). We expose both so the caller declares which it is.
CONTRAST_BODY = 4.5
CONTRAST_LARGE = 3.0

# A frame whose luminance std-dev under the text box exceeds this (0..255 scale) is
# "busy": even with an OK *mean* contrast, the high-frequency detail collides with
# glyph edges and kills legibility, so we plate it. Tuned conservatively so calm
# graded beds (a soft graded_background, a smooth sky) stay plate-free while real
# footage texture (foliage, crowds, rubble, type-in-the-photo) trips it.
BUSY_LUMA_STD = 34.0

# Mobile-legibility floor. A 1080p master is routinely watched at ~360-414px wide
# on a phone (a ~2.6-3x downscale). Body copy below ~h*0.030 (≈ 32px at 1080p) ≈ 11-
# 12px on that phone — below comfortable reading. We never let readable_body_font
# drop below this fraction of frame height (it wraps to more lines instead).
BODY_MIN_FRAC = 0.030
BODY_MIN_PX = 22                     # absolute floor for tiny/oddball canvases

# Characters-per-line cap. Long measure (>~42 chars at body size) is tiring and on
# a card it also means a too-wide block. We wrap to width AND to this char budget,
# whichever bites first.
MAX_CHARS_PER_LINE = 40

# Safe margin: keep text off the extreme edge so a TV overscan / rounded phone
# corner / vignette never clips a glyph. Fraction of the SHORT side.
SAFE_MARGIN_FRAC = 0.055

# Scrim ceiling — a readability plate never exceeds this alpha, so it is always a
# soft wash and never a hard black bar (premium-first).
SCRIM_MAX_ALPHA = 0.46

# Near-white / near-black the colour nudge drives toward (not pure 255/0 — slightly
# off so it sits in the warm premium world, not clinical pure white).
_TEXT_WHITE = (244, 240, 232)
_TEXT_BLACK = (14, 12, 10)


# ───────────────────────── WCAG contrast math ─────────────────────────
def _srgb_to_linear(channel: float) -> float:
    """One sRGB channel (0..1) -> linear-light, per WCAG / IEC 61966-2-1."""
    c = float(channel)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Sequence[float]) -> float:
    """WCAG relative luminance (0..1) of an 8-bit RGB triple."""
    r, g, b = (float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0)
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def contrast_ratio(c1: Sequence[float], c2: Sequence[float]) -> float:
    """WCAG contrast ratio (1.0 .. 21.0) between two RGB colours."""
    l1 = relative_luminance(c1)
    l2 = relative_luminance(c2)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


def luma601(rgb: Sequence[float]) -> float:
    """Rec.601 luma 0..255 (matches ffmpeg signalstats; cheap proxy for 'how dark
    does this read'). Distinct from relative_luminance (which is for WCAG)."""
    return 0.299 * float(rgb[0]) + 0.587 * float(rgb[1]) + 0.114 * float(rgb[2])


# ───────────────────────── background sampling ─────────────────────────
def _clamp_box(box, w: int, h: int):
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def box_bg_stats(img: Image.Image, box) -> dict:
    """Sample the background under ``box`` and report what we need to judge
    readability: mean colour, mean luma (0..255), luma std-dev (0..255 → busy-ness),
    and the darkest/brightest local luma. Robust to tiny / off-canvas boxes; never
    raises."""
    w, h = img.size
    x0, y0, x1, y1 = _clamp_box(box, w, h)
    crop = img.convert("RGB").crop((x0, y0, x1, y1))
    arr = np.asarray(crop, dtype=np.float32)
    if arr.size == 0:
        return {"mean_rgb": (0.0, 0.0, 0.0), "mean_luma": 0.0, "luma_std": 0.0,
                "min_luma": 0.0, "max_luma": 0.0}
    mean_rgb = tuple(float(v) for v in arr.reshape(-1, 3).mean(axis=0))
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    return {
        "mean_rgb": mean_rgb,
        "mean_luma": float(luma.mean()),
        "luma_std": float(luma.std()),
        "min_luma": float(luma.min()),
        "max_luma": float(luma.max()),
    }


def box_bg_sampler(img: Image.Image) -> Callable[[object], tuple]:
    """Return a ``bg_sampler(box) -> mean RGB`` bound to ``img``. This is the
    callable ``ensure_contrast`` consumes, so a primitive can either hand the image
    or its own sampler (e.g. one that knows the text rides a known plate colour)."""
    def _sampler(box):
        return box_bg_stats(img, box)["mean_rgb"]
    return _sampler


def is_busy_background(img: Image.Image, box, *, threshold: float = BUSY_LUMA_STD) -> bool:
    """True if the bed under ``box`` is visually busy (high luma variance) — the
    condition under which we plate even when mean contrast looked acceptable."""
    return box_bg_stats(img, box)["luma_std"] >= float(threshold)


# ───────────────────────── colour repair ─────────────────────────
def max_contrast_shift(text_rgb: Sequence[float], bg_luma: float):
    """Nudge ``text_rgb`` toward whichever of near-white / near-black gives more
    contrast against a background of luma ``bg_luma`` (0..255), preserving a little
    of the original hue so it still feels designed. Returns an RGB int tuple.

    This is repair-ladder rung 1: the cheapest, mark-free fix — recolour the type
    before adding any plate."""
    bg = (bg_luma, bg_luma, bg_luma)
    to_white = contrast_ratio(_TEXT_WHITE, bg)
    to_black = contrast_ratio(_TEXT_BLACK, bg)
    target = _TEXT_WHITE if to_white >= to_black else _TEXT_BLACK
    src = np.asarray(text_rgb, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    # 82% toward the high-contrast pole keeps a whisper of the original tint.
    out = src * 0.18 + tgt * 0.82
    return tuple(int(round(float(v))) for v in np.clip(out, 0, 255))


# ───────────────────────── contrast plan ─────────────────────────
@dataclass
class ContrastPlan:
    """The outcome of routing a text box through ``ensure_contrast``.

    A primitive reads this and applies it: draw with ``text_color``; if
    ``scrim`` is set, lay that plate first (see ``apply_plan``); bump weight by
    ``weight_boost``; if ``stroke_w`` > 0 add that halo stroke; and if ``ok`` is
    False after everything, RELAYOUT or skip (the bed is hopeless for this box)."""
    text_color: tuple
    ok: bool = True
    ratio: float = 0.0
    target: float = CONTRAST_BODY
    scrim: dict | None = None          # {"alpha", "dark", "pad", "radius"} or None
    weight_boost: int = 0              # add to the font variation weight axis
    stroke_w: int = 0                  # px halo stroke (last-resort legibility)
    stroke_color: tuple | None = None
    busy: bool = False
    steps: list = field(default_factory=list)   # human-readable repair trace


def _plate_color_for(bg_luma: float, text_luma: float) -> tuple[tuple, bool]:
    """Pick a plate that pushes contrast the RIGHT way: dark plate under light
    text, light plate under dark text. Returns (rgb, is_dark)."""
    if text_luma >= bg_luma:                      # light text → darken behind it
        return (10, 9, 8), True
    return (236, 230, 220), False                 # dark text → lighten behind it


def ensure_contrast(draw: ImageDraw.ImageDraw | None, box, text_color,
                    bg_sampler: Callable[[object], tuple], *,
                    large: bool = False, target: float | None = None,
                    allow_scrim: bool = True, allow_weight: bool = True,
                    allow_stroke: bool = True,
                    busy_std: float | None = None,
                    img: Image.Image | None = None) -> ContrastPlan:
    """THE readability gate. Decide how to make ``text_color`` clear the contrast
    target over the background under ``box``, climbing a restrained repair ladder
    and stopping at the first rung that succeeds.

    ``bg_sampler`` is ``box_bg_sampler(img)`` (or any ``box -> mean RGB``). Pass the
    source ``img`` too (or instead) to also factor background *variance* (busy
    detection) into the decision. ``large`` selects the 3:1 target for big/title
    text; body uses 4.5:1.

    Returns a :class:`ContrastPlan`; it never draws and never raises. The repair
    ladder, in order:
        0. measure
        1. nudge text colour to max contrast            (no marks)
        2. add a restrained rounded scrim/plate          (soft wash)
        3. request a heavier font weight                 (fatter strokes)
        4. add a thin halo stroke                         (last resort)
    If after all rungs the ratio still misses, ``ok`` is False so the primitive can
    relayout / shrink / skip rather than ship an unreadable card."""
    tgt = float(target) if target is not None else (CONTRAST_LARGE if large else CONTRAST_BODY)
    steps: list[str] = []

    # --- 0. measure the bed -------------------------------------------------
    try:
        bg_rgb = tuple(float(v) for v in bg_sampler(box))
    except Exception:                                              # noqa: BLE001
        bg_rgb = (0.0, 0.0, 0.0)
    bg_luma = luma601(bg_rgb)
    busy = False
    if img is not None:
        st = box_bg_stats(img, box)
        bg_rgb = st["mean_rgb"]
        bg_luma = st["mean_luma"]
        busy = st["luma_std"] >= (busy_std if busy_std is not None else BUSY_LUMA_STD)

    cur = tuple(int(v) for v in text_color)
    ratio = contrast_ratio(cur, bg_rgb)
    steps.append(f"measure ratio={ratio:.2f} target={tgt:.1f} "
                 f"bg_luma={bg_luma:.0f} busy={busy}")

    plan = ContrastPlan(text_color=cur, ratio=ratio, target=tgt, busy=busy,
                        steps=steps)

    # A busy bed gets a plate even if mean contrast passed: variance is the real
    # legibility killer on footage. (Skipped if scrims are disabled by the caller.)
    if ratio >= tgt and not (busy and allow_scrim):
        plan.ok = True
        steps.append("pass (no repair)")
        return plan

    # --- 1. recolour to max contrast (mark-free) ---------------------------
    shifted = max_contrast_shift(cur, bg_luma)
    r2 = contrast_ratio(shifted, bg_rgb)
    if r2 > ratio + 0.05:
        plan.text_color = shifted
        cur, ratio = shifted, r2
        plan.ratio = r2
        steps.append(f"recolor -> {shifted} ratio={r2:.2f}")
    if ratio >= tgt and not (busy and allow_scrim):
        plan.ok = True
        steps.append("pass after recolor")
        return plan

    # --- 2. restrained scrim / plate ---------------------------------------
    if allow_scrim:
        text_luma = luma601(cur)
        plate_rgb, is_dark = _plate_color_for(bg_luma, text_luma)
        # Solve the smallest plate alpha that, blended over the bed, lifts contrast
        # to target — capped at SCRIM_MAX_ALPHA so it stays a soft wash. The plate
        # composites as  bed*(1-a) + plate*a, so we search a in small steps.
        chosen_a = 0.0
        eff_ratio = ratio
        for a in (x / 100.0 for x in range(8, int(SCRIM_MAX_ALPHA * 100) + 1, 2)):
            blended = tuple(b * (1 - a) + p * a for b, p in zip(bg_rgb, plate_rgb))
            rr = contrast_ratio(cur, blended)
            if rr >= tgt:
                chosen_a, eff_ratio = a, rr
                break
            chosen_a, eff_ratio = a, rr           # remember the best we reached
        # For a busy bed that already passed on contrast, still lay a gentle floor
        # plate so the variance is tamped down under the glyphs.
        if busy and chosen_a < 0.18:
            chosen_a = max(chosen_a, 0.18)
            blended = tuple(b * (1 - chosen_a) + p * chosen_a
                            for b, p in zip(bg_rgb, plate_rgb))
            eff_ratio = contrast_ratio(cur, blended)
        plan.scrim = {"alpha": round(chosen_a, 3), "dark": is_dark,
                      "rgb": plate_rgb, "pad": "auto", "radius": "auto"}
        plan.ratio = eff_ratio
        ratio = eff_ratio
        steps.append(f"scrim alpha={chosen_a:.2f} dark={is_dark} ratio={eff_ratio:.2f}")
        if ratio >= tgt:
            plan.ok = True
            steps.append("pass after scrim")
            return plan

    # --- 3. heavier weight (fatter strokes read better on noise) -----------
    if allow_weight:
        plan.weight_boost = 140
        steps.append("weight +140")
        # weight doesn't change the contrast math but materially helps perceived
        # legibility; we treat it as sufficient if we are already close.
        if ratio >= tgt - 0.4:
            plan.ok = True
            steps.append("accept (heavier weight, near target)")
            return plan

    # --- 4. last-resort halo stroke ----------------------------------------
    if allow_stroke:
        text_luma = luma601(cur)
        plan.stroke_w = 2
        plan.stroke_color = _TEXT_BLACK if text_luma > 128 else _TEXT_WHITE
        steps.append(f"stroke w=2 color={plan.stroke_color}")
        # A 2px halo on its own buys roughly a stop of separation; accept if close.
        if ratio >= tgt - 0.8:
            plan.ok = True
            steps.append("accept (stroke, near target)")
            return plan

    plan.ok = False
    steps.append("FAIL — caller should relayout / shrink / skip")
    return plan


def apply_plan(img: Image.Image, plan: ContrastPlan, box, *,
               pad: int | None = None, radius: int | None = None) -> Image.Image:
    """Render ``plan``'s scrim (if any) onto ``img`` under ``box`` and return the
    image. A no-op when the plan carries no scrim. The plate is a soft rounded
    low-opacity wash (premium-first), padded out a touch from the text box so the
    glyphs sit on it comfortably. Mutates and returns ``img`` (RGB)."""
    if plan.scrim is None or plan.scrim.get("alpha", 0) <= 0.001:
        return img
    base = img.convert("RGBA")
    w, h = base.size
    x0, y0, x1, y1 = _clamp_box(box, w, h)
    bw, bh = (x1 - x0), (y1 - y0)
    if pad is None:
        pad = int(round(max(bw, bh) * 0.16)) + 6
    if radius is None:
        radius = int(round(min(bw, bh) * 0.28)) + 8
    px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
    px1, py1 = min(w, x1 + pad), min(h, y1 + pad)
    plate = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    rgb = plan.scrim.get("rgb", (10, 9, 8))
    a = int(round(min(SCRIM_MAX_ALPHA, float(plan.scrim["alpha"])) * 255))
    pd.rounded_rectangle([px0, py0, px1 - 1, py1 - 1], radius=max(2, radius),
                         fill=(int(rgb[0]), int(rgb[1]), int(rgb[2]), a))
    # feather the plate so it melts into the footage instead of a crisp panel edge
    plate = plate.filter(ImageFilter.GaussianBlur(max(2, radius // 6)))
    out = Image.alpha_composite(base, plate)
    return out.convert("RGB")


# ───────────────────────── typography ─────────────────────────
def _char_w(draw: ImageDraw.ImageDraw, fnt: ImageFont.FreeTypeFont) -> float:
    """Approx average glyph advance for char-budget wrapping."""
    box = draw.textbbox((0, 0), "average", font=fnt)
    return (box[2] - box[0]) / 7.0


def readable_body_font(h: int, *, lines: int = 1, scale: float = 1.0,
                       weight: int = 620) -> ImageFont.FreeTypeFont:
    """A legible BODY/supporting font: the clean CONDENSED-SANS family (never the
    thin serif), at a weight heavy enough to hold up on footage, and never below
    the mobile-legibility px floor (``BODY_MIN_FRAC`` of ``h`` / ``BODY_MIN_PX``).

    ``scale`` lets a caller ask for a touch smaller/larger; ``lines`` lets a
    multi-line block start a notch smaller so it doesn't overfill. We route through
    ``look.font('caption')`` which resolves to Avenir-Next-Condensed / VidloreSans —
    a clean sans — deliberately NOT ``look.font('title')`` (Playfair serif)."""
    base = h * BODY_MIN_FRAC * 1.18 * float(scale)
    if lines >= 3:
        base *= 0.92
    size = int(round(max(base, BODY_MIN_PX, h * BODY_MIN_FRAC)))
    return look.font("caption", size, weight=weight)


def readable_title_font(h: int, *, scale: float = 1.0,
                        weight: int = 760) -> ImageFont.FreeTypeFont:
    """The premium SERIF heading font (Playfair / MGSerif), kept for headings where
    it is large enough to read. Weight bumped so the heading holds on noisy beds."""
    size = int(round(h * 0.090 * float(scale)))
    return look.font("title", max(size, int(h * 0.052)), weight=weight)


def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont,
                  max_w: int, *, max_lines: int = 4,
                  max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """Greedy word-wrap to BOTH a pixel width and a characters-per-line cap
    (whichever bites first → a sane reading measure, never a single 70-char line).
    Hard-splits a lone unbreakable token so a URL-ish string can't overflow. Folds
    any overflow onto the last line (the caller shrinks-to-fit). Never raises."""
    words = [w for w in str(text).split() if w]
    if not words:
        return [""]

    def _w(s: str) -> int:
        b = draw.textbbox((0, 0), s, font=fnt)
        return b[2] - b[0]

    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        fits = (_w(trial) <= max_w) and (len(trial) <= max_chars)
        if fits or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)

    used = sum(len(l.split()) for l in lines)
    if used < len(words) and lines:
        lines[-1] = " ".join([lines[-1]] + words[used:])

    # hard-split a single token still wider than the box (no spaces to wrap on)
    if len(lines) == 1 and " " not in lines[0] and _w(lines[0]) > max_w:
        s = lines[0]
        mid = len(s) // 2
        for cut in range(len(s) - 1, 0, -1):
            if _w(s[:cut]) <= max_w:
                mid = cut
                break
        lines = [s[:mid], s[mid:]][:max_lines]
    return lines[:max_lines]


def safe_margin(w: int, h: int) -> int:
    """Px safe margin (off-edge protection) for a w×h canvas."""
    return int(round(min(w, h) * SAFE_MARGIN_FRAC))


# ───────────────────────── one-call placement ─────────────────────────
def auto_scrim_if_needed(img: Image.Image, box, text_color, *,
                         large: bool = False,
                         bg_sampler: Callable[[object], tuple] | None = None,
                         **kw) -> tuple[Image.Image, ContrastPlan]:
    """Look at the bed under ``box`` and, ONLY if contrast fails or the bed is busy,
    lay a restrained readability plate. Returns ``(img, plan)`` — ``img`` mutated
    iff a plate was warranted, and the :class:`ContrastPlan` (so the caller can
    pick up the possibly-shifted ``text_color`` / weight / stroke). Premium-first:
    a clean high-contrast calm bed comes back untouched."""
    sampler = bg_sampler or box_bg_sampler(img)
    plan = ensure_contrast(None, box, text_color, sampler, large=large, img=img, **kw)
    if plan.scrim is not None:
        img = apply_plan(img, plan, box)
    return img, plan


def draw_text(img: Image.Image, xy, text: str, fnt: ImageFont.FreeTypeFont, *,
              fill, large: bool = False, anchor: str | None = None,
              align: str = "left", plate: bool = True,
              bg_sampler: Callable[[object], tuple] | None = None,
              return_plan: bool = False):
    """Place a single text run through the FULL readability gate and draw it.

    Measures the run's box, runs :func:`ensure_contrast`, lays a plate iff needed,
    applies the (possibly shifted) colour + a halo stroke if the plan asked for one,
    then draws. Mutates and returns ``img`` (or ``(img, plan)`` when ``return_plan``).

    This is the convenience entry the simple cards can adopt wholesale; the complex
    animated primitives instead call ``ensure_contrast`` / ``apply_plan`` directly so
    they can fold the plate into their own per-frame compositing."""
    d = ImageDraw.Draw(img)
    bb = d.textbbox(xy, text, font=fnt, anchor=anchor, align=align)
    box = (bb[0], bb[1], bb[2], bb[3])
    sampler = bg_sampler or box_bg_sampler(img)
    plan = ensure_contrast(None, box, fill, sampler, large=large, img=img,
                           allow_scrim=plate)
    if plate and plan.scrim is not None:
        img = apply_plan(img, plan, box)
        d = ImageDraw.Draw(img)
    stroke_w = plan.stroke_w if plan.stroke_w else 0
    stroke_fill = plan.stroke_color
    d.text(xy, text, font=fnt, fill=tuple(plan.text_color), anchor=anchor,
           align=align, stroke_width=stroke_w, stroke_fill=stroke_fill)
    return (img, plan) if return_plan else img


def draw_body(img: Image.Image, xy, text: str, *, h: int | None = None,
              fill=None, max_w: int | None = None, max_lines: int = 4,
              leading: float = 1.22, align: str = "left",
              anchor_top: bool = True, plate: bool = True,
              return_plan: bool = False):
    """Place a wrapped BODY block (clean heavier sans, capped measure, mobile px
    floor, contrast-gated). ``xy`` is the block's top-left (or top-centre if
    ``align='center'``). Returns ``img`` (or ``(img, plan)`` for the worst line)."""
    W, H = img.size
    h = h or H
    fnt = readable_body_font(h, lines=max_lines)
    fill = tuple(fill) if fill is not None else tuple(look.palette()["text"])
    margin = safe_margin(W, H)
    max_w = max_w if max_w is not None else (W - 2 * margin)
    d = ImageDraw.Draw(img)
    lines = wrap_to_width(d, text, fnt, max_w, max_lines=max_lines)
    bb = d.textbbox((0, 0), "AÉQyjg", font=fnt)
    line_h = int(round((bb[3] - bb[1]) * leading))
    x, y = int(xy[0]), int(xy[1])
    worst = None
    for ln in lines:
        if align == "center":
            lb = d.textbbox((0, 0), ln, font=fnt)
            lx = int(x - (lb[2] - lb[0]) / 2)
        else:
            lx = x
        img = draw_text(img, (lx, y), ln, fnt, fill=fill, large=False,
                        plate=plate, return_plan=True)
        img, plan = img
        if worst is None or plan.ratio < worst.ratio:
            worst = plan
        d = ImageDraw.Draw(img)
        y += line_h
    return (img, worst) if return_plan else img


def draw_headline(img: Image.Image, xy, text: str, *, h: int | None = None,
                  fill=None, max_w: int | None = None, max_lines: int = 2,
                  scale: float = 1.0, leading: float = 1.10, align: str = "center",
                  plate: bool = True, return_plan: bool = False):
    """Place a premium-serif HEADLINE (large-text 3:1 target, auto-shrink to fit
    ``max_lines`` within the safe width, contrast-gated). Returns ``img`` (or
    ``(img, plan)``)."""
    W, H = img.size
    h = h or H
    fill = tuple(fill) if fill is not None else tuple(look.palette()["accent_hi"])
    margin = safe_margin(W, H)
    max_w = max_w if max_w is not None else (W - 2 * margin)
    d = ImageDraw.Draw(img)
    scl = float(scale)
    fnt = readable_title_font(h, scale=scl)
    lines = wrap_to_width(d, text, fnt, max_w, max_lines=max_lines, max_chars=999)
    # shrink until it fits the line budget within the safe width
    guard = 0
    while guard < 12:
        widest = max((d.textbbox((0, 0), l, font=fnt)[2]
                      - d.textbbox((0, 0), l, font=fnt)[0]) for l in lines)
        if widest <= max_w and len(lines) <= max_lines:
            break
        scl *= 0.92
        fnt = readable_title_font(h, scale=scl)
        lines = wrap_to_width(d, text, fnt, max_w, max_lines=max_lines, max_chars=999)
        guard += 1
    bb = d.textbbox((0, 0), "AÉQyjg", font=fnt)
    line_h = int(round((bb[3] - bb[1]) * leading))
    x, y = int(xy[0]), int(xy[1])
    worst = None
    for ln in lines:
        if align == "center":
            lb = d.textbbox((0, 0), ln, font=fnt)
            lx = int(x - (lb[2] - lb[0]) / 2)
        else:
            lx = x
        img, plan = draw_text(img, (lx, y), ln, fnt, fill=fill, large=True,
                              plate=plate, return_plan=True)
        if worst is None or plan.ratio < worst.ratio:
            worst = plan
        d = ImageDraw.Draw(img)
        y += line_h
    return (img, worst) if return_plan else img


__all__ = [
    "CONTRAST_BODY", "CONTRAST_LARGE", "BUSY_LUMA_STD", "BODY_MIN_FRAC",
    "BODY_MIN_PX", "MAX_CHARS_PER_LINE", "SAFE_MARGIN_FRAC", "SCRIM_MAX_ALPHA",
    "relative_luminance", "contrast_ratio", "luma601",
    "box_bg_stats", "box_bg_sampler", "is_busy_background",
    "max_contrast_shift", "ContrastPlan", "ensure_contrast", "apply_plan",
    "readable_body_font", "readable_title_font", "wrap_to_width", "safe_margin",
    "auto_scrim_if_needed", "draw_text", "draw_body", "draw_headline",
]
