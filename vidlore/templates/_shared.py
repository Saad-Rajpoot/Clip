"""SHARED DOCUMENTARY VISUAL LANGUAGE (Phase 2 foundation).

This module is the SINGLE source of the design DNA that every template
inherits — typography scale, colour depth, easing/timing presets,
reusable PIL composition primitives, and motion-expression helpers
for assemble.py.

The goal: when the viewer sees ANY template render, it should
subconsciously read as "same documentary system" — consistent
typography weight, consistent palette depth, consistent reveal
pacing, consistent easing on every animation. Templates pick from
this shared vocabulary; they don't reinvent it.

NO BREAKING CHANGES policy: existing templates already in
production are NOT refactored to use this module on day one — they
work, leave them alone. NEW templates start using it, and existing
templates can opt in gradually template-by-template when each is
next polished.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# =========================================================================
# TYPOGRAPHY SCALE
# A consistent type ladder used by every template. Each role has a single
# size — templates pick the role, never the raw number, so a future
# global retune touches one place.
# =========================================================================

TYPE_DISPLAY = 88     # mega numbers (stat dashboard, hero number reveal)
TYPE_H1      = 72     # primary headlines (timeline header, map place name)
TYPE_H2      = 54     # secondary headlines (evidence/comparison)
TYPE_H3      = 42     # tertiary (subhead lines, captions on cards)
TYPE_BODY    = 32     # paragraph body text (documents, dossiers)
TYPE_CAPTION = 24     # small labels (stat labels, badge labels)
TYPE_MICRO   = 20     # micro tags (file numbers, timestamps, agency)


# =========================================================================
# TIMING PRESETS (seconds)
# Reveal pacing presets that match documentary editing — slower than
# social-media motion, faster than slide-deck animation.
# =========================================================================

REVEAL_FAST   = 0.35    # quick element pop (badge, dot marker)
REVEAL_MEDIUM = 0.55    # standard text fade-in
REVEAL_SLOW   = 0.85    # cinematic headline reveal
HOLD_SHORT    = 0.30    # pre-reveal footage hold
HOLD_LONG     = 0.60    # documentary breathing room before overlay
TAIL_FADE     = 0.45    # tail fade-out duration


# =========================================================================
# EASING — both PIL-time progress curves AND ffmpeg-expression strings.
# =========================================================================

def ease_out_cubic(p: float) -> float:
    """0..1 → 0..1 with cubic ease-out (decelerating)."""
    return 1 - (1 - p) ** 3


def ease_out_quart(p: float) -> float:
    """0..1 → 0..1 with quartic ease-out (slower start, gentler land)."""
    return 1 - (1 - p) ** 4


def ease_in_out(p: float) -> float:
    """0..1 → 0..1 with symmetric ease in+out."""
    return p * p * (3 - 2 * p)


def ff_ease_out_cubic(p_expr: str) -> str:
    """ffmpeg-expression for cubic ease-out. `p_expr` is an expression
    that evaluates to 0..1 over the reveal window."""
    return f"(1-pow(1-{p_expr}\\,3))"


def ff_ease_out_quart(p_expr: str) -> str:
    return f"(1-pow(1-{p_expr}\\,4))"


def ff_progress(rk_start: float, dur: float) -> str:
    """ffmpeg expression for a clamped 0..1 progress over [rk_start,
    rk_start+dur]. Used by drawtext alpha/y expressions."""
    return f"min(1\\,max(0\\,(t-{rk_start:.2f})/{dur}))"


def ff_alpha_fade_in_out(
    appear: float, fade_in: float, hold_until: float, fade_out: float
) -> str:
    """ffmpeg alpha= expression for: 0 before `appear`, ramp up over
    `fade_in`, hold at 1 until `hold_until`, ramp down over `fade_out`.
    Used by every drawtext that needs a clean fade-in/hold/fade-out."""
    end_of_fade_in = appear + fade_in
    end_of_fade_out = hold_until + fade_out
    return (
        f"if(lt(t,{appear:.2f})\\,0\\,"
        f"if(lt(t,{end_of_fade_in:.2f})\\,"
        f"(t-{appear:.2f})/{fade_in:.2f}\\,"
        f"if(lt(t,{hold_until:.2f})\\,1\\,"
        f"max(0\\,({end_of_fade_out:.2f}-t)/{fade_out:.2f}))))"
    )


# =========================================================================
# COLOUR / PALETTE depth helpers
# =========================================================================

# =========================================================================
# VISUAL VARIATION ENGINE — Phase 1
# -------------------------------------------------------------------------
# Pro documentary editors maintain ONE cinematic identity while making each
# scene feel visually fresh. They decide per-scene (palette, layout, shape,
# motion) based on the scene's content, emotion and what came before — never
# by spinning the same template wheel. This engine codifies those decisions:
#
#   • PALETTES — 8 documentary palettes (intel blue, sepia archive, military
#     green, dark investigative, warm historical, intel amber, red emergency,
#     newspaper mono). The doc's THEME picks a family; scene ROLE biases
#     toward emergency/sepia; RECENCY enforces 3-scene cooldown.
#   • LAYOUTS  — per-kind layout pools (centered/left/split/stacked/asymm.).
#   • SHAPES   — per-kind card-shape pools (soft / hard tactical / paper
#     snippet / evidence tag / transparent overlay / side-callout).
#   • MOTION + OVERLAY + TYPOGRAPHY — picked here, wired by callers (Phase 1
#     wires palette + bg + a few shapes; later phases wire the rest).
#   • RECENCY  — each dimension has a cooldown; a variant won't re-pick
#     within N scenes (palette 3, layout 2, shape 2). Picks stay deterministic
#     (same brief -> same sequence) so caching still works.
#
# The renderer calls `set_scene_context(sc, theme)` once before drawing each
# card; every helper here reads the picked StylePack automatically — no
# per-template plumbing required for the palette + bg layer.
# =========================================================================

# ---- 8 PALETTES (rgb + bg gradient + key-light tint) --------------------
PALETTES: dict[str, dict] = {
    "intel_blue": {
        "accent": (110, 170, 230), "accent2": (220, 180, 80),
        "ink": (245, 248, 255),    "ink_dim": (170, 190, 220),
        "bg_top": (16, 24, 42),    "bg_bot": (5, 8, 16),
        "key": (54, 80, 130, 255), "key_y": 0.30},
    "sepia_archive": {
        "accent": (200, 150, 70),  "accent2": (235, 215, 175),
        "ink": (240, 230, 210),    "ink_dim": (180, 160, 130),
        "bg_top": (36, 28, 20),    "bg_bot": (12, 9, 6),
        "key": (130, 92, 44, 230), "key_y": 0.33},
    "military_green": {
        "accent": (130, 180, 100), "accent2": (230, 220, 170),
        "ink": (235, 240, 220),    "ink_dim": (165, 180, 145),
        "bg_top": (22, 30, 20),    "bg_bot": (6, 12, 6),
        "key": (84, 138, 76, 235), "key_y": 0.32},
    "dark_investigative": {
        "accent": (220, 90, 80),   "accent2": (225, 225, 225),
        "ink": (240, 240, 245),    "ink_dim": (155, 155, 165),
        "bg_top": (14, 14, 18),    "bg_bot": (2, 2, 5),
        "key": (74, 36, 36, 215),  "key_y": 0.28},
    "warm_historical": {
        "accent": (220, 170, 95),  "accent2": (190, 130, 80),
        "ink": (245, 232, 210),    "ink_dim": (180, 155, 120),
        "bg_top": (30, 23, 18),    "bg_bot": (8, 6, 4),
        "key": (148, 100, 60, 230), "key_y": 0.31},
    "intel_amber": {
        "accent": (240, 185, 90),  "accent2": (225, 215, 190),
        "ink": (245, 240, 225),    "ink_dim": (180, 170, 145),
        "bg_top": (18, 19, 24),    "bg_bot": (4, 5, 8),
        "key": (210, 145, 50, 230), "key_y": 0.29},
    "red_emergency": {
        "accent": (220, 75, 75),   "accent2": (240, 200, 160),
        "ink": (245, 235, 230),    "ink_dim": (180, 150, 145),
        "bg_top": (22, 13, 13),    "bg_bot": (6, 4, 4),
        "key": (180, 48, 48, 225), "key_y": 0.27},
    "newspaper_mono": {
        "accent": (35, 35, 35),    "accent2": (110, 110, 110),
        "ink": (28, 28, 32),       "ink_dim": (130, 130, 135),
        "bg_top": (246, 244, 238), "bg_bot": (218, 216, 210),
        "key": (190, 188, 180, 200), "key_y": 0.30},
}

# Topic-keyword -> preferred palettes (first 3 = primary candidates).
_TOPIC_PALETTES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("cold war", "kgb", "cia", "intelligence", "spy", "espionage",
      "operation", "defector"),
     ("intel_blue", "sepia_archive", "intel_amber", "dark_investigative")),
    (("war", "battle", "military", "soldier", "regiment", "naval"),
     ("military_green", "sepia_archive", "warm_historical", "dark_investigative")),
    (("crime", "murder", "killer", "investigation", "evidence", "case"),
     ("dark_investigative", "red_emergency", "intel_blue", "newspaper_mono")),
    (("history", "ancient", "century", "dynasty", "empire", "kingdom",
      "ledger", "report", "archive", "manuscript"),
     ("warm_historical", "sepia_archive", "newspaper_mono", "intel_amber")),
    (("news", "breaking", "press", "front page", "newspaper", "journal"),
     ("newspaper_mono", "intel_blue", "warm_historical", "intel_amber")),
    (("disaster", "emergency", "crash", "explosion", "outbreak", "warning"),
     ("red_emergency", "dark_investigative", "intel_amber", "intel_blue")),
]

# Per-kind layout candidates (controlled variation pools).
_KIND_LAYOUTS: dict[str, tuple[str, ...]] = {
    "comparison":     ("split", "asymmetric", "stacked"),
    "cause_effect":   ("split", "asymmetric", "stacked"),
    "stat_dashboard": ("centered", "asymmetric", "split"),
    "number":         ("centered", "asymmetric"),
    "bullet_list":    ("left", "stacked", "centered"),
    "quote_highlight": ("centered", "left"),
    "document":       ("left",),          # already filled article
    "timeline":       ("centered",),
}

# Per-kind shape pools.
_KIND_SHAPES: dict[str, tuple[str, ...]] = {
    "comparison":     ("hard_tactical", "soft_card", "evidence_tag"),
    "cause_effect":   ("hard_tactical", "soft_card", "side_callout"),
    "stat_dashboard": ("hard_tactical", "soft_card", "asymmetric_panel"),
    "number":         ("hard_tactical", "transparent_overlay"),
    "bullet_list":    ("soft_card", "paper_snippet", "transparent_overlay"),
    "quote_highlight": ("highlighted_text", "side_callout", "soft_card"),
    "document":       ("paper_snippet",),
}

# Motion + overlay pools (informational in Phase 1 — wired in later phases).
_MOTIONS  = ("slow_push", "slide", "masked_reveal", "wipe", "fade_up",
             "parallax", "stagger", "snap_hook")
# Overlay families — the assembler injects ONE per scene via the
# `_overlay_finish_for_scene` helper. Subtle on purpose: real documentary
# texture realism, NOT TikTok effects. Names map to ffmpeg filter snippets
# in assemble.py (`_OVERLAY_FILTERS`).
_OVERLAYS = ("archival_grain", "crt_scan", "paper_tex", "dust_speckle",
             "projector_flicker", "broadcast_noise")

# Palette-preferred overlay families — keeps texture coherent with palette
# family (intel + scan = tactical; sepia + paper = period; mono + paper =
# journalistic). Per-scene rotation within the preferred set, recency-aware.
_OVERLAY_FOR_PALETTE: dict[str, tuple[str, ...]] = {
    "intel_blue":         ("crt_scan", "projector_flicker", "broadcast_noise"),
    "intel_amber":        ("crt_scan", "archival_grain", "broadcast_noise"),
    "sepia_archive":      ("paper_tex", "archival_grain", "dust_speckle"),
    "warm_historical":    ("archival_grain", "paper_tex", "dust_speckle"),
    "dark_investigative": ("crt_scan", "broadcast_noise", "archival_grain"),
    "military_green":     ("archival_grain", "dust_speckle", "crt_scan"),
    "red_emergency":      ("broadcast_noise", "archival_grain"),
    "newspaper_mono":     ("paper_tex", "archival_grain"),
}

# Typography packs — each palette picks from a PREFERRED set so type
# always feels coherent with the palette family (sepia + typewriter feels
# period; intel + mono feels tactical; newspaper + serif feels journalistic).
# Names map to FONT_PACKS in footage.py which holds the actual font paths.
_TYPO_FOR_PALETTE: dict[str, tuple[str, ...]] = {
    "intel_blue":         ("intel_mono", "premium_sans"),
    "intel_amber":        ("intel_mono", "premium_sans"),
    "sepia_archive":      ("typewriter", "archive_serif"),
    "warm_historical":    ("archive_serif", "typewriter"),
    "dark_investigative": ("premium_sans", "intel_mono"),
    "military_green":     ("military_stencil", "broadcast_sans"),
    "red_emergency":      ("broadcast_sans", "premium_sans"),
    "newspaper_mono":     ("newspaper_serif", "premium_sans"),
}


class StylePack:
    """Per-scene decisions; the cinematic identity of ONE card."""
    __slots__ = ("palette_name", "p", "layout", "shape",
                 "motion", "overlay", "typography")

    def __init__(self, palette_name, palette, layout, shape,
                 motion, overlay, typography):
        self.palette_name = palette_name
        self.p = palette
        self.layout = layout
        self.shape = shape
        self.motion = motion
        self.overlay = overlay
        self.typography = typography


# Per-render state. Reset by `reset_variation()` at the start of each film.
_SCENE_CTX: dict = {"index": 0, "theme": {}, "topic_blob": ""}
_RECENT: dict = {"palette": [], "layout": [], "shape": []}
_COOL: dict = {"palette": 3, "layout": 2, "shape": 2}
_PACK_CACHE: dict = {}


def reset_variation(topic_blob: str = "") -> None:
    """Call once at the START of a film render — clears recency + cache so
    each film picks its own variant sequence."""
    _SCENE_CTX["topic_blob"] = (topic_blob or "").lower()
    _RECENT["palette"].clear()
    _RECENT["layout"].clear()
    _RECENT["shape"].clear()
    _PACK_CACHE.clear()


def set_scene_context(sc, theme=None) -> None:
    """Call once per scene from the renderer, BEFORE drawing the card."""
    try:
        _SCENE_CTX["index"] = int(getattr(sc, "index", 0) or 0)
    except Exception:                                       # noqa: BLE001
        _SCENE_CTX["index"] = 0
    if theme is not None:
        _SCENE_CTX["theme"] = theme or {}
    # absorb scene-level topic words for richer palette picking
    blob = _SCENE_CTX.get("topic_blob", "") or ""
    extra = " ".join([
        " ".join(getattr(sc, "keywords", None) or []),
        getattr(sc, "query", "") or "",
        getattr(sc, "narration", "") or "",
    ]).lower()
    _SCENE_CTX["scene_blob"] = extra
    if blob and len(blob) < 4000:
        _SCENE_CTX["topic_blob"] = (blob + " " + extra)[:8000]


def _pick(pool: tuple, dim: str, seed_idx: int):
    """Pick from a pool while honoring recency (don't re-pick anything used
    within the last `_COOL[dim]` scenes). Falls through if everything is
    cooling — keeps determinism by rotating the pool with `seed_idx`."""
    recent = _RECENT.get(dim, [])
    cool_n = _COOL.get(dim, 2)
    rotated = pool[seed_idx % len(pool):] + pool[:seed_idx % len(pool)]
    fresh = [p for p in rotated if p not in recent[-cool_n:]]
    pick = fresh[0] if fresh else rotated[0]
    recent.append(pick)
    if len(recent) > 12:
        del recent[:-12]
    return pick


def _palette_for_scene(theme) -> str:
    """Choose a palette family: theme accent steers the warmth/coolness;
    scene topic keywords boost specific palettes; scene role biases (climax
    -> emergency / amber; reflection -> sepia / warm). Recency-filtered."""
    blob = _SCENE_CTX.get("scene_blob", "") + " " + \
           _SCENE_CTX.get("topic_blob", "")
    candidates: list[str] = []
    for keys, pals in _TOPIC_PALETTES:
        if any(k in blob for k in keys):
            candidates.extend(pals)
            break
    # default doc family if no topic matched
    if not candidates:
        candidates = ["warm_historical", "intel_blue", "sepia_archive",
                      "intel_amber"]
    # de-dupe preserving order
    seen, cands = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            cands.append(c)
    return _pick(tuple(cands), "palette", _SCENE_CTX["index"])


def scene_pack(theme=None) -> StylePack:
    """Decide-or-recall this scene's StylePack. Deterministic per render."""
    idx = _SCENE_CTX.get("index", 0)
    if idx in _PACK_CACHE:
        return _PACK_CACHE[idx]
    pname = _palette_for_scene(theme or _SCENE_CTX.get("theme", {}))
    p = PALETTES.get(pname, PALETTES["intel_blue"])
    # layout + shape pools depend on this scene's graphic_kind, but `accent`
    # and bg helpers don't know it — default pools used here; per-template
    # callers may override via `scene_layout(kind)` / `scene_shape(kind)`.
    layout = _pick(("centered", "left", "split", "stacked", "asymmetric"),
                   "layout", idx)
    shape = _pick(("soft_card", "hard_tactical", "paper_snippet",
                   "evidence_tag", "transparent_overlay"), "shape", idx)
    motion = _MOTIONS[idx % len(_MOTIONS)]
    # palette-driven overlay (NOT random) so texture matches palette family
    ov_pool = _OVERLAY_FOR_PALETTE.get(pname, ("archival_grain",))
    overlay = ov_pool[idx % len(ov_pool)]
    # palette-driven typography (NOT random) so font feels coherent with bg
    typo_pool = _TYPO_FOR_PALETTE.get(pname, ("premium_sans",))
    typo = typo_pool[idx % len(typo_pool)]
    pack = StylePack(pname, p, layout, shape, motion, overlay, typo)
    _PACK_CACHE[idx] = pack
    return pack


def scene_layout(kind: str) -> str:
    """Kind-aware layout pick (recency-filtered). Cached per (scene_idx, kind)
    so re-calls within the same scene return the SAME layout — otherwise the
    second call (e.g. inside the renderer after a test pre-call) would see
    fresh recency state and pick differently."""
    idx = _SCENE_CTX.get("index", 0)
    key = ("layout", idx, kind or "")
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]
    pool = _KIND_LAYOUTS.get(kind or "", ("centered",))
    pick = _pick(pool, "layout", idx)
    _PACK_CACHE[key] = pick
    return pick


# Visual-class buckets — every concrete shape belongs to a CLASS (panel /
# paper / text-only / evidence). Anti-repetition: when two adjacent scenes
# share a CLASS, the next pick is biased toward a different class so the
# viewer subconsciously sees creative direction, not the same card system.
_VISUAL_CLASS_OF = {
    "soft_card":           "panel",
    "hard_tactical":       "panel",
    "asymmetric_panel":    "panel",
    "paper_snippet":       "paper",
    "torn_paper":          "paper",
    "sticky":              "paper",
    "evidence_tag":        "evidence",
    "side_callout":        "callout",
    "highlighted_text":    "text_only",
    "transparent_overlay": "text_only",
    "underline_only":      "text_only",
}


def _recent_classes(n: int = 3) -> list[str]:
    return [_VISUAL_CLASS_OF.get(s, "x")
            for s in _RECENT.get("shape", [])[-n:]]


def scene_shape(kind: str) -> str:
    """Kind-aware card-shape pick. First filters by per-shape recency, then
    by VISUAL-CLASS recency so two `card_panel` scenes in a row push the
    next pick toward `paper` / `text_only` / `evidence` — controlled
    variation, not random. Cached per (scene_idx, kind) so re-calls within
    the same scene return the same shape."""
    idx = _SCENE_CTX.get("index", 0)
    key = ("shape", idx, kind or "")
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]
    pool = _KIND_SHAPES.get(kind or "", ("soft_card",))
    cls_recent = _recent_classes(2)
    if cls_recent:
        diverse = [s for s in pool
                   if _VISUAL_CLASS_OF.get(s, "x") not in cls_recent]
        if diverse:
            pool = tuple(diverse)
    pick = _pick(pool, "shape", idx)
    _PACK_CACHE[key] = pick
    return pick


# ---- BACK-COMPAT accent() — Look-DNA-aware -----------------------------
def accent(theme: dict) -> tuple[int, int, int]:
    """Accent picker for renderers.

    Priority order:
      1. Active Look DNA `accent_rgb` (stable per-channel identity).
      2. Theme-level accent (set by pipeline.py when channel active).
      3. Per-scene rotated palette from StylePack (legacy path).

    Without a channel, rotation still cycles 8 palettes per scene so
    legacy renders look identical.  With a channel active, the accent
    LOCKS to the channel signature colour so every Midnight card is
    cool blue, every Atlas card is red, every Amber card is amber.
    """
    try:
        from .. import footage as _fg
        pal = _fg._look_card_palette({})
        if pal.get("channel_id"):
            return tuple(pal.get("accent", (210, 170, 90)))
    except Exception:                                       # noqa: BLE001
        pass
    if theme:
        ta = theme.get("accent")
        if isinstance(ta, (tuple, list)) and len(ta) >= 3:
            return (int(ta[0]), int(ta[1]), int(ta[2]))
    return tuple(scene_pack(theme).p["accent"])  # type: ignore[return-value]


def accent_static(theme: dict) -> tuple[int, int, int]:
    """The UN-rotated theme accent — used by elements that must stay constant
    across the whole film (series watermark, captions title, etc.)."""
    a = (theme or {}).get("accent")
    if isinstance(a, (tuple, list)) and len(a) >= 3:
        return (int(a[0]), int(a[1]), int(a[2]))
    return (210, 170, 90)


def with_alpha(rgb: tuple, a: int) -> tuple:
    """(r,g,b) + alpha → RGBA tuple."""
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))


# =========================================================================
# READABILITY — WCAG relative luminance + contrast, and background-aware ink.
# Used so EVERY text layer adapts to its background: a light parchment /
# white-clinical card gets DARK type, a dark card gets LIGHT type, and a
# headline never renders near-white on cream again (the "HOW COPPER STOPS
# SLUGS" class). Pure-Python (no deps); safe to call anywhere.
# =========================================================================

def relative_luminance(rgb) -> float:
    """WCAG 2.0 relative luminance of an (r,g,b) in 0..255."""
    def _lin(c):
        c = max(0.0, min(1.0, c / 255.0))
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (rgb[0], rgb[1], rgb[2])
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg, bg) -> float:
    """WCAG contrast ratio (1..21) between two (r,g,b) colours."""
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


def is_light_card_bg() -> bool:
    """True when the ACTIVE card background (Look DNA bg_mode) is a LIGHT
    surface (parchment / white-clinical / matte-light) and therefore needs
    DARK type. Dark/navy panels return False."""
    pal = look_palette()
    bg_mode = (pal.get("bg_mode") or "dark_panel")
    if bg_mode in ("white_clinical", "paper_scan"):
        return True
    if bg_mode == "matte_flat":
        base = pal.get("paper_rgb") or pal.get("matte_rgb") or (20, 20, 24)
        return relative_luminance(base) > 0.5
    return False


def card_bg_rgb() -> tuple:
    """Representative (r,g,b) of the active card BACKGROUND, so callers can
    pick a contrast-safe ink. Light modes return their paper/white colour;
    dark modes return the navy base."""
    pal = look_palette()
    bg_mode = (pal.get("bg_mode") or "dark_panel")
    if bg_mode == "paper_scan":
        return tuple((pal.get("paper_rgb") or (242, 230, 208))[:3])
    if bg_mode == "white_clinical":
        return tuple((pal.get("paper_rgb") or (245, 246, 248))[:3])
    if bg_mode == "matte_flat":
        return tuple((pal.get("matte_rgb") or pal.get("paper_rgb") or (18, 18, 22))[:3])
    return tuple(NAVY_BASE)


def pick_ink(bg_rgb=None, *, light=(245, 248, 255), dark=(28, 26, 24),
             min_ratio: float = 4.5) -> tuple:
    """Background-aware text colour: returns `dark` on a light background and
    `light` on a dark one, choosing whichever gives the higher contrast and
    nudging toward pure black/white if neither clears `min_ratio`."""
    bg = bg_rgb if bg_rgb is not None else card_bg_rgb()
    c_dark = contrast_ratio(dark, bg)
    c_light = contrast_ratio(light, bg)
    ink = dark if c_dark >= c_light else light
    if max(c_dark, c_light) < min_ratio:           # push to the extreme
        ink = (16, 15, 14) if relative_luminance(bg) > 0.5 else (250, 250, 252)
    return tuple(ink)


def pick_body_ink(bg_rgb=None, *, min_ratio: float = 4.5) -> tuple:
    """Body / secondary text ink that is MUTED vs the title (for hierarchy) but
    still clears the WCAG body-copy threshold on the given card background.

    The Look palettes define a `mute_rgb` tuned for DARK panels (a light grey on
    navy); on a LIGHT card (paper_scan / parchment) that same mute is a pale tan
    that drops body copy to ~3:1 and reads as unreadable (the "How copper stops
    slugs" step descriptions). This derives the body ink from the ACTUAL card
    background instead: start at the full-contrast title ink, mute it ~22 %
    toward the background for visual hierarchy, and fall back to full contrast
    if muting would drop below `min_ratio`. Guarantees readable body copy on any
    background while keeping title > body hierarchy."""
    bg = bg_rgb if bg_rgb is not None else card_bg_rgb()
    ink = pick_ink(bg, min_ratio=min_ratio)
    muted = tuple(int(round(ink[i] * 0.78 + bg[i] * 0.22)) for i in range(3))
    return muted if contrast_ratio(muted, bg) >= min_ratio else tuple(ink)


def safe_accent_on(bg_rgb, accent_rgb, *, min_ratio: float = 3.0) -> tuple:
    """Return `accent_rgb` if it clears `min_ratio` on `bg_rgb`; otherwise a
    darkened (light bg) or lightened (dark bg) version that does. Keeps the
    brand hue while guaranteeing a readable headline rule / bracket."""
    if contrast_ratio(accent_rgb, bg_rgb) >= min_ratio:
        return tuple(accent_rgb[:3])
    if relative_luminance(bg_rgb) > 0.5:           # light bg -> darken accent
        for f in (0.7, 0.55, 0.42, 0.3):
            cand = tuple(int(c * f) for c in accent_rgb[:3])
            if contrast_ratio(cand, bg_rgb) >= min_ratio:
                return cand
        return (40, 34, 24)
    for f in (1.4, 1.8, 2.4):                      # dark bg -> lighten accent
        cand = tuple(min(255, int(c * f)) for c in accent_rgb[:3])
        if contrast_ratio(cand, bg_rgb) >= min_ratio:
            return cand
    return (236, 224, 198)


# Documentary base palette — the "data templates" share this.
NAVY_BASE      = (10, 18, 36)
NAVY_GRADIENT  = (6, 8, 14)              # bottom-of-gradient direction
TEXT_PRIMARY   = (245, 248, 255)
TEXT_SECONDARY = (180, 200, 230)
TEXT_DIM       = (140, 160, 195)
BLACK_BACKING  = (0, 0, 0)


# =========================================================================
# LOOK-DNA AWARE PALETTE ACCESSORS
# These read the active Look DNA (via footage._look_card_palette) so a
# Midnight Pacific data-card ends up with cool desaturated panel text,
# an Amber Chronicles card gets warm sepia type, and an Atlas Explained
# card gets near-black on white — from the SAME `TEXT_PRIMARY` call.
#
# Templates that already inherit from this module via the
# `TEXT_PRIMARY`/`TEXT_SECONDARY` constants keep working unchanged on
# the legacy (no-channel) path because `look_palette()` returns those
# exact baked defaults when no Look DNA is active.
# =========================================================================

def look_palette() -> dict:
    """Lazy-import shim around `footage._look_card_palette` so this
    module can be imported before `footage` finishes initialising.
    Returns the same dict that footage._look_card_palette returns.
    Never raises."""
    try:
        from .. import footage as _fg
        return _fg._look_card_palette({})
    except Exception:                                       # noqa: BLE001
        return {}


def text_primary() -> tuple:
    """Channel-aware primary type colour for opaque data cards.
    Inherits dark-mode default (245,248,255) when no channel."""
    pal = look_palette()
    bg_mode = (pal.get("bg_mode") or "dark_panel")
    if bg_mode in ("white_clinical", "paper_scan"):
        return tuple(pal.get("ink_rgb", (33, 32, 34)))
    return tuple(pal.get("panel_text", TEXT_PRIMARY))


def text_secondary() -> tuple:
    pal = look_palette()
    bg_mode = (pal.get("bg_mode") or "dark_panel")
    if bg_mode in ("white_clinical", "paper_scan"):
        return tuple(pal.get("mute_rgb", (122, 120, 124)))
    return tuple(pal.get("panel_text2", TEXT_SECONDARY))


def panel_card_bg() -> tuple:
    """Background colour for inset stat / data sub-cards drawn on top
    of `make_navy_grid_bg`.  Atlas (white_clinical) wants a slightly
    darker neutral inset; Midnight (dark_panel) wants a lifted
    cool-blue overlay; Amber (paper_scan) wants a warm parchment block.
    """
    pal = look_palette()
    bg_mode = (pal.get("bg_mode") or "dark_panel")
    if bg_mode == "white_clinical":
        return (242, 242, 244, 255)
    if bg_mode == "paper_scan":
        return (236, 220, 192, 245)
    # dark / matte
    return (255, 255, 255, 14)


# =========================================================================
# PIL COMPOSITION PRIMITIVES — reusable bg + headline + vignette helpers
# =========================================================================

def _make_paper_scan_bg(W: int, H: int, pal: dict) -> Image.Image:
    """AMBER CHRONICLES — warm sepia paper / antique parchment back-
    plate.  Heavy fibre grain, soft uneven scan lighting, strong edge
    vignette so the panel feels like a museum-grade scanned page."""
    paper = pal.get("paper_rgb", (242, 230, 208))
    bg = Image.new("RGB", (W, H), tuple(paper))
    # top-down warm gradient (lighter top, darker bottom — scan vignette)
    grad = Image.new("L", (1, H), 0)
    gd = ImageDraw.Draw(grad)
    for yy in range(H):
        gd.point((0, yy), fill=int(255 - 70 * ((yy / H) ** 1.30)))
    grad = grad.resize((W, H))
    darker = Image.new("RGB", (W, H),
        (max(0, paper[0] - 36), max(0, paper[1] - 38), max(0, paper[2] - 40)))
    bg = Image.composite(bg, darker, grad)
    # off-centre warm bloom — uneven scan
    bloom = Image.new("L", (W, H), 0)
    ImageDraw.Draw(bloom).ellipse(
        [W * 0.18, -H * 0.20, W * 0.82, H * 0.96], fill=78)
    bloom = bloom.filter(ImageFilter.GaussianBlur(int(W * 0.18)))
    bg = bg.convert("RGBA")
    warm = Image.new("RGBA", (W, H),
        (min(255, paper[0] + 8), min(255, paper[1] + 8),
         min(255, paper[2] + 2), 0))
    warm.putalpha(bloom)
    bg = Image.alpha_composite(bg, warm)
    # heavy paper grain (fibre grain at two scales)
    try:
        gn = Image.effect_noise((W, H), 12).convert("L")
        tint = Image.new("RGBA", (W, H), (90, 60, 30, 24))
        bg = Image.alpha_composite(bg, Image.composite(
            tint, Image.new("RGBA", (W, H), (0, 0, 0, 0)),
            gn.point(lambda p: 24 if p > 130 else 0)))
        gn2 = Image.effect_noise((W // 2, H // 2), 18).convert("L").resize((W, H))
        tint2 = Image.new("RGBA", (W, H), (120, 80, 40, 14))
        bg = Image.alpha_composite(bg, Image.composite(
            tint2, Image.new("RGBA", (W, H), (0, 0, 0, 0)),
            gn2.point(lambda p: 18 if p > 150 else 0)))
    except Exception:                                       # noqa: BLE001
        pass
    # strong vignette
    vstr = int(pal.get("vignette_strength", 135))
    return apply_vignette(bg, strength=vstr, blur=int(W * 0.20))


def _make_clinical_white_bg(W: int, H: int, pal: dict) -> Image.Image:
    """ATLAS EXPLAINED — bright editorial dataviz backplate.  Flat
    near-white surface with a barely-perceptible top-down tint and an
    OPTIONAL very faint dot-grid to imply 'data-canvas' without ever
    reading as a chart background.  No vignette — Atlas wants edge-to-
    edge brightness so the dense graphics own the frame."""
    top = pal.get("panel_top", (252, 252, 252))
    bot = pal.get("panel_bot", (228, 228, 232))
    bg = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(1, H - 1)
        e = t * t * (3 - 2 * t)
        d.line([(0, y), (W, y)], fill=(
            int(top[0] + (bot[0] - top[0]) * e),
            int(top[1] + (bot[1] - top[1]) * e),
            int(top[2] + (bot[2] - top[2]) * e)))
    bg = bg.convert("RGBA")
    # subtle dot grid (only used in dataviz mode — explainer literal)
    dot = Image.new("L", (W, H), 0)
    dd = ImageDraw.Draw(dot)
    step = max(60, W // 30)
    for xx in range(step, W, step):
        for yy in range(step, H, step):
            dd.point((xx, yy), fill=255)
    dot = dot.filter(ImageFilter.GaussianBlur(0.6))
    dots_rgba = Image.merge("RGBA",
        (Image.new("L", (W, H), 0),
         Image.new("L", (W, H), 0),
         Image.new("L", (W, H), 0),
         dot.point(lambda p: 22 if p > 200 else 0)))
    bg = Image.alpha_composite(bg, dots_rgba)
    # very soft top-corner warm bloom — implies a studio over-shoulder
    key = pal.get("key_light", (255, 240, 235, 28))
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse(
        [-W * 0.30, -H * 0.40, W * 0.55, H * 0.50], fill=int(key[3]))
    glow = glow.filter(ImageFilter.GaussianBlur(int(W * 0.18)))
    light = Image.new("RGBA", (W, H), tuple(key[:3]) + (0,))
    light.putalpha(glow)
    bg = Image.alpha_composite(bg, light)
    vstr = int(pal.get("vignette_strength", 35))
    return apply_vignette(bg, strength=vstr, blur=int(W * 0.22))


def _make_dark_panel_bg(W: int, H: int, pal: dict) -> Image.Image:
    """MIDNIGHT PACIFIC — cool desaturated cinematic panel.  Honors
    look.panel_top / panel_bot / key_light so a channel can swap the
    base hue and key-light tint without touching the renderer.  This
    is the BACKWARDS-COMPATIBLE path: with no channel active, the
    palette returns the original navy + warm bloom defaults."""
    top = pal.get("panel_top", (10, 18, 36))
    bot = pal.get("panel_bot", (6, 8, 14))
    bg = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(1, H - 1)
        e = t * t * (3 - 2 * t)
        d.line([(0, y), (W, y)], fill=(
            int(top[0] + (bot[0] - top[0]) * e),
            int(top[1] + (bot[1] - top[1]) * e),
            int(top[2] + (bot[2] - top[2]) * e)))
    bg = bg.convert("RGBA")
    # cinematic key-light bloom
    glow = Image.new("L", (W, H), 0)
    cx, cy, rad = W // 2, int(H * 0.35), int(W * 0.50)
    ImageDraw.Draw(glow).ellipse(
        [cx - rad, cy - rad, cx + rad, cy + rad], fill=66)
    glow = glow.filter(ImageFilter.GaussianBlur(int(W * 0.12)))
    key = pal.get("key_light", (60, 110, 175, 36))
    light = Image.new("RGBA", (W, H), tuple(key[:3]) + (0,))
    light.putalpha(Image.eval(glow,
        lambda p: int(p * (key[3] / 66.0))))
    bg = Image.alpha_composite(bg, light)
    # film grain
    try:
        noise = Image.effect_noise((W, H), 14).convert("L")
        tint = Image.new("RGBA", (W, H), (150, 170, 210, 14))
        bg = Image.alpha_composite(
            bg, Image.composite(tint,
                Image.new("RGBA", (W, H), (0, 0, 0, 0)),
                noise.point(lambda p: 12 if p > 150 else 0)))
    except Exception:                                       # noqa: BLE001
        pass
    vstr = int(pal.get("vignette_strength", 95))
    return apply_vignette(bg, strength=vstr, blur=int(W * 0.18))


def make_navy_grid_bg(W: int, H: int) -> Image.Image:
    """Build the canonical 'data template' background.

    Dispatches on Look DNA `bg_mode`:
      • dark_panel    — cool cinematic navy panel (Midnight Pacific,
                        default for legacy / no-channel renders)
      • paper_scan    — warm sepia parchment (Amber Chronicles)
      • white_clinical — bright editorial dataviz (Atlas Explained)
      • matte_flat    — solid dark flat surface (dark-mode channels)

    This is the SINGLE point that 5+ data templates inherit — timeline,
    map_reveal, comparison, stat_dashboard, process_diagram, chapter_
    marker, photo_collage — so a channel's bg personality flows through
    all of them without touching their renderers.  When NO channel is
    active, falls back to the legacy per-scene StylePack palette so
    existing renders stay byte-identical.
    """
    pal = look_palette()
    bg_mode = pal.get("bg_mode")
    if bg_mode == "paper_scan":
        return _make_paper_scan_bg(W, H, pal)
    if bg_mode == "white_clinical":
        return _make_clinical_white_bg(W, H, pal)
    if bg_mode == "matte_flat":
        # flat matte panel — pure dark surface, no bloom, no grain
        top = pal.get("panel_top", (16, 16, 22))
        bg = Image.new("RGBA", (W, H), tuple(top) + (255,))
        vstr = int(pal.get("vignette_strength", 100))
        return apply_vignette(bg, strength=vstr, blur=int(W * 0.18))
    if bg_mode == "dark_panel":
        return _make_dark_panel_bg(W, H, pal)

    # ── LEGACY PATH (no channel active) ────────────────────────────
    # Per-scene bg sourced from the picked StylePack palette so the
    # whole card (bg + accent + ink) reads as ONE coherent palette per
    # scene — not a random rotation. 8 palettes × recency tracking =
    # visible variation while staying inside one documentary identity.
    _pk = scene_pack()
    _v = {"top": _pk.p["bg_top"], "bot": _pk.p["bg_bot"],
          "key": _pk.p["key"],    "key_y": _pk.p["key_y"]}
    top, bot = _v["top"], _v["bot"]
    bg = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(1, H - 1)
        e = t * t * (3 - 2 * t)                # smoothstep — richer falloff
        d.line([(0, y), (W, y)], fill=(
            int(top[0] + (bot[0] - top[0]) * e),
            int(top[1] + (bot[1] - top[1]) * e),
            int(top[2] + (bot[2] - top[2]) * e)))
    bg = bg.convert("RGBA")
    # soft cinematic key-light bloom — position + tone vary per scene too
    glow = Image.new("L", (W, H), 0)
    gd = ImageDraw.Draw(glow)
    cx, cy, rad = W // 2, int(H * _v["key_y"]), int(W * 0.50)
    gd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=66)
    glow = glow.filter(ImageFilter.GaussianBlur(int(W * 0.12)))
    light = Image.new("RGBA", (W, H), _v["key"])
    clear = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bg = Image.alpha_composite(bg, Image.composite(light, clear, glow))
    # NO blueprint grid — a clean deep gradient + bloom reads premium. A
    # whisper of fine film grain adds richness without any 'graph paper'.
    try:
        import random as _rnd
        noise = Image.effect_noise((W, H), 14).convert("L")
        tint = Image.new("RGBA", (W, H), (150, 170, 210, 14))
        bg = Image.alpha_composite(
            bg, Image.composite(tint, clear, noise.point(
                lambda p: 12 if p > 150 else 0)))
        _ = _rnd
    except Exception:                                       # noqa: BLE001
        pass
    return apply_vignette(bg, strength=95, blur=int(W * 0.18))


def apply_vignette(img: Image.Image, strength: int = 110,
                   blur: int = 220) -> Image.Image:
    """Soft radial vignette that darkens the corners. Cinematic edge
    falloff. Returns a new image (does not mutate)."""
    W, H = img.size
    vg = Image.new("L", (W, H), 255)
    ImageDraw.Draw(vg).ellipse(
        [-W // 5, -H // 5, W + W // 5, H + H // 5], fill=0)
    vg = vg.filter(ImageFilter.GaussianBlur(blur))
    dark = Image.new("RGBA", (W, H), (0, 0, 0, strength))
    return Image.alpha_composite(img,
        Image.composite(dark, Image.new("RGBA", (W, H), (0, 0, 0, 0)), vg))


def draw_headline(
    d: ImageDraw.ImageDraw,
    text: str,
    font_obj,
    accent_rgb: tuple,
    y: int,
    canvas_w: int,
    with_brackets: bool = True,
    bracket_len: int = 40,
    bracket_gap: int = 20,
) -> tuple[int, int, int, int]:
    """Draw the canonical centred headline with optional theme-accent
    brackets on either side. Returns the bbox (x0, y0, x1, y1) of the
    actual text drawn (so callers can position an underline below it
    or measure layout space).

    Used by timeline, map_reveal, comparison, stat_dashboard,
    photo_collage so the headline pattern is identical everywhere.

    TEXT-SAFETY (general): the headline is shrink-to-fit. If the supplied
    font makes it wider than the safe text column, the font is reduced
    (reloaded from the same face) until it fits; if it still can't fit at
    the floor size, it's word-clipped with an ellipsis. So a long headline
    like 'THE CONGRESS FOR CULTURAL FREEDOM' never renders as
    '...CULTURAL FREE'."""
    text = (text or "").upper()
    # safe text column leaves room for the accent brackets + margins.
    safe_w = int(canvas_w * 0.84)
    if with_brackets:
        safe_w -= 2 * (bracket_len + bracket_gap)
    safe_w = max(120, safe_w)
    if text:
        try:                                               # shrink same face
            _path, _sz = font_obj.path, int(font_obj.size)
            while _sz > 22 and d.textlength(text, font=font_obj) > safe_w:
                _sz -= 4
                font_obj = ImageFont.truetype(_path, _sz)
        except Exception:                                  # noqa: BLE001
            pass
        if d.textlength(text, font=font_obj) > safe_w:     # floor: word-clip
            while text and d.textlength(text + "…", font=font_obj) > safe_w:
                parts = text.rsplit(" ", 1)
                text = parts[0] if len(parts) > 1 else text[:-1]
            text = (text + "…") if text else "…"
    bb = d.textbbox((0, 0), text, font=font_obj)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    tx = (canvas_w - tw) // 2
    # BACKGROUND-AWARE INK (readability): on a light parchment / white card the
    # headline must be DARK, not the near-white TEXT_PRIMARY (the "HOW COPPER
    # STOPS SLUGS" was rendered near-white on cream). pick_ink/ safe_accent_on
    # read the active Look-DNA bg_mode and guarantee contrast on every theme.
    _cardbg = card_bg_rgb()
    _ink = pick_ink(_cardbg)
    _brk = safe_accent_on(_cardbg, accent_rgb)
    if with_brackets and text:
        bracket_y = y + th // 2 + 6
        d.line([(tx - bracket_gap - bracket_len, bracket_y),
                (tx - bracket_gap, bracket_y)],
               fill=with_alpha(_brk, 220), width=3)
        d.line([(tx + tw + bracket_gap, bracket_y),
                (tx + tw + bracket_gap + bracket_len, bracket_y)],
               fill=with_alpha(_brk, 220), width=3)
    d.text((tx, y), text, fill=with_alpha(_ink, 255), font=font_obj)
    return (tx, y, tx + tw, y + th)


def draw_accent_underline(
    d: ImageDraw.ImageDraw,
    centre_x: int,
    y: int,
    width: int,
    accent_rgb: tuple,
    thickness: int = 4,
    alpha: int = 220,
    gap: int = 12,
) -> None:
    """Short accent line centred under an element. Used by every
    template that needs the "this section is anchored here" cue.

    ``gap`` is added to the supplied ``y`` so the line sits a few pixels
    BELOW where the caller would otherwise place it. The default 12 px
    is the global breathing-room fix -- every existing caller now gets
    a proper editorial gap between the headline glyphs and the accent
    bar without needing per-callsite edits. Pass ``gap=0`` to opt out
    when the caller already includes its own gap (e.g. drawing under a
    badge / box where 0 is desired)."""
    y_drawn = int(y) + int(gap)
    d.rectangle(
        [centre_x - width // 2, y_drawn,
         centre_x + width // 2, y_drawn + thickness],
        fill=with_alpha(accent_rgb, alpha))


def rtl_panel(text: str, panel_left: int, panel_right: int, *,
              padding: int = 32, bar_w: int = 6,
              frame_w: int = 1920) -> dict:
    """Return the RTL-aware geometry for a panel that displays `text`.

    Templates call this ONCE per text element and use the returned dict
    to position the accent bar, anchor the text, and pick PIL anchor=.
    For Latin / CJK text the geometry is byte-identical to the existing
    LEFT-anchored layout so existing renders never regress.  For Arabic
    / Hebrew / Urdu the accent bar mirrors to the right edge, text
    anchor flips to "ra", and the text-X moves to (panel_right - padding).

    Returns:
        {
          "is_rtl":     bool,
          "anchor":     "la" (LTR) or "ra" (RTL),
          "text_x":     int   — where to place the text X (use with anchor)
          "bar_x1":     int   — left edge of accent bar
          "bar_x2":     int   — right edge of accent bar
          "padding":    int   — same as input, for caller convenience
        }

    The accent bar always sits ON the panel edge the eye enters from --
    left for LTR (we read left-to-right), right for RTL.
    """
    try:
        from .. import lang as _lang
        is_rtl = _lang.is_rtl(text or "")
    except Exception:                                          # noqa: BLE001
        is_rtl = False
    if is_rtl:
        return {
            "is_rtl":  True,
            "anchor":  "ra",
            "text_x":  int(panel_right - padding),
            "bar_x1":  int(panel_right - bar_w),
            "bar_x2":  int(panel_right),
            "padding": padding,
        }
    return {
        "is_rtl":  False,
        "anchor":  "la",
        "text_x":  int(panel_left + padding),
        "bar_x1":  int(panel_left),
        "bar_x2":  int(panel_left + bar_w),
        "padding": padding,
    }


def rtl_bullet_x(text: str, list_left: int, list_right: int, *,
                 padding: int = 30, dot_w: int = 12) -> dict:
    """RTL-aware geometry for a single bulleted list ITEM.

    Returns:
        {
          "is_rtl":     bool,
          "anchor":     PIL anchor for the text line ("la" / "ra"),
          "text_x":     X coordinate to draw the text at (paired w/ anchor),
          "dot_x1":     left edge of the accent dot rectangle,
          "dot_x2":     right edge of the accent dot rectangle,
        }
    """
    try:
        from .. import lang as _lang
        is_rtl = _lang.is_rtl(text or "")
    except Exception:                                          # noqa: BLE001
        is_rtl = False
    if is_rtl:
        return {
            "is_rtl":  True,
            "anchor":  "ra",
            "text_x":  int(list_right - padding),
            "dot_x1":  int(list_right),
            "dot_x2":  int(list_right + dot_w),
        }
    return {
        "is_rtl":  False,
        "anchor":  "la",
        "text_x":  int(list_left + padding),
        "dot_x1":  int(list_left),
        "dot_x2":  int(list_left + dot_w),
    }


def wrap_text(text: str, max_chars_per_line: int, max_lines: int) -> list[str]:
    """Word-wrap helper — used by evidence, timeline event labels,
    comparison values, dossier bodies, future templates.  Splits at
    word boundary; words longer than max_chars are hard-cut with an
    ellipsis so they always fit.

    Script-aware: CJK (Japanese / Korean / Chinese) has no spaces, so
    the word-split would return one giant "word" that always overflows.
    For CJK we wrap CHARACTER-by-character with a soft preference for
    breaking after CJK punctuation (、。・！？).  Latin / RTL languages
    keep the original word-based wrapping (unchanged byte-for-byte)."""
    if not text:
        return []
    # CJK detection — any codepoint in Hiragana / Katakana / Hangul /
    # CJK Unified Ideographs triggers char-wrap.
    is_cjk = any(0x3040 <= ord(c) <= 0xD7AF or 0x4E00 <= ord(c) <= 0x9FFF
                 for c in text)
    if is_cjk:
        cut_punct = set("、。・！？；：「」『』")
        lines: list[str] = []
        cur = ""
        for ch in text:
            cur += ch
            if ch in cut_punct or len(cur) >= max_chars_per_line:
                lines.append(cur.strip())
                cur = ""
        if cur.strip():
            lines.append(cur.strip())
        return lines[:max_lines]
    # Latin / RTL: original word-based wrap
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= max_chars_per_line:
            cur = (cur + " " + w) if cur else w
        else:
            if cur:
                lines.append(cur)
            cur = w if len(w) <= max_chars_per_line \
                  else w[: max_chars_per_line - 1] + "…"
    if cur:
        lines.append(cur)
    return lines[:max_lines]


# =========================================================================
# MOTION DRIFT — small ambient sin-wave drift for "alive" feel.
# Different sin periods per layer → parallax depth without footage
# actually moving. Returns ffmpeg-expression strings.
# =========================================================================

# Period table — different layers use different periods so they
# breathe out of sync and read as DEPTH not synchronised wobble.
DRIFT_PERIODS = {
    "frame":   (4.6, 3.9),    # (x_period, y_period) seconds
    "title":   (2.6, 4.2),
    "badge":   (5.1, 3.4),
    "marker":  (3.8, 5.7),
    "caption": (4.4, 3.1),
}


def ff_drift_x(layer: str, t0: float, amp: int = 3) -> str:
    """ffmpeg expression for subtle x-drift after element appears.
    `t0` is when the element finished its reveal; drift starts then."""
    period_x = DRIFT_PERIODS.get(layer, (4.0, 4.0))[0]
    return f"+{amp}*sin((t-{t0:.2f})/{period_x})"


def ff_drift_y(layer: str, t0: float, amp: int = 2) -> str:
    period_y = DRIFT_PERIODS.get(layer, (4.0, 4.0))[1]
    return f"+{amp}*sin((t-{t0:.2f})/{period_y})"


# =========================================================================
# VARIATION SEED — deterministic per-scene variation so the same scene
# always renders the same layout (cacheable, debuggable) but DIFFERENT
# scenes get different variants.
# =========================================================================

# DOC_005 — per-VIDEO layout drift. scene_variant is deterministic on
# (scene_index, salt), so two DIFFERENT videos used to run the IDENTICAL
# layout sequence (e.g. lower-third positions 0,1,2,3,0,1… every video →
# template feel). A per-render variation salt (installed once at render
# start from the brief hash) mixes in so each video gets its OWN stable
# layout sequence — varied across videos, still deterministic within a
# video. Uses a STABLE hash (not Python's per-process hash()) so re-runs
# of the same video are identical. Unset (tools/tests) → legacy behaviour.
_VARIANT_SALT_INT = 0


def set_variant_salt(s: str) -> None:
    """Install a per-render layout-variation salt (e.g. the brief hash).
    Empty/falsy restores exact legacy variant sequences."""
    global _VARIANT_SALT_INT
    if s:
        import hashlib
        _VARIANT_SALT_INT = int(
            hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8], 16)
    else:
        _VARIANT_SALT_INT = 0


def scene_variant(scene_index: int, n_variants: int,
                  salt: str = "") -> int:
    """Pick a variant index 0..n-1 deterministically from the scene
    index. Use this to spread layouts across scenes without randomness
    (so renders are stable across re-runs). `salt` lets a template
    have multiple independent variation axes (e.g. variant for
    position, variant for typography weight). A per-render variation
    salt (set_variant_salt) shifts the sequence per video so layouts
    don't repeat identically across different videos."""
    n = max(1, n_variants)
    if not _VARIANT_SALT_INT:
        # legacy (no per-video drift) — byte-identical to the original.
        seed = (scene_index * 2654435761) ^ hash(salt)
        return abs(seed) % n
    # DOC_005 per-video layout drift (fixed 2026-05-30: the old end-XOR
    # collided on low bits for small n → different videos got IDENTICAL
    # sequences). A salt-keyed AFFINE sequence `(i*step + phase) % n` with
    # `step` coprime to n gives: NO same-variant back-to-back within a
    # video AND a well-distributed DISTINCT sequence per video. Deterministic
    # (sha1, stable across re-runs) and axis-independent (the `salt` arg
    # picks a different step/phase per variation axis).
    import hashlib
    import math
    salt_h = (int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
              if salt else 0)
    sd = (_VARIANT_SALT_INT ^ (salt_h * 2654435761)) & 0x7FFFFFFF
    steps = [s for s in range(1, n) if math.gcd(s, n) == 1] or [1]
    step = steps[(sd >> 8) % len(steps)]
    phase = sd % n
    return (scene_index * step + phase) % n


def stable_jitter(scene_index: int, salt: str, lo: float,
                  hi: float) -> float:
    """Deterministic float in [lo, hi] keyed on (scene_index, salt).
    Use for tiny per-scene drift (±2px x offset, ±3deg rotation) so
    cards feel hand-placed rather than mathematically aligned."""
    rng = random.Random((scene_index, salt))
    return lo + rng.random() * (hi - lo)
