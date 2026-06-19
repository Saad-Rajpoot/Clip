"""Dark-niche text-card style gating (reusable, engine-level).

Problem this solves permanently: in a dark documentary (spy / mystery / true-crime
/ intelligence / covert), a full-screen near-WHITE "statement" text card
(footage._render_statement_card) breaks the crushed-black visual language with a
high-key flash. This module decides, for any scene's full-screen text card:

  • whether the niche is a DARK niche (cinematic low-key world), and
  • whether a requested LIGHT/high-key card should be allowed or REPLACED with a
    dark variant (text-on-black mono / muted parchment / restrained spotlight), and
  • allows a bright CONTRAST beat only when the editorial recipe explicitly permits
    it AND it is still rare in the video — always logged as an intentional beat.

It does NOT change card geometry/typography — only which lightness VARIANT is used
for a dark-niche video. Pure logic → fully unit-testable. The footage.py hook
wraps it defensively and falls back to the old behaviour on any error.
"""
from __future__ import annotations

import re

# Niches whose established look is low-key/cinematic — a bright full-frame card
# is jarring here unless deliberately authorised.
_DARK_NICHES = {"spy", "mystery", "crime", "true_crime", "intelligence",
                "covert", "espionage", "spy_intel", "geopolitics", "war",
                "conspiracy", "noir", "investigation", "investigative"}

# Palettes that are themselves dark/low-key (all four MG palettes use a dark bg,
# but ember/cold/parchment especially read "dark cinematic").
_DARK_PALETTES = {"ember_red", "cold_steel", "parchment_sepia", "amber_gold"}

# Full-screen text-card kinds that render LIGHT / near-white (high-key).
_LIGHT_CARD_KINDS = {"statement", "big_text", "light_statement", "airy_statement"}
# The dark replacement kind (footage._render_text_on_black_card).
DARK_TEXT_CARD_KIND = "text_on_black"

# At most this many bright contrast beats per video (rarity guard).
MAX_CONTRAST_BEATS = 1


def _norm(s) -> str:
    return re.sub(r"[^a-z_]+", "_", str(s or "").strip().lower()).strip("_")


def is_dark_niche(niche, *, palette: str | None = None) -> bool:
    n = _norm(niche)
    if n in _DARK_NICHES:
        return True
    if any(k in n for k in ("spy", "crime", "myster", "covert", "intel", "espionage")):
        return True
    # a dark palette on an unknown niche still wants dark cards (defensive)
    return False


def is_light_card_kind(kind) -> bool:
    return _norm(kind) in _LIGHT_CARD_KINDS


def resolve_card_variant(niche, kind, *, palette: str | None = None,
                         recipe_allows_contrast: bool = False,
                         contrast_used: int = 0) -> dict:
    """Decide the lightness variant for a full-screen text card.

    Returns {allow_light, kind, contrast_beat, reason}:
      • allow_light=True  → keep the requested (light) card.
      • allow_light=False → render the dark variant; `kind` is overridden to
        DARK_TEXT_CARD_KIND so the caller routes to the dark renderer.
      • contrast_beat=True → a bright card was deliberately permitted (rare) and
        should be logged as an intentional contrast beat."""
    k = _norm(kind)
    dark = is_dark_niche(niche, palette=palette)
    if not is_light_card_kind(k):
        return {"allow_light": False, "kind": kind, "contrast_beat": False,
                "reason": "not a light full-frame card — unchanged"}
    if not dark:
        return {"allow_light": True, "kind": kind, "contrast_beat": False,
                "reason": f"niche {_norm(niche)} not dark — light card allowed"}
    # dark niche + light card requested
    if recipe_allows_contrast and contrast_used < MAX_CONTRAST_BEATS:
        return {"allow_light": True, "kind": kind, "contrast_beat": True,
                "reason": (f"dark niche {_norm(niche)}: recipe-authorised RARE "
                           f"contrast beat ({contrast_used + 1}/{MAX_CONTRAST_BEATS})")}
    return {"allow_light": False, "kind": DARK_TEXT_CARD_KIND, "contrast_beat": False,
            "reason": (f"dark niche {_norm(niche)}: light card → dark variant "
                       f"({DARK_TEXT_CARD_KIND}) to preserve low-key look")}


def palette_card_compatible(palette: str, light_card: bool) -> tuple[bool, str]:
    """A light/high-key card on a dark palette is a compatibility mismatch."""
    if light_card and _norm(palette) in _DARK_PALETTES:
        return False, f"light card on dark palette {palette} (mismatch)"
    return True, "ok"
