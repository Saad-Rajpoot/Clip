"""Theme presets — Vidlore's Crime / History / Modern / Minimalist / Standard.

This module is the SINGLE SOURCE OF TRUTH for how a theme paints the video.
Every renderer (assemble.drawtext, footage PIL cards, captions ASS styler,
slide fallback gradient) pulls its colours, fonts, card treatment and
overlay effects from here — so picking a theme in the UI changes the WHOLE
look of the video, not just the accent dot.

Backward compatibility: the original five keys (`grade`, `caption`, `grad`,
`accent`, `broll`) are still present on every theme so older callers keep
working. New rich keys (`palette`, `fonts`, `card_style`, `stab_style`,
`overlay_effects`, `description`, `preview_grad`) are layered on top.
"""
from __future__ import annotations

# ASS colours are &HAABBGGRR (alpha, blue, green, red), AA=00 opaque.
_CAPTION_BASE = dict(
    font="Arial",
    size=54,
    bold=True,
    border_style=1,
    outline_w=3,
    shadow=1,
    margin_v=70,
)

# Per-theme FONT CANDIDATE LISTS. The assemble/footage renderer walks
# each list in order and picks the first file that exists, falling back
# to the bundled VidloreSans so a render NEVER fails on a missing font.
# These lists deliberately mix Win + macOS + common Linux locations.
_FONT_MONO = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/courbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
]
_FONT_SERIF = [
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/Library/Fonts/Georgia.ttf",
    "C:/Windows/Fonts/georgiab.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf",
]
_FONT_GEO = [   # geometric / clean sans for "modern" titles
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_LIGHT = [   # ultra-light for "minimalist"
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "C:/Windows/Fonts/segoeuil.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

THEMES = {
    "crime": {
        # ---------- legacy keys (kept) ----------
        "grade": "eq=contrast=1.20:saturation=0.66:brightness=-0.03,"
                 "colorbalance=rs=-0.06:bs=0.10",
        "caption": {**_CAPTION_BASE, "primary": "&H00FFFFFF",
                    "outline": "&H00101010", "back": "&H99000000",
                    "size": 56, "font": "Courier New"},
        "grad": ((10, 14, 22), (40, 16, 18)),
        "accent": (220, 50, 50),
        "broll": "city night dark surveillance",
        # ---------- NEW rich spec ----------
        "description": ("A dark and intense theme perfect for true crime, "
                        "mystery, and investigative content with dramatic "
                        "visual elements."),
        "palette": {
            "primary": (220, 50, 50),         # blood red
            "secondary": (210, 170, 70),      # warning amber
            "bg": (10, 14, 22),
            "panel": (18, 22, 30),
            "text": (240, 240, 240),
            "muted": (140, 140, 150),
        },
        "fonts": {
            "display": _FONT_MONO,            # typewriter / surveillance feel
            "body": _FONT_MONO,
            "caption": _FONT_MONO,
        },
        "card_style": "dossier",              # typewriter document w/ classification
        "stab_style": "redact",               # boxed red underline like a redaction
        "overlay_effects": ["film_grain", "vignette_strong"],
        "preview_grad": ((10, 14, 22), (60, 18, 22)),
    },
    "history": {
        "grade": "eq=contrast=1.10:saturation=0.82:brightness=0.01,"
                 "colorbalance=rs=0.08:gs=0.03:bs=-0.09",
        "caption": {**_CAPTION_BASE, "primary": "&H00E8F2FF",
                    "outline": "&H00203040", "back": "&H73000000",
                    "font": "Georgia"},
        "grad": ((28, 22, 14), (58, 44, 26)),
        "accent": (210, 170, 90),
        "broll": "ancient ruins archival sepia",
        "description": ("A classic and timeless theme ideal for historical "
                        "documentaries, educational content, and period pieces."),
        "palette": {
            "primary": (210, 170, 90),        # antique gold
            "secondary": (140, 70, 40),       # burnt sienna
            "bg": (28, 22, 14),
            "panel": (44, 34, 22),
            "text": (240, 230, 210),
            "muted": (160, 140, 110),
        },
        "fonts": {
            "display": _FONT_SERIF,
            "body": _FONT_SERIF,
            "caption": _FONT_SERIF,
        },
        "card_style": "parchment",            # archive-exhibit, aged paper
        "stab_style": "classical",            # serif underline, dignified
        "overlay_effects": ["paper_grain", "vignette_soft"],
        "preview_grad": ((40, 30, 18), (90, 70, 38)),
    },
    "modern": {
        "grade": "eq=contrast=1.14:saturation=1.16:brightness=0.02,"
                 "colorbalance=bs=0.06",
        "caption": {**_CAPTION_BASE, "primary": "&H00FFFFFF",
                    "outline": "&H00C04000", "back": "&H66000000",
                    "size": 58, "font": "Helvetica Neue"},
        "grad": ((16, 20, 30), (24, 40, 64)),
        "accent": (0, 160, 255),
        "broll": "modern city skyline tech",
        "description": ("A sleek and contemporary theme featuring clean lines "
                        "and vibrant colours, perfect for tech, lifestyle, "
                        "and business content."),
        "palette": {
            "primary": (0, 160, 255),         # electric cyan
            "secondary": (255, 110, 64),      # warm coral pop
            "bg": (16, 20, 30),
            "panel": (24, 32, 48),
            "text": (245, 248, 255),
            "muted": (150, 165, 185),
        },
        "fonts": {
            "display": _FONT_GEO,
            "body": _FONT_GEO,
            "caption": _FONT_GEO,
        },
        "card_style": "glass",                # frosted panel with bright edge
        "stab_style": "chip",                 # pill/chip with glow
        "overlay_effects": ["edge_glow"],
        "preview_grad": ((14, 24, 48), (20, 80, 150)),
    },
    "minimalist": {
        "grade": "eq=contrast=1.04:saturation=0.94:brightness=0.0",
        "caption": {**_CAPTION_BASE, "primary": "&H00FFFFFF",
                    "outline": "&H00000000", "back": "&H00000000",
                    "border_style": 1, "outline_w": 2, "shadow": 0,
                    "size": 50, "font": "Helvetica"},
        "grad": ((18, 18, 20), (32, 32, 36)),
        "accent": (235, 235, 235),
        "broll": "abstract texture minimal",
        "description": ("A clean and simple theme with subtle animations, "
                        "ideal for corporate presentations, product showcases, "
                        "and educational content."),
        "palette": {
            "primary": (235, 235, 235),       # near-white
            "secondary": (240, 80, 80),       # one accent pop, that's it
            "bg": (18, 18, 20),
            "panel": (28, 28, 32),
            "text": (240, 240, 240),
            "muted": (140, 140, 145),
        },
        "fonts": {
            "display": _FONT_LIGHT,
            "body": _FONT_LIGHT,
            "caption": _FONT_LIGHT,
        },
        "card_style": "card_clean",           # plain white card, no chrome
        "stab_style": "underline",            # one thin line, no box
        "overlay_effects": [],                # nothing — that's the point
        "preview_grad": ((20, 20, 22), (44, 44, 50)),
    },
    "standard": {
        "grade": "eq=contrast=1.08:saturation=1.0:brightness=0.0",
        "caption": {**_CAPTION_BASE, "primary": "&H00FFFFFF",
                    "outline": "&H00202020", "back": "&H73000000"},
        "grad": ((20, 24, 30), (36, 42, 52)),
        "accent": (90, 150, 220),
        "broll": "documentary landscape",
        "description": ("A versatile, well-balanced theme with neutral styling "
                        "that adapts seamlessly to any content category."),
        "palette": {
            "primary": (90, 150, 220),        # balanced blue
            "secondary": (240, 180, 80),      # complementary warm
            "bg": (20, 24, 30),
            "panel": (32, 38, 48),
            "text": (240, 240, 240),
            "muted": (150, 158, 170),
        },
        "fonts": {
            "display": [],   # empty -> renderer uses bundled VidloreSans
            "body": [],
            "caption": [],
        },
        "card_style": "balanced",
        "stab_style": "balanced",
        "overlay_effects": [],
        "preview_grad": ((24, 36, 56), (40, 80, 120)),
    },
}


# ffmpeg filter snippets each "overlay_effects" name maps to. Each value
# is a comma-joined chain that runs AFTER the theme's base colour grade
# on every regular footage clip (vintage / archival clips skip it — they
# already have their own period look). Keep these CONSERVATIVE — a
# noticeable but not gimmicky touch per theme:
#   • film_grain      – temporal noise, "35mm dossier surveillance" feel
#   • vignette_strong – pronounced corner darken, claustrophobic
#   • vignette_soft   – gentle corner darken, dignified
#   • paper_grain     – soft static noise + slight desat, archive paper
#   • edge_glow       – tiny brightness lift + light sharpen, tech punch
_EFFECT_FILTERS = {
    "film_grain":      "noise=alls=18:allf=t",
    "vignette_strong": "vignette=PI/4:eval=init",
    "vignette_soft":   "vignette=PI/6:eval=init",
    "paper_grain":     "noise=alls=8:allf=t,eq=saturation=0.92",
    "edge_glow":       "eq=brightness=0.015:saturation=1.06,"
                       "unsharp=5:5:0.4:3:3:0",
}


def effects_filters(t: dict) -> str:
    """Resolve a theme's `overlay_effects` list into an ffmpeg filter
    chain segment. Returns '' (empty) for themes with no effects, or
    ',filter1,filter2' (with leading comma) ready to be appended to an
    existing filter chain like the colour grade — so callers can write::

        grade = theme["grade"] + effects_filters(theme)

    Unknown effect names are silently skipped (a stale config never
    breaks a render)."""
    parts = []
    for name in (t.get("overlay_effects") or []):
        f = _EFFECT_FILTERS.get(name)
        if f:
            parts.append(f)
    return ("," + ",".join(parts)) if parts else ""


def theme(name: str) -> dict:
    """Return the rich theme dict, falling back to 'standard' for unknown
    names so a render NEVER crashes on a typo or legacy theme value."""
    return THEMES.get(name, THEMES["standard"])


def theme_names() -> list[str]:
    """Stable order for the UI picker — matches Vidlore's grid order."""
    return ["crime", "history", "modern", "minimalist", "standard"]


def theme_meta(name: str) -> dict:
    """Lightweight subset for the web UI picker — just what the card needs."""
    t = theme(name)
    return {
        "name": name,
        "title": name.capitalize() + " theme",
        "description": t.get("description", ""),
        "primary": t["palette"]["primary"] if "palette" in t else t["accent"],
        "secondary": (t["palette"]["secondary"] if "palette" in t
                      else t["accent"]),
        "grad": t.get("preview_grad", t["grad"]),
    }
