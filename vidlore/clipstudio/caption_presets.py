"""Caption preset registry — the SINGLE source of truth for how ClipStudio captions look.

ClipStudio burns two kinds of caption:
  1. the main NARRATION caption (the engine's kinetic word-by-word caption, styled from the
     theme's ``caption`` dict + ``accent`` colour — see vidlore/captions.py:write_ass), and
  2. the real-audio BREAKOUT dialogue caption (a separate ASS overlay burned by
     build._breakout_caption_ass).

Both were hard-coded to one appearance. This module defines a small family of professionally
designed PRESETS; a render picks ONE preset and BOTH caption kinds are styled from it so a video
carries one consistent design (never a random mix). Every ASS styling constant lives here — the
portal, build, and breakout functions all read from this registry, they never hand-roll styling.

Resolution order (resolve_style): explicit value → VIDLORE_CLIPSTUDIO_CAPTION_STYLE env → default
('professional'). An unknown value falls back to 'professional' and is reported (invalid=True) so
the caller can log it. Enabled/disabled is separate (captions_enabled / VIDLORE_CLIPSTUDIO_CAPTIONS).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_STYLE = "professional"
VALID_STYLES = ("professional", "minimal", "cinematic", "documentary", "focus")


def _ass_color(rgb, alpha: int = 0x00) -> str:
    """ASS colour string &HAABBGGRR. alpha 0x00 = fully opaque … 0xFF = fully transparent."""
    r, g, b = (int(c) & 0xFF for c in rgb)
    return f"&H{alpha & 0xFF:02X}{b:02X}{g:02X}{r:02X}"


def _fmt(x) -> str:
    """Compact ASS number (2 -> '2', 2.5 -> '2.5')."""
    return str(int(x)) if float(x) == int(x) else str(x)


@dataclass(frozen=True)
class CaptionPreset:
    name: str
    label: str                       # portal display name
    blurb: str                       # one-line design summary (portal + docs)
    # ---- main narration caption (maps to captions.write_ass `style` dict + `accent`) ----
    font: str
    size: int
    bold: bool
    primary_rgb: tuple               # base text colour
    accent_rgb: tuple                # active / emphasis word colour
    outline_rgb: tuple
    outline_w: float
    shadow: float
    border_style: int                # 1 = outline+drop-shadow · 3 = opaque box (uses back)
    back_rgb: tuple
    back_alpha: int                  # 0x00 opaque … 0xFF transparent
    margin_v: int
    max_lines: int
    # ---- breakout dialogue caption variant (same design family, gold karaoke fill) ----
    bk_font: str
    bk_size: int
    bk_sung_rgb: tuple               # word already spoken (karaoke-filled)
    bk_unsung_rgb: tuple             # word not yet spoken
    bk_outline_rgb: tuple
    bk_outline_w: float
    bk_shadow: float
    bk_border_style: int
    bk_back_alpha: int
    bk_margin_v: int

    # --- narration: theme caption dict + accent tuple (consumed by build_video → assemble) ---
    def theme_caption(self) -> tuple[dict, tuple]:
        cap = {
            "font": self.font,
            "size": self.size,
            "bold": self.bold,
            "primary": _ass_color(self.primary_rgb),
            "outline": _ass_color(self.outline_rgb),
            "back": _ass_color(self.back_rgb, self.back_alpha),
            "border_style": self.border_style,
            "outline_w": self.outline_w,
            "shadow": self.shadow,
            "margin_v": self.margin_v,
            "max_lines": self.max_lines,
            # a ClipStudio-selected preset is EXACT: write_ass must not let Look-DNA / legacy
            # subtitle_style silently re-skin its font / size / margin / weight / emphasis.
            "preset_locked": True,
        }
        return cap, tuple(self.accent_rgb)

    # --- breakout: the ASS [V4+ Styles] `Style: BK,...` line (same family, distinct karaoke) ---
    def breakout_style_line(self, *, margin_v: int | None = None) -> str:
        mv = self.bk_margin_v if margin_v is None else int(margin_v)
        return (
            "Style: BK,{font},{size},{sung},{unsung},{outline},{back},-1,0,{bs},{ow},{sh},2,"
            "120,120,{mv},1".format(
                font=self.bk_font, size=self.bk_size,
                sung=_ass_color(self.bk_sung_rgb),
                unsung=_ass_color(self.bk_unsung_rgb),
                outline=_ass_color(self.bk_outline_rgb),
                back=_ass_color((0, 0, 0), self.bk_back_alpha),
                bs=self.bk_border_style, ow=_fmt(self.bk_outline_w), sh=_fmt(self.bk_shadow),
                mv=mv))

    # --- portal preview: CSS-ish approximation of the real typography (compact realistic sample) ---
    def preview(self) -> dict:
        def _rgb(t):
            return f"rgb({int(t[0])},{int(t[1])},{int(t[2])})"
        # translucent backplate only when the preset draws a box (border_style 3), else a soft shadow
        if self.border_style == 3:
            box = f"rgba(0,0,0,{max(0.0, 1.0 - self.back_alpha / 255.0):.2f})"
            text_shadow = "none"
        else:
            box = "transparent"
            _sh = min(0.85, 0.35 + self.shadow * 0.18)
            _ow = max(0.5, self.outline_w * 0.45)
            text_shadow = (f"0 0 {_ow:.1f}px {_rgb(self.outline_rgb)}, "
                           f"0 1px 2px rgba(0,0,0,{_sh:.2f})")
        return {
            "color": _rgb(self.primary_rgb),
            "accent": _rgb(self.accent_rgb),
            "weight": "800" if self.bold else "500",
            "backplate": box,
            "text_shadow": text_shadow,
            "family": self.font,
        }


# ── the five presets ─────────────────────────────────────────────────────────────────────────
CAPTION_PRESETS: dict[str, CaptionPreset] = {
    # PROFESSIONAL — the new default. Premium movie-analysis / documentary look: bold clean white
    # sans, restrained warm-gold active word, crisp dark outline + soft shadow. Loud-proof.
    "professional": CaptionPreset(
        name="professional", label="Professional",
        blurb="Premium doc look · bold white sans · restrained gold active word · crisp outline",
        font="Arial", size=54, bold=True,
        primary_rgb=(255, 255, 255), accent_rgb=(255, 196, 84),
        outline_rgb=(12, 12, 12), outline_w=2.6, shadow=1.2, border_style=1,
        back_rgb=(0, 0, 0), back_alpha=0x82, margin_v=72, max_lines=2,
        bk_font="Arial Black", bk_size=62, bk_sung_rgb=(255, 200, 60), bk_unsung_rgb=(230, 230, 230),
        bk_outline_rgb=(0, 0, 0), bk_outline_w=3.4, bk_shadow=1.4, bk_border_style=1,
        bk_back_alpha=0x78, bk_margin_v=150),

    # MINIMAL — elegant white, thin outline + soft shadow, no box, near-white accent. Quietest.
    "minimal": CaptionPreset(
        name="minimal", label="Minimal",
        blurb="Elegant white · thin outline · no box · barely-there emphasis · least distracting",
        font="Arial", size=50, bold=True,
        primary_rgb=(248, 248, 248), accent_rgb=(244, 236, 214),
        outline_rgb=(18, 18, 18), outline_w=1.4, shadow=0.6, border_style=1,
        back_rgb=(0, 0, 0), back_alpha=0x9A, margin_v=66, max_lines=2,
        bk_font="Arial", bk_size=56, bk_sung_rgb=(250, 244, 224), bk_unsung_rgb=(210, 210, 210),
        bk_outline_rgb=(16, 16, 16), bk_outline_w=1.8, bk_shadow=0.8, bk_border_style=1,
        bk_back_alpha=0x9A, bk_margin_v=150),

    # CINEMATIC — refined film-subtitle: lighter weight than professional, soft warm accent, gentle
    # shadow, lifted higher so it always clears the cinematic letterbox bars. Balanced two lines.
    "cinematic": CaptionPreset(
        name="cinematic", label="Cinematic",
        blurb="Refined film subtitle · lighter weight · soft warm accent · sits above letterbox",
        font="Arial", size=52, bold=False,
        primary_rgb=(240, 244, 250), accent_rgb=(232, 200, 128),
        outline_rgb=(16, 20, 28), outline_w=1.8, shadow=1.0, border_style=1,
        back_rgb=(0, 0, 0), back_alpha=0x88, margin_v=104, max_lines=2,
        bk_font="Arial", bk_size=58, bk_sung_rgb=(236, 206, 132), bk_unsung_rgb=(224, 228, 234),
        bk_outline_rgb=(12, 16, 24), bk_outline_w=2.2, bk_shadow=1.2, bk_border_style=1,
        bk_back_alpha=0x82, bk_margin_v=176),

    # DOCUMENTARY — clean lower-third with a restrained translucent-dark backplate (border_style 3),
    # compact typography, strong readability for long-form analysis. Not a big social caption.
    "documentary": CaptionPreset(
        name="documentary", label="Documentary",
        blurb="Lower-third · translucent dark backplate · compact type · strong readability",
        font="Arial", size=46, bold=True,
        primary_rgb=(255, 255, 255), accent_rgb=(236, 206, 110),
        outline_rgb=(0, 0, 0), outline_w=0.6, shadow=0.0, border_style=3,
        back_rgb=(0, 0, 0), back_alpha=0x5A, margin_v=64, max_lines=2,
        bk_font="Arial", bk_size=52, bk_sung_rgb=(240, 210, 110), bk_unsung_rgb=(236, 236, 236),
        bk_outline_rgb=(0, 0, 0), bk_outline_w=0.8, bk_shadow=0.0, bk_border_style=3,
        bk_back_alpha=0x50, bk_margin_v=150),

    # FOCUS — word-synchronised emphasis dialled up: bright, energetic active-word gold for hooks,
    # bold base, strong outline. Still professional (no bounce/flash/scale-pulse/emoji/random colour).
    "focus": CaptionPreset(
        name="focus", label="Focus",
        blurb="Word-synced emphasis · bright active-word gold · energetic for hooks, still clean",
        font="Arial", size=56, bold=True,
        primary_rgb=(255, 255, 255), accent_rgb=(255, 178, 40),
        outline_rgb=(10, 10, 10), outline_w=2.8, shadow=1.2, border_style=1,
        back_rgb=(0, 0, 0), back_alpha=0x80, margin_v=74, max_lines=2,
        bk_font="Arial Black", bk_size=64, bk_sung_rgb=(255, 184, 40), bk_unsung_rgb=(226, 226, 226),
        bk_outline_rgb=(0, 0, 0), bk_outline_w=3.6, bk_shadow=1.4, bk_border_style=1,
        bk_back_alpha=0x74, bk_margin_v=150),
}


def resolve_style(value: str | None = None) -> tuple[CaptionPreset, bool]:
    """Resolve a caption preset. Order: explicit `value` → VIDLORE_CLIPSTUDIO_CAPTION_STYLE env →
    DEFAULT_STYLE ('professional'). Returns (preset, invalid) — `invalid` is True when a non-empty
    value did not name a known preset (the caller logs it and the default is used). Case/space
    tolerant; None/'' are NOT 'invalid' (they just mean 'use the default')."""
    raw = value
    if raw is None or not str(raw).strip():
        raw = os.environ.get("VIDLORE_CLIPSTUDIO_CAPTION_STYLE", "")
    key = str(raw or "").strip().lower()
    if not key:
        return CAPTION_PRESETS[DEFAULT_STYLE], False
    preset = CAPTION_PRESETS.get(key)
    if preset is None:
        return CAPTION_PRESETS[DEFAULT_STYLE], True
    return preset, False


def captions_enabled(value: bool | None = None) -> bool:
    """Resolve the caption ON/OFF flag: explicit bool wins; else VIDLORE_CLIPSTUDIO_CAPTIONS
    (0/false/no/off = OFF); default ON."""
    if value is not None:
        return bool(value)
    env = os.environ.get("VIDLORE_CLIPSTUDIO_CAPTIONS", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return True


def preset_choices() -> list[dict]:
    """Portal-facing list of presets (name, label, blurb, preview) in display order."""
    out = []
    for k in VALID_STYLES:
        p = CAPTION_PRESETS[k]
        out.append({"name": p.name, "label": p.label, "blurb": p.blurb, "preview": p.preview()})
    return out
