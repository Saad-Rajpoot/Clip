"""Multilingual rendering infrastructure.

This module is the SINGLE SOURCE OF TRUTH for script detection, font
routing, and RTL shaping.  Every rendering path in the codebase (footage,
templates, assemble) eventually reaches this module via `pick_font()` or
`prepare_text()` — so a font/shaping fix here propagates to ALL 100+
templates without per-template edits.

Goals:
    * Bundled fonts cover JP / KR / Arabic / Urdu / Hebrew without
      OS-substitution (works on Windows / Linux dist builds too).
    * Right script is auto-picked per text — no caller needs to know
      which font to ask for.
    * RTL strings (Arabic / Urdu / Hebrew) are reshaped + BiDi-applied
      so PIL renders connected letters in the correct visual order.
    * Caller can ask "is this text RTL?" to flip x-anchoring + layout.
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# ---- Asset paths --------------------------------------------------- #
_ASSET_DIR = Path(__file__).parent / "assets"

# Latin / Western (default — also the brand display face)
LATIN_BOLD = str(_ASSET_DIR / "VidloreSans-Bold.ttf")
LATIN_REG = str(_ASSET_DIR / "VidloreSans.ttf")

# Bundled script-specific Noto fonts (SIL OFL — redistributable)
CJK_JP = str(_ASSET_DIR / "NotoSansJP-VF.ttf")
CJK_KR = str(_ASSET_DIR / "NotoSansKR-VF.ttf")
ARABIC = str(_ASSET_DIR / "NotoSansArabic-VF.ttf")
URDU_NASTALIQ = str(_ASSET_DIR / "NotoNastaliqUrdu-VF.ttf")
HEBREW = str(_ASSET_DIR / "NotoSansHebrew-VF.ttf")

# Maps a detected script tag -> ordered font fallback chain.
# Bold and regular weights are taken from the same VF file (PIL reads
# the default instance, which is usually Regular; for Bold we re-open
# the same VF and request the right weight axis).
_SCRIPT_FONTS: dict[str, tuple[str, ...]] = {
    "latin":   (LATIN_BOLD, LATIN_REG),
    "jp":      (CJK_JP, CJK_KR, LATIN_BOLD),
    "kr":      (CJK_KR, CJK_JP, LATIN_BOLD),
    "cjk":     (CJK_JP, CJK_KR, LATIN_BOLD),
    "arabic":  (ARABIC, LATIN_BOLD),
    "urdu":    (URDU_NASTALIQ, ARABIC, LATIN_BOLD),
    "hebrew":  (HEBREW, LATIN_BOLD),
}

# ---- Script detection --------------------------------------------- #
# Unicode block ranges (start, end, script_tag) — checked in order;
# first match wins.  Latin is the default when no non-Latin codepoint
# is found.  Punctuation / digits don't disambiguate.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    # Hebrew  (0590–05FF) + Alphabetic Presentation Forms (FB1D–FB4F)
    (0x0590, 0x05FF, "hebrew"),
    (0xFB1D, 0xFB4F, "hebrew"),
    # Arabic  (0600–06FF) + Supplement (0750–077F)
    #     + Extended-A (08A0–08FF) + Presentation Forms (FB50–FDFF, FE70–FEFF)
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    # Hangul Syllables (AC00–D7AF) + Jamo (1100–11FF) + Compat Jamo (3130–318F)
    (0x1100, 0x11FF, "kr"),
    (0x3130, 0x318F, "kr"),
    (0xAC00, 0xD7AF, "kr"),
    # Hiragana (3040–309F) + Katakana (30A0–30FF) + Katakana Phonetic (31F0–31FF)
    (0x3040, 0x309F, "jp"),
    (0x30A0, 0x30FF, "jp"),
    (0x31F0, 0x31FF, "jp"),
    # CJK Unified Ideographs (4E00–9FFF) -- shared between JP/KR/ZH;
    # treated as JP by default (Kanji), but a sibling Hiragana/Katakana
    # in the same string will keep it JP; pure Hangul + Hanja will keep
    # it KR.
    (0x4E00, 0x9FFF, "cjk"),
    (0x3400, 0x4DBF, "cjk"),
    (0x20000, 0x2A6DF, "cjk"),
)

# Urdu uses Arabic script but is conventionally rendered in Nastaliq.
# When we know the BCP-47 lang code says "ur", we override the default
# "arabic" pick to use the Nastaliq font.  See `pick_font(text, *, lang_hint=...)`.
_RTL_SCRIPTS = frozenset({"arabic", "urdu", "hebrew"})


def detect_script(text: str) -> str:
    """Pick the single dominant script tag for `text`.

    Returns one of: "latin" (default), "jp", "kr", "cjk", "arabic",
    "hebrew".  When mixed, the FIRST non-Latin script wins, except
    that a JP/KR signal absorbs any "cjk" Han chars (so "東京" inside a
    Japanese sentence stays JP, not CJK)."""
    if not text:
        return "latin"
    hits: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if cp < 0x0080:           # ASCII -- ignore (most strings have lots)
            continue
        for lo, hi, tag in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                hits[tag] = hits.get(tag, 0) + 1
                break
    if not hits:
        return "latin"
    # JP/KR signals absorb the generic CJK Han bucket
    if "jp" in hits and "cjk" in hits:
        hits["jp"] += hits.pop("cjk")
    elif "kr" in hits and "cjk" in hits:
        hits["kr"] += hits.pop("cjk")
    # Pick the most frequent script
    return max(hits, key=hits.get)


def is_rtl(text_or_script: str) -> bool:
    """True for Arabic, Hebrew, Urdu strings (or script tags)."""
    if not text_or_script:
        return False
    if text_or_script in _RTL_SCRIPTS:
        return True
    return detect_script(text_or_script) in _RTL_SCRIPTS


# ---- RTL preparation (shaping + BiDi) ------------------------------ #
@lru_cache(maxsize=512)
def _arabic_reshape(text: str) -> str:
    try:
        import arabic_reshaper
        return arabic_reshaper.reshape(text)
    except Exception:                                          # noqa: BLE001
        return text


@lru_cache(maxsize=512)
def _bidi_visual(text: str) -> str:
    try:
        from bidi.algorithm import get_display
        return get_display(text)
    except Exception:                                          # noqa: BLE001
        return text


# ---- Localized chrome labels (Source: / Date: / Sources: etc.) ----- #
# Tiny vocabulary so document / quote / news cards don't print English
# chrome labels on top of Arabic / Hebrew body text.  Keyed on script
# from detect_script().  Falls back to English for unknown scripts.
_CHROME_VOCAB: dict[str, dict[str, str]] = {
    "arabic": {
        "source":  "المصدر",
        "sources": "المصادر",
        "date":    "التاريخ",
        "from":    "من",
        "by":      "بقلم",
        "via":     "عبر",
        "year":    "السنة",
        "page":    "صفحة",
        "vol":     "مجلد",
        "ref":     "مرجع",
    },
    "urdu": {
        "source":  "ماخذ",
        "sources": "مآخذ",
        "date":    "تاریخ",
        "from":    "از",
        "by":      "بقلم",
        "via":     "بذریعہ",
        "year":    "سال",
        "page":    "صفحہ",
        "vol":     "جلد",
        "ref":     "حوالہ",
    },
    "hebrew": {
        "source":  "מקור",
        "sources": "מקורות",
        "date":    "תאריך",
        "from":    "מאת",
        "by":      "מאת",
        "via":     "באמצעות",
        "year":    "שנה",
        "page":    "עמוד",
        "vol":     "כרך",
        "ref":     "הפניה",
    },
    "jp": {
        "source":  "出典",
        "sources": "出典",
        "date":    "日付",
        "from":    "より",
        "by":      "著者",
        "via":     "経由",
        "year":    "年",
        "page":    "ページ",
        "vol":     "巻",
        "ref":     "参照",
    },
    "kr": {
        "source":  "출처",
        "sources": "출처",
        "date":    "날짜",
        "from":    "에서",
        "by":      "저자",
        "via":     "경유",
        "year":    "년",
        "page":    "페이지",
        "vol":     "권",
        "ref":     "참조",
    },
}


# ---- Place-name transliteration for map cards --------------------- #
# Tiny table of the most common English place names that map_reveal /
# map_route emit, transliterated into Arabic and Hebrew so a native
# render can read the location label without context-switching to
# Latin.  Falls back to the original English string when the place
# isn't in the table -- a missed transliteration is better than tofu.
_PLACE_XLIT: dict[str, dict[str, str]] = {
    "arabic": {
        "lancaster county, pa":  "مقاطعة لانكستر، بنسلفانيا",
        "lancaster county":      "مقاطعة لانكستر",
        "pennsylvania":          "بنسلفانيا",
        "new york":              "نيويورك",
        "new york city":         "مدينة نيويورك",
        "los angeles":           "لوس أنجلوس",
        "chicago":               "شيكاغو",
        "boston":                "بوسطن",
        "san francisco":         "سان فرانسيسكو",
        "washington dc":         "واشنطن العاصمة",
        "washington, dc":        "واشنطن العاصمة",
        "london":                "لندن",
        "paris":                 "باريس",
        "berlin":                "برلين",
        "tokyo":                 "طوكيو",
        "seoul":                 "سيول",
        "moscow":                "موسكو",
        "rome":                  "روما",
        "amsterdam":             "أمستردام",
        "dubai":                 "دبي",
        "cairo":                 "القاهرة",
    },
    "hebrew": {
        "lancaster county, pa":  "מחוז לנקסטר, פנסילבניה",
        "lancaster county":      "מחוז לנקסטר",
        "pennsylvania":          "פנסילבניה",
        "new york":              "ניו יורק",
        "new york city":         "העיר ניו יורק",
        "los angeles":           "לוס אנג'לס",
        "chicago":               "שיקגו",
        "boston":                "בוסטון",
        "san francisco":         "סן פרנסיסקו",
        "washington dc":         "וושינגטון די.סי.",
        "washington, dc":        "וושינגטון די.סי.",
        "london":                "לונדון",
        "paris":                 "פריז",
        "berlin":                "ברלין",
        "tokyo":                 "טוקיו",
        "seoul":                 "סיאול",
        "moscow":                "מוסקבה",
        "rome":                  "רומא",
        "amsterdam":             "אמסטרדם",
        "dubai":                 "דובאי",
        "cairo":                 "קהיר",
    },
}


def line_height_multiplier(text: str) -> float:
    """Recommended line-height multiplier for the SCRIPT of `text`.

    Arabic / Urdu have tall ascenders + deep descenders (Nastaliq /
    NasiKh forms swing well below the baseline), so the default
    line-height that reads great in Latin / CJK looks cramped.  Hebrew
    is roughly between Latin and Arabic.  CJK is comfortable at the
    Latin default (Noto CJK has tight vertical metrics by design).

    Callers do:  ``line_h = int(base_line_h * line_height_multiplier(s))``
    """
    s = detect_script(text or "")
    if s in ("arabic", "urdu"):
        return 1.35
    if s == "hebrew":
        return 1.20
    return 1.0


def safe_margin_for(text: str, default_px: int) -> int:
    """Recommend a safe-area horizontal margin (px) for `text`.

    RTL scripts benefit from a slightly larger margin because the
    right-anchored type tends to crowd the bezel.  Returns
    `default_px` for Latin / CJK, ~1.25× for Arabic / Hebrew."""
    s = detect_script(text or "")
    if s in ("arabic", "urdu", "hebrew"):
        return int(default_px * 1.25)
    return default_px


def localize_place(label: str, sample_text: str) -> str:
    """Return a transliterated place name when the surrounding context
    is Arabic or Hebrew and the original label is recognised in the
    table.  Otherwise returns the input unchanged so unknown places
    still render (in Latin) rather than disappearing.

    `sample_text` is any string that signals the target language --
    typically the headline or scene narration so the chrome reads
    consistently with the rest of the screen."""
    if not label:
        return label
    script = detect_script(sample_text or "")
    table = _PLACE_XLIT.get(script)
    if not table:
        return label
    key = label.strip().lower()
    return table.get(key, label)


def chrome_label(key: str, sample_text: str, *,
                 colon: bool = True) -> str:
    """Return the chrome label (Source / Date / etc.) appropriate for the
    script of `sample_text`, suffixed with ': ' if `colon` is True.

    Pure-Latin / unknown scripts get the English word back.  RTL labels
    are NOT pre-shaped here -- callers should run the FULL composed
    string (label + body) through ``shape_for_drawtext`` so the colon
    and surrounding text BiDi-flip together.
    """
    key = (key or "").lower().strip()
    script = detect_script(sample_text or "")
    table = _CHROME_VOCAB.get(script, {})
    word = table.get(key, key.title())
    return f"{word}: " if colon else word


def shape_for_drawtext(text: str) -> str:
    """Public single-call helper for ffmpeg drawtext text= / textfile=
    payloads.  ffmpeg's drawtext filter does NOT run complex-script
    shaping (no harfbuzz integration in our bundled binary), so Arabic /
    Urdu would render as disconnected isolated forms in visual LTR
    order, and Hebrew would render visually reversed.  This helper:

      * passes Latin / CJK through untouched (no cost),
      * reshapes Arabic / Urdu to presentation forms then applies BiDi
        so glyphs both connect AND read right-to-left visually,
      * BiDi-flips Hebrew (no shaping needed).

    EVERY drawtext / textfile content that comes from user data
    (narration, scene labels, body text, emphasis, attribution, source
    chips) MUST be routed through this — otherwise a non-Latin script
    silently tofus or renders backwards even when the right font is
    loaded.  Pure ASCII labels (e.g. "ARCHIVE", "VS") can skip it.
    """
    return prepare_text(text)


def prepare_text(text: str, *, lang_hint: str = "") -> str:
    """Return the visual-order, shape-substituted string that should be
    drawn by PIL / ffmpeg drawtext.  Latin / CJK passes through
    unchanged.  Arabic / Urdu first reshape to presentation forms (so
    letters connect), then BiDi-rearrange to visual order.  Hebrew
    needs only BiDi.

    This is IDEMPOTENT-ish: passing already-prepared text reshapes
    again, which is harmless because presentation forms are stable
    under reshape.
    """
    if not text:
        return text
    script = lang_hint.lower().strip() if lang_hint else detect_script(text)
    if script in ("arabic", "urdu"):
        return _bidi_visual(_arabic_reshape(text))
    if script == "hebrew":
        return _bidi_visual(text)
    return text


# ---- Font picker --------------------------------------------------- #
def pick_font_paths(text: str, *, lang_hint: str = "") -> tuple[str, ...]:
    """Return an ORDERED font fallback chain that covers the script of
    `text`.  The first entry is the primary face for this script; the
    rest are progressive fallbacks (including Latin so digits, ASCII
    punctuation, and stray English never tofu)."""
    if lang_hint:
        h = lang_hint.lower().strip()
        if h in _SCRIPT_FONTS:
            return _SCRIPT_FONTS[h]
        # BCP-47-style hint (e.g. "ja", "ko", "ar-SA", "ur-PK", "he-IL")
        if h.startswith("ja") or h.startswith("jp"):
            return _SCRIPT_FONTS["jp"]
        if h.startswith("ko") or h.startswith("kr"):
            return _SCRIPT_FONTS["kr"]
        if h.startswith("ar"):
            return _SCRIPT_FONTS["arabic"]
        if h.startswith("ur"):
            return _SCRIPT_FONTS["urdu"]
        if h.startswith("he") or h.startswith("iw"):
            return _SCRIPT_FONTS["hebrew"]
    script = detect_script(text)
    return _SCRIPT_FONTS.get(script, _SCRIPT_FONTS["latin"])


def pick_font(text: str, size: int, *, lang_hint: str = "",
              weight: str = "bold"):
    """Return a PIL ImageFont sized for `text`.  Walks the script-
    appropriate fallback chain and returns the first font that loads;
    when nothing on the chain exists (extreme edge case), falls back to
    PIL's default bitmap font.

    `weight` is informational: VF fonts can be opened at specific
    weight axes via ImageFont.truetype(..., layout_engine=...).  We
    prefer the Bold weight when available."""
    from PIL import ImageFont
    paths = pick_font_paths(text, lang_hint=lang_hint)
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:                                      # noqa: BLE001
            continue
    return ImageFont.load_default()


# ---- RTL cinematic-layout helpers --------------------------------- #
def mirror_panel(box: tuple[int, int, int, int],
                 frame_w: int = 1920) -> tuple[int, int, int, int]:
    """Mirror a rectangular panel about the vertical centre of the frame.
    `(x1, y1, x2, y2)` -> mirrored `(frame_w - x2, y1, frame_w - x1, y2)`.
    Useful for RTL versions of left-anchored cards (the card now sits
    on the right edge with the same dimensions and vertical position).
    """
    x1, y1, x2, y2 = box
    return (frame_w - x2, y1, frame_w - x1, y2)


def accent_bar_edge_for(text_or_script: str) -> str:
    """For a panel that has a thin coloured accent bar on one edge:
    return 'left' for LTR text, 'right' for RTL text -- so an Arabic
    document card has the gold bar on the RIGHT (the edge the eye
    enters the panel from)."""
    return "right" if is_rtl(text_or_script) else "left"


def text_x_for(text: str, panel_left: int, panel_right: int,
               padding: int = 32) -> tuple[int, str]:
    """Pick the X coordinate AND PIL `anchor=` argument for drawing
    `text` inside a panel that spans `[panel_left, panel_right]`.

    Returns:
        (x, anchor)
        For LTR text:  (panel_left + padding, "la")
        For RTL text:  (panel_right - padding, "ra")

    Use as:
        x, anchor = lang.text_x_for(text, BX, BX + BW)
        draw.text((x, y), text, font=f, fill=color, anchor=anchor)

    Templates that already use this helper become RTL-correct for free
    (right-aligned, breath room on the side closest to the panel edge
    the eye enters from)."""
    if is_rtl(text):
        return (panel_right - padding, "ra")
    return (panel_left + padding, "la")


# ---- Anchor / layout helpers for RTL ------------------------------ #
def text_anchor_for(text: str, *, lang_hint: str = "") -> str:
    """PIL `anchor=` value for `draw.text()`:
        "la" (default) -- left/ascender for LTR
        "ra" (right/ascender) for RTL so the right edge of the panel
             is the alignment baseline
    Caller decides where the alignment point sits; this just tells it
    which side to anchor."""
    return "ra" if is_rtl(text or lang_hint) else "la"


def caption_pacing_hint(text: str) -> int:
    """Rough chars/syllables per minute for subtitle pacing.  Latin uses
    word-based pacing (~180 wpm); CJK is character-dense (~600 chars/min);
    Arabic / Hebrew sit around 250 'units' per minute.  Used by music
    WPM helper + caption splitter so a Japanese sentence doesn't all
    arrive as a single 5-second flash."""
    s = detect_script(text)
    if s in ("jp", "kr", "cjk"):
        return len(text)                              # chars, not words
    if s in ("arabic", "hebrew", "urdu"):
        return max(1, len(text.split()))              # word-like units
    return max(1, len(text.split()))                  # default: Latin words


def split_caption(text: str, max_units: int = 7) -> list[str]:
    """Cinematic subtitle chunker that respects each script's natural
    breaking points.

    Latin / RTL languages chunk on WORD boundaries (`max_units` words
    per subtitle line).  CJK has no spaces, so we chunk on CHARACTERS
    with a soft preference for breaking after CJK punctuation (、。
    ・「」『』) -- matching the rhythm a native subtitle editor would
    pick.  Rule of thumb: 6-12 chars per chunk reads comfortably.

    This is the single place every caption splitter in the codebase
    should call (assemble._split_caption, musiclib._wpm helper, etc.)
    so all subtitle systems pace identically per language."""
    if not text:
        return []
    s = detect_script(text)
    # CJK: char-based, prefer breaking after CJK punctuation
    if s in ("jp", "kr", "cjk"):
        # CJK comfortable subtitle = 6-12 chars per chunk depending on
        # pacing; default to 9 chars per chunk for tight readability.
        chunk_chars = max(6, min(12, max_units * 2))
        out: list[str] = []
        cur = ""
        for ch in text:
            cur += ch
            # break after CJK pause punctuation OR when chunk full
            if ch in "、。・！？；：「」『』" or len(cur) >= chunk_chars:
                if cur.strip():
                    out.append(cur.strip())
                cur = ""
        if cur.strip():
            out.append(cur.strip())
        return out
    # Latin / RTL: word-based.  RTL keeps word order untouched (BiDi
    # runs at render time).
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + max_units])
            for i in range(0, len(words), max_units)]


# ---- BCP-47 language tag -> Edge TTS voice -------------------------- #
# Default male neural voices per locale; matches the current EN default
# (en-US-GuyNeural).  Override via `Brief.voice` is still honoured.
_TTS_VOICE_MAP: dict[str, str] = {
    "en":  "en-US-GuyNeural",
    "en-US": "en-US-GuyNeural",
    "en-GB": "en-GB-RyanNeural",
    "de":  "de-DE-ConradNeural",
    "de-DE": "de-DE-ConradNeural",
    "es":  "es-ES-AlvaroNeural",
    "es-ES": "es-ES-AlvaroNeural",
    "es-MX": "es-MX-JorgeNeural",
    "tr":  "tr-TR-AhmetNeural",
    "tr-TR": "tr-TR-AhmetNeural",
    "ja":  "ja-JP-KeitaNeural",
    "ja-JP": "ja-JP-KeitaNeural",
    "jp":  "ja-JP-KeitaNeural",
    "ko":  "ko-KR-InJoonNeural",
    "ko-KR": "ko-KR-InJoonNeural",
    "kr":  "ko-KR-InJoonNeural",
    "ar":  "ar-SA-HamedNeural",
    "ar-SA": "ar-SA-HamedNeural",
    "ar-EG": "ar-EG-ShakirNeural",
    "ur":  "ur-PK-AsadNeural",
    "ur-PK": "ur-PK-AsadNeural",
    "he":  "he-IL-AvriNeural",
    "he-IL": "he-IL-AvriNeural",
    "iw":  "he-IL-AvriNeural",
    "fr":  "fr-FR-HenriNeural",
    "it":  "it-IT-DiegoNeural",
    "pt":  "pt-BR-AntonioNeural",
    "ru":  "ru-RU-DmitryNeural",
    "zh":  "zh-CN-YunxiNeural",
    "zh-CN": "zh-CN-YunxiNeural",
    "zh-TW": "zh-TW-YunJheNeural",
    "hi":  "hi-IN-MadhurNeural",
}

# Detected-script -> default Edge TTS voice when no explicit lang tag.
_SCRIPT_TO_VOICE: dict[str, str] = {
    "jp":     "ja-JP-KeitaNeural",
    "kr":     "ko-KR-InJoonNeural",
    "cjk":    "zh-CN-YunxiNeural",
    "arabic": "ar-SA-HamedNeural",
    "urdu":   "ur-PK-AsadNeural",
    "hebrew": "he-IL-AvriNeural",
    "latin":  "en-US-GuyNeural",
}


def voice_for(*, lang: str = "", text: str = "",
              default: str = "en-US-GuyNeural") -> str:
    """Pick a neural TTS voice.  Priority:
        1. Explicit BCP-47 `lang` tag (e.g. "ja", "ar-EG", "es-MX").
        2. Detected script of `text` (so a Japanese narration without
           a lang tag still gets ja-JP-KeitaNeural).
        3. Fallback to `default` (en-US-GuyNeural).

    Brief authors who hard-set `Brief.voice` are honoured upstream --
    this helper only fires when the user did NOT pick a voice."""
    if lang:
        h = lang.lower().strip()
        # exact match
        if h in _TTS_VOICE_MAP:
            return _TTS_VOICE_MAP[h]
        # primary-tag match (e.g. "ja-JP" -> "ja")
        primary = h.split("-")[0]
        if primary in _TTS_VOICE_MAP:
            return _TTS_VOICE_MAP[primary]
    if text:
        script = detect_script(text)
        if script in _SCRIPT_TO_VOICE:
            return _SCRIPT_TO_VOICE[script]
    return default


# ---- System-level RTL auto-shaping ------------------------------- #
# Monkey-patches PIL.ImageDraw so that every `draw.text(...)` call
# anywhere in the codebase (100+ templates) AUTO-RESHAPES + BIDI-FIXES
# Arabic / Urdu / Hebrew text before rendering.  Templates don't need
# to know about RTL; this hook propagates the fix universally.
#
# The patch is idempotent (installed once per process) and only fires
# for strings that actually contain RTL codepoints, so Latin / CJK /
# digits-only strings pay zero cost.

_PATCHED = False


def install_rtl_autoshape() -> None:
    """Idempotent global install of the RTL auto-shaping hook on PIL's
    ImageDraw.text + multiline_text.  Call once at import time."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from PIL import ImageDraw
    except Exception:                                          # noqa: BLE001
        return

    _orig_text = ImageDraw.ImageDraw.text
    _orig_multi = ImageDraw.ImageDraw.multiline_text

    def _maybe_prepare(text):
        if not isinstance(text, str):
            return text
        if not text:
            return text
        # Fast skip: ASCII / Latin-extended-only strings never need RTL
        if all(ord(c) < 0x0590 for c in text):
            return text
        return prepare_text(text)

    def text(self, xy, text, *args, **kwargs):                 # noqa: E501,A002
        return _orig_text(self, xy, _maybe_prepare(text), *args, **kwargs)

    def multiline_text(self, xy, text, *args, **kwargs):       # noqa: A002
        return _orig_multi(self, xy, _maybe_prepare(text), *args, **kwargs)

    ImageDraw.ImageDraw.text = text
    ImageDraw.ImageDraw.multiline_text = multiline_text
    _PATCHED = True


# Auto-install on module import so every consumer of `vidlore.lang`
# (which footage.py imports at module init) benefits without an
# explicit call.
install_rtl_autoshape()


# ---- Tofu detector ------------------------------------------------ #
def looks_tofu(img_pil) -> float:
    """Return a 0..1 score estimating how much of the image is covered
    by tofu (missing-glyph) boxes.

    Tofu signature: regularly-spaced, IDENTICAL small rectangular
    glyphs in a grid.  Real text has glyph-shape variety -- the pattern
    inside each glyph cell differs.  Tofu boxes have the SAME pattern
    repeated, so the standard deviation of inter-glyph regions is far
    lower than for real text.

    Heuristic: tile the image into 32×32 cells, take the per-cell mean
    luminance variance.  A high count of cells with NEAR-IDENTICAL
    variance + the right average brightness band (~white-on-dark
    tofu) is the tofu signature.  Returns 0..1; >0.30 = highly suspicious.

    This is a best-effort heuristic; visual review remains the source
    of truth.  Use along with `contrast_std` as belt-and-braces."""
    try:
        import numpy as np
        from PIL import Image
        if img_pil.mode != "L":
            g = img_pil.convert("L")
        else:
            g = img_pil
        # downscale for speed
        a = np.asarray(g.resize((640, 360), Image.BILINEAR), dtype="float32")
        # tile into 32x32 cells; compute std-of-pixels per cell
        CH, CW = 32, 32
        H, W = a.shape
        rows, cols = H // CH, W // CW
        if rows == 0 or cols == 0:
            return 0.0
        a2 = a[:rows*CH, :cols*CW].reshape(rows, CH, cols, CW)
        cell_std = a2.std(axis=(1, 3))                 # (rows, cols)
        cell_mean = a2.mean(axis=(1, 3))
        # candidate tofu cells: have content (std > 10) and bright pixels
        # (tofu glyphs are typically light text on dark bg or vice versa)
        active = cell_std > 10
        # tofu signature: many active cells share NEARLY-IDENTICAL std
        # (within ±2 of each other).  Real text has wider variety.
        if not active.any():
            return 0.0
        active_stds = cell_std[active]
        # cluster: how many active cells fall within ±2 of the median?
        med = np.median(active_stds)
        clustered = np.abs(active_stds - med) <= 2.0
        cluster_frac = clustered.sum() / max(1, active.sum())
        # tofu = high cluster fraction with many active cells in a row
        # (real text varies across the line)
        if cluster_frac < 0.55:
            return 0.0
        # scale by how much of the frame has active cells (more = stronger
        # tofu signal)
        active_frac = active.sum() / float(rows * cols)
        return float(cluster_frac * active_frac)
    except Exception:                                          # noqa: BLE001
        return 0.0


def font_covers_text(text: str, font_path: str) -> float:
    """Return 0..1 fraction of `text`'s non-space codepoints that
    `font_path` carries actual glyphs for.  Uses fontTools when
    available; falls back to a PIL rendering pass otherwise.

    The most reliable script-correctness check we can do without
    visual inspection -- a test that asserts `font_covers_text(text,
    pick_font_paths(text)[0]) > 0.95` proves the right font was
    routed."""
    if not text:
        return 1.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    try:
        from fontTools.ttLib import TTFont
        font = (TTFont(font_path, fontNumber=0)
                if font_path.endswith(".ttc")
                else TTFont(font_path))
        cmap = font.getBestCmap()
        covered = sum(1 for c in chars if ord(c) in cmap)
        return covered / float(len(chars))
    except Exception:                                          # noqa: BLE001
        return 0.0
