"""Step 1 of the pipeline: turn the 4-Pillars brief into a TTS-ready
narration script broken into scenes, each with visual search keywords.

Two providers (Vidlore 'hybrid' parity):
  * Anthropic LLM  -> writes the script automatically (if ANTHROPIC_API_KEY)
  * manual script  -> you supply a .txt; we scene-split + keyword it
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .brief import Brief
from .config import Config

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

_STOPWORDS = set(
    """the a an and or but if then than that this these those of to in on at by for
    with from as is are was were be been being it its it's into over under about
    after before between out up down off again further once here there all any both
    each few more most other some such no nor not only own same so too very can will
    just don should now you your we our they their he she his her them who whom which
    what when where why how also been had has have having do does did doing would could
    one two three first new like get got make made back even still much many us i'm
    you're they're we're isn't aren't didn't doesn't don't""".split()
)


@dataclass
class Scene:
    index: int
    narration: str
    keywords: list[str] = field(default_factory=list)
    # LLM "visual director" output: a precise cinematic shot description
    # for THIS sentence (what must literally be on screen). Empty on the
    # keyless/edited path -> footage falls back to the keyword heuristic.
    visual: str = ""
    # Emotional energy 1 (calm) .. 5 (climax) — drives pacing, motion and
    # sound. 0 = unknown (treated as 2/neutral).
    intensity: int = 0
    # The single most emotionally important word/short phrase in this
    # scene — punched in the captions at the moment it is spoken.
    emphasis: str = ""
    # Storyboard shot grammar so consecutive scenes are not visually
    # monotone: establishing | aerial | detail | reaction | portrait |
    # archival | macro | wide | tracking. Drives motion variety.
    shot_type: str = ""
    # Narrative beat from the showrunner arc (hook/problem/evidence/
    # reaction/escalation/turn/reveal/climax/payoff/resolution …). Drives
    # FOOTAGE-DURATION pacing (Issue #5): reveal/climax = hold long so
    # the impact lands; escalation = faster cutting; evidence = brisk.
    role: str = ""
    # Optional premium motion-graphic the editor decides this scene
    # needs. kind in {location, stat, number, label, document, ""}.
    # text = the headline/number/label line; body = (document only) a
    # short excerpt shown on the evidence card.
    graphic_kind: str = ""
    graphic_text: str = ""
    graphic_body: str = ""

    @property
    def query(self) -> str:
        return " ".join(self.keywords[:3]) if self.keywords else self.narration[:60]

    @property
    def energy(self) -> int:
        return self.intensity if 1 <= self.intensity <= 5 else 2


# ── EDITORIAL MODE ──────────────────────────────────────────────────
# Forensic comparison against top-channel docs (Vox / Johnny Harris /
# Netflix Explained / MagnatesMedia) revealed two opposing editorial
# regimes inside a single video:
#
#   DENSITY    — high cut rate (12-15 cuts/min), text overlays on
#                ~60-70% of frames, brighter + more saturated palette.
#                Used for HOOK / PROBLEM / EVIDENCE / PROOF / STATS
#                beats where the editor is packing information in.
#
#   RESTRAINT  — long held shots (10-15s), silence pockets ≥1s, no
#                or minimal on-screen text, desaturated darker grade.
#                Used for QUOTE / CLIMAX / REVEAL / REACTION /
#                RESOLUTION / PAYOFF beats where the editor lets the
#                moment breathe.
#
# An AI-tool-feeling render uses ONE mode the whole way (all-density
# = YouTube-AI-doc clutter; all-restraint = too still).  A human
# editor switches between them per scene-role.  `editorial_mode(sc)`
# is the SINGLE source of truth every downstream renderer should
# consult before deciding cut count, graphic density, transition
# style, grade, or whether to fire an emphasis stab.

_ROLE_DENSITY: frozenset = frozenset({
    "hook", "tease", "open", "cold_open",
    "problem", "stakes", "context",
    "evidence", "proof", "data", "stats",
    "explanation", "explain", "how_it_works",
    "escalation", "build",
})
_ROLE_RESTRAINT: frozenset = frozenset({
    "quote", "monologue", "interview",
    "reveal", "turn", "twist",
    "climax", "peak",
    "reaction", "aftermath",
    "resolution", "payoff", "outro", "close",
    "reflection", "silence",
})


def editorial_mode(sc) -> str:
    """Return 'density' or 'restraint' for a scene.

    Priority:
      1. intensity 5 (climax) -> ALWAYS restraint (the silence reveal)
      2. role hit on the restraint list -> restraint
      3. role hit on the density list -> density
      4. fallback: intensity-driven (1-2 = restraint, 3+ = density)

    Designed so that the cinematic differentiator (silence/long-hold
    on the climax word) is sacred — even an "evidence" role flips to
    restraint if intensity is at peak 5."""
    intensity = 0
    try:
        intensity = int(getattr(sc, "intensity", 0) or 0)
    except (TypeError, ValueError):
        pass
    if intensity >= 5:
        return "restraint"
    role = (getattr(sc, "role", "") or "").lower().strip()
    if role in _ROLE_RESTRAINT:
        return "restraint"
    if role in _ROLE_DENSITY:
        return "density"
    # Fallback: anything else (or no role set) — let intensity decide.
    return "density" if intensity >= 3 else "restraint"


def mode_weights(mode: str) -> dict:
    """Per-mode editorial weights downstream renderers can multiply
    against their existing knobs.  Centralised so tuning is one place.

    DENSITY favours: more beats per scene, more graphic chips,
    brighter/saturated palette, harder cuts.
    RESTRAINT favours: longer holds, silence, desat, dissolves.
    """
    if mode == "density":
        return {
            "beats_multiplier": 1.5,        # +50 % beats inside scene
            "graphics_weight": 0.5,         # consume half a graphic-cap slot
            "emphasis_fire_prob": 1.0,      # always fire the word-stab
            "grade_brightness_lift": 0.04,  # +4 % luma
            "grade_saturation_lift": 0.08,  # +8 % sat
            "transition_dissolve_bias": 0.15,  # prefer harder cuts
            "music_density_floor": 0.85,    # keep the bed loud
        }
    # restraint (default)
    return {
        "beats_multiplier": 0.7,            # -30 % beats (longer holds)
        "graphics_weight": 1.0,             # full cap slot
        "emphasis_fire_prob": 0.55,         # only ~half the words stab
        "grade_brightness_lift": -0.02,     # -2 % luma (a touch darker)
        "grade_saturation_lift": -0.04,     # -4 % sat
        "transition_dissolve_bias": 0.65,   # prefer motivated dissolves
        "music_density_floor": 0.55,        # quieter bed, more room for silence
    }


@dataclass
class Script:
    title: str
    scenes: list[Scene]

    @property
    def full_text(self) -> str:
        return " ".join(s.narration for s in self.scenes)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
# MULTILINGUAL sentence split: after a CJK ender (。．！？… — no space
# follows in Japanese/Chinese) OR after ASCII .!? when whitespace
# follows. The old ASCII-only `(?<=[.!?])\s+(?=[A-Z0-9"'])` found ZERO
# breaks in Japanese (enders are 。！？, no spaces) so a whole 19-min
# script collapsed into ~2 mega-scenes -> no per-scene visual/keywords/
# emphasis -> "no text + irrelevant footage".
_SENT_SPLIT = re.compile(r"(?<=[。．！？…])|(?<=[.!?])(?=\s)")


def _wlen(s: str) -> int:
    """Script-aware 'length' for scene sizing. Space-delimited text →
    word count; CJK / no-space scripts (Japanese, Chinese) → an
    approx from character count (~3 chars ≈ one pacing unit), so a
    normal Japanese sentence stays its OWN scene instead of every
    sentence merging into one giant block."""
    t = s.split()
    return len(t) if len(t) > 1 else max(1, len(s) // 3)
_TTS_NOISE = re.compile(
    r"^\s*(\[[^\]]*\]|\([^)]*\)|#{1,6}\s|[-*]\s|\d+\.\s|"
    r"(scene|narrator|voice ?over|vo|host|title|hook|intro|outro|cta)\s*[:\-])",
    re.I,
)


def _clean_for_tts(text: str) -> str:
    """Strip stage directions, speaker labels, markdown, urls (Vidlore's
    SCRIPT_NOT_TTS_READY rule)."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\[[^\]]*\]", "", line)          # [SHOW X]
        line = re.sub(r"https?://\S+", "", line)         # urls
        line = re.sub(r"^\s*#{1,6}\s*", "", line)        # md headings
        line = re.sub(r"^\s*[-*]\s+", "", line)          # bullets
        line = re.sub(r"^\s*\d+\.\s+", "", line)         # numbered
        line = re.sub(
            r"^\s*(scene\s*\d*|narrator|voice ?over|vo|host|title|hook|"
            r"intro|outro|cta)\s*[:\-]\s*",
            "",
            line,
            flags=re.I,
        )
        line = re.sub(r"\*\*|\*|__|`", "", line)          # md emphasis
        line = re.sub(r"\s{2,}", " ", line).strip()
        if line:
            out.append(line)
    return " ".join(out)


# Words that look "important" but return junk/no stock footage. A pro
# editor never searches these — they cut to the *thing being described*.
_ABSTRACT = set(
    """story stories history mystery mysteries legend legends myth myths truth
    truths question questions answer answers theory theories idea ideas reason
    reasons fact facts evidence proof account accounts record records report
    reports research researcher researchers scholar scholars team teams people
    person population world worlds society civilization civilisation culture
    century centuries decade decades year years time times moment moments day
    days night nights way ways thing things nothing everything something anyone
    everyone someone life lives death deaths fate destiny power knowledge
    information example examples case cases event events change changes problem
    problems result results process system systems number numbers amount level
    name names word words language meaning belief beliefs fear fears hope hopes
    sense future past present league region regional common era""".split()
)

# Concrete, depictable subjects a camera can actually show. If the
# narration mentions any of these, that is what the footage should be.
_VISUAL_LEXICON = set(
    """ocean sea seas wave waves water coast coastline coastal beach shore
    cliff cliffs island islands river rivers lake lakes waterfall flood
    tsunami storm storms rain lightning thunder hurricane wind snow ice
    glacier mountain mountains valley canyon desert dune forest forests
    jungle tree trees field fields grass volcano lava earthquake eruption
    cave caves rock rocks stone stones sand mud dust fog mist sky clouds
    sun moon stars night sunrise sunset fire flames smoke ash ruins ruin
    temple temples statue statues monument pyramid castle palace fortress
    wall walls tower column columns pillar arch gate bridge road roads
    street streets path stairs city cities town village house houses
    building buildings church cathedral tomb grave cemetery skull bones
    skeleton ship ships boat boats sail fleet harbor harbour port anchor
    shipwreck submarine sonar drill excavation dig artifact artifacts gold
    silver coin coins treasure jewel crown sword shield armor armour spear
    arrow gun rifle cannon soldier soldiers army warrior knight battle war
    explosion blood horse horses dog wolf lion snake snakes bird birds
    eagle fish shark whale crowd market money cash bank stocks chart map
    compass clock train car cars truck plane aircraft rocket factory
    machine engine gears bridge tunnel mine farm crops harvest""".split()
)


def _keywords(sentence: str, title: str) -> list[str]:
    """Concrete-visual keyword extraction for stock/B-roll search.

    A real editor cuts to *what the line is about*, not its proper nouns
    ("Helike") or longest words ("whispering"). So we (1) pull words that
    are in a concrete visual lexicon first, (2) then other non-abstract
    content words, and (3) only use a proper noun as a last resort.
    """
    low = sentence.lower()
    words = re.findall(r"[a-z]{3,}", low)

    visual: list[str] = []
    for w in words:
        if w in _VISUAL_LEXICON and w not in visual:
            visual.append(w)

    content: list[str] = []
    for w in words:
        if (
            len(w) >= 4
            and w not in _STOPWORDS
            and w not in _ABSTRACT
            and w not in _VISUAL_LEXICON
            and w not in content
        ):
            content.append(w)
    content.sort(key=len, reverse=True)

    kw: list[str] = []
    for token in visual[:3] + content[:2]:
        if token and token not in kw:
            kw.append(token)
    # Proper nouns / dates never match stock footage, so they are never
    # used as search terms — the theme B-roll fallback covers name-only
    # lines instead. ``content`` is the last resort before that.
    return kw[:4] or content[:2] or ["landscape"]


_SHOT_TYPES = {
    "establishing", "aerial", "wide", "detail", "macro",
    "reaction", "portrait", "archival", "tracking",
}
# --------------------------------------------------------------------------- #
# TEMPLATE REGISTRY INTEGRATION
# --------------------------------------------------------------------------- #
# Previously _GRAPHIC_KINDS / _BODY_KINDS / per-kind caps were hardcoded
# here AND the editor-brain prompt's per-kind rules were inlined into
# _EDITOR_RULES. That meant adding a new template touched 4 places.
# The new architecture: every template self-describes in
# `vidlore/templates/__init__.py`. Adding a Template entry there
# automatically updates the LLM's kind enum, the per-kind rules, the
# body-kind set and the caps applied by `_apply_graphic_caps`. One
# place to add, document and inspect — same modular shape that a
# future Remotion port will need.
from . import templates as _tpl

# ---- GLOBALLY BANNED FULL-SCREEN INFOGRAPHIC CARD TYPES (v15) ----
# These produce "PowerPoint slide / AI auto-template" look — blue-bg full-
# screen cards that feel cheap and repetitive. Permanently disabled across
# ALL niches. Information is delivered instead via: document (editorial paper
# with highlighter), location (lower-third overlay over footage), quote,
# footage + lower-thirds, and map/timeline where contextually earned.
#
# Screenshots that triggered this ban:
#   • stat_dashboard  → "THE PEST PROBLEM / 3 this month" blue grid card
#   • era_banner      → "LANCASTER COUNTY · 1860s" full-screen blue card
#   • comparison      → "THEIR GARDEN VS YOURS" blue VS split card
#
_BANNED_TEMPLATES: frozenset = frozenset({
    "era_banner",     # full-screen era label → use location lower-third + footage
    "comparison",     # blue VS split card → use document / quote_highlight instead
    "stat_dashboard", # blue multi-stat grid → use document with figures instead
})

_GRAPHIC_KINDS = set(_tpl.all_names()) - _BANNED_TEMPLATES
_BODY_KINDS = _tpl.body_kinds() - _BANNED_TEMPLATES
# V2.9 — Section-C STRUCTURED-asset graphic kinds. These map to MG primitives
# (silhouette_scale_compare / footage_route_trace), NOT vidlore/templates cards,
# so they aren't in _GRAPHIC_KINDS / _BODY_KINDS. They carry inherently
# structured data (scale-true sizes / on-screen route coords) the LLM must
# supply explicitly in the body (items=/points=) — there is no prose to mine.
# Whitelisted so _scene_graphic keeps the kind + body; pipeline._mg_structured_
# assets() does the real parse + clamps. The pipeline adapter requires a
# parseable body, so a hint without valid structured data drops cleanly.
_STRUCTURED_KINDS = frozenset({"scale_compare", "size_compare", "route_trace"})
# V3.3 — EMISSION VOCABULARY UNLOCK. graphic_kind strings that route (via
# director._GK_AFFINITY) to premium MG primitives whose pipeline adapter branch
# already parses graphic_body, but which the editor LLM was never OFFERED. Unlocked
# here as SUGGESTED kinds the LLM may request ONLY from REAL data already stated in
# the scene (the _MG_VOCAB block teaches strict no-invent rules + body formats).
# DELIBERATELY EXCLUDES every map/route/geo card (map_heat_spread, velocity_route_map,
# world_map_arc, map_badge_node) — those need verified geography and stay
# manual/editor-only to prevent fabricated boundaries/routes.
_MG_UNLOCK_KINDS = frozenset({
    "bar_chart",     # → statistic_bar_reveal   (>=2 real labelled values)
    "versus",        # → comparison_split        (2 real subjects; NOT banned 'comparison')
    "balance",       # → vs_balance_scale         (a real two-sided tradeoff)
    "composition",   # → composition_stack        (a real share/breakdown)
    # RC5.1 — 'process' (→ bullet_list) REMOVED. The cheap four-box
    # process template was cut from production (gone from REGISTRY). The
    # scriptwriter must never be offered/accept it: an LLM 'process' kind now
    # falls through _scene_graphic to no graphic (footage carries the beat).
    "hierarchy",     # → org_hierarchy_tree       (real named parent→children)
    "before_after",  # → before_after_slider      (a real then/now)
    "decision",      # → flowchart_decision       (a real decision + outcome)
    "sankey",        # → sankey_flow              (a real allocation)
    "eras",          # → era_band_timeline        (real dated periods)
    "headlines",     # → headline_montage         (real/paraphrased script lines)
    "gauge",         # → spectrum_meter           (a real measured value)
})
_CAPS: dict[str, int] = {k: v for k, v in _tpl.caps().items()
                         if k not in _BANNED_TEMPLATES}
_MIN_GAPS: dict[str, int] = {k: v for k, v in _tpl.min_gaps().items()
                              if k not in _BANNED_TEMPLATES}
# Legacy aliases — some upstream code still imports the old names.
_MAX_EVIDENCE_PER_VIDEO = _CAPS.get("evidence", 3)
_MAX_RANKING_PER_VIDEO = _CAPS.get("ranking", 5)

# ---- editor robustness knobs --------------------------------------- #
# Small batches are far more reliable than one huge call (a single broken
# char no longer loses the whole film); failed scenes split-and-retry.
_EDITOR_BATCH = 6
_MAX_RETRY = 2


# ---- CHARACTER-INTRO DETECTOR --------------------------------------- #
# Regex patterns that catch "...his name is X..." / "When X, an ..." /
# "X, a ..." / "Meet X" / "...named X..." style sentences.  Each pattern
# extracts (NAME, ROLE) where ROLE is an appositive description.
#
# Common false-positive names (not a person) are filtered out below.
_FAKE_NAMES: frozenset[str] = frozenset({
    "god", "lord", "christ", "jesus", "buddha", "allah",
    "america", "europe", "asia", "africa", "england", "britain",
    "russia", "china", "japan", "germany", "france", "italy", "spain",
    "amish", "mennonite", "quaker", "muslim", "hindu", "jewish",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "pennsylvania", "virginia", "california", "texas", "florida",
    "washington", "lincoln", "lancaster", "chester", "harvard", "yale",
    "world", "earth", "moon", "sun",
    "now", "then", "today", "tomorrow", "yesterday",
    "no", "yes", "the",
    # English connectors/openers that often start a sentence with
    # capital + comma -- not actual names of people.
    "when", "while", "before", "after", "though", "although", "if",
    "as", "since", "because", "but", "however", "still", "yet",
    "meanwhile", "instead", "then", "once", "until", "unless",
    "where", "what", "why", "how", "who", "and", "or", "so", "for",
    "in", "on", "at", "by", "from", "into", "across", "during",
    "later", "soon", "today", "tonight", "tomorrow",
})

# (a) "When NAME, an/a [appositive], ..."  ([Ww]hen so we accept "When"
#     at sentence start OR mid-sentence "when".  No re.I on the whole
#     regex -- that would break [A-Z] in the NAME capture group.)
_CHARINTRO_WHEN = re.compile(
    r"\b[Ww]hen\s+([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)"
    r"\s*,\s*(an?\s+[^,.;]{6,60})\s*,",
)
# (b) Sentence-start "NAME, an/a [appositive], ..."
_CHARINTRO_LEAD = re.compile(
    r"(?:^|\.\s+|—\s*)"
    r"([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)"
    r"\s*,\s*(an?\s+[^,.;]{6,60})\s*,",
)
# (c) "His/Her name is NAME[.]"  (highest-quality signal)
_CHARINTRO_NAMED = re.compile(
    r"\b(?:his|her)\s+name\s+is\s+([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)",
    re.I,
)
# (d) "Meet NAME, ..." / "This is NAME, ..."
_CHARINTRO_MEET = re.compile(
    r"\b(?:meet|this\s+is)\s+([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)"
    r"\s*[,.]",
    re.I,
)
# (e) "...a man/woman named NAME"  ([Aa] so we accept both "A" at
#     sentence start and lowercase "a" mid-sentence -- without
#     re.I, which would corrupt the [A-Z] name capture)
_CHARINTRO_AMAN = re.compile(
    r"\b[Aa]\s+(?:[Mm]an|[Ww]oman|[Ff]armer|[Tt]eacher|[Ss]cientist|"
    r"[Ii]nspector|[Oo]fficer|[Ee]lder|[Gg]randfather|[Gg]randmother|"
    r"[Vv]eteran|[Ww]itness|[Kk]eeper|[Dd]octor|[Nn]urse|[Ss]oldier|"
    r"[Ss]ailor|[Pp]riest|[Mm]onk|[Mm]iner|[Hh]unter|[Ww]riter|"
    r"[Rr]eporter|[Hh]ermit)\s+[Nn]amed\s+"
    r"([A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?)",
)

# ---- UNICODE / non-Latin script intro patterns --------------------- #
# These trigger when the narration is in JP / KR / AR / UR / HE and
# uses a script-native intro phrase.  Each pattern captures a short
# name token (1-3 chars CJK, 2-12 Arabic/Hebrew word).  ROLE is left
# empty for these -- the name reveal card just shows the name when
# the LLM didn't supply a separate role.
#
# Japanese:  "エリ、" / "エリは" / "エリという" / "彼の名はエリ"
# Korean:    "엘리, " / "엘리는" / "그의 이름은 엘리"
# Arabic:    "اسمه إيلي" / "إيلي،" / "هذا إيلي"
# Hebrew:    "שמו אלי" / "אלי," / "זה אלי"
_CHARINTRO_JP = re.compile(
    r"(?:彼の名は|彼女の名は|名は|の名は)([぀-ヿ一-鿿]{2,8})"
    r"|(?:^|[、。「])([゠-ヿ]{2,6})[、と]",
)
_CHARINTRO_KR = re.compile(
    # Korean names are 2-3 hangul syllables; stop the capture at any
    # non-name particle (는/은/이/이라고/입니다 etc).
    r"(?:이름은|이름이|이름이라고|불리는)\s*([가-힯]{1,4})"
    r"|(?:^|[,。])\s*([가-힯]{1,4})\s*[,은는이라고]",
)
_CHARINTRO_AR = re.compile(
    # "اسمه إيلي" (his name is X) / "اسمها إيلي" (her name is X).
    # Stop at Arabic comma U+060C, full-stop U+06D4 and whitespace --
    # don't consume the role appositive that follows the comma.
    r"(?:اسمه|اسمها|يدعى|تدعى|يسمى|تسمى|باسم|"
    r"اس\s*کا\s*نام|اس\s*کی\s*نام)\s+"
    r"([ء-يٮ-ۓ]{2,15})",
)
_CHARINTRO_HE = re.compile(
    # "שמו אלי" (his name is X) / "שמה אלי" (her name is X)
    r"(?:שמו|שמה|בשם)\s+"
    r"([א-ת]{2,12}(?:\s+[א-ת]{2,12})?)",
)


def _detect_character_intro(text: str) -> tuple[str, str] | None:
    """Return (NAME, ROLE) when this narration introduces a character,
    else None.  Conservative: only fires when the pattern is unambiguous,
    so it never promotes a noisy mention to a name_reveal card.

    ROLE is cleaned (leading 'a/an' stripped, length-capped) and may be
    empty when the source pattern doesn't carry an appositive (e.g.
    "Meet Eli." -- name only)."""
    if not text:
        return None
    t = text.strip()
    # Non-Latin script first — these patterns are SPECIFIC to a script,
    # so they only fire when the text contains those codepoints, and a
    # match is high-confidence.  Latin patterns below catch the rest.
    #
    # Common pronouns / topic-markers that the loose regex can mis-grab
    # as names ("그의" his / "彼の" his / "هذا" this / "זה" this).  These
    # are filtered here so they never trigger a name_reveal.
    _NON_LATIN_FAKE = {
        "그의", "그녀의", "그", "그녀", "이",
        "彼の", "彼", "彼女", "私", "あなた",
        "هذا", "هذه", "ذلك", "تلك", "هو", "هي",
        "זה", "זאת", "הוא", "היא",
    }
    for pat in (_CHARINTRO_JP, _CHARINTRO_KR,
                _CHARINTRO_AR, _CHARINTRO_HE):
        m = pat.search(t)
        if m:
            nm = (m.group(1) or (m.group(2) if m.lastindex
                  and m.lastindex >= 2 else "") or "").strip()
            if nm and nm not in _NON_LATIN_FAKE:
                return (nm, "")
    # ordered by signal strength -- strong first so a "named X" line
    # doesn't get pre-empted by an incidental leading match
    candidates: list[tuple[str, str]] = []
    for pat, role_group in (
        (_CHARINTRO_NAMED, None),
        (_CHARINTRO_MEET, None),
        (_CHARINTRO_WHEN, 2),
        (_CHARINTRO_LEAD, 2),
        (_CHARINTRO_AMAN, None),
    ):
        m = pat.search(t)
        if not m:
            continue
        nm_raw = m.group(1).strip()
        # First-word lowercase form for filter check
        first = nm_raw.split()[0].lower()
        if first in _FAKE_NAMES:
            continue
        role = ""
        if role_group is not None:
            role = m.group(role_group).strip()
            # strip leading article
            role = re.sub(r"^an?\s+", "", role, flags=re.I).strip()
            # title-case but keep small words lower-case
            role = role.rstrip(",.;:")
        candidates.append((nm_raw, role))
        # strong patterns (NAMED/MEET) win immediately
        if pat in (_CHARINTRO_NAMED, _CHARINTRO_MEET):
            return (nm_raw, role)
    return candidates[0] if candidates else None


def _promote_character_intros(scenes: list) -> int:
    """Walk all scenes in order; the FIRST scene that introduces a given
    character gets its `graphic_kind` promoted to `name_reveal` with the
    detected NAME + ROLE.  Subsequent re-mentions of the same name are
    NOT re-promoted (one reveal per character per video).

    Two tiers of override:
      • STRONG signal ("His/Her name is X" via _CHARINTRO_NAMED) → ALWAYS
        wins, even over editor-chosen high-value kinds. The viewer
        needs to see who the character is; a map or comparison can
        live on an adjacent scene.
      • WEAK signal (other intro patterns) → only promotes when the
        scene's existing kind is empty / low-signal.

    Returns the count of scenes promoted."""
    if "name_reveal" not in _GRAPHIC_KINDS:
        return 0
    # graphic_kinds we usually won't override (editor chose deliberately).
    # But STRONG signals bypass this — see below.
    _PROTECT = {
        "document", "quote_highlight", "map_route", "map_reveal",
        "map_pin_cluster", "timeline", "process_diagram",
        "scientific_explainer", "comparison", "stat_dashboard",
        "classified_dossier", "title_card", "chapter_marker",
        "redacted_stamp",
    }
    seen_names: set[str] = set()
    promoted = 0
    for sc in scenes:
        gk = (getattr(sc, "graphic_kind", "") or "").lower()
        if gk == "name_reveal":           # already a reveal; track its name
            nm = (getattr(sc, "graphic_text", "") or "").strip().lower()
            if nm:
                seen_names.add(nm.split()[0])
            continue
        narr = getattr(sc, "narration", "") or ""
        # STRONG signal: explicit "His/Her name is X" — overrides even
        # protected kinds. Bug fix 2026-05-26: pipeline was picking
        # map_reveal over name_reveal in a character intro scene,
        # leaving the protagonist without a portrait card.
        is_strong = bool(_CHARINTRO_NAMED.search(narr) or
                          _CHARINTRO_MEET.search(narr))
        if gk in _PROTECT and not is_strong:
            continue
        result = _detect_character_intro(narr)
        if not result:
            continue
        nm, rl = result
        first_lc = nm.split()[0].lower()
        if first_lc in seen_names:        # already revealed earlier
            continue
        sc.graphic_kind = "name_reveal"
        sc.graphic_text = nm.upper()[:24]
        sc.graphic_body = rl.upper()[:48] if rl else ""
        seen_names.add(first_lc)
        promoted += 1
    return promoted


def _auto_fallback(sc) -> bool:
    """FALLBACK SYSTEM — last-resort, keyword-driven editing for a scene the
    LLM editor never resolved (after salvage + retries). Guarantees sane
    shot_type / keywords / intensity so the film is NEVER 'footage-only and
    empty', and assigns a tasteful graphic ONLY on a strong, unambiguous
    signal (quote, archival document, a stated figure). In-place; returns
    True (the scene is now resolved)."""
    txt = (sc.narration or "").strip()
    # always give the scene usable craft defaults
    if not sc.keywords:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", txt)
        sc.keywords = list(dict.fromkeys(words))[:4] or ([txt[:30]] if txt
                                                         else ["scene"])
    if not sc.shot_type:
        sc.shot_type = "wide"
    if not sc.intensity:
        sc.intensity = 2
    if sc.graphic_kind or not txt:
        return True
    low = txt.lower()
    # 0) CHARACTER INTRODUCTION -> name_reveal (highest priority).
    # The LLM frequently skips name_reveal even when a sentence clearly
    # introduces a character ("When Eli, an Amish grandfather, ...").
    # We pattern-match the most common intro forms and promote the
    # scene to a name_reveal card.  Name extraction is conservative --
    # we only match short proper names (1-2 capitalised words) and a
    # short appositive role description.
    _name_role = _detect_character_intro(txt)
    if _name_role and "name_reveal" in _GRAPHIC_KINDS:
        _nm, _rl = _name_role
        sc.graphic_kind = "name_reveal"
        sc.graphic_text = _nm.upper()[:24]
        sc.graphic_body = _rl.upper()[:48] if _rl else ""
        return True
    # 1) a real quotation -> quote_highlight
    m = re.search(r'["“”]([^"“”]{14,140})'
                  r'["“”]', txt)
    if m and "quote_highlight" in _GRAPHIC_KINDS:
        sc.graphic_kind = "quote_highlight"
        sc.graphic_text = m.group(1).strip()[:140]
        return True
    # 2) archival/report language -> document evidence page. V3.3.2: derive the
    # source label from the ACTUAL cited term in the narration (no invented
    # "ARCHIVE RECORD" headline) so the card never fabricates a source line it
    # cannot support; the body stays the verbatim narration.
    _mdoc = re.search(r"\b(report|archive|archives|document|records?|memo|"
                      r"dossier|files?|study|findings|declassified|"
                      r"manuscript|ledger)\b", low)
    if _mdoc and "document" in _GRAPHIC_KINDS:
        sc.graphic_kind = "document"
        sc.graphic_text = _mdoc.group(1).upper()[:24]
        sc.graphic_body = txt[:240]
        return True
    # 3) number auto-assign REMOVED (v15) — was too aggressive: matched any
    # 3+ digit number and slapped a full-screen stat card on it. Lets footage
    # breathe instead. The LLM still picks "number" for truly critical
    # standalone figures; deterministic fallback defers to clean footage.
    return True


def _clip_words(s: str, maxlen: int) -> str:
    """Trim a card label/title to <= maxlen WITHOUT slicing a word in half.
    A hard `text[:64]` produced cards like '...would force Washingto'; this
    backs up to the last whole word and adds an ellipsis so the card always
    reads cleanly. Short text (<= maxlen) is returned untouched."""
    s = (s or "").strip()
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen].rstrip()
    if maxlen < len(s) and not s[maxlen].isspace():   # sliced mid-word → back up
        sp = cut.rfind(" ")
        if sp >= 1:
            cut = cut[:sp]
    cut = cut.rstrip(" ,;:-—–.")
    return (cut + "…") if cut else s[:maxlen].strip()


def clean_card_text(text: str, *, maxlen: int = 64) -> str:
    """Render-time guard for REUSED scripts whose graphic_text was already hard-
    sliced (e.g. a resumed script.json holding '...would force Washingto'). When
    a label looks like a mid-word [:N] cut — long, no terminal punctuation, a
    multi-letter trailing fragment — drop the partial word and add an ellipsis.
    Conservative: short or punctuated labels are left exactly as-is."""
    s = (text or "").strip()
    if not s or len(s) > maxlen + 4:
        # over budget → normal word-clip; very short → nothing to do
        return _clip_words(s, maxlen) if len(s) > maxlen else s
    if len(s) >= maxlen - 8 and s[-1].isalpha() and s[-1:].islower():
        # likely a lowercase mid-sentence cut (titles/acronyms end upper/punct)
        toks = s.split(" ")
        if len(toks) >= 3 and len(toks[-1]) >= 4:
            return " ".join(toks[:-1]).rstrip(" ,;:-—–.") + "…"
    return s


def _parse_extra(s: dict) -> tuple[str, str, str, str]:
    """(shot_type, graphic_kind, graphic_text, graphic_body) from one LLM
    scene dict, validated so a bad value never breaks the render."""
    st = str(s.get("shot_type", "") or "").strip().lower()
    if st not in _SHOT_TYPES:
        st = ""
    g = s.get("graphic") or {}
    if not isinstance(g, dict):
        g = {}
    gk = str(g.get("kind", "") or "").strip().lower()
    gt = str(g.get("text", "") or "").strip()
    gb = str(g.get("body", "") or "").strip()
    if gk in _STRUCTURED_KINDS:
        # body-driven Section-C kinds: require a parseable structured body
        # (graphic_text is OPTIONAL — it becomes the card title; the pipeline
        # adapter also reads an explicit `title=` from the body). A hint with no
        # usable items=/points= data drops cleanly rather than firing empty.
        _key = "items=" if gk in ("scale_compare", "size_compare") else "points="
        if _key not in gb:
            return st, "", "", ""
        return st, gk, _clip_words(gt, 64), gb[:300]
    if gk in _MG_UNLOCK_KINDS:
        # V3.3 unlocked MG kinds — body-driven (bars=/pair=/segments=/steps=/…).
        # The pipeline adapter parses + clamps + drops a body with no usable data,
        # and the director's required_inputs gate falls back to footage if the
        # primitive's inputs aren't filled — so an empty/garbage body fails safe.
        # graphic_text is the card title/lead (optional).
        return st, gk, _clip_words(gt, 64), gb[:300]
    if gk not in _GRAPHIC_KINDS or not gt:
        return st, "", "", ""
    return st, gk, _clip_words(gt, 64), (gb[:300] if gk in _BODY_KINDS else "")


def script_from_text(title: str, text: str, *, max_words_per_scene: int = 32) -> Script:
    """FOOTAGE↔VOICEOVER SYNC FIX: one scene ≈ ONE sentence (not the old
    2-sentence / 32-word blocks). Each scene gets ONE footage subject and
    its Whisper-aligned boundary, so the visual is on screen exactly
    while THAT sentence is spoken — instead of one clip lagging across
    10-16s of evolving narration ("voiceover says it, footage shows it
    late"). Very short fragments are merged forward so there are no
    1-second micro-shots; a rare over-long run-on still splits on words."""
    clean = _clean_for_tts(text)
    sentences = [s.strip() for s in _SENT_SPLIT.split(clean) if s.strip()]
    _MIN_W = 7                       # below this, merge into the scene
    chunks: list[str] = []
    for sent in sentences:
        # _wlen = script-aware size (words OR CJK chars), so a normal
        # Japanese sentence is NOT seen as "0 words" and merged away.
        if chunks and (_wlen(sent) < _MIN_W
                       or _wlen(chunks[-1]) < _MIN_W):
            sep = "" if not chunks[-1][-1:].isascii() else " "
            chunks[-1] = chunks[-1] + sep + sent
        else:
            chunks.append(sent)
    # safety: a single monster run-on -> hard split so a scene is never
    # absurdly long. Space-delimited -> by words; CJK -> by characters.
    out: list[str] = []
    cap = int(max_words_per_scene * 1.6)
    for ch in chunks:
        w = ch.split()
        if len(w) > 1 and len(w) > cap:                # spaced text
            for k in range(0, len(w), max_words_per_scene):
                out.append(" ".join(w[k:k + max_words_per_scene]))
        elif len(w) <= 1 and len(ch) > max_words_per_scene * 6:  # CJK
            step = max_words_per_scene * 5
            for k in range(0, len(ch), step):
                out.append(ch[k:k + step])
        else:
            out.append(ch)
    scenes = [Scene(i, c, _keywords(c, title)) for i, c in enumerate(out)]
    if not scenes:
        raise ValueError("Script is empty after cleaning.")
    return Script(title=title, scenes=scenes)


# --------------------------------------------------------------------------- #
# LLM provider
# --------------------------------------------------------------------------- #
def _channel_scene_count(brief: Brief) -> tuple[int, int, str]:
    """Apply Look-DNA `scene_count_mult` to the brief's target story
    points, and return (lo, hi, pacing_hint).

    Atlas Explained wants ~1.8× scenes for the same brief (Vox / Harris
    density).  Amber Chronicles wants ~0.6× (slow museum holds).
    Midnight Pacific stays at the brief default.  Without an active
    channel, returns the brief's untouched values + empty hint —
    legacy renders are byte-identical.

    IMP_026 — the base range is now `brief.target_scenes` (shot beats),
    DECOUPLED from `target_points` (story acts). A '6-8 min' doc is 25-40
    scenes (not 4-5), so the narration is long enough to actually fill the
    requested minutes."""
    lo, hi = brief.target_scenes
    mult = 1.0
    hint = ""
    try:
        from .look_dna import current as _ld_current, look_get
        if _ld_current() is not None:
            mult = float(look_get("scene_count_mult", 1.0) or 1.0)
            hint = str(look_get("pacing_hint", "") or "").strip()
    except Exception:                                       # noqa: BLE001
        pass
    if mult <= 0:
        mult = 1.0
    # Apply mult.  Floor at 2 keeps the doc from collapsing into a
    # single monologue, ceiling at 4× the brief default prevents
    # runaway 60-scene 2-min explosions.  Atlas with mult 2.4 on a
    # (2,3) brief expands to (5,7); Amber with mult 0.4 compresses
    # to (2,2) — a clear 2-3× structural spread the LLM cannot
    # self-normalise.
    cap_hi = max(hi, int(round(hi * 4.0)))
    lo2 = max(2, int(round(lo * mult)))
    hi2 = max(lo2, min(cap_hi, int(round(hi * mult))))
    if hi2 == lo2:
        hi2 = lo2 + 1 if mult >= 1.0 else lo2
    return lo2, hi2, hint


def _system_prompt(brief: Brief) -> str:
    tmpl = (_PROMPT_DIR / "four_pillars_documentary.md").read_text(encoding="utf-8")
    lo, hi, _hint = _channel_scene_count(brief)
    return tmpl.format(
        fmt=brief.fmt,
        minutes=brief.spec["minutes"],
        words=brief.target_words,
        points_lo=lo,
        points_hi=hi,
    )


def _enforce_scene_count(scenes: list, brief: Brief) -> list:
    """Clamp the LLM-produced scene list to the channel's target range.

    Reads (lo, hi) from `_channel_scene_count(brief)` — the same
    channel-aware range the system prompt asked the LLM for.  The LLM
    routinely under- or over-shoots, so this is the hard enforcement
    layer that makes Atlas STRUCTURALLY denser and Amber STRUCTURALLY
    sparser regardless of what Claude felt like producing.

    Strategy:
      • If too few (LLM under-shot):  split the LONGEST scene at the
        nearest sentence boundary until we reach `lo`.  Split scenes
        inherit the parent's role/intensity/visual/shot_type/graphic.
        Emphasis is moved to whichever child contains the emphasis
        token; the other child clears it.
      • If too many (LLM over-shot): merge adjacent LOW-intensity
        scenes pairwise (longer narration is the survivor; keywords
        unioned; intensities averaged).  Never merge across a
        climax/reveal/turn boundary.
      • Within range: untouched.

    Logged: a one-line "[enforce] scenes N -> M" so we can SEE the
    channel doing its work in the render log.
    """
    if not scenes:
        return scenes
    lo, hi, _ = _channel_scene_count(brief)
    n = len(scenes)
    if lo <= n <= hi:
        return scenes
    import re as _re

    def _split_one(scene) -> tuple | None:
        """Split this scene's narration at the best sentence midpoint.
        Returns (left_scene, right_scene) or None if not splittable."""
        sents = _re.split(r"(?<=[.!?])\s+", (scene.narration or "").strip())
        if len(sents) < 2:
            return None
        # split point: closest to middle by word-count
        words = [s.split() for s in sents]
        cum = 0
        total = sum(len(w) for w in words)
        best_i = None
        best_d = total
        for i in range(1, len(sents)):
            cum += len(words[i - 1])
            d = abs(cum - total / 2)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            return None
        left_text  = " ".join(sents[:best_i]).strip()
        right_text = " ".join(sents[best_i:]).strip()
        if not left_text or not right_text:
            return None
        emph = (scene.emphasis or "").strip()
        def _has(text: str, ph: str) -> bool:
            return bool(ph) and ph.lower() in text.lower()
        left = Scene(
            scene.index, left_text, list(scene.keywords or []),
            visual=scene.visual, intensity=scene.intensity,
            emphasis=emph if _has(left_text, emph) else "",
            shot_type=scene.shot_type, role=scene.role,
            graphic_kind=scene.graphic_kind,
            graphic_text=scene.graphic_text,
            graphic_body=scene.graphic_body,
        )
        right = Scene(
            scene.index, right_text, list(scene.keywords or []),
            visual=scene.visual, intensity=scene.intensity,
            emphasis=emph if _has(right_text, emph) else "",
            shot_type=scene.shot_type, role=scene.role,
            graphic_kind="", graphic_text="", graphic_body="",
        )
        return left, right

    def _merge_pair(a, b):
        """Merge two adjacent scenes into one (a dominates)."""
        merged_text = (a.narration.rstrip(".!?") + ". " + b.narration).strip()
        merged_kw = list(a.keywords or [])
        for k in (b.keywords or []):
            if k not in merged_kw:
                merged_kw.append(k)
        out = Scene(
            a.index, merged_text, merged_kw[:6],
            visual=a.visual or b.visual,
            intensity=max(a.intensity, b.intensity),
            emphasis=a.emphasis or b.emphasis,
            shot_type=a.shot_type or b.shot_type,
            role=a.role or b.role,
            graphic_kind=a.graphic_kind or b.graphic_kind,
            graphic_text=a.graphic_text or b.graphic_text,
            graphic_body=a.graphic_body or b.graphic_body,
        )
        return out

    HARD_ROLES = frozenset(("climax", "reveal", "turn", "twist", "peak"))

    work = list(scenes)
    if n < lo:
        # SPLIT loop: pick the longest splittable scene and split it
        # until we hit `lo` or no scene can be split further.
        guard = 0
        while len(work) < lo and guard < 50:
            guard += 1
            # pick longest by narration word count
            cand_idx = max(range(len(work)),
                            key=lambda i: len((work[i].narration or "").split()))
            split = _split_one(work[cand_idx])
            if split is None:
                # try the next-longest
                ordered = sorted(range(len(work)),
                                 key=lambda i: -len((work[i].narration or "").split()))
                done = False
                for i in ordered[1:]:
                    s = _split_one(work[i])
                    if s is not None:
                        work[i:i + 1] = list(s)
                        done = True
                        break
                if not done:
                    break
            else:
                work[cand_idx:cand_idx + 1] = list(split)
    elif n > hi:
        # MERGE loop: find adjacent pair with lowest combined intensity
        # (neither in HARD_ROLES) and merge.
        guard = 0
        while len(work) > hi and guard < 50:
            guard += 1
            best = None
            best_score = 10 ** 9
            for i in range(len(work) - 1):
                a, b = work[i], work[i + 1]
                if a.role in HARD_ROLES or b.role in HARD_ROLES:
                    continue
                score = (a.intensity or 2) + (b.intensity or 2)
                if score < best_score:
                    best = i
                    best_score = score
            if best is None:
                # nothing safe to merge — give up gracefully
                break
            work[best:best + 2] = [_merge_pair(work[best], work[best + 1])]

    # renumber indices
    for i, s in enumerate(work):
        s.index = i

    if len(work) != n:
        print(
            f"  [enforce] channel scene count {n} -> {len(work)} "
            f"(target {lo}-{hi})", flush=True)
    return work


def _count_words(scenes: list) -> int:
    return sum(len((getattr(s, "narration", "") or "").split()) for s in scenes)


def _expand_scenes_llm(brief: Brief, cfg: Config, existing: list,
                       deficit_words: int) -> list:
    """IMP_026 — ask the LLM for ADDITIONAL scene beats to deepen a too-short
    script (no filler, no repetition). Returned scenes are inserted before the
    closing beat. Best-effort: returns [] on any failure."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    except Exception:                                          # noqa: BLE001
        return []
    have = "\n".join(f"- {(s.narration or '')[:150]}" for s in existing)
    n_new = max(3, int(round(deficit_words / 38.0)))           # ~38 words/beat
    sysp = (
        "You are expanding an existing documentary script that is too SHORT "
        "for its target runtime. Add NEW scene beats that DEEPEN the story — "
        "context, setup, character detail, visual evidence, consequences, "
        "tension and reveal beats — that slot BETWEEN the existing beats. "
        "Never repeat or restate an existing beat, never add filler, keep the "
        "same factual subject and tone. Output pure spoken narration only."
    )
    user = (
        f"TITLE: {brief.title}\n\nEXISTING BEATS (do NOT repeat these):\n"
        f"{have}\n\nAdd about {n_new} NEW scene beats (~{deficit_words} words "
        "total) that belong in the MIDDLE / BUILD of the story (never a new "
        "opening or ending). Return ONLY JSON: {\"scenes\":[{\"narration\":"
        "\"...\",\"keywords\":[\"..\",\"..\"],\"visual\":\"..\",\"intensity\":"
        "1-5,\"emphasis\":\"..\"}, ...]}. 1-2 tight sentences per beat. "
        "'visual' = one vivid photorealistic shot for that exact line; "
        "'keywords' = 2-4 filmable nouns; 'emphasis' = the most charged word "
        "appearing verbatim in that beat."
    )
    try:
        msg = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=min(12000, 2000 + n_new * 200),
            system=sysp,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in msg.content
                      if getattr(b, "type", "") == "text")
        data = _parse_json(raw)
    except Exception as e:                                     # noqa: BLE001
        print(f"  [duration] expansion call failed: {e}", flush=True)
        return []
    out = []
    for s in data.get("scenes", []):
        nar = _clean_for_tts(s.get("narration", ""))
        if not nar:
            continue
        try:
            inten = int(s.get("intensity", 0) or 0)
        except (TypeError, ValueError):
            inten = 0
        st, gk, gt, gb = _parse_extra(s)
        out.append(Scene(
            0, nar,
            [k for k in s.get("keywords", []) if k] or _keywords(nar, brief.title),
            visual=str(s.get("visual", "") or "").strip(), intensity=inten,
            emphasis=str(s.get("emphasis", "") or "").strip(),
            shot_type=st, graphic_kind=gk, graphic_text=gt, graphic_body=gb))
    return out


def _validate_and_expand(scenes: list, brief: Brief, cfg: Config) -> list:
    """IMP_026 — enforce the WORD/DURATION target (the real runtime driver):
    count words, estimate minutes, and if the script is too SHORT expand it
    with capped follow-up LLM calls so a 6-min request becomes a ~6-min video.
    Splitting alone can't fix this — it only re-divides existing words."""
    from .brief import WORDS_PER_SECOND
    target = brief.target_words
    lo_min, hi_min = brief.target_minutes

    def _mins(w: int) -> float:
        return w / (WORDS_PER_SECOND * 60.0)

    words = _count_words(scenes)
    print(f"  [duration] script {words}w ~= {_mins(words):.1f} min "
          f"(target ~{target}w / {lo_min:.0f}-{hi_min:.0f} min, "
          f"{len(scenes)} scenes)", flush=True)
    if not getattr(cfg, "has_llm", False):
        return scenes
    # IMP_026 — bigger docs need more top-up passes to reach their word target
    # (a 35-min/5250w doc can't get there in 2 passes). Each pass is gated on
    # the 0.85x threshold, so extra passes only run while genuinely short.
    max_tries = 2 if target <= 2500 else (3 if target <= 4200 else 5)
    tries = 0
    while words < 0.85 * target and tries < max_tries:
        new = _expand_scenes_llm(brief, cfg, scenes, int(target - words))
        if not new:
            break
        scenes = (scenes[:-1] + new + scenes[-1:]) if len(scenes) >= 2 \
            else (scenes + new)
        for i, s in enumerate(scenes):
            s.index = i
        words = _count_words(scenes)
        tries += 1
        print(f"  [duration] expanded -> {words}w ~= {_mins(words):.1f} min "
              f"({len(scenes)} scenes)", flush=True)
    if words < 0.6 * target:
        print(f"  [duration] WARNING: still SHORT ({words}w vs {target}w) — "
              f"~{_mins(words):.1f} min, under requested "
              f"{lo_min:.0f}-{hi_min:.0f} min.", flush=True)
    elif words > 1.6 * target:
        print(f"  [duration] note: LONG ({words}w vs {target}w); scene-merge "
              "trims beats but runtime may exceed request.", flush=True)
    return scenes


def _llm_script(brief: Brief, cfg: Config, mode=None) -> Script:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    # Channel-native scene count + editorial pacing hint (P3 lever)
    lo, hi, pacing_hint = _channel_scene_count(brief)
    # When a channel is active, also bend the per-scene structure:
    #   • Atlas (mult ≥ 1.4) — punchy short sentences, dense beats.
    #   • Amber (mult ≤ 0.8) — longer reflective sentences, fewer
    #     beats per scene, generous breathing room.
    #   • Default — keep the 1-3 sentences-per-scene legacy rule.
    # IMP_026 — single LLM response can't carry 100+ rich scenes, so cap the
    # ask here; the duration-validation gate expands long docs with follow-up
    # calls until the WORD target (the real runtime driver) is met.
    ask_hi = min(hi, 42)
    ask_lo = min(lo, ask_hi)
    acts_lo, acts_hi = brief.target_points
    tw = brief.target_words
    # words-per-scene budget — and the sentence-length hint is DERIVED from it
    # so the constraints are consistent (the old 'few scenes + 1-3 sentences'
    # silently capped total words far below the target → half-length videos).
    _wper = tw / max(1.0, (ask_lo + ask_hi) / 2.0)
    wps_budget = int(round(_wper))
    if _wper <= 30:
        _sentences_hint = "1 tight sentence per scene"
    elif _wper <= 55:
        _sentences_hint = "1-2 sentences per scene"
    elif _wper <= 85:
        _sentences_hint = "2-3 sentences per scene"
    else:
        _sentences_hint = "2-4 reflective sentences per scene"
    _hint_line = (
        f"\nEDITORIAL PACING HINT: {pacing_hint}\n"
        if pacing_hint else ""
    )
    user = (
        "TITLE: %s\n\nCREATIVE BRIEF (4 Pillars):\n%s\n\n"
        "Return ONLY JSON: {\"title\": \"...\", \"scenes\": [{"
        '"narration": "...", "keywords": ["...","..."], '
        '"visual": "...", "intensity": 1-5, "emphasis": "..."}, ...]}. '
        "narration = pure spoken text (no labels/stage directions). "
        "**LENGTH IS A HARD REQUIREMENT: the full script MUST total about "
        "%d spoken words — this sets the video RUNTIME, so do NOT under-write. "
        "Structure the story as %d-%d major ACTS, then break each act into "
        "SEVERAL short scene beats: produce between %d and %d scenes total, "
        "%s (~%d words each). Many tight beats — never a handful of long "
        "monologues.** Every beat adds NEW information (never restate a prior "
        "beat). Build the length from MEANINGFUL sub-scenes — context, setup, "
        "character detail, visual evidence, consequences, tension beats, "
        "reveal beats — never filler or repetition.%s\n"
        "You are also the VISUAL DIRECTOR. For every scene write a "
        "'visual': one vivid, photorealistic SHOT that literally depicts "
        "exactly what THAT sentence is about — name the concrete subject, "
        "setting, era, time of day, weather, camera shot (wide/aerial/"
        "close-up/tracking) and mood. It must feel intentional and "
        "emotionally synced to the line, never generic. Example: narration "
        "'the sea pulled back then returned as a wall of water' -> visual "
        "'aerial shot of the ocean violently receding from an ancient "
        "Greek shoreline at dusk then a towering tsunami wave surging "
        "back, dark stormy sky, cinematic, photorealistic'. No on-screen "
        "text/logos. 'keywords' = 2-4 concrete filmable nouns (no proper "
        "nouns/dates/abstractions). 'intensity' = emotional energy of the "
        "scene: 1 calm/expository, 3 building, 5 climax/shock/reveal. "
        "'emphasis' = the SINGLE most emotionally charged word (or 2-word "
        "phrase) that appears VERBATIM in this scene's narration — it gets "
        "punched on screen the instant it is spoken (e.g. 'vanished', "
        "'no survivors', 'buried alive'). Pick what a human editor would "
        "hit the viewer with."
        % (brief.title, brief.prompt, tw, acts_lo, acts_hi,
           ask_lo, ask_hi, _sentences_hint, wps_budget, _hint_line)
    )
    msg = client.messages.create(
        model=cfg.anthropic_model,
        # IMP_026 — scale output budget with the scene ask so a dense doc's
        # JSON is never truncated mid-array (which silently dropped scenes →
        # short scripts).
        max_tokens=min(16000, 3500 + ask_hi * 220),
        system=_system_prompt(brief),
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    data = _parse_json(raw)
    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        nar = _clean_for_tts(s.get("narration", ""))
        if not nar:
            continue
        try:
            inten = int(s.get("intensity", 0) or 0)
        except (TypeError, ValueError):
            inten = 0
        st, gk, gt, gb = _parse_extra(s)
        scenes.append(
            Scene(
                len(scenes),
                nar,
                [k for k in s.get("keywords", []) if k]
                or _keywords(nar, brief.title),
                visual=str(s.get("visual", "") or "").strip(),
                intensity=inten,
                emphasis=str(s.get("emphasis", "") or "").strip(),
                shot_type=st, graphic_kind=gk, graphic_text=gt,
                graphic_body=gb,
            )
        )
    if not scenes:
        raise RuntimeError("LLM returned no usable scenes.")
    # IMP_026 — enforce the WORD/DURATION target FIRST (expand a short draft
    # with follow-up calls) so the runtime matches the request; THEN clamp the
    # scene count. Splitting in _enforce_scene_count only re-divides existing
    # words, so it can never fix a too-short script on its own.
    scenes = _validate_and_expand(scenes, brief, cfg)
    # ── P3-PLUS — HARD SCENE-COUNT ENFORCEMENT ─────────────────────
    # LLMs self-normalise scene count: ask for 11 scenes in a 2-min
    # brief and Claude returns 7, ask for 3 and it returns 6.  The
    # channel pacing knob loses its bite.  Post-process the LLM's
    # output to clamp scene count into the channel target — splitting
    # long scenes at sentence boundaries when too few, merging
    # adjacent low-intensity scenes when too many.
    scenes = _enforce_scene_count(scenes, brief)
    # CRITICAL FIX (Issue: user's auto-generated renders had 0 graphics
    # / 0 shot_types / 0 roles): the LLM-generated path was producing
    # narration + visual + emphasis only — it never asked the model for
    # shot_type or graphic kind, and never ran the per-scene editor
    # brain. Run that pass NOW so this path produces the same rich
    # editing decisions (ranking / evidence / location / explainer /
    # callout / document / portrait / chart / number / label, plus
    # shot_type variance and story-arc roles) as the paste-script path.
    try:
        _apply_editor_decisions(brief.title, scenes, cfg)
    except Exception as e:                                 # noqa: BLE001
        print(f"  [editor] decisions pass failed ({e}); "
              "scenes keep narration-only", flush=True)
    try:
        _geo_enrich(scenes, mode)
    except Exception as e:                                 # noqa: BLE001
        print(f"  [geo] enrichment skipped ({e})", flush=True)
    _apply_graphic_caps(scenes, mode)
    return Script(title=data.get("title") or brief.title, scenes=scenes)


def _strip_fences(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    return raw


def _repair_json(s: str) -> str:
    """Best-effort textual repair of common LLM JSON breakage:
    smart quotes, trailing commas, stray control chars, and TRUNCATION
    (unbalanced brackets/strings get closed). Returns a string to retry
    json.loads on — never raises."""
    if not s:
        return s
    # normalise unicode quote glyphs the model sometimes emits
    s = (s.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'"))
    # strip ASCII control chars that aren't valid raw inside JSON strings
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    # remove trailing commas before a closing } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # balance truncation: walk once, tracking strings, and append the
    # closers still open at end-of-text (drops a dangling trailing comma).
    stack, in_str, esc = [], False, False
    for ch in s:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    for opener in reversed(stack):
        s += "}" if opener == "{" else "]"
    return s


def _loads_robust(raw: str):
    """Parse JSON tolerantly: strict -> outermost-object -> repaired.
    Returns the parsed object, or None if nothing parses."""
    raw = _strip_fences(raw)
    for cand in (raw,
                 (re.search(r"[{\[].*[}\]]", raw, re.S) or [None])
                 and (re.search(r"[{\[].*[}\]]", raw, re.S).group(0)
                      if re.search(r"[{\[].*[}\]]", raw, re.S) else None),
                 _repair_json(raw)):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:                              # noqa: BLE001
            continue
    return None


def _salvage_objects(raw: str, require_key: str = "i") -> list:
    """PARTIAL-SALVAGE layer. Walk the text and extract EVERY balanced
    {...} object (respecting strings/escapes), parse each one independently
    (with repair), and keep the dicts that carry `require_key`. Recovers
    valid scene decisions even when the surrounding array is truncated or
    one object mid-stream is malformed. De-duplicates by the key value."""
    raw = _strip_fences(raw)
    objs, stack, in_str, esc = [], [], False, False
    for idx, ch in enumerate(raw):
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop()
            objs.append(raw[start:idx + 1])
    out, seen = [], set()
    for o in objs:
        d = None
        try:
            d = json.loads(o)
        except Exception:                              # noqa: BLE001
            try:
                d = json.loads(_repair_json(o))
            except Exception:                          # noqa: BLE001
                d = None
        if isinstance(d, dict) and require_key in d:
            k = d.get(require_key)
            if k not in seen:
                seen.add(k)
                out.append(d)
    return out


def _parse_json(raw: str) -> dict:
    """Back-compat strict-ish parse used by non-editor callers (story arc,
    script writer). Tolerant, but raises if nothing at all parses."""
    obj = _loads_robust(raw)
    if obj is None:
        raise ValueError("no parseable JSON in response")
    return obj


def _editor_debug_dump(tag: str, raw: str) -> str:
    """Best-effort: write a raw broken LLM response to a debug file for
    production debugging. Returns the path (or '')."""
    try:
        import tempfile
        import time as _t
        d = Path(tempfile.gettempdir()) / "vidlore_editor_debug"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{tag}_{int(_t.time() * 1000)}.txt"
        p.write_text(raw or "", encoding="utf-8")
        return str(p)
    except Exception:                                  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Editing analysis of a USER-PROVIDED script (no writing — the user's
# words are kept verbatim; the LLM only acts as the editor/director)
# --------------------------------------------------------------------------- #
_EDITOR_RULES = (
    "Vary 'st' across consecutive scenes (no repeats back-to-back). "
    "GRAPHICS are what make this look like a premium studio documentary "
    "— use them with PURPOSE (roughly 1 per 3-5 scenes, never two in a "
    "row, vary the kind; most scenes still = none so footage breathes): "
    "* number  -> a single striking standalone number/year/price/"
    "multiplier the line lands on (t e.g. '1970','$20','3X'; b=''). "
    "* chart   -> the line states a proportion/percentage/'half'/'most'/"
    "'X out of Y' (t = the figure e.g. '47%'; b = a SHORT UPPERCASE "
    "headline e.g. 'LOSING HALF OUR HARVEST'). "
    "* portrait-> a key PERSON is introduced/named (t = THE NAME, short; "
    "b = one vivid photoreal description of them: age, face, clothing, "
    "setting, mood — for the portrait photo). "
    "* callout -> a concrete THING in the footage should be labelled "
    "(t = the short label e.g. 'APHID CLUSTER'; b=''). "
    "* document-> narration cites a report/letter/study/record/ledger/"
    "'the entry reads' (t = short SOURCE title e.g. 'PENN STATE AG. "
    "FIELD REPORT 1923'; b = a <=240-char excerpt as if quoting it — an "
    "archive photo is auto-generated). "
    "* explainer-> narration INTRODUCES an important object / tool / "
    "structure / invention / concept / system the viewer needs clarified "
    "(a clean educational 'what is this' card with a photo). t = the "
    "THING'S NAME, short bold (e.g. 'BADGIRS', 'THE ANTIKYTHERA "
    "MECHANISM'); b = ONE short plain caption that clarifies it (e.g. "
    "'The ancient wind towers that cooled cities without electricity'). "
    "Use when a concept genuinely needs a beat to land. "
    "* ranking-> narration introduces ONE entry of a ranked list / "
    "countdown / Top-N reveal ('coming in at number four...', 'our #2 "
    "pick...', 'the third spot goes to...'). Cinematic ranked card: a "
    "white framed border around the footage + a '#N' badge slides up "
    "at the bottom. Use ONLY for genuine ranked content (Top 5, Best "
    "of, countdown). t = the rank label, just the NUMBER or '#N' "
    "(e.g. '4', '#2', 'NUMBER ONE'); b = the item's short title "
    "(<=40 chars, e.g. 'MOUNT RAINIER WILDFLOWER MEADOW'). Max 5 per "
    "video. Skip for unranked enumerations. "
    "* evidence-> narration delivers MULTIPLE proof points / a "
    "trust-building list / 3-5 key reasons / a research summary the "
    "viewer must absorb together (a side-by-side card: footage left, "
    "white panel right with headline + bullets revealed one by one). "
    "Use SPARINGLY: at most 1-3 per video, only when a panel genuinely "
    "improves clarity (e.g. 'WHY TRUST THIS LIST?', 'THE EVIDENCE', "
    "'KEY FINDINGS'). t = headline question or label, short caps "
    "<=30 chars; b = 3-5 short bullets separated by '|' (each <=50 "
    "chars, no period), e.g. 'Ranks plants by speed & safety|Backed by "
    "peer-reviewed studies|From alpine blooms to yard weeds'. NEVER use "
    "for a single fact — use number/document/explainer for those. "
    "* location-> a real place/era reveal (t 'LANCASTER, PA · 1923'; "
    "b=''). * label -> a key term first introduced (b=''). Else k='none'. "
    "* scale_compare-> RARE (<=1 per video): narration compares the SIZE of "
    "two real things with KNOWN real dimensions (a carrier vs a destroyer, a "
    "blue whale vs a bus). t = short caps title (e.g. 'SIZE COMPARISON'); "
    "b = 'items=NameA:sizeA:noteA|NameB:sizeB:noteB' — sizes are REAL numbers "
    "in the SAME unit (e.g. metres). NEVER invent sizes; if you don't know "
    "them, use a different kind. "
    "* route_trace-> RARE (<=1 per video): narration follows a physical "
    "JOURNEY/path traceable across ONE shot (a supply line, an escape, an "
    "expedition leg). t = short caps title (e.g. 'THE ROUTE'); b = "
    "'points=x:y:Start|x:y|x:y:End' — x,y are 0..1 SCREEN positions for the "
    "on-footage path (NOT geographic coords), 2-5 points ordered by the story, "
    "label only first & last. "
    "kw = concrete filmable nouns, no proper nouns/dates/abstractions. "
    "int 1 calm..5 climax; pace the arc, don't keep it flat."
)

# Append rules for any template registered in vidlore/templates/
# that isn't already explained in the hand-tuned text above (new
# templates auto-teach the LLM with no edit here).
# NOTE: _BANNED_TEMPLATES are excluded so the LLM never sees their rules.
_extra_rules = [
    "* " + _t.llm_rule for _t in _tpl.all_templates()
    if _t.name not in _BANNED_TEMPLATES          # never teach the LLM banned kinds
    and f"* {_t.name}->" not in _EDITOR_RULES
    and f"* {_t.name} ->" not in _EDITOR_RULES
    and f"* {_t.name:<8s}->" not in _EDITOR_RULES
]
if _extra_rules:
    _EDITOR_RULES = _EDITOR_RULES + " " + " ".join(_extra_rules)

# V3.3 — MOTION-GRAPHICS VOCABULARY (safe unlock). Teaches the editor LLM the
# premium structured cards it may REQUEST, behind one hard factual-safety rule.
# Body grammar is `key=value` with `|`-separated items; the pipeline adapter parses
# + clamps, and any card whose REAL inputs aren't present simply falls back to
# footage — so omitting a card is always safe.
_MG_VOCAB = (
    " MOTION-GRAPHICS VOCABULARY (use SPARINGLY — footage-first; at most one of "
    "these per ~4 scenes). HARD FACTUAL RULE: NEVER invent a statistic, "
    "percentage, value, ranking, comparison, quote, date, size, or route to "
    "trigger a card. Request one ONLY when the REAL data is ALREADY stated in "
    "THIS scene's narration, and copy those EXACT values; if you would have to "
    "make a number up, DO NOT request the card (return no graphic). "
    "VAGUE QUANTIFIERS ARE NOT DATA — words like 'most', 'many', 'several', "
    "'a huge increase', 'a large share', 'millions', 'rapidly', 'major', 'one "
    "of the biggest', 'years later' contain NO usable number, date, or split. "
    "On any scene whose only quantities are vague like these, use NO data card. "
    "This ban covers EVERY number / statistic / chart / percentage / currency / "
    "ranking / count / measurement / date / timeline / comparison card — "
    "including bar_chart, versus, balance, composition, sankey, gauge, eras, "
    "ranking, chart, number, stat, stat_dashboard, stat_insight, "
    "vertical_bar_chart, donut_chart, progress_bar, line_chart, timeline, "
    "mini_timeline, typing_date, comparison, currency_stat, numerical_ratio, "
    "demographic_split, document_stack, tally_counter, score_display, "
    "speedometer, heatmap_grid, stats_bar, calendar_grid, receipt and "
    "era_banner — ALL forbidden. Return no graphic and let the footage carry "
    "it. A data card REQUIRES explicit figures, named subjects, or explicit "
    "dates written verbatim in the scene; copy those EXACT values and never "
    "round, infer, or invent a default. "
    "* bar_chart -> animated bars; ONLY when the scene states >=2 labelled "
    "numeric values. t=title b='bars=Pipelines:40|Rail:25|Refineries:35;suffix=%'. "
    "Not for a single number (use number) and never an invented split. "
    "* versus -> contrast TWO real named subjects. t=title "
    "b='pair=Standard Oil|The Independents;values=90|10;suffix=%' (values optional, "
    "only if REAL). "
    "* balance -> a real two-sided tradeoff. b='pair=Speed|Safety;values=7|3'. "
    "* composition -> a real share/breakdown of ONE whole. "
    "b='segments=Crude:50|Refined:30|Export:20;suffix=%'. "
    "* hierarchy -> a real named structure. t=root b='children=Domestic|Export|Pipelines'. "
    "* before_after -> a real then/now. b='before=1865;after=1882'. "
    "* decision -> a real decision + real outcome. t=question b='yes=Absorbed|no=Driven out|chosen=yes'. "
    "* sankey -> a real allocation. b='branches=Reinvest:50|Dividends:30|Reserves:20'. "
    "* eras -> real DATED periods. b='eras=Rise:1865-1882|Peak:1882-1904|Fall:1904-1911'. "
    "* headlines -> real/paraphrased press lines FROM the script; never fabricate "
    "outlets or quotes. b='headlines=Trust on trial|Court orders breakup'. "
    "* gauge -> a real measured value on a real scale. b='value=72;bands=LOW|HIGH;readout=72 dB'. "
    "* scale_compare -> real TRUE-SCALE sizes (NEVER guess sizes). b='items=Carrier:333:333 m|Bus:12:12 m'. "
    "* route_trace -> a visual path over the CURRENT shot using SCREEN coords 0..1 "
    "(NOT geography). b='points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End'. "
    "Do NOT request maps, geographic routes between places, or spread/heat cards "
    "— those require verified geography and are added manually, not by you."
)
_EDITOR_RULES = _EDITOR_RULES + _MG_VOCAB


_STORY_ROLES = (
    "hook", "context", "problem", "stakes", "evidence", "reaction",
    "escalation", "turn", "reveal", "proof", "climax", "payoff",
    "resolution",
)

# The showrunner LLM rarely returns these EXACT tokens (it says "intro",
# "rising action", "the reveal", "conclusion" …). The old code nulled
# anything not an exact match, so EVERY role came back "" and the whole
# role-driven pacing / hook+payoff treatment silently never fired (the
# "AI arranging clips" tell). Map free-form labels onto the canon.
_ROLE_SYNONYMS = {
    "hook": ("hook", "intro", "introduction", "opening", "open",
             "teaser", "cold", "attention", "grabber"),
    "context": ("context", "setup", "set-up", "background", "premise",
                "exposition", "establishing", "scene-set"),
    "problem": ("problem", "conflict", "issue", "challenge", "tension-"),
    "stakes": ("stakes", "consequence", "consequences", "matters",
               "urgency", "why"),
    "evidence": ("evidence", "example", "demonstration", "data", "case",
                 "illustration"),
    "reaction": ("reaction", "response", "human", "emotional", "witness",
                 "testimony"),
    "escalation": ("escalation", "buildup", "build-up", "build", "rising",
                   "intensify", "tension", "ramp", "mounting"),
    "turn": ("turn", "twist", "pivot", "shift", "turning"),
    "reveal": ("reveal", "revelation", "discovery", "unveil", "exposed",
               "uncover"),
    "proof": ("proof", "validation", "confirmation", "study",
              "scientific", "science", "verification"),
    "climax": ("climax", "peak", "height", "crescendo", "apex",
               "culmination"),
    "payoff": ("payoff", "pay-off", "result", "outcome", "takeaway",
               "lesson", "reward", "implication"),
    "resolution": ("resolution", "conclusion", "ending", "outro",
                   "closing", "wrap", "denouement", "final", "finale",
                   "summary", "close", "end"),
}


def _norm_role(r: str) -> str:
    """Loosely map the LLM's free-form beat label onto _STORY_ROLES so
    the showrunner's structure actually reaches the editor (was nulled
    on any non-exact match)."""
    r = (r or "").strip().lower()
    if not r:
        return ""
    if r in _STORY_ROLES:
        return r
    toks = set(re.findall(r"[a-z]+", r))
    for canon, trigs in _ROLE_SYNONYMS.items():
        for tg in trigs:
            if tg in toks or tg in r:
                return canon
    return ""


def _story_arc(title: str, scenes: list[Scene], client, model: str) -> dict:
    """SHOWRUNNER pass — Issue #4 (cinematic storytelling flow).

    The per-scene editor runs in blind 22-scene batches, so nothing ever
    saw the WHOLE film: intensity was decided locally and the result felt
    like 'clip after clip' with no real escalation. This single compact
    pass reads the entire documentary at once and designs its dramatic
    ARC — each scene's narrative ROLE (hook / problem / evidence /
    reaction / escalation / reveal / payoff …) and a target emotional
    intensity that deliberately rises toward the reveal, releases to
    breathe, and lands a payoff. Because every downstream system (beat
    pacing, Ken-Burns, cut plan) keys off `intensity`, a real arc here
    makes the EDIT itself escalate.

    Compact one-liners in, tiny JSON out, so it scales to any length.
    Returns {index: (role, intensity)}; {} on any failure (caller then
    keeps the per-scene estimate — no regression)."""
    n = len(scenes)
    lines = "\n".join(
        f"{s.index}|{' '.join(s.narration.split()[:14])}" for s in scenes
    )
    roles = "|".join(_STORY_ROLES)
    system = (
        "You are the SHOWRUNNER and story director of a documentary. You "
        "see the ENTIRE piece at once and design its dramatic ARC like a "
        "film director: a hook, rising tension, evidence and human "
        "reactions, the reveal/climax given its own beat, brief releases "
        "so it can breathe, then a deliberate payoff. Emotional intensity "
        "must form ONE coherent CURVE across the whole film — never flat, "
        "never random scene-to-scene noise."
    )
    user = (
        f"TITLE: {title}\nSCENES (number|gist):\n{lines}\n\n"
        "For EVERY scene give its narrative role and target emotional "
        "intensity 1(calm)..5(peak), shaped as ONE arc for the whole "
        "film: ease in, escalate toward the central reveal, release "
        "after big beats so it breathes, end on a deliberate note. NOT a "
        "flat line, NOT random. Most scenes 2-3; reserve 5 for the true "
        f'climax. Return ONLY JSON {{"a":[{{"i":<n>,"r":"{roles}",'
        '"x":1-5}, ... exactly one per scene]}.'
    )
    msg = client.messages.create(
        model=model, max_tokens=4096, system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content
                  if getattr(b, "type", "") == "text")
    data = _parse_json(raw)
    arr = data.get("a") or data.get("arc") or data.get("scenes") or []
    out: dict = {}
    for d in arr:
        if not isinstance(d, dict):
            continue
        try:
            i = int(d.get("i", -1))
            x = int(d.get("x", 0) or 0)
        except (TypeError, ValueError):
            continue
        r = _norm_role(str(d.get("r", "") or ""))
        if 0 <= i < n and 1 <= x <= 5:
            out[i] = (r, x)
    return out


def _move_title(scenes: list, src: int, dst: int) -> None:
    scenes[dst].graphic_kind = "title_card"
    scenes[dst].graphic_text = scenes[src].graphic_text
    scenes[dst].graphic_body = scenes[src].graphic_body
    scenes[src].graphic_kind = ""
    scenes[src].graphic_text = ""
    scenes[src].graphic_body = ""


def _relocate_title_card(scenes: list) -> None:
    """IMP_004 — NO-OP BY DEFAULT (2026-05-30, user directive: NO automatic
    title relocation whatsoever).

    Vidlore never moves, delays, or forces a title card by default.  The
    title stays EXACTLY where the script / editor placed it — or, if the
    editor placed none, the video simply has no title card.  Whether a
    title appears and where it appears is purely an editorial / script
    decision, including a title on scene 0.

    Two legacy behaviours survive ONLY as explicit opt-ins (default off):
      • VIDLORE_TITLE_COLD_OPEN=1 → if a low-intensity title sits on scene
        0, nudge it to the next free scene so the film opens on footage.
      • VIDLORE_RELOCATE_TITLE=1  → the old push-0/1/2 → 3+ delayed-title.
    With neither flag set (the DEFAULT), this function does nothing.
    """
    import os
    if len(scenes) < 2:
        return
    _legacy = os.environ.get("VIDLORE_RELOCATE_TITLE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    _coldopen = os.environ.get(
        "VIDLORE_TITLE_COLD_OPEN", "0").strip().lower() in (
        "1", "true", "yes", "on")
    if not (_legacy or _coldopen):
        return                                  # DEFAULT — no relocation at all
    ti = next((i for i, s in enumerate(scenes)
               if (s.graphic_kind or "") == "title_card"), None)
    if ti is None:
        return

    # OPT-IN legacy: the old generic delayed-title (push 0/1/2 → 3+).
    if _legacy:
        if ti >= 3 or int(getattr(scenes[ti], "intensity", 0) or 0) >= 4:
            return
        hi = len(scenes)
        target = next((j for j in range(3, min(6, hi))
                       if not (scenes[j].graphic_kind or "")), None)
        if target is None:
            target = next((j for j in range(min(6, hi) - 1, ti, -1)
                           if not (scenes[j].graphic_kind or "")), None)
        if target is None or target == ti:
            return
        _move_title(scenes, ti, target)
        return

    # OPT-IN cold-open: only a low-intensity title literally on scene 0.
    if ti != 0:
        return
    if int(getattr(scenes[0], "intensity", 0) or 0) >= 4:
        return
    target = next((j for j in (1, 2) if j < len(scenes)
                   and not (scenes[j].graphic_kind or "")), None)
    if target is None:
        return
    _move_title(scenes, 0, target)


def _promote_statement_cards(scenes: list) -> None:
    """IMP_008 — the full-screen 'statement' card (light/near-black bg, one
    bold line) is a deliberate contemplative pause on a PIVOTAL sentence —
    a documentary device the editor LLM consistently UNDER-selects. So we
    deterministically promote the single strongest thesis / revelation beat
    to a statement card when it would otherwise play as plain footage.

    Conservative by design:
      • at most ONE promotion per video (skip if the editor already placed
        a statement);
      • never scene 0/1 (protect the cold-open hook);
      • never clobber an existing graphic on that scene;
      • only fires on a genuinely landable line — short (<12 words) AND
        high intensity (>=4), or a single-sentence revelation/turn/climax;
      • the downstream per-kind cap + global density cap still have final
        say, so this can only ADD restraint-friendly emphasis, never spam.
    """
    if len(scenes) < 4:
        return
    if any((getattr(s, "graphic_kind", "") or "") == "statement"
           for s in scenes):
        return                                  # editor already placed one
    _PEAK = {"reveal", "climax", "turn", "twist", "thesis"}
    best_i, best_score = -1, 0.0
    for i, s in enumerate(scenes):
        if i < 2:                               # protect the cold-open hook
            continue
        if (getattr(s, "graphic_kind", "") or ""):
            continue                            # don't displace a real card
        narr = re.sub(r"\s+", " ", (getattr(s, "narration", "") or "")).strip()
        if not narr:
            continue
        wc = len(narr.split())
        inten = int(getattr(s, "intensity", 0) or 0)
        role = (getattr(s, "role", "") or "").lower()
        n_sent = sum(narr.count(p) for p in (".", "!", "?")) or 1
        score = 0.0
        if wc < 12 and inten >= 4:              # short punch at high energy
            score += 2.0
        if role in _PEAK and n_sent <= 1:       # single-sentence revelation
            score += 1.6
        if role in _PEAK:
            score += 0.8
        if wc <= 8:                             # very tight = very landable
            score += 0.6
        if score > best_score:
            best_score, best_i = score, i
    if best_i >= 0 and best_score >= 2.0:
        s = scenes[best_i]
        s.graphic_kind = "statement"
        # Card text = the FIRST sentence (the line meant to land), not the
        # whole paragraph; the renderer wraps/fits and recovers full
        # sentences, so a short clean line stays clean.
        narr = re.sub(r"\s+", " ", (s.narration or "")).strip()
        m = re.search(r"[.!?]", narr)
        s.graphic_text = _clip_words((narr[:m.end()] if m else narr).strip(), 140)
        s.graphic_body = ""


# ---- NATURAL MAP TRIGGERING (geo-enrichment) ----------------------- #
# The LLM editor reliably renders maps but UNDER-SELECTS them. This
# deterministic pass reads each undecided scene for STRONG geographic
# signals and assigns the right map template — only on confident matches
# (gazetteer-validated), capped + spaced, so it never feels forced.
_GAZ_AREA = frozenset(w.lower() for w in (
    # countries / nations (common subset)
    "Afghanistan Albania Algeria Argentina Armenia Australia Austria "
    "Azerbaijan Bangladesh Belarus Belgium Bolivia Bosnia Brazil Bulgaria "
    "Cambodia Cameroon Canada Chile China Colombia Croatia Cuba Cyprus "
    "Czechia Denmark Egypt England Estonia Ethiopia Finland France Georgia "
    "Germany Greece Hungary Iceland India Indonesia Iran Iraq Ireland "
    "Israel Italy Japan Jordan Kazakhstan Kenya Korea Kuwait Laos Latvia "
    "Lebanon Libya Lithuania Luxembourg Malaysia Mexico Mongolia Morocco "
    "Myanmar Nepal Netherlands Nigeria Norway Pakistan Palestine Peru "
    "Philippines Poland Portugal Qatar Romania Russia Rwanda Scotland "
    "Serbia Singapore Slovakia Slovenia Somalia Spain Sudan Sweden "
    "Switzerland Syria Taiwan Tajikistan Tanzania Thailand Tunisia Turkey "
    "Turkmenistan Uganda Ukraine Uzbekistan Venezuela Vietnam Wales Yemen "
    "Zimbabwe "
    # historic regions / lands (treated as AREAS -> map_region)
    "Mesopotamia Persia Babylon Byzantium Anatolia Bavaria Bohemia "
    "Catalonia Galilee Gaul Kashmir Kurdistan Manchuria Normandy Prussia "
    "Punjab Sahara Siberia Sinai Tibet Transylvania"
).split())

# Cities -> a single POINT (map_reveal), never a region highlight.
_GAZ_CITIES = frozenset(w.lower() for w in (
    "Amsterdam Athens Baghdad Bangkok Beijing Beirut Berlin Bogota Boston "
    "Brussels Budapest Cairo Chicago Constantinople Copenhagen Damascus "
    "Delhi Dubai Dublin Edinburgh Florence Geneva Hamburg Hanoi Havana "
    "Helsinki Hiroshima HongKong Istanbul Jakarta Jerusalem Kabul Karachi "
    "Kiev Kyiv Lagos Lima Lisbon London Madrid Manila Marseille Mecca "
    "Melbourne Milan Moscow Mumbai Munich Nagasaki Nairobi Naples "
    "NewYork Odessa Osaka Oslo Paris Prague Pyongyang Rome Saigon "
    "Samarkand Seoul Shanghai Singapore Stalingrad Stockholm Sydney "
    "Tehran TelAviv Tokyo Toronto Tripoli Venice Vienna Warsaw Washington "
    "Xian Zurich"
).split())

_GAZ = _GAZ_AREA | _GAZ_CITIES

_PLACE = r"[A-Z][a-zA-Z.'’-]+(?:\s+[A-Z][a-zA-Z.'’-]+){0,2}"
_ROUTE_RE = re.compile(
    rf"\bfrom\s+({_PLACE})\s+(?:to|through|via|into|toward|towards)\s+"
    rf"({_PLACE})", re.UNICODE)
_REGION_RE1 = re.compile(
    rf"\b({_PLACE})\s+(?:was|is|became|remained)\s+(?:a|an|the)\s+"
    r"(?:divided\s+|occupied\s+|contested\s+)?(?:country|nation|empire|"
    r"kingdom|territory|region|province|state|republic|land)\b")
_REGION_RE2 = re.compile(
    r"\b(?:the\s+)?(?:region|territory|province|empire|kingdom|land)\s+of\s+"
    rf"({_PLACE})")
_POINT_RE = re.compile(
    rf"\b(?:in|at|near|the city of|capital of|outside|out of|across|"
    rf"back to|reached|arrived in|fled to|escaped to)\s+({_PLACE})")
# AREA / territory context — a gazetteer place near any of these strong
# region cues reads as a TERRITORY, not a single point.
_REGION_CUE = re.compile(
    r"\b(wall|border|frontier|divided|occupied|annexed|territory|homeland|"
    r"surveilled|patch of land|iron curtain|behind the curtain|"
    r"the (?:east|west|north|south)\b|controlled|regime|nation|empire)\b",
    re.I)
_STOPWORDS = {"the", "a", "an", "of", "and", "but", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday",
              "january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november",
              "december", "god", "lord", "president", "general", "doctor"}


def _clean_place(p: str) -> str:
    p = re.sub(r"[^A-Za-z .'’-]", "", (p or "")).strip(" .'-’")
    # drop a trailing stopword (e.g. "Berlin the" -> "Berlin")
    toks = p.split()
    while toks and toks[-1].lower() in _STOPWORDS:
        toks.pop()
    return " ".join(toks)


def _is_place(p: str) -> bool:
    if not p:
        return False
    toks = [t for t in p.split() if t]
    if any(t.lower() in _STOPWORDS for t in toks):
        return False
    flat = "".join(toks).lower()
    return (p.lower() in _GAZ or flat in _GAZ
            or any(t.lower() in _GAZ for t in toks))


def _is_area(p: str) -> bool:
    """A country / nation / historic region (deserves map_region), as
    opposed to a single city (which deserves a point map_reveal)."""
    toks = [t for t in (p or "").split() if t]
    flat = "".join(toks).lower()
    return ((p or "").lower() in _GAZ_AREA or flat in _GAZ_AREA
            or any(t.lower() in _GAZ_AREA for t in toks))


def _first_gaz_place(txt: str) -> str:
    """First capitalized token-group in the text that is a known place."""
    for m in re.finditer(_PLACE, txt):
        p = _clean_place(m.group(0))
        if _is_place(p):
            return p
    return ""


# Soft / auto-assigned graphic kinds a STRONG geo signal (a real route or a
# territory) is allowed to replace with a map — geography docs were getting
# crowded out because document/number/quote auto-fire on the same scenes. A
# map of a courier route is more valuable than a generic ARCHIVE-RECORD card.
# High-value special panels (timeline, dossier, stat_dashboard, ...) are NOT
# in this set, so they are never overridden.
_GEO_SOFT = frozenset({"", "quote_highlight", "document", "number",
                       "bullet_list", "section_title"})


def _geo_enrich(scenes: list, mode=None, *, budget: int = 3) -> int:
    """Assign map templates on strong geo signals. Returns the number
    injected. Conservative: gazetteer-validated, capped, spaced. Fills
    UNDECIDED scenes, and may also replace a SOFT auto-assigned kind
    (document/number/quote) when the geo signal is STRONG (a real route
    between two places, or a territory/region) — never a high-value panel,
    and a lone city-point only ever fills a fully undecided scene."""
    avoid = set(getattr(mode, "avoid_templates", ()) or ())
    # Is this a GEOGRAPHY-HEAVY documentary? Count distinct gazetteer places
    # across the whole script. If many (>=4), geography is central to the
    # story, so a lone city locator is worth a map even over a soft auto-card;
    # in a normal doc an incidental city mention should NOT trigger a map.
    _places: set = set()
    for _sc in scenes:
        for _m in re.finditer(_PLACE, _sc.narration or ""):
            _p = _clean_place(_m.group(0))
            if _is_place(_p):
                _places.add(_p.lower())
    geo_heavy = len(_places) >= 4
    placed, last = 0, -10
    for i, sc in enumerate(scenes):
        if placed >= budget:
            break
        cur = sc.graphic_kind or ""
        if (i - last) < 2:
            continue
        if cur not in _GEO_SOFT:           # never override a high-value panel
            continue
        soft_override = cur != ""          # replacing an auto-assigned card
        txt = sc.narration or ""

        def _set(kind, text):
            nonlocal placed, last
            sc.graphic_kind, sc.graphic_text, sc.graphic_body = kind, text, ""
            sc._geo = True                         # protect from density cap
            placed += 1
            last = i

        # ROUTE (strong) — A->B between two real places. May override a soft.
        m = _ROUTE_RE.search(txt)
        if m and "map_route" not in avoid and "map_route" in _GRAPHIC_KINDS:
            a, b = _clean_place(m.group(1)), _clean_place(m.group(2))
            if a and b and a.lower() != b.lower() and (
                    _is_place(a) or _is_place(b)):
                _set("map_route", f"{a}|{b}")
                continue

        # REGION (strong) — strict ("X was a territory") OR a known AREA near a
        # cue (wall / border / divided / occupied). May override a soft kind.
        m = _REGION_RE1.search(txt) or _REGION_RE2.search(txt)
        region_place = _clean_place(m.group(1)) if m else ""
        strict = bool(region_place and _is_place(region_place))
        if not strict and _REGION_CUE.search(txt):
            region_place = _first_gaz_place(txt)
        if region_place and _is_place(region_place):
            if (strict or _is_area(region_place)) \
                    and "map_region" not in avoid \
                    and "map_region" in _GRAPHIC_KINDS:
                _set("map_region", region_place)
                continue
            # a lone city point — fills undecided scenes; may override a soft
            # card only in a GEOGRAPHY-HEAVY doc (a locator earns its place).
            if ((geo_heavy or not soft_override) and not _is_area(region_place)
                    and "map_reveal" not in avoid
                    and "map_reveal" in _GRAPHIC_KINDS):
                _set("map_reveal", region_place)   # city -> point
                continue

        # POINT reveal (weak) — a single city locator. Fills undecided scenes
        # (matched via the strict _POINT_RE preposition pattern); in a
        # geography-heavy doc it may also override a soft auto-card, using the
        # broader first-gazetteer-place so a key city (e.g. the opening
        # locator) still earns its map even without an "in/at X" phrasing.
        if "map_reveal" not in avoid and "map_reveal" in _GRAPHIC_KINDS:
            place = ""
            for m in _POINT_RE.finditer(txt):
                cand = _clean_place(m.group(1))
                if _is_place(cand):
                    place = cand
                    break
            if not place and geo_heavy:
                place = _first_gaz_place(txt)
            # reduce a phrase like "Then Prague" / "CIA's West Berlin" to its
            # actual gazetteer city token so the map geocodes cleanly.
            if place and place.lower() not in _GAZ:
                for _tok in place.split():
                    if _tok.lower() in _GAZ:
                        place = _tok
                        break
            if place and _is_place(place) and (geo_heavy or not soft_override):
                _set("map_reveal", place)
    if placed:
        print(f"  [geo] {placed} map scene(s) auto-enriched from geography",
              flush=True)
    return placed


# control / territory relationship cue — the "who-controls-where" signal that
# turns a person + a place into a *tether* beat (not just an incidental
# mention). Deliberately demanding so a passing geography reference never
# trips it: there must be a verb of rule / command / basing.
_FIGLOC_CTRL = re.compile(
    r"\b(rul(?:e|ed|er|es)|controll?ed?|controls|seized|command(?:ed|s)?|"
    r"reign(?:ed|s)?|dictator|presiden(?:t|cy)|regime|empire|stronghold|"
    r"based\s+(?:in|out\s+of)|headquarter\w*|conquer(?:ed|s)?|occupied|"
    r"governed?|presided|throne|seat\s+of\s+power|operated\s+(?:in|from))\b",
    re.I)


def _promote_figure_locator(scenes: list, avoid: set | None = None) -> None:
    """IMP_021 — the 'who-controls-where' tether (a small portrait badge tied
    by one leader line to a marker on the REAL map of the place a key figure
    controlled). Like the statement card and name_reveal, this rare,
    high-value panel is consistently UNDER-selected by the editor LLM, so we
    deterministically promote ONE strong beat when the doc clearly earns it.

    TIGHT, low-false-positive gates (only a genuine person<->place doc fires):
      • template enabled, not banned/avoided, and not already used by the LLM;
      • the doc must ALREADY carry a name_reveal — i.e. a validated
        protagonist exists; we reuse THAT name so the badge matches the film's
        subject (no risk of inventing a person);
      • the host scene must carry no graphic yet, sit past the cold-open
        (idx >= 2), and its narration must contain BOTH a control/territory
        cue AND a gazetteer-validated place;
      • at most ONE promotion; the scene is marked `_geo` so the global
        density cap keeps it (a located leader is a signature beat).
    The downstream per-kind cap (max 1) + min_gap still have the final say.
    """
    avoid = avoid or set()
    if "figure_locator" in avoid or "figure_locator" not in _GRAPHIC_KINDS:
        return
    if len(scenes) < 4:
        return
    if any((getattr(s, "graphic_kind", "") or "") == "figure_locator"
           for s in scenes):
        return                                  # editor already placed one
    # protagonist = an existing name_reveal's NAME (strongest), else the first
    # narration that the conservative character-intro detector recognises as a
    # real introduction. Either way the name is validated, never invented.
    person = ""
    for s in scenes:
        if (getattr(s, "graphic_kind", "") or "") == "name_reveal":
            nm = (getattr(s, "graphic_text", "") or "").strip()
            if "::" in nm:
                nm = nm.split("::", 1)[0].strip()
            if nm:
                person = nm
                break
    if not person:
        for s in scenes:
            hit = _detect_character_intro(getattr(s, "narration", "") or "")
            if hit and hit[0].strip():
                person = hit[0].strip()
                break
    if not person:
        return                                  # not a person-centric doc
    best_i, best_place, best_score = -1, "", 0.0
    for i, s in enumerate(scenes):
        if i < 2:                               # protect the cold-open
            continue
        if (getattr(s, "graphic_kind", "") or ""):
            continue                            # never clobber a real card
        narr = re.sub(r"\s+", " ", (getattr(s, "narration", "") or "")).strip()
        if not narr or not _FIGLOC_CTRL.search(narr):
            continue
        place = _first_gaz_place(narr)
        if not place or not _is_place(place):
            continue
        role = (getattr(s, "role", "") or "").lower()
        inten = int(getattr(s, "intensity", 0) or 0)
        score = 1.0
        if role in {"reveal", "turn", "escalation", "climax", "rise"}:
            score += 1.2
        if inten >= 4:
            score += 0.8
        score += max(0.0, 0.6 - i * 0.03)       # prefer an earlier locator
        if score > best_score:
            best_score, best_i, best_place = score, i, _clean_place(place)
    if best_i >= 0 and best_place:
        s = scenes[best_i]
        s.graphic_kind = "figure_locator"
        s.graphic_text = person.upper()[:20]
        s.graphic_body = best_place
        try:
            s._geo = True                       # protect from the density cap
        except Exception:                       # noqa: BLE001
            pass
        print(f"  [figloc] tether promoted: {s.graphic_text} -> "
              f"{best_place} (scene {best_i})", flush=True)


def _recipe_density_mult() -> float:
    """REC density wiring — the per-video recipe's graphics-density bias as
    a BOUNDED multiplier on the global graphics cap (max_g = n*0.45*density).
    A 'density' recipe shows a few more cards; a 'restraint' recipe stays
    minimal.  Clamped to a gentle ±22% so it can NEVER spam — scene role,
    per-template max_per_video, and min_gap_scenes still win.  1.0 when no
    recipe.  VIDLORE_RECIPE_DENSITY=0 disables."""
    import os
    if os.environ.get("VIDLORE_RECIPE_DENSITY", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return 1.0
    try:
        from .look_dna import look_get as _lg
        d = _lg("density")
        if isinstance(d, (int, float)) and d > 0:
            return round(max(0.78, min(1.22, float(d))), 3)
    except Exception:                                              # noqa: BLE001
        pass
    return 1.0


def _apply_graphic_caps(scenes: list, mode=None) -> None:
    """Enforce registry-driven caps + variation spacing so the
    rare/special panels stay rare AND no template repeats back-to-back.

    Two rules, both read from `vidlore/templates/__init__.py`:

      • max_per_video — hard cap. Once a kind has fired N times in
        this video, every subsequent occurrence is silently downgraded
        to no-graphic (keeps the panel feeling special).

      • min_gap_scenes — variation spacer. If template X fired at
        scene i, the same template won't fire again until at least
        `min_gap_scenes` scenes later. Prevents "every other scene is
        a callout" template fatigue.

    Caps win over gaps (a kind capped out is downgraded regardless of
    gap). Downgrades clear graphic_kind/text/body so the assembler
    treats the scene as a clean footage cut (template #19 from the
    user wishlist: "not every scene needs effects").

    STYLE MODE BIAS (when `mode` is given). A Style Mode's *template
    personality* is enforced here so the choice shows up on screen, not
    just in the pacing/music:
      • avoid_templates — kinds that clash with the personality are
        downgraded to clean footage BEFORE anything else (a Netflix
        Historical Epic never flashes an SMS bubble or a breaking-news
        banner; a True Crime board never shows an era banner).
      • graphic_density — scales the global density cap (>1 a few more
        cards, <1 more restraint).
      • fav_templates — when the density cap has to drop graphics, the
        mode's signature templates are protected first.
    """
    fav: set = set()
    avoid: set = set()
    density = 1.0
    type_stamps = False
    if mode is not None:
        fav = set(getattr(mode, "fav_templates", ()) or ())
        avoid = set(getattr(mode, "avoid_templates", ()) or ())
        density = float(getattr(mode, "graphic_density", 1.0) or 1.0)
        type_stamps = bool(getattr(mode, "type_stamps", False))
    # REC density wiring — the per-video recipe nudges graphics FREQUENCY
    # within a bounded range (applies with or without a StyleMode). The
    # per-template caps + min_gap + scene-role rules below still win.
    density *= _recipe_density_mult()

    # (-1) GLOBAL BAN — strip permanently disabled full-screen infographic
    # cards (era_banner, comparison, stat_dashboard) from any scene,
    # regardless of niche or style mode. This catches leftovers from reused
    # script.json files built before v15, or any LLM edge-case that slips
    # through the kind-enum and rules filters above.
    for sc in scenes:
        if (sc.graphic_kind or "") in _BANNED_TEMPLATES:
            sc.graphic_kind = ""
            sc.graphic_text = ""
            sc.graphic_body = ""

    # (0-) TYPEWRITER STAMPS — in modes that lean archival (Homestead Gold),
    # a place/era 'location' anchor is rendered as the TYPING-DATE typewriter
    # (char-by-char reveal + synced typing clicks) instead of a plain title
    # bar — the faceless-doc 'archive stamp' feel the user expects.
    if type_stamps:
        for sc in scenes:
            if (sc.graphic_kind or "") == "location" and (sc.graphic_text
                                                           or "").strip():
                sc.graphic_kind = "typing_date"
                # typing_date uses t=text, b=optional context (keep any)
                if not (sc.graphic_body or "").strip():
                    sc.graphic_body = ""

    # (0) personality clash — drop off-brand templates up front so the
    # density cap below then chooses only from on-personality graphics.
    if avoid:
        for sc in scenes:
            if (sc.graphic_kind or "") in avoid:
                sc.graphic_kind = ""
                sc.graphic_text = ""
                sc.graphic_body = ""

    # (0b) TITLE-CARD COLD-OPEN LEAD-IN. The title card is an opaque
    # full-frame beat; if it lands on the FIRST scene the video literally
    # opens on the title (no hook), which feels cheap. A premium doc
    # opens with a few seconds of footage + the hook line, THEN drops the
    # title. So if scene 0 carries the title card, slide it to the next
    # free scene (within the first three) — footage now opens the film.
    _relocate_title_card(scenes)

    # IMP_008 — promote one pivotal beat to a full-screen statement card
    # (unless this personality bans the template). Runs BEFORE the caps so
    # the promotion is subject to the same density discipline as any card.
    if "statement" not in (avoid or set()):
        _promote_statement_cards(scenes)

    # IMP_021 — promote one 'who-controls-where' tether (portrait badge ->
    # marker on the real map of the place a key figure controlled). The
    # editor LLM under-selects this rare panel; the promoter only fires when
    # the doc already has a protagonist AND a scene names a control cue + a
    # real place. Runs BEFORE the caps so it obeys the same density rules.
    _promote_figure_locator(scenes, avoid)
    # COLD-OPEN GUARD — figure_locator is a full-frame opaque map composite;
    # if the editor LLM placed it on scene 0 the film would open ON the map
    # instead of footage + the hook line (same reason _relocate_title_card
    # exists). Downgrade a scene-0 tether to clean footage — the subject
    # still gets a portrait via name_reveal, and the film opens on footage.
    if scenes and (scenes[0].graphic_kind or "") == "figure_locator":
        scenes[0].graphic_kind = ""
        scenes[0].graphic_text = ""
        scenes[0].graphic_body = ""

    seen: dict[str, int] = {}                  # kind -> count so far
    last_idx: dict[str, int] = {}              # kind -> last scene index
    for i, sc in enumerate(scenes):
        k = sc.graphic_kind or ""
        if not k:
            continue
        cap = _CAPS.get(k)
        gap = _MIN_GAPS.get(k, 0)
        capped_out = cap is not None and seen.get(k, 0) >= cap
        too_close = (k in last_idx and (i - last_idx[k]) < gap)
        if capped_out or too_close:
            sc.graphic_kind = ""
            sc.graphic_text = ""
            sc.graphic_body = ""
            continue
        seen[k] = seen.get(k, 0) + 1
        last_idx[k] = i

    # GLOBAL DENSITY CAP (Visual Restraint — full-doc QA finding). The
    # per-kind caps above stop ONE template repeating, but on a short
    # script the LLM happily puts a DIFFERENT card on every scene — 8/8
    # scenes carded came back as an "effect demo reel", and the wall of
    # cards also defeated caption suppression. A real editor uses graphics
    # sparingly: keep the cold-open title, then only the highest-value
    # beats, never two cards back-to-back.
    #
    # P1.3 (2026-06-05) — FOOTAGE-FIRST restraint. A real render came back
    # with too many FULL-SCREEN cards back-to-back (slideshow feel: "$37
    # MILLION" → "$40 MILLION" → "SIX DECADES ZERO PROGRESS"). The reference
    # competitor keeps cards to ~15-20% of runtime, footage-first. Lowering
    # the per-scene multiplier from 0.45 → 0.34 makes carded scenes the
    # exception (this is a coarse SCENE-COUNT gate covering ALL card kinds
    # incl. footage-backed overlays; the precise FULL-SCREEN time budget +
    # consecutive/cooldown rules run in pipeline.py's overlay-restraint pass
    # where the ordered scene list AND per-scene seconds are available).
    n = len(scenes)
    gidx = [i for i, sc in enumerate(scenes) if sc.graphic_kind]
    max_g = max(2, round(n * 0.34 * density))
    if len(gidx) > max_g:
        _PRIO = {
            "climax": 6, "reveal": 6, "turn": 5, "evidence": 4,
            "proof": 4, "stakes": 4, "problem": 3, "escalation": 3,
            "hook": 3, "payoff": 2, "context": 2, "reaction": 2,
            "resolution": 2,
        }
        keep: set = set()
        # always keep one cold-open title card if present
        for i in gidx:
            if scenes[i].graphic_kind == "title_card":
                keep.add(i)
                break
        # PROTECT name_reveal — every character intro deserves its single
        # portrait moment. Without this, density cull strips name_reveal
        # whenever it sits next to a title_card (very common because both
        # tend to land near the cold-open), and the protagonist's face
        # never appears on screen. Bug fix 2026-05-26.
        for i in gidx:
            if scenes[i].graphic_kind == "name_reveal":
                keep.add(i)
        # PROTECT deliberately geo-enriched maps — they were placed on a
        # strong, confident geographic signal, so the documentary's
        # geography survives the density cull (natural map triggering).
        for i in gidx:
            if getattr(scenes[i], "_geo", False):
                keep.add(i)
        # then highest story-value beats, spaced so no two cards touch.
        # A Style Mode's signature (fav) templates get a priority bump so
        # they survive the cull ahead of equal-value off-personality cards.
        cands = sorted(
            (i for i in gidx if i not in keep),
            key=lambda i: (
                -(_PRIO.get(scenes[i].role or "", 2)
                  + (2 if scenes[i].graphic_kind in fav else 0)),
                i,
            ),
        )
        for i in cands:
            if len(keep) >= max_g:
                break
            if any(abs(i - j) <= 1 for j in keep):
                continue                       # no back-to-back cards
            keep.add(i)
        for i in gidx:
            if i not in keep:
                scenes[i].graphic_kind = ""
                scenes[i].graphic_text = ""
                scenes[i].graphic_body = ""


# --------------------------------------------------------------------------- #
# SHOT VARIETY ENGINE  (Human-Editor Intelligence #1)
# --------------------------------------------------------------------------- #
# A real editor varies the *framing scale*, not just the shot label. Three
# "wides" in a row (establishing -> wide -> aerial) read as monotonous even
# though the labels differ, and so does detail -> macro -> detail (all
# tight). Grouping shot_types into SCALE FAMILIES lets the engine break the
# visual rhythm a viewer actually feels, instead of only blocking identical
# labels (the old guard's blind spot).
_SHOT_FAMILY = {
    "establishing": "wide", "wide": "wide", "aerial": "wide",
    "detail": "tight", "macro": "tight",
    "reaction": "face", "portrait": "face",
    "tracking": "motion",
    "archival": "archival",
}

# Which framings FIT a beat emotionally, best first — how a documentary
# editor actually reaches for a shot: faces on human/emotional beats, tight
# inserts to LAND an impact, wide/aerial to let calm or scale breathe,
# tracking to build momentum. Used to pick an emotion-appropriate substitute
# when the variety engine has to break a repetition (never a blind pool).
_DEFAULT_PREF = ["wide", "detail", "establishing", "reaction",
                 "tracking", "macro", "aerial", "portrait"]
_ROLE_SHOT_PREF = {
    "hook":        ["establishing", "aerial", "wide", "detail",
                    "tracking", "reaction", "macro", "portrait"],
    "context":     ["wide", "establishing", "detail", "aerial",
                    "tracking", "reaction", "portrait", "macro"],
    "problem":     ["detail", "tracking", "wide", "reaction",
                    "macro", "establishing", "portrait", "aerial"],
    "stakes":      ["tracking", "detail", "wide", "reaction",
                    "aerial", "macro", "establishing", "portrait"],
    "evidence":    ["detail", "wide", "macro", "establishing",
                    "reaction", "tracking", "portrait", "aerial"],
    "reaction":    ["reaction", "portrait", "detail", "wide",
                    "macro", "tracking", "establishing", "aerial"],
    "escalation":  ["tracking", "detail", "macro", "wide",
                    "reaction", "aerial", "portrait", "establishing"],
    "turn":        ["detail", "tracking", "reaction", "wide",
                    "macro", "aerial", "portrait", "establishing"],
    "reveal":      ["detail", "macro", "aerial", "reaction",
                    "wide", "tracking", "portrait", "establishing"],
    "proof":       ["detail", "wide", "macro", "establishing",
                    "tracking", "reaction", "portrait", "aerial"],
    "climax":      ["macro", "detail", "reaction", "aerial",
                    "tracking", "wide", "portrait", "establishing"],
    "payoff":      ["wide", "establishing", "aerial", "reaction",
                    "portrait", "detail", "tracking", "macro"],
    "resolution":  ["wide", "establishing", "aerial", "portrait",
                    "reaction", "detail", "tracking", "macro"],
}


def _shot_pref(role: str, inten: int) -> list:
    """Emotion-fit framing order for a beat: the role sets the base taste;
    intensity nudges it (a hot beat leans tighter / impact, a quiet beat
    leans wider to breathe). Stable nudge — never a full reshuffle."""
    base = _ROLE_SHOT_PREF.get((role or "").lower(), _DEFAULT_PREF)
    inten = max(1, min(5, int(inten or 3)))
    if inten >= 4:
        boost = {"macro", "detail", "reaction", "aerial"}
    elif inten <= 2:
        boost = {"wide", "establishing", "aerial"}
    else:
        return list(base)
    return ([s for s in base if s in boost]
            + [s for s in base if s not in boost])


def _vary_shot_types(scenes: list) -> None:
    """SHOT VARIETY ENGINE. Walk the cut like an editor reviewing the
    timeline: keep the LLM's framing when it already varies, but break a
    repetition the viewer would feel — an identical shot back-to-back, or a
    THIRD shot of the same scale family in a row (wide->wide->wide,
    detail->macro->detail). The replacement is the most emotion-appropriate
    framing (per role+intensity) that avoids the previous/next shot and the
    recent scale families, so adjacent scenes feel intentionally varied.
    'archival' is never touched or introduced (it must stay archival for
    the vintage treatment). Mutates in place; deterministic."""
    fam = _SHOT_FAMILY
    n = len(scenes)

    def F(t: str) -> str:
        return fam.get(t or "", "")

    for k in range(n):
        cur = scenes[k].shot_type or ""
        if not cur or cur == "archival":
            continue
        prev_t = scenes[k - 1].shot_type if k - 1 >= 0 else ""
        prev2_t = scenes[k - 2].shot_type if k - 2 >= 0 else ""
        nxt_t = scenes[k + 1].shot_type if k + 1 < n else ""
        pf, p2f, cf = F(prev_t), F(prev2_t), F(cur)
        exact_repeat = bool(cur) and cur == prev_t
        scale_run3 = bool(cf) and cf == pf == p2f      # would be 3rd in a row
        if not (exact_repeat or scale_run3):
            continue                                   # editor's choice stands
        prefs = _shot_pref(scenes[k].role, scenes[k].intensity)
        recent_fams = {pf, p2f}
        nxt_f = F(nxt_t)
        chosen = ""
        # pass 1 — break the rhythm cleanly: a fresh scale family, not
        # colliding with the neighbours, and not about to start a new run.
        for c in prefs:
            ff = F(c)
            if c in (prev_t, nxt_t) or ff in recent_fams or ff == nxt_f:
                continue
            chosen = c
            break
        # pass 2 — relax the next-shot/look-ahead constraint.
        if not chosen:
            for c in prefs:
                if c == prev_t or F(c) == pf:
                    continue
                chosen = c
                break
        # pass 3 — at minimum, never the identical shot twice.
        if not chosen:
            for c in prefs:
                if c != prev_t:
                    chosen = c
                    break
        if chosen:
            scenes[k].shot_type = chosen


def _apply_editor_decisions(title: str, scenes: list, cfg: Config) -> None:
    """The EDITOR-BRAIN core, factored out so both code paths use it:
      * paste-script path  -> `analyze_script` calls this
      * LLM-generated path -> `_llm_script` calls this

    Previously only the paste path ran this, so videos generated from a
    prompt got narration + visual but NO shot_type / graphic / role /
    pacing decisions — that's why the user's last render had 0/13
    scenes with graphics and 0/13 with shot_type. Wiring this helper
    into BOTH paths fixes that completely.

    Operates IN-PLACE on the `scenes` list (mutates `.shot_type`,
    `.graphic_kind`, `.graphic_text`, `.graphic_body`, `.role`,
    `.intensity`, `.visual`, `.emphasis`, `.keywords`).
    """
    import anthropic

    if not scenes:
        return
    n = len(scenes)
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    # Issue #4 — design the whole-film dramatic arc FIRST, so the blind
    # per-scene batches below serve a real story (problem -> evidence ->
    # reaction -> reveal -> payoff) instead of "clip after clip".
    arc: dict = {}
    try:
        arc = _story_arc(title, scenes, client, cfg.anthropic_model)
    except Exception as e:  # noqa: BLE001
        print(f"  [story] arc pass failed ({e}); per-scene pacing only",
              flush=True)
    if arc:
        peak = max((v[1] for v in arc.values()), default=0)
        npk = sum(1 for v in arc.values() if v[1] >= 4)
        print(f"  [story] dramatic arc designed for {len(arc)}/{n} "
              f"scenes (peak {peak}/5, {npk} high-tension beats)",
              flush=True)

    system = (
        "You are a senior NETFLIX-grade documentary EDITOR. The narration "
        "is already cut into numbered scenes — you do NOT rewrite a single "
        "word. Each scene is tagged [role · arcN/5] = its place in the "
        "film's dramatic arc, decided by the showrunner. Your shot "
        "grammar, on-screen graphic and emphasis MUST serve that beat "
        "(hook=intrigue, problem/stakes=pressure, evidence/proof="
        "document or chart, reaction=human face, escalation=tighter "
        "shots, reveal/climax=give it room then hit, payoff/resolution="
        "wide and calm). Your 'int' MUST equal the scene's arc number so "
        "pacing escalates as designed. Vary the visual language."
    )

    def _sline(s: Scene) -> str:
        if s.index in arc:
            r, ax = arc[s.index]
            tag = f"[{r or 'beat'} arc{ax}/5]"
            return f"{s.index}|{tag} {s.narration}"
        return f"{s.index}| {s.narration}"

    valid_kinds = set(_tpl.all_names()) | _MG_UNLOCK_KINDS | _STRUCTURED_KINDS

    def _build_user(batch: list[Scene], strict: bool) -> str:
        lines = "\n".join(_sline(s) for s in batch)
        user = (
            f"TITLE: {title}\n\nSCENES (decide editing for EACH; the "
            f"returned 'i' MUST match the number shown):\n{lines}\n\n"
            'Return ONLY JSON {"d":[{"i":<scene number>,'
            '"kw":["concrete","filmable","nouns"],'
            '"vis":"one vivid photorealistic SHOT depicting exactly what '
            'this scene says — subject, setting, era, time of day, '
            'weather, camera move, lens, mood",'
            '"st":"establishing|aerial|wide|detail|macro|reaction|'
            'portrait|archival|tracking",'
            '"int":1-5,'
            '"emph":"single most charged word that appears VERBATIM in '
            'this scene",'
            '"g":{"k":"' + "|".join(
                [k for k in _tpl.llm_kind_enum().split("|")
                 if k not in _BANNED_TEMPLATES]   # never offer banned kinds to LLM
                + sorted(_MG_UNLOCK_KINDS | _STRUCTURED_KINDS)  # V3.3 unlocked MG kinds
            ) +
            '","t":"SHORT line / number / name","b":'
            '"see per-kind rule"}}, ... EXACTLY one object per scene '
            "above]} . "
            "LANGUAGE: the narration may be in ANY language. 'kw' and "
            "'vis' MUST be written in ENGLISH (they drive English "
            "stock-footage / image search — non-English here breaks "
            "matching). 'emph' MUST be copied VERBATIM from the scene in "
            "its ORIGINAL language (it is matched to the spoken words). "
            "Graphic 't' and 'b' in the SAME language as the narration "
            "(the viewer reads them on screen). " + _EDITOR_RULES
        )
        if strict:
            user += (
                "\n\nCRITICAL OUTPUT CONTRACT: respond with ONE compact, "
                "100% VALID JSON object and NOTHING else — no markdown, no "
                "code fences, no commentary before or after. Double-quote "
                "every key and string, NO trailing commas, escape any "
                "quotes inside strings, and ensure every { [ is closed. "
                "Keep each 'vis' under 200 characters so the response is "
                "not truncated."
            )
        return user

    def _decide(batch: list[Scene], strict: bool = False):
        """Call the LLM for a batch. Returns (decisions:list[dict], raw:str).
        Robustly recovers decisions even from broken JSON."""
        try:
            msg = client.messages.create(
                model=cfg.anthropic_model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user",
                           "content": _build_user(batch, strict)}],
            )
            raw = "".join(b.text for b in msg.content
                          if getattr(b, "type", "") == "text")
        except Exception as e:                         # noqa: BLE001
            print(f"  [editor] LLM call failed ({e})", flush=True)
            return [], ""
        obj = _loads_robust(raw)
        decisions = []
        if isinstance(obj, dict):
            decisions = (obj.get("d") or obj.get("scenes")
                         or obj.get("decisions") or [])
        elif isinstance(obj, list):
            decisions = obj
        decisions = [d for d in decisions if isinstance(d, dict)]
        # PARTIAL SALVAGE: if strict parse gave fewer than the batch, walk
        # the raw text and recover every well-formed decision object.
        if len(decisions) < len(batch):
            salv = _salvage_objects(raw, require_key="i")
            have = {d.get("i") for d in decisions}
            for d in salv:
                if d.get("i") not in have:
                    decisions.append(d)
                    have.add(d.get("i"))
        return decisions, raw

    def _apply_one(sc: Scene, d: dict) -> bool:
        """VALIDATION + apply one decision in-place. Invalid template kinds
        are dropped (scene keeps its other decisions, never crashes).
        Returns True if a usable decision was applied."""
        used = False
        kw = [str(k) for k in (d.get("kw") or []) if k]
        if kw:
            sc.keywords = kw
            used = True
        vis = str(d.get("vis", "") or "").strip()
        if vis:
            sc.visual = vis
            used = True
        try:
            iv = int(d.get("int", 0) or 0)
            sc.intensity = max(1, min(5, iv)) if iv else sc.intensity
        except (TypeError, ValueError):
            pass
        emph = str(d.get("emph", "") or "").strip()
        if emph:
            sc.emphasis = emph
        g = d.get("g") if isinstance(d.get("g"), dict) else {}
        st, gk, gt, gb = _parse_extra({
            "shot_type": d.get("st", ""),
            "graphic": {"kind": g.get("k", ""), "text": g.get("t", ""),
                        "body": g.get("b", "")},
        })
        if st:
            sc.shot_type = st
            used = True
        # VALIDATION: only accept a graphic kind the registry knows.
        if gk and gk in valid_kinds:
            sc.graphic_kind = gk
            sc.graphic_text = gt
            sc.graphic_body = gb
            used = True
        elif gk:
            print(f"  [editor] scene {sc.index}: dropped unknown graphic "
                  f"kind '{gk}'", flush=True)
        return used

    applied: set = set()

    def _run(batch: list[Scene], depth: int = 0) -> None:
        if not batch:
            return
        decisions, raw = _decide(batch, strict=depth > 0)
        by_i = {}
        for d in decisions:
            try:
                by_i.setdefault(int(d.get("i", -1)), d)
            except (TypeError, ValueError):
                continue
        idxset = {s.index for s in batch}
        for sc in batch:
            d = by_i.get(sc.index)
            if d and _apply_one(sc, d):
                applied.add(sc.index)
        missing = [s for s in batch if s.index not in applied
                   and s.index in idxset]
        lo, hi = batch[0].index, batch[-1].index
        print(f"  [editor] batch {lo}-{hi}: "
              f"{len(batch) - len(missing)}/{len(batch)} ok"
              + (f" · retrying {len(missing)} (depth {depth})"
                 if missing and depth < _MAX_RETRY else ""), flush=True)
        if not missing:
            return
        if depth < _MAX_RETRY:
            # RETRY only the failed scenes — split to shrink the batch.
            if len(missing) > 1:
                mid = len(missing) // 2
                _run(missing[:mid], depth + 1)
                _run(missing[mid:], depth + 1)
            else:
                _run(missing, depth + 1)
        else:
            dbg = _editor_debug_dump(f"batch_{lo}_{hi}", raw)
            print(f"  [editor] {len(missing)} scene(s) unresolved after "
                  f"retries -> heuristic fallback"
                  + (f"; raw saved {dbg}" if dbg else ""), flush=True)

    # ADAPTIVE BATCHING — smaller batches are far more reliable than one
    # huge call (a single broken char no longer loses the whole film).
    CHUNK = _EDITOR_BATCH
    nbatches = (n + CHUNK - 1) // CHUNK
    for c0 in range(0, n, CHUNK):
        _run(scenes[c0:c0 + CHUNK], depth=0)

    # FALLBACK SYSTEM — any scene the editor never resolved gets a
    # lightweight keyword-driven decision so the film NEVER collapses to
    # footage-only (and always has sane shot_type / keywords / graphics).
    fb = 0
    for sc in scenes:
        if sc.index not in applied:
            if _auto_fallback(sc):
                fb += 1

    # CHARACTER-INTRO PROMOTION -- the LLM editor frequently skips
    # name_reveal even when the script clearly introduces a character
    # ("When Eli, an Amish grandfather, ..."). This pass walks the whole
    # script in order, detects intro patterns, and promotes the FIRST
    # appearance of each character to a name_reveal card. Subsequent
    # mentions of the same name are not re-revealed.
    promoted = _promote_character_intros(scenes)

    print(f"  [editor] {len(applied)}/{n} scenes via LLM"
          + (f" + {fb} via heuristic fallback" if fb else "")
          + (f" + {promoted} character-intro promotions" if promoted else "")
          + f" (batch size {CHUNK})", flush=True)

    # The showrunner arc is the AUTHORITY on pacing: overwrite the blind
    # per-scene intensity with the deliberate global curve (every
    # downstream pacing system reads .intensity, so this is what makes
    # the EDIT escalate toward the reveal and breathe after it). Scenes
    # the arc didn't cover keep their per-scene estimate.
    if arc:
        for sc in scenes:
            if sc.index in arc:
                sc.role = arc[sc.index][0]        # Issue #5: drives hold
                sc.intensity = arc[sc.index][1]

    # ROBUSTNESS — the showrunner's role LABELS are flaky (the model
    # phrases them freely; even with _norm_role a whole run can come
    # back unmapped, which SILENTLY disabled Issue #5/#11/#12 — caught
    # repeatedly in testing). The intensity NUMBERS are always reliable,
    # so DERIVE any missing role deterministically from the arc's shape
    # & position. Roles can now never be empty -> the editor-intelligence
    # pacing always fires.
    n_sc = len(scenes)
    if n_sc:
        ii = [max(1, min(5, s.intensity or 2)) for s in scenes]
        peak = max(range(n_sc), key=lambda k: ii[k])
        for i, sc in enumerate(scenes):
            if sc.role:
                continue
            if i == 0:
                sc.role = "hook"
            elif i == peak:
                sc.role = "climax"
            elif i == peak - 1 and peak >= 2:
                sc.role = "reveal"
            elif i >= n_sc - 1:
                sc.role = "resolution"
            elif i >= n_sc - 2:
                sc.role = "payoff"
            elif i < peak and ii[i] >= ii[i - 1]:
                sc.role = "escalation"          # rising toward the peak
            elif i < peak:
                sc.role = "context"             # early/flat setup
            else:
                sc.role = "proof"               # post-peak: evidence/info

    # Issue #7 / Human-Editor #1 — SHOT VARIETY ENGINE: break framing
    # repetition the viewer actually feels (identical shot back-to-back OR
    # a third same-scale shot in a row), substituting an emotion-fit
    # framing. Far smarter than the old "no two identical labels" guard,
    # which let wide->aerial->establishing (three wides) slip through.
    _vary_shot_types(scenes)
    # (no return — helper mutates `scenes` in place; both callers will
    # wrap them in a Script and apply the graphic-caps separately).


def analyze_script(title: str, text: str, cfg: Config, mode=None) -> Script:
    """Editor brain on a USER-PROVIDED script (paste-narration path).
    Verbatim local split + editor brain + graphic caps."""
    base = script_from_text(title, text)        # verbatim local split
    scenes = base.scenes
    if cfg.has_llm:
        _apply_editor_decisions(title, scenes, cfg)
    try:
        _geo_enrich(scenes, mode)
    except Exception as e:                                 # noqa: BLE001
        print(f"  [geo] enrichment skipped ({e})", flush=True)
    # CHARACTER-INTRO PROMOTION — the brief-only path (_llm_script) runs
    # this; the user-script path (analyze_script) was missing it,
    # leaving every "His name is X" scene WITHOUT a name_reveal card
    # even though the script clearly introduced the character.
    # Bug fix 2026-05-26: same promotion logic here so portrait cards
    # render reliably whether the user supplies their own script or
    # the LLM writes one from the brief.
    promoted = _promote_character_intros(scenes)
    if promoted:
        print(f"  [editor] {promoted} character-intro promotion(s)",
              flush=True)
    _apply_graphic_caps(scenes, mode)
    if not scenes:
        raise RuntimeError("Editor produced no usable scenes.")
    return Script(title=title, scenes=scenes)


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
# Curated MOTION-GRAPHICS EXPLAINER menu — the cinematic storytelling kinds
# the weak-footage substitution may route to. The LLM picks the BEST per
# scene and is told to VARY them (MagnatesMedia / Johnny Harris range).
#
# v15: removed "comparison", "stat_dashboard", "number" (blue full-screen
# infographic cards — too template-tool). Replaced with editorial
# alternatives: document (paper+highlighter), cause_effect, quote_highlight.
# NOTE: the numbered four-box "process_diagram" step card AND the two-box
# "cause_effect" (CAUSE->EFFECT) card were both RETIRED — they looked template-y
# and repeated across videos. Neither is offered to the director, and both
# renderers are gated off in footage.py. "how it works / mechanism / cause->
# effect" beats now route to `document` (the clean paper+highlighter card).
_EXPLAINER_MENU = (
    "timeline", "network_graph",
    "document", "map_route", "quote_highlight",
)


def generate_explainers(title: str, weak: list, cfg: Config) -> dict:
    """AI VISUAL-EXPLANATION DIRECTOR (Phase B #2).

    `weak` = list of (scene_index, narration) for scenes whose footage is
    too weak to carry the idea. For EACH, the model chooses the single best
    explainer KIND and writes SHORT, STRUCTURED, visual-first content (not a
    narration sentence), choosing DIFFERENT kinds across scenes so the doc
    doesn't collapse into repeated kinetic-text cards. Returns
    {index: (kind, text, body)}; scenes it can't place are simply omitted
    (the caller falls back to the deterministic synthesiser).
    """
    if not weak or not getattr(cfg, "has_llm", False):
        return {}
    try:
        import anthropic
    except Exception:                                       # noqa: BLE001
        return {}
    by = {t.name: t for t in _tpl.all_templates()}
    rules = [f"- {by[k].llm_rule}" for k in _EXPLAINER_MENU if k in by]
    menu = "\n".join(rules)
    lines = "\n".join(f"{i}| {nar}" for i, nar in weak)
    system = (
        "You are the VISUAL-EXPLANATION DIRECTOR for a premium documentary "
        "(MagnatesMedia / Johnny Harris / Netflix explainer). The scenes "
        "below have NO usable footage, so each must be told with a "
        "MOTION-GRAPHIC EXPLAINER. For EACH scene, choose the SINGLE BEST "
        "way to VISUALISE the idea and write SHORT, STRUCTURED, visual-first "
        "content — labels, figures, keywords — NEVER a narration sentence. "
        "Match the visual to the idea: sequence/dates->timeline; how it "
        "works / mechanism / steps->document; X caused Y->document; "
        "influence / spread / connections->network_graph; A vs B->comparison; data / "
        "figures->stat_dashboard or number; movement / route->map_route. Use "
        "DIFFERENT kinds across scenes — variety is MANDATORY, never repeat "
        "the same kind twice in a row. Use 'quote_highlight' ONLY for a "
        "genuinely emotional statement, and keep it <=8 words. KEEP EVERY "
        "ON-SCREEN LABEL SHORT so it never overflows: network node names "
        "<=2 words, cause/effect and comparison box phrases <=4 words, "
        "step labels <=2 words, headlines <=5 words. Short, punchy, visual."
    )
    user = (
        f"TITLE: {title}\n\nEXPLAINER MENU (kind -> when to use + EXACT "
        f"t/b content format):\n{menu}\n\nWEAK-FOOTAGE SCENES (pick the best "
        f"explainer for EACH; the returned 'i' MUST match the number "
        f"shown):\n{lines}\n\n"
        'Return ONLY JSON {"items":[{"i":<scene number>,"kind":"<one menu '
        'kind>","t":"<short line / number / name>","b":"<structured content '
        'in that kind\'s format, SHORT>"}, ...]} . Write t/b in the SAME '
        "language as the narration."
    )
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    msg = client.messages.create(
        model=cfg.anthropic_model, max_tokens=2200, system=system,
        messages=[{"role": "user", "content": user}])
    raw = "".join(b.text for b in msg.content
                  if getattr(b, "type", "") == "text")
    data = _parse_json(raw)
    allow = set(_EXPLAINER_MENU)
    out: dict = {}
    prev_kind = None
    for it in data.get("items", []):
        try:
            i = int(it.get("i"))
        except (TypeError, ValueError):
            continue
        k = str(it.get("kind", "")).strip().lower()
        t = str(it.get("t", "") or "").strip()
        b = str(it.get("b", "") or "").strip()
        if k not in allow or (not t and not b):
            continue
        out[i] = (k, t, b)
        prev_kind = k
    return out


def build_script(brief: Brief, cfg: Config, script_file: str | None = None) -> Script:
    # Resolve the Style Mode once here so its TEMPLATE PERSONALITY
    # (fav/avoid + graphic density) shapes graphic selection during
    # script build — same deterministic resolution the pipeline uses for
    # pacing/music, so the on-screen graphics match the chosen show.
    try:
        from .style_modes import resolve_style
        mode = resolve_style(getattr(brief, "style", "auto"),
                             theme=brief.theme, title=brief.title,
                             prompt=brief.prompt)
    except Exception:                                       # noqa: BLE001
        mode = None
    if script_file:
        text = Path(script_file).read_text(encoding="utf-8")
        # User supplied the script -> if a key is set, the LLM acts as
        # EDITOR (verbatim words, smart cut). Otherwise heuristic split.
        if cfg.has_llm:
            try:
                return analyze_script(brief.title, text, cfg, mode)
            except Exception as e:  # noqa: BLE001
                print(f"  [editor] LLM analysis failed ({e}); "
                      "heuristic split", flush=True)
            return script_from_text(brief.title, text)
        return script_from_text(brief.title, text)
    # VOICEOVER-FIRST: the user uploaded their own narration but did NOT
    # paste a matching script. If we let the LLM invent a script from the
    # brief, the footage matches the INVENTED words while the voice says
    # something else (visuals look unrelated) AND word-alignment can't
    # anchor (the script ≠ the audio) so timing drifts. Fix: transcribe
    # the voiceover and make THAT the script — now footage matches what
    # is actually said and alignment is exact (script == audio).
    vo = getattr(brief, "voiceover", None)
    if vo and Path(vo).exists():
        try:
            from .align import transcript_text
            text = transcript_text(vo)
        except Exception as e:                              # noqa: BLE001
            print(f"  [voiceover] transcription failed ({e})", flush=True)
            text = ""
        if text and len(text.split()) >= 8:
            print(f"  [voiceover] transcribed {len(text.split())} words "
                  "→ using your spoken narration as the script", flush=True)
            if cfg.has_llm:
                try:
                    return analyze_script(brief.title, text, cfg, mode)
                except Exception as e:                      # noqa: BLE001
                    print(f"  [editor] analysis failed ({e}); "
                          "heuristic split", flush=True)
            return script_from_text(brief.title, text)
        print("  [voiceover] transcription empty/too short — "
              "falling back to brief-written script", flush=True)
    if cfg.has_llm:
        return _llm_script(brief, cfg, mode)
    raise RuntimeError(
        "No script source. Either set ANTHROPIC_API_KEY for auto-scripting "
        "or pass --script path/to/script.txt"
    )


# --------------------------------------------------------------------------- #
# Reviewed-script round-trip (script preview/edit step)
# --------------------------------------------------------------------------- #
def script_from_json(data: dict) -> Script:
    """Rebuild a Script from a persisted script.json (unedited path — keeps
    the original LLM scene split and keywords)."""
    title = data.get("title") or "video"
    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        nar = (s.get("narration", "") or "").strip()
        if not nar:
            continue
        try:
            inten = int(s.get("intensity", 0) or 0)
        except (TypeError, ValueError):
            inten = 0
        scenes.append(
            Scene(
                len(scenes),
                nar,
                [k for k in s.get("keywords", []) if k]
                or _keywords(nar, title),
                visual=str(s.get("visual", "") or "").strip(),
                intensity=inten,
                emphasis=str(s.get("emphasis", "") or "").strip(),
                shot_type=str(s.get("shot_type", "") or "").strip(),
                role=str(s.get("role", "") or "").strip(),
                graphic_kind=str(s.get("graphic_kind", "") or "").strip(),
                graphic_text=str(s.get("graphic_text", "") or "").strip(),
                graphic_body=str(s.get("graphic_body", "") or "").strip(),
            )
        )
    if not scenes:
        raise ValueError("script.json has no usable scenes.")
    return Script(title=title, scenes=scenes)


def script_from_edited_txt(text: str, fallback_title: str) -> Script:
    """Parse a (possibly hand-edited) script.txt back into a Script.

    The pipeline writes script.txt as: first block = title, then one
    blank-line-separated block per scene. The user is free to rewrite the
    narration, merge/split scenes (by adding/removing blank lines), or
    change the title — those edits are respected here, and keywords are
    re-derived from the edited narration.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        raise ValueError("script.txt is empty after review.")
    if len(blocks) == 1:
        title, scene_blocks = fallback_title, blocks
    else:
        title = " ".join(blocks[0].split()) or fallback_title
        scene_blocks = blocks[1:]
    scenes: list[Scene] = []
    for blk in scene_blocks:
        narration = _clean_for_tts(blk)
        if not narration:
            continue
        scenes.append(Scene(len(scenes), narration, _keywords(narration, title)))
    if not scenes:
        raise ValueError("No narration left in script.txt after cleaning.")
    return Script(title=title, scenes=scenes)
