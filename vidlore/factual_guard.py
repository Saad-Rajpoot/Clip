"""vidlore/factual_guard.py — V3.3.2 CENTRALIZED FACTUAL-VALIDATION LAYER.

ONE place that decides whether a *fact-bearing* motion-graphic card is backed by
explicit evidence in the scene's narration. Pure functions, **never raises**, no
I/O, and ZERO dependency on `asset_qa` (a separate, untouched asset-quality layer).
Imported by:
  * `pipeline.py`     — enforcement sweep + manifest rejection logging (authority)
  * `script_gen.py`   — prompt teaching reads the same VAGUE_QUANTIFIERS list
  * the test harness  — `tools/test_v332_legacy_factual_guard.py`

THE RULE (V3.3.2)
-----------------
A fact-bearing card may render ONLY when the REAL data is ALREADY stated in the
scene's narration:

  1. VAGUE GATE (pure number/date/currency/measurement/ranking cards): if the
     narration contains NO explicit quantity (no digit, no spelled magnitude/
     proportion) — only vague quantifiers like "most", "many", "a huge increase",
     "millions", "years later" — the card is DROPPED → footage-only fallback.

  2. GROUNDING GATE (every fact-bearing card): a card may NOT introduce a hard
     number / year / percentage / currency amount that is ABSENT from the
     narration (a planted/invented value). If it does → DROPPED → footage-only.

Vague quantifiers are NOT data. Explicit values ("$12 million in 1980",
"90 percent", "from 12 to 84") are preserved exactly — the guard never mutates a
value, it only KEEPS or DROPS the whole card.

This is the legacy-template counterpart to the prompt-level `_MG_VOCAB` rule in
`script_gen.py`: the prompt *discourages* the LLM from fabricating; this layer
*enforces* it in code for every emission path (LLM brief, paste-script, reload,
heuristic fallback), so a fabricated card can never reach the renderer.

Design notes
------------
* "Substantive" numbers only are grounded/flagged: 2+ digit integers, any number
  bound to %/currency/magnitude, and 4-digit years. A bare single digit (a step
  count, "2X" for "doubled", a "3-part" process) is intentionally NOT flagged, so
  legitimate structural cards are never false-dropped.
* Spelled magnitudes in the narration ("ninety percent", "three million", "half")
  count as evidence AND ground the digitised value (ninety→90), so a faithful
  card built from a spelled number is kept.
* Geometric / coordinate / manual-geo / pure-style kinds are EXEMPT — their
  numbers are screen positions, coordinates or technical readouts, not fabricable
  documentary statistics (and the map family is manual-only by design).
* The guard FAILS OPEN on any internal error (returns ok=True): a guard bug must
  never crash or silently gut a render. The *decision* still fails CLOSED (no
  evidence → drop).
"""
from __future__ import annotations

import re

VERSION = "v3.3.2"

# --------------------------------------------------------------------------- #
# FACT-BEARING KIND SETS  (curated from the full templates/__init__.py audit —
# 96 templates — plus the V3.3 unlocked vocabulary and Section-C structured kinds)
# --------------------------------------------------------------------------- #

# STRICT — pure number / statistic / currency / measurement / ranking / count
# cards. These get the VAGUE GATE *and* grounding. A scene whose only quantities
# are vague gets NO such card.
NUMERIC_DATA_KINDS: frozenset = frozenset({
    # legacy templates
    "number", "stat", "chart", "progress_bar", "line_chart", "stat_dashboard",
    "speedometer", "stat_insight", "receipt", "vertical_bar_chart",
    "heatmap_grid", "numerical_ratio", "document_stack", "currency_stat",
    "score_display", "tally_counter", "demographic_split", "stats_bar",
    "donut_chart", "ranking", "hashtag_trend", "map_region",
    # V3.3 unlocked MG vocabulary (numeric)
    "bar_chart", "composition", "sankey", "gauge",
})

# STRICT — date / year / timeline cards. Also get the VAGUE GATE + grounding.
DATE_DATA_KINDS: frozenset = frozenset({
    "timeline", "mini_timeline", "typing_date", "calendar_grid", "era_banner",
    # V3.3 unlocked
    "eras", "before_after",
})

# GROUNDING ONLY — a qualitative contrast of two NAMED subjects is allowed even
# with no numbers, but INVENTED comparison values are blocked.
COMPARISON_KINDS: frozenset = frozenset({
    "comparison", "versus", "vs", "balance", "cause_effect",
})

# GROUNDING ONLY — claim / relationship / definition / process cards. Free text
# is allowed; a planted hard number / date not in the narration is blocked.
CLAIM_KINDS: frozenset = frozenset({
    "document", "did_you_know", "evidence", "bullet_list", "glossary",
    "define_the_term", "footnote", "process_diagram", "network_graph",
    "relationship_tree", "family_tree_3gen", "conspiracy_board",
    # V3.3 unlocked
    "hierarchy", "process", "decision", "headlines",
})

# GROUNDING ONLY — attributed quote / transcript / press cards. The quote text is
# the LLM's (paraphrase allowed); a planted hard figure/date is blocked.
QUOTE_KINDS: frozenset = frozenset({
    "quote_highlight", "quote_stream", "pull_quote_portrait", "long_quote",
    "letter", "diary_entry", "audio_waveform", "voice_memo",
    "subtitle_translation", "inscription", "social_post", "news_article",
    "newspaper", "news_broadcast", "classified", "press_release",
    "email_screenshot", "sms_text", "speech_bubble", "headline_crawl",
    "statement",
})

# EXEMPT — numbers here are screen coords / map coordinates / technical readouts
# / manual-only geography / pure styling. Never guarded (would false-drop), and
# the map/route family is editor-manual-only by V3.3 design anyway.
GEOMETRIC_EXEMPT: frozenset = frozenset({
    "route_trace", "scale_compare", "size_compare",
    "map_route", "map_reveal", "map_pin_cluster", "globe_map",
    "map_heat_spread", "map_badge_node", "velocity_route_map", "world_map_arc",
    "gps_stamp", "compass_bearing", "military_hud", "crosshair_lock",
    "surveillance", "clock_face", "radio_dial", "figure_locator", "countdown",
})

# Cards that get the strict vague gate (must have explicit data in narration).
STRICT_EVIDENCE_KINDS: frozenset = NUMERIC_DATA_KINDS | DATE_DATA_KINDS
# Cards that get grounding only (qualitative content allowed, planted values not).
GROUNDING_ONLY_KINDS: frozenset = COMPARISON_KINDS | CLAIM_KINDS | QUOTE_KINDS
# Everything the guard inspects.
FACT_BEARING_KINDS: frozenset = STRICT_EVIDENCE_KINDS | GROUNDING_ONLY_KINDS

# --------------------------------------------------------------------------- #
# VAGUE QUANTIFIERS — words/phrases that carry NO usable number, date or split
# (STEP 3). Kept as data for the prompt teaching + tests. The numeric logic does
# not need this list (vague words contain no digit and no spelled magnitude bound
# to a unit, so they naturally fail `_narration_has_data`), but it documents the
# contract and is asserted by the test harness.
# --------------------------------------------------------------------------- #
VAGUE_QUANTIFIERS: frozenset = frozenset({
    "most", "many", "several", "huge", "massive", "large", "small", "rapidly",
    "dramatically", "significantly", "one of the biggest", "millions",
    "billions", "dozens", "hundreds", "countless", "years later", "long ago",
    "a large share", "a major increase", "a dramatic decline", "numerous",
    "some", "almost all", "nearly everyone", "a huge increase", "major",
    "a lot", "lots", "tons of", "plenty", "widespread", "vast", "enormous",
    "soared", "skyrocketed", "plummeted", "exploded", "surged",
})

# --------------------------------------------------------------------------- #
# NUMBER EXTRACTION / EVIDENCE DETECTION
# --------------------------------------------------------------------------- #
_DIGIT_RE = re.compile(r"\d")

# A spelled cardinal bound to a magnitude/unit, OR a real proportion word.
# "five million", "ninety percent", "twelve years", "half", "tripled" → data.
# Bare "millions"/"one of the biggest"/"many years" → NOT matched (vague).
_CARDINALS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    "fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion"
)
_MAGNITUDE = (
    "percent|hundred|thousand|million|billion|trillion|dollars?|cents?|euros?|"
    "pounds?|years?|months?|weeks?|days?|hours?|minutes?|seconds?|miles?|"
    "kilometres?|kilometers?|metres?|meters?|tons?|tonnes?|degrees?|"
    "people|times"
)
# Standalone spelled cardinals that are real quantities on their own
# ("ninety to ten", "three rivals"). Excludes "one" (usually "one of the…",
# an article/pronoun, not a documentary quantity) and the bare plural magnitude
# words ("millions"/"hundreds"), which STEP 3 treats as vague.
_STANDALONE_CARD = (
    "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    "forty|fifty|sixty|seventy|eighty|ninety"
)
_SPELLED_DATA_RE = re.compile(
    r"\b(?:(?:" + _CARDINALS + r")\s+(?:" + _MAGNITUDE + r")"     # "five million"
    r"|(?:" + _STANDALONE_CARD + r")"                             # bare "ninety", "ten"
    r"|half|a\s+third|two\s+thirds|a\s+quarter|three\s+quarters|quarter\s+of"
    r"|doubled?|tripled?|quadrupled?|tenfold|hundredfold)\b",
    re.IGNORECASE,
)

# Substantive numeric tokens that must be grounded if they appear ON a card:
#   percentages, currency amounts, 4-digit years, magnitude-bound numbers,
#   and any 2+ digit integer/decimal. Single bare digits are intentionally
#   ignored (structural: step counts, "2X", "3-part").
_PCT_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(?:%|percent)", re.IGNORECASE)
_CUR_RE = re.compile(r"[$€£¥]\s?(\d[\d,]*\.?\d*)")
_MAGNUM_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(?:%|percent|million|billion|thousand|hundred|trillion|"
    r"k\b|m\b|bn\b)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(1[0-9]\d\d|20\d\d)\b")
_BIGINT_RE = re.compile(r"\b\d[\d,]*\.?\d*\b")

# Spelled cardinal → digit string, so a card that digitises a spelled narration
# number ("ninety percent" → "90") is grounded.
_SPELL2DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "half": "50", "quarter": "25",
    "third": "33", "double": "2", "triple": "3",
}
_SPELL_WORD_RE = re.compile(
    r"\b(" + "|".join(_SPELL2DIGIT.keys()) + r")\b", re.IGNORECASE)


def _norm_num(tok: str) -> str:
    """Normalise a numeric token: strip commas, drop a trailing '.0'."""
    t = (tok or "").replace(",", "").strip()
    if t.endswith(".0"):
        t = t[:-2]
    if "." in t:
        t = t.rstrip("0").rstrip(".") or t
    return t


def _hard_numbers(s: str) -> set:
    """Substantive numeric tokens in `s` (percentages, currency, years,
    magnitude-bound and 2+ digit numbers). Bare single digits are ignored."""
    s = s or ""
    out: set = set()
    for rx in (_PCT_RE, _CUR_RE, _MAGNUM_RE):
        for m in rx.findall(s):
            out.add(_norm_num(m if isinstance(m, str) else m[0]))
    for y in _YEAR_RE.findall(s):
        out.add(_norm_num(y))
    for m in _BIGINT_RE.findall(s):
        n = _norm_num(m)
        if len(n.replace(".", "")) >= 2:        # 2+ significant digits only
            out.add(n)
    return {n for n in out if n}


def _spelled_digits(narr: str) -> set:
    """Digit strings implied by spelled cardinals present in the narration."""
    return {_SPELL2DIGIT[w.lower()] for w in _SPELL_WORD_RE.findall(narr or "")}


def _narration_has_data(narr: str) -> bool:
    """True if the narration states ANY explicit quantity (digit OR a spelled
    magnitude/proportion). Vague quantifiers alone → False."""
    t = narr or ""
    return bool(_DIGIT_RE.search(t)) or bool(_SPELLED_DATA_RE.search(t))


def _ungrounded_numbers(narr: str, text: str, body: str) -> list:
    """Substantive numbers that appear on the CARD but NOT in the narration —
    i.e. planted / invented values. Structural 0 and 100 (scale max / percent
    base) are always allowed."""
    src = _hard_numbers(narr) | _spelled_digits(narr) | {"0", "100"}
    card = _hard_numbers(f"{text or ''}\n{body or ''}")
    return sorted(n for n in card if n not in src)


# --------------------------------------------------------------------------- #
# PUBLIC API
# --------------------------------------------------------------------------- #
def is_fact_bearing(kind: str) -> bool:
    k = (kind or "").strip().lower()
    return bool(k) and k in FACT_BEARING_KINDS and k not in GEOMETRIC_EXEMPT


def classify(kind: str) -> str:
    """Coarse category for reporting/manifest ('numeric'|'date'|'comparison'|
    'claim'|'quote'|'exempt'|'not_fact_bearing')."""
    k = (kind or "").strip().lower()
    if not k or k in GEOMETRIC_EXEMPT:
        return "exempt"
    if k in NUMERIC_DATA_KINDS:
        return "numeric"
    if k in DATE_DATA_KINDS:
        return "date"
    if k in COMPARISON_KINDS:
        return "comparison"
    if k in CLAIM_KINDS:
        return "claim"
    if k in QUOTE_KINDS:
        return "quote"
    return "not_fact_bearing"


def guard(kind: str, narration: str, text: str = "", body: str = "") -> tuple:
    """Decide whether a fact-bearing card is evidence-backed.

    Returns (ok: bool, reason: str). ok=True → keep the card; ok=False → drop it
    to footage-only. NEVER raises (fails OPEN on unexpected internal error so a
    guard bug cannot break a render). Non-fact-bearing / exempt kinds always
    return (True, 'ok:not_guarded').
    """
    try:
        k = (kind or "").strip().lower()
        if not k or k in GEOMETRIC_EXEMPT or k not in FACT_BEARING_KINDS:
            return True, "ok:not_guarded"
        narr = narration or ""
        # 1) VAGUE GATE — pure number/date cards need an explicit quantity.
        if k in STRICT_EVIDENCE_KINDS and not _narration_has_data(narr):
            return False, "no_explicit_data:vague_only"
        # 2) GROUNDING GATE — no planted/invented hard values on any fact card.
        ung = _ungrounded_numbers(narr, text, body)
        if ung:
            return False, "ungrounded_values:" + ",".join(ung[:6])
        return True, "ok:evidence_present"
    except Exception:                                          # noqa: BLE001
        return True, "ok:guard_failopen"


# --------------------------------------------------------------------------- #
# RC4 — MACHINE-PAYLOAD GUARD for HUMAN text fields.
#
# A card's human-readable text fields (title/text/headline/quote/verdict/
# definition/caption) must never surface internal/serialized data. The trigger
# bug: a route card's packed waypoint DSL ("Tehran@35.69,51.39|Baghdad@33.34,
# 44.40|...", produced by editor_manifest._card_encode) was copied into the
# SHARED `title` asset key, and a sibling title-only primitive (act_chapter_card)
# rendered that raw geo payload as its on-screen title. `is_machine_payload`
# detects such leaked machine data so the pipeline can blank the field (or drop
# the card to footage-only) for ANY text card — not just the one that triggered
# it. It must NOT flag a legitimate human title that merely contains a digit, a
# year, a single stray '@', or one '|'.
# --------------------------------------------------------------------------- #

# A "NAME@<num>,<num>" waypoint token (the packed route place format). Anchored
# coordinate pair so a casual "email@domain" or a lone "@handle" never matches.
_GEO_WAYPOINT_RE = re.compile(r"[^@|]+@-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?")

# key=value DSL idioms that only ever appear in a packed graphic body, never at
# the start of a human title.
_DSL_KEY_RE = re.compile(
    r"(?:^|[|;·])\s*(?:stops|bars|events|points|branches|coords|route|region|"
    r"share|values|steps|children|pair|focus|tag|suffix|prefix)\s*=",
    re.IGNORECASE)


def is_machine_payload(s) -> bool:
    """True if `s` is clearly internal/machine data sitting in a HUMAN text
    field (a leaked DSL/serialization), so it must never render as on-screen
    copy. NEVER raises (fails CLOSED-safe → False so a guard quirk can't blank a
    legitimate title). Deliberately conservative: a normal title with one digit,
    a 4-digit year, a single stray '@', or a single '|' is NOT flagged."""
    try:
        if not isinstance(s, str):
            return False
        t = s.strip()
        if not t:
            return False
        # 1) Packed route waypoints: >=1 "NAME@lat,lon" token AND a '|' separator
        #    (the multi-stop packed list). A single coordinate token without a
        #    pipe is allowed through here and caught only if it's the whole field.
        wps = _GEO_WAYPOINT_RE.findall(t)
        if wps and ("|" in t or len(wps) >= 2):
            return True
        # A lone "NAME@lat,lon" that IS essentially the entire field is also a
        # leaked coordinate payload, not a human title.
        if len(wps) == 1 and len(wps[0]) >= len(t) - 2:
            return True
        # 2) key=value DSL idioms (stops=, bars=, events=, …) at a token start.
        if _DSL_KEY_RE.search(t):
            return True
        # 3) Raw JSON / serialized arrays / dict reprs.
        if (t.startswith('{"') or t.startswith("[{") or t.startswith("[[")
                or t.endswith('"}') or t.endswith("]}")
                or t.startswith("{'") or t.endswith("'}")):
            return True
        # 4) Unresolved placeholders / null-coordinate artifacts.
        if ("{{" in t or "<unresolved" in t.lower() or "None@" in t
                or "none@" in t):
            return True
        return False
    except Exception:                                          # noqa: BLE001
        return False


def sanitize_card_text(s):
    """Return clean human text unchanged, or None when `s` is a machine payload
    (so the caller blanks that field / drops the card to footage-only). NEVER
    raises. Empty/None in → None out."""
    try:
        if not isinstance(s, str):
            return None
        if not s.strip():
            return None
        if is_machine_payload(s):
            return None
        return s
    except Exception:                                          # noqa: BLE001
        return s if isinstance(s, str) and s.strip() else None


# Convenience aggregate for tests / drift checks.
ALL_GUARDED_KINDS: frozenset = FACT_BEARING_KINDS
