"""MASTER DOCUMENTARY STYLE MODES.

A Style Mode is a documentary's *cinematic personality* — not just its
colours, but its editorial behaviour: pacing psychology, transition bias,
music intensity, atmosphere presence, camera restraint, and which template
families it leans on. The SAME topic rendered in two modes should FEEL like
two different shows.

A mode is a set of scalar/knob biases applied on top of the existing
engines (it parameterises them, it doesn't replace them). `resolve_style`
picks a mode: an explicit choice always wins; otherwise it's auto-inferred
from the theme/topic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StyleMode:
    name: str
    label: str
    # ---- look / sound ------------------------------------------------ #
    music_theme: str = ""          # override the music theme ('' = keep brief)
    # ---- pacing psychology ------------------------------------------- #
    beat_target: float = 3.4       # avg beat length (s); higher = slower cuts
    beat_min: float = 2.4
    # ---- transition bias --------------------------------------------- #
    dissolve_gap: int = 3          # min scenes between dissolves (lower = more)
    dissolve_drop: int = 2         # energy-drop size that triggers a dissolve
    #                                (lower = dissolves fire on gentler exhales)
    # ---- camera / restraint ------------------------------------------ #
    hold_bias: float = 1.0         # >1 = more near-still holds (lingering)
    drift_scale: float = 1.0       # <1 = slower, grander camera drift
    # ---- audio personality ------------------------------------------- #
    music_intensity: float = 1.0   # >1 = bigger arc swells
    atmos_scale: float = 1.0       # >1 = more present environmental ambience
    # ---- template personality ---------------------------------------- #
    graphic_density: float = 1.0   # >1 = allow a few more cards
    fav_templates: tuple = ()      # kinds this mode leans on (kept first)
    avoid_templates: tuple = ()    # kinds that clash with the personality
    # ---- colour identity --------------------------------------------- #
    accent: tuple = ()             # () = keep theme accent; else (r,g,b)
    #                                override applied to the theme so EVERY
    #                                graphic card shares one signature colour
    # ---- typewriter stamps ------------------------------------------- #
    type_stamps: bool = False      # True -> place/era 'location' anchors
    #                                are rendered as the TYPING-DATE
    #                                typewriter (char reveal + typing SFX),
    #                                the faceless-doc archive-stamp feel


# Neutral baseline — current engine behaviour, nothing biased.
STANDARD = StyleMode(name="standard", label="Standard Documentary")

# NETFLIX HISTORICAL EPIC — grand, warm, unhurried. Long lingering holds,
# sweeping music swells, reflective dissolves used generously, slow majestic
# camera, present-but-warm atmosphere; leans on timeline / era / map /
# chapter / archival language; avoids the cold modern surveillance/news look.
EPIC = StyleMode(
    name="epic",
    label="Netflix Historical Epic",
    music_theme="history",
    beat_target=4.3,               # slower, let images breathe
    beat_min=3.0,
    dissolve_gap=2,                # dissolves more often (reflective)
    dissolve_drop=1,               # and on gentler emotional exhales
    hold_bias=1.5,                 # lots of lingering holds
    drift_scale=0.78,              # slow, grand camera moves
    music_intensity=1.25,          # bigger sweeping swells
    atmos_scale=1.15,
    graphic_density=1.0,
    fav_templates=("timeline", "map_reveal", "map_route",
                   "chapter_marker", "quote_highlight", "photo_collage",
                   "document", "classified", "name_reveal"),
    avoid_templates=("surveillance", "breaking_news", "sms_text",
                     "social_post", "status_indicator", "radio_dial",
                     "headline_crawl", "email_screenshot"),
)

# TRUE CRIME / INVESTIGATION — cold, tense, methodical. Hard cuts dominate
# (dissolves are rare and reserved for a genuine emotional exhale), shots are
# held on evidence and faces to build dread, the camera is controlled rather
# than grand, music sits low and tense with sharp swells on a reveal, and an
# unsettling room-tone atmosphere is always present. Leans HARD on the
# investigation language — evidence, case files, classified/redacted
# documents, surveillance, call logs, timelines, maps, newspaper clippings,
# the verdict stamp; avoids the warm, grand, celebratory or cutesy panels
# that would break the cold forensic mood.
TRUE_CRIME = StyleMode(
    name="true_crime",
    label="True Crime / Investigation",
    music_theme="crime",
    beat_target=3.1,               # tense, deliberate — a touch quicker
    beat_min=2.1,                  # but allows sharp montage cuts
    dissolve_gap=4,                # hard cuts dominate; dissolves are rare
    dissolve_drop=3,               # and only on a real emotional exhale
    hold_bias=1.2,                 # hold on evidence / faces to build dread
    drift_scale=0.9,               # controlled, watchful camera (not grand)
    music_intensity=1.1,           # low tension with sharp reveal stings
    atmos_scale=1.2,               # ever-present unsettling room tone
    graphic_density=1.15,          # the case board IS the storytelling
    fav_templates=("evidence", "evidence_tag", "case_file", "classified",
                   "redacted", "conspiracy_board", "surveillance",
                   "call_log", "document", "document_stack", "timeline",
                   "map_pin_cluster", "newspaper", "news_article",
                   "verdict_stamp", "voice_memo", "gps_stamp"),
    avoid_templates=("era_banner", "award_ribbon", "inscription",
                     "photo_collage", "polaroid_stack", "did_you_know",
                     "hashtag_trend", "speech_bubble", "film_slate"),
)

# HOMESTEAD GOLD — the faceless "forgotten knowledge / old-ways" doc look
# (Hidden-Homestead style): warm GOLD accent on EVERY card, unhurried but
# punchy pacing, leans on the explainer language the channel lives on —
# big statement text, labelled diagrams, framed inserts, % bars, callout
# arrows, line charts, section titles, archival documents + source credits.
# Avoids the cold modern surveillance / breaking-news / HUD chrome.
HOMESTEAD = StyleMode(
    name="homestead",
    label="Homestead Gold (faceless-doc)",
    music_theme="history",
    beat_target=3.8,
    beat_min=2.6,
    dissolve_gap=3,
    dissolve_drop=2,
    hold_bias=1.25,
    drift_scale=0.85,
    music_intensity=1.12,
    atmos_scale=1.1,
    graphic_density=1.1,
    accent=(226, 188, 96),         # signature warm gold on every card
    type_stamps=True,              # place/era anchors TYPE on (with clicks)
    fav_templates=("statement", "framed_insert", "diagram_labels",
                   "progress_bar", "callout", "line_chart", "section_title",
                   "typing_date", "bullet_list", "vertical_bar",
                   "stat_insight", "document", "footnote",
                   "name_reveal", "lower_third", "timeline", "map_reveal"),
    avoid_templates=("surveillance", "breaking_news", "sms_text",
                     "status_indicator", "radio_dial", "military_hud",
                     "headline_crawl", "email_screenshot", "call_log"),
)

_MODES = {m.name: m for m in (STANDARD, EPIC, TRUE_CRIME, HOMESTEAD)}


def get_mode(name: str) -> StyleMode:
    return _MODES.get((name or "").strip().lower(), STANDARD)


def all_modes() -> list:
    return list(_MODES.values())


# Auto-inference: theme + topic words -> the best-fitting personality.
# Conservative — only fires EPIC for clearly historical/grand material;
# everything else stays STANDARD until more modes are built.
_EPIC_WORDS = (
    "ancient", "empire", "civilization", "civilisation", "dynasty",
    "century", "centuries", "millennia", "ruins", "archaeolog", "pharaoh",
    "medieval", "kingdom", "historical", "history", "lost city", "tomb",
    "legend", "era", "antiquity", "bronze age", "ad ", "bc ",
)

# Investigation / true-crime topic markers. Kept deliberately specific so a
# historical "murder of a king" still reads as EPIC unless crime language
# clearly dominates (theme=='crime' is the strongest signal).
_CRIME_WORDS = (
    "murder", "murderer", "killer", "serial", "homicide", "detective",
    "suspect", "trial", "verdict", "convicted", "crime", "criminal",
    "investigation", "cold case", "unsolved", "kidnap", "heist", "fraud",
    "conspiracy", "witness", "forensic", "manhunt", "disappearance",
    "missing person", "fbi", "police", "interrogation", "alibi",
)


# Homestead / forgotten-knowledge / old-ways topic markers (the faceless
# "this cheap trick the experts buried" garden-science doc). Checked before
# EPIC so a 'forgotten 1923 garden secret' reads as HOMESTEAD GOLD even
# though it also carries history words.
_HOMESTEAD_WORDS = (
    "garden", "gardening", "homestead", "amish", "pest", "pests", "bug",
    "beetle", "aphid", "slug", "insect", "soil", "compost", "crop",
    "harvest", "seed", "permaculture", "off-grid", "off grid", "survival",
    "remedy", "remedies", "old-time", "old ways", "forgotten", "grandparent",
    "grandmother", "grandfather", "homemade", "self-sufficient", "frugal",
    "pioneer", "almanac", "folk", "backyard", "orchard", "livestock",
)


def resolve_style(explicit: str = "", *, theme: str = "",
                  title: str = "", prompt: str = "") -> StyleMode:
    """Pick the Style Mode. An explicit, known choice ALWAYS wins;
    'auto'/'' falls back to inference from theme + topic words."""
    ex = (explicit or "").strip().lower()
    if ex and ex not in ("auto", "default") and ex in _MODES:
        return _MODES[ex]
    th = (theme or "").strip().lower()
    blob = f"{title} {prompt}".lower()
    # Explicit crime THEME is decisive.
    if th == "crime":
        return TRUE_CRIME
    # Homestead / old-ways garden-science is checked BEFORE crime words:
    # this channel frames its garden topics conspiratorially ("they buried
    # it", "suppressed") which would otherwise mis-trigger TRUE_CRIME on a
    # lone 'conspiracy'. The homestead domain words (garden/pest/soil/crop…)
    # are specific enough that a real crime doc won't carry them.
    if any(w in blob for w in _HOMESTEAD_WORDS):
        return HOMESTEAD
    # Clear crime topic language wins over generic history words.
    if any(w in blob for w in _CRIME_WORDS):
        return TRUE_CRIME
    if th == "history" or any(w in blob for w in _EPIC_WORDS):
        return EPIC
    return STANDARD
