"""Documentary-quality candidate filter.

Used by the orchestrator to RANK + REJECT music candidates BEFORE we burn
download bandwidth.  The goal is to keep the library skewed toward
cinematic / investigative / atmospheric / restrained-piano / orchestral
documentary music and away from corporate-stock, EDM, vlog, comedy, and
overdramatic-trailer noise.

The scorer reads any text blob (title + tags + genre + feel + description)
and returns a ``DocQuality`` with:

    * ``score``        -- floating preference, roughly -1.0 .. +1.0
    * ``verdict``      -- "prefer" / "neutral" / "reject"
    * ``reasons``      -- short human-readable explanation per hit

A candidate with verdict == "reject" is dropped pre-download.  The
score is also written into the sidecar JSON so the selection engine
(Stage 3) can tie-break in favour of higher doc-quality tracks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------- #
# REJECT --  if ANY of these tokens appear in the blob, score takes a hit.
# Heavy negatives (-0.5) for outright-disqualifiers (EDM, corporate motivational,
# wedding kitsch).  Soft negatives (-0.25) for "probably wrong" tags
# (lifestyle, happy upbeat) that a doc occasionally wants.
# ----------------------------------------------------------------------- #
_REJECT_HARD: tuple[str, ...] = (
    # EDM / dance club
    "edm", "dubstep", "drum and bass", "house music", "techno party",
    "club banger", "festival drop", "future house", "trance",
    # Corporate motivational stock
    "corporate motivational", "inspiring corporate", "uplifting business",
    "explainer", "presentation background", "advertising jingle",
    "tutorial music", "infomercial",
    # Trailer-spam
    "trailer impact", "trailer hits", "epic trailer drop", "hybrid trailer",
    "blockbuster trailer", "trailer sting", "trailer hit",
    # Comedy / kid / silly
    "comedy", "silly", "wacky", "circus", "polka", "kids ",
    "child", "cartoon", "humor", "humour", "quirky",
    # Wedding / ceremonial kitsch
    "wedding", "bridal", "ceremony",
    # Vocals-heavy / lyrical pop (always wrong as a doc bed)
    "lyrics", "vocal song", "pop vocal", "rap track", "hip hop song",
    # Reggae / ska / world-novelty
    "reggae", "ska", "calypso", "luau", "carnival",
)

_REJECT_SOFT: tuple[str, ...] = (
    "vlog", "lifestyle", "happy upbeat", "summer fun",
    "tropical", "feel good", "cheerful", "groovy pop",
    "country pop", "playful", "bouncy", "carefree",
    "morning happy", "fresh upbeat",
    # Generic "epic motivational" — overused, not real cinematic
    "motivational", "inspirational background", "uplifting positive",
)

# ----------------------------------------------------------------------- #
# PREFER -- the things we WANT.  Each hit adds +0.25 (capped overall).
# ----------------------------------------------------------------------- #
_PREFER_STRONG: tuple[str, ...] = (
    "cinematic", "documentary", "doc score", "film score",
    "investigative", "detective", "noir",
    "dark ambient", "dark atmospheric", "atmospheric",
    "tense ambient", "tension build", "slow build",
    "historical", "orchestral score", "strings ensemble",
    "emotional piano", "restrained piano", "solo piano",
    "underscore", "drone", "pad ", "score bed",
    "suspense", "thriller score", "espionage",
)

_PREFER_SOFT: tuple[str, ...] = (
    "instrumental", "ambient", "minimalist", "sparse",
    "reflective", "contemplative", "melancholy",
    "epic", "orchestral", "strings", "piano",
    "tension", "mysterious", "ominous", "subtle",
    "intimate", "introspective",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class DocQuality:
    score: float
    verdict: str                          # "prefer" | "neutral" | "reject"
    reasons: list[str] = field(default_factory=list)


def doc_quality(blob: str) -> DocQuality:
    """Score a text blob (title + tags + description, in any combination)."""
    t = " " + (blob or "").lower() + " "
    score = 0.0
    reasons: list[str] = []
    for kw in _REJECT_HARD:
        if kw in t:
            score -= 0.55
            reasons.append(f"-hard:{kw}")
    for kw in _REJECT_SOFT:
        if kw in t:
            score -= 0.25
            reasons.append(f"-soft:{kw}")
    for kw in _PREFER_STRONG:
        if kw in t:
            score += 0.30
            reasons.append(f"+strong:{kw}")
    for kw in _PREFER_SOFT:
        if kw in t:
            score += 0.12
            reasons.append(f"+soft:{kw}")

    score = max(-1.5, min(1.5, score))
    if score <= -0.40:
        verdict = "reject"
    elif score >= 0.20:
        verdict = "prefer"
    else:
        verdict = "neutral"
    return DocQuality(score=round(score, 3), verdict=verdict, reasons=reasons)


def is_reject(blob: str) -> bool:
    return doc_quality(blob).verdict == "reject"
