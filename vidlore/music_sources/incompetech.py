"""Incompetech (Kevin MacLeod) -- CC BY 4.0, bucket-balanced.

Reads the full catalogue from ``pieces.json`` then for EACH musiclib
category we score every track by feel/instrument/description fit + the
shared doc-quality scorer.  The yielded stream is **round-robin across
categories** so the orchestrator's per-category cap takes a balanced
slice -- you don't get 12 suspense tracks and 0 emotional_piano just
because the cinematic/dark cluster is biggest in the catalogue.

License is CC BY 4.0 across the entire catalogue.  Attribution is
written automatically into each sidecar JSON + LICENSES.md.
"""
from __future__ import annotations

from typing import Iterable
from urllib.parse import quote

from . import Candidate, register

_PIECES_JSON = "https://incompetech.com/music/royalty-free/pieces.json"
_MP3_BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 "
                   "Safari/537.36"),
    "Accept": "application/json, */*",
}

# Per-musiclib-category fit rules. Each entry lists FEEL tokens, INSTRUMENT
# tokens, and TITLE/DESC keyword hints -- a track scores against the union
# of all three. Categories overlap is fine: the highest-scoring bucket wins
# (round-robin keeps low-population buckets fed).
_CAT_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "historical_epic": {
        "feels": ("epic", "heroic", "dramatic", "majestic", "triumphant",
                  "grand"),
        "instruments": ("strings", "brass", "orchestra", "timpani", "choir",
                        "horn"),
        "keywords": ("epic", "heroic", "majestic", "cinematic",
                     "orchestral", "medieval", "renaissance", "ancient"),
    },
    "military_tension": {
        "feels": ("driving", "intense", "action", "tense", "determined",
                  "marching"),
        "instruments": ("snare", "timpani", "brass", "percussion", "drums"),
        "keywords": ("battle", "war", "march", "military", "drum",
                     "campaign", "soldier", "tactical"),
    },
    "emotional_piano": {
        "feels": ("sad", "emotional", "reflective", "melancholy", "tender",
                  "longing", "contemplative", "introspective"),
        "instruments": ("piano", "solo piano", "cello", "violin"),
        "keywords": ("piano", "tear", "lonely", "remember", "memory",
                     "farewell", "regret", "reflection"),
    },
    "aftermath": {
        "feels": ("uplifting", "peaceful", "calm", "reflective", "relaxed",
                  "warm", "hopeful"),
        "instruments": ("piano", "strings", "pad", "warm"),
        "keywords": ("dawn", "morning", "peace", "after", "rest", "calm",
                     "resolve", "release"),
    },
    "slow_reveal": {
        "feels": ("mysterious", "atmospheric", "reflective", "rising",
                  "expanding", "building"),
        "instruments": ("strings", "pad", "piano", "drone"),
        "keywords": ("rise", "reveal", "awakening", "unveil", "discovery",
                     "emerging", "ascent"),
    },
    "climax_build": {
        "feels": ("intense", "driving", "epic", "powerful", "dramatic",
                  "building", "rollicking", "action"),
        "instruments": ("strings", "brass", "drums", "timpani",
                        "percussion", "orchestra"),
        "keywords": ("battle", "climax", "rising", "ascent", "charge",
                     "intense", "powerful", "showdown"),
    },
    "ambient": {
        "feels": ("calming", "relaxed", "atmospheric", "mellow", "floating",
                  "ethereal", "spacious"),
        "instruments": ("pad", "drone", "synth", "soundscape", "strings"),
        "keywords": ("ambient", "drift", "float", "atmosphere", "space",
                     "underwater", "horizon"),
    },
    "dark_investigation": {
        "feels": ("dark", "creepy", "eerie", "mysterious", "noir",
                  "ominous", "brooding", "sinister"),
        "instruments": ("strings", "drone", "pad", "bass", "cello"),
        "keywords": ("dark", "noir", "shadow", "menace", "hidden",
                     "buried", "secret", "underworld", "ossuary"),
    },
    "suspense": {
        "feels": ("suspenseful", "tense", "building", "unsettling",
                  "creeping", "anxious", "tension"),
        "instruments": ("strings", "percussion", "drone", "pulse"),
        "keywords": ("chase", "stalk", "approach", "lurk", "edge",
                     "danger", "warning", "unanswered"),
    },
    "mystery": {
        "feels": ("mysterious", "enigmatic", "secretive", "atmospheric",
                  "exotic"),
        "instruments": ("strings", "pad", "celesta", "harp"),
        "keywords": ("mystery", "puzzle", "enigma", "whisper", "secret",
                     "shadow", "spy", "mirage"),
    },
    "tech_cyber": {
        "feels": ("driving", "electronic", "futuristic"),
        "instruments": ("synth", "electronic", "drum machine"),
        "keywords": ("cyber", "digital", "tech", "future", "machine",
                     "circuit", "neon", "synthwave"),
    },
    "archive_texture": {
        "feels": ("vintage", "noisy"),
        "instruments": ("tape", "vinyl"),
        "keywords": ("archive", "vintage", "static", "tape", "shortwave"),
    },
}


def _track_blob(t: dict) -> str:
    return " ".join(str(t.get(k) or "") for k in
                    ("feel", "genre", "title", "description", "instruments"))


def _doc_useful(t: dict) -> bool:
    from ..music_quality import doc_quality
    return doc_quality(_track_blob(t)).verdict != "reject"


def _quality_score(t: dict) -> float:
    from ..music_quality import doc_quality
    return doc_quality(_track_blob(t)).score


def _hms_to_sec(s: str) -> int:
    try:
        parts = [int(x) for x in (s or "").strip().split(":")]
    except Exception:                                       # noqa: BLE001
        return 0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


def _category_fit(t: dict, rules: dict) -> float:
    """How well a track matches one category (0..~2)."""
    feels = (t.get("feel") or "").lower()
    instruments = (t.get("instruments") or "").lower()
    text = (t.get("title", "") + " " + t.get("description", "")).lower()
    score = 0.0
    for f in rules["feels"]:
        if f in feels:
            score += 0.45
    for i in rules["instruments"]:
        if i in instruments:
            score += 0.22
    for k in rules["keywords"]:
        if k in text:
            score += 0.18
    return score


def _best_category(t: dict) -> tuple[str, float]:
    """Pick the highest-fitting musiclib category for a track."""
    best = ("auto", 0.0)
    for cat, rules in _CAT_RULES.items():
        s = _category_fit(t, rules)
        if s > best[1]:
            best = (cat, s)
    return best


def _make_candidate(t: dict, cat: str) -> Candidate:
    fn = t["filename"].strip()
    title = (t.get("title") or fn.rsplit(".", 1)[0]).strip()
    mp3 = _MP3_BASE + quote(fn)
    return Candidate(
        title=title,
        source="incompetech",
        url=("https://incompetech.com/music/royalty-free/index.html"),
        download_url=mp3,
        channel="Kevin MacLeod (incompetech)",
        license="Kevin MacLeod (incompetech.com) -- CC BY 4.0",
        attribution=("Music: Kevin MacLeod (incompetech.com), "
                     "Licensed under Creative Commons: By "
                     "Attribution 4.0 License -- "
                     "http://creativecommons.org/licenses/by/4.0/"),
        category_hint=cat,
        duration=_hms_to_sec(t.get("length") or ""),
        tags=[s.strip().lower() for s in
              (t.get("feel") or "").split(",") if s.strip()][:6],
    )


@register("incompetech")
def discover(limit: int = 30) -> Iterable[Candidate]:
    """Bucket each track into its best musiclib category, rank within bucket
    by (category_fit + doc_quality), then yield ROUND-ROBIN so every bucket
    contributes evenly to the orchestrator's per-cat-cap budget."""
    try:
        import requests
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! incompetech deps missing: {e}")
        return
    try:
        r = requests.get(_PIECES_JSON, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        pieces = r.json()
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! incompetech: pieces.json fetch failed: {e}")
        return
    if not isinstance(pieces, list):
        return

    # 1. filter -- doc-quality reject + duration window
    usable: list[dict] = []
    for t in pieces:
        if not isinstance(t, dict) or not t.get("filename"):
            continue
        if not _doc_useful(t):
            continue
        dur = _hms_to_sec(t.get("length") or "")
        if dur and not (45 <= dur <= 420):
            continue
        usable.append(t)

    # 2. bucket -- each track lands in its single best category
    buckets: dict[str, list[tuple[float, dict]]] = {
        c: [] for c in _CAT_RULES}
    for t in usable:
        cat, fit = _best_category(t)
        if cat == "auto" or fit < 0.30:
            continue              # weak fit -> skip (orchestrator can find via fallback)
        # score = category fit + doc-quality boost, deterministic uuid tiebreaker
        score = fit + max(0.0, _quality_score(t)) * 0.5
        buckets[cat].append((score, t))

    # 3. rank within each bucket
    for cat in buckets:
        buckets[cat].sort(key=lambda x: (-x[0], x[1].get("uuid") or ""))

    # 4. round-robin yield -- one from each bucket per pass until limit hit
    #    OR all buckets exhausted. Empty buckets are skipped silently.
    yielded = 0
    bucket_order = list(_CAT_RULES.keys())
    pass_no = 0
    while yielded < limit:
        wrote_any = False
        for cat in bucket_order:
            if yielded >= limit:
                return
            if pass_no >= len(buckets[cat]):
                continue
            _score, t = buckets[cat][pass_no]
            yield _make_candidate(t, cat)
            yielded += 1
            wrote_any = True
        if not wrote_any:
            return
        pass_no += 1
