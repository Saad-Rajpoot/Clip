"""Historical-footage period guard (reusable, engine-level).

Problem this solves permanently: a scene in a period documentary (Napoleon 1812,
Rome, medieval Europe, the Industrial Revolution …) must not be filled with an
obviously-modern stock clip — cars, glass skyscrapers, modern highways, a
contemporary city skyline, a drone city flyover. This module gives the footage
engine, for ANY scene and topic:

  • an ERA estimate (label + approximate year + confidence) from the scene's
    narration/keywords/title,
  • a PERIOD-RISK score for a candidate clip (does its title/snippet describe
    something that could not exist in that era?),
  • negative QUERY terms to keep the stock search period-appropriate,
  • and a safe FALLBACK ORDER (period-neutral landscape → archival/artwork →
    public-domain historical → AI historical → conservative generic atmosphere).

It deliberately works on TEXT (query/title/snippet/era), not pixels: frame-level
modern-object detection is a heavier future enhancement (see LIMITATIONS at the
bottom). Genuine landscapes/nature are never penalised. Pure logic → fully
unit-testable; the caller wraps calls defensively.
"""
from __future__ import annotations

import re

# A scene whose era is before this is "period-sensitive": modern markers that
# postdate the era are anachronistic and should be rejected/penalised. This is the
# MODERN-OBJECT gate (period_risk / period_sensitive) and is intentionally left at
# 1945 — a 1980s topic is not "anachronistic" if a clip merely shows a car.
MODERN_ERA_YEAR = 1945

# Upper bound for HISTORICAL-CONFLICT query biasing (RC4, 2026-06-05). A documentary
# whose topic resolves to a real past decade between 1945 and this year (e.g. the
# Iran–Iraq War → 1980, the Bosnian War → 1995) is NOT period-sensitive for modern
# objects, but its stock/image search should still be biased toward period archival
# footage rather than 2010s b-roll. This widens ONLY query_bias/reject_terms; it does
# NOT touch MODERN_ERA_YEAR, period_sensitive, or period_risk. A genuinely modern
# topic resolves to year 2010 (the `modern` bucket), which is > this bound and so is
# never biased.
HISTORICAL_BIAS_MAX_YEAR = 2006

# Modern visual markers → the earliest year each could plausibly appear. A marker
# is anachronistic for a scene whose era_year is clearly BEFORE the marker's year.
_MODERN_MARKERS = {
    "car": 1908, "cars": 1908, "automobile": 1908, "truck": 1920, "bus": 1925,
    "traffic": 1920, "highway": 1935, "freeway": 1945, "motorway": 1935,
    "skyscraper": 1900, "skyline": 1930, "glass building": 1955,
    "glass tower": 1960, "high-rise": 1950, "high rise": 1950, "downtown": 1930,
    "modern city": 1950, "city street": 1920, "neon": 1930, "billboard": 1925,
    "drone": 2012, "drone shot": 2012, "aerial city": 1990, "aerial view of city": 1990,
    "smartphone": 2008, "iphone": 2008, "laptop": 1995, "computer": 1985,
    "tv": 1950, "television": 1950, "airplane": 1915, "jet": 1950, "airport": 1950,
    "helicopter": 1945, "satellite": 1960, "led": 1995, "smartwatch": 2015,
    "subway train": 1905, "modern train": 1970, "bullet train": 1964,
    "solar panel": 1990, "wind turbine": 1990, "power lines": 1900,
    "parking lot": 1930, "shopping mall": 1960, "suburb": 1945,
    # ── 2026-06-03: SEARCH-SIDE additions. These are run over a stock clip's
    # public SLUG / title at fetch time (see footage._period_blocked) to drop a
    # wrong-era clip BEFORE download. Every term is UNAMBIGUOUSLY post-1945, so
    # it never false-rejects genuine period footage (a snowy field, marching
    # soldiers, an archival plate all score 0). Chosen from real Pexels slugs
    # that leaked onto the Napoleon-1812 sweep (a COVID-masked protest crowd, a
    # modern-Moscow street). Bare "city"/"street"/"mask" are deliberately NOT
    # here — they also describe period scenes (ancient city, gas mask) and would
    # over-filter; only the modern COMPOUNDS are listed.
    "covid": 2019, "covid 19": 2019, "coronavirus": 2019, "pandemic": 2019,
    "face mask": 2019, "face masks": 2019, "facemask": 2019,
    "surgical mask": 2019, "lockdown": 2020, "social distancing": 2020,
    "hand sanitizer": 2019, "ppe": 2019, "hazmat": 1980,
    "urban": 1920, "cityscape": 1930, "metropolis": 1910,
    "city streets": 1920, "city center": 1920, "city centre": 1920,
    "protest": 1960, "protester": 1960, "protesters": 1960,
    "protests": 1960, "protestors": 1960, "protesting": 1960,
    "placard": 1960, "placards": 1960,
    "social justice": 2000, "black lives matter": 2013, "blm": 2013,
    # ── 2026-06-03 (round 2): modern URBAN-TRANSIT vocabulary. A period sweep
    # still leaked a modern-Moscow tram/metro platform onto the burning-Moscow
    # beat — its slug carried none of the markers above. Trams/metros/escalators
    # postdate 1812; years are set so they're correctly NOT flagged for a WWII
    # (1942) or industrial (1850) scene where trains/trams already existed.
    "tram": 1830, "trams": 1830, "tramway": 1830, "streetcar": 1880,
    "trolley": 1900, "trolleybus": 1900, "metro": 1900, "subway": 1900,
    "metro station": 1900, "metro train": 1900, "subway station": 1900,
    "subway platform": 1900, "central station": 1900, "escalator": 1900,
    "turnstile": 1900, "train station": 1830, "railway station": 1830,
    "railway platform": 1830, "train platform": 1830,
    # Soviet-era markers are anachronistic for anything pre-1917 (a 1942 WWII
    # scene is NOT flagged — the USSR existed then).
    "soviet": 1917, "lenin": 1917,
    # electric festive lighting (a modern-Moscow "Christmas lights" b-roll leak);
    # the bare holiday word is NOT listed (it is not modern), only the lights.
    "christmas lights": 1900, "christmas decorations": 1900,
}

# ── RC5 (2026-06-05): SEARCH-SIDE modern-marker blacklist for a HISTORICAL-CONFLICT
# era (post-1945 but pre-2000, e.g. the Iran–Iraq War 1980, the Gulf War 1991, the
# Bosnian War 1995). For these topics MODERN_ERA_YEAR/period_risk stay silent (cars,
# cities, aircraft existed in the 1980s — they are NOT anachronistic), so the
# existing _MODERN_MARKERS gate cannot help. But the stock/web search for a 1980s
# war STILL surfaces 2010s-2020s b-roll, video-game/simulation captures, modern-
# military-UI loops, COVID crowds, and "4K cinematic" promo stock that read wrong on
# a period documentary. This set lists ONLY tokens that a real 1980s-2000 archival
# clip's slug/title could never legitimately carry — every entry post-dates ~2005 OR
# names a non-photographic medium (a game/render/simulation) — so it down-ranks the
# modern-looking filler without touching genuine period-conflict footage (a desert,
# a tank column, soldiers, a city under fire, an explosion all score 0 here). It is
# consumed ONLY on the QUERY/METADATA side (footage._period_blocked) and feeds the
# existing fail-closed ladder so a weak-period beat escalates to a period-grounded
# fal still rather than keeping a modern-looking asset for variety. NEVER a new pixel
# reject. Multi-word phrases match as substrings; single words on word boundaries.
_CONFLICT_MODERN_MARKERS = (
    # interactive / non-photographic media (a game capture or 3D render is never
    # archival war footage, regardless of decade)
    "video game", "videogame", "game ui", "game hud", "gameplay", "game play",
    "in-game", "game footage", "game engine", "first person shooter", "fps game",
    "war game", "wargame", "strategy game", "game simulation", "simulation game",
    "flight simulator", "combat simulator", "military simulation", "milsim",
    "3d render", "3d animation", "cgi animation", "rendered animation",
    "game screenshot", "screen recording",
    # explicitly-recent stock / promo vocabulary
    "4k stock", "4k cinematic", "stock footage 4k", "ultra hd", "8k", "60fps",
    "drone footage", "drone shot", "aerial drone", "fpv drone",
    "smartphone", "iphone", "android phone", "selfie", "social media",
    "tiktok", "instagram", "youtube short",
    # unambiguously post-2005 events / objects on a pre-2000 conflict topic
    "covid", "covid 19", "coronavirus", "pandemic", "lockdown", "face mask",
    "social distancing", "vaccine", "ukraine war", "russia ukraine", "russia-ukraine",
    "gaza 2023", "modern warfare", "modern combat", "modern military",
    "protest 2019", "protest 2020", "protest 2021", "black lives matter", "blm",
    "electric car", "tesla", "self driving", "5g", "drone strike fpv",
    # generic "this is present-day b-roll" giveaways
    "modern city", "present day", "nowadays", "21st century", "contemporary city",
    "city traffic", "rush hour traffic", "modern street", "modern skyline",
)


# Period-neutral terms that must NEVER be penalised (genuine landscapes / nature /
# timeless elements). If a candidate is dominated by these, period-risk stays low.
_NEUTRAL = ("landscape", "mountain", "mountains", "forest", "field", "fields",
            "meadow", "river", "lake", "ocean", "sea", "sky", "clouds", "storm",
            "snow", "winter", "fog", "mist", "sunrise", "sunset", "dawn", "dusk",
            "wilderness", "valley", "hills", "desert", "plains", "horizon",
            "candle", "fire", "flame", "smoke", "stars", "moon", "rain",
            "countryside", "wheat", "grass", "tree", "trees", "waves", "coast")

# Era buckets: label → (approx_year, trigger words). Year words/explicit years win.
_ERA_WORDS = {
    # Only UNAMBIGUOUS antiquity markers — bare place/nationality words like
    # "egypt"/"egyptian"/"rome"/"roman"/"greek" are removed because they also
    # describe MODERN nationalities (e.g. an "Egyptian-born" 1960s spy) and were
    # false-triggering 'ancient' on modern docs. BCE is handled by a regex below.
    "ancient": (-300, ("ancient", "roman empire", "ancient rome", "ancient greece",
                       "ancient egypt", "pharaoh", "mesopotamia", "babylon",
                       "gladiator", "legion", "legions", "antiquity",
                       "classical antiquity", "pyramids of", "centurion", "sparta",
                       "the colosseum")),
    "medieval": (1300, ("medieval", "middle ages", "knight", "knights", "crusade",
                        "crusader", "feudal", "castle", "monastery", "viking",
                        "plague", "black death", "dark ages", "monarchy",
                        "longbow", "siege", "norman", "byzantine")),
    "renaissance": (1550, ("renaissance", "tudor", "elizabethan", "reformation",
                           "1500s", "sixteenth century", "conquistador", "galleon")),
    "early_modern": (1750, ("colonial", "1700s", "eighteenth century", "baroque",
                            "enlightenment", "revolutionary war", "founding fathers",
                            "musket", "redcoat")),
    "napoleonic": (1812, ("napoleon", "napoleonic", "1812", "1815", "waterloo",
                          "borodino", "grande armee", "grande armée")),
    "industrial": (1850, ("industrial revolution", "1800s", "victorian",
                          "nineteenth century", "steam engine", "steam-powered",
                          "factory age", "robber baron", "gilded age", "telegraph",
                          "civil war", "1860s", "1870s", "1880s", "1890s")),
    "ww1": (1916, ("world war i", "world war one", "wwi", "great war", "1914",
                   "1916", "1918", "trench", "trenches", "western front")),
    "ww2": (1942, ("world war ii", "world war two", "wwii", "1939", "1940", "1941",
                   "1942", "1943", "1944", "1945", "blitz", "d-day", "holocaust",
                   "nazi", "wehrmacht", "pearl harbor")),
    "midcentury": (1960, ("cold war", "1950s", "1960s", "space race", "vietnam",
                          "cuban missile")),
    # ── 2026-06-05 (RC4): post-1945 CONFLICT buckets. Without these, a 1980s war
    # documentary (Iran–Iraq, the Gulf War …) whose narration says "modern"/"today"
    # fell through to the `modern` catch-all (year 2010) — so query_bias/reject_terms
    # pushed 2010s stock onto a 1980s topic. Every trigger is a SPECIFIC war phrase
    # (never bare "war"), so a genuine modern tech/startup video still lands "modern".
    # Years anchor the topic to its true decade so downstream search bias is period-apt.
    "late_cold_war": (1980, ("iran-iraq war", "iran iraq war", "iran–iraq",
                             "iran-iraq", "gulf war", "falklands", "falklands war",
                             "soviet-afghan", "soviet–afghan", "soviet afghan war",
                             "afghan war", "1970s", "1980s", "1980s war",
                             "desert storm", "contras", "iran-contra")),
    "modern_conflict": (1995, ("balkans war", "bosnia war", "bosnian war",
                               "rwandan genocide", "first gulf war", "1990s",
                               "kosovo war", "gulf war 1991")),
    "modern": (2010, ("today", "modern", "contemporary", "21st century",
                      "twenty-first century", "internet", "digital", "smartphone",
                      "social media", "startup", "2010s", "2020s")),
}


def _blob(*parts) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def detect_era(text: str = "", *, keywords=None, title: str = "") -> dict:
    """Estimate the era of a scene from its narration/keywords/title.

    Returns {label, year, confidence (0..1), reason, period_sensitive}. Explicit
    4-digit years (1500-2025) anchor the estimate; era words add confidence."""
    blob = _blob(text, " ".join(keywords or []), title)
    # explicit years carry the most signal
    years = [int(y) for y in re.findall(r"\b(1[0-9]\d{2}|20[0-2]\d)\b", blob)]
    old_years = [y for y in years if y < MODERN_ERA_YEAR]
    bce = bool(re.search(r"\b\d{1,4}\s*(?:bc|bce)\b", blob))
    hits = {}
    for label, (yr, words) in _ERA_WORDS.items():
        c = sum(1 for w in words if w in blob)
        if c:
            hits[label] = c
    label, year, conf, reason = "unknown", None, 0.0, "no period signal"
    if hits:
        label = max(hits, key=hits.get)
        year = _ERA_WORDS[label][0]
        conf = min(1.0, 0.45 + 0.18 * hits[label])
        reason = f"era words → {label} ({hits[label]} hit/s)"
    if bce:
        label, year, conf = "ancient", -300, max(conf, 0.8)
        reason = "BCE reference"
    if old_years:
        y = min(old_years)
        year = y if year is None else min(year, y)
        conf = max(conf, 0.7)
        reason = (reason + f"; explicit year {y}") if hits or bce else f"explicit year {y}"
        if label in ("unknown", "modern"):
            label = _year_to_label(y)
    elif years and not hits and not bce:
        label, year, conf, reason = "modern", max(years), 0.55, f"modern year {max(years)}"
    return {"label": label, "year": year, "confidence": round(conf, 2),
            "reason": reason,
            "period_sensitive": year is not None and year < MODERN_ERA_YEAR}


def _year_to_label(y: int) -> str:
    for label, (yr, _w) in sorted(_ERA_WORDS.items(), key=lambda kv: kv[1][0]):
        if y <= yr + 40:
            return label
    return "modern"


def period_risk(candidate_text: str, era: dict | None = None, *,
                era_year: int | None = None) -> dict:
    """How anachronistic does a candidate clip's title/snippet look for this era?

    Returns {risk (0..100), markers, neutral, reason}. Neutral landscape/nature
    terms keep risk low even if an incidental modern word appears."""
    if era_year is None and era:
        era_year = era.get("year")
    text = (candidate_text or "").lower()
    neutral_hits = [w for w in _NEUTRAL if re.search(r"\b" + re.escape(w) + r"\b", text)]
    if era_year is None or era_year >= MODERN_ERA_YEAR:
        return {"risk": 0, "markers": [], "neutral": neutral_hits,
                "reason": "scene not period-sensitive"}
    markers = []
    for marker, m_year in _MODERN_MARKERS.items():
        if era_year < m_year - 5 and re.search(r"\b" + re.escape(marker) + r"\b", text):
            markers.append((marker, m_year))
    # Explicit POST-1945 year or decade in the text ("1950s", "1972", "2020")
    # is an unambiguous anachronism for a pre-1945 scene. This catches the
    # mid-century "vintage" home-movie stock (a 1950s suburban street, a 1960s
    # film) that era-biased "vintage" queries ironically surface — its slug has
    # no modern NOUN, only the date. Genuine period slugs ("19th century",
    # "1812", "1920s") carry no 1945+ year and are never penalised; Pexels
    # clip-IDs are 7-8 digits and never match a \b-bounded 4-digit year.
    if era_year < MODERN_ERA_YEAR:
        for ym in set(re.findall(r"\b(19[5-9]\d|194[5-9]|20[0-4]\d)s?\b", text)):
            markers.append((ym + "s", int(ym)))
    if not markers:
        return {"risk": 0, "markers": [], "neutral": neutral_hits,
                "reason": "no anachronistic markers"}
    raw = sum(18 + max(0, (m_year - era_year)) // 40 for _m, m_year in markers)
    # genuine landscapes dampen the penalty (a "snowy mountain road" isn't modern)
    if neutral_hits and len(neutral_hits) >= len(markers):
        raw = int(raw * 0.5)
    risk = min(100, raw)
    return {"risk": risk, "markers": [m for m, _y in markers], "neutral": neutral_hits,
            "reason": f"anachronistic for ~{era_year}: " + ", ".join(m for m, _y in markers)}


def reject_terms(era: dict | None = None, *, era_year: int | None = None) -> list[str]:
    """Negative query terms to keep a stock search period-appropriate. Empty for
    modern/unknown scenes (so we never over-filter contemporary topics)."""
    if era_year is None and era:
        era_year = era.get("year")
    if era_year is None or era_year >= HISTORICAL_BIAS_MAX_YEAR:
        return []
    if era_year >= MODERN_ERA_YEAR:
        # 1946–2005 historical-conflict topic: bias away from present-day b-roll, but
        # keep cars/cities/aircraft (they existed) — only reject explicitly NOW terms.
        # RC5 additive: also push away the obvious non-photographic / recent-stock
        # vocabulary (a video-game/simulation capture, 4K promo footage, COVID
        # crowds) that the modern_search_markers blacklist screens on the slug side.
        return ["present day", "contemporary", "smartphone", "drone", "social media",
                "video game", "simulation", "4k stock", "modern military", "covid"]
    base = ["modern", "contemporary", "city", "skyline", "skyscraper", "car",
            "traffic", "highway", "smartphone", "drone", "aerial city"]
    if era_year < 1900:
        base += ["airplane", "neon", "billboard"]
    return base


def query_bias(era: dict | None = None, *, era_year: int | None = None) -> list[str]:
    """Positive period terms to append to a stock/image search."""
    if era_year is None and era:
        era_year = era.get("year")
    if era_year is None or era_year >= HISTORICAL_BIAS_MAX_YEAR:
        return []
    if era_year >= MODERN_ERA_YEAR:
        # 1946–2005 historical-conflict topic: archival/period footage of that decade.
        decade = (era_year // 10) * 10
        return ["archival", f"{decade}s", "documentary footage", "period news"]
    if era_year < 0:
        return ["ancient ruins", "classical antiquity", "historical painting"]
    if era_year < 1450:
        return ["medieval", "old painting", "historical engraving"]
    if era_year < 1750:
        return ["historical painting", "old master", "engraving"]
    if era_year < 1900:
        return ["vintage archival", "19th century", "period painting", "sepia"]
    return ["archival", "vintage", "black and white", "wartime archival"]


def historical_conflict(era: dict | None = None, *, era_year: int | None = None) -> bool:
    """True for a HISTORICAL-CONFLICT era: a real past decade in [1945, 2000).

    This is the band where `period_sensitive`/`period_risk` are deliberately SILENT
    (cars/cities/aircraft existed, so they're not anachronistic) yet the stock/web
    search must still be biased toward period archival footage and away from modern
    filler. A genuinely modern topic resolves to year 2010 (> the bound) and is
    never flagged; a pre-1945 topic is handled by the stronger period_risk gate.
    Pure, defensive — never raises (RC5, 2026-06-05)."""
    try:
        if era_year is None and isinstance(era, dict):
            era_year = era.get("year")
        return (isinstance(era_year, (int, float))
                and MODERN_ERA_YEAR <= era_year < 2000)
    except Exception:                                          # noqa: BLE001
        return False


def modern_search_markers(era: dict | None = None, *,
                          era_year: int | None = None) -> tuple:
    """Unambiguously-modern slug/title tokens to DOWN-RANK/REJECT on a historical-
    conflict (1945–1999) search. Empty for every other era so the caller never
    over-filters a pre-1945 topic (handled by period_risk) or a modern topic.

    Each token post-dates ~2005 OR names a non-photographic medium (game/render/
    simulation), so a genuine 1980s–1990s archival clip's slug can never carry one
    — period-neutral war footage (desert, tanks, soldiers, an explosion, a city
    under fire) is never penalised. METADATA/QUERY-side only; the caller feeds a
    match into the existing fail-closed ladder (escalate to a period fal still),
    NOT a new pixel reject. Defensive — never raises (RC5, 2026-06-05)."""
    try:
        if historical_conflict(era, era_year=era_year):
            return _CONFLICT_MODERN_MARKERS
        return ()
    except Exception:                                          # noqa: BLE001
        return ()


def conflict_modern_marker(text: str, era: dict | None = None, *,
                           era_year: int | None = None) -> str:
    """If `text` (a stock clip's slug/title/tags/url) carries an unambiguously-
    modern marker for a HISTORICAL-CONFLICT era, return that marker (truthy);
    else "". The metadata-side counterpart to period_risk for the 1945–1999 band.

    Period-NEUTRAL footage is never matched — only the explicit modern/game/
    recent-stock tokens in `_CONFLICT_MODERN_MARKERS` fire. Multi-word phrases
    match as substrings; single words on word boundaries (so 'fps game' fires but
    'game' alone inside 'gameel' does not, and '8k' fires while a clip-id does
    not). Defensive — never raises (RC5, 2026-06-05)."""
    try:
        markers = modern_search_markers(era, era_year=era_year)
        if not markers:
            return ""
        norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
        padded = f" {norm} "
        for tok in markers:
            if " " in tok:
                if tok in padded:
                    return tok
            elif f" {tok} " in padded:
                return tok
        return ""
    except Exception:                                          # noqa: BLE001
        return ""


def fallback_order(era: dict | None = None) -> list[str]:
    """Safe ordered fallbacks when no trustworthy period footage is found."""
    return ["period_neutral_landscape", "archival_artwork",
            "public_domain_historical", "ai_historical", "generic_atmosphere"]


def assess_scene(text: str = "", *, keywords=None, title: str = "") -> dict:
    """One-shot summary for logging / QA: era + whether to guard + the biases."""
    era = detect_era(text, keywords=keywords, title=title)
    return {"era": era,
            "period_sensitive": era["period_sensitive"],
            "reject_terms": reject_terms(era),
            "query_bias": query_bias(era),
            "fallback_order": fallback_order(era)}

# LIMITATIONS: this guard is TEXT-based (era ↔ query/title/snippet). It cannot see
# a modern car that the stock site failed to mention in the title; true frame-level
# modern-object detection (a vision classifier over sampled frames) is a heavier
# future enhancement. The fallback order + AI-historical path keep the result safe
# even when a modern clip slips through the text filter.
