"""Pre-photographic portrait intelligence (reusable, engine-level).

Problem this solves permanently: a documentary about a person who lived BEFORE
practical photography (e.g. Napoleon, †1821) must not be given a random modern
photo or a generic AI "officer" — it should be given a verified period PAINTING,
engraving, or public-domain illustration. This module decides, for any named
person and across all niches/topics:

  • whether the person is PRE-PHOTOGRAPHIC (so artwork must be preferred), and
  • the ordered SEARCH QUERIES that bias toward the right visual TYPE
    (painting → engraving → portrait → photograph), and
  • whether a candidate SOURCE is acceptable (strict name match, sane provenance,
    no "modern photo for a pre-photographic person"), with a logged reason.

It is pure logic + an optional best-effort Wikipedia birth-year lookup, so the
classification + query rules are fully unit-testable offline. footage.py consults
it; if this module ever raises, the caller falls back to its old behaviour.

Photography timeline used: the daguerreotype is 1839; portrait photography only
became practical/common in the mid-1850s. A person whose life ended before that
cannot have a real photograph, so we prefer contemporary artwork.
"""
from __future__ import annotations

import re

# Practical portrait-photography cutoff. Someone who died before this almost
# certainly has NO authentic photograph → prefer a contemporary painting/engraving.
PHOTO_PRACTICAL_YEAR = 1855
# Born this late ⇒ overwhelmingly photographed even if also painted.
PHOTO_CERTAIN_BIRTH_YEAR = 1830

# Curated, offline birth/death table for notable figures (BCE = negative). Extend
# freely — unknown names fall back to a live Wikipedia parse, then to "photographic"
# (the safe default: never force artwork on someone who may have real photos).
_FIGURE_YEARS = {
    "napoleon": (1769, 1821), "napoleon bonaparte": (1769, 1821),
    "julius caesar": (-100, -44), "caesar": (-100, -44),
    "augustus": (-63, 14), "marcus aurelius": (121, 180), "nero": (37, 68),
    "cleopatra": (-69, -30), "genghis khan": (1162, 1227),
    "kublai khan": (1215, 1294), "attila": (406, 453), "attila the hun": (406, 453),
    "alexander the great": (-356, -323), "alexander iii of macedon": (-356, -323),
    "leonidas": (-540, -480), "hannibal": (-247, -183), "spartacus": (-103, -71),
    "charlemagne": (748, 814), "william the conqueror": (1028, 1087),
    "richard the lionheart": (1157, 1199), "saladin": (1137, 1193),
    "joan of arc": (1412, 1431), "leonardo da vinci": (1452, 1519),
    "michelangelo": (1475, 1564), "christopher columbus": (1451, 1506),
    "henry viii": (1491, 1547), "elizabeth i": (1533, 1603),
    "william shakespeare": (1564, 1616), "galileo": (1564, 1642),
    "galileo galilei": (1564, 1642), "isaac newton": (1643, 1727),
    "louis xiv": (1638, 1715), "george washington": (1732, 1799),
    "benjamin franklin": (1706, 1790), "thomas jefferson": (1743, 1826),
    "mozart": (1756, 1791), "wolfgang amadeus mozart": (1756, 1791),
    "beethoven": (1770, 1827), "ludwig van beethoven": (1770, 1827),
    "catherine the great": (1729, 1796), "marie antoinette": (1755, 1793),
    "horatio nelson": (1758, 1805), "duke of wellington": (1769, 1852),
    "simon bolivar": (1783, 1830), "jane austen": (1775, 1817),
    "ludwig wittgenstein": (1889, 1951),
    # photographic-era anchors (used to prove the negative case in tests)
    "abraham lincoln": (1809, 1865), "charles darwin": (1809, 1882),
    "karl marx": (1818, 1883), "ulysses s. grant": (1822, 1885),
    "john d. rockefeller": (1839, 1937), "john d rockefeller": (1839, 1937),
    "andrew carnegie": (1835, 1919), "winston churchill": (1874, 1965),
    "albert einstein": (1879, 1955), "theodore roosevelt": (1858, 1919),
    "queen victoria": (1819, 1901), "al capone": (1899, 1947),
    "john f. kennedy": (1917, 1963), "john f kennedy": (1917, 1963),
}

_GENERIC = {"the", "of", "sir", "dr", "mr", "mrs", "ms", "jr", "sr", "king",
            "queen", "emperor", "general", "president", "lord", "lady", "saint",
            "st", "i", "ii", "iii", "iv", "v", "the great", "von", "van", "de"}


def normalize_name(name) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _tokens(s) -> list[str]:
    return [t for t in re.split(r"[\s\-.,_]+", normalize_name(s))
            if t and t not in _GENERIC]


def name_match(name: str, candidate_title: str) -> float:
    """0..1 token overlap; the SURNAME (last meaningful token) must be present.
    Mirrors footage._name_match so callers get one consistent strict gate."""
    nt, tt = _tokens(name), set(_tokens(candidate_title))
    if not nt or not tt:
        return 0.0
    if nt[-1] not in tt:
        return 0.0
    return sum(1 for t in nt if t in tt) / max(1, len(nt))


def figure_years(name) -> tuple[int, int] | None:
    """(birth, death) for a known figure, else None. BCE years are negative."""
    n = normalize_name(name)
    if n in _FIGURE_YEARS:
        return _FIGURE_YEARS[n]
    # try without a trailing epithet ("napoleon bonaparte" → "napoleon")
    toks = n.split()
    for k in (" ".join(toks[:2]), toks[0] if toks else ""):
        if k in _FIGURE_YEARS:
            return _FIGURE_YEARS[k]
    return None


def era_class(name, *, birth: int | None = None, death: int | None = None
              ) -> tuple[str, str]:
    """Classify ('pre_photographic' | 'photographic' | 'unknown', reason).

    A person is PRE-PHOTOGRAPHIC when they could not have a real photograph:
    death before ~1855, or (no death) birth before ~1790. Born after ~1830 ⇒
    photographic. Everything else (and unknown years) ⇒ 'photographic' as the
    SAFE default (we never force artwork onto someone who may have real photos)."""
    if birth is None and death is None:
        yrs = figure_years(name)
        if yrs:
            birth, death = yrs
    if birth is None and death is None:
        return "unknown", "no birth/death year known"
    if death is not None and death < PHOTO_PRACTICAL_YEAR:
        return "pre_photographic", f"died {death} < {PHOTO_PRACTICAL_YEAR} (pre-photography)"
    if death is None and birth is not None and birth < PHOTO_PRACTICAL_YEAR - 65:
        return "pre_photographic", f"born {birth}, pre-photographic era"
    if birth is not None and birth >= PHOTO_CERTAIN_BIRTH_YEAR:
        return "photographic", f"born {birth} >= {PHOTO_CERTAIN_BIRTH_YEAR} (photographic era)"
    if death is not None and death >= PHOTO_PRACTICAL_YEAR:
        return "photographic", f"lived past {PHOTO_PRACTICAL_YEAR} (likely photographed)"
    return "photographic", "default: assume photographable"


def prefers_artwork(name, *, birth: int | None = None, death: int | None = None) -> bool:
    """True ⇒ search for a contemporary PAINTING/ENGRAVING before any photo/AI."""
    cls, _ = era_class(name, birth=birth, death=death)
    return cls == "pre_photographic"


def portrait_queries(name, *, prephoto: bool | None = None,
                     birth: int | None = None, death: int | None = None) -> list[str]:
    """Ordered search queries that bias toward the correct visual TYPE.

    Pre-photographic → painting/engraving/portrait first (NEVER 'photograph').
    Photographic → portrait/photograph (the existing behaviour)."""
    nm = str(name or "").strip()
    if prephoto is None:
        prephoto = prefers_artwork(name, birth=birth, death=death)
    if prephoto:
        return [f"{nm} portrait painting", f"{nm} painting", f"{nm} engraving",
                f"{nm} bust", f"{nm} portrait", nm]
    return [f"{nm} portrait", f"{nm} photograph", nm]


# visual-type keywords used to recognise (and prefer) artwork in a source title.
_ARTWORK_WORDS = ("painting", "portrait", "engraving", "lithograph", "drawing",
                  "etching", "bust", "statue", "sculpture", "fresco", "illustration",
                  "miniature", "oil on", "mezzotint", "woodcut", "watercolor",
                  "watercolour", "daguerreotype")
_MODERN_PHOTO_WORDS = ("getty", "shutterstock", "reuters", "associated press",
                       "ap photo", "stock photo", "premiere", "red carpet",
                       "press conference", "interview", "cosplay", "reenactor",
                       "reenactment", "actor", "movie", "film still", "tv series")


def source_is_artwork(title: str, prov: dict | None = None) -> bool:
    t = (title or "").lower()
    if prov:
        t += " " + " ".join(str(prov.get(k, "")) for k in
                            ("source", "image_url", "license")).lower()
    return any(w in t for w in _ARTWORK_WORDS)


def acceptable_source(name: str, *, title: str = "", prov: dict | None = None,
                      name_match_score: float | None = None,
                      min_name_match: float = 0.5) -> tuple[bool, str]:
    """Strict reusable gate. Rejects weak name matches, obvious modern stock /
    reenactor / actor imagery, and (for pre-photographic people) modern photos.
    Returns (ok, reason) — always logged by the caller."""
    nm = name_match_score
    if nm is None:
        nm = name_match(name, title) if title else None
    if nm is not None and nm < min_name_match:
        return False, f"weak name match {nm:.2f} < {min_name_match}"
    blob = (title or "").lower()
    if prov:
        blob += " " + " ".join(str(prov.get(k, "")) for k in
                               ("source", "domain", "image_url")).lower()
    if any(w in blob for w in _MODERN_PHOTO_WORDS):
        return False, "modern stock / reenactor / actor imagery"
    if prefers_artwork(name) and not source_is_artwork(title, prov):
        # a pre-photographic person should be artwork; a non-artwork hit here is
        # only acceptable if it is plainly a museum/commons/archive scan
        dom = str((prov or {}).get("domain", "")).lower()
        trusted = any(d in dom for d in ("wikimedia", "wikipedia", "loc.gov",
                                         "metmuseum", "rijksmuseum", "britishmuseum",
                                         "nationalgallery", "npg.org", "archive.org"))
        if not trusted:
            return False, "pre-photographic person but source is not artwork/trusted"
    return True, "ok"


def wikipedia_birth_death(name: str, *, timeout: float = 6.0
                          ) -> tuple[int | None, int | None]:
    """Best-effort live lookup of (birth, death) from the Wikipedia summary for an
    UNKNOWN name. Network; returns (None, None) on any failure. Never raises."""
    try:
        import urllib.parse
        import urllib.request
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(name.replace(" ", "_")))
        req = urllib.request.Request(url, headers={"User-Agent": "Vidlore/portrait_intel"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            import json
            data = json.loads(r.read().decode("utf-8", "ignore"))
        text = (data.get("extract", "") or "") + " " + (data.get("description", "") or "")
        # birth: "(1769–1821)" or "born 1769" / "1769 –" ; death: second year
        m = re.search(r"\(\s*(?:born\s*)?(\d{3,4})\s*[–\-]\s*(\d{3,4})", text)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"\bborn\b[^\d]{0,12}(\d{3,4})", text)
        if m:
            return int(m.group(1)), None
        m = re.search(r"\((\d{3,4})\s*[–\-]\s*present", text)
        if m:
            return int(m.group(1)), None
    except Exception:                                              # noqa: BLE001
        return None, None
    return None, None


def assess(name: str, *, use_network: bool = False) -> dict:
    """One-shot summary for logging / QA: classification + recommended queries +
    whether artwork is preferred. `use_network` allows a Wikipedia year lookup for
    unknown names (off by default so it stays deterministic + offline-testable)."""
    yrs = figure_years(name)
    birth, death = (yrs if yrs else (None, None))
    if yrs is None and use_network:
        birth, death = wikipedia_birth_death(name)
    cls, reason = era_class(name, birth=birth, death=death)
    pref = cls == "pre_photographic"
    return {"name": name, "birth": birth, "death": death, "era_class": cls,
            "reason": reason, "prefers_artwork": pref,
            "queries": portrait_queries(name, prephoto=pref)}
