"""Web Image Discovery Engine — find topic-relevant photos on the open
web and route them through the existing image-to-Ken-Burns pipeline.

Pivoted from web_footage.py: video extraction from search-result pages
is unreliable (Pexels/Pixabay listings, Vimeo JS players, YouTube DRM,
archive.org filename guessing). Images, by contrast, are almost always
direct .jpg/.png URLs in result pages and OpenGraph meta — extraction
is reliable, and the existing _cover_to_canvas + Ken Burns assembly
turns a still into a cinematic clip indistinguishable from B-roll.

Pipeline:
    discover_pool() — DuckDuckGo image API + Wikimedia Commons image
                       search + (optional) SearXNG; returns Candidate[]
    score_candidate() — relevance, resolution, watermark hint, dedupe
    pick_for_scene()  — re-rank pool per scene, return best above min
    download_candidate() — fetch image to local cache, hash, verify
    save_manifest()   — emit web_image_manifest.json + sidecar list

Cache layout:
    ~/.vidlore/web_image_search_cache/   — DDG/SearXNG JSON results
    ~/.vidlore/web_image_cache/          — downloaded .jpg files
    ~/.vidlore/web_image_rejects/        — URLs we've already rejected

Env:
    WEB_IMAGE_ENGINE=1          # opt-in (default off in env)
    WEB_IMAGE_MIX=balanced       # off|light|balanced|heavy
    WEB_IMAGE_MIN_SCORE=         # override mix-driven default
    WEB_IMAGE_MAX_PROJECT_QUERIES=12
    WEB_IMAGE_MAX_SCENE_QUERIES=2
    WEB_IMAGE_MAX_TOTAL_QUERIES=40
    WEB_IMAGE_MAX_DOWNLOADS_PER_SCENE=2
    WEB_IMAGE_CACHE_TTL_DAYS=30
    WEB_IMAGE_WATERMARK_FILTER=balanced  # off|loose|balanced|strict
    SEARXNG_URL=                  # shared with web_footage
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# --------------------------------------------------------------------------- #
# Cache + session
# --------------------------------------------------------------------------- #
_CACHE_ROOT = Path(os.environ.get("VIDLORE_WEB_IMG_CACHE",
                                    str(Path.home() / ".vidlore"))).expanduser()
SEARCH_CACHE = _CACHE_ROOT / "web_image_search_cache"
IMAGE_CACHE  = _CACHE_ROOT / "web_image_cache"
REJECT_CACHE = _CACHE_ROOT / "web_image_rejects"
for d in (SEARCH_CACHE, IMAGE_CACHE, REJECT_CACHE):
    d.mkdir(parents=True, exist_ok=True)

_DEFAULT_TTL_DAYS = 30
_UA = {"User-Agent":
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"}

_session = requests.Session()
_session.headers.update(_UA)
_search_lock = threading.Lock()

# --------------------------------------------------------------------------- #
# Domain reputation — same philosophy as web_footage but image-tuned
# --------------------------------------------------------------------------- #
_DOMAIN_BOOST = {
    "wikimedia.org":         +28,
    "commons.wikimedia.org": +28,
    "en.wikipedia.org":      +22,
    "wikipedia.org":         +22,
    "loc.gov":               +24,
    "nasa.gov":              +24,
    "archives.gov":          +22,
    "nara.gov":              +22,
    "smithsonian.org":       +20,
    "si.edu":                +20,
    "europeana.eu":          +20,
    "flickr.com":            +16,   # Commons + creative commons heavy
    "unsplash.com":          +14,
    "pexels.com":            +14,
    "pixabay.com":           +14,
    "burst.shopify.com":     +12,
    "openverse.org":         +14,
    "rawpixel.com":          +10,
    "publicdomainpictures.net": +10,
    "freepik.com":           +6,
    "nytimes.com":           +8,
    "bbc.co.uk":             +8,
    "bbc.com":               +8,
    "nationalgeographic.com": +12,
    # TLD baselines
    "edu":  +12,
    "gov":  +14,
    # Risky / aggressive watermark sites — keep but penalise
    "shutterstock.com":   -12,
    "gettyimages.com":    -12,
    "istockphoto.com":    -12,
    "alamy.com":          -10,
    "depositphotos.com":  -8,
    "dreamstime.com":     -8,
    "stock.adobe.com":    -8,
}

# Domain substrings we hard-skip — DRM/paywall/known-low-quality.
# Substring match (not exact) so "www.pinterest.com" + "i.pinimg.com"
# (Pinterest's CDN) both get caught by "pinterest" / "pinimg".
_DOMAIN_BLACKLIST_PARTS = (
    "pinterest.", "pinimg.com",                            # mostly low-res repins
    "facebook.com", "instagram.com", "tiktok.com",         # auth required
    "twitter.com", "x.com", "t.co",
    "netflix.com", "amazon.com", "ebay.com",
)

# Paid-stock domains that publish WATERMARKED preview images. These
# previews are visibly stamped with the vendor's logo/text overlay —
# unusable in a real edit. We HARD-REJECT at discovery time rather
# than rely on the corner-edge heuristic, which can miss centered
# watermarks (verified in v3 audit: Alamy previews passed the corner
# check). NEVER remove watermarks to "rescue" these — that's a
# licensing violation; we just don't take them.
_WATERMARK_STOCK_DOMAINS = (
    "alamy.com",            "alamy.de",       "alamy.fr",
    "gettyimages.com",      "gettyimages.co.uk",
    "istockphoto.com",
    "shutterstock.com",
    "dreamstime.com",
    "depositphotos.com",
    "stock.adobe.com",      "adobe.com/stock",
    "stockphotos.com",
    "123rf.com",
    "fotolia.com",
    "bigstockphoto.com",
)

# Hotel/travel/booking aggregators — image content is usually generic
# postcard / room photos, rarely topic-specific documentary visuals.
# Soft penalty, not hard-reject (sometimes a regional travel page
# has the only good shot of a niche place).
_TRAVEL_HOTEL_DOMAINS = (
    "hotels.com", "booking.com", "expedia.com", "tripadvisor.com",
    "trivago.com", "agoda.com", "airbnb.com",
    "tripjive.com", "travellemming.com", "usatipps.de", "usatipps",
    "de.hotels", "hotels.de",
    "thrillophilia.com", "kayak.com",
)


def _is_blacklisted(dom: str) -> bool:
    d = (dom or "").lower()
    return any(p in d for p in _DOMAIN_BLACKLIST_PARTS)


def _is_watermark_stock(dom: str) -> bool:
    """Paid-stock domain that publishes visibly-watermarked previews."""
    d = (dom or "").lower()
    return any(p in d for p in _WATERMARK_STOCK_DOMAINS)


def _is_travel_hotel(dom: str) -> bool:
    """Hotel/travel/booking aggregator — usually generic content."""
    d = (dom or "").lower()
    return any(p in d for p in _TRAVEL_HOTEL_DOMAINS)

# Extensions we treat as valid image candidates
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class WebImageConfig:
    enabled: bool = False
    provider: str = "free"           # free|searxng
    mix: str = "balanced"            # off|light|balanced|heavy
    min_score: int = 55
    searxng_url: str = ""
    max_project_queries: int = 12
    max_scene_queries: int = 2
    max_total_queries: int = 40
    max_downloads_per_scene: int = 2
    cache_ttl_days: int = _DEFAULT_TTL_DAYS
    watermark_filter: str = "balanced"   # off|loose|balanced|strict
    min_width: int = 800
    min_height: int = 500
    # Pacing caps — prevent slideshow feel
    max_timeline_share: float = 0.40     # hard cap on % of beats with web img
    max_consecutive_beats: int = 3       # max scenes in a row using web img


def mix_target_fraction(mix: str) -> float:
    """Target % of timeline beats filled by web images. Enforced as a
    HARD CAP in fetch_footage — once we hit the target, the web-image
    tier stops picking and scenes fall through to stock video / AI image
    instead. Keeps the edit feeling like a documentary, not a slideshow.

    Calibrated 2026-05-26 after the first A/B test produced 76% web
    images (way over balanced=40% target) because no cap existed."""
    return {"off": 0.0, "light": 0.20, "balanced": 0.40, "heavy": 0.60} \
        .get(mix.lower(), 0.40)


def mix_max_consecutive(mix: str) -> int:
    """Max scenes in a row that can use a web image before forced
    fallthrough to stock/AI. Prevents the slideshow feel even when the
    topic is image-rich and the timeline-share cap hasn't fired yet."""
    return {"off": 0, "light": 2, "balanced": 3, "heavy": 4}.get(mix.lower(), 3)


def mix_min_score(mix: str, base: int = 50) -> int:
    """Mix slider = strictness knob. Pool is already quality-gated upstream
    (resolution + domain + watermark hint), so min_score is the relevance dial."""
    return {"off": 999, "light": base + 5,
            "balanced": base, "heavy": base - 5}.get(mix.lower(), base)


def cfg_from_env() -> WebImageConfig:
    def _b(k: str, d: bool = False) -> bool:
        v = os.environ.get(k, "").strip().lower()
        return v in ("1", "true", "yes", "on") if v else d
    def _i(k: str, d: int) -> int:
        try:    return int(os.environ.get(k, "").strip() or d)
        except ValueError: return d
    def _f(k: str, d: float) -> float:
        try:    return float(os.environ.get(k, "").strip() or d)
        except ValueError: return d
    mix = os.environ.get("WEB_IMAGE_MIX", "balanced").lower()
    return WebImageConfig(
        enabled=_b("WEB_IMAGE_ENGINE", False),
        provider=os.environ.get("WEB_IMAGE_PROVIDER", "free").lower(),
        mix=mix,
        min_score=_i("WEB_IMAGE_MIN_SCORE", mix_min_score(mix)),
        searxng_url=os.environ.get("SEARXNG_URL", "").rstrip("/"),
        max_project_queries=_i("WEB_IMAGE_MAX_PROJECT_QUERIES", 12),
        max_scene_queries=_i("WEB_IMAGE_MAX_SCENE_QUERIES", 2),
        max_total_queries=_i("WEB_IMAGE_MAX_TOTAL_QUERIES", 40),
        max_downloads_per_scene=_i("WEB_IMAGE_MAX_DOWNLOADS_PER_SCENE", 2),
        cache_ttl_days=_i("WEB_IMAGE_CACHE_TTL_DAYS", _DEFAULT_TTL_DAYS),
        watermark_filter=os.environ.get(
            "WEB_IMAGE_WATERMARK_FILTER", "balanced").lower(),
        # Pacing caps — mix-driven defaults, env overrides
        max_timeline_share=_f("WEB_IMAGE_MAX_TIMELINE_SHARE",
                               mix_target_fraction(mix)),
        max_consecutive_beats=_i("WEB_IMAGE_MAX_CONSECUTIVE_BEATS",
                                   mix_max_consecutive(mix)),
    )


# --------------------------------------------------------------------------- #
# Candidate record
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    title: str
    image_url: str                         # the direct .jpg/.png URL
    source_page: str = ""                  # the HTML page the image was on
    source_site: str = ""
    snippet: str = ""
    width: int = 0
    height: int = 0
    file_hash: str = ""
    relevance_score: int = 0
    rejection: str = ""
    query: str = ""
    selected_for_scene: int = -1
    local_path: str = ""
    downloaded: bool = False

    def to_dict(self) -> dict:
        return {
            "title": self.title, "image_url": self.image_url,
            "source_page": self.source_page, "source_site": self.source_site,
            "snippet": self.snippet, "width": self.width, "height": self.height,
            "file_hash": self.file_hash, "relevance_score": self.relevance_score,
            "rejection": self.rejection, "query": self.query,
            "selected_for_scene": self.selected_for_scene,
            "local_path": self.local_path, "downloaded": self.downloaded,
        }


# --------------------------------------------------------------------------- #
# Budget tracker
# --------------------------------------------------------------------------- #
class _Budget:
    def __init__(self, cfg: WebImageConfig) -> None:
        self.cfg = cfg
        self.project = 0
        self.scene = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self.project + self.scene

    def can_project(self) -> bool:
        with self._lock:
            return (self.project < self.cfg.max_project_queries
                    and self.total < self.cfg.max_total_queries)

    def can_scene(self) -> bool:
        with self._lock:
            return (self.scene < self.cfg.max_scene_queries
                    and self.total < self.cfg.max_total_queries)

    def spend_project(self) -> None:
        with self._lock:
            self.project += 1

    def spend_scene(self) -> None:
        with self._lock:
            self.scene += 1

    def hit_cache(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def snapshot(self) -> dict:
        return {
            "project": self.project, "scene": self.scene, "total": self.total,
            "cache": self.cache_hits,
            "cap_project": self.cfg.max_project_queries,
            "cap_scene": self.cfg.max_scene_queries,
            "cap_total": self.cfg.max_total_queries,
        }


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _query_cache_path(provider: str, query: str) -> Path:
    h = hashlib.sha256(f"{provider}::{query}".encode()).hexdigest()[:24]
    return SEARCH_CACHE / f"{provider}_{h}.json"


def _read_cache(p: Path, ttl_days: int) -> Optional[list[dict]]:
    if not p.exists():
        return None
    age = (time.time() - p.stat().st_mtime) / 86400.0
    if age > ttl_days:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(p: Path, data: list[dict]) -> None:
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _reject_seen(url: str) -> bool:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return (REJECT_CACHE / h).exists()


def _reject_mark(url: str, reason: str) -> None:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    try:
        (REJECT_CACHE / h).write_text(reason or "rejected", encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Search providers
# --------------------------------------------------------------------------- #
def _domain_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


def _search_duckduckgo_images(query: str, n: int = 20) -> list[dict]:
    """DuckDuckGo image search via their public JS endpoint.

    DDG's image search uses a 2-step flow:
      1. GET /?q=<query> — returns HTML with a vqd token
      2. GET /i.js?l=us-en&o=json&q=<query>&vqd=<token> — returns JSON
    """
    out: list[dict] = []
    try:
        # Step 1 — pull the vqd token from the search page
        r = _session.get("https://duckduckgo.com/",
                          params={"q": query, "iax": "images", "ia": "images"},
                          timeout=20)
        if r.status_code != 200:
            return out
        m = re.search(r"vqd=['\"]?(\d+-\d+(?:-\d+)?)['\"]?", r.text)
        if not m:
            m = re.search(r"vqd=([\d-]+)", r.text)
        if not m:
            return out
        vqd = m.group(1)
        # Step 2 — JSON image results
        time.sleep(0.4)             # be polite
        r2 = _session.get(
            "https://duckduckgo.com/i.js",
            params={"l": "us-en", "o": "json", "q": query, "vqd": vqd,
                    "f": ",,,", "p": "1", "v7exp": "a"},
            headers={**_UA, "Referer": "https://duckduckgo.com/"},
            timeout=25,
        )
        if r2.status_code != 200:
            return out
        try:
            data = r2.json()
        except Exception:
            return out
        for item in (data.get("results") or [])[:n]:
            url = item.get("image") or ""
            src = item.get("url") or ""
            if not url:
                continue
            out.append({
                "image_url": url,
                "source_page": src,
                "source_site": _domain_of(src) or _domain_of(url),
                "title": html.unescape(item.get("title") or ""),
                "snippet": "",
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
            })
    except Exception:
        return out
    return out


def _search_searxng_images(query: str, base: str, n: int) -> list[dict]:
    out: list[dict] = []
    if not base:
        return out
    try:
        r = _session.get(
            f"{base}/search",
            params={"q": query, "format": "json", "categories": "images"},
            timeout=25,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return out
    for item in (data.get("results") or [])[:n]:
        url = item.get("img_src") or item.get("thumbnail_src") or ""
        if not url:
            continue
        out.append({
            "image_url": url,
            "source_page": item.get("url", ""),
            "source_site": _domain_of(item.get("url", "")) or _domain_of(url),
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "width": int(item.get("img_width") or 0),
            "height": int(item.get("img_height") or 0),
        })
    return out


def _search_wikimedia_images(query: str, n: int = 12) -> list[dict]:
    """Wikimedia Commons direct API — returns full-resolution image URLs."""
    out: list[dict] = []
    try:
        r = _session.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": n,
                "prop": "imageinfo", "iiprop": "url|size|mime",
                "iiurlwidth": 1920,
            },
            timeout=20,
        )
        r.raise_for_status()
        pages = list((r.json().get("query", {}) or {}).get("pages", {}).values())
    except Exception:
        return out
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        mime = (ii.get("mime") or "").lower()
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        url = ii.get("thumburl") or ii.get("url") or ""
        if not url:
            continue
        out.append({
            "image_url": url,
            "source_page": ii.get("descriptionurl", "") or url,
            "source_site": "commons.wikimedia.org",
            "title": (p.get("title") or "").replace("File:", "")
                                              .rsplit(".", 1)[0],
            "snippet": "",
            "width": int(ii.get("width") or 0),
            "height": int(ii.get("height") or 0),
        })
    return out


def _search(query: str, cfg: WebImageConfig, budget: _Budget,
            project_level: bool) -> list[dict]:
    """Provider dispatcher with disk cache + budget enforcement.
    Always combines DDG with Wikimedia so we get both popular web images
    AND curated archival photos in a single search slot."""
    qclean = query.strip()
    if not qclean:
        return []
    if project_level:
        if not budget.can_project():
            return []
        budget.spend_project()
    else:
        if not budget.can_scene():
            return []
        budget.spend_scene()

    # Disk cache (provider-namespaced)
    cached_all = _read_cache(_query_cache_path("combined", qclean),
                              cfg.cache_ttl_days)
    if cached_all is not None:
        budget.hit_cache()
        return cached_all

    results: list[dict] = []
    # DDG image API (always tried — broadest pool)
    try:
        results.extend(_search_duckduckgo_images(qclean, n=20))
    except Exception:
        pass
    # Wikimedia Commons (always — curated, license-clean, high-res)
    try:
        results.extend(_search_wikimedia_images(qclean, n=10))
    except Exception:
        pass
    # SearXNG if configured
    if cfg.provider == "searxng" and cfg.searxng_url:
        try:
            results.extend(_search_searxng_images(qclean, cfg.searxng_url, 12))
        except Exception:
            pass

    _write_cache(_query_cache_path("combined", qclean), results)
    return results


# --------------------------------------------------------------------------- #
# Query generation
# --------------------------------------------------------------------------- #
_BAD_TOPIC_WORDS = {
    "she", "he", "they", "it", "we", "you", "i",
    "the", "this", "that", "those", "these",
    "but", "and", "or", "yet", "so",
    "after", "before", "during", "when", "while", "then",
    "now", "today", "yesterday", "tomorrow",
    "what", "where", "why", "how", "yes", "no",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "self", "sufficient", "true", "real", "new", "old", "best", "worst",
    "first", "last", "next", "every", "another", "other", "such",
    "some", "any", "all", "most", "many", "few", "very",
    "good", "great", "bad", "small", "big", "huge", "tiny",
}


def _topic_phrases(title: str, full_script: str) -> list[str]:
    text = f"{title}\n{full_script}"
    props = re.findall(r"\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){0,2})\b", text)
    props = [p for p in props
             if p.split()[0].lower() not in _BAD_TOPIC_WORDS]
    counts: dict[str, int] = {}
    for p in props:
        counts[p] = counts.get(p, 0) + 1
    multi = [p for p, c in counts.items() if c >= 2]
    singletons_multi_word = [p for p, c in counts.items()
                              if c == 1 and " " in p]
    strong = list(dict.fromkeys(multi + singletons_multi_word))
    if not strong:
        strong = list(dict.fromkeys(props))[:4]
    seen, out = set(), []
    for p in strong:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) >= 8:
            break
    return out


def _primary_anchor_words(brief_title: str, full_script: str,
                            topic_words: set[str]) -> set[str]:
    """Return the 2-3 most-mentioned content words from the script.

    These are the CENTRAL topic anchors (e.g. "amish" for an Amish
    farming doc; "elderly" / "japanese" for a Japanese elderly health
    doc). A candidate that doesn't contain at least one is almost
    certainly a weak match — letting it through is how we got
    travel/hotel pages matching only on "Pennsylvania".

    Methodology:
      1. Combine title + script into a content blob.
      2. Word-frequency-count the proper-noun + lowercase content tokens.
      3. Pick the 3 highest-count topic words.
      4. Always include any 4+ letter word that appears in the TITLE
         (titles concentrate the topic).

    Synonym tables are intentionally NOT built — that risks both
    false positives and brittleness across languages. The pure
    frequency signal is robust across English / Japanese / Korean /
    Turkish briefs (the script writer's own word choice carries the
    weight).
    """
    text = f"{brief_title}\n{brief_title}\n{full_script}"      # double-weight title
    counts: dict[str, int] = {}
    for w in re.findall(r"[a-zA-Z]{4,}", text.lower()):
        if w in _BAD_TOPIC_WORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    # Score = count, but require minimum 2 mentions to qualify
    candidates = [(c, w) for w, c in counts.items() if c >= 2]
    candidates.sort(reverse=True)
    primary: set[str] = set()
    for _, w in candidates[:6]:
        primary.add(w)
    # Always seed with title content words (so single-mention proper
    # nouns from the title don't get dropped).
    for w in _word_set(brief_title):
        if w not in _BAD_TOPIC_WORDS and len(w) >= 4:
            primary.add(w)
    # Strip the same generics we strip from topic_words
    _GENERIC = {"world", "history", "story", "stories", "documentary",
                 "video", "videos", "photo", "photos", "image", "images",
                 "picture", "pictures", "film", "movie", "channel",
                 "show", "shows", "people", "lives", "things"}
    primary -= _GENERIC
    # Cap at ~6 anchors so we keep the gate STRONG (1-of-6, not 1-of-20)
    if len(primary) > 6:
        # Keep the title-derived ones first, then top-by-frequency
        title_set = set(w for w in _word_set(brief_title)
                          if len(w) >= 4 and w not in _BAD_TOPIC_WORDS)
        ordered = (list(title_set) +
                   [w for _, w in candidates if w not in title_set])
        primary = set(ordered[:6])
    return primary


# Multilingual culture/audience -> native-language qualifiers. Keeps queries
# image-search friendly (people, places, scenes).
_AUDIENCE_NATIVE = {
    "ja": ["日本", "日本人", "高齢者"],            # Japan, Japanese, elderly
    "ko": ["한국", "한국인", "서울"],              # Korea, Korean, Seoul
    "zh": ["中国", "中国人"],
    "ar": ["العربية"],
    "tr": ["Türkiye", "Türk", "İstanbul"],
    "es": ["España", "español"],
    "pt": ["Brasil", "brasileiro"],
    "de": ["Deutschland", "deutsch"],
    "fr": ["France", "français"],
    "it": ["Italia", "italiano"],
    "ur": [],
    "hi": ["भारत", "भारतीय"],
}
_AUDIENCE_EN_PREFIX = {
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "tr": "Turkish", "es": "Spanish",
    "pt": "Brazilian", "de": "German", "fr": "French",
    "it": "Italian", "ur": "South Asian", "hi": "Indian",
    "en": "",
}


def build_project_queries(brief_title: str, full_script: str,
                            audience: str = "auto",
                            language: str = "en",
                            max_queries: int = 12) -> list[str]:
    """Generate ~10-12 broad image-search queries. Mix of:
       title + qualifier, topic + qualifier, audience-prefixed,
       optional native-language for non-English audiences."""
    topics = _topic_phrases(brief_title, full_script)
    en_prefix = _AUDIENCE_EN_PREFIX.get((audience or "").lower(), "")
    native_q = _AUDIENCE_NATIVE.get((audience or "").lower(), [])
    qualifiers = ("photo", "photograph", "high resolution photo",
                  "documentary photo")
    queries: list[str] = []
    seen: set[str] = set()
    def _add(q: str) -> None:
        s = q.strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            queries.append(s)
    # Title baseline
    _add(f"{brief_title} {qualifiers[0]}")
    _add(f"{brief_title} {qualifiers[2]}")
    # Topic × qualifier
    for t in topics[:5]:
        _add(f"{t} {qualifiers[0]}")
        _add(f"{t} {qualifiers[3]}")
    # Audience prefix
    if en_prefix:
        for t in topics[:3]:
            _add(f"{en_prefix} {t}")
    # Native-language slot (max 2 to stay within budget)
    for nq in native_q[:2]:
        if topics:
            _add(f"{topics[0]} {nq}")
    return queries[:max(1, int(max_queries))]


def build_scene_queries(scene_text: str, audience: str = "auto",
                          max_q: int = 2) -> list[str]:
    topics = _topic_phrases("", scene_text)
    en_prefix = _AUDIENCE_EN_PREFIX.get((audience or "").lower(), "")
    out: list[str] = []
    if topics:
        out.append(f"{topics[0]} photo")
        if en_prefix and len(out) < max_q:
            out.append(f"{en_prefix} {topics[0]}")
    return out[:max_q]


# --------------------------------------------------------------------------- #
# URL-pattern filter — keep only direct image URLs
# --------------------------------------------------------------------------- #
def _looks_like_image_url(url: str) -> bool:
    u = (url or "").lower().split("?", 1)[0]
    return any(u.endswith(e) for e in _IMAGE_EXTS)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_pool(brief_title: str, full_script: str,
                    cfg: WebImageConfig,
                    audience: str = "auto",
                    language: str = "en") -> tuple[list[Candidate], _Budget]:
    """Project-level discovery — runs ONCE per render."""
    budget = _Budget(cfg)
    queries = build_project_queries(brief_title, full_script,
                                      audience=audience, language=language,
                                      max_queries=cfg.max_project_queries)
    pool: list[Candidate] = []
    seen_urls: set[str] = set()
    dropped_blacklist = 0
    dropped_stock_wm = 0
    for q in queries:
        if not budget.can_project():
            break
        for r in _search(q, cfg, budget, project_level=True):
            url = r.get("image_url", "")
            if not url or url in seen_urls:
                continue
            dom = _domain_of(url) or _domain_of(r.get("source_page", ""))
            src_dom = _domain_of(r.get("source_page", ""))
            if _is_blacklisted(dom) or _is_blacklisted(src_dom):
                dropped_blacklist += 1
                continue
            # Hard-reject watermarked paid-stock domains. Their previews
            # have visible logo overlays we can't use without licensing.
            if _is_watermark_stock(dom) or _is_watermark_stock(src_dom):
                dropped_stock_wm += 1
                continue
            if not _looks_like_image_url(url):
                continue
            if _reject_seen(url):
                continue
            seen_urls.add(url)
            pool.append(Candidate(
                title=r.get("title", ""),
                image_url=url,
                source_page=r.get("source_page", ""),
                source_site=r.get("source_site", "") or dom,
                snippet=r.get("snippet", ""),
                width=r.get("width", 0),
                height=r.get("height", 0),
                query=q,
            ))
    # Topic-keyword pre-filter — DDG sometimes returns off-topic images
    brief_topics = _topic_phrases(brief_title, full_script)
    topic_words: set[str] = set()
    for t in brief_topics:
        topic_words |= _word_set(t)
    _GENERIC = {"world", "history", "story", "stories", "documentary",
                "video", "videos", "photo", "photos", "image", "images",
                "picture", "pictures", "film", "movie", "channel", "show",
                "day", "year", "people", "life", "thing", "time", "way",
                "place", "north", "south", "east", "west", "first", "last",
                "best", "free", "new", "old"}
    topic_words -= _GENERIC
    # Primary-anchor requirement: derive the most-mentioned topic
    # word(s) from the script (e.g. "amish" for Amish farming doc,
    # "elderly" for Japanese elderly doc). At least ONE primary
    # anchor must appear in title+snippet, otherwise the candidate
    # is a weak topic-tangent (hotel page that just mentions
    # "Pennsylvania", or "Japan travel guide" without "elderly").
    primary_anchors = _primary_anchor_words(brief_title, full_script,
                                              topic_words)
    # Scene-intent profile — picks the right negative-anchor list
    # for the brief's category (health/longevity vs war/history vs
    # agriculture). Without this, a Japan health doc happily picks
    # "Inside the Prisons of Japan" because "japan" matches the
    # primary anchor.
    intent = detect_intent_profile(brief_title, full_script)
    intent_prof = _INTENT_PROFILES.get(intent) if intent else None
    cleaned: list[Candidate] = []
    dropped_off_topic = 0
    dropped_no_anchor = 0
    dropped_neg_context = 0
    dropped_intent = 0
    dropped_video_thumb = 0
    dropped_poster_title = 0
    dropped_blog_domain = 0
    for c in pool:
        c_words = _word_set(c.title) | _word_set(c.snippet)
        if topic_words and c_words and not (c_words & topic_words):
            dropped_off_topic += 1
            continue
        if primary_anchors and c_words and not (c_words & primary_anchors):
            dropped_no_anchor += 1
            continue
        blob = f"{c.title or ''} {c.snippet or ''}".lower()
        if any(w in blob for w in _NEGATIVE_CONTEXT_WORDS):
            dropped_neg_context += 1
            continue
        if intent_prof:
            if any(w in blob for w in intent_prof["reject_words"]):
                dropped_intent += 1
                continue
        if _is_video_thumbnail(c.image_url):
            dropped_video_thumb += 1
            continue
        # v8 HARD-REJECT on poster/blog-title patterns — was a -20
        # score penalty in v7, but two known stragglers (japanhandbook
        # "Diet & Exercise Tips for Senior Expat Guide", drdanacohen
        # "Healthy Aging Tips for Maintaining Vitality") squeaked past
        # the score floor anyway. Now they're dropped at discovery.
        if _looks_like_poster_title(c.title):
            dropped_poster_title += 1
            continue
        # v8 HARD-REJECT on known blog-domain patterns — handbook,
        # cookbook, recipes, blog., journal., wellness., rdsic, etc.
        # These domains publish title-card / collage / screenshot
        # graphics almost exclusively for their featured-image slot.
        if _is_blog_domain(c.source_site):
            dropped_blog_domain += 1
            continue
        # Resolution gate (skip thumbnails)
        if c.width and c.height:
            if c.width < cfg.min_width or c.height < cfg.min_height:
                continue
        cleaned.append(c)
    # Pre-score against brief
    brief_text = f"{brief_title}\n{full_script}"
    for c in cleaned:
        c.relevance_score = score_candidate(c, brief_text, query_blob=c.query)
    intent_msg = f" intent={intent}" if intent else ""
    print(f"  [web-image] pool: {len(pool)} raw → {len(cleaned)} on-topic"
          f"{intent_msg} (dropped {dropped_blacklist} blacklist, "
          f"{dropped_stock_wm} stock-watermark, "
          f"{dropped_off_topic} off-topic, "
          f"{dropped_no_anchor} no-primary-anchor, "
          f"{dropped_neg_context} negative-context, "
          f"{dropped_intent} intent-reject, "
          f"{dropped_video_thumb} video-thumbnail, "
          f"{dropped_poster_title} poster-title, "
          f"{dropped_blog_domain} blog-domain)", flush=True)
    return cleaned, budget


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _word_set(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower())
            if w not in {"the", "and", "for", "with", "from", "this",
                          "that", "you", "are", "was", "were", "have",
                          "has", "had", "but", "not", "what", "when",
                          "where", "how", "why", "into", "about", "after",
                          "before", "photo", "photograph", "image"}}


# Title hints that strongly suggest a stock-watermarked image
_WATERMARK_HINT_WORDS = {
    "shutterstock", "getty", "istock", "alamy", "depositphotos",
    "dreamstime", "stock-photo", "stock photo", "preview", "watermark",
    "sample", "comp",
}

# Title patterns that signal "designed graphic / blog header / poster /
# infographic / clickbait" rather than a real documentary photograph.
# Extended in v8 (2026-05-26) to cover the rdsic + japanhandbook
# stragglers ("Diet and Exercise Tips for Senior Expat Guide",
# "Japanese people are famous for…").  v8: HARD-REJECT (not penalty).
_POSTER_TITLE_PATTERNS = (
    # Clickbait listicle / guide structure (concrete patterns only —
    # bare " best ", " top " over-match: "The BEST Free Stock" from
    # Pexels and "14 Best Cities" from real travel photos pass through
    # cleanly as long as the rest of the title isn't a guide/tips piece)
    "tips for", "tips to", "tips on", "tips and", "best tips",
    "ways to", "ways for", "things to", "reasons why",
    "guide to", "guide for", "guide on", " guide ", "guide:",
    "how to", "what to", "what are these",
    "top 10", "top 5", "top 7", "top 15", "top 20",
    "best ways", "best ideas",
    "expat guide", "senior expat", "expert tips",
    # Recipe / diet / food blog
    "add to your diet", "add to diet", "for your diet", "diet plan",
    "diet tips", "exercise tips", "recipes", "cookbook",
    "famous for",
    # Health-campaign style
    "embracing", "aging gracefully", "healthy aging month",
    "awareness month", "national month",
    "wellness tips",
    # Blog/infographic / SEO header signatures
    "infographic", "checklist",
    "ultimate guide", "complete guide", "beginner's guide",
    "step by step", "ebook", "free download", "download now",
    "unveiled", "strategies unveiled",
    # Stock/poster site clutter
    "premium vector", "vector illustration", "flat design",
    "concept art", "presentation template", "powerpoint",
    "background design", "wallpaper",
    # Video / cover style
    "video preview", "video cover",
    "title card", "cover image", "playlist cover",
    # Visible URLs in titles (signal: blog header image) — keep
    # ".com" because it's a strong signal even on its own
    ".com", ".org/blog", ".net/blog", " - blog",
)

# Domain substrings that publish lots of blog-header / collage style
# images.  Soft penalty (-15) — sometimes a recipe site has a clean
# food close-up, but the default expectation is title-card or collage.
_BLOG_DOMAIN_PARTS = (
    "handbook", "cookbook", "recipes", "recipe.", "foodnetwork",
    "blog.", "journal.", "magazine.", "wellness.",
    "tipgalore", "tipjive", "thetop", "lifehacker",
    "rdsic", "expatguide", "lifehack",
)


def _looks_like_poster_title(title: str) -> bool:
    """True when the title strongly suggests the image is a designed
    graphic (poster, blog header, infographic) rather than a real
    documentary photo."""
    t = (title or "").lower()
    if not t:
        return False
    return any(p in t for p in _POSTER_TITLE_PATTERNS)


def _is_blog_domain(dom: str) -> bool:
    d = (dom or "").lower()
    return any(p in d for p in _BLOG_DOMAIN_PARTS)

# Title words that strongly indicate off-topic / inappropriate context
# even when the candidate shares a topic anchor with the brief. Caught
# false-positives during the v2/v4 audits (Amish farming pulled in
# "Northern Lights PA"; Japan elderly health pulled in "Inside the
# Prisons of Japan" and "Architect of War"). Each match applies a -25
# penalty that pushes the candidate below typical thresholds.
# Extended 2026-05-26 v5 to cover the war/prison/military/political
# class that slipped through on the Japan health doc.
_NEGATIVE_CONTEXT_WORDS = {
    # weather / phenomena unrelated to most docs
    "aurora", "northern lights", "eclipse", "tornado", "hurricane",
    "blizzard", "wildfire",
    # seasonal scenery that's just postcard padding
    "foliage", "fall colors", "fall color", "autumn leaves",
    # crime / abuse / death / law-enforcement
    "abuse", "assault", "murder", "killed", "rape", "rapist",
    "scandal", "crime", "criminal", "drugs", "drug", "prison",
    "prisons", "jail", "jails", "inmate", "inmates", "police",
    "arrest", "arrested", "execution", "execute", "executed",
    "torture", "war crimes", "death row", "convict",
    # military / war (a recurring bug on regional health docs)
    "war", "wars", "warship", "warships", "battleship", "battleships",
    "warfare", "battle", "battles", "army", "navy", "soldier",
    "soldiers", "marine", "marines", "marines", "tank", "tanks",
    "rifle", "rifles", "weapon", "weapons", "bomb", "bombs",
    "bomber", "missile", "missiles", "ww1", "ww2", "wwi", "wwii",
    "world war", "civil war", "vietnam war", "korean war",
    "invasion", "invader", "general staff", "admiral", "emperor",
    "imperial", "regime", "dictator", "fascist", "nazi",
    "architect of war", "samurai war",
    # 3D / game / cartoon — unless topic is gaming
    "3d model", "turbosquid", "wireframe", "low-poly", "video game",
    "videogame", "anime", "manga",
    # politics / economic policy — usually off-topic for lifestyle
    "election", "elections", "campaign", "politician", "parliament",
    "summit", "treaty", "diplomatic", "economic cooperation",
    # off-topic industries
    "coal country", "coal mining", "fracking", "oil rig",
    # social media / shopping detritus
    "amazon prime", "ebay", "for sale", "buy now",
    # disaster / accident
    "sinkhole", "wreck", "crash", "accident", "explosion",
}

# Video-thumbnail CDNs — these serve video cover/poster images that
# almost always carry clickbait title-text overlays. HARD-REJECTED at
# every layer: discovery, scoring, download.  User direction
# 2026-05-26 v5+: "never use video thumbnails — only real documentary
# photos or actual scene frames extracted from inside the video".
# Frame extraction from video pages would require yt-dlp + ffmpeg
# scrubbing, which is out of scope here; for now we just never take
# the thumbnail / poster / cover image.
_VIDEO_THUMBNAIL_CDNS = (
    # YouTube
    "i.ytimg.com", "img.youtube.com", "i9.ytimg.com",
    "yt3.ggpht.com",                                   # channel art
    "i.ytimg.com/an_webp",
    # Vimeo
    "i.vimeocdn.com", "vimeo.com/video/", "vimeocdn.com",
    # Dailymotion
    "dailymotion.com/thumbnail", "dailymotion.com/sdn",
    "static1.dmcdn.net",
    # TikTok / IG reels
    "tiktok.com/api/img", "p16-sign", "scontent.cdninstagram.com",
    # Twitch / generic poster/cover paths
    "static-cdn.jtvnw.net",
)

# URL fragments that identify thumbnail / cover / poster images even
# from CDNs we haven't pre-listed (catches `/maxresdefault.jpg`,
# `/hqdefault.jpg`, `_thumb.jpg`, `?image=poster`, etc.)
_THUMBNAIL_URL_PATTERNS = (
    "/maxresdefault",        # YouTube high-res thumbnail
    "/hqdefault",            # YouTube standard thumbnail
    "/sddefault", "/mqdefault", "/default.jpg",
    "/0.jpg", "/1.jpg", "/2.jpg", "/3.jpg",   # YouTube fallback thumbs
    "/poster.", "/poster-", "_poster.", "-poster.",
    "/cover.", "/cover-", "_cover.", "-cover.",
    "/thumb.", "/thumb-", "_thumb.", "-thumb.",
    "/thumbnail.", "_thumbnail.",
    "video-thumb", "video_thumb",
    "videothumbnail", "video-thumbnail",
    "/preview.", "_preview.",
)


def _is_video_thumbnail(image_url: str) -> bool:
    """True when the URL screams 'video thumbnail / cover / poster'.

    Multi-layer check:
      1. Host on a known video-thumbnail CDN (YouTube, Vimeo, etc.)
      2. URL path contains a known thumbnail-filename pattern
      3. Generic substring fallback for `thumbnail`/`poster`/`cover`
         when paired with a video-host indicator
    """
    u = (image_url or "").lower()
    if not u:
        return False
    if any(c in u for c in _VIDEO_THUMBNAIL_CDNS):
        return True
    if any(p in u for p in _THUMBNAIL_URL_PATTERNS):
        return True
    return False


# Scene-intent profiles: when the brief topic falls in a known
# category, apply the matching negative anchor list so off-topic
# candidates from completely different categories get rejected even
# if they share the primary topic word.
#
# Example: an "elderly health" brief gets the HEALTH profile which
# excludes war/prison/military/political. A Japan-Tokyo gun brief
# would use a different profile.
_INTENT_PROFILES = {
    # Health / longevity / lifestyle docs — reject everything in
    # negative-list above PLUS the politics class.
    "health": {
        "trigger_words": {
            "health", "healthy", "elderly", "senior", "seniors",
            "aging", "longevity", "lifespan", "wellness", "lifestyle",
            "diet", "nutrition", "fitness", "doctor", "hospital",
            "clinic", "medicine", "medical",
        },
        # Reject candidates containing any of these (semantic
        # negative-profile, on top of the global _NEGATIVE list)
        "reject_words": {
            "prison", "prisons", "jail", "war", "military", "army",
            "navy", "soldier", "weapon", "battle", "invasion",
            "emperor", "imperial", "regime", "dictator", "ww2",
            "wwii", "world war", "samurai war", "architect of war",
            "execution", "torture", "yakuza", "gangster",
        },
        # Soft-require ≥1 of these for full-score (otherwise -10)
        "require_any": {
            "elderly", "senior", "seniors", "old", "aging", "ageing",
            "health", "healthy", "longevity", "lifespan", "doctor",
            "clinic", "hospital", "diet", "food", "nutrition",
            "walking", "yoga", "tai chi", "okinawa", "family",
            "lifestyle", "wellness", "care", "fitness", "exercise",
            "centenarian", "long life", "home", "garden", "meal",
        },
    },
    # History / war / military docs — opposite: REQUIRE war/army
    # anchor and reject lifestyle/wellness fluff.
    "history_war": {
        "trigger_words": {
            "war", "battle", "military", "army", "navy", "ww2",
            "ww1", "world war", "invasion", "occupation",
        },
        "reject_words": set(),
        "require_any": {
            "war", "battle", "army", "navy", "soldier", "weapon",
            "military", "general", "admiral", "veteran",
        },
    },
    # Farming / agriculture / rural lifestyle (catches Amish,
    # homesteading, etc).
    "agriculture": {
        "trigger_words": {
            "farm", "farming", "farmer", "agriculture", "amish",
            "rural", "homestead", "ranch", "harvest", "crop",
        },
        "reject_words": {
            "factory", "stock market", "wall street", "trading",
            "war", "prison", "military",
        },
        "require_any": {
            "farm", "farming", "farmer", "rural", "field", "barn",
            "horse", "plow", "harvest", "amish", "ranch", "soil",
            "crop", "agriculture", "homestead", "garden", "livestock",
        },
    },
}


def detect_intent_profile(brief_title: str, full_script: str) -> str | None:
    """Pick the intent profile whose trigger words appear most often
    in title+script. Returns None if no profile matches strongly."""
    text = f"{brief_title} {brief_title} {full_script}".lower()
    best, best_score = None, 0
    for name, prof in _INTENT_PROFILES.items():
        score = sum(text.count(t) for t in prof["trigger_words"])
        if score > best_score and score >= 2:
            best, best_score = name, score
    return best


def score_candidate(c: Candidate, scene_text: str,
                      query_blob: str = "") -> int:
    """Rule-based 0-100 score for an image candidate.
    Scoring stack:
      • +12 baseline (came from topic-matched search)
      • up to +45 keyword overlap (symmetric max)
      • up to +12 query echo
      • +6..+28 domain reputation
      • +12 high resolution (≥1600w)
      • +6 mid resolution (≥1200w)
      • -8 low resolution (<800w)
      • -8 portrait-only aspect (we want landscape for 16:9)
      • -15 watermark-hint words in title
      • +6 landscape aspect (good for 16:9 fill)"""
    title_words = _word_set(c.title)
    snippet_words = _word_set(c.snippet)
    scene_words = _word_set(scene_text)
    q_words = _word_set(query_blob or c.query)
    s = 12 if (c.query or q_words) else 0
    # Keyword overlap (symmetric max)
    cand_words = title_words | snippet_words
    if cand_words and scene_words:
        hits = len(cand_words & scene_words)
        ov = max(hits / max(1, len(scene_words)),
                 hits / max(1, len(cand_words)))
        s += int(min(45, ov * 100))
    # Query echo
    if q_words and title_words:
        q_ov = len(q_words & title_words) / max(1, len(q_words))
        s += int(min(12, q_ov * 18))
    # Domain reputation
    dom = c.source_site or _domain_of(c.image_url)
    rep = 0
    for d, b in _DOMAIN_BOOST.items():
        if d in dom:
            rep += b
            break
    if dom.endswith(".gov"): rep += 4
    if dom.endswith(".edu"): rep += 3
    s += rep
    # Resolution
    if c.width:
        if c.width >= 1600:                  s += 12
        elif c.width >= 1200:                s += 6
        elif c.width < 800:                  s -= 8
    if c.width and c.height:
        ar = c.width / c.height
        if 1.3 <= ar <= 2.2:                 s += 6     # landscape, 16:9-friendly
        elif ar < 0.8:                       s -= 8     # very portrait
    # Watermark hint in title
    title_low = (c.title or "").lower()
    if any(w in title_low for w in _WATERMARK_HINT_WORDS):
        s -= 15
    # Travel/hotel/booking page penalty — these are rarely topic-
    # specific even when they keyword-match. Soft (-10) so a genuine
    # niche-place photo on a regional travel page can still win if
    # its content match is strong, but hotel splashes won't dominate.
    if _is_travel_hotel(dom):
        s -= 10
    # Off-topic context penalty — strong (-25) so a contextually
    # wrong candidate (e.g. "Northern Lights PA", "coal country",
    # "Amish sexual abuse documentary") gets pushed below threshold
    # even when it shares a topic anchor word with the brief.
    snippet_low = (c.snippet or "").lower()
    blob_low = f"{title_low} {snippet_low}"
    for w in _NEGATIVE_CONTEXT_WORDS:
        if w in blob_low:
            s -= 25
            break
    # YouTube/video-thumbnail heavy penalty — these almost always
    # carry huge embedded clickbait text we can't strip. We don't
    # hard-reject (a real documentary's official poster is valid)
    # but push them below the typical pick threshold.
    if _is_video_thumbnail(c.image_url):
        s -= 25
    # Poster / blog-header / infographic title patterns — "Tips for",
    # "Healthy Aging Month", "Guide to", "Discover", "Best 10",
    # ".com" in title (visible URL = blog header). Strong -20.
    if _looks_like_poster_title(c.title):
        s -= 20
    return max(0, min(100, s))


# --------------------------------------------------------------------------- #
# Pacer — thread-safe slideshow-prevention gate
# --------------------------------------------------------------------------- #
class Pacer:
    """Gatekeeper that enforces the timeline-share cap and the
    max-consecutive-beats rule across parallel scene workers.

    Why it exists: fetch_footage runs _one() per (scene, beat) in a
    ThreadPoolExecutor. Without a shared, atomic check before each
    pick, the engine can either:
      (a) overshoot the target share (every parallel worker takes a
          web image because at-pick-time nobody else has yet), or
      (b) produce long consecutive runs of web images that read as
          a slideshow rather than a documentary.

    The pacer wraps two checks in one lock-protected call:
      • can_pick(scene_idx, total_scenes) → bool
      • commit(scene_idx) — caller invokes AFTER a successful download
        + canvas convert. If the candidate is rejected, no commit, so
        the budget is not consumed."""
    def __init__(self, cfg: WebImageConfig, total_scenes: int) -> None:
        self.cfg = cfg
        self.total = max(1, int(total_scenes))
        self.picks: set[int] = set()           # distinct scene indices that committed
        self.lock = threading.Lock()
        # cap = ceil(total * share), at least 1
        self.cap = max(1, int(round(self.total * cfg.max_timeline_share)))

    def can_pick(self, scene_idx: int) -> tuple[bool, str]:
        """Returns (allowed, reason_if_not). Reason is logged into the
        manifest so the audit trail shows WHY a pick was suppressed.

        Three gates, all must pass:
          (a) per-scene cap — at most ONE web image per scene; other
              beats fall through to stock/AI for visual variety
              (otherwise a 6-beat scene fills 6 web images and reads
              as a slideshow even with the timeline-share cap),
          (b) timeline-share cap — at most cap distinct scenes total,
          (c) consecutive-scenes cap — no run longer than N scenes."""
        with self.lock:
            if scene_idx in self.picks:
                return False, "scene_already_filled"
            if len(self.picks) >= self.cap:
                return False, (f"timeline_share_cap reached "
                                f"({len(self.picks)}/{self.cap})")
            n = max(1, int(self.cfg.max_consecutive_beats))
            window = list(range(scene_idx - n, scene_idx))
            if window and all(s in self.picks for s in window):
                return False, (f"consecutive_cap reached "
                                f"(prev {n} scenes are web-image)")
            return True, ""

    def commit(self, scene_idx: int) -> None:
        with self.lock:
            self.picks.add(scene_idx)

    def stats(self) -> dict:
        with self.lock:
            return {"committed": len(self.picks), "cap": self.cap,
                    "total_scenes": self.total,
                    "share": round(len(self.picks) / self.total, 3)}


# --------------------------------------------------------------------------- #
# Per-scene picker
# --------------------------------------------------------------------------- #
# Module-level lock guards both the pick+claim atomicity AND the
# decisions-list append. Without it, two parallel scene workers can
# both read used_urls (empty), both pick the same top candidate, and
# both commit — producing the duplicate-Brookings bug observed in
# the first A/B test.
_pick_lock = threading.Lock()


def pick_for_scene(scene_text: str, scene_idx: int,
                     pool: list[Candidate],
                     used_urls: set[str], used_hashes: set[str],
                     cfg: WebImageConfig,
                     decisions: list[dict] | None = None) -> Candidate | None:
    """Pick the best image for a scene and ATOMICALLY mark it used.
    eff = scene_s + min(15, brief_s - scene_s).

    The pick+claim happens under a module lock so parallel scene
    workers never both grab the same URL (closes the dedup race
    observed in the 2026-05-26 first A/B test where Brookings was
    selected for scenes 1 and 2 simultaneously)."""
    with _pick_lock:
        best: tuple[int, int, Candidate] | None = None
        for c in pool:
            if c.image_url in used_urls:
                continue
            if c.file_hash and c.file_hash in used_hashes:
                continue
            if c.rejection:
                continue
            scene_s = score_candidate(c, scene_text)
            brief_s = max(0, int(c.relevance_score or 0))
            eff = scene_s + min(15, max(0, brief_s - scene_s))
            if best is None or eff > best[0]:
                best = (eff, scene_s, c)
        if best is None:
            if decisions is not None:
                decisions.append({"scene": scene_idx, "selected": False,
                                   "reason": "empty_pool"})
            return None
        eff, scene_s, c = best
        if eff < cfg.min_score:
            if decisions is not None:
                decisions.append({
                    "scene": scene_idx, "selected": False,
                    "reason": f"below_min_score eff={eff} (scene={scene_s} "
                              f"brief={c.relevance_score}) min={cfg.min_score}",
                    "best_candidate": {
                        "image_url": c.image_url,
                        "source_site": c.source_site,
                        "title": (c.title or "")[:120],
                        "effective_score": eff,
                        "scene_score": scene_s,
                        "brief_score": int(c.relevance_score or 0),
                    },
                })
            return None
        # ATOMIC CLAIM — mark used immediately so no parallel worker
        # can re-pick this candidate.  Caller is responsible for
        # everything downstream (download, watermark, canvas convert).
        used_urls.add(c.image_url)
        c.relevance_score = eff
        c.selected_for_scene = scene_idx
        return c


# --------------------------------------------------------------------------- #
# Download + hash + dedupe
# --------------------------------------------------------------------------- #
_MAX_IMAGE_BYTES = 20 * 1024 * 1024     # 20 MB hard cap


def download_candidate(c: Candidate,
                        max_bytes: int = _MAX_IMAGE_BYTES,
                        timeout: int = 25) -> Path | None:
    """Download the image to local cache. Verifies it's a real image
    file (not HTML/text), respects size cap, computes md5 hash.

    Belt-and-suspenders block: refuses to download if the URL is a
    known video-thumbnail CDN or matches a thumbnail/cover/poster
    URL pattern. Discovery layer already filters these out, but this
    second check guarantees that no thumbnail ever lands on disk."""
    if _is_video_thumbnail(c.image_url):
        c.rejection = "rejected_youtube_thumbnail"
        _reject_mark(c.image_url, c.rejection)
        return None
    ext = ".jpg"
    u_low = c.image_url.lower().split("?", 1)[0]
    for e in _IMAGE_EXTS:
        if u_low.endswith(e):
            ext = ".jpg" if e == ".jpeg" else e
            break
    h = hashlib.sha256(c.image_url.encode()).hexdigest()[:24]
    cache_p = IMAGE_CACHE / f"{h}{ext}"
    if cache_p.exists() and cache_p.stat().st_size > 8_000:
        c.local_path = str(cache_p)
        c.downloaded = True
        try:
            with open(cache_p, "rb") as fh:
                c.file_hash = hashlib.md5(fh.read()).hexdigest()
        except Exception:
            pass
        return cache_p
    # Fresh download
    try:
        with _session.get(c.image_url, stream=True, timeout=timeout,
                            allow_redirects=True) as r:
            if r.status_code != 200:
                c.rejection = f"http {r.status_code}"
                _reject_mark(c.image_url, c.rejection)
                return None
            ct = (r.headers.get("content-type") or "").lower()
            if "image" not in ct:
                c.rejection = f"not an image (ct={ct[:30]})"
                _reject_mark(c.image_url, c.rejection)
                return None
            cl = int(r.headers.get("content-length") or 0)
            if cl > max_bytes:
                c.rejection = f"too large {cl}"
                _reject_mark(c.image_url, c.rejection)
                return None
            wrote = 0
            md5 = hashlib.md5()
            with open(cache_p, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    if not chunk:
                        continue
                    wrote += len(chunk)
                    if wrote > max_bytes:
                        c.rejection = "exceeded max_bytes during stream"
                        _reject_mark(c.image_url, c.rejection)
                        cache_p.unlink(missing_ok=True)
                        return None
                    md5.update(chunk)
                    fh.write(chunk)
            if wrote < 8_000:
                c.rejection = "tiny file (likely error page)"
                _reject_mark(c.image_url, c.rejection)
                cache_p.unlink(missing_ok=True)
                return None
            c.file_hash = md5.hexdigest()
    except Exception as e:                                  # noqa: BLE001
        c.rejection = f"download err {type(e).__name__}"
        _reject_mark(c.image_url, c.rejection)
        return None
    c.local_path = str(cache_p)
    c.downloaded = True
    return cache_p


# --------------------------------------------------------------------------- #
# Watermark / logo heuristic — runs on the downloaded image
# --------------------------------------------------------------------------- #
def detect_text_heavy(img_path: Path, strictness: str = "balanced") -> str:
    """Heuristic image-content text detector — rejects designed
    graphics that survived URL/title filtering.

    Real documentary photos have edge density distributed across the
    frame (faces, hands, foliage, textures).  Posters, blog headers,
    and infographics concentrate edge density in horizontal bands
    where the title text lives (top, center, bottom) while leaving
    the rest of the frame relatively smooth.  We detect that
    distribution.

    Specifically:
      1. Slice the image into 5 horizontal bands (top, upper-mid,
         centre, lower-mid, bottom).
      2. Run an edge filter on each.
      3. Compute mean edge density per band.
      4. If ANY band has very high density (>2× the median band)
         AND that band's density is over an absolute threshold,
         flag as text-heavy.
      5. Also check for the "logo strip" pattern — very high edge
         density in a top OR bottom 8% strip is a banner header.

    Empirically tuned against the v6 Japan picks (where 6/15 were
    poster/blog headers that should have been rejected).  Returns ""
    when clean, or a rejection reason string."""
    if strictness == "off":
        return ""
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return ""
    W, H = img.size
    if W < 400 or H < 300:
        return ""

    # 1) FIVE horizontal bands — text usually lives in one of them
    band_h = max(1, H // 5)
    band_edges: list[float] = []
    for i in range(5):
        y0 = i * band_h
        y1 = min(H, y0 + band_h)
        box = (0, y0, W, y1)
        try:
            crop = img.crop(box).filter(ImageFilter.FIND_EDGES).convert("L")
            band_edges.append(ImageStat.Stat(crop).mean[0])
        except Exception:
            band_edges.append(0.0)
    if not band_edges:
        return ""

    sorted_edges = sorted(band_edges)
    median = sorted_edges[len(sorted_edges) // 2]

    # WHOLE-IMAGE edge density — empirically (calibrated against the
    # v6 manifest 2026-05-26), real documentary photos sit at 13-25;
    # designed graphics / flat posters / blog headers with text on
    # smooth backgrounds sit at 2.5-8. Below the floor → graphic.
    try:
        whole_filt = img.filter(ImageFilter.FIND_EDGES).convert("L")
        whole_e = ImageStat.Stat(whole_filt).mean[0]
    except Exception:
        whole_e = 0.0

    # TOP/BOTTOM 12% STRIPS — banner / logo / phone-number bars
    # (the "+91 81795 08852" hospital ad pattern)
    strip_h = max(20, H // 8)
    top_e = bottom_e = 0.0
    try:
        top = img.crop((0, 0, W, strip_h)).filter(ImageFilter.FIND_EDGES).convert("L")
        top_e = ImageStat.Stat(top).mean[0]
        bot = img.crop((0, H - strip_h, W, H)).filter(ImageFilter.FIND_EDGES).convert("L")
        bottom_e = ImageStat.Stat(bot).mean[0]
    except Exception:
        pass

    # Thresholds calibrated against the v6 manifest (15 picks, mix of
    # real photos and blog headers).  Real photos pass cleanly:
    #   pexels=17.2  movingimage=20.3  rdsic=14.8  vecteezy=7.1*
    #   theplanetd=17.8  japanwondertravel=13.2
    # Designed posters fail:
    #   getdoc=4.8  agingresearch=4.1  freepik=4.4  lawire=3.6
    #   hospi=2.7  (flat backgrounds + text overlay)
    # *vecteezy Mt Fuji is borderline (sky + simple landscape) but
    # passes due to mountain edge — accept.
    cfg_thr = {
        "strict":   dict(flat_max=8.0,  strip_ratio=2.2),
        "balanced": dict(flat_max=6.5,  strip_ratio=2.5),
        "loose":    dict(flat_max=5.0,  strip_ratio=3.5),
    }
    thr = cfg_thr.get(strictness, cfg_thr["balanced"])

    # 0) WHITE-BACKGROUND COLLAGE / BLOG-SCREENSHOT detector — added
    # v8 to catch rdsic ("Healthy Japanese Food" article screenshot
    # with food collage on white) and similar layouts where the image
    # is dominated by very bright pixels with photos and text scattered.
    # Real documentary photos rarely have mean brightness > 195.
    try:
        mean_lum = ImageStat.Stat(img.convert("L")).mean[0]
    except Exception:
        mean_lum = 128.0
    if mean_lum > 195 and whole_e > 8:
        return (f"rejected_collage_screenshot mean_lum={mean_lum:.0f} "
                f"whole_edges={whole_e:.1f} (white-bg collage)")

    # 1) FLAT-BACKGROUND poster / graphic — total edge density below
    # the floor for real photos. This catches: agingresearch poster
    # (4.1), getdoc (4.8), freepik vector (4.4), lawire (3.6),
    # hospi silhouettes (2.7), the "EMBRACING HEALTHY AGING"
    # cartoon poster (low-detail flat design with text panels).
    if whole_e < thr["flat_max"]:
        return (f"rejected_designed_graphic whole_edges={whole_e:.1f} "
                f"(below {thr['flat_max']} = flat/posterish)")

    # 2) STRIP-BANNER detection — top or bottom 12% strip has 2.5×
    # the median band density (= text banner on photographic content).
    # Catches: discoveryeye "September is Healthy Aging Month" with
    # bottom-band text overlay on running-seniors photo.
    if median > 0:
        if top_e > 0 and top_e / max(1.0, median) > thr["strip_ratio"] and top_e > 12:
            return (f"rejected_text_strip top edges={top_e:.1f} "
                    f"ratio={top_e/median:.1f}x")
        if bottom_e > 0 and bottom_e / max(1.0, median) > thr["strip_ratio"] and bottom_e > 12:
            return (f"rejected_text_strip bottom edges={bottom_e:.1f} "
                    f"ratio={bottom_e/median:.1f}x")

    return ""


def detect_watermark(img_path: Path, strictness: str = "balanced") -> str:
    """Multi-region watermark heuristic.

    Catches three common stock-watermark patterns:
      1. **Corner stamps** — vendor logos in the four corners (legacy
         Pexels-like). Caught by edge-density on corner crops.
      2. **Centered diagonal text** — Alamy/Shutterstock-style large
         semi-transparent text running diagonally across the whole
         image. Caught by edge density in a center horizontal band +
         high inter-row variance (text has distinct rows of pixels).
      3. **Centered logo / brand strip** — Getty-style centered logo.
         Caught by elevated edge density in a small centered patch.

    We do NOT remove watermarks (licensing violation). When the
    confidence is high, we return a rejection reason so the caller
    drops the candidate and falls through to the next pick.

    'off' mode: never flags. 'loose' / 'balanced' / 'strict' tune the
    thresholds. Default is balanced."""
    if strictness == "off":
        return ""
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return ""
    W, H = img.size
    if W < 400 or H < 300:
        return ""

    # 1) CORNER STAMPS — four corner crops, edge-density mean
    cw, ch = max(80, W // 6), max(60, H // 6)
    corners = [
        (0, 0, cw, ch),
        (W - cw, 0, W, ch),
        (0, H - ch, cw, H),
        (W - cw, H - ch, W, H),
    ]
    corner_e = []
    for box in corners:
        try:
            crop = img.crop(box).filter(ImageFilter.FIND_EDGES)
            corner_e.append(ImageStat.Stat(crop.convert("L")).mean[0])
        except Exception:
            corner_e.append(0.0)
    max_corner = max(corner_e) if corner_e else 0.0

    # 2) CENTERED HORIZONTAL BAND — catches diagonal stock text
    #    (Alamy/Shutterstock big "PREVIEW" overlays span the center).
    #    A text strip has high LOCAL EDGE DENSITY + significant ROW-TO-
    #    ROW VARIANCE (consecutive text lines vs the smooth photo
    #    background underneath).
    band_h = max(80, H // 6)
    band_box = (0, (H - band_h) // 2, W, (H + band_h) // 2)
    band_edge = 0.0
    band_row_var = 0.0
    try:
        band = img.crop(band_box).filter(ImageFilter.FIND_EDGES).convert("L")
        band_edge = ImageStat.Stat(band).mean[0]
        # Row-variance: compute mean brightness per row, then variance
        try:
            import numpy as _np
            arr = _np.asarray(band, dtype=_np.float32)
            row_means = arr.mean(axis=1)
            band_row_var = float(row_means.var())
        except Exception:
            band_row_var = 0.0
    except Exception:
        pass

    # 3) CENTERED LOGO PATCH — small box dead-center.
    pw, ph = max(80, W // 7), max(60, H // 7)
    cx, cy = W // 2, H // 2
    centre_box = (cx - pw // 2, cy - ph // 2, cx + pw // 2, cy + ph // 2)
    centre_e = 0.0
    try:
        ctr = img.crop(centre_box).filter(ImageFilter.FIND_EDGES)
        centre_e = ImageStat.Stat(ctr.convert("L")).mean[0]
    except Exception:
        pass

    # Thresholds tuned against the v2 audit (where Alamy centered
    # watermarks were missed). 'balanced' chosen to keep false-
    # positive rate low on clean photos while catching obvious
    # stock overlays.
    cfg_thr = {
        "strict":   dict(corner=22.0, band=38.0, band_row=120.0, centre=30.0),
        "balanced": dict(corner=32.0, band=48.0, band_row=180.0, centre=42.0),
        "loose":    dict(corner=50.0, band=70.0, band_row=300.0, centre=65.0),
    }
    thr = cfg_thr.get(strictness, cfg_thr["balanced"])

    if max_corner > thr["corner"]:
        return f"watermark_corner edges={max_corner:.1f}"
    # Center diagonal text needs BOTH high edge density AND high
    # row-variance — together they're hard for natural photographs
    # to produce, but trivially produced by text overlays.
    if band_edge > thr["band"] and band_row_var > thr["band_row"]:
        return (f"watermark_center_text band_e={band_edge:.1f} "
                f"row_var={band_row_var:.0f}")
    if centre_e > thr["centre"]:
        return f"watermark_centre_logo edges={centre_e:.1f}"
    return ""


# --------------------------------------------------------------------------- #
# Manifest output
# --------------------------------------------------------------------------- #
def save_manifest(run_dir: Path, pool: list[Candidate],
                    decisions: list[dict],
                    budget: _Budget | None = None) -> Path:
    """Emit web_image_manifest.json + web_image_candidates.json sidecar."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pool_p = run_dir / "web_image_candidates.json"
    mani_p = run_dir / "web_image_manifest.json"
    try:
        pool_p.write_text(
            json.dumps([c.to_dict() for c in pool], indent=2,
                        ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    try:
        mani_p.write_text(
            json.dumps({
                "decisions": decisions,
                "budget": (budget.snapshot() if budget else {}),
                "cache_dirs": {
                    "search": str(SEARCH_CACHE),
                    "image": str(IMAGE_CACHE),
                    "rejects": str(REJECT_CACHE),
                },
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    return mani_p
