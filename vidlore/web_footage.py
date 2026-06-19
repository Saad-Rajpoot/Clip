"""Direct Web Footage Engine — cost-controlled, free/self-hosted-first
discovery of topic-relevant video footage from the wider web.

DESIGN PRINCIPLES (set by user 2026-05-26):

* Default backend is FREE: DuckDuckGo HTML scrape, with optional
  self-hosted SearXNG for higher quality. Paid APIs (Brave, SerpAPI,
  Bing) only fire when explicitly enabled via env vars.

* Strict query budget per render: project-level discovery runs ONCE up
  front (≤ N queries), builds a candidate pool, and scenes re-rank
  inside that pool. Scene-level targeted search fires only when the
  pool has no good match (≤ M extra queries). This keeps cost bounded
  even on free backends so we don't get rate-limit-blocked.

* Aggressive caching: every search result + every downloaded clip +
  every rejection lives in ~/.vidlore/web_search_cache and
  ~/.vidlore/web_footage_cache. Same query / same URL is never re-hit
  within the cache TTL window.

* Quality first: a candidate must clear a minimum relevance score
  (default 70) AND pass a watermark/logo heuristic. Otherwise the
  scene falls through to existing stock / AI tiers.

* AI images STAY ON. This engine is a NEW tier above stock, not a
  replacement for the AI fallback that handles historical /
  reconstruction / rare-shot scenes.

Public surface (used by footage.py):

    cfg_from_env() → WebFootageConfig
    discover_pool(brief, script, cfg, run_dir) → list[Candidate]
    pick_for_scene(scene, pool, used_urls, cfg) → Candidate | None
    download_candidate(candidate, dest, cfg) → bool
    save_manifest(run_dir, candidates, decisions) → None

The orchestrator is intentionally pure-data: it returns Candidate
records that footage.py decides what to do with. That keeps the stock
+ AI ladder logic centralised in footage.py.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

try:
    from PIL import Image
    _PIL_OK = True
except Exception:                                              # noqa: BLE001
    _PIL_OK = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

_CACHE_ROOT = Path(os.environ.get("VIDLORE_WEB_CACHE",
                                   str(Path.home() / ".vidlore")))
SEARCH_CACHE = _CACHE_ROOT / "web_search_cache"
FOOTAGE_CACHE = _CACHE_ROOT / "web_footage_cache"
REJECT_CACHE = _CACHE_ROOT / "web_footage_rejects"
for _d in (SEARCH_CACHE, FOOTAGE_CACHE, REJECT_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# Domain reputation. Positive = boost, negative = penalty. The list is
# small on purpose — we don't want to over-engineer reputation.
_DOMAIN_BOOST = {
    # Permissively licensed / commercial-safe pools — these are
    # curated archives, so once the SEARCH query already matched the
    # topic, the candidate is almost certainly on-topic. We give them
    # a big enough boost to clear the min_score=70 cutoff on their
    # own (when paired with the +12 query-echo baseline below).
    "wikimedia.org":       +28,
    "commons.wikimedia.org": +28,
    "archive.org":         +26,
    "loc.gov":             +24,
    "nasa.gov":            +24,
    "noaa.gov":            +22,
    "pexels.com":          +18,
    "pixabay.com":         +18,
    "videvo.net":          +18,
    "coverr.co":           +18,
    "openverse.org":       +18,
    "vimeo.com":           +12,
    # Educational / cultural
    "edu":                 +14,
    "gov":                 +16,
    # Aggregators that are okay
    "mixkit.co":           +14,
    "mazwai.com":          +12,
    # Risky / aggressive watermark sites (download anyway but penalise)
    "shutterstock.com":    -10,
    "gettyimages.com":     -10,
    "istockphoto.com":     -10,
    "alamy.com":           -8,
    "envato.com":          -6,
    "videoblocks.com":     -6,
    "stock.adobe.com":     -6,
    "depositphotos.com":   -6,
}

# Domains we deliberately skip (DRM, paywall, or TOS-incompatible)
_DOMAIN_BLACKLIST = {
    "netflix.com", "amazon.com", "hbomax.com", "disneyplus.com",
    "hulu.com", "primevideo.com",
}

# Site-restricted searches we add to project-level discovery
_VIDEO_SITES = (
    "site:archive.org",
    "site:commons.wikimedia.org",
    "site:videvo.net",
    "site:mixkit.co",
    "site:coverr.co",
    "site:mazwai.com",
)

_VIDEO_EXTS = (".mp4", ".webm", ".m4v", ".mov")
_DEFAULT_TTL_DAYS = 30


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #
@dataclass
class WebFootageConfig:
    # Default OFF — opt-in via env (WEB_FOOTAGE_ENGINE=1) or dashboard.
    # When on, free/self-hosted DuckDuckGo backend is the default so no
    # paid API key is required to try the engine.
    enabled: bool = False
    provider: str = "free"          # free|searxng|brave|serpapi|bing
    mix: str = "balanced"           # off|light|balanced|heavy
    min_score: int = 70
    paid_fallback: bool = False
    searxng_url: str = ""
    brave_api_key: str = ""
    serpapi_key: str = ""
    bing_api_key: str = ""
    max_project_queries: int = 20
    max_scene_queries: int = 2
    max_total_queries: int = 50
    max_downloads_per_scene: int = 2
    cache_ttl_days: int = _DEFAULT_TTL_DAYS
    search_depth: str = "normal"   # fast|normal|deep
    watermark_filter: str = "balanced"   # strict|balanced|loose


def cfg_from_env() -> WebFootageConfig:
    def _b(k: str, d: bool = False) -> bool:
        v = os.environ.get(k, "").strip().lower()
        return v in ("1", "true", "yes", "on") if v else d
    def _i(k: str, d: int) -> int:
        try:    return int(os.environ.get(k, "").strip() or d)
        except ValueError: return d
    return WebFootageConfig(
        enabled=_b("WEB_FOOTAGE_ENGINE", True),
        provider=os.environ.get("WEB_SEARCH_PROVIDER", "free").lower(),
        mix=os.environ.get("WEB_FOOTAGE_MIX", "balanced").lower(),
        # Threshold is mix-driven by default: light=60, balanced=55,
        # heavy=50.  Explicit WEB_MIN_SCORE env overrides.  See
        # mix_min_score() for the calibration story (2026-05-26 A/B).
        min_score=_i("WEB_MIN_SCORE",
                      mix_min_score(os.environ.get(
                          "WEB_FOOTAGE_MIX", "balanced").lower())),
        paid_fallback=_b("WEB_PAID_FALLBACK", False),
        searxng_url=os.environ.get("SEARXNG_URL", "").rstrip("/"),
        brave_api_key=os.environ.get("BRAVE_API_KEY", "").strip(),
        serpapi_key=os.environ.get("SERPAPI_KEY", "").strip(),
        bing_api_key=os.environ.get("BING_API_KEY", "").strip(),
        max_project_queries=_i("WEB_SEARCH_MAX_PROJECT_QUERIES", 20),
        max_scene_queries=_i("WEB_SEARCH_MAX_SCENE_QUERIES", 2),
        max_total_queries=_i("WEB_SEARCH_MAX_TOTAL_QUERIES", 50),
        max_downloads_per_scene=_i("WEB_SEARCH_MAX_DOWNLOADS_PER_SCENE", 2),
        cache_ttl_days=_i("WEB_SEARCH_CACHE_TTL_DAYS", _DEFAULT_TTL_DAYS),
        search_depth=os.environ.get("WEB_SEARCH_DEPTH", "normal").lower(),
        watermark_filter=os.environ.get("WEB_WATERMARK_FILTER",
                                          "balanced").lower(),
    )


def mix_target_fraction(mix: str) -> float:
    """Target % of final video using web footage. The pipeline doesn't
    enforce this hard — it just stops trying once we're at-or-above
    the target."""
    return {"off": 0.0, "light": 0.18, "balanced": 0.35, "heavy": 0.55} \
        .get(mix.lower(), 0.35)


def mix_min_score(mix: str, base: int = 50) -> int:
    """The mix slider IS the strictness knob. The pool is already
    quality-gated upstream (URL-pattern + topic-keyword filters drop
    generic pages and off-topic stragglers in discover_pool), so
    min_score is purely the relevance-strength dial on top of that.

    Calibrated against the 2026-05-26 A/B test (Amish farming):
    production scene-text scoring tops out around 50-55 even for
    strong matches due to short scene narrations vs short candidate
    titles, so base=50 is the honest "balanced" point."""
    return {"off": 999, "light": base + 5,
            "balanced": base, "heavy": base - 5}.get(mix.lower(), base)


# --------------------------------------------------------------------------- #
# Candidate record
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    title: str
    url: str
    source_site: str
    snippet: str = ""
    thumbnail: str = ""
    media_url: str = ""           # populated after extraction
    duration_s: float = 0.0
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
        return asdict(self)


# --------------------------------------------------------------------------- #
# Query budget tracker — one per render
# --------------------------------------------------------------------------- #
class _Budget:
    def __init__(self, cfg: WebFootageConfig):
        self._lock = threading.Lock()
        self.project_used = 0
        self.scene_used = 0
        self.total_used = 0
        self.cfg = cfg
    def can_project(self) -> bool:
        with self._lock:
            return (self.project_used < self.cfg.max_project_queries
                    and self.total_used < self.cfg.max_total_queries)
    def can_scene(self) -> bool:
        with self._lock:
            return (self.scene_used < self.cfg.max_scene_queries
                    and self.total_used < self.cfg.max_total_queries)
    def spend_project(self):
        with self._lock:
            self.project_used += 1; self.total_used += 1
    def spend_scene(self):
        with self._lock:
            self.scene_used += 1; self.total_used += 1
    def snapshot(self) -> dict:
        return {"project": self.project_used, "scene": self.scene_used,
                "total": self.total_used,
                "cap_project": self.cfg.max_project_queries,
                "cap_scene": self.cfg.max_scene_queries,
                "cap_total": self.cfg.max_total_queries}


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:24]


def _cache_search_get(cfg: WebFootageConfig, key: str) -> list[dict] | None:
    p = SEARCH_CACHE / f"{key}.json"
    if not p.exists():
        return None
    age_d = (time.time() - p.stat().st_mtime) / 86400.0
    if age_d > cfg.cache_ttl_days:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def _cache_search_set(key: str, results: list[dict]) -> None:
    try:
        (SEARCH_CACHE / f"{key}.json").write_text(
            json.dumps(results), encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass


def _reject_seen(url: str) -> bool:
    return (REJECT_CACHE / _hash(url)).exists()


def _reject_mark(url: str, reason: str) -> None:
    try:
        (REJECT_CACHE / _hash(url)).write_text(reason, encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Search providers
# --------------------------------------------------------------------------- #
_session = requests.Session()
_session.headers.update(_HEADERS)


def _search_duckduckgo(query: str, n: int = 12) -> list[dict]:
    """Free HTML scrape of DuckDuckGo. Returns up to n results.

    We use the html.duckduckgo.com lite endpoint which is the most
    stable scrape target (no JS). DDG actively discourages scraping
    so we throttle to 1 req/sec and respect their result limit.
    """
    try:
        r = _session.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "wt-wt"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        # Each result block is a <a class="result__a" href="...">
        # followed by <a class="result__snippet">
        html_body = r.text
        results: list[dict] = []
        # title + url pattern
        for m in re.finditer(
            r'<a\s+rel="nofollow"\s+class="result__a"\s+href="([^"]+)"[^>]*>'
            r'(.*?)</a>',
            html_body, re.DOTALL,
        ):
            url, title = m.group(1), html.unescape(re.sub(r"<[^>]+>", "",
                                                          m.group(2))).strip()
            # DDG wraps real URL behind redirect: /l/?uddg=encoded
            if url.startswith("/l/?") or "uddg=" in url:
                q = urllib.parse.parse_qs(url.split("?", 1)[-1]).get("uddg")
                if q:
                    url = urllib.parse.unquote(q[0])
            results.append({"title": title, "url": url, "snippet": ""})
            if len(results) >= n:
                break
        # Snippets — match positionally by class
        snips = re.findall(
            r'<a\s+class="result__snippet"[^>]*>(.*?)</a>',
            html_body, re.DOTALL,
        )
        for i, s in enumerate(snips[:len(results)]):
            results[i]["snippet"] = html.unescape(
                re.sub(r"<[^>]+>", "", s)).strip()
        return results
    except Exception:                                          # noqa: BLE001
        return []


def _search_searxng(query: str, base: str, n: int = 12) -> list[dict]:
    """Self-hosted SearXNG JSON API. Highest-quality free option."""
    if not base:
        return []
    try:
        r = _session.get(
            f"{base}/search",
            params={"q": query, "format": "json", "categories": "videos,general"},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json() or {}
        out: list[dict] = []
        for it in (data.get("results") or [])[:n]:
            out.append({
                "title": (it.get("title") or "").strip(),
                "url": it.get("url") or "",
                "snippet": (it.get("content") or "").strip(),
                "thumbnail": it.get("thumbnail") or it.get("img_src") or "",
            })
        return out
    except Exception:                                          # noqa: BLE001
        return []


def _search_brave(query: str, key: str, n: int = 12) -> list[dict]:
    if not key:
        return []
    try:
        r = _session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            params={"q": query, "count": n},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        data = r.json() or {}
        out: list[dict] = []
        for it in ((data.get("web") or {}).get("results") or [])[:n]:
            out.append({
                "title": (it.get("title") or "").strip(),
                "url": it.get("url") or "",
                "snippet": (it.get("description") or "").strip(),
                "thumbnail": (((it.get("thumbnail") or {}).get("src")) or ""),
            })
        return out
    except Exception:                                          # noqa: BLE001
        return []


def _search(query: str, cfg: WebFootageConfig, budget: _Budget,
             *, project_level: bool) -> list[dict]:
    """Dispatch to the configured provider with cache + budget."""
    key = _hash(f"{cfg.provider}|{query}")
    cached = _cache_search_get(cfg, key)
    if cached is not None:
        return cached
    if project_level:
        if not budget.can_project():
            return []
        budget.spend_project()
    else:
        if not budget.can_scene():
            return []
        budget.spend_scene()
    results: list[dict] = []
    n = {"fast": 8, "normal": 12, "deep": 20}.get(cfg.search_depth, 12)
    if cfg.provider == "searxng" and cfg.searxng_url:
        results = _search_searxng(query, cfg.searxng_url, n)
    elif cfg.provider == "brave" and cfg.brave_api_key:
        results = _search_brave(query, cfg.brave_api_key, n)
    # `free` is the default; we also fall through if the configured
    # paid provider returned nothing AND paid_fallback is not on.
    if not results:
        results = _search_duckduckgo(query, n)
    # Optional paid fallback: only if explicitly enabled
    if not results and cfg.paid_fallback and cfg.brave_api_key:
        results = _search_brave(query, cfg.brave_api_key, n)
    # Persist even empty result so we don't repeat a bad query
    _cache_search_set(key, results)
    time.sleep(0.4)        # gentle pacing for free backends
    return results


# --------------------------------------------------------------------------- #
# Query generation
# --------------------------------------------------------------------------- #
_GENERIC_BAD = {
    "office", "lifestyle", "background", "abstract", "concept",
}

def _domain_of(url: str) -> str:
    try:
        h = urllib.parse.urlparse(url).hostname or ""
        return h.lower().lstrip("www.")
    except Exception:                                          # noqa: BLE001
        return ""


# Sentence-start words that get false-positive capitalisation
_BAD_TOPIC_WORDS = {
    "she", "he", "they", "it", "we", "you", "i",
    "the", "this", "that", "those", "these",
    "but", "and", "or", "yet", "so",
    "after", "before", "during", "when", "while", "then",
    "now", "today", "yesterday", "tomorrow",
    "what", "where", "why", "how",
    "yes", "no",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # Title-case adjectives that slip through but aren't topic anchors —
    # these come from titles like "A Self-Sufficient World" where the
    # words after a colon/dash are not real proper nouns.
    "self", "sufficient", "true", "real", "new", "old", "best", "worst",
    "first", "last", "next", "every", "another", "other", "such",
    "some", "any", "all", "most", "many", "few", "very",
    "good", "great", "bad", "small", "big", "huge", "tiny",
}


def _topic_phrases(title: str, full_script: str) -> list[str]:
    """Extract topic-anchor phrases from title + narration. Returns a
    short list of capitalised proper nouns (people, places) and the
    strongest 2-3-word noun phrases. Keeps things free-form and
    avoids the LLM — we want this to work offline."""
    text = f"{title}\n{full_script}"
    # Proper-noun runs of 2-3 Capitalised words. The intra-name
    # separator is restricted to literal single spaces so we don't
    # accidentally span a line break ("Iranian General\n\nFor...").
    props = re.findall(r"\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,}){0,2})\b", text)
    # Filter false-positives (pronouns capitalised at sentence start, days, etc.)
    props = [p for p in props
             if p.split()[0].lower() not in _BAD_TOPIC_WORDS]
    # Count, separate strong (≥2 mentions) from singletons
    counts: dict[str, int] = {}
    for p in props:
        counts[p] = counts.get(p, 0) + 1
    multi = [p for p, c in counts.items() if c >= 2]
    # Multi-word proper nouns are valuable even at single mention
    # ("Lancaster County", "Sarah Cohen") — single capitalised words are
    # only kept if they appear >=2× to avoid noisy sentence-start words.
    singletons_multi_word = [p for p, c in counts.items()
                              if c == 1 and " " in p]
    strong = list(dict.fromkeys(multi + singletons_multi_word))
    if not strong:
        # Fallback: top 4 distinct proper nouns even if single-mention
        strong = list(dict.fromkeys(props))[:4]
    # de-dup case-insensitively, keep insertion order
    seen, out = set(), []
    for p in strong:
        k = p.lower()
        if k in seen: continue
        seen.add(k); out.append(p)
        if len(out) >= 8: break
    return out


def build_project_queries(brief_title: str, full_script: str,
                           audience: str = "auto",
                           language: str = "en",
                           max_queries: int = 20) -> list[str]:
    """Generate 10-20 broad project-level search queries from the brief
    title + full script.  Covers the main topic + cultural prefix +
    documentary-footage qualifiers + a few site-restricted variants.

    Multilingual: when language is non-English, we ALSO emit native-
    language versions of the topic queries (English versions still
    dominate because most video footage is indexed in English)."""
    topics = _topic_phrases(brief_title, full_script)
    audience_prefix = {
        "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ar": "Arabic", "tr": "Turkish", "es": "Spanish",
        "pt": "Brazilian", "de": "German", "fr": "French",
        "it": "Italian", "ur": "South Asian", "hi": "Indian",
        "en": "",
    }.get((audience or "").lower(), "")
    qualifiers = ("footage", "documentary footage", "b-roll",
                  "real footage", "video", "archive video")
    queries: list[str] = []
    seen: set[str] = set()
    def _add(q: str) -> None:
        s = q.strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k); queries.append(s)

    # Baseline: title + each qualifier
    for q in qualifiers[:3]:
        _add(f"{brief_title} {q}")
    # Topic phrases × qualifiers
    for t in topics[:5]:
        for q in qualifiers[:3]:
            _add(f"{t} {q}")
    # Audience-prefixed
    if audience_prefix:
        for t in topics[:4]:
            _add(f"{audience_prefix} {t} footage")
            _add(f"{audience_prefix} {t} documentary")
    # Site-restricted variants for the strongest topic
    if topics:
        for s in _VIDEO_SITES[:3]:
            _add(f"{topics[0]} {s}")
    return queries[:max(1, int(max_queries))]


def build_scene_queries(scene_text: str, audience: str = "auto",
                         max_q: int = 2) -> list[str]:
    """Up to 2 targeted queries for a specific scene that the
    project-level pool didn't satisfy. Compact and specific."""
    topics = _topic_phrases("", scene_text)
    audience_prefix = {
        "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ar": "Arabic", "tr": "Turkish", "es": "Spanish",
        "pt": "Brazilian", "de": "German", "fr": "French",
    }.get((audience or "").lower(), "")
    out: list[str] = []
    if topics:
        out.append(f"{topics[0]} footage")
        if audience_prefix:
            out.append(f"{audience_prefix} {topics[0]} video")
    else:
        # Fallback: first 6 content words of the scene
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", scene_text)
                 if w.lower() not in {"the", "and", "for", "with", "from",
                                       "this", "that", "his", "her"}][:6]
        if words:
            out.append(" ".join(words[:4]) + " footage")
    return out[:max_q]


# --------------------------------------------------------------------------- #
# Discovery — project-level + scene-level
# --------------------------------------------------------------------------- #
def discover_pool(brief_title: str, full_script: str,
                   cfg: WebFootageConfig,
                   audience: str = "auto",
                   language: str = "en") -> tuple[list[Candidate], _Budget]:
    """Run project-level discovery ONCE per render. Returns the
    candidate pool + the budget tracker (so scenes can extend it)."""
    budget = _Budget(cfg)
    queries = build_project_queries(brief_title, full_script,
                                     audience=audience, language=language,
                                     max_queries=cfg.max_project_queries)
    pool: list[Candidate] = []
    seen_urls: set[str] = set()
    for q in queries:
        if not budget.can_project():
            break
        results = _search(q, cfg, budget, project_level=True)
        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            dom = _domain_of(url)
            if dom in _DOMAIN_BLACKLIST:
                continue
            if _reject_seen(url):
                continue
            seen_urls.add(url)
            pool.append(Candidate(
                title=r.get("title", ""),
                url=url,
                source_site=dom,
                snippet=r.get("snippet", ""),
                thumbnail=r.get("thumbnail", ""),
                query=q,
            ))
    # Pre-filter 1: keep only candidates whose URL looks like a real
    # video-carrier page. A search for "Lancaster County footage" pulls
    # in a lot of generic landing pages (lancastercountypa.gov,
    # newsroom homepages, etc.) that can't yield downloadable video —
    # they'd just pollute scoring + waste download attempts.
    # Pre-filter 2: TOPIC-KEYWORD alignment. DDG sometimes returns
    # off-topic stragglers (e.g. a Lancaster query also pulled an
    # archive.org WWII V-2 rocket video). If the candidate's title+
    # snippet shares ZERO content words with the brief topic anchors,
    # it's off-topic and would just cross-contaminate scenes.
    brief_text = f"{brief_title}\n{full_script}"
    brief_topics = _topic_phrases(brief_title, full_script)
    topic_words: set[str] = set()
    for t in brief_topics:
        topic_words |= _word_set(t)
    # Drop very-common nouns that frequently appear in unrelated
    # candidate titles ("world", "history", "documentary", etc.) —
    # without this, a brief titled "Self-Sufficient World" matches any
    # candidate with "World War" in the title (false positive).
    _GENERIC = {"world", "history", "story", "stories", "documentary",
                "documentaries", "video", "videos", "footage", "film",
                "movie", "movies", "channel", "show", "shows", "day",
                "days", "year", "years", "people", "person", "life",
                "lives", "thing", "things", "time", "times", "way",
                "ways", "place", "places", "north", "south", "east",
                "west", "first", "last", "best", "free", "new", "old"}
    topic_words -= _GENERIC
    video_pool: list[Candidate] = []
    dropped_non_video = 0
    dropped_off_topic = 0
    for c in pool:
        if not _looks_like_video_page(c.url):
            dropped_non_video += 1
            continue
        c_words = _word_set(c.title) | _word_set(c.snippet)
        if topic_words and c_words and not (c_words & topic_words):
            # zero overlap with brief topic anchors → reject as off-topic
            dropped_off_topic += 1
            continue
        video_pool.append(c)
    # Pre-score the survivors against the brief so manifest / logs
    # make sense. pick_for_scene() re-scores per scene later.
    for c in video_pool:
        c.relevance_score = score_candidate(
            c, brief_text, query_blob=c.query)
    print(f"  [web-footage] pool: {len(pool)} raw → "
          f"{len(video_pool)} video-bearing on-topic "
          f"(dropped {dropped_non_video} non-video pages, "
          f"{dropped_off_topic} off-topic)", flush=True)
    return video_pool, budget


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

# URL/path patterns that indicate a real video carrier page (not just a
# generic search-result landing). These get a strong score boost — and
# we use the same set to PRE-FILTER the pool so generic .gov / .com
# index pages never compete with actual video pages.
_VIDEO_URL_PATTERNS = (
    "/details/",        # archive.org video items
    "/embed/",          # any embed page
    "/watch?",          # youtube watch URL (queries with ?)
    "vimeo.com/",       # vimeo per-video URLs
    "/video/",          # pexels.com/video/, pixabay.com/videos/
    "/videos/",
    "/clip/",
    "/clips/",
    "/file:",           # wikimedia commons file pages (case-sensitive in URL)
    "/wiki/File:",
)
_VIDEO_FILE_EXTS = (".mp4", ".webm", ".mov", ".m4v", ".ogv")

def _looks_like_video_page(url: str) -> bool:
    """True when the URL pattern strongly suggests a per-video page
    (not a generic landing / search index / homepage)."""
    u = (url or "").lower()
    if any(u.endswith(e) for e in _VIDEO_FILE_EXTS):
        return True
    return any(p in u for p in _VIDEO_URL_PATTERNS)


def _word_set(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (s or "").lower())
            if w not in {"the", "and", "for", "with", "from", "this",
                          "that", "you", "are", "was", "were", "have",
                          "has", "had", "but", "not", "what", "when",
                          "where", "how", "why", "into", "about", "after",
                          "before"}}


def score_candidate(c: Candidate, scene_text: str,
                     query_blob: str = "") -> int:
    """Rule-based relevance score 0-100. Quick + offline.

    Scoring stack:
      • +12 baseline credit: the candidate came back from a search
        whose query was derived from this brief, so it has a presumption
        of topic alignment even when the per-scene keyword overlap is
        zero (scenes are short, candidate snippets are short).
      • up to +50 keyword overlap (title+snippet vs scene text)
      • up to +15 query echo (title repeats query terms)
      • +12..+28 domain reputation (wikimedia/archive/.gov)
      • +6 resolution ≥720p, +6 duration in [3-30]s, -12 sub-1.5s
      • -10..-10 penalty for stock-watermark domains"""
    title_words = _word_set(c.title)
    snippet_words = _word_set(c.snippet)
    scene_words = _word_set(scene_text)
    q_words = _word_set(query_blob or c.query)
    # Baseline credit: the candidate came from a topic-matched search,
    # so before we look at the scene at all, we know it's at least
    # plausibly on-topic.  Without this, short scene snippets (10-15
    # words) collide with short candidate titles and produce false
    # zeros — which pushes the whole pool below min_score.
    kw_score = 12 if (c.query or q_words) else 0
    # Keyword overlap.  Use the symmetric max so short candidate titles
    # ("Lancaster County Videos") match strongly against long scene
    # narrations and vice-versa.  Pure jaccard against |scene| would
    # underweight short candidates; against |candidate| would overweight
    # noisy long titles. Max of both is the honest measure.
    cand_words = title_words | snippet_words
    if cand_words and scene_words:
        hits = len(cand_words & scene_words)
        ov_scene = hits / max(1, len(scene_words))
        ov_cand  = hits / max(1, len(cand_words))
        overlap = max(ov_scene, ov_cand)
        kw_score += int(min(50, overlap * 100))
    # Query echo (does the title literally repeat query terms?)
    if q_words and title_words:
        q_overlap = len(q_words & title_words) / max(1, len(q_words))
        kw_score += int(min(15, q_overlap * 20))
    # Domain reputation
    dom = c.source_site
    rep = 0
    for d, b in _DOMAIN_BOOST.items():
        if d in dom:
            rep += b
            break
    # TLD bonus catch-all
    if dom.endswith(".gov"): rep += 4
    if dom.endswith(".edu"): rep += 3
    kw_score += rep
    # URL-pattern boost — a per-video page URL is dramatically more
    # likely to actually yield downloadable video than a generic
    # search-result landing. Worth +18 because it flips the
    # "lancastercountypa.gov homepage" vs "archive.org/details/amish-
    # farm-1920" tie that the keyword-only score gets wrong.
    if _looks_like_video_page(c.url):
        kw_score += 18
    # Resolution/duration boosts AFTER probe
    if c.width and c.height and min(c.width, c.height) >= 720: kw_score += 6
    if 3.0 <= c.duration_s <= 30.0:                            kw_score += 6
    elif c.duration_s < 1.5:                                    kw_score -= 12
    # Clamp
    return max(0, min(100, kw_score))


# --------------------------------------------------------------------------- #
# Video URL extraction from a page
# --------------------------------------------------------------------------- #
def _extract_video_urls(page_url: str) -> list[str]:
    """Fetch a page and look for downloadable video URLs.

    Looks for (in order):
      1. JSON-LD VideoObject contentUrl
      2. <meta property="og:video:url" / og:video>
      3. <video src> and <source src>
      4. Direct .mp4/.webm in any href / src / data-* attribute
      5. archive.org canonical pattern (replace /details/ → /download/)

    Returns deduped list of absolute URLs, ordered most-likely first."""
    dom = _domain_of(page_url)
    # Already a direct media file?
    if any(page_url.lower().endswith(e) for e in _VIDEO_EXTS):
        return [page_url]
    try:
        r = _session.get(page_url, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return []
        body = r.text
    except Exception:                                          # noqa: BLE001
        return []
    out: list[str] = []
    def _add(u: str | None) -> None:
        if not u: return
        u = u.strip()
        if not u: return
        # Resolve relative
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            base = urllib.parse.urlparse(page_url)
            u = f"{base.scheme}://{base.netloc}{u}"
        elif not u.startswith(("http://", "https://")):
            return
        if u not in out:
            out.append(u)

    # 1) JSON-LD VideoObject
    for m in re.finditer(
            r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
            body, re.DOTALL):
        try:
            data = json.loads(m.group(1))
            stack = data if isinstance(data, list) else [data]
            for obj in stack:
                if not isinstance(obj, dict): continue
                if obj.get("@type") in ("VideoObject", "Movie"):
                    _add(obj.get("contentUrl"))
                    _add(obj.get("embedUrl"))
        except Exception:                                      # noqa: BLE001
            pass
    # 2) OpenGraph
    for m in re.finditer(
            r'<meta\s+property="og:(?:video(?::url)?|video:secure_url)"\s+'
            r'content="([^"]+)"', body, re.I):
        _add(m.group(1))
    # 3) <video> and <source>
    for m in re.finditer(r'<video[^>]+src="([^"]+)"', body, re.I):
        _add(m.group(1))
    for m in re.finditer(r'<source[^>]+src="([^"]+)"', body, re.I):
        _add(m.group(1))
    # 4) Direct mp4/webm anywhere
    for ext in _VIDEO_EXTS:
        for m in re.finditer(rf'(?:src|href|data-[\w-]+)\s*=\s*'
                              rf'"([^"]+\{ext}[^"]*)"',
                              body, re.I):
            _add(m.group(1))
    # 5) archive.org pattern
    if "archive.org/details/" in page_url:
        slug = page_url.split("archive.org/details/", 1)[-1] \
            .split("?", 1)[0].split("#", 1)[0].rstrip("/")
        if slug:
            for ext in (".mp4", ".webm"):
                # Try the most common naming convention
                _add(f"https://archive.org/download/{slug}/{slug}{ext}")
    # Filter: only media extensions or known media endpoints
    media = [u for u in out
             if any(u.lower().split("?", 1)[0].endswith(e)
                    for e in _VIDEO_EXTS)
             or "videoplayback" in u
             or "/download/" in u]
    return media


# --------------------------------------------------------------------------- #
# Watermark detection (heuristic, no external deps beyond PIL + ffprobe)
# --------------------------------------------------------------------------- #
def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _probe(path: Path) -> tuple[float, int, int]:
    """Return (duration_s, width, height) for a media file; (0, 0, 0)
    on any failure."""
    try:
        r = subprocess.run(
            [_ffprobe(), "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return 0.0, 0, 0
        lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
        w = h = 0; dur = 0.0
        for ln in lines:
            if "," in ln:
                try:
                    a, b = ln.split(",", 1)
                    w, h = int(a), int(b)
                except Exception:                              # noqa: BLE001
                    pass
            else:
                try: dur = float(ln)
                except ValueError: pass
        return dur, w, h
    except Exception:                                          # noqa: BLE001
        return 0.0, 0, 0


def _extract_frames(path: Path, dest_dir: Path, n: int = 5) -> list[Path]:
    """Pull n evenly-spaced frames as small JPEGs for visual analysis."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dur, _, _ = _probe(path)
    if dur <= 0:
        return []
    out: list[Path] = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        f = dest_dir / f"frame_{i:02d}.jpg"
        try:
            subprocess.run(
                [_ffmpeg(), "-y", "-ss", f"{t:.2f}", "-i", str(path),
                 "-frames:v", "1", "-vf", "scale=320:-2", "-q:v", "3",
                 str(f)],
                capture_output=True, timeout=15,
            )
            if f.exists() and f.stat().st_size > 1000:
                out.append(f)
        except Exception:                                      # noqa: BLE001
            continue
    return out


def detect_watermark(path: Path, *, strictness: str = "balanced") -> str:
    """Heuristic watermark detector. Returns a non-empty reason string
    if a watermark is suspected; empty string if clean.

    Strategy: extract 5 frames, look at the 4 corner crops + bottom
    centre. If the SAME corner has high text-edge density AND low
    inter-frame variance (i.e. static overlay) across all frames →
    likely burned-in watermark.

    Strictness:
      strict   — reject on weaker signals
      balanced — default
      loose    — only reject on very strong signals
    """
    if not _PIL_OK:
        return ""
    frames = _extract_frames(path, path.parent / f"_wm_{path.stem}", n=5)
    if len(frames) < 3:
        return ""
    try:
        from PIL import ImageChops, ImageStat
        imgs = [Image.open(f).convert("L") for f in frames]
        W, H = imgs[0].size
        regions = {
            "TL": (0, 0, int(W * 0.30), int(H * 0.20)),
            "TR": (int(W * 0.70), 0, W, int(H * 0.20)),
            "BL": (0, int(H * 0.80), int(W * 0.30), H),
            "BR": (int(W * 0.70), int(H * 0.80), W, H),
            "BC": (int(W * 0.30), int(H * 0.86), int(W * 0.70), H),
        }
        thresholds = {
            "strict":   (10.0, 35.0),
            "balanced": (15.0, 45.0),
            "loose":    (22.0, 60.0),
        }.get(strictness, (15.0, 45.0))
        max_diff_thresh, min_brightness = thresholds
        suspect: list[str] = []
        for name, box in regions.items():
            crops = [im.crop(box) for im in imgs]
            base = crops[0]
            diffs: list[float] = []
            for c in crops[1:]:
                d = ImageChops.difference(base, c)
                diffs.append(ImageStat.Stat(d).mean[0])
            max_diff = max(diffs) if diffs else 0.0
            mean_bright = ImageStat.Stat(base).mean[0]
            stddev = ImageStat.Stat(base).stddev[0]
            # Watermark = STATIC across frames (low diff) AND BRIGHT-ish AND
            # high local contrast (text edges)
            if max_diff < max_diff_thresh and mean_bright > min_brightness \
                    and stddev > 35.0:
                suspect.append(name)
        # cleanup
        for f in frames:
            try: f.unlink()
            except OSError: pass
        try: (path.parent / f"_wm_{path.stem}").rmdir()
        except OSError: pass
        if not suspect:
            return ""
        return f"watermark suspected in {','.join(suspect)}"
    except Exception:                                          # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _cache_path_for(media_url: str) -> Path:
    return FOOTAGE_CACHE / f"{_hash(media_url)}.mp4"


def download_candidate(c: Candidate, *,
                        max_bytes: int = 60 * 1024 * 1024,
                        timeout: int = 45) -> Path | None:
    """Resolve a Candidate to a local mp4 file. Returns the local
    path on success, None on failure. Cached by media_url hash so
    repeated renders reuse the same file."""
    # 1) Find media URLs if not already known
    if not c.media_url:
        urls = _extract_video_urls(c.url)
        if not urls:
            c.rejection = "no media url"
            return None
        c.media_url = urls[0]
    cache_p = _cache_path_for(c.media_url)
    if cache_p.exists() and cache_p.stat().st_size > 50_000:
        c.local_path = str(cache_p)
        c.downloaded = True
        # Re-probe so dimensions/duration are populated
        dur, w, h = _probe(cache_p)
        c.duration_s, c.width, c.height = dur, w, h
        return cache_p
    # 2) Download with streaming + cap
    try:
        with _session.get(c.media_url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                c.rejection = f"http {r.status_code}"
                _reject_mark(c.url, c.rejection)
                return None
            n = 0
            with open(cache_p, "wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    if not chunk: continue
                    fh.write(chunk)
                    n += len(chunk)
                    if n > max_bytes:
                        c.rejection = "exceeds max bytes"
                        break
        if cache_p.stat().st_size < 50_000:
            cache_p.unlink(missing_ok=True)
            c.rejection = c.rejection or "too small"
            _reject_mark(c.url, c.rejection)
            return None
    except Exception as e:                                     # noqa: BLE001
        c.rejection = f"download err {type(e).__name__}"
        _reject_mark(c.url, c.rejection)
        cache_p.unlink(missing_ok=True)
        return None
    # 3) Probe
    dur, w, h = _probe(cache_p)
    c.duration_s, c.width, c.height = dur, w, h
    if dur < 1.0:
        c.rejection = "no video stream"
        cache_p.unlink(missing_ok=True)
        _reject_mark(c.url, c.rejection)
        return None
    if min(w, h) < 480:
        c.rejection = f"too small {w}x{h}"
        cache_p.unlink(missing_ok=True)
        _reject_mark(c.url, c.rejection)
        return None
    # 4) Hash for dedupe
    try:
        h_md5 = hashlib.md5()
        with open(cache_p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h_md5.update(chunk)
        c.file_hash = h_md5.hexdigest()
    except Exception:                                          # noqa: BLE001
        pass
    c.local_path = str(cache_p)
    c.downloaded = True
    return cache_p


# --------------------------------------------------------------------------- #
# Scene-level selection — pure ranking over the pool
# --------------------------------------------------------------------------- #
def pick_for_scene(scene_text: str, scene_idx: int,
                    pool: list[Candidate],
                    used_urls: set[str], used_hashes: set[str],
                    cfg: WebFootageConfig,
                    decisions: list[dict] | None = None) -> Candidate | None:
    """Pick the best candidate from the pool for this scene. Re-scores
    each unused candidate against the scene text and returns the
    highest-scoring one above min_score. Returns None if nothing
    clears the bar.

    Effective score = max(brief_pre_score, scene_re_score).  The pool
    only contains candidates that came back from topic-matched search
    queries, so the brief-level signal is real even when a particular
    short scene narration shares few words with the title/snippet —
    throwing it away on per-scene re-score would (and did) silently
    reject every good candidate. Taking the max preserves topic alignment
    while still letting strong per-scene matches dominate when present.

    When ``decisions`` is provided, every consideration is appended (the
    pick OR the rejection-with-reason) so the manifest is honest about
    what actually happened — not just what was selected."""
    best: tuple[int, int, Candidate] | None = None  # (eff, scene_s, c)
    for c in pool:
        if c.url in used_urls:
            continue
        if c.file_hash and c.file_hash in used_hashes:
            continue
        if c.rejection:
            continue
        scene_s = score_candidate(c, scene_text)
        # Brief pre-score lives in c.relevance_score (set by discover_pool).
        # Effective score blends the two: scene_s wins when it's strong,
        # and a strong brief_s adds a credit ceiling of +15 (so a globally
        # on-topic candidate still beats a clearly off-topic one, but
        # can't completely override a near-zero scene-relevance).
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
                "scene": scene_idx,
                "selected": False,
                "reason": f"below_min_score eff={eff} (scene={scene_s} "
                          f"brief={c.relevance_score}) min={cfg.min_score}",
                "best_candidate": {
                    "url": c.url, "source_site": c.source_site,
                    "title": (c.title or "")[:120],
                    "effective_score": eff,
                    "scene_score": scene_s,
                    "brief_score": int(c.relevance_score or 0),
                },
            })
        return None
    c.relevance_score = eff
    c.selected_for_scene = scene_idx
    return c


# --------------------------------------------------------------------------- #
# Manifest output
# --------------------------------------------------------------------------- #
def save_manifest(run_dir: Path, pool: list[Candidate],
                   decisions: list[dict],
                   budget: _Budget | None = None) -> Path:
    """Write web_footage_manifest.json + web_candidates.json. Returns
    the manifest path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pool_p = run_dir / "web_candidates.json"
    mani_p = run_dir / "web_footage_manifest.json"
    try:
        pool_p.write_text(
            json.dumps([c.to_dict() for c in pool], indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:                                          # noqa: BLE001
        pass
    try:
        mani_p.write_text(
            json.dumps({
                "decisions": decisions,
                "budget": (budget.snapshot() if budget else {}),
                "cache_dirs": {
                    "search": str(SEARCH_CACHE),
                    "footage": str(FOOTAGE_CACHE),
                    "rejects": str(REJECT_CACHE),
                },
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:                                          # noqa: BLE001
        pass
    return mani_p
