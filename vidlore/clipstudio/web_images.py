"""Internet image search — exact-scene fallback when no YouTube footage fits a beat.

A focused, self-contained port of the proven scraping core from the sibling `vidlore`
(vidrush) engine's web_images.py: Bing Images (the user's preferred backend — Wikipedia
alone is too sparse), DuckDuckGo, and Wikimedia Commons, behind one `search_images()`
dispatcher with a disk cache, plus a hardened `download_image()`.

No dependency on the vidrush package — copied + trimmed so ClipStudio stays standalone.
Env knobs:  VIDLORE_CLIPSTUDIO_IMG_BING=0   (disable Bing)
            VIDLORE_CLIPSTUDIO_IMG_DDG=0    (disable DuckDuckGo)
            VIDLORE_CLIPSTUDIO_IMG_WIKI=0   (disable Wikimedia)
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

_UA = {"User-Agent":
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"}

# Domain substrings we hard-skip — auth-walled, low-res repin farms, or watermarked stock
# (we never strip a watermark to "rescue" an image — licensing + it looks unprofessional).
_BLACKLIST = (
    "pinterest.", "pinimg.com", "facebook.com", "instagram.com", "tiktok.com",
    "twitter.com", "x.com", "t.co", "netflix.com", "amazon.com", "ebay.com",
)
_WATERMARK_STOCK = (
    "alamy.", "gettyimages.", "istockphoto.com", "shutterstock.com", "dreamstime.com",
    "depositphotos.com", "stock.adobe.com", "stockphotos.com", "123rf.com",
    "fotolia.com", "bigstockphoto.com",
)
# NON-PHOTOGRAPHIC sources — video-game art/key-art, concept/fan art, wallpapers, toy/figure
# shops. We want LIVE-ACTION scene stills, never illustrated/CGI game art (the user's core
# complaint). gotconquest.com etc. are mobile-game promo sites.
_NONPHOTO_DOMAINS = (
    "gotconquest.com", "conquest.", "artstation.com", "deviantart.com",
    "wallpaperaccess.com", "wallpapercave.com", "wallhaven.cc", "hdqwalls.com",
    "wallpaperflare.com", "peakpx.com", "wallpapersden.com", "getwallpapers.com",
    "store.steampowered.com", "ign.com", "gamespot.com", "polygon.com",
    "redbubble.com", "etsy.com", "ebay.", "aliexpress.", "wikia.nocookie.net",
    # AI image GENERATORS + clipart/aggregator farms — never real show stills:
    "craiyon.com", "animalia-life.club", "inspiredpencil.com", "openart.ai",
    "midjourney.com", "lexica.art", "civitai.com", "playground.com",
    "stablediffusionweb.com", "dreamstudio.ai", "nightcafe.studio",
)


def _cache_root() -> Path:
    d = Path(os.environ.get("VIDLORE_CLIPSTUDIO_IMG_CACHE",
                            str(Path.home() / ".cache" / "clipstudio" / "web_images")))
    (d / "search").mkdir(parents=True, exist_ok=True)
    (d / "img").mkdir(parents=True, exist_ok=True)
    return d


def _env_flag(name: str, default: bool = True) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _domain_hit(host: str, needle: str) -> bool:
    """Label-boundary-safe domain match — 'x.com'/'t.co' must NOT match inside
    'screenrant.com'. A trailing-dot needle ('pinterest.') is a prefix-label match."""
    host = (host or "").lower()
    n = needle.lower()
    if n.endswith("."):                              # 'pinterest.' → any label starting pinterest
        return any(lbl.startswith(n[:-1]) for lbl in host.split("."))
    return host == n or host.endswith("." + n)


def is_unusable_domain(url: str) -> bool:
    """Auth-walled / repin-farm / watermarked-stock / non-photographic (game-art, wallpaper,
    merch) domains we never download from."""
    d = _domain_of(url)
    return (any(_domain_hit(d, p) for p in _BLACKLIST)
            or any(_domain_hit(d, p) for p in _WATERMARK_STOCK)
            or any(_domain_hit(d, p) for p in _NONPHOTO_DOMAINS))


# Pure AI-image GENERATORS. Unlike a clipart aggregator (which often hot-links a REAL still off a
# CDN — e.g. animalia-life.club linking static.hbo.com), a page on one of these IS the generated
# artwork: the image is synthetic no matter what CDN serves it. So this is checked against the
# SOURCE (referrer) page, not just the image host — an AI image parked on a generic avatar CDN
# (avatars.mds.yandex.net) but sourced from shedevrum.ai must still be rejected.
_AI_GENERATOR_SOURCES = (
    "shedevrum.ai", "craiyon.com", "openart.ai", "midjourney.com", "lexica.art",
    "civitai.com", "playground.com", "playgroundai.com", "stablediffusionweb.com",
    "dreamstudio.ai", "nightcafe.studio", "leonardo.ai", "starryai.com", "deepai.org",
    "creator.nightcafe.studio", "tensor.art", "mage.space", "getimg.ai",
)


def is_ai_generated_source(source_site: str) -> bool:
    """True if the referring PAGE is a known AI image generator — its images are synthetic
    regardless of which CDN hosts the file. (Clipart aggregators are NOT here: they frequently
    hot-link genuine stills, so they're judged by image host + the photographic CLIP guard.)"""
    d = (source_site or "").lower().strip()
    if "://" in d or "/" in d:
        d = _domain_of(d if "://" in d else "http://" + d)
    return any(_domain_hit(d, p) for p in _AI_GENERATOR_SOURCES)


# --------------------------------------------------------------------------- #
# wall-clock-bounded GET (a scalar timeout can't bound a slow-trickle host)
# --------------------------------------------------------------------------- #
_session = None
_lock = threading.Lock()


def _sess():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers.update(_UA)
    return _session


class _GResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def _force_close(r):
    try:
        r.raw.close()
    except Exception:
        pass


def _guarded_get(url, *, timeout=20, max_bytes=8_000_000, **kw):
    _cap = float(os.environ.get("VIDLORE_DL_MAX_SECONDS", "60") or 60)
    kw.pop("stream", None)
    with _sess().get(url, stream=True, timeout=(min(8, int(timeout)), timeout), **kw) as r:
        killer = threading.Timer(_cap, _force_close, (r,))
        killer.daemon = True
        killer.start()
        try:
            raw = r.raw.read(max_bytes + 1, decode_content=True) or b""
        finally:
            killer.cancel()
        sc = r.status_code
    try:
        text = raw[:max_bytes].decode(r.encoding or "utf-8", "replace")
    except (LookupError, TypeError):
        text = raw[:max_bytes].decode("utf-8", "replace")
    return _GResp(sc, text)


# --------------------------------------------------------------------------- #
# search backends
# --------------------------------------------------------------------------- #
def _search_bing(query: str, n: int = 24) -> list[dict]:
    """Bing Images — each `.iusc` anchor embeds an HTML-escaped m="{...}" JSON with
    murl (full-res image), purl (hosting page), t (title)."""
    out: list[dict] = []
    try:
        r = _guarded_get("https://www.bing.com/images/search",
                         params={"q": query, "form": "HDRSC2", "first": "1"},
                         headers={**_UA, "Referer": "https://www.bing.com/"}, timeout=20)
        if r.status_code != 200:
            return out
        page = r.text or ""
    except Exception:
        return out
    seen: set[str] = set()
    for blob in re.findall(r'\bm="(\{[^"]+\})"', page):
        raw = html.unescape(blob)
        try:
            meta = json.loads(raw)
            murl = (meta.get("murl") or "").strip()
            purl = (meta.get("purl") or "").strip()
            title = (meta.get("t") or meta.get("desc") or "").strip()
        except Exception:
            m1 = re.search(r'"murl"\s*:\s*"([^"]+)"', raw)
            m2 = re.search(r'"purl"\s*:\s*"([^"]+)"', raw)
            m3 = re.search(r'"t"\s*:\s*"([^"]*)"', raw)
            murl = m1.group(1) if m1 else ""
            purl = m2.group(1) if m2 else ""
            title = m3.group(1) if m3 else ""
        if not murl or not murl.lower().startswith(("http://", "https://")) or murl in seen:
            continue
        seen.add(murl)
        out.append({"image_url": murl, "source_page": purl,
                    "source_site": _domain_of(purl) or _domain_of(murl),
                    "title": title, "width": 0, "height": 0})
        if len(out) >= n:
            break
    return out


def _search_ddg(query: str, n: int = 24) -> list[dict]:
    """DuckDuckGo images — page yields a vqd token, then /i.js returns JSON."""
    out: list[dict] = []
    try:
        r = _guarded_get("https://duckduckgo.com/",
                         params={"q": query, "iax": "images", "ia": "images"}, timeout=20)
        if r.status_code != 200:
            return out
        m = re.search(r"vqd=['\"]?(\d+-\d+(?:-\d+)?)['\"]?", r.text) or \
            re.search(r"vqd=([\d-]+)", r.text)
        if not m:
            return out
        time.sleep(0.4)
        r2 = _guarded_get("https://duckduckgo.com/i.js",
                          params={"l": "us-en", "o": "json", "q": query, "vqd": m.group(1),
                                  "f": ",,,", "p": "1", "v7exp": "a"},
                          headers={**_UA, "Referer": "https://duckduckgo.com/"}, timeout=25)
        if r2.status_code != 200:
            return out
        data = json.loads(r2.text)
        for item in (data.get("results") or [])[:n]:
            url = item.get("image") or ""
            if not url:
                continue
            out.append({"image_url": url, "source_page": item.get("url", ""),
                        "source_site": _domain_of(item.get("url", "")) or _domain_of(url),
                        "title": html.unescape(item.get("title") or ""),
                        "width": int(item.get("width") or 0),
                        "height": int(item.get("height") or 0)})
    except Exception:
        return out
    return out


def _search_wikimedia(query: str, n: int = 10) -> list[dict]:
    out: list[dict] = []
    try:
        r = _guarded_get("https://commons.wikimedia.org/w/api.php",
                         params={"action": "query", "format": "json", "generator": "search",
                                 "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": n,
                                 "prop": "imageinfo", "iiprop": "url|size|mime",
                                 "iiurlwidth": 1920}, timeout=20)
        if r.status_code != 200:
            return out
        pages = list((json.loads(r.text).get("query", {}) or {}).get("pages", {}).values())
    except Exception:
        return out
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if (ii.get("mime") or "").lower() not in ("image/jpeg", "image/png", "image/webp"):
            continue
        url = ii.get("thumburl") or ii.get("url") or ""
        if not url:
            continue
        out.append({"image_url": url, "source_page": ii.get("descriptionurl", "") or url,
                    "source_site": "commons.wikimedia.org",
                    "title": (p.get("title") or "").replace("File:", "").rsplit(".", 1)[0],
                    "width": int(ii.get("width") or 0), "height": int(ii.get("height") or 0)})
    return out


def search_images(query: str, n: int = 30, ttl_days: int = 14) -> list[dict]:
    """Bing + DDG + Wikimedia, deduped by image URL, disk-cached. Blacklisted /
    watermarked-stock domains are dropped here so they never reach scoring."""
    q = (query or "").strip()
    if not q:
        return []
    cache = _cache_root() / "search" / (hashlib.sha256(q.encode()).hexdigest()[:24] + ".json")
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            if time.time() - d.get("t", 0) < ttl_days * 86400:
                # Re-apply the domain blacklist to the CACHED list too. The blacklist can GROW
                # between runs (we keep adding AI-art / clipart farms as we spot them); a still-
                # warm 14-day cache written before a domain was banned must not resurrect it.
                return [it for it in d.get("r", [])
                        if not is_unusable_domain(it.get("image_url") or "")]
        except Exception:
            pass
    results: list[dict] = []
    with _lock:                                  # serialize SERP hits — be polite
        if _env_flag("VIDLORE_CLIPSTUDIO_IMG_BING", True):
            try:
                results += _search_bing(q)
            except Exception:
                pass
        if _env_flag("VIDLORE_CLIPSTUDIO_IMG_DDG", True):
            try:
                results += _search_ddg(q)
            except Exception:
                pass
        if _env_flag("VIDLORE_CLIPSTUDIO_IMG_WIKI", True):
            try:
                results += _search_wikimedia(q)
            except Exception:
                pass
    seen: set[str] = set()
    deduped = []
    for it in results:
        u = it.get("image_url") or ""
        if not u or u in seen or is_unusable_domain(u):
            continue
        seen.add(u)
        deduped.append(it)
    try:
        cache.write_text(json.dumps({"t": time.time(), "r": deduped}))
    except Exception:
        pass
    return deduped


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def download_image(url: str, dest: Path, *, max_bytes: int = 12_000_000,
                   timeout: int = 25, min_w: int = 480, min_h: int = 360) -> Optional[Path]:
    """Download to dest, verifying it is a real raster image of usable size. Returns the
    path (a normalized .jpg) or None. Cached by URL hash so repeats are free."""
    import requests
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    cache_p = _cache_root() / "img" / f"{h}.jpg"
    final = Path(dest)
    if cache_p.exists() and cache_p.stat().st_size > 8000:
        if _validate_dims(cache_p, min_w, min_h):
            _copy(cache_p, final)
            return final
        cache_p.unlink(missing_ok=True)
        return None
    _cap = float(os.environ.get("VIDLORE_DL_MAX_SECONDS", "60") or 60)
    tmp = cache_p.with_suffix(".raw")
    try:
        with _sess().get(url, stream=True, timeout=(min(8, timeout), timeout),
                         allow_redirects=True) as r:
            if r.status_code != 200:
                return None
            ct = (r.headers.get("content-type") or "").lower()
            if "image" not in ct:
                return None
            cl = int(r.headers.get("content-length") or 0)
            if cl and cl > max_bytes:
                return None
            killer = threading.Timer(_cap, _force_close, (r,))
            killer.daemon = True
            killer.start()
            try:
                data = r.raw.read(max_bytes + 1, decode_content=True) or b""
            finally:
                killer.cancel()
        if len(data) < 8000 or len(data) > max_bytes:
            return None
        tmp.write_bytes(data)
    except Exception:
        tmp.unlink(missing_ok=True)
        return None
    # transcode→jpg + validate it actually decodes and is big enough
    if not _transcode_jpg(tmp, cache_p):
        tmp.unlink(missing_ok=True)
        return None
    tmp.unlink(missing_ok=True)
    if not _validate_dims(cache_p, min_w, min_h):
        cache_p.unlink(missing_ok=True)
        return None
    _copy(cache_p, final)
    return final


def _copy(src: Path, dst: Path):
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(src) != str(dst):
        shutil.copyfile(src, dst)


def _transcode_jpg(src: Path, dst: Path) -> bool:
    try:
        import cv2
        im = cv2.imread(str(src))
        if im is None:
            return False
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return dst.exists() and dst.stat().st_size > 4000
    except Exception:
        return False


def _validate_dims(p: Path, min_w: int, min_h: int) -> bool:
    try:
        import cv2
        im = cv2.imread(str(p))
        if im is None:
            return False
        h, w = im.shape[:2]
        return w >= min_w and h >= min_h
    except Exception:
        return False
