"""Pixabay Music source.

Pixabay does not expose a public REST API for its music catalogue (their
documented API covers images + video only), but the website serves an
internal JSON endpoint that returns track metadata + a direct mp3 URL.
We hit that endpoint directly with the same User-Agent + Origin headers a
browser would send.  Each track is licensed under the Pixabay Content
License -- free for commercial use, no attribution required.

If the endpoint shape changes, this source yields nothing rather than
crashing the orchestrator.
"""
from __future__ import annotations

import os
from typing import Iterable

from . import Candidate, register

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 "
                   "Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://pixabay.com",
    "Referer": "https://pixabay.com/",
}

# Mood-targeted queries.  Pixabay also accepts a "category=music" filter
# but mood/instrument keywords give us better targeted hits.
_QUERIES = (
    ("dark cinematic", "dark_investigation"),
    ("suspense thriller", "suspense"),
    ("mysterious", "mystery"),
    ("epic orchestral", "historical_epic"),
    ("military drum", "military_tension"),
    ("ambient drone", "ambient"),
    ("emotional piano", "emotional_piano"),
    ("corporate", "financial"),
    ("synthwave", "tech_cyber"),
    ("aftermath calm", "aftermath"),
)


def _endpoint(q: str, page: int = 1) -> str:
    # Internal JSON used by pixabay.com/music/search/<q>/
    # (subject to change without notice)
    from urllib.parse import quote_plus
    return ("https://pixabay.com/api/music/search/?q="
            f"{quote_plus(q)}&pagi={page}")


@register("pixabay")
def discover(limit: int = 20) -> Iterable[Candidate]:
    try:
        import requests
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! pixabay deps missing: {e}")
        return
    # Pixabay's *documented* API key (PIXABAY_API_KEY) doesn't cover music,
    # but setting it via header is still polite and prevents 429s.
    headers = dict(_HEADERS)
    key = os.environ.get("PIXABAY_API_KEY") or ""
    if key:
        headers["X-Pixabay-Key"] = key

    yielded = 0
    seen: set[str] = set()
    for q, hint in _QUERIES:
        if yielded >= limit:
            return
        try:
            r = requests.get(_endpoint(q), headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            data = r.json() if r.headers.get(
                "content-type", "").startswith("application/json") else {}
        except Exception as e:                              # noqa: BLE001
            print(f"  ! pixabay: q='{q}' failed: {e}")
            continue
        hits = (data or {}).get("hits") or (data or {}).get("results") or []
        if not isinstance(hits, list):
            continue
        for h in hits:
            if yielded >= limit:
                break
            if not isinstance(h, dict):
                continue
            mp3 = (h.get("audio_url") or h.get("preview_url")
                   or h.get("url") or "")
            if not mp3 or mp3 in seen:
                continue
            seen.add(mp3)
            title = (h.get("name") or h.get("title")
                     or h.get("display_name")
                     or mp3.rsplit("/", 1)[-1].split(".")[0])
            dur = int(h.get("duration") or 0)
            yield Candidate(
                title=str(title)[:120],
                source="pixabay",
                url=str(h.get("permalink") or "https://pixabay.com/music/"),
                download_url=mp3,
                channel=str(h.get("user_name") or h.get("user") or "Pixabay"),
                license=("Pixabay Content License -- "
                         "free for commercial use, no attribution required"),
                attribution="",
                category_hint=hint,
                duration=dur,
                tags=["pixabay"],
            )
            yielded += 1
