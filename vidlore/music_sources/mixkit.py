"""Mixkit free music source -- Mixkit Free License (commercial OK, no attribution).

Mixkit serves the FULL track audio from ``assets.mixkit.co/music/<id>/<id>.mp3``
(not ``-preview.mp3`` as an earlier draft assumed).  The track card on the
catalogue page carries a track-title in a nearby heading + the tag the
listing was filtered by, both of which we capture as routing hints.

Per-page yields ~30-70 tracks; we cap by ``limit`` and rotate target tags
so every musiclib category gets a balanced slice.
"""
from __future__ import annotations

import re
from typing import Iterable

from . import Candidate, register

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 "
                   "Safari/537.36"),
    "Accept": "text/html,application/json,*/*",
}

# Tag pages mapped to musiclib categories.  Yield round-robin across them
# so we don't drain one tag at the expense of others.
_BROWSE: tuple[tuple[str, str], ...] = (
    ("https://mixkit.co/free-stock-music/tag/cinematic/", "historical_epic"),
    ("https://mixkit.co/free-stock-music/tag/documentary/", "ambient"),
    ("https://mixkit.co/free-stock-music/tag/dark/", "dark_investigation"),
    ("https://mixkit.co/free-stock-music/tag/suspense/", "suspense"),
    ("https://mixkit.co/free-stock-music/tag/emotional/", "emotional_piano"),
    ("https://mixkit.co/free-stock-music/tag/epic/", "historical_epic"),
    ("https://mixkit.co/free-stock-music/tag/ambient/", "ambient"),
    ("https://mixkit.co/free-stock-music/tag/dramatic/", "climax_build"),
    ("https://mixkit.co/free-stock-music/tag/mysterious/", "mystery"),
    ("https://mixkit.co/free-stock-music/tag/sad/", "emotional_piano"),
    ("https://mixkit.co/free-stock-music/tag/orchestral/", "historical_epic"),
)

# assets.mixkit.co/music/<id>/<id>.mp3
_AUDIO_RE = re.compile(
    r'https://assets\.mixkit\.co/music/(\d+)/\1\.mp3', re.I)
# title lives in nearest preceding card heading
_CARD_TITLE_RE = re.compile(
    r'<a[^>]+class="[^"]*track[^"]*"[^>]*>([^<]+)</a>'
    r'|<h\d[^>]*>([^<]+)</h\d>', re.I)


def _harvest_page(html: str, hint: str) -> list[Candidate]:
    """Pull every assets.mixkit.co mp3 + nearest title from one page."""
    out: list[Candidate] = []
    seen: set[str] = set()
    # Pre-index titles by offset so we can back-pair each mp3 with the
    # nearest preceding title.
    titles: list[tuple[int, str]] = []
    for m in _CARD_TITLE_RE.finditer(html):
        t = (m.group(1) or m.group(2) or "").strip()
        if t and len(t) < 120:
            titles.append((m.start(), t))
    for am in _AUDIO_RE.finditer(html):
        mp3 = am.group(0)
        if mp3 in seen:
            continue
        seen.add(mp3)
        # back-locate nearest preceding title
        title = ""
        for off, t in reversed(titles):
            if off < am.start():
                title = t
                break
        if not title:
            mid = am.group(1)
            title = f"Mixkit Track {mid}"
        # apply quality filter on the title (so cheesy / lyrical pop
        # tagged as "cinematic" still gets dropped)
        from ..music_quality import doc_quality
        if doc_quality(title + " " + hint).verdict == "reject":
            continue
        out.append(Candidate(
            title=title[:120],
            source="mixkit",
            url=mp3,           # use mp3 url as dedup key (unique per track)
            download_url=mp3,
            channel="Mixkit",
            license=("Mixkit Free License -- free for commercial use, "
                     "no attribution required"),
            attribution="",
            category_hint=hint,
            tags=["mixkit"],
        ))
    return out


@register("mixkit")
def discover(limit: int = 30) -> Iterable[Candidate]:
    try:
        import requests
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! mixkit deps missing: {e}")
        return
    # Pull each tag page once, then ROUND-ROBIN yield so category balance
    # is preserved across the limit window.
    per_tag: list[list[Candidate]] = []
    for url, hint in _BROWSE:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=20)
        except Exception as e:                              # noqa: BLE001
            print(f"  ! mixkit: GET {url} failed: {e}")
            continue
        if r.status_code != 200:
            continue
        per_tag.append(_harvest_page(r.text, hint))

    yielded = 0
    pass_no = 0
    while yielded < limit and any(len(b) > pass_no for b in per_tag):
        for b in per_tag:
            if yielded >= limit:
                return
            if pass_no < len(b):
                yield b[pass_no]
                yielded += 1
        pass_no += 1
