"""Free Music Archive source.

FMA (freemusicarchive.org) hosts a wide CC catalogue.  Licenses VARY per
track, so we ONLY ingest tracks whose page exposes one of:

    * CC0
    * CC BY  (any version)
    * CC BY-SA  (any version)

CC BY-NC / NC-SA / ND are SKIPPED -- attribution is allowed but commercial
restrictions / no-derivatives don't fit a documentary score we may edit.

We browse curated FMA genre listings rather than blanket search so the
candidate quality stays high.  Each track's download URL is on its page;
we parse it from the standard ``<meta itemprop="audio">`` element.
"""
from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin

from . import Candidate, register

_BROWSE_URLS = (
    # FMA genre indexes that lean documentary-useful.
    "https://freemusicarchive.org/genre/Soundtrack/",
    "https://freemusicarchive.org/genre/Ambient/",
    "https://freemusicarchive.org/genre/Classical/",
    "https://freemusicarchive.org/genre/Instrumental/",
    "https://freemusicarchive.org/genre/Drone/",
)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 "
                   "Safari/537.36"),
}

_GENRE_CAT = {
    "soundtrack": "historical_epic",
    "ambient": "ambient",
    "classical": "emotional_piano",
    "instrumental": "auto",
    "drone": "ambient",
}

# License sniff -- order matters (most-permissive first)
_OK_LICENSES = (
    ("cc0", "CC0 1.0 (Public Domain Dedication)", False),
    ("publicdomain", "Public Domain", False),
    ("by-sa", "CC BY-SA 4.0", True),
    ("by/", "CC BY 4.0", True),
    ("by-", "CC BY (variant)", True),
)

_REJECT_TOKENS = ("nc", "nd")  # NonCommercial / NoDerivatives -- skip


def _classify_license(href: str) -> tuple[str, bool] | None:
    h = href.lower()
    if any(("/" + t + "/") in h or ("-" + t + "/") in h
           for t in _REJECT_TOKENS):
        return None
    for needle, label, attribution in _OK_LICENSES:
        if needle in h:
            return label, attribution
    return None


@register("fma")
def discover(limit: int = 20) -> Iterable[Candidate]:
    try:
        import requests
        from lxml import html as lxml_html
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! fma deps missing: {e}")
        return
    yielded = 0
    seen: set[str] = set()
    for browse in _BROWSE_URLS:
        if yielded >= limit:
            return
        cat = _GENRE_CAT.get(browse.rstrip("/").rsplit("/", 1)[-1].lower(),
                             "auto")
        try:
            r = requests.get(browse, headers=_HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:                              # noqa: BLE001
            print(f"  ! fma: GET {browse} failed: {e}")
            continue
        try:
            doc = lxml_html.fromstring(r.text)
        except Exception:                                   # noqa: BLE001
            continue
        # track pages live at /music/<artist>/<album>/<track-slug>/
        track_links = [a.get("href") for a in
                       doc.xpath("//a[contains(@href,'/music/')]")
                       if (a.get("href") or "").count("/") >= 4]
        for tlink in track_links:
            if yielded >= limit:
                break
            tpage = urljoin(browse, tlink)
            if tpage in seen:
                continue
            seen.add(tpage)
            try:
                tr = requests.get(tpage, headers=_HEADERS, timeout=20)
                if tr.status_code != 200:
                    continue
                td = lxml_html.fromstring(tr.text)
            except Exception:                               # noqa: BLE001
                continue
            # license link -- canonical CC URL
            lic_href = ""
            for la in td.xpath("//a[contains(@href,'creativecommons.org')]"):
                lic_href = (la.get("href") or "")
                if lic_href:
                    break
            classified = _classify_license(lic_href)
            if not classified:
                continue
            lic_label, attr_required = classified
            # download url
            mp3 = ""
            for meta in td.xpath("//meta[@itemprop='audio']"):
                mp3 = meta.get("content") or ""
                if mp3:
                    break
            if not mp3:
                for a in td.xpath("//a[contains(@href,'.mp3')]"):
                    mp3 = urljoin(tpage, a.get("href") or "")
                    if mp3:
                        break
            if not mp3:
                continue
            # title + artist
            title_el = td.xpath("//h1[1]")
            title = (title_el[0].text_content().strip()
                     if title_el else tpage.rsplit("/", 2)[-2])
            artist_el = td.xpath("//*[contains(@class,'artist')][1]")
            artist = (artist_el[0].text_content().strip()
                      if artist_el else "")
            attribution = (
                f"Music: {artist} -- {title} ({lic_label}, "
                f"freemusicarchive.org)") if attr_required else ""
            yield Candidate(
                title=re.sub(r"\s+", " ", title)[:120],
                source="fma",
                url=tpage,
                download_url=mp3,
                channel=artist,
                license=lic_label + " (Free Music Archive)",
                attribution=attribution,
                category_hint=cat,
                tags=["fma"],
            )
            yielded += 1
