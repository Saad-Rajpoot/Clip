"""YouTube Audio Library source -- community @audiolibrary channel.

The official Audio Library lives at studio.youtube.com/audiolibrary which is
OAuth-gated.  The @audiolibrary channel on YouTube mirrors that catalogue
publicly (250k+ subscribers, run by the "Audio Library -- Music for content
creators" team).  Each track on that channel is published with one of two
licence labels in the description:

    * "Free to Use (No Attribution Required)"  -- treated as CC0-like
    * "Free to Use (Attribution Required)"     -- attribution captured

We honour both, attaching the appropriate licence string to every Candidate.
A track that does NOT carry a recognised free-use marker is skipped.
"""
from __future__ import annotations

import re
from typing import Iterable

from . import Candidate, register

# The trusted public mirror playlists. Each entry maps a playlist URL to a
# CATEGORY HINT (auto-classify is still applied by the orchestrator; this is
# a starting bias). The user can extend this in
# vidlore/assets/music/_sources.yaml using the existing manual mechanism.
_PLAYLISTS = (
    # Channel home -- catch-all
    ("https://www.youtube.com/@audiolibrary/videos", "auto"),
    ("https://www.youtube.com/@AudioLibraryPlus/videos", "auto"),
)

_YT_CLIENTS = ("android", "ios", "tv", "mweb", "web_safari")

# Skip SFX uploads (the @audiolibrary channel is mostly sound effects, not
# music). The real YT Audio Library music section is OAuth-gated, so for now
# we drop anything that looks like an SFX clip and rely on the curated
# Incompetech / FMA / Pixabay sources for documentary-quality music.
_SFX_PATTERNS = (
    r"\bsound\s*effect\b",
    r"\bsfx\b",
    r"foley",
    r"/\s*sound\s*effect",
    r"ambience\s*(?:loop|sample)?",
)

# "Free to Use" markers we accept verbatim in the description / title.
_FREE_PATTERNS = (
    r"\bfree\s+to\s+use\b",
    r"\bno\s+copyright\b",
    r"\broyalty[\s-]?free\b",
    r"\bcc0\b",
    r"creative\s+commons",
)


def _is_sfx(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _SFX_PATTERNS)

# Attribution markers -- if present, capture the credit line.
_ATTR_PATTERNS = (
    r"attribution\s+required",
    r"credit\s+(?:required|the\s+artist)",
)


def _is_free(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _FREE_PATTERNS)


def _needs_attribution(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _ATTR_PATTERNS)


def _extract_credit(description: str) -> str:
    """Pull a likely 'Music by ...' / 'Track: ...' credit line."""
    for line in (description or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^(music\s+(by|from)|track|artist|composer)\s*[:-]",
                    s, re.I):
            return s
    return ""


@register("youtube_al")
def discover(limit: int = 20) -> Iterable[Candidate]:
    """Walk each trusted YouTube AL mirror playlist; yield up to ``limit``
    safe candidates."""
    try:
        import yt_dlp                                       # noqa: F401
    except Exception:                                       # noqa: BLE001
        return

    # FLAT pass first (cheap) -- we only enrich the ones we intend to keep
    # so that channel-page enumeration doesn't fetch full descriptions for
    # hundreds of tracks at a time.
    yielded = 0
    for play_url, hint in _PLAYLISTS:
        if yielded >= limit:
            return
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({
                "quiet": True, "no_warnings": True,
                "skip_download": True,
                "extract_flat": "in_playlist",
                "playlistend": max(limit * 2, 40),
                "extractor_args": {"youtube": {
                    "player_client": list(_YT_CLIENTS)}},
            }) as ydl:
                info = ydl.extract_info(play_url, download=False)
        except Exception as e:                              # noqa: BLE001
            print(f"  ! youtube_al: list failed for {play_url}: {e}")
            continue

        entries = (info or {}).get("entries", []) or []
        for e in entries:
            if not e or yielded >= limit:
                break
            vid = e.get("id") or ""
            title = (e.get("title") or vid).strip()
            if not vid:
                continue
            # cheap reject of SFX uploads BEFORE we pay for a description fetch
            if _is_sfx(title):
                continue
            # cheap free-use sniff on title
            blob = title
            # we ALWAYS pull description because YT AL mirror channels put
            # the credit line / "Free to use" tag in the description, not the
            # title. This is the heavy bit -- it's a single video lookup.
            try:
                with yt_dlp.YoutubeDL({
                    "quiet": True, "no_warnings": True,
                    "skip_download": True, "noplaylist": True,
                    "extractor_args": {"youtube": {
                        "player_client": list(_YT_CLIENTS)}},
                }) as ydl2:
                    full = ydl2.extract_info(
                        f"https://www.youtube.com/watch?v={vid}",
                        download=False) or {}
            except Exception:                               # noqa: BLE001
                full = {}
            desc = full.get("description") or ""
            blob = f"{title}\n{desc}"
            if not _is_free(blob):
                continue
            credit = _extract_credit(desc)
            attrib_required = _needs_attribution(blob)
            licence = ("YouTube Audio Library -- Free To Use"
                       + (" (Attribution Required)" if attrib_required
                          else " (No Attribution)"))
            attribution = credit if attrib_required else ""

            yield Candidate(
                title=title,
                source="youtube_al",
                url=f"https://www.youtube.com/watch?v={vid}",
                yt_id=vid,
                channel=(full.get("channel")
                         or full.get("uploader") or ""),
                license=licence,
                attribution=attribution,
                category_hint=hint,
                duration=int(full.get("duration") or 0),
                tags=list((full.get("tags") or [])[:8]),
            )
            yielded += 1
