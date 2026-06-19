"""Trusted royalty-free music source registry.

Each source module registers a ``discover()`` generator that yields
:class:`Candidate` objects.  The orchestrator (:mod:`vidlore.music_extract`)
walks every registered source in a round-robin so the library grows
**evenly across sources** -- no one channel dominates the eventual mix.

A new source is added by dropping a module into this package that calls
:func:`register` with a unique name.  See :mod:`vidlore.music_sources.incompetech`
for the simplest reference implementation.

LICENSING CONTRACT
==================
Every Candidate MUST carry a ``license`` string.  The orchestrator REFUSES
to ingest a candidate with no license set.  This is intentional -- safer
to skip a track than to land an un-attributed copyrighted file in the
library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

# Lazy-import marker: source modules are imported when the registry is first
# enumerated, not at package import time. Heavy network deps stay quiet
# until --auto actually runs.
_LAZY_MODULES = (
    "youtube_al",
    "incompetech",
    "fma",
    "mixkit",
    "pixabay",
)


@dataclass
class Candidate:
    """One discoverable track from one source. Network-light: the source
    fills in the metadata it can cheaply, and the orchestrator decides
    whether to actually download."""

    # identification
    title: str
    source: str                                  # name registered below
    url: str                                     # canonical landing page

    # download hints
    download_url: str | None = None              # direct audio file URL
    yt_id: str | None = None                     # use yt_dlp on this id

    # provenance / attribution (license REQUIRED -- empty = rejected)
    channel: str = ""
    license: str = ""
    attribution: str = ""

    # routing hints
    category_hint: str = "auto"                  # source-side mood guess
    duration: int = 0                            # seconds, 0 = unknown
    tags: list[str] = field(default_factory=list)

    def is_safe(self) -> bool:
        """True only when we have enough provenance to ingest the track."""
        return bool(self.license and (self.download_url or self.yt_id))


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
SourceFn = Callable[[int], Iterable[Candidate]]

_REGISTRY: dict[str, SourceFn] = {}


def register(name: str):
    """Decorator: ``@register("incompetech")`` on a generator
    ``def discover(limit: int) -> Iterable[Candidate]``."""
    def deco(fn: SourceFn) -> SourceFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def _ensure_loaded() -> None:
    """Lazy-import every source module so their @register decorators fire."""
    for mod in _LAZY_MODULES:
        if mod in _REGISTRY:
            continue
        try:
            __import__(f"{__name__}.{mod}", fromlist=["*"])
        except Exception as e:                             # noqa: BLE001
            # A broken source must NEVER take the orchestrator down.
            print(f"  ! source '{mod}' failed to load: {e}")


def available_sources() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def discover_all(per_source_limit: int = 25,
                 only: list[str] | None = None) -> list[Candidate]:
    """Round-robin every enabled source, take up to ``per_source_limit``
    candidates from each.  Order is INTERLEAVED so the orchestrator's per-
    category budgets are evenly fed even if it stops early."""
    _ensure_loaded()
    by_src: dict[str, list[Candidate]] = {}
    for name, fn in _REGISTRY.items():
        if only and name not in only:
            continue
        try:
            got = list(fn(per_source_limit))
        except Exception as e:                             # noqa: BLE001
            print(f"  ! source '{name}' raised: {e}")
            got = []
        # drop unsafe (no license / no download method) immediately
        safe = [c for c in got if c.is_safe()]
        if len(safe) < len(got):
            print(f"  . {name}: dropped "
                  f"{len(got) - len(safe)} unsafe candidate(s)")
        by_src[name] = safe
        print(f"  + {name}: {len(safe)} candidates")

    # interleave round-robin so per-source diversity is preserved
    out: list[Candidate] = []
    while any(by_src.values()):
        for name in list(by_src):
            if by_src[name]:
                out.append(by_src[name].pop(0))
    return out
