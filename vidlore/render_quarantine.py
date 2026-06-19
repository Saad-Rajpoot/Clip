"""Render-clip quarantine (V3.2.2).

A tiny, dependency-free, never-raising registry of footage clips that failed
validation or crashed assembly, so a malformed external clip:
  • is not handed to a downstream ffmpeg stage again this render, and
  • is not re-selected from cache on the NEXT render.

Keyed by both local path and (when known) source URL. Persists to a JSON sidecar
under the run dir's cache so it survives across renders; also keeps an in-process
set for the current render. Pure stdlib. Every public call swallows its own
exceptions — quarantine must never itself break a render.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# in-process quarantine (current render) — path strings + url strings
_MEM: set[str] = set()
_RECORDS: list[dict] = []
_SIDE: Path | None = None

# RC5 — GLOBAL (cross-project) junk registry. The per-project sidecar above bars
# a rejected asset only within ITS render dir; but a globally-obvious junk asset
# (a game/anime/UI/poster image identified by its source metadata) should never
# be re-served on ANY future project either. This second registry persists such
# url/signature entries to a stable user-level file and is merged into `_MEM` on
# attach, so `is_quarantined(url=...)` blocks it everywhere. Best-effort; if the
# home dir is unwritable it simply degrades to per-project behaviour.
_GLOBAL_MEM: set[str] = set()
_GLOBAL_RECORDS: list[dict] = []
_GLOBAL_SIDE: Path = Path(os.environ.get(
    "VIDLORE_RELEVANCE_QUARANTINE",
    os.path.join(os.path.expanduser("~"), ".vidlore",
                 "relevance_quarantine.json")))


def _norm(s) -> str:
    try:
        return str(s or "").strip()
    except Exception:                                          # noqa: BLE001
        return ""


def _load_global() -> None:
    """Load the cross-project junk registry into `_GLOBAL_MEM` / `_MEM`."""
    try:
        if _GLOBAL_SIDE.is_file():
            data = json.loads(_GLOBAL_SIDE.read_text() or "[]")
            for r in (data or []):
                v = _norm(r.get("asset_source_url")) \
                    or _norm(r.get("asset_local_path"))
                if v:
                    _GLOBAL_MEM.add(v)
                    _MEM.add(v)
            _GLOBAL_RECORDS[:] = list(data or [])
    except Exception:                                          # noqa: BLE001
        pass


def attach(cache_dir) -> None:
    """Point the quarantine at a run/cache dir + load any prior sidecar (the
    per-project corrupt-clip + relevance records) AND the cross-project junk
    registry, so globally-obvious junk stays blocked across projects."""
    global _SIDE
    try:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        _SIDE = d / "clip_quarantine.json"
        if _SIDE.is_file():
            data = json.loads(_SIDE.read_text() or "[]")
            for r in (data or []):
                for k in ("asset_local_path", "asset_source_url"):
                    v = _norm(r.get(k))
                    if v:
                        _MEM.add(v)
    except Exception:                                          # noqa: BLE001
        _SIDE = None
    _load_global()


def _persist() -> None:
    try:
        if _SIDE is not None:
            _SIDE.write_text(json.dumps(_RECORDS, indent=1))
    except Exception:                                          # noqa: BLE001
        pass


def quarantine(local_path="", *, source_url="", reason="",
               replacement_path="", replacement_source_type="",
               replacement_reason="", retry_count=0, timestamp="",
               global_junk=False) -> None:
    """Record a bad clip + its replacement. `timestamp` is passed in (callers
    own the clock so this stays deterministic / resume-safe).

    RC5: `global_junk=True` ALSO writes the asset's source_url to the
    cross-project junk registry so a globally-obvious junk asset (a game/anime/UI
    image identified by its metadata, never legitimate documentary footage) can
    never be re-served on ANY future render either. Relevance rejects of a
    project-specific asset use the default (per-project only)."""
    try:
        lp, su = _norm(local_path), _norm(source_url)
        if lp:
            _MEM.add(lp)
        if su:
            _MEM.add(su)
        _RECORDS.append({
            "asset_local_path": lp,
            "asset_source_url": su,
            "asset_validation_status": "rejected",
            "asset_rejection_reason": _norm(reason),
            "asset_quarantined": True,
            "asset_quarantine_timestamp": _norm(timestamp),
            "replacement_source_type": _norm(replacement_source_type),
            "replacement_path": _norm(replacement_path),
            "replacement_reason": _norm(replacement_reason),
            "assembly_retry_count": int(retry_count or 0),
        })
        _persist()
        if global_junk and su:
            _quarantine_global(su, reason)
        # NOTE: we deliberately do NOT rename/move the bad file. Strong
        # validation (`_clip_ready(strong=True)`) RE-REJECTS a corrupt clip
        # deterministically on every render → it can never be reused, so the
        # record + `is_quarantined()` skip are sufficient. Renaming risked a
        # "missing file" in any stage still holding the original path. The bytes
        # stay on disk for debugging; the run-dir cache is cleaned wholesale by
        # the normal storage-safety sweep.
    except Exception:                                          # noqa: BLE001
        pass


def _quarantine_global(source_url: str, reason: str) -> None:
    """Persist a globally-obvious junk asset to the cross-project registry."""
    try:
        su = _norm(source_url)
        if not su or su in _GLOBAL_MEM:
            return
        _GLOBAL_MEM.add(su)
        _MEM.add(su)
        _GLOBAL_RECORDS.append({
            "asset_source_url": su,
            "asset_rejection_reason": _norm(reason),
            "asset_quarantined": True,
            "scope": "global",
        })
        _GLOBAL_SIDE.parent.mkdir(parents=True, exist_ok=True)
        _GLOBAL_SIDE.write_text(json.dumps(_GLOBAL_RECORDS, indent=1))
    except Exception:                                          # noqa: BLE001
        pass


def is_quarantined(local_path="", source_url="") -> bool:
    try:
        lp, su = _norm(local_path), _norm(source_url)
        return bool((lp and lp in _MEM) or (su and su in _MEM))
    except Exception:                                          # noqa: BLE001
        return False


def records() -> list[dict]:
    return list(_RECORDS)


def reset() -> None:
    """Clear in-process state (tests). Does NOT delete the on-disk global
    registry; pass a tmp VIDLORE_RELEVANCE_QUARANTINE in tests to isolate it."""
    global _SIDE
    _MEM.clear()
    _RECORDS.clear()
    _GLOBAL_MEM.clear()
    _GLOBAL_RECORDS.clear()
    _SIDE = None
