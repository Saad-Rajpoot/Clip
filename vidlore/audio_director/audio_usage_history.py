"""Cross-video audio anti-repetition (engine-level, reusable).

Problem this solves permanently: across the many videos a channel publishes, the
soundtrack must not feel recycled. Track-level play counters already persist in
`musiclib`'s `_usage.json`; this module adds the two missing layers the audio
spec calls for —

  • CATEGORY-level cooldowns   (don't lead two consecutive videos with the same
    music category, even if the individual track differs),
  • SFX-FAMILY-level cooldowns (don't open every video with the same whoosh),

plus a DETERMINISTIC per-video seed so two videos on the same topic still vary
their music/SFX choices in a reproducible way (re-rendering the same video twice
yields the same soundtrack; a different video yields a different one).

History is a small JSON ring-buffer of the most recent videos, kept per *scope*
(a channel id, so different channels keep independent histories). Pure logic +
defensive JSON I/O — every caller wraps these in try/except so a failure here can
never break a render.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HISTORY_FILE = "audio_usage_history.json"
MAX_VIDEOS = 32                     # ring-buffer depth
SCHEMA_VERSION = 1


def _root(cfg=None) -> Path:
    """Library root (same dir as musiclib's _usage.json). Defensive fallback."""
    try:
        from .. import musiclib
        return musiclib.library_root(cfg)
    except Exception:                                          # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "assets" / "music"


def _path(cfg=None, scope: str = "default") -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (scope or "default"))
    base = _root(cfg)
    return base / (HISTORY_FILE if safe == "default" else f"audio_usage_{safe}.json")


def _empty() -> dict:
    return {"schema_version": SCHEMA_VERSION, "videos": []}


def load_history(cfg=None, scope: str = "default") -> dict:
    p = _path(cfg, scope)
    if not p.exists():
        return _empty()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "videos" not in d:
            return _empty()
        d.setdefault("schema_version", SCHEMA_VERSION)
        if not isinstance(d["videos"], list):
            d["videos"] = []
        return d
    except Exception:                                          # noqa: BLE001
        return _empty()


def save_history(history: dict, cfg=None, scope: str = "default") -> None:
    try:
        p = _path(cfg, scope)
        p.parent.mkdir(parents=True, exist_ok=True)
        # keep only the most recent MAX_VIDEOS records
        history = dict(history)
        history["videos"] = list(history.get("videos", []))[-MAX_VIDEOS:]
        p.write_text(json.dumps(history, indent=1, ensure_ascii=False),
                     encoding="utf-8")
    except Exception:                                          # noqa: BLE001
        pass


# ── deterministic per-video variation ────────────────────────────────────────
def video_seed(video_id: str, scope: str = "default", salt: str = "") -> int:
    """Stable 0..2^31 seed from (scope, video_id, salt). Same inputs -> same seed
    (reproducible renders); different video_id -> different variation."""
    key = f"{scope}|{video_id}|{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha1(key).digest()[:4], "big") & 0x7FFFFFFF


# ── cooldown queries (was X used within the last `gap` videos?) ───────────────
def _recent(history: dict, field: str, gap: int) -> set:
    seen: set = set()
    vids = history.get("videos", [])
    for rec in vids[-max(0, int(gap)):]:
        for v in (rec.get(field) or []):
            seen.add(v)
    return seen


def category_recently_used(history: dict, category: str, gap: int = 2) -> bool:
    return category in _recent(history, "categories", gap)


def track_recently_used(history: dict, track_id: str, gap: int = 3) -> bool:
    return track_id in _recent(history, "tracks", gap)


def sfx_family_recently_used(history: dict, family: str, gap: int = 1) -> bool:
    return family in _recent(history, "sfx_families", gap)


def lead_category_recently_used(history: dict, category: str, gap: int = 3) -> bool:
    """A stricter check for the *opening* category: avoid two videos that OPEN on
    the same music category within `gap` videos (the most noticeable repeat)."""
    vids = history.get("videos", [])
    for rec in vids[-max(0, int(gap)):]:
        if rec.get("lead_category") == category:
            return True
    return False


def filter_categories(history: dict, ordered: list[str], gap: int = 2) -> list[str]:
    """Reorder a fallback category chain so recently-used categories sink to the
    back (never dropped — we always keep a usable option)."""
    if not ordered:
        return ordered
    recent = _recent(history, "categories", gap)
    fresh = [c for c in ordered if c not in recent]
    stale = [c for c in ordered if c in recent]
    return fresh + stale if fresh else ordered


# ── recording ─────────────────────────────────────────────────────────────────
def record_video(history: dict, *, video_id: str, niche: str = "",
                 categories=None, tracks=None, sfx_families=None,
                 lead_category: str = "", date: str = "") -> dict:
    """Append a video's audio fingerprint. De-dupes a re-render of the same
    video_id (updates in place) so re-rendering doesn't inflate history."""
    rec = {
        "video_id": str(video_id),
        "niche": niche or "",
        "date": date or "",
        "lead_category": lead_category or "",
        "categories": sorted(set(categories or [])),
        "tracks": sorted(set(tracks or [])),
        "sfx_families": sorted(set(sfx_families or [])),
    }
    vids = [v for v in history.get("videos", []) if v.get("video_id") != rec["video_id"]]
    vids.append(rec)
    history["videos"] = vids[-MAX_VIDEOS:]
    return history


def summarize(history: dict) -> dict:
    vids = history.get("videos", [])
    cats: dict[str, int] = {}
    fams: dict[str, int] = {}
    for rec in vids:
        for c in rec.get("categories", []):
            cats[c] = cats.get(c, 0) + 1
        for f in rec.get("sfx_families", []):
            fams[f] = fams.get(f, 0) + 1
    return {"videos": len(vids),
            "distinct_categories": len(cats),
            "category_counts": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "sfx_family_counts": dict(sorted(fams.items(), key=lambda kv: -kv[1]))}
