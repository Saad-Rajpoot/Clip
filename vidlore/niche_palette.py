"""Niche-aware weighted palette selection (reusable, engine-level).

Problem this solves permanently: the old `video_palette(niche, seed)` used
`opts[seed % len(opts)]`, a flat 50/50 between a niche's two palettes — so a
TRUE-CRIME video had a 50% chance of `amber_gold` (a warm business gold) instead
of the genre-defining `ember_red`. This module replaces that with:

  • WEIGHTED selection — each niche leans toward its genre-defining palette(s)
    (crime → ember_red / cold_steel / muted dark, never warm business gold), while
  • preserving controlled per-video VARIATION (deterministic from the seed, so two
    different crime topics still differ), and
  • optional cross-video ANTI-REPEAT (two consecutive same-niche videos won't get
    the same palette), and
  • a logged selection REASON for the manifest.

Pure-deterministic given (niche, seed); persistence/anti-repeat are opt-in so the
core is fully unit-testable. The 4 base palettes live in motion_graphics/look.py
(amber_gold, ember_red, cold_steel, parchment_sepia) — unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# niche → {palette: weight}. The first/heaviest is the genre-defining choice;
# the rest provide controlled variation WITHIN a coherent mood (no warm business
# gold for crime/spy). Weights need not sum to 1.
_NICHE_WEIGHTS = {
    "crime":       {"ember_red": 0.62, "cold_steel": 0.22, "parchment_sepia": 0.16},
    "spy":         {"cold_steel": 0.50, "ember_red": 0.28, "parchment_sepia": 0.22},
    "mystery":     {"cold_steel": 0.45, "ember_red": 0.33, "parchment_sepia": 0.22},
    "geopolitics": {"cold_steel": 0.60, "ember_red": 0.20, "amber_gold": 0.20},
    "tech":        {"cold_steel": 0.70, "amber_gold": 0.30},
    "business":    {"amber_gold": 0.62, "parchment_sepia": 0.23, "cold_steel": 0.15},
    "history":     {"parchment_sepia": 0.66, "amber_gold": 0.20, "cold_steel": 0.14},
    "biography":   {"parchment_sepia": 0.55, "amber_gold": 0.45},
    "science":     {"cold_steel": 0.60, "amber_gold": 0.40},
    "_default":    {"amber_gold": 0.60, "parchment_sepia": 0.25, "cold_steel": 0.15},
}

# fold the many niche vocabularies (director / NIM / look_dna) into the keys above.
_ALIASES = {
    "true_crime": "crime", "truecrime": "crime", "murder": "crime",
    "spy_intel": "spy", "espionage": "spy", "intelligence": "spy",
    "covert": "spy", "investigative": "spy", "spy/mossad": "spy", "mossad": "spy",
    "war": "history", "ancient": "history", "medieval": "history",
    "geo": "geopolitics", "politics": "geopolitics", "conflict": "geopolitics",
    "finance": "business", "money": "business", "economics": "business",
    "bio": "biography", "profile": "biography",
}

_VALID = {"amber_gold", "ember_red", "cold_steel", "parchment_sepia"}
_HIST = Path(os.path.expanduser("~/.vidlore")) / "palette_history.json"


def normalize_niche(niche) -> str:
    n = re.sub(r"[^a-z_]+", "_", str(niche or "").strip().lower()).strip("_")
    if n in _NICHE_WEIGHTS:
        return n
    if n in _ALIASES:
        return _ALIASES[n]
    for k in _ALIASES:                                   # substring tolerance
        if k in n:
            return _ALIASES[k]
    for k in _NICHE_WEIGHTS:
        if k != "_default" and k in n:
            return k
    return "_default"


def _seed_unit(niche: str, seed: int) -> float:
    """Deterministic float in [0,1) from (niche, seed)."""
    h = hashlib.sha1(f"{niche}:{int(seed)}".encode()).hexdigest()
    return (int(h[:8], 16) & 0xFFFFFFFF) / float(0xFFFFFFFF)


def _ranked(weights: dict) -> list[tuple[str, float]]:
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))


def _weighted_pick(weights: dict, u: float) -> str:
    items = _ranked(weights)
    total = sum(w for _p, w in items) or 1.0
    acc = 0.0
    for p, w in items:
        acc += w / total
        if u <= acc + 1e-12:
            return p
    return items[-1][0]


def select_palette(niche, seed: int, *, recent: list[str] | None = None,
                   channel: str | None = None, persist: bool = False) -> tuple[str, str]:
    """Return (palette_name, reason). Deterministic given (niche, seed); if
    `recent` (most-recent-last) is supplied (or loaded via channel+persist), the
    pick is nudged so it differs from the last same-niche video when possible."""
    norm = normalize_niche(niche)
    weights = _NICHE_WEIGHTS.get(norm, _NICHE_WEIGHTS["_default"])
    u = _seed_unit(norm, seed)
    pick = _weighted_pick(weights, u)
    reason = f"niche={norm} weighted pick {pick} (u={u:.2f})"

    if recent is None and persist and channel:
        recent = _load_recent(channel, norm)
    if recent:
        last = recent[-1]
        if pick == last and len(weights) > 1:
            for alt, _w in _ranked(weights):           # next-best different choice
                if alt != last:
                    reason = f"niche={norm}: anti-repeat (last={last}) → {alt}"
                    pick = alt
                    break
    if pick not in _VALID:
        pick, reason = "amber_gold", reason + " [coerced→amber_gold]"
    if persist and channel:
        _record(channel, norm, pick)
    return pick, reason


def selection_for_manifest(niche, seed: int, **kw) -> dict:
    pal, reason = select_palette(niche, seed, **kw)
    return {"palette": pal, "reason": reason, "niche_norm": normalize_niche(niche),
            "weights": _NICHE_WEIGHTS.get(normalize_niche(niche))}


# ── optional cross-video anti-repeat memory (opt-in, never required) ──────────
def _load_recent(channel: str, niche: str, n: int = 4) -> list[str]:
    try:
        data = json.loads(_HIST.read_text())
        key = f"{channel}:{niche}"
        return [e["palette"] for e in data.get(key, [])][-n:]
    except Exception:                                              # noqa: BLE001
        return []


def _record(channel: str, niche: str, palette: str) -> None:
    try:
        _HIST.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(_HIST.read_text()) if _HIST.exists() else {}
        key = f"{channel}:{niche}"
        data.setdefault(key, []).append({"palette": palette})
        data[key] = data[key][-12:]
        _HIST.write_text(json.dumps(data))
    except Exception:                                              # noqa: BLE001
        pass
