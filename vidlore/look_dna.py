"""LOOK DNA — Per-Channel Persistent Cinematic Identity.

The architectural answer to Vidlore V2's promised "uniqueness constraint":
where they will (likely) ship per-render stochastic randomisation, we ship
**persistent per-channel DNA + per-render variation within ranges +
forensic anti-convergence feedback loop.**

A Look DNA file is a 24-dimension YAML at
``~/.vidlore/channels/<channel_name>.yaml`` that defines a channel's
permanent cinematic identity.  Every renderer downstream consults the
*resolved* look (DNA after seeded sampling) before falling back to the
legacy Style Mode / Theme defaults — so the same channel feels
consistent across many videos, different channels feel completely
different, and the architecture is **fully backward-compatible**:
no channel set → existing behaviour exactly.

Public API surface (the contract the rest of the pipeline reads):

    from vidlore.look_dna import resolve_look, ResolvedLook

    look = resolve_look(brief, channel_name="homestead_roots")
    #   ↑ load YAML, inherit preset, apply brief overrides, sample
    #     ranges with a deterministic seed, anti-converge against
    #     channel history, return a frozen ResolvedLook ready for
    #     every downstream consumer to read.

    look.beat_target        # 3.91 (sampled from {min:3.6, max:4.2})
    look.silence.tier3_pocket_s   # 1.12
    look.transition_palette.sample(rng)  # 'hard_cut' (weighted)

Storage layout::

    ~/.vidlore/
    ├── channels/
    │   ├── <name>.yaml         ← the DNA (source of truth)
    │   └── <name>/
    │       ├── history.jsonl   ← forensic fingerprint per render
    │       ├── seed_state.json ← variation cursor
    │       └── face_seeds.json ← character-name → AI face seed
    └── presets/
        ├── standard.yaml
        ├── homestead.yaml
        └── ... (built-in starters)

This module is intentionally self-contained — no imports from the rest
of vidlore.  Downstream consumers do all the integration.  That keeps
the DNA layer testable in isolation and impossible to break by
accident from a footage/assemble refactor.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml                                                    # type: ignore
except ImportError:                                                # pragma: no cover
    yaml = None    # falls back to JSON; YAML is recommended


# =====================================================================
# 1. SENTINEL TYPES — the building blocks that distinguish a fixed
#    value from a range or weighted bag.  Inline classes (not enums)
#    because we marshal them to/from YAML constantly.
# =====================================================================

class Range:
    """Continuous-uniform numeric range.  Sampled with rng.uniform()."""
    __slots__ = ("min", "max")
    def __init__(self, mn: float, mx: float) -> None:
        if mn > mx:
            mn, mx = mx, mn
        self.min, self.max = float(mn), float(mx)
    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.min, self.max)
    def midpoint(self) -> float:
        return (self.min + self.max) / 2.0
    def to_dict(self) -> dict:
        return {"min": self.min, "max": self.max}
    def __repr__(self) -> str:                                     # pragma: no cover
        return f"Range({self.min:.3f}, {self.max:.3f})"


class IntRange(Range):
    """Discrete-integer range.  Sampled with rng.randint()."""
    def sample(self, rng: random.Random) -> int:
        return rng.randint(int(self.min), int(self.max))
    def __repr__(self) -> str:                                     # pragma: no cover
        return f"IntRange({int(self.min)}, {int(self.max)})"


class WeightedBag:
    """Discrete weighted choice (e.g. transition palette).  Weights
    do not need to sum to 1.0 — they are normalised at sample time."""
    __slots__ = ("items",)
    def __init__(self, items: dict[str, float]) -> None:
        # Drop zero/negative weights; keep insertion order
        self.items = {k: float(v) for k, v in items.items() if float(v) > 0}
        if not self.items:
            raise ValueError("WeightedBag needs at least one positive-weight item")
    def sample(self, rng: random.Random) -> str:
        keys = list(self.items.keys())
        weights = list(self.items.values())
        return rng.choices(keys, weights=weights, k=1)[0]
    def top(self, n: int = 1) -> list[str]:
        """Return the n most-weighted items (for deterministic fallback)."""
        return [k for k, _ in sorted(
            self.items.items(), key=lambda kv: -kv[1])[:n]]
    def to_dict(self) -> dict:
        return dict(self.items)
    def __repr__(self) -> str:                                     # pragma: no cover
        return f"WeightedBag({self.items!r})"


def _coerce(value: Any) -> Any:
    """Best-effort YAML-dict → sentinel coercion.  A dict with EXACTLY
    {min, max} numeric keys becomes a Range; ints stay IntRange.  Other
    dicts stay dicts (they are nested groups like ``silence:``)."""
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {"min", "max"} and all(
                isinstance(value[k], (int, float)) for k in keys):
            if isinstance(value["min"], int) and isinstance(value["max"], int):
                return IntRange(value["min"], value["max"])
            return Range(value["min"], value["max"])
        # Weighted bag heuristic: ALL values are positive numbers and
        # the key set isn't a recognised structural one (avoids false
        # positives for, e.g., {brightness_lift: ..., saturation_lift: ...}
        # whose values happen to all be numbers).  We mark a bag by
        # convention: the dict must be the value of a key named
        # ``*_palette`` or ``*_bag``; otherwise leave as dict.
        return {k: _coerce(v) for k, v in value.items()}
    return value


def _uncoerce(value: Any) -> Any:
    """Inverse of _coerce for YAML serialisation."""
    if isinstance(value, (Range, IntRange)):
        return value.to_dict()
    if isinstance(value, WeightedBag):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _uncoerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_uncoerce(v) for v in value]
    return value


# =====================================================================
# 2. LookDNA — the source-of-truth dataclass (with ranges intact)
# =====================================================================

@dataclass
class LookDNA:
    """A channel's full DNA, ranges-and-all.  Call ``.sample()`` to
    collapse ranges to point values and produce a ``ResolvedLook``."""
    name: str = "unnamed"
    label: str = ""
    inherits: str = ""
    version: int = 1
    # Flat key→value bag — keeps the structure flexible so we can add
    # new dimensions without changing the dataclass shape.  The full
    # 24-dimension schema lives in the YAML; we don't enforce it here
    # so v2 can add fields without code changes.
    knobs: dict = field(default_factory=dict)

    # ---- loaders ---------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict) -> "LookDNA":
        d = dict(d or {})
        name = d.pop("name", "unnamed")
        label = d.pop("label", "")
        inherits = d.pop("inherits", "")
        version = int(d.pop("version", 1))
        knobs = {k: _coerce(v) for k, v in d.items()}
        return cls(name=name, label=label, inherits=inherits,
                   version=version, knobs=knobs)

    @classmethod
    def from_yaml_file(cls, path: Path) -> "LookDNA":
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Look DNA file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        if yaml is None:
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw) or {}
        return cls.from_dict(data)

    @classmethod
    def from_preset(cls, name: str) -> "LookDNA":
        """Load a built-in preset from ``vidlore/look_presets/<name>.yaml``."""
        here = Path(__file__).parent
        path = here / "look_presets" / f"{name}.yaml"
        if not path.exists():
            # The ``standard`` floor must always succeed.  Build it
            # programmatically as a last-resort fallback.
            if name == "standard":
                return cls.from_dict(_STANDARD_FLOOR_DICT())
            raise FileNotFoundError(
                f"Preset '{name}' not found at {path}")
        return cls.from_yaml_file(path)

    # ---- merge ------------------------------------------------------ #
    def merge(self, other: "LookDNA") -> "LookDNA":
        """Return a new LookDNA where `other`'s values override self's.

        Merge semantics (deliberate, documented):
          * scalars → child overrides parent
          * Range / IntRange / WeightedBag → child REPLACES parent
          * dict → recursive deep-merge
          * list → child REPLACES parent (NOT extends — avoids sneaky
            list-extension bugs across inheritance chains)
        """
        merged_knobs = _deep_merge(self.knobs, other.knobs)
        return LookDNA(
            name=other.name or self.name,
            label=other.label or self.label,
            inherits=other.inherits or self.inherits,
            version=max(self.version, other.version),
            knobs=merged_knobs,
        )

    # ---- sampling --------------------------------------------------- #
    def sample(self, rng: Optional[random.Random] = None) -> "ResolvedLook":
        """Collapse every Range / IntRange / WeightedBag to a point
        value using the given RNG (seeded for determinism)."""
        if rng is None:
            rng = random.Random()
        sampled = _sample_recursive(self.knobs, rng)
        return ResolvedLook(
            name=self.name, label=self.label, version=self.version,
            knobs=sampled,
        )

    # ---- persistence ----------------------------------------------- #
    def save(self, path: Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "label": self.label,
            "version": self.version,
        }
        if self.inherits:
            data["inherits"] = self.inherits
        data.update(_uncoerce(self.knobs))
        if yaml is None:
            payload = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            payload = yaml.safe_dump(data, sort_keys=False,
                                      allow_unicode=True, default_flow_style=False)
        _atomic_write(path, payload)
        return path

    # ---- utility --------------------------------------------------- #
    def get(self, dotted: str, default: Any = None) -> Any:
        """``look.get('silence.tier3_pocket_s')`` → walks knobs by dot path."""
        node: Any = self.knobs
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


# =====================================================================
# 3. ResolvedLook — the SAMPLED, frozen look that downstream reads
# =====================================================================

@dataclass(frozen=True)
class ResolvedLook:
    """The look after seeded sampling — all ranges collapsed to point
    values, all weighted bags realised to a single choice.  This is
    what every renderer downstream consults.  Frozen to guarantee
    nothing in the render pipeline can mutate it accidentally."""
    name: str
    label: str
    version: int
    knobs: dict        # nested dict, all leaves are scalars/lists/dicts

    def get(self, dotted: str, default: Any = None) -> Any:
        """Dotted-path lookup; same semantics as LookDNA.get."""
        node: Any = self.knobs
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label,
                "version": self.version, "knobs": dict(self.knobs)}


# =====================================================================
# 4. Deep merge + recursive sample helpers
# =====================================================================

def _deep_merge(parent: dict, child: dict) -> dict:
    """Child overrides parent.  Scalars/sentinels/lists: child wins.
    Dicts: recursive merge."""
    out = dict(parent)
    for k, v in (child or {}).items():
        if (k in out
                and isinstance(out[k], dict) and isinstance(v, dict)
                and not _is_sentinel_dict(v)
                and not _is_sentinel_dict(out[k])):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_sentinel_dict(value: Any) -> bool:
    """True if `value` is a coerced sentinel object (so we don't
    recursive-merge a Range as if it were a regular dict)."""
    return isinstance(value, (Range, IntRange, WeightedBag))


def _sample_recursive(node: Any, rng: random.Random) -> Any:
    if isinstance(node, (Range, IntRange)):
        return node.sample(rng)
    if isinstance(node, WeightedBag):
        # Resolved look keeps the BAG (so transition picker can still
        # sample per-cut) AND a default top-pick.
        return {"_bag": node.to_dict(), "default": node.top(1)[0]}
    if isinstance(node, dict):
        return {k: _sample_recursive(v, rng) for k, v in node.items()}
    if isinstance(node, list):
        return [_sample_recursive(v, rng) for v in node]
    return node


# =====================================================================
# 5. Storage roots + filesystem helpers
# =====================================================================

def _channels_root() -> Path:
    """Where channel YAMLs + per-channel state live.  Overridable via
    VIDLORE_LOOK_HOME for tests / multi-user setups."""
    env = os.environ.get("VIDLORE_LOOK_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".vidlore" / "channels"


def _channel_state_dir(name: str) -> Path:
    p = _channels_root() / _safe_name(name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_name(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_"
                 for c in (name or "").strip().lower())
    return s or "default"


def _atomic_write(path: Path, payload: str) -> None:
    """tempfile + rename — never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False,
            prefix=".tmp_", suffix=path.suffix,
            encoding="utf-8") as tf:
        tf.write(payload)
        tmp = Path(tf.name)
    tmp.replace(path)


# =====================================================================
# 6. Per-channel runtime state — cursor + history + face seeds
# =====================================================================

def _read_cursor(channel_name: str) -> int:
    if not channel_name:
        return 0
    p = _channel_state_dir(channel_name) / "seed_state.json"
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text()).get("cursor", 0))
    except Exception:                                              # noqa: BLE001
        return 0


def _bump_cursor(channel_name: str, stride: int = 1) -> int:
    if not channel_name:
        return 0
    current = _read_cursor(channel_name) + max(1, int(stride))
    p = _channel_state_dir(channel_name) / "seed_state.json"
    _atomic_write(p, json.dumps({"cursor": current, "ts": int(time.time())}))
    return current


def append_history(channel_name: str, entry: dict) -> None:
    """Append a fingerprint entry to the channel's history.jsonl.
    Called by the pipeline at end-of-render."""
    if not channel_name:
        return
    p = _channel_state_dir(channel_name) / "history.jsonl"
    line = json.dumps(entry, ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_recent_history(channel_name: str, n: int = 5) -> list[dict]:
    if not channel_name:
        return []
    p = _channel_state_dir(channel_name) / "history.jsonl"
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:                                              # noqa: BLE001
        return []
    out = []
    for ln in lines[-max(1, n):]:
        try:
            out.append(json.loads(ln))
        except Exception:                                          # noqa: BLE001
            continue
    return out


def read_face_seed(channel_name: str, character_name: str) -> Optional[int]:
    """Return the cached AI-face seed for a recurring character so
    every appearance of 'Eli' uses the same face across the channel."""
    if not channel_name or not character_name:
        return None
    p = _channel_state_dir(channel_name) / "face_seeds.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        v = d.get(character_name.strip().lower())
        return int(v) if v is not None else None
    except Exception:                                              # noqa: BLE001
        return None


def write_face_seed(channel_name: str, character_name: str,
                    seed: int) -> None:
    if not channel_name or not character_name:
        return
    p = _channel_state_dir(channel_name) / "face_seeds.json"
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except Exception:                                              # noqa: BLE001
        d = {}
    d[character_name.strip().lower()] = int(seed)
    _atomic_write(p, json.dumps(d, indent=2, ensure_ascii=False))


# =====================================================================
# 7. Fingerprint prediction + anti-convergence
# =====================================================================

# Approximate forward-prediction of forensic-tool fingerprints from a
# resolved look.  These are *intentionally* coarse — good enough to
# detect "is the next render going to look like the last 3?" without
# requiring an actual render.  Calibrated against the existing
# polished-EN baseline so the mappings start sensible.

_FP_AXES = (
    "median_shot_s", "cuts_per_min", "music_presence_pct",
    "n_long_silences", "mean_brightness", "mean_saturation",
    "mean_edge_density", "text_density_proxy",
)

# Per-axis weight in the distance calculation (sum doesn't need to be 1).
_FP_WEIGHTS = {
    "median_shot_s":       1.5,
    "cuts_per_min":        1.5,
    "music_presence_pct":  0.6,
    "n_long_silences":     1.0,
    "mean_brightness":     1.0,
    "mean_saturation":     1.2,
    "mean_edge_density":   1.0,
    "text_density_proxy":  1.5,
}

# Normalisation scales — divide raw diffs by these so axes are
# comparable in Euclidean space.
_FP_SCALES = {
    "median_shot_s":       3.0,
    "cuts_per_min":        8.0,
    "music_presence_pct":  20.0,
    "n_long_silences":     5.0,
    "mean_brightness":     0.20,
    "mean_saturation":     0.15,
    "mean_edge_density":   0.15,
    "text_density_proxy":  40.0,
}


def predict_fingerprint(resolved: ResolvedLook) -> dict:
    """Map a ResolvedLook → approximate forensic fingerprint.

    Used by anti_converge() to ask "if we render this look right now,
    will the fingerprint be too close to the channel's recent average?"
    without actually rendering.
    """
    def g(path: str, default: float = 0.0) -> float:
        v = resolved.get(path, default)
        if v is None:
            return default
        if isinstance(v, dict) and "default" in v:
            v = v["default"]
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    median = g("beat_target", 3.4)
    cuts_per_min = 60.0 / max(0.5, median)
    music_pct = 100.0 * g("music.density_floor", 0.85)
    # silences ≈ scaling on tier3 pocket size
    n_silences = max(0, int(g("silence.tier3_pocket_s", 0.0) * 6.0))
    brightness = 0.40 + g("grade.brightness_lift", 0.0)
    saturation = 0.10 + g("grade.saturation_lift", 0.0)
    edge_density = 0.12 + 0.08 * g("graphic_density", 1.0) - 1.0 * 0.0
    text_density = max(5.0, min(80.0,
        g("text.text_density_target_pct", 20.0)))
    return {
        "median_shot_s":      round(median, 3),
        "cuts_per_min":       round(cuts_per_min, 2),
        "music_presence_pct": round(music_pct, 1),
        "n_long_silences":    n_silences,
        "mean_brightness":    round(brightness, 4),
        "mean_saturation":    round(saturation, 4),
        "mean_edge_density":  round(edge_density, 4),
        "text_density_proxy": round(text_density, 1),
    }


def fp_distance(a: dict, b: dict) -> float:
    """Weighted-Euclidean distance between two fingerprints on the
    `_FP_AXES`.  Symmetric.  Lower = more similar."""
    if not a or not b:
        return float("inf")
    s = 0.0
    for axis in _FP_AXES:
        ax = a.get(axis)
        bx = b.get(axis)
        if ax is None or bx is None:
            continue
        scale = _FP_SCALES.get(axis, 1.0)
        weight = _FP_WEIGHTS.get(axis, 1.0)
        try:
            d = (float(ax) - float(bx)) / scale
        except (TypeError, ValueError):
            continue
        s += weight * d * d
    return s ** 0.5


def anti_converge(resolved: ResolvedLook,
                  channel_name: str,
                  *,
                  history_window: int = 5,
                  min_distance: float = 0.18,
                  max_perturbations: int = 6) -> ResolvedLook:
    """If predicted fingerprint is too close to the channel's recent
    history, perturb the resolved look along the channel's declared
    `forced_drift_axes` until the distance threshold is satisfied (or
    we hit max_perturbations and accept the current shift).
    """
    if not channel_name:
        return resolved
    history = read_recent_history(channel_name, n=history_window)
    if not history:
        return resolved
    recent_fps = [h.get("fp", {}) for h in history if h.get("fp")]
    if not recent_fps:
        return resolved
    predicted = predict_fingerprint(resolved)
    dists = [fp_distance(predicted, h) for h in recent_fps]
    closest = min(dists)
    if closest >= min_distance:
        return resolved                                   # already varied enough
    # Need to drift.  Pull the declared drift axes from the resolved
    # look itself (channel YAML supplied them).
    drift_axes = resolved.get("anti_convergence.forced_drift_axes", [])
    if not drift_axes:
        # Sensible fallback: shift the most common visible dimensions
        drift_axes = ["beat_target", "music.intensity",
                      "silence.tier3_pocket_s", "graphic_density"]
    # We don't actually mutate the frozen ResolvedLook — instead we
    # produce a NEW one with perturbed knobs.
    knobs = dict(resolved.knobs)
    rng = random.Random(int(time.time() * 1000) & 0x7FFFFFFF)
    for _ in range(max_perturbations):
        axis = rng.choice(drift_axes)
        knobs = _shift_axis(knobs, axis, magnitude=0.15, rng=rng)
        candidate = ResolvedLook(
            name=resolved.name, label=resolved.label,
            version=resolved.version, knobs=knobs)
        predicted = predict_fingerprint(candidate)
        closest = min(fp_distance(predicted, h) for h in recent_fps)
        if closest >= min_distance:
            return candidate
    # gave up — accept the partially-shifted candidate
    return ResolvedLook(
        name=resolved.name, label=resolved.label,
        version=resolved.version, knobs=knobs)


def _shift_axis(knobs: dict, dotted: str,
                magnitude: float, rng: random.Random) -> dict:
    """Mutate a copy of `knobs`: nudge the value at `dotted` by ±magnitude
    of the current value.  No-op for non-numeric axes."""
    out = json.loads(json.dumps(knobs, default=str))    # deep copy
    parts = dotted.split(".")
    node: Any = out
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            return knobs
        node = node[p]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        return knobs
    cur = node[leaf]
    if isinstance(cur, dict) and "default" in cur:
        # Re-sample the weighted bag
        try:
            bag = WeightedBag(cur.get("_bag", {}))
            cur["default"] = bag.sample(rng)
            node[leaf] = cur
        except Exception:                                          # noqa: BLE001
            pass
        return out
    try:
        v = float(cur)
        delta = magnitude * v * rng.choice([-1.0, 1.0])
        node[leaf] = round(v + delta, 4)
    except (TypeError, ValueError):
        # non-numeric — leave alone
        pass
    return out


# =====================================================================
# 8a. AUTO-LOOK (DOC_001) — niche-aware preset selection when no channel
#     is configured.  Without this, resolve_look() returns None for every
#     channel-less brief and the whole pipeline collapses to ONE fixed
#     neutral look → every video looks the same (the "template feel" root
#     cause).  Here we map the brief's topic to one of the curated look
#     presets so each video adopts a distinct, niche-appropriate cinematic
#     identity, while a per-brief hash picks deterministically among the
#     candidates (two different briefs in the same niche still differ via
#     seeded sampling of the preset's ranges).
# =====================================================================

# Topic keyword → niche, WEIGHTED (look selection only; NOT footage gating).
# DOC_019: specific CRIME / INTELLIGENCE / MYSTERY signals are weighted
# ABOVE generic biography markers ("rise and fall", "the story of") so a
# CRIME biography (Pablo Escobar / cartel boss / mafia don / fraudster)
# lands in true_crime — NOT generic biography or historical-epic. Business
# biographies keep their own specific signals (separate from criminal ones).
# classify_brief_niche() sums the weights of matched keywords (word-boundary)
# and picks the highest — deterministic.
_AUTO_LOOK_NICHE_WORDS: dict = {
    "true_crime": [
        ("cartel", 2.6), ("drug lord", 2.6), ("drug baron", 2.6), ("kingpin", 2.6),
        ("narco", 2.6), ("cocaine", 2.0), ("heroin", 2.0), ("drug trafficking", 2.4),
        ("mafia", 2.6), ("mob boss", 2.6), ("crime boss", 2.6), ("crime lord", 2.6),
        ("crime family", 2.6), ("crime syndicate", 2.6), ("organized crime", 2.6),
        ("criminal empire", 2.6), ("criminal underworld", 2.6), ("underworld", 2.2),
        ("gangster", 2.3), ("mobster", 2.3), ("serial killer", 2.6), ("hitman", 2.3),
        ("murder", 1.9), ("murderer", 2.0), ("homicide", 2.0), ("killer", 1.6),
        ("heist", 2.2), ("robbery", 1.8), ("smuggling", 1.9), ("trafficking", 1.9),
        ("fraudster", 2.3), ("ponzi", 2.4), ("racketeering", 2.4), ("extortion", 2.2),
        ("embezzle", 2.0), ("fraud", 1.5), ("scam", 1.5), ("assassination", 1.6),
        ("kidnapping", 1.8), ("manhunt", 2.0), ("most wanted", 2.2),
        ("notorious criminal", 2.4), ("crime", 1.3), ("criminal", 1.5),
        ("gang", 1.3), ("prison", 1.0), ("detective", 1.4), ("verdict", 1.2),
    ],
    "spy_intel": [
        ("espionage", 2.6), ("spy", 2.2), ("spies", 2.2), ("secret agent", 2.4),
        ("double agent", 2.6), ("intelligence agency", 2.4),
        ("intelligence operative", 2.6), ("intelligence officer", 2.4),
        ("cia", 2.0), ("kgb", 2.2), ("mossad", 2.2), ("mi6", 2.2), ("mi5", 2.2),
        ("nsa", 2.0), ("fbi", 1.6), ("stasi", 2.2), ("covert", 2.0),
        ("undercover", 2.0), ("defector", 2.0), ("informant", 1.8),
        ("counterintelligence", 2.6), ("surveillance", 1.6), ("classified", 1.5),
        ("cold war", 1.5), ("operative", 1.6), ("infiltrate", 1.8), ("sabotage", 1.8),
    ],
    "mystery": [
        ("unsolved", 2.4), ("unexplained", 2.2), ("disappearance", 2.4),
        ("vanished", 2.4), ("vanishing", 2.4), ("mystery", 2.0), ("mysterious", 1.8),
        ("conspiracy", 1.8), ("cover-up", 2.0), ("cover up", 2.0), ("enigma", 1.8),
        ("the secret", 1.3), ("strange", 1.1), ("bizarre", 1.4), ("paranormal", 2.0),
        ("haunting", 1.6), ("cryptic", 1.8), ("riddle", 1.6),
    ],
    "history": [
        ("ancient", 2.0), ("dynasty", 2.0), ("medieval", 2.0), ("civilization", 2.2),
        ("the roman empire", 2.4), ("roman empire", 2.4), ("rome", 1.5), ("roman", 1.4),
        ("egypt", 1.5), ("pharaoh", 2.0), ("mesopotamia", 2.2), ("byzantine", 2.2),
        ("renaissance", 2.0), ("colonial", 1.7), ("world war", 2.0), ("civil war", 2.0),
        ("revolution", 1.5), ("kingdom", 1.4), ("the fall of", 1.0), ("rise of the", 0.9),
        ("emperor", 1.5), ("conquest", 1.8), ("the empire", 1.2), ("middle ages", 2.2),
        ("antiquity", 2.2),
        # RC4 — bare war/conflict signals (MODEST weight) so a documentary war
        # brief lands on history (or geopolitics) instead of silently defaulting
        # to business. Kept low so a specific crime/spy/business term still wins.
        ("war", 1.6), ("invasion", 1.7), ("invaded", 1.6), ("ceasefire", 1.8),
        ("frontline", 1.7), ("offensive", 1.5), ("military", 1.2),
    ],
    "business": [
        ("startup", 2.2), ("corporation", 1.9), ("ceo", 1.9), ("entrepreneur", 2.2),
        ("billionaire", 1.7), ("tycoon", 1.9), ("mogul", 1.9), ("magnate", 2.0),
        ("business empire", 2.2), ("the company", 1.5), ("company", 1.3),
        ("bankruptcy", 1.8), ("stock market", 2.0), ("wall street", 2.0),
        ("monopoly", 1.8), ("ipo", 2.2), ("the brand", 1.6), ("tech giant", 2.2),
        ("founder", 1.4), ("industry", 1.3), ("acquisition", 1.8), ("revenue", 1.6),
        ("the collapse of", 1.0),
    ],
    "science": [
        ("physics", 2.2), ("quantum", 2.4), ("the universe", 2.0), ("outer space", 2.0),
        ("astronomy", 2.2), ("astrophysics", 2.4), ("biology", 2.0), ("chemistry", 2.0),
        ("scientific", 1.8), ("the science of", 2.0), ("black hole", 2.2),
        ("evolution", 1.8), ("the brain", 1.8), ("neuroscience", 2.2), ("genome", 2.0),
        ("particle", 2.0), ("experiment", 1.5), ("how does", 1.3), ("why does", 1.3),
        ("relativity", 2.2),
    ],
    "geopolitics": [
        ("geopolitics", 2.6), ("geopolitical", 2.6), ("the border", 1.7),
        ("territory", 1.6), ("the conflict", 1.6), ("map of", 1.8), ("alliance", 1.6),
        ("superpower", 2.0), ("strait of", 2.0), ("pipeline", 1.8), ("invaded", 1.7),
        ("annexed", 2.0), ("sanctions", 1.4), ("trade war", 2.0), ("nato", 1.7),
        ("diplomacy", 1.8), ("proxy war", 2.0),
        # RC4 — bare war/geopolitics signals (MODEST weight). A war/Gulf brief
        # should resolve to geopolitics/history, never the business default.
        ("war", 1.6), ("invasion", 1.7), ("ceasefire", 1.8), ("frontline", 1.7),
        ("offensive", 1.5), ("the gulf", 1.5), ("military", 1.2),
    ],
    # GENERIC biography markers — LOW weight. They apply to ANY person story
    # (crime, business, art, politics) so they must NOT outweigh a specific
    # crime/spy/business signal. Biography wins only when nothing specific does.
    "biography": [
        ("biography", 1.2), ("the life of", 1.0), ("life story", 1.0), ("biopic", 1.4),
        ("rise and fall", 0.6), ("the story of", 0.5), ("the man who", 0.7),
        ("the woman who", 0.7), ("legend of", 0.7), ("the legacy of", 0.8),
        ("portrait of", 0.8), ("the untold story", 0.6),
    ],
}

# Niche → ordered candidate presets (must exist in vidlore/look_presets/).
_AUTO_LOOK_PRESETS: dict = {
    "true_crime":  ["true_crime", "midnight_pacific"],
    "spy_intel":   ["midnight_pacific", "true_crime"],
    "history":     ["amber_chronicles", "netflix_epic"],
    "biography":   ["netflix_epic", "amber_chronicles"],
    "business":    ["atlas_explained", "netflix_epic"],
    "science":     ["atlas_explained"],
    "geopolitics": ["atlas_explained", "midnight_pacific"],
    "mystery":     ["midnight_pacific", "true_crime"],
    # No clear niche → still de-template by spreading across distinct looks.
    "_default":    ["netflix_epic", "amber_chronicles", "atlas_explained",
                    "true_crime", "midnight_pacific", "standard"],
}


def auto_look_disabled() -> bool:
    """VIDLORE_AUTO_LOOK=0/false/no/off → restore exact legacy behaviour."""
    return os.environ.get("VIDLORE_AUTO_LOOK", "1").strip().lower() in (
        "0", "false", "no", "off")


def classify_brief_niche(brief) -> str:
    """Cheap local no-API topic classifier for LOOK selection only.
    Returns a niche key or '_default'.  Word-boundary matched so short
    tokens don't false-match inside other words."""
    text = ((getattr(brief, "title", "") or "") + " "
            + (getattr(brief, "prompt", "") or "")).lower()
    if not text.strip():
        return "_default"
    scores: dict = {}
    best, best_score = "_default", 0.0
    for niche, rules in _AUTO_LOOK_NICHE_WORDS.items():
        score = 0.0
        for kw, wt in rules:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                score += wt
        scores[niche] = score
        if score > best_score + 1e-9:         # strict > → first niche wins ties
            best, best_score = niche, score
    # RC4 — LOW-CONFIDENCE / war-doc guard. The silent "business" default
    # mis-routed war/geopolitics docs. When business only wins by a THIN margin
    # over a documentary niche (history/geopolitics), prefer the documentary
    # niche; a decisive business brief (revenue/IPO/CEO) keeps its margin and is
    # untouched. A clearly weak business signal also yields to a present
    # documentary signal. Crime/spy/etc. winners are never overridden.
    if best == "business":
        doc_best, doc_score = "", 0.0
        for _n in ("history", "geopolitics"):
            if scores.get(_n, 0.0) > doc_score:
                doc_best, doc_score = _n, scores[_n]
        if doc_best and (best_score < 2.4 or doc_score >= best_score - 0.6):
            return doc_best
    return best


def auto_look_preset(brief) -> Optional[str]:
    """Pick a preset name for a channel-less brief, or None to keep the
    exact legacy (no-look) path.  Deterministic per brief."""
    if auto_look_disabled():
        return None
    niche = classify_brief_niche(brief)
    cands = _AUTO_LOOK_PRESETS.get(niche) or _AUTO_LOOK_PRESETS["_default"]
    try:
        avail = set(list_presets())
    except Exception:                                              # noqa: BLE001
        avail = set()
    cands = [c for c in cands if c in avail] or ["standard"]
    idx = int(_hash_brief(brief), 16) % len(cands)
    return cands[idx]


# =====================================================================
# 8b. DOC_002 — per-video IDENTITY variation within a look.
#     Two videos in the SAME niche pick the same preset (shared family
#     feel) — but should NOT be pixel-identical.  Here we vary the most
#     visible identity surfaces (map style, chart style, accent colour)
#     per-brief, drawing from niche-appropriate, tasteful pools.  All
#     values are the renderer's REAL accepted tokens (unknown → graceful
#     fallback) and only use bundled-safe assets.  Seed is decorrelated
#     from the preset pick so map/chart/accent don't move in lockstep.
# =====================================================================

# map_style accepted by mapgen.STYLE_PRESETS: satellite|dark|political|
# parchment|tactical|news  (unknown → satellite).
_AUTO_LOOK_MAP_STYLES: dict = {
    "true_crime":  ["dark", "tactical", "satellite"],
    "spy_intel":   ["tactical", "dark", "political", "satellite"],
    "mystery":     ["dark", "satellite", "tactical"],
    "history":     ["parchment", "political"],
    "biography":   ["parchment", "political", "satellite"],
    "business":    ["political", "news", "satellite"],
    "science":     ["political", "satellite"],
    "geopolitics": ["political", "tactical", "satellite"],
    "_default":    ["satellite", "dark", "political"],
}
# chart_style accepted by _look_card_palette: navy_grid|white_dataviz|
# sepia_textured|neutral.
_AUTO_LOOK_CHART_STYLES: dict = {
    "true_crime":  ["navy_grid", "neutral"],
    "spy_intel":   ["navy_grid", "neutral"],
    "mystery":     ["navy_grid", "neutral"],
    "history":     ["sepia_textured", "neutral"],
    "biography":   ["sepia_textured", "navy_grid"],
    "business":    ["white_dataviz", "navy_grid"],
    "science":     ["white_dataviz", "neutral"],
    "geopolitics": ["white_dataviz", "navy_grid"],
    "_default":    ["navy_grid", "neutral"],
}
# Per-niche accent COLOUR families (emotionally matched). A small seeded
# jitter gives per-video variation WITHIN the family — so two crime docs
# aren't the exact same red.  Kept tasteful (anchors are saturated but
# not neon; jitter is +/-10 per channel).
_AUTO_LOOK_ACCENTS: dict = {
    "true_crime":  [[210, 55, 55], [196, 72, 48], [184, 58, 64]],
    "spy_intel":   [[110, 150, 178], [96, 132, 150], [122, 142, 165]],
    "mystery":     [[120, 145, 170], [138, 120, 165], [108, 142, 150]],
    "history":     [[206, 158, 80], [196, 150, 92], [188, 146, 70]],
    "biography":   [[200, 165, 90], [208, 150, 84], [190, 158, 96]],
    "business":    [[235, 170, 70], [240, 150, 70], [225, 178, 60]],
    "science":     [[90, 185, 200], [110, 175, 210], [80, 195, 180]],
    "geopolitics": [[220, 120, 70], [210, 140, 80], [200, 132, 95]],
    "_default":    [[230, 175, 80], [200, 110, 90], [120, 150, 175]],
}


def _auto_look_identity(brief, niche: str) -> dict:
    """Per-video identity overrides (map_style, chart_style, accent_rgb)
    so two same-niche videos differ on the most visible surfaces.
    Deterministic per brief, decorrelated from the preset pick."""
    seed = (int(_hash_brief(brief), 16) ^ 0x6D5A3C) & 0x7FFFFFFF
    rng = random.Random(seed)
    maps = _AUTO_LOOK_MAP_STYLES.get(niche) or _AUTO_LOOK_MAP_STYLES["_default"]
    charts = _AUTO_LOOK_CHART_STYLES.get(niche) or _AUTO_LOOK_CHART_STYLES["_default"]
    accents = _AUTO_LOOK_ACCENTS.get(niche) or _AUTO_LOOK_ACCENTS["_default"]
    base = rng.choice(accents)
    accent = [max(20, min(255, c + rng.randint(-10, 10))) for c in base]
    return {
        "map_style": rng.choice(maps),
        "chart_style": rng.choice(charts),
        "accent_rgb": accent,
    }


def _auto_editorial_recipe(brief, niche: str) -> Optional[dict]:
    """Layer-2 editorial recipe for AUTO-LOOK: a deterministic per-video
    pick within the niche envelope (editorial_recipe.py).  Returns the
    recipe dict, or None when the engine is disabled / unavailable so the
    caller falls back to DOC_002 identity variation.

    A LOCKED recipe (Review-Editor 'Lock Look', carried on
    ``brief.extra['editorial_recipe_lock']``) is honoured verbatim so
    re-renders keep the same look.  Otherwise a fresh recipe is generated
    at the requested variation index (``brief.extra['editorial_variation']``,
    bumped by the Review-Editor 'Generate New Variation' control).

    SELECTIVE overrides (``brief.extra['recipe_overrides']`` — a dict like
    ``{"map_style": "...", "accent": [r,g,b], "beat_target": 3.2}``) are
    applied ON TOP of the generated/locked recipe, so the user can pin one
    axis (e.g. map style) while everything else stays auto.  All inputs
    are read defensively so an older brief.json never breaks."""
    try:
        from . import editorial_recipe as _er
    except Exception:                                              # noqa: BLE001
        return None
    if _er.disabled():
        return None
    extra = getattr(brief, "extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    locked = extra.get("editorial_recipe_lock")
    if isinstance(locked, dict) and locked.get("niche"):
        recipe = dict(locked)
    else:
        try:
            variation = int(extra.get("editorial_variation", 0) or 0)
        except Exception:                                          # noqa: BLE001
            variation = 0
        try:
            recipe = _er.generate_recipe(brief, niche or "_default",
                                         channel=None, variation=variation)
        except Exception:                                          # noqa: BLE001
            return None
    # selective per-axis overrides (Review-Editor) on top of the recipe
    ov = extra.get("recipe_overrides")
    if isinstance(ov, dict) and recipe is not None:
        for k, v in ov.items():
            if v is not None:
                recipe[k] = v
        recipe["_overridden"] = sorted(k for k, v in ov.items() if v is not None)
    return recipe


# =====================================================================
# 8. Top-level resolve_look() — the single entry point
# =====================================================================

def resolve_look(brief, *, channel_name: Optional[str] = None,
                 cli_overrides: Optional[dict] = None,
                 ) -> Optional[ResolvedLook]:
    """Single entry point for the entire pipeline.

    Returns None when there is no channel and the brief has no inline
    look overrides — pipeline then runs the legacy resolve_style path
    untouched (backward-compatible invariant I-1).

    Otherwise: build the LookDNA stack (standard floor → preset →
    channel → brief overrides → CLI overrides), sample with a seed
    derived from (channel cursor + brief hash), anti-converge against
    the channel's recent history, and return the frozen ResolvedLook.
    """
    # Channel name precedence: CLI > brief.channel
    name = (channel_name
            or getattr(brief, "channel", None)
            or getattr(brief, "channel_name", None)
            or "").strip().lower() or None

    brief_overrides = getattr(brief, "look_overrides", None) or {}
    preset_name = getattr(brief, "look_preset", None) or ""

    # No channel + no inline preset/overrides → AUTO-LOOK (DOC_001):
    # auto-select a niche-appropriate preset so each channel-less video
    # adopts a distinct cinematic identity instead of one fixed neutral
    # default.  VIDLORE_AUTO_LOOK=0 restores exact legacy (returns None).
    _auto_mode = False
    _auto_niche = ""
    if not name and not brief_overrides and not preset_name:
        auto = auto_look_preset(brief)
        if not auto:
            _record_look_decision(None, "", "legacy", False)
            return None
        preset_name, _auto_mode = auto, True
        _auto_niche = classify_brief_niche(brief)

    # 1. Standard floor (always exists)
    look = LookDNA.from_preset("standard")

    # 2. Named preset if requested (brief.look_preset)
    if preset_name:
        try:
            preset = LookDNA.from_preset(preset_name)
            look = look.merge(preset)
        except FileNotFoundError:
            pass   # ignore unknown preset, log later

    # 3. Channel YAML (and its declared `inherits` chain)
    if name:
        ch_path = _channels_root() / f"{_safe_name(name)}.yaml"
        if ch_path.exists():
            channel_dna = LookDNA.from_yaml_file(ch_path)
            if channel_dna.inherits:
                try:
                    base = LookDNA.from_preset(channel_dna.inherits)
                    look = look.merge(base)
                except FileNotFoundError:
                    pass
            look = look.merge(channel_dna)

    # 4. Brief-level inline overrides
    if brief_overrides:
        look = look.merge(LookDNA.from_dict(
            {"name": "brief_overrides", **brief_overrides}))

    # 5. CLI overrides (highest priority)
    if cli_overrides:
        look = look.merge(LookDNA.from_dict(
            {"name": "cli_overrides", **cli_overrides}))

    # 5b. AUTO-LOOK cost-safety: keep macro scene COUNT at the brief's
    # intent (the duration system stays in control).  Auto-look varies
    # the VISUAL / AUDIO / MOTION identity, NOT how many scenes the LLM
    # generates — a preset's 2.4× (atlas) or 0.6× (amber) scene_count_mult
    # would change LLM cost + video length unexpectedly for a user who
    # never opted into a channel.  Force neutral 1.0 in auto mode only.
    if _auto_mode:
        # preserve the chosen preset's name (downstream logs it as the
        # "editor signature"; a merge that set name would clobber it).
        #
        # EDITORIAL RECIPE (niche = a SOFT prior, NOT a fixed template):
        # build a DETERMINISTIC per-video recipe within the niche's
        # ENVELOPE (editorial_recipe.py) and let it drive the visible
        # identity + per-video pacing.  Same niche + different brief →
        # different recipe, so 20 spy docs each feel custom-edited instead
        # of sharing one preset.  The recipe SUBSUMES DOC_002 identity
        # (map / chart / accent) and adds tone + beat_target.  DOC_002
        # stays as the graceful fallback when the recipe engine is
        # disabled (VIDLORE_EDITORIAL_RECIPE=0) or unavailable.
        _recipe = _auto_editorial_recipe(brief, _auto_niche)
        if _recipe is not None:
            try:
                from . import editorial_recipe as _er
                _ident = {k: v for k, v
                          in _er.to_look_overrides(_recipe).items()
                          if v is not None}
            except Exception:                                      # noqa: BLE001
                _recipe, _ident = None, _auto_look_identity(brief, _auto_niche)
        else:
            _ident = _auto_look_identity(brief, _auto_niche)
        _record_look_recipe(_recipe)
        look = look.merge(LookDNA.from_dict(
            {"name": look.name, "scene_count_mult": 1.0, **_ident}))

    # 6. SAMPLE — deterministic seed from (channel cursor + brief hash)
    cursor = _read_cursor(name) if name else 0
    brief_hash = _hash_brief(brief)
    seed = (hash((cursor, brief_hash)) & 0x7FFFFFFF) or 1
    rng = random.Random(seed)
    resolved = look.sample(rng)

    # 7. ANTI-CONVERGE against history (only when we have a channel)
    if name:
        ac = resolved.get("anti_convergence", {}) or {}
        if ac and ac.get("history_window", 0):
            resolved = anti_converge(
                resolved, name,
                history_window=int(ac.get("history_window", 5)),
                min_distance=float(ac.get("min_distance", 0.18)),
            )

    # DOC_012 — record the decision so render_meta + the Review Editor can
    # show the "editor signature" (which look, detected niche, and why).
    _record_look_decision(
        resolved.name, _auto_niche,
        "auto" if _auto_mode else ("channel" if name else "override"),
        _auto_mode)
    return resolved


_LAST_LOOK_DECISION: dict = {}


def _record_look_decision(look_name, niche, source, auto) -> None:
    global _LAST_LOOK_DECISION
    _LAST_LOOK_DECISION = {
        "look": look_name, "niche": niche or "",
        "source": source, "auto": bool(auto),
    }


def last_look_decision() -> dict:
    """Most recent resolve_look() decision — for render_meta + the Review
    Editor 'editor signature' (which look, detected niche, why)."""
    return dict(_LAST_LOOK_DECISION)


_LAST_LOOK_RECIPE: dict = {}


def _record_look_recipe(recipe) -> None:
    global _LAST_LOOK_RECIPE
    _LAST_LOOK_RECIPE = dict(recipe) if isinstance(recipe, dict) else {}


def last_look_recipe() -> dict:
    """Most recent AUTO-LOOK editorial RECIPE (editorial_recipe.py) — the
    deterministic per-video pick within the niche envelope.  Consumed by
    render_meta + the Review-Editor recipe summary / 'Lock Look' control.
    Empty dict when the recipe engine is disabled or no auto-look ran."""
    return dict(_LAST_LOOK_RECIPE)


def commit_render(channel_name: Optional[str], brief, resolved: ResolvedLook,
                  fingerprint: dict) -> None:
    """Called by the pipeline AFTER a render completes.  Logs the
    fingerprint + bumps the variation cursor so the next render uses
    a different sample of the ranges."""
    if not channel_name:
        return
    name = _safe_name(channel_name)
    entry = {
        "ts": int(time.time()),
        "brief_hash": _hash_brief(brief),
        "fp": fingerprint,
        "look_name": resolved.name if resolved else "",
    }
    append_history(name, entry)
    stride = 1
    try:
        stride = int(resolved.get("variation.cursor_stride", 1))
    except Exception:                                              # noqa: BLE001
        pass
    _bump_cursor(name, stride=stride)


def _hash_brief(brief) -> str:
    title = getattr(brief, "title", "") or ""
    prompt = getattr(brief, "prompt", "") or ""
    h = hashlib.sha1((title + "\n" + prompt[:400]).encode("utf-8"))
    return h.hexdigest()[:8]


# =====================================================================
# 9. Built-in "standard" floor — guaranteed to load even without YAML
# =====================================================================

def _STANDARD_FLOOR_DICT() -> dict:
    """The neutral baseline.  Every field has a safe default so
    every downstream getter can rely on it.  Hard-coded so we never
    hit a missing-file failure for the floor."""
    return {
        "name": "standard",
        "label": "Standard (neutral baseline)",
        "version": 1,
        "tone": "neutral",
        "beat_target": {"min": 3.2, "max": 3.6},
        "beat_min":    {"min": 2.2, "max": 2.6},
        "hook_seconds": 42,
        "hook_beat_mult": {"min": 0.60, "max": 0.70},
        "climax_silence_mult": {"min": 1.00, "max": 1.10},
        "dissolve_gap": {"min": 3, "max": 4},
        "dissolve_drop": 2,
        "camera_easing": "ease_out_cubic",
        "drift_scale": {"min": 0.90, "max": 1.05},
        "hold_bias":   {"min": 1.00, "max": 1.20},
        "handheld_micro_float": True,
        "push_in_impact": {"min": 0.95, "max": 1.05},
        "graphic_density": {"min": 1.0, "max": 1.1},
        "density_per_min_max": {"min": 2, "max": 3},
        "fav_templates": [],
        "avoid_templates": [],
        # ── SCRIPT-AWARE PACING (P3) ────────────────────────────────
        # `scene_count_mult` multiplies the brief's target story-point
        # range BEFORE the LLM sees it.  Fast channels (Atlas) expand
        # the same brief into many short scenes; slow channels (Amber)
        # collapse into a few long ones.  This is the single biggest
        # macro-editorial lever — without it, every channel inherits
        # the brief's default scene count and pacing skeleton matches.
        # `pacing_hint` is a one-sentence editorial instruction
        # injected into the LLM prompt so the SCENE STRUCTURE itself
        # (sentences per scene, sentence length, intensity arc) bends
        # toward the channel's editorial identity, not just the count.
        "scene_count_mult": 1.0,
        "pacing_hint": "",
        "chart_style": "sans_clinical",
        "map_style":   "satellite",
        "annotation_voice": "clinical",
        "accent_rgb": [255, 210, 90],
        "grade": {
            "brightness_lift": {"min": -0.02, "max": 0.02},
            "saturation_lift": {"min": -0.04, "max": 0.02},
            "shadow_lift":     {"min": 0.0, "max": 0.03},
            "highlight_roll":  {"min": -0.02, "max": 0.0},
            "warm_shadow_split": 0.012,
        },
        "ai": {
            "per_video_floor": 0,
            "per_video_ceiling": {"min": 2, "max": 5},
            "fire_on_roles": ["climax", "reveal", "name_reveal"],
            "fire_on_intensity": 4,
            "portrait_style": "magnum_natgeo",
            "grain_strength": {"min": 0.06, "max": 0.10},
            "diffusion_fallback": False,
        },
        "archival": {"enabled": False, "bias_on_dates_before": 1950,
                     "source_priority": ["pexels_historical"]},
        "music": {
            "category_bias": [],
            "intensity": {"min": 0.95, "max": 1.10},
            "density_floor": {"min": 0.55, "max": 0.75},
            "swell_on_climax": "restrained",
            "cue_min_seconds": 14,
            "track_repeat_window_days": 7,
        },
        "silence": {
            "tier1_pocket_s": 0.0,
            "tier2_pocket_s": {"min": 0.45, "max": 0.55},
            "tier3_pocket_s": {"min": 0.95, "max": 1.15},
            "tier3_stretch_s": {"min": 0.35, "max": 0.55},
            "pre_roll_t3_s":   {"min": 0.45, "max": 0.55},
        },
        "sfx": {
            "whoosh_min_gap_s": {"min": 10, "max": 14},
            "boom_min_gap_s":   {"min": 28, "max": 36},
            "amplitude":        {"min": 0.45, "max": 0.62},
            "reveal_stinger":   "none",
        },
        "text": {
            "emphasis_fire_prob_density":   1.0,
            "emphasis_fire_prob_restraint": {"min": 0.45, "max": 0.60},
            "scrim_under_busy_shots": True,
            "text_density_target_pct": {"min": 12, "max": 22},
        },
        "fonts": {
            "display": ["VidloreSans"],
            "body":    ["VidloreSans"],
            "weight_bias": "bold",
            "letter_spacing_caps": 0.03,
        },
        "lower_third": {"variant": "scrim_panel_v1",
                        "duration_s": {"min": 4.0, "max": 5.5},
                        "accent_underline": True},
        "name_reveal": {"layout": "portrait_left_text_right_v1",
                        "serif_name": True, "gold_underline": True,
                        "scrim_panel_behind_text_only": True},
        "captions": {"enabled_default": False,
                     "ass_style_overrides": {}},
        "intro": {"cold_open_s": 6, "title_hold_s": 2.0,
                  "hook_compression": 1.4},
        "climax": {"hold_s": {"min": 6, "max": 12},
                   "pre_silence_s": {"min": 1.0, "max": 1.4}},
        "animation": {"default_easing": "ease_out_cubic",
                      "stagger_ms": {"min": 80, "max": 140},
                      "philosophy": "cinematic_restraint"},
        "voice": {"backend": "edge_tts", "voice_id": "en-US-GuyNeural",
                  "rate_pct": 0, "pitch_st": 0},
        "anti_convergence": {
            "history_window": 5, "min_distance": 0.18,
            "forced_drift_axes": [
                "beat_target", "music.intensity",
                "silence.tier3_pocket_s", "graphic_density"],
        },
        "variation": {"cursor_stride": 1, "deterministic": True},
    }


# =====================================================================
# 10. Channel management helpers (used by `vidlore channel ...` CLI)
# =====================================================================

def list_channels() -> list[str]:
    root = _channels_root()
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))


def list_presets() -> list[str]:
    here = Path(__file__).parent / "look_presets"
    if not here.exists():
        return ["standard"]
    return sorted(p.stem for p in here.glob("*.yaml"))


def channel_path(name: str) -> Path:
    return _channels_root() / f"{_safe_name(name)}.yaml"


def init_channel(name: str, *, inherits: str = "standard",
                 force: bool = False) -> Path:
    """Create a new ``<name>.yaml`` seeded from a preset.  Returns the
    path.  Refuses to overwrite unless ``force=True``."""
    p = channel_path(name)
    if p.exists() and not force:
        raise FileExistsError(
            f"Channel '{name}' already exists at {p}. Pass force=True to overwrite.")
    base = LookDNA.from_preset(inherits)
    base.name = _safe_name(name)
    base.label = f"{name} (inherits {inherits})"
    base.inherits = inherits
    base.save(p)
    return p


# ─────────────────────────────────────────────────────────────────────
# 11. CONTEXT-VAR PROVIDER  — the keystone for downstream integration
# ─────────────────────────────────────────────────────────────────────
#
# The pipeline calls ``set_current(look)`` right after resolving and
# ``clear_current()`` at the end of the render.  Any renderer anywhere
# in the codebase can then do::
#
#     from .look_dna import current as look_current
#     look = look_current()                # ResolvedLook | None
#     val  = look.get('silence.tier3_pocket_s', 1.0) if look else 1.0
#
# This is by far the smallest-blast-radius integration pattern: zero
# new positional args anywhere, every consumer is a 1-line opt-in,
# legacy (no channel) returns None and the existing code path runs
# untouched.

_CURRENT: Optional[ResolvedLook] = None


def set_current(look: Optional[ResolvedLook]) -> None:
    """Install the active look for this render thread."""
    global _CURRENT
    _CURRENT = look


def clear_current() -> None:
    """Remove the active look (called at end-of-render)."""
    global _CURRENT
    _CURRENT = None


def current() -> Optional[ResolvedLook]:
    """Read the active look.  Returns None when no channel is set —
    every consumer must handle this case by falling back to legacy."""
    return _CURRENT


def look_get(dotted: str, default: Any = None) -> Any:
    """Tiny convenience used at consumer sites:

        from .look_dna import look_get
        gap = look_get('sfx.whoosh_min_gap_s', default=12.0)
    """
    cur = current()
    if cur is None:
        return default
    val = cur.get(dotted, default)
    # collapsed WeightedBag landed as {_bag,...,default}; return the
    # already-realised 'default' key
    if isinstance(val, dict) and "default" in val:
        return val["default"]
    return val if val is not None else default


# Convenience for tests / debugging
__all__ = [
    "Range", "IntRange", "WeightedBag",
    "LookDNA", "ResolvedLook",
    "resolve_look", "commit_render",
    "predict_fingerprint", "fp_distance", "anti_converge",
    "list_channels", "list_presets", "init_channel", "channel_path",
    "read_face_seed", "write_face_seed",
    "set_current", "clear_current", "current", "look_get",
]
