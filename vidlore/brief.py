"""The creation brief = Vidlore's input screen (format, length, mode, the
4-Pillars prompt) plus the duration->content-density rules taken from the
official Vidlore prompting guide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Vidlore's documented duration buckets and how many main story points each
# needs (their "Density Formula"). Used to steer the script length.
# IMP_026 — duration system. `points` = number of major story ACTS (the
# narrative spine). `scenes` = the (decoupled) number of SHOT BEATS the doc is
# cut into — each act contains MANY scenes. `words` = total spoken-word target
# that ACTUALLY drives video length (~2.85 words/sec narration). Previously
# scene-count was hardwired to `points` (4-5) and "1-3 sentences/scene" capped
# a "6-8 min" doc at ~450 words ≈ 2.4 min — half the requested length. Now
# scene-count is its own per-duration target and a validation gate enforces the
# word total (expand if short / compress if long) so a 6-min request is a
# 6-min video.
# IMP_029 — word targets RE-CALIBRATED to the empirically-measured narration
# rate (~3.0-3.1 w/s; the rendered 6-8 doc was 1206 words -> 391.5s = 3.08 w/s).
# Targets now aim at the MIDPOINT minutes of each bucket so a render lands
# CENTRE-range, not the low edge. (Old 2.85-based targets undershot — esp.
# "18-20" at 3000w ≈ 16 min, below the 18-20 request.) words ≈ midpoint_min *
# 60 * WORDS_PER_SECOND.
DURATION_BUCKETS = {
    # quick QA bucket — short briefs for ~2 min renders (dev iteration)
    "1-2": dict(minutes=2, points=(2, 3), scenes=(5, 8), words=290),
    "6-8": dict(minutes=7, points=(4, 5), scenes=(25, 40), words=1300),
    "10-12": dict(minutes=11, points=(7, 8), scenes=(50, 80), words=2050),
    "18-20": dict(minutes=19, points=(12, 15), scenes=(100, 150), words=3450),
    "30-40": dict(minutes=35, points=(20, 30), scenes=(200, 300), words=6300),
}

# Empirically-measured narration rate (rendered doc: 1206 words / 391.5 s =
# 3.08 w/s on edge-tts). 3.0 is a touch conservative so the gate, which expands
# until words >= 0.85*target, never UNDER-shoots even on a slightly slower
# voice. Used for the validation gate's minute estimate + UI.
WORDS_PER_SECOND = 3.0

THEMES = ("crime", "history", "modern", "minimalist", "standard")
FORMATS = ("documentary", "top10")


@dataclass
class Brief:
    title: str
    prompt: str                       # the 4-Pillars creative brief
    fmt: str = "documentary"          # documentary | top10
    duration: str = "6-8"             # one of DURATION_BUCKETS
    theme: str = "history"            # one of THEMES
    voice: str | None = None          # override config voice
    captions: bool = False   # Vidlore runs clean footage; off by default
    music: str | None = None          # path to a local music file (optional)
    voiceover: str | None = None      # path to YOUR own narration audio
    background: str = "auto"          # one of backgrounds.BG_NAMES or "auto"
    style: str = "auto"               # Style Mode: 'auto'|'standard'|'epic'…
    # ── LOOK DNA hooks (see vidlore/look_dna.py) ──────────────────
    # `channel`     = ~/.vidlore/channels/<name>.yaml  is consulted
    # `look_preset` = bundled preset name (overrides channel inherits)
    # `look_overrides` = inline knob overrides (deep-merged on top)
    channel: str | None = None
    look_preset: str | None = None
    look_overrides: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration not in DURATION_BUCKETS:
            raise ValueError(
                "duration must be one of %s" % list(DURATION_BUCKETS)
            )
        if self.theme not in THEMES:
            raise ValueError("theme must be one of %s" % list(THEMES))
        if self.fmt not in FORMATS:
            raise ValueError("fmt must be one of %s" % list(FORMATS))
        # Background validation is deliberately lenient — an unknown
        # name silently falls back to the theme gradient at render time
        # (the renderer never crashes on a stale brief.json from an
        # older build). Lazy-import to avoid a circular dep at module
        # load while the package is being initialised.
        try:
            from .backgrounds import is_valid as _bg_ok
            if self.background and not _bg_ok(self.background):
                self.background = "auto"
        except Exception:                                  # noqa: BLE001
            pass

    @property
    def spec(self) -> dict:
        return DURATION_BUCKETS[self.duration]

    @property
    def target_words(self) -> int:
        return self.spec["words"]

    @property
    def target_points(self) -> tuple[int, int]:
        return self.spec["points"]

    @property
    def target_scenes(self) -> tuple[int, int]:
        """IMP_026 — shot-beat count, DECOUPLED from story acts (`points`).
        Falls back to points for any older bucket dict missing `scenes`."""
        return self.spec.get("scenes", self.spec["points"])

    @property
    def target_minutes(self) -> tuple[float, float]:
        """(lo, hi) minute range parsed from the bucket key (e.g. '6-8')."""
        try:
            a, b = self.duration.split("-")
            return (float(a), float(b))
        except Exception:                                  # noqa: BLE001
            m = float(self.spec.get("minutes", 5))
            return (m * 0.85, m * 1.15)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Brief":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
