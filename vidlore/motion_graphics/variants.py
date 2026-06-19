"""vidlore/motion_graphics/variants.py — V3.4 SEEDED VISUAL-VARIANT SELECTOR.

Deterministic, anti-repeat selection of a primitive's `layout_variant` per scene.
Pure functions, **never raises**, no I/O. Reuses the existing dispatch slot: the
chosen id flows through `director.plan()` → `inputs["layout"]` → the primitive's
`render(..., layout=...)`, and the deterministic `cache_key` already includes it.

Guarantees
----------
* Same (project seed, channel DNA, primitive, scene index, recent history) →
  the SAME variant (reproducible debugging).
* A DIFFERENT project seed yields tasteful variation wherever a primitive exposes
  ≥2 safe variants. Never uncontrolled randomness (no Math.random / time).
* Anti-repeat: rotate (deterministically) away from the variant used in the recent
  N scenes of the same primitive/family; if only one variant exists, keep it and
  log the reason. Selection runs AFTER the director's eligibility/cap/cooldown gate,
  so density restraint + cooldowns are unchanged.

The selector NEVER touches factual values, dates, labels, routes, coordinates or
scene order — it only picks a visual skin/layout id. A primitive whose `render`
ignores `layout` simply renders its default; the manifest still logs the choice.
"""
from __future__ import annotations

import hashlib

VERSION = "v3.4"

# Reference PRINCIPLE per variant id (grounded, no competitor copying). Logged to
# the manifest as `variant_reference_source` for provenance/audit.
VARIANT_REFERENCES = {
    # statistic_bar_reveal
    "columns": "MagnatesMedia: stats animate INTO frame as rising bars (principle).",
    "hero_bar": "Single-hero stat grammar (V3.3 #3): one dominant value owns the frame.",
    "horizontal_bars": "Ranked-list grammar: label-left / value-right horizontal rows.",
    # comparison_split
    "vs_center": "Documentary side-by-side contrast with a central pivot (principle).",
    "stacked_rows": "Over/under stacked contrast — a second comparison silhouette.",
    # map_route_spread (skin only; coords/labels/order immutable)
    "route_trace": "MagnatesMedia route-draw: a line crawls a period map (principle).",
    "dashed_march": "Supply/advance dash grammar: segmented march along the SAME path.",
    # gold_number_callout
    "hero_center": "Hero-figure: one giant centred numeral.",
    "corner_stat": "Lower-third stat: numeral pinned to a corner over breathing space.",
    # portrait_legend_reveal
    "archival_left": "Archival-left: portrait left, name/kicker block right.",
    # era_stamp_overlay
    "corner_br": "Corner year stamp (default bottom-right).",
    "corner_tr": "Corner year stamp rotated to top-right (anti-repeat alternative).",
}


def _h(*parts) -> int:
    """Stable 32-bit hash of the parts (deterministic, no RNG)."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


class VariantSelector:
    """One per video. Tracks recent variant history for anti-repeat."""

    HISTORY = 4               # remember the last N variants per primitive/family

    def __init__(self, *, project_seed: int = 0, channel_dna: str = "",
                 niche: str = ""):
        self.seed = int(project_seed) & 0xFFFFFFFF
        self.dna = str(channel_dna or niche or "")
        self._by_primitive: dict = {}     # primitive -> [variant ids, recent last]
        self._by_family: dict = {}        # family    -> [variant ids, recent last]

    def select(self, primitive: str, variants, *, family: str = "",
               scene_index: int = 0) -> dict:
        """Pick a variant for `primitive` at `scene_index`. Returns the full
        manifest-evidence dict (always includes `visual_variant_id`)."""
        try:
            vs = [str(v) for v in (variants or []) if v]
            if not vs:
                return self._evidence(primitive, family, "", scene_index,
                                      reason="no_variants", anti="none",
                                      fallback="primitive_default")
            base_i = _h(self.seed, self.dna, primitive, scene_index) % len(vs)
            if len(vs) == 1:
                chosen, reason, anti = vs[0], "single_variant", \
                    "single_variant_no_alternative"
            else:
                # Block ONLY the immediately-previous variant of this primitive
                # (and the family's last skin) — strong enough to kill adjacent
                # repeats, loose enough that the SEED still chooses among the
                # remaining candidates (so a different seed → different variation).
                recent = set(self._by_primitive.get(primitive, [])[-1:]) | \
                    set(self._by_family.get(family, [])[-1:] if family else [])
                order = [vs[(base_i + k) % len(vs)] for k in range(len(vs))]
                chosen = next((c for c in order if c not in recent), order[0])
                if chosen == order[0] and order[0] in recent:
                    anti = "all_recent_kept_seeded"
                elif chosen != order[0]:
                    anti = f"avoided_recent:{order[0]}"
                else:
                    anti = "none"
                reason = f"seeded_pick(base={vs[base_i]},chosen={chosen})"
            # record history
            self._by_primitive.setdefault(primitive, []).append(chosen)
            if family:
                self._by_family.setdefault(family, []).append(chosen)
            return self._evidence(primitive, family, chosen, scene_index,
                                  reason=reason, anti=anti, fallback="")
        except Exception:                                          # noqa: BLE001
            # never break the director: fall back to the default variant
            d = (list(variants)[0] if variants else "")
            return self._evidence(primitive, family, d, scene_index,
                                  reason="selector_error_default", anti="none",
                                  fallback="selector_error")

    def _evidence(self, primitive, family, chosen, scene_index, *,
                  reason, anti, fallback) -> dict:
        return {
            "parent_primitive": primitive,
            "visual_variant_id": chosen,
            "variant_seed": _h(self.seed, primitive, scene_index) & 0xFFFF,
            "variant_selection_reason": reason,
            "variant_reference_source": VARIANT_REFERENCES.get(chosen, ""),
            "variant_recent_history": list(
                self._by_primitive.get(primitive, []))[-self.HISTORY:],
            "anti_repeat_block_reason": anti,
            "fallback_variant_reason": fallback,
        }
