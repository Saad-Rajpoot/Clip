"""EDITORIAL RECIPE — niche as a SOFT prior, not a fixed template.

Two layers:
  • NICHE_ENVELOPES (Layer 1): per-niche ALLOWED sets/ranges — the tasteful
    boundary of what a documentary in that niche may do (mood, palette family,
    map/chart options, grade range, transition families, pacing range, density,
    music behaviour, lower-third families, ...).  NOT point values.
  • generate_recipe() (Layer 2): a DETERMINISTIC per-video pick of ONE concrete
    value from each set, seeded from the brief.  Same niche + different video →
    different recipe; same video re-rendered → identical recipe.  An anti-
    repetition history nudges new recipes away from recently-used ones.

This generalises DOC_002 (which varies map/chart/accent) to ALL identity +
behavioural axes, so 20 spy docs each get a different editorial recipe instead
of the same fixed preset.

Self-contained (stdlib + look-preset-style data).  `to_look_overrides()` emits
the subset of axes the renderer already consumes; the rest are recorded for
incremental wiring + the Review-Editor recipe summary.  Env-gated:
VIDLORE_EDITORIAL_RECIPE=0 disables (callers fall back to legacy auto-look).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

_HOME = Path(os.path.expanduser("~"))
_HISTORY = _HOME / ".vidlore" / "editorial_history.json"


# ── Layer 1: niche editorial envelopes (allowed sets / ranges) ──────────
# accents = curated palette FAMILY (tasteful anchors, jittered per video);
# transition_families = LIST OF SUBSETS (a video uses one subset, not all);
# ranges are [lo, hi] sampled uniformly.  Pools are seeded from the current
# look system + DOC_002 and will be widened from the forensic primitives.
NICHE_ENVELOPES: dict = {
    "true_crime": {
        "tones": ["tense", "dark"],
        "accents": [[210, 55, 55], [196, 72, 48], [184, 58, 64], [176, 66, 52]],
        "map_styles": ["dark", "tactical", "satellite"],
        "chart_styles": ["navy_grid", "neutral"],
        "grade_sat": [0.78, 0.90], "beat_target": [2.7, 3.4],
        "hold_mult": [1.05, 1.25], "music_bed": [0.80, 0.92], "density": [0.9, 1.15],
        "transition_families": [
            ["evidence_flash", "blur_cut", "page_wipe", "cut"],
            ["archive_flash", "blur_cut", "cut", "dissolve"]],
        "motion_bias": ["push_hold", "investigative_pan", "impact_push"],
        "lower_third_families": ["scrim_panel", "case_file_tag", "redacted_strip"],
        "subtitle_styles": ["bold_lower", "mono_lower"],
        "sfx_restraint": ["diegetic_light", "tension_bed"],
        "source_priority": ["archive", "ai", "web_image", "stock"],
    },
    "spy_intel": {
        "tones": ["tense", "dark"],
        "accents": [[110, 150, 178], [96, 132, 150], [122, 142, 165], [88, 120, 140]],
        "map_styles": ["tactical", "dark", "political", "satellite"],
        "chart_styles": ["navy_grid", "neutral"],
        "grade_sat": [0.74, 0.86], "beat_target": [3.2, 4.2],
        "hold_mult": [1.15, 1.35], "music_bed": [0.72, 0.86], "density": [0.8, 1.0],
        "transition_families": [
            ["slow_dissolve", "dissolve", "cut", "page_wipe"],
            ["blur_cut", "dissolve", "cut", "geo_push"]],
        "motion_bias": ["slow_push", "controlled_static", "drift"],
        "lower_third_families": ["scrim_panel", "classified_strip", "mono_tag"],
        "subtitle_styles": ["mono_lower", "bold_lower"],
        "sfx_restraint": ["diegetic_light", "tension_bed", "sparse"],
        "source_priority": ["archive", "ai", "stock", "web_image"],
    },
    "mystery": {
        "tones": ["tense", "dark"],
        "accents": [[120, 145, 170], [138, 120, 165], [108, 142, 150], [150, 140, 120]],
        "map_styles": ["dark", "satellite", "tactical"],
        "chart_styles": ["navy_grid", "neutral"],
        "grade_sat": [0.55, 0.78], "beat_target": [3.4, 4.6],
        "hold_mult": [1.2, 1.45], "music_bed": [0.66, 0.82], "density": [0.7, 0.95],
        "transition_families": [
            ["slow_dissolve", "dissolve", "cut"],
            ["dissolve", "cut", "fadeblack"]],
        "motion_bias": ["slow_push", "controlled_static", "drift"],
        "lower_third_families": ["text_on_black", "mono_tag", "scrim_panel"],
        "subtitle_styles": ["mono_lower", "minimal"],
        "sfx_restraint": ["sparse", "diegetic_light"],
        "source_priority": ["archive", "ai", "web_image", "stock"],
    },
    "history": {
        "tones": ["reverent", "wonder"],
        "accents": [[206, 158, 80], [196, 150, 92], [188, 146, 70], [180, 150, 100]],
        "map_styles": ["parchment", "political", "satellite"],
        "chart_styles": ["sepia_textured", "neutral"],
        "grade_sat": [0.82, 0.95], "beat_target": [4.6, 6.0],
        "hold_mult": [1.3, 1.55], "music_bed": [0.66, 0.82], "density": [0.7, 0.95],
        "transition_families": [
            ["film_dissolve", "slow_dissolve", "page_wipe", "cut"],
            ["slow_dissolve", "dissolve", "fadeblack", "cut"]],
        "motion_bias": ["slow_push", "drift", "controlled_static"],
        "lower_third_families": ["serif_engraved", "era_label", "scrim_panel"],
        "subtitle_styles": ["serif_lower", "minimal"],
        "sfx_restraint": ["sparse", "diegetic_light"],
        "source_priority": ["archive", "ai", "stock", "web_image"],
    },
    "biography": {
        "tones": ["reverent", "wonder"],
        "accents": [[200, 165, 90], [208, 150, 84], [190, 158, 96], [186, 162, 110]],
        "map_styles": ["parchment", "political", "satellite"],
        "chart_styles": ["sepia_textured", "navy_grid"],
        "grade_sat": [0.82, 0.95], "beat_target": [4.2, 5.6],
        "hold_mult": [1.2, 1.45], "music_bed": [0.70, 0.86], "density": [0.85, 1.1],
        "transition_families": [
            ["film_dissolve", "slow_dissolve", "cut"],
            ["slow_dissolve", "dissolve", "page_wipe", "cut"]],
        "motion_bias": ["slow_push", "drift", "reveal_push"],
        "lower_third_families": ["serif_engraved", "name_portrait", "scrim_panel"],
        "subtitle_styles": ["serif_lower", "bold_lower"],
        "sfx_restraint": ["diegetic_light", "sparse"],
        "source_priority": ["archive", "ai", "stock", "web_image"],
    },
    "business": {
        "tones": ["clinical", "wonder"],
        "accents": [[235, 170, 70], [240, 150, 70], [225, 178, 60], [245, 160, 64]],
        "map_styles": ["political", "news", "satellite"],
        "chart_styles": ["white_dataviz", "navy_grid"],
        "grade_sat": [0.86, 0.98], "beat_target": [2.6, 3.4],
        "hold_mult": [0.7, 0.95], "music_bed": [0.9, 1.0], "density": [1.0, 1.3],
        "transition_families": [
            ["geo_push", "page_wipe", "cut", "soft_dissolve"],
            ["page_wipe", "cut", "soft_dissolve"]],
        "motion_bias": ["active_pan", "reveal_push", "push_hold"],
        "lower_third_families": ["clean_bar", "data_pill", "scrim_panel"],
        "subtitle_styles": ["bold_lower", "clean_lower"],
        "sfx_restraint": ["light", "diegetic_light"],
        "source_priority": ["stock", "ai", "web_image", "archive"],
    },
    "science": {
        "tones": ["clinical", "wonder"],
        "accents": [[90, 185, 200], [110, 175, 210], [80, 195, 180], [120, 180, 215]],
        "map_styles": ["political", "satellite"],
        "chart_styles": ["white_dataviz", "neutral"],
        "grade_sat": [0.9, 1.02], "beat_target": [2.4, 3.2],
        "hold_mult": [0.75, 1.0], "music_bed": [0.85, 1.0], "density": [1.0, 1.3],
        "transition_families": [
            ["geo_push", "page_wipe", "cut", "soft_dissolve"],
            ["cut", "soft_dissolve", "page_wipe"]],
        "motion_bias": ["active_pan", "reveal_push"],
        "lower_third_families": ["clean_bar", "labeled_diagram", "data_pill"],
        "subtitle_styles": ["clean_lower", "bold_lower"],
        "sfx_restraint": ["light", "diegetic_light"],
        "source_priority": ["ai", "stock", "web_image", "archive"],
    },
    "geopolitics": {
        "tones": ["clinical", "tense"],
        "accents": [[220, 120, 70], [210, 140, 80], [200, 132, 95], [225, 128, 60]],
        "map_styles": ["political", "tactical", "satellite"],
        "chart_styles": ["white_dataviz", "navy_grid"],
        "grade_sat": [0.84, 0.96], "beat_target": [2.6, 3.6],
        "hold_mult": [0.85, 1.1], "music_bed": [0.85, 1.0], "density": [1.0, 1.3],
        "transition_families": [
            ["geo_push", "page_wipe", "cut", "soft_dissolve"],
            ["geo_push", "cut", "blur_cut"]],
        "motion_bias": ["active_pan", "map_drift", "push_hold"],
        "lower_third_families": ["clean_bar", "location_label", "data_pill"],
        "subtitle_styles": ["bold_lower", "clean_lower"],
        "sfx_restraint": ["light", "diegetic_light"],
        "source_priority": ["stock", "archive", "ai", "web_image"],
    },
}
NICHE_ENVELOPES["_default"] = NICHE_ENVELOPES["biography"]


# ── Forensic envelope expansions (Wave 0) ───────────────────────────────
# editorial_envelopes_forensic.json (bundled beside this module, synthesized
# by tools/deep_forensics.py over 4 reference docs — see
# research/forensic_refs/_SYNTHESIS.md) WIDENS each niche toward the real
# editorial range observed in pro documentaries.  It dict-MERGES over the
# hand-written base above: expansion keys overwrite (lists replace, ranges
# overwrite); keys absent from the expansion (chart_styles, sfx_restraint,
# source_priority) are preserved.  Only REAL renderer tokens are used.
# Graceful (missing/!parse → base unchanged) + env-gated
# (VIDLORE_EDITORIAL_FORENSIC=0 → base only) so it can never break a render.
_FORENSIC_ENV = Path(__file__).parent / "editorial_envelopes_forensic.json"


def _forensic_disabled() -> bool:
    return os.environ.get("VIDLORE_EDITORIAL_FORENSIC", "1").strip().lower() in (
        "0", "false", "no", "off")


def _apply_forensic_expansions(base: dict) -> dict:
    if _forensic_disabled():
        return base
    try:
        data = json.loads(_FORENSIC_ENV.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return base
    for niche, exp in data.items():
        if niche.startswith("_") or not isinstance(exp, dict):
            continue
        tgt = dict(base.get(niche) or base.get("_default") or {})
        for k, v in exp.items():
            if not k.startswith("_"):          # skip _notes provenance
                tgt[k] = v
        base[niche] = tgt
    base["_default"] = base.get("biography", base.get("_default"))
    return base


NICHE_ENVELOPES = _apply_forensic_expansions(NICHE_ENVELOPES)


def disabled() -> bool:
    return os.environ.get("VIDLORE_EDITORIAL_RECIPE", "1").strip().lower() in (
        "0", "false", "no", "off")


def _seed(brief, salt: str = "") -> int:
    title = (getattr(brief, "title", "") or "")
    prompt = (getattr(brief, "prompt", "") or "")[:400]
    chan = (getattr(brief, "channel", None) or "") or ""
    h = hashlib.sha1((title + "\n" + prompt + "\n" + chan + "\n" + salt).encode("utf-8"))
    return int(h.hexdigest()[:12], 16)


def _pick(rng, env, key, default=None):
    pool = env.get(key)
    if not pool:
        return default
    return rng.choice(pool)


def _rng_range(rng, env, key, default):
    rng_pair = env.get(key)
    if not rng_pair or len(rng_pair) != 2:
        return default
    lo, hi = rng_pair
    return round(rng.uniform(float(lo), float(hi)), 3)


def _jitter_accent(rng, base):
    return [max(20, min(255, int(c + rng.randint(-10, 10)))) for c in base]


def recipe_signature(recipe: dict) -> tuple:
    """Compact identity used by the anti-repetition history."""
    return (
        recipe.get("niche"), tuple(recipe.get("accent", [])[:3]),
        recipe.get("map_style"), recipe.get("chart_style"),
        tuple(recipe.get("transition_palette", [])[:3]),
        recipe.get("lower_third_family"), recipe.get("subtitle_style"),
        recipe.get("motion_bias"),
    )


def _similar(a: tuple, b: tuple) -> int:
    """Count of matching axes between two signatures (higher = more alike)."""
    return sum(1 for x, y in zip(a, b) if x == y and x is not None)


def _build(brief, niche: str, salt: str) -> dict:
    env = NICHE_ENVELOPES.get(niche) or NICHE_ENVELOPES["_default"]
    rng = random.Random(_seed(brief, salt))
    accent = _jitter_accent(rng, _pick(rng, env, "accents", [200, 160, 90]))
    return {
        "niche": niche,
        "tone": _pick(rng, env, "tones", "neutral"),
        "accent": accent,
        "map_style": _pick(rng, env, "map_styles", "satellite"),
        "chart_style": _pick(rng, env, "chart_styles", "navy_grid"),
        "grade_sat": _rng_range(rng, env, "grade_sat", 0.92),
        "beat_target": _rng_range(rng, env, "beat_target", 3.4),
        "hold_mult": _rng_range(rng, env, "hold_mult", 1.0),
        "music_bed": _rng_range(rng, env, "music_bed", 1.0),
        "density": _rng_range(rng, env, "density", 1.0),
        "transition_palette": _pick(rng, env, "transition_families", ["cut", "dissolve"]),
        "motion_bias": _pick(rng, env, "motion_bias", "drift"),
        "lower_third_family": _pick(rng, env, "lower_third_families", "scrim_panel"),
        "subtitle_style": _pick(rng, env, "subtitle_styles", "bold_lower"),
        "sfx_restraint": _pick(rng, env, "sfx_restraint", "diegetic_light"),
        # forensic Wave-0 axes (recorded; wired in later waves):
        "music_behavior": env.get("music_behavior", "balanced"),
        "rhythm_mode": (env.get("rhythm_profile") or {}).get("mode", "even"),
        # gilded warm-low-key sub-grade — only where the niche envelope
        # offers it (business/biography tycoon arcs) and only for SOME
        # videos, so the niche stays a SET (some clinical, some gilded).
        "grade_mode": ("gilded" if (env.get("grade_subgrade")
                       and rng.random() < 0.45) else "standard"),
        "source_priority": env.get("source_priority", []),
        "variation_salt": salt,
    }


def _load_history() -> list:
    try:
        return json.loads(_HISTORY.read_text(encoding="utf-8"))
    except Exception:                                              # noqa: BLE001
        return []


def generate_recipe(brief, niche: str, *, channel: Optional[str] = None,
                    variation: int = 0, avoid_recent: int = 6) -> dict:
    """Deterministic per-video editorial recipe within the niche envelope.
    `variation` re-rolls (Review-Editor 'Generate New Variation').  Anti-
    repetition: if the recipe is too similar to a recent one for the same
    channel/niche, deterministically re-roll up to a few times."""
    history = _load_history()
    recent = [h for h in history
              if h.get("niche") == niche
              and (not channel or h.get("channel") == channel)][-avoid_recent:]
    recent_sigs = [tuple(tuple(x) if isinstance(x, list) else x
                         for x in (h.get("sig") or ())) for h in recent]
    best = None
    for attempt in range(variation, variation + 5):
        rec = _build(brief, niche, salt=f"v{attempt}")
        sig = recipe_signature(rec)
        worst = max((_similar(sig, rs) for rs in recent_sigs), default=0)
        if worst <= 4:                          # distinct enough → accept
            rec["_attempt"] = attempt
            return rec
        if best is None or worst < best[0]:
            best = (worst, rec, attempt)
    rec = best[1]
    rec["_attempt"] = best[2]
    return rec


def record_recipe(recipe: dict, *, channel: Optional[str] = None,
                  cap: int = 200) -> None:
    """Append the recipe signature to the cross-video history (cooldowns)."""
    try:
        _HISTORY.parent.mkdir(parents=True, exist_ok=True)
        hist = _load_history()
        hist.append({"niche": recipe.get("niche"), "channel": channel or "",
                     "sig": list(recipe_signature(recipe))})
        _HISTORY.write_text(json.dumps(hist[-cap:]), encoding="utf-8")
    except Exception:                                              # noqa: BLE001
        pass


def to_look_overrides(recipe: dict) -> dict:
    """Emit the recipe axes the renderer consumes as resolved-look knobs.

    DOC_002 originals (tone/accent/map/chart) + per-video pacing
    (beat_target) were the first wired axes.  REC_WIRE_AXES adds the
    per-video values for knobs that previously had only a FIXED per-preset
    table in assemble.py — grade_sat (DOC_014), hold_mult (DOC_004),
    music_bed (DOC_017), density (HE#3b).  assemble.py now reads these from
    the resolved look first and falls back to its per-name table when
    absent (no recipe / channel mode → exact legacy behaviour).

    REC_BIMODAL wires rhythm_mode → assemble._rhythm_spread (shot-length
    spread); music_behavior → assemble._look_music_behavior (bed presence);
    sfx_restraint → assemble._look_sfx_restraint_mult (whoosh density);
    subtitle_style → captions.write_ass (per-video caption family);
    lower_third_family → footage._render_lower_third_card (per-video chrome).
    Still recorded-not-consumed: motion_bias (REC_MOTION_WIRE, deferred)."""
    return {
        "tone": recipe.get("tone"),
        "accent_rgb": recipe.get("accent"),
        "map_style": recipe.get("map_style"),
        "chart_style": recipe.get("chart_style"),
        # widened per-video pacing within the niche envelope
        "beat_target": recipe.get("beat_target"),
        # REC_WIRE_AXES — per-video values for the existing live look knobs
        "grade_sat": recipe.get("grade_sat"),
        "hold_mult": recipe.get("hold_mult"),
        "music_bed": recipe.get("music_bed"),
        "density": recipe.get("density"),
        # REC transition weighting — the per-video ALLOWED transition subset
        # (separate knob so it never clobbers the look's own transition_
        # palette WeightedBag).  assemble._edit_plan downgrades a flashy
        # transition (whip/blur_cut) NOT in this set to a clean cut.
        "transition_allowed": recipe.get("transition_palette"),
        # gilded warm sub-grade mode (assemble._look_grade_mode reads this)
        "grade_mode": recipe.get("grade_mode"),
        # REC_BIMODAL — per-video rhythm character; assemble._rhythm_spread
        # reads this to widen/narrow the shot-length spread (LENGTHS only, so
        # the footage fetcher and assembler still agree on beat COUNT).
        "rhythm_mode": recipe.get("rhythm_mode"),
        # per-niche music PRESENCE behaviour (momentum bed vs silence-
        # punctuated / voice-led recede); assemble._look_music_behavior reads
        # it as a bounded factor on the resting bed (distinct from music_bed
        # LEVEL).
        "music_behavior": recipe.get("music_behavior"),
        # per-video SFX restraint (sparse / light / diegetic_light /
        # tension_bed); assemble._look_sfx_restraint_mult widens the whoosh
        # min-gap (sparser → fewer whooshes) on top of the niche default.
        "sfx_restraint": recipe.get("sfx_restraint"),
        # per-video SUBTITLE family (minimal / bold_lower / serif_lower /
        # mono_lower); captions.write_ass varies size/position/emphasis/font
        # (readability-first, bottom-kept). Review Editor can override.
        "subtitle_style": recipe.get("subtitle_style"),
        # per-video LOWER-THIRD family (scrim_panel / case_file_tag /
        # dossier_id_card / era_label / name_portrait / serif_engraved);
        # footage._render_lower_third_card varies accent treatment + scrim.
        "lower_third_family": recipe.get("lower_third_family"),
    }
