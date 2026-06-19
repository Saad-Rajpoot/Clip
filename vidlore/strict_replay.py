"""V3.4.2 — STRICT REPLAY MODE (deterministic parity / debugging).

Fresh production renders stay visually varied (anti-repetition is untouched).
Strict replay is an OPT-IN mode that reproduces an EXISTING project as exactly as
the inputs allow, by composing mechanisms the engine already has — it adds no new
render behaviour and changes nothing unless `prepare()` is called / the
`VIDLORE_STRICT_REPLAY` env is set:

  • recipe + variation_salt  → pinned via `brief.extra['editorial_recipe_lock']`
    (look_dna honours a locked recipe verbatim → NO salt rotation),
  • script + narration       → reused via `VIDLORE_REUSE_SCRIPT_JSON=1` (no LLM),
  • cached TTS / footage / AI stills / archival → reused (content-hashed cache;
    fal/TTS are NOT called when a valid cached asset exists),
  • NO new web-footage / web-image fetches (engines forced off → cache only),
  • motion-graphics mode     → forced to the project's RECORDED mode,
  • AI-generated VIDEO        → stays OFF (never enabled here).

If an artifact is genuinely missing, the engine's normal narrowest-safe fallback
runs and the result is classified NON-exact. The classification is persisted so a
caller can tell an exact replay from a fallback / a fresh variation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

VERSION = "v3.4.2"

EXACT = "exact_replay"
FALLBACK = "replay_with_missing_asset_fallback"
FRESH = "fresh_variation_render"


def _load(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def prepare(run_dir, brief, *, mg_mode=None) -> dict:
    """Pin everything needed for an exact replay of the project at `run_dir`.
    Mutates os.environ + brief.extra (opt-in). Returns a 'pins' dict for logging
    + the manifest. Never raises — a missing field just leaves that pin unset."""
    run_dir = Path(run_dir)
    meta = _load(run_dir / "render_meta.json")
    settings = _load(run_dir / "render_settings.json")
    recipe = meta.get("editorial_recipe") or {}
    pins: dict = {"strict_replay": True, "version": VERSION}

    # 1. recipe + variation_salt — lock verbatim (defeats anti-repetition rotation)
    if isinstance(recipe, dict) and recipe.get("niche"):
        if not isinstance(getattr(brief, "extra", None), dict):
            brief.extra = {}
        brief.extra["editorial_recipe_lock"] = dict(recipe)
        pins["recipe_locked"] = True
        pins["variation_salt"] = recipe.get("variation_salt")
    else:
        pins["recipe_locked"] = False

    # 2. motion-graphics mode — recorded mode wins; explicit arg overrides; else OFF
    mg = settings.get("mg") if isinstance(settings, dict) else None
    if mg_mode is not None:
        mg = bool(mg_mode)
    if mg is None:
        mg = False                       # legacy project default (how it was rendered)
        pins["mg_source"] = "legacy_default_off"
    else:
        pins["mg_source"] = "recorded" if mg_mode is None else "explicit_arg"
    os.environ["VIDLORE_MOTION_GRAPHICS"] = "1" if mg else "0"
    pins["motion_graphics"] = bool(mg)

    # 3. exact script + narration (no LLM)
    os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"

    # 4. no NEW web fetches — rely on the project's content-hashed cache
    os.environ["WEB_FOOTAGE_ENGINE"] = "0"
    os.environ["WEB_IMAGE_ENGINE"] = "0"

    # 5. AI-generated VIDEO stays OFF (double-gated; never enabled here)
    os.environ.pop("VIDLORE_AI_VIDEO", None)
    os.environ.pop("VIDLORE_AI_VIDEO", None)
    pins["ai_video"] = False

    # 6. voice backend — match the recorded one (so cached TTS is reused, not re-synth)
    vb = settings.get("voice_backend") if isinstance(settings, dict) else None
    if vb:
        os.environ["VIDLORE_TTS_BACKEND"] = vb
        pins["voice_backend"] = vb

    os.environ["VIDLORE_STRICT_REPLAY"] = "1"
    return pins


def classify(orig_meta: dict, new_meta: dict, *, regen: dict | None = None,
             dur_eps: float = 0.05) -> tuple:
    """Classify a replay result against the original. `regen` counts the things
    that were re-made this run (llm, tts, fal, stock_refetch, ai_video). Returns
    (status, detail)."""
    regen = regen or {}
    o_rec = (orig_meta.get("editorial_recipe") or {})
    n_rec = (new_meta.get("editorial_recipe") or {})
    salt_match = o_rec.get("variation_salt") == n_rec.get("variation_salt")
    recipe_match = all(o_rec.get(k) == n_rec.get(k) for k in
                       ("niche", "accent", "beat_target", "density",
                        "transition_palette", "variation_salt"))
    scenes_match = orig_meta.get("scenes") == new_meta.get("scenes")
    od, nd = orig_meta.get("scene_durations") or [], new_meta.get("scene_durations") or []
    dur_match = (len(od) == len(nd) and
                 all(abs(float(a) - float(b)) <= dur_eps for a, b in zip(od, nd)))

    detail = {
        "salt_match": salt_match, "recipe_match": recipe_match,
        "scenes_match": scenes_match, "durations_match": dur_match,
        "regen": dict(regen),
    }
    # A fresh variation: the anti-rep recipe/salt moved, or the structure changed.
    if not (salt_match and recipe_match and scenes_match):
        return FRESH, detail
    # Structure pinned. Exact iff nothing of substance was regenerated.
    bad = (regen.get("llm", 0) or regen.get("ai_video", 0)
           or regen.get("stock_refetch", 0) or not dur_match)
    if bad:
        return FALLBACK, detail
    # TTS/fal re-runs that came from a genuine cache MISS count as fallback;
    # callers pass those in regen only when they were real (cache-miss) re-makes.
    if regen.get("tts", 0) or regen.get("fal", 0):
        return FALLBACK, detail
    return EXACT, detail


def write_status(run_dir, status: str, detail: dict, pins: dict) -> Path:
    """Persist the replay classification next to the render so a caller (and the
    manifest) records whether the result is exact / fallback / fresh."""
    run_dir = Path(run_dir)
    payload = {"schema": "replay_status/1", "version": VERSION,
               "replay_status": status, "detail": detail, "pins": pins}
    p = run_dir / "replay_status.json"
    try:
        p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return p
