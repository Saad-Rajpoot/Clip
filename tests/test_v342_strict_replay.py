# V3.4.2 STEP 4 — STRICT REPLAY capability regression (pure logic; no render).
# Run:  PYTHONPATH=. python tests/test_v342_strict_replay.py
import os
import json
import tempfile
from pathlib import Path

import vidlore.strict_replay as SR
from vidlore.brief import Brief

_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


RECIPE = {"niche": "spy_intel", "accent": [100, 140, 183], "beat_target": 3.885,
          "density": 0.942, "transition_palette": ["slow_dissolve", "dissolve", "cut"],
          "variation_salt": "v2"}
META = {"scenes": 26, "scene_durations": [8.257, 3.312, 3.0], "editorial_recipe": RECIPE}


def _proj(mg=False, recipe=RECIPE):
    d = Path(tempfile.mkdtemp())
    meta = dict(META); meta["editorial_recipe"] = recipe
    (d / "render_meta.json").write_text(json.dumps(meta))
    (d / "render_settings.json").write_text(json.dumps({"mg": mg, "voice_backend": "edge"}))
    return d


# --- classify() ------------------------------------------------------------
s, _ = SR.classify(META, META, regen={})
check("classify_exact", s == SR.EXACT)

new_fresh = json.loads(json.dumps(META))
new_fresh["editorial_recipe"] = dict(RECIPE, variation_salt="v0", accent=[1, 2, 3])
s, _ = SR.classify(META, new_fresh, regen={})
check("classify_fresh_on_salt_change", s == SR.FRESH)

s, _ = SR.classify(META, META, regen={"llm": 1})
check("classify_fallback_on_llm", s == SR.FALLBACK)
s, _ = SR.classify(META, META, regen={"ai_video": 1})
check("classify_fallback_on_ai_video", s == SR.FALLBACK)
s, _ = SR.classify(META, META, regen={"stock_refetch": 3})
check("classify_fallback_on_stock_refetch", s == SR.FALLBACK)
s, _ = SR.classify(META, META, regen={"tts": 5})
check("classify_fallback_on_tts_cache_miss", s == SR.FALLBACK)

new_dur = json.loads(json.dumps(META)); new_dur["scene_durations"] = [8.5, 3.3, 3.0]
s, _ = SR.classify(META, new_dur, regen={})
check("classify_fallback_on_duration_drift", s == SR.FALLBACK)

new_scenes = json.loads(json.dumps(META)); new_scenes["scenes"] = 25
s, _ = SR.classify(META, new_scenes, regen={})
check("classify_fresh_on_scene_count_change", s == SR.FRESH)


# --- prepare() -------------------------------------------------------------
for k in ("VIDLORE_MOTION_GRAPHICS", "VIDLORE_AI_VIDEO", "VIDLORE_AI_VIDEO",
          "WEB_FOOTAGE_ENGINE", "WEB_IMAGE_ENGINE", "VIDLORE_STRICT_REPLAY"):
    os.environ.pop(k, None)

d = _proj(mg=False)
b = Brief(title="t", prompt="reviewed")
pins = SR.prepare(d, b)
check("prepare_locks_recipe", b.extra.get("editorial_recipe_lock", {}).get("variation_salt") == "v2")
check("prepare_salt_pin", pins.get("variation_salt") == "v2")
check("prepare_mg_off_recorded", os.environ["VIDLORE_MOTION_GRAPHICS"] == "0")
check("prepare_no_web_fetch", os.environ["WEB_FOOTAGE_ENGINE"] == "0" and os.environ["WEB_IMAGE_ENGINE"] == "0")
check("prepare_reuse_script", os.environ["VIDLORE_REUSE_SCRIPT_JSON"] == "1")
check("prepare_ai_video_off", "VIDLORE_AI_VIDEO" not in os.environ and "VIDLORE_AI_VIDEO" not in os.environ)
check("prepare_strict_flag", os.environ["VIDLORE_STRICT_REPLAY"] == "1")
check("prepare_voice_backend", os.environ.get("VIDLORE_TTS_BACKEND") == "edge")

# MG-ON project → recorded ON
d2 = _proj(mg=True)
b2 = Brief(title="t", prompt="reviewed")
SR.prepare(d2, b2)
check("prepare_mg_on_recorded", os.environ["VIDLORE_MOTION_GRAPHICS"] == "1")

# explicit mg_mode arg overrides recorded
d3 = _proj(mg=True)
b3 = Brief(title="t", prompt="reviewed")
p3 = SR.prepare(d3, b3, mg_mode=False)
check("prepare_explicit_mg_override", os.environ["VIDLORE_MOTION_GRAPHICS"] == "0" and p3["mg_source"] == "explicit_arg")

# legacy project (no render_settings.json) → MG OFF default
d4 = Path(tempfile.mkdtemp())
(d4 / "render_meta.json").write_text(json.dumps(META))
b4 = Brief(title="t", prompt="reviewed")
p4 = SR.prepare(d4, b4)
check("prepare_legacy_mg_off", os.environ["VIDLORE_MOTION_GRAPHICS"] == "0" and p4["mg_source"] == "legacy_default_off")

# --- write_status() --------------------------------------------------------
d5 = _proj(mg=False)
SR.write_status(d5, SR.EXACT, {"salt_match": True}, {"strict_replay": True})
st = json.loads((d5 / "replay_status.json").read_text())
check("write_status_persisted", st["replay_status"] == SR.EXACT and st["schema"] == "replay_status/1")

print("\nV3.4.2 STRICT REPLAY: %d/%d PASS" % (_passed, _passed))
