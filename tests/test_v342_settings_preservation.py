# V3.4.2 STEP 3 — editor-rerender SETTINGS PRESERVATION regression.
# Proves _persist_render_settings / _restore_render_settings round-trip the
# original project mode (esp. MG on/off) so an editor rerender never silently
# resets to the new MG-ON default; legacy projects get a documented MG-OFF
# migration default; editor-set keys are never clobbered by the restore.
#
# Run:  PYTHONPATH=. python tests/test_v342_settings_preservation.py
import os
import json
import tempfile
from pathlib import Path

os.environ.setdefault("VIDLORE_AI_VIDEO", "0")
import vidlore.web as W
from vidlore.brief import Brief


class _Cfg:
    music_enabled = True
    sfx_enabled = False
    transitions_enabled = True
    overlays_enabled = True


_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


def _persist(mg_on, extra=None):
    d = Path(tempfile.mkdtemp())
    b = Brief(title="t", prompt="reviewed")
    b.extra = dict(extra or {})
    os.environ["VIDLORE_MOTION_GRAPHICS"] = "1" if mg_on else "0"
    W._persist_render_settings(d, b, _Cfg)
    return d


# 1 — MG-OFF project round-trips to MG OFF (the portal-style bug)
d = _persist(False, {"niche": "spy_intel", "sfx": True})
b = Brief(title="t", prompt="reviewed")          # load_brief gives empty extra
st = W._restore_render_settings(d, b)
check("1_mg_off_preserved", b.extra.get("mg") is False)
check("1_status_mentions_restore", "restored" in st and "mg=False" in st)
check("1_niche_preserved", b.extra.get("niche") == "spy_intel")
check("1_sfx_preserved", b.extra.get("sfx") is True)

# 2 — MG-ON project round-trips to MG ON
d = _persist(True, {})
b = Brief(title="t", prompt="reviewed")
W._restore_render_settings(d, b)
check("2_mg_on_preserved", b.extra.get("mg") is True)

# 3 — legacy project (no snapshot) → documented MG-OFF migration default + log
d = Path(tempfile.mkdtemp())
b = Brief(title="t", prompt="reviewed")
st = W._restore_render_settings(d, b)
check("3_legacy_mg_off", b.extra.get("mg") is False)
check("3_legacy_logged", "legacy" in st and "migration default MG OFF" in st)
check("3_legacy_marker", b.extra.get("_settings_migrated") == "legacy_no_snapshot_mg_off")

# 4 — editor-set key is NEVER clobbered by the restore (setdefault semantics)
d = _persist(True, {"music": True})
b = Brief(title="t", prompt="reviewed")
b.extra = {"music": False}                        # editor turned music OFF this session
W._restore_render_settings(d, b)
check("4_editor_override_wins", b.extra.get("music") is False)
check("4_restore_still_fills_mg", b.extra.get("mg") is True)

# 5 — AI-generated video stays OFF in the snapshot
d = _persist(True, {})
snap = json.loads((d / "render_settings.json").read_text())
check("5_ai_video_off_recorded", snap.get("ai_video") is False)
check("5_schema_present", snap.get("_schema") == "render_settings/1")

# 6 — persist is best-effort (never raises, even on a bad run_dir)
try:
    W._persist_render_settings(Path("/no/such/dir/xyz"), Brief(title="t", prompt="r"), _Cfg)
    check("6_persist_never_raises", True)
except Exception:
    check("6_persist_never_raises", False)

# 7 — restore is best-effort on a corrupt snapshot (falls back, no raise)
d = Path(tempfile.mkdtemp())
(d / "render_settings.json").write_text("{ this is not json")
b = Brief(title="t", prompt="reviewed")
st = W._restore_render_settings(d, b)
check("7_corrupt_snapshot_safe", isinstance(st, str))

# 8 — full realistic extra round-trips intact
full = {"niche": "spy_intel", "sfx": True, "wf_mix": "balanced", "wi_mix": "heavy",
        "voice_mode": "premium", "tts_voice": "deep_male_documentary", "look_preset": "true_crime"}
d = _persist(False, full)
b = Brief(title="t", prompt="reviewed")
W._restore_render_settings(d, b)
check("8_full_extra_roundtrip", all(b.extra.get(k) == v for k, v in full.items()))
check("8_full_mg_off", b.extra.get("mg") is False)

print("\nV3.4.2 SETTINGS PRESERVATION: %d/%d PASS" % (_passed, _passed))
