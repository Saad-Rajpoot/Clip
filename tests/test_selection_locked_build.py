"""Build-side proof that verified selection ownership cannot be bypassed."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = (ROOT / "vidlore" / "clipstudio" / "build.py").read_text(encoding="utf-8")


def _lock_block() -> str:
    start = BUILD.index("# VERIFIED-SELECTION LOCK")
    end = BUILD.index("windows_avail = list", start)
    return BUILD[start:end]


def test_selection_lock_reads_only_the_verified_cut_clip():
    block = _lock_block()
    assert "sel.clip_path" not in block  # getattr form is deliberate/fail-safe
    assert 'getattr(sel, "clip_path"' in block
    assert "_fit_verified_selection_clip(" in block
    assert "src.local_path" not in block
    assert "beat_windows" not in block


def test_missing_or_failed_owned_clip_blocks_instead_of_filling():
    block = _lock_block()
    assert block.count('kind="scene_lineage"') >= 4
    assert "_placeholder_clip" not in block
    assert "window[alt]" not in block
    assert '"walk"' not in block


def test_lock_is_unconditional_and_exits_before_legacy_reselector():
    block = _lock_block()
    assert "VIDLORE_" not in block
    assert "gbeat += k\n        continue" in block


def test_owned_derivative_holds_a_safe_in_window_frame_without_double_zoom():
    fit_start = BUILD.index("def _fit_verified_selection_clip(")
    fit_end = BUILD.index("# ── BREAKOUT ARTIFACT COMPOSITION", fit_start)
    fit = BUILD[fit_start:fit_end]
    block = _lock_block()

    assert "clip_duration * 0.88" in fit
    assert 'vf.extend([f"trim=end={safe_end:.3f}", "setpts=PTS-STARTPTS"])' in fit
    assert fit.index("trim=end=") < fit.index("tpad=stop_mode=clone")
    assert "_owned_zoom = 1.0" in block
    assert "1.12 if pos < _hook_n" not in block


def test_both_manifest_and_decoded_canary_are_mandatory():
    assert "_assert_scene_lineage(" in BUILD
    call = BUILD[BUILD.index("result = assemble("):]
    assert 'scene_lineage={"entries": _scene_lineage}' in call
    assert BUILD.index("_assert_scene_lineage(") < BUILD.index("result = assemble(")


def test_cross_scene_repair_inherits_real_donor_owner():
    # Branding, darkness and rejected-footage repairs cannot relabel a donor as the current beat;
    # manifest construction compares this inherited owner with the expected selection owner.
    assert "_lineage_derive(_got, _donor)" in BUILD
    assert "_lineage_derive(_got, _last_clean_r)" in BUILD
    manifest = BUILD[BUILD.index("# FINAL BUILD-SIDE LINEAGE MANIFEST"):]
    assert '"owner_beat": None' in manifest
    assert '"root_owner_beat": _root_l.get("owner_beat")' in manifest
