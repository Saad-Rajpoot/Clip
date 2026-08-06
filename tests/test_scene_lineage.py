from pathlib import Path

import pytest

from vidlore.clipstudio.scene_lineage import (
    assert_scene_lineage,
    selection_binding,
    validate_scene_lineage,
)


def _clip(tmp_path: Path, name: str) -> str:
    p = tmp_path / name
    p.write_bytes(b"real clip bytes")
    return str(p)


def _ordinary(tmp_path: Path, **kw):
    binding = selection_binding(34, "varys", 36.245, 38.163,
                                {"verdict": "keep", "status": "ok"})
    source = _clip(tmp_path, "varys_source.mp4")
    out = {
        "kind": "selection_derivative",
        "final_scene": 34,
        "original_beat": 34,
        "owner_beat": 34,
        "clip": 0,
        "file": _clip(tmp_path, "beat_034_0.mp4"),
        "via": "selection_derivative",
        "selected_source_id": "varys",
        "actual_source_id": "varys",
        "selection_source_path": source,
        "selected_window": ["varys", 36.245, 38.163],
        "actual_window": ["varys", 36.245, 38.163],
        "selection_binding": binding,
        "root_binding": binding,
        "validated": True,
    }
    out.update(kw)
    return out


def test_own_selection_derivative_passes(tmp_path):
    assert validate_scene_lineage([_ordinary(tmp_path)]) == []


@pytest.mark.parametrize("change,needle", [
    ({"owner_beat": 33}, "root belongs"),
    ({"actual_source_id": "olenna"}, "source changed"),
    ({"actual_window": ["olenna", 98.7, 101.0]}, "actual window"),
    ({"actual_window": ["varys", 36.245, 99.0]}, "actual window"),
    ({"via": "window[alt]"}, "unverified path"),
    ({"via": "walk"}, "unverified path"),
    ({"root_binding": "other"}, "root binding"),
    ({"kind": "placeholder"}, "unknown/untracked"),
])
def test_wrong_or_untracked_roots_fail(tmp_path, change, needle):
    probs = validate_scene_lineage([_ordinary(tmp_path, **change)])
    assert probs and needle in probs[0]["reason"]


def test_breakout_and_verified_image_are_explicit_special_roots(tmp_path):
    rows = [
        {"kind": "breakout", "final_scene": 150, "original_beat": 150,
         "owner_beat": 150, "clip": 0, "file": _clip(tmp_path, "breakout.mp4"),
         "validated": True},
        {"kind": "verified_image", "final_scene": 151, "original_beat": 150,
         "owner_beat": 150, "clip": 0, "file": _clip(tmp_path, "image.mp4"),
         "root_binding": "image-binding", "validated": True,
         "image_owner_kind": "verified_file", "image_sha256": "image-sha",
         "image_width": 1280, "image_height": 720, "preserved_original": True,
         "actual_image_source_id": "", "actual_image_shot_index": None},
    ]
    assert validate_scene_lineage(rows) == []


def _source_frame_image(tmp_path: Path, **kw):
    row = {
        "kind": "verified_image", "final_scene": 34, "original_beat": 34,
        "owner_beat": 34, "clip": 0, "file": _clip(tmp_path, "source_image.mp4"),
        "root_binding": "image-binding", "validated": True,
        "image_owner_kind": "source_frame", "image_sha256": "image-sha",
        "image_width": 1920, "image_height": 1080,
        "expected_image_source_id": "varys", "actual_image_source_id": "varys",
        "expected_image_shot_index": 7, "actual_image_shot_index": 7,
        "actual_image_time": 12.0,
        "source_native_width": 1920, "source_native_height": 1080,
        "preserved_original": False,
    }
    row.update(kw)
    return row


def test_source_frame_image_exact_owner_passes(tmp_path):
    assert validate_scene_lineage([_source_frame_image(tmp_path)]) == []


@pytest.mark.parametrize("change,needle", [
    ({"image_owner_kind": ""}, "unknown owner"),
    ({"image_sha256": ""}, "source-image hash"),
    ({"image_width": 1024, "image_height": 768}, "not publishable"),
    ({"actual_image_source_id": "olenna"}, "owner changed"),
    ({"actual_image_shot_index": 8}, "owner changed"),
    ({"source_native_height": 360}, "native 1280x720"),
    ({"source_native_width": 640, "source_native_height": 720}, "native 1280x720"),
    ({"actual_image_time": None}, "time is missing"),
])
def test_verified_image_manifest_rejects_unproved_owner_or_quality(tmp_path, change, needle):
    problems = validate_scene_lineage([_source_frame_image(tmp_path, **change)])
    assert problems and any(needle in p["reason"] for p in problems)


def _hold(tmp_path: Path, **kw):
    """Beat 87's aired input: a Ken-Burns hold on beat 86's own verified frame.

    Job 0ca9dc4c2f built the entire 152-scene video and then died here. The rejected-footage gate
    had legitimately covered beat 87 — whose footage the verifier rejected — with beat 86's clean
    same-scene frame (overlap 0.55, inside the freeze caps, recorded in
    rejected_footage_audit.json). Because a hold had no lineage kind of its own it inherited beat
    86's root wholesale, and the provenance gate read the sanctioned donation as a silent owner
    swap. Two deliberate features, mutually unsatisfiable, at the last gate before assemble.
    """
    binding = selection_binding(86, "kings_landing", 144.411, 145.913,
                                {"verdict": "keep", "status": "ok"})
    source = _clip(tmp_path, "kings_landing.mp4")
    out = {
        "kind": "scene_hold",
        "final_scene": 87,
        "original_beat": 87,
        "owner_beat": 86,
        "clip": 0,
        "file": _clip(tmp_path, "beat_087_0_hold.mp4"),
        "via": "editorial_hold",
        "hold_of_beat": 86,
        "hold_kind": "editorial_hold_kenburns",
        "hold_duration_s": 3.96,
        "hold_compat_evidence": {"scene_overlap": 0.55,
                                 "shared_tokens": ["dontos", "purple", "wedding"]},
        "selected_source_id": "kings_landing",
        "actual_source_id": "kings_landing",
        "selection_source_path": source,
        "selected_window": ["kings_landing", 144.411, 145.913],
        "actual_window": ["kings_landing", 144.411, 145.913],
        "selection_binding": binding,
        "root_binding": binding,
        "validated": True,
    }
    out.update(kw)
    return out


def test_declared_editorial_hold_on_a_donor_beat_passes(tmp_path):
    assert validate_scene_lineage([_hold(tmp_path)]) == []


@pytest.mark.parametrize("change,needle", [
    # The owner relaxation is bought with an explicit declaration — never inferred.
    ({"hold_of_beat": None}, "does not declare which beat"),
    ({"hold_of_beat": 12}, "claims donor beat"),
    ({"hold_of_beat": 87, "owner_beat": 87}, "names itself as its own donor"),
    ({"via": "selection_derivative"}, "unsanctioned path"),
    ({"hold_kind": "silent_swap"}, "unknown treatment"),
    ({"hold_compat_evidence": {}}, "no same-scene compatibility evidence"),
    ({"hold_duration_s": 0.0}, "positive finite"),
    ({"hold_duration_s": None}, "missing/malformed"),
    # And every byte-level rule still binds the donor's own verified selection.
    ({"actual_source_id": "olenna"}, "source changed"),
    ({"actual_window": ["kings_landing", 144.411, 99.0]}, "actual window"),
    ({"root_binding": "other"}, "root binding"),
    ({"validated": False}, "not validated"),
])
def test_a_hold_that_is_not_a_declared_sanctioned_hold_still_fails(tmp_path, change, needle):
    problems = validate_scene_lineage([_hold(tmp_path, **change)])
    assert problems and any(needle in p["reason"] for p in problems)


def test_the_owner_relaxation_does_not_leak_to_ordinary_footage(tmp_path):
    """Only a declared hold may air another beat's frame; a plain derivative may not."""
    problems = validate_scene_lineage([
        _ordinary(tmp_path, owner_beat=33, hold_of_beat=33,
                  hold_kind="editorial_hold", hold_duration_s=2.0,
                  hold_compat_evidence={"scene_overlap": 0.9})])
    assert problems and any("root belongs" in p["reason"] for p in problems)


def test_assertion_persists_failure_then_raises(tmp_path):
    audit = tmp_path / "scene_lineage.json"
    with pytest.raises(Exception, match="scene-lineage gate"):
        assert_scene_lineage([_ordinary(tmp_path, actual_source_id="olenna")], audit)
    assert '"passed": false' in audit.read_text()


def test_audit_write_failure_is_not_swallowed(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    with pytest.raises((OSError, FileExistsError)):
        assert_scene_lineage([_ordinary(tmp_path)], blocker / "audit.json")
