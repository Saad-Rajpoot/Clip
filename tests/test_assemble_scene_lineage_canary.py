from __future__ import annotations

import inspect
import importlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vidlore.assemble import assemble, _assert_lineage_repair_owner
from vidlore.ffmpeg_tool import ffmpeg_exe
from vidlore.footage import FootageItem
from vidlore.scene_lineage_canary import (
    SceneLineageError,
    _sample_local_frames,
    bind_encode_plan,
    fail_audit,
    new_audit,
    verify_encoded_plan,
    verify_delivered_output,
    verify_timeline_order,
    write_audit,
)


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")


@pytest.fixture(scope="module")
def textured_clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("lineage_canary")
    a = root / "a.mp4"
    b = root / "b.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(a),
    ])
    _run_ffmpeg([
        "-f", "lavfi", "-i", "smptebars=size=320x180:rate=30:duration=1.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(b),
    ])
    correct = root / "correct.mp4"
    swapped = root / "swapped.mp4"
    for first, second, dest in ((a, b, correct), (b, a, swapped)):
        _run_ffmpeg([
            "-i", str(first), "-i", str(second),
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-r", "30", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(dest),
        ])
    return root, a, b, correct, swapped


def _plan_row(bi: int, path: Path) -> dict:
    return {
        "bi": bi,
        "j": bi,
        "scene_index": bi,
        "m": 0,
        "bd": 1.5,
        "pad": 0.0,
        "item": FootageItem(bi, path, True),
        "seg": path,
        "mg_clip": None,
        "mg_off": 0.0,
    }


def _expectation(scene: int, path: Path) -> dict:
    return {
        "final_scene": scene,
        "clip": 0,
        "file": str(path),
        "kind": "selection_derivative",
        "media_kind": "video",
    }


def test_assemble_api_is_optional_and_backwards_compatible():
    params = inspect.signature(assemble).parameters
    assert params["scene_lineage"].default is None
    assert params["lineage_expectations"].default is None


def test_binding_uses_metadata_and_exact_path_as_primary_authority(textured_clips):
    _, own, other, _, _ = textured_clips
    plan = [_plan_row(34, own)]

    rows, failures = bind_encode_plan(plan, [_expectation(34, own)])
    assert failures == []
    assert rows[0]["passed"] is True
    assert plan[0]["_lineage"]["input_path"] == own.resolve()

    rows, failures = bind_encode_plan([_plan_row(34, own)], [_expectation(34, other)])
    assert rows[0]["passed"] is False
    assert any("path differs" in f["reason"] for f in failures)

    no_kind = _expectation(34, own)
    no_kind.pop("kind")
    _, failures = bind_encode_plan([_plan_row(34, own)], [no_kind])
    assert any("no kind" in f["reason"] for f in failures)


def test_input_mutated_after_bind_fails_even_when_path_is_unchanged(textured_clips, tmp_path):
    _, a, b, _, _ = textured_clips
    mutable = tmp_path / "owned.mp4"
    shutil.copyfile(a, mutable)
    plan = [_plan_row(0, mutable)]
    assert bind_encode_plan(plan, [_expectation(0, mutable)])[1] == []

    # Same expected path, foreign bytes.  The old canary sampled this file only
    # after encode and therefore let the replacement define its own expectation.
    shutil.copyfile(b, mutable)
    rows, failures, _ = verify_encoded_plan(plan)
    assert failures
    assert "changed after lineage binding" in rows[0]["error"]


def test_foreign_derivative_at_expected_filename_fails_selected_source_window(
        textured_clips):
    _, selected_source, foreign, _, _ = textured_clips
    plan = [_plan_row(0, foreign)]
    exp = _expectation(0, foreign)
    exp.update({
        "selected_source_id": "selected",
        "selection_source_path": str(selected_source),
        "selected_window": ["selected", 0.0, 1.5],
    })
    rows, failures = bind_encode_plan(plan, [exp])
    assert failures and rows[0]["passed"] is False
    assert any("selected source window" in reason for reason in rows[0]["reasons"])


def test_authorized_watermark_crop_is_compared_exactly_without_admitting_foreign_bytes(
        textured_clips):
    root, selected_source, foreign, _, _ = textured_clips
    cropped = root / "authorized_crop.mp4"
    _run_ffmpeg([
        "-i", str(selected_source),
        "-vf", "crop=iw*0.840:ih*0.840:0:0,scale=320:180",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cropped),
    ])
    exp = _expectation(0, cropped)
    exp.update({
        "selected_source_id": "selected",
        "selection_source_path": str(selected_source),
        "selected_window": ["selected", 0.0, 1.5],
        "selection_source_compare_filter": "crop=iw*0.840:ih*0.840:0:0",
    })
    plan = [_plan_row(0, cropped)]
    rows, failures = bind_encode_plan(plan, [exp])
    assert failures == []
    assert rows[0]["selected_source_comparison"]["authorized_source_filter"] \
        == "crop=iw*0.840:ih*0.840:0:0"

    foreign_exp = dict(exp, file=str(foreign))
    foreign_rows, foreign_failures = bind_encode_plan(
        [_plan_row(0, foreign)], [foreign_exp])
    assert foreign_failures and foreign_rows[0]["passed"] is False


def test_empty_or_partial_contract_fails_closed(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    plan = [_plan_row(0, clip), _plan_row(1, clip)]
    _, empty_failures = bind_encode_plan(plan, {})
    assert any("contract is empty" in f["reason"] for f in empty_failures)
    _, partial_failures = bind_encode_plan(plan, [_expectation(0, clip)])
    assert any(f.get("scene") == 1 and "no lineage expectation" in f["reason"]
               for f in partial_failures)


def test_encoded_canary_accepts_own_clip_and_rejects_swapped_output(textured_clips):
    _, a, b, _, _ = textured_clips
    plan = [_plan_row(0, a)]
    assert bind_encode_plan(plan, [_expectation(0, a)])[1] == []

    rows, failures, _ = verify_encoded_plan(plan)
    assert failures == []
    assert rows[0]["comparison"]["gross_mismatches"] == 0

    plan[0]["seg"] = b
    rows, failures, _ = verify_encoded_plan(plan)
    assert failures
    assert rows[0]["comparison"]["gross_mismatches"] >= 4


def test_emergency_slate_is_an_unconditional_lineage_failure(textured_clips):
    _, a, _, _, _ = textured_clips
    plan = [_plan_row(0, a)]
    assert bind_encode_plan(plan, [_expectation(0, a)])[1] == []
    plan[0]["_lineage_emergency_slate"] = True
    rows, failures, _ = verify_encoded_plan(plan)
    assert failures
    assert "emergency slate" in rows[0]["error"]


def test_scene_encoder_reports_its_internal_slate(monkeypatch, tmp_path):
    # `_scene_video` has an internal readiness fallback which used to return
    # None and be counted as a successful real encode by the outer pool.
    assembly = importlib.import_module("vidlore.assemble")
    quarantine = importlib.import_module("vidlore.render_quarantine")
    monkeypatch.setattr(quarantine, "is_quarantined", lambda *a, **k: True)
    monkeypatch.setattr(assembly, "_safe_slate", lambda *a, **k: True)
    missing = tmp_path / "quarantined.mp4"
    assert assembly._scene_video(
        FootageItem(7, missing, True), 1.0, "null", tmp_path / "seg.mp4") is True


def test_conformed_timeline_order_accepts_correct_and_rejects_swap(textured_clips):
    _, a, b, correct, swapped = textured_clips
    plan = [_plan_row(0, a), _plan_row(1, b)]
    assert bind_encode_plan(
        plan, [_expectation(0, a), _expectation(1, b)])[1] == []
    _, encoded_failures, banks = verify_encoded_plan(plan)
    assert encoded_failures == []

    rows, failures = verify_timeline_order(correct, plan, [1.5, 1.5], {}, banks, 30)
    assert failures == []
    assert all(r["passed"] and len(r["samples"]) == 2 for r in rows)

    rows, failures = verify_timeline_order(swapped, plan, [1.5, 1.5], {}, banks, 30)
    assert len(failures) == 4
    assert all(not r["passed"] for r in rows)


def test_postpass_delivered_artifact_is_rechecked_and_swap_blocks(textured_clips, tmp_path):
    _, a, b, correct, swapped = textured_clips
    plan = [_plan_row(0, a), _plan_row(1, b)]
    binding, bind_failures = bind_encode_plan(
        plan, [_expectation(0, a), _expectation(1, b)])
    assert bind_failures == []
    encoded, encoded_failures, banks = verify_encoded_plan(plan)
    assert encoded_failures == []
    timeline, timeline_failures = verify_timeline_order(
        correct, plan, [1.5, 1.5], {}, banks, 30)
    assert timeline_failures == []
    audit = new_audit(tmp_path / "final.mp4")
    audit.update({
        "status": "passed", "stage": "timeline_order", "binding": binding,
        "encoded_segments": encoded, "timeline_order": timeline,
    })
    audit_path = tmp_path / "scene_lineage_audit.json"
    write_audit(audit_path, audit)
    assert all(r["passed"] for r in verify_delivered_output(
        correct, audit_path, stage="assembled_output"))

    with pytest.raises(SceneLineageError, match="delivered_output"):
        verify_delivered_output(swapped, audit_path, stage="delivered_output")
    persisted = json.loads(audit_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["stage"] == "delivered_output"
    assert persisted["delivered_checks"]["delivered_output"]["passed"] is False


def test_black_repair_may_only_freeze_from_same_owned_beat():
    windows = [(0.0, 1.5, 0), (1.5, 3.0, 1)]
    _assert_lineage_repair_owner(1.80, 2.10, 2.40, windows, 30)
    with pytest.raises(SceneLineageError, match="neighbour donor"):
        _assert_lineage_repair_owner(1.80, 2.10, 1.40, windows, 30)
    with pytest.raises(SceneLineageError, match="crosses or lacks"):
        _assert_lineage_repair_owner(1.40, 1.70, 1.30, windows, 30)


def test_timeline_samples_exclude_transition_boundary_blend():
    # 0.4 s xfade at 30 fps occupies local frames 0..11 of the incoming
    # beat.  Canary frames begin two frames later and stay clear of its end.
    samples = _sample_local_frames(duration_frames=45, left_blend_frames=12)
    assert samples
    assert min(samples) >= 14
    assert max(samples) <= 42


def test_audit_is_persisted_on_pass_and_before_failure(tmp_path):
    audit_path = tmp_path / "scene_lineage_audit.json"
    payload = new_audit(tmp_path / "final.mp4")
    payload["stage"] = "binding"
    write_audit(audit_path, payload)
    assert json.loads(audit_path.read_text())["stage"] == "binding"

    with pytest.raises(SceneLineageError, match="scene-lineage canary failed"):
        fail_audit(audit_path, payload, "timeline_order", [{
            "stage": "timeline_order", "reason": "swapped beat",
        }])
    persisted = json.loads(audit_path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["failures"][0]["reason"] == "swapped beat"
