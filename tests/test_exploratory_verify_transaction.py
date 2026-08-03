"""Verifier/recovery transaction regressions.

Exploratory scoped recovery may mutate selection metadata for strict evaluation, but it must not
touch deterministic ``seg_NNN.mp4`` bytes until that selection has passed the publication contract.
"""
import copy
from types import SimpleNamespace

import pytest

from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import (
    ClipCandidate,
    ClipProject,
    ClipSelection,
    ScriptSegment,
    Shot,
    SourceVideo,
)


_KEEP = {
    "verdict": "keep",
    "confidence": 0.95,
    "matches_narration": True,
    "specific_enough": True,
    "correct_subject_visible": True,
    "wrong_subject_visible": False,
    "contradicts_narration": False,
    "quality_ok": True,
}
_REPLACE = {
    **_KEEP,
    "verdict": "replace",
    "matches_narration": False,
    "specific_enough": False,
    "correct_subject_visible": False,
}


def _promotion_fixture(tmp_path):
    proj = ClipProject(name="exploratory-verify", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source bytes")
    proj.sources = [SourceVideo(
        id="s1", url="u", title="Game of Thrones exact council scene",
        permission="owner", status="ok", local_path=str(media))]

    shots = []
    for idx in range(2):
        frame = tmp_path / f"shot_{idx}.jpg"
        frame.write_bytes(b"\xff\xd8\xff" + bytes([idx]))
        shots.append(Shot(source_id="s1", index=idx, start=idx * 3.0,
                          end=idx * 3.0 + 2.0, keyframe_path=str(frame)))

    seg = ScriptSegment(
        index=0, text="Varys learns the truth in the Red Keep.",
        expected_visual="Varys in the Red Keep",
        scene_query="Game of Thrones Varys Red Keep",
        required_entity="Varys", required_kind="character",
        visual_policy=P.EXACT, is_specific_claim=True)
    sel = ClipSelection(
        segment_index=0, source_id="s1", shot_index=0,
        in_point=0.0, out_point=2.0, confidence=0.7,
        alternates=[ClipCandidate(
            segment_index=0, source_id="s1", shot_index=1,
            score=0.9, in_point=3.0, out_point=5.0)])
    proj.segments = [seg]
    proj.selections = [sel]
    proj.meta["analysis"] = {
        "video_type": "multi_scene", "characters": [], "actors": []}
    return proj, seg, sel, shots


def _stub_real_promotion(monkeypatch, shots, cut_selection):
    by_shot = {(s.source_id, s.index): s for s in shots}

    def lookup(_proj):
        def get(source_id, shot_index):
            return by_shot.get((source_id, shot_index))
        get.all_shots = lambda source_id: [
            s for s in shots if s.source_id == source_id]
        return get

    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    monkeypatch.setattr(V, "_shot_lookup", lookup)
    monkeypatch.setattr(
        V, "verify_frame",
        lambda frame, *_args, **_kwargs: dict(
            _REPLACE if frame == shots[0].keyframe_path else _KEEP))
    monkeypatch.setattr(V._cut, "cut_selection", cut_selection)


@pytest.mark.parametrize("failure", ["none", "partial_none", "partial_exception"])
def test_materialized_promotion_cut_failure_restores_selection_and_existing_clip(
        tmp_path, monkeypatch, failure):
    proj, seg, sel, shots = _promotion_fixture(tmp_path)
    clip = proj.clips_dir / "seg_000.mp4"
    clip.write_bytes(b"ORIGINAL CLIP BYTES")
    sel.clip_path = str(clip)
    proj.save()

    def broken_cut(_proj, _sel, _cfg, **_kwargs):
        if failure.startswith("partial"):
            clip.write_bytes(b"PARTIAL PROMOTED BYTES")
        if failure == "partial_exception":
            raise RuntimeError("ffmpeg died after truncating output")
        return None

    _stub_real_promotion(monkeypatch, shots, broken_cut)
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"), progress=None)

    assert summary["replaced"] == 0
    assert summary["errored"] > 0
    assert summary["materialization_errors"] == 1
    assert (sel.source_id, sel.shot_index, sel.in_point, sel.out_point) == (
        "s1", 0, 0.0, 2.0)
    assert sel.clip_path == str(clip)
    assert sel.verifier["verdict"] == "replace", (
        "the real primary semantic rejection remains; only the uncommitted promotion rolls back")
    assert clip.read_bytes() == b"ORIGINAL CLIP BYTES"
    assert not clip.with_suffix(".mp4.verify_rollback.tmp").exists()

    persisted = ClipProject.load(tmp_path)
    disk_sel = persisted.selections[0]
    assert (disk_sel.source_id, disk_sel.shot_index,
            disk_sel.in_point, disk_sel.out_point) == ("s1", 0, 0.0, 2.0)
    assert disk_sel.clip_path == str(clip)
    assert clip.read_bytes() == b"ORIGINAL CLIP BYTES"


def test_failed_promotion_removes_partial_clip_when_no_original_existed(tmp_path, monkeypatch):
    proj, seg, sel, shots = _promotion_fixture(tmp_path)
    clip = proj.clips_dir / "seg_000.mp4"
    assert not clip.exists()
    proj.save()

    def partial_cut(_proj, _sel, _cfg, **_kwargs):
        clip.write_bytes(b"NEW PARTIAL FILE")
        return None

    _stub_real_promotion(monkeypatch, shots, partial_cut)
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"), progress=None)

    assert summary["materialization_errors"] == 1 and summary["errored"] > 0
    assert not clip.exists()
    assert (sel.source_id, sel.shot_index) == ("s1", 0)
    persisted = ClipProject.load(tmp_path)
    assert (persisted.selections[0].source_id,
            persisted.selections[0].shot_index) == ("s1", 0)


def test_successful_materialized_promotion_commits_metadata_clip_and_accounting(
        tmp_path, monkeypatch):
    proj, seg, sel, shots = _promotion_fixture(tmp_path)
    clip = proj.clips_dir / "seg_000.mp4"
    clip.write_bytes(b"ORIGINAL CLIP")
    sel.clip_path = str(clip)
    proj.save()

    def successful_cut(_proj, _sel, _cfg, **_kwargs):
        clip.write_bytes(b"COMPLETE PROMOTED CLIP")
        return clip

    _stub_real_promotion(monkeypatch, shots, successful_cut)
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"), progress=None)

    assert summary["replaced"] == 1
    assert summary["errored"] == summary["materialization_errors"] == 0
    assert (sel.source_id, sel.shot_index, sel.in_point, sel.out_point) == (
        "s1", 1, 3.0, 5.0)
    assert sel.clip_path == str(clip)
    assert clip.read_bytes() == b"COMPLETE PROMOTED CLIP"
    persisted = ClipProject.load(tmp_path)
    assert persisted.selections[0].shot_index == 1
    assert persisted.selections[0].clip_path == str(clip)


def test_exploratory_verifier_promotion_does_not_materialize_clip(tmp_path, monkeypatch):
    """Exploration stays off disk until one commit writes matching metadata and clip bytes."""
    proj, seg, sel, shots = _promotion_fixture(tmp_path)
    clip = proj.clips_dir / "seg_000.mp4"
    clip.write_bytes(b"ORIGINAL CLIP")
    sel.clip_path = str(clip)
    proj.save()
    snapshot = {s.segment_index: copy.deepcopy(s) for s in proj.selections}
    by_shot = {(s.source_id, s.index): s for s in shots}

    def lookup(_proj):
        def get(source_id, shot_index):
            return by_shot.get((source_id, shot_index))
        get.all_shots = lambda source_id: [
            s for s in shots if s.source_id == source_id]
        return get

    cut_calls = []

    def cut_selection(_proj, _sel, _cfg, **_kwargs):
        cut_calls.append(_sel.shot_index)
        clip.write_bytes(b"PROMOTED CLIP")
        return clip

    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    monkeypatch.setattr(V, "_shot_lookup", lookup)
    monkeypatch.setattr(
        V, "verify_frame",
        lambda frame, *_args, **_kwargs: dict(
            _REPLACE if frame == shots[0].keyframe_path else _KEEP))
    monkeypatch.setattr(V._cut, "cut_selection", cut_selection)

    result = V.verify_and_repair(
        proj, [seg], ClipConfig(),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"),
        materialize_promotions=False, persist_project=False, progress=None)

    assert result["replaced"] == 1
    assert sel.shot_index == 1, "strict evaluation still needs the promoted metadata"
    assert cut_calls == []
    assert clip.read_bytes() == b"ORIGINAL CLIP"
    assert (tmp_path / "verdict_cache.json").is_file(), (
        "exploratory mode must retain durable paid verifier verdicts")
    persisted_before_commit = ClipProject.load(tmp_path)
    assert persisted_before_commit.selections[0].shot_index == 0

    rematched = {s.segment_index: s for s in proj.selections}
    assert O._commit_scoped_recovery(
        proj, ClipConfig(), snapshot, rematched, {0}, log=lambda _m: None)
    assert cut_calls == [1]
    assert clip.read_bytes() == b"PROMOTED CLIP"
    persisted_after_commit = ClipProject.load(tmp_path)
    committed = persisted_after_commit.selections[0]
    assert (committed.shot_index, committed.in_point, committed.out_point) == (1, 3.0, 5.0)
    assert committed.clip_path == str(clip)


def test_rejected_scoped_promotion_leaves_existing_clip_bytes_untouched(
        tmp_path, monkeypatch):
    """A strict-rejected exploratory promotion must restore metadata without stale clip bytes."""
    proj, seg, original, _shots = _promotion_fixture(tmp_path)
    clip = proj.clips_dir / "seg_000.mp4"
    clip.write_bytes(b"ORIGINAL CLIP")
    original.clip_path = str(clip)
    proj.save()
    modes = []

    def rematch(_proj, _segs, _cfg, **_kwargs):
        _proj.selections = [ClipSelection(
            segment_index=0, source_id="s1", shot_index=0,
            in_point=0.0, out_point=2.0, confidence=0.8)]
        return _proj.selections

    def exploratory_verify(_proj, *_args, **kwargs):
        materialize = kwargs.get("materialize_promotions", True)
        persist = kwargs.get("persist_project", True)
        modes.append((materialize, persist))
        _proj.selections[0].shot_index = 1
        _proj.selections[0].in_point = 3.0
        _proj.selections[0].out_point = 5.0
        _proj.selections[0].verifier = dict(_KEEP)
        if materialize:
            clip.write_bytes(b"EXPLORATORY PROMOTION")
        if persist:
            _proj.save()
        return {"verified": 1, "replaced": 1, "failed": 0, "available": True,
                "errored": 1, "verifier_down": False}

    monkeypatch.setattr(O, "match_segments", rematch)
    monkeypatch.setattr(V, "verify_and_repair", exploratory_verify)
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio import relevance_contract as R
    monkeypatch.setattr(D, "discover_sources", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(R, "evaluate_selection_relevance", lambda *_args, **_kwargs: {
        "status": "blocked", "blocked_count": 1,
        "blockers": [{"segment_index": 0, "reasons": ["still_wrong"]}],
    })

    recovered = O._recover_unresolved_beats(
        proj, [seg], SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
        SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
        policy="approved_testing", log=lambda _message: None, only_indices={0},
        audit_filename="exploratory-transaction.json")

    assert recovered == 0
    assert modes == [(False, False)], (
        "exploratory verification must neither materialize nor persist promotions")
    restored = proj.selections[0]
    assert (restored.shot_index, restored.in_point, restored.out_point) == (0, 0.0, 2.0)
    assert clip.read_bytes() == b"ORIGINAL CLIP"
    persisted = ClipProject.load(tmp_path)
    disk_sel = persisted.selections[0]
    assert (disk_sel.shot_index, disk_sel.in_point, disk_sel.out_point) == (0, 0.0, 2.0)
    assert disk_sel.clip_path == str(clip)
