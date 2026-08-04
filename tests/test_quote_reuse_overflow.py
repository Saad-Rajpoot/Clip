"""Exact-quote repair may exceed only the soft reuse cap, never semantic gates."""
from __future__ import annotations

from types import SimpleNamespace

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import (
    ClipCandidate, ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo,
)


def _verdict(keep: bool) -> dict:
    return {
        "verdict": "keep" if keep else "replace",
        "matches_narration": keep,
        "correct_subject_visible": keep,
        "wrong_subject_visible": False,
        "contradicts_narration": not keep,
        "specific_enough": keep,
        "quality_ok": True,
        "era_ok": True,
        "confidence": .95,
        "reason": "exact quoted action" if keep else "different scene",
    }


def _project(tmp_path, *, quote_locked: bool = True):
    proj = ClipProject(name="quote-reuse-overflow", root=str(tmp_path))
    proj.ensure_dirs()
    shots = {}
    for sid, shot_index in (("wrong", 0), ("heavy", 1), ("light", 2), ("under", 3)):
        media = tmp_path / f"{sid}.mp4"
        frame = tmp_path / f"{sid}.jpg"
        media.write_bytes(f"media-{sid}".encode())
        frame.write_bytes(f"frame-{sid}".encode())
        proj.sources.append(SourceVideo(
            id=sid, url=f"u-{sid}", title=f"Game of Thrones {sid} scene",
            permission="owner", status="ok", local_path=str(media), duration=10.0,
            width=1920, height=1080))
        shots[(sid, shot_index)] = Shot(
            source_id=sid, index=shot_index, start=1.0, end=5.0,
            keyframe_path=str(frame), quality=.9)

    seg = ScriptSegment(
        index=0,
        text="He puts a blade to Ned Stark's throat and repeats his warning.",
        expected_visual="Littlefinger holds a blade to Ned Stark's throat.",
        required_entity="Petyr Baelish, Ned Stark", required_kind="character",
        scene_query="Littlefinger knife Ned throat I did warn you",
        quote="I did warn you not to trust me.", visual_policy=P.EXACT,
        is_specific_claim=True, est_duration=4.0)
    signals = {"quote_pool_exact": True, "dialogue": 1.0} if quote_locked else {}
    target = ClipSelection(
        segment_index=0, source_id="wrong", shot_index=0, in_point=1.0,
        out_point=5.0, confidence=.7, signals=dict(signals),
        beat_windows=[["wrong", 1.0, 5.0]])
    proj.segments = [seg]
    proj.selections = [target]
    proj.meta["analysis"] = {
        "video_type": "multi_scene", "characters": [], "actors": [],
    }

    def get_shot(sid, index):
        return shots.get((sid, index))

    get_shot.all_shots = lambda sid: [
        shot for (source_id, _index), shot in shots.items() if source_id == sid]
    return proj, seg, target, shots, get_shot


def _owner(segment_index: int, source_id: str, shot_index: int) -> ClipSelection:
    return ClipSelection(
        segment_index=segment_index, source_id=source_id, shot_index=shot_index,
        in_point=1.0, out_point=5.0, confidence=.9)


def _configure(monkeypatch, get_shot):
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_DEEP_BENCH", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VENUE_FALLBACK", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "0")


def _run(proj, seg, logs):
    return V.verify_and_repair(
        proj, [seg], ClipConfig(max_reuse_per_shot=2),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"),
        only_indices={0}, materialize_promotions=False, persist_project=False,
        progress=logs.append)


def test_exact_quote_overflow_uses_least_used_strict_candidate_and_audits_it(
        tmp_path, monkeypatch):
    proj, seg, target, _shots, get_shot = _project(tmp_path)
    # Both candidates are over the cap. The relevance-ranked candidate is more heavily reused;
    # the documented fallback must choose the least-used fully strict candidate instead.
    proj.selections.extend([
        _owner(10, "heavy", 1), _owner(11, "heavy", 1), _owner(12, "heavy", 1),
        _owner(20, "light", 2), _owner(21, "light", 2),
    ])
    target.alternates = [
        ClipCandidate(segment_index=0, source_id="heavy", shot_index=1, score=.99,
                      in_point=1.0, out_point=5.0,
                      signals={"quote_pool_exact": True, "dialogue": 1.0}),
        ClipCandidate(segment_index=0, source_id="light", shot_index=2, score=.90,
                      in_point=1.0, out_point=5.0,
                      signals={"quote_pool_exact": True, "dialogue": 1.0}),
    ]
    _configure(monkeypatch, get_shot)
    calls = []

    def judge(path, *_args, **_kwargs):
        calls.append(str(path))
        return _verdict(not str(path).endswith("wrong.jpg"))

    monkeypatch.setattr(V, "verify_frame", judge)
    logs = []
    summary = _run(proj, seg, logs)

    assert summary["replaced"] == 1
    assert (target.source_id, target.shot_index) == ("light", 2)
    assert not any(path.endswith("heavy.jpg") for path in calls), \
        "least-used ordering must not spend vision on a more heavily reused passing candidate"
    marker = V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT
    assert target.signals[marker] == 1.0
    assert target.verifier[marker] is True
    assert target.verifier["reuse_count_before"] == 2
    assert target.verifier["reuse_cap"] == 2
    assert any(marker in line and "reuse 2/2" in line for line in logs)


def test_under_cap_strict_quote_candidate_wins_before_any_overflow(
        tmp_path, monkeypatch):
    proj, seg, target, _shots, get_shot = _project(tmp_path)
    proj.selections.extend([_owner(10, "heavy", 1), _owner(11, "heavy", 1)])
    target.alternates = [
        ClipCandidate(segment_index=0, source_id="heavy", shot_index=1, score=.99,
                      in_point=1.0, out_point=5.0,
                      signals={"quote_pool_exact": True, "dialogue": 1.0}),
        ClipCandidate(segment_index=0, source_id="under", shot_index=3, score=.80,
                      in_point=1.0, out_point=5.0,
                      signals={"quote_pool_exact": True, "dialogue": 1.0}),
    ]
    _configure(monkeypatch, get_shot)
    calls = []

    def judge(path, *_args, **_kwargs):
        calls.append(str(path))
        return _verdict(not str(path).endswith("wrong.jpg"))

    monkeypatch.setattr(V, "verify_frame", judge)
    logs = []
    summary = _run(proj, seg, logs)

    assert summary["replaced"] == 1
    assert (target.source_id, target.shot_index) == ("under", 3)
    assert not any(path.endswith("heavy.jpg") for path in calls)
    assert V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT not in target.signals
    assert V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT not in target.verifier


def test_non_quote_selection_cannot_use_reuse_overflow(tmp_path, monkeypatch):
    proj, seg, target, _shots, get_shot = _project(tmp_path, quote_locked=False)
    proj.selections.extend([_owner(10, "heavy", 1), _owner(11, "heavy", 1)])
    target.alternates = [ClipCandidate(
        segment_index=0, source_id="heavy", shot_index=1, score=.99,
        in_point=1.0, out_point=5.0, signals={"dialogue": 1.0})]
    _configure(monkeypatch, get_shot)
    monkeypatch.setattr(
        V, "verify_frame",
        lambda path, *_args, **_kwargs: _verdict(not str(path).endswith("wrong.jpg")))
    logs = []

    summary = _run(proj, seg, logs)

    assert summary["replaced"] == 0
    assert (target.source_id, target.shot_index) == ("wrong", 0)
    assert V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT not in target.signals
    assert not any(V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT in line for line in logs)


def test_quote_locked_overflow_still_requires_candidate_quote_containment(
        tmp_path, monkeypatch):
    proj, seg, target, _shots, get_shot = _project(tmp_path)
    proj.selections.extend([_owner(10, "heavy", 1), _owner(11, "heavy", 1)])
    # The rejected primary is quote-locked, but this candidate carries no direct/transfer proof
    # and the fixture has no timed words from which containment could be re-derived.
    target.alternates = [ClipCandidate(
        segment_index=0, source_id="heavy", shot_index=1, score=.99,
        in_point=1.0, out_point=5.0, signals={"dialogue": 1.0})]
    _configure(monkeypatch, get_shot)
    calls = []

    def judge(path, *_args, **_kwargs):
        calls.append(str(path))
        return _verdict(not str(path).endswith("wrong.jpg"))

    monkeypatch.setattr(V, "verify_frame", judge)
    logs = []
    summary = _run(proj, seg, logs)

    assert summary["replaced"] == 0
    assert (target.source_id, target.shot_index) == ("wrong", 0)
    assert not any(path.endswith("heavy.jpg") for path in calls), \
        "quote containment must reject the overflow candidate before vision"
    assert V.REUSE_CAP_OVERFLOW_EXACT_CONTRACT not in target.signals
