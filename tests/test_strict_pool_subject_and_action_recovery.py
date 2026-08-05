from pathlib import Path
from types import SimpleNamespace as NS

from vidlore.clipstudio import image_fallback as IF
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


def _candidate(segment_index: int, sid: str, shot_index: int) -> ClipCandidate:
    return ClipCandidate(
        segment_index=segment_index, source_id=sid, shot_index=shot_index,
        score=0.9, in_point=float(shot_index) * 3.0,
        out_point=float(shot_index) * 3.0 + 2.5)


def _repair_fixture(tmp_path: Path, *, policy: str, shot_intent: str = ""):
    proj = ClipProject(name="strict-pool-repair", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}
    shots = {}
    for sid in ("wrong", "target"):
        media = tmp_path / f"{sid}.mp4"
        media.write_bytes(b"native media")
        proj.sources.append(SourceVideo(
            id=sid, title=f"{sid} scene", url=f"u:{sid}", local_path=str(media),
            status="ok", permission="owner"))
        rows = []
        for index in range(3):
            keyframe = tmp_path / f"{sid}_{index}.jpg"
            keyframe.write_bytes(b"frame")
            rows.append(Shot(
                source_id=sid, index=index, start=float(index) * 3.0,
                end=float(index) * 3.0 + 2.8, keyframe_path=str(keyframe),
                quality=0.9, luma_avg=40.0, subs_flag=0, graphics_flag=0,
                static_frac=0.0, pair_diff_max=5.0, pair_diff_mean=5.0))
        shots[sid] = rows

    def get_shot(sid, index):
        return next((shot for shot in shots.get(sid, []) if shot.index == index), None)

    get_shot.all_shots = lambda sid: shots.get(sid, [])
    exact = policy == "exact_scene"
    seg = ScriptSegment(
        index=0, text="The required subject appears in the exact action.",
        expected_visual="The required subject in the target scene",
        scene_query="Game of Thrones required subject target action",
        required_entity="Required Subject", required_kind="character",
        visual_policy=policy, is_specific_claim=exact, shot_intent=shot_intent,
        est_duration=2.5)
    sel = ClipSelection(
        segment_index=0, source_id="wrong", shot_index=0,
        in_point=0.0, out_point=2.5, confidence=0.8)
    proj.segments = [seg]
    proj.selections = [sel]
    return proj, seg, sel, get_shot


def _vision_verdict(path, *_args, **_kwargs):
    keep = Path(path).name.startswith("target_")
    return {
        "verdict": "keep" if keep else "replace",
        "matches_narration": keep,
        "correct_subject_visible": keep,
        "wrong_subject_visible": not keep,
        "contradicts_narration": not keep,
        "specific_enough": keep,
        "quality_ok": True,
        "confidence": 0.95,
        "reason": "required subject" if keep else "wrong subject",
    }


def test_scoped_character_subject_miss_can_use_indexed_pool_neighborhood(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _repair_fixture(tmp_path, policy="character_specific")
    calls = []

    def neighborhood(*_args, **kwargs):
        calls.append(dict(kwargs))
        return [_candidate(0, "target", 1)]

    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates", neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(V, "verify_frame", _vision_verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")

    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=0, materialize_promotions=False, persist_project=False,
        strict_pool_recovery=True)

    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 1)
    assert sel.verifier["correct_subject_visible"] is True
    assert calls and calls[0]["allow_indexed_pool_sources"] is True
    assert not calls[0].get("whole_source_probe", False)


def test_normal_character_verification_does_not_open_indexed_pool_neighborhood(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _repair_fixture(tmp_path, policy="character_specific")
    calls = []

    def neighborhood(*_args, **_kwargs):
        calls.append(True)
        return [_candidate(0, "target", 1)]

    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates", neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(V, "verify_frame", _vision_verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")

    V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=0, materialize_promotions=False, persist_project=False,
        strict_pool_recovery=False)

    assert calls == []
    assert (sel.source_id, sel.shot_index) == ("wrong", 0)


def test_scoped_recovery_exhausts_bounded_matcher_head_before_network(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _repair_fixture(tmp_path, policy="exact_scene")
    # Production keeps a bounded six-source head, while ordinary verification intentionally asks
    # only three for latency.  The real ee93371e41 miss was a strict-passing frame at head rank 6.
    sel.alternates = [
        _candidate(0, "wrong", 1),
        _candidate(0, "wrong", 2),
        _candidate(0, "wrong", 0),
        _candidate(0, "wrong", 1),
        _candidate(0, "wrong", 2),
        _candidate(0, "target", 1),
    ]

    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(V, "verify_frame", _vision_verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")

    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=3, materialize_promotions=False, persist_project=False,
        strict_pool_recovery=True)

    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 1)
    assert sel.verifier["verdict"] == "keep"
    assert not sel.verifier.get("downgraded")


def test_exact_action_uses_five_call_whole_source_probe_after_local_miss(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _repair_fixture(
        tmp_path, policy="exact_scene", shot_intent="action")
    calls = []

    def neighborhood(*_args, **kwargs):
        whole = bool(kwargs.get("whole_source_probe"))
        calls.append((whole, kwargs.get("cap"), kwargs.get("source_cap")))
        return [_candidate(0, "target" if whole else "wrong", 2 if whole else 1)]

    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates", neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(V, "verify_frame", _vision_verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")

    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=0, materialize_promotions=False, persist_project=False,
        strict_pool_recovery=True)

    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 2)
    assert calls == [(False, 12, 4), (True, 5, 1)]


def test_whole_source_action_probe_cannot_promote_wqc_rejected_candidate(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _repair_fixture(
        tmp_path, policy="exact_scene", shot_intent="action")

    def neighborhood(*_args, **kwargs):
        if kwargs.get("whole_source_probe"):
            return [_candidate(0, "target", 2)]
        return []

    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates", neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(V, "verify_frame", _vision_verdict)
    monkeypatch.setattr(
        "vidlore.clipstudio.match.validate_candidate_window",
        lambda cand, *_args, **_kwargs: (
            "rejected", "unreadable(shot 2)", {
                "policy": "exact",
                "orig": (cand.in_point, cand.out_point),
                "final": (cand.in_point, cand.out_point),
                "preserved": False,
            }))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1")

    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=0, materialize_promotions=False, persist_project=False,
        strict_pool_recovery=True)

    assert summary["replaced"] == 0
    assert (sel.source_id, sel.shot_index) == ("wrong", 0)


def test_whole_source_probe_is_native_anchor_bound_disjoint_and_capped(
        tmp_path, monkeypatch):
    proj = ClipProject(name="whole-source-probe", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {
        "movie_title": "Game of Thrones",
        "anchor_scenes": [{
            "name": "Oberyn death in the Mountain trial by combat",
            "query": "Game of Thrones Oberyn Mountain death S04E08",
            "episode": "S04E08 The Mountain and the Viper",
        }],
    }
    media = tmp_path / "fight.mp4"
    media.write_bytes(b"native hd source")
    proj.sources = [SourceVideo(
        id="fight", title="Game of Thrones S04E08 Oberyn vs Mountain full fight",
        url="u:fight", local_path=str(media), status="ok", permission="owner",
        extra={"query": "Game of Thrones Oberyn Mountain death S04E08"})]
    shots = []
    for index in range(36):
        keyframe = tmp_path / f"fight_{index:02d}.jpg"
        keyframe.write_bytes(b"frame")
        shots.append(Shot(
            source_id="fight", index=index, start=float(index) * 3.0,
            end=float(index) * 3.0 + 2.8, keyframe_path=str(keyframe),
            quality=0.9, luma_avg=40.0, subs_flag=0, graphics_flag=0,
            static_frac=0.0, pair_diff_max=5.0, pair_diff_mean=5.0))

    def get_shot(sid, index):
        return next((shot for shot in shots if sid == "fight" and shot.index == index), None)

    get_shot.all_shots = lambda sid: shots if sid == "fight" else []
    seg = ScriptSegment(
        index=0, text="He never got it.",
        expected_visual="Oberyn's fatal death as the Mountain crushes his head",
        scene_query="Game of Thrones Oberyn Mountain death head crushed",
        required_entity="Oberyn Martell", required_kind="character",
        visual_policy="exact_scene", is_specific_claim=True, shot_intent="action",
        est_duration=2.0)
    sel = ClipSelection(
        segment_index=0, source_id="fight", shot_index=10,
        in_point=30.0, out_point=32.5, confidence=0.8,
        deep_alternates=[ClipCandidate(
            segment_index=0, source_id="fight", shot_index=20, score=0.9,
            in_point=60.0, out_point=62.5, signals={"clip": 0.95})])
    proj.segments = [seg]
    proj.selections = [sel]

    relevance = {27: 0.99, 28: 0.98, 29: 0.97, 30: 0.96, 31: 0.95, 32: 0.94}
    shots[27].luma_avg = 5.0
    shots[27].luma_hi = 20.0
    monkeypatch.setattr(
        IF, "_shot_relevance",
        lambda shot, *_a, **_k: relevance.get(shot.index, 0.1))
    from vidlore.clipstudio import quality_contract as QC
    monkeypatch.setattr(QC, "probe_native_video_info",
                        lambda _path: {"width": 1920, "height": 1080})

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=99, radius=6, source_cap=8,
        allow_indexed_pool_sources=True, whole_source_probe=True)

    assert len(out) == 5
    assert [candidate.shot_index for candidate in out] == [28, 29, 30, 31, 32]
    assert all(candidate.signals["strict_whole_source_probe"] is True for candidate in out)
    assert all(abs(candidate.shot_index - 10) > 6
               and abs(candidate.shot_index - 20) > 6 for candidate in out)
    assert V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=5, radius=6,
        allow_indexed_pool_sources=False, whole_source_probe=True) == []
