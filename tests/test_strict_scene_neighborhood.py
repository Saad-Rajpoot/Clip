"""Focused regression tests for verify's bounded strict scene-neighborhood rung.

The matcher should keep its global variety/ranking behavior.  These tests cover the local failure
mode instead: a scene-affine source is already retained, but the exact action is a nearby shot that
the shallow verifier bench never received.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace as NS

from vidlore.clipstudio import image_fallback as IF
from vidlore.clipstudio import match as M
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import (ClipCandidate, ClipProject, ClipSelection, ScriptSegment,
                                       Shot, SourceVideo)


def _cand(sid: str, shot_index: int) -> ClipCandidate:
    return ClipCandidate(segment_index=0, source_id=sid, shot_index=shot_index,
                         in_point=float(shot_index), out_point=float(shot_index) + 2.5,
                         score=0.5)


def _fixture(tmp_path: Path):
    proj = ClipProject(name="neighbor", root=str(tmp_path))
    proj.ensure_dirs()
    sources = []
    for sid, title, query in (
        ("wrong", "Sansa confronts Littlefinger before the trial",
         "Sansa accuses Littlefinger trial Great Hall"),
        ("target", "Death of Lord Petyr Baelish Littlefinger",
         "Littlefinger trial Sansa Great Hall"),
    ):
        media = tmp_path / f"{sid}.mp4"
        media.write_bytes(b"media")
        sources.append(SourceVideo(id=sid, title=title, url=f"u:{sid}", local_path=str(media),
                                   status="ok", permission="owner", extra={"query": query}))
    proj.sources = sources
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}

    shots = {}
    for sid, stop in (("wrong", 13), ("target", 26)):
        rows = []
        for i in range(stop):
            kf = tmp_path / f"{sid}_{i:02d}.jpg"
            kf.write_bytes(b"jpg" + bytes([i]))
            rows.append(Shot(source_id=sid, index=i, start=float(i) * 3.0,
                             end=float(i) * 3.0 + 2.8, keyframe_path=str(kf),
                             luma_avg=40.0, subs_flag=0, graphics_flag=0,
                             static_frac=0.0, pair_diff_max=5.0, pair_diff_mean=5.0))
        shots[sid] = rows

    def get_shot(sid, index):
        return next((s for s in shots.get(sid, []) if s.index == index), None)

    get_shot.all_shots = lambda sid: shots.get(sid, [])
    seg = ScriptSegment(
        index=0, text="the cost arrives later",
        expected_visual="Littlefinger falls to his knees in the Great Hall as Sansa stares him down",
        required_entity="Petyr Baelish", required_kind="character",
        scene_query="Game of Thrones Sansa accuses Littlefinger trial Great Hall",
        visual_policy="exact_scene", is_specific_claim=True, est_duration=2.5)
    # target#5 is the retained normal seed; target#20 is a distant deep sibling and must not widen
    # the same source's scan. The exact target#11 is +6 from the normal seed.
    sel = ClipSelection(segment_index=0, source_id="wrong", shot_index=5,
                        in_point=15.0, out_point=17.5, confidence=0.8,
                        alternates=[_cand("wrong", 4), _cand("target", 5)],
                        deep_alternates=[_cand("target", 20)])
    proj.selections = [sel]
    return proj, seg, sel, get_shot


def test_default_pool_is_bounded_and_keeps_the_measured_plus_six_target(tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)

    def relevance(sh, _kf, _query, **_kwargs):
        if sh.source_id == "wrong":
            return 0.99 - sh.index / 1000.0       # cross-source visual noise is deliberately high
        # target#11 is only sixth within its correct source, matching the measured trial failure.
        return {0: .90, 1: .80, 2: .70, 3: .60, 4: .55, 11: .50}.get(sh.index, .10)

    monkeypatch.setattr(IF, "_shot_relevance", relevance)
    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(),
        exclude={("wrong", 5), ("wrong", 4), ("target", 5)})
    keys = [(c.source_id, c.shot_index) for c in out]

    assert len(out) == 12, "the default vision-attempt pool must stay capped at twelve"
    assert ("target", 11) in keys, \
        "the +6 exact shot must receive one of the correct source's six reserved strict slots"
    assert ("target", 12) not in keys, "the default radius must remain bounded at +6"
    assert not any(sid == "target" and ix >= 14 for sid, ix in keys), \
        "a distant deep seed must not widen a source that already has a normal alternate seed"
    target_rows = [c for c in out if c.source_id == "target"]
    assert [c.signals["visual_relevance"] for c in target_rows] == sorted(
        (c.signals["visual_relevance"] for c in target_rows), reverse=True), \
        "persisted/live CLIP relevance must order candidates within the affine source"
    assert all(c.signals.get("strict_scene_neighborhood") for c in out)


def test_candidate_cap_is_explicit_and_hard_bounded(tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: -1.0)

    small = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=3)
    huge = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=10_000, radius=10_000,
        source_cap=10_000)

    assert len(small) == 3
    assert len(huge) <= 24, "even hostile environment values must not unbound vision attempts"
    # With CLIP unavailable, the fallback within a source is deterministic nearest-shot ordering.
    target = [c for c in huge if c.source_id == "target"]
    distances = [c.signals["neighbor_distance"] for c in target]
    assert distances == sorted(distances)


def test_neighborhood_never_resurrects_match_gated_pixels(tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)
    target = {shot.index: shot for shot in get_shot.all_shots("target")}
    # Give every forbidden frame a higher visual score than the one good neighbor.  If even one
    # hard predicate is missing, it will enter the bounded strict-vision pool.
    target[4].quality = 0.15                    # below a configured (not default) floor
    target[6].luma_avg = target[6].luma_min = 60.0
    target[6].luma_hi = 61.0                    # featureless lit card
    target[7].quality = 0.0                     # genuine zero must not become the 1.0 default
    target[8].subs_flag = 1                     # persisted burned-sub band
    target[9].scores = {"bonus_tail": 1}       # post-scene featurette tail
    target[10].graphics_flag = 2                # hard designed graphic
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BLACK_FLOOR", "0.20")
    monkeypatch.setattr(
        IF, "_shot_relevance",
        lambda sh, *_a, **_k: 0.99 if sh.source_id == "target"
        and sh.index in {4, 6, 7, 8, 9, 10}
        else 0.25)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(),
        exclude={("wrong", 5), ("wrong", 4), ("target", 5)})
    keys = {(candidate.source_id, candidate.shot_index) for candidate in out}

    assert not ({("target", i) for i in {4, 6, 7, 8, 9, 10}} & keys)
    assert ("target", 11) in keys, "a clean sibling beside the gated frames must remain eligible"


def test_neighborhood_reconstructs_quote_signals_from_the_candidate_window(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)
    seg.quote = "Lady Stark is here in King's Landing"
    monkeypatch.setattr(
        IF, "_shot_relevance",
        lambda sh, *_a, **_k: 0.9 if sh.source_id == "target" and sh.index == 11 else 0.5)
    monkeypatch.setattr(
        M, "locate_beat_moment",
        lambda _proj, sid, _seg: (33.4, 34.7, 0.95) if sid == "target" else None)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(),
        exclude={("wrong", 5), ("wrong", 4), ("target", 5)})
    exact = next(candidate for candidate in out
                 if candidate.source_id == "target" and candidate.shot_index == 11)

    assert exact.in_point <= 33.4 and exact.out_point >= 34.7
    assert exact.signals["dialogue"] == 1.0
    assert exact.signals["moment_lock"] == 1.0
    assert exact.signals["moment_ratio"] == 0.95
    assert exact.signals["quality"] == round(get_shot("target", 11).quality, 3)


def test_neighborhood_uses_its_own_cap_and_promotes_strictly_before_downgrade(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)
    # The ordinary verifier gets only one alternate. The exact moment is fourth in the new rung;
    # inheriting max_replacements=1 would make this test fail and recreate the production bug.
    neighborhood = [_cand("target", i) for i in (6, 7, 8, 11)]
    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates",
                        lambda *_a, **_k: neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)

    calls = []

    def verdict(path, *_args, **_kwargs):
        calls.append(Path(path).name)
        keep = Path(path).name == "target_11.jpg"
        return {
            "verdict": "keep" if keep else "replace",
            "matches_narration": keep,
            "correct_subject_visible": keep,
            "wrong_subject_visible": False,
            "contradicts_narration": False,
            "specific_enough": keep,
            "quality_ok": True,
            "confidence": 0.95,
            "reason": "exact" if keep else "wrong moment",
        }

    monkeypatch.setattr(V, "verify_frame", verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=1, materialize_promotions=False, persist_project=False)

    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 11)
    assert sel.verifier.get("verdict") == "keep"
    assert not sel.verifier.get("downgraded"), "the exact rescue must not be relabeled contextual"
    assert "target_11.jpg" in calls, \
        "the neighborhood must search beyond max_replacements=1 using its own bounded cap"

    src = inspect.getsource(V.verify_and_repair)
    strict_pos = src.index("STRICT SCENE-NEIGHBORHOOD EXPANSION")
    contextual_pos = src.index("EXACT→CONTEXTUAL DOWNGRADE", strict_pos)
    assert strict_pos < contextual_pos
    assert "attempt_cap=_n_cap" in src[strict_pos:contextual_pos]


def test_strict_promotions_keep_searching_until_named_look_target_is_visible(
        tmp_path, monkeypatch):
    proj, seg, sel, get_shot = _fixture(tmp_path)
    seg.text = "Keep your eye on the dagger."
    neighborhood = [_cand("target", 6), _cand("target", 11)]
    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates",
                        lambda *_a, **_k: neighborhood)
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    asked = []

    def verdict(path, *_args, **kwargs):
        name = Path(path).name
        asked.append((name, kwargs.get("must_see", "")))
        primary = name == "wrong_05.jpg"
        visible = name == "target_11.jpg"
        return {
            # Deliberately inconsistent keep+target_visible=false on the earlier alternates.  The
            # deterministic acceptance contract must reject that malformed optimism and continue.
            "verdict": "replace" if primary else "keep",
            "matches_narration": not primary,
            "correct_subject_visible": not primary,
            "wrong_subject_visible": False,
            "contradicts_narration": False,
            "specific_enough": not primary,
            "quality_ok": True,
            "target_visible": visible,
            "confidence": 0.95,
            "reason": "target visible" if visible else "named target absent",
        }

    monkeypatch.setattr(V, "verify_frame", verdict)
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=2, materialize_promotions=False, persist_project=False)

    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 11)
    assert sel.verifier.get("target_visible") is True
    assert sel.verifier["selection_evidence"]["must_see"] == "the dagger"
    assert all(must_see == "the dagger" for name, must_see in asked
               if name != "wrong_05.jpg"), \
        "every strict replacement must answer the same named-target question as the primary"
    assert "target_06.jpg" in {name for name, _ in asked}, \
        "a malformed keep with target_visible=false must not stop the strict search"
