"""Regression coverage for exact silent-reaction scene binding.

The measured production failure put Tywin's unrelated marriage conversation over a claim that he
silently grants Tyrion a trial by combat.  Vision hallucinated three strict keeps.  These tests
prove that local timed scene evidence, not a right-actor face, now decides that narrow shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from vidlore.clipstudio import image_fallback as IF
from vidlore.clipstudio import index as I
from vidlore.clipstudio import relevance_contract as R
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


def _source(tmp_path: Path, sid: str, title: str) -> SourceVideo:
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes((sid + "-native-media").encode())
    return SourceVideo(
        id=sid,
        title=title,
        url=f"u:{sid}",
        local_path=str(media),
        status="ok",
        permission="owner",
        width=1280,
        height=720,
        extra={"query": title},
    )


def _shots(tmp_path: Path, sid: str, *, dialogue: str = "") -> list[Shot]:
    rows = []
    for index in range(30):
        keyframe = tmp_path / f"{sid}_{index:02d}.jpg"
        keyframe.write_bytes(b"jpeg" + bytes([index]))
        rows.append(Shot(
            source_id=sid,
            index=index,
            start=float(index) * 4.0,
            end=float(index) * 4.0 + 3.5,
            keyframe_path=str(keyframe),
            transcript=(dialogue if index == 20 else ""),
            face_ids=["Charles Dance"] if index == 24 else [],
            quality=0.9,
            luma_avg=45.0,
            luma_hi=150.0,
            luma_min=30.0,
            luma_min_black_frac=0.1,
            subs_flag=0,
            graphics_flag=0,
            static_frac=0.0,
            pair_diff_max=5.0,
            pair_diff_mean=5.0,
        ))
    return rows


def _write_current_index(
        proj: ClipProject, sid: str, shots: list[Shot], cfg: ClipConfig) -> None:
    proj.shots_path(sid).write_text(
        json.dumps([shot.to_dict() for shot in shots]), encoding="utf-8")
    meta = {
        "schema": I.INDEX_SCHEMA,
        "words": True,
        "asr_prompt_fingerprint": I.asr_semantic_fingerprint(proj, cfg),
    }
    (Path(proj.index_dir) / f"{sid}.index.meta.json").write_text(
        json.dumps(meta), encoding="utf-8")


def _fixture(tmp_path: Path):
    cfg = ClipConfig()
    proj = ClipProject(name="reaction-context", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {
        "movie_title": "Game of Thrones",
        "characters": [
            {"name": "Tywin Lannister", "actor": "Charles Dance"},
            {"name": "Tyrion Lannister", "actor": "Peter Dinklage"},
            {"name": "Sansa Stark", "actor": "Sophie Turner"},
        ],
    }
    proj.sources = [
        _source(tmp_path, "marriage", "Tywin commands Tyrion to marry Sansa"),
        _source(tmp_path, "trial", "The Epic Trial by Combat of Tyrion Lannister"),
    ]
    shots = {
        "marriage": _shots(tmp_path, "marriage", dialogue="You will marry Sansa Stark."),
        "trial": _shots(tmp_path, "trial", dialogue="I demand a trial by combat."),
    }
    for sid, rows in shots.items():
        _write_current_index(proj, sid, rows, cfg)
    seg = ScriptSegment(
        index=0,
        text="He does not argue, and he does not refuse.",
        expected_visual=(
            "Tywin Lannister grants Tyrion's demand for a trial by combat without argument"),
        scene_query="Game of Thrones Tywin grants trial by combat Tyrion",
        required_entity="Tywin Lannister",
        required_kind="character",
        visual_policy="exact_scene",
        is_specific_claim=True,
        shot_intent="reaction",
        entities=["Tyrion Lannister"],
        est_duration=4.0,
    )
    return proj, seg, shots, cfg


def _selection(sid: str) -> ClipSelection:
    return ClipSelection(
        segment_index=0,
        source_id=sid,
        shot_index=24,
        in_point=96.0,
        out_point=99.5,
        confidence=1.0,
        identity="Charles Dance",
    )


def _positive_verifier() -> dict:
    return {
        "status": "ok",
        "verdict": "keep",
        "matches_narration": True,
        "specific_enough": True,
        "correct_subject_visible": True,
        "wrong_subject_visible": False,
        "contradicts_narration": False,
        "quality_ok": True,
        "selection_evidence": {"is_specific": True},
    }


def test_wrong_marriage_scene_fails_but_trial_reaction_has_timed_context(tmp_path):
    proj, seg, _rows, cfg = _fixture(tmp_path)

    wrong = V._exact_reaction_context_evidence(proj, _selection("marriage"), seg, cfg=cfg)
    correct = V._exact_reaction_context_evidence(proj, _selection("trial"), seg, cfg=cfg)

    assert wrong["required"] is True and wrong["passed"] is False
    assert wrong["reason"] == "exact_reaction_context_unproven"
    assert wrong["matched_terms"] == []
    assert correct["passed"] is True
    assert set(correct["matched_terms"]) == {"demand", "trial", "combat"}
    assert correct["context_shots"] == [20, 20]
    assert "tywin" not in correct["target_terms"]
    assert "tyrion" not in correct["target_terms"]


def test_final_publication_gate_recomputes_reaction_context(tmp_path, monkeypatch):
    proj, seg, _rows, cfg = _fixture(tmp_path)
    proj.segments = [seg]
    sel = _selection("marriage")
    sel.verifier = _positive_verifier()
    proj.selections = [sel]
    # The test is about the new independent context fact; bind-shape validation has its own suite.
    monkeypatch.setattr(R, "selection_verifier_evidence_reason", lambda *_a, **_k: "")

    blocked = R.evaluate_selection_relevance(proj, [seg], cfg=cfg)
    assert blocked["blocked_count"] == 1
    assert "exact_reaction_context_unproven" in blocked["blockers"][0]["reasons"]
    assert blocked["blockers"][0]["exact_reaction_context"]["passed"] is False

    sel.source_id = "trial"
    sel.verifier = _positive_verifier()
    passed = R.evaluate_selection_relevance(proj, [seg], cfg=cfg)
    assert passed["blocked_count"] == 0
    assert passed["checked"][0]["exact_reaction_context"]["passed"] is True


def test_stale_asr_sidecar_cannot_prove_reaction_context(tmp_path):
    proj, seg, _rows, cfg = _fixture(tmp_path)
    meta_path = Path(proj.index_dir) / "trial.index.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["asr_prompt_fingerprint"] = "stale"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    evidence = V._exact_reaction_context_evidence(
        proj, _selection("trial"), seg, cfg=cfg)

    assert evidence["passed"] is False
    assert evidence["reason"] == "exact_reaction_context_asr_provenance_invalid"


def test_scene_affinity_needs_event_terms_not_two_character_names(tmp_path):
    proj, seg, _rows, _cfg = _fixture(tmp_path)
    marriage = ClipCandidate(
        segment_index=0, source_id="marriage", shot_index=24,
        in_point=96.0, out_point=99.5, score=1.0)
    trial = ClipCandidate(
        segment_index=0, source_id="trial", shot_index=24,
        in_point=96.0, out_point=99.5, score=0.8)

    ordered = V._scene_affinity_order([marriage, trial], seg, proj, "neither")

    assert [candidate.source_id for candidate in ordered] == ["trial", "marriage"]


def test_reaction_timed_anchor_exposes_fourth_forward_edit_under_same_cap(
        tmp_path, monkeypatch):
    proj, seg, rows, cfg = _fixture(tmp_path)
    selected = ClipSelection(
        segment_index=0,
        source_id="marriage",
        shot_index=2,
        in_point=8.0,
        out_point=11.5,
        confidence=0.8,
        deep_alternates=[ClipCandidate(
            segment_index=0,
            source_id="trial",
            shot_index=2,
            in_point=8.0,
            out_point=11.5,
            score=0.8,
            signals={"clip": 0.88},
        )],
    )

    def get_shot(sid, index):
        return next((shot for shot in rows[sid] if shot.index == index), None)

    get_shot.all_shots = lambda sid: rows[sid]
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.6)

    candidates = V._strict_scene_neighborhood_candidates(
        selected, seg, proj, get_shot, cfg, cap=12, source_cap=1)
    timed = [candidate for candidate in candidates
             if candidate.signals.get("strict_scene_timed_text_region")]

    assert len(candidates) <= 12
    assert any(candidate.source_id == "trial" and candidate.shot_index == 24
               for candidate in timed), \
        "the +4 Tywin reaction must be visible inside the unchanged five-call reserve"
    assert len(timed) <= 5
