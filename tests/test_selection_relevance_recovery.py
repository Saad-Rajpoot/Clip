"""Scoped final recovery for the mandatory semantic publication contract."""
from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo


GOOD = {
    "status": "ok", "verdict": "keep", "matches_narration": True,
    "specific_enough": True, "correct_subject_visible": True,
    "wrong_subject_visible": False, "contradicts_narration": False,
    "quality_ok": True, "target_visible": True,
}


def _fixture(tmp_path, verifier):
    proj = ClipProject(name="semantic-retry", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source")
    proj.sources = [SourceVideo(
        id="s1", url="u", title="Game of Thrones exact scene", permission="owner",
        status="ok", local_path=str(media))]
    frame = tmp_path / "shot_0000.jpg"
    frame.write_bytes(b"necklace-scene-frame")
    shot = Shot(source_id="s1", index=0, start=0.0, end=2.0,
                keyframe_path=str(frame))
    proj.shots_path("s1").write_text(json.dumps([shot.to_dict()]))
    proj.meta["analysis"] = {"video_type": "multi_scene", "characters": [], "actors": []}
    seg = ScriptSegment(
        index=0, text="Olenna removes the stone from Sansa's necklace.",
        expected_visual="Olenna's hand removes a stone from Sansa's necklace",
        required_entity="necklace stone", required_kind="object",
        scene_query="Game of Thrones Olenna Sansa necklace stone",
        visual_policy=P.EXACT, is_specific_claim=True)
    sel = ClipSelection(
        segment_index=0, source_id="s1", shot_index=0, in_point=0.0, out_point=2.0,
        confidence=0.8, verifier=dict(verifier))
    # Full semantic verdicts model a previously completed verification pass.  Deliberately partial
    # ones remain unbound so the scoped-reverify test exercises legacy/missing evidence repair.
    if all(k in sel.verifier for k in (
            "matches_narration", "specific_enough", "wrong_subject_visible", "quality_ok")):
        V.bind_selection_verifier_evidence(
            proj, sel, seg, sel.verifier, shot=shot, model="vision",
            is_specific=True, multiframe=True, faceid_names=[], era="", must_see="")
    proj.segments = [seg]
    proj.selections = [sel]
    return proj, [seg], sel


def _call(proj, segs):
    return O._retry_selection_relevance(
        proj, segs, ClipConfig(), SimpleNamespace(movie_title="Game of Thrones"),
        SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
        policy="approved_testing", log=lambda _m: None)


def test_missing_evidence_gets_scoped_reverify_before_acquisition(tmp_path):
    proj, segs, sel = _fixture(tmp_path, {"status": "ok", "verdict": "keep"})

    def reverify(_proj, _segs, _cfg, _eng, *, only_indices, progress):
        assert only_indices == {0}
        sel.verifier = dict(GOOD)
        V.bind_selection_verifier_evidence(
            proj, sel, segs[0], sel.verifier, model="vision", is_specific=True,
            multiframe=True, faceid_names=[], era="", must_see="")
        return {"verifier_down": False}

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair", side_effect=reverify) as rv, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills:
        audit = _call(proj, segs)
    assert audit["status"] == "pass"
    rv.assert_called_once()
    recover.assert_not_called()
    stills.assert_not_called()


def test_explicit_negative_skips_duplicate_reverify_and_recovers_only_blocker(tmp_path):
    proj, segs, sel = _fixture(
        tmp_path, {**GOOD, "verdict": "replace", "matches_narration": False,
                   "specific_enough": False, "correct_subject_visible": False})

    def recover(_proj, _segs, _analysis, _cfg, _eng, **kw):
        assert kw["only_indices"] == {0}
        assert kw["audit_filename"] == "semantic_recovery_audit.json"
        sel.verifier = dict(GOOD)
        V.bind_selection_verifier_evidence(
            proj, sel, segs[0], sel.verifier, model="vision", is_specific=True,
            multiframe=True, faceid_names=[], era="", must_see="")
        return 1

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair") as rv, \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=recover) as rec, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills:
        audit = _call(proj, segs)
    assert audit["status"] == "pass"
    rv.assert_not_called(), "an explicit cached rejection must not be re-asked unchanged"
    rec.assert_called_once()
    stills.assert_not_called()


def test_legacy_still_is_reverified_on_its_actual_bytes_before_acquisition(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    still = tmp_path / "legacy-web.jpg"
    still.write_bytes(b"actual-image")
    sel.image_path = str(still)
    sel.image_meta = {"source": "web-exact-scene", "relevance_class": "exact_scene"}

    with mock.patch("vidlore.clipstudio.verify.verify_frame", return_value=dict(GOOD)) as judge, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as fallback:
        audit = _call(proj, segs)
    assert audit["status"] == "pass"
    judge.assert_called_once()
    recover.assert_not_called()
    fallback.assert_not_called()
    assert sel.image_meta["still_semantic_verified"] is True
    assert sel.image_meta["exact_still_verified"] is True
    assert sel.image_meta["still_image_sha256"]


def test_failed_retry_stays_blocked_and_unchanged_resume_does_not_repeat(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    with mock.patch.object(O, "_recover_unresolved_beats", return_value=0) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks", return_value=0) as stills:
        first = _call(proj, segs)
        second = _call(proj, segs)
    assert first["status"] == second["status"] == "blocked"
    assert recover.call_count == 1
    assert stills.call_count == 1
    marker = proj.meta["selection_relevance_recovery"]
    assert marker["before"] == marker["after"] == [0]
    assert marker["post_fingerprint"]


def test_orchestrate_runs_strict_recovery_then_confirmed_gap_ladder_then_asserts():
    src = inspect.getsource(O.produce_auto)
    branch = src[src.index('== "selection_relevance"'):src.index(
        '== "rejected_footage"')]
    assert "_retry_selection_relevance(" in branch
    assert "_assert_sr(" in branch

    retry = inspect.getsource(O._retry_selection_relevance)
    strict_recover = retry.index("_recover_unresolved_beats(")
    strict_images = retry.index("_fill_image_fallbacks(")
    gap_ladder = retry.index("heal_selection_relevance_gaps(")
    final_contract = retry.index("final = _R_sr.evaluate_selection_relevance", gap_ladder)
    assert strict_recover < gap_ladder and strict_images < gap_ladder < final_contract

    fallback = inspect.getsource(O._fill_image_fallbacks)
    assert '"still_semantic_verified"' in fallback
    assert '"still_verifier"' in fallback
    assert '"still_image_sha256"' in fallback
    assert "is_specific=True" in fallback


def test_semantic_gap_ladder_requires_actual_frame_confirmation(tmp_path):
    from vidlore.clipstudio import selfheal as S
    proj, segs, _sel = _fixture(
        tmp_path, {**GOOD, "verdict": "replace", "matches_narration": False,
                   "specific_enough": False, "correct_subject_visible": False})
    audit = {
        "blockers": [{"segment_index": 0,
                      "reasons": ["verdict_replace", "matches_narration_false"],
                      "quote_evidence": {"branch": "paraphrase"}}]}
    assert S.semantic_gap_candidates(proj, audit) == (
        [], "no_confirmed_actual_frame_gap_audit")

    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    assert S.semantic_gap_candidates(proj, audit) == ([0], "confirmed_actual_frame_audit")

    # Either half of the evidence changing makes the authorization stale and leaves the strict
    # publication blocker in place.
    segs[0].required_entity = "A different promised subject"
    assert S.semantic_gap_candidates(proj, audit)[1] == "stale_gap_review_beat_changed"
    segs[0].required_entity = "Jaime Lannister"
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    proj.sources[0].checksum = "pool changed"
    assert S.semantic_gap_candidates(proj, audit)[1] == "stale_gap_review_source_pool_changed"


def test_real_quote_and_technical_fault_never_buy_a_gap_downgrade(tmp_path):
    from vidlore.clipstudio import selfheal as S
    proj, _segs, _sel = _fixture(tmp_path, GOOD)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, proj.segments, [0], method="actual_frame_and_pool_audit")
    audit = {"blockers": [
        {"segment_index": 0, "reasons": ["exact_quote_dialogue_signal_below_floor"],
         "quote_evidence": {"branch": "verbatim"}},
        {"segment_index": 1, "reasons": ["verifier_evidence_mismatch", "verdict_replace"],
         "quote_evidence": {"branch": "paraphrase"}},
    ]}
    assert S.semantic_gap_candidates(proj, audit)[0] == []


def test_gap_ladder_exception_rolls_back_abstract_mutation(monkeypatch, tmp_path):
    from vidlore.clipstudio import selfheal as S
    proj, segs, sel = _fixture(
        tmp_path, {**GOOD, "verdict": "replace", "matches_narration": False,
                   "specific_enough": False, "correct_subject_visible": False})
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    audit = {"blockers": [{
        "segment_index": 0, "reasons": ["verdict_replace", "matches_narration_false"],
        "quote_evidence": {"branch": "paraphrase"}}]}
    before_seg, before_sel = copy.deepcopy(vars(segs[0])), copy.deepcopy(vars(sel))

    def explode_after_mutation(_proj, seg, _sel, *_a, **_kw):
        seg.visual_policy = "abstract_effect"
        seg.required_entity = ""
        raise RuntimeError("vision transport failed")

    monkeypatch.setattr(S, "_soften_and_retry", explode_after_mutation)
    with pytest.raises(RuntimeError, match="vision transport"):
        S.heal_selection_relevance_gaps(
            proj, segs, None, audit, policy="approved_testing", eng=SimpleNamespace())
    assert vars(segs[0]) == before_seg
    assert vars(sel) == before_sel


def test_web_image_installer_requires_and_returns_actual_pixel_verdict(tmp_path):
    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio import ocr

    seg = ScriptSegment(
        index=2, text="Watch the chalice itself.", expected_visual="A close-up of the chalice",
        required_entity="chalice", required_kind="object", scene_query="Purple Wedding chalice",
        visual_policy=P.EXACT, is_specific_claim=True)
    analysis = SimpleNamespace(movie_title="Game of Thrones", episode_hint="S04E02")
    candidate = {"image_url": "https://img/x.jpg", "source_site": "example.org",
                 "source_page": "https://example.org/got", "title": "Game of Thrones chalice"}

    def download(_url, dest):
        dest.write_bytes(b"actual-web-image-pixels")
        return dest

    bad = {**GOOD, "matches_narration": False, "specific_enough": False,
           "correct_subject_visible": False}
    calls = []

    def judge(*args, **kwargs):
        calls.append(kwargs)
        return dict(bad if len(calls) == 1 else GOOD)

    patches = (
        mock.patch.object(IF._wi, "search_images", return_value=[candidate]),
        mock.patch.object(IF._wi, "download_image", side_effect=download),
        mock.patch.object(IF._wi, "is_ai_generated_source", return_value=False),
        mock.patch.object(IF, "_web_candidate_ok", return_value=True),
        mock.patch.object(IF, "_aspect", return_value=1.7),
        mock.patch.object(IF, "_has_center_seam", return_value=False),
        mock.patch.object(IF, "_photographic_ok", return_value=True),
        mock.patch.object(IF, "_clip_relevance", return_value=0.8),
        mock.patch.object(IF, "_face_verdict", return_value="skip"),
        mock.patch.object(ocr, "available", return_value=False),
        mock.patch("vidlore.clipstudio.verify.verify_frame", side_effect=judge),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
            patches[6], patches[7], patches[8], patches[9], patches[10]:
        assert IF.fetch_scene_image(
            seg, analysis, tmp_path / "bad", eng_cfg=SimpleNamespace(anthropic_model="vision")) is None
        good = IF.fetch_scene_image(
            seg, analysis, tmp_path / "good", eng_cfg=SimpleNamespace(anthropic_model="vision"))

    assert good and good["strict_verifier"]["matches_narration"] is True
    assert good["image_sha256"]
    assert calls and all(c["is_specific"] is True and c["venue_fallback"] is False for c in calls)
