"""Scoped final recovery for the mandatory semantic publication contract."""
from __future__ import annotations

import copy
import inspect
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import index as IX
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig, load_clip_config
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo


GOOD = {
    "status": "ok", "verdict": "keep", "matches_narration": True,
    "specific_enough": True, "correct_subject_visible": True,
    "wrong_subject_visible": False, "contradicts_narration": False,
    "quality_ok": True, "target_visible": True,
}

VERIFY_OK = {"available": True, "errored": 0, "verifier_down": False}

INCONCLUSIVE_VERIFY_SUMMARIES = [
    pytest.param({}, "verifier_available_missing", id="empty-summary"),
    pytest.param({"available": True, "errored": 1, "verifier_down": False},
                 "verifier_errored:1", id="errored"),
    pytest.param({"available": False, "errored": 0, "verifier_down": False},
                 "verifier_available_false", id="unavailable"),
    pytest.param({"available": True, "errored": 0, "verifier_down": True},
                 "verifier_down", id="breaker-down"),
]


def _stamp_current_asr_provenance(proj, sid, cfg=None):
    cfg = cfg or load_clip_config()
    (proj.index_dir / f"{sid}.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA,
        "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, cfg),
    }))


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
    _stamp_current_asr_provenance(proj, "s1")
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


def _append_bound_rejected_beat(proj, segs, *, index=1):
    seg = ScriptSegment(
        index=index, text="The crown lies on the table.",
        expected_visual="the royal crown resting on the council table",
        required_entity="royal crown", required_kind="object",
        scene_query="Game of Thrones royal crown council table",
        visual_policy=P.EXACT, is_specific_claim=True)
    rejected = {
        **GOOD, "verdict": "replace", "matches_narration": False,
        "specific_enough": False, "correct_subject_visible": False,
    }
    sel = ClipSelection(
        segment_index=index, source_id="s1", shot_index=0,
        in_point=0.0, out_point=2.0, confidence=0.7, verifier=dict(rejected))
    shot = Shot.from_dict(json.loads(proj.shots_path("s1").read_text())[0])
    V.bind_selection_verifier_evidence(
        proj, sel, seg, sel.verifier, shot=shot,
        model="vision", is_specific=True, multiframe=True,
        faceid_names=[], era="", must_see="")
    segs.append(seg)
    proj.segments = list(segs)
    proj.selections.append(sel)
    return seg, sel, shot


def _install_completed_exact_downgrade(proj, seg, sel, *, contextual=True):
    """Persist the exact audit shape produced by verify's completed lenient fallback."""
    sel.verifier = {
        **GOOD,
        "downgraded": "exact→contextual" if contextual else "exact→generic_filler",
        "relevance_class": "contextual_fallback" if contextual else "generic_filler",
    }
    shot = Shot.from_dict(json.loads(proj.shots_path(sel.source_id).read_text())[0])
    V.bind_selection_verifier_evidence(
        proj, sel, seg, sel.verifier, shot=shot, model="vision",
        is_specific=False, multiframe=True, faceid_names=[], era="", must_see="")
    return sel


def _call(proj, segs):
    return O._retry_selection_relevance(
        proj, segs, ClipConfig(), SimpleNamespace(movie_title="Game of Thrones"),
        SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
        policy="approved_testing", log=lambda _m: None)


def test_retry_uses_active_custom_asr_cfg_without_futile_recovery(tmp_path):
    """A current custom-model index must not look stale under the default environment.

    The semantic retry used to omit ``cfg`` at each contract evaluation.  Quote typing then loaded
    a second, default ``ClipConfig`` and rejected a perfectly current custom Whisper index as an ASR
    fingerprint mismatch, sending a passing beat through reverify/acquisition/still recovery.
    """
    proj, segs, sel = _fixture(tmp_path, GOOD)
    seg = segs[0]
    seg.quote = "Chaos isn't a pit. Chaos is a ladder."
    sel.signals = {"dialogue": 1.0, "moment_lock": 1.0}
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.2 + i * .12, 0.3 + i * .12, word] for i, word in enumerate(words)
    ]))
    custom_cfg = ClipConfig(whisper_model="small.en", whisper_compute="float16")
    default_cfg = ClipConfig(whisper_model="base", whisper_compute="int8")
    assert IX.asr_semantic_fingerprint(proj, custom_cfg) != \
        IX.asr_semantic_fingerprint(proj, default_cfg)
    _stamp_current_asr_provenance(proj, "s1", custom_cfg)
    V.bind_selection_verifier_evidence(
        proj, sel, seg, sel.verifier, model="vision", is_specific=True,
        multiframe=True, faceid_names=[], era="", must_see="")

    with mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair") as reverify, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        audit = O._retry_selection_relevance(
            proj, segs, custom_cfg, SimpleNamespace(movie_title="Game of Thrones"),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None)

    assert audit["status"] == "pass"
    assert audit["quote_branch_counts"] == {
        "verbatim": 1, "paraphrase": 0, "indeterminate": 0}
    evidence = audit["checked"][0]["quote_evidence"]
    assert evidence["asr_prompt_fingerprint_expected"] == \
        IX.asr_semantic_fingerprint(proj, custom_cfg)
    assert evidence["asr_provenance_invalid_source_count"] == 0
    recover.assert_not_called()
    reverify.assert_not_called()
    stills.assert_not_called()
    ladder.assert_not_called()


def _write_mock_recovery_page(proj, kw, *, deferred=(), page_completed=True,
                              page_error="", current_error=""):
    """Give mocked recovery the same nonce/scope-bound receipt required in production."""
    requested = sorted(kw["only_indices"])
    deferred = sorted(deferred)
    page_scope = [i for i in requested if i not in set(deferred)]
    audit = {
        "schema_version": O._SEMANTIC_RECOVERY_PAGE_SCHEMA,
        "request_id": kw["audit_request_id"],
        "requested_scope": requested,
        "page_scope": page_scope,
        "page_completed": page_completed,
        "page_error": page_error,
        "deferred": deferred,
        "deferred_retriable": deferred,
        "current_pool_rematch": {
            "attempted": page_scope, "recovered": [], "still_unresolved": requested,
            "deferred": deferred, "error": current_error,
        },
    }
    (proj.output_dir / kw["audit_filename"]).write_text(json.dumps(audit))


def _quote_cache_fixture(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    segs[0].quote = "Who does this belong to?"
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.0, 0.2, "winter"], [0.2, 0.4, "is"], [0.4, 0.6, "coming"],
    ]), encoding="utf-8")
    _stamp_current_asr_provenance(proj, "s1")
    return proj, segs


def test_retry_reuses_quote_pool_classification_while_generation_is_stable(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    proj, segs = _quote_cache_fixture(tmp_path)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "0",
           "VIDLORE_CLIPSTUDIO_SELFHEAL": "0"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted), \
            mock.patch.object(R, "_quote_pool_branches",
                              wraps=R._quote_pool_branches) as branch_builder, \
            mock.patch.object(R, "evaluate_selection_relevance",
                              wraps=R.evaluate_selection_relevance) as evaluations:
        audit = _call(proj, segs)

    assert audit["status"] == "blocked"
    assert evaluations.call_count >= 4, "the recovery path must exercise repeated strict audits"
    assert branch_builder.call_count == 1, (
        "selection/verifier-only audits must reuse the request-local whole-pool classification")


def test_quote_pool_cache_cannot_be_replaced_with_caller_authored_branches(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    proj, segs = _quote_cache_fixture(tmp_path)
    injected = {0: {"branch": "paraphrase", "verbatim_required": False}}

    with pytest.raises(TypeError, match="request-local classification cache"):
        R.evaluate_selection_relevance(
            proj, segs, cfg=ClipConfig(), quote_pool_cache=injected)

    class ForgedCache(R._RequestQuotePoolClassificationCache):
        def contracts_for(self, *_args, **_kwargs):
            return injected

    with pytest.raises(TypeError, match="request-local classification cache"):
        R.evaluate_selection_relevance(
            proj, segs, cfg=ClipConfig(), quote_pool_cache=ForgedCache())


def test_retry_invalidates_quote_pool_classification_when_index_generation_changes(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    proj, segs = _quote_cache_fixture(tmp_path)

    def grow_index(_proj, _segs, _analysis, _cfg, _eng, **kw):
        words = _proj.index_dir / "s1.words.json"
        words.write_text(words.read_text(encoding="utf-8") + " ", encoding="utf-8")
        _write_mock_recovery_page(_proj, kw)
        return 0

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "0",
           "VIDLORE_CLIPSTUDIO_SELFHEAL": "0"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=grow_index), \
            mock.patch.object(R, "_quote_pool_branches",
                              wraps=R._quote_pool_branches) as branch_builder:
        _call(proj, segs)

    assert branch_builder.call_count == 2, (
        "the post-recovery audit must rebuild branches exactly once for the new indexed pool")


def test_retry_invalidates_quote_pool_classification_after_specificity_softening(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S
    proj, segs = _quote_cache_fixture(tmp_path)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    def soften(_proj, _segs, *_args, **_kwargs):
        _segs[0].visual_policy = P.ABSTRACT
        return {"candidate_count": 1, "softened_count": 1}

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "0",
           "VIDLORE_CLIPSTUDIO_SELFHEAL": "1",
           "VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN": "1"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted), \
            mock.patch.object(S, "heal_selection_relevance_gaps", side_effect=soften), \
            mock.patch.object(R, "_quote_pool_branches",
                              wraps=R._quote_pool_branches) as branch_builder:
        audit = _call(proj, segs)

    assert audit["status"] == "pass"
    assert branch_builder.call_count == 2, (
        "the final audit must not reuse a pre-softening quote-policy classification")


def test_missing_evidence_gets_scoped_reverify_before_acquisition(tmp_path):
    proj, segs, sel = _fixture(tmp_path, {"status": "ok", "verdict": "keep"})

    def reverify(_proj, _segs, _cfg, _eng, *, only_indices, progress):
        assert only_indices == {0}
        sel.verifier = dict(GOOD)
        V.bind_selection_verifier_evidence(
            proj, sel, segs[0], sel.verifier, model="vision", is_specific=True,
            multiframe=True, faceid_names=[], era="", must_see="")
        return dict(VERIFY_OK)

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair", side_effect=reverify) as rv, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills:
        audit = _call(proj, segs)
    assert audit["status"] == "pass"
    rv.assert_called_once()
    recover.assert_not_called()
    stills.assert_not_called()


@pytest.mark.parametrize("reason", [
    "exact_verifier_evidence_not_strict",
    "exact_moving_verdict_was_downgraded",
    "exact_moving_relevance_is_contextual",
])
def test_exact_non_strict_provenance_gets_scoped_reverify(reason):
    audit = {"blockers": [{"segment_index": 59, "reasons": [reason]}]}
    assert O._unverifiable_relevance_indices(audit) == {59}


def test_native_still_rejection_forces_fresh_moving_reverify():
    audit = {"blockers": [{
        "segment_index": 36,
        "reasons": ["verifier_stale_native_still_conflict"],
    }]}
    assert O._unverifiable_relevance_indices(audit) == {36}


def test_invalid_still_recovery_is_not_suppressed_by_moving_technical_fault():
    audit = {"blockers": [{
        "segment_index": 56,
        "reasons": [
            "invalid_still:owned source-frame still is 512x288",
            "verifier_evidence_mismatch",
        ],
    }]}
    assert O._invalid_still_recovery_indices(audit) == {56}


@pytest.mark.parametrize("semantic", [True, False])
@pytest.mark.parametrize("reason", [
    "invalid_still:owned source-frame still is 512x288; native materialization required",
    "invalid_still:owned native semantic binding is stale: beat-question changed",
    "invalid_still:owned source-frame still indexed provenance is invalid",
])
def test_every_owned_invalid_still_uses_native_refresh_lane(semantic, reason):
    meta = {
        "source": "source-frame-recovery", "src": "owner", "shot": 7,
        "still_semantic_verified": semantic,
    }
    assert O._owned_native_still_recovery_candidate(meta, [reason]) is True


def test_exact_non_strict_provenance_with_semantic_negative_still_skips_reverify():
    audit = {"blockers": [{
        "segment_index": 59,
        "reasons": ["exact_verifier_evidence_not_strict", "matches_narration_false"],
    }]}
    assert O._unverifiable_relevance_indices(audit) == set()


@pytest.mark.parametrize("contextual", [True, False], ids=["contextual", "generic"])
def test_completed_bound_exact_downgrade_is_content_and_skips_redundant_reverify(
        tmp_path, contextual):
    from vidlore.clipstudio import relevance_contract as R

    proj, segs, sel = _fixture(tmp_path, GOOD)
    _install_completed_exact_downgrade(proj, segs[0], sel, contextual=contextual)
    audit = R.evaluate_selection_relevance(proj, segs)
    entry = audit["blockers"][0]

    assert R.completed_deliberate_exact_downgrade(entry) is True
    assert O._unverifiable_relevance_indices(audit) == set()
    assert O._persistent_verifier_technical_indices(audit) == set()


def test_completed_downgrade_mixed_with_real_binding_fault_stays_technical(tmp_path):
    from vidlore.clipstudio import relevance_contract as R

    proj, segs, sel = _fixture(tmp_path, GOOD)
    _install_completed_exact_downgrade(proj, segs[0], sel)
    audit = R.evaluate_selection_relevance(proj, segs)
    entry = audit["blockers"][0]
    entry["reasons"].append("verifier_evidence_mismatch")

    assert R.completed_deliberate_exact_downgrade(entry) is False
    assert O._unverifiable_relevance_indices(audit) == {0}
    assert O._persistent_verifier_technical_indices(audit) == {0}


def test_completed_downgrade_plus_real_quote_floor_remains_content_recoverable(tmp_path):
    """A selected-window quote miss is content, even beside the deliberate downgrade shape."""
    from vidlore.clipstudio import relevance_contract as R

    proj, segs, sel = _fixture(tmp_path, GOOD)
    _install_completed_exact_downgrade(proj, segs[0], sel)
    audit = R.evaluate_selection_relevance(proj, segs)
    entry = audit["blockers"][0]
    entry["reasons"].append("exact_quote_dialogue_signal_below_floor")
    entry["quote_evidence"] = {"branch": "verbatim"}

    assert R.completed_deliberate_exact_downgrade(entry) is True
    assert O._unverifiable_relevance_indices(audit) == set()
    assert O._persistent_verifier_technical_indices(audit) == set()


def test_retry_routes_completed_downgrade_to_content_recovery_without_reverify(tmp_path):
    proj, segs, sel = _fixture(tmp_path, GOOD)
    _install_completed_exact_downgrade(proj, segs[0], sel)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        assert kw["only_indices"] == {0}
        _write_mock_recovery_page(_proj, kw)
        return 0

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "0",
           "VIDLORE_CLIPSTUDIO_SELFHEAL": "0"}
    with mock.patch.dict(os.environ, env), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair") as reverify, \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover:
        audit = _call(proj, segs)

    assert audit["status"] == "blocked"
    reverify.assert_not_called()
    recover.assert_called_once()
    assert proj.meta["selection_relevance_recovery"]["technical_blockers"] == []


def test_absent_moving_source_does_not_enter_impossible_scoped_reverify():
    audit = {"blockers": [{
        "segment_index": 59,
        "reasons": ["moving_source_absent", "verifier_evidence_unrecomputable"],
    }]}
    assert O._unverifiable_relevance_indices(audit) == set()


def test_persistent_technical_partition_keeps_missing_footage_in_content_lane():
    audit = {"blockers": [
        {
            "segment_index": 11,
            "reasons": ["verifier_evidence_mismatch", "matches_narration_false"],
        },
        {
            "segment_index": 12,
            "reasons": ["moving_source_absent", "verifier_evidence_unrecomputable"],
        },
    ]}
    assert O._persistent_verifier_technical_indices(audit) == {11}


@pytest.mark.parametrize("summary,expected_reason", INCONCLUSIVE_VERIFY_SUMMARIES)
def test_missing_evidence_inconclusive_summary_is_retryable_before_recovery(
        tmp_path, summary, expected_reason):
    proj, segs, _sel = _fixture(tmp_path, {"status": "ok", "verdict": "keep"})

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                    return_value=summary) as verifier, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match=expected_reason):
            _call(proj, segs)

    verifier.assert_called_once()
    recover.assert_not_called()
    stills.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta


def test_success_summary_with_fresh_malformed_keep_is_retryable_before_recovery(tmp_path):
    """A successful batch count is not proof that its positive JSON was publication-complete."""
    proj, segs, sel = _fixture(tmp_path, {"status": "ok", "verdict": "keep"})

    def malformed_reverify(_proj, _segs, _cfg, _eng, *, only_indices, progress):
        assert only_indices == {0}
        malformed = dict(GOOD)
        malformed["matches_naration"] = malformed.pop("matches_narration")
        sel.verifier = malformed
        V.bind_selection_verifier_evidence(
            proj, sel, segs[0], sel.verifier, model="vision", is_specific=True,
            multiframe=True, faceid_names=[], era="", must_see="")
        return dict(VERIFY_OK)

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                    side_effect=malformed_reverify) as verifier, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as stills, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match="remained technically inconclusive"):
            _call(proj, segs)

    verifier.assert_called_once()
    recover.assert_not_called()
    stills.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta


def test_isolated_error_summary_does_not_override_a_clear_final_audit(tmp_path):
    """The final current facts, not a historical batch count, decide technical failure."""
    proj, segs, sel = _fixture(tmp_path, {"status": "ok", "verdict": "keep"})

    def repair_then_report_isolated_error(
            _proj, _segs, _cfg, _eng, *, only_indices, progress):
        assert only_indices == {0}
        sel.verifier = dict(GOOD)
        V.bind_selection_verifier_evidence(
            proj, sel, segs[0], sel.verifier, model="vision", is_specific=True,
            multiframe=True, faceid_names=[], era="", must_see="")
        return {"available": True, "errored": 1, "verifier_down": False}

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                    side_effect=repair_then_report_isolated_error), \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        audit = _call(proj, segs)

    assert audit["status"] == "pass"
    recover.assert_not_called()
    images.assert_not_called()
    ladder.assert_not_called()


def test_global_verifier_outage_aborts_before_independent_content_recovery(tmp_path):
    """Partitioning is for isolated beat errors, never permission to spend against a dead backend."""
    proj, segs, _technical_sel = _fixture(
        tmp_path, {"status": "ok", "verdict": "keep"})
    _append_bound_rejected_beat(proj, segs)
    outage = {"available": True, "errored": 0, "verifier_down": True}

    with mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                    return_value=outage) as verifier, \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match="globally inconclusive: verifier_down"):
            _call(proj, segs)

    assert verifier.call_args.kwargs["only_indices"] == {0}
    recover.assert_not_called()
    images.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta


def test_technical_beat_fails_closed_after_independent_content_recovery_is_saved(tmp_path):
    """One verifier fault must not abort a conclusive recovery for another beat."""
    proj, segs, _technical_sel = _fixture(
        tmp_path, {"status": "ok", "verdict": "keep"})
    content_seg, content_sel, shot = _append_bound_rejected_beat(proj, segs)

    def recover_content(_proj, _segs, _analysis, _cfg, _eng, **kw):
        assert kw["only_indices"] == {1}, (
            "the technically inconclusive beat must never enter content recovery")
        _write_mock_recovery_page(_proj, kw)
        content_sel.verifier = dict(GOOD)
        V.bind_selection_verifier_evidence(
            proj, content_sel, content_seg, content_sel.verifier, shot=shot,
            model="vision", is_specific=True, multiframe=True,
            faceid_names=[], era="", must_see="")
        return 1

    inconclusive = {"available": True, "errored": 1, "verifier_down": False}
    env = {
        "VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "0",
        "VIDLORE_CLIPSTUDIO_SELFHEAL": "1",
        "VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN": "1",
    }
    with mock.patch.dict(os.environ, env), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=inconclusive) as verifier, \
            mock.patch.object(O, "_recover_unresolved_beats",
                              side_effect=recover_content) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        for _attempt in range(2):
            with pytest.raises(
                    O.PipelineError, match=r"beat\(s\) \[0\].*verifier_errored:1"):
                _call(proj, segs)

    assert verifier.call_count == 2, (
        "the content exhaustion marker must never suppress a fresh technical retry")
    assert verifier.call_args.kwargs["only_indices"] == {0}
    recover.assert_called_once(), "the already repaired content beat must remain bounded"
    images.assert_not_called()
    ladder.assert_not_called()
    assert content_sel.verifier["matches_narration"] is True, (
        "the independent conclusive content repair must survive the final technical failure")
    final_audit = json.loads(
        (proj.output_dir / "selection_relevance_audit.json").read_text())
    assert [entry["segment_index"] for entry in final_audit["blockers"]] == [0]
    marker = proj.meta["selection_relevance_recovery"]
    assert marker["before"] == [1]
    assert marker["after"] == []
    assert [entry["segment_index"] for entry in marker["technical_blockers"]] == [0]


def test_explicit_negative_skips_duplicate_reverify_and_recovers_only_blocker(tmp_path):
    proj, segs, sel = _fixture(
        tmp_path, {**GOOD, "verdict": "replace", "matches_narration": False,
                   "specific_enough": False, "correct_subject_visible": False})

    def recover(_proj, _segs, _analysis, _cfg, _eng, **kw):
        assert kw["only_indices"] == {0}
        assert kw["audit_filename"] == "semantic_recovery_audit.json"
        _write_mock_recovery_page(_proj, kw)
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


@pytest.mark.parametrize("failure", [
    "exception", "none", "partial_keep", "malformed_keep", "missing_subject", "error_keep",
])
def test_legacy_still_inconclusive_verifier_is_retryable_before_any_fallback(tmp_path, failure):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    still = tmp_path / "legacy-unverified.jpg"
    still.write_bytes(b"legacy actual bytes")
    sel.image_path = str(still)
    sel.image_meta = {"source": "web-exact-scene", "relevance_class": "exact_scene"}
    before_meta = copy.deepcopy(sel.image_meta)

    def inconclusive(*_args, **_kwargs):
        if failure == "exception":
            raise RuntimeError("still verifier transport failed")
        if failure == "none":
            return None
        if failure == "partial_keep":
            return {"status": "ok", "verdict": "keep"}
        if failure == "malformed_keep":
            return {**GOOD, "matches_narration": "yes"}
        if failure == "missing_subject":
            out = dict(GOOD)
            out.pop("correct_subject_visible")
            return out
        return {**GOOD, "status": "error", "verdict": "keep"}

    with mock.patch("vidlore.clipstudio.verify.verify_frame", side_effect=inconclusive), \
            mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as fallback, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match="legacy-still verifier"):
            _call(proj, segs)

    recover.assert_not_called()
    fallback.assert_not_called()
    ladder.assert_not_called()
    assert sel.image_meta == before_meta
    assert "selection_relevance_recovery" not in proj.meta


def test_failed_retry_stays_blocked_and_unchanged_resume_does_not_repeat(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks", return_value=0) as stills:
        first = _call(proj, segs)
        second = _call(proj, segs)
    assert first["status"] == second["status"] == "blocked"
    assert recover.call_count == 1
    assert stills.call_count == 1
    marker = proj.meta["selection_relevance_recovery"]
    assert marker["before"] == marker["after"] == [0]
    assert marker["post_fingerprint"]


@pytest.mark.parametrize("index_kind", ["shots", "words"])
def test_index_only_pool_growth_reopens_exhausted_semantic_generation(tmp_path, index_kind):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    media = proj.sources[0].local_path
    media_stat = (os.stat(media).st_size, os.stat(media).st_mtime_ns)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks", return_value=0):
        _call(proj, segs)
        _call(proj, segs)
        assert recover.call_count == 1, "unchanged exhausted generation is finite"
        old_pool = O._semantic_recovery_pool_fingerprint(proj)

        if index_kind == "words":
            (proj.index_dir / "s1.words.json").write_text(json.dumps([
                {"word": "fresh", "start": 0.0, "end": 0.4},
            ]), encoding="utf-8")
        else:
            shots_path = proj.shots_path("s1")
            shots = json.loads(shots_path.read_text(encoding="utf-8"))
            shots.append(Shot(
                source_id="s1", index=99, start=3.0, end=5.0,
                keyframe_path=str(tmp_path / "fresh-index-frame.jpg")).to_dict())
            shots_path.write_text(json.dumps(shots), encoding="utf-8")

        assert O._semantic_recovery_pool_fingerprint(proj) != old_pool
        assert (os.stat(media).st_size, os.stat(media).st_mtime_ns) == media_stat
        _call(proj, segs)
        _call(proj, segs)

    assert recover.call_count == 2, "index-only growth earns exactly one fresh strict generation"


@pytest.mark.parametrize("failure", [
    "helper_exception", "missing_audit", "corrupt_audit", "mismatched_audit",
    "incomplete_page", "current_pool_error",
])
def test_inconclusive_recovery_page_never_authorizes_fallback_or_exhaustion(tmp_path, failure):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)

    def inconclusive(_proj, _segs, _analysis, _cfg, _eng, **kw):
        audit_path = _proj.output_dir / kw["audit_filename"]
        if failure == "helper_exception":
            raise RuntimeError("recovery helper crashed")
        if failure == "missing_audit":
            return 0
        if failure == "corrupt_audit":
            audit_path.write_text("{not-json", encoding="utf-8")
            return 0
        if failure == "mismatched_audit":
            wrong = dict(kw)
            wrong["audit_request_id"] = "stale-request"
            _write_mock_recovery_page(_proj, wrong)
            return 0
        if failure == "incomplete_page":
            _write_mock_recovery_page(_proj, kw, page_completed=False)
            return 0
        _write_mock_recovery_page(
            _proj, kw, page_completed=True, current_error="vision rematch timed out")
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=inconclusive) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError):
            _call(proj, segs)

    recover.assert_called_once()
    images.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta
    assert segs[0].visual_policy == P.EXACT


@pytest.mark.parametrize("failing_helper", ["match", "verify"])
def test_real_recovery_helper_error_is_retryable_and_never_softens(tmp_path, failing_helper):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)

    def rematch(*_args, **_kwargs):
        if failing_helper == "match":
            raise RuntimeError("matcher temporarily unavailable")
        return proj.selections

    def reverify(*_args, **_kwargs):
        if failing_helper == "verify":
            raise RuntimeError("verifier temporarily unavailable")
        return {}

    with mock.patch.object(O, "match_segments", side_effect=rematch), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair", side_effect=reverify), \
            mock.patch("vidlore.clipstudio.discover.discover_sources", return_value=[]), \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError):
            _call(proj, segs)

    page = json.loads(
        (proj.output_dir / "semantic_recovery_audit.json").read_text(encoding="utf-8"))
    assert page["page_completed"] is False
    expected_word = "matcher" if failing_helper == "match" else "verifier"
    assert expected_word in page["current_pool_rematch"]["error"].lower()
    images.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta


def test_exact_image_fallback_exception_is_retryable_before_ladder_or_marker(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks",
                              side_effect=RuntimeError("image verifier transport failed")) as images, \
            mock.patch("vidlore.clipstudio.selfheal.heal_selection_relevance_gaps") as ladder:
        for _attempt in range(2):
            with pytest.raises(O.PipelineError, match="exact-image fallback failed"):
                _call(proj, segs)
            assert "selection_relevance_recovery" not in proj.meta

    assert recover.call_count == 2, "no exhaustion marker means Resume retries strict recovery"
    assert images.call_count == 2
    ladder.assert_not_called()
    assert segs[0].visual_policy == P.EXACT


def test_web_image_technical_nonverdict_stops_real_strict_fallback_before_marker(
        tmp_path):
    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio import ocr
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    # Model the strict footage-gap shape: no moving clip and no local still, so the real web pass
    # is the final exact-preserving rung.
    sel.source_id = ""
    sel.shot_index = -1
    sel.clip_path = ""
    before_seg, before_sel = copy.deepcopy(vars(segs[0])), copy.deepcopy(vars(sel))

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    candidate = {
        "image_url": "https://img/exact.jpg", "source_site": "example.org",
        "source_page": "https://example.org/game-of-thrones",
        "title": "Game of Thrones exact necklace scene",
    }

    def download(_url, dest):
        dest.write_bytes(b"candidate pixels awaiting verifier")
        return dest

    analysis = SimpleNamespace(
        movie_title="Game of Thrones", episode_hint="",
        char_to_actor=lambda: {})
    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1",
           "VIDLORE_CLIPSTUDIO_WEB_IMAGE_FALLBACK": "1"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch.object(IF, "pick_source_still", return_value=None), \
            mock.patch.object(IF, "pick_pool_still", return_value=None), \
            mock.patch.object(IF._wi, "search_images", return_value=[candidate]), \
            mock.patch.object(IF._wi, "download_image", side_effect=download), \
            mock.patch.object(IF._wi, "is_ai_generated_source", return_value=False), \
            mock.patch.object(IF, "_web_candidate_ok", return_value=True), \
            mock.patch.object(IF, "_aspect", return_value=1.7), \
            mock.patch.object(IF, "_has_center_seam", return_value=False), \
            mock.patch.object(IF, "_photographic_ok", return_value=True), \
            mock.patch.object(IF, "_clip_relevance", return_value=0.8), \
            mock.patch.object(IF, "_face_verdict", return_value="skip"), \
            mock.patch.object(ocr, "available", return_value=False), \
            mock.patch("vidlore.clipstudio.verify.verify_frame", return_value=None), \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match="exact-image fallback failed.*"
                                                    "SceneImageTechnicalError"):
            O._retry_selection_relevance(
                proj, segs, ClipConfig(), analysis,
                SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
                policy="approved_testing", log=lambda _m: None)

    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta
    assert "exact_scene_missing" not in sel.flag_reasons
    assert vars(segs[0]) == before_seg
    assert vars(sel) == before_sel


def test_all_web_search_providers_technical_raise_without_caching_empty_result(tmp_path):
    from vidlore.clipstudio import image_fallback as IF

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path / "project", bad)
    sel.source_id = ""
    sel.shot_index = -1
    sel.clip_path = ""
    before_seg, before_sel = copy.deepcopy(vars(segs[0])), copy.deepcopy(vars(sel))
    analysis = SimpleNamespace(
        movie_title="Game of Thrones", episode_hint="",
        char_to_actor=lambda: {})
    cache_dir = tmp_path / "web-cache"
    env = {
        "VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1",
        "VIDLORE_CLIPSTUDIO_WEB_IMAGE_FALLBACK": "1",
        "VIDLORE_CLIPSTUDIO_IMG_BING": "1",
        "VIDLORE_CLIPSTUDIO_IMG_DDG": "1",
        "VIDLORE_CLIPSTUDIO_IMG_WIKI": "1",
        "VIDLORE_CLIPSTUDIO_IMG_CACHE": str(cache_dir),
    }
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(IF, "pick_source_still", return_value=None), \
            mock.patch.object(IF, "pick_pool_still", return_value=None), \
            mock.patch.object(IF._wi, "_guarded_get",
                              side_effect=TimeoutError("all image providers timed out")):
        with pytest.raises(IF.SceneImageTechnicalError, match="image search failed"):
            O._fill_image_fallbacks(
                proj, segs, analysis, None, {}, lambda _m: None,
                eng_cfg=SimpleNamespace(anthropic_model="vision"),
                fail_on_web_technical=True)

    assert not list((cache_dir / "search").glob("*.json")), (
        "an all-provider outage must not create a long-lived empty search cache")
    assert "selection_relevance_recovery" not in proj.meta
    assert vars(segs[0]) == before_seg
    assert vars(sel) == before_sel


def test_strict_still_all_unverified_retries_then_mixed_reject_allows_ladder_marker(tmp_path):
    from vidlore.clipstudio import image_fallback as IF
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    sel.flag_reasons = ["verifier_failed"]
    sel.flagged = True
    first_shot = json.loads(proj.shots_path("s1").read_text(encoding="utf-8"))[0]
    second_frame = tmp_path / "shot_0001.jpg"
    second_frame.write_bytes(b"second strict still candidate")
    second_shot = Shot(source_id="s1", index=1, start=3.0, end=5.0,
                       keyframe_path=str(second_frame))
    proj.shots_path("s1").write_text(
        json.dumps([first_shot, second_shot.to_dict()]), encoding="utf-8")
    candidates = [
        (first_shot["keyframe_path"], "s1", 0, 0.92, ""),
        (str(second_frame), "s1", 1, 0.88, ""),
    ]
    outage = {"active": True}

    def pick_candidate(_seg, _pool, used_keys, *_args, **_kwargs):
        return next((c for c in candidates if (c[1], c[2]) not in used_keys), None)

    def judge(path, *_args, **_kwargs):
        # First Resume: every candidate is a transport non-verdict. Second Resume: candidate 0
        # remains unverified but candidate 1 supplies an explicit semantic reject, making the mixed
        # batch a conclusive content result rather than an outage.
        if outage["active"] or str(path) == first_shot["keyframe_path"]:
            return None
        return dict(bad)

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    analysis = SimpleNamespace(movie_title="Game of Thrones", char_to_actor=lambda: {})

    def retry():
        return O._retry_selection_relevance(
            proj, segs, ClipConfig(), analysis, SimpleNamespace(anthropic_model="vision"),
            faceid_obj=None, refs={}, roster=[], policy="approved_testing", log=lambda _m: None)

    env = {
        "VIDLORE_CLIPSTUDIO_WEB_IMAGE_FALLBACK": "0",
        "VIDLORE_CLIPSTUDIO_VISION_RETRY_SEC": "0",
        "VIDLORE_CLIPSTUDIO_STILL_CANDIDATES": "2",
    }
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover, \
            mock.patch("vidlore.clipstudio.llm.has_llm", return_value=True), \
            mock.patch.object(IF, "pick_source_still", return_value=None), \
            mock.patch.object(IF, "pick_pool_still", side_effect=pick_candidate), \
            mock.patch("vidlore.clipstudio.verify.verify_frame", side_effect=judge), \
            mock.patch.object(S, "heal_selection_relevance_gaps",
                              return_value={"candidate_count": 1, "softened_count": 0}) as ladder:
        with pytest.raises(O.PipelineError, match="technically inconclusive"):
            retry()
        assert ladder.call_count == 0
        assert "selection_relevance_recovery" not in proj.meta
        assert "exact_scene_missing" not in sel.flag_reasons

        outage["active"] = False
        final = retry()

    assert final["status"] == "blocked"
    assert recover.call_count == 2, "the technical still page is retried on Resume"
    ladder.assert_called_once()
    assert proj.meta["selection_relevance_recovery"]["deferred"] == []
    assert segs[0].visual_policy == P.EXACT


def test_gap_ladder_exception_rolls_back_and_resume_retries_without_exhaustion_marker(tmp_path):
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    before_seg = copy.deepcopy(vars(segs[0]))
    before_sel = copy.deepcopy(vars(sel))

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    def mutate_then_fail(_proj, seg, selection, *_args, **_kwargs):
        seg.visual_policy = P.ABSTRACT
        seg.required_entity = ""
        selection.image_meta = {"partial_ladder_mutation": True}
        raise RuntimeError("ladder verifier transport failed")

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted) as recover, \
            mock.patch.object(O, "_fill_image_fallbacks", return_value=0), \
            mock.patch.object(S, "_soften_and_retry", side_effect=mutate_then_fail) as soften:
        for _attempt in range(2):
            with pytest.raises(O.PipelineError, match="specificity ladder failed"):
                _call(proj, segs)
            assert vars(segs[0]) == before_seg
            assert vars(sel) == before_sel
            assert "selection_relevance_recovery" not in proj.meta
            assert "selection_relevance_gap_softening" not in proj.meta

    assert recover.call_count == 2, "the failed page remains eligible on the next Resume"
    assert soften.call_count == 2
    assert not (proj.output_dir / "semantic_gap_softening_audit.json").exists()
    persisted = ClipProject.load(tmp_path)
    assert persisted.segments[0].visual_policy == P.EXACT
    assert persisted.selections[0].source_id == "s1"


def test_gap_ladder_scene_lineage_corruption_remains_nonretryable(tmp_path):
    """A broken owner/hash invariant must not be relabelled as a retryable verifier outage."""
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")

    def exhausted(_proj, _segs, _analysis, _cfg, _eng, **kw):
        _write_mock_recovery_page(_proj, kw)
        return 0

    deterministic = V.NonRetryableBuildError(
        "indexed keyframe hash no longer matches", kind="scene_lineage")
    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=exhausted), \
            mock.patch.object(O, "_fill_image_fallbacks", return_value=0), \
            mock.patch.object(
                S, "heal_selection_relevance_gaps", side_effect=deterministic):
        with pytest.raises(V.NonRetryableBuildError, match="keyframe hash") as caught:
            _call(proj, segs)

    assert caught.value.kind == "scene_lineage"
    assert "selection_relevance_recovery" not in proj.meta
    assert "selection_relevance_gap_softening" not in proj.meta


def test_preflight_recovery_router_preserves_typed_lineage_and_unknown_fallback():
    original = V.NonRetryableBuildError(
        "selection relevance still blocked", kind="selection_relevance")
    lineage = V.NonRetryableBuildError(
        "indexed keyframe hash no longer matches", kind="scene_lineage")

    with pytest.raises(V.NonRetryableBuildError, match="keyframe hash") as caught:
        O._raise_semantic_recovery_failure(original, lineage, lambda _message: None)
    assert caught.value is lineage
    assert caught.value.kind == "scene_lineage"

    messages = []
    with pytest.raises(V.NonRetryableBuildError, match="selection relevance") as restored:
        O._raise_semantic_recovery_failure(
            original, RuntimeError("untyped repair plumbing fault"), messages.append)
    assert restored.value is original
    assert messages and "technical failure" in messages[0]


def test_pool_bound_softening_stays_for_same_pool_then_restores_and_blocks_on_new_pool(
        tmp_path):
    """A footage-gap downgrade is valid for one indexed pool, never an authored-script rewrite."""
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    seg = segs[0]
    sel.image_meta = {"pre_softening": {"reviewed": True}}
    original_seg = copy.deepcopy(vars(seg))
    original_image_path = sel.image_path
    original_image_meta = copy.deepcopy(sel.image_meta)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    audit = R.evaluate_selection_relevance(proj, segs)
    assert audit["blocked_count"] == 1

    softened_frame = tmp_path / "softened-abstract.jpg"
    softened_frame.write_bytes(b"softened-frame")

    def successful_softening(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        target.required_entity = ""
        target.required_kind = ""
        target.quote = ""
        target.scene_query = ""
        target.expected_visual = "Neutral atmosphere"
        target.is_specific_claim = False
        selection.image_path = str(softened_frame)
        selection.image_meta = {"selfheal_rung": "abstract", "installed": True}
        return True

    with mock.patch.object(S, "_soften_and_retry", side_effect=successful_softening):
        payload = S.heal_selection_relevance_gaps(
            proj, segs, ClipConfig(), audit, policy="approved_testing",
            eng=SimpleNamespace(anthropic_model="vision"), log=lambda _m: None)

    assert payload["schema_version"] == S._SOFTENING_SCHEMA
    row = next(r for r in payload["beats"] if r["status"] == "softened")
    assert row["pool_fingerprint"] == S._gap_pool_fingerprint(proj)[0]
    assert row["original"]["expected_visual"] == original_seg["expected_visual"]
    assert row["original"]["is_specific_claim"] is True
    assert row["original"]["image_path"] == original_image_path
    assert row["original"]["image_meta"] == original_image_meta

    # No source/index change: the reviewed fallback is still valid and the assertion remains a pass.
    same_pool = R.assert_selection_relevance(proj, segs)
    assert same_pool["blocked_count"] == 0
    assert seg.visual_policy == P.ABSTRACT
    assert proj.meta["selection_relevance_gap_softening"]["active"] is True

    # A new searchable source invalidates the old footage-gap premise. The assertion must restore
    # every authored/selection field first, then expose the original exact beat to strict recovery.
    late_media = tmp_path / "late_source.mp4"
    late_media.write_bytes(b"new-footage")
    late_frame = tmp_path / "late_shot.jpg"
    late_frame.write_bytes(b"late-frame")
    proj.sources.append(SourceVideo(
        id="late", url="late", title="new exact-scene candidate", permission="owner",
        status="ok", local_path=str(late_media)))
    proj.shots_path("late").write_text(json.dumps([Shot(
        source_id="late", index=0, start=0.0, end=2.0,
        keyframe_path=str(late_frame)).to_dict()]))
    (proj.index_dir / "late.words.json").write_text("[]")

    with pytest.raises(V.NonRetryableBuildError) as exc:
        R.assert_selection_relevance(proj, segs)

    assert exc.value.kind == "selection_relevance"
    for field in ("visual_policy", "required_entity", "required_kind", "quote",
                  "scene_query", "expected_visual", "is_specific_claim"):
        assert getattr(seg, field) == original_seg[field]
    assert sel.image_path == original_image_path
    assert sel.image_meta == original_image_meta
    restored_marker = proj.meta["selection_relevance_gap_softening"]
    assert restored_marker["active"] is False
    assert restored_marker["restored_count"] == 1
    assert restored_marker["beats"][0]["status"] == "restored_pool_changed"
    persisted = ClipProject.load(tmp_path)
    assert persisted.segments[0].visual_policy == P.EXACT
    assert persisted.selections[0].image_meta == original_image_meta


def test_banning_softened_source_invalidates_pool_binding_and_restores_exact_beat(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    seg = segs[0]
    original = S._capture_softening_state(seg, sel)
    pool_fp, pool_n = S._gap_pool_fingerprint(proj)
    seg.visual_policy = P.ABSTRACT
    seg.required_entity = ""
    seg.required_kind = ""
    seg.scene_query = ""
    seg.expected_visual = "Neutral atmosphere"
    seg.is_specific_claim = False
    sel.image_path = str(tmp_path / "shot_0000.jpg")
    sel.image_meta = {"selfheal_rung": "abstract", "installed": True}
    S._record_phase1_softening(
        proj, seg, sel, original, basis="test_bound_softening",
        pool_fingerprint=pool_fp, pool_source_count=pool_n)

    # No bytes or timestamps change. Eligibility alone must change the searchable-pool identity.
    proj.meta["banned_sources"] = ["s1"]
    assert S._gap_pool_fingerprint(proj)[0] != pool_fp
    with pytest.raises(V.NonRetryableBuildError) as exc:
        R.assert_selection_relevance(proj, segs)

    assert exc.value.kind == "selection_relevance"
    assert seg.visual_policy == P.EXACT
    assert sel.image_path == original["image_path"]
    assert sel.image_meta == original["image_meta"]
    assert proj.meta["selection_relevance_gap_softening"]["active"] is False


def test_unchanged_resume_rotates_real_capped_recovery_once_then_stops(tmp_path):
    """Exercise the actual capped recovery audit and persisted semantic marker.

    With no changing footage, three blockers under cap=2 have exactly two useful pages. The third
    Resume must return to the strict gate without replaying page one. Adding source bytes changes the
    generation fingerprint and is allowed to start a fresh finite rotation.
    """
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    frame = tmp_path / "shot_0000.jpg"
    shot = Shot(source_id="s1", index=0, start=0.0, end=2.0,
                keyframe_path=str(frame))
    for idx in (1, 2):
        seg = ScriptSegment(
            index=idx, text=f"Blocked exact narration {idx}",
            expected_visual=f"Exact missing scene {idx}",
            required_entity=f"target {idx}", required_kind="object",
            scene_query=f"Game of Thrones exact missing scene {idx}",
            visual_policy=P.EXACT, is_specific_claim=True)
        sel = ClipSelection(
            segment_index=idx, source_id="s1", shot_index=0,
            in_point=0.0, out_point=2.0, confidence=0.8, verifier=dict(bad))
        V.bind_selection_verifier_evidence(
            proj, sel, seg, sel.verifier, shot=shot, model="vision",
            is_specific=True, multiframe=True, faceid_names=[], era="", must_see="")
        proj.segments.append(seg)
        proj.selections.append(sel)
        segs.append(seg)
    # Legacy recovery may already have touched the head. Semantic generations are independently
    # finite and must still begin at their deterministic first page.
    proj.meta["recovery_attempted"] = [0, 1]
    from vidlore.clipstudio import selfheal as S
    # The viewer confirmed only the cap-tail beat as a genuine gap. That authorization must not let
    # it jump ahead of its scheduled strict-positive page.
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [2], method="actual_frame_and_pool_audit")

    env = {
        "VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS": "2",
        "VIDLORE_CLIPSTUDIO_SELFHEAL": "1",
        "VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1",
    }
    image_scopes = []
    ladder_calls = []

    def record_image_scope(_proj, target_segs, *_args, **_kwargs):
        image_scopes.append([s.index for s in target_segs])
        return 0

    def exhausted_ladder(_proj, seg, *_args, **_kwargs):
        ladder_calls.append(seg.index)
        return False

    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections) \
            as matcher, \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.discover.discover_sources", return_value=[]), \
            mock.patch.object(O, "_fill_image_fallbacks", side_effect=record_image_scope), \
            mock.patch.object(S, "_soften_and_retry", side_effect=exhausted_ladder):
        first = _call(proj, segs)
        round1 = json.loads(
            (proj.output_dir / "semantic_recovery_audit.json").read_text())
        assert image_scopes == []
        assert ladder_calls == [], "any deferred page preserves every blocker at exact specificity"
        second = _call(proj, segs)
        round2 = json.loads(
            (proj.output_dir / "semantic_recovery_audit.json").read_text())
        assert image_scopes == [[0, 1, 2]]
        assert ladder_calls == [2], "beat 2 becomes eligible only after its own page is exhausted"
        third = _call(proj, segs)
        assert ladder_calls == [2] and len(image_scopes) == 1

        # A real source-state change starts a new generation; its first page is bounded again.
        new_media = tmp_path / "new_generation.mp4"
        new_media.write_bytes(b"new indexed-pool generation")
        proj.sources.append(SourceVideo(
            id="new_generation", url="u2", title="Game of Thrones new exact scene source",
            permission="owner", status="ok", local_path=str(new_media)))
        changed = _call(proj, segs)
        changed_round = json.loads(
            (proj.output_dir / "semantic_recovery_audit.json").read_text())

    assert all(a["status"] == "blocked" and a["blocked_count"] == 3
               for a in (first, second, third, changed)), "the publication gate never weakens"
    assert round1["current_pool_rematch"]["attempted"] == [0, 1]
    assert round1["deferred_retriable"] == [2]
    assert round2["current_pool_rematch"]["attempted"] == [2]
    assert round2["deferred"] == round2["deferred_retriable"] == []
    assert matcher.call_count == 3, "third unchanged Resume must not run recovery"
    assert changed_round["current_pool_rematch"]["attempted"] == [0, 1]
    assert proj.meta["selection_relevance_recovery"]["deferred"] == [2]


def test_tail_page_pool_growth_reopens_prior_blockers_once_then_exhausts(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    base_frame = tmp_path / "shot_0000.jpg"
    base_shot = Shot(source_id="s1", index=0, start=0.0, end=2.0,
                     keyframe_path=str(base_frame))
    for idx in (1, 2):
        seg = ScriptSegment(
            index=idx, text=f"Still blocked narration {idx}",
            expected_visual=f"Missing exact scene {idx}", required_entity=f"target {idx}",
            required_kind="object", scene_query=f"exact scene target {idx}",
            visual_policy=P.EXACT, is_specific_claim=True)
        sel = ClipSelection(
            segment_index=idx, source_id="s1", shot_index=0, in_point=0.0,
            out_point=2.0, confidence=0.8, verifier=dict(bad))
        V.bind_selection_verifier_evidence(
            proj, sel, seg, sel.verifier, shot=base_shot, model="vision",
            is_specific=True, multiframe=True, faceid_names=[], era="", must_see="")
        proj.segments.append(seg)
        proj.selections.append(sel)
        segs.append(seg)

    scopes = []
    image_scopes = []
    new_shot = None

    def recover(_proj, _segs, _analysis, _cfg, _eng, **kw):
        nonlocal new_shot
        scope = sorted(kw["only_indices"])
        scopes.append(scope)
        if len(scopes) == 1:
            assert scope == [0, 1, 2]
            _write_mock_recovery_page(_proj, kw, deferred=[2])
        elif len(scopes) == 2:
            assert scope == [2]
            media = tmp_path / "source-b.mp4"
            media.write_bytes(b"new pool generation containing beat zero")
            frame = tmp_path / "source-b-shot.jpg"
            frame.write_bytes(b"source-b exact beat zero frame")
            new_shot = Shot(source_id="source_b", index=0, start=4.0, end=6.0,
                            keyframe_path=str(frame))
            _proj.sources.append(SourceVideo(
                id="source_b", url="u-source-b", title="Exact scene for beat zero",
                permission="owner", status="ok", local_path=str(media)))
            _proj.shots_path("source_b").write_text(json.dumps([new_shot.to_dict()]))
            _write_mock_recovery_page(_proj, kw)
        else:
            assert scope == [0, 1], "only prior out-of-page blockers see the new generation"
            recovered = ClipSelection(
                segment_index=0, source_id="source_b", shot_index=0, in_point=4.0,
                out_point=6.0, confidence=0.96, verifier=dict(GOOD))
            V.bind_selection_verifier_evidence(
                _proj, recovered, segs[0], recovered.verifier, shot=new_shot,
                model="vision", is_specific=True, multiframe=True,
                faceid_names=[], era="", must_see="")
            _proj.selections = [recovered] + [
                s for s in _proj.selections if s.segment_index != 0]
            _write_mock_recovery_page(_proj, kw)
        return 0

    def record_images(_proj, target_segs, *_args, **_kwargs):
        image_scopes.append([s.index for s in target_segs])
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=recover) as recovery, \
            mock.patch.object(O, "_fill_image_fallbacks", side_effect=record_images):
        first = _call(proj, segs)
        first_marker = copy.deepcopy(proj.meta["selection_relevance_recovery"])
        assert image_scopes == []
        assert all(s.visual_policy == P.EXACT for s in segs)
        second = _call(proj, segs)
        second_marker = copy.deepcopy(proj.meta["selection_relevance_recovery"])
        assert image_scopes == []
        assert all(s.visual_policy == P.EXACT for s in segs), \
            "the head must survive exact until it sees the source added by the tail"
        third = _call(proj, segs)
        fourth = _call(proj, segs)

    assert first["blocked_count"] == second["blocked_count"] == 3
    assert first_marker["deferred"] == [2]
    assert second_marker["pool_changed_during_page"] is True
    assert second_marker["completed_page_scope"] == [2]
    assert second_marker["deferred"] == [0, 1]
    assert scopes == [[0, 1, 2], [2], [0, 1]]
    assert recovery.call_count == 3, "the unchanged post-growth generation exhausts finitely"
    assert image_scopes == [[1, 2]], "fallback starts only after the stable full strict walk"
    assert third["blocked_count"] == fourth["blocked_count"] == 2
    assert {e["segment_index"] for e in third["blockers"]} == {1, 2}
    assert proj.meta["selection_relevance_recovery"]["deferred"] == []


def test_single_preflight_drain_finishes_tail_and_pool_growth_rebound(tmp_path):
    """One portal build must finish the audited walk instead of failing at a page boundary.

    Page one leaves a capped tail. Page two grows the source pool, which legitimately re-opens the
    already-attempted head. The same preflight invocation must consume page three before the strict
    assertion is allowed to classify the still-blocked content as terminal.
    """
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio.verify import NonRetryableBuildError

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    _append_bound_rejected_beat(proj, segs, index=1)
    _append_bound_rejected_beat(proj, segs, index=2)
    calls = []
    rebound_calls = []

    def record_marker(audit, *, deferred, completed):
        blockers = sorted(int(e["segment_index"]) for e in audit["blockers"])
        content = O._selection_relevance_audit_without(audit, set())
        proj.meta["selection_relevance_recovery"] = {
            "schema_version": R.SCHEMA_VERSION,
            "before": [0, 1, 2],
            "after": blockers,
            "post_fingerprint": O._selection_relevance_retry_fingerprint(
                proj, segs, content),
            "deferred": list(deferred),
            "pool_fingerprint": O._semantic_recovery_pool_fingerprint(proj),
            "pool_changed_during_page": len(calls) == 2,
            "completed_page_scope": list(completed),
            "technical_blockers": [],
        }
        proj.save()

    def recover_one_page():
        calls.append(len(calls) + 1)
        if len(calls) == 2:
            # The tail page finds new indexed bytes. Model the production rebound: the old head has
            # never searched this generation and therefore becomes the next deferred scope.
            media = tmp_path / "page-growth.mp4"
            media.write_bytes(b"new strict recovery pool generation")
            proj.sources.append(SourceVideo(
                id="page_growth", url="u:page-growth", title="New exact scene source",
                permission="owner", status="ok", local_path=str(media)))
        audit = R.evaluate_selection_relevance(proj, segs)
        assert audit["blocked_count"] == 3
        if len(calls) == 1:
            record_marker(audit, deferred=[2], completed=[0, 1])
        elif len(calls) == 2:
            record_marker(audit, deferred=[0, 1], completed=[2])
        else:
            record_marker(audit, deferred=[], completed=[0, 1])
        return audit

    env = {
        "VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS": "2",
        "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY_MAX_PAGES": "6",
    }
    logs = []
    with mock.patch.dict(os.environ, env):
        final = O._drain_semantic_recovery_pages(
            proj, segs, recover_one_page,
            rebind_page=lambda: rebound_calls.append(len(calls)), log=logs.append)

    assert calls == [1, 2, 3]
    assert rebound_calls == [1, 2, 3]
    assert proj.meta["selection_relevance_recovery"]["deferred"] == []
    assert final["blocked_count"] == 3
    assert any("continuing 1 audited deferred" in line for line in logs)
    assert any("continuing 2 audited deferred" in line for line in logs)
    with pytest.raises(NonRetryableBuildError):
        R.assert_selection_relevance(proj, segs)


def test_single_preflight_drain_rejects_repeated_deferred_state(tmp_path):
    """A broken page cursor is a technical stop, never an infinite autonomous retry."""
    from vidlore.clipstudio import relevance_contract as R

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    calls = []

    def stuck_page():
        calls.append(len(calls) + 1)
        audit = R.evaluate_selection_relevance(proj, segs)
        content = O._selection_relevance_audit_without(audit, set())
        proj.meta["selection_relevance_recovery"] = {
            "schema_version": R.SCHEMA_VERSION,
            "before": [0], "after": [0], "deferred": [0],
            "post_fingerprint": O._selection_relevance_retry_fingerprint(
                proj, segs, content),
            "pool_fingerprint": O._semantic_recovery_pool_fingerprint(proj),
            "completed_page_scope": [], "technical_blockers": [],
        }
        return audit

    with mock.patch.dict(os.environ, {
            "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY_MAX_PAGES": "6"}):
        with pytest.raises(O.PipelineError, match="made no forward progress"):
            O._drain_semantic_recovery_pages(
                proj, segs, stuck_page, rebind_page=lambda: None, log=lambda _m: None)
    assert calls == [1, 2]


def test_single_preflight_drain_has_finite_whole_walk_guard(tmp_path):
    """Fresh pool bytes cannot turn autonomous pagination into an unbounded downloader."""
    from vidlore.clipstudio import relevance_contract as R

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    calls = []

    def always_growing_page():
        calls.append(len(calls) + 1)
        media = tmp_path / f"growth-{len(calls)}.mp4"
        media.write_bytes(f"pool generation {len(calls)}".encode())
        proj.sources.append(SourceVideo(
            id=f"growth_{len(calls)}", url=f"u:growth:{len(calls)}", title="New scene",
            permission="owner", status="ok", local_path=str(media)))
        audit = R.evaluate_selection_relevance(proj, segs)
        content = O._selection_relevance_audit_without(audit, set())
        proj.meta["selection_relevance_recovery"] = {
            "schema_version": R.SCHEMA_VERSION,
            "before": [0], "after": [0], "deferred": [0],
            "post_fingerprint": O._selection_relevance_retry_fingerprint(
                proj, segs, content),
            "pool_fingerprint": O._semantic_recovery_pool_fingerprint(proj),
            "completed_page_scope": [], "technical_blockers": [],
        }
        return audit

    with mock.patch.dict(os.environ, {
            "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY_MAX_PAGES": "2"}):
        with pytest.raises(O.PipelineError, match="finite 2-page guard"):
            O._drain_semantic_recovery_pages(
                proj, segs, always_growing_page,
                rebind_page=lambda: None, log=lambda _m: None)
    assert calls == [1, 2]


def test_page_growth_restores_hidden_29th_blocker_into_same_bounded_generation(tmp_path):
    """A pool-bound abstract beat must join pagination as soon as a page grows the pool.

    This reproduces the 101-beat run's 28→29 transition: 28 strict blockers enter semantic
    recovery while beat 24 is legitimately abstract for the old pool.  The first bounded page adds
    indexed footage, invalidating beat 24's softening.  Its restored strict blocker must be included
    in that page's persisted post-state/deferred tail, so Resume advances through the tail instead
    of treating the same pool as a fresh generation and restarting all 29 blockers.
    """
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    base_frame = tmp_path / "shot_0000.jpg"
    base_shot = Shot(source_id="s1", index=0, start=0.0, end=2.0,
                     keyframe_path=str(base_frame))
    for idx in range(1, 29):
        seg = ScriptSegment(
            index=idx, text=f"Strict blocked narration {idx}",
            expected_visual=f"Exact missing scene {idx}", required_entity=f"target {idx}",
            required_kind="object", scene_query=f"exact scene target {idx}",
            visual_policy=P.EXACT, is_specific_claim=True)
        sel = ClipSelection(
            segment_index=idx, source_id="s1", shot_index=0, in_point=0.0,
            out_point=2.0, confidence=0.8, verifier=dict(bad))
        V.bind_selection_verifier_evidence(
            proj, sel, seg, sel.verifier, shot=base_shot, model="vision",
            is_specific=True, multiframe=True, faceid_names=[], era="", must_see="")
        proj.segments.append(seg)
        proj.selections.append(sel)
        segs.append(seg)

    softened_idx = 24
    softened_seg = next(s for s in segs if s.index == softened_idx)
    softened_sel = next(s for s in proj.selections if s.segment_index == softened_idx)
    original = S._capture_softening_state(softened_seg, softened_sel)
    old_pool_fp, old_pool_n = S._gap_pool_fingerprint(proj)
    softened_seg.visual_policy = P.ABSTRACT
    softened_seg.required_entity = ""
    softened_seg.required_kind = ""
    softened_seg.scene_query = ""
    softened_seg.expected_visual = "Neutral atmosphere"
    softened_seg.is_specific_claim = False
    softened_sel.image_meta = {"selfheal_rung": "abstract", "installed": True}
    S._record_phase1_softening(
        proj, softened_seg, softened_sel, original, basis="reproduced_old_pool_gap",
        pool_fingerprint=old_pool_fp, pool_source_count=old_pool_n)

    initial = R.evaluate_selection_relevance(proj, segs)
    assert initial["blocked_count"] == 28
    assert softened_idx not in {e["segment_index"] for e in initial["blockers"]}

    scopes = []

    def bounded_pages(_proj, _segs, _analysis, _cfg, _eng, **kw):
        scope = sorted(kw["only_indices"])
        scopes.append(scope)
        if len(scopes) == 1:
            assert len(scope) == 28 and softened_idx not in scope
            media = tmp_path / "page-growth.mp4"
            media.write_bytes(b"new indexed footage")
            frame = tmp_path / "page-growth-shot.jpg"
            frame.write_bytes(b"new indexed frame")
            _proj.sources.append(SourceVideo(
                id="page_growth", url="u-page-growth", title="New exact scene source",
                permission="owner", status="ok", local_path=str(media)))
            _proj.shots_path("page_growth").write_text(json.dumps([Shot(
                source_id="page_growth", index=0, start=0.0, end=2.0,
                keyframe_path=str(frame)).to_dict()]))
            (_proj.index_dir / "page_growth.words.json").write_text("[]")
            # Model a cap=8 page: twenty original blockers remain in its audited tail.
            _write_mock_recovery_page(_proj, kw, deferred=scope[8:])
        else:
            # The second Resume must receive only the prior generation's deferred scope. Advance
            # another bounded page without changing the pool again.
            _write_mock_recovery_page(_proj, kw, deferred=scope[8:])
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=bounded_pages), \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        first = _call(proj, segs)
        first_marker = copy.deepcopy(proj.meta["selection_relevance_recovery"])
        first_row = next(r for r in
                         proj.meta["selection_relevance_gap_softening"]["beats"]
                         if r["segment_index"] == softened_idx)

        assert first["blocked_count"] == 29
        assert first_marker["before"] == sorted(i for i in range(29) if i != softened_idx)
        assert first_marker["after"] == list(range(29))
        assert first_row["status"] == "restored_pool_changed"
        assert first_row["active"] is False
        assert len(first_marker["completed_page_scope"]) == 8
        assert len(first_marker["deferred"]) == 21
        assert softened_idx in first_marker["deferred"]
        assert first_marker["post_fingerprint"] == \
            O._selection_relevance_retry_fingerprint(proj, segs, first)
        assert first_marker["pool_fingerprint"] == \
            O._semantic_recovery_pool_fingerprint(proj)

        expected_resume_scope = first_marker["deferred"]
        second = _call(proj, segs)
        second_marker = copy.deepcopy(proj.meta["selection_relevance_recovery"])

    assert scopes[1] == expected_resume_scope
    assert len(scopes[1]) == 21 < second["blocked_count"] == 29
    assert len(second_marker["deferred"]) == 13
    assert set(second_marker["deferred"]).issubset(set(expected_resume_scope))
    images.assert_not_called()
    ladder.assert_not_called()


@pytest.mark.parametrize("recovery_marker", [
    pytest.param(None, id="marker-absent"),
    pytest.param({
        "schema_version": -1, "post_fingerprint": "stale",
        "deferred": [24], "after": [24], "pool_fingerprint": "old-pool",
    }, id="marker-stale"),
])
def test_semantic_preflight_precedes_legacy_preassembly_without_valid_marker(
        tmp_path, recovery_marker):
    """Absent/stale pagination metadata cannot let legacy repair deadlock semantic retry.

    This reproduces the scale-run ordering failure: a rematch invalidates the progress marker while
    the strict beat is still blocked, and the older predictor also wants to heal it. The semantic
    assertion/retry must run first unconditionally; only after it passes may legacy self-heal run.
    build_video is still called and models its own semantic assertion before its independent
    rejected-footage gate, proving neither publication contract was bypassed.
    """
    from vidlore.clipstudio import analyze as AN
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import discover as DS
    from vidlore.clipstudio import download as DL
    from vidlore.clipstudio import faceid as FI
    from vidlore.clipstudio import index as INDEX
    from vidlore.clipstudio import ledger as LG
    from vidlore.clipstudio import review as RV
    from vidlore.clipstudio.analyze import ScriptAnalysis

    seg = ScriptSegment(
        index=24, text="Beat twenty four strict narration", scene_query="missing exact scene",
        expected_visual="Missing exact scene", required_entity="Dontos",
        required_kind="character", visual_policy=P.EXACT, is_specific_claim=True)
    candidate = DS.SourceCandidate(url="https://video/source", id="src", title="Scene")

    def analyze(*_args, **_kwargs):
        return (ScriptAnalysis(topic="t", movie_title="Game of Thrones",
                               video_type="multi_scene", actors=[], characters=[]), [seg])

    def download(proj, *_args, **_kwargs):
        proj.sources = [SourceVideo(
            id="src", url=candidate.url, title=candidate.title, permission="owner",
            status="ok", local_path=str(tmp_path / "source.mp4"), duration=30.0)]
        (tmp_path / "source.mp4").write_bytes(b"source")

    def match(proj, *_args, **_kwargs):
        proj.selections = [ClipSelection(
            segment_index=24, source_id="src", shot_index=0, in_point=0.0, out_point=3.0,
            confidence=0.8, flag_reasons=["verifier_failed"],
            verifier={"status": "ok", "verdict": "replace"},
            beat_windows=[["src", 0.0, 3.0]])]
        if recovery_marker is not None:
            proj.meta["selection_relevance_recovery"] = copy.deepcopy(recovery_marker)

    events = []
    recovered = {"done": False}
    from vidlore.clipstudio.verify import NonRetryableBuildError

    clear = {"status": "pass", "blocked_count": 0, "blockers": []}

    def strict_assert(*_args, **_kwargs):
        events.append("semantic-assert-pass" if recovered["done"] else "semantic-assert-block")
        if not recovered["done"]:
            raise NonRetryableBuildError(
                "strict selection relevance still blocked", kind="selection_relevance")
        return clear

    def retry(*_args, **_kwargs):
        events.append("semantic-retry")
        recovered["done"] = True
        return clear

    def predictor(*_args, **_kwargs):
        events.append("legacy-predictor")
        return "pre-assembly blocker scene(s) [24]"

    def legacy_selfheal_fn(*_args, **_kwargs):
        events.append("legacy-selfheal")
        return None

    def authoritative_build(*_args, **_kwargs):
        events.append("build")
        # The real build_video starts with this same assertion. Keep it explicit while mocking the
        # expensive encoder so the regression proves the authoritative assertion remains ordered.
        strict_assert()
        raise NonRetryableBuildError(
            "authoritative rejected footage still blocked", kind="rejected_footage")

    def finalize(*_args, **_kwargs):
        events.append("qc")
        return {"flagged_for_review": 1, "segments": 1, "mean_confidence": 0.8}

    env = {
        "VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED": "0",
        "VIDLORE_CLIPSTUDIO_INDEX_OVERLAP": "0",
        "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY": "1",
        "VIDLORE_CLIPSTUDIO_SELFHEAL_BUILD_RETRY": "0",
    }
    legacy_selfheal = mock.Mock(side_effect=legacy_selfheal_fn)
    retry_mock = mock.Mock(side_effect=retry)
    from contextlib import ExitStack
    from vidlore.clipstudio import relevance_contract as R
    patches = [
        mock.patch.dict(os.environ, env),
        mock.patch.object(AN, "analyze_script", side_effect=analyze),
        mock.patch.object(DS, "discover_sources", return_value=[candidate]),
        mock.patch.object(DL, "download_candidates", side_effect=download),
        mock.patch.object(FI, "available", return_value=False),
        mock.patch.object(INDEX, "clip_available", return_value=True),
        mock.patch.object(O, "asr_pool_current", return_value=True),
        mock.patch.object(O, "index_all"),
        mock.patch.object(O, "_ensure_anchor_coverage"),
        mock.patch.object(O, "match_segments", side_effect=match),
        mock.patch.object(O, "cut_all"),
        mock.patch.object(O, "_recover_unresolved_beats"),
        mock.patch.object(O, "_fill_image_fallbacks"),
        mock.patch.object(O, "_purge_unwanted_sources", return_value=0),
        mock.patch.object(RV, "write_review", return_value="review.html"),
        mock.patch.object(LG, "finalize", side_effect=finalize),
        mock.patch.object(R, "assert_selection_relevance", side_effect=strict_assert),
        mock.patch.object(B, "preassemble_release_block_reason", side_effect=predictor),
        mock.patch.object(O, "_run_preassemble_selfheal", legacy_selfheal),
        mock.patch.object(O, "_retry_selection_relevance", retry_mock),
        mock.patch.object(O, "build_video", side_effect=authoritative_build),
    ]
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        with pytest.raises(NonRetryableBuildError) as exc:
            O.produce_auto(
                tmp_path, topic="t", script_text="A sufficiently real narration script.",
                movie_hint="Game of Thrones", policy="approved_testing", max_sources=1,
                do_build=True, verify=False)

    assert exc.value.kind == "rejected_footage"
    retry_mock.assert_called_once()
    legacy_selfheal.assert_called_once()
    assert events == [
        "semantic-assert-block", "semantic-retry", "semantic-assert-pass",
        "qc", "legacy-predictor", "legacy-selfheal", "qc",
        "build", "semantic-assert-pass",
    ]


def test_scoped_recovery_reuses_current_pool_before_demanding_a_new_url(tmp_path):
    """A contract blocker is eligible even when the legacy verifier says ``keep``.

    This is the exact shape of a quote-window failure: the dedicated source is already indexed,
    the selected picture looks plausible, but the timed dialogue is outside the window.  Recovery
    must rematch the current pool and retain only the scoped beat that clears the strict contract.
    """
    proj, segs, old_target = _fixture(tmp_path, GOOD)
    other_seg = ScriptSegment(index=1, text="Context line", visual_policy=P.FILLER)
    old_other = ClipSelection(
        segment_index=1, source_id="s1", shot_index=0, in_point=0.0, out_point=1.0,
        confidence=0.9, verifier=dict(GOOD))
    proj.segments.append(other_seg)
    segs.append(other_seg)
    proj.selections.append(old_other)

    def rematch(_proj, _segs, _cfg, **_kw):
        _proj.selections = [
            ClipSelection(segment_index=0, source_id="dedicated_quote_source", shot_index=9,
                          in_point=12.0, out_point=14.0, confidence=0.95,
                          verifier=dict(GOOD)),
            ClipSelection(segment_index=1, source_id="churned_non_target", shot_index=4,
                          in_point=8.0, out_point=9.0, confidence=0.7,
                          verifier=dict(GOOD)),
            ClipSelection(segment_index=99, source_id="invented_non_target", shot_index=2,
                          in_point=3.0, out_point=4.0, confidence=0.7,
                          verifier=dict(GOOD)),
        ]
        return _proj.selections

    def cut_one(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(f"clip:{sel.source_id}".encode())
        return path

    strict_clear = {"blockers": [], "blocked_count": 0, "status": "pass"}
    with mock.patch.object(O, "match_segments", side_effect=rematch) as matcher, \
            mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=cut_one) as cutter, \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)) as verify, \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=strict_clear), \
            mock.patch("vidlore.clipstudio.discover.discover_sources") as discover:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-current-pool.json")

    assert recovered == 1
    matcher.assert_called_once()
    verify.assert_called_once()
    assert verify.call_args.kwargs["only_indices"] == {0}
    assert verify.call_args.kwargs["materialize_promotions"] is False
    assert verify.call_args.kwargs["persist_project"] is False
    discover.assert_not_called(), "an already-indexed recovery must not demand a novel URL"
    by_idx = {s.segment_index: s for s in proj.selections}
    assert by_idx[0].source_id == "dedicated_quote_source"
    assert by_idx[1].source_id == old_other.source_id, "non-target matcher churn must roll back"
    assert 99 not in by_idx, "a rematched entry absent from the snapshot must never leak in"
    assert cutter.call_count == 1, "only the accepted scoped selection gets fresh clip bytes"
    audit = json.loads((proj.output_dir / "strict-current-pool.json").read_text())
    assert audit["current_pool_rematch"]["recovered"] == [0]


@pytest.mark.parametrize("summary,expected_reason", INCONCLUSIVE_VERIFY_SUMMARIES)
def test_current_pool_inconclusive_verifier_rolls_back_and_marks_page_incomplete(
        tmp_path, summary, expected_reason):
    proj, segs, original = _fixture(tmp_path, GOOD)

    def exploratory_rematch(_proj, *_args, **_kwargs):
        _proj.selections = [ClipSelection(
            segment_index=0, source_id="exploratory", shot_index=7,
            in_point=9.0, out_point=11.0, confidence=0.97, verifier=dict(GOOD))]
        return _proj.selections

    with mock.patch.object(O, "match_segments", side_effect=exploratory_rematch), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=summary) as verifier, \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance") \
            as contract, \
            mock.patch("vidlore.clipstudio.discover.discover_sources") as discover, \
            mock.patch("vidlore.clipstudio.cut.cut_selection") as cutter:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-current-inconclusive.json", audit_request_id="current-page")

    assert recovered == 0
    verifier.assert_called_once()
    contract.assert_not_called(), "no strict acceptance may run on an inconclusive verdict batch"
    discover.assert_not_called()
    cutter.assert_not_called()
    assert proj.selections[0].source_id == original.source_id == "s1"
    page = json.loads(
        (proj.output_dir / "strict-current-inconclusive.json").read_text(encoding="utf-8"))
    assert page["page_completed"] is False
    assert expected_reason in page["current_pool_rematch"]["error"]
    assert expected_reason in page["page_error"]


def _targeted_recovery_analysis():
    return SimpleNamespace(
        movie_title="Game of Thrones", year="", video_type="multi_scene",
        anchor_scenes=[], key_scenes=[], characters=[], actors=[], locations=[], events=[],
        visual_keywords=[], emotional_moments=[], episode_hint="", synopsis="", tone="",
        char_to_actor=lambda: {})


def test_targeted_discovery_all_technical_statuses_leave_page_retryable(tmp_path):
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    yt_calls, archive_calls = [], []

    def yt_down(query, _n):
        yt_calls.append(query)
        return [], D.STATUS_TRANSPORT

    def archive_down(query, _n):
        archive_calls.append(query)
        return [], D.STATUS_TIMEOUT

    query = segs[0].scene_query
    env = {"VIDLORE_CLIPSTUDIO_DISCOVER_WORKERS": "1",
           "VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch.object(D, "_ytsearch_ex", side_effect=yt_down), \
            mock.patch.object(D, "_archive_search_ex", side_effect=archive_down), \
            mock.patch("time.sleep", return_value=None), \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        with pytest.raises(O.PipelineError, match="TargetedDiscoveryTechnicalError"):
            O._retry_selection_relevance(
                proj, segs, ClipConfig(), _targeted_recovery_analysis(),
                SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
                policy="approved_testing", log=lambda _m: None)

    assert yt_calls.count(query) == archive_calls.count(query) == 3
    images.assert_not_called()
    ladder.assert_not_called()
    assert "selection_relevance_recovery" not in proj.meta
    page = json.loads((proj.output_dir / "semantic_recovery_audit.json").read_text())
    assert page["page_completed"] is False
    assert "TargetedDiscoveryTechnicalError" in page["page_error"]


def test_targeted_download_all_failed_rolls_back_rows_and_resume_retries(tmp_path):
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    candidate = SimpleNamespace(
        url="https://video/retry-download", title="Game of Thrones Olenna necklace stone",
        id="retry-download", query=segs[0].scene_query, provider="youtube",
        permission_hint="", relevance=0.95, anchor_verified=False)
    attempts = []

    def fail_download(cand, sid, perm, note, _proj, _cfg, progress=None):
        attempts.append(cand.url)
        return SourceVideo(
            id=sid, url=cand.url, title=cand.title, permission=perm,
            permission_note=note, status="download_failed", error="timed out after retries")

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1", "VIDLORE_HD_403_SWEEP": "0"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]) as discover, \
            mock.patch("vidlore.clipstudio.download._download_one",
                       side_effect=fail_download), \
            mock.patch("time.sleep", return_value=None), \
            mock.patch.object(O, "index_all") as indexer, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        for _resume in range(2):
            with pytest.raises(O.PipelineError, match="targeted downloads were technically"):
                O._retry_selection_relevance(
                    proj, segs, ClipConfig(), _targeted_recovery_analysis(),
                    SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
                    policy="approved_testing", log=lambda _m: None)
            assert all(s.url != candidate.url for s in proj.sources)
            assert "selection_relevance_recovery" not in proj.meta

    assert attempts == [candidate.url, candidate.url]
    assert discover.call_count == 2, "the failed URL must remain eligible on Resume"
    indexer.assert_not_called()
    images.assert_not_called()
    ladder.assert_not_called()
    page = json.loads((proj.output_dir / "semantic_recovery_audit.json").read_text())
    assert page["page_completed"] is False
    assert page["download_failures"][0]["error"] == "timed out after retries"


def test_targeted_downloader_exception_after_row_mutation_rolls_back_for_resume(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    candidate = SimpleNamespace(
        url="https://video/raised-download", title="Game of Thrones Olenna necklace stone",
        id="raised-download", query=segs[0].scene_query, provider="youtube")
    attempts = []

    def mutate_then_raise(_proj, *_args, **_kwargs):
        attempts.append(candidate.url)
        _proj.sources.append(SourceVideo(
            id="raised_download_row", url=candidate.url, title=candidate.title,
            permission="owner", status="download_failed", error="worker future exploded"))
        _proj.save()
        raise RuntimeError("downloader crashed after manifest mutation")

    still_blocked = {"status": "blocked", "blocked_count": 1,
                     "blockers": [{"segment_index": 0}]}
    with mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=still_blocked), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]) as discover, \
            mock.patch("vidlore.clipstudio.download.download_candidates",
                       side_effect=mutate_then_raise):
        for _resume in range(2):
            assert O._recover_unresolved_beats(
                proj, segs, _targeted_recovery_analysis(), ClipConfig(),
                SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
                policy="approved_testing", log=lambda _m: None, only_indices={0},
                audit_filename="raised-download.json",
                audit_request_id=f"raised-{_resume}") == 0
            assert all(s.url != candidate.url for s in proj.sources)
            persisted = ClipProject.load(tmp_path)
            assert all(s.url != candidate.url for s in persisted.sources)

    assert attempts == [candidate.url, candidate.url]
    assert discover.call_count == 2
    page = json.loads((proj.output_dir / "raised-download.json").read_text())
    assert page["page_completed"] is False
    assert "downloader crashed after manifest mutation" in page["page_error"]


def test_targeted_download_mixed_success_and_failure_is_still_inconclusive(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    candidates = [
        SimpleNamespace(
            url=f"https://video/mixed-{kind}",
            title=f"Game of Thrones Olenna necklace {kind}", id=f"mixed-{kind}",
            query=segs[0].scene_query, provider="youtube", permission_hint="",
            relevance=0.95, anchor_verified=False)
        for kind in ("good", "failed")
    ]

    def mixed_download(cand, sid, perm, note, _proj, _cfg, progress=None):
        if cand.url.endswith("good"):
            media = tmp_path / f"{sid}.mp4"
            media.write_bytes(b"usable new source")
            return SourceVideo(
                id=sid, url=cand.url, title=cand.title, permission=perm,
                permission_note=note, status="ok", local_path=str(media), height=1080)
        return SourceVideo(
            id=sid, url=cand.url, title=cand.title, permission=perm,
            permission_note=note, status="download_failed", error="transport reset")

    still_blocked = {"status": "blocked", "blocked_count": 1,
                     "blockers": [{"segment_index": 0}]}
    env = {"VIDLORE_HD_403_SWEEP": "0"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=still_blocked), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=candidates), \
            mock.patch("vidlore.clipstudio.download._download_one",
                       side_effect=mixed_download), \
            mock.patch("time.sleep", return_value=None), \
            mock.patch.object(O, "index_all") as indexer:
        recovered = O._recover_unresolved_beats(
            proj, segs, _targeted_recovery_analysis(), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="mixed-download.json", audit_request_id="mixed-download")

    assert recovered == 0
    assert all(s.url not in {c.url for c in candidates} for s in proj.sources)
    indexer.assert_not_called()
    page = json.loads((proj.output_dir / "mixed-download.json").read_text())
    assert page["page_completed"] is False
    assert len(page["download_failures"]) == 1


def test_targeted_source_with_zero_index_rolls_back_and_resume_retries(tmp_path):
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    candidate = SimpleNamespace(
        url="https://video/retry-index", title="Game of Thrones Olenna necklace stone",
        id="retry-index", query=segs[0].scene_query, provider="youtube")
    downloads = []

    def download_ok(_proj, *_args, **_kwargs):
        downloads.append(candidate.url)
        media = tmp_path / "retry-index.mp4"
        media.write_bytes(b"readable media with no searchable shots")
        _proj.sources.append(SourceVideo(
            id="retry_index_source", url=candidate.url, title=candidate.title,
            permission="owner", status="ok", local_path=str(media)))
        return _proj.sources

    env = {"VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK": "1"}
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]) as discover, \
            mock.patch("vidlore.clipstudio.download.download_candidates",
                       side_effect=download_ok), \
            mock.patch.object(O, "index_all",
                              return_value={"retry_index_source": []}) as indexer, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        for _resume in range(2):
            with pytest.raises(O.PipelineError, match="no searchable shots"):
                O._retry_selection_relevance(
                    proj, segs, ClipConfig(), _targeted_recovery_analysis(),
                    SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
                    policy="approved_testing", log=lambda _m: None)
            assert all(s.url != candidate.url for s in proj.sources)
            assert "selection_relevance_recovery" not in proj.meta

    assert downloads == [candidate.url, candidate.url]
    assert discover.call_count == indexer.call_count == 2
    images.assert_not_called()
    ladder.assert_not_called()
    page = json.loads((proj.output_dir / "semantic_recovery_audit.json").read_text())
    assert page["page_completed"] is False
    assert page["index_results"] == [{"id": "retry_index_source", "shots": 0}]


def test_targeted_index_mixed_searchable_and_empty_rolls_back_whole_attempt(tmp_path):
    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    candidate = SimpleNamespace(
        url="https://video/mixed-index", title="Game of Thrones Olenna necklace stone",
        id="mixed-index", query=segs[0].scene_query, provider="youtube")
    new_urls = {"https://video/index-good", "https://video/index-empty"}

    def download_two(_proj, *_args, **_kwargs):
        for sid, url in (("index_good", "https://video/index-good"),
                         ("index_empty", "https://video/index-empty")):
            media = tmp_path / f"{sid}.mp4"
            media.write_bytes(b"new source bytes")
            _proj.sources.append(SourceVideo(
                id=sid, url=url, title=sid, permission="owner", status="ok",
                local_path=str(media)))
        return _proj.sources

    still_blocked = {"status": "blocked", "blocked_count": 1,
                     "blockers": [{"segment_index": 0}]}
    with mock.patch.object(O, "match_segments", side_effect=lambda *a, **k: proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=still_blocked), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]), \
            mock.patch("vidlore.clipstudio.download.download_candidates",
                       side_effect=download_two), \
            mock.patch.object(O, "index_all", return_value={
                "index_good": [object()], "index_empty": []}), \
            mock.patch("vidlore.clipstudio.cut.cut_selection") as cutter:
        recovered = O._recover_unresolved_beats(
            proj, segs, _targeted_recovery_analysis(), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="mixed-index.json", audit_request_id="mixed-index")

    assert recovered == 0
    assert all(s.url not in new_urls for s in proj.sources)
    cutter.assert_not_called()
    page = json.loads((proj.output_dir / "mixed-index.json").read_text())
    assert page["page_completed"] is False
    assert page["index_results"] == [
        {"id": "index_good", "shots": 1}, {"id": "index_empty", "shots": 0}]


def test_new_source_recovery_uses_strict_contract_not_legacy_keep_predicate(tmp_path):
    """A visual ``keep`` is not recovery when a real quote still misses its timed window."""
    proj, segs, old_target = _fixture(tmp_path, GOOD)
    segs[0].quote = "Chaos isn't a pit. Chaos is a ladder."

    candidate = SimpleNamespace(
        url="https://video/new", title="Game of Thrones Chaos is a Ladder", id="new",
        query="Chaos is a ladder", provider="youtube")

    def rematch(_proj, _segs, _cfg, **_kw):
        # First call (current pool) remains wrong; after download, select the newly indexed source.
        sid = "new_source" if any(s.id == "new_source" for s in _proj.sources) else "s1"
        _proj.selections = [ClipSelection(
            segment_index=0, source_id=sid, shot_index=1, in_point=4.0, out_point=7.0,
            confidence=0.9, verifier=dict(GOOD))]
        return _proj.selections

    def download(_proj, _cands, _cfg, **_kw):
        media = tmp_path / "new.mp4"
        media.write_bytes(b"new source")
        _proj.sources.append(SourceVideo(
            id="new_source", url=candidate.url, title=candidate.title, permission="owner",
            status="ok", local_path=str(media)))
        return _proj.sources

    def cut_one(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(f"clip:{sel.source_id}".encode())
        return path

    strict_blocked = {"blockers": [{"segment_index": 0}], "blocked_count": 1,
                      "status": "blocked"}
    strict_clear = {"blockers": [], "blocked_count": 0, "status": "pass"}
    with mock.patch.object(O, "match_segments", side_effect=rematch) as matcher, \
            mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=cut_one), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)) as verifier, \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       side_effect=[strict_blocked, strict_clear]) as contract, \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]), \
            mock.patch("vidlore.clipstudio.download.download_candidates",
                       side_effect=download), \
            mock.patch.object(O, "index_all", return_value={"new_source": [object()]}):
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-new-source.json")

    assert recovered == 1
    assert matcher.call_count == 2
    assert verifier.call_count == 2
    assert all(call.kwargs["materialize_promotions"] is False
               and call.kwargs["persist_project"] is False
               for call in verifier.call_args_list)
    assert contract.call_count == 2
    assert proj.selections[0].source_id == "new_source"
    audit = json.loads((proj.output_dir / "strict-new-source.json").read_text())
    assert audit["recovered"] == [0]


@pytest.mark.parametrize("summary,expected_reason", INCONCLUSIVE_VERIFY_SUMMARIES)
def test_new_source_inconclusive_verifier_rolls_back_and_marks_page_incomplete(
        tmp_path, summary, expected_reason):
    proj, segs, original = _fixture(tmp_path, GOOD)
    candidate = SimpleNamespace(
        url="https://video/new-inconclusive", title="Exact scene candidate", id="new-candidate",
        query="exact scene", provider="youtube")

    def rematch(_proj, *_args, **_kwargs):
        if any(s.id == "new_inconclusive" for s in _proj.sources):
            _proj.selections = [ClipSelection(
                segment_index=0, source_id="new_inconclusive", shot_index=3,
                in_point=5.0, out_point=7.0, confidence=0.98, verifier=dict(GOOD))]
        return _proj.selections

    def download(_proj, *_args, **_kwargs):
        media = tmp_path / "new-inconclusive.mp4"
        media.write_bytes(b"new source awaiting a real verdict")
        _proj.sources.append(SourceVideo(
            id="new_inconclusive", url=candidate.url, title=candidate.title,
            permission="owner", status="ok", local_path=str(media)))
        return _proj.sources

    still_blocked = {"status": "blocked", "blocked_count": 1,
                     "blockers": [{"segment_index": 0}]}
    with mock.patch.object(O, "match_segments", side_effect=rematch) as matcher, \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       side_effect=[dict(VERIFY_OK), summary]) as verifier, \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=still_blocked) as contract, \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       return_value=[candidate]), \
            mock.patch("vidlore.clipstudio.download.download_candidates", side_effect=download), \
            mock.patch.object(O, "index_all",
                              return_value={"new_inconclusive": [object()]}), \
            mock.patch("vidlore.clipstudio.cut.cut_selection") as cutter:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-new-inconclusive.json", audit_request_id="new-page")

    assert recovered == 0
    assert matcher.call_count == verifier.call_count == 2
    assert contract.call_count == 1, "new footage is never contract-cleared without a verdict"
    cutter.assert_not_called()
    assert proj.selections[0].source_id == original.source_id == "s1"
    assert any(s.id == "new_inconclusive" for s in proj.sources)
    page = json.loads(
        (proj.output_dir / "strict-new-inconclusive.json").read_text(encoding="utf-8"))
    assert page["page_completed"] is False
    assert page["current_pool_rematch"]["error"] == ""
    assert expected_reason in page["page_error"]


def test_scoped_commit_rolls_back_metadata_and_clip_bytes_on_partial_cut(tmp_path):
    proj, segs, old0 = _fixture(tmp_path, GOOD)
    seg1 = ScriptSegment(index=1, text="Second exact line", scene_query="second scene",
                         required_entity="Ned Stark", required_kind="character",
                         visual_policy=P.EXACT, is_specific_claim=True)
    old1 = ClipSelection(segment_index=1, source_id="s1", shot_index=0,
                         in_point=0.0, out_point=1.0, confidence=0.8,
                         verifier=dict(GOOD))
    proj.segments.append(seg1)
    proj.selections.append(old1)
    snapshot = {s.segment_index: copy.deepcopy(s) for s in proj.selections}
    rematched = {
        0: ClipSelection(segment_index=0, source_id="new0", shot_index=3,
                         in_point=4.0, out_point=6.0, confidence=0.9),
        1: ClipSelection(segment_index=1, source_id="new1", shot_index=4,
                         in_point=7.0, out_point=9.0, confidence=0.9),
    }
    old_bytes = {0: b"old-clip-zero", 1: b"old-clip-one"}
    for idx, payload in old_bytes.items():
        (proj.clips_dir / f"seg_{idx:03d}.mp4").write_bytes(payload)

    def partial_cut(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(f"new:{sel.source_id}".encode())
        return path if sel.segment_index == 0 else None

    with mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=partial_cut):
        ok = O._commit_scoped_recovery(
            proj, ClipConfig(), snapshot, rematched, {0, 1}, log=lambda _m: None)

    assert ok is False
    assert {s.segment_index: s.source_id for s in proj.selections} == {0: "s1", 1: "s1"}
    for idx, payload in old_bytes.items():
        assert (proj.clips_dir / f"seg_{idx:03d}.mp4").read_bytes() == payload


@pytest.mark.parametrize("allocation_failure", ["mkdir", "mkdtemp"])
def test_scoped_commit_allocation_failure_restores_metadata_without_touching_clips(
        tmp_path, allocation_failure):
    proj, _segs, _old = _fixture(tmp_path, GOOD)
    snapshot = {s.segment_index: copy.deepcopy(s) for s in proj.selections}
    rematched = {
        0: ClipSelection(segment_index=0, source_id="exploratory", shot_index=8,
                         in_point=12.0, out_point=14.0, confidence=0.95),
    }
    clip = proj.clips_dir / "seg_000.mp4"
    clip.write_bytes(b"known-old-clip-bytes")
    # This is the real entry state: match/verify has already installed exploratory metadata while
    # the deterministic clip filename still contains the snapshot's bytes.
    proj.selections = list(rematched.values())
    proj.save()

    if allocation_failure == "mkdtemp":
        allocation_patch = mock.patch("tempfile.mkdtemp", side_effect=OSError("disk full"))
    else:
        real_mkdir = type(proj.output_dir).mkdir

        def fail_output_dir(path, *args, **kwargs):
            if path == proj.output_dir:
                raise OSError("output directory unavailable")
            return real_mkdir(path, *args, **kwargs)

        # Install a real descriptor function so ``path`` is the Path instance (a MagicMock class
        # method would consume/bind arguments differently and also break rollback's project save).
        allocation_patch = mock.patch("pathlib.Path.mkdir", new=fail_output_dir)

    with allocation_patch, \
            mock.patch("vidlore.clipstudio.cut.cut_selection") as cutter:
        ok = O._commit_scoped_recovery(
            proj, ClipConfig(), snapshot, rematched, {0}, log=lambda _m: None)

    assert ok is False
    cutter.assert_not_called()
    assert [(s.segment_index, s.source_id) for s in proj.selections] == [(0, "s1")]
    assert clip.read_bytes() == b"known-old-clip-bytes"
    persisted = ClipProject.load(tmp_path)
    assert [(s.segment_index, s.source_id) for s in persisted.selections] == [(0, "s1")]


def test_current_pool_recovery_honors_cap_and_audits_deferred_scope(tmp_path):
    proj, first_segs, first_sel = _fixture(tmp_path, GOOD)
    segs = []
    sels = []
    for idx in range(5):
        segs.append(ScriptSegment(
            index=idx, text=f"Exact line {idx}", scene_query=f"exact scene {idx}",
            required_entity="Ned Stark", required_kind="character",
            visual_policy=P.EXACT, is_specific_claim=True))
        sels.append(ClipSelection(
            segment_index=idx, source_id="s1", shot_index=0, in_point=0.0,
            out_point=2.0, confidence=0.8, verifier=dict(GOOD)))
    proj.segments = segs
    proj.selections = sels
    all_blocked = {"status": "blocked", "blocked_count": 5,
                   "blockers": [{"segment_index": i} for i in range(5)]}

    with mock.patch.dict(os.environ, {"VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS": "2"}), \
            mock.patch.object(O, "match_segments", return_value=proj.selections), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)) as verify, \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=all_blocked), \
            mock.patch("vidlore.clipstudio.discover.discover_sources", return_value=[]):
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices=set(range(5)),
            audit_filename="strict-cap.json")

    assert recovered == 0
    assert verify.call_args.kwargs["only_indices"] == {0, 1}
    audit = json.loads((proj.output_dir / "strict-cap.json").read_text())
    assert audit["current_pool_rematch"]["attempted"] == [0, 1]
    assert audit["deferred"] == [2, 3, 4]
    assert audit["still_unresolved"] == [0, 1, 2, 3, 4]


def test_text_only_strict_beat_is_retrievable_in_current_pool_page(tmp_path):
    proj, segs, old0 = _fixture(tmp_path, GOOD)
    text_only = ScriptSegment(
        index=1, text="That name is Tyrion Lannister.", scene_query="",
        expected_visual="", required_entity="", required_kind="", visual_policy=P.EXACT,
        is_specific_claim=True)
    segs.append(text_only)
    proj.segments.append(text_only)
    proj.selections.append(ClipSelection(
        segment_index=1, source_id="s1", shot_index=0, in_point=0.0, out_point=1.0,
        confidence=0.8, verifier=dict(GOOD)))

    def rematch(_proj, _segs, _cfg, **_kw):
        _proj.selections = [
            ClipSelection(segment_index=0, source_id="current_pool_exact", shot_index=4,
                          in_point=4.0, out_point=6.0, confidence=0.9,
                          verifier=dict(GOOD)),
            ClipSelection(segment_index=1, source_id="churned", shot_index=2,
                          in_point=2.0, out_point=3.0, confidence=0.7,
                          verifier=dict(GOOD)),
        ]
        return _proj.selections

    def cut_one(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(b"new exact clip")
        return path

    with mock.patch.object(O, "match_segments", side_effect=rematch), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value={"status": "pass", "blocked_count": 0, "blockers": []}), \
            mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=cut_one), \
            mock.patch("vidlore.clipstudio.discover.discover_sources") as discover:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0, 1},
            audit_filename="strict-text-only.json")

    assert recovered == 2
    discover.assert_not_called()
    audit = json.loads((proj.output_dir / "strict-text-only.json").read_text())
    assert audit["current_pool_rematch"]["attempted"] == [0, 1]
    assert audit["recovered"] == [0, 1]
    assert audit["deferred"] == audit["deferred_retriable"] == []
    assert audit["still_unresolved"] == []


@pytest.mark.parametrize(
    "expected_visual,narration,expected_query",
    [
        ("Olenna removes the necklace stone", "", "Olenna removes the necklace stone"),
        ("", "Who does this belong to?", "Who does this belong to?"),
    ],
    ids=["expected-visual-only", "narration-only"],
)
def test_recovery_discovery_receives_effective_query_on_disposable_segment_copy(
        tmp_path, expected_visual, narration, expected_query):
    proj, segs, _old = _fixture(tmp_path, GOOD)
    original = segs[0]
    original.scene_query = ""
    original.required_entity = ""
    original.expected_visual = expected_visual
    original.text = narration
    captured = []

    def capture_discovery(_analysis, _cfg, *, segments, progress, extra_queries,
                          required_queries):
        captured.extend(segments)
        assert extra_queries == required_queries == [expected_query]
        return []

    still_blocked = {"status": "blocked", "blocked_count": 1,
                     "blockers": [{"segment_index": 0}]}
    with mock.patch.object(O, "match_segments"), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value=still_blocked), \
            mock.patch("vidlore.clipstudio.discover.discover_sources",
                       side_effect=capture_discovery):
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-effective-query.json")

    assert recovered == 0
    assert len(captured) == 1
    assert captured[0] is not original
    assert captured[0].scene_query == expected_query
    assert original.scene_query == ""
    assert proj.segments[0] is original
    assert original.expected_visual == expected_visual
    assert original.text == narration
    audit = json.loads((proj.output_dir / "strict-effective-query.json").read_text())
    assert audit["attempts"][0]["queries"] == [expected_query]


def test_only_all_four_empty_recovery_fields_are_non_retrievable(tmp_path):
    proj, segs, _sel = _fixture(tmp_path, GOOD)
    seg = segs[0]
    seg.scene_query = seg.required_entity = seg.expected_visual = seg.text = ""

    with mock.patch.object(O, "match_segments") as matcher:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-truly-empty.json")

    assert recovered == 0
    matcher.assert_not_called()
    audit = json.loads((proj.output_dir / "strict-truly-empty.json").read_text())
    assert audit["page_completed"] is True
    assert audit["page_scope"] == []
    assert audit["deferred"] == [0]
    assert audit["deferred_retriable"] == []


def test_current_pool_recovery_clears_a_real_timed_quote_window_contract(tmp_path):
    """Exercise the real quote contract, not a mocked blocker list.

    The old window has a visually positive verdict but no quote audio. The same indexed source has
    the exact line later; rematching to that timed span must be accepted, cut, and audited.
    """
    from vidlore.clipstudio import relevance_contract as R

    proj = ClipProject(name="quote-window", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"show source")
    proj.sources = [SourceVideo(
        id="s1", url="u", title="Game of Thrones Chaos is a Ladder scene",
        permission="owner", status="ok", local_path=str(media))]
    frames = []
    shots = []
    for idx, (start, end) in enumerate(((0.0, 2.0), (10.0, 12.0))):
        frame = tmp_path / f"shot_{idx}.jpg"
        frame.write_bytes(f"frame-{idx}".encode())
        frames.append(frame)
        shots.append(Shot(source_id="s1", index=idx, start=start, end=end,
                          keyframe_path=str(frame)))
    proj.shots_path("s1").write_text(json.dumps([s.to_dict() for s in shots]))
    quote = "Chaos isn't a pit. Chaos is a ladder."
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [10.2 + i * .12, 10.3 + i * .12, word] for i, word in enumerate(words)
    ]))
    _stamp_current_asr_provenance(proj, "s1")
    seg = ScriptSegment(
        index=0, text="Which is why genius is the wrong word.", quote=quote,
        scene_query="Game of Thrones Littlefinger chaos is a ladder speech Varys",
        expected_visual="Littlefinger delivers the chaos is a ladder speech to Varys",
        required_entity="Petyr Baelish", required_kind="character",
        visual_policy=P.EXACT, is_specific_claim=True)
    old = ClipSelection(
        segment_index=0, source_id="s1", shot_index=0, in_point=0.0, out_point=2.0,
        confidence=0.9, signals={"dialogue": 0.0}, verifier=dict(GOOD))
    V.bind_selection_verifier_evidence(
        proj, old, seg, old.verifier, shot=shots[0], model="vision", is_specific=True,
        multiframe=True, faceid_names=[], era="", must_see="")
    proj.segments = [seg]
    proj.selections = [old]
    initial = R.evaluate_selection_relevance(proj, [seg])
    assert initial["blockers"][0]["reasons"] == [
        "exact_quote_dialogue_signal_below_floor"]

    def rematch(_proj, _segs, _cfg, **_kw):
        new = ClipSelection(
            segment_index=0, source_id="s1", shot_index=1, in_point=10.0, out_point=12.0,
            confidence=0.95, signals={"dialogue": 1.0, "moment_lock": 1.0},
            verifier=dict(GOOD))
        V.bind_selection_verifier_evidence(
            _proj, new, seg, new.verifier, shot=shots[1], model="vision",
            is_specific=True, multiframe=True, faceid_names=[], era="", must_see="")
        _proj.selections = [new]
        return _proj.selections

    def cut_one(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(b"quote-window-clip")
        return path

    with mock.patch.object(O, "match_segments", side_effect=rematch), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=cut_one), \
            mock.patch("vidlore.clipstudio.discover.discover_sources") as discover:
        recovered = O._recover_unresolved_beats(
            proj, [seg], SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="strict-real-quote.json")

    assert recovered == 1
    discover.assert_not_called()
    assert proj.selections[0].in_point == 10.0
    final = R.evaluate_selection_relevance(proj, [seg])
    assert final["status"] == "pass"
    assert final["quote_branch_counts"] == {
        "verbatim": 1, "paraphrase": 0, "indeterminate": 0}


def test_quote_window_builder_contains_full_phrase_and_skips_sd_equal_ratio(tmp_path):
    """Whole-pool existence evidence becomes a bounded HD window, not an arbitrary SD pick."""
    from vidlore.clipstudio import relevance_contract as R

    proj, segs, old = _fixture(tmp_path, GOOD)
    seg = segs[0]
    seg.quote = "Chaos isn't a pit. Chaos is a ladder."
    seg.est_duration = 1.5
    old.flagged = True
    old.flag_reasons = ["verifier_failed", "exact_scene_missing"]
    old.in_point, old.out_point = 0.0, 2.0
    hd_media = tmp_path / "quote_hd.mp4"
    hd_media.write_bytes(b"hd quote bytes")
    hd_alt_media = tmp_path / "quote_hd_alt.mp4"
    hd_alt_media.write_bytes(b"hd recap bytes")
    proj.sources.append(SourceVideo(
        id="quote_hd", url="u-hd",
        title="Game of Thrones Olenna Sansa necklace stone chaos is a ladder",
        permission="owner", status="ok", local_path=str(hd_media), duration=30.0))
    proj.sources.append(SourceVideo(
        id="quote_hd_alt", url="u-hd-alt", title="generic recap",
        permission="owner", status="ok", local_path=str(hd_alt_media), duration=30.0))
    for sid in ("s1", "quote_hd", "quote_hd_alt"):
        frame = tmp_path / f"{sid}_quote.jpg"
        frame.write_bytes(b"quote frame")
        shot = Shot(source_id=sid, index=4, start=8.0, end=18.0,
                    keyframe_path=str(frame), quality=(0.6 if sid == "quote_hd_alt" else 0.8))
        proj.shots_path(sid).write_text(json.dumps([shot.to_dict()]))

    branch = {
        "authored_quote": seg.quote, "branch": "verbatim", "verbatim_required": True,
        "pool_match": {"source_id": "s1", "source_title": "SD first",
                       "timed_asr_span": [10.0, 15.2], "timed_asr_ratio": 1.0},
        "pool_matches": [
            {"source_id": "s1", "source_title": "SD first",
             "timed_asr_span": [10.0, 15.2], "timed_asr_ratio": 1.0},
                {"source_id": "quote_hd", "source_title": "HD second",
                 "timed_asr_span": [10.0, 15.2], "timed_asr_ratio": 1.0},
                {"source_id": "quote_hd_alt", "source_title": "HD recap",
                 "timed_asr_span": [10.0, 15.2], "timed_asr_ratio": 1.0},
        ],
    }

    def dimensions(path):
        return ({"width": 640, "height": 360} if path.name == "source.mp4"
                else {"width": 1920, "height": 1080})

    with mock.patch.object(R, "_quote_pool_branches", return_value={0: branch}), \
            mock.patch("vidlore.clipstudio.ingest.probe", side_effect=dimensions):
        built, audit = O._quote_window_recovery_selections(
            proj, segs, ClipConfig(), {0})

    assert set(built) == {0}
    picked = built[0]
    assert picked.source_id == "quote_hd"
    assert picked.in_point <= 10.0 and picked.out_point >= 15.2
    assert picked.duration >= 5.2 - 1e-6, "a short narration beat must not truncate the real quote"
    assert picked.flagged is False
    assert picked.flag_reasons == []
    assert picked.beat_windows == [["quote_hd", 10.0, 15.2]]
    assert len(picked.alternates) == 1
    assert picked.alternates[0].source_id == "quote_hd_alt"
    rows = audit["beats"][0]["candidates"]
    assert next(row for row in rows if row["source_id"] == "s1")["status"] == \
        "skipped_non_hd_or_unprobeable"
    assert next(row for row in rows if row["source_id"] == "quote_hd")["status"] == \
        "candidate"


def test_quote_window_rung_commits_before_broad_rematch_and_keeps_page_receipt(tmp_path):
    proj, segs, old = _fixture(tmp_path, GOOD)
    old.flagged = True
    old.flag_reasons = ["verifier_failed", "exact_scene_missing"]
    direct = copy.deepcopy(old)
    direct.in_point, direct.out_point = 10.0, 15.2
    direct.clip_path = ""
    direct.flagged = False
    direct.flag_reasons = []
    direct.verifier = {}
    direct.beat_windows = [["s1", 10.0, 15.2]]
    direct.signals = {"dialogue": 1.0, "moment_lock": 1.0, "quote_pool_exact": True}
    rung_audit = {"attempted": [0], "recovered": [], "still_unresolved": [0],
                  "error": "", "beats": [{"segment_index": 0, "status": "candidate_ready"}]}

    def cut_one(_proj, sel, _cfg, **_kw):
        path = _proj.clips_dir / f"seg_{sel.segment_index:03d}.mp4"
        path.write_bytes(b"fresh quote-window clip")
        return path

    with mock.patch.object(O, "_quote_window_recovery_selections",
                           return_value=({0: direct}, rung_audit)), \
            mock.patch("vidlore.clipstudio.verify.verify_and_repair",
                       return_value=dict(VERIFY_OK)), \
            mock.patch("vidlore.clipstudio.relevance_contract.evaluate_selection_relevance",
                       return_value={"status": "pass", "blocked_count": 0, "blockers": []}), \
            mock.patch("vidlore.clipstudio.cut.cut_selection", side_effect=cut_one), \
            mock.patch.object(O, "match_segments") as broad, \
            mock.patch("vidlore.clipstudio.discover.discover_sources") as discover:
        recovered = O._recover_unresolved_beats(
            proj, segs, SimpleNamespace(movie_title="Game of Thrones"), ClipConfig(),
            SimpleNamespace(anthropic_model="vision"), faceid_obj=None, refs={}, roster=[],
            policy="approved_testing", log=lambda _m: None, only_indices={0},
            audit_filename="quote-rung-page.json", audit_request_id="quote-page")

    assert recovered == 1
    broad.assert_not_called()
    discover.assert_not_called()
    assert proj.selections[0].in_point == 10.0
    assert proj.selections[0].flagged is False
    assert proj.selections[0].flag_reasons == []
    assert proj.selections[0].beat_windows == [["s1", 10.0, 15.2]]
    page = json.loads((proj.output_dir / "quote-rung-page.json").read_text())
    assert page["current_pool_quote_windows"]["recovered"] == [0]
    assert page["current_pool_rematch"]["attempted"] == page["page_scope"] == [0]
    assert page["current_pool_rematch"]["broad_attempted"] == []
    assert page["page_completed"] is True


def test_orchestrate_runs_strict_recovery_then_confirmed_gap_ladder_then_asserts():
    src = inspect.getsource(O.produce_auto)
    shared = src[src.index("def _recover_selection_relevance_or_raise"):
                 src.index("# STRICT SEMANTIC PREFLIGHT")]
    assert "_retry_selection_relevance(" in shared
    assert "_assert_sr_recovered(" in shared
    assert src.index("_assert_sr_preflight(") < src.index(
        "preassemble_release_block_reason")
    build_catch = src[src.index("except RuntimeError as _be"):
                      src.index("_pm_stage.write_report")]
    assert '== "selection_relevance"' not in build_catch
    assert "_recover_selection_relevance_or_raise" not in build_catch

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


def _bind_schema2_exhausted_gap(proj, segs, idx, tmp_path):
    """Bind the existing total-absence proof for orchestration-scope tests."""
    from vidlore.clipstudio import selfheal as S

    seg = next(s for s in segs if s.index == idx)
    pool_fp, pool_n = S._gap_absence_pool_fingerprint(proj)
    evidence = tmp_path / f"beat-{idx}-strict-exhaustion.json"
    evidence.write_text(json.dumps({
        "schema_version": 2,
        "status": "complete",
        "pool_scope": S._GAP_ABSENCE_POOL_SCOPE,
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
        "beats": {str(idx): {
            "beat_fingerprint": S._gap_beat_fingerprint(seg),
            "classification": "footage_gap",
            "actual_frame_pool_audit": True,
            "whole_pool_reviewed": True,
            "correct_footage_present_in_pool": False,
            "pipeline_bug_ruled_out": True,
            "strict_acquisition_status": "exhausted",
            "technical_status": "complete",
        }},
    }, sort_keys=True))
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [idx], method="actual_frame_and_pool_audit",
        source=str(evidence), strict_acquisition_exhausted_beats=[idx])
    return evidence


def test_validated_exhausted_gap_skips_duplicate_acquisition_and_exact_image_fallback(
        tmp_path):
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    _bind_schema2_exhausted_gap(proj, segs, 0, tmp_path)

    def settle_gap(_proj, targets, *_args, **_kwargs):
        seg = targets[0]
        seg.visual_policy = P.ABSTRACT
        seg.required_entity = ""
        seg.required_kind = ""
        seg.quote = ""
        seg.scene_query = ""
        seg.is_specific_claim = False
        return {"candidate_count": 1, "softened_count": 1}

    with mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps",
                              side_effect=settle_gap) as ladder:
        final = _call(proj, segs)

    assert final["status"] == "pass"
    recover.assert_not_called()
    images.assert_not_called()
    ladder.assert_called_once()


def test_new_bound_gap_review_runs_ladder_after_same_generation_was_already_exhausted(
        tmp_path):
    """A completed strict marker predates operator review and must not deadlock that review."""
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    _bind_schema2_exhausted_gap(proj, segs, 0, tmp_path)
    current = R.evaluate_selection_relevance(proj, segs)
    assert current["blocked_count"] == 1
    proj.meta["selection_relevance_recovery"] = {
        "schema_version": R.SCHEMA_VERSION,
        "before": [0],
        "after": [0],
        "post_fingerprint": O._selection_relevance_retry_fingerprint(proj, segs, current),
        "deferred": [],
        "gap_softening": {},
    }

    result_payload = {"candidate_count": 1, "softened_count": 1,
                      "beats": [{"segment_index": 0, "status": "softened"}]}

    def settle_gap(_proj, targets, *_args, **_kwargs):
        seg = targets[0]
        seg.visual_policy = P.ABSTRACT
        seg.required_entity = ""
        seg.required_kind = ""
        seg.quote = ""
        seg.scene_query = ""
        seg.is_specific_claim = False
        return result_payload

    with mock.patch.object(O, "_recover_unresolved_beats") as recover, \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps",
                              side_effect=settle_gap) as ladder:
        final = _call(proj, segs)

    assert final["status"] == "pass"
    recover.assert_not_called()
    images.assert_not_called()
    ladder.assert_called_once()
    assert proj.meta["selection_relevance_recovery"]["gap_softening"] == result_payload


def test_pool_growth_revalidates_exhausted_gap_and_returns_it_to_strict_recovery(
        tmp_path):
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    _append_bound_rejected_beat(proj, segs, index=1)
    _bind_schema2_exhausted_gap(proj, segs, 0, tmp_path)
    scopes = []

    def grow_pool(_proj, _segs, _analysis, _cfg, _eng, **kw):
        scopes.append(sorted(kw["only_indices"]))
        assert scopes[-1] == [1], "the bound exhausted gap must skip this duplicate page"
        media = tmp_path / "new-pool-source.mp4"
        media.write_bytes(b"new searchable pool bytes")
        frame = tmp_path / "new-pool-frame.jpg"
        frame.write_bytes(b"new indexed frame")
        _proj.sources.append(SourceVideo(
            id="new_pool", url="new", title="Game of Thrones new exact scene",
            permission="owner", status="ok", local_path=str(media)))
        _proj.shots_path("new_pool").write_text(json.dumps([Shot(
            source_id="new_pool", index=0, start=0.0, end=2.0,
            keyframe_path=str(frame)).to_dict()]))
        (_proj.index_dir / "new_pool.words.json").write_text("[]")
        _write_mock_recovery_page(_proj, kw)
        return 0

    with mock.patch.object(O, "_recover_unresolved_beats", side_effect=grow_pool), \
            mock.patch.object(O, "_fill_image_fallbacks") as images, \
            mock.patch.object(S, "heal_selection_relevance_gaps") as ladder:
        final = _call(proj, segs)

    assert final["status"] == "blocked" and final["blocked_count"] == 2
    assert scopes == [[1]]
    assert proj.meta["selection_relevance_recovery"]["deferred"] == [0]
    images.assert_not_called()
    ladder.assert_not_called(), "a changed pool invalidates the gap proof before softening"


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


def test_character_verbatim_quote_and_missing_branch_both_fail_closed(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, _sel = _fixture(tmp_path, bad)
    seg = segs[0]
    seg.visual_policy = P.CHARACTER
    seg.quote = "Chaos isn't a pit. Chaos is a ladder."
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word] for i, word in enumerate(words)
    ]))
    # The beat contract changed after the fixture bound its verifier; rebind the current negative
    # so the only reason for denial is quote typing, not stale selection evidence.
    V.bind_selection_verifier_evidence(
        proj, _sel, seg, _sel.verifier, model="vision", is_specific=True,
        multiframe=True, faceid_names=[], era="", must_see="")
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")

    audit = R.evaluate_selection_relevance(proj, segs)
    assert audit["blockers"][0]["quote_evidence"]["branch"] == "verbatim"
    assert S.semantic_gap_candidates(proj, audit)[0] == []

    missing = copy.deepcopy(audit)
    missing["blockers"][0]["quote_evidence"] = {}
    assert S.semantic_gap_candidates(proj, missing)[0] == []


def test_phase2_denies_mixed_or_unknown_evidence_reasons(tmp_path):
    from vidlore.clipstudio import selfheal as S

    proj, segs, _sel = _fixture(tmp_path, GOOD)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    for reasons in (
        ["verdict_absent", "exact_verifier_evidence_not_strict"],
        ["verdict_replace", "future_evidence_schema_fault"],
    ):
        audit = {"blockers": [{
            "segment_index": 0, "reasons": reasons, "quote_evidence": {}}]}
        assert S.semantic_gap_candidates(proj, audit)[0] == []


@pytest.mark.parametrize("contextual", [True, False], ids=["contextual", "generic"])
def test_phase2_bound_review_admits_completed_deliberate_exact_downgrade(
        tmp_path, contextual):
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    proj, segs, sel = _fixture(tmp_path, GOOD)
    _install_completed_exact_downgrade(proj, segs[0], sel, contextual=contextual)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")
    audit = R.evaluate_selection_relevance(proj, segs)

    assert R.completed_deliberate_exact_downgrade(audit["blockers"][0]) is True
    assert S.semantic_gap_candidates(proj, audit) == ([0], "confirmed_actual_frame_audit")


def test_phase2_confirmed_paraphrase_with_only_semantic_negatives_remains_eligible(tmp_path):
    from vidlore.clipstudio import relevance_contract as R
    from vidlore.clipstudio import selfheal as S

    bad = {**GOOD, "verdict": "replace", "matches_narration": False,
           "specific_enough": False, "correct_subject_visible": False}
    proj, segs, sel = _fixture(tmp_path, bad)
    seg = segs[0]
    seg.quote = "The essayist's paraphrase is not spoken dialogue."
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1, 0.2, "completely"], [0.3, 0.4, "unrelated"],
    ]))
    V.bind_selection_verifier_evidence(
        proj, sel, seg, sel.verifier, model="vision", is_specific=True,
        multiframe=True, faceid_names=[], era="", must_see="")
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, segs, [0], method="actual_frame_and_pool_audit")

    audit = R.evaluate_selection_relevance(proj, segs)

    assert audit["blockers"][0]["quote_evidence"]["branch"] == "paraphrase"
    assert set(audit["blockers"][0]["reasons"]) <= set(S._SEMANTIC_NEGATIVE_REASONS)
    assert S.semantic_gap_candidates(proj, audit) == ([0], "confirmed_actual_frame_audit")


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
        mock.patch.object(IF._wi, "search_images", side_effect=[[], [candidate], [candidate]]),
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
            seg, analysis, tmp_path / "empty", eng_cfg=SimpleNamespace(anthropic_model="vision"),
            raise_on_technical=True) is None
        assert IF.fetch_scene_image(
            seg, analysis, tmp_path / "bad", eng_cfg=SimpleNamespace(anthropic_model="vision"),
            raise_on_technical=True) is None
        good = IF.fetch_scene_image(
            seg, analysis, tmp_path / "good", eng_cfg=SimpleNamespace(anthropic_model="vision"))

    assert good and good["strict_verifier"]["matches_narration"] is True
    assert good["image_sha256"]
    assert calls and all(c["is_specific"] is True and c["venue_fallback"] is False for c in calls)
