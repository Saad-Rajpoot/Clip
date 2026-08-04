"""When the exact moment is not in reach, settle for the right subject — do not fail the video.

The owner's standing rule for this pipeline, in their own words: if no exact scene is available,
do not insist on it, make do with a similar one. Until now nothing implemented the second half.
An exact_scene beat whose moment could not be found kept demanding it through every fallback and
then release-blocked the whole render.

Job 6a26707939 is the measured case. Four beats survived acquisition and blocked the video; three
of them (24, 82, 149) describe the Bolton flayed-man banner coming down at Winterfell. The frame is
IN the pool — game_of_thrones_jon_sn_123ebf87 shot 64 is the banners lying in the snow — and CLIP
cannot retrieve it: within its own 67-shot file it ranks 48th, 33rd and 40th on those three
queries, while the Stark-banner shot beside it ranks 1st on all three. Demanding the exact moment
there is demanding something retrieval cannot deliver at any bench depth.

Softening is the LAST thing tried, only on a beat that would otherwise block the render, and it is
labelled honestly: the resulting still is a `contextual_fallback`, never `exact`.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import index as IX
from vidlore.clipstudio import selfheal as S
from vidlore.clipstudio.config import ClipConfig, load_clip_config
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.models import (
    ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo,
)


class Seg:
    def __init__(self, **kw):
        self.index = kw.pop("index", 24)
        self.text = kw.pop("text", "The flayed man banners brought down to the ground.")
        self.quote = kw.pop("quote", "")
        self.expected_visual = kw.pop("expected_visual", "Bolton banners falling at Winterfell")
        self.scene_query = kw.pop("scene_query", "Game of Thrones Bolton banner falls Winterfell")
        self.required_kind = kw.pop("required_kind", "object")
        self.required_entity = kw.pop("required_entity", "flayed man banner")
        self.visual_policy = kw.pop("visual_policy", "exact_scene")
        self.is_specific_claim = kw.pop("is_specific_claim", True)


def _phase1_fixture(tmp_path, *segments):
    index_dir = tmp_path / "index"
    output_dir = tmp_path / "output"
    index_dir.mkdir()
    output_dir.mkdir()
    proj = NS(
        root=str(tmp_path), index_dir=index_dir, output_dir=output_dir,
        meta={}, sources=[], segments=list(segments),
        selections=[NS(segment_index=s.index, image_path="", image_meta={})
                    for s in segments])
    proj.shots_path = lambda sid: index_dir / f"{sid}.shots.json"
    return proj


def _stub_phase1_search(monkeypatch):
    """Leave only the policy-mutation decision live; no model/network/disk recovery work."""
    from vidlore.clipstudio import config as C
    monkeypatch.setattr(C, "engine_config", lambda: NS())
    monkeypatch.setattr(S, "_clean_pool", lambda _proj: [])
    monkeypatch.setattr(S, "still_recover", lambda *a, **k: False)
    monkeypatch.setattr(S, "_venue_cache_save", lambda _proj: None)


def _phase1_evidence_fixture(tmp_path, *, index=2, quote="", words=None,
                             verdict_patch=None, cfg=None, evidence_is_specific=True):
    """A bound phase-1 review plus recomputable current-window verifier evidence."""
    proj = ClipProject(name="phase1-evidence", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "show.mp4"
    media.write_bytes(b"show-footage")
    source = SourceVideo(
        id="show", url="u", title="Game of Thrones scene", permission="owner",
        status="ok", local_path=str(media))
    proj.sources = [source]
    shots = []
    for shot_index in range(2):
        frame = tmp_path / f"shot_{shot_index}.jpg"
        frame.write_bytes(f"frame-{shot_index}".encode())
        shots.append(Shot(
            source_id="show", index=shot_index, start=float(shot_index * 2),
            end=float(shot_index * 2 + 2), keyframe_path=str(frame), transcript=""))
    proj.shots_path("show").write_text(json.dumps([shot.to_dict() for shot in shots]))
    if words is not None:
        (proj.index_dir / "show.words.json").write_text(json.dumps(words))

    seg = ScriptSegment(
        index=index, text="The requested exact event happens.", quote=quote,
        expected_visual="The exact requested event", scene_query="Game of Thrones exact event",
        required_entity="Ned Stark", required_kind="character",
        visual_policy=P.EXACT, is_specific_claim=True)
    verdict = {
        "status": "ok", "verdict": "replace", "matches_narration": False,
        "specific_enough": False, "correct_subject_visible": False,
        "wrong_subject_visible": False, "contradicts_narration": False,
        "quality_ok": True, "era_ok": True,
    }
    verdict.update(verdict_patch or {})
    sel = ClipSelection(
        segment_index=index, source_id="show", shot_index=0, in_point=0.0,
        out_point=2.0, confidence=0.8, verifier=verdict,
        flag_reasons=["verifier_failed"])
    proj.meta["analysis"] = {"video_type": "multi_scene", "characters": [], "actors": []}
    if words is not None:
        (proj.index_dir / "show.index.meta.json").write_text(json.dumps({
            "schema": IX.INDEX_SCHEMA,
            "words": True,
            "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(
                proj, cfg or load_clip_config()),
        }))
    V.bind_selection_verifier_evidence(
        proj, sel, seg, verdict, shot=shots[0], model="vision-test",
        is_specific=evidence_is_specific,
        multiframe=True, faceid_names=[], era=V._project_beat_era(proj, seg),
        must_see=P.deictic_target(seg))
    proj.segments = [seg]
    proj.selections = [sel]
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg], [index], method="actual_frame_and_pool_audit")
    return proj, seg, sel


def _valid_keep(**patch):
    verdict = {
        "verdict": "keep", "confidence": 1.0,
        "matches_narration": True, "specific_enough": True, "quality_ok": True,
        "wrong_subject_visible": False, "contradicts_narration": False,
        "correct_subject_visible": True,
    }
    verdict.update(patch)
    return verdict


def test_an_unreachable_exact_beat_is_softened_to_the_right_subject():
    seg = Seg()
    assert P.policy_of(seg) == P.EXACT
    assert S._soften_to_character(seg, lambda *a: None) is True
    assert P.policy_of(seg) == P.CHARACTER


def test_an_unfindable_PROP_requirement_is_dropped():
    """Softening the policy alone does nothing when the requirement IS the unfindable thing. Every
    still candidate is checked against required_entity, so beats 82 and 149 ('flayed man banner',
    'Bolton banners') and 160 ('tent') kept failing at the looser policy exactly as they had at the
    strict one — measured, all three survived the first softening pass and blocked the render."""
    seg = Seg()
    S._soften_to_character(seg, lambda *a: None)
    assert seg.required_entity == ""
    assert seg.required_kind == ""


@pytest.mark.parametrize("kind", ["character", "actor"])
def test_a_PERSON_requirement_is_never_dropped(kind):
    """"Any Melisandre shot" is an honest fallback; "any shot at all" on a beat about a person is
    how a wrong-character leak gets in, and that is the one class the identity gate exists to stop."""
    seg = Seg(required_kind=kind, required_entity="Melisandre")
    S._soften_to_character(seg, lambda *a: None)
    assert seg.required_entity == "Melisandre"
    assert seg.required_kind == kind


def test_the_beat_keeps_pointing_at_show_footage_not_an_abstract_effect():
    seg = Seg()
    S._soften_to_character(seg, lambda *a: None)
    assert seg.visual_policy == P.CHARACTER
    assert P.policy_of(seg) != P.ABSTRACT, \
        "scene-describing narration must still earn a specific label after the requirement is dropped"


def test_a_beat_that_is_not_exact_is_left_alone():
    for pol in ("character_specific", "generic_filler", "abstract_effect"):
        seg = Seg(visual_policy=pol, is_specific_claim=False, quote="", text="filler line")
        assert S._soften_to_character(seg, lambda *a: None) is False


def test_softening_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN", "0")
    assert S._soften_to_character(Seg(), lambda *a: None) is False


def test_it_says_out_loud_what_it_gave_up_on():
    """A silent downgrade is how a pipeline loses relevance without anyone noticing."""
    lines = []
    S._soften_to_character(Seg(), lines.append)
    assert lines and "exact" in lines[0].lower()
    assert "24" in lines[0], "the log must name the beat"


# ------------------------------------------------------------------ where it sits in the ladder
def test_softening_is_the_LAST_thing_tried():
    """It must run after the normal still search, after acquisition and after region-frames — the
    point is to stop asking, not to skip the honest attempts.  The only earlier call is guarded by
    a current audit proving those strict acquisition attempts already completed for this pool."""
    src = inspect.getsource(S.heal_blocked_beats)
    still_calls = [i for i in range(len(src)) if src.startswith("still_recover", i)]
    i_still = still_calls[1]  # ordinary strict search; [0] is the gate-forbidden special case
    i_acq = src.index("fresh = acquire_for_beat")
    i_region = src.index("_region_frames_recover")
    i_ordinary_soft = src.rindex("_soften_and_retry")
    reviewed_guards = [
        i for i in range(len(src))
        if src.startswith("_phase1_reviewed_exhaustion_authorization", i)
    ]
    assert i_ordinary_soft > i_still and i_ordinary_soft > i_acq \
        and i_ordinary_soft > i_region
    assert len(reviewed_guards) == 2
    assert reviewed_guards[0] < i_still < reviewed_guards[1] < i_acq


def test_the_retry_does_not_re_acquire_or_rebuild_the_pool():
    """Softening is about lowering the demand, not spending more. A second acquisition round here
    would double the cost of every blocked beat for no new footage."""
    src = inspect.getsource(S._soften_and_retry)
    assert "acquire_for_beat" not in src
    assert "_clean_pool" not in src


def test_the_softened_search_looks_deeper_than_the_exact_one():
    src = inspect.getsource(S._soften_and_retry)
    assert "SELFHEAL_SOFT_CANDS" in src


@pytest.mark.parametrize("path", ["with acquisition", "without acquisition"])
def test_both_failure_paths_reach_the_softening(path):
    """A beat can arrive at 'unresolved' with acquisition disabled or with acquisition exhausted;
    neither may fall through to a release block without trying the looser bar."""
    src = inspect.getsource(S.heal_blocked_beats)
    unresolved = [i for i in range(len(src)) if src.startswith("unresolved this pass", i)]
    assert len(unresolved) >= 2, "both give-up sites must exist"
    for u in unresolved:
        head = src[:u]
        assert "_soften_and_retry" in head, "every give-up site must be preceded by the retry"


def test_the_still_is_labelled_a_contextual_fallback_not_an_exact_hit():
    """Honest accounting: the audit must still say the exact moment was never found."""
    src = inspect.getsource(S._soften_to_character)
    assert "contextual_fallback" in src, \
        "the docstring must state the class so nobody later mistakes this for an exact match"


# ------------------------------------------------------------------ the second rung
def test_the_ladder_falls_through_to_abstract_when_the_narration_itself_is_unfillable():
    """Dropping the requirement is not enough when the NARRATION names the unfindable thing. The
    still verifier judges candidates against the beat's sentence, so beat 24 — "The flayed man
    banners brought down to the ground" — cannot be satisfied by any frame in the pool however loose
    the policy is. Measured: it stayed unresolved through the character rung and blocked the render
    a third time. `_soften_to_abstract` is the pipeline's own designed escape for that case."""
    src = inspect.getsource(S._soften_and_retry)
    i_char = src.index("_soften_to_character")
    i_abs = src.index("_soften_to_abstract")
    assert i_abs > i_char, "abstract is the LAST rung, never the first"
    assert src.count("still_recover") == 2, "each rung must actually search again"


def test_the_abstract_rung_is_only_reached_after_the_character_rung_fails():
    """`and` short-circuits: a beat the character rung fills must never be rewritten as visual rest."""
    src = inspect.getsource(S._soften_and_retry)
    head = src[:src.index("_soften_to_abstract")]
    assert "return True" in head, "a successful character rung must return before the abstract one"


def test_invalid_existing_still_cannot_short_circuit_recovery(monkeypatch, tmp_path):
    from types import SimpleNamespace as NS

    stale = tmp_path / "stale.jpg"
    stale.write_bytes(b"contextual pixels")
    sel = NS(segment_index=24, image_path=str(stale), image_meta={
        "source": "source-frame-recovery", "relevance_class": "contextual_fallback",
        "still_verified": True})
    seg = Seg()
    proj = NS()
    monkeypatch.setattr(S, "_clean_pool", lambda _p: [])
    # With no candidates, success is impossible; the point is that the legacy path is detached
    # instead of being treated as a resolved exact beat merely because the file exists.
    assert S.still_recover(
        proj, seg, sel, NS(), pool=[], used_paths=set(), log=lambda _m: None) is False
    assert sel.image_path == "" and sel.image_meta == {}


def test_failed_abstract_search_restores_the_original_strict_request(monkeypatch):
    seg = Seg()
    sel = NS(segment_index=24, image_path="", image_meta={})
    monkeypatch.setattr(S, "still_recover", lambda *a, **k: False)
    before = (seg.visual_policy, seg.required_entity, seg.required_kind,
              seg.expected_visual, seg.scene_query, seg.quote)
    assert S._soften_and_retry(NS(), seg, sel, NS(), [], set(), lambda _m: None) is False
    after = (seg.visual_policy, seg.required_entity, seg.required_kind,
             seg.expected_visual, seg.scene_query, seg.quote)
    assert after == before


def test_unjudged_ladder_candidate_raises_and_restores_full_phase1_state(
        monkeypatch, tmp_path):
    """``verify_frame=None`` is infrastructure uncertainty, never evidence of a footage gap."""
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    before_used = set(used)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    # The venue pass installs a real candidate; the strict publication-strength recheck receives no
    # answer. This exercises selection + used-path rollback, not merely segment-field rollback.
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: _valid_keep())
    monkeypatch.setattr(V, "verify_frame", lambda *_a, **_k: None)

    with pytest.raises(S.InconclusiveStillVerificationError, match="returned no verdict"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert used == before_used


def test_venue_error_keep_is_inconclusive_and_cannot_soften(monkeypatch, tmp_path):
    """A stale/default keep inside an error wrapper is not a venue judgment."""
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: {
        "status": "error", "verdict": "keep", "confidence": 1.0,
    })

    with pytest.raises(S.InconclusiveStillVerificationError, match="status.*error"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert used == {"already-used.jpg"}
    assert "selection_relevance_gap_softening" not in proj.meta


def test_venue_malformed_keep_is_inconclusive_and_cannot_soften(monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    malformed = _valid_keep()
    malformed.pop("matches_narration")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: malformed)

    with pytest.raises(S.InconclusiveStillVerificationError,
                       match="missing/malformed.*matches_narration"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg and vars(sel) == before_sel
    assert used == {"already-used.jpg"}
    assert "selection_relevance_gap_softening" not in proj.meta


def test_strict_confirm_error_keep_is_not_laundered_and_cannot_soften(
        monkeypatch, tmp_path):
    """The publication-strength recheck must inspect status before writing still evidence."""
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: _valid_keep())
    monkeypatch.setattr(V, "verify_frame", lambda *_a, **_k: {
        "status": "error", "verdict": "keep", "confidence": 1.0,
        "matches_narration": True, "specific_enough": True, "quality_ok": True,
        "wrong_subject_visible": False, "contradicts_narration": False,
        "correct_subject_visible": True,
    })

    with pytest.raises(S.InconclusiveStillVerificationError, match="status.*error"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert used == {"already-used.jpg"}
    assert "selection_relevance_gap_softening" not in proj.meta


def test_strict_confirm_malformed_keep_is_inconclusive_and_cannot_soften(
        monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    malformed = _valid_keep()
    malformed.pop("contradicts_narration")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: _valid_keep())
    monkeypatch.setattr(V, "verify_frame", lambda *_a, **_k: malformed)

    with pytest.raises(S.InconclusiveStillVerificationError,
                       match="missing/malformed.*contradicts_narration"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg and vars(sel) == before_sel
    assert used == {"already-used.jpg"}
    assert "selection_relevance_gap_softening" not in proj.meta


@pytest.mark.parametrize("outcome", ["explicit_replace", "no_candidates"])
def test_conclusive_ladder_exhaustion_returns_false_and_restores(outcome, monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj) if outcome == "explicit_replace" else []
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify",
                        lambda *_a, **_k: {"verdict": "replace", "reason": "wrong scene"})

    assert S._soften_and_retry(
        proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
        lambda _m: None) is False

    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert used == {"already-used.jpg"}


def test_phase1_acquisition_technical_error_propagates_without_softening_marker(
        monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    _stub_phase1_search(monkeypatch)
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    soften_calls = []

    def fail_acquire(*_args, **_kwargs):
        # Prove the per-beat transaction restores incidental mutations too, not just the policy.
        seg.visual_policy = P.ABSTRACT
        sel.image_path = "partial.jpg"
        raise S.InconclusiveAcquisitionError(seg.index, "download", detail="network")

    monkeypatch.setattr(S, "acquire_for_beat", fail_acquire)
    monkeypatch.setattr(S, "_soften_and_retry",
                        lambda *_a, **_k: soften_calls.append(True) or True)

    with pytest.raises(S.InconclusiveAcquisitionError, match="download"):
        S.heal_blocked_beats(
            proj, [seg], None, blocked=[seg.index], policy="approved_testing",
            allow_acquire=True, log=lambda _m: None)

    assert vars(seg) == before_seg and vars(sel) == before_sel
    assert soften_calls == []
    assert "selection_relevance_gap_softening" not in proj.meta
    assert not (proj.output_dir / "semantic_gap_softening_audit.json").exists()


# ------------------------------------------------------- phase-1 authorization (101-beat incident)
def test_phase1_never_softens_an_unreviewed_beat_24(monkeypatch, tmp_path):
    """The structural predictor surfaced beat 24 even though the viewer's confirmed-gap set did not.
    Phase 1 must keep searching strictly, then leave it blocked; it has no authority to rewrite it."""
    seg = Seg(index=24)
    proj = _phase1_fixture(tmp_path, seg)
    pool_fp, pool_n = S._gap_pool_fingerprint(proj)
    proj.meta["selection_relevance_gap_review"] = {
        "schema_version": 2,
        "confirmed_gap_beats": [2],
        "beat_fingerprints": {"2": "different reviewed beat"},
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
    }
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: True)
    mutation_calls = []
    monkeypatch.setattr(S, "_soften_to_abstract",
                        lambda *a, **k: mutation_calls.append("abstract"))
    monkeypatch.setattr(S, "_soften_and_retry",
                        lambda *a, **k: mutation_calls.append("ladder") or True)
    before = copy.deepcopy(vars(seg))
    lines = []

    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=False, log=lines.append)

    assert resolved == 0 and mutation_calls == []
    assert vars(seg) == before
    assert any("specificity softening DENIED" in line
               and "beat_not_confirmed_as_footage_gap" in line for line in lines)


def test_phase1_never_softens_reviewed_beats_after_the_pool_changes(monkeypatch, tmp_path):
    """Beats 2/80 were reviewed against 87 sources, then phase 1 ran against 102. A review of the
    old pool is not evidence that the newly expanded pool still lacks the requested scene."""
    seg2, seg80 = Seg(index=2), Seg(index=80)
    proj = _phase1_fixture(tmp_path, seg2, seg80)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg2, seg80], [2, 80], method="actual_frame_and_pool_audit")
    late_media = tmp_path / "late_source.mp4"
    late_media.write_bytes(b"new pool bytes")
    proj.sources.append(NS(
        id="late_source", status="ok", checksum="late-checksum",
        local_path=str(late_media)))
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: True)
    mutation_calls = []
    monkeypatch.setattr(S, "_soften_to_abstract",
                        lambda *a, **k: mutation_calls.append("abstract"))
    monkeypatch.setattr(S, "_soften_and_retry",
                        lambda *a, **k: mutation_calls.append("ladder") or True)
    before = [copy.deepcopy(vars(seg2)), copy.deepcopy(vars(seg80))]
    lines = []

    resolved = S.heal_blocked_beats(
        proj, [seg2, seg80], None, blocked=[2, 80], policy="approved_testing",
        allow_acquire=False, log=lines.append)

    assert resolved == 0 and mutation_calls == []
    assert [vars(seg2), vars(seg80)] == before
    denials = [line for line in lines if "stale_gap_review_source_pool_changed" in line]
    assert len(denials) == 2 and all("strict authored policy preserved" in x for x in denials)


def test_phase1_ladder_runs_when_beat_and_current_pool_are_bound(monkeypatch, tmp_path):
    """The repair is an authorization gate, not a global softening kill switch."""
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, index=2)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    called = []

    def authorized_ladder(_proj, target, *_args, **_kwargs):
        called.append(target.index)
        target.visual_policy = "abstract_effect"
        return True

    monkeypatch.setattr(S, "_soften_and_retry", authorized_ladder)
    lines = []
    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[2], policy="approved_testing",
        allow_acquire=False, log=lines.append)

    assert resolved == 1 and called == [2]
    assert seg.visual_policy == "abstract_effect"
    assert not any("specificity softening DENIED" in line for line in lines)
    marker = proj.meta["selection_relevance_gap_softening"]
    assert marker["schema_version"] == S._SOFTENING_SCHEMA
    assert marker["active"] is True
    row = marker["beats"][0]
    assert row["phase"] == "phase1_preassembly"
    assert row["pool_fingerprint"] == S._gap_pool_fingerprint(proj)[0]
    assert row["original"]["visual_policy"] == P.EXACT
    assert row["original"]["expected_visual"] == "The exact requested event"
    assert row["original"]["is_specific_claim"] is True
    assert row["original"]["image_path"] == ""
    assert row["original"]["image_meta"] == {}


def test_phase1_uses_active_custom_asr_cfg_for_quote_typing(monkeypatch, tmp_path):
    """Custom-model evidence stays current instead of becoming an indeterminate quote blocker."""
    custom_cfg = ClipConfig(whisper_model="small.en", whisper_compute="float16")
    words = [[0.1 + i * .1, 0.2 + i * .1, word]
             for i, word in enumerate("An unrelated real line from the show".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="The essayist paraphrases this moment.", words=words,
        cfg=custom_cfg)
    assert IX.asr_semantic_fingerprint(proj, custom_cfg) != \
        IX.asr_semantic_fingerprint(
            proj, ClipConfig(whisper_model="base", whisper_compute="int8"))
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    softened = []

    def ladder(_proj, target, *_args, **_kwargs):
        softened.append(target.index)
        target.visual_policy = P.ABSTRACT
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    lines = []
    resolved = S.heal_blocked_beats(
        proj, [seg], custom_cfg, blocked=[seg.index], policy="approved_testing",
        allow_acquire=False, log=lines.append)

    assert resolved == 1 and softened == [seg.index]
    assert not any("phase1_quote_pool_classification_indeterminate" in line for line in lines)


def _write_completed_exhaustion_evidence(
        proj, seg, *, source="actual-frame-pool-audit.json", beat_patch=None, top_patch=None):
    evidence = Path(source)
    if not evidence.is_absolute():
        evidence = Path(proj.root) / evidence
    pool_fp, pool_n = S._gap_absence_pool_fingerprint(proj)
    beat = {
        "beat_fingerprint": S._gap_beat_fingerprint(seg),
        "classification": "footage_gap",
        "actual_frame_pool_audit": True,
        "whole_pool_reviewed": True,
        "correct_footage_present_in_pool": False,
        "pipeline_bug_ruled_out": True,
        "strict_acquisition_status": "exhausted",
        "technical_status": "complete",
    }
    beat.update(beat_patch or {})
    artifact = {
        "schema_version": 2, "status": "complete",
        "pool_scope": S._GAP_ABSENCE_POOL_SCOPE,
        "pool_fingerprint": pool_fp, "pool_source_count": pool_n,
        "beats": {str(seg.index): beat},
    }
    artifact.update(top_patch or {})
    evidence.write_text(json.dumps(artifact, sort_keys=True))
    return evidence, artifact


def _bind_completed_exhaustion(proj, seg, *, source="actual-frame-pool-audit.json"):
    evidence, _artifact = _write_completed_exhaustion_evidence(
        proj, seg, source=source)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg], [seg.index], method="actual_frame_and_pool_audit",
        source=str(evidence), strict_acquisition_exhausted_beats=[seg.index])
    return evidence


@pytest.mark.parametrize(("top_patch", "reason"), [
    ({"schema_version": 1}, "schema_not_2"),
    ({"pool_scope": ""}, "pool_scope_not_all_source_ok_indexed"),
])
def test_legacy_or_unscoped_exhaustion_evidence_cannot_be_reinterpreted(
        top_patch, reason, tmp_path):
    """Schema 1 cryptographically covered only the eligible pool; it cannot silently inherit the
    broader all-indexed meaning that makes matcher-only bans safe to ignore."""
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, index=24)
    evidence, _artifact = _write_completed_exhaustion_evidence(
        proj, seg, top_patch=top_patch)

    with pytest.raises(ValueError, match=reason):
        S.make_selection_relevance_gap_review(
            proj, [seg], [seg.index], method="actual_frame_and_pool_audit",
            source=str(evidence), strict_acquisition_exhausted_beats=[seg.index])


def test_bound_completed_exhaustion_runs_ladder_before_acquire(monkeypatch, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, index=24)
    _bind_completed_exhaustion(proj, seg)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    acquired = []
    monkeypatch.setattr(
        S, "acquire_for_beat",
        lambda *_a, **_k: acquired.append(True) or pytest.fail("must not reacquire"))

    def ladder(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        selection.image_meta = {"selfheal_rung": "abstract"}
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)

    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=True, log=lambda _m: None)

    assert resolved == 1 and acquired == []
    marker = proj.meta["selection_relevance_gap_softening"]
    row = next(r for r in marker["beats"] if r["status"] == "softened")
    assert row["basis"] == "phase1_bound_strict_acquisition_exhausted"
    assert row["original"]["visual_policy"] == P.EXACT
    assert row["new"]["visual_policy"] == P.ABSTRACT


def test_match_auto_ban_does_not_stale_completed_whole_indexed_exhaustion(
        monkeypatch, tmp_path):
    """Reproduces the 101-beat run: match re-derived one extra source-level rejection after the
    actual-frame audit, shrinking the publishable pool 102→101 before beat 24 reached self-heal.

    A ban cannot make missing footage appear, so absence evidence is bound to every indexed OK
    source.  The active softening remains bound to the narrower publishable pool, where a later ban
    can still restore it if its installed source becomes ineligible.
    """
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, index=24)
    aux_media = tmp_path / "auxiliary.mp4"
    aux_media.write_bytes(b"already-reviewed auxiliary footage")
    proj.sources.append(SourceVideo(
        id="auxiliary", url="aux", title="Already reviewed scene", permission="owner",
        status="ok", local_path=str(aux_media)))
    proj.shots_path("auxiliary").write_text(json.dumps([]))
    _bind_completed_exhaustion(proj, seg)
    review_fp = proj.meta["selection_relevance_gap_review"]["pool_fingerprint"]
    publishable_before = S._gap_pool_fingerprint(proj)[0]

    # This is the matcher mutation that made the real run's valid review stale before acquisition.
    proj.meta["auto_rejected_sources"] = ["auxiliary"]
    assert S._gap_pool_fingerprint(proj)[0] != publishable_before
    assert S._gap_pool_fingerprint(proj)[0] != review_fp
    assert S._gap_absence_pool_fingerprint(proj)[0] == \
        proj.meta["selection_relevance_gap_review"]["absence_pool_fingerprint"]
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    acquired = []
    monkeypatch.setattr(
        S, "acquire_for_beat",
        lambda *_a, **_k: acquired.append(True) or pytest.fail("must not reacquire"))

    def ladder(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        selection.image_meta = {"selfheal_rung": "abstract"}
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=True, log=lambda _m: None)

    assert resolved == 1 and acquired == []
    row = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
               if r["status"] == "softened")
    assert row["basis"] == "phase1_bound_strict_acquisition_exhausted"
    assert row["pool_fingerprint"] == S._gap_pool_fingerprint(proj)[0]


def test_reviewed_exhaustion_is_consumed_before_an_ordinary_beat_mutates_pool(
        monkeypatch, tmp_path):
    """The scale path can report several blockers in ordinary beat order.  A preceding unreviewed
    beat must not add a source and stale a later completed whole-pool audit before its ladder runs.
    """
    proj, reviewed, _reviewed_sel = _phase1_evidence_fixture(tmp_path, index=24)
    _bind_completed_exhaustion(proj, reviewed)
    ordinary = ScriptSegment(
        index=7, text="An ordinary unresolved event.", quote="",
        expected_visual="An ordinary exact event", scene_query="ordinary exact scene",
        required_entity="Arya Stark", required_kind="character",
        visual_policy=P.EXACT, is_specific_claim=True)
    ordinary_sel = ClipSelection(
        segment_index=7, source_id="show", shot_index=1, in_point=2.0,
        out_point=4.0, confidence=0.4, verifier={"status": "ok", "verdict": "replace"},
        flag_reasons=["verifier_failed"])
    proj.segments = [ordinary, reviewed]
    proj.selections.append(ordinary_sel)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    events = []

    def ladder(_proj, target, selection, *_args, **_kwargs):
        events.append(f"ladder:{target.index}")
        target.visual_policy = P.ABSTRACT
        selection.image_meta = {"selfheal_rung": "abstract"}
        return True

    def acquire(_proj, target, *_args, **_kwargs):
        events.append(f"acquire:{target.index}")
        late_media = tmp_path / "late-acquired.mp4"
        late_media.write_bytes(b"newly indexed source bytes")
        _proj.sources.append(SourceVideo(
            id="late-acquired", url="late", title="New scene", permission="owner",
            status="ok", local_path=str(late_media)))
        _proj.shots_path("late-acquired").write_text(json.dumps([]))
        return []

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    monkeypatch.setattr(S, "acquire_for_beat", acquire)

    resolved = S.heal_blocked_beats(
        proj, [ordinary, reviewed], None, blocked=[7, 24],
        policy="approved_testing", allow_acquire=True, log=lambda _m: None)

    assert resolved == 1
    assert events[:2] == ["ladder:24", "acquire:7"]
    row = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
               if r["status"] == "softened")
    assert row["segment_index"] == 24
    assert row["pool_fingerprint"] != S._gap_pool_fingerprint(proj)[0]


def test_exhausted_review_ladder_failure_restores_and_never_reacquires(
        monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path, index=24)
    _bind_completed_exhaustion(proj, seg)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    before_seg, before_sel = copy.deepcopy(vars(seg)), copy.deepcopy(vars(sel))
    acquired = []
    monkeypatch.setattr(S, "acquire_for_beat",
                        lambda *_a, **_k: acquired.append(True) or [])

    def exhausted(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        selection.image_path = "partial.jpg"
        return False

    monkeypatch.setattr(S, "_soften_and_retry", exhausted)

    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=True, log=lambda _m: None)

    assert resolved == 0 and acquired == []
    assert vars(seg) == before_seg and vars(sel) == before_sel
    assert "selection_relevance_gap_softening" not in proj.meta


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("classification", "pipeline_bug", "classification_not_footage_gap"),
    ("actual_frame_pool_audit", False, "actual_frame_pool_audit_absent"),
    ("whole_pool_reviewed", False, "whole_pool_review_absent"),
    ("correct_footage_present_in_pool", True, "pool_absence_not_proven"),
    ("pipeline_bug_ruled_out", False, "pipeline_bug_not_ruled_out"),
    ("strict_acquisition_status", "incomplete", "status_not_exhausted"),
    ("technical_status", "download_error", "technical_status_not_complete"),
    ("evidence_source", "", "evidence_source_missing"),
    ("evidence_sha256", "", "evidence_sha256_missing_or_malformed"),
    ("evidence_sha256", "not-a-hash", "evidence_sha256_missing_or_malformed"),
])
def test_unsafe_exhaustion_record_never_gets_early_authority(
        field, value, reason, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    _bind_completed_exhaustion(proj, seg)
    proj.meta["selection_relevance_gap_review"]["strict_acquisition_exhaustion"][
        str(seg.index)][field] = value

    ok, why = S._phase1_reviewed_exhaustion_authorization(proj, seg)

    assert ok is False and reason in why


def test_stale_completed_exhaustion_cannot_soften(monkeypatch, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    _bind_completed_exhaustion(proj, seg)
    late_media = tmp_path / "late-source.mp4"
    late_media.write_bytes(b"new searchable pool bytes")
    proj.sources.append(SourceVideo(
        id="late", url="late", title="Late exact scene", permission="owner",
        status="ok", local_path=str(late_media)))

    ok, why = S._phase1_reviewed_exhaustion_authorization(proj, seg)

    assert ok is False and why == "strict_acquisition_evidence_source_pool_changed"


def test_changed_retrieval_embeddings_stale_completed_exhaustion(tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    embeds = proj.embeds_path("show")
    embeds.write_bytes(b"original-searchable-clip-matrix")
    _bind_completed_exhaustion(proj, seg)

    # The source, shots, words, and authored beat are unchanged; only the retrieval matrix changed.
    # A whole-pool exhaustion decision made against the old ranking world must no longer authorize
    # skipping strict acquisition.
    embeds.write_bytes(b"replacement-searchable-clip-matrix-with-different-size")
    ok, why = S._phase1_reviewed_exhaustion_authorization(proj, seg)

    assert ok is False and why == "strict_acquisition_evidence_source_pool_changed"


def test_tampered_exhaustion_evidence_artifact_cannot_soften(tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    _bind_completed_exhaustion(proj, seg)
    record = proj.meta["selection_relevance_gap_review"]["strict_acquisition_exhaustion"][
        str(seg.index)]
    Path(record["evidence_source"]).write_text("tampered after review binding")

    ok, why = S._phase1_reviewed_exhaustion_authorization(proj, seg)

    assert ok is False and why == "strict_acquisition_evidence_artifact_hash_mismatch"


def test_old_exhaustion_artifact_cannot_be_rebound_to_a_later_pool(tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    evidence, _artifact = _write_completed_exhaustion_evidence(proj, seg)
    late_media = tmp_path / "late-source-before-review.mp4"
    late_media.write_bytes(b"new searchable pool bytes")
    proj.sources.append(SourceVideo(
        id="late-before-review", url="late", title="Late exact scene", permission="owner",
        status="ok", local_path=str(late_media)))

    with pytest.raises(ValueError, match="evidence_source_pool_changed"):
        S.make_selection_relevance_gap_review(
            proj, [seg], [seg.index], method="actual_frame_and_pool_audit",
            source=str(evidence), strict_acquisition_exhausted_beats=[seg.index])


def test_completed_exhaustion_cannot_soften_a_verbatim_quote(tmp_path):
    quote = "Chaos isn't a pit. Chaos is a ladder."
    words = [[0.1 + i * .1, 0.2 + i * .1, word]
             for i, word in enumerate("Chaos isn't a pit Chaos is a ladder".split())]
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, quote=quote, words=words)
    _bind_completed_exhaustion(proj, seg)

    ok, why = S._phase1_reviewed_exhaustion_authorization(proj, seg)

    assert ok is False and why == "phase1_verbatim_quote_promise"


def test_exhaustion_declaration_requires_confirmed_beat_and_evidence_source(tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path)
    with pytest.raises(ValueError, match="subset"):
        S.make_selection_relevance_gap_review(
            proj, [seg], [], method="audit", source="audit.json",
            strict_acquisition_exhausted_beats=[seg.index])
    with pytest.raises(ValueError, match="evidence source"):
        S.make_selection_relevance_gap_review(
            proj, [seg], [seg.index], method="audit",
            strict_acquisition_exhausted_beats=[seg.index])
    with pytest.raises(ValueError, match="existing file"):
        S.make_selection_relevance_gap_review(
            proj, [seg], [seg.index], method="audit", source=str(tmp_path / "missing.json"),
            strict_acquisition_exhausted_beats=[seg.index])


def test_beat80_pipeline_or_technical_evidence_cannot_skip_acquisition(
        monkeypatch, tmp_path):
    """The 101-beat incident's honest gaps must not become authority for its pipeline bugs.

    Even a correctly hashed file is negative evidence when its own authoritative beat row says the
    pool contains a pipeline/technical failure.  The maker rejects it, and hand-binding the same row
    cannot bypass the runtime validator: normal acquisition runs and its technical error propagates.
    """
    proj, seg, sel = _phase1_evidence_fixture(tmp_path, index=80)
    evidence, artifact = _write_completed_exhaustion_evidence(
        proj, seg, source="beat80-pipeline-bug.json", beat_patch={
            "classification": "pipeline_bug",
            "pipeline_bug_ruled_out": False,
            "strict_acquisition_status": "incomplete",
            "technical_status": "download_error",
        })
    with pytest.raises(ValueError, match="classification_not_footage_gap"):
        S.make_selection_relevance_gap_review(
            proj, [seg], [80], method="actual_frame_and_pool_audit",
            source=str(evidence), strict_acquisition_exhausted_beats=[80])

    review = S.make_selection_relevance_gap_review(
        proj, [seg], [80], method="actual_frame_and_pool_audit", source=str(evidence))
    review["strict_acquisition_exhaustion"] = {
        "80": {
            **artifact["beats"]["80"],
            "evidence_source": str(evidence.resolve()),
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    }
    proj.meta["selection_relevance_gap_review"] = review
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    soften_calls = []
    monkeypatch.setattr(
        S, "_soften_and_retry", lambda *_a, **_k: soften_calls.append(True) or True)

    def fail_acquire(*_args, **_kwargs):
        raise S.InconclusiveAcquisitionError(80, "download", detail="network")

    monkeypatch.setattr(S, "acquire_for_beat", fail_acquire)
    before_seg, before_sel = copy.deepcopy(vars(seg)), copy.deepcopy(vars(sel))
    before_meta = copy.deepcopy(proj.meta)

    with pytest.raises(S.InconclusiveAcquisitionError, match="download"):
        S.heal_blocked_beats(
            proj, [seg], None, blocked=[80], policy="approved_testing",
            allow_acquire=True, log=lambda _m: None)

    assert soften_calls == []
    assert vars(seg) == before_seg and vars(sel) == before_sel
    assert proj.meta == before_meta
    assert "selection_relevance_gap_softening" not in proj.meta


def test_phase1_unfillable_mutation_is_recorded_and_failed_attempt_restores(
        monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path, index=2)
    from vidlore.clipstudio import config as C
    monkeypatch.setattr(C, "engine_config", lambda: NS(anthropic_model="vision-test"))
    monkeypatch.setattr(S, "_clean_pool", lambda _proj: [])
    monkeypatch.setattr(S, "_venue_cache_save", lambda _proj: None)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: True)
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))

    # First pass: neither the dedicated abstract rung nor the final ladder has a candidate. Both
    # mutations must roll all the way back and no durable marker may claim success.
    monkeypatch.setattr(S, "still_recover", lambda *a, **k: False)
    assert S.heal_blocked_beats(
        proj, [seg], None, blocked=[2], policy="approved_testing",
        allow_acquire=False, log=lambda _m: None) == 0
    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert "selection_relevance_gap_softening" not in proj.meta

    # Second pass: the gate-forbidden abstract rung installs a candidate. This separate policy
    # mutation site must carry the same complete, pool-bound reversible record as the main ladder.
    def abstract_only_recovery(_proj, target, selection, *_args, **_kwargs):
        if P.policy_of(target) != P.ABSTRACT:
            return False
        selection.image_path = str(tmp_path / "abstract.jpg")
        selection.image_meta = {"installed": True}
        return True

    monkeypatch.setattr(S, "still_recover", abstract_only_recovery)
    assert S.heal_blocked_beats(
        proj, [seg], None, blocked=[2], policy="approved_testing",
        allow_acquire=False, log=lambda _m: None) == 1
    marker = proj.meta["selection_relevance_gap_softening"]
    row = next(r for r in marker["beats"] if r["status"] == "softened")
    assert row["basis"] == "phase1_gate_forbidden_content"
    assert row["original"]["visual_policy"] == P.EXACT
    assert row["original"]["expected_visual"] == before_seg["expected_visual"]
    assert row["original"]["is_specific_claim"] is True
    assert row["original"]["image_meta"] == before_sel["image_meta"]


def _run_phase1_authorization(monkeypatch, proj, seg):
    """Run the real authorization while stubbing only footage search and policy mutation."""
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)
    called = []

    def ladder(_proj, target, *_args, **_kwargs):
        called.append(target.index)
        target.visual_policy = P.ABSTRACT
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    lines = []
    resolved = S.heal_blocked_beats(
        proj, [seg], None, blocked=[seg.index], policy="approved_testing",
        allow_acquire=False, log=lines.append)
    return resolved, called, lines


def test_phase1_bound_review_cannot_soften_a_verbatim_quote(monkeypatch, tmp_path):
    quote = "Chaos isn't a pit. Chaos is a ladder."
    words = [
        [10.0 + i * .15, 10.1 + i * .15, word]
        for i, word in enumerate("Chaos isn't a pit Chaos is a ladder".split())
    ]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote=quote, words=words)

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("phase1_verbatim_quote_promise" in line for line in lines)


def test_phase1_bound_review_cannot_soften_an_indeterminate_quote(monkeypatch, tmp_path):
    # No timed-ASR index means the pool cannot establish whether this authored phrase is dialogue.
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="Chaos isn't a pit. Chaos is a ladder.", words=None)

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("phase1_quote_pool_classification_indeterminate" in line for line in lines)


def test_phase1_bound_review_cannot_soften_a_technical_verifier_fault(monkeypatch, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, verdict_patch={"status": "error"})

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("phase1_technical_or_evidence_blocker:verifier_error" in line
               for line in lines)


def test_phase1_bound_review_cannot_soften_stale_selection_evidence(monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    sel.verifier["selection_evidence"]["fingerprint"] = "stale-window-proof"

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("phase1_technical_or_evidence_blocker:verifier_evidence_mismatch" in line
               for line in lines)


def test_phase1_bound_paraphrase_with_pure_content_negative_can_still_soften(
        monkeypatch, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="The essayist's paraphrase is not spoken dialogue.",
        words=[[0.1, 0.2, "completely"], [0.3, 0.4, "unrelated"]])

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 1 and called == [seg.index]
    assert P.policy_of(seg) == P.ABSTRACT
    assert not any("specificity softening DENIED" in line for line in lines)


@pytest.mark.parametrize("relevance_class,downgraded", [
    ("contextual_fallback", "exact→contextual"),
    ("generic_filler", "exact→generic_filler"),
])
def test_phase1_bound_review_admits_completed_deliberate_exact_downgrade(
        monkeypatch, tmp_path, relevance_class, downgraded):
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path,
        verdict_patch={
            "status": "ok", "verdict": "keep", "matches_narration": True,
            "specific_enough": True, "correct_subject_visible": True,
            "wrong_subject_visible": False, "contradicts_narration": False,
            "quality_ok": True, "era_ok": True, "downgraded": downgraded,
            "relevance_class": relevance_class,
        },
        evidence_is_specific=False,
    )

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 1 and called == [seg.index]
    assert P.policy_of(seg) == P.ABSTRACT
    assert not any("specificity softening DENIED" in line for line in lines)
