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
    # Authored quotes are decoder hints and therefore part of the current ASR fingerprint.
    proj.segments = [seg]
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


def _bind_current_quote_retrieval_review(proj, seg, words, *, cfg=None):
    """Make a quoted fixture satisfy the separate retrieval-generation precondition."""
    active_cfg = cfg or load_clip_config()
    assert IX._save_quote_retrieval_words(proj, proj.sources[0], active_cfg, words)
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg], [seg.index], method="actual_frame_and_pool_audit", cfg=active_cfg)


def _quote_binding_refresh_fixture(monkeypatch, tmp_path):
    """A bound paraphrase review whose only current change is the global retrieval generation."""
    seg = Seg(
        index=77, quote="I'm taking you to the docks.", required_kind="character",
        required_entity="Jaime Lannister")
    proj = _phase1_fixture(tmp_path, seg)
    old_generation, new_generation = "1" * 64, "2" * 64

    def pool_binding(generation):
        return {
            "generation_fingerprint": generation,
            "eligible_source_count": 0,
            "sidecar_pool_fingerprint": ("3" if generation == old_generation else "4") * 64,
            "complete": True,
            "invalid_sources": [],
        }

    def branch_binding(generation):
        quote = " ".join(seg.quote.strip().split())
        return {
            "binding_schema_version": 1,
            "authored_quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            "branch": "paraphrase",
            "branch_reason": "no_qualifying_timed_asr_phrase_match",
            "quote_retrieval_fingerprint_expected": generation,
            "confirmation_generation_fingerprint": "5" * 64,
            "confirmation_attempts_shape_valid": True,
            "confirmation_attempt_count": 0,
            "confirmation_confirmed_count": 0,
            "confirmation_rejected_count": 0,
            "confirmation_inconclusive_count": 0,
            "retrieval_truncated_stream_count": 0,
            "confirmation_artifact_keys": [],
            "confirmation_result_identities": [],
            "confirmation_result_fingerprint": "6" * 64,
        }

    state = {
        "pool": pool_binding(old_generation),
        "branch": branch_binding(old_generation),
    }
    monkeypatch.setattr(
        S, "_quote_retrieval_pool_binding",
        lambda *_args, **_kwargs: copy.deepcopy(state["pool"]))
    monkeypatch.setattr(
        S, "_quote_review_branch_bindings",
        lambda *_args, **_kwargs: {str(seg.index): copy.deepcopy(state["branch"])})
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg], [seg.index], method="actual_frame_and_pool_audit")
    state.update({
        "pool": pool_binding(new_generation),
        "branch": branch_binding(new_generation),
    })
    return proj, seg, state, old_generation, new_generation


def test_gap_review_refreshes_only_unrelated_global_quote_generation(
        monkeypatch, tmp_path):
    proj, seg, _state, old_generation, new_generation = \
        _quote_binding_refresh_fixture(monkeypatch, tmp_path)
    before = copy.deepcopy(proj.meta["selection_relevance_gap_review"])

    result = S.refresh_selection_relevance_gap_review_quote_bindings(proj)

    assert result == {
        "refreshed": True,
        "reason": "unrelated_global_quote_retrieval_generation_changed",
        "previous_generation_fingerprint": old_generation,
        "current_generation_fingerprint": new_generation,
        "reviewed_beats": [seg.index],
    }
    after = proj.meta["selection_relevance_gap_review"]
    assert after["quote_retrieval_binding"]["generation_fingerprint"] == new_generation
    assert after["quote_branch_bindings"][str(seg.index)][
        "quote_retrieval_fingerprint_expected"] == new_generation
    assert after["quote_binding_refresh"]["previous_generation_fingerprint"] == old_generation
    for field in (
            "schema_version", "method", "source", "confirmed_gap_beats",
            "beat_fingerprints", "pool_fingerprint", "pool_source_count",
            "absence_pool_scope", "absence_pool_fingerprint", "absence_pool_source_count"):
        assert after[field] == before[field], f"refresh must not rewrite reviewed fact {field}"
    assert S._gap_review_quote_binding_reason(proj, seg, after) == ""


@pytest.mark.parametrize("changed", [
    "beat", "pool", "verbatim", "indeterminate", "result", "incomplete", "strict",
])
def test_gap_review_quote_refresh_fails_closed_on_any_real_fact_change(
        changed, monkeypatch, tmp_path):
    proj, seg, state, old_generation, _new_generation = \
        _quote_binding_refresh_fixture(monkeypatch, tmp_path)
    before = copy.deepcopy(proj.meta["selection_relevance_gap_review"])
    if changed == "beat":
        seg.text += " Changed after the actual-frame review."
    elif changed == "pool":
        proj.sources.append(SourceVideo(
            id="new-source", url="u", title="new indexed source", permission="owner",
            status="ok", local_path=str(tmp_path / "new-source.mp4")))
    elif changed in ("verbatim", "indeterminate"):
        state["branch"]["branch"] = changed
    elif changed == "result":
        state["branch"]["confirmation_result_fingerprint"] = "9" * 64
    elif changed == "incomplete":
        state["pool"]["complete"] = False
        state["pool"]["invalid_sources"] = [
            {"source_id": "show", "reason": "sidecar_missing"}]
    elif changed == "strict":
        proj.meta["selection_relevance_gap_review"]["strict_acquisition_exhaustion"] = {
            str(seg.index): {"evidence_sha256": "7" * 64}}
        before = copy.deepcopy(proj.meta["selection_relevance_gap_review"])

    result = S.refresh_selection_relevance_gap_review_quote_bindings(proj)

    assert result["refreshed"] is False
    assert proj.meta["selection_relevance_gap_review"] == before
    assert proj.meta["selection_relevance_gap_review"][
        "quote_retrieval_binding"]["generation_fingerprint"] == old_generation


def _patch_confirmed_quote(monkeypatch, proj, seg, *, cfg=None):
    from vidlore.clipstudio import relevance_contract as R

    decoder = R._quote_confirmation_decoder_fingerprint(cfg or load_clip_config())
    key = hashlib.sha256(f"{seg.index}\x1f{seg.quote}".encode()).hexdigest()

    def confirmed(_proj, _src, _quote, prompted_span, _cfg, **_kwargs):
        return {
            "schema_version": R.QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": R.QUOTE_CONFIRMATION_ALGORITHM,
            "status": "confirmed", "artifact_key": key,
            "decoder_fingerprint": decoder,
            "prompted_span": list(prompted_span),
            "confirmed_span": list(prompted_span),
            "timed_asr_ratio": float(prompted_span[2]),
            "match_method": "fuzzy_phrase_timed_asr+unprompted_confirmation",
        }

    monkeypatch.setattr(R, "_confirm_prompted_quote_span_unprompted", confirmed)


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
        or src.startswith("_phase1_reviewed_exhaustion_details", i)
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
    from vidlore.clipstudio import build as B

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
    monkeypatch.setattr(
        B, "_rescue_still_fullres",
        lambda *_a, **_k: (_ for _ in ()).throw(
            V.VisionBackendError("native still verifier returned no verdict", kind="down")))

    with pytest.raises(S.InconclusiveStillVerificationError, match="returned no verdict"):
        S._soften_and_retry(
            proj, seg, sel, NS(anthropic_model="vision-test"), pool, used,
            lambda _m: None)

    assert vars(seg) == before_seg
    assert vars(sel) == before_sel
    assert used == before_used


@pytest.mark.parametrize(("policy", "expected_specific"), [
    (P.EXACT, True),
    (P.CHARACTER, False),
])
def test_concrete_still_recheck_asks_the_current_authorized_policy(
        policy, expected_specific, monkeypatch, tmp_path):
    """The character rung must not silently re-demand the missing exact moment it surrendered.

    This is the 101-beat footage-gap shape: beat 69 can use a clean related ship frame and beat 87
    can use a clean Ned frame only after the bound review authorizes EXACT -> CHARACTER.  The actual
    image still has to affirm narration, subject, quality, and non-contradiction and is hash-bound;
    only the verifier's specificity question follows the now-current policy.
    """
    frame = tmp_path / "related-show-frame.jpg"
    frame.write_bytes(b"real related show pixels")
    seg = Seg(
        visual_policy=policy, is_specific_claim=(policy == P.EXACT),
        required_kind="character", required_entity="Ned Stark")
    sel = NS(
        segment_index=seg.index, image_path=str(frame),
        image_meta={"source": "source-frame-recovery"})
    proj = NS(meta={"analysis": {"episode_hint": ""}})
    asked = []

    def verify(*_args, **kwargs):
        asked.append(kwargs.get("is_specific"))
        return _valid_keep(status="ok")

    monkeypatch.setattr(V, "verify_frame", verify)
    ok, why = S._strictly_confirm_concrete_still(
        proj, seg, sel, NS(anthropic_model="vision-test"), require_conclusive=True)

    assert (ok, why) == (True, "")
    assert asked == [expected_specific]
    assert sel.image_meta["still_semantic_verified"] is True
    assert sel.image_meta["still_image_sha256"] == hashlib.sha256(frame.read_bytes()).hexdigest()
    assert sel.image_meta["relevance_class"] == (
        "exact_scene" if expected_specific else "contextual_fallback")


def test_owned_thumbnail_is_materialized_and_verified_on_native_pixels(
        monkeypatch, tmp_path):
    """The ladder's 512px search image is only a locator, never the pixels authorized to air."""
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import relevance_contract as R

    thumbnail = tmp_path / "jaime-thumb.jpg"
    thumbnail.write_bytes(b"indexed search thumbnail")
    native = tmp_path / "jaime-native.jpg"
    native.write_bytes(b"freshly extracted native source pixels")
    native_hash = hashlib.sha256(native.read_bytes()).hexdigest()
    seg = Seg(
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister")
    sel = NS(
        segment_index=seg.index, image_path=str(thumbnail),
        image_meta={
            "source": "source-frame-recovery", "src": "owner", "shot": 7,
            "relevance_class": "contextual_fallback",
        })
    proj = NS(meta={"analysis": {"episode_hint": ""}})
    expected_proj = proj
    rescue_calls = []

    def rescue(project, selection, judged_path, _log, **kwargs):
        rescue_calls.append((project, selection, judged_path, kwargs))
        return {
            "path": str(native),
            "file_sha256": native_hash,
            "semantic_strict_reason": "",
            "semantic_verifier": _valid_keep(
                status="ok", vision_served_by="vision-live"),
            "semantic_model": "vision-live",
            "semantic_question_fingerprint": "native-question-fp",
            "indexed_keyframe_sha256": "indexed-thumb-fp",
            "owner_source_content_fingerprint": "owner-content-fp",
            "owner_time": 12.0,
        }

    def coverage(selection, current_seg, *, proj=None):
        assert selection.image_path == str(native)
        assert current_seg is seg
        assert proj is expected_proj
        assert selection.image_meta["still_image_sha256"] == native_hash
        assert selection.image_meta["native_semantic_materialized"] is True
        assert selection.image_meta["native_semantic_question_fingerprint"] == \
            "native-question-fp"
        return True, "source-frame-recovery"

    monkeypatch.setattr(B, "_rescue_still_fullres", rescue)
    monkeypatch.setattr(R, "verified_still_coverage", coverage)
    monkeypatch.setattr(
        V, "verify_frame",
        lambda *_a, **_k: pytest.fail("the indexed thumbnail was judged as publication pixels"))

    ok, why = S._strictly_confirm_concrete_still(
        proj, seg, sel, NS(anthropic_model="vision-test"), require_conclusive=True)

    assert (ok, why) == (True, "")
    assert len(rescue_calls) == 1
    assert rescue_calls[0][0:3] == (proj, sel, str(thumbnail))
    assert rescue_calls[0][3]["seg"] is seg
    assert rescue_calls[0][3]["allow_semantic_reject"] is True
    assert rescue_calls[0][3]["refresh_semantic_verdict"] is True
    assert sel.image_meta["still_verifier"]["vision_served_by"] == "vision-live"
    assert sel.image_meta["exact_still_verified"] is False


def test_owned_thumbnail_native_resolution_rejection_is_a_candidate_miss(
        monkeypatch, tmp_path):
    """An SD owner rejects this candidate without weakening the native-HD publication gate."""
    from vidlore.clipstudio import build as B

    thumbnail = tmp_path / "sd-owner-thumb.jpg"
    thumbnail.write_bytes(b"indexed thumbnail")
    seg = Seg(
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister")
    sel = NS(
        segment_index=seg.index, image_path=str(thumbnail),
        image_meta={"source": "source-frame", "src": "sd-owner", "shot": 3})

    def reject(*_args, **_kwargs):
        raise V.NonRetryableBuildError(
            "source-frame owner is 640x360", kind="native_resolution")

    monkeypatch.setattr(B, "_rescue_still_fullres", reject)

    ok, why = S._strictly_confirm_concrete_still(
        NS(meta={"analysis": {}}), seg, sel, NS(anthropic_model="vision-test"),
        require_conclusive=True)

    assert ok is False
    assert "640x360" in why
    assert sel.image_path == str(thumbnail)


def test_owned_thumbnail_native_verifier_outage_remains_inconclusive(
        monkeypatch, tmp_path):
    """Backend uncertainty is not converted into either a content rejection or a softening pass."""
    from vidlore.clipstudio import build as B

    thumbnail = tmp_path / "unjudged-thumb.jpg"
    thumbnail.write_bytes(b"indexed thumbnail")
    seg = Seg(
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister")
    sel = NS(
        segment_index=seg.index, image_path=str(thumbnail),
        image_meta={"source": "source-frame-recovery", "src": "owner", "shot": 3})

    def unavailable(*_args, **_kwargs):
        raise V.VisionBackendError("vision service unavailable", kind="down")

    monkeypatch.setattr(B, "_rescue_still_fullres", unavailable)

    with pytest.raises(
            S.InconclusiveStillVerificationError,
            match="native concrete-still verification.*vision service unavailable"):
        S._strictly_confirm_concrete_still(
            NS(meta={"analysis": {}}), seg, sel, NS(anthropic_model="vision-test"),
            require_conclusive=True)


def test_owned_thumbnail_lineage_corruption_remains_nonretryable(
        monkeypatch, tmp_path):
    from vidlore.clipstudio import build as B

    thumbnail = tmp_path / "corrupt-owner-thumb.jpg"
    thumbnail.write_bytes(b"indexed thumbnail")
    seg = Seg(
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister")
    sel = NS(
        segment_index=seg.index, image_path=str(thumbnail),
        image_meta={"source": "source-frame-recovery", "src": "owner", "shot": 3})

    def corrupt(*_args, **_kwargs):
        raise V.NonRetryableBuildError(
            "indexed keyframe hash changed", kind="scene_lineage")

    monkeypatch.setattr(B, "_rescue_still_fullres", corrupt)

    with pytest.raises(V.NonRetryableBuildError, match="keyframe hash") as caught:
        S._strictly_confirm_concrete_still(
            NS(meta={"analysis": {}}), seg, sel, NS(anthropic_model="vision-test"),
            require_conclusive=True)
    assert caught.value.kind == "scene_lineage"


@pytest.mark.parametrize(("policy", "expected_venue"), [
    (P.EXACT, True),
    (P.CHARACTER, False),
])
def test_preliminary_still_question_follows_the_current_policy(
        policy, expected_venue, monkeypatch, tmp_path):
    frame = tmp_path / "candidate.jpg"
    frame.write_bytes(b"show pixels")
    seg = Seg(
        visual_policy=policy, is_specific_claim=(policy == P.EXACT),
        required_kind="character", required_entity="Jaime Lannister")
    asked = []

    def verify(*_args, **kwargs):
        asked.append(kwargs.get("venue_fallback"))
        return _valid_keep()

    monkeypatch.setattr(V, "verify_frame", verify)
    verdict = S._venue_verify(
        str(frame), seg, [], NS(anthropic_model="vision-test"), proj=None, cache=None)

    assert verdict["verdict"] == "keep"
    assert asked == [expected_venue]


def test_character_preliminary_keep_requires_the_named_subject():
    seg = Seg(
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister")
    atmosphere_only = _valid_keep(correct_subject_visible=False)

    assert S._preliminary_still_keep(
        atmosphere_only, seg, require_conclusive=True) is False
    assert S._preliminary_still_keep(
        _valid_keep(), seg, require_conclusive=True) is True


def test_character_still_ranking_drops_the_abandoned_exact_storyboard(
        monkeypatch, tmp_path):
    from vidlore.clipstudio import image_fallback as IF

    frame = tmp_path / "jaime.jpg"
    frame.write_bytes(b"show pixels")
    shot = Shot(source_id="show", index=0, start=0.0, end=2.0,
                keyframe_path=str(frame), transcript="")
    seg = Seg(
        text="That order changes Jaime's position.",
        visual_policy=P.CHARACTER, is_specific_claim=False,
        required_kind="character", required_entity="Jaime Lannister",
        scene_query="Jaime attacks Ned on horseback in the street",
        expected_visual="Jaime orders Ned's men killed during the street attack")
    sel = NS(segment_index=seg.index, image_path="", image_meta={})
    queries = []

    def relevance(_shot, _path, query, **_kwargs):
        queries.append(query)
        return 0.9

    monkeypatch.setattr(IF, "_shot_relevance", relevance)
    monkeypatch.setattr(S, "_venue_verify",
                        lambda *_a, **_k: {"verdict": "replace", "confidence": 1.0})
    proj = NS(root=str(tmp_path), index_dir=tmp_path, output_dir=tmp_path)

    assert S.still_recover(
        proj, seg, sel, NS(), pool=[("show", shot)], used_paths=set(),
        log=lambda _m: None) is False
    assert queries == ["Jaime Lannister That order changes Jaime's position."]
    assert "horseback" not in queries[0] and "street attack" not in queries[0]


def test_character_rung_continues_after_first_final_proof_rejects(monkeypatch):
    seg = Seg(required_kind="character", required_entity="Jaime Lannister")
    sel = NS(segment_index=seg.index, image_path="", image_meta={})
    used = {"already-used.jpg"}
    candidates = iter(["false-positive.jpg", "jaime.jpg"])
    recovery_calls = []

    def recover(_proj, _seg, selection, _eng, *, used_paths, **_kwargs):
        try:
            path = next(candidates)
        except StopIteration:
            return False
        recovery_calls.append(path)
        selection.image_path = path
        selection.image_meta = {"source": "source-frame-recovery"}
        used_paths.add(path)
        return True

    proofs = iter([(False, "Jaime is not identifiable"), (True, "")])
    monkeypatch.setattr(S, "still_recover", recover)
    monkeypatch.setattr(S, "_strictly_confirm_concrete_still",
                        lambda *_a, **_k: next(proofs))

    assert S._soften_and_retry(
        NS(), seg, sel, NS(), [], used, lambda _m: None) is True
    assert recovery_calls == ["false-positive.jpg", "jaime.jpg"]
    assert sel.image_path == "jaime.jpg"
    assert sel.image_meta["selfheal_rung"] == "character_specific"
    assert "false-positive.jpg" not in used
    assert "jaime.jpg" in used


def test_softening_candidate_count_excludes_inactive_history(monkeypatch):
    monkeypatch.setattr(S, "_gap_pool_fingerprint", lambda _proj: ("pool-fp", 91))
    historical = {
        "schema_version": S._SOFTENING_SCHEMA,
        "beats": [{
            "segment_index": 24,
            "status": "restored_pool_changed",
            "active": False,
        }],
    }
    current = [
        {"segment_index": 69, "status": "softened", "active": True},
        {"segment_index": 87, "status": "still_blocked", "active": False},
    ]

    payload = S._merge_softening_payload(
        NS(sources=[]), current, basis="confirmed_actual_frame_audit",
        existing=historical)

    assert payload["candidate_count"] == 2
    assert payload["history_count"] == 3
    assert payload["candidates"] == [69]
    assert payload["restored_count"] == 1


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
    from vidlore.clipstudio import build as B

    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: _valid_keep())
    monkeypatch.setattr(
        B, "_rescue_still_fullres",
        lambda *_a, **_k: (_ for _ in ()).throw(
            V.VisionBackendError("native still verifier status is error", kind="down")))

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
    from vidlore.clipstudio import build as B

    proj, seg, sel = _phase1_evidence_fixture(tmp_path)
    pool = S._clean_pool(proj)
    used = {"already-used.jpg"}
    before_seg = copy.deepcopy(vars(seg))
    before_sel = copy.deepcopy(vars(sel))
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", "1")
    monkeypatch.setattr("vidlore.clipstudio.image_fallback._shot_relevance",
                        lambda *_a, **_k: 0.9)
    monkeypatch.setattr(S, "_venue_verify", lambda *_a, **_k: _valid_keep())
    monkeypatch.setattr(
        B, "_rescue_still_fullres",
        lambda *_a, **_k: (_ for _ in ()).throw(V.VisionBackendError(
            "native still verifier missing/malformed field contradicts_narration",
            kind="down")))

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
    _bind_current_quote_retrieval_review(proj, seg, words, cfg=custom_cfg)
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


def _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg, *,
                                   source="native-hd-gap-audit.json",
                                   dimensions=(640, 360)):
    """Bind the schema-4 shape used when exact quote footage exists only in native SD."""
    from vidlore.clipstudio import quality_contract as Q
    from vidlore.clipstudio import relevance_contract as R

    src = proj.sources[0]
    src.checksum = "current-native-sd-source-checksum"
    proj.meta["auto_rejected_sources"] = [src.id]
    proj.meta["auto_rejected_reasons"] = {src.id: "sub_native_hd"}
    current_dimensions = {"width": dimensions[0], "height": dimensions[1]}
    monkeypatch.setattr(Q, "probe_native_video_info", lambda _path: dict(current_dimensions))

    # Fixture media are intentionally tiny byte sentinels, not decodable videos.  Supply the
    # independent confirmation result this test is about and retain the patch for every runtime
    # revalidation performed after the evidence artifact is bound.
    confirmation_key = hashlib.sha256(
        f"{src.id}\x1f{seg.quote}".encode("utf-8")).hexdigest()
    confirmation_decoder = R._quote_confirmation_decoder_fingerprint(load_clip_config())

    def confirmed(_proj, _src, _quote, prompted_span, _cfg, **_kwargs):
        return {
            "schema_version": R.QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": R.QUOTE_CONFIRMATION_ALGORITHM,
            "status": "confirmed",
            "artifact_key": confirmation_key,
            "decoder_fingerprint": confirmation_decoder,
            "prompted_span": list(prompted_span),
            "confirmed_span": list(prompted_span),
            "timed_asr_ratio": float(prompted_span[2]),
            "match_method": "exact_contiguous_timed_asr+unprompted_confirmation",
        }

    monkeypatch.setattr(R, "_confirm_prompted_quote_span_unprompted", confirmed)
    general_words = IX.load_words(proj, src.id)
    assert IX._save_quote_retrieval_words(
        proj, src, load_clip_config(), general_words)
    contract = R._quote_pool_branches(proj, [seg])[seg.index]
    assert contract["branch"] == "verbatim"
    matches = list(contract.get("pool_matches") or [contract["pool_match"]])
    assert [m["source_id"] for m in matches] == [src.id]
    match = matches[0]
    pool_fp, pool_n = S._gap_absence_pool_fingerprint(proj)
    evidence = Path(proj.root) / source
    beat = {
        "beat_fingerprint": S._gap_beat_fingerprint(seg),
        "classification": S._NATIVE_HD_GAP_CLASSIFICATION,
        "gap_scope": S._NATIVE_HD_GAP_CLASSIFICATION,
        "actual_frame_pool_audit": True,
        "whole_pool_reviewed": True,
        "correct_footage_present_in_pool": True,
        "correct_publishable_footage_present_in_pool": False,
        "pipeline_bug_ruled_out": True,
        "strict_acquisition_status": "exhausted",
        "technical_status": "complete",
        "publication_min_short_edge": Q.MIN_NATIVE_SHORT_EDGE,
        "publication_min_long_edge": Q.MIN_NATIVE_LONG_EDGE,
        "nonpublishable_exact_sources": [{
            "source_id": src.id,
            "source_checksum": src.checksum,
            "auto_reject_reason": "sub_native_hd",
            "native_width": dimensions[0],
            "native_height": dimensions[1],
            "actual_frame_target_verified": True,
            "timed_asr_span": match["timed_asr_span"],
            "timed_asr_ratio": match["timed_asr_ratio"],
            "quote_confirmation_artifact_key":
                match["unprompted_confirmation"]["artifact_key"],
            "quote_confirmation_decoder_fingerprint":
                match["unprompted_confirmation"]["decoder_fingerprint"],
        }],
    }
    evidence.write_text(json.dumps({
        "schema_version": S._NATIVE_HD_GAP_EVIDENCE_SCHEMA,
        "status": "complete",
        "pool_scope": S._GAP_ABSENCE_POOL_SCOPE,
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
        "beats": {str(seg.index): beat},
    }, sort_keys=True))
    proj.meta["selection_relevance_gap_review"] = S.make_selection_relevance_gap_review(
        proj, [seg], [seg.index], method="actual_frame_and_pool_audit",
        source=str(evidence), strict_acquisition_exhausted_beats=[seg.index])
    return evidence, current_dimensions


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
        selection.image_path = str(tmp_path / "shot_0.jpg")
        selection.image_meta = {
            "source": "source-frame-recovery", "src": "show", "shot": 0,
            "selfheal_rung": "abstract",
        }
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
    assert row["authorization"]["authorization_kind"] == "total_absence"
    assert row["authorization"]["strict_acquisition_exhausted"] is True
    current = S.restore_stale_selection_relevance_softenings(proj, [seg])
    assert current["restored"] == [] and current["unchanged"] == [24]
    assert seg.visual_policy == P.ABSTRACT


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
    from vidlore.clipstudio import relevance_contract as R
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert S.semantic_gap_candidates(proj, audit) == (
        [24], "current_all_indexed_strict_exhaustion_evidence")
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


def test_active_strict_exhaustion_softening_rebinds_on_publishable_only_drift(
        monkeypatch, tmp_path):
    proj, seg, _sel = _phase1_evidence_fixture(tmp_path, index=24)
    aux_media = tmp_path / "auxiliary.mp4"
    aux_media.write_bytes(b"already-reviewed auxiliary footage")
    proj.sources.append(SourceVideo(
        id="auxiliary", url="aux", title="Already reviewed scene", permission="owner",
        status="ok", local_path=str(aux_media)))
    proj.shots_path("auxiliary").write_text(json.dumps([]))
    _bind_completed_exhaustion(proj, seg)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)

    def ladder(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        selection.image_path = str(tmp_path / "shot_0.jpg")
        selection.image_meta = {
            "source": "source-frame-recovery", "src": "show", "shot": 0,
            "selfheal_rung": "abstract",
        }
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    assert S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=True, log=lambda _m: None) == 1
    old_fp = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
                  if r["status"] == "softened")["pool_fingerprint"]

    proj.meta["auto_rejected_sources"] = ["auxiliary"]
    current_fp = S._gap_pool_fingerprint(proj)[0]
    assert current_fp != old_fp
    assert S._gap_absence_pool_fingerprint(proj)[0] == \
        proj.meta["selection_relevance_gap_review"]["absence_pool_fingerprint"]

    result = S.restore_stale_selection_relevance_softenings(proj, [seg])

    assert result["restored"] == []
    assert result["unchanged"] == result["rebound"] == [24]
    assert seg.visual_policy == P.ABSTRACT
    row = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
               if r["status"] == "softened")
    assert row["pool_fingerprint"] == current_fp
    assert row["rebound_from_pool_fingerprint"] == old_fp
    assert row["rebound_reason"] == "current_all_indexed_strict_exhaustion_evidence"
    marker = proj.meta["selection_relevance_gap_softening"]
    assert marker["pool_fingerprint"] == current_fp
    assert marker["pool_source_count"] == S._gap_pool_fingerprint(proj)[1]


def test_active_strict_exhaustion_softening_restores_if_fallback_owner_is_banned(
        monkeypatch, tmp_path):
    proj, seg, sel = _phase1_evidence_fixture(tmp_path, index=24)
    aux_media = tmp_path / "auxiliary.mp4"
    aux_media.write_bytes(b"already-reviewed auxiliary footage")
    proj.sources.append(SourceVideo(
        id="auxiliary", url="aux", title="Already reviewed scene", permission="owner",
        status="ok", local_path=str(aux_media)))
    proj.shots_path("auxiliary").write_text(json.dumps([]))
    _bind_completed_exhaustion(proj, seg)
    _stub_phase1_search(monkeypatch)
    monkeypatch.setattr(S, "beat_unfillable", lambda _seg: False)

    def ladder(_proj, target, selection, *_args, **_kwargs):
        target.visual_policy = P.ABSTRACT
        selection.image_path = str(tmp_path / "shot_0.jpg")
        selection.image_meta = {
            "source": "source-frame-recovery", "src": "show", "shot": 0,
            "selfheal_rung": "abstract",
        }
        return True

    monkeypatch.setattr(S, "_soften_and_retry", ladder)
    assert S.heal_blocked_beats(
        proj, [seg], None, blocked=[24], policy="approved_testing",
        allow_acquire=True, log=lambda _m: None) == 1
    proj.meta["auto_rejected_sources"] = ["show"]
    assert S._gap_absence_pool_fingerprint(proj)[0] == \
        proj.meta["selection_relevance_gap_review"]["absence_pool_fingerprint"]

    result = S.restore_stale_selection_relevance_softenings(proj, [seg])

    assert result["restored"] == [24]
    assert result["rebound"] == []
    assert seg.visual_policy == P.EXACT
    assert sel.image_path == ""
    row = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
               if r["segment_index"] == 24)
    assert row["status"] == "restored_evidence_stale"
    assert row["restore_reason"] == "softened_still_owner_banned"


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

    assert ok is False and why == "gap_review_quote_retrieval_binding_missing_or_incomplete"


def test_schema4_native_hd_gap_can_surrender_only_its_bound_verbatim_quote(
        monkeypatch, tmp_path):
    from vidlore.clipstudio import relevance_contract as R

    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words,
        verdict_patch={
            "status": "ok", "verdict": "keep", "matches_narration": True,
            "specific_enough": True, "correct_subject_visible": True,
            "downgraded": "exact→contextual", "relevance_class": "contextual_fallback",
        }, evidence_is_specific=False)
    evidence, _dimensions = _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)

    details, why = S._phase1_reviewed_exhaustion_details(proj, seg)
    assert why == "authorized_by_bound_completed_strict_acquisition_exhaustion"
    assert details["authorization_kind"] == S._NATIVE_HD_GAP_CLASSIFICATION
    assert details["surrender_verbatim_quote"] is True
    assert details["evidence_sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()

    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["blockers"][0]["quote_evidence"]["branch"] == "verbatim"
    assert S.semantic_gap_candidates(proj, audit)[0] == [48]

    observed = []

    def authorized_ladder(_proj, target, selection, *_args, **kwargs):
        observed.append(kwargs.get("surrender_verbatim_quote"))
        assert S._soften_to_character(
            target, lambda _m: None,
            surrender_verbatim_quote=kwargs["surrender_verbatim_quote"])
        selection.image_meta = {"selfheal_rung": "character_specific"}
        return True

    monkeypatch.setattr(S, "_clean_pool", lambda _proj: [])
    monkeypatch.setattr(S, "_venue_cache_save", lambda _proj: None)
    monkeypatch.setattr(S, "_soften_and_retry", authorized_ladder)
    payload = S.heal_selection_relevance_gaps(
        proj, [seg], ClipConfig(), audit, policy="approved_testing",
        eng=NS(anthropic_model="vision-test"), log=lambda _m: None)

    assert observed == [True]
    assert seg.quote == "" and seg.visual_policy == P.CHARACTER
    row = next(r for r in payload["beats"] if r["status"] == "softened")
    assert row["original"]["quote"] == quote and row["new"]["quote"] == ""
    assert row["quote_branch"] == "verbatim" and row["surrendered_quote"] is True
    assert row["authorization"]["authorization_kind"] == \
        S._NATIVE_HD_GAP_CLASSIFICATION
    assert row["authorization"]["evidence_sha256"] == \
        hashlib.sha256(evidence.read_bytes()).hexdigest()


def test_legacy_schema3_native_gap_evidence_is_not_reinterpreted_after_confirmation(
        monkeypatch, tmp_path):
    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)
    evidence, _dimensions = _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)
    artifact = json.loads(evidence.read_text())
    artifact["schema_version"] = 3
    evidence.write_text(json.dumps(artifact, sort_keys=True))

    validated, reason = S._strict_acquisition_evidence(
        proj, [seg], [seg.index], source=str(evidence))

    assert validated is None
    assert reason == "strict_acquisition_evidence_schema_not_2_or_4"


def test_native_gap_quote_surrender_invalidates_when_confirmation_artifact_changes(
        monkeypatch, tmp_path):
    from vidlore.clipstudio import relevance_contract as R

    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)
    _evidence, _dimensions = _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)
    old_confirmation = R._confirm_prompted_quote_span_unprompted

    def changed_confirmation(*args, **kwargs):
        result = dict(old_confirmation(*args, **kwargs))
        result["artifact_key"] = "b" * 64
        return result

    monkeypatch.setattr(R, "_confirm_prompted_quote_span_unprompted", changed_confirmation)
    details, reason = S._phase1_reviewed_exhaustion_details(proj, seg)

    assert details is None
    assert reason == "strict_acquisition_native_gap_quote_confirmation_artifact_changed"


@pytest.mark.parametrize(("field", "reason"), [
    ("retrieval_truncated_stream_count",
     "strict_acquisition_native_gap_retrieval_scan_truncated"),
    ("unprompted_confirmation_inconclusive_count",
     "strict_acquisition_native_gap_confirmation_inconclusive"),
])
def test_native_gap_quote_surrender_requires_every_candidate_stream_conclusive(
        monkeypatch, tmp_path, field, reason):
    """One confirmed SD hit cannot hide a possible unscanned or unjudged HD occurrence."""
    from vidlore.clipstudio import relevance_contract as R

    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)
    _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)
    complete_branches = R._quote_pool_branches

    def incomplete_branches(*args, **kwargs):
        branches = complete_branches(*args, **kwargs)
        branches[seg.index][field] = 1
        return branches

    monkeypatch.setattr(R, "_quote_pool_branches", incomplete_branches)
    details, denial = S._phase1_reviewed_exhaustion_details(proj, seg)

    assert details is None
    assert denial == reason


@pytest.mark.parametrize("stale_kind", ["evidence_hash", "source_became_hd"])
def test_active_native_gap_softening_restores_quote_when_objective_proof_stales(
        monkeypatch, tmp_path, stale_kind):
    from vidlore.clipstudio import relevance_contract as R

    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words,
        verdict_patch={
            "status": "ok", "verdict": "keep", "matches_narration": True,
            "specific_enough": True, "correct_subject_visible": True,
            "downgraded": "exact→contextual", "relevance_class": "contextual_fallback",
        }, evidence_is_specific=False)
    evidence, dimensions = _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)
    audit = R.evaluate_selection_relevance(proj, [seg])

    def authorized_ladder(_proj, target, selection, *_args, **kwargs):
        assert S._soften_to_character(
            target, lambda _m: None,
            surrender_verbatim_quote=kwargs["surrender_verbatim_quote"])
        selection.image_meta = {"selfheal_rung": "character_specific"}
        return True

    monkeypatch.setattr(S, "_clean_pool", lambda _proj: [])
    monkeypatch.setattr(S, "_venue_cache_save", lambda _proj: None)
    monkeypatch.setattr(S, "_soften_and_retry", authorized_ladder)
    S.heal_selection_relevance_gaps(
        proj, [seg], ClipConfig(), audit, policy="approved_testing",
        eng=NS(anthropic_model="vision-test"), log=lambda _m: None)
    assert seg.quote == "" and seg.visual_policy == P.CHARACTER
    assert S.restore_stale_selection_relevance_softenings(proj, [seg])["restored"] == []

    if stale_kind == "evidence_hash":
        evidence.write_text(evidence.read_text() + "\n")
    else:
        dimensions.update(width=1280, height=720)

    restored = S.restore_stale_selection_relevance_softenings(proj, [seg])
    assert restored["restored"] == [48]
    assert seg.visual_policy == P.EXACT and seg.quote == quote
    row = next(r for r in proj.meta["selection_relevance_gap_softening"]["beats"]
               if r["segment_index"] == 48)
    assert row["status"] == "restored_evidence_stale" and row["active"] is False
    post = R.evaluate_selection_relevance(proj, [seg])
    assert post["status"] == "blocked" and post["blocked_count"] == 1


def test_one_schema4_artifact_binds_total_absence_and_native_hd_absence_together(
        monkeypatch, tmp_path):
    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, native_seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)
    evidence, _dimensions = _bind_native_hd_gap_exhaustion(
        monkeypatch, proj, native_seg)
    total_seg = ScriptSegment(
        index=4, text="She puts it on the table and asks one question.", quote="",
        expected_visual="Catelyn places the dagger on the table",
        scene_query="Game of Thrones Catelyn dagger table",
        required_entity="Catelyn Stark", required_kind="character",
        visual_policy=P.EXACT, is_specific_claim=True)
    proj.segments = [total_seg, native_seg]

    artifact = json.loads(evidence.read_text())
    artifact["beats"]["4"] = {
        "beat_fingerprint": S._gap_beat_fingerprint(total_seg),
        "classification": "footage_gap",
        "gap_scope": S._TOTAL_ABSENCE_GAP_SCOPE,
        "actual_frame_pool_audit": True,
        "whole_pool_reviewed": True,
        "correct_footage_present_in_pool": False,
        "pipeline_bug_ruled_out": True,
        "strict_acquisition_status": "exhausted",
        "technical_status": "complete",
    }
    evidence.write_text(json.dumps(artifact, sort_keys=True))
    review = S.make_selection_relevance_gap_review(
        proj, proj.segments, [4, 48], method="actual_frame_and_pool_audit",
        source=str(evidence), strict_acquisition_exhausted_beats=[4, 48])
    proj.meta["selection_relevance_gap_review"] = review

    total, total_reason = S._reviewed_exhaustion_evidence(proj, total_seg)
    native, native_reason = S._reviewed_exhaustion_evidence(proj, native_seg)
    assert total_reason == native_reason == "strict_acquisition_evidence_valid"
    assert total["authorization_kind"] == "total_absence"
    assert total["surrender_verbatim_quote"] is False
    assert native["authorization_kind"] == S._NATIVE_HD_GAP_CLASSIFICATION
    assert native["surrender_verbatim_quote"] is True
    assert review["confirmed_gap_beats"] == [4, 48]
    assert set(review["strict_acquisition_exhaustion"]) == {"4", "48"}


@pytest.mark.parametrize(
    "changed", ["hd", "unprobeable", "reject_reason", "reject_membership"],
    ids=["became-hd", "probe-failed", "wrong-reject-reason", "not-currently-rejected"],
)
def test_schema4_native_gap_fails_closed_when_objective_sd_proof_changes(
        monkeypatch, tmp_path, changed):
    from vidlore.clipstudio import quality_contract as Q

    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)
    _evidence, dimensions = _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)

    if changed == "hd":
        dimensions.update(width=1280, height=720)
    elif changed == "unprobeable":
        dimensions.clear()
    elif changed == "reject_reason":
        proj.meta["auto_rejected_reasons"][proj.sources[0].id] = "subtitled_copy"
    else:
        proj.meta["auto_rejected_sources"] = []

    details, why = S._phase1_reviewed_exhaustion_details(proj, seg)
    assert details is None
    assert {
        "hd": "source_is_native_hd",
        "unprobeable": "source_unprobeable",
        "reject_reason": "source_not_rejected_as_sd",
        "reject_membership": "source_not_currently_rejected",
    }[changed] in why


def test_schema4_native_gap_denies_quote_surrender_when_an_hd_source_has_unknown_asr(
        monkeypatch, tmp_path):
    quote = "Kill his men."
    words = [[0.1 + i * .2, 0.2 + i * .2, word]
             for i, word in enumerate("Kill his men".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, index=48, quote=quote, words=words)

    unknown_media = tmp_path / "hd-unknown.mp4"
    unknown_media.write_bytes(b"publishable hd source with unknown asr")
    unknown_frame = tmp_path / "hd-unknown.jpg"
    unknown_frame.write_bytes(b"frame")
    unknown = SourceVideo(
        id="hd_unknown", url="unknown", title="Game of Thrones possible street scene",
        local_path=str(unknown_media), permission="owner", status="ok",
        checksum="unknown-hd-checksum", width=1280, height=720,
    )
    proj.sources.append(unknown)
    proj.shots_path(unknown.id).write_text(json.dumps([Shot(
        source_id=unknown.id, index=0, start=0.0, end=2.0,
        keyframe_path=str(unknown_frame)).to_dict()]))
    # Deliberately no current words/index metadata: absence cannot be proven for this HD source.
    prior_review = copy.deepcopy(proj.meta.get("selection_relevance_gap_review"))
    with pytest.raises(ValueError, match="native_gap_asr_pool_incomplete"):
        _bind_native_hd_gap_exhaustion(monkeypatch, proj, seg)
    assert proj.meta.get("selection_relevance_gap_review") == prior_review


def test_character_softening_preserves_quote_without_schema4_authorization():
    quote = "Kill his men."
    ordinary = Seg(quote=quote, required_kind="character", required_entity="Jaime Lannister")
    assert S._soften_to_character(ordinary, lambda _m: None) is True
    assert ordinary.quote == quote

    authorized = Seg(quote=quote, required_kind="character", required_entity="Jaime Lannister")
    assert S._soften_to_character(
        authorized, lambda _m: None, surrender_verbatim_quote=True) is True
    assert authorized.quote == ""


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
        S._soften_to_abstract(target, lambda _message: None,
                              cause="test exhausted paraphrase")
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
    _patch_confirmed_quote(monkeypatch, proj, seg)
    _bind_current_quote_retrieval_review(proj, seg, words)

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("current_quote_branch_not_conclusive_paraphrase" in line for line in lines)


def test_phase1_bound_review_cannot_soften_an_indeterminate_quote(monkeypatch, tmp_path):
    # No timed-ASR index means the pool cannot establish whether this authored phrase is dialogue.
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="Chaos isn't a pit. Chaos is a ladder.", words=None)

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 0 and called == []
    assert P.policy_of(seg) == P.EXACT
    assert any("gap_review_quote_retrieval_binding_missing_or_incomplete" in line
               for line in lines)


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
    _bind_current_quote_retrieval_review(
        proj, seg, [[0.1, 0.2, "completely"], [0.3, 0.4, "unrelated"]])

    resolved, called, lines = _run_phase1_authorization(monkeypatch, proj, seg)

    assert resolved == 1 and called == [seg.index]
    assert P.policy_of(seg) == P.ABSTRACT
    assert seg.quote == ""
    restored = S.restore_stale_selection_relevance_softenings(proj, [seg])
    assert restored["restored"] == [] and restored["unchanged"] == [seg.index]
    assert P.policy_of(seg) == P.ABSTRACT and seg.quote == ""
    assert not any("specificity softening DENIED" in line for line in lines)


def test_gap_review_paraphrase_binding_includes_confirmation_generation(
        monkeypatch, tmp_path):
    from vidlore.clipstudio import relevance_contract as R

    words = [[0.1, 0.2, "completely"], [0.3, 0.4, "unrelated"]]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="The essayist's paraphrase is not spoken dialogue.", words=words)
    _bind_current_quote_retrieval_review(proj, seg, words)
    stored = proj.meta["selection_relevance_gap_review"]["quote_branch_bindings"][
        str(seg.index)]
    assert len(stored["confirmation_generation_fingerprint"]) == 64
    current_branches = R._quote_pool_branches

    def changed_confirmation_generation(*args, **kwargs):
        branches = current_branches(*args, **kwargs)
        branches[seg.index]["confirmation_decoder_fingerprint_expected"] = "b" * 64
        return branches

    monkeypatch.setattr(R, "_quote_pool_branches", changed_confirmation_generation)
    ok, reason = S._phase1_softening_authorization(proj, seg)

    assert ok is False
    assert reason == "stale_gap_review_quote_confirmation_generation"


@pytest.mark.parametrize(
    "changed", ["artifact", "result", "result_hash", "becomes_verbatim"])
def test_active_paraphrase_softening_restores_when_confirmation_binding_changes(
        monkeypatch, tmp_path, changed):
    """Sidecars alone cannot keep an old paraphrase decision alive after confirmation changes."""
    from vidlore.clipstudio import relevance_contract as R

    quote = "Chaos isn't a pit. Chaos is a ladder."
    words = [[0.1 + i * .1, 0.2 + i * .1, word]
             for i, word in enumerate("Chaos isn't a pit Chaos is a ladder".split())]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote=quote, words=words)
    decoder = R._quote_confirmation_decoder_fingerprint(load_clip_config())
    state = {
        "status": "rejected", "reason": "unprompted_phrase_not_found_near_hint",
        "artifact_key": "a" * 64,
        "result_content_sha256": "c" * 64,
    }

    def confirmation(_proj, _src, _quote, prompted_span, _cfg, **_kwargs):
        result = {
            "schema_version": R.QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": R.QUOTE_CONFIRMATION_ALGORITHM,
            "status": state["status"],
            "reason": state["reason"],
            "artifact_key": state["artifact_key"],
            "result_content_sha256": state["result_content_sha256"],
            "decoder_fingerprint": decoder,
            "prompted_span": list(prompted_span),
            "match_method": "exact_contiguous_timed_asr+unprompted_confirmation",
        }
        if state["status"] == "confirmed":
            result["confirmed_span"] = list(prompted_span)
            result["timed_asr_ratio"] = float(prompted_span[2])
        return result

    monkeypatch.setattr(R, "_confirm_prompted_quote_span_unprompted", confirmation)
    _bind_current_quote_retrieval_review(proj, seg, words)
    sidecar = proj.index_dir / "show.quote_retrieval.json"
    sidecar_before = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    resolved, called, _lines = _run_phase1_authorization(monkeypatch, proj, seg)
    assert resolved == 1 and called == [seg.index] and seg.quote == ""
    row = next(row for row in proj.meta["selection_relevance_gap_softening"]["beats"]
               if row["segment_index"] == seg.index)
    assert row["quote_branch_binding"]["branch"] == "paraphrase"
    assert row["quote_branch_binding"]["confirmation_artifact_keys"] == ["a" * 64]

    if changed == "artifact":
        state["artifact_key"] = "b" * 64
    elif changed == "result":
        state["reason"] = "same_branch_but_new_independent_result"
    elif changed == "result_hash":
        # The semantic status/span can stay identical while the persisted decoder bytes change.
        # Active softening must bind the independently validated result content, not merely the
        # input/cache key, or stale evidence could survive an artifact rewrite.
        state["result_content_sha256"] = "d" * 64
    else:
        state.update(status="confirmed", reason="")

    restored = S.restore_stale_selection_relevance_softenings(proj, [seg])

    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == sidecar_before
    assert restored["restored"] == [seg.index]
    assert seg.visual_policy == P.EXACT and seg.quote == quote
    row = next(row for row in proj.meta["selection_relevance_gap_softening"]["beats"]
               if row["segment_index"] == seg.index)
    expected_reason = ("current_quote_branch_not_paraphrase"
                       if changed == "becomes_verbatim" else "quote_branch_binding_changed")
    assert row["restore_reason"] == expected_reason


@pytest.mark.parametrize("changed", ["sidecar", "prompt_generation"])
def test_active_paraphrase_softening_restores_when_quote_retrieval_binding_changes(
        monkeypatch, tmp_path, changed):
    words = [[0.1, 0.2, "completely"], [0.3, 0.4, "unrelated"]]
    proj, seg, _sel = _phase1_evidence_fixture(
        tmp_path, quote="The essayist's paraphrase is not spoken dialogue.", words=words)
    _bind_current_quote_retrieval_review(proj, seg, words)
    resolved, called, _lines = _run_phase1_authorization(monkeypatch, proj, seg)
    assert resolved == 1 and called == [seg.index] and seg.quote == ""

    if changed == "sidecar":
        sidecar = proj.index_dir / "show.quote_retrieval.json"
        sidecar.write_text(sidecar.read_text() + "\n")
    else:
        proj.segments.append(ScriptSegment(
            index=999, text="Another authored beat", quote="A newly authored retrieval hint.",
            expected_visual="another moment", scene_query="another moment",
            visual_policy=P.EXACT, is_specific_claim=True))

    restored = S.restore_stale_selection_relevance_softenings(proj, [seg])

    assert restored["restored"] == [seg.index]
    assert seg.visual_policy == P.EXACT
    assert seg.quote == "The essayist's paraphrase is not spoken dialogue."
    row = next(row for row in proj.meta["selection_relevance_gap_softening"]["beats"]
               if row["segment_index"] == seg.index)
    assert row["status"] == "restored_evidence_stale"
    assert "quote_retrieval_" in row["restore_reason"]


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
