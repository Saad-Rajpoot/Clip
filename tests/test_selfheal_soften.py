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

import inspect

import pytest

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import selfheal as S


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
    point is to stop asking, not to skip the honest attempts."""
    src = inspect.getsource(S.heal_blocked_beats)
    i_still = src.index("still_recover")
    i_acq = src.index("acquire_for_beat")
    i_region = src.index("_region_frames_recover")
    i_soft = src.index("_soften_and_retry")
    assert i_soft > i_still and i_soft > i_acq and i_soft > i_region


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
    from types import SimpleNamespace as NS

    seg = Seg()
    sel = NS(segment_index=24, image_path="", image_meta={})
    monkeypatch.setattr(S, "still_recover", lambda *a, **k: False)
    before = (seg.visual_policy, seg.required_entity, seg.required_kind,
              seg.expected_visual, seg.scene_query, seg.quote)
    assert S._soften_and_retry(NS(), seg, sel, NS(), [], set(), lambda _m: None) is False
    after = (seg.visual_policy, seg.required_entity, seg.required_kind,
             seg.expected_visual, seg.scene_query, seg.quote)
    assert after == before
