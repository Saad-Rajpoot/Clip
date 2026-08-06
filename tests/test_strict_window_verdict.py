"""The strict per-window decision, now callable — and provably the same decision.

It used to live inside `_try_promote_inner`, unreachable from anywhere else. That is why nothing
could audit a pool the way the publication gate judges it: an external check had to re-implement the
bar, and a re-implementation that drifts is worse than no check. `_try_promote_inner` now calls
`strict_window_verdict`; there is exactly one copy.

The parity tests below re-derive the OLD inline expression from the same inputs and assert the two
agree on every combination — including the ones that matter most: a transport error is not a
rejection, and a `keep` without affirmative target evidence does not satisfy a beat that points at
something.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import verify as V


# ------------------------------------------------------------------ there is only one copy
def test_try_promote_calls_the_extracted_judge():
    src = inspect.getsource(V)
    i = src.index("def _try_promote_inner(")
    body = src[i:i + 6000]
    assert "strict_window_verdict(" in body


def test_the_old_inline_decision_is_gone_from_the_promotion_loop():
    """No second copy: the loop must not still assemble `_accept` itself."""
    src = inspect.getsource(V)
    i = src.index("def _try_promote_inner(")
    body = src[i:i + 6000]
    assert "_strict_keep_rejection_reason(" not in body
    assert "_exact_contextual_ok(" not in body


# ------------------------------------------------------------------ fail-closed
def test_a_transport_error_is_incomplete_not_a_rejection():
    """A caller counting rejections must never count a verifier that did not answer."""
    got = V.strict_window_verdict(None, object(), object(), object(), object(), {},
                                  downgrade=False, exact=True, character=False, must_see="")
    assert got["accept"] is False
    assert got["status"] == "incomplete"
    assert got["reason"] == "verifier_transport_error"


def test_an_exception_while_judging_is_incomplete_not_a_rejection():
    class Boom:
        def source(self, _sid):
            raise RuntimeError("index unavailable")
    got = V.strict_window_verdict({"verdict": "keep"}, object(), object(), Boom(), object(), {},
                                  downgrade=False, exact=True, character=False, must_see="")
    assert got["accept"] is False and got["status"] == "incomplete"
    assert got["reason"].startswith("strict_judgement_error:")


def test_incomplete_is_distinguishable_from_judged():
    assert {"judged", "incomplete"} >= {
        V.strict_window_verdict(None, 0, 0, 0, 0, {}, downgrade=False, exact=False,
                                character=False, must_see="")["status"]}


# ------------------------------------------------------------------ the decision is unchanged
class _Seg:
    index = 3
    text = "Tyrion pours the wine."
    required_kind = "character"
    required_entity = "Tyrion Lannister"
    visual_policy = "exact_scene"
    quote = ""
    scene_query = ""


class _Alt:
    source_id = "s1"
    shot_index = 2
    in_point = 1.0
    out_point = 3.0


class _Proj:
    def source(self, _sid):
        return type("S", (), {"title": "Game of Thrones scene"})()


def _old_inline(av, seg, alt, proj, cfg, c2a, *, downgrade, exact, character, must_see):
    """The expression exactly as `_try_promote_inner` used to evaluate it."""
    src = proj.source(alt.source_id)
    title = ((getattr(src, "title", "") or "") + " " + (alt.source_id or ""))
    conflict = V._contradiction_reason(seg, av, title, c2a)
    if conflict:
        av["contradicts_narration"] = True
        av["contradiction_reason"] = conflict
    if downgrade:
        return V._exact_contextual_ok(av, seg, title, c2a)
    reject = ""
    if exact:
        reject = V._strict_keep_rejection_reason(av, seg, title, c2a, must_see=must_see)
        if not reject:
            reaction = V._exact_reaction_context_evidence(proj, alt, seg, cfg=cfg)
            if reaction.get("required") and not reaction.get("passed"):
                reject = str(reaction.get("reason") or "exact_reaction_context_unproven")
    elif character:
        reject = V._character_keep_rejection_reason(av, seg, title, c2a, must_see=must_see)
    return (av.get("verdict") == "keep" and not conflict and not reject
            and (not exact or not must_see or av.get("target_visible") is True))


_VERDICTS = [
    {"verdict": "keep", "matches_narration": True, "correct_subject_visible": True,
     "wrong_subject_visible": False, "specific_enough": True, "target_visible": True},
    {"verdict": "keep", "matches_narration": True, "correct_subject_visible": True,
     "wrong_subject_visible": False, "specific_enough": True},              # no target evidence
    {"verdict": "keep", "matches_narration": False, "correct_subject_visible": True,
     "wrong_subject_visible": False, "specific_enough": True},
    {"verdict": "keep", "matches_narration": True, "correct_subject_visible": False,
     "wrong_subject_visible": True, "specific_enough": True},
    {"verdict": "keep", "matches_narration": True, "correct_subject_visible": True,
     "wrong_subject_visible": False, "specific_enough": False},
    {"verdict": "replace", "matches_narration": False, "correct_subject_visible": False,
     "wrong_subject_visible": True, "specific_enough": False},
    {"verdict": "keep", "contradicts_narration": True, "matches_narration": True,
     "correct_subject_visible": True, "wrong_subject_visible": False, "specific_enough": True},
]


@pytest.mark.parametrize("av", _VERDICTS)
@pytest.mark.parametrize("downgrade,exact,character",
                         [(False, True, False), (False, False, True),
                          (False, False, False), (True, True, False)])
@pytest.mark.parametrize("must_see", ["", "the chalice"])
def test_refactored_and_inline_agree(av, downgrade, exact, character, must_see):
    seg, alt, proj, cfg, c2a = _Seg(), _Alt(), _Proj(), None, {}
    want = _old_inline(dict(av), seg, alt, proj, cfg, dict(c2a), downgrade=downgrade,
                       exact=exact, character=character, must_see=must_see)
    got = V.strict_window_verdict(dict(av), seg, alt, proj, cfg, dict(c2a), downgrade=downgrade,
                                  exact=exact, character=character, must_see=must_see)
    assert bool(got["accept"]) == bool(want), (av, downgrade, exact, character, must_see)


# ------------------------------------------------------------------ it explains itself
@pytest.mark.parametrize("field", [
    "verdict", "matches_narration", "correct_subject_visible", "wrong_subject_visible",
    "specific_enough", "evidence", "reason", "model", "prompt_version", "window", "rung", "status",
])
def test_the_result_carries_what_an_audit_needs(field):
    got = V.strict_window_verdict(dict(_VERDICTS[0]), _Seg(), _Alt(), _Proj(), None, {},
                                  downgrade=False, exact=True, character=False, must_see="")
    assert field in got


def test_the_window_identity_is_recorded():
    got = V.strict_window_verdict(dict(_VERDICTS[0]), _Seg(), _Alt(), _Proj(), None, {},
                                  downgrade=False, exact=True, character=False, must_see="")
    assert got["window"] == {"source_id": "s1", "shot_index": 2,
                             "in_point": 1.0, "out_point": 3.0}


def test_a_rejection_always_names_a_reason():
    got = V.strict_window_verdict(dict(_VERDICTS[5]), _Seg(), _Alt(), _Proj(), None, {},
                                  downgrade=False, exact=True, character=False, must_see="")
    assert got["accept"] is False and got["reason"]
