"""Regressions for explicit symbolic-metaphor contracts versus objective deixis."""
from vidlore.clipstudio import policy as P
from vidlore.clipstudio.models import ScriptSegment


def _power_beat(**overrides):
    values = {
        "index": 170,
        "text": "But Tywin Lannister's power did not die in that room.",
        "expected_visual": "Symbolic image of Tywin's power crumbling, perhaps a falling statue",
        "shot_intent": "symbolic",
        "visual_policy": P.EXACT,
        "is_specific_claim": False,
    }
    values.update(overrides)
    return ScriptSegment(**values)


def test_explicit_unanchored_symbolic_metaphor_resolves_abstract_and_stays_stable():
    beat = _power_beat()

    assert P.is_deictic(beat) is True
    assert P.policy_of(beat) == P.ABSTRACT
    assert P.finalize_beats([beat]) == {P.ABSTRACT: 1}
    for _ in range(3):
        beat = ScriptSegment.from_dict(beat.to_dict())
        assert P.policy_of(beat) == P.ABSTRACT
        assert P.finalize_beats([beat]) == {P.ABSTRACT: 1}


def test_symbolic_label_or_storyboard_alone_never_overrides_objective_deixis():
    controls = [
        ScriptSegment(
            index=0,
            text="Somewhere in those ninety seconds, one word gets spoken.",
            expected_visual="Symbolic image of time passing",
            shot_intent="symbolic", visual_policy=P.EXACT),
        ScriptSegment(
            index=1,
            text="The point is what everyone else at that table heard.",
            expected_visual="Symbolic image of uneasy listeners",
            shot_intent="symbolic", visual_policy=P.EXACT),
        _power_beat(expected_visual="Tywin in the room"),
        _power_beat(shot_intent="emotional_closeup"),
    ]

    assert all(P.is_deictic(beat) for beat in controls)
    assert all(P.policy_of(beat) == P.EXACT for beat in controls)


def test_any_concrete_contract_keeps_symbolic_metaphor_exact():
    controls = [
        _power_beat(required_entity="Tywin Lannister", required_kind="character"),
        _power_beat(scene_query="Tywin's final room"),
        _power_beat(quote="You are no son of mine"),
        _power_beat(is_specific_claim=True),
    ]

    assert all(P.policy_of(beat) == P.EXACT for beat in controls)


def test_symbolic_storyboard_cannot_hide_its_own_deictic_anchor():
    storyboards = [
        "Symbolic image of his legacy lingering at that exact table",
        "Symbolic image of his legacy lingering at that bridge",
        "Symbolic image of his legacy lingering on this battlefield",
        "Symbolic image of his legacy lingering around that dagger",
    ]

    for expected in storyboards:
        beat = _power_beat(
            text="His legacy did not die at that table.",
            expected_visual=expected,
        )
        assert P.policy_of(beat) == P.EXACT


def test_symbolic_metaphor_with_a_concrete_action_clause_stays_exact():
    beat = _power_beat(
        text="Tywin's power did not die when Tyrion fired the crossbow in that room.",
    )

    assert P.policy_of(beat) == P.EXACT
