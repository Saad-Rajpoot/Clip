"""Regression: the 56e0467283 fix pass (relevance / verifier / timeline / breakouts).

Every case is anchored to a MEASURED defect in the shipped 15:24 render, not a hypothetical.
Where a test asserts a number, that number came from the audit of the aired video.

    python3 tests/test_relevance_fix_pass.py

Pure logic — no render, no network, no LLM.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import policy  # noqa: E402

FAILS = []


def _seg(text, **kw):
    d = dict(text=text, visual_policy="", quote="", is_specific_claim=False,
             required_entity="", required_kind="", scene_query="", expected_visual="")
    d.update(kw)
    return NS(**d)


# ---------------------------------------------------------------------------
# F3 — deictic inheritance. Lines are VERBATIM from the shipped render.
# ---------------------------------------------------------------------------
def test_deixis_promotes_the_lines_that_actually_failed():
    # 10:08 — shipped as generic_filler, aired Ned Stark's execution (S01E09).
    # _ABSTRACT_RX also matches "the point is", which pushed this AWAY from exact.
    s = _seg("The point is what everyone else at that table heard.", visual_policy="generic_filler")
    assert policy.is_deictic(s), "'that table' not detected as deictic"
    assert policy.policy_of(s) == policy.EXACT, "deixis must outrank the LLM's generic_filler label"

    # 0:36 — shipped as abstract_effect, aired the Hound + Arya at the Inn at the Crossroads.
    s = _seg("Somewhere in those ninety seconds, one word gets spoken.",
             visual_policy="abstract_effect")
    assert policy.policy_of(s) == policy.EXACT, "'those ninety seconds' must be exact"

    # 14:50 — the thesis payoff.
    s = _seg("then Joffrey's real death didn't happen at a wedding. It happened at that table.")
    assert policy.policy_of(s) == policy.EXACT

    # 11:14 — shipped as character_specific, aired the Hound + Arya.
    s = _seg("Go to the books though, and you find out that nobody in that room was laughing.",
             visual_policy="character_specific")
    assert policy.policy_of(s) == policy.EXACT

    # 9:02 — the promised payoff; narrator points AT the line the viewer never heard.
    s = _seg("There it is. That's the word I told you to listen for. Nightshade.",
             visual_policy="generic_filler")
    assert policy.policy_of(s) == policy.EXACT

    # 6:10 — an explicit instruction to re-watch the anchor scene.
    s = _seg("Watch it again sometime, and don't look at Joffrey at all.")
    assert policy.policy_of(s) == policy.EXACT


def test_deixis_does_not_swallow_generic_narration():
    """The guard-rail on the guard-rail: over-promoting would strip filler from beats that never
    needed exact footage and block the render on them."""
    for t in ("That's the tragedy of it.",
              "This is why he never had to raise his voice.",
              "And that's exactly the point.",
              "It's really about control, not titles.",
              "Think about what that means for a moment."):
        assert not policy.is_deictic(_seg(t)), f"false deictic hit: {t!r}"

    s = _seg("Westeros was a place where power was always contested.", visual_policy="generic_filler")
    assert policy.policy_of(s) == policy.FILLER, "generic filler must survive"

    s = _seg("In the end, nothing would ever be the same.")
    assert policy.policy_of(s) == policy.ABSTRACT, "abstract beats must stay abstract"


def test_deixis_survives_finalize_and_is_idempotent():
    s = _seg("The point is what everyone else at that table heard.", visual_policy="generic_filler")
    policy.finalize_beats([s])
    assert s.visual_policy == policy.EXACT
    assert policy.policy_of(s) == policy.EXACT, "must be stable on a second pass"


# ---------------------------------------------------------------------------
# F6 — word-level ASR + quote-span location.
# The word stream below is the REAL one from tywin_lannister_dismis_f4c81b75 (shipped index +
# isolated whisper of the source), garble included. These are the two lines the render lost.
# ---------------------------------------------------------------------------
_REAL_ASR_CHUNKS = [
    (121.25, 122.46, "I am the"), (122.46, 124.17, "king."),
    (124.17, 126.58, "I will punish you. Any man who must"),
    (126.58, 130.05, "say I am the king is no true king."),
    (130.05, 132.42, "I'll make sure you understand"),
    (132.42, 134.51, "up and I've won your war for you."),
    (134.51, 140.31, "My father won the real war. He killed Prince Rhaegar. He took the crown while"),
    (140.31, 142.89, "you hid on a costly rock."),
    (154.99, 158.70, "The king is tired. See him to"),
    (158.70, 162.75, "his chambers. Come on off. I'm not tired. We have"),
    (162.75, 163.96, "so much to celebrate."), (163.96, 166.00, "A wedding to plan."),
    (166.00, 169.67, "You must rest. Grand"), (169.67, 171.38, "Maester. Perhaps"),
    (171.38, 173.55, "a messence of nightshade to help him"),
    (173.55, 178.68, "sleep. I'm not fired."),
]


def _real_words():
    out = []
    for a, b, txt in _REAL_ASR_CHUNKS:
        ws = txt.split()
        step = (b - a) / len(ws)
        out.extend((round(a + i * step, 3), round(a + (i + 1) * step, 3), w)
                   for i, w in enumerate(ws))
    return out


def test_thesis_line_locatable_across_a_shot_boundary():
    """'Any man who must say I am the king is no true king' straddles the cut at 126.58, so it
    lands as '...Any man who must' + 'say I am the king...'. No single shot contains it, which is
    why per-shot substring search never found the most iconic line in the video."""
    from vidlore.clipstudio.index import find_quote_span
    r = find_quote_span(_real_words(), "Any man who must say 'I am the king' is no true king.")
    assert r is not None, "thesis line not located"
    s, e, ratio = r
    assert abs(s - 125.6) <= 2.0, f"span start {s} not at the real line start (~125.6)"
    assert e >= 129.0, f"span end {e} truncates the line"
    assert ratio >= 0.8, f"phrase ratio too low: {ratio}"


def test_payoff_line_locatable_through_asr_garble_and_four_shots():
    """The nightshade line spans 4 shots AND is garbled ('a messence of nightshade', 'I'm not
    fired'). No breakout candidate was ever generated for it in the shipped render."""
    from vidlore.clipstudio.index import find_quote_span
    r = find_quote_span(_real_words(), "Perhaps some essence of nightshade to help him sleep.")
    assert r is not None, "payoff line not located through garble"
    s, e, ratio = r
    assert abs(s - 170.0) <= 2.0, f"span start {s} wrong"
    assert e >= 173.5, f"span end {e} cuts the line short of 'sleep'"
    assert ratio >= 0.75, f"phrase ratio too low: {ratio}"


def test_single_garbled_word_cannot_anchor_a_match():
    """A lone fuzzy token must never carry a match — otherwise 'sleep' anchors anywhere. The
    acceptance is per-PHRASE; fuzziness is only ever one term inside that score."""
    from vidlore.clipstudio.index import find_quote_span
    assert find_quote_span(_real_words(), "sleep") is None
    assert find_quote_span(_real_words(), "Winter is coming to the North this year") is None
    assert find_quote_span(_real_words(), "dragons burned the fleet at anchor") is None


TESTS = [
    test_deixis_promotes_the_lines_that_actually_failed,
    test_deixis_does_not_swallow_generic_narration,
    test_deixis_survives_finalize_and_is_idempotent,
    test_thesis_line_locatable_across_a_shot_boundary,
    test_payoff_line_locatable_through_asr_garble_and_four_shots,
    test_single_garbled_word_cannot_anchor_a_match,
]


if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
