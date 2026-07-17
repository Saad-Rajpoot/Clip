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


TESTS = [
    test_deixis_promotes_the_lines_that_actually_failed,
    test_deixis_does_not_swallow_generic_narration,
    test_deixis_survives_finalize_and_is_idempotent,
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
