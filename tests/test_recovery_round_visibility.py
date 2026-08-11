"""A recovery round may be slow. It may not be invisible.

Measured on job 218acdfe10: fifteen bounded-rediscovery rounds, **216 minutes**, in stretches of up
to 841 seconds with not one line of log. Every step inside a round — search, download, index,
rematch, scoped reverify — was called with `progress=None`, so from outside the render was
indistinguishable from a wedged process. It was reported as exactly that, twice.

The rounds themselves are NOT waste, and this file records the number so nobody "optimises" them
away from memory: those 216 minutes recovered **16 beats with real verified footage**. A naive
stop-after-two-dry-rounds rule would have quit at round 6 and lost seven of them — rounds 7, 9, 11
and 15 each produced beats after a dry spell. Cutting recovery does not save time; it ships beats
with no footage.

So the fix is visibility and attribution: each step says what it is doing and how long it took, and
the next person deciding where a round's fourteen minutes go reads it out of build.log instead of
guessing.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import orchestrate as O


def _round_source() -> str:
    """From the round's opening log line to the end of its enclosing top-level function.

    Slicing at the next `def ` would stop at the first NESTED helper and hand back a few hundred
    characters that assert almost nothing — so the boundary is a top-level def, at column zero.
    """
    src = inspect.getsource(O)
    start = src.index("recovery: {len(unresolved)} unresolved beat(s)")
    end = src.index("\ndef ", start)
    body = src[start:end]
    assert len(body) > 5000, "the slice missed the round body — the guard would be vacuous"
    return body


def _code_only(body: str) -> str:
    """Comments explain the defect by NAME, so a substring check over them proves nothing."""
    return "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))


def test_no_step_of_a_recovery_round_is_silent():
    body = _code_only(_round_source())
    assert "progress=None" not in body, \
        "a step of a recovery round is still invisible — this is the 841-second black box"


def test_every_expensive_step_reports_what_it_is_and_how_long_it_took():
    body = _round_source()
    for stage in ("search", "download", "index", "rematch", "reverify"):
        assert f'_rlog("{stage}")' in body, f"{stage} does not say what it is doing"
        assert f'_rdone("{stage}")' in body, f"{stage} does not report its duration"


def test_the_lines_are_attributable_to_the_round_and_the_step():
    body = _round_source()
    assert 'log(f"recovery/{stage}: {m}")' in body, \
        "a bare progress line cannot be told apart from the main pipeline's"
    assert "into this round" in body, \
        "a step duration is only useful next to how far into the round it happened"


def test_the_rounds_themselves_are_not_quietly_capped():
    """The measured yield curve is not monotonic — rounds 7, 9, 11 and 15 each recovered beats
    after a dry round — so a dry-streak cutoff would trade footage for wall-clock. If one is ever
    added it must be a deliberate, documented decision, not a side effect of a speed pass."""
    body = _round_source()
    for smell in ("dry_streak", "consecutive_dry", "max_dry"):
        assert smell not in body, f"an undocumented recovery cutoff appeared: {smell}"


def test_the_timer_cannot_crash_a_render():
    """_rdone on a stage that never started must still produce a line, not a KeyError."""
    body = _round_source()
    assert "_stage_t.get(stage, _t_round)" in body, \
        "a missing stage stamp must fall back, never raise inside a render"
