"""Running out of repair pages is a content verdict, not a plumbing fault.

Job f840b0cb49 spent 8h12m getting to the point where 41 of 254 beats still failed the semantic
publication contract, ran all 16 recovery pages against them (`targeted rediscovery found no NEW
source`, every round), and then died — in REVIEW mode, one stage away from a deliverable draft.

The escape in `_produce_auto` keys on `kind == "selection_relevance"`, and the pagination guard
raised a bare `PipelineError`, which carries no kind. So review mode treated "the repair machinery
is out of moves" as if it were a broken pipe and re-raised, throwing away everything.

The exhaustion errors now say what they are. Nothing is softened by this: strict mode raises exactly
as before, the beats stay blocked and audited, and the draft is still named REVIEW_DRAFT. The
neighbouring guards for a malformed or unrecognised receipt stay deliberately untyped — those are
technical faults and must remain fatal in every mode.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio.orchestrate import PipelineError, _pagination_terminal_error


def _cursor(reason: str, *, deferred=(129, 130, 141), max_pages=16) -> dict:
    return {
        "status": "current",
        "deferred": list(deferred),
        "receipt": {"terminal_reason": reason, "max_pages": max_pages},
    }


# The two ways the walk can end. Both mean: beats still fail the contract, no moves left.
@pytest.mark.parametrize("reason", ["finite_guard:16:129,130,141", "no_progress:129,130,141"])
def test_exhaustion_is_typed_as_a_relevance_stop(reason):
    err = _pagination_terminal_error(_cursor(reason))
    assert isinstance(err, PipelineError)
    assert getattr(err, "kind", "") == "selection_relevance"


@pytest.mark.parametrize("reason", ["finite_guard:16:129,130,141", "no_progress:129,130,141"])
def test_review_mode_escape_accepts_it(reason):
    """The exact predicate `_produce_auto` applies before it continues into build."""
    err = _pagination_terminal_error(_cursor(reason))
    assert isinstance(err, RuntimeError)                      # reaches `except RuntimeError`
    assert getattr(err, "kind", "") == "selection_relevance"  # and satisfies the review condition


def test_strict_mode_still_raises():
    """Typing the error must not make it optional — `block` mode has no escape to fall into."""
    err = _pagination_terminal_error(_cursor("finite_guard:16:129,130,141"))
    with pytest.raises(PipelineError):
        raise err


def test_the_blocked_beats_are_still_named():
    err = _pagination_terminal_error(_cursor("finite_guard:16:129,130,141"))
    assert "129, 130, 141" in str(err)
    assert "16-page" in str(err)


def test_a_technical_pipeline_error_stays_untyped():
    """Tagging is opt-in. A malformed-receipt / unreadable-audit fault must NOT become deliverable."""
    assert getattr(PipelineError("semantic recovery pagination terminal receipt is malformed"),
                   "kind", "") == ""


@pytest.mark.parametrize("reason", ["", "configured_rounds_completed", "no_untried_quality"])
def test_a_walk_that_has_not_terminated_yields_nothing(reason):
    assert _pagination_terminal_error(_cursor(reason)) is None
