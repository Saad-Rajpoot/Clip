"""A cursor left behind by an older build must be discarded, not fatal.

`_semantic_recovery_cursor` has a branch for exactly this: an older-schema cursor is stale state,
and the correct response is to run a fresh bounded page and replace it. That branch stamped the
CURRENT schema onto the old marker and then ran the CURRENT validator over it — a validator that
demands fields (`deferred`, `completed_page_scope`, `pool_fingerprint`, `gap_softening`) added long
after such a marker was written. So every cursor older than those additions raised instead of being
discarded, and the render died.

MEASURED: the short end-to-end test crashed with

    PipelineError: semantic recovery cursor is missing 'deferred'

against a real on-disk marker

    {"schema_version": 3, "before": [2, 5], "after": [2, 5], "post_fingerprint": "d27ac3..."}

while the current schema is 13 — state left by an earlier build permanently breaking that job.
Validating what is about to be thrown away is what turned recoverable state into a fatal error.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import orchestrate as O


SRC = inspect.getsource(O._semantic_recovery_cursor)


def test_the_stale_branch_no_longer_forges_the_current_schema():
    assert 'structural_marker["schema_version"] = current_schema' not in SRC


def test_the_stale_branch_does_not_demand_fields_a_new_schema_added():
    i = SRC.index("allow_stale and isinstance(marker_schema, int)")
    j = SRC.index('return {"status": "stale"', i)
    assert "_semantic_recovery_marker_fields(" not in SRC[i:j], \
        "a marker being discarded must not be validated against the current field set"


def test_a_stale_cursor_is_still_structurally_checked():
    """Discarding is not the same as ignoring: a corrupt old marker must still raise."""
    i = SRC.index("allow_stale and isinstance(marker_schema, int)")
    j = SRC.index('return {"status": "stale"', i)
    assert '_semantic_cursor_int_list(marker, "before", require_nonempty=True)' in SRC[i:j]


@pytest.mark.parametrize("bad", [
    {"before": []},                       # empty is not a "complete positive" cursor
    {"before": "2,5"},                    # not a list
    {"before": [2, 2]},                   # duplicates
    {"before": [-1]},                     # negative index
    {"before": [True]},                   # bool is not an index
    {},                                   # missing entirely
])
def test_a_corrupt_old_cursor_still_raises(bad):
    with pytest.raises(O.PipelineError):
        O._semantic_cursor_int_list(bad, "before", require_nonempty=True)


def test_the_real_marker_that_crashed_the_render_now_validates():
    """The exact dict read off disk in the failing run."""
    marker = {"schema_version": 3, "before": [2, 5], "after": [2, 5],
              "post_fingerprint": "d27ac38fdf238daf34edd8fe4b37da8daca0c80385af9305ff7610d8d353a8e9"}
    assert O._semantic_cursor_int_list(marker, "before", require_nonempty=True) == [2, 5]
    with pytest.raises(O.PipelineError):
        O._semantic_cursor_int_list(marker, "deferred")   # the field it used to die on


def test_a_stale_cursor_reports_no_progress():
    """It must not be mistaken for completed work — that would reset a finite recovery budget."""
    i = SRC.index('return {"status": "stale"')
    assert '"deferred": []' in SRC[i:i + 120]
