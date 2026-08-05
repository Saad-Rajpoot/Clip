"""An authorized transform is described by its geometry, not by how it was spelled.

Lineage proves a delivered frame belongs to its source window, and the pipeline is allowed to have
punched a corner crop into that frame on the way — so the comparison must apply the same crop, so
the crop must be DECLARED. A declared transform is an input, and an input is something a bug or an
attacker could use to make foreign pixels compare equal. The original defence was four literal
filter strings. Airtight, and brittle where it matters: `iw*0.84` and `iw*0.840` are the same
geometry, and one of them was refused, so any change in how the filter is formatted downstream
silently converts a provable frame into a blocked render.

The declaration is now PARSED into canonical geometry and checked against a strict schema instead
of string-matched. Nothing is widened. What is accepted is what was accepted before: a pure corner
crop, at a fraction this pipeline actually produces, anchored to a real corner. The expression is
never executed — parsing only decides whether the declaration describes a transform the pipeline was
allowed to have applied.
"""
from __future__ import annotations

import pytest

from vidlore.scene_lineage_canary import (
    _AUTHORIZED_SOURCE_COMPARE_FILTERS,
    canonical_source_compare_transform as canon,
    source_compare_filter_authorized as ok,
)


# ------------------------------------------------------------------ nothing already trusted breaks
@pytest.mark.parametrize("expr", sorted(_AUTHORIZED_SOURCE_COMPARE_FILTERS))
def test_every_previously_authorized_crop_still_passes(expr):
    assert ok(expr), expr
    assert canon(expr)["kind"] == "corner_crop"


def test_all_four_corners_are_recognised_as_distinct_geometry():
    seen = {(canon(e)["x_frac"], canon(e)["y_frac"])
            for e in _AUTHORIZED_SOURCE_COMPARE_FILTERS}
    assert len(seen) == 4, "each corner must canonicalise to its own origin"


# ------------------------------------------------------------------ the brittleness that is fixed
@pytest.mark.parametrize("expr", [
    "crop=iw*0.84:ih*0.84:0:0",
    "crop=iw*.840:ih*.840:0:0",
    "  crop=iw*0.840:ih*0.840:0:0  ",
])
def test_an_equivalent_spelling_of_the_same_geometry_passes(expr):
    assert ok(expr), expr
    c = canon(expr)
    assert c["w_frac"] == pytest.approx(0.840) and c["x_frac"] == 0.0


def test_equivalent_spellings_canonicalise_identically():
    assert canon("crop=iw*0.84:ih*0.84:0:0") == canon("crop=iw*0.840:ih*0.840:0:0")


# ------------------------------------------------------------------ what must still be refused
@pytest.mark.parametrize("expr", [
    "crop=iw*0.700:ih*0.700:0:0",                       # unauthorized fraction
    "crop=iw*0.840:ih*0.900:0:0",                       # non-square fraction
    "crop=iw*0.840:ih*0.840:iw*0.200:0",                # origin is not the corner
    "crop=iw*0.840:ih*0.840:0:ih*0.100",                # origin is not the corner
    "crop=iw*0.840:ih*0.840:10:0",                      # absolute pixel origin
    "crop=iw*0.840:ih*0.840:iw*0.160:ih*0.160,eq=gamma=2",   # chained second filter
    "eq=gamma=2,crop=iw*0.840:ih*0.840:0:0",            # chained first filter
    "scale=1920:1080",                                  # a different transform entirely
    "crop=iw*0.840:ih*0.840:0:0;drawbox=0:0:10:10:red", # graph separator
    "movie=/etc/passwd",                                # a filter naming a file
    "crop=iw*0.840:ih*0.840",                           # incomplete geometry
    "CROP=iw*0.840:ih*0.840:0:0",                       # not the crop filter
    "crop=iw*0.840:ih*0.840:0:0 ; ",                    # trailing graph fragment
])
def test_anything_that_is_not_an_authorized_corner_crop_is_refused(expr):
    assert not ok(expr), expr
    assert canon(expr) is None


def test_altered_geometry_fails_even_one_digit_out():
    assert ok("crop=iw*0.840:ih*0.840:iw*0.160:ih*0.160")
    assert not ok("crop=iw*0.840:ih*0.840:iw*0.161:ih*0.160")


def test_an_empty_or_missing_declaration_is_not_an_authorization():
    for expr in ("", "   ", None):
        assert canon(expr) is None


# ------------------------------------------------------------------ the gate reads the parser
def test_the_lineage_gate_uses_the_parser_not_the_literal_set():
    import inspect

    from vidlore import scene_lineage_canary as C
    src = inspect.getsource(C)
    assert "not source_compare_filter_authorized(" in src, \
        "the gate must validate through the schema"
    i = src.index("selection source comparison declares an unauthorized filter")
    assert "source_compare_filter_authorized" in src[max(0, i - 400):i], \
        "the refusal must be the parser's verdict"


def test_the_parser_never_executes_the_expression():
    """It decides whether a declaration is permitted; it must not run anything."""
    import inspect

    from vidlore import scene_lineage_canary as C
    src = inspect.getsource(C.canonical_source_compare_transform)
    for forbidden in ("subprocess", "eval(", "exec(", "ffmpeg"):
        assert forbidden not in src, forbidden


def test_the_canonical_form_carries_a_versioned_schema():
    assert canon("crop=iw*0.840:ih*0.840:0:0")["schema"] == "source_compare_transform/1"
