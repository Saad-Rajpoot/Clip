"""A review draft exists to be looked at. It must not be withheld over one imperfect beat.

Job 0ca9dc4c2f died at five separate end-of-build gates in a row, each throwing away a fully
rendered video. Four were bugs. The fifth was not a bug at all — the caption readability gate had
simply never been told that a review draft is allowed to be imperfect, while the footage,
unverified-exact and black-frame gates all already knew. Counting afterwards found the same
omission in twelve more content-quality gates, and the worst of them sit in the still passes, which
is exactly where a draft's unresolved beats end up.

The distinction is made once, in `content_defect_is_deliverable`, and it is NOT "how bad is it":

  INTEGRITY — the artifact is wrong or unprovable. Wrong owner, broken lineage, a frame that changed
  under the verifier, a malformed binding. These say "this may not be our footage at all" and stay
  fatal in BOTH modes.

  CONTENT QUALITY — the artifact is honestly ours and imperfect. A sub-HD still, an unresolved beat,
  a verdict we could not obtain. These are what a human opens a draft to see.

Production ('block', the default) is unchanged: everything here is still fail-closed.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import build as B


@pytest.fixture
def review(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "warn")


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block")


# ------------------------------------------------------------------ production is unchanged
@pytest.mark.parametrize("reason", [
    "image native-resolution gate: verified still is 512x288",
    "image semantic gate: beat 12 has an unresolved question",
    "caption readability: cue 163 at 24.53 CPS",
    "image-lineage gate: indexed owner changed while its still was judged",
])
def test_production_delivers_nothing_imperfect(production, reason):
    assert B.content_defect_is_deliverable(reason) is False


def test_production_is_the_default_when_the_env_is_unset(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", raising=False)
    assert B.review_draft_mode() is False
    assert B.content_defect_is_deliverable("anything at all") is False


# ------------------------------------------------------------------ integrity stays fatal in BOTH
@pytest.mark.parametrize("reason", [
    "image-lineage gate: source-frame ownership metadata is partial",
    "image-lineage gate: indexed owner 'x' changed while its still was judged",
    "scene-lineage manifest: aired input has no verified owner",
    "decoded-frame canary: delivered frame does not match its expected moment",
    "image-lineage gate: source-frame shot id 'abc' is malformed",
    "image-lineage gate: native still bytes vanished before semantic verification",
    "provenance: sha256 mismatch on the owner artifact",
    "index binding is unreadable",
])
def test_an_integrity_fault_is_never_deliverable(review, reason):
    """'This may not be our footage' is not a quality defect at any strictness."""
    assert B.content_defect_is_deliverable(reason) is False


# ------------------------------------------------------------------ quality defects are deliverable
@pytest.mark.parametrize("reason", [
    "image native-resolution gate: verified still is 512x288; a real 1280x720 owner is required",
    "image semantic gate: beat 12 has an unresolved question",
    "image semantic gate: native still failed strict verification",
    "image native-resolution gate: extracted still is 640x360",
])
def test_a_quality_defect_reaches_the_reviewer(review, reason):
    assert B.content_defect_is_deliverable(reason) is True


# ------------------------------------------------------------------ the still pass degrades per beat
def test_the_still_preflight_degrades_one_beat_instead_of_the_render():
    src = inspect.getsource(B)
    i = src.index("if not content_defect_is_deliverable(_img_pf_exc):")
    tail = src[i:i + 1400]
    assert "raise" in tail, "an undeliverable fault must still stop the render"
    assert "_review_draft.append(" in tail, "a delivered defect must be reported"
    assert '_sel_img_pf.image_path = ""' in tail, "the unusable still is dropped, not aired"


def test_the_failure_is_recorded_before_it_is_forgiven():
    src = inspect.getsource(B)
    i = src.index("if not content_defect_is_deliverable(_img_pf_exc):")
    head = src[max(0, i - 900):i]
    assert "_image_lineage_failures.append(" in head
    assert "_persist_image_lineage_audit(" in head


def test_the_relaxed_invariant_is_named_not_hidden():
    """It knowingly relaxes the aired-still rule; the code must say so where it does it."""
    src = inspect.getsource(B)
    i = src.index("if not content_defect_is_deliverable(_img_pf_exc):")
    assert "semantic replacement" in src[i:i + 1600]


# ------------------------------------------------------------------ what must never be forgiven
def test_a_missing_or_unalignable_voiceover_still_fails(review):
    """A render once shipped 20 minutes with captions and music and NO narration. That is not a
    'small issue' a reviewer can look past — it is the wrong deliverable."""
    src = inspect.getsource(B)
    for msg in ("uploaded voiceover could not produce a complete positive-duration word",
                "voiceover alignment failed"):
        i = src.index(msg)
        window = src[max(0, i - 400):i]
        assert "content_defect_is_deliverable" not in window, msg


def test_structural_caption_faults_are_not_covered_by_this_relaxation():
    """Backwards/overlapping cues and captions over a breakout's real audio make the subtitle file
    wrong, not merely hard to read — the caption gate keeps them fatal in both modes."""
    from vidlore import captions as C
    src = inspect.getsource(C)
    assert "review" in src.lower()


# ------------------------------------------------------------------ unknown gates fail closed
#
# The first classifier matched keywords in the message and mis-filed "verified still is 512x288;
# a real 1280x720 owner is required" as an ownership fault, because the word "owner" happened to
# appear. Gates name themselves, and that name is the stable fact.
def test_classification_is_by_gate_name_not_by_keywords():
    src = inspect.getsource(B.content_defect_is_deliverable)
    assert "startswith" in src
    assert "_DELIVERABLE_GATE_PREFIXES" in src


def test_a_gate_nobody_has_classified_yet_is_fatal(review):
    """A gate added tomorrow must fail closed until someone opts it in on purpose."""
    assert B.content_defect_is_deliverable("brand new gate: something went wrong") is False
    assert B.content_defect_is_deliverable("") is False
    assert B.content_defect_is_deliverable(None) is False


def test_the_deliverable_list_is_short_and_explicit():
    assert len(B._DELIVERABLE_GATE_PREFIXES) <= 5
    for p in B._DELIVERABLE_GATE_PREFIXES:
        assert p.endswith(":"), p


def test_a_resolution_defect_is_not_an_ownership_defect(review):
    """The exact message that broke the first classifier."""
    assert B.content_defect_is_deliverable(
        "image native-resolution gate: verified still is 512x288; "
        "a real 1280x720 owner is required") is True
