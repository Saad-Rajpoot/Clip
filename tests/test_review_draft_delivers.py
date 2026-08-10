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
from vidlore.clipstudio import release_policy as RP
from vidlore.clipstudio.verify import NonRetryableBuildError


def _exc(kind: str, msg: str = "gate failed"):
    """A terminal exception carrying exactly the identity the classifier reads."""
    return NonRetryableBuildError(msg, kind=kind)


@pytest.fixture
def review(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "warn")


@pytest.fixture
def production(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block")


# ------------------------------------------------------------------ production is unchanged
@pytest.mark.parametrize("kind", [
    "native_resolution", "image_semantic", "caption_readability", "scene_lineage",
])
def test_production_delivers_nothing_imperfect(production, kind):
    assert B.content_defect_is_deliverable(_exc(kind)) is False


def test_production_is_the_default_when_the_env_is_unset(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", raising=False)
    assert B.review_draft_mode() is False
    assert B.content_defect_is_deliverable(_exc("selection_relevance")) is False


# ------------------------------------------------------------------ integrity stays fatal in BOTH
@pytest.mark.parametrize("kind", [
    "scene_lineage",               # broken/unprovable ownership of an aired frame
    "selection_relevance_audit",   # the audit artifact itself is untrustworthy
    "voiceover_alignment",         # narration cannot be placed against the timeline
    "montage", "animal",
])
def test_an_integrity_fault_is_never_deliverable(review, kind):
    """'This may not be our footage' is not a quality defect at any strictness."""
    assert B.content_defect_is_deliverable(_exc(kind)) is False


@pytest.mark.parametrize("kind", ["down", "billing", "auth", "native_resolution_probe",
                                  "breakout_qa_probe", "final_ad_unverified"])
def test_a_technical_fault_is_never_deliverable(review, kind):
    """'We could not measure it' is never content, even when the thing measured IS content."""
    assert B.content_defect_is_deliverable(_exc(kind)) is False


# ------------------------------------------------------------------ quality defects are deliverable
@pytest.mark.parametrize("kind", sorted(RP.CONTENT_KINDS))
def test_a_quality_defect_reaches_the_reviewer(review, kind):
    assert B.content_defect_is_deliverable(_exc(kind)) is True


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
def test_classification_is_by_kind_not_by_message_text():
    """Gate-name PREFIX matching replaced keyword matching, and was itself replaced: prose gets
    reworded (by the time this was audited, 'caption readability:' was a registered prefix no gate
    raised any more) and prefixes cannot be ENUMERATED, so a gate that forgot to declare one looked
    exactly like a gate with nothing to declare. `kind` is a machine field a census test can walk."""
    src = inspect.getsource(B.content_defect_is_deliverable)
    assert "startswith" not in src
    assert "_DELIVERABLE_GATE_PREFIXES" not in inspect.getsource(B)
    assert list(inspect.signature(B.content_defect_is_deliverable).parameters) == ["exc"]


def test_a_gate_nobody_has_classified_yet_is_fatal(review):
    """A gate added tomorrow must fail closed until someone opts it in on purpose."""
    assert B.content_defect_is_deliverable(_exc("a_gate_invented_next_year")) is False
    assert B.content_defect_is_deliverable(_exc("")) is False
    assert B.content_defect_is_deliverable(RuntimeError("brand new gate: something broke")) is False
    assert B.content_defect_is_deliverable(None) is False


def test_the_deliverable_list_is_short_and_explicit():
    """Every content kind is registered on purpose, and every registered kind is really used."""
    assert 0 < len(RP.CONTENT_KINDS) <= 12
    for k in RP.CONTENT_KINDS:
        assert RP.KIND_CLASS[k] == "content"
    assert RP.CONTENT_KINDS.isdisjoint(
        {k for k, v in RP.KIND_CLASS.items() if v in ("integrity", "technical")})


def test_a_resolution_defect_is_not_an_ownership_defect(review):
    """The exact message that broke the FIRST classifier — the word 'owner' in a resolution gate.
    Under kind-based routing the message cannot mislead anyone, which is the point."""
    e = _exc("native_resolution",
             "image native-resolution gate: verified still is 512x288; "
             "a real 1280x720 owner is required")
    assert B.content_defect_is_deliverable(e) is True
    assert RP.gate_class(e) == "content"
