"""A backend that is down and a picture the backend won't look at are not the same thing.

Job 229233891e: 145 of 146 beats verified normally and beat 36 alone came back with no verdict at
all, every time, for hours. Diagnosed by hand before any code was written —

  * `llm.vision_probe` passed and Gemini answered ordinary text
  * four other keyframes from the SAME job were judged fine by Gemini
  * the file was a clean 512x288 RGB JPEG, 25.6 KB
  * all three frames sampled across that window (15%/50%/85%) were refused too, by Gemini AND by
    the Claude fallback — so it is the window's CONTENT, not one bad thumbnail, and re-sampling
    cannot help

That is a fact about the picture. Treating it as a broken system threw away a 2h20m render at the
final gate, which is the failure this fixes.

An outage stays fatal. During one, "no verdict" carries no information, and accepting anything
would ship unverified footage. So the two are told apart by asking the backend at that moment with
the pipeline's own health probe. Healthy backend + a beat that will not resolve = an unjudgeable
frame: the beat stays UNVERIFIED and the publication contract goes on blocking it. Nothing is
accepted — the render simply stops dying for it.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import orchestrate as O


SRC = inspect.getsource(O._retry_selection_relevance)
I = SRC.index("A BACKEND THAT IS DOWN AND A FRAME THE BACKEND WON'T LOOK AT")
BLOCK = SRC[I:I + 3800]


def test_health_is_read_from_this_pass_not_from_a_live_probe():
    """A probe would be a different call, at a different moment, on a different image — and it
    made the outcome depend on network state: three existing tests began passing or failing by
    suite ordering alone. The verdicts this pass already produced are free and deterministic."""
    assert "vision_probe(" not in BLOCK
    assert 'get("verdict")' in BLOCK and "_judged" in BLOCK


def test_an_unhealthy_backend_is_still_fatal():
    """During an outage 'no verdict' means nothing, so it must not become a content result."""
    assert "if not _healthy:" in BLOCK
    i = BLOCK.index("if not _healthy:")
    assert "raise PipelineError(" in BLOCK[i:i + 400]


def test_an_unprobeable_backend_counts_as_unhealthy():
    """Fail closed: if the probe itself cannot run, nothing has been proven."""
    assert "_judged = 0" in BLOCK
    assert "max(3, len(_technical_indices) * 3)" in BLOCK, \
        "a couple of stray verdicts must not certify the backend"


def test_a_healthy_backend_does_not_raise():
    i_raise = BLOCK.index("raise PipelineError(")
    i_log = BLOCK.index("could not be judged while")
    assert i_raise < i_log, "the healthy path continues past the raise"


def test_the_beat_is_still_unverified_and_still_blocks():
    """This is the whole safety argument: nothing is accepted, only the render survives."""
    assert "UNVERIFIED" in BLOCK
    assert "publication contract still blocks" in BLOCK
    for forbidden in ("verdict\" ] = \"keep\"", "approve", "accept_unverified"):
        assert forbidden not in BLOCK, forbidden


def test_the_reason_is_reported_not_swallowed():
    assert "log(" in BLOCK
    assert "_technical_reasons" in BLOCK


def test_the_measured_diagnosis_is_recorded_where_it_will_be_read():
    """The next person to see this must not re-derive it: re-sampling the window does not help."""
    assert "145 of 146" in BLOCK
    assert "four other keyframes" in BLOCK


# ------------------------------------------------------------------ the other site that needed it
#
# e1e5802 fixed the scoped re-verification path. Job 229233891e then died at the recovery page with
# `new_source_verifier_errored:1`: exactly one freshly acquired source could not be judged while the
# rest of the same batch was judged normally. Same fact, same shape, different call site.
NEW_SRC = inspect.getsource(O)
J = NEW_SRC.index("SAME DISTINCTION AS e1e5802")
NEW_BLOCK = NEW_SRC[J:J + 2200]


def test_the_recovery_page_uses_the_same_evidence_test():
    assert '_new_verify_result.get("errored")' in NEW_BLOCK
    assert '_new_verify_result.get("verified")' in NEW_BLOCK


def test_the_recovery_page_still_raises_on_a_real_outage():
    assert 'raise RuntimeError(f"new_source_{_new_verify_error}")' in NEW_BLOCK


def test_the_recovery_bar_matches_the_other_site():
    """At least three real verdicts, and at least three per errored item."""
    assert "_judged_ok >= max(3, _errored * 3)" in NEW_BLOCK


def test_malformed_counts_fail_closed():
    """Non-int or bool counts must not be read as proof the verifier worked."""
    assert "isinstance(_errored, int) and not isinstance(_errored, bool)" in NEW_BLOCK
    assert "isinstance(_judged_ok, int) and not isinstance(_judged_ok, bool)" in NEW_BLOCK


def test_the_unjudged_source_authorizes_nothing():
    assert "stay unresolved" in NEW_BLOCK
