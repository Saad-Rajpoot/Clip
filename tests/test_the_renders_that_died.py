"""The four portal renders that died, pinned by name.

This file is the regression bar for one promise: a portal render must produce a video. Each case
below is a real job that burned hours and ended with no file, taken from its own
output/incident_report.json. If one of these ever goes red again, a render is about to be lost.

    0ca9dc4c2f   x3   PipelineError  semantic scoped re-verification remained technically
                                     inconclusive for beat(s) [88, 89]
    f840b0cb49        PipelineError  semantic recovery pagination reached its finite 16-page guard
    0321078108        PipelineError  semantic recovery pagination reached its finite 14-page guard
    229233891e        RuntimeError   source fingerprint unavailable before indexing <source>

The first three are CONTENT verdicts: beats failed the semantic publication contract and the repair
machinery had no move left. The project on disk still held a complete set of selections, so a marked
draft was achievable every time — the render died anyway because the error carried no identity and
the drivers read "no identity" as "crash".

The fourth is deliberately NOT fixed. A source whose fingerprint cannot be read before indexing is a
TECHNICAL fault; indexing footage we cannot identify is how unprovable material reaches a video, and
"we could not measure it" is never content. It stays fatal, and this file says so on purpose.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import release_policy as RP
from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio.verify import is_content_stop


def test_f840b0cb49_and_0321078108_pagination_exhaustion():
    """Two renders, 8h12m and 6h20m, both killed by the repair walk running out of pages."""
    for pages, deferred in ((16, [129, 130, 141]), (14, [99, 121, 123])):
        err = O._pagination_terminal_error({
            "status": "current", "deferred": deferred,
            "receipt": {"terminal_reason": f"finite_guard:{pages}:x", "max_pages": pages}})
        assert err is not None
        assert RP.gate_class(err) == "content"
        assert is_content_stop(err) is True, f"the {pages}-page guard would kill a render again"


def test_0ca9dc4c2f_inconclusive_reverification():
    """Died here three times. The raise is inside a long function, so this pins its shape: the
    content kind is attached, and ONLY inside the guard that proves the backend answered."""
    src = inspect.getsource(O._retry_selection_relevance)
    i = src.index("if not _healthy:")
    block = src[i:i + 2200]
    assert "if _judged > 0:" in block, "the outage guard is gone"
    j = block.index("if _judged > 0:")
    assert '_inconclusive.kind = "selection_relevance"' in block[j:j + 200], \
        "the kind must be attached only when the backend demonstrably answered"
    assert "raise _inconclusive" in block


def test_0ca9dc4c2f_a_total_outage_is_still_infrastructure():
    """The other half of the same fork. Zero verdicts is positive evidence of an outage, and an
    outage needs the backend restored — not a draft full of beats nobody looked at."""
    err = O.PipelineError("semantic scoped re-verification remained technically inconclusive")
    assert RP.gate_class(err) == "technical" or RP.gate_class(err) == "integrity"
    assert is_content_stop(err) is False


def test_229233891e_unreadable_fingerprint_stays_fatal():
    """Not every render death is a bug to route around. Indexing footage we cannot identify is how
    unprovable material reaches a video."""
    err = RuntimeError("source fingerprint unavailable before indexing small_council_meeting__x")
    assert RP.gate_class(err) == "integrity"
    assert is_content_stop(err) is False


def test_both_drivers_would_now_offer_the_draft():
    """The portal and the CLI resume both gate on the same predicate; neither may drift again."""
    from pathlib import Path
    from vidlore.clipstudio import web as W
    portal = inspect.getsource(W)
    resume = (Path(__file__).resolve().parents[1] / "tools" / "resume_job.py").read_text("utf-8")
    for name, src in (("portal", portal), ("resume_job", resume)):
        assert "is_content_stop" in src, f"{name} no longer uses the shared predicate"
