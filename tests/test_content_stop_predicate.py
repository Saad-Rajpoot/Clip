"""Every driver that can offer a review draft must agree on what earns one.

The portal and tools/resume_job.py each answered "is this failure eligible for an automatic review
draft?" on their own, with `isinstance(exc, NonRetryableBuildError)`. Both therefore missed the
semantic-recovery pagination guard, which is every bit a content verdict — beats fail the
publication contract and the repair walk is out of pages — but is raised as a PipelineError by
machinery that lives in orchestrate, not verify.

Job 0321078108 died on exactly that after 6h20m with the portal's auto-review sitting right there,
ineligible, and job f840b0cb49 died the same way the day before.

One predicate now, `verify.is_content_stop`, used by both. Integrity failures and infrastructure
failures stay out of it.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from vidlore.clipstudio.verify import (NonRetryableBuildError, VisionBackendError, is_content_stop)
from vidlore.clipstudio.orchestrate import PipelineError, _pagination_terminal_error


ROOT = Path(__file__).resolve().parents[1]


def _terminal(reason="finite_guard:14:99,121"):
    return _pagination_terminal_error(
        {"status": "current", "deferred": [99, 121],
         "receipt": {"terminal_reason": reason, "max_pages": 14}})


def test_the_pagination_guard_now_earns_a_review_draft():
    assert is_content_stop(_terminal()) is True
    assert is_content_stop(_terminal("no_progress:99,121")) is True


def test_a_content_verdict_still_earns_one():
    assert is_content_stop(NonRetryableBuildError("footage gap", kind="selection_relevance")) is True
    assert is_content_stop(NonRetryableBuildError("held frame", kind="scene_lineage")) is True


def test_infrastructure_never_earns_one():
    """A dead vision backend needs the backend back, not a draft."""
    assert is_content_stop(VisionBackendError("quota exhausted", kind="billing")) is False
    assert is_content_stop(VisionBackendError("no key", kind="auth")) is False


def test_an_untyped_technical_fault_never_earns_one():
    """The malformed-receipt / unreadable-audit guards raise bare PipelineErrors on purpose."""
    assert is_content_stop(PipelineError("pagination terminal receipt is malformed")) is False
    assert is_content_stop(RuntimeError("ffmpeg exited 1")) is False
    assert is_content_stop(OSError("disk full")) is False


def test_an_integrity_kind_on_a_foreign_error_is_not_smuggled_in():
    """Only selection_relevance crosses the class boundary; integrity kinds must not."""
    err = PipelineError("lineage broke")
    err.kind = "scene_lineage"
    assert is_content_stop(err) is False


def test_both_drivers_use_the_one_predicate():
    from vidlore.clipstudio import web as W
    portal = inspect.getsource(W)
    resume = (ROOT / "tools" / "resume_job.py").read_text(encoding="utf-8")
    for name, src in (("portal", portal), ("resume_job", resume)):
        assert "is_content_stop" in src, f"{name} does not use the shared predicate"
        assert "isinstance(_ce, _NRB2)" not in src, f"{name} still has its own copy"
        assert "isinstance(e, NonRetryableBuildError)" not in src, f"{name} still has its own copy"
