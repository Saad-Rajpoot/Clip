"""An unreachable provider is unreachable for every query, not for each one separately.

Caught live, mid-render, on job d835faa83e: the process had sat on ONE log line for two hours at 0%
CPU with a single socket in SYN_SENT to 207.241.224.2 — www.archive.org. From that machine
archive.org never answered (curl: HTTP 000 after 12s); YouTube answered in 0.7s.

Every query paid archive's 20-second timeout TWICE — once in the parallel fan-out, then again in
the SERIAL bounded-backoff retry, because a timeout is retryable. A backfill round of ~235 queries
spends about 98 minutes that way, on a host that was never going to answer. It is also the better
half of the 8.1 hours the previous run lost inside recovery/search, which had been put down to
provider throttling.

So a provider that fails `_PROVIDER_DOWN_AFTER` times in a row is skipped for the rest of that
search. Scope is deliberately ONE discover_sources() call: the next search re-probes, so a network
that comes back is used again — no state to reset, nothing to leak between renders in a long-lived
portal.

This weakens nothing. archive.org is a bonus pool, the required-query contract is satisfied by ANY
provider answering, and the provider is named out loud when it is dropped.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio import discover as D
from vidlore.clipstudio.config import ClipConfig


class _Analysis:
    movie_title = "Game of Thrones"
    video_type = "multi_scene"
    anchor_scenes = []
    actors = []
    characters = []
    keywords = []
    topic = "t"


def _wire(monkeypatch, n_queries=40, *, ar_status=None, yt_status=None):
    qs = [f"q{i}" for i in range(n_queries)]
    monkeypatch.setattr(D, "build_queries", lambda _a, _s: list(qs))
    monkeypatch.setattr(D, "anchor_queries", lambda _a, _s: [])
    monkeypatch.setattr("time.sleep", lambda *_a: None)          # retry backoff, not a real wait
    calls = {"yt": 0, "ar": 0}

    def yt(_q, _n):
        calls["yt"] += 1
        return [], (yt_status or D.STATUS_EMPTY)

    def ar(_q, _n):
        calls["ar"] += 1
        return [], (ar_status or D.STATUS_EMPTY)

    monkeypatch.setattr(D, "_ytsearch_ex", yt)
    monkeypatch.setattr(D, "_archive_search_ex", ar)
    return calls, qs


def _run(monkeypatch, **kw):
    lines: list = []
    D.discover_sources(_Analysis(), ClipConfig(), segments=None, progress=lines.append, **kw)
    return lines


# ---------------------------------------------------------------- the two-hour stall
def test_a_dead_provider_is_called_a_bounded_number_of_times(monkeypatch):
    """THE test. 40 queries × 2 calls each (fan-out + serial retry) is what cost two hours."""
    calls, _ = _wire(monkeypatch, 40, ar_status=D.STATUS_TIMEOUT)
    _run(monkeypatch)
    assert calls["ar"] <= D._PROVIDER_DOWN_AFTER + 2, \
        f"archive was called {calls['ar']} times against a host that never answers"
    assert calls["yt"] == 40, "the working provider must still answer every query"


def test_the_dead_provider_is_named_out_loud_once(monkeypatch):
    _wire(monkeypatch, 40, ar_status=D.STATUS_TIMEOUT)
    lines = _run(monkeypatch)
    said = [ln for ln in lines if "unreachable" in ln]
    assert len(said) == 1, "an outage must be reported, and reported once"
    assert "archive.org" in said[0] and "timeout" in said[0]


def test_a_healthy_provider_is_never_dropped(monkeypatch):
    calls, _ = _wire(monkeypatch, 30)
    lines = _run(monkeypatch)
    assert calls["ar"] == 30 and calls["yt"] == 30
    assert not [ln for ln in lines if "unreachable" in ln]


def test_one_flaky_query_does_not_condemn_a_provider(monkeypatch):
    """A single failure among successes must not trip it — the counter is CONSECUTIVE."""
    seq = {"n": 0}

    def flaky(_q, _n):
        seq["n"] += 1
        return [], (D.STATUS_TIMEOUT if seq["n"] % 4 == 0 else D.STATUS_EMPTY)

    _wire(monkeypatch, 30)
    monkeypatch.setattr(D, "_archive_search_ex", flaky)
    lines = _run(monkeypatch)
    assert not [ln for ln in lines if "unreachable" in ln]


# ---------------------------------------------------------------- nothing is weakened
def test_the_required_contract_is_still_satisfied_by_the_live_provider(monkeypatch):
    """archive down + youtube answering = a conclusive answer. Raising here would turn an outage
    on a bonus pool into a failed render."""
    _wire(monkeypatch, 12, ar_status=D.STATUS_TIMEOUT)
    D.discover_sources(_Analysis(), ClipConfig(), segments=None,
                       required_queries=["q0"], progress=None)      # must not raise


def test_both_providers_down_is_still_a_failure(monkeypatch):
    """The breaker must not swallow a real, total outage."""
    _wire(monkeypatch, 12, ar_status=D.STATUS_TRANSPORT, yt_status=D.STATUS_TRANSPORT)
    with pytest.raises(D.TargetedDiscoveryTechnicalError):
        D.discover_sources(_Analysis(), ClipConfig(), segments=None,
                           required_queries=["q0"], progress=None)


def test_the_breaker_does_not_survive_the_call(monkeypatch):
    """Per-search scope: a network that recovers must be used again, with no reset step and no
    state leaking between renders in a long-lived portal."""
    calls, _ = _wire(monkeypatch, 20, ar_status=D.STATUS_TIMEOUT)
    _run(monkeypatch)
    first = calls["ar"]
    _run(monkeypatch)
    assert calls["ar"] > first, "the next search must re-probe the provider"
