"""A query answered once in a render is not asked again.

Job d835faa83e spent **8.1 of its 22.7 hours inside `recovery/search`**. Fourteen bounded-recovery
rounds, each rebuilding the identical query list — 54 queries for 8 unresolved beats, 25 of them
the anchor queries the main discovery had already run — median 45 minutes per round, worst 77. The
main discovery had answered 200 queries in five minutes; by the fourteenth round the provider was
throttling every repeat and each one went through the bounded serial retry.

None of it could have helped. A repeated query returns the same URLs, and the recovery call site
filters those against `have_urls` before it does anything with them, which is why round after round
logged "targeted rediscovery found no NEW source".

The measurement that found this came from instrumentation added one commit earlier — before it,
those fourteen rounds were 841-second silences and the guess (written down in that commit) was that
the verify bench dominated recovery. It did not: `recovery/reverify` totalled 1.8 minutes.

What must stay true, and is tested here: only CONCLUSIVELY answered queries are remembered, so a
provider outage can never be memoised into a satisfied contract.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import discover as D


class _Analysis:
    movie_title = "Game of Thrones"
    video_type = "multi_scene"
    anchor_scenes = []
    actors = []
    characters = []
    keywords = []
    topic = "t"


def _harness(monkeypatch, issued, status=None):
    """Answer every query conclusively (and offline), recording what was actually issued.

    Patched at `_ytsearch_ex` / `_archive_search_ex` — the real provider entry points. An earlier
    draft patched names that do not exist, so the fixtures went to the live network and hung.
    """
    status = status or D.STATUS_EMPTY                  # "empty" IS a conclusive answer
    monkeypatch.setattr(D, "build_queries", lambda _a, _s: ["alpha", "beta", "gamma"])
    monkeypatch.setattr(D, "anchor_queries", lambda _a, _s: ["alpha"])

    def fake(q, _n):
        issued.append(q)
        return [], status

    monkeypatch.setattr(D, "_ytsearch_ex", fake)
    monkeypatch.setattr(D, "_archive_search_ex", lambda _q, _n: ([], status))
    return issued


def _call(monkeypatch, **kw):
    issued: list = []
    _harness(monkeypatch, issued)
    cfg = _cfg()
    D.discover_sources(_Analysis(), cfg, segments=None, **kw)
    return issued


def _cfg():
    from vidlore.clipstudio.config import ClipConfig
    return ClipConfig()


# ---------------------------------------------------------------- the memo
def test_the_parameter_exists_and_is_a_caller_owned_set():
    sig = inspect.signature(D.discover_sources)
    assert "searched" in sig.parameters
    assert sig.parameters["searched"].default is None, "opt-in; unchanged for every other caller"


def test_a_remembered_query_is_not_issued_again(monkeypatch):
    memo = {"alpha", "beta"}
    issued = _call(monkeypatch, searched=memo)
    assert "alpha" not in issued and "beta" not in issued, "already answered — must not be re-asked"
    assert "gamma" in issued, "an unasked query must still be asked"


def test_conclusive_answers_are_added_to_the_memo(monkeypatch):
    memo: set = set()
    _call(monkeypatch, searched=memo)
    assert {"alpha", "beta", "gamma"} <= memo, "the render must remember what it asked"


def test_without_the_memo_nothing_changes(monkeypatch):
    issued = _call(monkeypatch)
    assert {"alpha", "beta", "gamma"} <= set(issued), "the default path is untouched"


def test_the_skip_is_reported(monkeypatch, capsys):
    lines: list = []
    issued: list = []
    _harness(monkeypatch, issued)
    D.discover_sources(_Analysis(), _cfg(), segments=None, progress=lines.append,
                       searched={"alpha", "beta"})
    head = next(ln for ln in lines if ln.startswith("discover: "))
    assert "already answered this render, not re-issued" in head
    assert "2" in head


def test_a_render_with_nothing_skipped_does_not_mention_it(monkeypatch):
    lines: list = []
    issued: list = []
    _harness(monkeypatch, issued)
    D.discover_sources(_Analysis(), _cfg(), segments=None, progress=lines.append, searched=set())
    head = next(ln for ln in lines if ln.startswith("discover: "))
    assert "already answered" not in head


# ---------------------------------------------------------------- the contract still holds
def test_a_skipped_required_query_counts_as_answered(monkeypatch):
    """It WAS answered — earlier in this same render. Raising here would turn a saving into a
    spurious technical failure."""
    issued: list = []
    _harness(monkeypatch, issued)
    D.discover_sources(_Analysis(), _cfg(), segments=None,
                       required_queries=["alpha"], searched={"alpha"})   # must not raise


def test_an_outage_is_still_a_failure_and_is_never_memoised(monkeypatch):
    """The one way this could go wrong: remembering a query that never got a real answer."""
    monkeypatch.setattr(D, "build_queries", lambda _a, _s: ["alpha"])
    monkeypatch.setattr(D, "anchor_queries", lambda _a, _s: [])
    monkeypatch.setattr(D, "_ytsearch_ex", lambda _q, _n: ([], D.STATUS_TRANSPORT))
    monkeypatch.setattr(D, "_archive_search_ex", lambda _q, _n: ([], D.STATUS_TRANSPORT))
    monkeypatch.setattr("time.sleep", lambda *_a: None)   # the retry backoff, not a real wait
    memo: set = set()
    with pytest.raises(D.TargetedDiscoveryTechnicalError):
        D.discover_sources(_Analysis(), _cfg(), segments=None,
                           required_queries=["alpha"], searched=memo)
    assert "alpha" not in memo, "an unanswered query must never be remembered as answered"


# ---------------------------------------------------------------- every searcher shares one memo
def test_all_three_discovery_call_sites_share_the_render_s_memo():
    """Main discovery, the backfill pass and the recovery rounds are one render. If the main pass
    did not record its queries, recovery would re-issue all 25 anchor searches — the exact 8.1
    hours this exists to stop."""
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O)
    calls = [seg for seg in src.split("discover_sources(")[1:]]
    assert len(calls) >= 3, "expected the main, backfill and recovery searches"
    wired = sum(1 for seg in calls if "searched=" in seg[:400])
    assert wired >= 3, f"only {wired} of {len(calls)} discovery calls share the memo"
    assert src.count('proj.meta["searched_queries"]') >= 3, \
        "the memo must be persisted, or a resume starts re-asking from zero"
