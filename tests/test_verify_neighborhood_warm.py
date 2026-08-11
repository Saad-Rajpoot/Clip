"""The ±6-shot bench asks its questions together and decides them one at a time — identically.

Measured on job 218acdfe10's resume, this rung was the single most expensive thing in the render:
65 benches, median 73s each, forty of them over a minute — **68 minutes** spent asking twelve
independent questions one after another at ~5s of network latency apiece, while the very same
stage's prefetch pool answered 230 beats in 286s. The rung could not be staged with the existing
waves because its candidate pool is derived INSIDE the decision loop, from `strict_tried`.

So the pool is warmed concurrently at the moment it is built, and the serial walk then hits cache.
Everything this file exists to prove is about what did NOT change:

  * the same beat ends on the same shot with the same verdict, warm or serial;
  * no image is judged twice — the walk really is reading the cache, not re-asking;
  * a warm that FAILS is not cached, so that one candidate is simply asked again by the walk and
    the outcome is unchanged (this is the failure mode that must never poison a decision);
  * the kill switch and the serial (`workers=1`) path restore the exact previous behaviour.

Window-QC, the quote lock, the reuse cap, `strict_window_verdict` and the circuit breaker all
still run in `_try_promote_inner`, untouched and in the same order.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_strict_scene_neighborhood import _cand, _fixture      # noqa: E402

from vidlore.clipstudio import image_fallback as IF             # noqa: E402
from vidlore.clipstudio import verify as V                      # noqa: E402
from vidlore.clipstudio.config import ClipConfig                # noqa: E402

# The exact moment is FOURTH in the rung — the shape of the measured production failure, and the
# only shape where "stop at the first accept" and "warm them all" can disagree.
_POOL = [("target", 6), ("target", 7), ("target", 8), ("target", 11)]
_WINNER = "target_11.jpg"


def _run(tmp_path, monkeypatch, *, workers: str, warm: str = "1", explode=(), pool=None):
    """Drive verify_and_repair over the neighborhood rung; return (summary, sel, calls)."""
    proj, seg, sel, get_shot = _fixture(tmp_path)
    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates",
                        lambda *_a, **_k: [_cand(s, i) for s, i in (pool or _POOL)])
    monkeypatch.setattr(V, "_shot_lookup", lambda _proj: get_shot)
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    calls, boom = [], list(explode)

    def verdict(path, *_args, **_kwargs):
        name = Path(path).name
        calls.append(name)
        if name in boom:
            boom.remove(name)                       # fail ONCE, exactly like a transport blip
            raise RuntimeError("transport blip")
        keep = name == _WINNER
        return {"verdict": "keep" if keep else "replace", "matches_narration": keep,
                "correct_subject_visible": keep, "wrong_subject_visible": False,
                "contradicts_narration": False, "specific_enough": keep, "quality_ok": True,
                "confidence": 0.95, "reason": "exact" if keep else "wrong moment"}

    monkeypatch.setattr(V, "verify_frame", verdict)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", workers)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_NEIGHBORHOOD", warm)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(), NS(anthropic_model="m", anthropic_key="k"),
        max_replacements=1, materialize_promotions=False, persist_project=False)
    return summary, sel, calls


# ---------------------------------------------------------------- the decision does not move
def test_the_warm_bench_reaches_the_same_shot_as_the_serial_bench(tmp_path, monkeypatch):
    """THE test. Concurrency may buy wall-clock; it may not buy a different video."""
    serial = _run(tmp_path / "a", monkeypatch, workers="1")
    warmed = _run(tmp_path / "b", monkeypatch, workers="8")

    for (summary, sel, _), label in ((serial, "serial"), (warmed, "warmed")):
        assert summary["replaced"] == 1, label
        assert (sel.source_id, sel.shot_index) == ("target", 11), label
        assert sel.verifier.get("verdict") == "keep", label
        assert not sel.verifier.get("downgraded"), f"{label}: exact rescue relabelled contextual"

    assert set(serial[2]) == set(warmed[2]), \
        "the warmed run must ask the SAME questions, not different ones"


def _bench(calls):
    """Only the ±6-shot candidates.

    The primary shot's image legitimately appears twice in a run — once for the STRICT question
    (is_specific=True) and once for the lenient generic-filler re-ask (is_specific=False). They are
    two different questions about the same pixels, keyed differently on purpose, so counting
    filenames across the whole run would conflate them and this file would be asserting a fiction.
    Verified by diffing the two fingerprints: `is_specific` is the only field that differs.
    """
    return [c for c in calls if c.startswith("target_")]


def test_the_walk_reads_the_cache_instead_of_asking_again(tmp_path, monkeypatch):
    """If the warm keyed its questions differently, every candidate would be paid for twice —
    slower AND dearer than before. No bench candidate may be judged twice."""
    _s, _sel, calls = _run(tmp_path, monkeypatch, workers="8")
    bench = _bench(calls)
    dupes = [c for c in set(bench) if bench.count(c) > 1]
    assert not dupes, f"asked twice (warm keyed differently from the serial ask): {dupes}"


def test_the_warm_really_runs_and_what_it_costs(tmp_path, monkeypatch):
    """Non-vacuous proof, and an honest statement of the trade.

    Put the winner FIRST. The serial walk then accepts on candidate #1 and never asks the other
    three; the warm asks all four before the walk starts. So the call counts diverge — which is
    exactly what proves the warm ran, and exactly what it costs: candidates the walk would have
    skipped. Measured on the render that motivated this, the walk EXHAUSTED every candidate in 47
    of 65 benches, so this best case for serial is the rare one.
    """
    winner_first = [("target", 11), ("target", 6), ("target", 7), ("target", 8)]
    serial = _run(tmp_path / "a", monkeypatch, workers="1", pool=winner_first)
    warmed = _run(tmp_path / "b", monkeypatch, workers="8", pool=winner_first)

    assert _bench(serial[2]) == [_WINNER], "serial stops at the first accept"
    assert len(_bench(warmed[2])) == 4, "the warm asks the whole bench up front — or it never ran"
    assert set(_bench(warmed[2])) == {f"target_{i:02d}.jpg" for _s, i in winner_first}
    # and the extra questions buy nothing but time: the decision is the same one
    for summary, sel, _ in (serial, warmed):
        assert summary["replaced"] == 1
        assert (sel.source_id, sel.shot_index) == ("target", 11)


def test_a_failed_warm_costs_one_retry_and_nothing_else(tmp_path, monkeypatch):
    """A warm that raises must not be cached — the walk asks that candidate again and decides
    exactly as it always did. This is the failure mode that must never poison a verdict."""
    summary, sel, calls = _run(tmp_path, monkeypatch, workers="8", explode=[_WINNER])
    assert calls.count(_WINNER) == 2, "the failed warm must be re-asked by the serial walk"
    assert summary["replaced"] == 1
    assert (sel.source_id, sel.shot_index) == ("target", 11)
    assert sel.verifier.get("verdict") == "keep"


# ---------------------------------------------------------------- it can be turned off
def test_the_kill_switch_restores_the_fully_serial_bench(tmp_path, monkeypatch):
    off = _run(tmp_path / "a", monkeypatch, workers="8", warm="0")
    serial = _run(tmp_path / "b", monkeypatch, workers="1")
    assert off[0]["replaced"] == serial[0]["replaced"] == 1
    assert (off[1].source_id, off[1].shot_index) == (serial[1].source_id, serial[1].shot_index)
    # the walk stops at its first accept, so with the warm off the bench is asked candidate by
    # candidate in the same order as the fully-serial run, and stops in the same place
    assert _bench(off[2]) == _bench(serial[2]), \
        "with the warm off the bench call sequence must be byte-identical to serial"


def test_one_worker_never_warms(tmp_path, monkeypatch):
    """`=1` is the fully-serial contract the breaker and outage suites depend on: the walk stops
    at its first accept, so nothing beyond the winner is ever asked."""
    _s, _sel, calls = _run(tmp_path, monkeypatch, workers="1")
    assert len(calls) == len(set(calls)), "serial must not double-ask either"
    assert calls[-1] == _WINNER, "the serial walk stops at the first accepted candidate"
    assert "target_11.jpg" in calls


# ---------------------------------------------------------------- structure
def test_the_warm_runs_before_the_walk_and_mutates_nothing():
    src = inspect.getsource(V.verify_and_repair)
    warm_at = src.index("_warm_scene_neighborhood(_npool, seg, _exact)")
    walk_at = src.index("_try_promote(downgrade=False, pool=_npool", warm_at)
    assert warm_at < walk_at, "the warm must precede the decision walk"

    body = src[src.index("def _warm_scene_neighborhood"):]
    body = body[:body.index("\n    for sel in proj.selections:")]
    # the docstring NAMES the things it promises not to touch, so check the CODE
    code = body[body.index('"""', body.index('"""') + 3) + 3:]
    for forbidden in ("strict_tried", "failed_wins", "_reuse[", "sel.source_id =",
                      "replaced +=", "swapped ="):
        assert forbidden not in code, f"the warm must not touch {forbidden}"
    assert "_look_scope[\"on\"] = True" in body, \
        "a strict rung asks the named-look question; warming without it pays twice"
    assert "finally:" in body and "_prior_scope_n" in body, "the look scope must be restored"


def test_the_breaker_and_the_switch_are_both_honoured():
    body = inspect.getsource(V.verify_and_repair)
    body = body[body.index("def _warm_scene_neighborhood"):]
    assert "_breaker_open" in body[:2000], "a warm must not run against a downed backend"
    assert "VERIFIER_BREAKER_TRIP" in body[:4000], "repeated transport failures must abort the warm"
    assert "VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_NEIGHBORHOOD" in body[:2000]


@pytest.mark.parametrize("turbo,expected", [("0", 4), ("1", 12)])
def test_the_prefetch_pool_widens_under_turbo(monkeypatch, turbo, expected):
    from vidlore.clipstudio.config import verify_prefetch_workers
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", turbo)
    assert verify_prefetch_workers() == expected


def test_an_explicit_worker_count_always_wins(monkeypatch):
    from vidlore.clipstudio.config import verify_prefetch_workers
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    assert verify_prefetch_workers() == 1, "the serial contract must remain reachable"
