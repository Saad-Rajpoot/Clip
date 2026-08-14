"""Per-beat enrichment asks its batches together, and still assembles them in order.

Measured on job f556c8c761: 224 beats is 13 batches, each a deepseek-v4-pro generation of up to
3,840 tokens taking 12-27 minutes — roughly FOUR HOURS of a render spent before a single source has
been searched for, waiting on calls that have nothing to do with each other. Every batch is built
from the same global-context header plus its own slice of beats; none reads another's answer.

What must not move, and is what this file actually guards:

  * the ORDER of `beats` — it is consumed positionally downstream, so completion order must never
    become output order;
  * the QUESTION — same prompt, same model, same token budget, same two-try retry;
  * the FALLBACK — a batch that fails twice still drops to heuristic visuals for its beats only,
    and never takes a neighbouring batch down with it.

The default width is deliberately modest. A batch that fails does not just cost time: its beats
lose their scene queries, and relevance is the thing this pipeline exists to protect. Buying speed
with rate-limit failures would be a bad trade, not a fast one.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import analyze as A


class _Beat:
    def __init__(self, i, text):
        self.index, self.text = i, text


def _beats(n):
    return [_Beat(i, f"line {i}") for i in range(n)]


_HIGH_LEVEL = ('{"movie_title":"Game of Thrones","synopsis":"s","tone":"t",'
               '"video_type":"multi_scene","actors":[],"characters":[],"anchor_scenes":[]}')


def _run(monkeypatch, n, *, fail_starts=(), record=None, delay=0.0, **_kw):
    """Drive the REAL entry point, `_llm_analyze`, with both stages mocked.

    Stage 1 (the high-level call) and stage 2 (the per-beat batches) go through the same
    `_llm.complete`; they are told apart by the "BEATS:" payload only stage 2 carries.
    """
    import json as _j
    import re as _re
    import time as _t
    from vidlore.clipstudio import llm as _real_llm

    def complete(*, system="", messages, max_tokens=1024, eng_cfg=None, model=""):
        body = messages[0]["content"]
        if "BEATS:" not in body:
            return _HIGH_LEVEL
        # the prompt carries a JSON SCHEMA example before "BEATS:", so parse what follows the
        # marker rather than the first bracket in the body
        idxs = [int(x["i"]) for x in _j.loads(body.split("BEATS:", 1)[1].strip())]
        if record is not None:
            record.append(idxs[0])
        if delay:
            _t.sleep(delay)
        if idxs[0] in fail_starts:
            return "not json at all"
        return _j.dumps([{"i": i, "scene_query": f"q{i}", "visual_policy": "generic_filler",
                          "emotion": "", "specific": False} for i in idxs])

    monkeypatch.setattr(_real_llm, "complete", complete)
    monkeypatch.setattr(_real_llm, "beat_model", lambda: "deepseek-v4-pro")
    return A._llm_analyze("script text", "topic", "Game of Thrones", _beats(n), None, None)


# ---------------------------------------------------------------- order is the contract
def test_beats_come_back_in_script_order_not_completion_order(monkeypatch):
    """THE test. Downstream consumes `beats` positionally; completion order is arbitrary."""
    out = _run(monkeypatch, 90)                       # 5 batches
    got = [b["i"] for b in out["beats"]]
    assert got == list(range(90)), "batches were assembled in the order they finished"


def test_every_batch_is_asked(monkeypatch):
    seen: list = []
    _run(monkeypatch, 90, record=seen)
    assert sorted(seen) == [0, 18, 36, 54, 72], f"a batch was skipped: {sorted(seen)}"


def test_a_short_script_still_works(monkeypatch):
    out = _run(monkeypatch, 5)
    assert [b["i"] for b in out["beats"]] == list(range(5))


# ---------------------------------------------------------------- failure stays local
def test_one_failed_batch_loses_only_its_own_beats(monkeypatch):
    """A batch that fails twice falls back to heuristic visuals for ITS beats — and must not take
    the batches around it with it."""
    out = _run(monkeypatch, 90, fail_starts=(36,))
    got = [b["i"] for b in out["beats"]]
    assert got == list(range(36)) + list(range(54, 90)), \
        "a neighbouring batch was lost with the failing one"


def test_a_raising_batch_is_treated_exactly_like_a_bad_parse(monkeypatch):
    """A transport error inside one batch must degrade to heuristics for its beats, exactly as an
    unparseable reply does — never abort the whole analyze stage."""
    from vidlore.clipstudio import llm as _real_llm

    def complete(*, system="", messages, max_tokens=1024, eng_cfg=None, model=""):
        if "BEATS:" not in messages[0]["content"]:
            return _HIGH_LEVEL
        raise RuntimeError("transport died")

    monkeypatch.setattr(_real_llm, "complete", complete)
    monkeypatch.setattr(_real_llm, "beat_model", lambda: "m")
    out = A._llm_analyze("script", "t", "Game of Thrones", _beats(36), None, None)
    assert out is not None and out["beats"] == []


# ---------------------------------------------------------------- the question did not change
def test_the_prompt_model_and_retry_are_untouched():
    src = inspect.getsource(A)
    i = src.index("def _one_batch")
    body = src[i:i + 4000]
    assert "for _try in range(2)" in body, "the two-try retry disappeared"
    assert "model=_llm.beat_model()" in body, "the per-beat model changed"
    assert "min(8000, 600 + len(chunk) * 180)" in body, "the token budget changed"


def test_the_pool_is_bounded_and_modest_by_default(monkeypatch):
    from vidlore.clipstudio.config import _workers
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "0")
    assert _workers("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", 4, 8) == 4
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "1")
    assert _workers("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", 4, 8) == 8


def test_one_worker_restores_the_serial_walk(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", "1")
    seen: list = []
    out = _run(monkeypatch, 54, record=seen)
    assert seen == [0, 18, 36], "with one worker the batches must run in script order"
    assert [b["i"] for b in out["beats"]] == list(range(54))


def test_concurrency_actually_overlaps(monkeypatch):
    """Non-vacuous: with a real per-call delay, 5 batches at 4 workers must beat 5 serial ones."""
    import time
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", "4")
    t0 = time.time()
    _run(monkeypatch, 90, delay=0.20)
    concurrent = time.time() - t0

    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_ANALYZE_WORKERS", "1")
    t0 = time.time()
    _run(monkeypatch, 90, delay=0.20)
    serial = time.time() - t0
    assert concurrent < serial * 0.75, f"no overlap: {concurrent:.2f}s vs serial {serial:.2f}s"
