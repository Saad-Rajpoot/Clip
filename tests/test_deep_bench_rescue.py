"""The verifier must exhaust a DEEP bench before it settles for a contextual fallback.

Measured on job 69d80e9dd4_v4, cross-referencing the render log against the frame-level relevance
eval:

    verify REPLACED with a passing alternate :  38 beats   eval mean 5.92
    verify left the pick alone              : 117 beats   eval mean 5.25
    verify DOWNGRADED exact -> contextual   : 113 beats   eval mean 4.34   <-- worst group

113 of 268 beats — 42% of the video — are ones where the verifier correctly recognised the footage
was not the exact scene and then kept it anyway, relabelled "contextual_fallback". It only ever saw
6 alternates out of a ~4000-shot pool before giving up. So match now keeps a deeper ranked bench and
the verifier tries the SAME strict bar against it before settling.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import match as M
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.models import ClipCandidate, ClipSelection


def _cand(i):
    return ClipCandidate(segment_index=0, source_id=f"src{i}", shot_index=i,
                         in_point=0.0, out_point=2.0, score=0.5)


def test_selection_carries_a_deep_bench_and_round_trips():
    s = ClipSelection(segment_index=0, source_id="a", shot_index=0, in_point=0, out_point=2,
                      confidence=0.5, alternates=[_cand(1)], deep_alternates=[_cand(7), _cand(8)])
    d = s.to_dict()
    assert len(d["deep_alternates"]) == 2
    r = ClipSelection.from_dict(d)
    assert [c.source_id for c in r.deep_alternates] == ["src7", "src8"]
    assert isinstance(r.deep_alternates[0], ClipCandidate)


def test_a_project_saved_before_the_deep_bench_still_loads():
    d = ClipSelection(segment_index=0, source_id="a", shot_index=0, in_point=0, out_point=2,
                      confidence=0.5).to_dict()
    d.pop("deep_alternates")
    assert ClipSelection.from_dict(d).deep_alternates == []


def test_match_fills_the_bench_beyond_the_normal_alternates():
    src = inspect.getsource(M.match_segments)
    assert "deep_alternates" in src, "match must keep a deeper bench for the verifier"
    assert "alternates[cfg.candidates_per_segment:_deep_n]" in src, \
        "the bench must start AFTER the normal alternates, not duplicate them"


def test_the_bench_is_tunable_and_can_be_switched_off():
    src = inspect.getsource(M.match_segments)
    assert "VIDLORE_CLIPSTUDIO_DEEP_ALTERNATES" in src
    assert "if _deep_n > cfg.candidates_per_segment" in src, \
        "setting the depth at or below candidates_per_segment must leave the bench empty"


def test_verify_tries_the_bench_at_the_STRICT_bar_before_downgrading():
    """A lenient retry would just re-accept the same wrong scene; the point is to find the real one."""
    src = inspect.getsource(V.verify_and_repair)
    i = src.index("VIDLORE_CLIPSTUDIO_DEEP_BENCH")
    j = src.index("_downgrade_on", i)
    seg = src[i:j]
    assert "_try_promote(downgrade=False" in seg, \
        "the deep-bench attempt must use the strict bar, not the contextual one"
    assert "deep_alternates" in seg


def test_the_bench_is_only_reached_when_the_ORIGINAL_pick_is_unusable():
    """Trying the bench on a pick that is already fine actively costs quality.

    Measured over the first 11 rescues: the group rose 4.18 -> 5.36 and the two deictic beats went
    1 -> 6 and 2 -> 9, but two beats already scoring 9 dropped to 4 and 5. The verifier only knows
    "not the exact scene", never "already good", so it traded a strong clip for a strict-passing
    weaker one. _contextual_subject_ok separates the cases exactly — the rescued beats failed it,
    the regressed ones passed it — so the bench must sit down the else-branch."""
    src = inspect.getsource(V.verify_and_repair)
    assert "elif not _deep_bench():" in src, \
        "the bench must be reached only where the original has already been judged unusable"
    # the plain contextual-keep branch (no missed look target) must never consult the bench
    i = src.index("elif _orig_ok:")
    j = src.index("elif not _deep_bench():")
    assert "_deep_bench()" not in src[i:j], \
        "a pick that passes the contextual bar must be kept, never sent to the bench"


def test_deep_bench_has_a_kill_switch():
    src = inspect.getsource(V.verify_and_repair)
    assert 'environ.get(\n                    "VIDLORE_CLIPSTUDIO_DEEP_BENCH", "1")' in src \
        or '"VIDLORE_CLIPSTUDIO_DEEP_BENCH", "1"' in src


def test_an_empty_bench_changes_nothing():
    """Beats the verifier never questions, and old projects, must behave byte-identically."""
    src = inspect.getsource(V.verify_and_repair)
    i = src.index("VIDLORE_CLIPSTUDIO_DEEP_BENCH")
    seg = src[i:src.index("_downgrade_on", i)]
    assert "if _bench and" in seg, "an empty bench must skip the attempt entirely"


def test_the_ranked_list_is_built_deeper_than_the_beat_will_use():
    """The first version truncated to candidates_per_segment where the list is BUILT, so the deep
    slice below it was always empty and the rescue path was dead code — measured: bench mean 0.0
    candidates per beat, 0 rescues over a full 268-beat verify."""
    src = inspect.getsource(M.match_segments)
    i = src.index("alt_best.values()")
    seg = src[max(0, i - 900):i + 400]
    assert "_keep_n" in seg, "the ranked list must be built to the DEEP depth, not the shallow one"
    assert "reverse=True)[:cfg.candidates_per_segment]" not in seg, \
        "truncating at build time empties the deep bench"
    assert "max(cfg.candidates_per_segment" in seg, \
        "the deep depth must never be shallower than the normal alternates"


def test_normal_consumers_still_see_only_candidates_per_segment():
    """Building deeper must not widen what every other stage reads — only the bench."""
    src = inspect.getsource(M.match_segments)
    assert "alternates=alternates[:cfg.candidates_per_segment]" in src, \
        "the selection's own alternates must stay at the configured depth"
    assert "for c in [cand] + alternates[:cfg.candidates_per_segment]" in src, \
        "beat_windows must stay at the configured depth"
