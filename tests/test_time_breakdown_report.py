"""Every render must say where its hours went, without anyone parsing a log by hand.

This report exists because the same investigation was done three times. Each time the answer came
from hand-parsing build.log with a throwaway script, and each time it was the same two facts: the
machine had idle-slept (118 minutes of a 7-hour render), and one rung was asking twelve independent
questions one after another (68 minutes). perf_report.json had the stage durations the whole time —
but nobody opens a JSON while watching a render crawl.

So the render prints it. Ranked, as a share of the total, with the slept time called out by name.
It is strictly observational and it must never be able to break a render: a broken report is a
nuisance, a report that raises is a lost render.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import orchestrate as O


class _PM:
    """Stand-in for perf_metrics with a fixed stage table."""

    def __init__(self, stages, *, boom=False):
        self._stages = stages
        self._boom = boom
        self.closed = False

    def stage(self, _name):
        self.closed = True

    def snapshot(self):
        if self._boom:
            raise RuntimeError("perf backend is down")
        return {"stages": self._stages}


_REAL = [{"stage": "AI verify + repair", "dur_s": 15406.0},
         {"stage": "deep index", "dur_s": 708.0},
         {"stage": "match", "dur_s": 293.0},
         {"stage": "cut", "dur_s": 181.0},
         {"stage": "done", "dur_s": 0.0}]


def _run(stages, slept=0.0, **kw):
    out = []
    rows = O._log_time_breakdown(out.append, _PM(stages), slept, **kw)
    return out, rows


# ---------------------------------------------------------------- it answers the question
def test_the_biggest_stage_is_named_first_with_its_share():
    lines, rows = _run(_REAL)
    assert rows["AI verify + repair"] == 15406.0
    body = [ln for ln in lines if ln.startswith("time:   ")]
    assert "AI verify + repair" in body[0], "the worst stage must lead"
    assert "256.8 min" in body[0], "minutes, not seconds"
    assert "93%" in body[0], "a share, so a big number in a bigger render is not mistaken for fine"


def test_the_slept_time_is_called_out_by_name():
    lines, _ = _run(_REAL, slept=7080.0)
    head = lines[0]
    assert "ASLEEP" in head and "118 min" in head
    assert any("KEEP_AWAKE" in ln for ln in lines), \
        "naming the fix beside the symptom is the whole point"


def test_a_render_that_never_slept_says_nothing_about_sleep():
    lines, _ = _run(_REAL, slept=0.0)
    assert not any("ASLEEP" in ln for ln in lines)
    assert not any("KEEP_AWAKE" in ln for ln in lines)


def test_repeated_stage_marks_are_summed_not_listed_twice():
    """A resumed render re-enters stages; four separate 'match' rows are one line, not four."""
    lines, rows = _run([{"stage": "match", "dur_s": 60.0}] * 4)
    assert rows["match"] == 240.0
    assert len([ln for ln in lines if "match" in ln]) == 1


def test_zero_length_stages_are_not_printed():
    lines, _ = _run(_REAL)
    assert not any(" done" in ln for ln in lines), "a 0.0s bookkeeping stage is noise"


# ---------------------------------------------------------------- it can never break a render
def test_a_broken_perf_backend_does_not_raise():
    out = []
    rows = O._log_time_breakdown(out.append, _PM([], boom=True), 0.0)
    assert rows == {}
    assert any("unavailable" in ln for ln in out)


def test_a_log_that_itself_raises_is_survived():
    def hostile(_m):
        raise IOError("log file is gone")

    O._log_time_breakdown(hostile, _PM(_REAL), 500.0)      # must not raise


def test_no_stages_recorded_prints_nothing_rather_than_dividing_by_zero():
    lines, rows = _run([])
    assert rows == {} and lines == []


def test_the_open_stage_is_closed_before_it_is_read():
    """The last stage is still running when the report is taken; unclosed, it reads as 0 and the
    stage that took the longest would be missing from its own breakdown."""
    pm = _PM(_REAL)
    O._log_time_breakdown(lambda _m: None, pm, 0.0)
    assert pm.closed, "the final stage was never closed, so its duration is lost"


# ---------------------------------------------------------------- wired into every render
def test_every_render_prints_it():
    src = inspect.getsource(O._produce_auto)
    assert "_log_time_breakdown(log, _pm_stage" in src
    assert src.index("perf_report.json") < src.index("_log_time_breakdown("), \
        "print it where the report is already written, at the end of the pipeline"
