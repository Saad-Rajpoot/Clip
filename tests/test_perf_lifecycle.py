"""P1.4 — per-render metrics lifecycle: a second job never inherits the first job's
numbers; stage durations are monotonic; reports are atomic and identity-stamped.

    python3 tests/test_perf_lifecycle.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import perf_metrics as P                 # noqa: E402

FAILS = []


def test_start_run_isolates_jobs():
    P.start_run("job-1", "/proj/a")
    P.incr("x", 5)
    P.stage("match")
    P.stage("cut")
    s1 = P.snapshot()
    assert s1["run_id"] == "job-1" and s1["counts"]["x"] == 5
    assert [st["stage"] for st in s1["stages"]][:1] == ["match"]
    P.start_run("job-2", "/proj/b")
    s2 = P.snapshot()
    assert s2["run_id"] == "job-2" and s2["project"] == "/proj/b"
    assert s2["counts"] == {}, "job-2 must not inherit job-1's counters"
    assert s2["stages"] == [], "job-2 must not inherit job-1's stages"


def test_end_run_closes_open_stage_and_durations_nonnegative():
    P.start_run("job-3", "")
    P.stage("verify")
    P.end_run()
    s = P.snapshot()
    assert [st["stage"] for st in s["stages"]] == ["verify"]
    assert all(st["dur_s"] >= 0 for st in s["stages"]), "monotonic durations only"


def test_timed_accumulates_and_never_swallows():
    P.start_run("job-4", "")
    with P.timed("op"):
        pass
    try:
        with P.timed("op"):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("timed must not swallow exceptions")
    s = P.snapshot()
    assert s["times_n"]["op"] == 2 and s["times_s"]["op"] >= 0.0


def test_report_is_atomic_and_stamped():
    P.start_run("job-5", "/proj/c")
    P.incr("y")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "perf_report.json")
    P.write_report(p)
    rep = json.load(open(p))
    assert rep["run_id"] == "job-5" and rep["counts"]["y"] == 1
    assert not os.path.exists(p + ".tmp"), "atomic write must not leave a temp file"


TESTS = [
    test_start_run_isolates_jobs,
    test_end_run_closes_open_stage_and_durations_nonnegative,
    test_timed_accumulates_and_never_swallows,
    test_report_is_atomic_and_stamped,
]

if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
