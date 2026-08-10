"""A dead render must say so, and a live one must never be called dead.

Job 2e0d34d9b1 died at 16:37:45 when CTranslate2's thread pool aborted the process (SIGABRT, no
Python traceback — libc++ declares condition_variable::wait _NOEXCEPT, so the throw is an immediate
std::terminate that nothing in Python or C++ can catch). The render ran in a daemon thread inside
the portal, so the portal, the render and the in-memory _JOBS registry all went at once. /status
then answered 404 forever and the page's `catch(e){}` swallowed it. The user watched a job that
had been dead for SIX HOURS.

The mirror-image failure is worse, and this file guards it first: real stages in this project go
silent for 1906s, 4031s and 5022s at a stretch. Any liveness rule derived from log recency would
declare healthy renders dead and send an operator to kill work that was fine. So liveness is
decided by evidence a dead process cannot produce — an flock the kernel releases, and os.kill(pid,
0) — and the heartbeat ticks on its own thread, only ever downgrading a LIVE pid to "unresponsive".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from vidlore.clipstudio import run_state as RS


def _mk(tmp_path, **over):
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    doc = {"schema": RS.SCHEMA, "jid": "j1", "pid": os.getpid(), "driver": "portal",
           "started": time.time(), "phase": "5/9 · deep index", "status": RS.RUNNING,
           "heartbeat_ts": time.time(), "code_rev": "abc1234"}
    doc.update(over)
    RS.state_path(tmp_path).write_text(json.dumps(doc), encoding="utf-8")
    return doc


# --------------------------------------------------------------- the disaster test, first
def test_a_live_job_silent_for_100_minutes_is_never_dead(tmp_path):
    """THE test. Measured silent gaps in this project's own build.logs: 1906s, 4031s, 5022s. A live
    render that has not logged for 6000s is normal, and calling it dead would be catastrophic."""
    lock = RS.RunLock(tmp_path)
    assert lock.acquire()
    try:
        doc = _mk(tmp_path, heartbeat_ts=time.time())      # heartbeat is fresh; the LOG is silent
        assert RS.liveness(tmp_path, doc) == RS.RUNNING
        assert RS.reconcile(tmp_path).get("status") == RS.RUNNING
    finally:
        lock.release()


def test_a_stale_heartbeat_on_a_live_pid_is_unresponsive_not_dead(tmp_path):
    lock = RS.RunLock(tmp_path)
    assert lock.acquire()
    try:
        doc = _mk(tmp_path, heartbeat_ts=time.time() - 300.0)
        assert RS.liveness(tmp_path, doc) == RS.UNRESPONSIVE
        assert RS.reconcile(tmp_path).get("status") == RS.RUNNING, "must not be written off"
    finally:
        lock.release()


# --------------------------------------------------------------- and a real death is reported
def test_a_process_that_dies_mid_render_is_reported_died(tmp_path):
    """Spawn a child that takes the lock, declares itself running, then dies the way the real one
    did — os._exit, no unwinding, no chance to record anything."""
    code = (
        "import sys, os, json, time\n"
        f"sys.path.insert(0, {str(RS.__file__).rsplit('/vidlore/', 1)[0]!r})\n"
        "from vidlore.clipstudio import run_state as RS\n"
        f"d = {str(tmp_path)!r}\n"
        "lock = RS.RunLock(d); lock.acquire()\n"
        "RS.write(d, 'j1', driver='portal', phase='5/9 · deep index')\n"
        "sys.stdout.write('up'); sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.read(2) == b"up"
        assert RS.liveness(tmp_path) == RS.RUNNING, "while the child holds the lock it is alive"
        proc.kill()
        proc.wait(timeout=10)
        time.sleep(0.3)                                    # let the kernel drop the flock
        assert RS.liveness(tmp_path) == RS.DIED
        doc = RS.reconcile(tmp_path)
        assert doc["status"] == RS.DIED
        assert doc["last_phase"] == "5/9 · deep index", "the phase it died in must survive"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_dead_pid_is_died_even_without_a_usable_lock(tmp_path):
    """exFAT / network volumes make flock a no-op — this project has a documented USB path."""
    _mk(tmp_path, pid=999_999_999)                         # a pid that cannot exist
    RS.lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    assert RS._pid_alive(999_999_999) is False
    assert RS.liveness(tmp_path) == RS.DIED


def test_no_state_file_is_unknown_not_a_verdict(tmp_path):
    (tmp_path / "output").mkdir(parents=True)
    assert RS.liveness(tmp_path) == RS.UNKNOWN


# --------------------------------------------------------------- the heartbeat is independent
def test_the_heartbeat_ticks_without_any_pipeline_progress(tmp_path):
    _mk(tmp_path, heartbeat_ts=time.time() - 50.0)
    hb = RS.Heartbeat(tmp_path, every=0.05).start()
    try:
        time.sleep(0.4)
    finally:
        hb.stop()
    age = time.time() - float(RS.read(tmp_path)["heartbeat_ts"])
    assert age < 5.0, "the heartbeat thread must advance with no log activity at all"


def test_writing_the_state_is_not_best_effort(tmp_path):
    """A silently absent state file reproduces the exact six-hour bug, so this one write raises."""
    target = tmp_path / "output"
    target.mkdir()
    target.chmod(0o500)
    try:
        with pytest.raises(Exception):
            RS.write(tmp_path, "j1", driver="portal")
    finally:
        target.chmod(0o700)


def test_the_state_records_which_native_libraries_ran(tmp_path):
    """The render that died loaded ctranslate2 4.7.1 from a system interpreter's user site while
    the repo's own venv held 4.8.0, and nothing in the job record said which had run."""
    RS.write(tmp_path, "j1", driver="portal", code_rev="abc1234")
    doc = RS.read(tmp_path)
    assert "ctranslate2" in doc["versions"] and "faster_whisper" in doc["versions"]
    assert doc["executable"] and doc["pid"] == os.getpid() and doc["code_rev"] == "abc1234"


# --------------------------------------------------------------- the portal contract
def test_status_answers_from_disk_when_the_registry_lost_the_job():
    """404-forever is what let the page lie. Disk knows, so /status must ask it."""
    import inspect
    from vidlore.clipstudio import web as W
    src = inspect.getsource(W.status)
    assert "_disk_job_view(jid)" in src
    view = inspect.getsource(W._disk_job_view)
    assert "reconcile" in view and "liveness" in view
    assert "no Python traceback is possible" in view, "the error must explain a native death"


def test_the_page_stops_polling_silently():
    import inspect
    from vidlore.clipstudio import web as W
    src = inspect.getsource(W)
    assert "if(!r.ok){" in src, "a failed fetch must be seen"
    assert "}catch(e){}" not in src, "the silent catch is what hid the dead portal"
    assert "the portal is not responding" in src
    assert "the render process died" in src
