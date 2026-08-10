"""Is this render actually alive? — answered from disk and the kernel, never from a timer.

A portal render died at 16:37 on 2026-08-10 and the browser went on showing it as running for SIX
HOURS. Nothing lied on purpose; nothing was in a position to tell the truth:

  * the render runs in a daemon THREAD inside the portal process (web.py), so a native abort in a
    C++ library — CTranslate2's thread pool, in that case — kills the portal, the render and the
    in-memory _JOBS registry together, with no Python traceback and no incident report;
  * every fact about a running job lived only in that registry, so once the process was gone
    /status returned 404 forever and the page's `catch(e){}` swallowed it in silence.

So the run's identity is written to disk, and liveness is decided by evidence the process cannot
fake once it is dead:

    1. an flock held for the render's whole life — if an observer can take it, the writer is gone;
    2. os.kill(pid, 0) where flock is unavailable (exFAT / network volumes: this project has a
       documented USB deployment path);
    3. only THEN a heartbeat age, and only to say "unresponsive" — never "dead".

THE HEARTBEAT TICKS ON ITS OWN THREAD, not from pipeline progress. That is the single most
important detail here: measured silent gaps in real build.logs reach 1906s, 4031s and 5022s, so any
liveness rule derived from log recency would declare healthy renders dead — the mirror image of the
bug being fixed, and a worse one, because it would send an operator to kill a job that was working.

When neither flock nor kill can be trusted the verdict is `unknown` and says so. It is never
silently rounded to `running` (the six-hour lie) or to `died` (killing good work).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

SCHEMA = "run_state/1"
HEARTBEAT_S = 10.0
# Generous on purpose: it only ever downgrades a LIVE pid to "unresponsive", and a wedged native
# library can hold a thread far longer than a stall in ordinary Python.
STALE_HEARTBEAT_S = 90.0

RUNNING, DIED, UNRESPONSIVE, UNKNOWN = "running", "died", "unresponsive", "unknown"


def state_path(project_dir) -> Path:
    return Path(project_dir) / "output" / "run_state.json"


def lock_path(project_dir) -> Path:
    return Path(project_dir) / "output" / "run.lock"


def _versions() -> dict:
    """The native libraries whose thread pools can abort this process, recorded per run.

    The render that died loaded ctranslate2 4.7.1 from the user site-packages of a system
    interpreter, while the repo's own .venv held 4.8.0 — and nothing in the job record said which
    one had run. A crash you cannot attribute to a library version is a crash you investigate twice.
    """
    out = {}
    for mod in ("ctranslate2", "faster_whisper", "onnxruntime", "cv2", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:                                  # noqa: BLE001 — absence is a fact too
            out[mod] = "absent"
    return out


def write(project_dir, jid: str, *, driver: str, phase: str = "starting",
          status: str = RUNNING, code_rev: str = "", extra: dict | None = None) -> Path:
    """Record who is running this job. Raises if it cannot be written.

    Deliberately NOT best-effort. Every other sidecar in this file is written inside a
    `try/except: pass` so a log failure can never break a render — but this one IS the record that
    a render happened at all, and a silently absent state file reproduces the exact bug this module
    exists to remove.
    """
    p = state_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": SCHEMA, "jid": str(jid), "pid": os.getpid(), "driver": str(driver),
        "started": time.time(), "phase": str(phase), "status": str(status),
        "heartbeat_ts": time.time(), "code_rev": str(code_rev),
        "executable": __import__("sys").executable, "versions": _versions(),
    }
    doc.update(extra or {})
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p


def read(project_dir) -> dict:
    try:
        return json.loads(state_path(project_dir).read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001 — absent/corrupt is unknown
        return {}


def touch(project_dir, *, phase: str | None = None, status: str | None = None) -> None:
    """Advance the heartbeat (and optionally the phase). Best-effort: never breaks a render."""
    try:
        doc = read(project_dir)
        if not doc:
            return
        doc["heartbeat_ts"] = time.time()
        if phase is not None:
            doc["phase"] = str(phase)[:300]
        if status is not None:
            doc["status"] = str(status)
        p = state_path(project_dir)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:                                      # noqa: BLE001
        pass


class Heartbeat:
    """Ticks run_state.json on its OWN thread while the render works.

    Progress-driven heartbeats cannot work here: real stages go quiet for over an hour (measured
    1906s / 4031s / 5022s in this project's own logs). A thread that only writes a timestamp keeps
    saying "the process is still alive" through the longest legitimate silence, which is exactly the
    claim being made — and nothing more.
    """

    def __init__(self, project_dir, every: float = HEARTBEAT_S):
        self._dir, self._every = project_dir, float(every)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> "Heartbeat":
        if self._t is None:
            self._t = threading.Thread(target=self._run, name="run-heartbeat", daemon=True)
            self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            touch(self._dir)

    def stop(self, *, status: str | None = None, phase: str | None = None) -> None:
        self._stop.set()
        if status is not None or phase is not None:
            touch(self._dir, status=status, phase=phase)


class RunLock:
    """An flock held for the render's whole life. Its release IS the death certificate.

    A pid can be recycled and a timestamp can be stale, but a lock the kernel drops when the owning
    process dies cannot lie. Held by EVERY driver — the portal and tools/resume_job.py — because
    both can own a project dir, and one taking the lock while the other holds it is also how a
    second writer on one job is refused rather than silently corrupting it.
    """

    def __init__(self, project_dir):
        self.path = lock_path(project_dir)
        self._fh = None
        self.usable = True

    def acquire(self) -> bool:
        """True if held. False if another live driver holds it. `usable` False if flock is a no-op
        on this filesystem, in which case liveness falls back to the pid check."""
        try:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False                                   # someone else is genuinely running it
        except Exception:                                  # noqa: BLE001 — exFAT/NFS/Windows
            self.usable = False
            return True

    def release(self) -> None:
        try:
            if self._fh is not None:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                self._fh.close()
        except Exception:                                  # noqa: BLE001
            pass
        self._fh = None


def _lock_is_free(project_dir) -> bool | None:
    """True when nobody holds the run lock, False when someone does, None when flock is unusable."""
    p = lock_path(project_dir)
    if not p.exists():
        return None                                        # never taken: says nothing either way
    try:
        import fcntl
        with open(p, "a+") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                return True
            except BlockingIOError:
                return False
    except Exception:                                      # noqa: BLE001
        return None


def _pid_alive(pid) -> bool | None:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                        # exists, owned by someone else
    except Exception:                                      # noqa: BLE001
        return None


def liveness(project_dir, doc: dict | None = None) -> str:
    """running | died | unresponsive | unknown — in that order of evidence."""
    doc = doc if doc is not None else read(project_dir)
    if not doc:
        return UNKNOWN
    if str(doc.get("status") or "") != RUNNING:
        return str(doc.get("status") or UNKNOWN)

    free = _lock_is_free(project_dir)
    if free is True:
        return DIED                                        # the kernel dropped it: the writer is gone
    if free is False:
        alive = True                                       # somebody holds it, so somebody is running
    else:
        alive = _pid_alive(doc.get("pid"))
        if alive is False:
            return DIED
        if alive is None:
            return UNKNOWN

    # Only now, and only to say "not answering" — never to declare a live process dead. Renders go
    # quiet for over an hour legitimately; this measures the heartbeat thread, not the pipeline.
    try:
        age = time.time() - float(doc.get("heartbeat_ts") or 0.0)
    except (TypeError, ValueError):
        return UNKNOWN
    return UNRESPONSIVE if age > STALE_HEARTBEAT_S else RUNNING


def reconcile(project_dir) -> dict:
    """Rewrite a `running` state whose owner is provably gone. Returns the resulting doc."""
    doc = read(project_dir)
    if not doc:
        return {}
    if liveness(project_dir, doc) == DIED and str(doc.get("status") or "") == RUNNING:
        doc["status"] = DIED
        doc["died_detected"] = time.time()
        doc["last_phase"] = doc.get("phase")
        try:
            p = state_path(project_dir)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            os.replace(tmp, p)
        except Exception:                                  # noqa: BLE001
            pass
    return doc
