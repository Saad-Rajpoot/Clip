"""Decision-neutral performance instrumentation with a PER-RENDER lifecycle.

Counters and wall-clock timers for the pipeline's hot paths — CLIP embedding calls,
verifier calls by rung, verdict-cache hits/misses, still-candidate scans, discovery
query latency/provider errors/serial retries, ffmpeg/ffprobe process spawns, and
pipeline-stage durations.

LIFECYCLE. The portal is a long-lived process serving many renders; without explicit
scoping a second job would inherit (and mis-report) the first job's counts. So:

    start_run(run_id, project)   reset everything, stamp identity, arm the run
    snapshot() / write_report()  read the CURRENT run's numbers (id included)
    end_run()                    close the open stage; numbers remain readable

Durations use time.monotonic() — wall-clock steps (NTP, DST) can never produce negative
or inflated stage times. Reports are written atomically (tmp+replace).

STRICTLY OBSERVATIONAL: every public function is a cheap update behind one lock and
swallows its own failures; nothing here may influence a decision, an ordering, a cache
content, or an output byte. Counting is always on; VIDLORE_PERF=1 additionally writes
the JSON report at the end of produce_auto (and VIDLORE_PERF_PATH at interpreter exit).
"""
from __future__ import annotations

import json
import os
import threading
import time

_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}
_TIMES: dict[str, float] = {}          # name -> accumulated seconds
_TIME_N: dict[str, int] = {}           # name -> number of observations
_STAGES: list[dict] = []               # [{stage, dur_s}]
_stage_cur: dict | None = None         # {stage, t0(monotonic)}
_RUN: dict = {"run_id": "", "project": "", "started_at": ""}
_t0_mono = time.monotonic()
_audit_installed = False


def start_run(run_id: str = "", project: str = "") -> None:
    """Begin a fresh measurement scope: clears every counter/timer/stage from any prior
    run in this process and stamps the run identity. Call once per render job."""
    global _stage_cur, _t0_mono
    try:
        with _LOCK:
            _COUNTS.clear()
            _TIMES.clear()
            _TIME_N.clear()
            _STAGES.clear()
            _stage_cur = None
            _t0_mono = time.monotonic()
            _RUN["run_id"] = str(run_id or "")
            _RUN["project"] = str(project or "")
            _RUN["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _install_ffmpeg_audit()
    except Exception:
        pass


def reset() -> None:
    """Clear all state without stamping a new identity."""
    start_run(_RUN.get("run_id", ""), _RUN.get("project", ""))


def end_run() -> None:
    """Close the open stage so its duration is final. Numbers remain readable."""
    global _stage_cur
    try:
        now = time.monotonic()
        with _LOCK:
            if _stage_cur is not None:
                _STAGES.append({"stage": _stage_cur["stage"],
                                "dur_s": round(now - _stage_cur["t0"], 3)})
                _stage_cur = None
    except Exception:
        pass


def incr(name: str, n: int = 1) -> None:
    try:
        with _LOCK:
            _COUNTS[name] = _COUNTS.get(name, 0) + int(n)
    except Exception:
        pass


def observe(name: str, seconds: float) -> None:
    try:
        with _LOCK:
            _TIMES[name] = _TIMES.get(name, 0.0) + float(seconds)
            _TIME_N[name] = _TIME_N.get(name, 0) + 1
    except Exception:
        pass


class timed:
    """Context manager: `with perf.timed("verify.vision_call"): ...` — accumulates
    monotonic wall time under the name and counts one observation. Never raises."""

    def __init__(self, name: str):
        self.name = name
        self._s = 0.0

    def __enter__(self):
        self._s = time.monotonic()
        return self

    def __exit__(self, *exc):
        observe(self.name, time.monotonic() - self._s)
        return False                                     # never swallow the caller's exception


def stage(name: str) -> None:
    """Mark a pipeline-stage transition. Duration of the previous stage is recorded when
    the next one starts (end_run() closes the last)."""
    global _stage_cur
    try:
        now = time.monotonic()
        with _LOCK:
            if _stage_cur is not None:
                _STAGES.append({"stage": _stage_cur["stage"],
                                "dur_s": round(now - _stage_cur["t0"], 3)})
            _stage_cur = {"stage": str(name), "t0": now}
    except Exception:
        pass


def _install_ffmpeg_audit() -> None:
    """Count ffmpeg/ffprobe process spawns via the interpreter's audit hook — zero changes
    at any call site. Reads argv only; never blocks or filters anything. Installed once per
    process (hooks cannot be removed); counts land in whichever run is active."""
    global _audit_installed
    if _audit_installed:
        return
    import sys

    def _hook(event, args):
        if event == "subprocess.Popen":
            try:
                exe = str((args[1] or [""])[0] if args[1] else args[0] or "")
                base = os.path.basename(exe).lower()
                if "ffmpeg" in base:
                    incr("subprocess.ffmpeg")
                elif "ffprobe" in base:
                    incr("subprocess.ffprobe")
                else:
                    incr("subprocess.other")
            except Exception:
                pass

    try:
        sys.addaudithook(_hook)
        _audit_installed = True
    except Exception:
        pass


def enabled() -> bool:
    return os.environ.get("VIDLORE_PERF", "").strip() in ("1", "true", "yes")


def snapshot() -> dict:
    try:
        with _LOCK:
            stages = list(_STAGES)
            if _stage_cur is not None:
                stages = stages + [{"stage": _stage_cur["stage"],
                                    "dur_s": round(time.monotonic() - _stage_cur["t0"], 3)}]
            return {
                "run_id": _RUN.get("run_id", ""),
                "project": _RUN.get("project", ""),
                "started_at": _RUN.get("started_at", ""),
                "uptime_s": round(time.monotonic() - _t0_mono, 3),
                "counts": dict(sorted(_COUNTS.items())),
                "times_s": {k: round(v, 3) for k, v in sorted(_TIMES.items())},
                "times_n": dict(sorted(_TIME_N.items())),
                "stages": stages,
            }
    except Exception:
        return {}


def write_report(path) -> None:
    """Best-effort atomic JSON dump; never raises."""
    try:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(snapshot(), indent=1), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


if enabled():
    _install_ffmpeg_audit()
    _pp = os.environ.get("VIDLORE_PERF_PATH", "").strip()
    if _pp:
        import atexit
        atexit.register(write_report, _pp)
