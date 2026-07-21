"""Decision-neutral performance instrumentation.

Counters and wall-clock timers for the pipeline's hot paths — CLIP embedding calls,
verifier calls by rung, verdict-cache hits/misses, still-candidate scans, discovery
query latency, ffmpeg process spawns, and pipeline-stage durations.

STRICTLY OBSERVATIONAL. Nothing here may influence a decision, an ordering, a cache
content, or an output byte:
  * every public function is a cheap counter/timer update behind one lock;
  * every function swallows its own failures (a metrics bug must never break a render);
  * the ffmpeg spawn counter uses a sys.addaudithook subprocess listener — installed
    only when reporting is enabled, and it only READS the argv it is handed.

Enable reporting with VIDLORE_PERF=1 (report JSON is written next to the project's
build log by orchestrate, plus at interpreter exit to VIDLORE_PERF_PATH if set).
Counting itself is always on — it is nanoseconds per event and having the numbers
ALWAYS available lets a slow render be diagnosed after the fact.
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
_STAGES: list[dict] = []               # [{stage, dur_s, t0}]
_stage_cur: dict | None = None
_t0 = time.time()
_audit_installed = False


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
    """Context manager: `with perf.timed("verify.rung.venue"): ...` — accumulates
    wall time under the name and counts one observation. Never raises."""

    def __init__(self, name: str):
        self.name = name
        self._s = 0.0

    def __enter__(self):
        self._s = time.time()
        return self

    def __exit__(self, *exc):
        observe(self.name, time.time() - self._s)
        return False                                     # never swallow the caller's exception


def stage(name: str) -> None:
    """Mark a pipeline-stage transition. Duration of the previous stage is recorded
    when the next one starts (or at snapshot time for the last)."""
    global _stage_cur
    try:
        now = time.time()
        with _LOCK:
            if _stage_cur is not None:
                _stage_cur["dur_s"] = round(now - _stage_cur["t0"], 3)
                _STAGES.append(_stage_cur)
            _stage_cur = {"stage": str(name), "t0": now}
    except Exception:
        pass


def _install_ffmpeg_audit() -> None:
    """Count ffmpeg/ffprobe process spawns via the interpreter's audit hook — zero
    changes at any call site. Reads argv only; never blocks or filters anything."""
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
                stages = stages + [{**_stage_cur,
                                    "dur_s": round(time.time() - _stage_cur["t0"], 3)}]
            return {
                "uptime_s": round(time.time() - _t0, 3),
                "counts": dict(sorted(_COUNTS.items())),
                "times_s": {k: round(v, 3) for k, v in sorted(_TIMES.items())},
                "times_n": dict(sorted(_TIME_N.items())),
                "stages": [{"stage": s["stage"], "dur_s": s.get("dur_s", 0.0)} for s in stages],
            }
    except Exception:
        return {}


def write_report(path) -> None:
    """Best-effort JSON dump; never raises."""
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
