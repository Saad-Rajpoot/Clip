"""Hold the machine awake for the length of a render.

The pipeline has detected idle sleep for a while — `⏸ system SLEPT ~17 min mid-render` — and then
done nothing about it. Measured on job 218acdfe10's resume: five sleeps, 17+17+17+17+50 minutes,
**118 minutes of a 7-hour render spent switched off**. Every one of them landed inside the verify
stage, where the process is waiting on vision API calls and looks perfectly idle to the OS, because
that is exactly what it is: a program whose next second of work depends on a remote answer.

So the render takes a power assertion, the same one `caffeinate` takes, for as long as it runs.

What this does NOT do: keep a laptop awake when the lid is shut. macOS clamshell sleep is not an
idle timer and no user-space assertion overrides it. `-i -m -s` covers idle sleep, disk sleep and
system sleep — the repeating 17-minute pattern above — and the display is deliberately left free to
switch off, because a dark screen costs nothing and a bright one for six hours costs the battery.

Failure is never fatal. If `caffeinate` is missing, or the platform has no equivalent, the render
runs exactly as it does today and says so once. A render must not depend on a power hint.
"""
from __future__ import annotations

import os
import subprocess
import sys

ENV_OFF = "VIDLORE_CLIPSTUDIO_KEEP_AWAKE"

# idle sleep, disk sleep, system sleep. NOT -d: letting the display sleep is free.
_CAFFEINATE = ("caffeinate", "-i", "-m", "-s")

# Windows: ES_CONTINUOUS | ES_SYSTEM_REQUIRED — "keep running, I am doing real work".
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def disabled() -> bool:
    return os.environ.get(ENV_OFF, "1").strip().lower() in ("0", "false", "no", "off")


class KeepAwake:
    """Start a power assertion; stop it on close. Safe to stop twice, safe to never start."""

    def __init__(self):
        self._proc = None
        self._windows = False
        self.how = ""

    def start(self, log=None) -> "KeepAwake":
        emit = log or (lambda m: None)
        if disabled():
            self.how = "off (%s=0)" % ENV_OFF
            return self
        try:
            if sys.platform == "darwin":
                # -w <pid>: caffeinate exits on its own if this render dies without unwinding —
                # a native abort leaves no chance to clean up, and a stray assertion would then
                # keep the machine awake forever.
                self._proc = subprocess.Popen(
                    [*_CAFFEINATE, "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.how = "caffeinate -imsw %d" % os.getpid()
            elif os.name == "nt":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
                self._windows = True
                self.how = "SetThreadExecutionState(ES_SYSTEM_REQUIRED)"
            else:
                self.how = "unsupported platform — not held"
        except Exception as exc:                           # noqa: BLE001 — a power hint, not a gate
            self.how = f"unavailable ({type(exc).__name__}) — not held"
        if self._proc is not None or self._windows:
            emit(f"keep-awake: holding a power assertion for this render ({self.how}); "
                 f"idle sleep will not pause it. Closing the lid still sleeps the machine.")
        else:
            emit(f"keep-awake: {self.how} — the machine may idle-sleep mid-render")
        return self

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:                              # noqa: BLE001
                try:
                    proc.kill()
                except Exception:                          # noqa: BLE001
                    pass
        if self._windows:
            self._windows = False
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            except Exception:                              # noqa: BLE001
                pass

    @property
    def held(self) -> bool:
        return bool(self._windows or (self._proc is not None and self._proc.poll() is None))

    def __enter__(self) -> "KeepAwake":
        return self.start()

    def __exit__(self, *_a) -> bool:
        self.stop()
        return False
