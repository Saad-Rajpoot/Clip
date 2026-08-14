"""Publish a render artifact atomically, without a fixed temp name that can be poisoned.

Job f3daa0ecce died after 9.2 hours:

    FATAL: semantic recovery page audit is missing or corrupt:
      PermissionError: [Errno 1] Operation not permitted: '.../semantic_recovery_audit.json'

Nothing was out of disk (420 GB free), the directory was writable seconds later, the file was owned
by the user, and iCloud Desktop sync was off. What the leftover temp files carried was
`com.apple.macl` — the extended attribute macOS attaches to a TCC (privacy) access decision. The
render was writing into ~/Desktop, which is one of the three TCC-governed trees.

The mechanism, and the reason a retry loop alone would NOT have saved that render: the writers all
reused ONE fixed temp name, `<file>.tmp`. Once that inode carried a decision that denied this
process, every subsequent attempt targeted the same poisoned name and got the same EPERM. Backing
off and trying again into it could not have worked — and a TCC denial is cached per responsible
process for its lifetime, so sleeping does not change the answer either. (Deleting those stale
`.tmp` files is what let the render resume.)

So the temp is unique per attempt. That alone removes the failure that actually happened, and it
also removes the cross-process clobber that other call sites had already hand-patched with their
own uuid suffix.

The bounded retry is the smaller half, and deliberately narrow:

  * only OSError is retried — a TypeError from an unserializable payload is a programming bug
  * errnos a pause cannot fix (ENOSPC, EROFS, EDQUOT, EXDEV, EISDIR, …) raise IMMEDIATELY
  * EPERM/EACCES get fewer attempts than a generic hiccup, because a privacy denial is a decision,
    not a hiccup
  * every attempt after the first is SHOUTED, because this project has already lost months to a
    fail-open catch that looked benign
  * exhaustion always RAISES, with the original error chained

`best_effort=True` exists for the sidecars that are already `except Exception: pass` today; it
returns None instead of raising and is never the default. A gate's evidence file must keep failing
closed: a render may not claim protection it could not write down.
"""
from __future__ import annotations

import errno
import os
import tempfile
import time
from pathlib import Path

# Measured: the incident needed a different temp NAME, not more attempts. These are for a genuine
# transient (a fsync racing a backup snapshot), and are kept small so a real misconfiguration
# surfaces in seconds rather than being ground away.
_ATTEMPTS = 4
_EPERM_ATTEMPTS = 2          # a TCC denial is cached per process — more attempts buy nothing
_BACKOFF_S = (0.05, 0.25, 1.0)

# A pause cannot create disk space, make a read-only volume writable, or fix a path bug.
_TERMINAL_ERRNOS = frozenset(filter(None, (
    getattr(errno, name, None) for name in
    ("ENOSPC", "EROFS", "EDQUOT", "ENAMETOOLONG", "EISDIR", "ENOTDIR", "EXDEV", "EINVAL", "EFAULT")
)))
_SLOW_ERRNOS = frozenset(filter(None, (getattr(errno, n, None) for n in ("EPERM", "EACCES"))))

_LOG = None
_COUNTER = None


def set_log(fn) -> None:
    """Install the process-wide sink for retry notices (inversion — this module imports nothing)."""
    global _LOG
    _LOG = fn


def set_counter(fn) -> None:
    global _COUNTER
    _COUNTER = fn


def _say(msg: str, log=None) -> None:
    sink = log or _LOG
    if sink is None:
        print(f"[atomic_io] {msg}", flush=True)
        return
    try:
        sink(msg)
    except Exception:                                    # noqa: BLE001 — a notice, never a gate
        pass


def _count(name: str) -> None:
    if _COUNTER is None:
        return
    try:
        _COUNTER(name)
    except Exception:                                    # noqa: BLE001
        pass


def _attempts_for(exc: OSError) -> int:
    return _EPERM_ATTEMPTS if getattr(exc, "errno", None) in _SLOW_ERRNOS else _ATTEMPTS


def _publish_once(path: Path, data: bytes) -> None:
    """mkstemp → write → fsync → replace, as ONE unit. A unique temp every time.

    Retrying `os.replace` alone against a temp an earlier `finally` may have unlinked is how you
    atomically publish truncated JSON, so the whole transaction is the retried unit.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = ""                                         # replace consumed it
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:                              # cleanup must never mask the real error
                pass


def atomic_write_text(path, text: str, *, best_effort: bool = False, encoding: str = "utf-8",
                      mkparents: bool = True, label: str = "", log=None):
    """Publish `text` at `path` atomically. Raises on exhaustion unless `best_effort`.

    `text` is serialized by the CALLER, once, before this call: several callers mutate the dict they
    are dumping while a page runs, and a helper that re-serialized per attempt could publish
    different bytes on attempt 3 than it prepared on attempt 1.
    """
    path = Path(path)
    if path.is_symlink():
        raise OSError(errno.EPERM, "refusing to publish through a symlink", str(path))
    if mkparents:
        path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode(encoding)
    what = label or path.name

    last: OSError | None = None
    attempt = 0
    while True:
        attempt += 1
        try:
            _publish_once(path, data)
            if path.stat().st_size != len(data):
                raise OSError(errno.EIO, "short write after replace", str(path))
            if attempt > 1:
                _say(f"atomic-write: {what} succeeded on attempt {attempt} — the first "
                     f"{attempt - 1} failed ({type(last).__name__}: {last}); if this repeats, the "
                     f"output directory is the problem, not the render", log)
                _count("atomic_io.recovered")
            return path
        except OSError as exc:
            last = exc
            cap = _attempts_for(exc)
            code = getattr(exc, "errno", None)
            fatal = code in _TERMINAL_ERRNOS or attempt >= cap
            _count("atomic_io.attempt_failed")
            if not fatal:
                _say(f"atomic-write: {what} attempt {attempt}/{cap} failed "
                     f"({type(exc).__name__}: {exc}) — retrying with a NEW temp name", log)
                time.sleep(_BACKOFF_S[min(attempt - 1, len(_BACKOFF_S) - 1)])
                continue
            _count("atomic_io.failed")
            hint = ""
            if code in _SLOW_ERRNOS:
                hint = (" — on macOS this is the Privacy (TCC) denial signature for the Desktop, "
                        "Documents and Downloads folders; grant the interpreter Full Disk Access "
                        "or move the render output root outside those trees")
            _say(f"atomic-write: {what} FAILED after {attempt} attempt(s) "
                 f"({type(exc).__name__}: {exc}){hint}", log)
            if best_effort:
                return None
            raise
