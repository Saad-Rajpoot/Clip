"""A poisoned temp name must not be able to end a nine-hour render.

Job f3daa0ecce died after 9.2 hours:

    FATAL: semantic recovery page audit is missing or corrupt:
      PermissionError: [Errno 1] Operation not permitted

420 GB free, the directory writable seconds later, the file owned by the user, iCloud Desktop sync
off. What the leftover temp files carried was `com.apple.macl` — the extended attribute macOS
attaches to a TCC privacy decision — and the render was writing into ~/Desktop, one of the three
TCC-governed trees.

The mechanism is the part worth pinning here, because it is the part a retry loop would NOT have
fixed: every writer reused ONE fixed temp name, `<file>.tmp`. Once that inode carried a decision
denying this process, each further attempt aimed at the same poisoned name and got the same EPERM.
Deleting those stale files is what let the render resume.

So the temp is unique per attempt, and the retry is the smaller half — narrow, errno-typed, loud,
and it always ends in a raise. What must never regress: an audit whose existence IS a gate's
evidence still fails closed, and on the already-failing path the lineage VERDICT still outranks its
receipt.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from vidlore import atomic_io as A
from vidlore import scene_lineage_canary as C


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(A, "_BACKOFF_S", (0.0, 0.0, 0.0))     # no real sleeping in tests
    A.set_log(lambda _m: None)
    yield
    A.set_log(None)


# ---------------------------------------------------------------- the failure that happened
def test_the_temp_name_is_different_every_attempt(tmp_path, monkeypatch):
    """THE test. A fixed `<file>.tmp` is what made one denial permanent."""
    seen: list[str] = []
    real = A.tempfile.mkstemp

    def spy(*a, **k):
        fd, name = real(*a, **k)
        seen.append(Path(name).name)
        return fd, name

    monkeypatch.setattr(A.tempfile, "mkstemp", spy)
    target = tmp_path / "audit.json"
    A.atomic_write_text(target, '{"a":1}')
    A.atomic_write_text(target, '{"a":2}')
    assert len(seen) == 2 and seen[0] != seen[1], f"same temp name reused: {seen}"
    assert json.loads(target.read_text()) == {"a": 2}


def test_a_poisoned_leftover_temp_no_longer_blocks_the_write(tmp_path):
    """Reproduces the incident's shape: a stale `<file>.tmp` sitting in the directory. The old
    writer targeted exactly that name; this one must not care that it is there."""
    target = tmp_path / "semantic_recovery_audit.json"
    poisoned = tmp_path / "semantic_recovery_audit.json.tmp"
    poisoned.write_text("stale")
    os.chmod(poisoned, 0o400)
    try:
        A.atomic_write_text(target, '{"page_completed": true}')
        assert json.loads(target.read_text())["page_completed"] is True
    finally:
        os.chmod(poisoned, 0o600)


def test_a_transient_failure_is_retried_and_reported(tmp_path, capsys):
    calls = {"n": 0}
    real = A._publish_once

    def flaky(path, data):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(errno.EIO, "transient")
        return real(path, data)

    A.set_log(None)                                   # let it print, so we can read the notice
    orig = A._publish_once
    A._publish_once = flaky
    try:
        A.atomic_write_text(tmp_path / "x.json", "{}")
    finally:
        A._publish_once = orig
    out = capsys.readouterr().out
    assert calls["n"] == 3
    assert "attempt 1/" in out and "NEW temp name" in out
    assert "succeeded on attempt 3" in out, "a recovered write must not be silent"


# ---------------------------------------------------------------- it still fails closed
def test_exhaustion_raises_the_original_error(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_publish_once",
                        lambda *_a: (_ for _ in ()).throw(PermissionError(errno.EPERM, "TCC")))
    with pytest.raises(PermissionError):
        A.atomic_write_text(tmp_path / "x.json", "{}")


def test_a_privacy_denial_says_what_to_do(tmp_path, monkeypatch, capsys):
    A.set_log(None)
    monkeypatch.setattr(A, "_publish_once",
                        lambda *_a: (_ for _ in ()).throw(PermissionError(errno.EPERM, "denied")))
    with pytest.raises(PermissionError):
        A.atomic_write_text(tmp_path / "x.json", "{}", label="lineage audit")
    out = capsys.readouterr().out
    assert "Full Disk Access" in out and "Desktop" in out, \
        "the one error that cost nine hours must name its own fix"


@pytest.mark.parametrize("code", ["ENOSPC", "EROFS", "EXDEV", "EISDIR"])
def test_an_unfixable_errno_is_not_slept_on(tmp_path, monkeypatch, code):
    """A pause cannot create disk space or make a read-only volume writable."""
    n = getattr(errno, code, None)
    if n is None:
        pytest.skip(f"{code} not on this platform")
    calls = {"n": 0}

    def boom(*_a):
        calls["n"] += 1
        raise OSError(n, code)

    monkeypatch.setattr(A, "_publish_once", boom)
    with pytest.raises(OSError):
        A.atomic_write_text(tmp_path / "x.json", "{}")
    assert calls["n"] == 1, f"{code} was retried {calls['n']} times"


def test_a_privacy_denial_is_retried_less_than_a_hiccup(tmp_path, monkeypatch):
    """A TCC decision is cached per process — more attempts buy nothing but delay."""
    assert A._EPERM_ATTEMPTS < A._ATTEMPTS
    for code, expect in ((errno.EPERM, A._EPERM_ATTEMPTS), (errno.EIO, A._ATTEMPTS)):
        calls = {"n": 0}

        def boom(*_a, _c=code):
            calls["n"] += 1
            raise OSError(_c, "x")

        monkeypatch.setattr(A, "_publish_once", boom)
        with pytest.raises(OSError):
            A.atomic_write_text(tmp_path / "x.json", "{}")
        assert calls["n"] == expect


def test_a_programming_bug_is_never_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*_a):
        calls["n"] += 1
        raise TypeError("payload is not serializable")

    monkeypatch.setattr(A, "_publish_once", boom)
    with pytest.raises(TypeError):
        A.atomic_write_text(tmp_path / "x.json", "{}")
    assert calls["n"] == 1


def test_best_effort_returns_none_and_is_never_the_default(tmp_path, monkeypatch):
    import inspect
    assert inspect.signature(A.atomic_write_text).parameters["best_effort"].default is False
    monkeypatch.setattr(A, "_publish_once",
                        lambda *_a: (_ for _ in ()).throw(OSError(errno.EIO, "x")))
    assert A.atomic_write_text(tmp_path / "x.json", "{}", best_effort=True) is None


def test_a_symlink_is_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(OSError):
        A.atomic_write_text(link, "{}")


def test_no_temp_file_survives_a_success(tmp_path):
    A.atomic_write_text(tmp_path / "x.json", "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# ---------------------------------------------------------------- the gate contract holds
def test_the_lineage_verdict_outranks_its_receipt(tmp_path, monkeypatch):
    """If the audit cannot be persisted on the FAILURE path, the caller must still be told what was
    wrong with the render — with the IO error chained, not substituted."""
    monkeypatch.setattr(C, "atomic_write_text",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError(errno.EPERM, "TCC")))
    with pytest.raises(C.SceneLineageError) as ei:
        C.fail_audit(tmp_path / "a.json", C.new_audit(tmp_path / "out.mp4"), "timeline_order",
                     [{"reason": "scene hold claims donor beat 177"}])
    msg = str(ei.value)
    assert "1 violation(s)" in msg and "donor beat 177" in msg, "the verdict was lost"
    assert "could not be written" in msg, "the audit failure must still be reported"
    assert isinstance(ei.value.__cause__, PermissionError), "the IO error must be chained"


def test_a_healthy_failure_path_still_persists_the_audit(tmp_path):
    audit = tmp_path / "a.json"
    with pytest.raises(C.SceneLineageError):
        C.fail_audit(audit, C.new_audit(tmp_path / "out.mp4"), "binding",
                     [{"reason": "frame does not match its plan"}])
    rec = json.loads(audit.read_text())
    assert rec["status"] == "failed" and rec["stage"] == "binding"
    assert rec["failures"][0]["reason"] == "frame does not match its plan"


def test_a_successful_audit_write_is_unchanged(tmp_path):
    audit = tmp_path / "canary.json"
    C.write_audit(audit, {"schema": "assemble_scene_lineage/1", "status": "ok"})
    assert json.loads(audit.read_text())["status"] == "ok"
