"""A render must record which code produced it, and say so when the portal is stale.

Job 0321078108 burned 6h20m and died on a defect that had been fixed in the repo three hours before
it started. The portal is a long-lived process — it imports the package once and keeps that code
until someone restarts it — and that server had been up for two days. Nothing anywhere said so:
build.log recorded a render with no way to tell which code produced it, so the failure looked like
a new bug instead of a stale process.

Every render now stamps the commit the SERVER PROCESS is running, and re-reads the checkout at that
moment so a drifted portal announces itself in the first line it writes.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import web as W


def test_the_render_banner_carries_the_code_revision():
    src = inspect.getsource(W._run_job)
    i = src.index("===== render start")
    banner = src[i:i + 220]
    assert "code=" in banner and "_running_code_revision()" in banner


def test_the_revision_is_a_real_commit_here():
    rev = W._read_code_revision()
    assert rev and rev != "unknown", "this repo is a git checkout; the stamp should resolve"
    assert len(rev.split("+")[0]) >= 7


def test_a_stale_process_says_so(monkeypatch):
    """The whole point: the process's imported revision differing from the checkout is the signal."""
    monkeypatch.setattr(W, "_CODE_REVISION", "aaaaaaa")
    monkeypatch.setattr(W, "_read_code_revision", lambda: "bbbbbbb")
    out = W._running_code_revision()
    assert "STALE" in out and "aaaaaaa" in out and "bbbbbbb" in out
    assert "RESTART THE PORTAL" in out


def test_a_current_process_stays_quiet(monkeypatch):
    monkeypatch.setattr(W, "_CODE_REVISION", "aaaaaaa")
    monkeypatch.setattr(W, "_read_code_revision", lambda: "aaaaaaa")
    assert W._running_code_revision() == "aaaaaaa"
    assert "STALE" not in W._running_code_revision()


def test_no_git_is_not_a_render_failure(monkeypatch):
    """A log line may never break a render — and 'unknown' must not be reported as staleness."""
    monkeypatch.setattr(W, "_CODE_REVISION", "unknown")
    monkeypatch.setattr(W, "_read_code_revision", lambda: "bbbbbbb")
    assert W._running_code_revision() == "unknown"
    monkeypatch.setattr(W, "_CODE_REVISION", "aaaaaaa")
    monkeypatch.setattr(W, "_read_code_revision", lambda: "unknown")
    assert W._running_code_revision() == "aaaaaaa"


def test_the_reader_swallows_every_fault():
    src = inspect.getsource(W._read_code_revision)
    assert "except Exception" in src and 'return "unknown"' in src
    assert "timeout=" in src, "a hung git call must not stall a render"
