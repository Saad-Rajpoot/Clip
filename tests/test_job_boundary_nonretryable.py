"""A content failure must be machine-readably non-retryable at the JOB boundary.

Drives tools/rerender_project.py as a REAL subprocess and inspects what a driver would see. The
shipped render was restarted 8 times over 11 hours because a release-block (a judgment about the
CONTENT) was indistinguishable from a transient fault; the 8th attempt "passed" only because the
vision API had died by then and the gate had nothing left to reject.

    python3 tests/test_job_boundary_nonretryable.py

No LLM, no ffmpeg, no encode.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILS = []


def _run_with_injected_failure(tmp, exc_expr, ident=""):
    """Run rerender_project.py against a stub project, forcing `exc_expr` out of the first stage.

    The tool DAEMONIZES (double fork + setsid), so the parent always exits 0 and the exit code
    carries no information — the status FILE is the only contract a driver can read. That is
    exactly why the machine-readable status matters here rather than a return code: poll for it.

    sitecustomize is imported automatically by CPython at startup, so the injection lands BEFORE
    the tool's own imports — no need to make the tool testable by editing it."""
    pd = os.path.join(tmp, "proj" + str(ident))
    os.makedirs(os.path.join(pd, "output"), exist_ok=True)
    json.dump({"name": "t", "root": pd, "sources": [], "segments": [], "selections": [],
               "meta": {"analysis": {"movie_title": "Game of Thrones", "video_type": "single_scene"}}},
              open(os.path.join(pd, "project.json"), "w"))
    shim = os.path.join(tmp, "sitecustomize.py")
    with open(shim, "w") as fh:
        fh.write(
            "import sys\n"
            f"sys.path.insert(0, {ROOT!r})\n"
            "import vidlore.clipstudio.match as M\n"
            "from vidlore.clipstudio.verify import NonRetryableBuildError\n"
            f"def _boom(*a, **k):\n    raise {exc_expr}\n"
            "M.match_segments = _boom\n")
    env = dict(os.environ, PYTHONPATH=tmp + os.pathsep + ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "rerender_project.py"), pd],
                   capture_output=True, text=True, timeout=120, env=env, cwd=ROOT)
    sf = os.path.join(pd, "output", "_rerender.status.json")
    for _ in range(600):                      # the daemon is detached — wait for its verdict
        if os.path.exists(sf):
            try:
                return json.load(open(sf)), pd
            except Exception:
                pass                          # mid-write
        time.sleep(0.1)
    return None, pd


def test_content_failure_is_marked_non_retryable():
    tmp = tempfile.mkdtemp()
    try:
        st, pd = _run_with_injected_failure(
            tmp, "NonRetryableBuildError('rejected-footage gate: 1 beat(s) unresolved')")
        assert st is not None, "the job must persist a machine-readable status"
        assert st["status"] == "content_failure", st
        assert st["retryable"] is False, "a content verdict must never be marked retryable"
        assert st["error_type"] == "NonRetryableBuildError", st
        # no video may be produced
        assert not os.path.exists(os.path.join(pd, "output", "final.mp4")), "nothing may publish"
        # the legacy sentinel still exists for existing readers
        assert os.path.exists(os.path.join(pd, "output", "_rerender.done"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_transient_failure_stays_retryable():
    """The distinction has to cut both ways, or it is just a rename: a genuine transient fault must
    still be retryable, otherwise a network blip becomes a permanent stop."""
    tmp = tempfile.mkdtemp()
    try:
        st, _ = _run_with_injected_failure(tmp, "ConnectionError('read timeout')")
        assert st is not None and st["status"] == "error", st
        assert st["retryable"] is True, "a transient fault must remain retryable"
        assert st["error_type"] == "ConnectionError", st
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_unchanged_content_failure_is_not_auto_restarted():
    """Proof of the actual guarantee: a driver that honours `retryable` re-runs the transient case
    and stops on the content case. Re-running an unchanged content failure only rolls the dice
    until a gate goes quiet — which is what published the 15:24 render."""
    tmp = tempfile.mkdtemp()
    try:
        attempts = {"n": 0}

        def driver(exc_expr, max_attempts=8):
            for i in range(max_attempts):
                attempts["n"] += 1
                st, _ = _run_with_injected_failure(tmp, exc_expr, ident=f"{exc_expr[:4]}{i}")
                assert st is not None, "every attempt must persist a status"
                if not st.get("retryable"):
                    return "stopped"
            return "exhausted"

        attempts["n"] = 0
        assert driver("NonRetryableBuildError('release-block')") == "stopped"
        assert attempts["n"] == 1, (
            f"an unchanged content failure must be attempted ONCE, not retried; "
            f"got {attempts['n']} attempts (the shipped render took 8)")

        attempts["n"] = 0
        assert driver("ConnectionError('blip')", max_attempts=3) == "exhausted"
        assert attempts["n"] == 3, "a transient fault must still be retried"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_portal_status_endpoint_exposes_retryable():
    src = open(os.path.join(ROOT, "vidlore", "clipstudio", "web.py"), encoding="utf-8").read()
    assert '"status", "retryable", "error_type"' in src, \
        "the portal /status payload must carry the machine-readable outcome"
    assert "NonRetryableBuildError" in src, "the portal job boundary must classify content failures"
    assert 'j["retryable"] = _vision or (not _content)' in src, \
        "retryability must cover content failures AND the (retryable) vision-outage case"
    assert "VisionBackendError" in src and "vision_down" in src, \
        "the job boundary must classify a vision-backend outage as its own retryable status"


def test_portal_auto_review_draft_on_footage_gap():
    """A FOOTAGE GAP must not show a bare 'fail': the portal auto-builds a review draft (warn mode,
    resumed from cache) so the user always gets a video. A VISION outage stays a retryable failure
    (it needs billing/backend restore, not a draft)."""
    src = open(os.path.join(ROOT, "vidlore", "clipstudio", "web.py"), encoding="utf-8").read()
    assert "VIDLORE_CLIPSTUDIO_PORTAL_AUTO_REVIEW" in src, "auto-review must be env-toggleable"
    assert "_do_render(resume=True)" in src, "the auto-review retry must RESUME (assembly only)"
    assert 'os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"' in src, \
        "the auto-review retry must flip to warn mode"
    # eligibility: content gap yes, vision outage NO
    assert "isinstance(_ce, _NRB2)" in src and "not isinstance(_ce, _VBE2)" in src, \
        "auto-review must fire on a content gap but NOT on a vision outage"
    assert 'j["status"] = ("review_draft"' in src, "a delivered review draft gets its own status"
    assert "review_draft" in src and "REVIEW DRAFT" in src, "the UI must label the review draft"


TESTS = [
    test_content_failure_is_marked_non_retryable,
    test_transient_failure_stays_retryable,
    test_an_unchanged_content_failure_is_not_auto_restarted,
    test_portal_status_endpoint_exposes_retryable,
    test_portal_auto_review_draft_on_footage_gap,
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
