"""A render must run on the environment this project installs, and must say so before it starts.

Two renders died the same uncatchable death — CTranslate2's thread pool reaching
`std::condition_variable::wait`, which libc++ declares `_NOEXCEPT`, so the throw is an immediate
`std::terminate`: SIGABRT, no Python traceback, and in the portal's case the render, the portal and
the in-memory job registry go together.

Both crash reports name `libctranslate2.4.7.1.dylib`, loaded from a system interpreter's user
site-packages. The repo's `.venv` holds 4.8.0. Neither render ran on the project's own environment,
because starting the portal as `python3 -m vidlore.clipstudio.web` instead of via
ClipStudio-Portal.command silently swaps ctranslate2, onnxruntime, cv2 and numpy for whatever the
system happens to have.

`run_state` already recorded the version that ran — that is the only reason this was attributable at
all — but recording is not checking. This guard checks, at startup, where it costs a second.

It is deliberately narrow. It does NOT claim 4.8.0 fixes the abort; that is unproven. It claims a
render must run on the environment the project installs and tests, and when it cannot it must fail
in the first line of its log rather than in hour three.

Every test pins an explicit root, so the verdict never depends on the interpreter running pytest.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from vidlore.clipstudio import runtime_env as E


def _mk_venv(root: Path) -> Path:
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    return venv


# ---------------------------------------------------------------- the shipped configuration
def test_the_configuration_that_died_is_refused(tmp_path):
    """THE test: a project venv exists, and this interpreter is not it."""
    _mk_venv(tmp_path)
    assert E.running_in_project_venv(tmp_path) is False
    with pytest.raises(RuntimeError) as ei:
        E.enforce(tmp_path, driver="portal")
    msg = str(ei.value)
    assert "not the project environment" in msg
    assert "ClipStudio-Portal.command" in msg, "the message must say how to start it correctly"
    assert str(tmp_path / ".venv") in msg, "and which environment it should have been"
    assert sys.executable in msg, "and which one it actually is"


def test_the_project_venv_is_accepted(tmp_path, monkeypatch):
    venv = _mk_venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(venv))
    assert E.running_in_project_venv(tmp_path) is True
    assert E.foreign_reason(tmp_path) == ""
    assert "ctranslate2=" in E.enforce(tmp_path, driver="portal")


def test_a_checkout_with_no_venv_has_no_opinion(tmp_path):
    """Windows checkouts and the USB-stick copy have no .venv; inventing a rule for them would
    block a machine this project explicitly supports."""
    assert E.project_venv(tmp_path) is None
    assert E.foreign_reason(tmp_path) == ""
    assert E.enforce(tmp_path, driver="portal")            # returns a summary, raises nothing


def test_a_symlinked_venv_path_still_counts_as_the_same_venv(tmp_path, monkeypatch):
    """`pwd -P` in the launcher resolves symlinks; sys.prefix may not. Same place, same verdict."""
    real = tmp_path / "real"
    real.mkdir()
    _mk_venv(real)
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setattr(sys, "prefix", str(link / ".venv"))
    assert E.running_in_project_venv(real) is True


# ---------------------------------------------------------------- the override is explicit
def test_the_override_downgrades_the_refusal_and_records_itself(tmp_path, monkeypatch, capsys):
    _mk_venv(tmp_path)
    monkeypatch.setenv(E.OVERRIDE_ENV, "1")
    line = E.enforce(tmp_path, driver="portal")
    out = capsys.readouterr().out
    assert "not the project environment" in out, "an override must still be LOUD"
    assert E.OVERRIDE_ENV in line, "and the run must carry the fact that it was overridden"


def test_the_override_is_never_inferred(tmp_path, monkeypatch):
    _mk_venv(tmp_path)
    for value in ("", "0", "false", "no", "off", "maybe"):
        monkeypatch.setenv(E.OVERRIDE_ENV, value)
        with pytest.raises(RuntimeError):
            E.enforce(tmp_path, driver="portal")


def test_the_enforce_log_goes_where_the_caller_says(tmp_path, monkeypatch):
    _mk_venv(tmp_path)
    monkeypatch.setenv(E.OVERRIDE_ENV, "1")
    seen = []
    E.enforce(tmp_path, driver="resume_job", log=seen.append)
    assert seen and any("resume_job" in s for s in seen), "build.log must carry the warning too"


# ---------------------------------------------------------------- what the log must say
def test_the_summary_names_the_library_that_aborted():
    s = E.summary()
    for key in ("py=", "prefix=", "ctranslate2=", "onnxruntime=", "cv2=", "numpy="):
        assert key in s, f"{key} missing — this line is where an investigation starts"


def test_an_absent_library_is_reported_not_raised(monkeypatch):
    """A machine without cv2 must still be able to print its environment."""
    import builtins
    real = builtins.__import__

    def boom(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2 here")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert E.native_versions()["cv2"] == "absent"


def test_user_site_packages_is_called_out_by_name(tmp_path, monkeypatch):
    """The distinguishing fact of both crashes: the library came from the user site, not the venv."""
    _mk_venv(tmp_path)
    monkeypatch.setattr(E, "native_origin",
                        lambda mod="ctranslate2":
                        "/Users/x/Library/Python/3.9/lib/python/site-packages/ctranslate2/__init__.py")
    assert "user site-packages" in E.foreign_reason(tmp_path)


# ---------------------------------------------------------------- both drivers are wired
def _resume_main_source() -> str:
    """The resume driver's main(), not the file — the file's docstring mentions produce_auto, and
    an ordering assertion that matched prose would pass with the guard placed anywhere."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "resume_job.py"
    spec = importlib.util.spec_from_file_location("_resume_job_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return inspect.getsource(mod.main)


def test_both_render_drivers_enforce_before_doing_any_work():
    """A guard only one driver honours is not a guard. Same shape as the run-state contract test."""
    from vidlore.clipstudio import web as W
    portal = inspect.getsource(W.main)
    resume = _resume_main_source()
    for name, src in (("portal", portal), ("resume_job", resume)):
        assert "runtime_env" in src and "enforce(" in src, f"{name} does not check its environment"

    # and the check must come before the work: the portal must not bind its port first, and the
    # resume must not start rendering first.
    assert portal.index("enforce(") < portal.index("app.run("), "portal binds before checking"
    assert resume.index("enforce(") < resume.index("produce_auto("), "resume renders before checking"
    assert resume.index("enforce(") < resume.index("RunLock("), "resume locks before checking"


def test_the_environment_is_stamped_into_every_build_log():
    from vidlore.clipstudio import web as W
    assert "_env.summary()" in inspect.getsource(W._run_job)
    resume = (Path(__file__).resolve().parents[1] / "tools" / "resume_job.py").read_text("utf-8")
    assert "env_line" in resume and "env {" in resume
