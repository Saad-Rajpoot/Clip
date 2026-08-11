"""A render must not be switched off in the middle of itself.

The pipeline has DETECTED idle sleep for months — `⏸ system SLEPT ~17 min mid-render` — and done
nothing about it. Job 218acdfe10's resume: 17+17+17+17+50 minutes, **118 of 420**, every one of them
inside the verify stage, which waits on remote vision answers and therefore looks perfectly idle to
the OS. So the render now holds a power assertion for as long as it runs.

It is a hint, never a gate: if `caffeinate` is missing or the platform has no equivalent, the render
proceeds exactly as before. And it must ALWAYS be released — the portal is a long-lived process, so
a leaked assertion would keep the machine awake long after the render ended.
"""
from __future__ import annotations

import inspect
import sys

import pytest

from vidlore.clipstudio import keep_awake as K


@pytest.fixture(autouse=True)
def _default_on(monkeypatch):
    monkeypatch.delenv(K.ENV_OFF, raising=False)
    yield


# ---------------------------------------------------------------- it is held, and released
@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS")
def test_the_assertion_is_really_held_and_really_released():
    import subprocess
    k = K.KeepAwake().start()
    try:
        assert k.held, "nothing is holding the machine awake"
        out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True).stdout
        assert "PreventUserIdleSystemSleep" in out
    finally:
        k.stop()
    assert not k.held, "the assertion outlived the render"


def test_stopping_twice_and_never_starting_are_both_safe():
    k = K.KeepAwake()
    k.stop()
    k.stop()
    assert not k.held
    K.KeepAwake().start().stop()


def test_the_context_manager_releases_on_an_exception():
    k = K.KeepAwake()
    with pytest.raises(ValueError):
        with k:
            raise ValueError("render failed")
    assert not k.held, "a failed render must not leave the machine pinned awake"


# ---------------------------------------------------------------- it is a hint, not a gate
def test_it_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(K.ENV_OFF, "0")
    assert K.disabled() is True
    k = K.KeepAwake().start()
    assert not k.held
    assert "off" in k.how


def test_a_missing_caffeinate_never_breaks_the_render(monkeypatch):
    monkeypatch.setattr(K.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no caffeinate")))
    monkeypatch.setattr(sys, "platform", "darwin")
    k = K.KeepAwake().start()                              # must not raise
    assert not k.held
    assert "unavailable" in k.how
    k.stop()


def test_it_says_which_way_it_is_holding(monkeypatch):
    msgs = []
    K.KeepAwake().start(log=msgs.append).stop()
    assert msgs, "the log must record whether the machine is pinned awake or not"


def test_the_display_is_left_free_to_sleep():
    """-d would keep a laptop screen lit for six hours for no benefit at all."""
    assert "-d" not in K._CAFFEINATE
    assert set(K._CAFFEINATE[1:]) == {"-i", "-m", "-s"}


# ---------------------------------------------------------------- wired into every render
def _wrapper_source():
    """produce_auto sets `__wrapped__ = _produce_auto`, and inspect.getsource follows that — so
    asking for the function's source hands back the INNER pipeline instead of the accounting
    wrapper the assertion actually lives in. The code object does not unwrap."""
    from vidlore.clipstudio import orchestrate as O
    return inspect.getsource(O.produce_auto.__code__)


def test_the_render_holds_it_and_frees_it_on_every_exit():
    src = _wrapper_source()
    assert "KeepAwake" in src, "renders do not hold a power assertion"
    assert "_awake.stop()" in src
    start = src.index("_awake = ")
    fin = src.index("finally:", start)
    stop = src.index("_awake.stop()", fin)
    assert start < fin < stop, "the release must sit in the finally, not on the success path"


def test_it_wraps_the_whole_pipeline_not_one_stage():
    """Held in the wrapper, so a raise mid-render still frees it."""
    src = _wrapper_source()
    assert "_produce_auto(project_dir, **kw)" in src
    assert src.index("_awake = ") < src.index("_produce_auto(project_dir, **kw)")
