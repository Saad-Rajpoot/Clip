"""The HD venv must be the interpreter that actually runs — PYTHONPATH says otherwise.

`HD_PY` is a FOREIGN interpreter: a Python 3.11 venv (.hdvenv) launched from the 3.9 host, holding
the only yt-dlp new enough to see YouTube's SABR/PO-token HD formats. PYTHONPATH is consulted
before that venv's own site-packages, so any host entry shipping a `yt_dlp` package wins. The
portal launcher exports `PYTHONPATH="$DIR:$DIR/.clipstudio_libs"`, and .clipstudio_libs carries the
engine's pinned yt-dlp 2025.10.14 — precisely the SABR-blind build hd_download exists to bypass.

MEASURED on job 03768be9ac (2026-08-07): one video, one command, PYTHONPATH the only difference.
With the portal's value, `-m yt_dlp --version` reported 2025.10.14 and extraction died on "The page
needs to be reloaded"; cleared, 2026.07.04 and 1080p. In that render all 45 sources fell back to SD
and both 403 recovery sweeps recovered 0/43 over 2h10m of cooldown, because every retry re-ran the
same shadowed extractor. The symptom is a 403/unavailable wave — indistinguishable from YouTube
blocking the machine, which is why it survived two sweeps, an exit-node rotation and a self-update.

So: every subprocess launched through an HD interpreter goes through `_env_with_runtimes()`, and
that env carries no inherited interpreter search path.
"""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from vidlore.clipstudio import hd_download as H

_PKG = Path(inspect.getsourcefile(H)).parent


# ------------------------------------------------------------------ the env itself
def test_an_inherited_pythonpath_is_dropped(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/host/repo:/host/repo/.clipstudio_libs")
    assert "PYTHONPATH" not in H._env_with_runtimes()


def test_pythonhome_is_dropped_too(monkeypatch):
    """On a foreign interpreter it relocates the stdlib and breaks the venv outright."""
    monkeypatch.setenv("PYTHONHOME", "/host/python39")
    assert "PYTHONHOME" not in H._env_with_runtimes()


def test_an_absent_pythonpath_is_not_invented(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert "PYTHONPATH" not in H._env_with_runtimes()


def test_everything_else_still_reaches_the_child(monkeypatch):
    """Only the interpreter path is stripped — the proxy pool, keys and HOME must survive."""
    monkeypatch.setenv("PYTHONPATH", "/host/repo")
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "1.2.3.4:12323:u:p")
    env = H._env_with_runtimes()
    assert env["VIDLORE_HD_PROXIES"] == "1.2.3.4:12323:u:p"
    assert env.get("HOME") == __import__("os").environ.get("HOME")


def test_the_js_runtimes_are_still_put_on_path(monkeypatch):
    """Stripping PYTHONPATH must not cost us the deno/node PATH prepend the solver needs."""
    import os
    monkeypatch.setattr(H, "NODE_BIN", "/opt/node/bin/node", raising=False)
    monkeypatch.setattr(H, "DENO_BIN", "/opt/deno/bin/deno", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    assert H._env_with_runtimes()["PATH"].split(os.pathsep)[:3] == [
        "/opt/node/bin", "/opt/deno/bin", "/usr/bin"]


# ------------------------------------------------------------------ and it really reaches the child
def test_a_poisoned_pythonpath_cannot_reach_the_child(tmp_path, monkeypatch):
    """The behavioural version: the child's sys.path must not contain the host's entry."""
    if not (H.HD_PY and Path(H.HD_PY).exists()):
        pytest.skip("no HD interpreter on this machine")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    p = subprocess.run([H.HD_PY, "-c", "import sys, json; print(json.dumps(sys.path))"],
                       env=H._env_with_runtimes(), capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert str(tmp_path) not in json.loads(p.stdout)


# ------------------------------------------------------------------ no call site may forget it
def _hd_interpreter_calls(path: Path) -> list:
    """(lineno, has_env) for every subprocess call whose argv[0] is an HD interpreter.

    Recognises the spellings the tree actually uses: the module-local `HD_PY`, the cross-module
    `_hd.HD_PY` and a local `hd_py` copied off the module — passed inline, through a variable, or
    returned by a cmd-building helper."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def _is_hd_argv(node) -> bool:
        if not (isinstance(node, ast.List) and node.elts):
            return False
        head = node.elts[0]
        name = (head.id if isinstance(head, ast.Name)
                else head.attr if isinstance(head, ast.Attribute) else "")
        return name.lower() == "hd_py"

    hd_vars, hd_builders = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_hd_argv(node.value):
            hd_vars.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(isinstance(n, ast.Return) and _is_hd_argv(n.value) for n in ast.walk(node)):
                hd_builders.add(node.name)

    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("run", "Popen") and node.args):
            continue
        arg = node.args[0]
        is_hd = (_is_hd_argv(arg)
                 or (isinstance(arg, ast.Name) and arg.id in hd_vars)
                 or (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                     and arg.func.id in hd_builders))
        if is_hd:
            found.append((node.lineno, any(kw.arg == "env" for kw in node.keywords)))
    return found


@pytest.mark.parametrize("mod", ["hd_download.py", "discover.py", "selfheal.py"])
def test_every_hd_interpreter_call_passes_the_isolated_env(mod):
    calls = _hd_interpreter_calls(_PKG / mod)
    assert calls, f"{mod}: expected at least one HD-interpreter subprocess to guard"
    missing = [ln for ln, has_env in calls if not has_env]
    assert not missing, (
        f"{mod}:{missing} launches the HD interpreter without env=_env_with_runtimes(); it will "
        f"inherit PYTHONPATH and silently run the host's SABR-blind yt-dlp")


def test_the_probe_and_the_download_build_their_cmd_lists_but_still_pass_the_env():
    """Both build argv separately, so the AST guard above cannot see them — check the source."""
    for fn in (H.probe_max_height, H.download_hd):
        assert "_env_with_runtimes()" in inspect.getsource(fn), fn.__name__


def test_the_flag_probe_asks_the_venvs_yt_dlp():
    """A --help read through the host's yt-dlp reports the WRONG flag support: that is how
    --remote-components (the n-challenge solver) got gated off on a venv that supports it."""
    assert "_env_with_runtimes()" in inspect.getsource(H._flag_supported)


def test_the_self_update_reports_the_version_that_will_run():
    """pip installed 2026.07.04 into the venv and the very next line confirmed 2025.10.14 — which
    read as 'the update did nothing' and hid the shadowing instead of exposing it."""
    src = inspect.getsource(H.maybe_update_ytdlp)
    assert src.count("_env_with_runtimes()") >= 2, "both the install and the version check"


# ------------------------------------------------------------------ and it is on the record
def test_the_download_stage_names_the_extractor_it_will_run(monkeypatch):
    """The version installed is not evidence about the version that RUNS. One line per render."""
    seen = []
    monkeypatch.setattr(H, "HD_PY", "/venv/bin/python", raising=False)
    monkeypatch.setattr(H, "_STACK_LOGGED", [False], raising=False)
    monkeypatch.setattr(H.subprocess, "run",
                        lambda *a, **k: type("P", (), {"stdout": "2026.07.04\n", "returncode": 0})())
    assert H.log_hd_stack(seen.append) == "2026.07.04"
    assert seen and "2026.07.04" in seen[0] and "/venv/bin/python" in seen[0]
    assert H.log_hd_stack(seen.append) == "2026.07.04", "still reports"
    assert len(seen) == 1, "but says it once per process"


def test_naming_the_extractor_uses_the_isolated_env_too():
    """Asking through a poisoned PYTHONPATH is how 'now 2025.10.14' got printed right after
    installing 2026.07.04 — a diagnostic that lies is worse than none."""
    assert "_env_with_runtimes()" in inspect.getsource(H.log_hd_stack)


def test_the_download_stage_calls_it():
    from vidlore.clipstudio import download as D
    assert "log_hd_stack(log)" in inspect.getsource(D.download_candidates)
