"""Every page the portal serves must contain JavaScript that actually parses.

The portal's status page is written as a Python string that is emitted verbatim into a <script>
block, so a `\\n` that should reach the browser has to be written `\\\\n` in the source. The
liveness work added this line to the render-died branch:

    document.getElementById('log').textContent=(j.log||[]).join('\\n');

Python turned that `\\n` into a real newline, so the browser received a string literal opened on one
line and never closed — a SyntaxError that kills the WHOLE script block, not just that branch. The
poller never ran, and job 218acdfe10 sat at "Status: starting… / working…" for an hour while the
render was healthy and already past stage 4/9.

That is the same class of failure the liveness work existed to end: a page that says nothing while
the truth is one HTTP request away. A page whose script does not parse cannot report anything —
not progress, not death, not "lost contact" — so it is worse than the silent catch it replaced.

Node is authoritative when present. The fallback scanner is not decoration: it is checked against
the real defect below, so it cannot quietly degrade into a test that passes on anything.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

from vidlore.clipstudio import web as W

_NODE = shutil.which("node")


def _scripts(html: str) -> list:
    return re.findall(r"<script>(.*?)</script>", html, re.S)


def _unterminated_string_lines(js: str) -> list:
    """Lines that open a ' or " string and never close it.

    Template literals may legitimately span lines, so they are tracked but never reported. A regex
    literal containing a quote character would be misread, and none of the portal's do; node is the
    authority wherever it is installed.
    """
    bad, in_tpl = [], False
    for n, line in enumerate(js.splitlines(), 1):
        i, sq, dq, esc = 0, False, False, False
        while i < len(line):
            c = line[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif in_tpl:
                if c == "`":
                    in_tpl = False
            elif sq:
                if c == "'":
                    sq = False
            elif dq:
                if c == '"':
                    dq = False
            elif c == "'":
                sq = True
            elif c == '"':
                dq = True
            elif c == "`":
                in_tpl = True
            elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break                                   # line comment — nothing after it matters
            i += 1
        if sq or dq:
            bad.append((n, line.strip()[:90]))
    return bad


def _assert_parses(js: str, label: str) -> None:
    bad = _unterminated_string_lines(js)
    assert not bad, f"{label}: unterminated string literal at {bad}"
    if _NODE:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(js)
            path = f.name
        try:
            r = subprocess.run([_NODE, "--check", path], capture_output=True, text=True)
            assert r.returncode == 0, f"{label}: node --check failed\n{r.stderr[:800]}"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------- the pages
def test_the_job_status_page_script_parses():
    """THE test. This is the page that froze."""
    html = W._job_page("deadbeef01")
    blocks = _scripts(html)
    assert blocks, "the status page must ship a poller"
    for i, js in enumerate(blocks):
        _assert_parses(js, f"job page block {i}")


def test_the_create_form_script_parses():
    with W.app.test_request_context("/"):
        html = W.index()
    for i, js in enumerate(_scripts(html)):
        _assert_parses(js, f"index block {i}")


def test_the_job_page_parses_for_a_job_the_registry_never_saw():
    """A restarted portal renders this page from disk with no _JOBS entry — a different code path
    through the caption block, and it must emit valid script too."""
    assert "nosuchjob00" not in W._JOBS
    _assert_parses(_scripts(W._job_page("nosuchjob00"))[0], "disk-only job page")


def test_every_poller_branch_is_wired_to_a_real_job_id():
    """A leftover __JID__ would fetch a job that does not exist and 404 forever."""
    html = W._job_page("deadbeef01")
    assert "__JID__" not in html
    assert "/status/deadbeef01" in html and "/retry/deadbeef01" in html


# ---------------------------------------------------------------- the guard cannot rot
def test_the_scanner_catches_the_actual_defect():
    """Positive control: the exact text that shipped, byte for byte."""
    broken = "document.getElementById('log').textContent=(j.log||[]).join('\n');"
    assert _unterminated_string_lines(broken), "the scanner would not have caught the real bug"
    assert not _unterminated_string_lines(broken.replace("\n", "\\n"))


def test_the_scanner_does_not_object_to_the_portal_s_own_idioms():
    ok = "\n".join([
        "var s='it\\'s fine';",
        "var u=\"http://127.0.0.1:5151/status/x\";",     # // inside a string is not a comment
        "// a comment with an unmatched ' quote",
        "let m=(j.phase||'').match(/(\\d+)\\/9/);",
        "h+='<a class=dl href=\"#\">x</a>';",
    ])
    assert _unterminated_string_lines(ok) == []


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
def test_node_agrees_the_real_defect_is_a_syntax_error():
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write("var x=['a'].join('\n');")
        path = f.name
    try:
        assert subprocess.run([_NODE, "--check", path], capture_output=True).returncode != 0
    finally:
        os.unlink(path)
