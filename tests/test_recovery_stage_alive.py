"""R4-5 bounded recovery must actually RUN — plus the guard that stopped it from being silent.

A bare `os.environ` inside `_recover_unresolved_beats` (whose only import is `import os as _os`,
in a module with no module-level `import os`) made the whole stage raise NameError on EVERY
render for months. The caller's fail-open catch logged it as `recovery: skipped (NameError: ...)`
— indistinguishable from a benign environmental skip — so four blocked renders paid the expensive
self-heal / rebuild path while the cheap targeted recovery never ran once.

Locked here:
  1. the stage runs end-to-end on an unresolved beat without raising,
  2. the LOOK-RECOVERY branch (which held the bug) is really executed,
  3. no unbound `os` reference can come back anywhere in orchestrate.py (closure-aware AST scan),
  4. programming errors are logged LOUDLY and recorded, never as a quiet "skipped".

    python3 tests/test_recovery_stage_alive.py

No LLM, no ffmpeg, no network.
"""
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import orchestrate as O                      # noqa: E402
from vidlore.clipstudio import policy as P                           # noqa: E402
from vidlore.clipstudio.analyze import ScriptAnalysis                # noqa: E402
from vidlore.clipstudio.config import ClipConfig                     # noqa: E402
from vidlore.clipstudio.models import (ClipProject, ClipSelection,   # noqa: E402
                                       ScriptSegment)

ORCH = ROOT / "vidlore" / "clipstudio" / "orchestrate.py"


def _seg(idx, text, query="", entity="", policy="exact_scene"):
    return ScriptSegment(index=idx, text=text, scene_query=query, required_entity=entity,
                         required_kind="character", visual_policy=policy)


def _sel(idx, *, rejected=False, flags=None):
    return ClipSelection(segment_index=idx, source_id="src_a", shot_index=0, in_point=0.0,
                         out_point=3.0, confidence=0.8,
                         flag_reasons=list(flags or (["verifier_failed"] if rejected else [])),
                         verifier=({"status": "ok", "verdict": "replace"} if rejected
                                   else {"status": "ok", "verdict": "keep"}))


def _proj(tmp, segs, sels):
    p = ClipProject(name="t", root=tmp)
    p.ensure_dirs()
    p.segments = list(segs)
    p.selections = list(sels)
    p.meta["analysis"] = {"movie_title": "Game of Thrones", "video_type": "multi_scene",
                          "episode_hint": "", "characters": []}
    return p


def _run(proj, segs, log=None):
    """Drive the real stage with discovery stubbed to 'found nothing' (no network)."""
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    with mock.patch("vidlore.clipstudio.discover.discover_sources", return_value=[]) as disc:
        n = O._recover_unresolved_beats(
            proj, segs, analysis, ClipConfig(), {}, faceid_obj=None, refs={},
            roster=[], policy="approved_testing", log=(log or (lambda m: None)))
    return n, disc


class TestRecoveryRuns(unittest.TestCase):
    def test_unresolved_beat_does_not_raise(self):
        """The regression itself: this call raised NameError('os') on every render."""
        with tempfile.TemporaryDirectory() as td:
            segs = [_seg(0, "Ned kneels on the steps.",
                         query="Ned Stark execution Sept of Baelor", entity="Ned Stark")]
            proj = _proj(td, segs, [_sel(0, rejected=True)])
            n, disc = _run(proj, segs)
        self.assertIsInstance(n, int)
        self.assertEqual(n, 0)                      # nothing discovered → nothing recovered
        disc.assert_called_once()                   # it got PAST the buggy line to real work

    def test_look_recovery_branch_executes(self):
        """The bug sat inside the LOOK-RECOVERY branch — prove that branch really runs."""
        lines = []
        with tempfile.TemporaryDirectory() as td:
            segs = [_seg(0, "Ned kneels on the steps.",
                         query="Ned Stark execution Sept of Baelor", entity="Ned Stark"),
                    _seg(1, "Watch the chalice as it moves.",
                         query="Purple Wedding chalice", entity="")]
            self.assertTrue(P.deictic_target(segs[1]), "fixture must carry a deictic target")
            proj = _proj(td, segs, [_sel(0, rejected=True),
                                    _sel(1, flags=["look_target_missing"])])
            _run(proj, segs, log=lines.append)
        self.assertTrue(any("look-miss beat(s) added" in m for m in lines),
                        f"look-recovery branch never ran; log was {lines}")

    def test_no_unresolved_beats_is_a_cheap_noop(self):
        with tempfile.TemporaryDirectory() as td:
            segs = [_seg(0, "Ned kneels.", query="q", entity="Ned Stark")]
            proj = _proj(td, segs, [_sel(0)])       # verdict keep → nothing to recover
            n, disc = _run(proj, segs)
        self.assertEqual(n, 0)
        disc.assert_not_called()

    def test_kill_switch_still_short_circuits(self):
        with tempfile.TemporaryDirectory() as td:
            segs = [_seg(0, "Ned kneels.", query="q", entity="Ned Stark")]
            proj = _proj(td, segs, [_sel(0, rejected=True)])
            with mock.patch.dict(os.environ, {"VIDLORE_CLIPSTUDIO_RECOVERY": "0"}):
                n, disc = _run(proj, segs)
        self.assertEqual(n, 0)
        disc.assert_not_called()


class TestNoUnboundOsAnywhere(unittest.TestCase):
    """Static guard: the same class of bug must never come back silently in this module."""

    def test_every_os_reference_resolves(self):
        src = ORCH.read_text()
        tree = ast.parse(src)
        lines = src.splitlines()

        module_names = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    module_names.add(a.asname or a.name.split(".")[0])

        def binds(fn):
            """names bound in fn's OWN scope (never descending into nested scopes)"""
            out = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}

            def walk(node):
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            out.add(ch.name)
                        continue
                    if isinstance(ch, (ast.Import, ast.ImportFrom)):
                        for a in ch.names:
                            out.add(a.asname or a.name.split(".")[0])
                    if isinstance(ch, ast.Name) and isinstance(ch.ctx, ast.Store):
                        out.add(ch.id)
                    walk(ch)
            walk(fn)
            return out

        parent = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parent[ch] = node

        def enclosing_fns(node):
            out, cur = [], parent.get(node)
            while cur is not None:
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(cur)
                cur = parent.get(cur)
            return out

        unbound = []
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            own = binds(fn)
            outer = set()
            for s in enclosing_fns(fn):
                outer |= binds(s)
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id == "os"):
                    continue
                if "os" in own or "os" in outer or "os" in module_names:
                    continue
                if any("os" in binds(s) for s in enclosing_fns(node) if s is not fn):
                    continue
                unbound.append(f"{fn.name}() line {node.lineno}: "
                               f"{lines[node.lineno - 1].strip()[:80]}")
        self.assertEqual(unbound, [], "unbound `os` — every function must use its own alias "
                                      "(orchestrate.py has NO module-level `import os`)")


class TestFailOpenCatchIsLoudOnBugs(unittest.TestCase):
    def _log(self):
        out = []
        return out, out.append

    def test_programming_error_is_loud_and_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            proj = _proj(td, [], [])
            msgs, log = self._log()
            O._log_stage_skip(log, proj, "recovery", NameError("name 'os' is not defined"))
        joined = " ".join(msgs)
        self.assertIn("BUG", joined)
        self.assertIn("did NOT run", joined)
        self.assertNotIn("recovery: skipped", joined)     # never reads as a benign skip
        self.assertEqual(proj.meta["stage_bugs"][0]["stage"], "recovery")

    def test_environmental_error_stays_quiet(self):
        with tempfile.TemporaryDirectory() as td:
            proj = _proj(td, [], [])
            msgs, log = self._log()
            O._log_stage_skip(log, proj, "recovery", OSError("connection reset"))
        self.assertIn("recovery: skipped", " ".join(msgs))
        self.assertNotIn("BUG", " ".join(msgs))
        self.assertNotIn("stage_bugs", proj.meta)

    def test_all_fail_open_stage_catches_route_through_the_helper(self):
        src = ORCH.read_text()
        self.assertEqual(src.count('_log_stage_skip(log, proj, "'), 4,   # call sites, not the def
                         "recovery / image-fallback / pre-assemble gate / self-heal")
        for stage in ("recovery", "image-fallback", "pre-assemble gate", "self-heal"):
            self.assertIn(f'_log_stage_skip(log, proj, "{stage}"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
