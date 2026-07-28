"""Incident advisor — LLM triages unexpected failures, deterministic code acts from a
closed menu. Fully offline (mocked LLM).

    python3 tests/test_incident_advisor.py
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio import incident as INC              # noqa: E402
from vidlore.clipstudio import orchestrate as ORC           # noqa: E402
from vidlore.clipstudio.models import ClipProject           # noqa: E402
from vidlore.clipstudio.verify import (                     # noqa: E402
    NonRetryableBuildError, VisionBackendError)


def _proj(td):
    p = ClipProject(name="t", root=str(td))
    p.ensure_dirs()
    return p


class TestAdvise(unittest.TestCase):
    def test_menu_action_parsed_and_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            with mock.patch("vidlore.clipstudio.llm.complete_ex",
                            return_value=('{"action": "retry_after_wait", "wait_s": 60, '
                                          '"why": "YouTube throttle"}', {"served": "deepseek"})):
                v = INC.advise("download", RuntimeError("HTTP 429"), proj=p, log=lambda m: None)
            self.assertEqual(v["action"], "retry_after_wait")
            self.assertEqual(v["wait_s"], 60)
            rep = json.loads((p.output_dir / "incident_report.json").read_text())
            self.assertEqual(len(rep), 1)
            self.assertIn("context", rep[0])

    def test_non_menu_action_becomes_abort(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            with mock.patch("vidlore.clipstudio.llm.complete_ex",
                            return_value=('{"action": "disable_the_release_gate"}', {})):
                v = INC.advise("build", RuntimeError("x"), proj=p, log=lambda m: None)
            self.assertEqual(v["action"], "abort", "off-menu commands must never execute")

    def test_llm_unreachable_aborts(self):
        with mock.patch("vidlore.clipstudio.llm.complete_ex",
                        side_effect=RuntimeError("no api")):
            v = INC.advise("verify", RuntimeError("x"), proj=None, log=lambda m: None)
        self.assertEqual(v["action"], "abort")

    def test_kill_switch(self):
        with mock.patch.dict(os.environ, {"VIDLORE_CLIPSTUDIO_INCIDENT_ADVISOR": "0"}), \
                mock.patch("vidlore.clipstudio.llm.complete_ex") as llm:
            v = INC.advise("any", RuntimeError("x"), proj=None, log=lambda m: None)
        self.assertEqual(v["action"], "abort")
        llm.assert_not_called()

    def test_wait_clamped(self):
        with mock.patch("vidlore.clipstudio.llm.complete_ex",
                        return_value=('{"action": "retry_after_wait", "wait_s": 99999}', {})):
            v = INC.advise("s", RuntimeError("x"), proj=None, log=lambda m: None)
        self.assertLessEqual(v["wait_s"], 300)


class TestResilientWrapper(unittest.TestCase):
    def _run(self, td, produce_effects, advise_actions, env=None):
        calls = {"produce": 0, "kw": []}

        def fake_produce(project_dir, **kw):
            calls["produce"] += 1
            calls["kw"].append(dict(kw))
            eff = produce_effects[min(calls["produce"] - 1, len(produce_effects) - 1)]
            if isinstance(eff, Exception):
                raise eff
            return eff

        adv = iter(advise_actions)
        with mock.patch.object(ORC, "produce_auto", side_effect=fake_produce), \
                mock.patch.object(INC, "advise",
                                  side_effect=lambda *a, **k: next(adv)), \
                mock.patch.object(INC, "interventions_used", return_value=0), \
                mock.patch.dict(os.environ, env or {}):
            try:
                out = ORC.produce_auto_resilient(str(td), progress=lambda m: None)
                return out, calls, None
            except Exception as e:                       # noqa: BLE001
                return None, calls, e

    def test_retry_resumes_from_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            _proj(td)
            out, calls, err = self._run(
                td, [RuntimeError("flaky subprocess"), {"ok": True}],
                [{"action": "retry", "wait_s": 0, "why": "transient"}])
            self.assertIsNone(err)
            self.assertEqual(out, {"ok": True})
            self.assertEqual(calls["produce"], 2)
            self.assertTrue(calls["kw"][1].get("resume"), "retry must resume from checkpoints")

    def test_abort_propagates(self):
        with tempfile.TemporaryDirectory() as td:
            _proj(td)
            out, calls, err = self._run(
                td, [RuntimeError("disk full")],
                [{"action": "abort", "wait_s": 0, "why": "not transient"}])
            self.assertIsNotNone(err)
            self.assertEqual(calls["produce"], 1)

    def test_content_errors_never_advised(self):
        with tempfile.TemporaryDirectory() as td:
            _proj(td)
            for exc in (NonRetryableBuildError("footage gap"),
                        VisionBackendError("billing")):
                out, calls, err = self._run(td, [exc], [])
                self.assertIsNotNone(err)
                self.assertEqual(calls["produce"], 1,
                                 "content/billing failures keep their own machinery")

    def test_intervention_cap(self):
        with tempfile.TemporaryDirectory() as td:
            _proj(td)
            with mock.patch.object(ORC, "produce_auto",
                                   side_effect=RuntimeError("always fails")), \
                    mock.patch.object(INC, "advise",
                                      return_value={"action": "retry", "wait_s": 0,
                                                    "why": "t"}), \
                    mock.patch.object(INC, "interventions_used", side_effect=[0, 1, 2, 3]):
                with self.assertRaises(RuntimeError):
                    ORC.produce_auto_resilient(str(td), progress=lambda m: None)


class TestWiring(unittest.TestCase):
    def test_portal_uses_resilient_wrapper(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "web.py").read_text()
        self.assertIn("produce_auto_resilient as produce_auto", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
