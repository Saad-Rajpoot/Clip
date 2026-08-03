"""Record the exact owned file that enters assembly.

The old schema documented unverified alternate/window-walk substitutions after they happened.
Schema 2 forbids those substitutions: every ordinary row binds the aired derivative to the beat's
selected ``seg_NNN.mp4`` and immutable selection hash, then persists before the lineage gate.

    python3 -m pytest tests/test_aired_window_record.py -q

No network, no LLM.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "vidlore" / "clipstudio" / "build.py"


class TestTheRecordIsWired(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SRC.read_text()

    def test_a_row_is_appended_where_the_clip_is_actually_cut(self):
        """The record is made beside the selection-locked derivative, not at match time."""
        i_cut = self.src.index("_owned = _fit_verified_selection_clip(")
        i_rec = self.src.index("_aired_windows.append(", i_cut)
        self.assertLess(i_cut, i_rec)
        self.assertLess(i_rec - i_cut, 5000, "the record must sit with the derivative it describes")

    def test_it_carries_the_fields_an_audit_needs(self):
        blk = self.src.split("_aired_windows.append(", 1)[1][:1400]
        for f in ('"beat"', '"original_beat"', '"clip"', '"file"', '"root_file"',
                  '"source_id"', '"source_title"', '"in"', '"need"', '"via"', '"ok"',
                  '"selection_binding"'):
            self.assertIn(f, blk, f)

    def test_ordinary_path_cannot_reach_legacy_alternate_or_walk(self):
        lock = self.src.index("# VERIFIED-SELECTION LOCK")
        stop = self.src.index("windows_avail = list", lock)
        block = self.src[lock:stop]
        self.assertIn("_selected_clip", block)
        self.assertIn("gbeat += k\n        continue", block)
        self.assertNotIn("VIDLORE_", block, "selection ownership must have no opt-out")

    def test_it_is_persisted_before_the_gates_that_can_raise(self):
        """A render that dies at the A/V or black gate is exactly when the record is most wanted."""
        i_write = self.src.index('"aired_windows.json"')
        i_gate = self.src.index("_assert_scene_lineage(")
        i_assemble = self.src.index("result = assemble(")
        self.assertLess(i_write, i_gate)
        self.assertLess(i_gate, i_assemble)

    def test_persisting_it_is_fail_closed(self):
        blk = self.src.split('"aired_windows.json"')[0][-400:] + \
            self.src.split('"aired_windows.json"')[1][:700]
        self.assertNotIn("except Exception", blk)
        self.assertIn("Persistence is part of both invariants", self.src)

    def test_schema_and_summary_name_owned_derivatives(self):
        self.assertIn('"schema": "aired_windows/2"', self.src)
        self.assertIn("owned selection derivative(s)", self.src)


class TestAgainstARealRenderIfPresent(unittest.TestCase):
    """Once a render has been produced with this code, the artifact must describe it."""

    def setUp(self):
        self.p = Path("/Users/hussnain/Desktop/clipstudio_output/portal/fc41397ea5/output/"
                      "aired_windows.json")
        if not self.p.exists():
            self.skipTest("no render produced with the record yet")
        d = json.loads(self.p.read_text())
        if d.get("schema") != "aired_windows/2":
            self.skipTest("present render predates the selection-lock schema")

    def test_schema_and_shape(self):
        d = json.loads(self.p.read_text())
        self.assertEqual(d.get("schema"), "aired_windows/2")
        self.assertTrue(d.get("clips"))
        r = d["clips"][0]
        self.assertIn(r["via"], ("selection", "selection_derivative",
                                 "verified_image", "breakout"))
        self.assertTrue(r.get("lineage_validated"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
