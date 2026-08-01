"""Record what actually aired, because the ledger does not.

`ledger.jsonl` stores the MATCH stage's pick. Build then re-selects: a window airs only once so a
later beat is pushed onto an alternate, the look-variety sort reorders, the burned-text and darkness
probes skip candidates, and when nothing survives a shot-walk fills the beat. None of that was
written down.

The cost is not hypothetical. A 41-agent audit of a delivered render read the ledger, named a
source, and described a scene that is not on screen at that timecode — on two different beats. It
also had to abandon a whole line of investigation ("does the aired span run past the shot it was
verified on?") because the aired source and in-point were unrecoverable: the ledger is written at
one time and the clips at another, and only the clips are true.

`output/aired_windows.json` closes that. One row per cut clip: beat, clip index, file, source id and
title, in-point, requested length, and HOW it was chosen — window[0] (the verified pick), an
alternate, or the shot walk. Written before the final gates so it survives a gate raise.

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
        """Not at selection time — at the cut, which is the last place the truth is known."""
        i_cut = self.src.index("rc = _recut_to_duration(src.local_path, start, src_need")
        i_rec = self.src.index("_aired_windows.append(")
        self.assertLess(i_cut, i_rec)
        self.assertLess(i_rec - i_cut, 1400, "the record must sit with the cut it describes")

    def test_it_carries_the_fields_an_audit_needs(self):
        blk = self.src.split("_aired_windows.append(")[1][:600]
        for f in ('"beat"', '"clip"', '"file"', '"source_id"', '"source_title"',
                  '"in"', '"need"', '"via"', '"ok"'):
            self.assertIn(f, blk, f)

    def test_provenance_distinguishes_the_verified_pick_from_a_substitute(self):
        """The whole point: an auditor must be able to see WHICH beats left window[0]."""
        for tag in ('"window[0]"', '"window[alt]"', '"walk"'):
            self.assertIn(tag, self.src, tag)

    def test_the_walk_is_the_default_so_a_missed_branch_cannot_look_verified(self):
        """Fail-safe direction: if a new code path forgets to tag itself, it must read as the
        LEAST trustworthy origin, never as the verified one."""
        self.assertIn('_aired_via = "walk"', self.src)
        i_default = self.src.index('_aired_via = "walk"')
        i_window = self.src.index('_aired_via = ("window[0]"')
        self.assertLess(i_default, i_window)

    def test_it_is_persisted_before_the_gates_that_can_raise(self):
        """A render that dies at the A/V or black gate is exactly when the record is most wanted."""
        i_write = self.src.index('"aired_windows.json"')
        i_sync = self.src.index("_sync = _avsync(result)")
        self.assertLess(i_write, i_sync)

    def test_persisting_it_can_never_fail_a_render(self):
        blk = self.src.split('"aired_windows.json"')[0][-400:] + \
            self.src.split('"aired_windows.json"')[1][:700]
        self.assertIn("except Exception", blk)
        self.assertIn("aired-window record skipped", blk)

    def test_the_summary_counts_the_two_risky_origins(self):
        self.assertIn('a.get("via") == "window[alt]"', self.src)
        self.assertIn('a.get("via") == "walk"', self.src)


class TestAgainstARealRenderIfPresent(unittest.TestCase):
    """Once a render has been produced with this code, the artifact must describe it."""

    def setUp(self):
        self.p = Path("/Users/hussnain/Desktop/clipstudio_output/portal/fc41397ea5/output/"
                      "aired_windows.json")
        if not self.p.exists():
            self.skipTest("no render produced with the record yet")

    def test_schema_and_shape(self):
        d = json.loads(self.p.read_text())
        self.assertEqual(d.get("schema"), "aired_windows/1")
        self.assertTrue(d.get("clips"))
        r = d["clips"][0]
        self.assertIn(r["via"], ("window[0]", "window[alt]", "walk"))
        self.assertGreaterEqual(float(r["in"]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
