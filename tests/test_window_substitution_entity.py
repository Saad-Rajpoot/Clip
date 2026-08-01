"""A beat can air a window the verifier never confirmed, and lose its named subject doing it.

Found by the frame-level audit of job fc41397ea5, whose worst example burns the caption "Olenna
Tyrell" in the accent colour over 2.6 seconds of MARGAERY Tyrell smiling in a garden, under a line
calling her "a woman who had been dead since the season before" — the essay's punchline.

ROOT CAUSE, traced rather than guessed. `beat_windows[0]` is the pick the verifier confirmed; the
rest are match-ranked alternates that no verifier ever saw. A beat is pushed off [0] far more often
than it looks:

  - 31% of distinct windows appear in MORE THAN ONE beat's list (196 of 636 on this render), and a
    window airs only ONCE ever, so whichever beat comes first in the timeline claims it;
  - simulating that rule alone, 22 of 181 beats (12%) are forced onto an alternate BEFORE the
    look-variety sort, the burned-text probe or the darkness probe have even run.

Most of that is harmless and must stay harmless: 14 of the 22 substitutes came from a source whose
title still names the beat's required entity — frequently the same scene from a different upload,
which is what a deep pool is for. The damage is the other kind, measured at 3 of 181 beats: a beat
requiring "Qyburn" landing on a Gregor Clegane character study, one requiring "The Mountain" landing
on the Sept explosion, one requiring "Cersei Lannister" landing on an explosion b-roll upload.

WHY THE TITLE AND NOT FACE-ID: the identities persisted for these very windows are empty — `[]`,
`['']`, `['', '']` — so an identity test fails open on exactly the beats that need it. The uploader's
title is the evidence that actually exists.

    python3 -m pytest tests/test_window_substitution_entity.py -q

No network, no LLM.
"""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "vidlore" / "clipstudio"
JOB = Path("/Users/hussnain/Desktop/clipstudio_output/portal/fc41397ea5")

STOP = {"the", "of", "and"}


def toks(s):
    return {t for t in re.findall(r"[a-z']+", (s or "").lower())
            if len(t) > 2 and t not in STOP}


def carries(title, entity):
    e = toks(entity)
    return bool(e) and bool(e & toks(title))


class TestTheRuleItself(unittest.TestCase):
    """The predicate, on the exact titles from the render."""

    def test_a_substitute_that_keeps_the_subject_is_allowed(self):
        # beat 51: "Great Sept of Baelor" — the same event from another upload
        self.assertTrue(carries("Game of Thrones 6x10: Sept of Baelor Explosion (Additional)",
                                "Great Sept of Baelor"))

    def test_a_substitute_that_drops_the_subject_is_caught(self):
        # beat 54: the beat requires Qyburn; the substitute is a Gregor Clegane character study
        self.assertTrue(carries("Game Of Thrones: Cersei & Qyburn | Pinky & The Brain", "Qyburn"))
        self.assertFalse(carries("Game of Thrones: The Mountain - Gregor Clegane Character Study",
                                 "Qyburn"))

    def test_the_mountain_case(self):
        self.assertTrue(carries("Game of Thrones 6x08 - The Mountain vs Faith Militant",
                                "The Mountain"))
        self.assertFalse(carries("Game of Thrones 6x10   Cersei Blows Up the Sept", "The Mountain"))

    def test_the_show_name_alone_never_counts_as_the_subject(self):
        """Every title says 'Game of Thrones'; only tokens of the ENTITY are compared, so a shared
        franchise word can never make a substitute look correct."""
        self.assertFalse(carries("Game of Thrones 6x10 Cersei Blows Up the Sept", "Qyburn"))

    def test_short_words_and_articles_do_not_match(self):
        self.assertFalse(carries("The Sept of Baelor", "The"))
        self.assertFalse(carries("Cersei and the Mountain", "of"))


class TestTheWiring(unittest.TestCase):
    def test_it_only_defends_an_entity_the_VERIFIED_window_carried(self):
        """If beat_windows[0] never named the subject either, there is nothing to preserve and the
        rule must stay out of the way."""
        src = (SRC / "build.py").read_text()
        self.assertIn("_w0_holds = bool(_ent_toks) and bool(", src)
        self.assertIn("_title_toks(windows_avail[0][0])", src)

    def test_it_is_a_pass_one_skip_so_coverage_cannot_shrink(self):
        """Same contract as the legibility probe: pass 2 is the old loop, so a beat whose every
        alternate drops the subject still airs exactly what it airs today."""
        src = (SRC / "build.py").read_text()
        self.assertIn("if (_lpass == 1 and _w0_holds", src)
        self.assertIn("substitute drops the beat's named subject", src)

    def test_a_same_source_window_is_never_blocked(self):
        """Another window of the SAME upload is the same scene — the rule must not fire on it."""
        src = (SRC / "build.py").read_text()
        self.assertIn("_wh[0][0] != windows_avail[0][0]", src)


class TestMeasuredOnTheRender(unittest.TestCase):
    """The numbers in the docstring, re-derived from the job so they cannot rot silently."""

    @classmethod
    def setUpClass(cls):
        if not (JOB / "project.json").exists():
            raise unittest.SkipTest("job fc41397ea5 not on disk")
        cls.d = json.loads((JOB / "project.json").read_text())
        cls.titles = {s["id"]: (s.get("title") or "") for s in cls.d["sources"]}
        cls.segs = {s["index"]: s for s in cls.d["segments"]}

    def forced(self):
        aired, out = set(), []
        for sel in sorted(self.d.get("selections", []), key=lambda s: s["segment_index"]):
            ws = sel.get("beat_windows") or []
            if not ws:
                continue
            for i, w in enumerate(ws):
                k = (w[0], round(float(w[1]), 2))
                if k in aired:
                    continue
                aired.add(k)
                if i:
                    out.append((sel["segment_index"], i, ws))
                break
        return out

    def test_windows_really_are_shared_between_beats(self):
        owner = {}
        for sel in self.d.get("selections", []):
            for w in sel.get("beat_windows") or []:
                owner.setdefault((w[0], round(float(w[1]), 2)), set()).add(sel["segment_index"])
        shared = [k for k, v in owner.items() if len(v) > 1]
        self.assertGreater(len(shared) / max(1, len(owner)), 0.20,
                           "the air-once rule only bites because lists overlap")

    def test_the_air_once_rule_alone_displaces_about_an_eighth_of_beats(self):
        f = self.forced()
        self.assertGreaterEqual(len(f), 15)
        self.assertLessEqual(len(f), 40)

    def test_most_substitutions_are_harmless_and_must_stay_allowed(self):
        kept = sum(1 for b, i, ws in self.forced()
                   if carries(self.titles.get(ws[i][0], ""),
                              self.segs[b].get("required_entity") or ""))
        self.assertGreaterEqual(kept, 10, "the rule must not fire on same-scene substitutions")

    def test_the_rule_fires_on_only_a_handful(self):
        bad = [b for b, i, ws in self.forced()
               if carries(self.titles.get(ws[0][0], ""), self.segs[b].get("required_entity") or "")
               and not carries(self.titles.get(ws[i][0], ""),
                               self.segs[b].get("required_entity") or "")]
        self.assertLessEqual(len(bad), 6, f"expected a narrow rule, got {bad}")
        self.assertIn(54, bad, "the Qyburn beat is the reference case")


if __name__ == "__main__":
    unittest.main(verbosity=2)
