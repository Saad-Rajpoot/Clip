"""Three beats of non-show material, found by the paired frame A/B of arya_v2.

Each was re-pulled and looked at by a second agent, and each is ABSENT from the render arya_v2
replaces — so they arrived with this render, not with the script.

  beat  10  a fan-made portrait of Jon Snow with glowing wight-blue eyes and painted frost cracks,
            inside a source titled "Game of Thrones - Ice Dragon (Nights King kills and resurrects
            Viserion)" — an ordinary scene title. Nothing in the title marks it. VISION's problem.
  beat  42  BEHIND-THE-SCENES: the Winterfell yard with the film crew in frame — camera dolly,
            operator, equipment cases, modern jackets and baseball caps along the wall. The source,
            "Arya's 8 Years Journey on Game Of Thrones Killing The Night King", is a legitimate
            retrospective; blocking such titles would starve footage, so this is VISION's too — and
            the non-show hard rule had no clause for production footage, which is REAL footage OF
            the show and therefore slips every clause it did have.
  beat 158  a stylised still from "Jon Snow vs The Night's King Army Lightsaber Battle - Game Of
            Thrones + Star Wars" — a crossover fan edit, and the one of the three that a TITLE can
            catch.

    python3 -m pytest tests/test_nonshow_leaks_arya_v2.py -q

No network, no LLM.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.discover import (_NONSHOW_TITLE,                    # noqa: E402
                                        is_unwanted_source_title)
from vidlore.clipstudio import verify as V                                 # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"
LEAK = ("Jon Snow vs The Night’s King Army Lightsaber Battle - "
        "Game Of Thrones + Star Wars")


class TestCrossoverFanEdits(unittest.TestCase):
    def test_the_leaked_title_is_caught(self):
        self.assertTrue(_NONSHOW_TITLE.search(LEAK))
        self.assertTrue(is_unwanted_source_title(LEAK))

    def test_other_crossover_framings(self):
        for t in ("Game of Thrones x Marvel crossover",
                  "GoT mashup with Lord of the Rings",
                  "Game of Thrones vs Star Wars epic",
                  "Witcher x Game of Thrones",
                  "Game of Thrones meets Harry Potter"):
            self.assertTrue(_NONSHOW_TITLE.search(t), t)

    def test_a_franchise_NAME_alone_is_not_evidence(self):
        """The measured false positive: a genuine Long Night upload that merely CREDITS ITS MUSIC.
        Dropping it would starve exactly the footage this video is about."""
        t = ("Game of Thrones | The Long Night | Arya Battle Scene | "
             "Forge from Avengers: Endgame")
        self.assertFalse(_NONSHOW_TITLE.search(t))

    def test_ordinary_scene_titles_survive(self):
        for t in ("Game of Thrones S08E03 Arya Stark Kills Night King",
                  "GoT S08E03 - Arya kills the Night King",
                  "Game of Thrones 8x03 The Long Night",
                  "Arya's 8 Years Journey on Game Of Thrones Killing The Night King",
                  "Game of Thrones - Ice Dragon (Nights King kills and resurrects Viserion)"):
            self.assertFalse(_NONSHOW_TITLE.search(t), t)

    def test_an_episode_code_is_not_a_crossover_connector(self):
        """`x` joins franchises AND episode numbers — 8x03 must not read as a mashup."""
        self.assertFalse(_NONSHOW_TITLE.search("Game of Thrones 8x03 Arya vs the dead"))

    def test_props_that_cannot_exist_in_this_show_stand_alone(self):
        for t in ("Jon Snow lightsaber duel", "Arya jedi training", "Hogwarts letter arrives"):
            self.assertTrue(_NONSHOW_TITLE.search(t), t)

    def test_measured_on_the_real_pools(self):
        """1 of 225 titles in arya_v2, 0 of 119 in the render it replaces — the rule is narrow."""
        import json
        for job, want in (("arya_v2", 1), ("5cab63d801", 0)):
            p = Path(f"/Users/hussnain/Desktop/clipstudio_output/portal/{job}/project.json")
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            hits = [s for s in d["sources"] if _NONSHOW_TITLE.search(s.get("title") or "")]
            self.assertEqual(len(hits), want, [s.get("title") for s in hits])


class TestBehindTheScenesIsNotTheShow(unittest.TestCase):
    """The clause the non-show hard rule was missing. BTS is REAL footage OF the production, so it
    passes 'is this the real show, with the real actors' — which is why it slipped."""

    def test_the_rule_names_the_crew_and_the_kit(self):
        src = (SRC / "verify.py").read_text()
        rule = src.split("_nonshow = (")[1][:2600]
        low = rule.lower()
        self.assertIn("behind-the-scenes", low)
        for word in ("crew", "dolly", "rehearsal", "blooper", "out of character"):
            self.assertIn(word, low, word)

    def test_it_explains_WHY_this_class_slips(self):
        src = (SRC / "verify.py").read_text()
        rule = src.split("_nonshow = (")[1][:2600]
        self.assertIn("REAL footage OF the production", rule)

    def test_low_res_and_dark_are_still_explicitly_fine(self):
        """The clause must not become a resolution or exposure gate — the Long Night is dark."""
        src = (SRC / "verify.py").read_text()
        rule = src.split("_nonshow = (")[1][:2600]
        self.assertIn("merely low-res, ", rule)
        self.assertIn("dark, blurry", rule)

    def test_the_prompt_version_was_bumped_with_it(self):
        """The verdict cache keys on PROMPT_VERSION. Changing the prompt without bumping it serves
        answers to a different question — the whole point of the fingerprint."""
        major = int(V.PROMPT_VERSION.split("-", 1)[0].lstrip("v"))
        self.assertGreaterEqual(major, 8, V.PROMPT_VERSION)
        self.assertIn("PROMPT_VERSION", (SRC / "verify.py").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
