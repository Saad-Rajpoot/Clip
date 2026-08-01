"""The "wrong-character gate" could not detect a wrong character.

The owner reported that breakouts almost never appear and that relevant footage is missed for the
same reason. Both are true, and the cause is one expression: `face_ids & main_cast`.

MEASURED on a real render (4,586 shots, 110 sources, 74% at >=720p):
  - Face-ID names only 13.4% of shots. 52.6% have faces DETECTED but left unnamed; 34% have none.
  - **0 of 625 Face-ID name instances fall outside the main cast**, because `match()` is an argmax
    over a roster containing only main cast. So `face_ids & main_cast` is bit-identical to
    `face_ids != {}` — a "no Face-ID" gate wearing a wrong-character label.
  - It rejected 25 breakout candidates, every one merely UNIDENTIFIED. Of the 18 distinct shots,
    10 plainly show main cast (including Tommen on the Iron Throne abolishing trial by combat, the
    video's central scene, wanted by 7 beats); only 3 are genuinely off-cast. Precision 3/18 = 17%.
  - And it PASSED the real thing: 7 of 15 named candidates carry a main-cast face that is NOT the
    beat's required person. All 7 passed; 2 aired.

Three states now, never conflated — RIGHT, WRONG, UNKNOWN. Replayed over 638 candidate shots:
rejections labelled "wrong character" go 525 (all unknowns) -> 71 (all genuinely mismatched), 124
are labelled unidentified, and 330 shots stop being discarded.

THE FOOTAGE SIDE. `_face_targets` was an exact dict lookup, so 27 of 107 character-beats could never
match a Face-ID result (which holds ACTOR names): 'Cersei', 'Ser Gregor Clegane', 'Tommen'. On those
the +0.30 face bonus never fired AND the -0.50 wrongface penalty fired on shots of the CORRECT
person — 1,295 (beat, shot) pairs taking a 0.80 swing the wrong way. Resolution now reaches 102 of
107 beats, and enlarging the target set can only ever SHRINK the wrong set, never grow it.

MEASURED AND REJECTED — do not retry without new evidence: re-running Face-ID at native resolution
instead of the 512px keyframe names 23 more shots per 300, but 5 of those 23 are WRONG (21.7%)
against a 4.8% baseline, and the score does not separate right from wrong (wrong: 0.363-0.432;
right: 0.364-0.489). At 512 a hard face fails to embed and stays honestly unnamed; at native res it
yields a low-information embedding the closed-world argmax must name.

    python3 -m pytest tests/test_faceid_identity_gate.py -q

No network, no LLM.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.match import resolve_face_targets as R    # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"

# the real roster from the render this came from
CAST = {
    "cersei lannister": "Lena Headey",
    "high sparrow": "Jonathan Pryce",
    "tommen baratheon": "Dean-Charles Chapman",
    "kevan lannister": "Ian Gelder",
    "qyburn": "Anton Lesser",
    "ser gregor clegane / the mountain": "Hafþór Júlíus Björnsson",
    "margaery tyrell": "Natalie Dormer",
    "loras tyrell": "Finn Jones",
    "jaime lannister": "Nikolaj Coster-Waldau",
    "daenerys targaryen": "Emilia Clarke",
}


class TestTheResolver(unittest.TestCase):
    """Every entity the old exact-lookup could not reach."""

    def test_a_given_name_alone_resolves(self):
        for ent, want in (("Cersei", "lena headey"), ("Tommen", "dean-charles chapman"),
                          ("Daenerys", "emilia clarke"), ("Qyburn", "anton lesser")):
            t, full = R(ent, CAST)
            self.assertIn(want, t, ent)
            self.assertTrue(full, ent)

    def test_honorifics_are_stripped_from_both_sides(self):
        """'Ser Gregor Clegane' must meet a roster entry stored as 'Ser Gregor Clegane / The
        Mountain' — the alias needs stripping too, not just the query."""
        t, _ = R("Ser Gregor Clegane", CAST)
        self.assertIn("hafþór júlíus björnsson", t)
        self.assertIn("hafþór júlíus björnsson", R("The Mountain", CAST)[0])
        self.assertIn("dean-charles chapman", R("King Tommen", CAST)[0])

    def test_high_is_not_an_honorific(self):
        """'High Sparrow' IS a roster name; stripping 'high' turned it into 'sparrow' and it
        resolved to nothing."""
        self.assertIn("jonathan pryce", R("High Sparrow", CAST)[0])

    def test_a_shared_surname_never_resolves(self):
        """A false target is what lets a wrong character through, so these must stay empty."""
        for ent in ("Tyrell", "Lannister", "the High Septon", "Grand Maester Pycelle",
                    "Lord Tywin Lannister", "the Hound", "Sansa Stark"):
            self.assertEqual(R(ent, CAST)[0], set(), ent)

    def test_a_partial_list_is_flagged_so_WRONG_can_never_be_claimed(self):
        """When a beat names someone off-roster we cannot conclude a shot is the wrong person —
        only that we do not know. `full` is what stops the gate claiming otherwise."""
        t, full = R("Cersei and Missandei", CAST)
        self.assertIn("lena headey", t)
        self.assertFalse(full)
        self.assertFalse(R("Missandei, Jon, Olenna", CAST)[1])

    def test_empty_input_is_safe(self):
        self.assertEqual(R("", CAST), (set(), False))
        self.assertEqual(R("Cersei", {}), (set(), False))

    def test_enlarging_targets_can_only_shrink_the_wrong_set(self):
        """Monotonicity — the property that makes this change incapable of creating a wrong
        character anywhere: wrong = (conf & all_faces) - targets, guarded by not (conf & targets)."""
        allf = {v.lower() for v in CAST.values()} | set(CAST)
        conf = {"lena headey"}
        for small, big in ((set(), {"lena headey"}),
                           ({"anton lesser"}, {"anton lesser", "lena headey"})):
            w_small = bool((conf & allf) and not (conf & small))
            w_big = bool((conf & allf) and not (conf & big))
            self.assertTrue(w_small >= w_big)


class TestTheGateSeparatesThreeStates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (SRC / "build.py").read_text()

    def test_the_old_expression_is_gone(self):
        self.assertNotIn("_fids9 & _main_faces9", self.src)

    def test_wrong_needs_a_CONFIDENT_name_and_a_FULLY_resolved_target(self):
        self.assertIn("_conf9 and _tgt9 and _full9 and not (_conf9 & _tgt9)", self.src)

    def test_unknown_is_counted_and_named_separately(self):
        self.assertIn('_rej["unidentified"]', self.src)
        self.assertIn("unidentified, not wrong", self.src)
        self.assertIn("unidentified={_rej['unidentified']}", self.src)

    def test_unknown_survives_on_scene_evidence(self):
        """A source proven to BE this scene is evidence no face crop can give."""
        self.assertIn("_scene_proof9 = _verbatim_ok or (c[2].id in _tier1) or (c[2].id in _tier2)",
                      self.src)

    def test_only_confident_names_count(self):
        """The identities blob carries empty-string names for unconfident faces; they must not be
        read as an identification."""
        self.assertIn('if str(f).strip()', self.src.split("_conf9 = ")[1][:160])

    def test_the_verbatim_override_still_logs(self):
        """It had become unreachable under the new `if` — an elif on a condition that is true."""
        seg = self.src.split("audio-match overrides face guard")[0][-400:]
        self.assertIn("if not _conf9 and _verbatim_ok:", seg)


class TestTheBackfill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (SRC / "build.py").read_text()

    def test_selection_goes_deeper_than_the_air_cap(self):
        """8 were picked and 3 aired: five died after extraction and nothing replaced them, while
        32 candidates were never evaluated."""
        self.assertIn("len(picked) >= n_max + _bk_reserve", self.src)

    def test_the_reserve_never_airs_more_breakouts(self):
        self.assertIn("if len(out) >= n_max:", self.src)
        self.assertIn("the reserve exists to REPLACE deaths", self.src)

    def test_it_is_tunable_and_cannot_raise_NameError(self):
        """This function has no module-level `os`; a bare one raises inside a fail-open catch,
        which is how a whole stage died silently for months in this project."""
        self.assertIn("VIDLORE_CLIPSTUDIO_BREAKOUT_RESERVE", self.src)
        self.assertIn("import os as _os_bk", self.src)
        self.assertIn("_os_bk.environ.get(\"VIDLORE_CLIPSTUDIO_BREAKOUT_RESERVE\"", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
