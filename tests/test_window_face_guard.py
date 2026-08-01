"""The right shot, the wrong seconds — a wrong character no shot-level score can see.

MEASURED on portal job 409e284b60. Seven scenes aired a main-cast character the narration was not
talking about. Two of them are the interesting ones, because everything upstream was RIGHT:

    beat  71  required_entity 'Benjen Stark'  faceid 1.0  identity 'Joseph Mawle'
    beat 160  required_entity 'Jon Snow'      faceid 1.0  identity 'Kit Harington'

The chosen shot carried the correct actor and scored accordingly — and the seconds actually cut
from it showed somebody else. By the time a window is cut the scoring is over, so the shot-level
`wrongface` penalty cannot reach it. (`wrongface` was in fact None on all seven: on the other five
the shot had no confident face at all, or the beat named an object/event, so there was nothing for
a shot-level rule to fire on either.)

The guard treats seconds showing a confidently-named DIFFERENT main-cast member as dirty, so the
existing window-QC machinery moves the cut off them. Three properties make it safe:

  * THREE STATES, as everywhere else — wrong needs a CONFIDENT name, that name in the main cast,
    and the beat's entity FULLY resolved. An unnamed face is unknown, never wrong.
  * STRICT SHORTEN-ONLY — the window is computed twice, once with the identity spans and once
    exactly as before, and the identity pass is used only when it does not reject. It can move a
    cut; it can never cost a shot. Footage is never starved by it.
  * The moment rules are untouched: a shortened window must still keep the beat's moment.

    python3 -m pytest tests/test_window_face_guard.py -q

No network, no LLM.
"""
import itertools
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.match import (clean_cut_window, face_guard_for,      # noqa: E402
                                      _wrong_face_spans)

CAST = {"benjen stark": "Joseph Mawle", "jon snow": "Kit Harington",
        "lyanna stark": "Aisling Franciosi"}
ALL = {"benjen stark", "joseph mawle", "jon snow", "kit harington",
       "lyanna stark", "aisling franciosi"}
BENJEN = ({"joseph mawle"}, ALL, True)


def shot(i, a, z, names=(), confident=True):
    return NS(index=i, start=a, end=z,
              identities=[{"name": n, "confident": confident} for n in names],
              subs_flag=-1, dark_flag=-1, graphics_flag=-1, ocr_text_flag=-1,
              corner_masks=None, badge_flag=-1)


class TestItFindsTheWrongSeconds(unittest.TestCase):
    def test_a_span_showing_another_main_cast_member_is_dirty(self):
        shots = [shot(0, 0, 5, ["Joseph Mawle"]), shot(1, 5, 9, ["Kit Harington"])]
        got = _wrong_face_spans(shots, 0, 9, BENJEN)
        self.assertEqual([(s, e) for s, e, _ in got], [(5.0, 9.0)])

    def test_the_window_moves_off_them(self):
        shots = [shot(0, 0, 6, ["Joseph Mawle"]), shot(1, 6, 10, ["Kit Harington"])]
        t0, t1, act, why = clean_cut_window(shots, 0, 10, 3.0, anchor=(0, 6),
                                            face_guard=BENJEN)
        self.assertEqual(act, "shortened")
        self.assertEqual((t0, t1), (0.0, 6.0))
        self.assertIn("wrong-face", why)

    def test_without_the_guard_the_same_window_is_kept_whole(self):
        shots = [shot(0, 0, 6, ["Joseph Mawle"]), shot(1, 6, 10, ["Kit Harington"])]
        self.assertEqual(clean_cut_window(shots, 0, 10, 3.0, anchor=(0, 6))[2], "ok")


class TestTheThreeStatesAreNeverConflated(unittest.TestCase):
    def test_an_unnamed_face_is_unknown_not_wrong(self):
        shots = [shot(0, 0, 5, ["Joseph Mawle"]), shot(1, 5, 9, ["Kit Harington"], confident=False)]
        self.assertEqual(_wrong_face_spans(shots, 0, 9, BENJEN), [])

    def test_a_shot_with_no_identities_at_all_is_unknown(self):
        self.assertEqual(_wrong_face_spans([shot(0, 0, 9)], 0, 9, BENJEN), [])

    def test_a_name_outside_the_main_cast_is_not_evidence(self):
        """The roster is what makes 'wrong' meaningful; an off-roster name proves nothing."""
        shots = [shot(0, 0, 9, ["Some Extra"])]
        self.assertEqual(_wrong_face_spans(shots, 0, 9, BENJEN), [])

    def test_a_shot_showing_the_target_TOO_is_kept(self):
        shots = [shot(0, 0, 9, ["Joseph Mawle", "Kit Harington"])]
        self.assertEqual(_wrong_face_spans(shots, 0, 9, BENJEN), [])

    def test_a_partially_resolved_entity_can_never_call_a_face_wrong(self):
        guard = face_guard_for(NS(required_entity="Benjen and Missandei"), CAST, ALL)
        self.assertIsNone(guard)

    def test_an_object_or_event_beat_builds_no_guard(self):
        for ent in ("dragonglass", "Dragonpit wight demonstration", ""):
            self.assertIsNone(face_guard_for(NS(required_entity=ent), CAST, ALL), ent)

    def test_a_named_character_builds_one(self):
        g = face_guard_for(NS(required_entity="Benjen Stark"), CAST, ALL)
        #  BOTH spellings are targets: a shot's identities may carry the character name or the
        #  actor's, and treating either as "not the target" is how the correct person got
        #  penalised before (1,295 pairs, see test_faceid_identity_gate).
        self.assertEqual(g[0], {"joseph mawle", "benjen stark"})
        self.assertTrue(g[2])


class TestItCanNeverStarveFootage(unittest.TestCase):
    """The property the whole design rests on: rejection can only ever shrink, never grow."""

    def test_an_all_wrong_window_is_kept_rather_than_lost(self):
        shots = [shot(0, 0, 10, ["Kit Harington"])]
        t0, t1, act, why = clean_cut_window(shots, 0, 10, 3.0, anchor=(0, 10),
                                            face_guard=BENJEN)
        self.assertEqual(act, "ok")
        self.assertEqual((t0, t1), (0, 10))

    def test_a_window_too_short_after_trimming_is_kept_whole(self):
        shots = [shot(0, 0, 2, ["Joseph Mawle"]), shot(1, 2, 10, ["Kit Harington"])]
        self.assertEqual(clean_cut_window(shots, 0, 10, 5.0, anchor=(0, 10),
                                          face_guard=BENJEN)[2], "ok")

    def test_the_guard_never_rejects_anything_the_old_path_accepted(self):
        """Exhaustive over every arrangement of three shots and both identities."""
        rng = random.Random(5)
        for names in itertools.product([["Joseph Mawle"], ["Kit Harington"], []], repeat=3):
            for _ in range(3):
                b = sorted(rng.sample(range(1, 10), 2))
                shots = [shot(0, 0, b[0], names[0]), shot(1, b[0], b[1], names[1]),
                         shot(2, b[1], 10, names[2])]
                for min_len in (1.0, 3.0, 6.0):
                    old = clean_cut_window(shots, 0, 10, min_len, anchor=(0, 10))
                    new = clean_cut_window(shots, 0, 10, min_len, anchor=(0, 10),
                                           face_guard=BENJEN)
                    if old[2] != "rejected":
                        self.assertNotEqual(new[2], "rejected", (names, b, min_len))

    def test_no_guard_means_byte_identical_behaviour(self):
        rng = random.Random(9)
        for _ in range(40):
            b = sorted(rng.sample(range(1, 10), 2))
            shots = [shot(0, 0, b[0], ["Kit Harington"]), shot(1, b[0], b[1]),
                     shot(2, b[1], 10, ["Joseph Mawle"])]
            self.assertEqual(clean_cut_window(shots, 0, 10, 2.0, anchor=(0, 10)),
                             clean_cut_window(shots, 0, 10, 2.0, anchor=(0, 10),
                                              face_guard=None))


class TestItIsWiredAndSwitchable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "vidlore" / "clipstudio" / "match.py").read_text()

    def test_selection_arbitration_passes_a_guard(self):
        self.assertIn("face_guard=(face_guard_for(seg, char2actor, all_faces)", self.src)

    def test_alternates_are_judged_by_the_same_guard(self):
        seg = self.src.split("_a_act, _a_why, _a_meta = validate_candidate_window")[1][:200]
        self.assertIn("face_guard=face_guard", seg)

    def test_it_has_its_own_kill_switch(self):
        self.assertIn("VIDLORE_CLIPSTUDIO_WRONGFACE_WINDOW_GATE", self.src)

    def test_the_fallback_call_is_the_old_one_verbatim(self):
        """Pass 2 must be the identical algorithm with no identity spans — that is what makes
        'can never reject more' true by construction rather than by test."""
        seg = self.src.split("def clean_cut_window")[1].split("def _clean_cut_window_inner")[0]
        self.assertIn("return _clean_cut_window_inner(shots, t0, t1, min_len, anchor, "
                      "partial_corner, preserve, [])", seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
