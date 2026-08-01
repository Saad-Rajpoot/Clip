"""Bounded recovery kept searching for the same beats and never reached the rest.

MEASURED on portal job 409e284b60 (recovery_audit.json + build.log). 32 beats were unresolved and
the per-round cap is 8. The selection ranked by policy class then script order — a DETERMINISTIC
key — so each round took the same head of the same list:

    round 1  [90, 110, 166, 76, 89, 91, 12, 13]     3 recovered
    round 2  [90, 110,  76, 79, 89, 12, 13, 19]     "found no NEW source"

Six of round 2's eight were re-attempts, and its audit reads `candidates_found 21,
new_candidates 0`: re-issuing a query YouTube has already answered returns the sources that are
already downloaded. The cost was paid and nothing could come back.

Meanwhile these never got a single search, and they are the ones the finished video visibly
misses — every one names an iconic, findable moment:

    22  the Hound cuts a wight in half in the Dragonpit
    60  the Children of the Forest make the Night King
    94  Arya kills the Night King in the godswood
   122  the wight torso crawling on the Dragonpit sand
   144  Benjen Stark's last stand beyond the Wall
   168  Arya kills the Night King / the dagger

Fix: within its class, a beat that has never been searched outranks one already tried. Re-attempts
are kept — a later round has a bigger pool — but only after every beat that has had no turn.

    python3 -m pytest tests/test_recovery_rotation.py -q

No network, no LLM.
"""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.orchestrate import recovery_pick                  # noqa: E402

CHARACTER, FILLER, ABSTRACT, EXACT = "character", "filler", "abstract", "exact"


class Seg:
    def __init__(self, i, pol=FILLER, q="a query", ent=""):
        self.index, self._pol, self.scene_query, self.required_entity = i, pol, q, ent


def policy_mod(mapping):
    m = types.SimpleNamespace(CHARACTER=CHARACTER, FILLER=FILLER, ABSTRACT=ABSTRACT, EXACT=EXACT)
    m.policy_of = lambda s: mapping.get(getattr(s, "index", -1), FILLER)
    return m


#  the render's real unresolved list, with each beat's real policy class
REAL = [12, 13, 19, 22, 32, 33, 37, 54, 60, 72, 75, 76, 79, 82, 89, 90, 94, 95, 96, 109, 110,
        111, 115, 116, 122, 128, 140, 144, 148, 162, 163, 168]
CLASSES = {90: CHARACTER, 110: CHARACTER, 166: CHARACTER, 76: FILLER, 89: FILLER, 91: FILLER,
           12: EXACT, 13: EXACT, 79: FILLER, 19: EXACT}
SEGS = {i: Seg(i, CLASSES.get(i, FILLER)) for i in REAL}
POL = policy_mod(CLASSES)


class TestRotation(unittest.TestCase):
    def _old_pick(self, pool):
        """The ranking as it shipped: class, then script order. No memory of past rounds."""
        return sorted(pool, key=lambda i: ({CHARACTER: 0, FILLER: 1, ABSTRACT: 1,
                                            EXACT: 2}[CLASSES.get(i, FILLER)], i))[:8]

    def test_a_second_round_reaches_beats_the_first_never_searched(self):
        r1 = recovery_pick(REAL, SEGS, POL, set(), 8)
        left = [i for i in REAL if i not in (166, 91)]
        self.assertEqual(self._old_pick(left), r1,
                         "reproduce the bug first: the old ranking re-picks the same eight")
        r2 = recovery_pick(left, SEGS, POL, set(r1), 8)
        self.assertEqual(len(r2), 8)
        self.assertEqual(len(set(r1) & set(r2)), 2,
                         f"only the two CHARACTER beats may repeat — no fresh beat in that "
                         f"class outranks them: {r2}")
        self.assertTrue(set(r2) & {72, 75, 76, 79, 82, 89},
                        "beats that had never been searched must now get a turn")

    def test_three_rounds_reach_twenty_beats_instead_of_eight(self):
        """The whole point. Same cap, same cost, 2.5x the coverage."""
        def sweep(rotate):
            seen, tried = set(), set()
            for _ in range(3):
                pick = recovery_pick(REAL, SEGS, POL, tried, 8) if rotate \
                    else self._old_pick(REAL)
                seen |= set(pick)
                if rotate:
                    tried |= set(pick)
            return len(seen)
        self.assertEqual(sweep(False), 8)
        self.assertEqual(sweep(True), 20)

    def test_class_priority_still_wins_over_freshness(self):
        """A never-searched EXACT beat must NOT jump ahead of a re-attempted CHARACTER beat —
        character beats are the ones that block releases."""
        pick = recovery_pick([90, 12], SEGS, POL, {90}, 2)
        self.assertEqual(pick, [90, 12])

    def test_a_beat_with_no_query_material_never_burns_a_slot(self):
        segs = dict(SEGS)
        segs[999] = Seg(999, CHARACTER, q="", ent="")
        self.assertNotIn(999, recovery_pick([999] + REAL, segs, POL, set(), 8))

    def test_required_entity_alone_is_enough_to_qualify(self):
        segs = {7: Seg(7, CHARACTER, q="", ent="Benjen Stark")}
        self.assertEqual(recovery_pick([7], segs, POL, set(), 8), [7])

    def test_the_order_is_still_deterministic(self):
        a = recovery_pick(REAL, SEGS, POL, {12, 90}, 8)
        b = recovery_pick(list(reversed(REAL)), SEGS, POL, {12, 90}, 8)
        self.assertEqual(a, b)

    def test_it_never_returns_more_than_the_cap(self):
        self.assertEqual(len(recovery_pick(REAL, SEGS, POL, set(), 8)), 8)
        self.assertEqual(recovery_pick(REAL, SEGS, POL, set(), 0), [])

    def test_everything_already_tried_still_gets_retried(self):
        """Rotation de-prioritises; it must never starve. A later round has a bigger pool."""
        pick = recovery_pick(REAL, SEGS, POL, set(REAL), 8)
        self.assertEqual(len(pick), 8)


class TestItIsRecordedBeforeTheAttempt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "vidlore" / "clipstudio" / "orchestrate.py").read_text()

    def test_the_attempt_is_persisted_on_the_project(self):
        self.assertIn('proj.meta["recovery_attempted"] = sorted(_tried | set(unresolved))', self.src)

    def test_it_is_recorded_BEFORE_the_search_runs(self):
        """A round that dies mid-way must still spend the beat's turn, or a beat whose search
        reliably crashes monopolises the cap for ever."""
        fn = self.src.split("def _recover_unresolved_beats")[1]
        self.assertLess(fn.index('proj.meta["recovery_attempted"]'),
                        fn.index("cands = discover_sources("))

    def test_the_audit_says_which_beats_are_new_this_round(self):
        self.assertIn('audit["first_attempt_this_round"] = _fresh', self.src)
        self.assertIn("never searched before", self.src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
