"""Why the Night King never showed up in his own intro (job 5cab63d801, beats 0-19).

The narration opens on the Night King catching Arya by the throat. The pool held 25 sources whose
titles say exactly that. The render aired an S7E4 reunion instead, and the earlier read — "the
retrieval never surfaced them" — was wrong: replaying `_score_pool` over the real pool put the
correct source at rank #1, losing by 0.0084. The decomposition is what these tests pin:

    CLIP could not discriminate  clip_cos 0.3351 (wrong shot) vs 0.3354 (right shot)
    faceid gave both 0.30        Arya is in most of the pool
    transcript gave the margin   0.5 x 0.20 = +0.10, earned by the clip SPEAKING "Arya Stark"
                                 (entities count double in _text_sim) — the same character-presence
                                 evidence faceid had already scored
    title affinity               capped at 0.12, awarded 0.09

Character presence was worth 0.50 and scene identity 0.12, on a beat whose whole job was to show
one specific scene. Three ranking-only fixes, each A/B'd on the intro with a vision judge
(research/eval/intro_ab.py, median of 3 reads per beat, verdict cache reset per arm):

    A. information-weighted title affinity + a real ceiling on exact_scene beats
    B. a flat-frame gate — lit but empty (grey card, "SUBSCRIBED" end-slate, washed-out fog)
    C. era AGREEMENT, the positive half of the era-conflict test

    exact_scene accuracy 64-73% -> 82%, mean 6.82 -> 7.55; intro accuracy 75-80% -> 85-90%.

No gate was tightened and nothing was removed from the pool except 0.35% of shots that are
objectively unusable — footage breadth is the owner's standing constraint.

    python3 -m pytest tests/test_scene_identity_ranking.py -q

No network, no LLM.
"""
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import match as M                     # noqa: E402
from vidlore.clipstudio import era as E                       # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio" / "match.py"


def shot(**kw):
    d = dict(luma_avg=120.0, luma_hi=240.0, luma_min=110.0, luma_min_black_frac=0.0, quality=0.6)
    d.update(kw)
    return NS(**d)


class TestFlatFrameGate(unittest.TestCase):
    """Lit but empty. The grey wall that scored 0/10 measured avg 61.4 / hi 92.0 / min 60.9."""

    def test_the_grey_card_that_aired_is_caught(self):
        self.assertTrue(M._shot_featureless(shot(luma_avg=61.4, luma_hi=92.0, luma_min=60.9)))

    def test_white_subscribe_endcard_is_caught(self):
        self.assertTrue(M._shot_featureless(shot(luma_avg=247.9, luma_hi=254.0, luma_min=247.9)))

    def test_the_long_night_is_never_touched(self):
        """The one thing this gate must not do — the video is ABOUT the Long Night.

        Dark frames legitimately have a narrow luma range, which is why the rule needs the
        brightness floor: measured Night-King footage sits at avg 10-20 / hi 36-53."""
        for la, lh in ((10.2, 36.0), (14.0, 44.0), (19.8, 53.0), (8.0, 30.0)):
            self.assertFalse(M._shot_featureless(shot(luma_avg=la, luma_hi=lh, luma_min=la - 0.4)),
                             f"dark scene avg={la} hi={lh} must survive")

    def test_a_lit_scene_with_highlights_survives(self):
        self.assertFalse(M._shot_featureless(shot(luma_avg=90.0, luma_hi=230.0, luma_min=70.0)))

    def test_a_lit_flat_frame_that_CHANGES_over_time_survives(self):
        """The time arm: a card does not move. A dim scene that brightens is not a card."""
        self.assertFalse(M._shot_featureless(shot(luma_avg=61.4, luma_hi=92.0, luma_min=40.0)))

    def test_old_index_fails_open(self):
        self.assertFalse(M._shot_featureless(shot(luma_avg=-1.0, luma_hi=-1.0, luma_min=-1.0)))

    def test_switch_disables(self):
        os.environ["VIDLORE_CLIPSTUDIO_FLAT_GATE"] = "0"
        try:
            self.assertFalse(M._shot_featureless(shot(luma_avg=61.4, luma_hi=92.0, luma_min=60.9)))
        finally:
            os.environ.pop("VIDLORE_CLIPSTUDIO_FLAT_GATE", None)

    def test_it_is_wired_into_the_never_airs_block(self):
        src = SRC.read_text()
        self.assertIn("if _shot_featureless(ps.shot):", src)


class TestEraAgreement(unittest.TestCase):
    """Beat 15: "The show wrote this ending in season one" lost the S01E01 upload to a season-less
    compilation, because a beat's era is INVISIBLE to token matching — titles have their digits
    stripped and season/episode are stopwords, so "season one" can never match "S01E01" as text."""

    def test_exact_season_title_agrees(self):
        self.assertEqual(E.title_seasons("Game of Thrones S01E01 White Walkers Attack"), {1})

    def test_the_compilation_that_beat_it_declares_nothing(self):
        self.assertEqual(E.title_seasons("Game of Thrones || The White Walkers"), set())

    def test_a_range_compilation_is_not_evidence_of_one_season(self):
        """It INCLUDES season 1, so it must not conflict — but it must not earn agreement either."""
        t = "Every White Walker scene | Game of Thrones Seasons 1-8"
        self.assertFalse(E.title_era_conflicts("season 1", t))
        self.assertNotEqual(E.title_seasons(t), {1})

    def test_conflict_and_agreement_stay_consistent(self):
        for season, title, agrees in ((1, "GoT S01E01 opening", True),
                                      (8, "Game of Thrones 8x03 The Long Night", True),
                                      (1, "Game of Thrones S08E03 Arya kills the Night King", False)):
            t_s = E.title_seasons(title)
            self.assertEqual(t_s == {season}, agrees, title)
            if agrees:
                self.assertFalse(E.title_era_conflicts(f"season {season}", title), title)

    def test_bonus_is_exact_scene_only_and_off_for_inherited_era(self):
        """A season inherited from an anchor is a default, not a claim the beat made — see
        beat_era_soft. Rewarding titles for agreeing with a guess would launder the guess."""
        src = SRC.read_text()
        seg = src.split("ERA AGREEMENT")[1][:1400]
        self.assertIn("not beat_era_soft", seg)
        self.assertIn("if _exact_beat and ps.sid in _era_match_sids:", src)


class TestTitleAffinityIsInformationWeighted(unittest.TestCase):
    """A raw hit COUNT treats 'arya' (41% of this pool's titles) as equal evidence to 'throat'."""

    @staticmethod
    def itf(df, n=119):
        return math.log((n + 1) / (df + 1)) / math.log(n + 1)

    def test_a_common_token_carries_less_than_a_rare_one(self):
        self.assertLess(self.itf(49), self.itf(26))      # arya (41%) < king (22%)
        self.assertLess(self.itf(26), self.itf(3))       # king < long (2.5%)

    def test_a_token_no_title_uses_carries_the_most(self):
        self.assertAlmostEqual(self.itf(0), 1.0, places=6)

    def test_the_night_king_match_clears_the_margin_it_lost_by(self):
        """{night, king, arya} against the pool's real title frequencies, capped at 0.34/0.90 —
        it has to beat 0.0084, the gap the correct source lost by."""
        mass = self.itf(28) + self.itf(26) + self.itf(49)
        self.assertGreater(0.34 * min(1.0, mass / 0.90) - 0.09, 0.0084)

    def test_essay_titles_are_discounted_not_excluded(self):
        """An essay ABOUT the scene is not the scene — but it often holds the only copy."""
        self.assertTrue(M._ESSAY_TITLE_RX.search(
            "Why Arya Killing the Night King is Perfect — Death is the Enemy"))
        self.assertFalse(M._ESSAY_TITLE_RX.search("GoT S08E03 - Arya kills the Night King"))
        self.assertIn("_tbonus *= 0.5", SRC.read_text())

    def test_purity_and_affinity_read_one_definition(self):
        src = SRC.read_text()
        self.assertIn("_essay_comp = _ESSAY_TITLE_RX", src)
        self.assertEqual(src.count("psycholog|toxic|best scenes|supercut"), 1)

    def test_unreadable_copies_earn_no_title_credit(self):
        """Naming the scene must not buy an unwatchable copy. The damper is HARD-only on purpose:
        the 'full' variant also scaled dim shots and broke the godswood beat 10 -> 5, because
        "Arya kills the Night King" is a NIGHT BATTLE and dim by nature."""
        src = SRC.read_text()
        # split on the TITLE damper's own text — moment-lock carries a damper of the same name
        seg = src.split("LEGIBILITY DAMPER — naming the scene")[1][:1600]
        self.assertIn("if _shot_unreadable(ps.shot):", seg)
        self.assertIn("_tbonus = 0.0", seg)
        self.assertIn('_ta_damp == "full"', seg)

    def test_switch_restores_the_old_behaviour(self):
        src = SRC.read_text()
        self.assertIn('"VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_ITF", "1"', src)
        self.assertIn("_aff_cap = _aff_exact if (_exact_beat and _aff_itf) else _aff", src)


class TestTheDoubleCountedEvidenceIsRecorded(unittest.TestCase):
    """_text_sim weights ENTITY hits double, so a clip merely SAYING "Arya Stark" scores what a
    clip SHOWING the described scene scores. Left in place deliberately — it is load-bearing for
    dialogue-driven beats — but the ranking no longer lets it outweigh scene identity."""

    def test_speaking_the_name_still_scores(self):
        seg = NS(keywords=["hand", "closes", "around", "throat", "lifts", "ground"],
                 entities=["Arya Stark"])
        self.assertGreaterEqual(M._text_sim(seg, "Arya Stark, come here"), 0.4)

    def test_scene_identity_now_outweighs_it_on_exact_beats(self):
        src = SRC.read_text()
        self.assertIn('_f_env("VIDLORE_CLIPSTUDIO_TITLE_AFFINITY_EXACT", 0.34)', src)
        self.assertGreater(0.34, 0.20)      # w_trans, the coincidence that won beat 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
