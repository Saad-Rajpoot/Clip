"""Root causes behind a measured 5.43/10 relevance render (job 5cab63d801, 291 scenes audited).

Two independent defects, each proven from that render's own artifacts:

A. ERA POISON. The script is an Arya arc essay whose climax is S08E03; it names seasons 1, 3, 7
   AND 8 out loud. `verified_episode_hint` read only the FIRST season mention (parse_season
   returns the first hit), matched it against the S01E01 anchor hint, and stamped the hint
   VERIFIED — handing hard era power to season 1 over a whole-series video. 79 breakout
   candidates were then pre-filtered as "later-era-source" and only 3 of 68 survived.

B. HD COLLAPSE -> BLIND GATES. Every HD download failed with YouTube "Error code: 152", a
   PO-token/SABR rejection dressed as unavailability. `_classify_dl_err` returned "other", so the
   recovery sweep (which keyed on the literal string "403") never fired: 110/110 sources fell back
   and the whole 22-minute video was built from 360p upscaled to a 1080p canvas. That is not only
   a quality problem — Face-ID named just 15% of shots, and the vision verifier and CLIP graphics
   gate lost the detail they need, which is why 27% of scenes showed the wrong character and a
   fan-art poster passed as `exact_scene`. Re-probed afterwards, 11 of 14 of those same sources
   offer >=720p, so the failure was transient and recoverable.

    python3 -m pytest tests/test_relevance_root_causes.py -q

No network, no LLM.
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import era as E                       # noqa: E402
from vidlore.clipstudio import hd_download as HD              # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"

# the real failure string from job 5cab63d801's download_audit
ERR152 = ("hd: no HD file (ERROR: [youtube] 0YM7oOTZvVM: This video is unavailable. "
          "Error code: 152 - 18 Watch video on YouTube) — fallback")


class TestEraSpanScriptCannotVerifyAHint(unittest.TestCase):
    ARC = ("Eight seasons of prophecy. The show wrote this ending in season one. "
           "Season three, episode six. The Brotherhood. Season seven, episode four. "
           "The courtyard at Winterfell. Season eight, episode three. Arya has just come.")

    def test_script_seasons_finds_all_of_them(self):
        self.assertEqual(E.script_seasons(self.ARC), {1, 3, 7, 8})

    def test_parse_season_still_returns_only_the_first(self):
        """The old corroboration used this — which is correct for a TITLE and wrong for a SCRIPT."""
        self.assertEqual(E.parse_season(self.ARC), 1)

    def test_multi_era_script_cannot_corroborate(self):
        hint, ok, why = E.verified_episode_hint(NS(episode_hint="S01E01"), script_text=self.ARC)
        self.assertEqual(hint, "S01E01")
        self.assertFalse(ok, "a whole-series script must never verify one episode")
        self.assertIn("spans seasons", why)

    def test_single_era_script_still_verifies(self):
        s = "This is the small council scene from season three, episode ten."
        _h, ok, why = E.verified_episode_hint(NS(episode_hint="S03E10"), script_text=s)
        self.assertTrue(ok, why)

    def test_contradiction_still_wins(self):
        s = "The whole video is about season three, episode ten."
        _h, ok, why = E.verified_episode_hint(NS(episode_hint="S04E01"), script_text=s)
        self.assertFalse(ok)
        self.assertIn("CONTRADICTED", why)

    def test_no_era_claim_stays_uncorroborated(self):
        _h, ok, why = E.verified_episode_hint(NS(episode_hint="S02E09"),
                                              script_text="Arya walks into the hall.")
        self.assertFalse(ok)
        self.assertIn("uncorroborated", why)

    def test_word_and_numeric_forms_both_counted(self):
        self.assertEqual(E.script_seasons("season one and Season 8 and S03E02 and 7x05"),
                         {1, 8, 3, 7})

    def test_absurd_numbers_are_ignored(self):
        self.assertEqual(E.script_seasons("season 99 of nothing"), set())


class TestPoTokenRejectionIsTransient(unittest.TestCase):
    def test_error_152_is_not_unavailability(self):
        self.assertEqual(HD._classify_dl_err(ERR152), "throttle_403")
        self.assertTrue(HD.is_po_token_failure(ERR152))

    def test_403_unchanged(self):
        t = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        self.assertEqual(HD._classify_dl_err(t), "throttle_403")
        self.assertTrue(HD.is_po_token_failure(t))

    def test_genuine_unavailability_still_permanent(self):
        for t in ("ERROR: Private video. Sign in if you've been granted access",
                  "This video has been removed by the uploader",
                  "Video unavailable",
                  "This video is not available in your country"):
            self.assertEqual(HD._classify_dl_err(t), "unavailable", t)
            self.assertFalse(HD.is_po_token_failure(t), t)

    def test_unrelated_errors_stay_other(self):
        self.assertEqual(HD._classify_dl_err("ERROR: unable to rename file"), "other")

    def test_sweep_keys_on_the_class_not_the_literal_403(self):
        src = (SRC / "download.py").read_text()
        self.assertIn("_hd.is_po_token_failure((s.extra or {}).get(\"hd_fallback\") or \"\")", src)
        self.assertIn("hd_po_token_fallbacks", src, "the audit must count this class separately")

    def test_machine_wide_collapse_widens_the_sweep(self):
        """A 24-source cap suits a few flaky videos; 110/110 failing has ONE shared cause."""
        src = (SRC / "download.py").read_text()
        seg = src.split("MACHINE-WIDE COLLAPSE")[1][:900]
        self.assertIn("0.8 * len(_yt_all)", seg)
        self.assertIn("VIDLORE_HD_403_SWEEP_MAX_WIDE", seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
