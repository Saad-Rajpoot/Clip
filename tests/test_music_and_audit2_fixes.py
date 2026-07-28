"""Music-silent root cause + audit-2 leak fixes.

    python3 tests/test_music_and_audit2_fixes.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio.discover import (                   # noqa: E402
    is_unwanted_source_title as unw, source_has_bonus_tail)
from vidlore.clipstudio.match import _shot_dirty_reason     # noqa: E402
from vidlore.clipstudio.models import Shot                  # noqa: E402
from vidlore.clipstudio import build as B                   # noqa: E402


class TestMusicResolution(unittest.TestCase):
    def test_arc_cues_match_compose_score_contract(self):
        """The arc cues must be start/end SEGMENTS — the {'t': ...} points raised
        KeyError('end') on every render and the silent except hid it."""
        captured = {}

        def fake_compose(cues, total, dest, **kw):
            captured["cues"] = cues
            for c in cues:
                assert "start" in c and "end" in c and "category" in c
            return None                                  # let it fall through to fallback

        with tempfile.TemporaryDirectory() as td, \
                mock.patch("vidlore.musiclib.compose_score", side_effect=fake_compose), \
                mock.patch("vidlore.musiclib.library_root",
                           return_value=Path(td)):
            B._resolve_music(None, "history", 600.0, Path(td), log=lambda m: None)
        cues = captured["cues"]
        self.assertGreaterEqual(len(cues), 2)
        self.assertEqual(cues[0]["start"], 0.0)
        self.assertAlmostEqual(cues[-1]["end"], 600.0, places=1)
        for a, b in zip(cues, cues[1:]):
            self.assertAlmostEqual(a["end"], b["start"], places=3)

    def test_compose_failure_is_logged_not_swallowed(self):
        logs = []
        with tempfile.TemporaryDirectory() as td, \
                mock.patch("vidlore.musiclib.compose_score",
                           side_effect=KeyError("end")), \
                mock.patch("vidlore.musiclib.library_root", return_value=Path(td)):
            B._resolve_music(None, "history", 100.0, Path(td), log=logs.append)
        self.assertTrue(any("compose_score FAILED" in m for m in logs))

    def test_fallback_uses_env_aware_library_root(self):
        with tempfile.TemporaryDirectory() as td:
            lib = Path(td) / "lib" / "historical_epic"
            lib.mkdir(parents=True)
            (lib / "track.mp3").write_bytes(b"x")
            with mock.patch("vidlore.musiclib.compose_score", return_value=None), \
                    mock.patch("vidlore.musiclib.library_root",
                               return_value=Path(td) / "lib"):
                p = B._resolve_music(None, "history", 100.0, Path(td), log=lambda m: None)
            self.assertTrue(p and p.endswith("track.mp3"))

    def test_build_hard_fails_on_no_music(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "build.py").read_text()
        self.assertIn("refusing a \"\n            \"silent-music render", src.replace(
            "refusing a \"\n            \"silent-music render",
            "refusing a \"\n            \"silent-music render"))
        self.assertIn("VIDLORE_CLIPSTUDIO_ALLOW_NO_MUSIC", src)
        self.assertIn("silent-music render", src)


class TestMerchTitles(unittest.TestCase):
    def test_product_listings_rejected(self):
        for t in ("Aladean Vintage Chalice Medieval Goblets Celebrate Royalty King Queen "
                  "Wine Cups Handmade Style",
                  "Medieval goblet replica for sale — Etsy handmade",
                  "GoT Longclaw sword unboxing and review"):
            self.assertTrue(unw(t), t)

    def test_scene_titles_still_pass(self):
        for t in ("Game of Thrones - King Joffrey's Death (Poisoned at his wedding)",
                  "Olenna and Margaery garden scene 4x04",
                  "Tyrion's Speech - Laws of Gods and Men - S4Ep6"):
            self.assertFalse(unw(t), t)


class TestBonusTail(unittest.TestCase):
    def test_bonus_titles_detected(self):
        self.assertTrue(source_has_bonus_tail(
            "Game of Thrones - King Joffrey's Death + BONUS Scene [The Purple Wedding]"))
        self.assertFalse(source_has_bonus_tail("Game of Thrones 4x02 Joffrey death scene"))

    def test_tail_stamp_makes_shot_dirty(self):
        sh = Shot(source_id="s", index=9, start=200.0, end=205.0,
                  scores={"bonus_tail": 1}, subs_flag=0)
        self.assertEqual(_shot_dirty_reason(sh), "bonus-tail")
        clean = Shot(source_id="s", index=1, start=10.0, end=15.0, subs_flag=0)
        self.assertEqual(_shot_dirty_reason(clean), "")

    def test_pool_load_stamps_trailing_shots(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "match.py").read_text()
        self.assertIn("source_has_bonus_tail", src)
        self.assertIn('sh.scores["bonus_tail"] = 1', src)


class TestMinerFloor(unittest.TestCase):
    def test_zero_overlap_fallback_removed(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "build.py").read_text()
        self.assertNotIn("_mine_tier([s for s in srcs if s.id in _tier1], 0)", src,
                         "the zero-overlap last-resort tier must be gone")
        self.assertIn("VIDLORE_CLIPSTUDIO_BREAKOUT_MINE_MIN_OV", src)


class TestSelfhealSearch(unittest.TestCase):
    def test_ytsearch_filters_junk_and_durations(self):
        import vidlore.clipstudio.selfheal as SH
        fake_out = ("abc123\t120\tOlenna confession scene 4x04\n"
                    "def456\t20\tShort clip\n"
                    "ghi789\t3000\tFull episode reaction\n"
                    "jkl012\t180\tGoT chalice replica unboxing review\n")
        with mock.patch.object(SH.subprocess, "run",
                               return_value=mock.Mock(stdout=fake_out)), \
                mock.patch("vidlore.clipstudio.hd_download.HD_PY",
                           sys.executable):
            with mock.patch.object(SH.Path, "exists", return_value=True):
                out = SH._yt_search_candidates(["olenna confession"], log=lambda m: None)
        titles = [c.title for c in out]
        self.assertIn("Olenna confession scene 4x04", titles)
        self.assertNotIn("Short clip", titles)                  # too short
        self.assertNotIn("Full episode reaction", titles)       # too long + reaction
        self.assertNotIn("GoT chalice replica unboxing review", titles)  # merch gate

    def test_wired_ahead_of_discovery(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "selfheal.py").read_text()
        self.assertIn("cands = _yt_search_candidates(queries", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
