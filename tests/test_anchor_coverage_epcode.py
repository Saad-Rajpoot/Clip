"""Anchor-coverage gate (single_scene) + narrow wrong-episode soft penalty.

    python3 tests/test_anchor_coverage_epcode.py
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio import orchestrate as ORC           # noqa: E402
from vidlore.clipstudio.models import ClipProject, SourceVideo, SOURCE_OK  # noqa: E402
from vidlore.clipstudio.config import ClipConfig            # noqa: E402

FAILS = []
ANA = {"video_type": "single_scene", "movie_title": "Game of Thrones",
       "anchor_scenes": [{"name": "Jaime leaves Brienne in the night courtyard",
                          "query": "Jaime leaves Brienne crying courtyard night Winterfell",
                          "episode": "S08E04"}]}


def _proj(td, titles):
    p = ClipProject(name="t", root=str(td))
    p.ensure_dirs()
    for i, t in enumerate(titles):
        p.sources.append(SourceVideo(id=f"s{i}", url=f"https://y/{i}", title=t,
                                     status=SOURCE_OK, height=720))
    return p


class TestAnchorCoverage(unittest.TestCase):
    def test_covered_pool_makes_no_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td, ["Jaime leaves Brienne crying scene S08E04",
                           "Jaime says goodbye to Brienne night courtyard 8x04",
                           "Random Cersei scene"])
            with mock.patch("vidlore.clipstudio.selfheal._yt_search_candidates") as yts:
                n = ORC._ensure_anchor_coverage(p, ANA, ClipConfig(),
                                                policy="approved_testing", log=lambda m: None)
            self.assertEqual(n, 0)
            yts.assert_not_called()

    def test_uncovered_pool_triggers_direct_search(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td, ["Jaime and Brienne best moments compilation",
                           "Game of Thrones 8x02 knighting scene"])
            cands = [types.SimpleNamespace(url="https://y/new1",
                                           title="Jaime leaves Brienne crying S08E04 scene",
                                           id="new1", provider="youtube", query="q")]
            with mock.patch("vidlore.clipstudio.selfheal._yt_search_candidates",
                            return_value=cands) as yts, \
                    mock.patch("vidlore.clipstudio.download.download_candidates") as dl, \
                    mock.patch("vidlore.clipstudio.index.index_source", return_value=None):
                ORC._ensure_anchor_coverage(p, ANA, ClipConfig(),
                                            policy="approved_testing", log=lambda m: None)
            yts.assert_called_once()
            dl.assert_called_once()
            self.assertEqual(dl.call_args.kwargs.get("policy"), "approved_testing")

    def test_multi_scene_and_kill_switch_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td, [])
            multi = dict(ANA, video_type="multi_scene")
            self.assertEqual(ORC._ensure_anchor_coverage(p, multi, ClipConfig(),
                                                         policy="x", log=lambda m: None), 0)
            with mock.patch.dict(os.environ, {"VIDLORE_CLIPSTUDIO_ANCHOR_COVERAGE": "0"}):
                self.assertEqual(ORC._ensure_anchor_coverage(p, ANA, ClipConfig(),
                                                             policy="x", log=lambda m: None), 0)


class TestEpcodePenalty(unittest.TestCase):
    def test_narrow_scope_wiring(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "match.py").read_text()
        self.assertIn("anchor_ep is not None and _policy.policy_of(seg) == _policy.EXACT", src,
                      "penalty must be EXACT-beats-only")
        self.assertIn("VIDLORE_CLIPSTUDIO_EPCODE_PENALTY", src)
        self.assertIn('_t_ep is not None and tuple(_t_ep) != tuple(anchor_ep)', src,
                      "codeless titles must never be penalized")
        self.assertIn("if single_scene:", src.split("_anchor_ep_m = None")[1][:400],
                      "anchor code only computed for single_scene renders")

    def test_parse_variants(self):
        from vidlore.clipstudio.era import parse_episode
        self.assertEqual(parse_episode("GoT 8x02 knighting"), (8, 2))
        self.assertEqual(parse_episode("S08E04 Jaime leaves"), (8, 4))
        self.assertIsNone(parse_episode("Jaime Brienne moments"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAnchorEraSoftAffinity(unittest.TestCase):
    def test_wiring_and_semantics(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "match.py").read_text()
        # era-silent beats inherit the anchor season at HALF penalty, single_scene only
        self.assertIn('_beat_era_m = f"season {_a_season}"', src)
        self.assertIn("_era_soft_m = True", src)
        self.assertIn("if not _beat_era_m:", src)
        self.assertIn("_era_pen * (0.5 if beat_era_soft else 1.0)", src)
        # beats with their OWN era keep the full penalty path (soft only when era was empty)
        seg = src.split("ANCHOR-ERA SOFT AFFINITY")[1][:900]
        self.assertIn("if not _beat_era_m:", seg)

    def test_title_era_conflict_respects_undeclared(self):
        from vidlore.clipstudio import era as E
        # undeclared-title sources are NEVER era-penalized (the not-strict doctrine)
        self.assertFalse(E.title_era_conflicts("season 8", "Jaime and Brienne best moments"))
        self.assertTrue(E.title_era_conflicts("season 8", "Game of Thrones Season 1 Winterfell scenes"))
        self.assertFalse(E.title_era_conflicts("season 8", "GoT Seasons 1-8 compilation"))
