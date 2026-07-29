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
