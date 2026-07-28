"""Self-healing footage recovery loop — fully offline (mocked vision, discovery, download).

    python3 tests/test_selfheal_loop.py
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio import selfheal as SH               # noqa: E402
from vidlore.clipstudio.models import (                     # noqa: E402
    ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo, SOURCE_OK)
from vidlore.clipstudio.config import ClipConfig            # noqa: E402


def _seg(i, text="Joffrey drinks at the wedding", policy="exact_scene", ent="Joffrey",
         expected="Joffrey raising the chalice", query="Purple Wedding Joffrey chalice"):
    return ScriptSegment(index=i, text=text, expected_visual=expected, scene_query=query,
                         visual_policy=policy, required_entity=ent, required_kind="character")


def _proj(td):
    p = ClipProject(name="t", root=str(td))
    p.ensure_dirs()
    p.meta = {"analysis": {"movie_title": "Game of Thrones", "characters": [], "actors": []}}
    return p


def _add_source_with_shots(p, sid, n=3, luma=60.0):
    (p.index_dir).mkdir(parents=True, exist_ok=True)
    kfdir = p.index_dir / sid / "keyframes"
    kfdir.mkdir(parents=True, exist_ok=True)
    shots = []
    for i in range(n):
        kf = kfdir / f"shot_{i:04d}.jpg"
        kf.write_bytes(b"\xff\xd8\xff\xdbfake")
        shots.append(Shot(source_id=sid, index=i, start=2.0 * i, end=2.0 * i + 2,
                          keyframe_path=str(kf), luma_avg=luma, subs_flag=0,
                          pair_diff_max=25.0, static_frac=0.0).to_dict())
    (p.index_dir / f"{sid}.shots.json").write_text(json.dumps(shots))
    src = SourceVideo(id=sid, url=f"https://youtube.com/watch?v={sid}", title=sid,
                      status=SOURCE_OK, height=1080, local_path="")
    p.sources.append(src)
    return src


class TestParsing(unittest.TestCase):
    def test_parse_blocked(self):
        msg = ("3 verifier-rejected beat(s) have no valid same-scene fallback anywhere in the "
               "timeline — scene(s) [10, 11, 116]. Rediscovery needed.")
        self.assertEqual(SH.parse_blocked(msg), [10, 11, 116])
        self.assertEqual(SH.parse_blocked("no markers here"), [])

    def test_blocked_indexes_from_audit(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            (p.output_dir).mkdir(parents=True, exist_ok=True)
            (p.output_dir / "rejected_footage_audit.json").write_text(json.dumps(
                {"unresolved_release_block": [{"seg_index": 5}, {"seg_index": 9}]}))
            self.assertEqual(SH.blocked_indexes(p), [5, 9])


class TestUnfillable(unittest.TestCase):
    def test_bts_beats_detected_and_softened(self):
        s = _seg(1, text="That is a choice.",
                 expected="Brief shot of showrunners David Benioff and D.B. Weiss from "
                          "behind-the-scenes footage")
        self.assertTrue(SH.beat_unfillable(s))
        SH._soften_to_abstract(s, lambda m: None)
        self.assertEqual(s.visual_policy, "abstract_effect")
        self.assertEqual(s.required_entity, "")

    def test_normal_scene_beat_not_unfillable(self):
        self.assertFalse(SH.beat_unfillable(_seg(2)))


class TestStillRecovery(unittest.TestCase):
    def test_installs_on_venue_keep_with_honest_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            _add_source_with_shots(p, "good_src")
            seg = _seg(3)
            sel = ClipSelection(segment_index=3, source_id="x", shot_index=0,
                               in_point=0, out_point=2, confidence=0.5)
            p.selections = [sel]
            with mock.patch.object(SH, "_venue_verify",
                                   return_value={"verdict": "keep", "reason": "right scene"}), \
                    mock.patch("vidlore.clipstudio.image_fallback._clip_relevance",
                               return_value=0.8):
                ok = SH.still_recover(p, seg, sel, eng_cfg=None, log=lambda m: None)
            self.assertTrue(ok)
            self.assertTrue(sel.image_path)
            self.assertEqual(sel.image_meta["relevance_class"], "contextual_fallback")
            self.assertTrue(sel.image_meta["still_verified"])

    def test_no_install_when_vision_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            _add_source_with_shots(p, "good_src")
            seg = _seg(4)
            sel = ClipSelection(segment_index=4, source_id="x", shot_index=0,
                               in_point=0, out_point=2, confidence=0.5)
            p.selections = [sel]
            with mock.patch.object(SH, "_venue_verify",
                                   return_value={"verdict": "replace", "reason": "wrong"}), \
                    mock.patch("vidlore.clipstudio.image_fallback._clip_relevance",
                               return_value=0.8):
                ok = SH.still_recover(p, seg, sel, eng_cfg=None, log=lambda m: None)
            self.assertFalse(ok)
            self.assertFalse(getattr(sel, "image_path", ""))

    def test_banned_and_static_sources_never_considered(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            _add_source_with_shots(p, "banned_src")
            p.meta["banned_sources"] = ["banned_src"]
            pool = SH._clean_pool(p)
            self.assertEqual(pool, [])


class TestQueries(unittest.TestCase):
    def test_llm_queries_fall_back_deterministically(self):
        seg = _seg(5)
        with mock.patch("vidlore.clipstudio.llm.complete_ex",
                        side_effect=RuntimeError("no api")):
            qs = SH._llm_queries(seg, "Game of Thrones")
        self.assertTrue(qs and all("Game of Thrones" in q for q in qs))

    def test_llm_queries_used_when_available(self):
        seg = _seg(6)
        with mock.patch("vidlore.clipstudio.llm.complete_ex",
                        return_value=('["GoT Purple Wedding chalice 4x02", "Joffrey death scene"]',
                                      {"served": "deepseek"})):
            qs = SH._llm_queries(seg, "Game of Thrones")
        self.assertEqual(qs[0], "GoT Purple Wedding chalice 4x02")


class TestAcquisition(unittest.TestCase):
    def test_bounded_and_policy_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            seg = _seg(7)
            cfg = ClipConfig()
            cands = [types.SimpleNamespace(url=f"https://y/{i}",
                                           title=f"Purple Wedding chalice scene {i}")
                     for i in range(6)]
            seen = {}

            def fake_download(proj, new, cfg2, *, policy, progress=None):
                seen["n"] = len(new)
                seen["policy"] = policy
                for c in new:
                    proj.sources.append(SourceVideo(id=f"s{c.url[-1]}", url=c.url, title=c.title,
                                                    status=SOURCE_OK, height=720,
                                                    local_path=str(Path(td) / "f.mp4")))

            with mock.patch("vidlore.clipstudio.discover.discover_sources",
                            return_value=cands), \
                    mock.patch("vidlore.clipstudio.download.download_candidates",
                               side_effect=fake_download), \
                    mock.patch("vidlore.clipstudio.index.index_source", return_value=None), \
                    mock.patch.object(SH, "_llm_queries",
                                      return_value=["Purple Wedding chalice scene"]):
                fresh = SH.acquire_for_beat(p, seg, cfg, policy="approved_testing",
                                            log=lambda m: None)
            self.assertEqual(seen["n"], 2, "SELFHEAL_MAX_SRC=2 must bound the fetch")
            self.assertEqual(seen["policy"], "approved_testing")
            self.assertEqual(len(fresh), 2)


class TestLoop(unittest.TestCase):
    def test_rounds_stop_on_clear_and_on_no_progress(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            p.segments = [_seg(1)]
            p.selections = [ClipSelection(segment_index=1, source_id="x", shot_index=0,
                                          in_point=0, out_point=2, confidence=0.5)]
            cfg = ClipConfig()
            gate = ["blocked — scene(s) [1]. x", None]     # blocked once, then clear

            def fake_gate(*a, **k):
                return gate.pop(0) if gate else None

            with mock.patch("vidlore.clipstudio.build.preassemble_release_block_reason",
                            side_effect=fake_gate), \
                    mock.patch.object(SH, "heal_blocked_beats", return_value=1) as healed:
                out = SH.run(p, p.segments, cfg, None, policy="approved_testing",
                             log=lambda m: None)
            self.assertIsNone(out)
            self.assertEqual(healed.call_count, 1)

            gate2 = iter(["blocked — scene(s) [1]. x"] * 10)
            with mock.patch("vidlore.clipstudio.build.preassemble_release_block_reason",
                            side_effect=lambda *a, **k: next(gate2)), \
                    mock.patch.object(SH, "heal_blocked_beats", return_value=0):
                out2 = SH.run(p, p.segments, cfg, None, policy="approved_testing",
                              log=lambda m: None)
            self.assertIsNotNone(out2, "gate verdict must survive a no-progress heal")

    def test_kill_switch(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            p.segments = [_seg(1)]
            cfg = ClipConfig()
            with mock.patch.dict(os.environ, {"VIDLORE_CLIPSTUDIO_SELFHEAL": "0"}), \
                    mock.patch("vidlore.clipstudio.build.preassemble_release_block_reason",
                               return_value="still blocked"), \
                    mock.patch.object(SH, "heal_blocked_beats") as healed:
                out = SH.run(p, p.segments, cfg, None, policy="x", log=lambda m: None)
            self.assertEqual(out, "still blocked")
            healed.assert_not_called()


class TestWiring(unittest.TestCase):
    def test_orchestrate_wires_pre_gate_and_build_retry(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "orchestrate.py").read_text()
        self.assertIn("_selfheal.run(proj, segs, cfg, analysis, policy=policy", src)
        self.assertIn('"NO valid fallback" in str(_be)', src)
        self.assertIn("VIDLORE_CLIPSTUDIO_SELFHEAL_BUILD_RETRY", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
