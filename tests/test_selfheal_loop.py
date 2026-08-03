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

from vidlore.clipstudio import orchestrate as O              # noqa: E402
from vidlore.clipstudio import selfheal as SH                # noqa: E402
from vidlore.clipstudio.models import (                     # noqa: E402
    ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo, SOURCE_FAILED, SOURCE_OK)
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

    def test_stronger_ranked_non_ok_source_index_is_not_a_ladder_candidate(self):
        """A stale index is not permission to air bytes from a currently blocked source."""
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            blocked = _add_source_with_shots(p, "blocked_src", n=1)
            blocked.status = "download_failed"
            _add_source_with_shots(p, "good_src", n=1)
            seg = _seg(41)
            sel = ClipSelection(segment_index=41, source_id="x", shot_index=0,
                                in_point=0, out_point=2, confidence=0.5)
            p.selections = [sel]

            def relevance(_shot, path, *_args, **_kwargs):
                return 0.99 if "blocked_src" in str(path) else 0.10

            with mock.patch.object(SH, "_venue_verify",
                                   return_value={"verdict": "keep", "reason": "candidate"}), \
                    mock.patch("vidlore.clipstudio.image_fallback._shot_relevance",
                               side_effect=relevance) as ranked:
                ok = SH.still_recover(p, seg, sel, eng_cfg=None, log=lambda _m: None)

            self.assertTrue(ok)
            self.assertIn("good_src", sel.image_path)
            self.assertTrue(ranked.call_args_list)
            self.assertTrue(all("blocked_src" not in str(call.args[1])
                                for call in ranked.call_args_list))


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
            faceid_obj = object()
            refs = {"Joffrey": object()}
            roster = ["Jack Gleeson"]

            def fake_download(proj, new, cfg2, *, policy, progress=None):
                seen["n"] = len(new)
                seen["policy"] = policy
                for c in new:
                    proj.sources.append(SourceVideo(id=f"s{c.url[-1]}", url=c.url, title=c.title,
                                                    status=SOURCE_OK, height=720,
                                                    local_path=str(Path(td) / "f.mp4")))

            def fake_index(_proj, _source, _cfg, **kwargs):
                seen.setdefault("index_kwargs", []).append(kwargs)
                return [object()]

            with mock.patch("vidlore.clipstudio.discover.discover_sources",
                            return_value=cands), \
                    mock.patch("vidlore.clipstudio.download.download_candidates",
                               side_effect=fake_download), \
                    mock.patch("vidlore.clipstudio.index.index_source",
                               side_effect=fake_index), \
                    mock.patch.object(SH, "_llm_queries",
                                      return_value=["Purple Wedding chalice scene"]):
                fresh = SH.acquire_for_beat(
                    p, seg, cfg, policy="approved_testing",
                    faceid_obj=faceid_obj, refs=refs, roster=roster,
                    log=lambda m: None)
            self.assertEqual(seen["n"], 2, "SELFHEAL_MAX_SRC=2 must bound the fetch")
            self.assertEqual(seen["policy"], "approved_testing")
            self.assertEqual(len(fresh), 2)
            self.assertEqual(len(seen["index_kwargs"]), 2)
            for kwargs in seen["index_kwargs"]:
                self.assertIs(kwargs["faceid"], faceid_obj)
                self.assertIs(kwargs["references"], refs)
                self.assertIs(kwargs["roster"], roster)

    def test_discovery_exception_is_inconclusive_not_an_empty_pool(self):
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(71), ClipConfig()
            with mock.patch.object(SH, "_yt_search_candidates", return_value=[]), \
                    mock.patch.object(SH, "_llm_queries", return_value=["exact scene"]), \
                    mock.patch("vidlore.clipstudio.discover.discover_sources",
                               side_effect=TimeoutError("search timed out")):
                with self.assertRaisesRegex(SH.InconclusiveAcquisitionError, "discovery"):
                    SH.acquire_for_beat(
                        p, seg, cfg, policy="approved_testing", log=lambda _m: None)

    def test_real_required_query_all_provider_technical_is_inconclusive(self):
        """Exercise discover's real status fan-out, not a mocked top-level exception."""
        from vidlore.clipstudio import discover as D
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(711), ClipConfig()
            provider = lambda _q, _n: ([], D.STATUS_TRANSPORT)
            with mock.patch.object(SH, "_yt_search_candidates", return_value=[]), \
                    mock.patch.object(SH, "_llm_queries", return_value=["required exact scene"]), \
                    mock.patch.object(D, "build_queries", return_value=[]), \
                    mock.patch.object(D, "anchor_queries", return_value=[]), \
                    mock.patch.object(D, "_ytsearch_ex", side_effect=provider), \
                    mock.patch.object(D, "_archive_search_ex", side_effect=provider), \
                    mock.patch("time.sleep", return_value=None):
                with self.assertRaisesRegex(
                        SH.InconclusiveAcquisitionError, "TargetedDiscoveryTechnicalError"):
                    SH.acquire_for_beat(
                        p, seg, cfg, policy="approved_testing", log=lambda _m: None)

    def test_all_failed_downloads_are_inconclusive_not_policy_exhaustion(self):
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(72), ClipConfig()
            cand = types.SimpleNamespace(
                url="https://y/fail", title="Purple Wedding exact scene")

            def failed_download(proj, new, _cfg, **_kwargs):
                for item in new:
                    proj.sources.append(SourceVideo(
                        id="failed", url=item.url, title=item.title,
                        status=SOURCE_FAILED, error="HTTP 403"))

            with mock.patch.object(SH, "_yt_search_candidates", return_value=[]), \
                    mock.patch.object(SH, "_llm_queries", return_value=["exact scene"]), \
                    mock.patch("vidlore.clipstudio.discover.discover_sources",
                               return_value=[cand]), \
                    mock.patch("vidlore.clipstudio.download.download_candidates",
                               side_effect=failed_download):
                with self.assertRaisesRegex(SH.InconclusiveAcquisitionError, "download"):
                    SH.acquire_for_beat(
                        p, seg, cfg, policy="approved_testing", log=lambda _m: None)
            self.assertEqual(p.sources, [], "failed source rows must roll back for Resume")

    def test_mixed_success_and_failed_download_rolls_back_and_retries_same_urls(self):
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(721), ClipConfig()
            cands = [types.SimpleNamespace(url=f"https://y/{i}", title=f"exact scene {i}")
                     for i in range(2)]
            attempts = []

            def download(proj, new, _cfg, **_kwargs):
                attempts.append([c.url for c in new])
                first = len(attempts) == 1
                for i, item in enumerate(new):
                    proj.sources.append(SourceVideo(
                        id=f"s{i}", url=item.url, title=item.title,
                        status=(SOURCE_FAILED if first and i == 1 else SOURCE_OK),
                        error=("network" if first and i == 1 else ""),
                        local_path=("" if first and i == 1 else str(Path(td) / f"{i}.mp4"))))

            common = [
                mock.patch.object(SH, "_yt_search_candidates", return_value=[]),
                mock.patch.object(SH, "_llm_queries", return_value=["exact scene"]),
                mock.patch("vidlore.clipstudio.discover.discover_sources", return_value=cands),
                mock.patch("vidlore.clipstudio.download.download_candidates", side_effect=download),
                mock.patch("vidlore.clipstudio.index.index_source", return_value=[object()]),
            ]
            with common[0], common[1], common[2], common[3], common[4]:
                with self.assertRaises(SH.InconclusiveAcquisitionError):
                    SH.acquire_for_beat(
                        p, seg, cfg, policy="approved_testing", log=lambda _m: None)
                self.assertEqual(p.sources, [])
                fresh = SH.acquire_for_beat(
                    p, seg, cfg, policy="approved_testing", log=lambda _m: None)
            self.assertEqual(len(fresh), 2)
            self.assertEqual(attempts[0], attempts[1], "rollback must not suppress failed URLs")

    def test_duplicate_only_download_is_conclusive(self):
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(722), ClipConfig()
            cand = types.SimpleNamespace(url="https://y/dup", title="exact scene duplicate")

            def duplicate_download(proj, new, _cfg, **_kwargs):
                proj.sources.append(SourceVideo(
                    id="dup", url=new[0].url, title=new[0].title, status="duplicate"))

            with mock.patch.object(SH, "_yt_search_candidates", return_value=[]), \
                    mock.patch.object(SH, "_llm_queries", return_value=["exact scene"]), \
                    mock.patch("vidlore.clipstudio.discover.discover_sources",
                               return_value=[cand]), \
                    mock.patch("vidlore.clipstudio.download.download_candidates",
                               side_effect=duplicate_download):
                self.assertEqual(SH.acquire_for_beat(
                    p, seg, cfg, policy="approved_testing", log=lambda _m: None), [])

    def test_index_exception_is_inconclusive(self):
        self._assert_index_failure_is_inconclusive(RuntimeError("decoder unavailable"), "RuntimeError")

    def test_zero_shot_index_is_inconclusive(self):
        self._assert_index_failure_is_inconclusive([], "zero_shots")

    def _assert_index_failure_is_inconclusive(self, index_outcome, detail):
        with tempfile.TemporaryDirectory() as td:
            p, seg, cfg = _proj(td), _seg(73), ClipConfig()
            cand = types.SimpleNamespace(
                url="https://y/index", title="Purple Wedding exact scene")

            def successful_download(proj, new, _cfg, **_kwargs):
                for item in new:
                    proj.sources.append(SourceVideo(
                        id="downloaded", url=item.url, title=item.title,
                        status=SOURCE_OK, local_path=str(Path(td) / "downloaded.mp4")))

            def index_failure(proj, source, *_args, **_kwargs):
                (proj.index_dir / f"{source.id}.shots.json").write_text("partial")
                if isinstance(index_outcome, BaseException):
                    raise index_outcome
                return index_outcome

            patch_index = (mock.patch("vidlore.clipstudio.index.index_source",
                                      side_effect=index_failure)
                           if isinstance(index_outcome, BaseException)
                           else mock.patch("vidlore.clipstudio.index.index_source",
                                           side_effect=index_failure))
            with mock.patch.object(SH, "_yt_search_candidates", return_value=[]), \
                    mock.patch.object(SH, "_llm_queries", return_value=["exact scene"]), \
                    mock.patch("vidlore.clipstudio.discover.discover_sources",
                               return_value=[cand]), \
                    mock.patch("vidlore.clipstudio.download.download_candidates",
                               side_effect=successful_download), patch_index:
                with self.assertRaisesRegex(
                        SH.InconclusiveAcquisitionError, f"index.*{detail}"):
                    SH.acquire_for_beat(
                        p, seg, cfg, policy="approved_testing", log=lambda _m: None)
            self.assertEqual(p.sources, [])
            self.assertFalse((p.index_dir / "downloaded.shots.json").exists(),
                             "partial index must be purged on retryable abort")


class TestLoop(unittest.TestCase):
    def test_rounds_stop_on_clear_and_on_no_progress(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            p.segments = [_seg(1)]
            p.selections = [ClipSelection(segment_index=1, source_id="x", shot_index=0,
                                          in_point=0, out_point=2, confidence=0.5)]
            cfg = ClipConfig()
            faceid_obj = object()
            refs = {"Joffrey": object()}
            roster = ["Jack Gleeson"]
            gate = ["blocked — scene(s) [1]. x", None]     # blocked once, then clear

            def fake_gate(*a, **k):
                return gate.pop(0) if gate else None

            with mock.patch("vidlore.clipstudio.build.preassemble_release_block_reason",
                            side_effect=fake_gate), \
                    mock.patch.object(SH, "heal_blocked_beats", return_value=1) as healed:
                out = SH.run(
                    p, p.segments, cfg, None, policy="approved_testing",
                    faceid_obj=faceid_obj, refs=refs, roster=roster,
                    log=lambda m: None)
            self.assertIsNone(out)
            self.assertEqual(healed.call_count, 1)
            heal_kwargs = healed.call_args.kwargs
            self.assertIs(heal_kwargs["faceid_obj"], faceid_obj)
            self.assertIs(heal_kwargs["refs"], refs)
            self.assertIs(heal_kwargs["roster"], roster)

            gate2 = iter(["blocked — scene(s) [1]. x"] * 10)
            with mock.patch("vidlore.clipstudio.build.preassemble_release_block_reason",
                            side_effect=lambda *a, **k: next(gate2)), \
                    mock.patch.object(SH, "heal_blocked_beats", return_value=0):
                out2 = SH.run(p, p.segments, cfg, None, policy="approved_testing",
                              log=lambda m: None)
            self.assertIsNotNone(out2, "gate verdict must survive a no-progress heal")

    def test_heal_threads_main_pool_capabilities_to_targeted_acquisition(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            seg = _seg(12)
            p.segments = [seg]
            p.selections = [ClipSelection(
                segment_index=12, source_id="x", shot_index=0,
                in_point=0, out_point=2, confidence=0.5)]
            faceid_obj = object()
            refs = {"Joffrey": object()}
            roster = ["Jack Gleeson"]
            with mock.patch("vidlore.clipstudio.config.engine_config", return_value=None), \
                    mock.patch.object(SH, "_clean_pool", return_value=[]), \
                    mock.patch.object(SH, "still_recover", return_value=False), \
                    mock.patch.object(SH, "beat_unfillable", return_value=False), \
                    mock.patch.object(SH, "_venue_cache_save"), \
                    mock.patch.object(SH, "acquire_for_beat", return_value=[]) as acquired:
                SH.heal_blocked_beats(
                    p, [seg], ClipConfig(), blocked=[12], policy="approved_testing",
                    faceid_obj=faceid_obj, refs=refs, roster=roster,
                    log=lambda _m: None)

            acquire_kwargs = acquired.call_args.kwargs
            self.assertIs(acquire_kwargs["faceid_obj"], faceid_obj)
            self.assertIs(acquire_kwargs["refs"], refs)
            self.assertIs(acquire_kwargs["roster"], roster)

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


class TestRegionFrameTextSafety(unittest.TestCase):
    """Live-found gap: region frames bypass the persisted flags, so a frame with giant burned
    meme-text (Italian caption + cartoon watermark) and one with a French subtitle both
    passed the vision venue bar — text safety must be deterministic and pre-vision."""

    def test_dirty_frames_never_reach_vision(self):
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            seg = _seg(9)
            sel = ClipSelection(segment_index=9, source_id="x", shot_index=0,
                                in_point=0, out_point=2, confidence=0.5)
            src = types.SimpleNamespace(id="s1", local_path=str(Path(td) / "v.mp4"),
                                        duration=100.0)
            Path(src.local_path).write_bytes(b"fake")
            frames = [str(Path(td) / f"f{i}.jpg") for i in range(4)]
            with mock.patch("vidlore.clipstudio.selfheal.subprocess.run"), \
                    mock.patch.object(SH, "_frame_text_dirty", return_value=True) as dirty, \
                    mock.patch.object(SH, "_venue_verify") as venue, \
                    mock.patch("vidlore.clipstudio.selfheal.Path.exists", return_value=True), \
                    mock.patch("PIL.Image.open"), \
                    mock.patch("numpy.asarray", return_value=__import__("numpy").zeros((4, 4))):
                ok = SH._region_frames_recover(p, seg, sel, src, None, log=lambda m: None)
            self.assertFalse(ok)
            venue.assert_not_called()

    def test_unfetchable_frame_counts_dirty(self):
        self.assertTrue(SH._frame_text_dirty("/nonexistent/frame.jpg"),
                        "an uncheckable frame must never air")


class TestWiring(unittest.TestCase):
    def test_orchestrate_wires_pre_gate_and_build_retry(self):
        src = (Path(__file__).resolve().parents[1]
               / "vidlore" / "clipstudio" / "orchestrate.py").read_text()
        self.assertIn("return _selfheal.run(", src)
        self.assertIn("proj, segs, cfg, analysis, policy=policy", src)
        self.assertIn("faceid_obj=faceid_obj, refs=refs, roster=roster", src)
        # the catch routes on the TYPED exception kind — the old message-substring match
        # ("NO valid fallback") never matched the real gate message and was dead code
        self.assertIn('getattr(_be, "kind", "") == "rejected_footage"', src)
        self.assertNotIn('"NO valid fallback" in str(_be)', src)
        self.assertIn("VIDLORE_CLIPSTUDIO_SELFHEAL_BUILD_RETRY", src)

    def test_build_retry_catch_fires_on_real_gate_exception(self):
        """The REAL gate exception (as raised by build.py) must route into heal_blocked_beats —
        the exact contract the dead string-match silently broke."""
        from vidlore.clipstudio.verify import NonRetryableBuildError
        e = NonRetryableBuildError(
            "rejected-footage gate: 2 beat(s) unresolved (no valid editorial hold or "
            "contextual fallback) — rediscovery needed for scene(s) [71, 73]. This is a "
            "CONTENT failure: re-running the same render will not fix it.",
            kind="rejected_footage")
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(getattr(e, "kind", ""), "rejected_footage")
        # build.py raises with the kind set
        bsrc = (Path(__file__).resolve().parents[1]
                / "vidlore" / "clipstudio" / "build.py").read_text()
        self.assertIn('kind="rejected_footage"', bsrc)

    def test_preassemble_acquisition_failure_is_retryable_and_uncheckpoints_recovery(self):
        """The 101-beat failure: beat 80 download uncertainty must never become content."""
        from vidlore.clipstudio.verify import NonRetryableBuildError
        with tempfile.TemporaryDirectory() as td:
            p = _proj(td)
            O._stage_done(p, "recover", "recover-sig")
            logs = []
            technical = SH.InconclusiveAcquisitionError(
                80, "download", detail="all attempted sources failed")
            faceid_obj = object()
            refs = {"Petyr Baelish": object()}
            roster = ["Aidan Gillen"]
            with mock.patch.object(SH, "run", side_effect=technical) as runner:
                with self.assertRaises(SH.InconclusiveAcquisitionError) as raised:
                    O._run_preassemble_selfheal(
                        p, [], ClipConfig(), None, policy="approved_testing",
                        pre="blocked — scene(s) [80].", faceid_obj=faceid_obj,
                        refs=refs, roster=roster, log=logs.append)

            self.assertIs(raised.exception, technical)
            self.assertNotIsInstance(raised.exception, NonRetryableBuildError)
            persisted = ClipProject.load(td)
            self.assertIsNone(O._ckpt(persisted)["stages"].get("recover"),
                              "Resume must rerun recovery after technical acquisition failure")
            self.assertNotIn("selection_relevance_recovery", persisted.meta)
            self.assertNotIn("selection_relevance_gap_softening", persisted.meta)
            self.assertFalse(any("self-heal: skipped" in line for line in logs),
                             "technical acquisition must propagate, never be logged as a skip")
            run_kwargs = runner.call_args.kwargs
            self.assertIs(run_kwargs["faceid_obj"], faceid_obj)
            self.assertIs(run_kwargs["refs"], refs)
            self.assertIs(run_kwargs["roster"], roster)


if __name__ == "__main__":
    unittest.main(verbosity=2)
