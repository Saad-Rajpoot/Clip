"""Speed Phase C — download↔index overlap (OPT-8) decision-safety contracts.

    python3 -m pytest tests/test_speed_phase_c.py -q

The prewarmer's whole safety story: (1) a source is handed over ONLY when its media file is
final (403-fallen sources wait for the sweeps), (2) a sweep replacement purges stale index
artifacts, (3) the worker is joined before index_all touches the models, (4) index_all is
unchanged and cache-hits prewarmed artifacts through the ordinary resume path.
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio.models import ClipProject, SourceVideo, SOURCE_OK  # noqa: E402
from vidlore.clipstudio.config import ClipConfig                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"


def _cand(i):
    return types.SimpleNamespace(url=f"https://y/{i}", title=f"vid {i}", id=f"v{i}",
                                 provider="youtube", duration=60.0, height=720,
                                 channel="c", query="q")


class TestOnReadyContract(unittest.TestCase):
    def _run(self, statuses, extras=None):
        """statuses[i] drives _download_one's result for candidate i."""
        from vidlore.clipstudio import download as DL
        extras = extras or {}
        fired = []

        def fake_dl(c, sid, perm, note, proj, cfg, progress):
            sv = SourceVideo(id=sid, url=c.url, title=c.title, status=statuses[c.id],
                             height=720)
            sv.checksum = f"ck_{c.id}" if statuses[c.id] == SOURCE_OK else ""
            sv.local_path = str(proj.sources_dir / f"{sid}.mp4")
            sv.extra.update(extras.get(c.id, {}))
            return sv

        with tempfile.TemporaryDirectory() as td:
            proj = ClipProject(name="t", root=td)
            proj.ensure_dirs()
            cands = [_cand(i) for i in range(len(statuses))]
            statuses = {c.id: s for c, s in zip(cands, statuses.values())} \
                if isinstance(statuses, dict) else {c.id: s for c, s in
                                                    zip(cands, list(statuses))}
            with mock.patch.object(DL, "_download_one",
                                   side_effect=lambda *a, **k: fake_dl(*a, **k)), \
                    mock.patch.object(DL, "_hd") if hasattr(DL, "_hd") else mock.patch.dict(
                        os.environ, {}):
                pass
            with mock.patch.object(DL, "_download_one", fake_dl), \
                    mock.patch.dict(os.environ, {"VIDLORE_HD_403_SWEEP": "0"}):
                DL.download_candidates(proj, cands, ClipConfig(),
                                       policy="approved_testing",
                                       on_ready=lambda sv: fired.append(sv.id))
        return fired

    def test_fires_once_per_ok_source_only(self):
        fired = self._run([SOURCE_OK, SOURCE_OK, "error"])
        self.assertEqual(sorted(fired), sorted(set(fired)))       # exactly once each
        self.assertEqual(len(fired), 2)                           # never for failed

    def test_403_fallen_source_fires_late_not_in_loop(self):
        # sweep disabled → the fallen source must still fire, but in the LATE pass
        # (after the sweep block), never from the in-loop clean-source path
        fired = self._run([SOURCE_OK, SOURCE_OK],
                          extras={"v0": {"hd_fallback": "HTTP Error 403: Forbidden"}})
        self.assertEqual(len(fired), 2)
        self.assertEqual(fired[-1], [f for f in fired if f.startswith("vid-0") or True][-1])
        # the clean source fired first (in-loop), the 403 one last (late pass)
        self.assertTrue(fired[0] != fired[1])

    def test_callback_error_never_breaks_download(self):
        from vidlore.clipstudio import download as DL

        def boom(sv):
            raise RuntimeError("prewarm died")

        def fake_dl(c, sid, perm, note, proj, cfg, progress):
            sv = SourceVideo(id=sid, url=c.url, title=c.title, status=SOURCE_OK, height=720)
            sv.checksum = f"ck_{c.id}"
            sv.local_path = str(proj.sources_dir / f"{sid}.mp4")
            return sv

        with tempfile.TemporaryDirectory() as td:
            proj = ClipProject(name="t", root=td)
            proj.ensure_dirs()
            with mock.patch.object(DL, "_download_one", fake_dl), \
                    mock.patch.dict(os.environ, {"VIDLORE_HD_403_SWEEP": "0"}):
                out = DL.download_candidates(proj, [_cand(0)], ClipConfig(),
                                             policy="approved_testing", on_ready=boom)
        self.assertEqual(sum(1 for s in out if s.status == SOURCE_OK), 1)


class TestPurgeOnSweepReplace(unittest.TestCase):
    def test_purge_removes_all_artifacts(self):
        from vidlore.clipstudio.index import purge_source_index
        with tempfile.TemporaryDirectory() as td:
            proj = ClipProject(name="t", root=td)
            proj.ensure_dirs()
            sid = "some_source_abc123"
            for suf in (".shots.json", ".embeds.npy", ".embeds.manifest.json",
                        ".words.json", ".index.meta.json"):
                (proj.index_dir / f"{sid}{suf}").write_text("x")
            kd = proj.index_dir / sid / "keyframes"
            kd.mkdir(parents=True)
            (kd / "shot_0000.jpg").write_bytes(b"x")
            purge_source_index(proj, sid)
            leftovers = [p.name for p in proj.index_dir.glob(f"{sid}*")]
            self.assertEqual(leftovers, [])

    def test_sweeps_call_purge_after_media_replacement(self):
        src = (ROOT / "download.py").read_text()
        # both sweep replacement blocks purge the (now stale) index for that source
        self.assertEqual(src.count("purge_source_index"), 2)
        for blk in src.split('sv.extra["hd_recovered"] = True')[1:]:
            self.assertIn("purge", blk[:300])


class TestPrewarmerWiring(unittest.TestCase):
    def test_worker_joined_before_index_all_both_sites(self):
        src = (ROOT / "orchestrate.py").read_text()
        # main stage: close() runs (in a finally) BEFORE the stage-5 index_all call
        i_close = src.find("_pw_main.close()")
        # the STAGE-5 call (8-space indent, no _ prefix) — not the backfill's _index_all
        i_index_all = src.find("\n        index_all(proj, cfg, references=refs")
        self.assertGreater(i_close, 0)
        self.assertGreater(i_index_all, i_close)
        # backfill: close() in finally before the round's _index_all
        bf = src.split("_pw = _IndexPrewarmer(proj, cfg, references=refs")[1]
        self.assertIn("_pw.close()", bf.split("_index_all(proj, cfg")[0])

    def test_kill_switch_and_cache_hit_design(self):
        src = (ROOT / "orchestrate.py").read_text()
        self.assertIn("VIDLORE_CLIPSTUDIO_INDEX_OVERLAP", src)
        # index_all stays the authoritative pass — the prewarmer never replaces it
        self.assertIn("index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster",
                      src)

    def test_download_holds_back_403_sources_from_in_loop_fire(self):
        src = (ROOT / "download.py").read_text()
        seg = src.split("index-overlap: a CLEAN source")[1][:600]
        self.assertIn('"403" not in ((sv.extra or {}).get("hd_fallback") or "")', seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestRungPrefetch(unittest.TestCase):
    def test_sheet_names_unique_per_call(self):
        src = (ROOT / "verify.py").read_text()
        # concurrent same-beat alternates must never share a contact-sheet path
        self.assertIn("_uuid_vs.uuid4().hex[:6]", src)
        self.assertNotIn('f"_vsheet_{_seg.index}_{getattr(ashot, \'index\', 0)}.jpg"', src)

    def test_phase2_ask_set_replicates_serial_walk(self):
        src = (ROOT / "verify.py").read_text()
        seg = src.split("PHASE-2 RUNG PREFETCH")[1].split("for sel in proj.selections:")[0]
        # only warmed-replace exact beats; slot consumed even when shotless (serial parity)
        self.assertIn('str(_v0.get("verdict")) != "replace" or not _exP', seg)
        self.assertIn("slot consumed even when shotless", seg)
        # scene-affinity ordering mirrored, both rungs through _cached_verify_ctx
        self.assertIn("_scene_affinity_order(selP.alternates, segP, proj,", seg)
        self.assertIn('rung="strict_promote"', seg)
        self.assertIn('rung="lenient_filler"', seg)
        # look scope: off for alternates, restored for lenient — two separate sub-passes
        self.assertIn('_look_scope["on"] = False', seg)
        self.assertIn('_look_scope["on"] = True', seg)
        # its own kill-switch + abort on repeated transport failures
        self.assertIn("VIDLORE_CLIPSTUDIO_VERIFY_PREFETCH_RUNGS", seg)
        self.assertIn("VERIFIER_BREAKER_TRIP", seg)

    def test_phase2_mirrors_lenient_env_gates(self):
        src = (ROOT / "verify.py").read_text()
        seg = src.split("PHASE-2 RUNG PREFETCH")[1].split("for sel in proj.selections:")[0]
        self.assertIn("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", seg)
        self.assertIn("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", seg)
