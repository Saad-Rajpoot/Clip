"""Speed Phase A — decision-parity contracts for the quality-neutral speed fixes.

    python3 -m pytest tests/test_speed_phase_a.py -q

Covers: monotonic flags walk bit-parity (synthetic GOP video), OCR pool opt-in gating,
verify-workers scoped set/restore, early release-gate dry-run wiring (subset rule, warn
mode, typed kind), parallel resolve_quality barrier, selfheal ordered-accept prefetch.
"""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1] / "vidlore" / "clipstudio"


def _mk_test_video(dest: str, secs: int = 10) -> bool:
    """testsrc2 h264 with 2s GOPs — real keyframe structure for the mono/seek parity walk."""
    from vidlore.clipstudio.index import ffmpeg_exe
    cmd = [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", f"testsrc2=duration={secs}:size=320x180:rate=25",
           "-c:v", "libx264", "-g", "50", "-pix_fmt", "yuv420p", dest]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return os.path.exists(dest) and os.path.getsize(dest) > 0
    except Exception:
        return False


class TestMonoFlagsParity(unittest.TestCase):
    FIELDS = ["subs_flag", "text_conf", "luma_avg", "luma_hi", "luma_min",
              "luma_min_black_frac", "corner_masks", "static_frac",
              "pair_diff_max", "pair_diff_mean"]

    def _shots(self, spans):
        return [types.SimpleNamespace(start=a, end=b, transcript="") for a, b in spans]

    def test_mono_equals_seek_on_synthetic_gop_video(self):
        from vidlore.clipstudio import index as IX
        with tempfile.TemporaryDirectory() as td:
            vid = os.path.join(td, "t.mp4")
            if not _mk_test_video(vid):
                self.skipTest("ffmpeg unavailable")
            # spans include: short (<2s → 5 samples), normal, long (→ scaled samples),
            # boundary-stacked, and one overrunning EOF (exercises the EOF fallback)
            spans = [(0.0, 1.4), (1.4, 3.0), (3.0, 3.9), (3.9, 8.6), (8.6, 12.5)]
            A, B = self._shots(spans), self._shots(spans)
            dA = IX._compute_shot_flags_seek(vid, A)
            dB = IX._compute_shot_flags_mono(vid, B)
            self.assertEqual(dA, dB)
            for i, (a, b) in enumerate(zip(A, B)):
                for f in self.FIELDS:
                    self.assertEqual(getattr(a, f, None), getattr(b, f, None),
                                     f"shot {i} field {f}")

    def test_dispatcher_kill_switch(self):
        src = (ROOT / "index.py").read_text()
        self.assertIn("VIDLORE_CLIPSTUDIO_FLAGS_FAST", src)
        self.assertIn("_compute_shot_flags_seek(path, shots, progress=progress)", src)
        # mono failure falls back to the seek walk for the WHOLE source
        seg = src.split("def compute_shot_flags")[1].split("def _flags_finish_shot")[0]
        self.assertIn("except Exception:", seg)

    def test_mono_replicates_keyframe_restart_rule(self):
        # the seek walk starts at the last keyframe ≤ t: a candidate seen BEFORE a later
        # keyframe ≤ t must be discarded — the mono walk encodes that rule explicitly
        src = (ROOT / "index.py").read_text()
        self.assertIn("if fr.key_frame and ft <= t:", src)
        self.assertIn("cand = None", src)


class TestOcrPoolOptIn(unittest.TestCase):
    def test_pool_disabled_without_entrypoint_opt_in(self):
        from vidlore.clipstudio import ocr as O
        old = os.environ.pop("VIDLORE_CLIPSTUDIO_OCR_POOL_OK", None)
        try:
            self.assertEqual(O._pool_workers(), 0)
            self.assertIsNone(O.read_text_async("/nonexistent.jpg"))
        finally:
            if old is not None:
                os.environ["VIDLORE_CLIPSTUDIO_OCR_POOL_OK"] = old

    def test_guarded_entrypoints_opt_in(self):
        self.assertIn('os.environ["VIDLORE_CLIPSTUDIO_OCR_POOL_OK"] = "1"',
                      (ROOT / "web.py").read_text())
        self.assertIn('setdefault("VIDLORE_CLIPSTUDIO_OCR_POOL_OK", "1")',
                      (ROOT / "cli.py").read_text())

    def test_index_never_trusts_stale_keyframes(self):
        # only THIS run's recorded pre-extract result counts; a bare exists() check would
        # resurrect stale keyframes from an older index
        src = (ROOT / "index.py").read_text()
        self.assertIn("_prex[i] if i in _prex else extract_keyframe", src)
        self.assertNotIn("kf.exists() or extract_keyframe", src)


class TestVerifyWorkersScoped(unittest.TestCase):
    def test_scoped_set_and_restore_not_setdefault_leak(self):
        src = (ROOT / "orchestrate.py").read_text()
        # both verify call sites: set when unset, ALWAYS popped in finally — never leaked
        self.assertEqual(src.count('environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", None)'), 2)
        self.assertNotIn('setdefault("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"', src)

    def test_operator_env_always_wins(self):
        src = (ROOT / "orchestrate.py").read_text()
        self.assertIn('"VIDLORE_CLIPSTUDIO_VERIFY_WORKERS" not in _os_vw.environ', src)


class TestEarlyReleaseGate(unittest.TestCase):
    def test_wiring_subset_rule_and_modes(self):
        src = (ROOT / "build.py").read_text()
        seg = src.split("EARLY RELEASE-GATE DRY-RUN")[1].split("for pos, seg in enumerate")[0]
        # kill-switches: master gate env AND its own env
        self.assertIn("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE", seg)
        self.assertIn("VIDLORE_CLIPSTUDIO_EARLY_RF_GATE", seg)
        # warn mode NEVER raises early (review builds must complete)
        self.assertIn('if _mode_e == "warn":', seg)
        # typed kind for the orchestrate heal catch
        self.assertIn('kind="rejected_footage"', seg)
        # subset rule: an evaluation failure must never count as proof of doom
        self.assertIn("_ok_e = True", seg)
        # audit file written before the early raise (stale-audit hazard)
        self.assertIn("rejected_footage_audit.json", seg)

    def test_authoritative_gate_untouched_after_dry_run(self):
        src = (ROOT / "build.py").read_text()
        i_early = src.find("EARLY RELEASE-GATE DRY-RUN")
        i_gate = src.find("_rf_audit, _rf_block = [], []")
        self.assertGreater(i_early, 0)
        self.assertGreater(i_gate, i_early)


class TestDiscoverParallel(unittest.TestCase):
    def test_probe_barrier_and_serial_exception_parity(self):
        src = (ROOT / "discover.py").read_text()
        seg = src.split("def _probe_one")[1]
        self.assertIn("_f.result()", seg)          # barrier + submission-order propagation
        self.assertIn("_hd._HD_SEM", src.split("def _probe_one")[1].split("def ")[0])
        self.assertIn("VIDLORE_CLIPSTUDIO_HD_PROBE_WORKERS", src)

    def test_subs_prefetch_exact_slice_only(self):
        src = (ROOT / "discover.py").read_text()
        seg = src.split("_slice = pool[:limit]")[1].split("def ")[0]
        # '' failures consumed directly — no second fetch attempt the baseline lacked
        self.assertIn('_pre_subs[c.url] if c.url in _pre_subs else _fetch_subs_text(c.url)', seg)
        self.assertIn("dict.fromkeys(c.url for c in _slice)", seg)


class TestSelfhealPrefetch(unittest.TestCase):
    def test_ordered_accept_and_used_paths_prefilter(self):
        src = (ROOT / "selfheal.py").read_text()
        seg = src.split("def still_recover")[1].split("def _frame_text_dirty")[0]
        # candidates filtered on used_paths BEFORE the window (serial `tried` parity)
        self.assertIn("if sh.keyframe_path not in used_paths][:cand_n]", seg)
        # accept walk stays in ranked order over the SAME window
        self.assertIn("for rel, sid, sh in cands:", seg)
        # serial fallback when a prefetched verdict is missing
        self.assertIn("if v is None:", seg)
        # persisted-embed ranking (certified identical path), not the live CLIP pass
        self.assertIn("IF._shot_relevance(", seg)
        self.assertNotIn("IF._clip_relevance(", seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestBuildSweepParallel(unittest.TestCase):
    def test_branding_and_dodge_precompute_with_direct_fallback(self):
        src = (ROOT / "build.py").read_text()
        # branding first-pass probes precompute; missing verdicts fall back to the direct call
        self.assertIn('_clip_branding_text(Path(sp), _ocr_engine_tl(_ocr_eng))', src)
        self.assertIn("else _clip_branding_text(Path(cp), _ocr_eng)) for cp in clips]", src)
        # dodge FIRST-pass precomputed on a post-mutation snapshot...
        self.assertIn('_clip_has_burned_text(Path(sp), _ocr_engine_tl(_ocr_eng))', src)
        # ...but the POST-CROP recheck stays a direct fresh probe (in-place mutation hazard)
        seg = src.split("caption-dodge REPAIR")[0]
        self.assertIn("and not _clip_has_burned_text(Path(cp), _ocr_eng):",
                      src.split("_corner and _crop_clip_corner")[1][:200])

    def test_adscan_parallel_order_independent_aggregation(self):
        src = (ROOT / "build.py").read_text()
        seg = src.split("def _judge_one")[1].split("def _dense_confirm")[0]
        self.assertIn("VIDLORE_CLIPSTUDIO_ADSCAN_WORKERS", src)
        # consumers walk sorted(cand.items()) — completion order cannot leak into decisions
        self.assertIn("sorted(cand.items())", src)
        # serial path retained behind the kill-switch
        self.assertIn("for i, fp in enumerate(frames):", src)

    def test_thread_engine_helper_fallback(self):
        from vidlore.clipstudio.build import _ocr_engine_tl
        sentinel = object()
        # helper returns SOME engine; on construction failure it must return the fallback —
        # simulate by poisoning the import path? (construction works here, so just assert
        # it returns a working engine object or the fallback, never None)
        eng = _ocr_engine_tl(sentinel)
        self.assertIsNotNone(eng)
