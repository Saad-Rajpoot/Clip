"""HD 403 resilience: classification, retry policy, pot restart, and the recovery sweep.

Run standalone:  python3 tests/test_hd_403_resilience.py
"""
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio import hd_download as hd            # noqa: E402
from vidlore.clipstudio import download as dl               # noqa: E402
from vidlore.clipstudio.models import (                     # noqa: E402
    ClipProject, SourceVideo, SOURCE_OK)
from vidlore.clipstudio.config import ClipConfig            # noqa: E402


class TestClassify(unittest.TestCase):
    def test_403_is_transient(self):
        self.assertEqual(hd._classify_dl_err(
            "ERROR: unable to download video data: HTTP Error 403: Forbidden"),
            "throttle_403")

    def test_permanent_failures(self):
        for msg in ("Video unavailable", "Private video. Sign in if you've been granted access",
                    "ERROR: Sign in to confirm you're not a bot",
                    "This video has been removed by the uploader"):
            self.assertEqual(hd._classify_dl_err(msg), "unavailable", msg)

    def test_other(self):
        self.assertEqual(hd._classify_dl_err("read timed out"), "other")
        self.assertEqual(hd._classify_dl_err(""), "other")


class TestRetryPolicy(unittest.TestCase):
    """download_hd must NOT retry permanent failures, MUST back off + retry 403s."""

    def _run(self, stderr_text, retries=2):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=1, stderr=stderr_text, stdout="")

        msgs = []
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(hd.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(hd.time, "sleep"), \
                mock.patch.object(hd, "_restart_pot_server", return_value=True), \
                mock.patch.object(hd, "ensure_pot_server", return_value=True), \
                mock.patch.object(hd, "available", return_value=True), \
                mock.patch.object(hd, "HD_PY", "python3"):
            out = hd.download_hd("https://www.youtube.com/watch?v=x", str(Path(td) / "s1"),
                                 retries=retries, progress=msgs.append)
        return out, calls, msgs

    def test_unavailable_no_retry(self):
        out, calls, _ = self._run("ERROR: Private video", retries=3)
        self.assertIsNone(out)
        self.assertEqual(len(calls), 1, "permanent failure must not be retried")

    def test_403_retries_all_attempts(self):
        out, calls, msgs = self._run(
            "ERROR: unable to download video data: HTTP Error 403: Forbidden", retries=2)
        self.assertIsNone(out)
        self.assertEqual(len(calls), 3, "403 should use every attempt (retries+1)")
        self.assertTrue(any("[403]" in m for m in msgs),
                        f"fallback reason must carry the [403] tag: {msgs}")

    def test_403_fallback_tag_never_unavailable(self):
        _, _, msgs = self._run(
            "ERROR: unable to download video data: HTTP Error 403: Forbidden")
        self.assertFalse(any("[unavailable]" in m for m in msgs))


class TestPotRestartOnce(unittest.TestCase):
    def test_restart_gated_once_per_process(self):
        with mock.patch.object(hd, "_POT_403_RESTARTED", False), \
                mock.patch.object(hd.subprocess, "run") as run, \
                mock.patch.object(hd, "ensure_pot_server", return_value=True) as ens:
            run.return_value = types.SimpleNamespace(stdout="", returncode=0)
            self.assertTrue(hd._restart_pot_server())
            self.assertTrue(hd._POT_403_RESTARTED)
            self.assertFalse(hd._restart_pot_server(), "second restart must be refused")
            self.assertEqual(ens.call_count, 1)

    def test_env_kill_switch(self):
        with mock.patch.dict(os.environ, {"VIDLORE_HD_POT_RESTART_ON_403": "0"}), \
                mock.patch.object(hd, "_POT_403_RESTARTED", False):
            self.assertFalse(hd._restart_pot_server())
            self.assertFalse(hd._POT_403_RESTARTED)


class TestSweep(unittest.TestCase):
    """The end-of-stage sweep upgrades 403-fallen sources and never touches others."""

    def _proj(self, td):
        proj = ClipProject(name="t", root=str(td))
        proj.ensure_dirs()
        return proj

    def _src(self, proj, sid, fallback, height=360):
        f = proj.sources_dir / f"{sid}.mp4"
        f.write_bytes(b"legacy360")
        sv = SourceVideo(id=sid, url=f"https://www.youtube.com/watch?v={sid}",
                         title=sid, status=SOURCE_OK, height=height, width=640,
                         local_path=str(f), extra={})
        if fallback:
            sv.extra["hd_fallback"] = fallback
        return sv

    def _run_sweep(self, proj, hi_by_url):
        def fake_download_hd(url, stem, **kw):
            hi = hi_by_url.get(url)
            if hi:
                p = Path(stem + ".mp4")
                p.write_bytes(b"HD1080DATA")
                return dict(hi, path=str(p))
            return None

        cfg = ClipConfig()
        with mock.patch.object(dl, "_sha1_file", return_value="newsum"), \
                mock.patch.object(dl.time, "sleep"), \
                mock.patch("vidlore.clipstudio.hd_download.available", return_value=True), \
                mock.patch("vidlore.clipstudio.hd_download.download_hd",
                           side_effect=fake_download_hd):
            dl.download_candidates(proj, [], cfg, policy="approved_testing")

    def test_403_source_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            proj = self._proj(td)
            sv = self._src(proj, "aaa", "hd: no HD file [403](HTTP Error 403: Forbidden) — fallback")
            proj.sources = [sv]
            self._run_sweep(proj, {sv.url: {"height": 1080, "width": 1920,
                                            "duration": 10.0, "fps": 30.0}})
            self.assertEqual(sv.height, 1080)
            self.assertTrue(sv.extra.get("hd_recovered"))
            self.assertTrue(sv.extra.get("hd_path"))
            self.assertNotIn("hd_fallback", sv.extra)
            self.assertEqual(Path(sv.local_path).read_bytes(), b"HD1080DATA")

    def test_non_403_fallback_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            proj = self._proj(td)
            sv = self._src(proj, "bbb", "hd: no HD file [unavailable](Private video) — fallback")
            proj.sources = [sv]
            called = {}
            self._run_sweep(proj, called)
            self.assertEqual(sv.height, 360)
            self.assertNotIn("hd_recovered", sv.extra)

    def test_failed_retry_keeps_legacy_file(self):
        with tempfile.TemporaryDirectory() as td:
            proj = self._proj(td)
            sv = self._src(proj, "ccc", "HTTP Error 403: Forbidden")
            proj.sources = [sv]
            self._run_sweep(proj, {})            # retry yields nothing
            self.assertEqual(sv.height, 360)
            self.assertEqual(Path(sv.local_path).read_bytes(), b"legacy360")
            self.assertEqual(list(proj.sources_dir.glob("ccc.hdretry*")), [],
                             "temp namespace must be swept")

    def test_kill_switch(self):
        with tempfile.TemporaryDirectory() as td:
            proj = self._proj(td)
            sv = self._src(proj, "ddd", "HTTP Error 403: Forbidden")
            proj.sources = [sv]
            with mock.patch.dict(os.environ, {"VIDLORE_HD_403_SWEEP": "0"}):
                self._run_sweep(proj, {sv.url: {"height": 1080}})
            self.assertEqual(sv.height, 360)


if __name__ == "__main__":
    unittest.main(verbosity=2)
