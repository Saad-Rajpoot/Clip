"""Windows parity for the HD download path — the platform branches, proven from macOS.

The HD (SABR/PO-token) downloader is what puts 720-1080p footage on screen; on Windows three
POSIX-only assumptions made it fail, and every one of them failed SILENTLY:

  1. `_merge_ffmpeg_dir` synthesised a helper named `ffmpeg` with no `.exe`, so yt-dlp found no
     merger and every HD download arrived as video with NO AUDIO → ASR 0 words → dialogue-lock
     (the strongest scene signal) dead. It also only searched POSIX install dirs.
  2. The PO-token server's port reclaim shelled out to `lsof` and detached with
     `start_new_session` — neither exists on Windows, so a stale server could never be replaced
     and the 403-recovery restart was a no-op.
  3. Self-heal's footage rescue probed for a console-script FILE named `yt-dlp` next to the
     interpreter; on Windows that is `yt-dlp.exe` in `Scripts\\`, so the search returned no
     candidates and the empty result looked exactly like "nothing found".

These tests fake `platform.system()` both ways, so they assert the Windows behaviour AND that the
macOS behaviour is unchanged — the whole point being that a Windows fix must not touch the Mac.

    python3 -m pytest tests/test_windows_parity.py -q

No network, no ffmpeg, no yt-dlp.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import hd_download as HD                      # noqa: E402
from vidlore.clipstudio import selfheal as SH                         # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"


def _as(system: str):
    """Pretend we are on `system` for everything the module asks."""
    return mock.patch.object(HD.platform, "system", return_value=system)


class TestExeSuffix(unittest.TestCase):
    def test_suffix_per_platform(self):
        with _as("Windows"):
            self.assertEqual(HD._exe_suffix(), ".exe")
        with _as("Darwin"):
            self.assertEqual(HD._exe_suffix(), "")
        with _as("Linux"):
            self.assertEqual(HD._exe_suffix(), "")


class TestMergeFfmpegDir(unittest.TestCase):
    """yt-dlp joins the directory it is given with the LITERAL name ffmpeg/ffmpeg.exe."""

    def test_windows_hint_needs_dot_exe(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "ffmpeg").write_bytes(b"x")          # POSIX-named only
            # the bare-name file must NOT satisfy Windows...
            with _as("Windows"), \
                    mock.patch("vidlore.ffmpeg_tool.ytdlp_ffmpeg_dir", return_value=None), \
                    mock.patch("shutil.which", return_value=None), \
                    mock.patch("vidlore.clipstudio.config.ffmpeg_exe", return_value=""):
                self.assertNotEqual(HD._merge_ffmpeg_dir(td), td)
            with _as("Darwin"):
                # ...but it is exactly what macOS wants
                self.assertEqual(HD._merge_ffmpeg_dir(td), td)

    def test_windows_hint_with_exe_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "ffmpeg.exe").write_bytes(b"x")
            with _as("Windows"):
                self.assertEqual(HD._merge_ffmpeg_dir(td), td)

    def test_delegates_to_the_platform_correct_engine_helper(self):
        """The engine already solves suffix + symlink-fallback + PATH in one place; the old
        macOS-only duplicate is what broke Windows."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "ffmpeg.exe").write_bytes(b"x")
            with _as("Windows"), \
                    mock.patch("vidlore.ffmpeg_tool.ytdlp_ffmpeg_dir", return_value=td) as yfd:
                self.assertEqual(HD._merge_ffmpeg_dir(""), td)
                yfd.assert_called_once()

    def test_engine_helper_result_is_rejected_when_it_lacks_the_exe(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as sysdir:
            (Path(sysdir) / "ffmpeg.exe").write_bytes(b"x")
            with _as("Windows"), \
                    mock.patch("vidlore.ffmpeg_tool.ytdlp_ffmpeg_dir", return_value=td), \
                    mock.patch("shutil.which",
                               return_value=str(Path(sysdir) / "ffmpeg.exe")):
                self.assertEqual(HD._merge_ffmpeg_dir(""), sysdir)

    def test_last_resort_link_carries_the_suffix(self):
        with tempfile.TemporaryDirectory() as home:
            real = Path(home) / "ffmpeg-win-x86_64-v7.1.exe"     # imageio's VERSIONED name
            real.write_bytes(b"x")
            with _as("Windows"), \
                    mock.patch("vidlore.ffmpeg_tool.ytdlp_ffmpeg_dir", return_value=None), \
                    mock.patch("shutil.which", return_value=None), \
                    mock.patch("vidlore.clipstudio.config.ffmpeg_exe", return_value=str(real)), \
                    mock.patch("os.path.expanduser",
                               side_effect=lambda p: home if p == "~" else p):
                d = HD._merge_ffmpeg_dir("")
            self.assertTrue((Path(d) / "ffmpeg.exe").exists(),
                            "the synthesised helper must be named ffmpeg.exe on Windows")


class TestPotServerPortReclaim(unittest.TestCase):
    def test_windows_uses_netstat_and_taskkill(self):
        netstat = ("  Proto  Local Address      Foreign Address    State       PID\n"
                   "  TCP    127.0.0.1:4416     0.0.0.0:0          LISTENING   6120\n"
                   "  TCP    127.0.0.1:4416     127.0.0.1:5000     ESTABLISHED 7777\n"
                   "  TCP    127.0.0.1:9999     0.0.0.0:0          LISTENING   4242\n")
        with _as("Windows"), \
                mock.patch.object(subprocess, "run",
                                  return_value=mock.Mock(stdout=netstat)) as run:
            pids = HD._pids_on_port(4416)
            self.assertEqual(run.call_args[0][0][0], "netstat")
        # only the LISTENING socket on OUR port
        self.assertEqual(pids, ["6120"])
        with _as("Windows"), mock.patch.object(subprocess, "run") as run:
            HD._kill_pid("6120")
            self.assertEqual(run.call_args[0][0], ["taskkill", "/F", "/PID", "6120"])

    def test_posix_still_uses_lsof_and_sigterm(self):
        with _as("Darwin"), \
                mock.patch.object(subprocess, "run",
                                  return_value=mock.Mock(stdout="321\n322\n")) as run:
            self.assertEqual(HD._pids_on_port(4416), ["321", "322"])
            self.assertEqual(run.call_args[0][0][0], "lsof")
        with _as("Darwin"), mock.patch.object(os, "kill") as k:
            HD._kill_pid("321")
            k.assert_called_once()

    def test_reclaim_failure_is_swallowed(self):
        with _as("Windows"), mock.patch.object(subprocess, "run",
                                               side_effect=OSError("no netstat")):
            self.assertEqual(HD._pids_on_port(4416), [])
        with _as("Windows"), mock.patch.object(subprocess, "run",
                                               side_effect=OSError("boom")):
            HD._kill_pid("1")                                # must not raise

    def test_detach_kwargs_per_platform(self):
        with _as("Darwin"):
            self.assertEqual(HD._detach_kwargs(), {"start_new_session": True})
        with _as("Windows"):
            kw = HD._detach_kwargs()
            self.assertNotIn("start_new_session", kw, "setsid does not exist on Windows")
            self.assertIn("creationflags", kw)
            self.assertTrue(kw["creationflags"])

    def test_server_start_uses_the_platform_detach(self):
        src = (SRC / "hd_download.py").read_text()
        self.assertIn("**_detach_kwargs(),", src)
        self.assertNotIn("start_new_session=True,", src)


class TestSelfhealYtSearch(unittest.TestCase):
    def test_invokes_yt_dlp_as_a_module_not_a_console_script(self):
        with tempfile.TemporaryDirectory() as td:
            py = Path(td) / "python.exe"
            py.write_bytes(b"x")                             # no `yt-dlp` file anywhere
            with mock.patch.object(HD, "HD_PY", str(py)), \
                    mock.patch.object(subprocess, "run",
                                      return_value=mock.Mock(stdout="")) as run:
                SH._yt_search_candidates(["ned stark execution"], log=lambda m: None)
            argv = run.call_args[0][0]
            self.assertEqual(argv[:3], [str(py), "-m", "yt_dlp"],
                             "must run the module, so Scripts\\yt-dlp.exe is irrelevant")

    def test_missing_hd_python_is_logged_not_silent(self):
        msgs = []
        with mock.patch.object(HD, "HD_PY", ""):
            out = SH._yt_search_candidates(["q"], log=msgs.append)
        self.assertEqual(out, [])
        self.assertTrue(any("yt-search unavailable" in m for m in msgs),
                        f"an empty rescue must say why; got {msgs}")


class TestFaceIdMissingIsLoud(unittest.TestCase):
    def test_orchestrate_warns_when_faceid_is_unavailable(self):
        src = (SRC / "orchestrate.py").read_text()
        # stage 4 proper — NOT the earlier prewarmer block, which shares the log prefix
        seg = src.split('log("4/9 · build Face-ID references")')[1].split("5 — deep index")[0]
        self.assertIn("Face-ID UNAVAILABLE", seg)
        self.assertIn("VIDLORE_CLIPSTUDIO_MODELS", seg)
        self.assertIn("wrong-character footage can no longer be rejected", seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestHdAvailabilityDoesNotRequireNode(unittest.TestCase):
    """Node was a FALSE hard requirement: nothing runs it. Deno starts the PO-token server and
    serves yt-dlp-ejs's JS challenges; yt-dlp only gets the youtubepot-bgutilhttp endpoint. On a
    box with Deno but no Node, available() returned False and EVERY source silently fell back to
    ~360p — the biggest quality regression in the pipeline, for a dependency never used."""

    def _with(self, *, node, deno, hd_py="py", pot="/pot", enabled=True):
        return mock.patch.multiple(HD, NODE_BIN=node, DENO_BIN=deno, HD_PY=hd_py,
                                   POT_SERVER_DIR=pot, HD_ENABLED=enabled)

    def test_available_without_node(self):
        with self._with(node="", deno="/usr/local/bin/deno"):
            self.assertTrue(HD.available(), "Deno alone must be enough")

    def test_unavailable_without_deno(self):
        with self._with(node="/usr/bin/node", deno=""):
            self.assertFalse(HD.available(), "Deno genuinely runs the PO server")

    def test_still_needs_hd_python_and_pot_dir_and_the_kill_switch(self):
        with self._with(node="", deno="/d", hd_py=""):
            self.assertFalse(HD.available())
        with self._with(node="", deno="/d", pot=""):
            self.assertFalse(HD.available())
        with self._with(node="", deno="/d", enabled=False):
            self.assertFalse(HD.available())

    def test_path_extension_tolerates_a_missing_node(self):
        with self._with(node="", deno="/usr/local/bin/deno"):
            env = HD._env_with_runtimes()
        self.assertIn("PATH", env)
        self.assertIn("/usr/local/bin", env["PATH"])

    def test_windows_runtime_candidates_include_exe(self):
        src = (SRC / "hd_download.py").read_text()
        self.assertIn('".deno/bin/deno.exe"', src)
        self.assertIn('".local-node/bin/node.exe"', src)

    def test_launcher_provisions_hd_and_skips_node(self):
        bat = (ROOT / "run-windows.bat").read_text()
        self.assertIn(".hdvenv\\Scripts\\python.exe", bat)
        self.assertIn("yt-dlp yt-dlp-ejs bgutil-ytdlp-pot-provider", bat)
        self.assertIn("deno.land/install.ps1", bat)
        self.assertIn("VIDLORE_HD_SETUP", bat)               # kill switch
        self.assertNotIn("nodejs.org", bat)                  # node is NOT installed


class TestPreflightRunsOnLaunch(unittest.TestCase):
    """The health-check the owner used to type by hand now runs on every click."""

    def test_launcher_and_diagnose_both_invoke_it(self):
        bat = (ROOT / "run-windows.bat").read_text()
        diag = (ROOT / "windows-diagnose.bat").read_text()
        self.assertIn("clipstudio_preflight.py", bat)
        self.assertIn("clipstudio_preflight.py", diag)
        self.assertIn("VIDLORE_PREFLIGHT", bat)                     # kill switch
        # informational only — it must run BEFORE the portal, and must not gate it
        self.assertLess(bat.find("clipstudio_preflight"), bat.find("vidlore.clipstudio.web"))

    def test_preflight_reports_every_asset_that_breaks_a_render(self):
        import subprocess as sp
        env = dict(os.environ, VIDLORE_CLIP_DIR="/nope", VIDLORE_CLIPSTUDIO_MODELS="/nope",
                   VIDLORE_MUSIC_DIR="/nope", VIDLORE_HD_DOWNLOAD="0")
        r = sp.run([sys.executable, str(ROOT / "tools" / "clipstudio_preflight.py")],
                   capture_output=True, text=True, env=env, timeout=300)
        out = r.stdout
        for label in ("music", "CLIP models", "Face-ID", "HD download", "ffmpeg", "API keys"):
            self.assertIn(label, out, f"{label} row missing")
        self.assertIn("XX", out, "a missing blocker must be marked, not glossed over")
        self.assertEqual(r.returncode, 0, "must never block the launcher")

    def test_preflight_is_cheap_enough_to_run_every_launch(self):
        """It checks that model FILES exist rather than loading them — loading CLIP is ~600 MB."""
        src = (ROOT / "tools" / "clipstudio_preflight.py").read_text()
        self.assertNotIn("clip_available()", src)
        self.assertNotIn("_try_load", src)


class TestLauncherDenoProvisioning(unittest.TestCase):
    """Measured on a real Windows run: `.hdvenv` provisioned fine but Deno did NOT, and the
    failure was invisible because the PowerShell output went to >nul. The launcher now uses a
    deterministic ZIP install, shows every error, and exports the path explicitly."""

    def setUp(self):
        self.bat = (ROOT / "run-windows.bat").read_text()

    def test_deterministic_zip_first_then_official_installer(self):
        self.assertIn("deno-x86_64-pc-windows-msvc.zip", self.bat)
        self.assertIn("deno.land/install.ps1", self.bat)          # fallback kept
        self.assertLess(self.bat.find("deno-x86_64-pc-windows-msvc.zip"),
                        self.bat.find("Doosra tareeqa"))

    def test_deno_errors_are_never_suppressed(self):
        for line in self.bat.splitlines():
            if "deno" in line.lower() and "powershell" in line.lower():
                self.assertNotIn(">nul", line,
                                 "a silent Deno failure is exactly what hid this bug")

    def test_resolved_path_is_exported_not_left_to_path_lookup(self):
        self.assertIn('set "VIDLORE_HD_DENO=', self.bat)

    def test_no_bare_exclamation_in_messages(self):
        """EnableDelayedExpansion makes '!' a metacharacter — batch EATS it, so '[!] ...' printed
        as '[ ...' and truncated the remedy on the owner's screen."""
        self.assertNotIn("[!]", self.bat)

    def test_env_override_is_honoured_by_the_resolver(self):
        with mock.patch.dict(os.environ, {"VIDLORE_HD_DENO": "/some/deno"}):
            import importlib
            from vidlore.clipstudio import hd_download as fresh
            importlib.reload(fresh)
            self.assertEqual(fresh.DENO_BIN, "/some/deno")
        import importlib
        from vidlore.clipstudio import hd_download as restore
        importlib.reload(restore)


class TestFixDenoHelper(unittest.TestCase):
    """The owner could not act on a pasted PowerShell one-liner, so the remedy is a file they
    can double-click. It must be self-contained and must not close its own window on failure."""

    def setUp(self):
        self.bat = (ROOT / "windows" / "Fix-Deno.bat").read_text()

    def test_installs_deno_both_ways_and_reports(self):
        self.assertIn("deno-x86_64-pc-windows-msvc.zip", self.bat)
        self.assertIn("deno.land/install.ps1", self.bat)
        self.assertIn("--version", self.bat)               # proves it actually landed
        self.assertGreaterEqual(self.bat.count("pause"), 2, "window must stay open to be read")

    def test_no_delayed_expansion_so_messages_survive(self):
        self.assertNotIn("EnableDelayedExpansion", self.bat)

    def test_readme_points_at_it(self):
        self.assertIn("Fix-Deno.bat", (ROOT / "windows" / "README.txt").read_text())
