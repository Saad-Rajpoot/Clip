"""The HD path collapsed to 0p because the JS-challenge SOLVER SCRIPT is no longer bundled.

Found 2026-07-31 while pre-flighting a rerender: `probe_max_height` returned 0 for six pool URLs in
a row. `--list-formats` explained it — YouTube offered ONLY storyboard images:

    [youtube] [jsc:deno] Solving JS challenges using deno
    WARNING: n challenge solving failed: Some formats may be missing.
    WARNING: Only images are available for download.
    ... the EJS script distribution (deno) and NPM package (deno) were skipped. These may be
    required to solve JS challenges. You can enable these downloads with --remote-components
    ejs:github

Deno was installed and running; what was missing is the SCRIPT it executes (yt-dlp-ejs), which
recent yt-dlp fetches only when remote components are enabled. YouTube offers no media formats at
all until the "n challenge" is solved, so every format selector missed and the failure surfaced as
`Requested format is not available` — indistinguishable, to the fallback, from a video that simply
has no HD copy. The render would have been another 100% 360p upscale.

With `--remote-components ejs:github` the same five URLs probe 1080p.

This is the THIRD distinct cause of a total HD collapse in this project, after `Error code: 152`
(PO-token/SABR rejection read as unavailability) and `Could not copy Chrome cookie database`
(Windows profile lock). All three shared one shape: a TOOLING failure that reports itself as
"no HD available" and quietly degrades the whole render.

    python3 -m pytest tests/test_hd_js_challenge.py -q

No network, no subprocess.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import hd_download as HD          # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"

WARN_N = "WARNING: [youtube] icdJ4RLP-Xw: n challenge solving failed: Some formats may be missing."
WARN_IMG = "WARNING: Only images are available for download. use --list-formats to see them"
ERR_FMT = ("ERROR: [youtube] icdJ4RLP-Xw: Requested format is not available. "
           "Use --list-formats for a list of available formats")


class TestTheSolverScriptIsRequested(unittest.TestCase):
    def test_remote_components_on_by_default(self):
        self.assertEqual(HD._remote_components(), ["--remote-components", "ejs:github"])

    def test_it_can_be_pointed_at_the_npm_mirror_or_disabled(self):
        for val, want in (("ejs:npm", ["--remote-components", "ejs:npm"]), ("", [])):
            os.environ["VIDLORE_HD_REMOTE_COMPONENTS"] = val
            try:
                self.assertEqual(HD._remote_components(), want)
            finally:
                os.environ.pop("VIDLORE_HD_REMOTE_COMPONENTS", None)

    def test_both_call_sites_pass_it(self):
        """The probe and the download must agree — a probe that reads 0p writes off an HD source
        before a byte is fetched, which is the same collapse one stage earlier."""
        src = (SRC / "hd_download.py").read_text()
        self.assertEqual(src.count("*_remote_components()"), 2)
        probe = src.split("def probe_max_height")[1].split("\ndef ")[0]
        self.assertIn("*_remote_components()", probe)


class TestTheFailureIsRecognised(unittest.TestCase):
    def test_the_cause_lines_classify(self):
        for t in (WARN_N, WARN_IMG,
                  "ERROR: challenge solver script distribution missing",
                  "the EJS script distribution (deno) was skipped"):
            self.assertEqual(HD._classify_dl_err(t), "jschallenge", t)

    def test_the_bare_symptom_is_NOT_matched(self):
        """"Requested format is not available" has innocent explanations (a real format filter
        miss). Matching the symptom would mislabel those; the cause lines are unambiguous."""
        self.assertEqual(HD._classify_dl_err(ERR_FMT), "other")

    def test_a_stranded_source_gets_swept(self):
        self.assertTrue(HD.is_recoverable_hd_failure(WARN_N))
        self.assertTrue(HD.is_recoverable_hd_failure(WARN_IMG))

    def test_it_is_not_read_as_the_video_being_gone(self):
        for t in (WARN_N, WARN_IMG):
            self.assertNotEqual(HD._classify_dl_err(t), "unavailable", t)


class TestTheOtherClassesStillHold(unittest.TestCase):
    def test_no_regressions(self):
        for t, exp in (
                ("ERROR: unable to download video data: HTTP Error 403: Forbidden", "throttle_403"),
                ("ERROR: [youtube] X: This video is unavailable. Error code: 152", "throttle_403"),
                ("ERROR: Could not copy Chrome cookie database", "cookies"),
                ("ERROR: Failed to decrypt with DPAPI", "cookies"),
                ("ERROR: Private video. Sign in if you've been granted access", "unavailable"),
                ("ERROR: This video has been removed by the uploader", "unavailable"),
                ("ERROR: unable to rename file", "other"),
                ("ERROR: ffmpeg exited with code 1", "other")):
            self.assertEqual(HD._classify_dl_err(t), exp, t)


if __name__ == "__main__":
    unittest.main(verbosity=2)
