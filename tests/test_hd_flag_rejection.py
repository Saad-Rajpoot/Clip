"""One unknown CLI flag took a finished 12-minute render to 0% HD.

MEASURED on portal job 409e284b60 (2026-08-02): `hd_path_ok 0/72`, `sub_480p_sources 72`, 1% of
sources at >=720p — the whole video built from legacy ~360p upscaled onto a 1080p canvas. Every
source reported the same thing:

    hd: no HD file (__main__.py: error: no such option: --remote-components) — fallback

`--remote-components ejs:github` is an ENHANCEMENT: it lets yt-dlp fetch the yt-dlp-ejs solver so
the JS "n challenge" can be answered. It was added unconditionally. When the resolved interpreter's
yt-dlp does not know the option, yt-dlp exits IN ARGUMENT PARSING — before any network call — so:

  * every rung of the client ladder fails identically and instantly (they all carry the flag),
  * every retry re-sends it,
  * `_classify_dl_err` read the message as "other", so the recovery sweep never re-attempted, and
  * the per-source log line reads like the video simply has no HD copy.

The failure is not reproducible from the machine's current state (every yt-dlp installed here now
accepts the flag, and a live probe of a failed source returns 720p), so the fix cannot be "make
that one build work". It is that an optional flag must never be able to reach 0% — three defenses,
each of which alone is sufficient:

  1. SUPPORT PROBE   — the flag is only sent if `--help` advertises it (cached per process).
  2. RUNTIME RETRY   — a rejection drops the flag process-wide and retries the SAME rung, which
                       survives even a yt-dlp swapped underneath a running render.
  3. RELEASE BLOCK   — hd_path_ok == 0 marks the output a REVIEW DRAFT, so a 100%-SD render can
                       never again call itself final.

    python3 -m pytest tests/test_hd_flag_rejection.py -q

No network, no LLM, no yt-dlp.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import hd_download as HD                      # noqa: E402

# the exact stderr the render recorded, 72 times
REAL = "__main__.py: error: no such option: --remote-components"
REAL_FALLBACK = f"hd: no HD file ({REAL}) — fallback"


def _reset():
    HD._RC_OFF = False
    HD._FLAG_SUPPORT.clear()
    HD._COOKIES_OFF = False


class TestItIsRecognisedAtAll(unittest.TestCase):
    """The render's own message, and the two other spellings a different parser would emit."""

    def setUp(self):
        _reset()

    def test_the_real_message_classifies_as_badflag(self):
        self.assertEqual(HD._classify_dl_err(REAL), "badflag")
        self.assertEqual(HD._classify_dl_err(REAL_FALLBACK), "badflag")

    def test_the_offending_flag_is_extracted(self):
        self.assertEqual(HD._bad_flag_name(REAL), "--remote-components")
        self.assertEqual(HD._bad_flag_name("yt-dlp: error: unrecognized arguments: --foo-bar"),
                         "--foo-bar")
        self.assertEqual(HD._bad_flag_name("error: ambiguous option: --remote"), "--remote")
        self.assertEqual(HD._bad_flag_name("HTTP Error 403: Forbidden"), "")

    def test_it_is_classified_BEFORE_the_youtube_classes(self):
        """It cannot coexist with a real response — argparse dies before the request. A message
        carrying both must still be read as our own broken command line."""
        self.assertEqual(HD._classify_dl_err(REAL + "\nERROR: video unavailable"), "badflag")
        self.assertEqual(HD._classify_dl_err(REAL + "\nHTTP Error 403: Forbidden"), "badflag")

    def test_the_sweep_will_re_attempt_it(self):
        """It went unrecovered because 'other' is not recoverable — 0 of 72 were retried."""
        self.assertTrue(HD.is_recoverable_hd_failure(REAL_FALLBACK))

    def test_a_genuinely_missing_video_is_still_not_recoverable(self):
        self.assertFalse(HD.is_recoverable_hd_failure("ERROR: Private video. Sign in"))
        self.assertEqual(HD._classify_dl_err("ERROR: Video unavailable"), "unavailable")

    def test_the_word_option_in_prose_does_not_trigger_it(self):
        """The marker is a REJECTION, not the word 'option'. yt-dlp prints advice constantly."""
        for benign in ("Use --cookies-from-browser or --cookies for the authentication",
                       "this option is deprecated, use --remote-components instead",
                       "WARNING: Some formats may be missing"):
            self.assertNotEqual(HD._classify_dl_err(benign), "badflag", benign)


class TestDefenceOne_SupportProbe(unittest.TestCase):
    def setUp(self):
        _reset()
        self._real_run = HD.subprocess.run

    def tearDown(self):
        HD.subprocess.run = self._real_run
        _reset()

    def _fake_help(self, text, calls):
        def run(cmd, **kw):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")
        HD.subprocess.run = run

    def test_an_unsupported_flag_is_never_sent(self):
        self._fake_help("Usage: yt-dlp [OPTIONS] URL\n  --cookies FILE\n", [])
        self.assertEqual(HD._remote_components(), [])

    def test_a_supported_flag_is_sent(self):
        self._fake_help("  --remote-components COMPONENT   Remote components to allow\n", [])
        self.assertEqual(HD._remote_components(), ["--remote-components", "ejs:github"])

    def test_the_probe_runs_once_per_process(self):
        """A --help per download would add a subprocess to every source in the pool."""
        calls = []
        self._fake_help("  --remote-components COMPONENT\n", calls)
        for _ in range(5):
            HD._remote_components()
        self.assertEqual(len(calls), 1, calls)

    def test_the_probe_asks_the_interpreter_that_will_run_the_download(self):
        calls = []
        self._fake_help("  --remote-components COMPONENT\n", calls)
        HD._remote_components()
        self.assertEqual(calls[0][:4], [HD.HD_PY, "-m", "yt_dlp", "--help"])

    def test_a_probe_that_cannot_run_fails_OPEN(self):
        """Evidence about the probe is not evidence about support; defence 2 still covers it."""
        def boom(cmd, **kw):
            raise OSError("no such interpreter")
        HD.subprocess.run = boom
        self.assertEqual(HD._remote_components(), ["--remote-components", "ejs:github"])

    def test_empty_help_output_fails_OPEN(self):
        self._fake_help("", [])
        self.assertEqual(HD._remote_components(), ["--remote-components", "ejs:github"])

    def test_the_env_override_still_wins_without_probing(self):
        calls = []
        self._fake_help("  --remote-components COMPONENT\n", calls)
        HD.os.environ["VIDLORE_HD_REMOTE_COMPONENTS"] = ""
        try:
            self.assertEqual(HD._remote_components(), [])
            self.assertEqual(calls, [])                  # disabled means no subprocess at all
        finally:
            HD.os.environ.pop("VIDLORE_HD_REMOTE_COMPONENTS", None)


class TestDefenceTwo_RuntimeRetry(unittest.TestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def test_a_rejection_drops_the_flag_for_the_rest_of_the_run(self):
        msgs = []
        self.assertTrue(HD._disable_unsupported_flag("--remote-components", REAL, msgs.append))
        self.assertTrue(HD._RC_OFF)
        self.assertEqual(HD._remote_components(), [])    # and never probes again
        self.assertTrue(any("--remote-components" in m for m in msgs))

    def test_only_the_first_caller_earns_the_retry(self):
        """Downloads run in parallel; a burst must not spend every attempt on the same discovery."""
        self.assertTrue(HD._disable_unsupported_flag("--remote-components", REAL))
        self.assertFalse(HD._disable_unsupported_flag("--remote-components", REAL))

    def test_an_unnamed_rejection_is_treated_as_the_optional_flag(self):
        self.assertTrue(HD._disable_unsupported_flag("", "error: no such option"))
        self.assertTrue(HD._RC_OFF)

    def test_a_cookie_flag_rejection_routes_to_the_cookie_switch(self):
        self.assertTrue(HD._disable_unsupported_flag("--cookies-from-browser", "x"))
        self.assertTrue(HD._COOKIES_OFF)
        self.assertFalse(HD._RC_OFF)                     # and does not disarm the solver

    def test_an_ESSENTIAL_flag_is_reported_not_silently_dropped(self):
        msgs = []
        self.assertFalse(HD._disable_unsupported_flag("--merge-output-format", "x", msgs.append))
        self.assertFalse(HD._RC_OFF)
        self.assertTrue(any("not optional" in m for m in msgs))

    def test_the_retry_keeps_the_SAME_client_rung(self):
        """The ladder answers 403s. Burning a rung on our own flag leaves nothing for a throttle —
        the same reasoning already applied to the cookie retry."""
        src = (ROOT / "vidlore" / "clipstudio" / "hd_download.py").read_text()
        seg = src.split('if last_class == "badflag":')[1].split('if last_class == "cookies":')[0]
        self.assertIn("continue", seg)
        self.assertNotIn("rung = ", seg)

    def test_an_undroppable_flag_stops_retrying_immediately(self):
        """Identical retries against an unparseable command line are pure latency."""
        src = (ROOT / "vidlore" / "clipstudio" / "hd_download.py").read_text()
        seg = src.split('if last_class == "badflag":')[1].split('if last_class == "cookies":')[0]
        self.assertIn("break", seg)


class TestDefenceThree_ReleaseBlock(unittest.TestCase):
    """A 100%-SD render passes every per-beat gate: the beats are right, the footage is just
    360p for twelve minutes. Only the aggregate can see it."""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "vidlore" / "clipstudio" / "build.py").read_text()

    def test_total_collapse_marks_the_output_a_review_draft(self):
        self.assertIn('int(_da.get("hd_path_ok") or 0) == 0', self.src)
        self.assertIn("_review_draft.append(_msg_hd)", self.src)

    def test_it_is_checked_BEFORE_the_rename(self):
        self.assertLess(self.src.index("_review_draft.append(_msg_hd)"),
                        self.src.index("if _review_draft:\n        try:\n            _draft"))

    def test_partial_degradation_does_NOT_block_a_release(self):
        """Some uploads genuinely have no HD copy. Holding those renders back would make the
        block meaningless — the trigger is collapse, not degradation."""
        self.assertNotIn('_da.get("hd_fallback")', self.src.split("_msg_hd")[0][-900:])

    def test_a_tiny_pool_does_not_trip_it(self):
        self.assertIn("_yt_n >= 5", self.src)

    def test_the_check_can_never_take_a_render_down(self):
        seg = self.src.split("_msg_hd = ")[0][-400:]
        self.assertIn("try:", seg)
        self.assertIn("hd-collapse check skipped", self.src)


class TestTheSelfUpdateIsNoLongerSilent(unittest.TestCase):
    """A failed `pip install -U` logged nothing AND stamped the weekly marker, so seven days of
    renders could inherit a stale HD stack with no trace."""

    def test_failure_is_logged(self):
        src = (ROOT / "vidlore" / "clipstudio" / "hd_download.py").read_text()
        seg = src.split("def maybe_update_ytdlp")[1].split("\ndef ")[0]
        self.assertIn("self-update FAILED", seg)
        self.assertIn("p.returncode", seg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
