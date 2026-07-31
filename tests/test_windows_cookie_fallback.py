"""A soft dependency took a whole Windows render down to 360p.

The owner's first render on the Windows bundle (job 957f56f925) built its entire 106.8s video from
SD footage upscaled to a 1080p canvas. download_audit: hd_path_ok 0, hd_fallback 42,
sub_480p_sources 42. Every one of them failed with the same line:

    ERROR: Could not copy Chrome cookie database.
    See https://github.com/yt-dlp/yt-dlp/issues/7271 for more info

Nothing was wrong with the videos, the PO-token server, or the network. yt-dlp aborts the whole
download when it cannot READ the browser profile, and on Windows it usually cannot: Chrome holds an
exclusive lock on its cookie DB while running, and since Chrome 127 App-Bound Encryption puts the
values out of reach even when the file can be copied (that failure reports as a DPAPI decrypt error
and never says the word "cookie").

Cookies are an OPTIMISATION — the PO-token path fetches public videos without them. Three fixes:

  1. `_classify_dl_err` gets a 'cookies' class, so the failure is no longer 'other'.
  2. The download retry drops cookies process-wide and retries the SAME client rung — the ladder
     exists to answer 403s and must not be burned on our own flag.
  3. `probe_max_height` shares `_cookie_args()` instead of hardcoding --cookies-from-browser; it
     used to die the same way and return 0, so discovery rated every 1080p upload as SD before a
     single byte was fetched.

    python3 -m pytest tests/test_windows_cookie_fallback.py -q

No network, no subprocess.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import hd_download as HD          # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"

# the exact string from job 957f56f925's download_audit
WIN_ERR = ("hd: no HD file (ERROR: Could not copy Chrome cookie database. See  "
           "https://github.com/yt-dlp/yt-dlp/issues/7271  for more info) — fallback")
DPAPI_ERR = ("ERROR: Failed to decrypt with DPAPI. See  "
             "https://github.com/yt-dlp/yt-dlp/issues/7271  for more info")


class TestTheWindowsFailureIsRecognised(unittest.TestCase):
    def test_the_render_killing_error_is_classified_as_cookies(self):
        self.assertEqual(HD._classify_dl_err(WIN_ERR), "cookies")
        self.assertTrue(HD.is_cookie_failure(WIN_ERR))

    def test_chrome_127_app_bound_encryption_counts_too(self):
        """It never says "cookie" — matching only that word would miss half of Windows."""
        self.assertNotIn("cookie", DPAPI_ERR.lower())
        self.assertEqual(HD._classify_dl_err(DPAPI_ERR), "cookies")

    def test_other_wordings_of_the_same_failure(self):
        for t in ("ERROR: could not find chrome cookies database",
                  "ERROR: Permission denied while reading the cookie database",
                  "ERROR: unable to extract cookies from browser"):
            self.assertEqual(HD._classify_dl_err(t), "cookies", t)

    def test_it_is_not_mistaken_for_the_video_being_gone(self):
        """The costly half of the bug: a tooling failure read as "no HD available"."""
        self.assertNotEqual(HD._classify_dl_err(WIN_ERR), "unavailable")

    def test_the_bot_check_that_ADVISES_cookies_must_not_disable_them(self):
        """yt-dlp's bot check literally prints "Use --cookies-from-browser". Matching the flag
        name, or testing this class before the sign-in patterns, would switch cookies off at the
        one moment they are the remedy."""
        t = "ERROR: Sign in to confirm you're not a bot. Use --cookies-from-browser"
        self.assertEqual(HD._classify_dl_err(t), "unavailable")
        self.assertFalse(HD.is_cookie_failure(t))


class TestTheOtherClassesAreUntouched(unittest.TestCase):
    def test_403_and_error_152_still_throttle(self):
        for t in ("ERROR: unable to download video data: HTTP Error 403: Forbidden",
                  "ERROR: [youtube] 0YM7oOTZvVM: This video is unavailable. Error code: 152"):
            self.assertEqual(HD._classify_dl_err(t), "throttle_403", t)

    def test_genuine_unavailability_still_permanent(self):
        for t in ("ERROR: Private video. Sign in if you've been granted access",
                  "This video has been removed by the uploader",
                  "Video unavailable"):
            self.assertEqual(HD._classify_dl_err(t), "unavailable", t)

    def test_unrelated_errors_stay_other(self):
        for t in ("ERROR: unable to rename file", "ERROR: ffmpeg exited with code 1",
                  "ERROR: unable to open database file"):
            self.assertEqual(HD._classify_dl_err(t), "other", t)

    def test_a_cookie_failure_is_not_a_po_token_failure(self):
        self.assertFalse(HD.is_po_token_failure(WIN_ERR))


class TestCookiesAreASoftDependency(unittest.TestCase):
    def setUp(self):
        self._was = HD._COOKIES_OFF
        HD._COOKIES_OFF = False

    def tearDown(self):
        HD._COOKIES_OFF = self._was

    def test_cookies_are_used_by_default(self):
        self.assertTrue(HD._cookie_args())

    def test_disabling_yields_a_clean_command(self):
        HD._disable_cookies("test")
        self.assertEqual(HD._cookie_args(), [])

    def test_only_the_first_caller_flips_the_switch(self):
        """The free retry is granted once. Without this, a parallel burst of cookie failures would
        each spend every attempt on one client rung."""
        self.assertTrue(HD._disable_cookies("first"))
        self.assertFalse(HD._disable_cookies("second"))

    def test_a_cookies_file_wins_over_the_live_profile(self, ):
        """The wiki-endorsed stable method — live profiles get their cookies rotated mid-run."""
        import os
        f = ROOT / "tests" / "_fake_cookies.txt"
        f.write_text("# Netscape HTTP Cookie File\n")
        os.environ["VIDLORE_HD_COOKIES_FILE"] = str(f)
        try:
            self.assertEqual(HD._cookie_args()[0], "--cookies")
        finally:
            os.environ.pop("VIDLORE_HD_COOKIES_FILE", None)
            f.unlink(missing_ok=True)


def code_only(text: str) -> str:
    """Drop comment lines — these assertions are about what RUNS, and the comments here quote the
    very flag names and failure strings the code must no longer use."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


class TestTheWiring(unittest.TestCase):
    def test_the_probe_no_longer_hardcodes_the_browser_flag(self):
        """It returned 0 on Windows, so discovery rated 1080p uploads as SD before downloading."""
        src = (SRC / "hd_download.py").read_text()
        probe = code_only(src.split("def probe_max_height")[1].split("\ndef ")[0])
        self.assertNotIn("--cookies-from-browser", probe)
        self.assertIn("*_cookie_args()", probe)
        self.assertIn('_classify_dl_err(p.stderr or "") == "cookies"', probe)

    def test_the_retry_does_not_burn_a_client_rung(self):
        src = (SRC / "hd_download.py").read_text()
        self.assertIn("rung = 0", src)
        self.assertIn("cmd = _cmd_for(rung)", src)
        branch = code_only(
            src.split('if last_class == "cookies":')[1].split("continue")[0])
        self.assertIn("_disable_cookies(last", branch)
        self.assertNotIn("rung", branch, "the rung must not advance before the cookie retry")

    def test_stranded_sources_are_swept(self):
        """Downloads run in parallel, so a burst can fall back before the switch flips."""
        self.assertTrue(HD.is_recoverable_hd_failure(WIN_ERR))
        self.assertTrue(HD.is_recoverable_hd_failure(
            "ERROR: unable to download video data: HTTP Error 403: Forbidden"))
        self.assertFalse(HD.is_recoverable_hd_failure("ERROR: Private video."))
        self.assertIn("_hd.is_recoverable_hd_failure(", (SRC / "download.py").read_text())

    def test_the_audit_counts_this_class_on_its_own(self):
        self.assertIn("hd_cookie_fallbacks", (SRC / "download.py").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
