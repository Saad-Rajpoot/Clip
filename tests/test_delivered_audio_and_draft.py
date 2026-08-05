"""Two ways a bad render looked like a good one, both found by the Windows-render audit.

A. THE AUDIO TRACK LIED ABOUT ITS OWN TIMELINE, AND THE GATE READ THE LIE.
   Job 957f56f925 delivered 106.688s of actual audio under a 106.800s video — 124 ms truncated —
   and logged `delivered A/V sync OK`. It passed because ONE AAC frame declared 5488 samples
   instead of 1024, and that 93 ms of invented timeline padded the container's declared duration
   back inside the 33 ms tolerance. The two defects hid each other, and the gate could not see
   either because it reads `stream=duration`, which is exactly the number the lie inflates.

   Not a Windows quirk: a macOS render carries the same class at 47 ms (measured on
   arya_intro_render). The trigger was NOT reproduced — a synthetic amix with a short bed and the
   same dropout_transition emits a clean timeline — so `asetpts=N/SR/TB` before the encoder is
   prevention aimed at the class, and `_audio_frame_anomalies` is what actually catches it.

B. A RELEASE-BLOCKED DRAFT WAS WRITTEN TO `final.mp4`.
   In warn mode the render completes on purpose, so the weak beats can be watched and audited
   rather than the whole render thrown away. But it landed under the same filename a passing render
   uses, with no marker in the picture — the only signals were build.log, review.html and the
   portal's job status, none of which travel with the mp4. One reached another machine on a USB
   stick and was read as finished (7 unresolved beats, ~46% relevance).

    python3 -m pytest tests/test_delivered_audio_and_draft.py -q

No network. The audio tests synthesise their own files with ffmpeg and are skipped if it is absent.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.assemble import _audio_frame_anomalies      # noqa: E402

SRC = ROOT / "vidlore"


def _ffmpeg():
    try:
        from vidlore.clipstudio.config import ffmpeg_exe
        return str(ffmpeg_exe())
    except Exception:
        return ""


class TestTheAudioTimelineDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = _ffmpeg()
        if not cls.ff or not Path(cls.ff).exists():
            raise unittest.SkipTest("ffmpeg unavailable")
        cls.tmp = tempfile.mkdtemp(prefix="av_gate_")
        cls.clean = os.path.join(cls.tmp, "clean.m4a")
        subprocess.run([cls.ff, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "sine=f=300:d=6:r=48000",
                        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000", cls.clean],
                       check=True)

    def test_a_clean_track_reports_no_anomaly(self):
        r = _audio_frame_anomalies(self.clean)
        self.assertTrue(r, "the detector must produce evidence for a readable file")
        self.assertEqual(r["anomalies"], 0)
        self.assertEqual(r["modal_samples"], 1024)
        self.assertAlmostEqual(r["true_media_s"], 6.0, delta=0.15)

    def test_the_last_frame_is_exempt(self):
        """A short tail is normal AAC padding, not a broken timeline."""
        r = _audio_frame_anomalies(self.clean)
        self.assertEqual(r["anomalies"], 0)

    def test_an_unreadable_file_says_nothing_rather_than_crashing(self):
        bad = os.path.join(self.tmp, "not_media.m4a")
        Path(bad).write_bytes(b"not a container")
        self.assertEqual(_audio_frame_anomalies(bad), {})

    def test_the_shipped_windows_render_would_now_be_caught(self):
        """The real artifact, if it is still on the USB stick. Its one bad frame is 93 ms — over
        the 64 ms audible floor, so it must be fatal, not a warning."""
        p = Path("/Volumes/USB/957f56f925/output/final.mp4")
        if not p.exists():
            self.skipTest("the Windows render is not mounted")
        r = _audio_frame_anomalies(p)
        self.assertEqual(r["anomalies"], 1)
        self.assertAlmostEqual(r["gap_s"] * 1000, 93.0, delta=1.0)
        self.assertGreaterEqual(r["gap_s"] * 1000, 64.0)


class TestTheGateWiring(unittest.TestCase):
    def test_the_duration_check_alone_is_no_longer_the_whole_gate(self):
        src = (SRC / "assemble.py").read_text()
        seg = src.split("def assert_delivered_av_sync")[1]
        self.assertIn("_audio_frame_anomalies(out_path)", seg)
        self.assertIn("VIDLORE_AUDIO_GAP_FATAL_MS", seg)

    def test_small_gaps_warn_and_large_ones_raise(self):
        """Renders on BOTH platforms have carried small ones for months; failing a four-hour render
        over 3 ms would be the wrong trade, and staying silent is how this shipped."""
        src = (SRC / "assemble.py").read_text()
        seg = src.split("def assert_delivered_av_sync")[1]
        self.assertIn('res["audio_frames"]["warning"] = _msg', seg)
        self.assertIn("raise TimelineSyncError", seg.split("_gap_ms >= _fatal_ms")[1][:200])

    def test_timestamps_are_rebuilt_from_samples_before_the_encoder(self):
        src = (SRC / "assemble.py").read_text()
        self.assertIn('_APTS = "asetpts=N/SR/TB"', src)
        self.assertEqual(src.count("{_LIMIT},{_APTS}[a]"), 2,
                         "both the voice-only and the mixed chain must be normalised")


class TestReviewDraftsAreMarked(unittest.TestCase):
    def test_every_warn_gate_records_a_reason(self):
        """Four gates now mark a draft, each seeing something the others cannot: semantic
        relevance, the early footage gate, the authoritative post-assembly gate, and a total HD
        collapse, which no per-beat check can see because every beat is correct and merely 360p
        (measured: a 12-minute render shipped as final with hd_path_ok 0/72)."""
        src = (SRC / "clipstudio" / "build.py").read_text()
        self.assertEqual(src.count("_review_draft.append("), 4,
                         "each independent warn gate must record its own reason")
        self.assertIn("_review_draft.append(_msg_hd)", src)

    def test_the_file_is_renamed_and_the_new_path_returned(self):
        """The portal's download link follows res['output'], so the rename must reach the caller."""
        src = (SRC / "clipstudio" / "build.py").read_text()
        seg = src.split("REVIEW DRAFT — rename")[1][:1400]
        self.assertIn('.REVIEW_DRAFT"', seg)
        self.assertIn("result.replace(_draft)", seg)
        self.assertIn("result = _draft", seg)

    def test_the_srt_follows_its_video(self):
        src = (SRC / "clipstudio" / "build.py").read_text()
        self.assertIn('for _side in (".srt",):', src)

    def test_a_failed_rename_never_loses_the_render(self):
        src = (SRC / "clipstudio" / "build.py").read_text()
        seg = src.split("REVIEW DRAFT — rename")[1][:1600]
        self.assertIn("except Exception as _e_rd", seg)
        self.assertIn("is NOT for publication", seg)

    def test_it_happens_after_the_av_gate(self):
        """Renaming first would leave the gate measuring a path that no longer exists."""
        src = (SRC / "clipstudio" / "build.py").read_text()
        self.assertLess(src.index("_sync = _avsync(result)"),
                        src.index("REVIEW DRAFT — rename"))

    def test_a_passing_render_is_untouched(self):
        src = (SRC / "clipstudio" / "build.py").read_text()
        self.assertIn("if _review_draft:", src)
        self.assertIn("_review_draft: list = []", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
