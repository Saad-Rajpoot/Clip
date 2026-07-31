"""Decide legibility BEFORE the cut, and answer with a window the beat already approved.

The paired A/B blamed "the right source, the wrong seconds". Tracing what actually aired corrected
that: of the four named beats, two were not dark cuts at all but the FREEZE DONORS that replaced
them. The measured render froze 31 near-black clips, 30 onto a donor from a DIFFERENT SCENE — so
the damage is not that darkness aired, it is that darkness was discovered after cutting, when the
only remedy left was a bright frame of another scene.

TWO IDEAS WERE MEASURED AND REJECTED before this one:

  Slide within the shot. 175 of 272 beats (64%) air LONGER than their selected shot, median 1.03s
  beyond it — there is usually no room. Replaying an honest slide (the real wqc_render_window with
  illegibility spans injected, anchor-overlap and moment-preserve untouched) rescued 1 beat and
  regressed 1. Net zero.

  A new luma threshold on the aired window. On this material mean, p99.5 and lit-pixel fraction do
  not separate readable from unreadable: a plainly readable Arya-and-wight frame measures mean 24.4
  / p99.5 78, while an unreadable black wide measures 25.4 / 145. A mean floor at 25 would flag
  9.24% of the video — and the Long Night is the subject.

What works instead needs NO new number: ask `_clip_too_dark`'s exact question of the SOURCE window
before cutting, and when the answer is bad take the next window from the beat's OWN relevance-ranked
list. Of 28 first choices that probe dark, 27 have a legible alternate already in that list. The
other 244 beats issue one probe and change nothing.

    python3 -m pytest tests/test_aired_window_legibility.py -q

Synthesises its own clips with ffmpeg; skipped if ffmpeg is absent.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.build import (_source_window_too_dark,      # noqa: E402
                                      _clip_too_dark, _SRCDARK_MEMO)
from vidlore.clipstudio.match import _luma, _shot_unreadable        # noqa: E402

SRC = ROOT / "vidlore" / "clipstudio"


def _ff():
    try:
        from vidlore.clipstudio.config import ffmpeg_exe
        return str(ffmpeg_exe())
    except Exception:
        return ""


def _synth(ff, path, spec):
    """spec: list of (seconds, gray 0-255) segments concatenated into one clip."""
    parts = []
    for i, (dur, gray) in enumerate(spec):
        parts.append(f"color=c=gray:s=640x360:d={dur}:r=25,"
                     f"lutyuv=y={gray}[v{i}]")
    chain = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(spec))) + \
        f"concat=n={len(spec)}:v=1:a=0[out]"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c=black:s=16x16:d=0.04", "-filter_complex", chain,
                    "-map", "[out]", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    str(path)], capture_output=True, timeout=120)


class TestThePredicate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = _ff()
        if not cls.ff or not Path(cls.ff).exists():
            raise unittest.SkipTest("ffmpeg unavailable")
        cls.tmp = Path(tempfile.mkdtemp(prefix="wlegib_"))
        cls.bright = cls.tmp / "bright.mp4"
        cls.dark = cls.tmp / "dark.mp4"
        cls.dip = cls.tmp / "dip.mp4"          # 0.5s dip — under the gate's own run tolerance
        cls.runclip = cls.tmp / "run.mp4"      # 1.2s run — over it
        # NB: not `cls.run` — that shadows TestCase.run and unittest calls the Path
        _synth(cls.ff, cls.bright, [(4.0, 200)])
        _synth(cls.ff, cls.dark, [(4.0, 8)])
        _synth(cls.ff, cls.dip, [(1.5, 200), (0.5, 6), (2.0, 200)])
        _synth(cls.ff, cls.runclip, [(1.2, 200), (1.2, 6), (1.6, 200)])

    def setUp(self):
        _SRCDARK_MEMO.clear()

    def test_bright_window_passes(self):
        self.assertFalse(_source_window_too_dark(self.bright, 0.0, 3.5))

    def test_dark_window_is_caught(self):
        self.assertTrue(_source_window_too_dark(self.dark, 0.0, 3.5))

    def test_a_short_dip_is_tolerated(self):
        """The shipping gate deliberately allows brief fades; this must not be stricter than it."""
        self.assertFalse(_source_window_too_dark(self.dip, 0.0, 3.8))

    def test_a_sustained_run_is_caught(self):
        self.assertTrue(_source_window_too_dark(self.runclip, 0.0, 3.8))

    def test_it_agrees_with_the_post_cut_gate_it_mirrors(self):
        """The whole design rests on the source-side answer predicting the cut-clip verdict."""
        for f in (self.bright, self.dark, self.dip, self.runclip):
            self.assertEqual(_source_window_too_dark(f, 0.0, 3.5), _clip_too_dark(f), f.name)

    def test_it_reads_the_WINDOW_not_the_whole_file(self):
        """A file that is dark early and bright later must pass when the window is the later part."""
        f = self.tmp / "half.mp4"
        _synth(self.ff, f, [(2.0, 6), (3.0, 200)])
        self.assertTrue(_source_window_too_dark(f, 0.0, 2.0))
        self.assertFalse(_source_window_too_dark(f, 2.2, 2.5))

    def test_unmeasurable_fails_open(self):
        bad = self.tmp / "not_media.mp4"
        bad.write_bytes(b"nope")
        self.assertFalse(_source_window_too_dark(bad, 0.0, 3.0))

    def test_the_memo_probes_once(self):
        _SRCDARK_MEMO.clear()
        _source_window_too_dark(self.bright, 0.0, 3.5)
        n = len(_SRCDARK_MEMO)
        _source_window_too_dark(self.bright, 0.0, 3.5)
        self.assertEqual(len(_SRCDARK_MEMO), n)
        self.assertEqual(n, 1)


class TestTheWiring(unittest.TestCase):
    def test_pass_two_is_the_old_loop_so_the_option_set_only_grows(self):
        src = (SRC / "build.py").read_text()
        self.assertIn("for _lpass in (1, 2):", src)
        self.assertIn("if (_lpass == 1 and _legib_on and _wsp", src)

    def test_a_beat_whose_every_window_is_dark_keeps_todays_choice(self):
        """Coverage over cleanliness — nothing is ever dropped for being dark."""
        src = (SRC / "build.py").read_text()
        self.assertIn("keeping the ranked first choice", src)
        self.assertIn("black-repair sweep", src, "the post-cut backstop must still be named")

    def test_the_walk_probe_is_bounded(self):
        src = (SRC / "build.py").read_text()
        walk = src.split("def _next_distinct_shot")[1].split("\n    def ")[0]
        self.assertIn("_lg_left = 6", walk)
        self.assertIn("_lg_left -= 1", walk)
        self.assertIn("return after_t", walk, "today's fallback must remain")

    def test_the_walk_is_told_how_long_the_cut_is(self):
        src = (SRC / "build.py").read_text()
        self.assertIn("_next_distinct_shot(sid, max(_playhead.get(sid, 0.0),", src)
        self.assertIn("need=per_beat)", src)

    def test_one_kill_switch_covers_both_insertion_points(self):
        src = (SRC / "build.py").read_text()
        self.assertEqual(src.count('"VIDLORE_CLIPSTUDIO_WINDOW_LEGIBILITY", "1"'), 2)

    def test_no_new_thresholds_were_invented(self):
        """It must ask the SHIPPING gate's question — a new number here is a coin flip on this
        material, which is why the threshold approach was rejected."""
        src = (SRC / "build.py").read_text()
        fn = src.split("def _source_window_too_dark")[1][:2400]
        self.assertIn("floor: float = 50.0", fn)
        self.assertIn("min_dark_run: float = 0.8", fn)
        self.assertIn("crop=iw*0.9:ih*0.84:iw*0.05:ih*0.08,fps=2,scale=320:-1", fn)


class TestTheLumaSentinel(unittest.TestCase):
    """`float(getattr(shot, name, -1.0) or -1.0)` read a genuine 0.0 as "not computed", so a
    literally black shot took the fail-open path out of every luma gate."""

    def test_a_real_zero_survives_as_zero(self):
        self.assertEqual(_luma(NS(luma_avg=0.0), "luma_avg"), 0.0)

    def test_a_missing_field_is_still_the_sentinel(self):
        self.assertEqual(_luma(NS(), "luma_avg"), -1.0)
        self.assertEqual(_luma(NS(luma_avg=None), "luma_avg"), -1.0)

    def test_a_black_shot_is_now_unreadable(self):
        self.assertTrue(_shot_unreadable(
            NS(luma_avg=0.0, luma_hi=0.0, luma_min=0.0, luma_min_black_frac=1.0, quality=0.1)))

    def test_an_old_index_still_fails_open(self):
        self.assertFalse(_shot_unreadable(
            NS(luma_avg=-1.0, luma_hi=-1.0, luma_min=-1.0, luma_min_black_frac=-1.0, quality=0.5)))
        self.assertFalse(_shot_unreadable(NS(quality=0.5)))

    def test_a_legitimately_dark_scene_still_passes(self):
        """The Long Night is the subject of the video that surfaced this."""
        self.assertFalse(_shot_unreadable(
            NS(luma_avg=12.0, luma_hi=140.0, luma_min=10.0, luma_min_black_frac=0.1, quality=0.4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
