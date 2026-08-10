"""A near-black clip is repaired from its OWN footage, never from the beat next door.

The dark-clip pass used to freeze-replace an unreadable clip with the PREVIOUS beat's clip whenever
the beat had no second clip of its own. Two things are wrong with that. The content is a different
scene — the exact defect the aired-window legibility work removed (31 freezes, 30 cross-scene). And
the scene-lineage contract cannot even record it: an ordinary derivative must own the beat it airs
on, so the entry came out `owner_beat=18, original_beat=19` and the gate refused the render.

MEASURED on job 03768be9ac: that cost 3h47m and a fully cut 154-beat video, for ONE clip — and the
clip did not need a donor at all. Its luma_hi ran 124,124,125,146,194,33,34: legible for 2.5s of
3.5s, dark only in the tail. Its own frames were the right answer the whole time.

So the pass now airs the beat's longest legible run and holds that run's last frame across the dark
remainder — bounded (the run must be ≥1s and cover half the beat; the held part may not exceed the
sanctioned single-hold cap), re-probed with `_clip_too_dark`, and owned by the beat itself. A beat
with nothing legible of its own reaches the gates as the footage gap it is.

    python3 -m pytest tests/test_dark_clip_self_repair.py -q

Synthesises its own clips with ffmpeg; skipped if ffmpeg is absent.
"""
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio import build as B                                  # noqa: E402
from vidlore.clipstudio.build import (_clip_too_dark, _clip_luma_profile,  # noqa: E402
                                     _self_clean_repair)
from vidlore.clipstudio.ingest import probe                                # noqa: E402
from vidlore.clipstudio.scene_lineage import (selection_binding,           # noqa: E402
                                              validate_scene_lineage)


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
        parts.append(f"color=c=gray:s=640x360:d={dur}:r=25,lutyuv=y={gray}[v{i}]")
    chain = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(spec))) + \
        f"concat=n={len(spec)}:v=1:a=0[out]"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=16x16:d=0.04", "-filter_complex", chain,
                    "-map", "[out]", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    str(path)], capture_output=True, timeout=120)


class TestSelfRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = _ff()
        if not cls.ff or not Path(cls.ff).exists():
            raise unittest.SkipTest("ffmpeg unavailable")
        cls.tmp = Path(tempfile.mkdtemp(prefix="selfclean_"))
        # the shape that cost the render: legible, then a dark tail
        cls.tail = cls.tmp / "tail.mp4"
        _synth(cls.ff, cls.tail, [(2.5, 200), (1.0, 6)])
        # the mirror image — the legible run is NOT at t=0
        cls.head = cls.tmp / "head.mp4"
        _synth(cls.ff, cls.head, [(1.0, 6), (2.5, 200)])
        cls.allblack = cls.tmp / "allblack.mp4"
        _synth(cls.ff, cls.allblack, [(3.5, 6)])
        cls.sliver = cls.tmp / "sliver.mp4"           # legible for 1.0s of 4.0s
        _synth(cls.ff, cls.sliver, [(1.0, 200), (3.0, 6)])

    def _repair(self, src, want, **kw):
        out = self.tmp / f"fix_{Path(src).stem}_{want}_{len(kw)}.mp4"
        return _self_clean_repair(Path(src), out, want, **kw)

    # ------------------------------------------------------------------ it repairs the real shape
    def test_a_dark_tail_is_repaired_from_the_beats_own_frames(self):
        got = self._repair(self.tail, 3.5)
        self.assertIsNotNone(got, "a clip legible for 2.5s of 3.5s must be repairable")
        self.assertFalse(_clip_too_dark(Path(got)), "the repaired clip must pass the dark probe")

    def test_the_repaired_clip_fills_the_whole_beat(self):
        """A short clip would desync every later beat — the hold exists to preserve timing."""
        for want in (2.6, 3.5, 4.0):
            got = self._repair(self.tail, want)
            self.assertIsNotNone(got, want)
            self.assertAlmostEqual(probe(Path(got)).get("duration", 0.0), want, delta=0.08,
                                   msg=f"asked {want}s")

    def test_every_frame_of_the_repair_is_legible(self):
        got = self._repair(self.tail, 3.5)
        prof = _clip_luma_profile(Path(got))
        self.assertTrue(prof, "measurable")
        self.assertTrue(all(h >= 50.0 for h in prof), prof)

    def test_the_legible_run_need_not_start_at_zero(self):
        got = self._repair(self.head, 3.5)
        self.assertIsNotNone(got, "the run is at the END of this clip")
        self.assertFalse(_clip_too_dark(Path(got)))

    # ------------------------------------------------------------------ and refuses to paper over
    def test_a_clip_with_nothing_legible_is_refused(self):
        self.assertTrue(_clip_too_dark(self.allblack))
        self.assertIsNone(self._repair(self.allblack, 3.5),
                          "no legible region → the beat is a footage gap, not a freeze")

    def test_a_legible_sliver_does_not_justify_a_beat_long_freeze(self):
        self.assertIsNone(self._repair(self.sliver, 4.0),
                          "1s legible of 4s is a mostly-dark beat")

    def test_the_held_remainder_respects_the_single_hold_cap(self):
        self.assertIsNotNone(self._repair(self.tail, 3.5, max_hold_sec=2.5))
        self.assertIsNone(self._repair(self.tail, 3.5, max_hold_sec=0.2),
                          "a 1.0s hold must be refused under a 0.2s cap")

    def test_an_unreadable_file_is_refused_not_guessed(self):
        missing = self.tmp / "nope.mp4"
        self.assertIsNone(self._repair(missing, 3.0))

    # ------------------------------------------------------------------ the donor rule itself
    def test_the_dark_pass_has_no_cross_scene_donor_left(self):
        src = inspect.getsource(B.build_video)
        self.assertNotIn("_last_clean_d", src,
                         "the previous-beat donor is unrepresentable in the lineage contract")

    def test_a_self_repaired_clip_can_never_become_a_donor(self):
        """It is a held frame, not footage — donating from it would chain freezes."""
        src = inspect.getsource(B.build_video)
        self.assertIn('"_selfclean" in _s', src)

    def test_the_repair_is_wired_into_the_dark_pass(self):
        self.assertIn("_self_clean_repair(", inspect.getsource(B.build_video))


class TestLineageOfTheRepair(unittest.TestCase):
    """The contract's own view: own-beat derivative accepted, neighbour's root refused."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="scl_"))
        cls.clip = cls.tmp / "beat_019_0_selfclean.mp4"
        cls.clip.write_bytes(b"x" * 32)
        cls.source = cls.tmp / "src.mp4"
        cls.source.write_bytes(b"y" * 32)

    def _entry(self, owner, original):
        b = selection_binding(original, "src_a", 96.763, 101.351, {"verdict": "keep"})
        return {
            "kind": "selection_derivative", "owner_beat": owner, "original_beat": original,
            "selected_source_id": "src_a", "actual_source_id": "src_a",
            "selected_window": ["src_a", 96.763, 101.351],
            "actual_window": ["src_a", 96.763, 101.351],
            "selection_source_path": str(self.source),
            "selection_binding": b, "root_binding": b,
            "via": "selection_derivative", "validated": True,
            "final_scene": original, "clip": 0, "file": str(self.clip),
        }

    def test_a_repair_owned_by_its_own_beat_passes(self):
        self.assertEqual(validate_scene_lineage([self._entry(19, 19)]), [])

    def test_the_neighbours_root_is_still_a_violation(self):
        problems = validate_scene_lineage([self._entry(18, 19)])
        self.assertTrue(any("root belongs to beat 18, not this beat 19" in p["reason"]
                            for p in problems), problems)


if __name__ == "__main__":
    unittest.main()


class TestTheBrandingPassHasTheSameRule(unittest.TestCase):
    """The branding sweep carried the identical latent fallback one pass earlier.

    It was written down as latent after job 0ca9dc4c2f and then fired for real in the near-black
    pass on 03768be9ac. Same contract, same verdict: a freeze donated by the previous beat names
    that beat as its root and the final provenance gate refuses the finished video."""

    def test_no_cross_beat_donor_in_the_branding_pass(self):
        src = inspect.getsource(B.build_video)
        self.assertNotIn("_last_clean =", src)
        self.assertNotIn("else _last_clean", src)

    def test_a_branded_beat_with_no_clean_clip_of_its_own_takes_the_placeholder(self):
        """The placeholder path already existed and already release-blocks honestly."""
        src = inspect.getsource(B.build_video)
        self.assertIn("no clean donor of its own", src)
        self.assertIn("_placeholder_clip(proj, seg.index)", src)
