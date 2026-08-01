"""A player badge on the SIDE border, at mid-height, that no corner detector could ever see.

MEASURED on portal job 409e284b60. An orange rounded square carrying a white 'm' and a running
timer — '53:5…' at one beat, '54:0…' at another, '54:4…' at a third — is burned into the right
border of four different re-uploads of the same screen recording, and aired on four beats. Two
independent reasons the existing pixel gate could not catch it, both geometric:

  * it sits at y 46-56% of frame height, and the corner patches are the outer 18% x 12% boxes, so
    not one pixel of the badge falls inside any of them;
  * its digits change every second, so a test that asks a whole patch to be static fails anyway.

`_source_edge_logo` keeps the corner detector's statistic — an overlay's edges land on the same
pixels while a scene's edges move — and changes only the geometry: the outer 9% of width over the
FULL height, plus one extra condition that does all the discriminating, COMPACTNESS. A pillarbox
seam is a perfect static edge too; what a badge has and a seam does not is a bounded footprint.

CALIBRATED on that render's 69 indexed sources with >=8 keyframes. It fires on 5, and every one
carries a real burned-in overlay, verified by eye:

    benjen_stark_saves_jon_7d24f0b7   r   the player badge
    uncle_benjen_saves_jon_f8952110   r   the player badge
    game_of_thrones_benjen_2360f6e5   l   'SPHINX TV'
    jon_snow_and_benjen_st_3a640f30   l   'FAVORITE FLASHBACKS FRENZY'
    game_of_thrones_s08e03_676094b2   r   a bottom-edge channel mark

0 false positives. The other two badge sources show it on only 25% of keyframes — the overlay
auto-hides — so no source-level detector can reach them; that is a per-shot question.

    python3 -m pytest tests/test_edge_badge_gate.py -q

No network, no LLM. Synthesised frames only.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.clipstudio.match import _source_edge_logo, _EDGE_LOGO_CACHE     # noqa: E402

W, H = 640, 360


class Shot:
    def __init__(self, p):
        self.keyframe_path = str(p)
        self.subs_flag = -1
        self.corner_masks = None
        self.phash = None


def _write(frames, tmp):
    out = []
    for i, a in enumerate(frames):
        p = Path(tmp) / f"kf_{i:03d}.jpg"
        Image.fromarray(a.astype("uint8")).save(p, quality=95)
        out.append(Shot(p))
    return out


def scene(rng, n=16):
    """Moving footage: broad smooth regions with a subject that moves between frames — real
    frames are mostly low-frequency, which is why an overlay's hard edges stand out at all.
    Uniform noise would put a strong edge on every pixel and hide the very thing under test."""
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for _ in range(n):
        a = 60 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
        cx, cy = rng.integers(80, W - 80), rng.integers(60, H - 60)
        a += 120 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 55.0 ** 2)))
        out.append(np.clip(a, 0, 255))
    return out


def stamp(frames, x0, x1, y0, y1, changing_digits=False):
    """Burn a bright bounded rectangle at the same pixels of every frame."""
    for i, a in enumerate(frames):
        a[y0:y1, x0:x1] = 20.0
        a[y0 + 3:y1 - 3, x0 + 3:x1 - 3] = 240.0
        if changing_digits:                       # the running timer: bottom third re-rolls
            yd = y1 - (y1 - y0) // 3
            a[yd:y1 - 2, x0 + 3:x1 - 3] = 40.0 + (i * 37 % 180)
    return frames


class TestItFindsTheBadge(unittest.TestCase):
    def setUp(self):
        _EDGE_LOGO_CACHE.clear()
        self.rng = np.random.default_rng(7)

    def test_a_mid_height_right_border_badge_is_found(self):
        with tempfile.TemporaryDirectory() as t:
            f = stamp(scene(self.rng), W - 34, W - 2, 168, 196)
            self.assertEqual(_source_edge_logo(_write(f, t)), "r")

    def test_it_survives_a_running_timer_inside_the_badge(self):
        """The real one re-rolls its digits every second; only the frame stays put."""
        with tempfile.TemporaryDirectory() as t:
            f = stamp(scene(self.rng), W - 34, W - 2, 168, 200, changing_digits=True)
            self.assertEqual(_source_edge_logo(_write(f, t)), "r")

    def test_a_left_border_badge_is_found(self):
        with tempfile.TemporaryDirectory() as t:
            f = stamp(scene(self.rng), 2, 34, 300, 330)
            self.assertEqual(_source_edge_logo(_write(f, t)), "l")


class TestItDoesNotFireOnLookalikes(unittest.TestCase):
    """0 false positives on 69 real sources is the bar; these are the shapes that threaten it."""

    def setUp(self):
        _EDGE_LOGO_CACHE.clear()
        self.rng = np.random.default_rng(11)

    def test_a_full_height_pillarbox_seam_is_not_a_badge(self):
        """A perfect static edge, present on every frame — separated only by compactness."""
        with tempfile.TemporaryDirectory() as t:
            f = scene(self.rng)
            for a in f:
                a[:, :18] = 0.0
                a[:, 18:20] = 255.0
            self.assertEqual(_source_edge_logo(_write(f, t)), "")

    def test_clean_moving_footage_is_not_a_badge(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(_source_edge_logo(_write(scene(self.rng), t)), "")

    def test_a_static_card_source_is_not_a_badge(self):
        """Identical frames make every edge persist; that is _source_is_static's call."""
        with tempfile.TemporaryDirectory() as t:
            one = scene(self.rng, 1)[0]
            self.assertEqual(_source_edge_logo(_write([one.copy() for _ in range(16)], t)), "")

    def test_a_badge_on_only_a_quarter_of_frames_is_left_alone(self):
        """The two sources whose overlay auto-hides. A source-level crop would punch in on the
        whole source to fix a quarter of it — that is a per-shot decision, not this one."""
        with tempfile.TemporaryDirectory() as t:
            f = scene(self.rng)
            stamp(f[:4], W - 34, W - 2, 168, 196)
            self.assertEqual(_source_edge_logo(_write(f, t)), "")

    def test_too_few_keyframes_is_no_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            f = stamp(scene(self.rng, 5), W - 34, W - 2, 168, 196)
            self.assertEqual(_source_edge_logo(_write(f, t)), "")


class TestItIsCachedAndWired(unittest.TestCase):
    def test_the_result_is_memoised_per_source(self):
        _EDGE_LOGO_CACHE.clear()
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as t:
            shots = _write(stamp(scene(rng), W - 34, W - 2, 168, 196), t)
            self.assertEqual(_source_edge_logo(shots), "r")
            for s in shots:                                   # unreadable now
                Path(s.keyframe_path).unlink()
            self.assertEqual(_source_edge_logo(shots), "r")   # served from cache

    def test_build_consults_it_only_after_the_corner_detector(self):
        src = (ROOT / "vidlore" / "clipstudio" / "build.py").read_text()
        seg = src.split("def _watermarked_source_corners")[1].split("\ndef ")[0]
        self.assertLess(seg.index("_source_corner_logo(shots)"), seg.index("_source_edge_logo(shots)"))
        self.assertIn("continue", seg.split("_source_corner_logo(shots)")[1]
                      .split("_source_edge_logo")[0])

    def test_a_side_maps_to_a_crop_that_drops_that_side(self):
        src = (ROOT / "vidlore" / "clipstudio" / "build.py").read_text()
        self.assertIn('_EDGE2CORNER = {"l": "bl", "r": "br"}', src)
        #  the crop filter keeps x=0 (the LEFT of frame) for br/tr — i.e. it drops the right side
        seg = src.split("def _watermark_crop_filter")[1].split("\ndef ")[0]
        self.assertIn('x = "0" if corner in ("br", "tr")', seg)

    def test_it_has_its_own_kill_switch(self):
        src = (ROOT / "vidlore" / "clipstudio" / "build.py").read_text()
        self.assertIn('VIDLORE_CLIPSTUDIO_EDGE_LOGO_GATE', src)
        self.assertIn("if edge_on:", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
