"""A held tail repeats the window's last frame — so the bank must contain it.

Job 229233891e died at the scene-lineage canary with 4 violations: "planned derivative visually
mismatches its selected source window". It was a false positive, and proving that took measuring
rather than guessing:

  * the CUT clip (seg_NNN.mp4) passed against its window; only the OWNED derivative failed
  * that derivative is far longer than its window — 1.63s of footage filling a 5.50s beat — so it
    plays the window and then FREEZES on its last frame
  * sampled uniformly, 10 of its 14 samples were byte-identical: dhash 0.4609, colour 0.1368, over
    and over, the held image
  * the bank's fractions stop at 0.88 of the span, so its last sample was 189.54s
  * a fine scan of the source put the held frame at 189.76s — INSIDE the window (188.13-189.77),
    matching at dhash 0.0117 / colour 0.0041

The hold was correct. The bank simply could not see the frame it repeats, so every copy of it read
as foreign and one unmatched image was counted ten times.

The fix ADDS the window's final moment. It removes nothing, so nothing previously caught can slip
past — and the control says so: foreign rejection is 4 of 7 before and after.

The other repair — collapsing the repeated samples — was tried first and DISCARDED, because the
same control measured what it cost: it fixed the false positive and halved foreign detection
(4/7 -> 2/7), since that sensitivity came from the repeats themselves.
"""
from __future__ import annotations

import inspect

from vidlore import scene_lineage_canary as C


SRC = inspect.getsource(C._uniform_times)


def test_a_window_bank_includes_its_final_moment():
    assert "if duration and _end < ceil:" in SRC
    i = SRC.index("if duration and _end < ceil:")
    assert "times.append(" in SRC[i:i + 1400]


def test_the_final_sample_is_not_a_whole_frame_short():
    """1/30 is a frame at 30fps and lands 0.02s before the held frame on the measured case."""
    assert "start + span - 0.01" in SRC


def test_a_full_file_bank_is_unchanged():
    """Only a WINDOW gets the extra moment; sampling a whole clip is untouched."""
    i = SRC.index("if duration and _end < ceil:")
    guard = SRC[:i]
    assert "times = [" in guard


def test_nothing_is_removed_from_the_bank():
    """The repair may only add evidence — a removed sample is a weakened gate."""
    assert "fractions" in SRC
    for shrink in ("del ", ".pop(", "[:-1]", "[1:]"):
        assert shrink not in SRC, shrink


def test_the_collapse_alternative_is_recorded_as_rejected():
    """So nobody re-tries it: the control measured it halving foreign detection."""
    whole = inspect.getsource(C)
    assert "collapsing the repeated samples" in whole
    assert "4/7 -> 2/7" in whole


def test_the_measured_case_is_written_down():
    assert "189.76" in SRC and "189.54" in SRC


def test_times_stay_sorted_and_unique():
    assert "sorted(set(" in SRC
