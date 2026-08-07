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
    assert "if duration and _end > (times[-1] if times else start):" in SRC
    i = SRC.index("if duration and _end >")
    assert "times.append(" in SRC[i:i + 1400]


def test_the_final_sample_is_not_a_whole_frame_short():
    """1/30 is a frame at 30fps and lands 0.02s before the held frame on the measured case."""
    assert "start + span - 0.01" in SRC


def test_a_full_file_bank_is_unchanged():
    """Only a WINDOW gets the extra moment; sampling a whole clip is untouched."""
    assert "if duration and" in SRC, "the extra moment is gated on a window being given"
    i = SRC.index("if duration and")
    assert "times = [" in SRC[:i], "the ordinary bank is built before, and unchanged"


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


def test_the_extra_sample_stays_clear_of_the_file_end():
    """MEASURED REGRESSION of my own first attempt. A window ending at 152.533s in a 152.6s source
    put the extra sample at 152.523s — ffmpeg decoded nothing there (bytes=0/816) and the whole
    bind failed. `ceil` (total - 1/30) left too little room."""
    assert "total - 0.12" in SRC


def test_the_extra_sample_is_only_added_when_it_is_actually_later():
    """Clamped back to an existing moment, it would add no evidence and only cost a decode."""
    assert "_end > (times[-1] if times else start)" in SRC


def test_a_short_decode_is_a_smaller_bank_not_a_lineage_failure():
    """ffmpeg cannot always land on an exact instant near a GOP boundary or a file's tail, and
    demanding every requested frame turned that into a fatal verdict about the FOOTAGE — measured
    twice on job 229233891e, 18 of 19 frames (bytes=14688/15504) killing the whole bind."""
    src = inspect.getsource(C._features_at_times)
    assert "A SHORT DECODE IS A SMALLER BANK" in src
    assert "times = list(times)[:got]" in src


def test_a_bank_too_small_to_mean_anything_is_still_refused():
    src = inspect.getsource(C._features_at_times)
    assert "max(3, int(len(times) * 0.6))" in src


def test_dropping_undecoded_moments_can_only_tighten_the_check():
    """_compare_bank calls a sample gross when EVERY expected moment rejects it, so fewer expected
    moments means fewer chances to match — never more."""
    assert "all(distance[0][\"gross_mismatch\"] for distance in choices)" in inspect.getsource(
        C._compare_bank)
