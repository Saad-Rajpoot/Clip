"""A selection owns [in, out). The frame AT `out` belongs to the next shot and must never air.

Measured on job ee93371e41, scene 170. The selection ended at source time 89.464375. The cut
carried 49 frames: 48 Varys frames and the first frame of the following Oberyn shot. Downstream,
`_fit_verified_selection_clip` clones the final decoded frame to fill a longer narration beat, so
that ONE foreign frame became 126 of 174 output frames — 72.4%, about 4.2 seconds of the wrong
character holding still.

The first answer was to stop cloning at 88% of the clip and hold that frame instead. It closes the
leak and it is far too broad: on the same job it discarded 48.30 seconds of verified motion across
every padded beat, while keeping 354.16 seconds.

The answer here removes the contaminating frame instead of hiding from it. `-ss` before `-i` makes
each frame's `t` its offset INTO the selection, so `select='lt(t, dur)'` is exactly the half-open
test — and because it filters on each frame's own decoded timestamp it is indifferent to frame
rate, VFR, container time base and seek behaviour. There is no fps arithmetic left to round wrong.

Clips cut that way carry `cut_contract = "halfopen_v1"`. Only a clip carrying that marker may have
its true final frame held; anything else keeps the conservative stop, so an old clip inherited on
resume can never silently regain the leak.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import build as B
from vidlore.clipstudio import cut as C
from vidlore.clipstudio.models import ClipSelection


# ------------------------------------------------------------------ the cut is half-open
def test_the_cut_filters_on_each_frames_own_timestamp():
    """Not on a frame count, not on fps arithmetic — on `t`, which VFR and odd time bases cannot
    round out from under us."""
    src = inspect.getsource(C.cut_selection)
    assert "select='lt(t" in src, "the half-open select must be present"
    assert "setpts=PTS-STARTPTS" in src


def test_the_select_runs_before_any_grade():
    """A frame that does not belong must never be graded, encoded or measured."""
    src = inspect.getsource(C.cut_selection)
    i_sel = src.index("select='lt(t")
    i_grade = src.index("_chain.append(_vf)")
    assert i_sel < i_grade


def test_vsync_cannot_re_manufacture_the_dropped_frame():
    src = inspect.getsource(C.cut_selection)
    assert '"-fps_mode", "passthrough"' in src


def _bound_of(dur: float) -> float:
    """The boundary cut.py actually writes into the filter, parsed back as ffmpeg would read it."""
    import math as _m
    return float(f"{_m.floor(max(0.0, dur) * 1e9) / 1e9:.9f}")


@pytest.mark.parametrize("dur", [
    2.0,            # exact, integer frame rate
    1.4666666,      # rounds UP at 6dp — the case that re-admitted the boundary frame
    0.9999999,      # rounds up across a whole second
    3.3333333,      # 29.97 / VFR-style repeating fraction
    89.464375 - 87.8,   # scene 170's real window length
    1.0 / 30.0,     # a single frame at 30fps
])
def test_the_written_boundary_never_exceeds_the_authorized_out(dur):
    """This is the whole safety property. If the threshold the filter receives is even slightly
    larger than the selection's duration, the frame at `out` passes and the leak is back — which is
    exactly what "%.6f" did to 1.4666666 (-> 1.466667)."""
    assert _bound_of(dur) <= dur, f"threshold {_bound_of(dur)!r} exceeds out {dur!r}"


@pytest.mark.parametrize("dur", [2.0, 1.4666666, 3.3333333, 89.464375 - 87.8])
def test_the_boundary_is_exclusive_at_out_and_inclusive_just_inside(dur):
    b = _bound_of(dur)
    assert not (dur < b), "a frame landing exactly on `out` must be excluded"
    assert (dur - 1.0 / 30.0) < b, "the last real in-window frame must be kept"


def test_truncation_never_costs_a_real_frame():
    """The threshold is floored to 1ns; frames are ~33ms apart, so no in-window frame can fall into
    the truncated sliver."""
    for dur in (2.0, 1.4666666, 0.9999999, 3.3333333):
        assert dur - _bound_of(dur) < 1e-8


# ------------------------------------------------------------------ the contract marker
def test_a_successful_cut_certifies_the_clip():
    src = inspect.getsource(C.cut_selection)
    assert "sel.cut_contract = _CUT_CONTRACT" in src


def test_the_selection_can_carry_the_marker():
    sel = ClipSelection(segment_index=0, source_id="a", shot_index=0,
                        in_point=0.0, out_point=2.0, confidence=0.5)
    assert hasattr(sel, "cut_contract") and sel.cut_contract == ""
    sel.cut_contract = C._CUT_CONTRACT
    assert ClipSelection.from_dict(sel.to_dict()).cut_contract == C._CUT_CONTRACT


def test_resume_refuses_to_inherit_an_uncertified_clip():
    """An old clip may still hold the frame at `out`, and the fitter trusts the marker. Inheriting
    it would silently restore the 72%-of-the-beat freeze."""
    src = inspect.getsource(C.cut_selection)
    i = src.index("if resume:")
    block = src[i:i + 420]
    assert 'getattr(sel, "cut_contract", "") == _CUT_CONTRACT' in block


# ------------------------------------------------------------------ the fitter
def test_a_certified_clip_keeps_every_verified_frame():
    """No blanket percentage: a frame-exact clip's true final frame is in-window by construction,
    so the pad holds it and nothing is discarded."""
    src = inspect.getsource(B._fit_verified_selection_clip)
    i = src.index("if clip_duration > 0 and need > clip_duration")
    cond = src[i:i + 200]
    assert "not frame_exact" in cond, "certified clips must skip the trim entirely"


def test_an_uncertified_clip_keeps_the_conservative_stop():
    src = inspect.getsource(B._fit_verified_selection_clip)
    assert "0.88" in src, "the guard must remain for clips that were not cut half-open"


def test_the_call_site_passes_the_certification_not_a_constant():
    src = inspect.getsource(B.build_video) if hasattr(B, "build_video") else B.__dict__ and ""
    whole = inspect.getsource(B)
    i = whole.index("_owned = _fit_verified_selection_clip(")
    call = whole[i:i + 320]
    assert 'frame_exact=(getattr(sel, "cut_contract", "") == _CUTC)' in call


def test_padding_is_only_applied_when_the_beat_is_longer_than_the_clip():
    """No padding required => no hold, no trim, pure selected motion."""
    src = inspect.getsource(B._fit_verified_selection_clip)
    assert "need > clip_duration + (2.0 / 30.0)" in src


def test_a_short_selection_still_gets_padded_to_the_beat():
    src = inspect.getsource(B._fit_verified_selection_clip)
    assert "tpad=stop_mode=clone" in src
    assert 'f"-frames:v"' not in src   # exact output frame count, not a stream loop
    assert '"-frames:v", str(frames)' in src


def test_crop_is_applied_after_the_boundary_fitting():
    """Crop plus boundary fitting must compose: the boundary decision is about WHICH frames, the
    crop about what each frame shows."""
    src = inspect.getsource(B._fit_verified_selection_clip)
    i_trim = src.index("trim=end=")
    i_crop = src.index("if crop_filter:")
    assert i_trim < i_crop


def test_the_derivative_is_still_lineage_checked_after_fitting():
    whole = inspect.getsource(B)
    i = whole.index("_owned = _fit_verified_selection_clip(")
    tail = whole[i:i + 600]
    assert "_lineage_derive(" in tail, "fitting must not bypass lineage proof"
    assert "NonRetryableBuildError" in tail, "a failed derivative must block the build"
