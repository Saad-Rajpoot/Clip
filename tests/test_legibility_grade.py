"""Dark canon should be LIT, not rejected and not replaced.

Measured on job 6a26707939, over the 501 frames the renderer would actually have shown: median
beat mean-luma 31.2, a quarter of all beats under 20, p90 luma 63.7 in the median beat. A
frame-by-frame vision audit of the same render flagged 44 beats too_dark and put 22 of them in the
<=4/10 bucket ("an essentially empty dark rectangle with a speck in it").

Two responses were already tried and both are worse than grading:

  * REJECT dark windows. Measured and rejected before this: the pool median is what it is, so a
    luma floor evicts correct, canonical night scenes and the beat falls back to generic filler,
    which scores 5.82 against exact_scene's 5.85 — a relevance loss bought with a legibility gain.
  * REPLACE near-black cut clips with a freeze from a neighbouring beat. That is what the pipeline
    does today, and it puts the WRONG CONTENT on screen to fix a brightness problem.

Grading is presentation-only: it runs after the window is chosen, changes no ranking, no verdict
and no selection, and cannot move a beat onto different footage. Because build's near-black sweep
probes the CUT clip, a window that becomes legible here also stops being freeze-replaced — a
relevance gain with no relevance risk.

Trigger and instrument both matter. The trigger is build's measured legibility score
max(YAVG, spread), because brightness alone rejects candle-lit scenes wholesale while genuine dead
frames fail BOTH mean and contrast. The instrument is gamma, because its slope is steepest in the
shadows — it opens crushed blacks instead of flattening the picture, and it cannot clip highlights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vidlore.clipstudio import cut as C


# --------------------------------------------------------------- when the grade must NOT fire
def test_a_normally_lit_window_is_left_alone():
    assert C.legibility_gamma(120.0, 140.0) == 1.0


def test_a_dim_but_high_contrast_scene_is_left_alone():
    """The Olenna-Jaime confession probes YAVG 28-45 and is perfectly readable on screen: a lit
    face against shadow spreads the histogram. Contrast is what legibility means, so a wide spread
    exempts the window however low its mean."""
    assert C.legibility_gamma(30.0, 95.0) == 1.0


def test_the_floor_is_the_score_not_the_mean():
    """max(YAVG, spread) — either one clearing the floor is enough."""
    assert C.legibility_gamma(C._GRADE_FLOOR + 1, 5.0) == 1.0
    assert C.legibility_gamma(5.0, C._GRADE_FLOOR + 1) == 1.0


# --------------------------------------------------------------- when it must fire
@pytest.mark.parametrize("yavg,spread", [
    (24.8, 21.9),   # beat 162, Melisandre in Stannis's tent
    (26.7, 12.5),   # beat 21, Stannis by the bookcase — the lowest contrast in the render
    (13.0, 15.0),   # the near-black end of the distribution
])
def test_a_genuinely_dead_window_is_lifted(yavg, spread):
    g = C.legibility_gamma(yavg, spread)
    assert g > 1.0
    assert g <= C._GRADE_MAX_GAMMA


def test_the_lift_actually_reaches_the_target():
    """Arithmetic, not vibes: applying the returned gamma to the measured mean must land near the
    readable target. v**(1/g) == t/255 by construction."""
    for yavg in (13.0, 20.0, 27.0, 35.0):
        g = C.legibility_gamma(yavg, 10.0)
        if g <= 1.0:
            continue
        out = 255.0 * (yavg / 255.0) ** (1.0 / g)
        assert abs(out - C._GRADE_TARGET) < 1.0, f"{yavg} -> {out}"


def test_the_gamma_is_capped_so_night_never_becomes_grey_noise():
    """A 4/255 frame is dead footage, not a grading problem — lift it as far as the cap and no
    further, and let the near-black sweep deal with what is still unusable."""
    assert C.legibility_gamma(2.0, 3.0) == C._GRADE_MAX_GAMMA


def test_darker_windows_get_more_lift_than_lighter_ones():
    assert C.legibility_gamma(14.0, 10.0) > C.legibility_gamma(30.0, 10.0)


# --------------------------------------------------------------- wiring
def test_the_filter_is_off_when_the_kill_switch_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_LEGIBILITY_GRADE", "0")
    vf, note = C.legibility_filter(tmp_path / "nope.mp4", 0.0, 2.0)
    assert vf == "" and note == ""


def test_an_unprobeable_window_is_never_treated_as_dark(monkeypatch, tmp_path):
    """A failed probe is 'unknown'. Guessing 'dark' would grade footage nobody measured."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_LEGIBILITY_GRADE", raising=False)
    monkeypatch.setattr(C, "_luma_stats", lambda *a, **k: None)
    vf, note = C.legibility_filter(tmp_path / "nope.mp4", 0.0, 2.0)
    assert vf == ""


def test_the_filter_names_the_gamma_and_what_it_measured(monkeypatch, tmp_path):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_LEGIBILITY_GRADE", raising=False)
    monkeypatch.setattr(C, "_luma_stats", lambda *a, **k: (24.8, 21.9))
    vf, note = C.legibility_filter(tmp_path / "x.mp4", 0.0, 2.0)
    assert vf.startswith("eq=gamma=")
    assert "24.8" in note and "21.9" in note


def test_the_cut_applies_the_filter_and_records_it():
    """It must reach ffmpeg AND the selection, so an audit can tell a graded beat from a shot one."""
    import inspect
    src = inspect.getsource(C.cut_selection)
    assert "legibility_filter" in src
    # the cut now always carries a -vf chain (the half-open boundary select leads it), so the grade
    # is appended to that chain rather than being the whole filter argument
    assert "_chain.append(_vf)" in src, "the grade must reach the cut command"
    assert '"-vf", ",".join(_chain)' in src
    assert "sel.legibility_grade = _note" in src, "a graded beat must say so in the project file"


def test_the_grade_cannot_change_which_footage_airs():
    """The standing rule for this change: presentation only. cut_selection may not touch the
    selection's source, shot or in/out points."""
    import inspect
    src = inspect.getsource(C.cut_selection)
    for forbidden in ("sel.source_id =", "sel.shot_index =", "sel.in_point =", "sel.out_point ="):
        assert forbidden not in src, f"cut must not reassign {forbidden.split()[0]}"


def test_breakout_extraction_applies_the_same_grade_and_reports_it(monkeypatch, tmp_path):
    """Breakouts bypass cut_selection, so _extract_breakout itself must grade its exact window."""
    from types import SimpleNamespace as NS
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import ingest as I

    source = tmp_path / "source.mp4"
    video = tmp_path / "breakout.mp4"
    audio = tmp_path / "breakout.wav"
    source.write_bytes(b"source")
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(list(cmd))
        target = str(cmd[-1])
        if target == str(video):
            video.write_bytes(b"video")
        elif target == str(audio):
            audio.write_bytes(b"audio")
        return NS(returncode=0, stdout=b"", stderr=b"")

    def fake_probe(path):
        path = Path(path)
        if path == source:
            return {"duration": 20.0, "width": 1920, "height": 1080}
        return {"duration": 3.0, "width": 1920, "height": 1080}

    note = "legibility grade gamma=1.259 (YAVG29.5, spread19.9)"
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr(B, "_dialogue_aware_dur", lambda *a, **k: (3.0, "spoken line"))
    monkeypatch.setattr(I, "probe", fake_probe)
    monkeypatch.setattr(C, "legibility_filter", lambda *a, **k: ("eq=gamma=1.259", note))

    quality = {}
    got = B._extract_breakout(str(source), 2.0, 4.0, video, audio, src_w=1920,
                              quality_meta=quality)

    assert got == 3.0
    video_cmd = next(c for c in commands if c[-1] == str(video))
    vf = video_cmd[video_cmd.index("-vf") + 1]
    assert "eq=gamma=1.259" in vf
    assert vf.index("eq=gamma=1.259") < vf.index("scale=1920:1080"), \
        "the shadow lift must run on the native window before normalization/upscale"
    assert quality["legibility_grade"] == note, \
        "the accepted breakout audit must be able to explain its presentation grade"
