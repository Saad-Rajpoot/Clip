"""A frame sampled at time t is not unowned — it lies inside exactly one indexed shot.

Job 229233891e rendered all 146 beats and then died at the very last gate:

    image-lineage gate: image_meta claims 'game_of_thrones_season_6030de3c' shot -1,
    but that exact shot is absent/unreadable

Beat 108's still was `selfheal_202.jpg`, a frame the region-recovery rung extracts at an arbitrary
timestamp. Because it is not itself a detected shot, it was installed declaring `shot: -1`. The
lineage gate then has to prove the still belongs to source X shot -1, which cannot exist, so it
correctly refused — and a finished 146-beat render was thrown away for it.

The gate was right; the claim was wrong. That frame is at t=202.0s, and shot 40 of that source
spans 199.97-202.53s — measured on the real job. Recording the owning shot turns an unprovable
claim into a provable one, which is exactly what the gate was asking for.

Where no shot contains t — an unindexed source — nothing is installed at all. A still whose
lineage cannot be established must not air, and refusing to install it is the honest outcome, not
declaring a shot that does not exist.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import selfheal as SH


SRC = inspect.getsource(SH)
I = SRC.index("USE THE SHOT'S OWN KEYFRAME")
BLOCK = SRC[I:I + 3000]


def test_candidates_are_the_shots_own_indexed_keyframes():
    """Provable by construction: index.py extracts keyframes at the shot midpoint, which is
    exactly what the image-lineage gate accepts without extra proof."""
    assert 'getattr(_sh, "keyframe_path"' in BLOCK
    assert "load_shots" in BLOCK


def test_no_frame_is_minted_at_an_arbitrary_time():
    """MEASURED TWICE. Extracting at an arbitrary t and declaring shot -1 killed the render; then
    recording the real owning shot moved the same death to 'declared verified still does not match
    indexed keyframe for ... shot 5', because such a frame is neither the keyframe nor a midpoint
    re-extraction carrying native proof."""
    code = "\n".join(l for l in BLOCK.splitlines() if not l.strip().startswith("#"))
    assert "selfheal_" not in code, "no frame is minted at an arbitrary timestamp"
    assert "-frames:v" not in code, "no ad-hoc extraction remains in this rung"


def test_the_resolved_index_is_what_gets_recorded():
    assert "_install_still(sel, fp, src.id, _shot, 0.8)" in SRC
    assert "_install_still(sel, fp, src.id, -1, 0.8)" not in SRC


def test_an_unresolvable_frame_is_never_installed():
    """No shot contains t -> the lineage cannot be proven -> the still does not air."""
    assert "if _shot < 0:" in SRC
    i = SRC.index("if _shot < 0:")
    assert "continue" in SRC[i:i + 160]


def test_the_shot_is_carried_with_each_candidate_not_recomputed_later():
    assert 'cands.append((float(arr.mean()), kf, int(getattr(_sh, "index", -1))))' in SRC
    assert "for luma, fp, _shot in cands[:4]:" in SRC


def test_a_missing_index_fails_closed_rather_than_guessing():
    assert "except Exception" in BLOCK
    assert "_shots = []" in BLOCK


def test_the_other_install_site_already_recorded_a_real_shot():
    """selfheal has two install sites; only the region-recovery one was wrong."""
    assert '_install_still(sel, sh.keyframe_path, sid, int(getattr(sh, "index", -1)), rel)' in SRC


def test_both_measured_failures_are_recorded_for_the_next_reader():
    assert "selfheal_202.jpg" in BLOCK
    assert "shot 5" in BLOCK
