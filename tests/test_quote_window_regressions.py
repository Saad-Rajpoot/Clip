"""Focused regressions for quote-window lineage during matching.

These tests use no network, model, media decoder, or persisted job state.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from vidlore.clipstudio import index as IX
from vidlore.clipstudio import match as M
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import ClipCandidate, Shot


def test_quote_locator_does_not_bridge_a_long_silence():
    # Measured source failure: the subtitle stream contains a stray "I", then 15.75 seconds of
    # silence/non-dialogue before the compact six-word line.  Concatenating the stream gave the
    # impossible 202.62–220.31 span a perfect score and made every four-second selection fail.
    words = [
        (202.62, 202.80, "I"),
        (218.37, 218.55, "did"),
        (218.58, 218.82, "warn"),
        (218.85, 219.02, "you"),
        (219.05, 219.22, "not"),
        (219.25, 219.40, "to"),
        (219.43, 219.70, "trust"),
        (219.74, 220.31, "me"),
    ]

    span = IX.find_quote_span(words, "I did warn you not to trust me", min_ratio=0.78)

    assert span is not None
    assert span[0] == 218.37
    assert span[1] == 220.31
    assert span[2] >= 0.78
    assert span[1] - span[0] < 3.0


def test_quote_locator_retains_only_gapped_verbatim_phrase_as_fallback():
    words = [
        (10.0, 10.2, "chaos"), (10.3, 10.5, "is"), (10.6, 10.8, "a"),
        (16.0, 16.2, "ladder"), (16.3, 16.5, "for"), (16.6, 16.8, "everyone"),
    ]

    span = IX.find_quote_span(words, "chaos is a ladder for everyone", min_ratio=0.78)

    assert span == (10.0, 16.8, 1.0)


def test_quote_locator_requires_a_substantive_token_not_common_prefix_only():
    # Actual beat-65 false positive: slack shortened the candidate to "he was a", giving the
    # authored "He was a monster" a 0.857 score even though ASR says a different noun.
    words = [
        (29.80, 29.94, "He"), (29.95, 30.08, "was"),
        (30.09, 30.20, "a"), (30.21, 30.48, "fighter"),
    ]

    assert IX.find_quote_span(words, "He was a monster", min_ratio=0.78) is None


def _pool_shot(sid, index, start, end, *, quality, transcript="same iconic scene dialogue"):
    shot = Shot(source_id=sid, index=index, start=start, end=end, quality=quality,
                transcript=transcript)
    return M._PoolShot(sid, shot, np.array([1.0, 0.0], dtype=np.float32))


def test_clean_copy_swap_cannot_move_a_quote_locked_winner_to_another_moment():
    cfg = ClipConfig()
    seg = NS(index=85, est_duration=4.0, quote="Chaos isn't a pit. Chaos is a ladder.",
             expected_visual="", text="")
    # The measured winner was the immediately preceding shot.  The moment scorer intentionally
    # treats a shot ending <=1.5s before dialogue as full-strength editorial pre-roll.
    dirty = _pool_shot("dirty_copy", 1, 20.0, 30.0, quality=0.50)
    # Visually duplicate and cleaner, but it is an EARLIER angle: this source's actual quote is at
    # 138–143, outside this 77–82 candidate shot.
    wrong_clean = _pool_shot("clean_copy", 10, 77.0, 82.5, quality=1.0)
    cand = ClipCandidate(segment_index=85, source_id=dirty.sid, shot_index=dirty.shot.index,
                         score=0.9, in_point=22.0, out_point=26.0,
                         signals={"moment_lock": 0.85, "moment_ratio": 1.0})
    best = (0.90, 0.90, dirty, cand)
    scored = [(0.90, 0.0, {}, dirty), (0.90, 0.0, {}, wrong_clean)]

    def located(_proj, sid, _seg, _anchors):
        return (31.8, 35.0, 1.0) if sid == "dirty_copy" else (138.0, 143.0, 1.0)

    with patch.object(M, "locate_beat_moment", side_effect=located):
        got, note = M._clean_copy_swap(
            seg, best, scored,
            src_dirty={"dirty_copy": {"corner": True}, "clean_copy": {}},
            src_height={"dirty_copy": 720, "clean_copy": 1080}, cfg=cfg,
            proj=object(), beat_quote=seg.quote,
        )

    assert got[2].sid == "dirty_copy"
    assert note is None


def test_clean_copy_swap_still_allows_the_same_located_moment():
    cfg = ClipConfig()
    seg = NS(index=85, est_duration=4.0, quote="Chaos isn't a pit. Chaos is a ladder.",
             expected_visual="", text="")
    dirty = _pool_shot("dirty_copy", 1, 20.0, 30.0, quality=0.50)
    clean = _pool_shot("clean_copy", 18, 138.0, 144.0, quality=1.0)
    cand = ClipCandidate(segment_index=85, source_id=dirty.sid, shot_index=dirty.shot.index,
                         score=0.9, in_point=22.0, out_point=26.0,
                         signals={"moment_lock": 0.85, "moment_ratio": 1.0})
    best = (0.90, 0.90, dirty, cand)
    scored = [(0.90, 0.0, {}, dirty), (0.90, 0.0, {}, clean)]

    def located(_proj, sid, _seg, _anchors):
        return (31.8, 35.0, 1.0) if sid == "dirty_copy" else (138.2, 143.2, 1.0)

    with patch.object(M, "locate_beat_moment", side_effect=located):
        got, note = M._clean_copy_swap(
            seg, best, scored,
            src_dirty={"dirty_copy": {"corner": True}, "clean_copy": {}},
            src_height={"dirty_copy": 720, "clean_copy": 1080}, cfg=cfg,
            proj=object(), beat_quote=seg.quote,
        )

    assert got[2].sid == "clean_copy"
    assert got[3].in_point <= 138.2
    assert got[3].out_point >= (138.2 + 143.2) / 2.0
    assert note and "clean-copy swap" in note


def test_clean_copy_swap_cannot_degrade_direct_quote_to_preroll_only():
    cfg = ClipConfig()
    seg = NS(index=85, est_duration=4.0, quote="Chaos is a ladder for everyone.",
             expected_visual="", text="")
    direct = _pool_shot("dirty_copy", 2, 20.0, 30.0, quality=0.50)
    preroll = _pool_shot("clean_copy", 10, 77.0, 82.0, quality=1.0)
    cand = ClipCandidate(segment_index=85, source_id=direct.sid, shot_index=direct.shot.index,
                         score=0.9, in_point=22.0, out_point=27.0,
                         signals={"moment_lock": 1.0, "moment_ratio": 1.0})
    best = (0.90, 0.90, direct, cand)
    scored = [(0.90, 0.0, {}, direct), (0.90, 0.0, {}, preroll)]

    def located(_proj, sid, _seg, _anchors):
        return (24.0, 26.0, 1.0) if sid == "dirty_copy" else (82.7, 85.0, 1.0)

    with patch.object(M, "locate_beat_moment", side_effect=located):
        got, note = M._clean_copy_swap(
            seg, best, scored,
            src_dirty={"dirty_copy": {"corner": True}, "clean_copy": {}},
            src_height={"dirty_copy": 720, "clean_copy": 1080}, cfg=cfg,
            proj=object(), beat_quote=seg.quote,
        )

    assert got[2].sid == "dirty_copy"
    assert note is None


def test_clean_copy_swap_preserves_publication_tolerated_quote_containment():
    cfg = ClipConfig()
    seg = NS(index=85, est_duration=4.0, quote="Chaos is a ladder for everyone.",
             expected_visual="", text="")
    direct = _pool_shot("dirty_copy", 2, 20.0, 30.0, quality=0.50)
    # This source's line spans a long dramatic pause.  Its candidate shot overlaps the phrase, but
    # the normal 4.6s trim cannot contain the 8s ASR span even with the contract's 0.75s tolerance.
    clean = _pool_shot("clean_copy", 18, 77.0, 93.0, quality=1.0)
    cand = ClipCandidate(segment_index=85, source_id=direct.sid, shot_index=direct.shot.index,
                         score=0.9, in_point=22.0, out_point=25.5,
                         signals={"moment_lock": 1.0, "moment_ratio": 1.0})
    best = (0.90, 0.90, direct, cand)
    scored = [(0.90, 0.0, {}, direct), (0.90, 0.0, {}, clean)]

    def located(_proj, sid, _seg, _anchors):
        return (24.0, 26.0, 1.0) if sid == "dirty_copy" else (82.0, 90.0, 1.0)

    with patch.object(M, "locate_beat_moment", side_effect=located):
        got, note = M._clean_copy_swap(
            seg, best, scored,
            src_dirty={"dirty_copy": {"corner": True}, "clean_copy": {}},
            src_height={"dirty_copy": 720, "clean_copy": 1080}, cfg=cfg,
            proj=object(), beat_quote=seg.quote,
        )

    assert got[2].sid == "dirty_copy"
    assert note is None
