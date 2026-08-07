"""Two correct rules, in direct contradiction — and the whole video was paying for it.

A breakout's caption may only show words the ASR is confident about: "a missing caption line beats
a wrong one", because a mis-transcribed line burned on screen reads as somebody else's subtitle.
And a breakout may only air if it is captioned through its final spoken word.

When the ASR garbles the tail, the first rule GUARANTEES the second one fails. Measured on job
229233891e scene 110: line one captioned perfectly (align 1.00), line two dropped at ASR confidence
0.09 against the 0.45 floor ('that men shall tremble. from both'), coverage 8 of 14 words = 57%,
and the render died — 145 other beats thrown away for one garbled half-line.

The contradiction is only irreconcilable at RENDER scope. At BREAKOUT scope it resolves itself: a
breakout that cannot be fully captioned does not get captioned. Neither rule is bent — no partial
caption is burned, no unverified line is shown — and the video survives.

What deliberately does NOT change: the breakout's `_caps` metadata stays, because that is what
keeps the main narration caption off the breakout's real-audio window. Only the burn is skipped.
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import build as B


SRC = inspect.getsource(B)
I = SRC.index("TWO CORRECT RULES, IN DIRECT CONTRADICTION")
BLOCK = SRC[I:I + 2200]


def test_a_partial_caption_breakout_no_longer_kills_the_render():
    assert "NonRetryableBuildError" not in BLOCK
    assert "_breakout_caps_enabled = False" in BLOCK


def test_no_partial_caption_is_ever_burned():
    """The rule is kept, not bent — the pass is dropped whole."""
    assert "_breakout_caption_burn_ok = False" in BLOCK
    assert "_caps_pre = []" in BLOCK


def test_the_burn_is_skipped_when_the_preflight_failed():
    i = SRC.index("if _caps and not _breakout_caption_burn_ok:")
    assert "_caps = []" in SRC[i:i + 500]


def test_the_suppression_metadata_survives():
    """Without it the main narration caption would print over the breakout's real audio."""
    assert "_breakout_caps" in SRC
    assert "stays suppressed over its window" in SRC


def test_the_flag_defaults_to_allowing_the_burn():
    assert "_breakout_caption_burn_ok = True" in SRC
    assert SRC.index("_breakout_caption_burn_ok = True") < SRC.index(
        "if _caps and not _breakout_caption_burn_ok:")


def test_the_drop_is_reported_loudly():
    assert "log(" in BLOCK
    assert "DROPPING the breakout caption pass" in BLOCK


def test_the_measured_case_is_written_down():
    assert "8/14" in BLOCK and "0.09" in BLOCK
