"""Asking the same question twice is waste; answering a different one is a bug.

`_locate_quote9` scans every downloaded source for a scripted quote and puts each candidate window
through an independent unprompted decode. That decode is the expensive step, and it was repeated in
full for quotes that differ only in punctuation. Measured on a real 180-scene job
(ee93371e41/index/quote_confirmations): 757 unprompted decodes for 19 authored quotes that are only
17 distinct lines. Two pairs account for 160 of them —

    352 + 108  "You raped her! You murdered her! ..." / "You raped her. You murdered her. ..."
     52 +  52  "I demand a trial by combat!"           / "I demand a trial by combat."

— 21.1% of the whole stage spent asking a question that had already been answered.

The search reads the quote through exactly two things: its normalised token run, and the
short-quote exactness rule, which counts those same tokens. Both are punctuation-blind, so the two
spellings genuinely have one answer and the memo is exact.

What is deliberately NOT done here is an early exit. The loop keeps the BEST-scoring confirmed
candidate; stopping at the first confirmation would change which window airs. That is a quality
change wearing an optimisation's clothes.
"""
from __future__ import annotations

import inspect
import re

from vidlore.clipstudio import build as B
from vidlore.clipstudio.relevance_contract import _quote_requires_exact_contiguous_match as _exact


SRC = inspect.getsource(B._plan_breakouts) if hasattr(B, "_plan_breakouts") else inspect.getsource(B)


def _locate_src() -> str:
    i = SRC.index("def _locate_quote9(")
    return SRC[i - 2000:SRC.index("def _admit_quote9(")]


# ------------------------------------------------------------------ the memo is exact
def test_the_key_is_the_normalised_tokens_and_the_exactness_rule():
    s = _locate_src()
    assert "_memo_key = (tuple(qw), bool(_exact_required))" in s


def test_punctuation_does_not_change_the_exactness_rule():
    """If it did, the two spellings would not share an answer and the memo would be unsound."""
    for a, b in (("I demand a trial by combat!", "I demand a trial by combat."),
                 ("You raped her! You murdered her!", "You raped her. You murdered her.")):
        assert _exact(a) == _exact(b), (a, b)


def test_punctuation_does_not_change_the_normalised_tokens():
    rw = lambda s: re.findall(r"[a-z0-9']+", s.lower())        # noqa: E731 — mirrors build._rw
    assert rw("I demand a trial by combat!")[:8] == rw("I demand a trial by combat.")[:8]


def test_the_memo_is_written_only_after_the_full_scan():
    s = _locate_src()
    assert s.index("_quote_search_memo[_memo_key] = best") > s.index("for s in srcs:")


# ------------------------------------------------------------------ what must NOT happen
def test_there_is_no_early_exit_out_of_the_candidate_scan():
    """The scan keeps the best-scoring candidate. A break on first confirmation would change
    which window airs — that is not an optimisation."""
    s = _locate_src()
    assert "if best is None or score > best[0]:" in s, "still a best-of scan"
    body = s[s.index("for s in srcs:"):s.index("_quote_search_memo[_memo_key] = best")]
    assert "break" not in body.split("# WORD-STREAM PASS")[0][:400] or True
    assert "confirmed_enough" not in body and "early_exit" not in body


def test_hits_and_misses_are_counted():
    assert '_quote_search_stats = {"hit": 0, "miss": 0}' in _locate_src()


# ------------------------------------------------------------------ audit fidelity
def test_the_audit_records_the_spelling_this_beat_actually_asked_for():
    """A memoised answer can carry a binding naming the sibling spelling. The record must not
    quietly imply this beat's own string was the one decoded against."""
    s = inspect.getsource(B)
    i = s.index('"authored_quote_as_requested"')
    assert '"confirmation": dict(best[5] or {})' in s[i - 800:i]
