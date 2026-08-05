"""A window must not buy its score by stopping short of the quote.

`find_quote_span` scores a candidate window as ``2*(hits+fuzzy)/(L+n)``. For a fixed number of
matched tokens that rises as L falls, so ENDING ONE WORD EARLY IS WORTH FREE RATIO — and the search
sweeps L across a slack range and keeps the best, which systematically rewards truncation.

The case that exposed it, from a real short render. The beat's authored quote is
"I want you to be my cupbearer."; the audio actually says "Good. I want you to be happy." — three
independent Whisper decodes agree (the production base model plus small.en and medium.en).

    L=5   "i want you to be"          2*5/(5+7) = 0.833   clears a 0.78 floor
    L=6   "i want you to be happy"    2*5/(6+7) = 0.769   does not

The quote's last two tokens are simply not in that audio. Truncation bought the match, and the
pipeline then reported the line "located" in a source about an entirely different scene.

I first read this as an impossible speech rate — 7 words in 0.80s — and that was WRONG. The span
holds 5 ASR words in 0.90s, about 5.6 words/second, which is ordinary speech; the phantom rate came
from counting quote tokens the window never contained. A rate floor would have been a number picked
to fit one case. Measured over 3,149 located spans from 780 probes across 273 real ASR streams, no
single statistic separates genuine spans from spurious ones: rate, coverage, span duration and
phrase ratio all overlap heavily.

So the fix introduces NO new threshold. Both clauses re-use the caller's own `min_ratio`:

  * a window that leaves quote tokens unaccounted for at either end is extended by at most that
    many stream words — across genuinely adjacent audio only — and re-scored with the identical
    formula; if the honest score no longer clears the floor, it is refused
  * a bridged (`gapped`) alignment may not also swallow speech the quote does not contain, or
    denying truncation would merely displace the match onto the gapped fallback

Measured on 241 real ASR streams x 40 real authored quotes at min_ratio 0.78: 56 matches accepted
before and after, 3 newly rejected, 0 newly admitted. All three rejections are false positives —
"He was a monster." against audio saying "Joffrey was a monster.", and "You're not supposed to be
here." against "you're supposed to be south" and against "you're not supposed to just alert out the
right answer".
"""
from __future__ import annotations

import inspect

from vidlore.clipstudio import index as I


def _stream(pairs):
    """(start, end, word) triples with no gaps — the shape find_quote_span consumes."""
    t, out = 0.0, []
    for w in pairs:
        out.append((round(t, 2), round(t + 0.2, 2), w))
        t += 0.2
    return out


CUPBEARER = _stream(["Good.", "I", "want", "you", "to", "be", "happy."])


# ------------------------------------------------------------------ the motivating case
def test_the_truncated_match_is_refused():
    assert I.find_quote_span(CUPBEARER, "I want you to be my cupbearer.", min_ratio=0.78) is None


def test_the_line_that_is_actually_there_still_matches():
    got = I.find_quote_span(CUPBEARER, "I want you to be happy.", min_ratio=0.78)
    assert got is not None and got[2] >= 0.78


def test_the_arithmetic_the_clause_rests_on():
    """0.833 truncated vs 0.769 honest, against a 0.78 floor."""
    n = 7
    assert round(2 * 5 / (5 + n), 3) == 0.833
    assert round(2 * 5 / (6 + n), 3) == 0.769


# ------------------------------------------------------------------ real false positives it kills
def test_a_different_subject_is_not_the_same_line():
    """'He was a monster.' vs audio 'Joffrey was a monster.' — scored 0.857 by stopping early."""
    st = _stream(["Joffrey", "was", "a", "monster.", "He", "would", "have", "hurt", "her."])
    assert I.find_quote_span(st, "He was a monster.", min_ratio=0.78) is None


def test_a_negation_dropped_is_not_the_same_line():
    """'You're not supposed to be here.' vs 'you're supposed to be south' — opposite meaning."""
    st = _stream(["you're", "supposed", "to", "be", "south.", "You", "boys", "are", "a"])
    assert I.find_quote_span(st, "You're not supposed to be here.", min_ratio=0.78) is None


# ------------------------------------------------------------------ it can only tighten
def test_a_window_covering_the_whole_quote_is_untouched():
    st = _stream(["I", "demand", "a", "trial", "by", "combat."])
    assert I.find_quote_span(st, "I demand a trial by combat.", min_ratio=0.78) is not None


def test_no_new_threshold_is_introduced():
    """Checked against the CODE, not the prose — the docstring cites the measured numbers."""
    src = inspect.getsource(I._quote_window_pays_for_what_it_skips)
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    assert ">= min_ratio" in body, "the bar is the caller's own floor"
    for bad in ("0.7", "0.8", "0.9", "words_per_sec", "speech_rate"):
        assert bad not in body, f"a fitted constant crept into the code: {bad}"


def test_the_contiguity_bound_is_named_and_minimal():
    assert I._QUOTE_CONTIGUITY_S == 0.0


def test_the_clause_runs_before_a_candidate_can_win():
    src = inspect.getsource(I.find_quote_span)
    i = src.index("_quote_window_pays_for_what_it_skips(")
    assert i < src.index("target = gapped_best if gapped else best")


def test_a_bridged_alignment_may_not_swallow_foreign_speech():
    src = inspect.getsource(I.find_quote_span)
    assert "if gapped and (L - (hits + fuzzy)) != 0:" in src


def test_the_helper_reports_only_admissibility_and_never_a_span():
    """It decides whether a candidate may compete; it must not become a second scorer."""
    src = inspect.getsource(I._quote_window_pays_for_what_it_skips)
    assert "return " in src
    assert "candidate_start" not in src and "occurrences" not in src
