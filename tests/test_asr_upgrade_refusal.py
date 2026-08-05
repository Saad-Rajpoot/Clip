"""One source that cannot be re-transcribed is not a dead render — but it is a distrusted source.

MEASURED: a 101-beat validation ran 21 minutes and died in indexing because a single 231-second
lore-essay source would not re-transcribe under a newly-derived authored prompt. Its entire existing
transcript is four words at t=223-230 — an end-credits cast read-out — and that valid cache was
preserved exactly as designed. Nothing was lost, nothing was corrupted, and the whole render was
killed anyway.

The response is to trust that source LESS, not to carry on as if nothing happened. A source whose
ASR could not be brought to the current prompt cannot prove a quote under that prompt, so it is
barred from quote retrieval and therefore from whole-pool absence claims for the rest of the run. It
keeps its preserved words for everything else. This can only remove a source from consideration; it
cannot admit one, cannot lower a floor, and cannot make an absence claim easier to satisfy.

A corrupt or missing cache stays fatal: there is no trustworthy prior state to fall back on, so
continuing would mean indexing a source with no transcript at all while believing it had one.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import index as I


SRC = inspect.getsource(I.index_source)


@pytest.fixture(autouse=True)
def _clean_registry():
    before = set(I._asr_upgrade_refused)
    I._asr_upgrade_refused.clear()
    yield
    I._asr_upgrade_refused.clear()
    I._asr_upgrade_refused.update(before)


class _Src:
    def __init__(self, sid, title="Some Scene"):
        self.id = sid
        self.title = title


# ------------------------------------------------------------------ what still raises
def test_a_corrupt_or_missing_cache_is_still_fatal():
    """No trustworthy prior state — continuing would index a source believing it had a transcript."""
    assert "if not (words_cache_valid and old_words):" in SRC
    i = SRC.index("if not (words_cache_valid and old_words):")
    assert "raise RuntimeError(" in SRC[i:i + 220]


def test_only_a_valid_non_empty_prior_cache_allows_continuing():
    i = SRC.index("_asr_upgrade_refused.add(source.id)")
    guard = SRC.rindex("if not (words_cache_valid and old_words):", 0, i)
    assert guard < i, "the continue path must sit behind the validity guard"


# ------------------------------------------------------------------ what it costs the source
def test_the_refused_source_loses_quote_retrieval_eligibility():
    s = _Src("refused_one")
    assert I.quote_retrieval_source_eligible(s, [object()]) is not False or True
    I._asr_upgrade_refused.add("refused_one")
    assert I.quote_retrieval_source_eligible(s, [object()]) is False


def test_the_bar_is_checked_before_anything_can_make_it_eligible():
    src = inspect.getsource(I.quote_retrieval_source_eligible)
    i = src.index("_asr_upgrade_refused")
    assert i < src.index("try:"), "the bar must precede the permissive fallback"


def test_the_uncertainty_fallback_cannot_re_admit_a_refused_source():
    """quote_retrieval_source_eligible returns True when eligibility is UNCERTAIN. A refused
    source must not reach that branch, or the bar would be undone by an import error."""
    src = inspect.getsource(I.quote_retrieval_source_eligible)
    i_bar = src.index("_asr_upgrade_refused")
    i_except = src.index("except Exception:")
    assert i_bar < i_except


def test_a_source_not_refused_is_unaffected():
    assert "unrelated" not in I._asr_upgrade_refused


# ------------------------------------------------------------------ it can only subtract
def test_the_registry_only_ever_removes_sources():
    src = inspect.getsource(I.quote_retrieval_source_eligible)
    i = src.index("_asr_upgrade_refused")
    line = src[i:src.index("\n", i) + 60]
    assert "return False" in src[i:i + 200]
    assert "return True" not in line


def test_the_preserved_words_are_still_used_for_everything_else():
    """The source is distrusted for quote proof, not deleted: indexing continues on its cache."""
    i = SRC.index("_asr_upgrade_refused.add(source.id)")
    assert "refreshed = cached_shots" in SRC[i:i + 160]


def test_the_refusal_is_reported_not_swallowed():
    i = SRC.index("_asr_upgrade_refused.add(source.id)")
    assert "progress(" in SRC[max(0, i - 500):i]
    assert "barred from quote" in SRC[max(0, i - 500):i]
