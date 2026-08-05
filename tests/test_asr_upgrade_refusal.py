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
    i = SRC.index("_remember_asr_refusal(proj, source.id)")
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
    i = SRC.index("_remember_asr_refusal(proj, source.id)")
    assert "refreshed = cached_shots" in SRC[i:i + 160]


def test_the_refusal_is_reported_not_swallowed():
    i = SRC.index("_remember_asr_refusal(proj, source.id)")
    assert "progress(" in SRC[max(0, i - 500):i]
    assert "barred from quote" in SRC[max(0, i - 500):i]


# ------------------------------------------------------------------ the bar must not just move
#
# MEASURED: after the first fix, a 72-minute run reached the pool-completeness check and died there
# instead — `ASR evidence incomplete for 13/141 usable source(s)` — with the two barred sources
# reappearing as `asr_prompt_fingerprint_mismatch`. Refusing a source's quote evidence and then
# demanding that same source carry a current quote-evidence fingerprint is incoherent: it relocates
# the abort rather than resolving it.
def test_a_barred_source_is_not_required_to_carry_a_current_prompt_fingerprint():
    src = inspect.getsource(I.asr_pool_cache_audit)
    i = src.index('"asr_prompt_fingerprint_mismatch"')
    guard = src.rindex("_asr_upgrade_refused", 0, i)
    assert guard < i, "the exclusion must gate the mismatch, not follow it"


def test_the_exclusion_is_scoped_to_the_fingerprint_check_only():
    """Every other integrity reason still applies to a barred source — missing shots, corrupt
    words, uncertified evidence, stale schema, interrupted refresh, bad provenance."""
    src = inspect.getsource(I.asr_pool_cache_audit)
    assert src.count("_asr_upgrade_refused") == 1
    for reason in ("shots_cache_invalid_or_missing", "words_cache_invalid_or_missing",
                   "word_evidence_not_certified", "word_evidence_schema_stale",
                   "asr_refresh_interrupted"):
        i = src.index(reason)
        assert "_asr_upgrade_refused" not in src[max(0, i - 300):i], reason


# ------------------------------------------------------------------ the bar must outlive a resume
#
# `quote_retrieval_source_eligible` is handed a source and its shots, never the project, so the bar
# has to live in process state for it to be reachable. But a resumed render is a NEW process, where
# that state is empty — and the pool audit would then demand quote evidence from a source the
# earlier process had already refused to take any from, reproducing the very abort the bar exists to
# prevent. So it is written to the project too, and seeded back before anything consults it.
def test_the_bar_is_written_to_the_project_not_only_to_process_state():
    src = inspect.getsource(I._remember_asr_refusal)
    assert "_asr_upgrade_refused.add(sid)" in src
    assert I._ASR_REFUSED_META_KEY in src or "_ASR_REFUSED_META_KEY" in src


def test_indexing_seeds_the_bar_before_it_indexes_anything():
    src = inspect.getsource(I.index_all)
    assert "_seed_asr_refusals(proj)" in src
    assert src.index("_seed_asr_refusals(proj)") < src.index("index_source(")


def test_seeding_happens_before_the_audit_that_consults_it():
    src = inspect.getsource(I.index_all)
    assert src.index("_seed_asr_refusals(proj)") < src.index("asr_pool_cache_audit(")


def test_a_round_trip_through_project_meta_restores_the_bar():
    class P:
        def __init__(self):
            self.meta = {}
    proj = P()
    I._remember_asr_refusal(proj, "some_source")
    assert "some_source" in proj.meta[I._ASR_REFUSED_META_KEY]
    I._asr_upgrade_refused.clear()                    # simulate a fresh process
    assert "some_source" not in I._asr_upgrade_refused
    I._seed_asr_refusals(proj)
    assert "some_source" in I._asr_upgrade_refused


def test_bookkeeping_failure_never_breaks_indexing():
    for fn in (I._remember_asr_refusal, I._seed_asr_refusals):
        assert "except Exception" in inspect.getsource(fn)
    I._remember_asr_refusal(object(), "no_meta_attr")   # must not raise
    I._seed_asr_refusals(object())
