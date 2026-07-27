"""Every LLM/vision call must be counted, and counting must never be able to break a render.

A render makes hundreds of Gemini vision calls — 512 fresh on job 69d80e9dd4_v4 with a warm verdict
cache, ~2000 cold — and none of it was recorded anywhere. "What did this render cost?" had no
answer, and "should verify run a second round?" was an unpriced decision. Measured after this
landed: 878 in / 98 out per verifier call, $0.0005 a call, ~$1 for a cold render.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio import llm


@pytest.fixture(autouse=True)
def _clean():
    llm.reset_usage()
    yield
    llm.reset_usage()


def test_records_calls_and_tokens():
    llm.record_usage("gemini-2.5-flash", prompt=878, completion=98)
    llm.record_usage("gemini-2.5-flash", prompt=878, completion=98)
    s = llm.usage_summary()
    assert s["calls"] == 2
    assert s["prompt"] == 1756 and s["completion"] == 196
    assert s["models"]["gemini-2.5-flash"]["calls"] == 2


def test_usd_uses_the_per_model_rate():
    llm.record_usage("gemini-2.5-flash", prompt=1_000_000, completion=1_000_000)
    s = llm.usage_summary()
    assert s["usd"] == pytest.approx(0.30 + 2.50, abs=1e-6)


def test_price_is_overridable_per_model(monkeypatch):
    """Rate cards change; a stale constant must not silently misreport spend."""
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_PRICE_GEMINI_2_5_FLASH_IN", "0.10")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_PRICE_GEMINI_2_5_FLASH_OUT", "1.00")
    llm.record_usage("gemini-2.5-flash", prompt=1_000_000, completion=1_000_000)
    assert llm.usage_summary()["usd"] == pytest.approx(1.10, abs=1e-6)


def test_unknown_model_is_counted_but_priced_at_zero():
    """An unpriced model must still show its call count rather than vanish from the report."""
    llm.record_usage("some-new-model", prompt=500, completion=50)
    s = llm.usage_summary()
    assert s["calls"] == 1 and s["prompt"] == 500
    assert s["models"]["some-new-model"]["usd"] == 0.0


def test_accounting_never_raises():
    """Booking is best-effort — a junk value must not take a render down with it."""
    for bad in (None, "x", object(), float("nan")):
        llm.record_usage("gemini-2.5-flash", prompt=bad, completion=bad)   # type: ignore[arg-type]
    llm.record_usage(None, prompt=1, completion=1)                          # type: ignore[arg-type]
    assert llm.usage_summary()["calls"] >= 1


def test_usage_extraction_handles_both_provider_shapes():
    class _Gem:
        class usage_metadata:                     # noqa: N801 - mirrors the SDK attribute name
            prompt_token_count = 878
            candidates_token_count = 98

    class _Claude:
        class usage:
            input_tokens = 1200
            output_tokens = 300

    assert llm._usage_from(_Gem()) == (878, 98)
    assert llm._usage_from(_Claude()) == (1200, 300)
    assert llm._usage_from(object()) == (0, 0), "a response with no usage must not raise"


def test_reset_clears_between_renders():
    llm.record_usage("gemini-2.5-flash", prompt=10, completion=1)
    llm.reset_usage()
    assert llm.usage_summary()["calls"] == 0


def test_every_provider_branch_books_its_usage():
    """A branch that forgets to book makes the total silently understate the bill."""
    import inspect
    src = inspect.getsource(llm)
    for fn in ("_gemini_complete", "_claude_complete", "_deepseek_complete"):
        i = src.index(f"def {fn}")
        j = src.index("\ndef ", i + 1)
        assert "record_usage(" in src[i:j], f"{fn} does not record its token usage"
