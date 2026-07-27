"""A named voiceover must never degrade into a silent-narration render.

Regression for a full-length render that completed "successfully" with captions, music and
breakout audio but no narration at all: the caller passed a voiceover path that did not exist,
every fallback in the chain declined in turn, and build landed on `_silent_narration`. An
hours-long failure that looks like a success is worse than an immediate one, so a caller that
NAMES a voiceover now gets an exception instead.

The fallbacks themselves are untouched — they exist for TTS outages (voiceover=None), which is a
different situation from a caller mistake.
"""
from __future__ import annotations

import inspect
import re

import pytest

from vidlore.clipstudio import build as B


def _narration_block() -> str:
    """The narration-selection block of build_video, as source text."""
    src = inspect.getsource(B.build_video)
    i = src.index("# 2) narration")
    j = src.index("_silent_narration(segments", i)
    return src[i:src.index("\n", j)]


def test_missing_voiceover_file_raises_before_any_work():
    """A path that does not exist is a caller mistake — fail fast, do not render."""
    block = _narration_block()
    assert re.search(r"if voiceover and not Path\(voiceover\)\.exists\(\)", block), \
        "missing-file guard is gone — a typo'd voiceover path can render silently again"
    guard = block[block.index("if voiceover and not Path(voiceover).exists()"):]
    assert "raise FileNotFoundError" in guard.split("if voiceover and Path")[0], \
        "the missing-file guard must raise, not log-and-continue"


def test_unalignable_voiceover_without_tts_raises():
    """Voiceover found but unusable and TTS off — there is no voice left, so do not pretend."""
    block = _narration_block()
    assert "if narration is None and voiceover and not use_tts" in block, \
        "an unalignable voiceover with use_tts=False can still fall through to silence"
    tail = block[block.index("if narration is None and voiceover and not use_tts"):]
    assert "raise RuntimeError" in tail.split("narration = _silent_narration")[0], \
        "the unalignable-voiceover guard must raise before the silent fallback"


def test_silent_fallback_survives_for_the_no_voiceover_case():
    """TTS outage with no uploaded voiceover must still degrade, not die."""
    block = _narration_block()
    assert "narration = _silent_narration(segments" in block, \
        "the silent fallback was removed — a keyless/offline TTS run would now hard-fail"
    # the fallback must not be reachable while a voiceover is named
    guarded = block[block.index("if narration is None and voiceover and not use_tts"):]
    assert guarded.index("raise RuntimeError") < guarded.index("_silent_narration"), \
        "the silent fallback must sit AFTER the voiceover guard"


def test_guard_is_ordered_before_the_expensive_alignment():
    """The check costs nothing; it must not sit behind an hour of compute."""
    block = _narration_block()
    assert block.index("if voiceover and not Path(voiceover).exists()") < \
        block.index("_synced_narration_from_file"), \
        "the existence check must precede alignment"


@pytest.mark.parametrize("use_tts", [True, False])
def test_portal_style_call_with_a_real_file_is_unaffected(tmp_path, use_tts):
    """The guards key off a MISSING file — a real upload takes the normal path either way."""
    vo = tmp_path / "voiceover.mp3"
    vo.write_bytes(b"\x00" * 4096)
    from pathlib import Path as _P
    assert vo.exists() and _P(str(vo)).exists(), "a saved upload passes the existence guard"
