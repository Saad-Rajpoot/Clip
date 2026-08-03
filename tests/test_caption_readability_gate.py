import json

import pytest

from vidlore.captions import (
    _caption_schedule,
    _group,
    assert_caption_schedule,
    caption_schedule_problems,
)
from vidlore.tts import WordTiming


def _words(tokens, starts, dur=0.16):
    return [WordTiming(t, s, s + dur) for t, s in zip(tokens, starts)]


def test_adjacent_fast_fragments_merge_without_losing_tokens():
    # Two legacy groups (6 + 4 words) are individually too fast because the
    # first ends at an arbitrary group boundary.  Together they are readable.
    toks = "this deliberately paced sentence has ten words and enough time now".split()
    starts = [i * 0.28 for i in range(len(toks))]
    schedule = _caption_schedule(_words(toks, starts, 0.18))
    flattened = [w.word for r in schedule for w in r["words"]]
    assert flattened == toks
    assert all(r["end"] > r["start"] for r in schedule)
    assert not caption_schedule_problems(schedule)


def test_zero_duration_alignment_fails_closed():
    words = [WordTiming("itemise", 423.94, 423.94),
             WordTiming("him", 423.94, 423.94)]
    probs = caption_schedule_problems(_caption_schedule(words))
    assert any("non-positive word span" in p["reason"] for p in probs)


def test_unreadably_fast_audio_fails_the_hard_cps_gate():
    words = _words(["extraordinarily", "compressed", "caption"], [1.0, 1.05, 1.10], 0.04)
    probs = caption_schedule_problems(_caption_schedule(words), hard_cps=25.0)
    assert any("CPS" in p["reason"] for p in probs)


def test_real_pause_is_not_consumed_to_fake_readability():
    words = [WordTiming("first", 0.0, 0.25), WordTiming("sentence.", 0.3, 0.7),
             WordTiming("second", 4.0, 4.3)]
    schedule = _caption_schedule(words)
    assert len(schedule) == 2
    assert schedule[1]["start"] - schedule[0]["end"] > 1.0


def test_schedule_never_overlaps_and_uses_each_word_once():
    words = _words([f"w{i}" for i in range(18)], [i * 0.22 for i in range(18)], 0.14)
    schedule = _caption_schedule(words)
    assert [id(w) for r in schedule for w in r["words"]] == [id(w) for w in words]
    assert all(schedule[i]["end"] <= schedule[i + 1]["start"] + 1e-6
               for i in range(len(schedule) - 1))


def test_mechanical_boundary_does_not_end_on_a_dangling_pronoun():
    toks = "is exactly how Varys learns she is in the city".split()
    words = _words(toks, [i * .32 for i in range(len(toks))], .22)
    cues = _group(words, max_words=6)
    assert cues[0][-1].word != "she"
    assert [w.word for cue in cues for w in cue] == toks


def test_authored_sentence_punctuation_is_a_preferred_boundary():
    toks = ["One", "claim", "ends.", "Another", "one", "starts"]
    words = _words(toks, [i * .35 for i in range(len(toks))], .2)
    cues = _group(words, max_words=6)
    assert [w.word for w in cues[0]] == ["One", "claim", "ends."]


def test_publication_gate_persists_failure_before_raising(tmp_path):
    words = _words(["extraordinarily", "compressed", "caption"],
                   [1.0, 1.05, 1.10], 0.04)
    audit = tmp_path / "caption_readability_audit.json"
    with pytest.raises(RuntimeError, match="caption readability gate"):
        assert_caption_schedule(words, audit)
    data = json.loads(audit.read_text())
    assert data["passed"] is False
    assert data["problem_count"] >= 1
    assert any("CPS" in p["reason"] for p in data["problems"])


def test_srt_and_ass_share_the_schedule_and_build_preflights_breakouts():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assemble = (root / "vidlore" / "assemble.py").read_text()
    build = (root / "vidlore" / "clipstudio" / "build.py").read_text()
    assert "schedule = assert_caption_schedule(" in assemble
    assert 'path.with_name("caption_readability_audit.json")' in assemble
    assert "_breakout_ass_preflight = _breakout_caption_ass(" in build
    assert 'kind="breakout_caption"' in build
    assert ".FAILED_BREAKOUT_CAPTION" in build
