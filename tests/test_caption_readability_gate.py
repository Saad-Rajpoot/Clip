import json

import pytest

from vidlore.captions import (
    _caption_schedule,
    _group,
    assert_caption_schedule,
    caption_schedule_problems,
    write_ass,
)
from vidlore.tts import WordTiming


def _words(tokens, starts, dur=0.16):
    return [WordTiming(t, s, s + dur) for t, s in zip(tokens, starts)]


def _real_breakout_burst():
    # Production cue 18, shifted to a compact fixture. The 300 ms after the
    # verified dialogue window is clean, but an unknown long gap may not donate it.
    spec = [
        ("the", .00, .06), ("one", .06, .22), ("who", .22, .36),
        ("was", .36, .50), ("always", .50, .74), ("three", .74, 1.08),
        ("moves", 1.08, 1.38), ("ahead", 1.38, 1.66), ("of", 1.66, 1.80),
        ("people", 1.80, 2.06), ("who", 2.06, 2.26), ("were", 2.26, 2.36),
        ("physically", 2.36, 2.66), ("stronger,", 2.66, 3.32),
        ("better", 3.76, 3.92), ("born,", 3.92, 4.28),
        ("and", 4.46, 4.82), ("better", 4.82, 5.04),
        ("protected.", 5.04, 5.52), ("Watch", 6.20, 6.42),
        ("it", 6.42, 6.58), ("again.", 6.58, 6.90),
    ]
    return [WordTiming("Past.", 0.0, 0.30)] + [
        WordTiming(token, 1.30 + start, 1.30 + end)
        for token, start, end in spec
    ]


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


def test_verified_breakout_lead_repairs_real_burst_on_serialized_clock():
    words = _real_breakout_burst()
    untrusted = _caption_schedule(words)
    assert any("CPS" in p["reason"] for p in caption_schedule_problems(untrusted))

    windows = [(0.40, 1.00)]
    repaired = _caption_schedule(words, protected_windows=windows)
    assert caption_schedule_problems(repaired) == []
    assert [w for cue in repaired for w in cue["words"]] == words
    assert any(len(cue["words"]) > 12 for cue in repaired)
    assert all(len(cue["words"]) <= 16 and len(cue["text"]) <= 84 for cue in repaired)
    assert all(abs(cue["start"] * 100 - round(cue["start"] * 100)) < 1e-8
               and abs(cue["end"] * 100 - round(cue["end"] * 100)) < 1e-8
               for cue in repaired)
    assert all(not (cue["start"] < 1.00 and cue["end"] > 0.40) for cue in repaired)
    first_after = next(cue for cue in repaired if cue["words"][0].word == "the")
    assert 1.00 <= first_after["start"] < 1.30


def test_local_repair_is_deterministic_and_does_not_mutate_words():
    words = _real_breakout_burst()
    original = [(w.word, w.start, w.end) for w in words]
    a = _caption_schedule(words, protected_windows=[(0.40, 1.00), (20.0, 21.0)])
    b = _caption_schedule(words, protected_windows=[(20.0, 21.0), (0.40, 1.00)])
    fingerprint = lambda schedule: [
        ([id(w) for w in cue["words"]], cue["text"], cue["start"], cue["end"])
        for cue in schedule
    ]
    assert fingerprint(a) == fingerprint(b)
    assert [(w.word, w.start, w.end) for w in words] == original


def test_impossible_audio_still_fails_with_a_nearby_protected_window(tmp_path):
    words = [WordTiming("extraordinarily", 1.0 + i * .02, 1.01 + i * .02)
             for i in range(18)]
    audit = tmp_path / "caption_readability_audit.json"
    with pytest.raises(RuntimeError, match="caption readability gate"):
        assert_caption_schedule(
            words, audit, protected_windows=[(0.50, 0.99)])
    data = json.loads(audit.read_text())
    assert data["passed"] is False
    assert data["word_count"] == 18
    assert sum(cue["words"] for cue in data["cues"]) == 18
    assert any("CPS" in problem["reason"] for problem in data["problems"])


@pytest.mark.parametrize("window", [
    [(2.0, 1.0)], [(float("nan"), 1.0)], [(0.0, float("inf"))],
    [(1.0,)], [(-1.0, 1.0)], [(1.0, 1.0)],
])
def test_malformed_protected_window_never_donates_dwell_and_is_audited(tmp_path, window):
    words = _real_breakout_burst()
    audit = tmp_path / "caption_readability_audit.json"
    with pytest.raises(RuntimeError, match="invalid protected caption window"):
        assert_caption_schedule(words, audit, protected_windows=window)
    data = json.loads(audit.read_text())
    assert data["passed"] is False
    assert data["protected_windows"] == []
    assert any("invalid protected" in p["reason"] for p in data["problems"])
    assert any("CPS" in p["reason"] for p in data["problems"])


def test_burn_blocked_window_splits_and_clamps_even_when_no_word_is_removed(tmp_path):
    words = [WordTiming("hello", 0.0, 0.2), WordTiming("world", 0.8, 1.0)]
    blocked = [(0.35, 0.65)]
    audit = tmp_path / "caption_burn_readability_audit.json"
    schedule = assert_caption_schedule(words, audit, blocked_windows=blocked)
    assert len(schedule) == 2
    assert schedule[0]["end"] <= 0.35
    assert schedule[1]["start"] >= 0.65
    assert all(not (cue["start"] < 0.65 and cue["end"] > 0.35) for cue in schedule)
    assert json.loads(audit.read_text())["blocked_windows"] == [[0.35, 0.65]]


def test_burn_blocked_window_stays_empty_after_intersecting_word_is_removed(tmp_path):
    original = [WordTiming("hello", 0.0, 0.2), WordTiming("middle", 0.4, 0.6),
                WordTiming("world", 0.8, 1.0)]
    blocked = [(0.35, 0.65)]
    kept = [word for word in original
            if not any(word.start < end and word.end > start for start, end in blocked)]
    schedule = assert_caption_schedule(
        kept, tmp_path / "caption_burn_readability_audit.json",
        blocked_windows=blocked)
    assert [w.word for cue in schedule for w in cue["words"]] == ["hello", "world"]
    assert all(not (cue["start"] < 0.65 and cue["end"] > 0.35) for cue in schedule)


def test_non_centisecond_block_is_quantized_inward(tmp_path):
    words = [WordTiming("hello", 0.0, 0.2), WordTiming("world", 0.656, 1.0)]
    schedule = assert_caption_schedule(
        words, tmp_path / "caption_burn_readability_audit.json",
        blocked_windows=[(0.355, 0.655)])
    assert schedule[0]["end"] <= 0.35
    assert schedule[1]["start"] >= 0.66
    assert all(not (cue["start"] < 0.655 and cue["end"] > 0.355)
               for cue in schedule)


def test_ass_last_word_cannot_reenter_an_approved_blocked_window(tmp_path):
    from vidlore.themes import theme
    words = [WordTiming("hello", 0.0, 0.355), WordTiming("world", 0.655, 1.0)]
    blocked = [(0.355, 0.655)]
    schedule = assert_caption_schedule(
        words, tmp_path / "caption_burn_readability_audit.json",
        blocked_windows=blocked)
    ass = write_ass(words, tmp_path / "captions.ass", style=theme("history")["caption"],
                    schedule=schedule)

    def seconds(value):
        hours, minutes, secs = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(secs)

    events = []
    for line in ass.read_text().splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(",", 3)
            events.append((seconds(fields[1]), seconds(fields[2])))
    assert events
    assert all(not (start < 0.655 and end > 0.355) for start, end in events)


def test_empty_burn_stream_still_persists_its_blocked_window_audit(tmp_path):
    audit = tmp_path / "caption_burn_readability_audit.json"
    assert assert_caption_schedule([], audit, blocked_windows=[(0.35, 0.65)]) == []
    data = json.loads(audit.read_text())
    assert data["passed"] is True
    assert data["word_count"] == 0
    assert data["blocked_windows"] == [[0.35, 0.65]]


def test_srt_returns_the_approved_breakout_schedule(tmp_path):
    from vidlore.assemble import _srt
    words = _real_breakout_burst()
    path = tmp_path / "final.srt"
    schedule = _srt(words, path, protected_windows=[(0.40, 1.00)])
    audit = json.loads((tmp_path / "caption_readability_audit.json").read_text())
    assert path.exists()
    assert audit["passed"] is True
    assert audit["protected_windows"] == [[0.4, 1.0]]
    assert audit["max_cps"] <= 20.0
    assert len(schedule) == audit["cue_count"]


def test_ass_rejects_an_approved_schedule_for_another_word_stream(tmp_path):
    from vidlore.themes import theme
    words = _real_breakout_burst()
    schedule = _caption_schedule(words, protected_windows=[(0.40, 1.00)])
    clones = [WordTiming(w.word, w.start, w.end) for w in words]
    with pytest.raises(RuntimeError, match="does not own"):
        write_ass(clones, tmp_path / "bad.ass", style=theme("history")["caption"],
                  schedule=schedule)


def test_srt_and_ass_share_the_schedule_and_build_preflights_breakouts():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assemble = (root / "vidlore" / "assemble.py").read_text()
    build = (root / "vidlore" / "clipstudio" / "build.py").read_text()
    assert "schedule = assert_caption_schedule(" in assemble
    assert 'path.with_name("caption_readability_audit.json")' in assemble
    assert "protected_windows=breakout_windows" in assemble
    assert "schedule=_approved_caption_schedule" in assemble
    assert "schedule=_burn_schedule" in assemble
    assert "blocked_windows=_drop_wins" in assemble
    assert "word_start < end and word_end > start" in assemble
    assert "_normalize_caption_windows(" in assemble
    assert 'workdir / "caption_burn_readability_audit.json"' in assemble
    assert "_breakout_ass_preflight = _breakout_caption_ass(" in build
    assert 'kind="breakout_caption"' in build
    assert ".FAILED_BREAKOUT_CAPTION" in build
