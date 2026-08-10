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


# --- review-draft reporting -------------------------------------------------------------------
# Job 0ca9dc4c2f finished all 152 scenes, passed every provenance gate, and was thrown away over
# ONE cue of 245 reading 24.53 CPS against a 20.00 ceiling. The narration itself outran the ceiling
# there: reading speed over a span is chars/duration, so no split or merge can lower it, and
# _repair_fast_cues is right to refuse ("impossible speech stays a hard failure"). The footage,
# unverified-exact and black-frame gates all already resolve exactly this tension in favour of
# delivering a draft a human can watch; this one gate had no such path.

def test_publication_stays_strict_by_default(tmp_path):
    """The default is unchanged: a too-fast cue still destroys a publishable render."""
    words = _words(["extraordinarily", "compressed", "caption"], [1.0, 1.05, 1.10], 0.04)
    with pytest.raises(RuntimeError, match="caption readability gate"):
        assert_caption_schedule(words, tmp_path / "audit.json")


def test_review_draft_reports_an_unreadably_fast_cue_instead_of_losing_the_render(tmp_path):
    words = _words(["extraordinarily", "compressed", "caption"], [1.0, 1.05, 1.10], 0.04)
    audit = tmp_path / "caption_readability_audit.json"

    schedule = assert_caption_schedule(words, audit, on_violation="report")

    # The draft is delivered — with the finding recorded just as loudly as a hard failure would.
    assert schedule and all(rec.get("words") for rec in schedule)
    data = json.loads(audit.read_text())
    assert data["passed"] is False
    assert data["problem_count"] >= 1
    assert any("CPS" in p["reason"] for p in data["problems"])
    # Every spoken word still reaches the viewer; nothing was trimmed to manufacture a pass.
    assert sum(len(rec["words"]) for rec in schedule) == len(words)


@pytest.mark.parametrize("window", [[(2.0, 1.0)], [(1.0, 1.0)], [(0.0, float("inf"))]])
def test_review_draft_never_reports_away_a_structurally_broken_schedule(tmp_path, window):
    """Only 'too fast to read' is a judgment. A malformed schedule corrupts the subtitle files."""
    words = _real_breakout_burst()
    with pytest.raises(RuntimeError, match="invalid protected caption window"):
        assert_caption_schedule(words, tmp_path / "audit.json",
                                protected_windows=window, on_violation="report")


def test_review_draft_still_fails_a_caption_sitting_on_protected_real_audio(tmp_path):
    """A caption over a breakout's own dialogue is wrong output, not merely hard to read."""
    words = _real_breakout_burst()
    audit = tmp_path / "audit.json"
    protected = [(1.30, 8.30)]
    with pytest.raises(RuntimeError, match="caption readability gate"):
        assert_caption_schedule(words, audit, protected_windows=protected,
                                on_violation="report")
    data = json.loads(audit.read_text())
    assert any("protected real-audio" in p["reason"] for p in data["problems"])


def test_the_approved_schedule_is_not_re_judged_more_strictly_than_it_was_approved(tmp_path):
    """Every re-validation of an approved schedule must carry the verdict that approved it.

    The burn and SRT writers re-check the schedule the gate already ruled on. Deciding it again
    under the publication policy rejects the exact schedule the review draft was built around —
    which is how job 0ca9dc4c2f died a second time on the same single 24.53 CPS cue, one layer
    further down, after the render had already passed every provenance gate.
    """
    from vidlore.themes import theme

    words = _words(["extraordinarily", "compressed", "caption"], [1.0, 1.05, 1.10], 0.04)
    approved = assert_caption_schedule(words, tmp_path / "audit.json", on_violation="report")

    # Publication still refuses it...
    with pytest.raises(RuntimeError, match="not publication-safe"):
        write_ass(words, tmp_path / "strict.ass", style=theme("history")["caption"],
                  schedule=approved)

    # ...and the review draft that approved it can still burn it.
    out = write_ass(words, tmp_path / "review.ass", style=theme("history")["caption"],
                    schedule=approved, on_violation="report")
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------- protected-window edge (job 03768be9ac)
def _protected_pair(first_word_start, window=(408.68, 416.45)):
    """Cue 180/181 of the real render: narration, a real-audio breakout, narration again."""
    before = [WordTiming("him.", 406.86, 407.30), WordTiming("He", 407.60, 407.90),
              WordTiming("gave", 407.90, 408.20), WordTiming("her", 408.20, window[0])]
    after = [WordTiming("the", first_word_start, first_word_start + 0.20),
             WordTiming("iron", first_word_start + 0.20, first_word_start + 0.60),
             WordTiming("coin,", first_word_start + 0.60, first_word_start + 1.10),
             WordTiming("she", first_word_start + 1.40, first_word_start + 1.70),
             WordTiming("could", first_word_start + 1.70, first_word_start + 2.10),
             WordTiming("come.", first_word_start + 2.10, first_word_start + 2.80)]
    return before + after, [list(window)]


def _overlaps(schedule, window):
    lo, hi = window
    return [i for i, r in enumerate(schedule)
            if float(r["start"]) < hi - 1e-6 and float(r["end"]) > lo + 1e-6]


def test_a_cue_a_hair_inside_a_protected_window_is_pushed_off_it():
    """10 ms — a third of a frame — killed a five-hour render inside assemble."""
    words, protected = _protected_pair(416.44)
    schedule = _caption_schedule(words, protected_windows=protected)
    assert not _overlaps(schedule, protected[0]), \
        [(r["start"], r["end"], r["text"]) for r in schedule]


def test_that_push_lands_exactly_on_the_window_end(tmp_path):
    words, protected = _protected_pair(416.44)
    schedule = assert_caption_schedule(words, tmp_path / "cap.json",
                                       protected_windows=protected)
    started = [r for r in schedule if float(r["start"]) >= protected[0][1] - 1e-9]
    assert started, "the narration cue must survive, just later"
    assert min(float(r["start"]) for r in started) == pytest.approx(416.45, abs=0.005)


def test_it_keeps_every_spoken_word(tmp_path):
    words, protected = _protected_pair(416.44)
    schedule = assert_caption_schedule(words, tmp_path / "cap.json",
                                       protected_windows=protected)
    assert [w.word for r in schedule for w in r["words"]] == [w.word for w in words]
    assert all(float(r["end"]) > float(r["start"]) for r in schedule)


def test_a_cue_that_starts_well_inside_the_window_still_blocks():
    """Narration genuinely under borrowed dialogue is a content fault, not a rounding one —
    shoving the caption to the window end would desync it from the word being spoken."""
    words, protected = _protected_pair(414.00)
    schedule = _caption_schedule(words, protected_windows=protected)
    assert _overlaps(schedule, protected[0]), "must not be papered over"


def test_a_cue_starting_after_the_window_is_unchanged():
    words, protected = _protected_pair(416.60)
    assert not _overlaps(_caption_schedule(words, protected_windows=protected), protected[0])


def test_the_slop_is_bounded_to_one_frame_ish():
    from vidlore.captions import _CAPTION_PROTECTED_EDGE_SLOP
    assert 0.01 <= _CAPTION_PROTECTED_EDGE_SLOP <= 0.06
