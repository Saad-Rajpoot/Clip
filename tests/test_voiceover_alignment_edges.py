from __future__ import annotations

from types import SimpleNamespace

from vidlore.captions import _caption_schedule, caption_schedule_problems
from vidlore.clipstudio.build import (
    _align_words_to_hyp,
    _restore_secure_script_tokens,
    _truncated_voiceover_tail_evidence,
)


def _hyp(words: list[str], *, step: float = 0.32, start: float = 0.05):
    return [
        (word, start + i * step, start + (i + 1) * step)
        for i, word in enumerate(words)
    ]


def _assert_valid_caption_timing(words: list[str], aligned: list[tuple[float, float]]) -> None:
    assert len(words) == len(aligned)
    assert all(end > start >= 0.0 for start, end in aligned)
    assert all(aligned[i][0] >= aligned[i - 1][0] for i in range(1, len(aligned)))
    timed = [
        SimpleNamespace(word=word, start=start, end=end)
        for word, (start, end) in zip(words, aligned)
    ]
    assert caption_schedule_problems(
        _caption_schedule(timed), hard_cps=float("inf")
    ) == []


def test_truncated_trailing_words_do_not_replace_accurate_asr():
    # Exact production edge: stronger ASR hears only "liar some" before the recording ends, so
    # base ASR's "liars" is not enough lexical evidence to force both authored words on screen.
    prefix = "where a lie will eventually surface and cost the".split()
    script = prefix + ["liar", "something"]
    heard = prefix + ["liars"]

    aligned = _align_words_to_hyp(script, _hyp(heard))

    assert aligned is None


def test_leading_token_fusion_donates_real_lexically_supported_prefix_span():
    script = ["Littlefinger"] + "then eight exact anchor words follow in order now".split()
    heard = ["Little", "finger"] + "then eight exact anchor words follow in order now".split()

    aligned = _align_words_to_hyp(script, _hyp(heard))

    assert aligned is not None
    _assert_valid_caption_timing(script, aligned)
    assert aligned[0][0] == _hyp(heard)[0][1]
    assert aligned[0][1] == _hyp(heard)[2][1]


def test_unrelated_positive_duration_edge_cannot_bypass_asr_fallback():
    common = "eight exact anchor words prove the middle content matches well".split()
    assert _align_words_to_hyp(
        ["Authored", "opening"] + common,
        _hyp(["Different"] + common),
    ) is None
    assert _align_words_to_hyp(
        common + ["Authored", "ending"],
        _hyp(common + ["Different"]),
    ) is None


def test_unheard_edge_words_still_fail_instead_of_getting_invented_time():
    common = "eight exact anchor words prove the middle content matches well".split()
    assert len(common) >= 8

    # No hypothesis material exists before/after the anchors in either case.  These are genuine
    # content gaps, so the existing ASR fallback contract remains intact.
    assert _align_words_to_hyp(["unheard"] + common, _hyp(common)) is None
    assert _align_words_to_hyp(common + ["unheard"], _hyp(common)) is None


def test_render_scale_unsupported_tail_cannot_replace_asr():
    # At production scale, an unsupported authored tail still cannot displace accurate ASR.
    common = [f"word{i}" for i in range(1063)]
    script = common + ["liar", "something"]
    heard = common + ["liars"]

    aligned = _align_words_to_hyp(script, _hyp(heard))

    assert aligned is None


def test_near_exact_script_with_unproven_speech_at_eof_is_truncated():
    common = [f"word{i}" for i in range(100)]
    script = common + ["liar", "something"]
    heard = _hyp(common + ["liars"], step=0.1, start=0.0)

    evidence = _truncated_voiceover_tail_evidence(
        script, heard, heard[-1][2] + 0.02)

    assert evidence is not None
    assert evidence["script_tail"] == ["liar", "something"]
    assert evidence["asr_tail"] == ["liars"]


def test_tail_mismatch_with_silence_or_low_overlap_is_not_called_truncated():
    common = [f"word{i}" for i in range(100)]
    heard = _hyp(common + ["different"], step=0.1, start=0.0)
    assert _truncated_voiceover_tail_evidence(
        common + ["authored"], heard, heard[-1][2] + 0.50
    ) is None
    assert _truncated_voiceover_tail_evidence(
        ["unrelated"] * 100, heard, heard[-1][2] + 0.02
    ) is None
    # Complete but deliberately paraphrased EOF is not truncation just because there is no pad.
    complete_paraphrase = _hyp(common + ["big"], step=0.1, start=0.0)
    assert _truncated_voiceover_tail_evidence(
        common + ["large"], complete_paraphrase, complete_paraphrase[-1][2] + 0.02
    ) is None


def test_script_guided_asr_restore_is_one_to_one_and_preserves_real_edits():
    common = [f"anchor{i}" for i in range(100)]
    script = common + ["ridden", "tourney", "unfalsifiable", "Olenna's", "out-think",
                       "king's", "Stannis", "boundary", "is", "liar", "something"]
    heard = common + ["written", "turnie", "unfalseifiable", "Olenna", "outthink",
                      "kings", "Stanis", "boundary", "has", "liars"]
    scene = SimpleNamespace(words=[
        SimpleNamespace(word=w, start=i * 0.3, end=(i + 1) * 0.3)
        for i, w in enumerate(heard)
    ])
    narration = SimpleNamespace(scenes=[scene])

    fixed = _restore_secure_script_tokens(
        narration, script, protected_terms={"Olenna", "Stannis"})

    assert fixed == 4
    assert [w.word for w in scene.words][len(common):] == [
        "written", "turnie", "unfalseifiable", "Olenna's", "out-think",
        "king's", "Stannis", "boundary", "has", "liars",
    ]


def test_script_guided_asr_restore_preserves_additions_deletions_and_unrelated_words():
    script = "alpha produced beta authored gamma".split()
    heard = "alpha beta and different gamma".split()
    scene = SimpleNamespace(words=[
        SimpleNamespace(word=w, start=i * 0.3, end=(i + 1) * 0.3)
        for i, w in enumerate(heard)
    ])

    assert _restore_secure_script_tokens(SimpleNamespace(scenes=[scene]), script) == 0
    assert [w.word for w in scene.words] == heard


def test_script_guided_restore_never_rewrites_confusable_generic_words():
    common = [f"anchor{i}" for i in range(100)]
    script = common + ["horse", "three", "quite", "trial"]
    heard = common + ["house", "there", "quiet", "trail"]
    scene = SimpleNamespace(words=[
        SimpleNamespace(word=w, start=i * 0.3, end=(i + 1) * 0.3)
        for i, w in enumerate(heard)
    ])

    assert _restore_secure_script_tokens(SimpleNamespace(scenes=[scene]), script) == 0
    assert [w.word for w in scene.words] == heard


def test_script_guided_restore_never_rewrites_one_known_name_to_another():
    common = [f"anchor{i}" for i in range(100)]
    script = common + ["Tyrion"]
    heard = common + ["Tywin"]
    scene = SimpleNamespace(words=[
        SimpleNamespace(word=w, start=i * 0.3, end=(i + 1) * 0.3)
        for i, w in enumerate(heard)
    ])

    assert _restore_secure_script_tokens(
        SimpleNamespace(scenes=[scene]), script,
        protected_terms={"Tyrion", "Tywin"}) == 0
    assert [word.word for word in scene.words] == heard
