"""Captions must be spelled the way the script spells them.

Job f840b0cb49 shipped twenty minutes of video in which the narrator's own script says "Jon" and
every caption said "John" — thirty-one times — plus "Various" for Varys and "Aris" for Aerys. The
script was correct throughout. Measured on that job's real voiceover and its real Whisper output:

    3088 script words, 3094 ASR words
    2982 script words (96.6%) anchored EXACTLY

An alignment that good should have produced captions from the SCRIPT, with ASR supplying only the
timings. Instead `_align_words_to_hyp` returned None and the whole file was captioned from raw ASR.
Four independent defects, each sufficient on its own:

  1. PREFIX GUARD. The script's first word is "Varys"; Whisper heard "Various".
     SequenceMatcher("varys", "various").ratio() == 0.667 against a 0.72 floor, so the guard
     returned None — discarding a 96.6% alignment because of ONE word at position 0.
  2. DEGENERATE ASR TIMESTAMPS. Whisper emitted 3 zero-duration words out of 3094 (hyp[2909]
     'than…' at 1097.98 → 1097.98). An exact anchor inherits that span, and the caller's
     all-positive validation then rejects everything.
  3. INTERIOR GAP FILL. Where one anchor ends at or after the next begins, the gap words were
     handed (t, t). The edges had already been fixed for exactly this; the interior had not.
  4. PROPER-NOUN PROTECTION. The fallback repair excluded names shorter than four characters, so
     "Jon" — the most-spoken name in the video — could never be corrected, and its 0.72 similarity
     floor also refused varys/various and aerys/aris (both 0.667).

What is NOT relaxed: the tail guard. A short unmatched TAIL is how a truncated voiceover announces
itself, and that must keep failing. Only the prefix is forgiven, and only when the two streams are
overwhelmingly the same recording.
"""
from __future__ import annotations

import math
from difflib import SequenceMatcher

import pytest

from vidlore.clipstudio.build import _align_words_to_hyp, _restore_secure_script_tokens


def _stream(words, start=0.0, step=0.4):
    """A dense (word, start, end) hypothesis with sane timings."""
    out, t = [], start
    for w in words:
        out.append((w, round(t, 3), round(t + step * 0.8, 3)))
        t += step
    return out


def _ok(aligned, flat, total):
    """The caller's exact acceptance test."""
    return (bool(aligned) and len(aligned) == len(flat)
            and all(math.isfinite(s) and math.isfinite(e) for s, e in aligned)
            and all(e > s for s, e in aligned)
            and all(aligned[i][0] >= aligned[i - 1][0] - 0.25 for i in range(1, len(aligned)))
            and aligned[-1][1] >= 0.5 * total)


# Long on purpose. Review proved the earlier 31-word body left `_overwhelming` False in the
# refusal fixtures, so the bounds they claimed to pin were never exercised — and it showed the
# project's own older regression test passes only because ITS body is ten words. Production
# scripts are thousands of words; the fixtures have to be too.
BODY = (("dies in the open on a beach below Dragonstone and the war goes on without him "
         "while the north remembers what the south forgot about winter and about fire").split()
        * 40)


# ---------------------------------------------------------------- 1. the prefix guard
def test_the_measured_similarity_that_broke_it():
    """Pin the number, so nobody 'simplifies' the floor without meeting this case again."""
    assert SequenceMatcher(None, "varys", "various").ratio() == pytest.approx(0.667, abs=0.005)
    assert SequenceMatcher(None, "aerys", "aris").ratio() == pytest.approx(0.667, abs=0.005)
    assert SequenceMatcher(None, "jon", "john").ratio() > 0.72          # this one was length-gated


def test_a_misheard_opening_name_no_longer_costs_the_whole_file():
    flat = ["Varys"] + BODY
    hyp = _stream(["Various"] + BODY)
    aligned = _align_words_to_hyp(flat, hyp)
    assert aligned is not None, "one mis-heard first word must not veto the alignment"
    assert _ok(aligned, flat, hyp[-1][2])


def test_a_genuinely_different_recording_is_still_refused():
    """The guard's real job. Low global agreement -> no relaxation, ASR keeps the captions."""
    flat = "a completely unrelated authored paragraph about something else entirely".split()
    hyp = _stream("the quick brown fox jumped over a sleeping dog near water".split())
    assert _align_words_to_hyp(flat, hyp) is None


def test_a_long_unmatched_prefix_is_still_refused():
    """Overwhelming agreement forgives a couple of words, never a paragraph.

    The fixture is deliberately at production scale: 1240 body words means six unmatched head words
    are 0.5% of the stream, so `_overwhelming` really is True and the count bound is what refuses
    this. On the old 31-word body the fixture never reached the relaxed path at all."""
    flat = "one two three four five six".split() + BODY
    hyp = _stream("alpha beta gamma delta epsilon zeta".split() + BODY)
    assert _align_words_to_hyp(flat, hyp) is None


@pytest.mark.parametrize("authored,heard", [
    (["In", "sixteen", "sixty", "six"], ["hey", "guys", "welcome", "back"]),   # ad-libbed intro
    (["Varys", "was", "the", "master"], ["Thank", "you."]),   # head-clipped + Whisper hallucination
    (["Zebra"], ["Bureaucracy"]),                             # a single unrelated word
])
def test_a_short_but_unrelated_prefix_is_refused_at_production_scale(authored, heard):
    """The case an earlier attempt got WRONG. It bypassed the lexical test entirely when agreement
    was overwhelming and the prefix short, which review executed and showed would burn four
    unrelated authored words over the opening of every caption track. The floor moves; the test is
    never skipped, so zero lexical support is still zero."""
    flat = authored + BODY
    hyp = _stream(heard + BODY)
    assert _align_words_to_hyp(flat, hyp) is None


# ---------------------------------------------------------------- 2. degenerate ASR timestamps
def test_a_zero_duration_asr_word_does_not_discard_the_alignment():
    """Whisper really does emit these — 3 of 3094 on the measured job."""
    flat = ["Varys"] + BODY
    hyp = _stream(["Varys"] + BODY)
    hyp[7] = (hyp[7][0], hyp[7][1], hyp[7][1])            # zero-duration, exactly as observed
    aligned = _align_words_to_hyp(flat, hyp)
    assert aligned is not None
    assert all(e > s for s, e in aligned), "every span must be positive after repair"
    assert _ok(aligned, flat, hyp[-1][2])


def test_the_repair_is_a_hair_not_a_guess():
    """A widened span must not move the caption or break start-time monotonicity."""
    flat = ["Varys"] + BODY
    hyp = _stream(["Varys"] + BODY)
    hyp[7] = (hyp[7][0], hyp[7][1], hyp[7][1])
    aligned = _align_words_to_hyp(flat, hyp)
    s, e = aligned[7]
    assert 0 < (e - s) <= 0.02, "sub-frame, so it cannot shift a cue"
    assert s == pytest.approx(hyp[7][1]), "the start is untouched — only the end is widened"


# ---------------------------------------------------------------- 3. interior gap fill
def test_touching_anchor_timestamps_give_the_gap_ordered_distinct_starts():
    """Asserts what only the INTERIOR fill can provide. Merely checking positivity was vacuous —
    the global degenerate-span backstop repairs (t, t) first, so the test passed with the interior
    fix reverted. Strictly increasing STARTS across a multi-word gap is the real property."""
    flat = ["Varys"] + BODY
    hyp = _stream(["Varys"] + BODY)
    for k in (5, 6, 7):                                    # three unmatched words in the gap
        hyp[k] = (f"mangled{k}", hyp[k][1], hyp[k][2])
    hyp[4] = (hyp[4][0], hyp[4][1], hyp[8][1])             # anchor 4 ends where anchor 8 starts
    aligned = _align_words_to_hyp(flat, hyp)
    assert aligned is not None
    assert all(e > s for s, e in aligned)
    starts = [aligned[i][0] for i in (5, 6, 7)]
    assert starts == sorted(starts) and len(set(starts)) == 3, \
        "the gap words must be spread, not stacked on one instant"


# ---------------------------------------------------------------- 4. proper-noun protection
class _W:
    def __init__(self, word):
        self.word = word


class _Sc:
    def __init__(self, words):
        self.words = [_W(w) for w in words]


class _Nar:
    def __init__(self, words):
        self.scenes = [_Sc(words)]


def _repair(script_words, heard_words, protected):
    nar = _Nar(heard_words)
    n = _restore_secure_script_tokens(nar, script_words, None, protected_terms=protected)
    return n, [w.word for w in nar.scenes[0].words]


PAD = ("and the war goes on without him while the north remembers what the south forgot "
       "about winter and about fire and about the long night that follows").split()


def test_a_three_letter_name_is_now_protected():
    """'Jon' is three characters. The old >=4 floor is why 31 captions said 'John'."""
    n, out = _repair(["Jon"] + PAD, ["John"] + PAD, {"Jon"})
    assert n == 1 and out[0] == "Jon"


def test_a_named_character_below_the_generic_similarity_floor_is_repaired():
    for authored, heard in (("Varys", "Various"), ("Aerys", "Aris")):
        n, out = _repair([authored] + PAD, [heard] + PAD, {authored})
        assert n == 1 and out[0] == authored, f"{heard} -> {authored}"


def test_an_unnamed_word_is_never_rewritten_however_similar():
    """horse/house scores 0.80 — well above every floor. Only the roster makes a token eligible."""
    assert SequenceMatcher(None, "horse", "house").ratio() > 0.72
    n, out = _repair(["horse"] + PAD, ["house"] + PAD, set())
    assert n == 0 and out[0] == "house"
    n, out = _repair(["horse"] + PAD, ["house"] + PAD, {"Varys"})   # roster present, word not in it
    assert n == 0 and out[0] == "house"


def test_a_different_recording_still_keeps_its_own_words():
    """The 0.90 same-narration gate is untouched: script spelling is evidence only when these are
    overwhelmingly the same narration."""
    n, out = _repair(["Jon"] + PAD, ["John"] + "totally different words spoken here".split(),
                     {"Jon"})
    assert n == 0 and out[0] == "John"


# ------------------------------------------- 5. what the roster must never make eligible
#
# Review executed the naive version of the widened roster and proved three regressions. Each one
# gets a test, because each was a real rewrite of a word the recording actually said.
# What the roster really contains after the fix: character-name tokens only. "John"/"Bradley" are
# ACTOR tokens and are deliberately absent — see test_an_actor_name_is_not_in_the_caption_roster.
ROSTER = {"The", "Hound", "Jon", "Snow", "Varys", "Ser", "Kit", "Ian"}


@pytest.mark.parametrize("heard", ["they", "then", "there", "these", "them"])
def test_a_lowercase_function_word_is_never_rewritten(heard):
    """'The Hound' and 'The Night King' put the article into the roster at >=3 chars, and the
    script's own lowercase 'the' then overwrote what the narrator actually said (0.75-0.857)."""
    n, out = _repair(["the"] + PAD, [heard] + PAD, ROSTER)
    assert n == 0 and out[0] == heard


@pytest.mark.parametrize("authored,heard", [("her", "Ser"), ("bit", "Kit"), ("and", "Ian")])
def test_a_lowercase_word_colliding_with_a_short_roster_name_is_left_alone(authored, heard):
    n, out = _repair([authored] + PAD, [heard] + PAD, ROSTER)
    assert n == 0 and out[0] == heard


def test_the_measured_jon_repair_works_with_the_real_roster():
    """With character names only, "John" is not a rostered rival and the repair fires."""
    n, out = _repair(["Jon"] + PAD, ["John"] + PAD, ROSTER)
    assert n == 1 and out[0] == "Jon"


def test_an_actor_token_in_the_roster_would_still_block_it():
    """Why the roster narrowing is the fix and not a nicety: put the actor token back and the
    absolute name-swap rule — correctly — refuses to repair Jon at all."""
    n, out = _repair(["Jon"] + PAD, ["John"] + PAD, ROSTER | {"John"})
    assert n == 0 and out[0] == "John"


def test_a_rival_the_script_really_uses_still_blocks():
    """Two roster names that both appear in the script must never be swapped for each other."""
    script = ["Tywin"] + PAD + ["Tyrion"]
    heard = ["Tyrion"] + PAD + ["Tyrion"]
    n, out = _repair(script, heard, {"Tywin", "Tyrion"})
    assert out[0] == "Tyrion", "a name the script uses elsewhere is a rival, not a mishearing"


def test_capitalisation_is_what_separates_the_two_tiers():
    """A rostered token used lowercase mid-sentence gets the GENERIC floors, so the relaxed tier is
    genuinely narrower than the guard's own membership test rather than the dead branch review
    found. Proven with a pair that ONLY the relaxed tier admits: ned/net is 3 characters and scores
    0.667 — under both generic floors, over both relaxed ones."""
    from vidlore.clipstudio.build import _CAPTION_STOPWORDS
    assert "the" in _CAPTION_STOPWORDS and "jon" not in _CAPTION_STOPWORDS
    assert SequenceMatcher(None, "ned", "net").ratio() == pytest.approx(0.667, abs=0.005)
    n, out = _repair(["ned"] + PAD, ["net"] + PAD, {"Ned"})        # lowercase in script
    assert n == 0 and out[0] == "net", "generic floors apply: 3 chars and 0.667 are both under"
    n, out = _repair(["Ned"] + PAD, ["Net"] + PAD, {"Ned"})        # capitalised in script
    assert n == 1 and out[0] == "Ned"


def test_an_actor_name_is_not_in_the_caption_roster_at_all():
    """The roster is built from CHARACTERS. Adding actors made "John" (from "John Bradley") a
    rostered rival, which blocked Jon->John outright — the review caught my own fix disabling
    itself. The name-swap rule stays absolute; the roster is what got narrowed."""
    import inspect
    from vidlore.clipstudio import build as B
    src = inspect.getsource(B.build_video)
    i = src.index("_proper_caption_terms = set()")
    block = src[i:i + 1800]
    assert '"characters"' in block
    assert '"actors"' not in block, "actor names must not enter the caption-repair roster"
