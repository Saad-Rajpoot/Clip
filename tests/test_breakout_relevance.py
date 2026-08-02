"""A breakout must belong AT THE BEAT it interrupts. 100% relevant or it does not air.

Job benjen_v2 — a 12:55 essay about BENJEN STARK, White Walkers and dragonglass — aired seven
real-audio breakouts. FOUR of them play a Season-1 CERSEI-and-NED conversation: Robert coming home
drunk, Jaime, "do you love your children", the Iron Throne. Every one of those was genuine
in-character dialogue, so the pre-existing dialogue-vs-narration gate passed them cleanly.

Why they got there (measured, not assumed):
  * The evidence miner binds a shot to whichever beat shares the most content words, floor 2, with
    ±2-char prefix fuzz. Beat 112 won its Cersei line on exactly two tokens — "children's"↔children
    (the Children of the Forest vs Cersei's children) and "heart"↔heart (a dragonglass shard in the
    heart vs "with all my heart").
  * Beat 112 CARRIES the right quote — Benjen's own "They drove a dragonglass dagger into my heart"
    — and that quote locates in ZERO of the job's 102 word streams (best ratio 0.471 against a 0.72
    floor). Nothing bound the quote to the audio, so a mined line served a quoted beat.
  * The compilation "Cersei and Jaime Lannister - All Scenes Part 1/8" entered the miner's trusted
    tier on two title tokens, 'scene' and 'cersei', taken from the Dragonpit anchor query.

Every one of those is a PROXY — a title token, an overlap count, a stoplist, prefix fuzz. Each was
defeated by an input that satisfies the proxy without satisfying the intent, and tightening any of
them just moves the next leak. So the gate here judges the two artifacts a viewer judges: the words
that will be HEARD and the words the NARRATION says.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import build as B
from vidlore.clipstudio import llm as L


# --------------------------------------------------------------------------- harness
def _judge(ident=None, rel=None, *, has_llm=True, replies=None, **kw):
    """Run the admission gate against scripted provider replies.

    The gate asks TWO questions in two separate calls, so the harness scripts them separately:
    `ident` answers stage 1 (blind identification) and `rel` answers stage 2 (does it belong).
    `replies=` feeds one flat list to every call, for the malformed/short-circuit paths.
    """
    seq = list(replies) if replies is not None else None
    ident = list(ident or [])
    rel = list(rel or [])
    calls = {"flat": 0, "i": 0, "r": 0}
    orig = (L.has_llm, L.complete)
    L.has_llm = lambda *a, **k: has_llm

    def _complete(*, system="", **k):
        if seq is not None:
            i = calls["flat"]
            calls["flat"] += 1
            return seq[i] if i < len(seq) else seq[-1]
        if "Identify it" in system:                     # stage 1's system prompt
            i = calls["i"]
            calls["i"] += 1
            return ident[i] if i < len(ident) else (ident or [_i()])[-1]
        i = calls["r"]
        calls["r"] += 1
        return rel[i] if i < len(rel) else (rel or [_b()])[-1]

    L.complete = _complete
    try:
        return B._breakout_window_admissible(
            "a real spoken line of some length", "Game of Thrones", **kw)
    finally:
        L.has_llm, L.complete = orig


AIRED = "a real spoken line of some length"


def _i(speaker="Benjen Stark", in_story=True, conf=0.95, scene="the Wall", intelligible=True,
       line=AIRED):
    """A stage-1 identification reply. `line` defaults to a verbatim quote of the harness audio."""
    return ('{"intelligible":%s,"line":"%s","speaker":"%s","scene":"%s","in_story":%s,'
            '"confidence":%s}'
            % (str(intelligible).lower(), line, speaker, scene, str(in_story).lower(), conf))


def _b(belongs=True, conf=0.95, why="same scene"):
    """A stage-2 relevance reply."""
    return '{"belongs":%s,"why":"%s","confidence":%s}' % (str(belongs).lower(), why, conf)


@pytest.fixture(autouse=True)
def _three_samples(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_SAMPLES", "3")
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_MIN_CONF", raising=False)
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_CHECK", raising=False)


# --------------------------------------------------------------------------- the defect itself
def test_an_off_topic_in_character_line_is_refused():
    """The Cersei/Ned lines were REAL dialogue — being in-character was never the question."""
    ok, why, _ = _judge(ident=[_i(speaker="Cersei Lannister",
                                  scene="the godswood, talking to Ned about Robert")] * 3,
                        rel=[_b(belongs=False)] * 3,
                        beat_text="Canon calls the shard in his heart the Children's work.")
    assert ok is False
    assert "off-topic" in why and "Cersei" in why


def test_a_belonging_line_airs():
    ok, why, v = _judge(ident=[_i()] * 3, rel=[_b()] * 3,
                        beat_text="A White Walker put an ice sword through his stomach.")
    assert ok is True and "Benjen Stark" in why
    assert len(v) == 3


# --------------------------------------------------------------------------- unanimity
def test_one_dissenting_sample_blocks_the_breakout():
    """Measured: beat 156's Cersei line admitted on 2 of 6 single samples. Unanimity caught it —
    this gate must never be run at N=1."""
    ok, why, _ = _judge(ident=[_i()] * 3,
                        rel=[_b(belongs=True), _b(belongs=False), _b(belongs=True)],
                        beat_text="anything")
    assert ok is False
    assert "1/3" in why or "2/3" in why


def test_all_three_must_agree_to_admit():
    ok, _, _ = _judge(ident=[_i()] * 3, rel=[_b()] * 3, beat_text="x")
    assert ok is True


def test_one_dissenting_identification_also_blocks_it():
    """Unanimity applies to BOTH questions — a window two samples call dialogue and one calls an
    essayist is exactly the window that must not air."""
    ok, _, _ = _judge(ident=[_i(), _i(speaker="narrator", in_story=False), _i()],
                      rel=[_b()] * 3)
    assert ok is False


# --------------------------------------------------------------------------- no quote exemption
def test_a_located_quote_does_not_buy_past_the_relevance_judgment():
    """MEASURED REGRESSION, job 6a26707939 scene 18. The beat promised "I have seen the future in
    the flames."; find_quote_span matched it at phrase ratio 0.8 against audio that actually says
    "I don't know, your grace. I CAN'T see the future in the flames" — the negation. All three
    judges said belongs=false and the old exemption admitted it anyway. A fuzzy locator is not an
    author. Scene 123 was the same shape in the same render."""
    ok, why, _ = _judge(ident=[_i(speaker="Stannis Baratheon")] * 3,
                        rel=[_b(belongs=False, why="the line states the opposite")] * 3,
                        quote_authored=True,
                        promised_quote="I have seen the future in the flames.")
    assert ok is False, "a located quote must never override a unanimous not-belongs"
    assert "off-topic" in why


def test_a_genuine_quote_anchor_still_airs_on_merit():
    """Removing the exemption must not cost the case it was meant to protect: when the promised line
    really is the line, stage 2 is told in as many words that it belongs."""
    ok, why, _ = _judge(ident=[_i(speaker="Stannis Baratheon")] * 3, rel=[_b()] * 3,
                        quote_authored=True,
                        promised_quote="Death is the enemy. The first enemy and the last.")
    assert ok is True and "quote-anchored" in why


def test_a_confident_narrator_is_refused_quote_or_no_quote():
    """A rival essayist's voice-over must never air."""
    for anchored in (True, False):
        ok, why, _ = _judge(ident=[_i(speaker="narrator", in_story=False, conf=0.9)] * 3,
                            rel=[_b()] * 3, quote_authored=anchored)
        assert ok is False and "narrator" in why


def test_an_unlocated_quote_is_no_help_either():
    """Beat 112 carries "They drove a dragonglass dagger into my heart" — a real, on-topic,
    human-authored quote — and it locates in ZERO of 102 sources."""
    ok, _, _ = _judge(ident=[_i(speaker="Cersei Lannister")] * 3, rel=[_b(belongs=False)] * 3,
                      quote_authored=False,          # _span was None
                      promised_quote="They drove a dragonglass dagger into my heart.")
    assert ok is False


def test_the_call_site_records_the_span_but_does_not_exempt_on_it():
    src = inspect.getsource(B._select_breakouts)
    assert "quote_authored=bool(_span)" in src
    assert "quote_authored=(bool(_span) or not _relevance_on)" not in src, \
        "the located span must not be routed into an exemption again"
    assert "relevance_required=_relevance_on" in src, \
        "only the env kill-switch may drop the relevance half"


# --------------------------------------------------------------- garbled audio cannot be relevant
def test_unintelligible_audio_is_refused_not_reconstructed():
    """MEASURED, job 6a26707939 scene 113. The clip's ASR was "We'll use to hear it, cut some black.
    I will do us" and all three single-stage samples replied "Melisandre declares Jon Snow is the
    prince that was promised", belongs=true — a paraphrase of the BEAT they had been shown in the
    same prompt, not of anything in the audio."""
    ok, why, _ = _judge(ident=[_i(intelligible=False)] * 3, rel=[_b()] * 3,
                        beat_text="Melisandre believed Jon was the prince that was promised.",
                        promised_quote="The prince that was promised will bring the dawn.")
    assert ok is False
    assert "intelligible" in why


def test_one_mute_sample_does_not_veto_a_clip_the_others_can_place():
    """MEASURED FALSE NEGATIVE, job 6a26707939 beat 53. The one genuinely correct breakout in that
    render — "Stannis says he will risk everything" over "This is the right time, and I will risk
    everything ... we march to victory" — shares its window with ASR garble ("we kind of want to
    supply line some clears"), and one live sample in three called the whole thing unintelligible.
    Real ASR is partly garbled almost every time. A gate that only ever says no is not a fixed gate,
    so identification goes by majority; relevance below still needs all three."""
    ok, why, _ = _judge(ident=[_i(), _i(intelligible=False), _i()], rel=[_b()] * 3)
    assert ok is True, why


def test_a_mute_majority_still_refuses():
    ok, why, _ = _judge(ident=[_i(), _i(intelligible=False), _i(intelligible=False)],
                        rel=[_b()] * 3)
    assert ok is False and "intelligible" in why


def test_a_single_confident_narrator_reading_still_vetoes():
    """Identification is a majority everywhere EXCEPT here: a rival essayist's voice-over airing
    inside our own video is the one failure with no upside, so one confident call is enough."""
    ok, why, _ = _judge(ident=[_i(), _i(speaker="narrator", in_story=False, conf=0.9), _i()],
                        rel=[_b()] * 3)
    assert ok is False and "narrator" in why


def test_relevance_still_needs_every_sample_even_when_identification_was_split():
    ok, _, _ = _judge(ident=[_i(), _i(intelligible=False), _i()],
                      rel=[_b(), _b(belongs=False), _b()])
    assert ok is False


def test_a_sample_that_cannot_quote_the_audio_does_not_count_as_hearing_it():
    """The deterministic half of the anti-confabulation floor. A judge may only claim it can hear
    something if it can quote it, and code checks the quote against the actual transcript — no
    amount of prompt-following guarantees that on its own."""
    ok, why, _ = _judge(ident=[_i(line="the prince that was promised will bring the dawn")] * 3,
                        rel=[_b()] * 3)
    assert ok is False and "does not contain" in why


def test_quoting_only_a_word_or_two_is_not_enough_to_verify():
    ok, why, _ = _judge(ident=[_i(line="a real")] * 3, rel=[_b()] * 3)
    assert ok is False and "intelligible" in why


def test_a_grounded_quote_from_a_partly_garbled_window_still_counts():
    """Real windows mix garble and clear speech; only the clear part has to check out."""
    ok, _, _ = _judge(ident=[_i(line="spoken line of some")] * 3, rel=[_b()] * 3)
    assert ok is True


def test_stage_two_is_given_the_verified_line_not_just_the_raw_transcript():
    seen = []

    orig = (L.has_llm, L.complete)
    L.has_llm = lambda *a, **k: True

    def _complete(*, system="", messages=None, **k):
        u = (messages or [{}])[0].get("content", "")
        if "Identify it" in system:
            return _i(line="spoken line of some")
        seen.append(u)
        return _b()

    L.complete = _complete
    try:
        B._breakout_window_admissible(AIRED, "Game of Thrones", beat_text="x")
    finally:
        L.has_llm, L.complete = orig
    assert seen and all("CLEAREST LINE IN THE CLIP" in u for u in seen)


def test_stage_one_is_blind_to_the_beat():
    """The confabulation had one cause: the transcript and the answer were in the same prompt."""
    seen = {}

    orig = (L.has_llm, L.complete)
    L.has_llm = lambda *a, **k: True

    def _complete(*, system="", messages=None, **k):
        u = (messages or [{}])[0].get("content", "")
        if "Identify it" in system:
            seen.setdefault("ident", []).append(u)
            return _i()
        seen.setdefault("rel", []).append(u)
        return _b()

    L.complete = _complete
    try:
        B._breakout_window_admissible(
            "a real spoken line of some length", "Game of Thrones",
            beat_text="Melisandre believed Jon was the prince that was promised.",
            beat_subject="character: Melisandre",
            promised_quote="The prince that was promised will bring the dawn.")
    finally:
        L.has_llm, L.complete = orig

    assert seen["ident"], "stage 1 must run"
    for u in seen["ident"]:
        assert "prince that was promised" not in u, "stage 1 saw the answer it was meant to find"
        assert "Melisandre" not in u, "stage 1 saw the beat's subject"
    assert any("prince that was promised" in u for u in seen["rel"]), \
        "stage 2 must still see what the narration promised"
    assert any("CLIP IDENTIFIED AS" in u for u in seen["rel"]), \
        "stage 2 must be anchored on the blind identification, not just the raw transcript"


# --------------------------------------------------------------------------- fail-closed
@pytest.mark.parametrize("replies,has_llm,frag", [
    (["not json at all"] * 3, True, "parsed"),
    ([_i(), "garbage", _i()], True, "parsed"),
    ([_i()] * 3, False, "no LLM"),
    ([_i(conf=0.4)] * 3, True, "not identifiable"),
])
def test_every_uncertain_path_refuses_the_breakout(replies, has_llm, frag):
    ok, why, _ = _judge(replies=replies, has_llm=has_llm)
    assert ok is False and frag in why


def test_a_malformed_relevance_reply_refuses_the_breakout():
    """Stage 2 fails closed on its own — an unparseable belongs reply is not a yes."""
    ok, why, _ = _judge(ident=[_i()] * 3, rel=[_b(), "not json", _b()])
    assert ok is False and "parsed" in why


def test_too_short_is_refused_without_calling_the_provider():
    ok, why, v = B._breakout_window_admissible("ok", "Game of Thrones")
    assert ok is False and "too short" in why and v == []


def test_a_code_fault_is_raised_not_swallowed():
    """`recovery: skipped (NameError)` hid a dead stage for months. A fail-closed catch must never
    become the place a typo goes to die."""
    orig = (L.has_llm, L.complete)
    L.has_llm = lambda *a, **k: True

    def _boom(**k):
        raise NameError("typo_in_the_gate")

    L.complete = _boom
    try:
        with pytest.raises(NameError):
            B._breakout_window_admissible("a real spoken line here", "Game of Thrones")
    finally:
        L.has_llm, L.complete = orig


# --------------------------------------------------------------------------- wiring / rules
def test_relevance_half_has_its_own_kill_switch():
    src = inspect.getsource(B._select_breakouts)
    assert "VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_CHECK" in src
    assert "relevance_required=_relevance_on" in src, \
        "the kill switch must drop stage 2 only — identification always runs"


def test_the_kill_switch_drops_relevance_but_keeps_identification():
    ok, why, _ = _judge(ident=[_i()] * 3, rel=[_b(belongs=False)] * 3, relevance_required=False)
    assert ok is True and "identified" in why
    ok, _, _ = _judge(ident=[_i(speaker="narrator", in_story=False)] * 3,
                      rel=[_b()] * 3, relevance_required=False)
    assert ok is False, "even with relevance off, a rival essayist must never air"


def test_rejections_are_counted_and_persisted_for_offline_audit():
    src = inspect.getsource(B._select_breakouts)
    assert '"off_topic": 0' in src, "off-topic rejections need their own counter"
    assert '_rej["off_topic"] += 1' in src
    assert "admission_verdicts" in src, \
        "every verdict must reach breakout_audit.json so 'why did that air?' is answerable offline"


def test_the_gate_only_removes_breakouts():
    """Standing rule: shorten-only, never starve. The gate must `continue` past a bad candidate so
    the reserve can backfill — it must never abort selection or strand the beat."""
    src = inspect.getsource(B._select_breakouts)
    i = src.index('_rej["off_topic"] += 1')
    tail = src[i:i + 400]
    assert "continue" in tail
    assert "return" not in tail.split("continue")[0]


def test_the_judge_sees_the_narration_not_just_the_audio():
    """The whole point: the old gate could not have caught this, because it never saw the beat."""
    sig = inspect.signature(B._breakout_window_admissible)
    for p in ("beat_text", "beat_subject", "promised_quote", "quote_authored"):
        assert p in sig.parameters
    src = inspect.getsource(B._breakout_window_admissible)
    assert "NARRATION AT THIS MOMENT" in src


def test_an_unlocatable_promised_line_is_reported_not_swallowed():
    """A beat that promises a line whose footage does not exist is a SCRIPT/FOOTAGE gap the owner
    can act on. It used to fall through in silence — no log, no counter — which is why a Cersei
    conversation could air under Benjen's own "They drove a dragonglass dagger into my heart"
    (that quote locates in zero of the job's 102 word streams) and nobody could see why."""
    src = inspect.getsource(B._select_breakouts)
    i = src.index('phrase match {_qr})')
    tail = src[i:i + 1200]
    assert "elif _qtext:" in tail, "a promised line that does not locate must say so"
    assert "NOT spoken in" in tail
