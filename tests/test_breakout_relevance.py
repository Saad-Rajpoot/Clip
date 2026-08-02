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
def _judge(replies, *, has_llm=True, **kw):
    """Run the admission gate against a scripted list of provider replies (one per sample)."""
    seq = list(replies)
    calls = {"n": 0}
    orig = (L.has_llm, L.complete)
    L.has_llm = lambda *a, **k: has_llm

    def _complete(**k):
        i = calls["n"]
        calls["n"] += 1
        return seq[i] if i < len(seq) else seq[-1]

    L.complete = _complete
    try:
        return B._breakout_window_admissible(
            "a real spoken line of some length", "Game of Thrones", **kw)
    finally:
        L.has_llm, L.complete = orig


def _v(speaker="Benjen Stark", in_story=True, belongs=True, conf=0.95, scene="the Wall"):
    return ('{"speaker":"%s","scene":"%s","in_story":%s,"belongs":%s,"confidence":%s}'
            % (speaker, scene, str(in_story).lower(), str(belongs).lower(), conf))


@pytest.fixture(autouse=True)
def _three_samples(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_SAMPLES", "3")
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_MIN_CONF", raising=False)
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_CHECK", raising=False)


# --------------------------------------------------------------------------- the defect itself
def test_an_off_topic_in_character_line_is_refused():
    """The Cersei/Ned lines were REAL dialogue — being in-character was never the question."""
    ok, why, _ = _judge([_v(speaker="Cersei Lannister", belongs=False,
                            scene="the godswood, talking to Ned about Robert")] * 3,
                        beat_text="Canon calls the shard in his heart the Children's work.")
    assert ok is False
    assert "off-topic" in why and "Cersei" in why


def test_a_belonging_line_airs():
    ok, why, v = _judge([_v()] * 3, beat_text="A White Walker put an ice sword through his stomach.")
    assert ok is True and "Benjen Stark" in why
    assert len(v) == 3


# --------------------------------------------------------------------------- unanimity
def test_one_dissenting_sample_blocks_the_breakout():
    """Measured: beat 156's Cersei line admitted on 2 of 6 single samples. Unanimity caught it —
    this gate must never be run at N=1."""
    ok, why, _ = _judge([_v(belongs=True), _v(belongs=False), _v(belongs=True)],
                        beat_text="anything")
    assert ok is False
    assert "1/3" in why or "2/3" in why


def test_all_three_must_agree_to_admit():
    ok, _, _ = _judge([_v(), _v(), _v()], beat_text="x")
    assert ok is True


# --------------------------------------------------------------------------- the quote exemption
def test_a_located_quote_exempts_the_relevance_judgment():
    """When the beat's scripted quote was LOCATED in this very source, an author already chose this
    line for this beat — a 'does it belong' re-litigation would only add false negatives."""
    ok, why, _ = _judge([_v(belongs=False)] * 3, quote_authored=True,
                        promised_quote="Death is the enemy. The first enemy and the last.")
    assert ok is True and "quote-anchored" in why


def test_the_exemption_cannot_rescue_a_confident_narrator():
    """A rival essayist's voice-over must never air, quote or no quote."""
    ok, why, _ = _judge([_v(speaker="narrator", in_story=False, conf=0.9)] * 3,
                        quote_authored=True)
    assert ok is False


def test_an_unlocated_quote_gets_no_exemption():
    """THE SMOKING GUN. Beat 112 carries "They drove a dragonglass dagger into my heart" — a real,
    on-topic, human-authored quote — and it locates in ZERO sources. The exemption keys on the
    LOCATED span, never on seg.quote being non-empty, so the beat is still judged and its Cersei
    line is still refused."""
    ok, _, _ = _judge([_v(speaker="Cersei Lannister", belongs=False)] * 3,
                      quote_authored=False,          # _span was None
                      promised_quote="They drove a dragonglass dagger into my heart.")
    assert ok is False


def test_the_call_site_keys_the_exemption_on_the_span_not_the_quote():
    src = inspect.getsource(B._select_breakouts)
    assert "quote_authored=(bool(_span)" in src, \
        "the exemption must key on the LOCATED span; seg.quote being non-empty proves nothing"


# --------------------------------------------------------------------------- fail-closed
@pytest.mark.parametrize("replies,has_llm,frag", [
    (["not json at all"] * 3, True, "parsed"),
    ([_v(), "garbage", _v()], True, "parsed"),
    ([_v()] * 3, False, "no LLM"),
    ([_v(conf=0.4)] * 3, True, "off-topic"),
])
def test_every_uncertain_path_refuses_the_breakout(replies, has_llm, frag):
    ok, why, _ = _judge(replies, has_llm=has_llm)
    assert ok is False and frag in why


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
    assert "or not _relevance_on" in src, \
        "with the kill switch off every window must be treated as quote-authored (old behaviour)"


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
