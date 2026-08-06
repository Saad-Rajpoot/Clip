"""When the narration tells the viewer to LOOK at something, name the thing.

This is the defect the video's owner has raised more than any other: "jab VO mein dagger dekhne ko
kaha jata hai, toh screen par wohi aa raha hai kya". Measured on the v4 render, of the beats that
instruct the viewer to look, the named thing was clearly visible in 8, partly in 10, and absent in
12 — including both beats the essay turns on.

policy.is_deictic already existed but answers a different question: it spots a pointing determiner
bound to a scene noun ("that table") and only decides POLICY. It cannot say WHAT to look for.
`deictic_target` extracts the noun phrase itself so the verifier can require it on screen.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from vidlore.clipstudio import policy as P


def t(text):
    return P.deictic_target(NS(text=text))


# --------------------------------------------------------------- the real lines from the script
@pytest.mark.parametrize("line,want", [
    ("Keep your eye on the dagger in that room.", "the dagger"),
    ("And keep your eye on Bran's face while his sister reads out the charges.", "Bran's face"),
    ("Watch Bran's face while his sister reads out the charges.", "Bran's face"),
    ("To answer that, watch the trial the way Bran watched it.", "the trial"),
    ("Now go and stand in the room that crowns him. The Dragonpit. Count the chairs.",
     "the chairs"),
    ("Look at her hands.", "her hands"),
    ("Notice the blood on the floor.", "the blood"),
])
def test_extracts_the_named_target(line, want):
    assert t(line) == want


# --------------------------------------------------------------- what must NOT become a target
@pytest.mark.parametrize("line", [
    "But notice the division of labour.",          # abstract noun, nothing to point a camera at
    "Now watch his strategy in the next thirty seconds, because this is the tell.",
    "Now look at the shape of the lie he tells her.",  # figurative shape, not a visible prop
    "Watch it again — not as a fan, as a lawyer.",  # no named thing
    "That is the tragedy of it.",                   # bare anaphora
    "I'm not going to dwell on Baelish's face in that moment.",   # not an instruction to look
    "The point is what everyone else at that table heard.",
    "",
])
def test_rejects_lines_with_no_visual_target(line):
    assert t(line) == ""


# --------------------------------------------------------------- phrase boundaries
def test_phrase_is_cut_at_a_preposition():
    """Greedy capture turned 'the dagger in that room' into a phrase whose head was 'room'."""
    assert t("Keep your eye on the dagger in that room.") == "the dagger"


def test_phrase_is_cut_at_a_second_determiner():
    """'watch the trial the way Bran watched it' ran on to 'the way', whose head reads as abstract
    and lost the beat entirely."""
    assert t("watch the trial the way Bran watched it") == "the trial"


def test_possessive_proper_noun_starts_a_phrase():
    """'Bran's face' is the single most-requested case and has no article to anchor on."""
    assert t("Watch Bran's face.") == "Bran's face"
    assert t("Look at Sansa's hands.") == "Sansa's hands"


def test_trailing_punctuation_is_stripped():
    assert t("Count the chairs.") == "the chairs"
    assert t("Watch the dagger!") == "the dagger"


def test_target_is_bounded_in_length():
    long = "watch the very long and rather overspecified ceremonial dagger"
    assert len(t(long).split()) <= 4


def test_has_deictic_target_mirrors_the_extractor():
    assert P.has_deictic_target(NS(text="Watch Bran's face.")) is True
    assert P.has_deictic_target(NS(text="That is the tragedy.")) is False


def test_missing_text_attribute_is_safe():
    assert P.deictic_target(NS()) == ""


def test_is_deictic_still_answers_its_own_question():
    """The pre-existing scene-pointing detector must be untouched by this addition."""
    assert P.is_deictic(NS(text="what everyone else at that table heard")) is True
    assert P.is_deictic(NS(text="Watch Bran's face.")) is False, \
        "instructed looking is a separate signal — it must not silently re-classify policy"


# --------------------------------------------------------------- the gate that uses it
def test_verifier_is_told_what_to_look_for_and_keys_its_cache_on_it():
    """A verdict cached WITHOUT the look question answers a different question and must not be
    reused, or the gate goes silently inert on every warm render."""
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_frame)
    assert "must_see" in src and "target_visible" in src, \
        "the verifier must be asked whether the named target is on screen"
    fp = inspect.getsource(V.verdict_fingerprint)
    assert "must_see" in fp and 'parts.append("look:"' in fp, \
        "the verdict cache must key on the look target"


def test_a_missed_target_blocks_the_contextual_downgrade():
    """The downgrade keeps a clip because the required SUBJECT is present — which is exactly how
    'keep your eye on the dagger' ships over a dagger-less clip. When the narration names a target
    and the verifier could not see it, that shortcut must be closed."""
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_and_repair)
    assert "_look_missed" in src
    assert "if _look_missed and _orig_ok:" in src, \
        "a usable pick must be KEPT when the target is missing, not gambled on the bench"
    assert src.index("_look_missed = ") < src.index("_orig_ok = _exact_contextual_ok(v"), \
        "the look verdict must be computed before the downgrade decision"


def test_a_missed_target_never_throws_away_a_usable_pick():
    """Refusing the downgrade unconditionally measured -4.00: beats scoring 8/9/9 fell to 4/2/3
    because the replacement was worse than what it discarded, and the deictic target ended up LESS
    visible. The owner's rule is explicit — "agar koi exact scene dastyab na ho, to wo uski zid na
    kare". So: look harder, then settle, and let the still pass cover the moment."""
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_and_repair)
    i = src.index("if _look_missed and _orig_ok:")
    j = src.index("elif _look_missed:")
    branch = src[i:j]
    assert "_deep_bench()" not in branch, \
        "a usable pick must never be routed to the bench — measured -4.00 when it was"
    assert "_keep_contextual(" in branch, "the usable pick must be kept"
    # the beat still gets covered: the flag routes it to the still pass
    assert "look_target_missing" in src


def test_look_gate_has_a_kill_switch():
    import inspect
    from vidlore.clipstudio import verify as V
    assert "VIDLORE_CLIPSTUDIO_LOOK_GATE" in inspect.getsource(V.verify_and_repair)


# --------------------------------------------------------------- era on the fallback path
def test_contextual_fallback_refuses_a_wrong_era_clip():
    """"Guzara kar le" must still mean a SENSIBLE substitute. The verifier is told a different
    season is wrong even when the character matches, but that only drove `verdict`; the contextual
    downgrade then re-admitted the clip on "the subject is visible", which is how season-1 child
    Bran shipped under season-8 Dragonpit lines (16-18 wrong_era beats on the frame eval)."""
    from vidlore.clipstudio.verify import _contextual_subject_ok as ok
    assert ok({"correct_subject_visible": True}) is True
    assert ok({"correct_subject_visible": True, "era_ok": True}) is True
    assert ok({"correct_subject_visible": True, "era_ok": False}) is False, \
        "a clearly wrong-era clip is not a legitimate contextual fallback"
    assert ok({"matches_narration": True, "era_ok": False}) is False


def test_a_verdict_cached_before_era_ok_still_passes():
    """Reading era_ok as 'not disproven' lets the gate tighten on fresh verdicts without
    invalidating a whole render's verdict cache (~$1 of vision calls)."""
    from vidlore.clipstudio.verify import _contextual_subject_ok as ok
    assert ok({"correct_subject_visible": True}) is True          # no era_ok key at all
    assert ok({"correct_subject_visible": True, "era_ok": None}) is True


def test_era_ok_is_only_asked_when_the_beat_declares_an_era():
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_frame)
    assert '\'"era_ok": true/false, \' if era_hint else ""' in src.replace('"', "'").replace("''", "'") \
        or "if era_hint else" in src, "era_ok must not be demanded when there is no era to check"


def test_a_missed_look_target_routes_the_beat_to_a_still():
    """If nothing in the pool shows the named thing, a screenshot OF THAT MOMENT beats moving
    footage of the wrong one — the owner's own rule. But the flag must survive the trip: verify
    records it, the still pass reads it."""
    import inspect
    from vidlore.clipstudio import verify as V, orchestrate as O
    vsrc = inspect.getsource(V.verify_and_repair)
    assert '"look_target_missing"' in vsrc, "verify must flag the miss"
    osrc = inspect.getsource(O)
    assert '"look_target_missing" in (sel.flag_reasons or [])' in osrc, \
        "the still pass must read the flag"
    assert "or weak or look_missed)" in osrc, \
        "a missed look target must make the beat want a still"


def test_the_look_flag_is_not_duplicated_on_re_verify():
    """verify_and_repair can run twice (recovery re-verifies); the flag list must not grow."""
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_and_repair)
    assert 'if "look_target_missing" not in (sel.flag_reasons or []):' in src


def test_no_blanket_luma_floor_was_added():
    """The pool IS this dark — median luma 31.9, and 31.1% of its 4688 shots sit under luma 20.
    The v4 picks (32% under 20) mirror it exactly, so a floor at 20 would delete a third of the
    pool and push beats away from the candle-lit trial the essay is actually about. Darkness that
    hides the SUBJECT is handled by _shot_unreadable, the moment-lock damper and the look gate."""
    import inspect
    from vidlore.clipstudio import match as M
    src = inspect.getsource(M)
    assert "LUMA_FLOOR" not in src.upper().replace("_MOMENT_LUMA_OK", ""), \
        "a blanket luma floor would cost more relevance than it buys on this material"


def test_the_look_question_is_preserved_for_strict_replacements_only():
    """A strict replacement inherits "show the dagger" or it can stop on a generic keep and leave
    a later publication blocker even when a subsequent candidate contains the target.  The softer
    contextual rung still disables the question; if strict search finds nothing, the existing
    usable-pick/still path remains the no-footage fallback instead of gambling on a replacement."""
    import inspect
    from vidlore.clipstudio import verify as V
    src = inspect.getsource(V.verify_and_repair)
    assert '_look_scope = {"on": True}' in src
    assert 'if not _look_scope["on"]:' in src, "_must_see must honour the scope"
    i = src.index("def _try_promote(")
    j = src.index("def _try_promote_inner(")
    wrapper = src[i:j]
    assert '_look_scope["on"] = not downgrade' in wrapper and "finally:" in wrapper, \
        "strict promotion must keep the look question; contextual promotion may soften it"
    # the strict per-window decision now lives in `strict_window_verdict`, called from
    # `_try_promote_inner` — one judge, so look for the invariant where it is defined
    assert 'av.get("target_visible") is True' in inspect.getsource(V.strict_window_verdict), \
        "a malformed keep without affirmative target evidence must not stop strict search"
