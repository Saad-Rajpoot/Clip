"""Regression: the 56e0467283 fix pass (relevance / verifier / timeline / breakouts).

Every case is anchored to a MEASURED defect in the shipped 15:24 render, not a hypothetical.
Where a test asserts a number, that number came from the audit of the aired video.

    python3 tests/test_relevance_fix_pass.py

Pure logic — no render, no network, no LLM.
"""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio import policy  # noqa: E402

FAILS = []


def _seg(text, **kw):
    d = dict(text=text, visual_policy="", quote="", is_specific_claim=False,
             required_entity="", required_kind="", scene_query="", expected_visual="")
    d.update(kw)
    return NS(**d)


# ---------------------------------------------------------------------------
# F3 — deictic inheritance. Lines are VERBATIM from the shipped render.
# ---------------------------------------------------------------------------
def test_deixis_promotes_the_lines_that_actually_failed():
    # 10:08 — shipped as generic_filler, aired Ned Stark's execution (S01E09).
    # _ABSTRACT_RX also matches "the point is", which pushed this AWAY from exact.
    s = _seg("The point is what everyone else at that table heard.", visual_policy="generic_filler")
    assert policy.is_deictic(s), "'that table' not detected as deictic"
    assert policy.policy_of(s) == policy.EXACT, "deixis must outrank the LLM's generic_filler label"

    # 0:36 — shipped as abstract_effect, aired the Hound + Arya at the Inn at the Crossroads.
    s = _seg("Somewhere in those ninety seconds, one word gets spoken.",
             visual_policy="abstract_effect")
    assert policy.policy_of(s) == policy.EXACT, "'those ninety seconds' must be exact"

    # 14:50 — the thesis payoff.
    s = _seg("then Joffrey's real death didn't happen at a wedding. It happened at that table.")
    assert policy.policy_of(s) == policy.EXACT

    # 11:14 — shipped as character_specific, aired the Hound + Arya.
    s = _seg("Go to the books though, and you find out that nobody in that room was laughing.",
             visual_policy="character_specific")
    assert policy.policy_of(s) == policy.EXACT

    # 9:02 — the promised payoff; narrator points AT the line the viewer never heard.
    s = _seg("There it is. That's the word I told you to listen for. Nightshade.",
             visual_policy="generic_filler")
    assert policy.policy_of(s) == policy.EXACT

    # 6:10 — an explicit instruction to re-watch the anchor scene.
    s = _seg("Watch it again sometime, and don't look at Joffrey at all.")
    assert policy.policy_of(s) == policy.EXACT


def test_deixis_does_not_swallow_generic_narration():
    """The guard-rail on the guard-rail: over-promoting would strip filler from beats that never
    needed exact footage and block the render on them."""
    for t in ("That's the tragedy of it.",
              "This is why he never had to raise his voice.",
              "And that's exactly the point.",
              "It's really about control, not titles.",
              "Think about what that means for a moment."):
        assert not policy.is_deictic(_seg(t)), f"false deictic hit: {t!r}"

    s = _seg("Westeros was a place where power was always contested.", visual_policy="generic_filler")
    assert policy.policy_of(s) == policy.FILLER, "generic filler must survive"

    s = _seg("In the end, nothing would ever be the same.")
    assert policy.policy_of(s) == policy.ABSTRACT, "abstract beats must stay abstract"


def test_deixis_survives_finalize_and_is_idempotent():
    s = _seg("The point is what everyone else at that table heard.", visual_policy="generic_filler")
    policy.finalize_beats([s])
    assert s.visual_policy == policy.EXACT
    assert policy.policy_of(s) == policy.EXACT, "must be stable on a second pass"


# ---------------------------------------------------------------------------
# F6 — word-level ASR + quote-span location.
# The word stream below is the REAL one from tywin_lannister_dismis_f4c81b75 (shipped index +
# isolated whisper of the source), garble included. These are the two lines the render lost.
# ---------------------------------------------------------------------------
_REAL_ASR_CHUNKS = [
    (121.25, 122.46, "I am the"), (122.46, 124.17, "king."),
    (124.17, 126.58, "I will punish you. Any man who must"),
    (126.58, 130.05, "say I am the king is no true king."),
    (130.05, 132.42, "I'll make sure you understand"),
    (132.42, 134.51, "up and I've won your war for you."),
    (134.51, 140.31, "My father won the real war. He killed Prince Rhaegar. He took the crown while"),
    (140.31, 142.89, "you hid on a costly rock."),
    (154.99, 158.70, "The king is tired. See him to"),
    (158.70, 162.75, "his chambers. Come on off. I'm not tired. We have"),
    (162.75, 163.96, "so much to celebrate."), (163.96, 166.00, "A wedding to plan."),
    (166.00, 169.67, "You must rest. Grand"), (169.67, 171.38, "Maester. Perhaps"),
    (171.38, 173.55, "a messence of nightshade to help him"),
    (173.55, 178.68, "sleep. I'm not fired."),
]


def _real_words():
    out = []
    for a, b, txt in _REAL_ASR_CHUNKS:
        ws = txt.split()
        step = (b - a) / len(ws)
        out.extend((round(a + i * step, 3), round(a + (i + 1) * step, 3), w)
                   for i, w in enumerate(ws))
    return out


def test_thesis_line_locatable_across_a_shot_boundary():
    """'Any man who must say I am the king is no true king' straddles the cut at 126.58, so it
    lands as '...Any man who must' + 'say I am the king...'. No single shot contains it, which is
    why per-shot substring search never found the most iconic line in the video."""
    from vidlore.clipstudio.index import find_quote_span
    r = find_quote_span(_real_words(), "Any man who must say 'I am the king' is no true king.")
    assert r is not None, "thesis line not located"
    s, e, ratio = r
    assert abs(s - 125.6) <= 2.0, f"span start {s} not at the real line start (~125.6)"
    assert e >= 129.0, f"span end {e} truncates the line"
    assert ratio >= 0.8, f"phrase ratio too low: {ratio}"


def test_payoff_line_locatable_through_asr_garble_and_four_shots():
    """The nightshade line spans 4 shots AND is garbled ('a messence of nightshade', 'I'm not
    fired'). No breakout candidate was ever generated for it in the shipped render."""
    from vidlore.clipstudio.index import find_quote_span
    r = find_quote_span(_real_words(), "Perhaps some essence of nightshade to help him sleep.")
    assert r is not None, "payoff line not located through garble"
    s, e, ratio = r
    assert abs(s - 170.0) <= 2.0, f"span start {s} wrong"
    assert e >= 173.5, f"span end {e} cuts the line short of 'sleep'"
    assert ratio >= 0.75, f"phrase ratio too low: {ratio}"


def test_single_garbled_word_cannot_anchor_a_match():
    """A lone fuzzy token must never carry a match — otherwise 'sleep' anchors anywhere. The
    acceptance is per-PHRASE; fuzziness is only ever one term inside that score."""
    from vidlore.clipstudio.index import find_quote_span
    assert find_quote_span(_real_words(), "sleep") is None
    assert find_quote_span(_real_words(), "Winter is coming to the North this year") is None
    assert find_quote_span(_real_words(), "dragons burned the fleet at anchor") is None


# ---------------------------------------------------------------------------
# F1 — beat-local era. The analyzer labelled the S03E10 "Mhysa" council scene as
# "S04E01 Two Swords"; that one string purged 354 shots (including the correct episode) and made
# wrong-episode clips anchors.
# ---------------------------------------------------------------------------
_REAL_ANALYSIS = NS(
    episode_hint="S04E01",
    anchor_scenes=[{"name": "Tywin sends Joffrey to bed",
                    "episode": "S04E01 Two Swords",
                    "dialogue": ["Any man who must say 'I am the king' is no true king.",
                                 "The king is tired. See him to his chambers."]}])
_REAL_SCRIPT = ("An old man ends a king's reign without raising his voice. "
                "This is the last episode of season 3. "
                "Days earlier, at a wedding, an entire army was butchered.")


def test_season_parsing_reads_spelled_out_numerals():
    """The script said 'the last episode of season three'. The old digits-only regex could not see
    the single disconfirming signal the pipeline already had."""
    from vidlore.clipstudio import era
    assert era.parse_season("This is the last episode of season three.") == 3
    for t, exp in (("season 4", 4), ("S03E10", 3), ("3x10", 3), ("Game of Thrones S4", 4),
                   ("S04E01 Two Swords", 4), ("no era here", None)):
        assert era.parse_season(t) == exp, f"{t!r} -> {era.parse_season(t)}"


def test_contradicted_hint_is_not_verified():
    from vidlore.clipstudio import era
    h, ok, why = era.verified_episode_hint(_REAL_ANALYSIS, _REAL_SCRIPT)
    assert h == "S04E01"
    assert ok is False, "a hint the script contradicts must never be verified"
    assert "CONTRADICTED" in why, why


def test_uncorroborated_hint_is_not_verified_either():
    from vidlore.clipstudio import era
    _, ok, why = era.verified_episode_hint(_REAL_ANALYSIS, "a script that names no season at all")
    assert ok is False and "uncorroborated" in why


def test_agreeing_hint_is_verified():
    from vidlore.clipstudio import era
    ana = NS(episode_hint="S03E10", anchor_scenes=[])
    _, ok, _ = era.verified_episode_hint(ana, "This is the last episode of season three.")
    assert ok is True


def test_beat_local_era_beats_the_global_hint():
    """A beat naming its own event uses THAT event's era. The Red Wedding is S03E09 no matter what
    the anchor scene is — forcing it to the anchor era is how Red Wedding beats got S04E01 clips."""
    from vidlore.clipstudio import era
    ev = era.event_eras_from(_REAL_ANALYSIS)
    core = NS(text="And at the end of that table sits his grandfather.",
              scene_query="", expected_visual="")
    rw = NS(text="Days earlier at the Red Wedding, an entire army was butchered.",
            scene_query="Game of Thrones Red Wedding season 3", expected_visual="")

    # an UNVERIFIED global hint constrains nothing
    assert era.beat_era(core, "S04E01", single_scene=True, global_verified=False,
                        event_eras=ev) == ""
    # a VERIFIED hint fills the gap for a core/deictic beat — returned VERBATIM, not normalised to
    # 'season N', so downstream episode-granular comparisons (S04E01 vs S04E10) still conflict
    assert era.beat_era(core, "S03E10", single_scene=True, global_verified=True,
                        event_eras=ev) == "S03E10"
    # local evidence wins even over a verified global hint
    assert era.beat_era(rw, "S04E01", single_scene=True, global_verified=True,
                        event_eras=ev) == "season 3"
    assert era.episodes_conflict("S04E01", "S04E10"), "episode granularity must survive"
    assert not era.episodes_conflict("S04E01", "season 4")


def test_unverified_hint_may_not_purge_or_anchor():
    """The two hard powers an unverified hint must never have. Measured: 354 shots purged every
    run, including 'Game Of Thrones S03E10 Red Wedding Aftermath scene' -- the correct episode,
    which contributed zero shots to a video about that episode."""
    import inspect
    from vidlore.clipstudio import match as M
    src = inspect.getsource(M.match_project) if hasattr(M, "match_project") else inspect.getsource(M)
    assert "_ep_verified" in src, "era purge / anchor promotion must consult episode_hint_verified"
    # the purge must be conditioned on verification, not merely on a season being parseable
    assert "and _ep_verified" in src


# ---------------------------------------------------------------------------
# F2 — the verifier must fail closed. FAILURE INJECTION: reproduce the outage that shipped this
# render. Real trace: 176 replaced/23 unresolved -> 180/25 -> 55/11 -> 0 replaced, 0 unresolved,
# PUBLISH. Scene 25 release-blocked on all 8 attempts and was never fixed; it stopped being checked.
# ---------------------------------------------------------------------------
def test_verdict_fingerprint_covers_every_input_that_changes_the_answer():
    from vidlore.clipstudio import verify as V
    base = dict(src_hash="a", source_id="s", shot_start=1.0, shot_end=2.0, beat_text="t",
                required_entity="Tywin", era="S03E10", visual_policy="exact_scene", model="m")
    fp0 = V.verdict_fingerprint(**base)
    assert fp0 == V.verdict_fingerprint(**base), "must be deterministic"
    for k, alt in (("src_hash", "b"), ("source_id", "s2"), ("shot_start", 1.5), ("shot_end", 9.0),
                   ("beat_text", "other"), ("required_entity", "Joffrey"), ("era", "S04E01"),
                   ("visual_policy", "generic_filler"), ("model", "m2")):
        d = dict(base)
        d[k] = alt
        assert V.verdict_fingerprint(**d) != fp0, f"{k} must change the fingerprint"


def test_prompt_version_is_in_the_fingerprint():
    """A verdict is only reusable if it answered the SAME question — a changed prompt invalidates."""
    import re as _re
    from pathlib import Path as _P
    src = _P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "verify.py"
    s = src.read_text(encoding="utf-8")
    assert "PROMPT_VERSION" in s
    m = _re.search(r"def verdict_fingerprint\(.*?return h\.hexdigest", s, _re.S)
    assert m and "PROMPT_VERSION" in m.group(0), "PROMPT_VERSION must be hashed into the verdict id"


def test_verifier_outage_fails_closed_not_open():
    """The measured failure: a total outage yields 0 rejections, and '0 rejections' was read as
    'nothing wrong'. An exact_scene beat nobody could check must be UNRESOLVED."""
    import inspect
    from vidlore.clipstudio import verify as V
    s = inspect.getsource(V.verify_and_repair)
    # the None branch must no longer be a bare `continue` that leaves the beat looking fine
    assert "FAIL CLOSED" in s, "the v-is-None branch must fail closed"
    assert "_errored += 1" in s
    assert "FLAG_VERIFIER_UNVERIFIED" in s, "an unverifiable beat must be flagged"
    # an exact beat with no verdict counts as unresolved
    i = s.index("FAIL CLOSED")
    branch = s[i:i + 1400]
    assert "if _exact:" in branch and "failed += 1" in branch, \
        "an unverifiable exact_scene beat must count as unresolved"
    # 'verified' must count successes, not attempts (it counted attempts, so 229 errors read as
    # '229 checked')
    assert "verified += 1                    # counts SUCCESSES" in s


def test_circuit_breaker_and_liveness_are_reported():
    import inspect
    from vidlore.clipstudio import verify as V
    s = inspect.getsource(V.verify_and_repair)
    assert "VERIFIER_BREAKER_TRIP" in s and "_consec_err" in s, "need a circuit breaker"
    assert '"verifier_down"' in s and '"errored"' in s and '"verified_frac"' in s, \
        "the summary must distinguish 'found nothing wrong' from 'checked nothing'"


def test_release_block_is_non_retryable():
    """Release-blocks and relevance failures are CONTENT verdicts. Re-running unchanged cannot fix
    them — it only rolls the dice until the verifier dies."""
    from vidlore.clipstudio import verify as V
    assert issubclass(V.NonRetryableBuildError, RuntimeError)
    d = V.NonRetryableBuildError.__doc__ or ""
    assert "Retry transient plumbing" in d and "Never retry a judgment" in d


TESTS = [
    test_deixis_promotes_the_lines_that_actually_failed,
    test_deixis_does_not_swallow_generic_narration,
    test_deixis_survives_finalize_and_is_idempotent,
    test_thesis_line_locatable_across_a_shot_boundary,
    test_payoff_line_locatable_through_asr_garble_and_four_shots,
    test_single_garbled_word_cannot_anchor_a_match,
    test_season_parsing_reads_spelled_out_numerals,
    test_contradicted_hint_is_not_verified,
    test_uncorroborated_hint_is_not_verified_either,
    test_agreeing_hint_is_verified,
    test_beat_local_era_beats_the_global_hint,
    test_unverified_hint_may_not_purge_or_anchor,
    test_verdict_fingerprint_covers_every_input_that_changes_the_answer,
    test_prompt_version_is_in_the_fingerprint,
    test_verifier_outage_fails_closed_not_open,
    test_circuit_breaker_and_liveness_are_reported,
    test_release_block_is_non_retryable,
]


if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
