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


def test_build_blocks_on_unverified_exact_beats():
    """The 56e0467283 scenario exactly: 229 beats errored -> 0 rejections -> the rejected-footage
    gate found nothing to block -> published, with 178 exact_scene beats whose relevance_class was
    literally 'unverified'. A vision outage must not be indistinguishable from a clean pass."""
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "UNVERIFIED-EXACT GATE" in s, "need a gate for beats nobody could check"
    i = s.index("UNVERIFIED-EXACT GATE")
    g = s[i:i + 2200]
    assert '("error", "unavailable")' in g, "must catch both error and unavailable"
    assert "verify_strict" in g, "must apply to exact_scene beats"
    assert "NonRetryableBuildError" in g, "an unverifiable render must not invite a blind retry"
    assert "image_path" in g, "a validated still legitimately covers the beat"
    # and it must be a SEPARATE gate from rejected-footage: unverified != proven wrong, so it must
    # not be freeze-replaced
    assert "must not be freeze-replaced" in g


def test_rejected_footage_gate_is_non_retryable_too():
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    i = s.index("rejected-footage gate:")
    assert "NonRetryableBuildError" in s[max(0, i - 900):i + 200], \
        "the release-block must be non-retryable"


# ---------------------------------------------------------------------------
# F5 — the transition clock. 46 transitions planned, 44 emitted; the 2 dropped left their padding
# in the stream = 1.200s, exactly concat(925.100) - final(923.900), and the 1.13/1.18s by which
# both breakout pictures lagged their own audio.
# ---------------------------------------------------------------------------
def _select_pairs(cands):
    """Mirror of the selection in assemble.py (pass 2). Kept in the test so the RULE is pinned even
    if the surrounding code is refactored."""
    taken, out = set(), {}
    for bi in sorted(cands):
        if bi in taken or (bi + 1) in taken:
            continue
        out[bi] = cands[bi]
        taken.update((bi, bi + 1))
    return out


def test_adjacent_transition_tails_cannot_both_pad():
    """The bug: beats 1 and 2 are BOTH motivated tails. Greedy merge emits trans_1 (consuming beats
    1+2) and skips beat 2 — but beat 2 was already padded at plan time, ffmpeg's xfade copies input
    2 whole, and trans_2 is never emitted. Nothing consumes that pad."""
    cands = {1: (0.55, "fade"), 2: (0.55, "fade"), 5: (0.6, "fadeblack")}
    sel = _select_pairs(cands)
    assert 1 in sel and 2 not in sel, "beat 2 is consumed by trans_1; it must not also start one"
    assert 5 in sel, "a non-colliding candidate is unaffected"
    # every selected pair is disjoint → every pad is consumed by exactly one xfade
    spans = [(bi, bi + 1) for bi in sel]
    flat = [x for sp in spans for x in sp]
    assert len(flat) == len(set(flat)), "pairs must not overlap"


def test_pads_are_derived_from_selected_pairs_only():
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    i = s.index("PASS 1 — collect CANDIDATE tails")
    blk = s[i:i + 3000]
    assert "No padding is committed here" in blk
    assert "_cand_tails[bi] = (xf, name, styj)" in blk, "pass 1 must only collect"
    assert "for bi, (xf, _n) in trans_tails.items():" in blk and "beat_pad[bi] = xf" in blk, \
        "pads must be derived from the SELECTED pairs, after the pairing is resolved"
    # the pad must NOT be set inside the candidate loop any more
    p1 = blk[:blk.index("PASS 2")]
    assert "beat_pad[bi] = xf" not in p1, "pass 1 must not commit padding"


def test_frame_allocation_carries_so_drift_cannot_accumulate():
    """The SECOND drift source, found by the sync invariant during the acceptance render (the
    pairing fix was necessary but not sufficient): each scene snapped to its OWN frame count, so
    every scene contributed up to half a frame of independent error and ~43 of them random-walked
    to +0.227s -- video 188.467s vs composed audio 188.240s.

    Allocating against the RUNNING clock makes the total exact by construction: each scene absorbs
    its predecessor's remainder instead of starting a fresh one."""
    import random
    FPS = 30.0

    def old(groups):
        out = []
        for g in groups:
            tot = max(len(g), int(round(sum(g) * FPS)))
            fr = [max(1, int(round(x * FPS))) for x in g]
            k = fr.index(max(fr))
            fr[k] = max(1, fr[k] + (tot - sum(fr)))
            out += [f / FPS for f in fr]
        return out

    def new(groups):
        out, emitted, cum = [], 0, 0.0
        for g in groups:
            cum += sum(g)
            tot = max(len(g), int(round(cum * FPS)) - emitted)
            fr = [max(1, int(round(x * FPS))) for x in g]
            k = fr.index(max(fr))
            fr[k] = max(1, fr[k] + (tot - sum(fr)))
            emitted += sum(fr)
            out += [f / FPS for f in fr]
        return out

    random.seed(7)
    one, worst_old, worst_new = 1 / FPS, 0.0, 0.0
    for _ in range(200):
        groups = [[random.uniform(0.8, 6.0) for _ in range(random.randint(1, 3))]
                  for _ in range(random.randint(35, 60))]
        tgt = sum(sum(g) for g in groups)
        worst_old = max(worst_old, abs(sum(old(groups)) - tgt))
        worst_new = max(worst_new, abs(sum(new(groups)) - tgt))
    assert worst_old > one, "the old allocator must demonstrably breach one frame (it did: +227ms)"
    assert worst_new <= one + 1e-9, f"carry must bound TOTAL drift under a frame, got {worst_new}"

    # and the shipped code must actually carry
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    assert "CARRY THE ROUNDING" in s
    i = s.index("CARRY THE ROUNDING")
    blk = s[i:i + 1800]
    assert "int(round(_cum_t * FPS)) - _emitted_f" in blk, "frames must derive from the running clock"
    assert "_emitted_f += sum(_fr)" in blk


def test_video_total_conforms_to_the_narration_clock():
    """The fifth and deepest drift source: the carry snaps beat_durs to the SCENE durations, but
    those diverge from the composed narration the viewer hears by a sub-frame per scene, and over
    ~40 scenes that reached +0.127s -- identical across four renders because it is neither the
    cold-open nor the pads. The audio is the authority, so the whole-frame delta is absorbed into
    the longest beat; the totals then match by construction."""
    FPS = 30.0

    def conform(beat_durs, narr_total):
        tgt, cur = int(round(narr_total * FPS)), int(round(sum(beat_durs) * FPS))
        if beat_durs and tgt != cur:
            k = max(range(len(beat_durs)), key=lambda i: beat_durs[i])
            adj = (tgt - cur) / FPS
            if beat_durs[k] + adj >= 0.2:
                beat_durs[k] = round((beat_durs[k] + adj) * FPS) / FPS
        return beat_durs

    import random
    random.seed(3)
    worst = 0.0
    for _ in range(300):
        n = random.randint(30, 60)
        bd = [round(random.uniform(0.5, 6.0) * FPS) / FPS for _ in range(n)]
        narr = sum(bd) + random.uniform(-0.2, 0.2)     # narration differs sub-frame per scene
        out = conform(list(bd), narr)
        worst = max(worst, abs(sum(out) - round(narr * FPS) / FPS))
    assert worst < 1e-6, f"conform must make video total == narration exactly, off by {worst}"

    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    assert "CONFORM THE VIDEO TO THE COMPOSED-AUDIO CLOCK" in s
    # must target the composed-audio FILE the invariant probes (not narration.total, which equals
    # the scene sum the video already has — that was a silent no-op), and absorb into a real beat,
    # not the word-less breakout pseudo-scene
    assert "_probe_duration(_audio_f)" in s, "conform must probe the composed audio file"
    assert '_is_breakout_beat' in s and 'getattr(_sc, "words"' in s, \
        "conform must exclude the word-less breakout beat"


def test_sync_invariant_compares_against_composed_audio():
    """'video == composed audio, ±1 frame'. Comparing against the RAW pre-breakout narration would
    be meaningless — the composed track is the clock the captions and breakout splices key to."""
    from pathlib import Path as _P
    from vidlore import assemble as A
    assert issubclass(A.TimelineSyncError, RuntimeError)
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "assemble.py").read_text(encoding="utf-8")
    assert "_assert_video_audio_sync(video_only, narration, workdir)" in s, "invariant must be wired"
    i = s.index("def _assert_video_audio_sync")
    fn = s[i:i + 1800]
    assert "tol_frames: float = 1.0" in fn and "tol = tol_frames / float(FPS)" in fn
    assert "never the raw" in fn, "must document that the reference is the COMPOSED audio"
    # a constant offset must not be offered as a fix — the drift is variable
    d = A.TimelineSyncError.__doc__ or ""
    assert "variable" in d and "second bug" in d


# ---------------------------------------------------------------------------
# F4 — positive evidence. 121 exact beats were downgraded to contextual and kept because no WRONG
# character was confirmed — vacuously true, since Face-ID had no reference for Joffrey, Varys or
# Pycelle. Absence of a wrong face is not presence of the right one.
# ---------------------------------------------------------------------------
_C2A = {"joffrey baratheon": "Jack Gleeson", "sandor clegane": "Rory McCann"}
_SEG_J = NS(required_entity="Joffrey Baratheon", required_kind="character",
            text="Joffrey comes apart.", scene_query="", expected_visual="")
_TITLE = "Game of Thrones Season 3 council"


def test_empty_faceid_is_unknown_not_innocent():
    """The shipped rule: 'no confirmed wrong character' + right era -> keep. With the leads
    unidentifiable that was satisfied by every frame in existence."""
    from vidlore.clipstudio import verify as V
    got = V._present_unconfirmed_ok({"correct_subject_visible": False}, _SEG_J, _TITLE, [],
                                    "season 3", frozenset(), char2actor=_C2A)
    assert got is False, "an empty Face-ID must be UNKNOWN, never a pass"


def test_positive_faceid_confirmation_allows_the_contextual_downgrade():
    from vidlore.clipstudio import verify as V
    got = V._present_unconfirmed_ok({"correct_subject_visible": False,
                                     "matches_narration": True, "specific_enough": True},
                                    _SEG_J, _TITLE,
                                    ["jack gleeson"], "season 3", frozenset(), char2actor=_C2A)
    assert got is True, ("Face-ID placing the entity plus an affirmative narration/specificity "
                         "judgment IS positive evidence")


def test_wrong_subject_visible_is_a_hard_rejection():
    from vidlore.clipstudio import verify as V
    got = V._present_unconfirmed_ok({"correct_subject_visible": False, "wrong_subject_visible": True},
                                    _SEG_J, _TITLE, ["jack gleeson"], "season 3", frozenset(),
                                    char2actor=_C2A)
    assert got is False, "wrong_subject_visible=true must reject even with the right face present"


def test_a_different_identified_person_still_blocks():
    from vidlore.clipstudio import verify as V
    got = V._present_unconfirmed_ok({"correct_subject_visible": False}, _SEG_J, _TITLE,
                                    ["rory mccann"], "season 3", frozenset(), char2actor=_C2A)
    assert got is False


def test_correct_actor_face_is_not_a_wrong_character():
    """Latent bug that fixing the Face-ID reference builder would have ACTIVATED: Face-ID reports
    ACTORS, beats name CHARACTERS. Without the roster a perfect Joffrey frame ('jack gleeson')
    reads as a confirmed WRONG character. It never bit only because Face-ID resolved nobody."""
    from vidlore.clipstudio import verify as V
    assert V._confirmed_wrong_character(_SEG_J, ["jack gleeson"], frozenset(), _C2A) is False
    assert V._confirmed_wrong_character(_SEG_J, ["rory mccann"], frozenset(), _C2A) is True


def test_faceid_recovers_oversized_reference_stills():
    """YuNet is fixed-scale: 3858x4804 -> 0 faces, the same photo at <=1024 -> 1. Three of eight
    leads were unidentifiable this way, including Joffrey."""
    from pathlib import Path as _P
    from vidlore.clipstudio import faceid as F
    assert F.DET_MAX_SIDE >= 512
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "faceid.py").read_text(
        encoding="utf-8")
    i = s.index("def _detect(self, img)")
    fn = s[i:i + 1600]
    assert "_detect_at(img)" in fn, "must try native size FIRST so working cases cannot regress"
    assert "DET_MAX_SIDE" in fn and "INTER_AREA" in fn, "must retry downscaled on a miss"
    assert "faces[:, :14] /= s" in fn, "boxes+landmarks must map back to original coordinates"


# ---------------------------------------------------------------------------
# F7 — quote-anchored breakout windows.
# ---------------------------------------------------------------------------
def test_breakout_window_contains_the_complete_iconic_line():
    """Shipped: breakout #1 cut at source 130.00 -- the shot boundary -- 0.36s AFTER the thesis
    line ended at 129.64, with a forward-only window, so the line was unreachable by construction.
    Breakout #2 ended 5.15s before the nightshade payoff."""
    from vidlore.clipstudio.index import find_quote_span
    from vidlore.clipstudio.build import _BK_LEAD_S, _BK_TAIL_S
    words = _real_words()
    for q, shot_start in (("Any man who must say 'I am the king' is no true king.", 130.05),
                          ("Perhaps some essence of nightshade to help him sleep.", 169.67)):
        sp = find_quote_span(words, q)
        assert sp is not None, f"not located: {q!r}"
        qs, qe, _ = sp
        start = max(0.0, qs - _BK_LEAD_S)
        min_dur = (qe - start) + _BK_TAIL_S
        assert start <= qs, "window must open at or before the line"
        assert start + min_dur >= qe, "window must contain the COMPLETE line"
    # the thesis window must reach BACK before its shot boundary — the old code never could
    qs, _, _ = find_quote_span(words, "Any man who must say 'I am the king' is no true king.")
    assert max(0.0, qs - _BK_LEAD_S) < 130.05, "window must precede the shot boundary"


def test_breakout_in_point_is_the_quote_not_the_shot():
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "QUOTE-ANCHORED WINDOW" in s
    i = s.index("QUOTE-ANCHORED WINDOW")
    # slice to the END of the anchoring block (the extract call) rather than a fixed character
    # count — the block legitimately grew when the "promised line is not spoken here" branch was
    # added, and a fixed window silently pushed `min_dur=_bk_min` out of view.
    blk = s[i:s.index("real = _extract_breakout(", i) + 400]
    assert "confirmed_span" in blk, (
        "the in-point must come from an independently confirmed quote-audio span")
    assert "_confirmation_record9" in blk
    assert "_bk_start = max(0.0, _qs - _BK_LEAD_S)" in blk
    assert "min_dur=_bk_min" in blk, "the window must be forbidden from ending before the line does"
    # the post-extract window gate must validate the window that ACTUALLY aired
    assert "_w0, _w1 = float(_bk_start)" in s, "the aired-window gate must use the anchored start"


def test_min_dur_floor_prevents_truncating_the_line():
    """_dialogue_aware_dur ends on 'a complete spoken line' -- which can be an EARLIER line than the
    quote's last word. Without a floor it would still truncate the payoff."""
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "_lo = max(3.0, min(float(min_dur or 0.0), _hi))" in s
    i = s.index("_lo = max(3.0, min(float(min_dur or 0.0), _hi))")
    assert "nightshade" in s[max(0, i - 700):i], "the floor must be documented by the failure it fixes"


def test_quote_anchored_window_skips_the_silence_trim():
    """The leading-silence probe exists to skip dead air at a SHOT's head. With a quote-anchored
    in-point there is none, and skipping ahead would clip the first word."""
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "if min_dur <= 0 and _m0 and float(_m0.group(1)) <= 0.15:" in s


def test_split_line_earns_the_verbatim_strength_it_actually_has():
    """Why the payoff never aired, precisely.

    A candidate WAS generated for the nightshade quote — but per-shot matching could only find a
    3-word sub-window ("run=3"), because the line is split across 4 shots and garbled. run=3 fails
    _verbatim_bypass_ok (needs >=4 words + >=70% coverage), so the candidate fell back to the
    Face-ID wrong-character gate; Face-ID could confirm nobody (no reference for Joffrey/Pycelle,
    and the council shots carry faces=[]), so it was skipped as "no confirmed main character" —
    one of the audit's wrong_char=45 rejections.

    The word stream scores the same line at 0.889 → run=7 → the bypass is earned honestly. The fix
    works THROUGH the safety gate, not around it."""
    import re as _re
    from vidlore.clipstudio.build import _verbatim_bypass_ok
    from vidlore.clipstudio.index import find_quote_span
    q = "Perhaps some essence of nightshade to help him sleep."
    qw = [w for w in _re.findall(r"[a-z0-9']+", q.lower())][:8]

    assert _verbatim_bypass_ok(qw, 3) is False, "run=3 must NOT bypass Face-ID (that is the guard)"
    sp = find_quote_span(_real_words(), q)
    assert sp is not None
    run = max(3, int(round(sp[2] * len(qw))))
    assert run >= 4, f"word-stream run {run} still too weak"
    assert _verbatim_bypass_ok(qw, run) is True, "an honestly-matched full line must earn the bypass"


def test_candidate_generation_uses_the_word_stream():
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "WORD-STREAM PASS first" in s
    i = s.index("WORD-STREAM PASS first")
    blk = s[i:i + 4200]
    assert "_quote_candidate_spans9" in blk
    assert "_confirm_quote_hit9" in blk, (
        "prompt-assisted retrieval must not grant quote privileges without no-prompt confirmation")
    assert "_texty9(_sh)" in blk, "burned-text shots must still be excluded"
    assert "continuous word stream is better evidence" in blk


def _cap_words(t):
    return [(w, i * 0.4, i * 0.4 + 0.35, 0.9) for i, w in enumerate(t.split())]


def test_breakout_caption_corrects_the_measured_meaning_inversion():
    """The burned caption read 'and I've won your war for you'; Tywin says 'WHEN I've won your war
    for you' — a conditional threat rendered as a past-tense boast, over the very line the
    narration then unpacks. No word-level test could catch it: 'and' vs 'when' is 0.29
    character-wise. Only the surrounding phrase can."""
    from vidlore.clipstudio.build import _correct_breakout_words
    asr = _cap_words("I'll make sure you understand that, and I've won your war for you.")
    out = _correct_breakout_words(
        asr, "I'll make sure you understand that, when I've won your war for you.")
    assert "when" in " ".join(w[0] for w in out).lower()
    assert len(out) == len(asr), "must never insert or delete words"
    assert all(a[1:] == b[1:] for a, b in zip(asr, out)), "timings must stay the ASR's"


def test_breakout_caption_never_takes_the_script_blindly():
    """A breakout that drifted onto different dialogue must keep its own words — the script
    proposes, the audio disposes."""
    from vidlore.clipstudio.build import _correct_breakout_words
    asr = _cap_words("My father won the real war. He killed Prince Rhaegar.")
    out = _correct_breakout_words(asr, "The king is tired. See him to his chambers.")
    assert [w[0] for w in out] == [w[0] for w in asr]


def test_breakout_caption_will_not_rewrite_a_semantic_opposite():
    """'I am the king' vs a known 'I am the queen' aligns at 0.75 — but king/queen is 0.44
    phonetically, so the audio keeps its word. Semantic opposites are what must never be rewritten."""
    from vidlore.clipstudio.build import _correct_breakout_words
    out = _correct_breakout_words(_cap_words("I am the king"), "I am the queen")
    assert "king" in [w[0] for w in out]


def test_breakout_caption_fixes_garble_with_phonetic_corroboration():
    from vidlore.clipstudio.build import _correct_breakout_words
    out = _correct_breakout_words(_cap_words("a messence of nightshade to help him sleep"),
                                  "some essence of nightshade to help him sleep")
    assert "essence" in " ".join(w[0] for w in out), "'messence'->'essence' is a 0.93 near-miss"


def test_aired_coverage_matches_how_candidates_are_found():
    """The aired-coverage gate must judge the same way find_quote_span FINDS — else a line located
    through ASR garble is dropped here on that same garble. Two real bugs, one measured:
      1. the greedy matcher burned its pointer on a missing word, so "Perhaps SOME essence" vs aired
         "Perhaps A messence" scored 1/6 and dropped the video's nightshade payoff;
      2. exact word-match rejected the 'messence'/'essence' near-miss.
    """
    from vidlore.clipstudio.build import _ordered_coverage
    w = str.split
    # the exact failing case — must now clear the 0.70 verbatim floor
    cov = _ordered_coverage(w("Perhaps some essence of nightshade to help him sleep"),
                            w("Perhaps a messence of nightshade to help him sleep"))
    assert cov >= 0.70, f"garble-tolerant, pointer-safe coverage must pass the payoff, got {cov}"
    # a genuinely different line must still score low — the tolerance must not become a pass-all
    assert _ordered_coverage(w("The king is tired see him to his chambers"),
                             w("My father won the real war he killed Prince Rhaegar")) < 0.5
    # a TRUNCATED aired window (only the prefix) must stay below floor — we won't air half a line
    assert _ordered_coverage(w("any man who must say I am the king is no true king"),
                             w("any man who must")) < 0.70


def test_promised_payoff_is_detected_and_the_tease_is_not():
    """The video spends 9 minutes promising one word and never plays it. The promise beat
    ('listen for something') names nothing; the PAYOFF beat ("That's the word I told you to listen
    for. Nightshade.") names it."""
    from vidlore.clipstudio.build import _promised_terms
    segs = [NS(index=0, text="Before we go in, I want you to listen for something."),
            NS(index=1, text="Somewhere in those ninety seconds, one word gets spoken."),
            NS(index=2, text="It's the name of a poison."),
            NS(index=3, text="That's the word I told you to listen for. Nightshade.")]
    assert _promised_terms(segs) == {"nightshade"}, _promised_terms(segs)
    # a tease alone promises nothing concrete
    assert _promised_terms(segs[:3]) == set()


def test_promised_payoff_is_priority_not_a_bypass():
    """Priority reorders candidates; every safety gate still runs. A promised line that cannot be
    aired safely still must not air."""
    from pathlib import Path as _P
    s = (_P(__file__).resolve().parents[1] / "vidlore" / "clipstudio" / "build.py").read_text(
        encoding="utf-8")
    assert "PROMISED PAYOFF" in s
    i = s.index("PROMISED PAYOFF")
    blk = s[i:i + 1400]
    assert "Priority ONLY, never a bypass" in blk
    assert "_keeps_a_promise" in blk and "cands.sort" in blk, "must act on the SORT, not the gates"


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
    test_build_blocks_on_unverified_exact_beats,
    test_rejected_footage_gate_is_non_retryable_too,
    test_adjacent_transition_tails_cannot_both_pad,
    test_pads_are_derived_from_selected_pairs_only,
    test_frame_allocation_carries_so_drift_cannot_accumulate,
    test_video_total_conforms_to_the_narration_clock,
    test_sync_invariant_compares_against_composed_audio,
    test_empty_faceid_is_unknown_not_innocent,
    test_positive_faceid_confirmation_allows_the_contextual_downgrade,
    test_wrong_subject_visible_is_a_hard_rejection,
    test_a_different_identified_person_still_blocks,
    test_correct_actor_face_is_not_a_wrong_character,
    test_faceid_recovers_oversized_reference_stills,
    test_breakout_window_contains_the_complete_iconic_line,
    test_breakout_in_point_is_the_quote_not_the_shot,
    test_min_dur_floor_prevents_truncating_the_line,
    test_quote_anchored_window_skips_the_silence_trim,
    test_split_line_earns_the_verbatim_strength_it_actually_has,
    test_candidate_generation_uses_the_word_stream,
    test_breakout_caption_corrects_the_measured_meaning_inversion,
    test_breakout_caption_never_takes_the_script_blindly,
    test_breakout_caption_will_not_rewrite_a_semantic_opposite,
    test_breakout_caption_fixes_garble_with_phonetic_corroboration,
    test_aired_coverage_matches_how_candidates_are_found,
    test_promised_payoff_is_detected_and_the_tease_is_not,
    test_promised_payoff_is_priority_not_a_bypass,
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
