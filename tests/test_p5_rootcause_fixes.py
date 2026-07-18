"""Regression: the competitor-comparison (P5) root-cause fix batch.

Every case is anchored to a MEASURED defect in the olenna_baseline render, not a hypothetical:

  A1  rhetorical connectors LLM-labelled exact_scene → 11 unsatisfiable beats release-blocked
      the render after ~7h of doomed fallback work.
  B1  the analysis LLM invents per-beat quotes (a confession beat carried a quote whose words
      are never spoken on screen) → the promised payoff produced ZERO breakout candidates.
  B2  scenes the script CITES (a confession, a trial testimony) were queried and found, but the
      global ranking dropped every candidate at the download cut — 0/45 sources contained them.
      A junk parody ("... Abridged") and foreign-sub uploads also slipped the title filter.
  B3  a flat luma floor (62) rejected 7 breakout candidates whose FACE was already confirmed by
      Face-ID — a confirmed main-cast face at YAVG 45-59 is legible by construction.

    python3 tests/test_p5_rootcause_fixes.py

Pure logic — no render, no network, no LLM (LLM calls are monkeypatched).
"""
import os
import re
import sys
import types
from pathlib import Path
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
# A1 — rhetorical-connector demotion guard (policy.py)
# ---------------------------------------------------------------------------
def test_a1_connectors_demote_to_abstract():
    # the four connector SHAPES that release-blocked the baseline (text generalized)
    for t in ("But that raises the next obvious question.",
              "So where does it even come from?",
              "Which brings us back to the feast.",
              "The answer starts years earlier."):
        s = _seg(t, visual_policy=policy.EXACT)
        assert policy.policy_of(s) == policy.ABSTRACT, \
            f"connector kept exact_scene: {t!r}"
    # character_specific connectors demote too
    s = _seg("And that's not a coincidence.", visual_policy=policy.CHARACTER)
    assert policy.policy_of(s) == policy.ABSTRACT


def test_a1_specific_beats_keep_their_labels():
    # ANY concrete hook (quote / entity) keeps the specific label — the guard must be earned
    s = _seg("But that raises the next question.", visual_policy=policy.EXACT,
             quote="You are no king of mine.")
    assert policy.policy_of(s) == policy.EXACT, "a quote must defeat the demotion guard"
    s = _seg("So where does it even come from?", visual_policy=policy.EXACT,
             required_entity="the strangler poison", required_kind="object")
    assert policy.policy_of(s) == policy.EXACT, "an entity must defeat the demotion guard"
    # a scene-describing beat with no connector phrasing is untouched
    s = _seg("The king collapses at the feast, clawing at his throat.",
             visual_policy=policy.EXACT)
    assert policy.policy_of(s) == policy.EXACT


def test_a1_deixis_still_outranks_everything():
    s = _seg("Now look at who is standing behind that table.", visual_policy=policy.EXACT)
    assert policy.policy_of(s) == policy.EXACT, "deixis must never be demoted"


def test_a1_short_question_without_hooks_demotes():
    s = _seg("But why would she do it?", visual_policy=policy.EXACT)
    assert policy.policy_of(s) == policy.ABSTRACT


# ---------------------------------------------------------------------------
# B2a — junk-title filter additions (discover.py _REJECT_TITLE)
# ---------------------------------------------------------------------------
def test_b2_junk_title_additions():
    from vidlore.clipstudio.discover import _REJECT_TITLE
    bad = [
        # a parody satisfied the trial anchor slot in the baseline
        "Hero's Trial 2: Medieval Boogaloo | Example Show Abridged - S04E06",
        # foreign-sub uploads poison ASR/word-stream matching
        "Hero arrests the archivist | Example Show 2x03 (Legendado)",
        "Example Show 4x02 La Boda Subtitulado",
        "Example Show Königsmord Deutsch HD",
        "Example Show 1x05 VOSTFR",
        "Example Show dublado em português",
    ]
    good = [
        "Epic Hero Speech During Trial",
        "The Royal Wedding - Example Show S04E02",
        "The queen confesses to the knight - Example Show",
    ]
    for t in bad:
        assert re.search(_REJECT_TITLE, t.lower()), f"junk title kept: {t!r}"
    for t in good:
        assert not re.search(_REJECT_TITLE, t.lower()), f"legit title rejected: {t!r}"


# ---------------------------------------------------------------------------
# B2b — key-scene download coverage (discover.py selection pass)
# ---------------------------------------------------------------------------
def _discover_fixture(key_scenes, env=None):
    """Run discover_sources against a synthetic pool where the dominant scene floods the
    download cut and the script-cited scenes rank below it (the measured starvation shape)."""
    from vidlore.clipstudio import discover as D
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.analyze import ScriptAnalysis

    def _c(i, title, ch, views):
        return D.SourceCandidate(url=f"https://x/{i}", id=f"v{i}", title=title,
                                 provider="youtube", duration=300.0, height=1080,
                                 channel=ch, view_count=views)
    pool = [
        _c(i, f"Royal Wedding Feast Scene - Example Kingdom HD part {i}", f"ch{i}", 50000 + i)
        for i in range(8)
    ] + [
        _c(100, "The Queen Confesses to the Knight - Example Kingdom", "chA", 900),
        _c(101, "The Archivist Trial Testimony - Example Kingdom", "chB", 800),
    ]
    ana = ScriptAnalysis(
        topic="how the wedding happened", movie_title="Example Kingdom",
        video_type="multi_scene",
        anchor_scenes=[{"name": "royal wedding feast",
                        "query": "Example Kingdom royal wedding feast scene"}],
        key_scenes=list(key_scenes))
    cfg = ClipConfig()
    cfg.discover_target = 4
    cfg.discover_max_per_channel = 2
    cfg.discover_resolve_quality = False
    cfg.discover_per_query = 4

    logs = []
    envset = {"VIDLORE_CLIPSTUDIO_SUB_VERIFY": "0"}
    envset.update(env or {})
    saved = {k: os.environ.get(k) for k in envset}
    _yt, _ar = D._ytsearch, D._archive_search
    D._ytsearch = lambda q, n: list(pool)
    D._archive_search = lambda q, n: []
    try:
        os.environ.update(envset)
        final = D.discover_sources(ana, cfg, segments=None,
                                   progress=lambda m: logs.append(str(m)))
    finally:
        D._ytsearch, D._archive_search = _yt, _ar
        for k, v in saved.items():
            (os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
    return final, "\n".join(logs)


def test_b2_cited_scenes_earn_a_download_slot():
    final, logs = _discover_fixture(
        ["the queen confesses to the knight", "the archivist trial testimony"])
    titles = " | ".join(c.title.lower() for c in final)
    assert "confesses" in titles, f"cited confession scene starved out of the cut: {titles}"
    assert "trial testimony" in titles, f"cited trial scene starved out of the cut: {titles}"
    assert "+key-scene" in logs, "coverage additions must be logged"


def test_b2_key_scene_coverage_is_bounded_and_optional():
    final, logs = _discover_fixture(
        ["the queen confesses to the knight", "the archivist trial testimony"],
        env={"VIDLORE_CLIPSTUDIO_KEYSCENE_COVERAGE": "0"})
    assert "+key-scene" in logs or True  # additions disabled → no log line
    assert "+key-scene" not in logs, "env=0 must disable key-scene additions"
    titles = " | ".join(c.title.lower() for c in final)
    assert "confesses" not in titles, "with the quota off, the starvation must reproduce"


def test_b2_anchor_covered_key_scene_adds_nothing():
    # a key_scene that IS the anchor scene must not force an extra slot
    final, logs = _discover_fixture(["the royal wedding feast"])
    assert "+key-scene" not in logs, "anchor-equivalent key scene must be skipped"


# ---------------------------------------------------------------------------
# B1 + B3 — breakout fixes (build.py _select_breakouts)
# ---------------------------------------------------------------------------
def _breakout_fixture(*, quote, asr, seg0_text, llm_reply=None, face_ids=(),
                      luma=80.0, characters=None, aired=None, env=None, seg_idx=0):
    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import llm as L
    import vidlore.clipstudio.index as _idxmod

    Path("/tmp/x.mp4").write_bytes(b"x")
    shots = [
        NS(index=0, start=12.0, end=20.0, transcript=asr,
           face_ids=list(face_ids), local_path="/tmp/x.mp4"),
        NS(index=1, start=22.0, end=26.0, transcript="", face_ids=[], local_path="/tmp/x.mp4"),
        NS(index=2, start=30.0, end=34.0, transcript="he walks in",
           face_ids=[], local_path="/tmp/x.mp4"),
    ]
    src = NS(id="s1", status="ok", local_path="/tmp/x.mp4",
             title="The queen power scene HD", width=1920, extra={"anchor_verified": True})
    proj = NS(sources=[src], meta={"analysis": {
        "characters": characters if characters is not None
        else [{"name": "Queen Regent", "actor": "Jane Example"}],
        "anchor_scenes": [{"name": "queen power scene",
                           "query": "queen power scene", "dialogue": []}],
        "movie_title": "Example Kingdom", "episode_hint": ""}})
    segs = [NS(index=i,
               text=(seg0_text if i == seg_idx else f"narration filler beat number {i} words here now"),
               quote=(quote if i == seg_idx else "")) for i in range(8)]

    llm_calls = []

    def _fake_complete(**kw):
        llm_calls.append(kw)
        return llm_reply or ""

    _orig_ls = _idxmod.load_shots
    _orig = (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
             B._asr_wav_words, L.complete)
    _idxmod.load_shots = lambda p, sid: shots
    B._extract_breakout = lambda *a, **k: 5.0
    B._breakout_window_luma = lambda *a, **k: luma
    B._frame_has_burned_text = lambda *a, **k: False
    _airtx = aired if aired is not None else asr
    B._asr_wav_words = lambda p: (_airtx.split(), _airtx, 5.0)
    L.complete = _fake_complete
    logs = []
    envset = dict(env or {})
    saved = {k: os.environ.get(k) for k in envset}
    try:
        os.environ.update(envset)
        out = B._select_breakouts(proj, segs, 200.0, Path("/tmp"),
                                  lambda m: logs.append(str(m)))
    finally:
        _idxmod.load_shots = _orig_ls
        (B._extract_breakout, B._breakout_window_luma, B._frame_has_burned_text,
         B._asr_wav_words, L.complete) = _orig
        for k, v in saved.items():
            (os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v))
    return out, "\n".join(logs), llm_calls


_SC = "seize him cut his throat stop wait i have changed my mind"


def test_b1_hallucinated_quote_corrected_and_located():
    # the scripted quote is an invention (locates nowhere); the corrective re-ask returns the
    # real on-screen line, which locates in the footage ASR → the payoff airs after all
    out, logs, calls = _breakout_fixture(
        quote="You cannot escape justice now.", asr=_SC,
        seg0_text="the narration describes the arrest in the throne room",
        llm_reply="0. Seize him. Cut his throat.")
    assert len(calls) == 1, f"exactly ONE corrective LLM call, got {len(calls)}"
    assert "quote corrected" in logs, "correction must be logged"
    assert any(o["seg_index"] == 0 for o in out), "corrected quote must produce the breakout"


def test_b1_second_hallucination_changes_nothing():
    # the re-ask replies with another line that ALSO is not in the ASR → still zero candidates;
    # the footage ASR stays ground truth, so a second hallucination cannot smuggle a breakout in
    out, logs, calls = _breakout_fixture(
        quote="You cannot escape justice now.", asr=_SC,
        seg0_text="the narration describes the arrest in the throne room",
        llm_reply="0. Guards, arrest this criminal at once.")
    assert len(calls) == 1
    assert not any(o["seg_index"] == 0 for o in out), \
        "an unlocatable correction must not create a candidate"
    assert "recovered 0" in logs


def test_b1_none_reply_and_env_off():
    # 'NONE' reply → no candidate, no crash
    out, logs, calls = _breakout_fixture(
        quote="You cannot escape justice now.", asr=_SC,
        seg0_text="the narration describes the arrest in the throne room",
        llm_reply="0. NONE")
    assert not any(o["seg_index"] == 0 for o in out)
    # env off → the LLM is never called
    out, logs, calls = _breakout_fixture(
        quote="You cannot escape justice now.", asr=_SC,
        seg0_text="the narration describes the arrest in the throne room",
        env={"VIDLORE_CLIPSTUDIO_QUOTE_CORRECT": "0"})
    assert not calls, "QUOTE_CORRECT=0 must disable the corrective call"


def test_b1_locatable_quote_never_triggers_the_reask():
    # a quote that locates on the first pass must not consume the corrective call
    out, logs, calls = _breakout_fixture(
        quote="Seize him. Cut his throat.", asr=_SC,
        seg0_text="the narration describes the arrest in the throne room")
    assert any(o["seg_index"] == 0 for o in out)
    assert not calls, "a located quote must not trigger the re-ask"


def test_b3_face_confirmed_candidate_survives_dim_footage():
    # weak verbatim match (3-word run → no Face-ID bypass) + the shot's face IS a confirmed
    # main-cast face + dim frame (45 < flat floor 62): the confirmed face is the legibility
    # proof, so the candidate must air
    out, logs, calls = _breakout_fixture(
        quote="You know nothing about my plans", asr="you know nothing else matters here tonight",
        aired="you know nothing about my plans tonight",
        seg0_text="the narration describes the confrontation", seg_idx=3,
        face_ids=("queen regent",), luma=45.0)
    assert any(o["seg_index"] == 3 for o in out), \
        "face-confirmed candidate must survive the luma gate at 45"


def test_b3_unconfirmed_dim_candidate_still_rejected():
    # same dim frame, no cast list → no face confirmation, no verbatim strength → the flat
    # floor still applies; near-dark unknown footage must never air (the gate is NOT weakened)
    out, logs, calls = _breakout_fixture(
        quote="You know nothing about my plans", asr="you know nothing else matters here tonight",
        aired="you know nothing about my plans tonight",
        seg0_text="the narration describes the confrontation", seg_idx=3,
        characters=[], face_ids=(), luma=45.0)
    assert not any(o["seg_index"] == 3 for o in out), \
        "dim footage without face confirmation must still be rejected"
    assert "dark" in logs.lower() or "luma" in logs.lower()


# ---------------------------------------------------------------------------
# C1 — scene-affinity ordering of verifier-replacement alternates (verify.py)
# ---------------------------------------------------------------------------
def _aff_proj(sources):
    class P:
        meta = {"analysis": {"movie_title": "Example Kingdom"}}
        def source(self, sid):
            return sources.get(sid)
    return P()


def test_c1_scene_affine_sources_ordered_first():
    from vidlore.clipstudio.verify import _scene_affinity_order
    sources = {
        "dinner": NS(title="King & Queen Dinner Scene Example Kingdom S03E01", extra={}),
        "wed1": NS(title="The Royal Wedding Feast - Example Kingdom S04E02", extra={}),
        "ver1": NS(title="some compilation upload", extra={"anchor_verified": True}),
        "misc": NS(title="Example Kingdom best moments", extra={}),
    }
    # relevance-ranked as match left them: the wrong-scene dinner first (the measured failure)
    alts = [NS(source_id="dinner", shot_index=1), NS(source_id="wed1", shot_index=4),
            NS(source_id="misc", shot_index=2), NS(source_id="ver1", shot_index=7)]
    seg = _seg("he complains the pie is dry and demands wine",
               scene_query="Example Kingdom royal wedding feast pie wine")
    out = _scene_affinity_order(alts, seg, _aff_proj(sources), "wed1")
    order = [a.source_id for a in out]
    assert order[0] in ("wed1", "ver1") and order[1] in ("wed1", "ver1"), \
        f"scene-affine sources must lead: {order}"
    assert order[0] == "wed1", f"stable sort must keep relevance order within a tier: {order}"
    assert order[-1] in ("dinner", "misc"), f"wrong-scene source must not lead: {order}"


def test_c1_original_source_outranks_strangers():
    # the verifier rejected one FRAME of the original source — its other shots are tier 1
    from vidlore.clipstudio.verify import _scene_affinity_order
    sources = {
        "orig": NS(title="untitled clip", extra={}),
        "other": NS(title="another untitled clip", extra={}),
    }
    alts = [NS(source_id="other", shot_index=0), NS(source_id="orig", shot_index=3)]
    seg = _seg("beat with no scene query")
    out = _scene_affinity_order(alts, seg, _aff_proj(sources), "orig")
    assert [a.source_id for a in out] == ["orig", "other"]


def test_c1_no_signal_preserves_relevance_order():
    # no scene_query tokens, no anchor flags, no original in the list → order unchanged
    from vidlore.clipstudio.verify import _scene_affinity_order
    sources = {f"s{i}": NS(title=f"clip number {i}", extra={}) for i in range(4)}
    alts = [NS(source_id=f"s{i}", shot_index=i) for i in range(4)]
    seg = _seg("a beat", scene_query="")
    out = _scene_affinity_order(alts, seg, _aff_proj(sources), "absent")
    assert [a.source_id for a in out] == ["s0", "s1", "s2", "s3"]


TESTS = [
    test_a1_connectors_demote_to_abstract,
    test_a1_specific_beats_keep_their_labels,
    test_a1_deixis_still_outranks_everything,
    test_a1_short_question_without_hooks_demotes,
    test_b2_junk_title_additions,
    test_b2_cited_scenes_earn_a_download_slot,
    test_b2_key_scene_coverage_is_bounded_and_optional,
    test_b2_anchor_covered_key_scene_adds_nothing,
    test_b1_hallucinated_quote_corrected_and_located,
    test_b1_second_hallucination_changes_nothing,
    test_b1_none_reply_and_env_off,
    test_b1_locatable_quote_never_triggers_the_reask,
    test_b3_face_confirmed_candidate_survives_dim_footage,
    test_b3_unconfirmed_dim_candidate_still_rejected,
    test_c1_scene_affine_sources_ordered_first,
    test_c1_original_source_outranks_strangers,
    test_c1_no_signal_preserves_relevance_order,
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
