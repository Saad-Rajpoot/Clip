"""Resume checkpoints + fail-fast rejected-footage feasibility gate.

Two guarantees are locked here:

1. STAGE CHECKPOINTS — a render that died (or content-blocked) at assembly must resume from where
   it stopped, not redo analyze→discover→download→index→match→verify. The checkpoint layer skips a
   stage iff we are resuming, its input signature is unchanged, and its persisted output is present.

2. FAIL-FAST FEASIBILITY — the end-of-assembly rejected-footage release gate is deterministic from
   the (frozen) selections, so it is surfaced BEFORE the ~20-minute per-beat re-encode a doomed
   render would throw away. The predictor must be SOUND: it may only ever report a SUBSET of the
   real gate's blocks (never wrongly kill a render the gate would pass).

    python3 tests/test_resume_and_feasibility.py

No LLM, no ffmpeg, no network.
"""
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio import build as B
from vidlore.clipstudio import cut as CUT
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, SourceVideo

FAILS = []


# --------------------------------------------------------------------------- helpers
def _seg(idx, scene, entity, policy="exact_scene"):
    return ScriptSegment(index=idx, text=f"beat {idx}", scene_query=scene,
                         required_entity=entity, required_kind="character", visual_policy=policy)


def _sel(idx, src, *, rejected=False, image="", identity=""):
    fr = ["verifier_failed"] if rejected else []
    return ClipSelection(segment_index=idx, source_id=src, shot_index=0, in_point=0.0,
                         out_point=3.0, confidence=0.8, identity=identity, image_path=image,
                         flag_reasons=fr,
                         verifier=({"status": "ok", "verdict": "replace"} if rejected
                                   else {"status": "ok", "verdict": "keep"}),
                         beat_windows=[[src, 0.0, 3.0]])


def _proj(tmp, segs, sels, video_type="multi_scene"):
    p = ClipProject(name="t", root=tmp)
    p.ensure_dirs()
    p.segments = list(segs)
    p.selections = list(sels)
    p.meta["analysis"] = {"movie_title": "Game of Thrones", "video_type": video_type,
                          "episode_hint": "", "characters": []}
    return p


# --------------------------------------------------------------------------- checkpoint layer
def test_signature_is_stable_and_sensitive():
    a = O._sig("script text", "topic", "movie")
    assert a == O._sig("script text", "topic", "movie"), "same inputs → same signature"
    assert a != O._sig("script text CHANGED", "topic", "movie"), "any input change → new signature"
    s1 = O._seg_sig([_seg(0, "throne room", "tyrion")])
    s2 = O._seg_sig([_seg(0, "throne room", "tyrion")])
    s3 = O._seg_sig([_seg(0, "dragon pit", "tyrion")])
    assert s1 == s2 and s1 != s3, "seg signature tracks the beats that drive matching"


def test_stage_skip_only_when_resuming_matching_and_artifact_present():
    tmp = tempfile.mkdtemp()
    try:
        p = ClipProject(name="t", root=tmp)
        p.ensure_dirs()
        sig = O._sig("inputs")
        # not recorded yet → never skippable
        assert O._stage_skip(p, "match", sig, resume=True) is False
        O._stage_done(p, "match", sig)
        # recorded + resuming + matching sig + artifact present → skip
        assert O._stage_skip(p, "match", sig, resume=True, artifact_ok=True) is True
        # never skip when NOT resuming (a fresh run always recomputes)
        assert O._stage_skip(p, "match", sig, resume=False, artifact_ok=True) is False
        # changed inputs → re-run
        assert O._stage_skip(p, "match", O._sig("OTHER"), resume=True, artifact_ok=True) is False
        # artifact vanished (e.g. selections empty) → re-run even though checkpoint exists
        assert O._stage_skip(p, "match", sig, resume=True, artifact_ok=False) is False
        # the checkpoint survives a save/load round-trip (a resume is a fresh process)
        p2 = ClipProject.load(tmp)
        assert O._stage_skip(p2, "match", sig, resume=True, artifact_ok=True) is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_checkpoint_version_bump_invalidates_old_records():
    tmp = tempfile.mkdtemp()
    try:
        p = ClipProject(name="t", root=tmp)
        p.ensure_dirs()
        sig = O._sig("x")
        O._stage_done(p, "match", sig)
        p.meta["pipeline"]["version"] = O._PIPELINE_CKPT_VERSION - 1   # simulate an older schema
        # _ckpt() must discard the stale block → nothing skippable
        assert O._stage_skip(p, "match", sig, resume=True, artifact_ok=True) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- fail-fast feasibility
def test_doomed_beat_with_no_compatible_predecessor_blocks_early():
    tmp = tempfile.mkdtemp()
    try:
        segs = [_seg(0, "throne room tyrion trial", "tyrion lannister"),
                _seg(1, "throne room tyrion trial", "tyrion lannister"),      # rejected, but same scene as 0
                _seg(2, "dragon battle meereen sky", "daenerys targaryen")]   # rejected + unrelated scene
        sels = [_sel(0, "srcA"), _sel(1, "srcB", rejected=True), _sel(2, "srcC", rejected=True)]
        p = _proj(tmp, segs, sels)
        reason = B.preassemble_release_block_reason(p, segs)
        assert reason, "a rejected beat with no same-scene predecessor must be flagged pre-assembly"
        assert "2" in reason and "1" not in reason.split("scene(s)")[1], \
            f"only beat 2 is doomed (beat 1 is rescued by same-scene beat 0): {reason}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rejected_beat_rescued_by_same_scene_clip_is_not_flagged():
    tmp = tempfile.mkdtemp()
    try:
        segs = [_seg(0, "throne room tyrion trial", "tyrion lannister"),
                _seg(1, "throne room tyrion trial", "tyrion lannister")]
        sels = [_sel(0, "srcA"), _sel(1, "srcB", rejected=True)]              # 1 rejected but 0 covers it
        assert B.preassemble_release_block_reason(_proj(tmp, segs, sels), segs) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rejected_beat_with_installed_still_is_not_rejected():
    tmp = tempfile.mkdtemp()
    try:
        # beat 1 was verifier_failed but image-fallback installed a still → it airs, so it is NOT a
        # rejected beat and must not trip the gate (this is exactly the 8b outcome resume preserves).
        segs = [_seg(0, "throne room", "tyrion"),
                _seg(1, "dragon pit meereen", "daenerys")]
        sels = [_sel(0, "srcA"), _sel(1, "srcB", rejected=True, image="/tmp/still.jpg")]
        assert B.preassemble_release_block_reason(_proj(tmp, segs, sels), segs) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_disabled_env_returns_none():
    tmp = tempfile.mkdtemp()
    old = os.environ.get("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE")
    try:
        segs = [_seg(0, "a b c", "x"), _seg(1, "totally different", "y")]
        sels = [_sel(0, "srcA"), _sel(1, "srcB", rejected=True)]
        p = _proj(tmp, segs, sels)
        assert B.preassemble_release_block_reason(p, segs), "sanity: doomed with gate on"
        os.environ["VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE"] = "0"
        assert B.preassemble_release_block_reason(p, segs) is None, "disabled gate must not block"
    finally:
        if old is None:
            os.environ.pop("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE", None)
        else:
            os.environ["VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_predictor_is_sound_never_over_claims():
    """The core safety property: for every beat the predictor flags, EVERY preceding air-worthy beat
    is genuinely same-scene-INCOMPATIBLE (via the same primitive the real gate uses). So the
    predictor's block set is a subset of the gate's — it can only save wasted work, never wrongly
    kill a render the gate would have passed."""
    tmp = tempfile.mkdtemp()
    try:
        segs = [_seg(0, "throne room tyrion", "tyrion"),
                _seg(1, "dragon pit daenerys", "daenerys"),          # rejected, unrelated → doomed
                _seg(2, "throne room tyrion", "tyrion")]             # rejected, same as 0 → rescued
        sels = [_sel(0, "srcA"), _sel(1, "srcB", rejected=True), _sel(2, "srcC", rejected=True)]
        p = _proj(tmp, segs, sels)
        reason = B.preassemble_release_block_reason(p, segs)
        assert reason and "1" in reason, "beat 1 is genuinely doomed"
        # Independently re-derive: beat 1's only preceding air-worthy beat is 0, and it is incompatible.
        ok0, _ = B._hold_scene_compat(segs[0], segs[1], sels[0], sels[1],
                                      single_scene=False, global_era="", overlap_min=0.4)
        assert ok0 is False, "the flagged beat truly has no compatible predecessor"
        # beat 2 IS rescued by beat 0 (same scene) — so it must NOT be in the block set
        ok2, _ = B._hold_scene_compat(segs[0], segs[2], sels[0], sels[2],
                                      single_scene=False, global_era="", overlap_min=0.4)
        assert ok2 is True and "2" not in reason.split("scene(s)")[1], \
            "a beat with a compatible predecessor must never be flagged"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- cut resume skip
def test_cut_resume_reuses_existing_clip_without_reencoding():
    tmp = tempfile.mkdtemp()
    called = {"ffmpeg": 0}
    orig_run = CUT.subprocess.run
    try:
        p = ClipProject(name="t", root=tmp)
        p.ensure_dirs()
        src_file = os.path.join(tmp, "movie.mp4")
        open(src_file, "wb").write(b"\x00" * 32)
        p.sources = [SourceVideo(id="srcA", url="local:x", local_path=src_file, duration=60.0)]
        sel = _sel(0, "srcA")
        existing = p.clips_dir / "seg_000.mp4"
        existing.write_bytes(b"already-cut-clip")
        CUT.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("ffmpeg must NOT run when resuming an already-cut clip"))
        cfg = type("Cfg", (), {"min_clip_sec": 1.0})()
        out = CUT.cut_selection(p, sel, cfg, resume=True)
        assert str(out) == str(existing), "resume must return the existing clip"
        assert existing.read_bytes() == b"already-cut-clip", "the existing clip must be untouched"
        assert called["ffmpeg"] == 0
    finally:
        CUT.subprocess.run = orig_run
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- wiring (source asserts)
def test_orchestrator_and_cut_expose_resume():
    import inspect
    assert "resume" in inspect.signature(O.produce_auto).parameters
    assert "resume" in inspect.signature(CUT.cut_all).parameters
    src = open(os.path.join(ROOT, "vidlore", "clipstudio", "orchestrate.py"), encoding="utf-8").read()
    # the fail-fast gate must run before assembly and raise the same non-retryable error
    assert "preassemble_release_block_reason" in src
    assert "Aborting BEFORE assembly" in src
    # match must checkpoint (closing the historic match→cut save gap)
    assert '_stage_done(proj, "match"' in src


def test_portal_exposes_resume_and_retry():
    src = open(os.path.join(ROOT, "vidlore", "clipstudio", "web.py"), encoding="utf-8").read()
    assert '@app.route("/retry/<jid>"' in src, "the portal must offer a resume/retry route"
    assert "resume=True" in src, "retry must resume rather than restart from scratch"
    assert 'kwargs={**params, "resume": True' in src, "retry must replay the original launch params"
    assert "retryResume()" in src and "retryDraft()" in src, "the job page must surface both buttons"
    assert 'os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"]' in src, \
        "review mode must be plumbed for footage-gap drafts"


def test_produce_auto_resume_skips_completed_stages_end_to_end():
    """Drive the REAL produce_auto orchestration twice on one dir — a full run, then resume=True —
    stubbing only the heavy leaf ops (LLM/discover/download/index/ffmpeg/vision). The resume run must
    SKIP analyze/discover/download/index/match/verify/recover (their stubs are not called again) and
    only re-run assembly. A third Resume simulates damaged/stale ASR artifacts under the SAME stage
    signature and must repair index then re-run match plus every dependent footage stage; the next
    Resume sees current ASR and skips them again. Finally, changing only the installed Whisper
    version invalidates the whole chain independently. This exercises the actual control flow,
    analysis rehydration, artifact validity, and signature matching — not just helper units."""
    from vidlore.clipstudio import analyze as AN, discover as DS, download as DL
    from vidlore.clipstudio import faceid as FI, verify as VF, review as RV, index as IX, ledger as LG
    from vidlore.clipstudio import llm as LM
    from vidlore.clipstudio.analyze import ScriptAnalysis

    calls = {}
    cut_resume_args = []
    asr_state = {"current": False}
    anchor_state = {"break_asr": False}
    def _count(name):
        calls[name] = calls.get(name, 0) + 1

    def fake_analyze(script_text, **k):
        _count("analyze")
        a = ScriptAnalysis(topic="t", movie_title="Movie", video_type="multi_scene",
                           actors=["Actor"], characters=[])
        segs = [_seg(0, "throne room", "tyrion"), _seg(1, "throne room", "tyrion")]
        return a, segs

    def fake_discover(analysis, cfg, **k):
        _count("discover")
        return [DS.SourceCandidate(url="https://y/1", id="srcA", title="Scene")]

    def fake_download(proj, candidates, cfg, **k):
        _count("download")
        proj.sources = [SourceVideo(id="srcA", url="https://y/1", local_path="/tmp/x.mp4",
                                    title="Scene", duration=60.0, status="ok")]

    def fake_match(proj, segs, cfg, **k):
        _count("match")
        proj.selections = [_sel(s.index, "srcA") for s in segs]   # clean → no rejected beats

    def fake_index(*_a, **_k):
        _count("index")
        asr_state["current"] = True

    def fake_anchor(*_a, **_k):
        if anchor_state["break_asr"]:
            asr_state["current"] = False
        return 0

    def fake_build(proj, segs, cfg, **k):
        _count("build")
        out = proj.output_dir / "final.mp4"
        out.write_text("video")
        return out

    def fake_cut(*_a, **kwargs):
        _count("cut")
        cut_resume_args.append(kwargs.get("resume"))
        return 0

    patches = [
        (AN, "analyze_script", fake_analyze),
        (DS, "discover_sources", fake_discover),
        (DL, "download_candidates", fake_download),
        (FI, "available", lambda: False),
        (IX, "clip_available", lambda: True),
        (IX, "_faster_whisper_version", lambda: "1.2.1"),
        (VF, "verify_and_repair", lambda *a, **k: _count("verify")),
        (LM, "vision_probe", lambda *a, **k: (True, "ok")),   # healthy backend → preflight passes
        (RV, "write_review", lambda *a, **k: "review.html"),
        (LG, "finalize", lambda proj, segs, cfg: {"flagged_for_review": 0, "segments": len(segs),
                                                  "mean_confidence": 1.0}),
        (O, "asr_pool_current", lambda *_a, **_k: asr_state["current"]),
        (O, "index_all", fake_index),
        (O, "_ensure_anchor_coverage", fake_anchor),
        (O, "match_segments", fake_match),
        (O, "cut_all", fake_cut),
        (O, "build_video", fake_build),
        (O, "_recover_unresolved_beats", lambda *a, **k: _count("recover")),
        (O, "_fill_image_fallbacks", lambda *a, **k: _count("fallback")),
        (O, "_purge_unwanted_sources", lambda *a, **k: 0),
    ]
    saved = [(m, n, getattr(m, n, None)) for m, n, _ in patches]
    tmp = tempfile.mkdtemp()
    try:
        for m, n, fn in patches:
            setattr(m, n, fn)
        kw = dict(topic="t", script_text="a real script line", movie_hint="Movie",
                  policy="approved_testing", max_sources=4, do_build=True, verify=True)
        # 1) full run
        O.produce_auto(tmp, **kw)
        after_first = dict(calls)
        assert after_first.get("analyze") == 1 and after_first.get("match") == 1
        assert after_first.get("build") == 1
        assert cut_resume_args == [False], "a fresh match must always cut fresh clips"
        # 2) resume — every completed stage must be skipped; only assembly re-runs
        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("analyze") == 1, f"analyze must NOT re-run on resume: {calls}"
        assert calls.get("discover") == 1, f"discover must NOT re-run: {calls}"
        assert calls.get("download") == 1, f"download must NOT re-run: {calls}"
        assert calls.get("match") == 1, f"match must NOT re-run (selections cached): {calls}"
        assert calls.get("verify") == 1, f"verify must NOT re-run (verdicts cached): {calls}"
        assert calls.get("recover") == 1, f"recovery must NOT re-run: {calls}"
        assert calls.get("index", 0) == 1, f"index must be skipped when all footage stages cached: {calls}"
        assert calls.get("build") == 2, f"assembly MUST re-run on resume: {calls}"

        # 3) Simulate an interrupted/invalid cut checkpoint while the match checkpoint remains
        # valid. This is the one safe reuse path: selections are unchanged, so cut_all may retain
        # already-created seg_NNN clips and only encode whichever files are absent.
        cached = ClipProject.load(tmp)
        cached.meta["pipeline"]["stages"].pop("cut", None)
        cached.save()
        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("match") == 1, f"valid cached selections must remain reusable: {calls}"
        assert calls.get("cut") == 2 and cut_resume_args[-1] is True, \
            "a partial cut may reuse clips only when match was genuinely skipped"
        assert calls.get("build") == 3

        # 4) The checkpoint signature is unchanged, but its ASR artifacts are no longer valid.
        # Resume must re-enter indexing AND keep the old match checkpoint invalid for this run even
        # after targeted ASR repair makes the cache current again. Since match ran, the old clips
        # must NOT be reused even though this is a job-level Resume.
        asr_state["current"] = False
        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("index") == 2, f"invalid ASR artifacts must re-enter indexing: {calls}"
        assert calls.get("match") == 2, f"repaired ASR must feed a fresh match: {calls}"
        assert calls.get("cut") == 3 and calls.get("verify") == 2, \
            f"ASR artifact repair must cascade through dependent footage stages: {calls}"
        assert cut_resume_args[-1] is False, \
            "a rematch can change source windows, so its deterministic clip paths must be recut"
        assert calls.get("recover") == 2 and calls.get("build") == 4

        # 5) The repair is now current and newly checkpointed, so the next Resume is cheap again.
        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("index") == 2 and calls.get("match") == 2, calls
        assert calls.get("verify") == 2 and calls.get("recover") == 2, calls
        assert calls.get("build") == 5

        # 6) Changing only the ASR runtime identity invalidates index→match and the whole footage
        # dependency chain. Download remains cached; index_all performs the targeted ASR refresh.
        IX._faster_whisper_version = lambda: "1.3.0"
        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("download") == 1, f"same source files remain reusable: {calls}"
        assert calls.get("index") == 3, f"changed ASR identity must re-enter indexing: {calls}"
        assert calls.get("match") == 3, f"stale transcript-driven selections must be rematched: {calls}"
        assert calls.get("cut") == 4 and calls.get("verify") == 3, \
            f"ASR invalidation must cascade through dependent footage stages: {calls}"
        assert cut_resume_args[-1] is False, "every rematch must force fresh cuts"
        assert calls.get("recover") == 3 and calls.get("build") == 6, \
            f"recovery and assembly must consume the refreshed selections: {calls}"

        # 7) A source admitted after the main index audit can fail ASR. This must stop BEFORE match
        # and before any new downstream checkpoint is recorded; disabling only Resume reuse is not
        # enough because the current run would otherwise consume the incomplete pool.
        anchor_state["break_asr"] = True
        IX._faster_whisper_version = lambda: "1.4.0"
        with pytest.raises(O.PipelineError, match="post-acquisition ASR evidence incomplete"):
            O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("index") == 4, calls
        assert calls.get("match") == 3 and calls.get("cut") == 4, \
            f"post-acquisition ASR failure must stop before match/cut: {calls}"
        assert calls.get("verify") == 3 and calls.get("recover") == 3, calls
        assert calls.get("build") == 6, calls
    finally:
        for m, n, orig in saved:
            if orig is not None:
                setattr(m, n, orig)
        shutil.rmtree(tmp, ignore_errors=True)


def test_vision_error_classifier():
    from vidlore.clipstudio import llm
    assert llm.classify_vision_error("429 RESOURCE_EXHAUSTED prepayment credits are depleted") == "billing"
    assert llm.classify_vision_error("Your credit balance is too low to access the API") == "billing"
    assert llm.classify_vision_error("past due invoice, account is not active") == "billing"
    assert llm.classify_vision_error("401 invalid api key") == "auth"
    assert llm.classify_vision_error("permission denied 403") == "auth"
    assert llm.classify_vision_error("connection reset / timeout") == "transient"
    assert llm.classify_vision_error("503 model overloaded") == "transient"
    # CRITICAL: a RATE-LIMIT 429 (quota per minute / RESOURCE_EXHAUSTED without money words) must be
    # TRANSIENT, not billing — misclassifying it hard-failed a funded render with 'out of credits'.
    assert llm.classify_vision_error(
        "429 Quota exceeded for quota metric 'GenerateContent requests per minute'") == "transient"
    assert llm.classify_vision_error(
        "429 RESOURCE_EXHAUSTED. Resource has been exhausted (check quota).") == "transient"
    assert llm.classify_vision_error("429 Too Many Requests") == "transient"


def test_vision_backend_error_is_retryable_and_distinct():
    from vidlore.clipstudio.verify import VisionBackendError, NonRetryableBuildError
    e = VisionBackendError("out of credits", kind="billing")
    assert e.kind == "billing"
    # a vision outage is INFRA (retryable once restored), NOT a content verdict
    assert not isinstance(e, NonRetryableBuildError)
    assert isinstance(e, RuntimeError)


def test_vision_outage_does_not_checkpoint_verify_and_skips_the_grind():
    """When verify reports the backend down (breaker open), produce_auto must: raise
    VisionBackendError, NOT checkpoint verify as done (so Resume re-runs it), and NEVER run the
    image-fallback grind. This is the exact failure the user hit: hours of doomed stills + a
    Resume that skipped the dead verify and re-hit the wall."""
    from vidlore.clipstudio import analyze as AN, discover as DS, download as DL
    from vidlore.clipstudio import faceid as FI, verify as VF, review as RV, index as IX, ledger as LG
    from vidlore.clipstudio import llm as LM
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.verify import VisionBackendError

    calls = {}
    def _count(n): calls[n] = calls.get(n, 0) + 1

    def fake_analyze(script_text, **k):
        _count("analyze")
        return ScriptAnalysis(topic="t", movie_title="Movie", video_type="multi_scene",
                              actors=["Actor"], characters=[]), [_seg(0, "throne room", "tyrion")]

    def fake_download(proj, candidates, cfg, **k):
        proj.sources = [SourceVideo(id="srcA", url="https://y/1", local_path="/tmp/x.mp4",
                                    title="Scene", duration=60.0, status="ok")]

    def fake_match(proj, segs, cfg, **k):
        proj.selections = [_sel(s.index, "srcA") for s in segs]

    # verify reports the backend DOWN (breaker open) — the outage signature
    def fake_verify(proj, segs, cfg, eng, **k):
        _count("verify")
        return {"verified": 0, "errored": 178, "attempted": 178, "verifier_down": True,
                "available": True, "verified_frac": 0.0}

    patches = [
        (AN, "analyze_script", fake_analyze),
        (DS, "discover_sources", lambda a, c, **k: [DS.SourceCandidate(url="https://y/1", id="srcA", title="Scene")]),
        (DL, "download_candidates", fake_download),
        (FI, "available", lambda: False),
        (IX, "clip_available", lambda: True),
        (VF, "verify_and_repair", fake_verify),
        (LM, "vision_probe", lambda *a, **k: (False, "billing")),   # preflight + classify both see billing
        (RV, "write_review", lambda *a, **k: "review.html"),
        (LG, "finalize", lambda proj, segs, cfg: {"segments": len(segs), "mean_confidence": 1.0}),
        (O, "index_all", lambda *a, **k: None),
        (O, "match_segments", fake_match),
        (O, "cut_all", lambda *a, **k: 0),
        (O, "build_video", lambda *a, **k: _count("build")),
        (O, "_recover_unresolved_beats", lambda *a, **k: _count("recover")),
        (O, "_fill_image_fallbacks", lambda *a, **k: _count("fallback")),
        (O, "_purge_unwanted_sources", lambda *a, **k: 0),
    ]
    saved = [(m, n, getattr(m, n, None)) for m, n, _ in patches]
    tmp = tempfile.mkdtemp()
    try:
        for m, n, fn in patches:
            setattr(m, n, fn)
        kw = dict(topic="t", script_text="a real script line", movie_hint="Movie",
                  policy="approved_testing", max_sources=4, do_build=True, verify=True)
        raised = None
        try:
            O.produce_auto(tmp, **kw)
        except VisionBackendError as e:
            raised = e
        assert raised is not None, "a vision outage must raise VisionBackendError"
        assert raised.kind == "billing"
        assert calls.get("build", 0) == 0, "assembly must NOT run after a vision outage"
        assert calls.get("fallback", 0) == 0, "the image-fallback grind must be SKIPPED on outage"
        assert calls.get("recover", 0) == 0, "recovery must be skipped on outage"
        # verify was NOT checkpointed → a resume re-runs it (does not skip the dead verify)
        p = ClipProject.load(tmp)
        assert O._ckpt(p)["stages"].get("verify") is None, \
            "an errored verify must NOT checkpoint as done, or Resume would skip it forever"
    finally:
        for m, n, orig in saved:
            if orig is not None:
                setattr(m, n, orig)
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_materialization_failure_rolls_back_checkpoint_and_resume_retries():
    """A promoted alternate is not a completed verify result until its clip exists.

    The verifier owns the selection/clip rollback; orchestration owns the stage transaction.  A
    materialization failure must therefore stop before recovery/build, leave ``verify`` absent from
    the persisted checkpoint, and let Resume rerun only that unfinished stage.  The second pass
    below simulates repaired ffmpeg/disk infrastructure and must proceed normally.
    """
    from vidlore.clipstudio import analyze as AN, discover as DS, download as DL
    from vidlore.clipstudio import faceid as FI, verify as VF, review as RV, index as IX, ledger as LG
    from vidlore.clipstudio import llm as LM
    from vidlore.clipstudio.analyze import ScriptAnalysis

    calls = {}
    def _count(n): calls[n] = calls.get(n, 0) + 1

    def fake_analyze(script_text, **k):
        _count("analyze")
        return ScriptAnalysis(topic="t", movie_title="Movie", video_type="multi_scene",
                              actors=["Actor"], characters=[]), [_seg(0, "throne room", "tyrion")]

    def fake_download(proj, candidates, cfg, **k):
        _count("download")
        proj.sources = [SourceVideo(id="srcA", url="https://y/1", local_path="/tmp/x.mp4",
                                    title="Scene", duration=60.0, status="ok")]

    def fake_match(proj, segs, cfg, **k):
        _count("match")
        proj.selections = [_sel(s.index, "srcA") for s in segs]

    def fake_verify(proj, segs, cfg, eng, **k):
        _count("verify")
        if calls["verify"] == 1:
            return {"verified": 1, "errored": 1, "attempted": 2,
                    "verifier_down": False, "available": True,
                    "materialization_errors": 1}
        return {"verified": 1, "errored": 0, "attempted": 1,
                "verifier_down": False, "available": True,
                "materialization_errors": 0}

    def fake_build(proj, segs, cfg, **k):
        _count("build")
        out = proj.output_dir / "final.mp4"
        out.write_text("video")
        return out

    patches = [
        (AN, "analyze_script", fake_analyze),
        (DS, "discover_sources", lambda a, c, **k: [
            DS.SourceCandidate(url="https://y/1", id="srcA", title="Scene")]),
        (DL, "download_candidates", fake_download),
        (FI, "available", lambda: False),
        (IX, "clip_available", lambda: True),
        (VF, "verify_and_repair", fake_verify),
        (LM, "vision_probe", lambda *a, **k: (True, "ok")),
        (RV, "write_review", lambda *a, **k: "review.html"),
        (LG, "finalize", lambda proj, segs, cfg: {
            "flagged_for_review": 0, "segments": len(segs), "mean_confidence": 1.0}),
        (O, "asr_pool_current", lambda *_a, **_k: True),
        (O, "index_all", lambda *a, **k: _count("index")),
        (O, "match_segments", fake_match),
        (O, "cut_all", lambda *a, **k: (_count("cut"), 0)[1]),
        (O, "build_video", fake_build),
        (O, "_recover_unresolved_beats", lambda *a, **k: _count("recover")),
        (O, "_fill_image_fallbacks", lambda *a, **k: _count("fallback")),
        (O, "_purge_unwanted_sources", lambda *a, **k: 0),
    ]
    saved = [(m, n, getattr(m, n, None)) for m, n, _ in patches]
    tmp = tempfile.mkdtemp()
    try:
        for m, n, fn in patches:
            setattr(m, n, fn)
        kw = dict(topic="t", script_text="a real script line", movie_hint="Movie",
                  policy="approved_testing", max_sources=4, do_build=True, verify=True)

        raised = None
        try:
            O.produce_auto(tmp, **kw)
        except O.PipelineError as exc:
            raised = exc
        assert raised is not None, "a promotion cut failure must fail the verify stage"
        assert "materialization failed" in str(raised)
        assert calls.get("verify") == 1
        assert calls.get("recover", 0) == 0, "recovery must not reinterpret infra failure"
        assert calls.get("fallback", 0) == 0, "fallback must not hide infra failure"
        assert calls.get("build", 0) == 0, "assembly must not run after failed promotion"
        persisted = ClipProject.load(tmp)
        assert O._ckpt(persisted)["stages"].get("verify") is None, \
            "failed promotion materialization must not checkpoint verify as clean"

        O.produce_auto(tmp, resume=True, **kw)
        assert calls.get("verify") == 2, "Resume must retry the uncheckpointed verify stage"
        assert calls.get("analyze") == 1 and calls.get("download") == 1, \
            f"Resume must reuse completed upstream work: {calls}"
        assert calls.get("match") == 1 and calls.get("cut") == 1, \
            f"Resume must not rematch or recut upstream selections: {calls}"
        assert calls.get("recover") == 1 and calls.get("fallback") == 1
        assert calls.get("build") == 1
        persisted = ClipProject.load(tmp)
        assert O._ckpt(persisted)["stages"].get("verify") is not None, \
            "a clean retry must checkpoint verify"
    finally:
        for m, n, orig in saved:
            if orig is not None:
                setattr(m, n, orig)
        shutil.rmtree(tmp, ignore_errors=True)


def test_final_black_gate_honors_review_mode():
    """The final-video black gate must block in production but WARN+DELIVER in review mode — the
    same warn-mode contract the footage + unverified-exact gates follow. A 1s dark region must not
    withhold a 13-min review draft a human is meant to inspect."""
    import types, tempfile, shutil
    from pathlib import Path
    from vidlore.clipstudio import build as B
    saved = {k: getattr(B, k, None) for k in
             ("_probe_duration", "_scan_coverage_reason", "_final_timestamp_reachable",
              "_frame_luma_hi", "ffmpeg_exe")}
    saved_run = B.subprocess.run
    tmp = Path(tempfile.mkdtemp()); (tmp / "output").mkdir(); work = tmp / "output" / "work"; work.mkdir()
    B._probe_duration = lambda p: 600.0
    B._scan_coverage_reason = lambda *a, **k: None
    B._final_timestamp_reachable = lambda *a, **k: True
    B._frame_luma_hi = lambda fp: 10.0                       # every frame dark → sustained-dark run
    B.ffmpeg_exe = lambda: "/bin/true"
    def _run(cmd, *a, **k):
        dest = Path(cmd[-1]).parent; dest.mkdir(parents=True, exist_ok=True)
        for i in range(20): (dest / f"b_{i:05d}.jpg").write_bytes(b"x")
        return types.SimpleNamespace(returncode=0)
    B.subprocess.run = _run
    try:
        # BLOCK (production) → raises + quarantines
        res = tmp / "output" / "fb.mp4"; res.write_bytes(b"v")
        os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "block"
        blocked = False
        try:
            B._final_video_black_gate(res, work, log=lambda m: None)
        except RuntimeError:
            blocked = True
        assert blocked and not res.exists(), "production must hard-block + quarantine a dark region"
        # WARN (review draft) → delivers, no raise, logs the warning
        res2 = tmp / "output" / "fw.mp4"; res2.write_bytes(b"v")
        os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"
        logs = []
        out = B._final_video_black_gate(res2, work, log=logs.append)
        assert out == res2 and res2.exists(), "review mode must DELIVER the draft, not quarantine"
        assert any("BLACK-QA (mode=warn" in l for l in logs), "review mode must warn loudly"
    finally:
        for k, v in saved.items():
            if v is not None:
                setattr(B, k, v)
        B.subprocess.run = saved_run
        os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_and_outage_wiring_present():
    src = open(os.path.join(ROOT, "vidlore", "clipstudio", "orchestrate.py"), encoding="utf-8").read()
    assert "vision_probe" in src and "VISION PREFLIGHT" in src, "preflight probe must be wired"
    assert "verifier_down" in src, "the post-verify outage path must be handled"
    web = open(os.path.join(ROOT, "vidlore", "clipstudio", "web.py"), encoding="utf-8").read()
    assert "vision_down" in web and "VisionBackendError" in web, "portal must map the outage status"
    assert "vision_kind" in web


TESTS = [
    test_vision_error_classifier,
    test_vision_backend_error_is_retryable_and_distinct,
    test_vision_outage_does_not_checkpoint_verify_and_skips_the_grind,
    test_verify_materialization_failure_rolls_back_checkpoint_and_resume_retries,
    test_final_black_gate_honors_review_mode,
    test_preflight_and_outage_wiring_present,
    test_signature_is_stable_and_sensitive,
    test_stage_skip_only_when_resuming_matching_and_artifact_present,
    test_checkpoint_version_bump_invalidates_old_records,
    test_doomed_beat_with_no_compatible_predecessor_blocks_early,
    test_rejected_beat_rescued_by_same_scene_clip_is_not_flagged,
    test_rejected_beat_with_installed_still_is_not_rejected,
    test_gate_disabled_env_returns_none,
    test_predictor_is_sound_never_over_claims,
    test_cut_resume_reuses_existing_clip_without_reencoding,
    test_produce_auto_resume_skips_completed_stages_end_to_end,
    test_orchestrator_and_cut_expose_resume,
    test_portal_exposes_resume_and_retry,
]

if __name__ == "__main__":
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            FAILS.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - len(FAILS)}/{len(TESTS)} passed")
    sys.exit(1 if FAILS else 0)
