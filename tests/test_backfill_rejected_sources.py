"""A gate-rejected source must be REPLACED, not silently subtracted from the pool.

Measured on job 69d80e9dd4_v4: 11 of 84 ok sources were dropped by the pool gates (7 burned-caption
re-uploads, 2 promo-card compilations, 1 STAR India screener with a burned timecode, 1 talking-head).
Every one of those rejections is correct — the footage really is unusable. The bug was what came
next: discovery had already spent its budget, so the drops just made the pool thinner, and among the
casualties were the most on-topic upload in the whole pool ("The Trial of Petyr Baelish", 1080p,
23 min) and the only clip of the dagger handover. 207 beats then asked for the trial and were served
scene packs of neighbouring scenes — right character, wrong scene, 135 times over.

These tests pin the replacement pass: it runs before match, searches with the LOST upload's own
title, and cannot spin.
"""
from __future__ import annotations

import types

import pytest

from vidlore.clipstudio import orchestrate as O


class _Src:
    def __init__(self, sid, title, status="ok", url="", error=""):
        self.id, self.title, self.status = sid, title, status
        self.url = url or f"https://y/{sid}"
        self.error = error


class _Proj:
    def __init__(self, sources, rejected):
        self.sources = sources
        self.meta = {"auto_rejected_sources": list(rejected)}
        self.saved = 0

    def save(self):
        self.saved += 1


def _wire(monkeypatch, *, cands=None, downloads=None):
    """Stub the network edges; capture what the pass asked for."""
    seen = {"queries": [], "pool_calls": 0, "index_calls": 0, "downloaded": []}
    cands = cands or []
    downloads = downloads if downloads is not None else []

    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "_load_pool",
                        lambda *a, **k: seen.__setitem__("pool_calls", seen["pool_calls"] + 1))
    monkeypatch.setattr(M, "usable_shot_yield", lambda *_a, **_k: (1, 1))

    def _disc(analysis, cfg, *, segments=None, progress=None, extra_queries=None):
        seen["queries"].append(list(extra_queries or []))
        return list(cands)

    def _dl(proj, new, cfg, *, policy=None, limit=None, progress=None, on_ready=None):
        for s in downloads:
            proj.sources.append(s)
        seen["downloaded"].append(len(new))

    def _idx(proj, cfg, **k):
        seen["index_calls"] += 1

    monkeypatch.setattr("vidlore.clipstudio.discover.discover_sources", _disc)
    monkeypatch.setattr("vidlore.clipstudio.download.download_candidates", _dl)
    monkeypatch.setattr("vidlore.clipstudio.index.index_all", _idx)
    return seen


def _run(proj, **kw):
    # refs/faceid must be present: a replacement indexed without cast data cannot compete
    # (w_face is 0.30 of the score), so the pass refuses to run without them.
    kw.setdefault("refs", {"Aidan Gillen": object()})
    kw.setdefault("faceid_obj", object())
    return O._backfill_rejected_sources(
        proj, [], types.SimpleNamespace(movie_title="Game of Thrones", key_scenes=[]),
        types.SimpleNamespace(), roster=None,
        policy="approved_testing", max_sources=8, show_title="Game of Thrones",
        log=kw.pop("log", lambda m: None), **kw)


def test_refuses_to_run_without_face_id_references(monkeypatch):
    """A source indexed without Face-ID starts w_face (0.30) behind every incumbent, so it loses
    the very beats it was fetched for and the failure reads as 'the picker ignores good footage'."""
    # Keep this unit focused on the already-derived quality rejection. The real pool probe needs
    # indexed media and would correctly reclassify this tiny synthetic source fixture.
    monkeypatch.setattr("vidlore.clipstudio.match._load_pool", lambda *_a, **_k: None)
    msgs = []
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    n = O._backfill_rejected_sources(
        proj, [], types.SimpleNamespace(movie_title="Game of Thrones", key_scenes=[]),
        types.SimpleNamespace(), refs={}, faceid_obj=None, roster=None,
        policy="approved_testing", max_sources=8, show_title="Game of Thrones", log=msgs.append)
    assert n == 0
    assert any("no Face-ID references" in m for m in msgs)


def test_kill_switch_disables_the_pass(monkeypatch):
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", "0")
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("a", "The Trial of Petyr Baelish")], ["a"])
    assert _run(proj) == 0
    assert seen["pool_calls"] == 0, "kill switch must skip before doing any work"


def test_no_rejections_does_nothing(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("a", "clean scene pack")], [])
    assert _run(proj) == 0
    assert not seen["queries"], "nothing was lost, so nothing should be searched for"


def test_searches_with_the_lost_upload_s_own_title(monkeypatch):
    """No beat-derived query reproduces 'the 23-minute trial upload we just threw away'."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish | The One Detail Everyone Misses"),
                  _Src("d", "Bran and LittleFinger / Catspaw Dagger")], ["t", "d"])
    _run(proj)
    q = seen["queries"][0]
    assert any("Trial of Petyr Baelish" in x for x in q)
    assert any("Catspaw Dagger" in x for x in q)


def test_channel_furniture_is_stripped_from_the_query(monkeypatch):
    """'| 4K' / '| English Subtitles' would drag the search back to more re-uploads."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", "Littlefinger Trial [HBO] | English Subtitles | 4K")], ["t"])
    _run(proj)
    q = " ".join(seen["queries"][0]).lower()
    assert "littlefinger trial" in q
    assert "subtitle" not in q and "4k" not in q and "hbo" not in q


def test_admits_and_indexes_the_replacement(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    new = _Src("clean", "Littlefinger Trial FULL SCENE", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="Littlefinger Trial")],
                 downloads=[new])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    assert _run(proj) == 1
    assert seen["index_calls"] == 1, "a downloaded replacement is useless until it is indexed"


def test_does_not_re_search_the_same_lost_title_twice(monkeypatch):
    """Round 2 must not spend the budget re-asking for a title round 1 already chased."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "3")
    new = _Src("clean", "replacement", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="replacement")],
                 downloads=[new])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    _run(proj)
    flat = [q for rnd in seen["queries"] for q in rnd]
    assert len(flat) == len(set(flat)), f"a title was searched twice: {flat}"


def test_no_new_candidate_stops_cleanly(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch, cands=[])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert any("no NEW candidate" in m for m in msgs)
    assert seen["downloaded"] == [], "nothing new means nothing to download"
    assert proj.meta["backfill_audit"]["status"] == "complete"
    assert proj.meta["backfill_audit"]["reason"] == "no_new_candidate"


def test_a_failing_edge_degrades_instead_of_killing_the_render(monkeypatch):
    """Backfill is an improvement pass — it must never take the render down with it."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    _wire(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("vidlore.clipstudio.discover.discover_sources", _boom)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert any("discovery failed" in m for m in msgs)
    assert proj.meta["backfill_audit"]["status"] == "incomplete"
    assert proj.meta["backfill_audit"]["reason"].startswith("discovery_failed:")


def test_audit_is_persisted(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    _wire(monkeypatch, cands=[])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    _run(proj)
    a = proj.meta.get("backfill_audit")
    assert a and a["rounds"], "the pass must record what it looked for and what it found"
    assert a["rounds"][0]["rejected"] == ["t"]
    assert a["schema_version"] == 2 and a["status"] == "complete"


def test_footage_signatures_use_the_post_backfill_pool_and_ignore_duplicate_records():
    seg = types.SimpleNamespace(
        index=0, text="beat", visual_policy="exact_scene", required_entity="",
        required_kind="", scene_query="scene", quote="")
    one = _Src("one", "one")
    duplicate = _Src("one", "duplicate record sharing the same index namespace")
    two = _Src("two", "newly admitted clean copy")

    before = O._footage_stage_signatures(
        "download", [one], force_index=False, segments=[seg], verify=True)
    duplicate_only = O._footage_stage_signatures(
        "download", [one, duplicate], force_index=False, segments=[seg], verify=True)
    after = O._footage_stage_signatures(
        "download", [one, duplicate, two], force_index=False, segments=[seg], verify=True)

    assert duplicate_only == before
    assert after != before


def test_completed_backfill_checkpoint_skips_but_incomplete_audit_retries():
    proj = _Proj([], [])
    sig = O._sig("stable upstream", "backfillv3")
    proj.meta["backfill_audit"] = {
        "schema_version": 2, "status": "complete", "input_sig": sig}
    O._stage_done(proj, "backfill", sig)
    assert O._stage_skip(
        proj, "backfill", sig, resume=True,
        artifact_ok=O._backfill_audit_complete_for(proj, sig))

    proj.meta["backfill_audit"]["status"] = "incomplete"
    assert not O._stage_skip(
        proj, "backfill", sig, resume=True,
        artifact_ok=O._backfill_audit_complete_for(proj, sig))


def test_incomplete_backfill_forces_cached_footage_path_then_complete_resume_skips():
    """A failed pass must remain schedulable after that run checkpoints every later stage."""
    proj = _Proj([], [])
    seg = types.SimpleNamespace(
        index=0, text="beat", visual_policy="exact_scene", required_entity="",
        required_kind="", scene_query="scene", quote="")
    cfg = types.SimpleNamespace(discover_target=18, max_height=1080)
    analysis = types.SimpleNamespace(
        movie_title="Game of Thrones", video_type="multi_scene",
        actors=["Aidan Gillen"], characters=[], key_scenes=[])
    footage = O._footage_stage_signatures(
        "download", [], force_index=False, segments=[seg], verify=True)
    sig_match, sig_verify, sig_recover = footage[1], footage[3], footage[4]
    for stage, sig in (("match", sig_match), ("verify", sig_verify),
                       ("recover", sig_recover)):
        O._stage_done(proj, stage, sig)

    sig_backfill = O._backfill_input_signature(
        "download", [seg], policy="approved_testing", max_sources=8,
        show_title="Game of Thrones", enabled=True, rounds=2,
        cfg=cfg, analysis=analysis)
    proj.meta["backfill_audit"] = {
        "schema_version": 2, "status": "incomplete", "input_sig": sig_backfill}
    downstream = {
        "skip_match": O._stage_skip(proj, "match", sig_match, resume=True),
        "skip_verify": O._stage_skip(proj, "verify", sig_verify, resume=True),
        "skip_recover": O._stage_skip(proj, "recover", sig_recover, resume=True),
    }
    skip_backfill = O._stage_skip(
        proj, "backfill", sig_backfill, resume=True,
        artifact_ok=O._backfill_audit_complete_for(proj, sig_backfill))
    assert all(downstream.values()) and not skip_backfill
    assert O._footage_stages_required(
        **downstream, backfill_enabled=True, skip_backfill=skip_backfill), \
        "resume 1 must rebuild Face-ID/backfill context and retry the incomplete pass"

    proj.meta["backfill_audit"]["status"] = "complete"
    O._stage_done(proj, "backfill", sig_backfill)
    skip_backfill = O._stage_skip(
        proj, "backfill", sig_backfill, resume=True,
        artifact_ok=O._backfill_audit_complete_for(proj, sig_backfill))
    assert skip_backfill
    assert not O._footage_stages_required(
        **downstream, backfill_enabled=True, skip_backfill=skip_backfill), \
        "resume 2 may skip only after the same semantic inputs completed conclusively"


def test_backfill_signature_binds_cfg_analysis_and_gate_env_but_not_recovery_pool(monkeypatch):
    seg = types.SimpleNamespace(
        index=0, text="beat", visual_policy="exact_scene", required_entity="Petyr Baelish",
        required_kind="character", scene_query="Littlefinger trial", quote="")
    cfg = types.SimpleNamespace(
        discover_target=18, discover_min_height=300, max_height=1080,
        scene_threshold=27.0, detect_ocr=True)
    analysis = types.SimpleNamespace(
        movie_title="Game of Thrones", video_type="multi_scene",
        actors=["Aidan Gillen"], characters=[{"name": "Petyr", "actor": "Aidan Gillen"}],
        key_scenes=["Littlefinger trial"])

    def _make(current_cfg=cfg, current_analysis=analysis):
        return O._backfill_input_signature(
            "download", [seg], policy="approved_testing", max_sources=8,
            show_title="Game of Thrones", enabled=True, rounds=2,
            cfg=current_cfg, analysis=current_analysis)

    base = _make()
    cfg_changed = types.SimpleNamespace(**vars(cfg))
    cfg_changed.discover_min_height = 720
    assert _make(current_cfg=cfg_changed) != base
    analysis_changed = types.SimpleNamespace(**vars(analysis))
    analysis_changed.actors = ["Aidan Gillen", "Sophie Turner"]
    assert _make(current_analysis=analysis_changed) != base

    with monkeypatch.context() as gate_env:
        gate_env.setenv("VIDLORE_CLIPSTUDIO_OCR_GATE", "__changed_for_test__")
        assert _make() != base
    with monkeypatch.context() as yield_env:
        yield_env.setenv("VIDLORE_CLIPSTUDIO_UNREADABLE_HI", "__changed_for_test__")
        assert _make() != base
    with monkeypatch.context() as perf_env:
        perf_env.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "__irrelevant_changed__")
        assert _make() == base, "performance-only knobs must not replay a completed web search"

    # Targeted post-match recovery changes the searchable pool, not the global clean-copy search's
    # authored/config inputs. It must invalidate match signatures, but not replay backfill forever.
    proj = _Proj([_Src("original", "original")], [])
    before_pool_add = _make()
    proj.sources.append(_Src("targeted-recovery", "new exact-scene source"))
    assert _make() == before_pool_add


def test_real_config_performance_toggle_does_not_replay_completed_backfill(monkeypatch):
    import dataclasses
    from vidlore.clipstudio.config import ClipConfig

    for key in ("VIDLORE_CLIPSTUDIO_CUT_WORKERS",
                "VIDLORE_CLIPSTUDIO_WHISPER_THREADS",
                "VIDLORE_CLIPSTUDIO_CONCURRENCY"):
        monkeypatch.delenv(key, raising=False)
    seg = types.SimpleNamespace(
        index=0, text="beat", visual_policy="exact_scene", required_entity="",
        required_kind="", scene_query="scene", quote="")
    analysis = types.SimpleNamespace(
        movie_title="Game of Thrones", video_type="multi_scene",
        actors=[], characters=[], key_scenes=[])

    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "0")
    normal = ClipConfig()
    normal_sig = O._backfill_input_signature(
        "download", [seg], policy="approved_testing", max_sources=8,
        show_title="Game of Thrones", enabled=True, rounds=2,
        cfg=normal, analysis=analysis)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_MAX_CPU", "1")
    turbo = ClipConfig()
    turbo_sig = O._backfill_input_signature(
        "download", [seg], policy="approved_testing", max_sources=8,
        show_title="Game of Thrones", enabled=True, rounds=2,
        cfg=turbo, analysis=analysis)

    assert (normal.cut_workers, normal.whisper_cpu_threads, normal.download_concurrency) != \
        (turbo.cut_workers, turbo.whisper_cpu_threads, turbo.download_concurrency), \
        "the test must exercise genuinely different performance settings"
    assert turbo_sig == normal_sig

    changed_index_semantics = dataclasses.replace(
        turbo, target_clip_sec=turbo.target_clip_sec + 0.5)
    changed_sig = O._backfill_input_signature(
        "download", [seg], policy="approved_testing", max_sources=8,
        show_title="Game of Thrones", enabled=True, rounds=2,
        cfg=changed_index_semantics, analysis=analysis)
    assert changed_sig != turbo_sig, \
        "shot/index semantic changes must invalidate a completed backfill"


def test_unexpected_new_invocation_failure_rolls_back_and_cannot_bless_stale_audit(
        monkeypatch):
    proj = _Proj([], [])
    old_sig, new_sig = "old-inputs", "new-inputs"
    proj.meta["backfill_audit"] = {
        "schema_version": 2, "status": "complete", "input_sig": old_sig}
    O._stage_done(proj, "backfill", old_sig)
    proj.meta["banned_sources"] = ["preexisting-ban"]
    msgs = []
    purged = []
    monkeypatch.setattr(
        "vidlore.clipstudio.index.purge_source_index",
        lambda _proj, sid: purged.append(sid))

    def _boom():
        proj.sources.append(_Src("unscreened", "post-index mutation"))
        proj.meta["banned_sources"].append("partial-mutation")
        raise RuntimeError("unexpected programming or provider failure")

    assert O._run_backfill_invocation(proj, new_sig, _boom, log=msgs.append) is False
    audit = proj.meta["backfill_audit"]
    assert audit["input_sig"] == new_sig
    assert audit["status"] == "incomplete"
    assert audit["reason"] == "unexpected_failure:RuntimeError"
    assert not O._backfill_audit_complete_for(proj, new_sig)
    assert proj.meta["pipeline"]["stages"]["backfill"]["sig"] == old_sig
    assert proj.sources == []
    assert proj.meta["banned_sources"] == ["preexisting-ban"]
    assert purged == ["unscreened"]


def test_download_failed_status_is_incomplete_and_url_remains_retryable(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    url = "https://y/technical-failure"
    failed = _Src(
        "failed", "clean-copy candidate", status="download_failed", url=url,
        error="HTTP 403 while downloading media")
    seen = _wire(
        monkeypatch,
        cands=[types.SimpleNamespace(url=url, title="clean-copy candidate")],
        downloads=[failed])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])

    assert _run(proj, input_sig="attempt-one") == 0
    audit = proj.meta["backfill_audit"]
    assert audit["status"] == "incomplete"
    assert audit["reason"] == "download_failed_status:1"
    assert audit["input_sig"] == "attempt-one"
    assert seen["index_calls"] == 0
    assert all(s.url != url for s in proj.sources), \
        "failed URL must not enter `have` and suppress its retry on Resume"


def test_thrown_download_failure_rolls_back_rows_added_before_the_exception(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    url = "https://y/mutate-then-fail"
    _wire(
        monkeypatch,
        cands=[types.SimpleNamespace(url=url, title="candidate")],
        downloads=[])

    def _mutate_then_fail(proj, *_a, **_k):
        proj.sources.append(_Src("partial", "candidate", url=url))
        raise RuntimeError("transport collapsed after manifest mutation")

    monkeypatch.setattr("vidlore.clipstudio.download.download_candidates", _mutate_then_fail)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])

    assert _run(proj, input_sig="thrown") == 0
    assert proj.meta["backfill_audit"]["status"] == "incomplete"
    assert proj.meta["backfill_audit"]["reason"] == "download_failed:RuntimeError"
    assert all(s.url != url for s in proj.sources)


def test_partial_download_failure_rolls_back_the_whole_unscreened_batch(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    good_url, bad_url = "https://y/good", "https://y/bad"
    good = _Src("good", "usable clean copy", url=good_url)
    failed = _Src(
        "failed", "possibly exact clean copy", status="download_failed", url=bad_url,
        error="timed out")
    _wire(
        monkeypatch,
        cands=[types.SimpleNamespace(url=good_url, title="usable clean copy"),
               types.SimpleNamespace(url=bad_url, title="possibly exact clean copy")],
        downloads=[good, failed])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])

    assert _run(proj, input_sig="partial") == 0
    assert proj.meta["backfill_audit"]["status"] == "incomplete"
    assert all(s.url != good_url for s in proj.sources), \
        "a successful sibling has not passed the title/yield screen and must not pollute match"
    assert all(s.url != bad_url for s in proj.sources), \
        "the failed sibling must remain retryable even when another candidate succeeded"


def test_index_failure_rolls_back_batch_and_same_candidate_retries(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    url = "https://y/retry-after-index"
    replacement = _Src("replacement", "usable clean copy", url=url)
    seen = _wire(
        monkeypatch,
        cands=[types.SimpleNamespace(url=url, title="usable clean copy")],
        downloads=[replacement])
    index_attempts = []

    def _index_then_recover(*_args, **_kwargs):
        index_attempts.append(True)
        if len(index_attempts) == 1:
            raise RuntimeError("index backend unavailable")

    monkeypatch.setattr("vidlore.clipstudio.index.index_all", _index_then_recover)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])

    assert _run(proj, input_sig="same-semantic-inputs") == 0
    assert proj.meta["backfill_audit"]["status"] == "incomplete"
    assert proj.meta["backfill_audit"]["reason"] == "indexing_failed:RuntimeError"
    assert all(s.url != url for s in proj.sources), \
        "unindexed URL must not enter `have` and masquerade as an exhausted search"

    assert _run(proj, input_sig="same-semantic-inputs") == 1
    assert proj.meta["backfill_audit"]["status"] == "complete"
    assert seen["downloaded"] == [1, 1]
    assert len(index_attempts) == 2


def test_policy_and_checksum_exclusions_are_conclusive_download_outcomes(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    blocked_url, duplicate_url = "https://y/blocked", "https://y/duplicate"
    blocked = _Src(
        "blocked", "rights-blocked copy", status="blocked_no_permission", url=blocked_url)
    duplicate = _Src(
        "duplicate", "already represented bytes", status="duplicate", url=duplicate_url)
    _wire(
        monkeypatch,
        cands=[types.SimpleNamespace(url=blocked_url, title="rights-blocked copy"),
               types.SimpleNamespace(url=duplicate_url, title="already represented bytes")],
        downloads=[blocked, duplicate])
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])

    assert _run(proj, input_sig="conclusive") == 0
    audit = proj.meta["backfill_audit"]
    assert audit["status"] == "complete"
    assert audit["reason"] == "download_conclusive_policy_or_duplicate"
    assert audit["input_sig"] == "conclusive"


@pytest.mark.parametrize("title,want", [
    ("Arya Cuts Littlefinger's Throat - 1080p", "arya cuts littlefinger's throat"),
    ("Bran Stark Scene Pack (GOT S7)", "bran stark scene pack"),
])
def test_query_cleaning_keeps_the_scene_words(monkeypatch, title, want):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("t", title)], ["t"])
    _run(proj)
    assert want in " ".join(seen["queries"][0]).lower()


# ---------------------------------------------------------------- reason-aware replacement

def _proj_with_reasons(reasons):
    p = _Proj([_Src(sid, f"title of {sid}") for sid in reasons], list(reasons))
    p.meta["auto_rejected_reasons"] = dict(reasons)
    return p


def test_only_quality_rejects_are_replaced(monkeypatch):
    """A subtitled copy of the right scene is worth replacing. An interview is not — searching for
    a cleaner copy of a talking-head just buys another talking-head."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _proj_with_reasons({
        "subbed": "subtitled_copy",          # right footage, unusable copy  -> replace
        "wm": "watermarked",                 # ditto                          -> replace
        "promo": "promo_overlay",            # ditto                          -> replace
        "interview": "talking_head_visual",  # never wanted                   -> skip
        "reaction": "reaction",              # never wanted                   -> skip
        "hotd": "wrong_show",                # never wanted                   -> skip
        "fanart": "graphics",                # never wanted                   -> skip
    })
    _run(proj)
    q = " ".join(seen["queries"][0]).lower()
    for sid in ("subbed", "wm", "promo"):
        assert sid in q, f"{sid} is a quality reject and must be replaced"
    for sid in ("interview", "reaction", "hotd", "fanart"):
        assert sid not in q, f"{sid} is content we never wanted — do not go looking for more"


def test_unknown_reason_defaults_to_replaceable(monkeypatch):
    """An older project has no reason map; treat its rejects as quality rather than silently
    replacing nothing."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _Proj([_Src("legacy", "The Trial of Petyr Baelish")], ["legacy"])
    proj.meta.pop("auto_rejected_reasons", None)
    _run(proj)
    assert seen["queries"] and "Trial of Petyr Baelish" in " ".join(seen["queries"][0])


def test_all_content_rejects_means_no_search_at_all(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    seen = _wire(monkeypatch)
    proj = _proj_with_reasons({"a": "reaction", "b": "wrong_show"})
    msgs = []
    assert _run(proj, log=msgs.append) == 0
    assert not seen["queries"]
    assert any("no gate-rejected source to replace" in m for m in msgs)


# ---------------------------------------------------------------- yield-proven replacements

def test_a_replacement_with_no_usable_shots_is_not_counted(monkeypatch):
    """Clearing the SOURCE gates is not the same as being able to air.

    The live run fetched a replacement titled 'Littlefinger gives Catspaw dagger to Bran Stark' —
    precisely the footage 8 beats were asking for — and it was another screener with 'FOR INTERNAL
    VIEWING ONLY' burned into the picture. It passed every source-level gate and then lost all 11
    shots to the shot-level text gate, so it won zero beats while the pass reported '+3 clean
    sources'. A replacement has to prove a usable yield."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    dud = _Src("screener", "Littlefinger gives Catspaw dagger to Bran", url="https://y/screener")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/screener", title="dagger handover")],
                 downloads=[dud])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (0, 11))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 0, "a 0-yield source is not a replacement"
    assert any("0 usable shots" in m for m in msgs)
    assert "screener" in (proj.meta.get("banned_sources") or []), \
        "an unusable replacement must be banned, not left to pollute the still/breakout pools"


def test_a_replacement_with_usable_shots_counts(monkeypatch):
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    good = _Src("clean", "Littlefinger Trial FULL SCENE", url="https://y/clean")
    seen = _wire(monkeypatch,
                 cands=[types.SimpleNamespace(url="https://y/clean", title="trial")],
                 downloads=[good])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (33, 34))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    assert _run(proj, log=msgs.append) == 1
    assert any("usable source(s) indexed" in m for m in msgs)
    assert "clean" not in (proj.meta.get("banned_sources") or [])


def test_unmeasurable_yield_is_retryable_and_rolls_back_unscreened_batch(monkeypatch):
    """A technical yield failure is neither a ban nor proof that the source can air."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    s = _Src("odd", "some scene", url="https://y/odd")
    _wire(monkeypatch, cands=[types.SimpleNamespace(url="https://y/odd", title="x")], downloads=[s])
    from vidlore.clipstudio import match as M

    def _boom(*a, **k):
        raise RuntimeError("index unreadable")

    monkeypatch.setattr(M, "usable_shot_yield", _boom)
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    purged = []
    monkeypatch.setattr(
        "vidlore.clipstudio.index.purge_source_index",
        lambda _proj, sid: purged.append(sid))

    assert _run(proj, input_sig="yield-attempt") == 0
    assert proj.meta["backfill_audit"]["status"] == "incomplete"
    assert proj.meta["backfill_audit"]["reason"] == \
        "yield_measurement_failed:RuntimeError"
    assert all(source.id != "odd" for source in proj.sources)
    assert "odd" not in (proj.meta.get("banned_sources") or [])
    assert purged == ["odd"]


# ---------------------------------------------------------------- what a replacement may BE

def _title_filter(title):
    """Exercise the real predicate through a one-source backfill run."""
    import re, inspect
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O._backfill_rejected_sources)
    assert "_title_ok" in src, "the pass must screen replacement TITLES, not just shot yield"
    return src


@pytest.mark.parametrize("title,blocked", [
    # measured on job benjen_v2 — every one of these was ADMITTED into a Benjen Stark essay
    ("Cersei and Jaime Lannister - Game of Thrones - All Scenes Part 3/8", True),
    ("Cersei and Jaime Lannister - Game of Thrones - All Scenes Part 4/8", True),
    ("Games of Thrones - S07E07 - Behind the Scene - Dragon Pit Meeting", True),
    ("A tale of Benjen Stark -  A Game of Thrones fanfiction - Winter", True),
    # ...and these must still get through
    ("Game of Thrones 7x06 - Benjen Saves Jon Snow", False),
    ("Game of Thrones S7E6 - Beyond The Wall | Wight Bear attack", False),
    ("Night King Destroys The Wall - Game of Thrones S07E07", False),
])
def test_a_replacement_must_be_scene_footage_not_an_anthology(monkeypatch, title, blocked):
    """A usable shot yield proves a source CAN air, never that it SHOULD.

    The 13-minute "All Scenes Part 3/8" compilation cleared every per-shot gate with 162 usable
    shots and then fed the breakout miner a Season-1 Cersei/Ned conversation. The pass had searched
    with the REJECTED upload's own title — which is only a good query when that upload was genuinely
    wanted; this one was already marginal, so asking for a cleaner copy of it bought an anthology of
    a different story."""
    monkeypatch.delenv("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", raising=False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")
    new = _Src("cand", title, url="https://y/cand")
    _wire(monkeypatch,
          cands=[types.SimpleNamespace(url="https://y/cand", title=title)],
          downloads=[new])
    from vidlore.clipstudio import match as M
    monkeypatch.setattr(M, "usable_shot_yield", lambda proj, sid, cfg=None: (162, 178))
    proj = _Proj([_Src("t", "The Trial of Petyr Baelish")], ["t"])
    msgs = []
    n = _run(proj, log=msgs.append)
    if blocked:
        assert n == 0, f"{title!r} must never be admitted"
        assert any("rejected" in m for m in msgs)
    else:
        assert n == 1, f"{title!r} is legitimate scene footage and must pass"


def test_the_title_screen_runs_before_the_expensive_yield_probe():
    """No point decoding shots for a source we already know we will not use."""
    import inspect
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O._backfill_rejected_sources)
    assert src.index("_why_bad = _title_ok(") < src.index("usable_shot_yield(proj, s.id, cfg)")
