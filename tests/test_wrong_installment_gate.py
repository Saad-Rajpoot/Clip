"""Sibling-franchise titles must not enter a single-show footage pool.

Regression: a Game of Thrones job admitted
``House Of The Dragon Valyrian steel dagger timeline in Game Of Thrones``.  The sibling gate found
``House of the Dragon`` but then waived the rejection merely because the title also contained the
target phrase ``Game of Thrones``.  That is a mixed-installment title, not proof of clean GoT pixels.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from vidlore.clipstudio import discover as D
from vidlore.clipstudio import match as M
from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import ClipProject, SourceVideo


LEAK_TITLE = "House Of The Dragon Valyrian steel dagger timeline in Game Of Thrones"


def test_measured_mixed_title_is_wrong_for_game_of_thrones():
    assert D._wrong_installment("Game of Thrones", LEAK_TITLE)


def test_candidate_cannot_self_exempt_by_appending_the_target_show_name():
    for title in (
        "Game of Thrones vs House of the Dragon dragon battle comparison",
        "House of the Dragon scenes compared with Game of Thrones",
        "HOTD dagger history in Game of Thrones",
        "Game of Thrones timeline featuring Rhaenyra and Alicent",
    ):
        assert D._wrong_installment("Game of Thrones", title), title


def test_single_show_sources_and_shared_lore_names_are_untouched():
    kept = (
        "Game of Thrones S04E02 Joffrey death scene",
        "Game of Thrones Viserys Targaryen recounts the Dothraki sea",
        "Game of Thrones Aegon the Conqueror history mentioned by Shireen",
        "Daenerys enters the House of the Undying | Game of Thrones 2x10",
    )
    for title in kept:
        assert not D._wrong_installment("Game of Thrones", title), title


def test_each_installment_keeps_its_own_clean_sources():
    assert not D._wrong_installment(
        "House of the Dragon", "House of the Dragon S01E09 Aegon coronation scene")
    assert not D._wrong_installment(
        "Game of Thrones", "Game of Thrones S01E01 King Robert arrives at Winterfell")
    assert D._wrong_installment(
        "House of the Dragon", "Game of Thrones Daenerys rides Drogon scene")


def test_cross_installment_analysis_target_is_not_guessed_at():
    target = "Game of Thrones vs House of the Dragon"
    assert not D._wrong_installment(target, "House of the Dragon Rhaenyra scene")
    assert not D._wrong_installment(target, "Game of Thrones Daenerys scene")


def test_discovery_rejects_measured_title_before_download(monkeypatch):
    wrong = D.SourceCandidate(
        url="https://video/wrong", id="wrong", title=LEAK_TITLE,
        provider="youtube", duration=120.0, height=1080, query="dagger")
    clean = D.SourceCandidate(
        url="https://video/clean", id="clean",
        title="Game of Thrones S01E03 Littlefinger identifies the dagger scene",
        provider="youtube", duration=120.0, height=1080, query="dagger")
    monkeypatch.setattr(D, "build_queries", lambda *_a, **_k: ["dagger"])
    monkeypatch.setattr(D, "anchor_queries", lambda *_a, **_k: [])
    monkeypatch.setattr(
        D, "_ytsearch_ex", lambda *_a, **_k: ([wrong, clean], D.STATUS_OK))
    monkeypatch.setattr(
        D, "_archive_search_ex", lambda *_a, **_k: ([], D.STATUS_EMPTY))
    analysis = NS(
        movie_title="Game of Thrones", video_type="multi_scene", anchor_scenes=[],
        actors=[], characters=[], key_scenes=[], visual_keywords=[], year="",
        episode_hint="")
    cfg = ClipConfig()
    cfg.discover_resolve_quality = False

    got = D.discover_sources(analysis, cfg)

    assert [c.id for c in got] == ["clean"]
    assert wrong.reject_reason == "wrong-show (franchise sibling/prequel)"


def test_cached_source_is_rejected_by_match_and_shared_ban_reason(tmp_path):
    proj = ClipProject(name="wrong-show", root=str(tmp_path))
    proj.ensure_dirs()
    proj.sources = [SourceVideo(
        id="house_of_the_dragon_va_d546b1ad", url="https://video/wrong",
        title=LEAK_TITLE, permission="owner", status="ok")]

    pool = M._load_pool(proj, ClipConfig(), show_title="Game of Thrones")

    assert pool == []
    assert proj.meta["auto_rejected_sources"] == ["house_of_the_dragon_va_d546b1ad"]
    assert proj.meta["auto_rejected_reasons"] == {
        "house_of_the_dragon_va_d546b1ad": "wrong_show"}


def test_gate_upgrade_replays_match_and_downstream_without_reindexing(
        tmp_path, monkeypatch):
    """An old resume must reach _load_pool so cached mixed-installment rows are re-filtered."""
    proj = ClipProject(name="resume-gate", root=str(tmp_path))
    proj.ensure_dirs()
    seg = NS(
        index=0, text="beat", visual_policy="exact_scene", required_entity="",
        required_kind="", scene_query="scene", quote="")
    kwargs = dict(
        force_index=False, segments=[seg], verify=True, asr_signature="asr-v1")

    monkeypatch.setattr(O, "_MATCH_GATE_VERSION", "gatev2-graphics")
    old = O._footage_stage_signatures("download", [], **kwargs)
    for stage, sig in zip(("index", "match", "cut", "verify", "recover"), old):
        O._stage_done(proj, stage, sig)
    assert all(O._stage_skip(proj, stage, sig, resume=True)
               for stage, sig in zip(
                   ("index", "match", "cut", "verify", "recover"), old))

    monkeypatch.setattr(
        O, "_MATCH_GATE_VERSION", "gatev3-wrong-installment-mixed-title")
    current = O._footage_stage_signatures("download", [], **kwargs)

    assert current[0] == old[0], "the valid downloaded/indexed pool remains reusable"
    assert all(not O._stage_skip(proj, stage, sig, resume=True)
               for stage, sig in zip(
                   ("match", "cut", "verify", "recover"), current[1:]))


def test_gate_upgrade_invalidates_completed_backfill_audit(monkeypatch):
    """Discovery's old clean-copy audit cannot bless titles admitted under the old sibling gate."""
    seg = NS(
        index=0, text="beat", visual_policy="exact_scene", required_entity="",
        required_kind="", scene_query="scene", quote="")
    cfg = NS(discover_target=18, max_height=1080)
    analysis = NS(
        movie_title="Game of Thrones", video_type="multi_scene",
        actors=[], characters=[], key_scenes=[])

    def signature():
        return O._backfill_input_signature(
            "download", [seg], policy="approved_testing", max_sources=8,
            show_title="Game of Thrones", enabled=True, rounds=2,
            cfg=cfg, analysis=analysis)

    monkeypatch.setattr(
        O, "_BACKFILL_SIGNATURE_VERSION",
        "backfillv4-completion-aware-semantic-inputs")
    old = signature()
    monkeypatch.setattr(
        O, "_BACKFILL_SIGNATURE_VERSION",
        "backfillv5-wrong-installment-mixed-title")

    assert signature() != old
