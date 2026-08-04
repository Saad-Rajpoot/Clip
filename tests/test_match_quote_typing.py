"""Focused regressions for whole-pool quote typing in retrieval and safe era inheritance."""
import inspect
from types import SimpleNamespace as NS
from unittest import mock

from vidlore.clipstudio import era as E
from vidlore.clipstudio import match as M
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import ClipProject, ScriptSegment, Shot, SourceVideo


QUOTE = "Who does this belong to?"


def _seg(index=0, *, quote=QUOTE, text="ordinary narration", scene_query="the dagger meeting"):
    return ScriptSegment(index=index, text=text, quote=quote, scene_query=scene_query,
                         expected_visual="Catelyn examines the dagger",
                         visual_policy="exact_scene", est_duration=3.0)


def _pool(transcript=QUOTE):
    shot = Shot(source_id="show", index=0, start=0.0, end=5.0,
                transcript=transcript, quality=0.9)
    return [M._PoolShot("show", shot)]


def test_only_whole_pool_verbatim_authored_quote_can_boost_dialogue():
    cfg = ClipConfig()
    seg = _seg()
    verbatim = M._score_pool(
        seg, _pool(), None, cfg, set(), quote_branch="verbatim")[0]
    paraphrase = M._score_pool(
        seg, _pool(), None, cfg, set(), quote_branch="paraphrase")[0]
    indeterminate = M._score_pool(
        seg, _pool(), None, cfg, set(), quote_branch="indeterminate")[0]

    assert verbatim[2]["dialogue"] == 1.0
    assert verbatim[0] >= cfg.w_dialogue
    for row, branch in ((paraphrase, "paraphrase"),
                        (indeterminate, "indeterminate")):
        assert row[2]["dialogue"] == 0.0
        assert "moment_lock" not in row[2]
        assert row[2][f"quote_branch_{branch}"] == 1.0
        assert row[0] < verbatim[0]
        assert all(isinstance(value, (int, float)) for value in row[2].values())


def test_nonverbatim_authored_phrase_is_dropped_but_distinct_anchor_remains():
    anchor = "I did it to protect the woman I love."
    seg = _seg(text="He did it to protect the woman he loved.",
               quote="I did what I did to protect Sansa.")

    assert M.beat_quote_candidates(
        seg, [anchor], quote_branch="paraphrase") == [anchor]
    assert M.beat_quote_candidates(
        seg, [anchor], quote_branch="indeterminate") == [anchor]
    assert M.beat_quote_candidates(
        seg, [], quote_branch="paraphrase") == []

    seen = []

    def locate(_proj, _sid, phrase):
        seen.append(phrase)
        return (10.0, 12.0, 1.0)

    with mock.patch.object(M, "quote_span_in_source", side_effect=locate):
        span = M.locate_beat_moment(
            NS(meta={}), "show", seg, [anchor], quote_branch="paraphrase")
    assert span == (10.0, 12.0, 1.0)
    assert seen == [anchor], "the rejected authored paraphrase must never reach ASR lookup"


def test_matcher_quote_classifier_is_once_per_match_and_forwarded(tmp_path, monkeypatch):
    segs = [_seg(0), _seg(1, quote="writer paraphrase")]
    source = SourceVideo(id="show", url="local:test", title="clean show scene",
                         width=1920, height=1080)
    proj = ClipProject(name="q", root=str(tmp_path), sources=[source], segments=segs)
    pool = _pool()
    calls = []

    classifier = mock.Mock(return_value={
        0: {"branch": "verbatim"}, 1: {"branch": "paraphrase"}})

    def scored(seg, _pool_arg, _text_vec, _cfg, _faces, *args, **kwargs):
        branch = kwargs.get("quote_branch")
        calls.append((seg.index, branch))
        return [(0.5, 0.0, {f"quote_branch_{branch}": 1.0}, pool[0])]

    monkeypatch.setattr(M, "_load_pool", lambda *args, **kwargs: pool)
    monkeypatch.setattr(M, "_score_pool", scored)
    monkeypatch.setattr(M._index, "clip_available", lambda: False)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_CLEAN_COPY_GATE", "0")
    with mock.patch.object(R, "_quote_pool_branches", classifier):
        M.match_segments(proj, segs, ClipConfig())

    assert classifier.call_count == 1
    assert calls == [(0, "verbatim"), (1, "paraphrase")]
    assert proj.meta["matcher_quote_branches"] == {"0": "verbatim", "1": "paraphrase"}
    assert getattr(segs[0], "_matcher_quote_branch") == "verbatim"
    assert getattr(segs[1], "_matcher_quote_branch") == "paraphrase"


def test_quote_classifier_failure_is_indeterminate_and_never_permissive():
    seg = _seg()
    proj = NS(meta={})
    with mock.patch.object(R, "_quote_pool_branches", side_effect=RuntimeError("broken index")):
        branches = M._matcher_quote_pool_branches(proj, [seg], ClipConfig())
    assert branches == {0: "indeterminate"}
    assert proj.meta["matcher_quote_branches"] == {"0": "indeterminate"}
    assert "RuntimeError" in proj.meta["matcher_quote_branch_error"]
    assert M._dialogue_match(seg, QUOTE, quote_branch=branches[0]) == 0.0


def test_persisted_paraphrase_branch_survives_fresh_resume_in_verifier_repair():
    fresh_seg = _seg()
    proj = NS(meta={"matcher_quote_branches": {"0": "paraphrase"}})
    assert not hasattr(fresh_seg, "_matcher_quote_branch")
    branch = M._effective_matcher_quote_branch(fresh_seg, proj=proj)
    assert branch == "paraphrase"
    assert M._dialogue_match(fresh_seg, QUOTE, quote_branch=branch) == 0.0
    source = inspect.getsource(V._strict_scene_neighborhood_candidates)
    assert "_effective_matcher_quote_branch(seg, proj=proj)" in source
    assert "quote_branch=" in source


def test_character_only_overlap_cannot_infer_an_anchor_era():
    anchors = [
        {"name": "The Purple Wedding",
         "query": "Example Kingdom Purple Wedding Joffrey death S04E02",
         "episode": "S04E02"},
        {"name": "Olenna confesses to poisoning Joffrey",
         "query": "Example Kingdom Olenna confession Joffrey S07E03",
         "episode": "S07E03"},
    ]
    full = NS(movie_title="Example Kingdom", anchor_scenes=anchors,
              characters=[{"name": "Olenna Tyrell", "actor": "Diana Rigg"},
                          {"name": "Joffrey Baratheon", "actor": "Jack Gleeson"}],
              actors=["Diana Rigg", "Jack Gleeson"])
    lightweight = NS(movie_title="Example Kingdom", anchor_scenes=anchors)
    beat = _seg(scene_query="Example Kingdom Olenna Joffrey garden aftermath")
    actor_only = _seg(scene_query="Example Kingdom Diana Rigg portrait")

    assert E.beat_era(beat, "", single_scene=False, global_verified=False,
                      anchor_eras=E.anchor_token_eras(full)) == ""
    assert E.beat_era(actor_only, "", single_scene=False, global_verified=False,
                      anchor_eras=E.anchor_token_eras(full)) == ""
    # Production's lightweight shims omit the roster. Cross-anchor-common token removal still
    # prevents Joffrey + one character name from manufacturing a two-token S7 match.
    assert E.beat_era(beat, "", single_scene=False, global_verified=False,
                      anchor_eras=E.anchor_token_eras(lightweight)) == ""

    confession = _seg(scene_query="Example Kingdom Olenna confesses poisoning Joffrey")
    wedding = _seg(scene_query="Example Kingdom Purple Wedding feast")
    anchor_eras = E.anchor_token_eras(lightweight)
    assert E.beat_era(confession, "", single_scene=False, global_verified=False,
                      anchor_eras=anchor_eras) == "season 7"
    assert E.beat_era(wedding, "", single_scene=False, global_verified=False,
                      anchor_eras=anchor_eras) == "season 4"

    # The publication verifier must carry the same roster into era inference; otherwise match and
    # publication can disagree and the latter reintroduces the false hard season conflict.
    project = ClipProject(name="era", root="/nonexistent", segments=[beat], meta={
        "analysis": {
            "movie_title": full.movie_title,
            "video_type": "multi_scene",
            "anchor_scenes": anchors,
            "characters": full.characters,
            "actors": full.actors,
        }})
    assert V._project_beat_era(project, beat) == ""


def test_same_era_repeated_scene_tokens_remain_eligible():
    analysis = NS(movie_title="Example Kingdom", anchor_scenes=[
        {"name": "royal wedding feast", "query": "royal wedding feast S04E02",
         "episode": "S04E02"},
        {"name": "royal wedding aftermath", "query": "royal wedding aftermath S04E03",
         "episode": "S04E03"},
    ])
    anchor_eras = E.anchor_token_eras(analysis)
    assert all("royal" in tokens and "wedding" in tokens for tokens, _era in anchor_eras)
    beat = _seg(scene_query="Example Kingdom royal wedding crowd")
    assert E.beat_era(beat, "", single_scene=False, global_verified=False,
                      anchor_eras=anchor_eras) == "season 4"
