"""Regressions for the bounded timed-text passage inside strict neighborhood repair."""
from __future__ import annotations

from pathlib import Path

from vidlore.clipstudio import image_fallback as IF
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.models import (ClipCandidate, ClipProject, ClipSelection, ScriptSegment,
                                       Shot, SourceVideo)


def _candidate(sid: str, shot_index: int, *, score: float = 0.5, signals=None):
    return ClipCandidate(segment_index=0, source_id=sid, shot_index=shot_index,
                         in_point=float(shot_index), out_point=float(shot_index) + 2.5,
                         score=score, signals=dict(signals or {}))


def _source(tmp_path: Path, sid: str, title: str, query: str):
    media = tmp_path / f"{sid}.mp4"
    media.write_bytes(b"media")
    return SourceVideo(id=sid, title=title, url=f"u:{sid}", local_path=str(media),
                       status="ok", permission="owner", extra={"query": query})


def _shots(tmp_path: Path, sid: str, count: int = 28):
    rows = []
    for index in range(count):
        keyframe = tmp_path / f"{sid}_{index:02d}.jpg"
        keyframe.write_bytes(b"jpg" + bytes([index]))
        rows.append(Shot(source_id=sid, index=index, start=float(index) * 3.0,
                         end=float(index) * 3.0 + 2.8, keyframe_path=str(keyframe),
                         luma_avg=40.0, subs_flag=0, graphics_flag=0, static_frac=0.0,
                         pair_diff_max=5.0, pair_diff_mean=5.0))
    return rows


def _lookup(rows_by_source):
    def get_shot(sid, index):
        return next((shot for shot in rows_by_source.get(sid, [])
                     if shot.index == index), None)

    get_shot.all_shots = lambda sid: rows_by_source.get(sid, [])
    return get_shot


def _dontos_fixture(tmp_path: Path):
    proj = ClipProject(name="timed-text", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}
    proj.sources = [
        _source(tmp_path, "wrong", "Sansa attends the royal wedding",
                "Sansa Stark wedding scene"),
        _source(tmp_path, "exact", "Dontos gives Sansa the necklace",
                "Dontos Sansa poisoned necklace godswood"),
    ]
    shots = {sid: _shots(tmp_path, sid) for sid in ("wrong", "exact")}
    # The proper name is split exactly as Whisper split the measured production source.
    shots["exact"][20].transcript = "The necklace did not belong to his mother."
    shots["exact"][22].transcript = "Ser don toes is following someone else's plan."
    seg = ScriptSegment(
        index=0, text="He attached himself to the exit.",
        expected_visual=("Dontos Hollard hands Sansa Stark the poisoned amethyst necklace "
                         "in the godswood."),
        required_entity="necklace", required_kind="object",
        scene_query="Game of Thrones Dontos gives Sansa poisoned necklace godswood",
        visual_policy="exact_scene", is_specific_claim=True, shot_intent="action",
        est_duration=5.0)
    # With source_cap=1 the old path keeps only the head source.  The exact source is nevertheless
    # already retained on the deep bench and may replace, never enlarge, that one source slot.
    sel = ClipSelection(segment_index=0, source_id="wrong", shot_index=2,
                        in_point=6.0, out_point=8.5, confidence=0.8,
                        alternates=[_candidate("wrong", 1)],
                        deep_alternates=[_candidate("exact", 2)])
    proj.selections = [sel]
    return proj, seg, sel, shots, _lookup(shots)


def test_distant_compound_asr_passage_enters_unchanged_twelve_call_cap(
        tmp_path, monkeypatch):
    proj, seg, sel, _shots_by_source, get_shot = _dontos_fixture(tmp_path)
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12, source_cap=1)
    timed = [candidate for candidate in out
             if candidate.signals.get("strict_scene_timed_text_region")]

    assert len(out) == 12, "the transcript reserve must reuse the existing call budget"
    assert {candidate.source_id for candidate in out} == {"exact"}, \
        "strong text replaces the last source slot; it never increases source_cap"
    assert {candidate.shot_index for candidate in timed} == {19, 20, 21, 22, 23}
    assert {"dontos", "necklace"} <= set(timed[0].signals["timed_text_matches"])


def test_generic_talk_does_not_open_a_distant_region(tmp_path, monkeypatch):
    proj = ClipProject(name="generic", root=str(tmp_path))
    proj.ensure_dirs()
    proj.sources = [_source(tmp_path, "only", "A person talks in a room",
                            "person talks room")]
    shots = {"only": _shots(tmp_path, "only")}
    shots["only"][20].transcript = "A person talks in the room and looks outside."
    get_shot = _lookup(shots)
    seg = ScriptSegment(index=0, text="They talk.",
                        expected_visual="A person talks and looks around a room.",
                        required_entity="person", scene_query="person talks in room",
                        visual_policy="exact_scene", is_specific_claim=True, est_duration=3.0)
    sel = ClipSelection(segment_index=0, source_id="only", shot_index=2,
                        in_point=6.0, out_point=8.5, confidence=0.8)
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig())

    assert not any(candidate.signals.get("strict_scene_timed_text_region") for candidate in out)
    assert all(candidate.shot_index <= 8 for candidate in out), \
        "generic prose must leave the original +/-6 neighborhood unchanged"


def test_timed_reserve_still_excludes_match_gated_pixels(tmp_path, monkeypatch):
    proj, seg, sel, shots, get_shot = _dontos_fixture(tmp_path)
    shots["exact"][21].subs_flag = 1
    shots["exact"][21].quality = 0.99
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.99)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12, source_cap=1)
    keys = {(candidate.source_id, candidate.shot_index) for candidate in out}

    assert ("exact", 21) not in keys, "burned-subtitle shots remain hard excluded"
    assert ("exact", 20) in keys and ("exact", 22) in keys
    assert len(out) <= 12


def test_old_deep_reserve_is_preserved_without_stronger_timed_text(tmp_path, monkeypatch):
    proj = ClipProject(name="old-deep", root=str(tmp_path))
    proj.ensure_dirs()
    proj.sources = [_source(tmp_path, "only", "Littlefinger trial in the Great Hall",
                            "Sansa accuses Littlefinger trial Great Hall")]
    shots = {"only": _shots(tmp_path, "only")}
    get_shot = _lookup(shots)
    seg = ScriptSegment(index=0, text="The verdict arrives.",
                        expected_visual="Littlefinger kneels during Sansa's Great Hall trial.",
                        required_entity="Littlefinger",
                        scene_query="Sansa accuses Littlefinger trial Great Hall",
                        visual_policy="exact_scene", is_specific_claim=True, est_duration=3.0)
    sel = ClipSelection(
        segment_index=0, source_id="only", shot_index=2, in_point=6.0, out_point=8.5,
        confidence=0.8,
        deep_alternates=[_candidate("only", 20, score=0.8, signals={"clip": 0.88})])
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12)
    deep = [candidate for candidate in out
            if candidate.signals.get("strict_scene_deep_region")]

    assert [candidate.shot_index for candidate in deep] == [20, 19, 21, 18, 22]
    assert not any(candidate.signals.get("strict_scene_timed_text_region") for candidate in out)


def test_one_asr_word_cannot_count_as_two_morphology_anchors(tmp_path, monkeypatch):
    proj = ClipProject(name="morph-alias", root=str(tmp_path))
    proj.ensure_dirs()
    proj.sources = [_source(tmp_path, "only", "Knight betrayed the king",
                            "knight betray king")]
    shots = {"only": _shots(tmp_path, "only", count=30)}
    shots["only"][12].transcript = "He betrayed them."
    get_shot = _lookup(shots)
    seg = ScriptSegment(index=0, text="The betrayal begins.",
                        expected_visual="The knight betrayed the king in the courtyard.",
                        required_entity="king", required_kind="event",
                        scene_query="knight betray king courtyard",
                        visual_policy="exact_scene", is_specific_claim=True, est_duration=3.0)
    sel = ClipSelection(
        segment_index=0, source_id="only", shot_index=2, in_point=6.0, out_point=8.5,
        confidence=0.8,
        deep_alternates=[_candidate("only", 20, score=0.8, signals={"clip": 0.88})])
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12)

    assert not any(candidate.signals.get("strict_scene_timed_text_region") for candidate in out)
    assert any(candidate.signals.get("strict_scene_deep_region") for candidate in out), \
        "one word matching betray/betrayed must not displace the old deep reserve"


def test_retained_reserve_appends_when_source_cap_has_spare_room(tmp_path, monkeypatch):
    proj = ClipProject(name="sparse-source-cap", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}
    proj.sources = [
        _source(tmp_path, "original", "Ned Stark betrayed in the throne room",
                "Ned Stark betrayed closeup"),
        _source(tmp_path, "exact", "Littlefinger Against The Three-Eyed Raven",
                "Littlefinger trial scene"),
    ]
    shots = {sid: _shots(tmp_path, sid, count=14) for sid in ("original", "exact")}
    get_shot = _lookup(shots)
    seg = ScriptSegment(
        index=0, text="There is a real tragedy inside that scene.",
        expected_visual="Ned Stark's painful expression as Littlefinger's knife betrays him.",
        required_entity="Ned Stark", required_kind="character",
        scene_query="Game of Thrones Ned Stark betrayed closeup",
        visual_policy="exact_scene", is_specific_claim=True,
        shot_intent="emotional_closeup", est_duration=5.77)
    exact = _candidate("exact", 9, score=0.74, signals={"clip": 0.93})
    sel = ClipSelection(segment_index=0, source_id="original", shot_index=2,
                        in_point=6.0, out_point=8.5, confidence=0.8,
                        alternates=[exact])
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12, source_cap=4)
    keys = {(candidate.source_id, candidate.shot_index) for candidate in out}

    assert ("exact", 9) in keys
    assert any(candidate.source_id == "original" for candidate in out), \
        "spare source capacity must preserve the original neighborhood"
    assert len({candidate.source_id for candidate in out}) == 2
    assert len(out) <= 12


def test_timed_action_shot_after_dialogue_keeps_its_resolving_tail(tmp_path, monkeypatch):
    proj = ClipProject(name="action-tail", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}
    proj.sources = [_source(tmp_path, "throne", "Ned betrayed in the throne room",
                            "gold cloaks betray Ned Stark throne room")]
    shots = {"throne": _shots(tmp_path, "throne", count=66)}
    shots["throne"][61].start = 199.367
    shots["throne"][61].end = 204.1
    shots["throne"][61].transcript = "Tell your men to lay down their swords."
    shots["throne"][62].start = 204.1
    shots["throne"][62].end = 205.4
    shots["throne"][62].transcript = ""
    shots["throne"][63].start = 205.4
    shots["throne"][63].end = 216.667
    shots["throne"][63].transcript = ""
    shots["throne"][64].start = 216.667
    shots["throne"][64].end = 221.133
    shots["throne"][64].transcript = "I did warn you not to trust me."
    get_shot = _lookup(shots)
    seg = ScriptSegment(
        index=0, text="The gold cloaks he relies on are bought.",
        expected_visual=("The gold cloaks draw their swords on Ned Stark in the throne room, "
                         "betraying his trust."),
        required_entity="gold cloaks, Ned Stark", required_kind="event",
        scene_query="Game of Thrones gold cloaks betray Ned Stark throne room",
        visual_policy="exact_scene", is_specific_claim=True, shot_intent="action",
        est_duration=6.92)
    sel = ClipSelection(
        segment_index=0, source_id="throne", shot_index=17,
        in_point=51.0, out_point=53.5, confidence=0.8,
        deep_alternates=[_candidate(
            "throne", 61, score=0.75, signals={"clip": 0.88, "transcript": 0.22})])
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), cap=12)
    action = next(candidate for candidate in out if candidate.shot_index == 63)

    assert action.signals.get("strict_scene_timed_text_region") is True
    assert action.signals.get("timed_text_tail_window") is True
    assert action.out_point == 216.667
    assert action.in_point > 208.0, \
        "the overlapping deep/timed reserve must retain the resolving tail, not its midpoint"


def test_unseen_fifth_normal_alternate_gets_one_slot_inside_same_cap(tmp_path, monkeypatch):
    proj = ClipProject(name="fifth-alternate", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"movie_title": "Game of Thrones"}
    source_ids = ["primary", "tried1", "tried2", "tried3", "tried4", "exact", "later"]
    proj.sources = [
        _source(tmp_path, sid,
                ("Littlefinger Against The Three-Eyed Raven" if sid == "exact"
                 else f"Ned Stark throne room context {sid}"),
                ("Littlefinger trial scene" if sid == "exact"
                 else "Ned Stark betrayed throne room"))
        for sid in source_ids
    ]
    shots = {sid: _shots(tmp_path, sid, count=14) for sid in source_ids}
    get_shot = _lookup(shots)
    seg = ScriptSegment(
        index=0, text="There is a real tragedy inside that scene.",
        expected_visual="Ned Stark's painful expression as Littlefinger's knife betrays him.",
        required_entity="Ned Stark", required_kind="character",
        scene_query="Game of Thrones Ned Stark betrayed closeup",
        visual_policy="exact_scene", is_specific_claim=True,
        shot_intent="emotional_closeup", est_duration=5.77)
    tried = [_candidate(f"tried{number}", 3, score=0.90 - number / 100.0)
             for number in range(1, 5)]
    exact = _candidate("exact", 9, score=0.74, signals={"clip": 0.93})
    later = _candidate("later", 4, score=0.60)
    sel = ClipSelection(segment_index=0, source_id="primary", shot_index=2,
                        in_point=6.0, out_point=8.5, confidence=0.8,
                        alternates=tried + [exact, later])
    excluded = {("primary", 2)} | {(candidate.source_id, candidate.shot_index)
                                   for candidate in tried}
    monkeypatch.setattr(IF, "_shot_relevance", lambda *_a, **_k: 0.5)

    out = V._strict_scene_neighborhood_candidates(
        sel, seg, proj, get_shot, ClipConfig(), exclude=excluded,
        cap=12, source_cap=2)
    exact_out = next(candidate for candidate in out
                     if (candidate.source_id, candidate.shot_index) == ("exact", 9))

    assert exact_out.signals.get("strict_scene_retained_alternate") is True
    assert len(out) <= 12, "the fifth alternate consumes, never adds, a strict call slot"
