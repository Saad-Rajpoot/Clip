"""Targeted regressions for the b79 viewer-eye relevance failures.

No network, LLM, ranking replay, or render. These fixtures exercise the policy and verifier
acceptance contracts that previously admitted raven→dragon and named-death contradictions.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import verify as V


def _seg(text: str, **kw):
    base = dict(text=text, quote="", visual_policy="", is_specific_claim=False,
                required_entity="", required_kind="", expected_visual="", scene_query="")
    base.update(kw)
    return NS(**base)


def test_concrete_animal_claim_cannot_remain_llm_filler():
    raven = _seg("A raven takes weeks.", visual_policy=P.FILLER, is_specific_claim=True,
                 required_entity="raven", required_kind="animal",
                 expected_visual="A raven in flight against a cloudy sky")
    assert P.policy_of(raven) == P.EXACT
    P.finalize_beats([raven])
    assert raven.visual_policy == P.EXACT
    assert P.policy_of(raven) == P.EXACT, "the guard must be stable after persistence"


def test_direct_visual_object_animal_or_place_overrides_character_label():
    cup = _seg("Almost nobody watched the cup.", visual_policy=P.CHARACTER,
               required_entity="chalice", required_kind="object")
    raven = _seg("You can see the raven on screen.", visual_policy=P.CHARACTER,
                 required_entity="raven", required_kind="animal")
    keep = _seg("Look at Winterfell's broken gate.", visual_policy=P.CHARACTER,
                required_entity="Winterfell", required_kind="place")
    for beat in (cup, raven, keep):
        assert P.policy_of(beat) == P.EXACT

    # General subject commentary stays CHARACTER; only a direct visual assertion is promoted.
    thematic = _seg("The chalice mattered to the conspiracy.", visual_policy=P.CHARACTER,
                    required_entity="chalice", required_kind="object")
    assert P.policy_of(thematic) == P.CHARACTER


def test_quote_cannot_remain_llm_filler_but_unanchored_commentary_can():
    quoted = _seg("Then he says it.", visual_policy=P.FILLER,
                  quote="Chaos is a ladder.")
    assert P.policy_of(quoted) == P.EXACT

    # is_specific_claim is noisy in persisted analyses. Without a quote/entity/kind this remains
    # filler, preventing a broad policy re-roll of abstract commentary.
    commentary = _seg("And that changes everything.", visual_policy=P.FILLER,
                      is_specific_claim=True)
    assert P.policy_of(commentary) == P.FILLER


_EXPLICIT_MISMATCH = {
    "verdict": "replace", "matches_narration": False, "specific_enough": False,
    "correct_subject_visible": True, "wrong_subject_visible": False,
    "quality_ok": True, "confidence": 0.8,
}
_POSITIVE_EXACT = {
    "verdict": "replace", "matches_narration": True, "specific_enough": True,
    "correct_subject_visible": True, "wrong_subject_visible": False,
    "quality_ok": True, "confidence": 0.8,
}


def test_exact_contextual_requires_positive_match_and_specificity():
    seg = _seg("Jaime rides Ned down in the street.", required_entity="Jaime Lannister",
               required_kind="character")
    # Preserve non-exact behavior: right-character contextual footage remains eligible there.
    assert V._contextual_subject_ok(_EXPLICIT_MISMATCH)
    # But an exact rejection cannot rewrite its own explicit mismatch+insufficiency to keep.
    assert not V._exact_contextual_ok(_EXPLICIT_MISMATCH, seg)
    assert V._exact_contextual_ok(_POSITIVE_EXACT, seg)
    assert not V._exact_contextual_ok(
        {**_POSITIVE_EXACT, "contradicts_narration": True}, seg)


def test_named_absence_showing_that_person_is_a_direct_contradiction():
    seg = _seg("and Baelish is not even in the room.",
               required_entity="Petyr Baelish", required_kind="character")
    why = V._direct_negative_contradiction(seg, {"correct_subject_visible": True})
    assert why and "absent" in why
    assert not V._exact_contextual_ok(_POSITIVE_EXACT, seg), why

    # Narrow scope: general figurative negation must not trigger an absence contradiction.
    control = _seg("Baelish is not in control.", required_entity="Petyr Baelish",
                   required_kind="character")
    assert V._direct_negative_contradiction(
        control, {"correct_subject_visible": True}) == ""


def test_source_title_is_negative_evidence_for_a_different_named_death_only():
    roster = {
        "joffrey baratheon": "Jack Gleeson",
        "petyr baelish": "Aidan Gillen",
        "catelyn stark": "Michelle Fairley",
    }
    jon = _seg("Jon Arryn the Hand of the King dies of a sudden illness",
               required_entity="Jon Arryn's body", required_kind="character",
               expected_visual="Jon Arryn's shrouded body lying in state")
    bad_title = "Game of Thrones - King Joffrey's Death (Poisoned at his wedding)"
    why = V._source_title_named_death_conflict(jon, bad_title, roster)
    assert why and "joffrey baratheon" in why.lower()
    assert not V._exact_contextual_ok(_POSITIVE_EXACT, jon, bad_title, roster)

    joffrey = _seg("Joffrey dies at his wedding", required_entity="Joffrey Baratheon",
                   required_kind="character")
    assert V._source_title_named_death_conflict(joffrey, bad_title, roster) == ""
    assert V._source_title_named_death_conflict(jon, bad_title, {}) == "", \
        "without roster context a title must not guess character identity"
    assert V._source_title_named_death_conflict(
        jon, "Joffrey reacts to Jon Arryn's death", roster) == "", \
        "a foreign name elsewhere in the title is not automatically the person who died"


def _verify_fixture(tmp_path, policy_name: str):
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo

    proj = ClipProject(name="contract", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"0" * 2048)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8\xff")
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones council scene",
                                permission="owner", status="ok", local_path=str(media))]
    seg = ScriptSegment(index=0, text="Jaime rides Ned down in the street.",
                        required_entity="Jaime Lannister", required_kind="character",
                        visual_policy=policy_name, is_specific_claim=True)
    shot = Shot(source_id="s1", index=0, start=0.0, end=2.0, keyframe_path=str(frame))
    proj.selections = [ClipSelection(segment_index=0, source_id="s1", shot_index=0,
                                     in_point=0.0, out_point=2.0, confidence=0.8)]
    proj.meta["analysis"] = {"video_type": "multi_scene", "characters": [], "actors": []}
    return proj, [seg], shot, ClipConfig()


def test_exact_repair_loop_does_not_rewrite_explicit_mismatch(tmp_path, monkeypatch):
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.EXACT)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    monkeypatch.setattr(V, "verify_frame", lambda *a, **k: dict(_EXPLICIT_MISMATCH))
    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)
    selection = proj.selections[0]
    assert summary["failed"] == 1
    assert selection.verifier["verdict"] == "replace"
    assert V.FLAG_EXACT_MISSING in selection.flag_reasons


def test_nonexact_contextual_leniency_is_preserved(tmp_path, monkeypatch):
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.CHARACTER)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    monkeypatch.setattr(V, "verify_frame", lambda *a, **k: dict(_EXPLICIT_MISMATCH))
    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)
    assert summary["failed"] == 0
    assert proj.selections[0].verifier["verdict"] == "keep"
    assert "non-exact" in proj.selections[0].verifier["relaxed"]


def test_verifier_judges_selected_trim_not_target_elsewhere_in_same_shot(tmp_path, monkeypatch):
    """A target at 15s in a 0-20s shot cannot approve the aired 1-3s trim."""
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.EXACT)
    shot.start, shot.end = 0.0, 20.0
    sel = proj.selections[0]
    sel.in_point, sel.out_point = 1.0, 3.0
    calls = []

    def sampled(src, start, end, dest):
        calls.append((float(start), float(end)))
        # Model the described target as visible only near 15s, outside the selected trim.
        Path(dest).write_bytes(b"target" if start <= 15.0 <= end else b"no-target")
        return dest

    def judge(frame, *_args, **_kwargs):
        if Path(frame).read_bytes() == b"target":
            return {**_POSITIVE_EXACT, "verdict": "keep", "contradicts_narration": False}
        return {**_EXPLICIT_MISMATCH, "correct_subject_visible": False,
                "contradicts_narration": False}

    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    monkeypatch.setattr(V, "_action_contact_sheet", sampled)
    monkeypatch.setattr(V, "verify_frame", judge)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "0")
    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)

    assert calls and set(calls) == {(1.0, 3.0)}
    assert summary["failed"] == 1 and sel.verifier["verdict"] == "replace"
    evidence = sel.verifier["selection_evidence"]
    assert evidence["selection_in"] == 1.0 and evidence["selection_out"] == 3.0
    assert evidence["multiframe"] is True and ":1.000-3.000" in evidence["image_id"]
