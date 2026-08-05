"""Targeted regressions for the b79 viewer-eye relevance failures.

No network, LLM, ranking replay, or render. These fixtures exercise the policy and verifier
acceptance contracts that previously admitted raven→dragon and named-death contradictions.
"""
from __future__ import annotations

import base64
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import verify as V
from vidlore.clipstudio import analyze as A
from vidlore.clipstudio import relevance_contract as R


def test_action_contact_sheet_preserves_three_moments_but_caps_provider_aspect(
        tmp_path, monkeypatch):
    """A 3x16:9 strip must remain multiframe without Gemini's measured 5.33:1 block."""
    from PIL import Image

    src = tmp_path / "source.mp4"
    src.write_bytes(b"fixture")
    dest = tmp_path / "sheet.jpg"
    colours = ((220, 20, 20), (20, 220, 20), (20, 20, 220))

    def fake_extract(argv, **_kwargs):
        frame_path = Path(argv[-1])
        frame_no = int(frame_path.stem.rsplit("_", 1)[-1])
        Image.new("RGB", (426, 240), colours[frame_no]).save(frame_path)
        return NS(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_extract)
    assert V._action_contact_sheet(str(src), 817.0, 820.0, dest) == dest

    with Image.open(dest).convert("RGB") as sheet:
        assert sheet.size == (1278, 320)
        assert sheet.width / sheet.height <= V._CONTACT_SHEET_MAX_ASPECT
        assert sheet.crop((0, 40, 426, 280)).size == (426, 240)
        assert sheet.crop((426, 40, 852, 280)).size == (426, 240)
        assert sheet.crop((852, 40, 1278, 280)).size == (426, 240)
        for neutral in (sheet.getpixel((639, 10)), sheet.getpixel((639, 310))):
            assert max(neutral) - min(neutral) <= 2
            assert all(10 <= channel <= 22 for channel in neutral)
        y = sheet.height // 2
        samples = [sheet.getpixel((213 + 426 * i, y)) for i in range(3)]
    assert samples[0][0] > 180 and samples[0][1] < 60 and samples[0][2] < 60
    assert samples[1][1] > 180 and samples[1][0] < 60 and samples[1][2] < 60
    assert samples[2][2] > 180 and samples[2][0] < 60 and samples[2][1] < 60


def test_contact_sheet_padding_does_not_change_single_frame_verifier_payload(
        tmp_path, monkeypatch):
    """The provider workaround is sheet-only; ordinary keyframe bytes stay byte-identical."""
    from PIL import Image
    from vidlore.clipstudio import llm

    frame = tmp_path / "keyframe.jpg"
    Image.new("RGB", (426, 240), (31, 73, 119)).save(frame, quality=91)
    original = frame.read_bytes()
    captured = {}

    def answer(**kwargs):
        captured.update(kwargs)
        return ('{"matches_narration":false,"correct_subject_visible":false,'
                '"wrong_subject_visible":false,"contradicts_narration":false,'
                '"specific_enough":false,"quality_ok":true,"confidence":0.5,'
                '"verdict":"replace","reason":"fixture"}', {"served": "fixture"})

    monkeypatch.setattr(llm, "complete_ex", answer)
    assert V.verify_frame(
        frame, "A generic line.", "", "", [], NS(), multiframe=False) is not None
    image_part = captured["messages"][0]["content"][0]
    assert base64.b64decode(image_part["source"]["data"]) == original
    assert frame.read_bytes() == original
    prompt = captured["messages"][0]["content"][1]["text"]
    assert "START -> MIDDLE -> END contact sheet" not in prompt


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


def test_character_general_directives_cannot_reintroduce_exact_storyboard_specificity():
    measured = (
        ("For seven seasons, the agreement about Petyr Baelish was almost total.",
         "Petyr Baelish smug in his brothel"),
        ("There is a second objection here, and it deserves a straight answer.",
         "Littlefinger alone in his brothel looking at camera"),
        ("Baelish alone treated the absence of checking as a permanent resource.",
         "Littlefinger at a window over King's Landing"),
    )
    for text, invented_storyboard in measured:
        beat = _seg(text, is_specific_claim=True)
        A._apply_beat_direction(beat, {
            "expected_visual": invented_storyboard,
            "scene_query": "Game of Thrones " + invented_storyboard,
            "required_entity": "Petyr Baelish", "required_kind": "character",
            "visual_policy": "character_specific", "specific": True,
        })
        assert beat.visual_policy == P.CHARACTER
        assert beat.is_specific_claim is False
        assert P.policy_of(beat) == P.CHARACTER


def test_unanchored_rewatch_is_abstract_but_a_narrated_target_remains_exact():
    unanchored = _seg(
        "Watch it again and something uncomfortable comes apart.",
        visual_policy=P.EXACT,
        expected_visual="invented rewind montage of four unrelated scenes")
    assert not P.is_deictic(unanchored)
    assert P.policy_of(unanchored) == P.ABSTRACT

    anchored = _seg("Watch it again sometime, and don't look at Joffrey at all.",
                    visual_policy=P.EXACT)
    assert P.is_deictic(anchored)
    assert P.policy_of(anchored) == P.EXACT

    normalized = _seg("Watch it again and something uncomfortable comes apart.")
    A._apply_beat_direction(normalized, {
        "expected_visual": "an editorial rewind effect", "scene_query": "invented scene",
        "required_entity": "invented subject", "required_kind": "scene",
        "visual_policy": "abstract_effect", "specific": True,
    })
    assert normalized.visual_policy == P.ABSTRACT
    assert normalized.is_specific_claim is False
    assert normalized.scene_query == normalized.required_entity == normalized.required_kind == ""


def test_character_general_verifier_omits_aspirational_storyboard_and_stays_positive(
        tmp_path, monkeypatch):
    from vidlore.clipstudio import llm as L
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    prompts = []

    def complete_ex(**kwargs):
        prompts.append(kwargs["messages"][0]["content"][1]["text"])
        return ('{"matches_narration":true,"correct_subject_visible":true,'
                '"wrong_subject_visible":false,"contradicts_narration":false,'
                '"specific_enough":true,"quality_ok":true,"confidence":0.9,'
                '"verdict":"keep","reason":"right subject"}', {"served": "test"})

    monkeypatch.setattr(L, "complete_ex", complete_ex)
    V.verify_frame(
        frame, "The agreement about Petyr Baelish was almost total.", "Petyr Baelish",
        "character", [], NS(), is_specific=False,
        expected_visual="Baelish smirking alone inside his brothel",
        scene_query="Game of Thrones Littlefinger brothel smirk")
    assert "Baelish smirking alone" not in prompts[0]
    assert "Target scene:" not in prompts[0]
    assert "specific_enough=true" in prompts[0]

    V.verify_frame(
        frame, "Baelish reveals the dagger in his brothel.", "Petyr Baelish",
        "character", [], NS(), is_specific=True,
        expected_visual="Baelish reveals the dagger in his brothel",
        scene_query="Game of Thrones Littlefinger dagger brothel")
    assert "Baelish reveals the dagger" in prompts[1]
    assert "Target scene:" in prompts[1]


def test_lenient_fingerprint_ignores_unused_storyboard_but_strict_does_not():
    base = dict(
        src_hash="a", source_id="s", shot_start=0.0, shot_end=2.0,
        beat_text="Petyr Baelish was underestimated", required_entity="Petyr Baelish",
        required_kind="character", expected_visual="invented room one",
        scene_query="invented query one", visual_policy=P.CHARACTER,
        faceid_names=[], multiframe=True, image_id="sheet:x", model="vision")
    lenient_a = V.verdict_fingerprint(**base, is_specific=False)
    lenient_b = V.verdict_fingerprint(
        **{**base, "expected_visual": "invented room two",
           "scene_query": "invented query two"}, is_specific=False)
    strict_a = V.verdict_fingerprint(**base, is_specific=True)
    strict_b = V.verdict_fingerprint(
        **{**base, "expected_visual": "invented room two",
           "scene_query": "invented query two"}, is_specific=True)
    assert lenient_a == lenient_b
    assert strict_a != strict_b
    assert V.PROMPT_VERSION == "v9-2026-08", "unchanged exact questions keep their warm cache"
    assert V.LENIENT_PROMPT_VERSION == "v10-2026-08"


def test_locative_absence_storyboard_is_not_a_subject_presence_contradiction(
        tmp_path, monkeypatch):
    """Showing Baelish elsewhere proves he was absent from the feast; it does not negate the VO."""
    from vidlore.clipstudio import llm as L

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    prompts = []

    def complete_ex(**kwargs):
        prompts.append(kwargs["messages"][0]["content"][1]["text"])
        return ('{"matches_narration":true,"correct_subject_visible":true,'
                '"wrong_subject_visible":false,"contradicts_narration":false,'
                '"specific_enough":true,"quality_ok":true,"confidence":0.9,'
                '"verdict":"keep","reason":"shown elsewhere"}', {"served": "test"})

    monkeypatch.setattr(L, "complete_ex", complete_ex)
    common = dict(
        narration="and Baelish is not even in the room.",
        required_entity="Petyr Baelish", required_kind="character", faceid_names=[],
        eng_cfg=NS(), is_specific=True,
        expected_visual="Littlefinger stands on a ship with Sansa, outside King's Landing.",
        scene_query="Game of Thrones Littlefinger Sansa escape ship")
    assert V.verify_frame(frame, **common)["verdict"] == "keep"
    assert "ABSENCE-ELSEWHERE CONTEXT" in prompts[0]
    assert "Seeing the subject at the storyboard's different location SUPPORTS" in prompts[0]

    base_fp = dict(
        src_hash="a", source_id="s", shot_start=0.0, shot_end=2.0,
        beat_text=common["narration"], required_entity=common["required_entity"],
        required_kind="character", expected_visual=common["expected_visual"],
        scene_query=common["scene_query"], visual_policy=P.EXACT, is_specific=True,
        faceid_names=[], multiframe=True, image_id="sheet:x", model="vision")
    special = V.verdict_fingerprint(**base_fp)
    ordinary = V.verdict_fingerprint(
        **{**base_fp, "beat_text": "Baelish is not the killer."})
    assert special != ordinary, "the conditional prompt clause must invalidate only its own cache"
    assert not V._absence_elsewhere_instruction(
        "Baelish is not the killer.", "Petyr Baelish",
        expected_visual=common["expected_visual"], scene_query=common["scene_query"])
    assert not V._absence_elsewhere_instruction(
        common["narration"], common["required_entity"],
        expected_visual="Baelish stands in the council room.",
        scene_query="Game of Thrones council room")
    assert not V._absence_elsewhere_instruction(
        "Baelish accused Varys, who was not in the room.", "Petyr Baelish",
        expected_visual=common["expected_visual"], scene_query=common["scene_query"])
    assert not V._absence_elsewhere_instruction(
        common["narration"], common["required_entity"],
        expected_visual="Baelish waits in the room aboard a ship.",
        scene_query="Game of Thrones council room escape ship")
    for excluded, storyboard in (
        ("harbor", "harbour"), ("boat", "ship"), ("castle", "keep"),
        ("room", "chamber"),
    ):
        assert not V._absence_elsewhere_instruction(
            f"Baelish is not in the {excluded}.", common["required_entity"],
            expected_visual=f"Baelish waits in the {storyboard}.",
            scene_query=f"Game of Thrones Baelish {storyboard}")

    seg = NS(text=common["narration"], required_entity=common["required_entity"],
             required_kind="character", expected_visual=common["expected_visual"],
             scene_query=common["scene_query"])
    assert V._direct_negative_contradiction(seg, {
        "correct_subject_visible": True, "contradicts_narration": False}) == ""
    seg.expected_visual = "Baelish stands in the council room."
    seg.scene_query = "Game of Thrones council room"
    assert "absent" in V._direct_negative_contradiction(seg, {
        "correct_subject_visible": True, "contradicts_narration": False})


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


def test_exact_storyboard_co_character_conflicts_with_different_title_cast():
    roster = {
        "petyr baelish": "Aidan Gillen",
        "catelyn stark": "Michelle Fairley",
    }
    beat = _seg(
        "Now look at the shape of the lie he tells her.",
        visual_policy=P.EXACT, is_specific_claim=True,
        required_entity="Petyr Baelish", required_kind="character",
        expected_visual="Littlefinger faces Catelyn before telling the dagger lie",
        scene_query="Game of Thrones Littlefinger brothel Catelyn dagger lie",
    )
    wrong = '"Why do they call you Littlefinger?" Game of Thrones S01E04 Arya Stark'
    why = V._source_title_exact_cast_conflict(beat, wrong, roster)
    assert why and "catelyn stark" in why and "arya stark" in why
    assert V._source_title_exact_cast_conflict(
        beat, "Catelyn confronts Littlefinger about the dagger", roster) == ""
    assert V._source_title_exact_cast_conflict(
        beat, "Sansa reacts while Catelyn confronts Littlefinger", roster) == ""
    assert V._source_title_exact_cast_conflict(
        beat, "Littlefinger lies to Lady Stark about the dagger", roster) == ""
    assert V._source_title_exact_cast_conflict(
        beat, "How Littlefinger betrayed House Stark", roster) == ""
    assert V._source_title_exact_cast_conflict(
        beat, "The dagger that started The Stark-Lannister war", roster) == ""

    contextual = _seg(**beat.__dict__)
    contextual.visual_policy = P.CHARACTER
    contextual.is_specific_claim = False
    assert V._source_title_exact_cast_conflict(contextual, wrong, roster) == ""


def test_cast_warning_resolution_must_name_expected_pixel_cast_not_only_set_boolean():
    roster = {
        "joffrey baratheon": "Jack Gleeson",
        "catelyn stark": "Michelle Fairley",
    }
    beat = _seg(
        "the weapon they left behind is the only thing she has.",
        visual_policy=P.EXACT, is_specific_claim=True,
        required_entity="Valyrian steel dagger", required_kind="object",
        expected_visual=("Exact scene: Game of Thrones catspaw Valyrian steel dagger left "
                         "behind. Also show: Catelyn."),
        scene_query="Game of Thrones catspaw Valyrian steel dagger left behind",
    )
    wrong_title = "All scenes of Joffrey Baratheon"
    base = {
        **_POSITIVE_EXACT,
        "verdict": "keep",
        "source_title_conflict_resolved": True,
        "reason": "The frame clearly shows the Valyrian steel dagger.",
    }

    rejected = V._strict_keep_rejection_reason(base, beat, wrong_title, roster)
    accepted = V._strict_keep_rejection_reason(
        {**base, "reason": "The selected pixels show Catelyn Stark beside the dagger."},
        beat, wrong_title, roster)

    assert "did not identify any expected co-character" in rejected
    assert accepted == ""

    negative = V._strict_keep_rejection_reason(
        {**base, "reason": "The dagger is visible, but Catelyn Stark is not present."},
        beat, wrong_title, roster)
    bare_name = V._strict_keep_rejection_reason(
        {**base, "reason": "The selected scene concerns Catelyn Stark."},
        beat, wrong_title, roster)
    faceid = V._strict_keep_rejection_reason(
        {**base, "reason": "The dagger is visible.", "selection_evidence": {
            "faceid_names": ["Catelyn Stark"]}}, beat, wrong_title, roster)

    assert "affirmative pixel evidence" in negative
    assert "affirmative pixel evidence" in bare_name
    assert faceid == ""

    for uncertain_reason in (
            "The pixels cannot confirm Catelyn Stark.",
            "It is unclear whether Catelyn Stark is visible.",
            "Catelyn Stark may not be present."):
        assert "affirmative pixel evidence" in V._strict_keep_rejection_reason(
            {**base, "reason": uncertain_reason}, beat, wrong_title, roster)


def test_cast_warning_honorific_surname_is_not_proof_for_an_ambiguous_family():
    roster = {
        "catelyn stark": "Michelle Fairley",
        "arya stark": "Maisie Williams",
        "sansa stark": "Sophie Turner",
    }
    beat = _seg(
        "Look at the dagger scene.", visual_policy=P.EXACT, is_specific_claim=True,
        required_entity="Valyrian steel dagger", required_kind="object",
        expected_visual="Catelyn Stark examines the dagger",
        scene_query="Catelyn Stark dagger scene")
    verdict = {
        **_POSITIVE_EXACT,
        "verdict": "keep",
        "source_title_conflict_resolved": True,
        "reason": "The selected pixels clearly show Lady Stark beside the dagger.",
    }

    assert "affirmative pixel evidence" in V._cast_warning_resolution_reason(
        verdict, beat, roster)


def test_project_roster_keeps_name_only_rows_without_none_actor_alias():
    proj = type("P", (), {"meta": {"analysis": {"characters": [
        {"name": "Catelyn Stark", "actor": None},
        {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
    ]}}})()
    assert V._project_char2actor(proj) == {
        "catelyn stark": "", "petyr baelish": "Aidan Gillen"}


def test_exact_title_cast_warning_is_resolved_from_pixels_and_fingerprinted(
        tmp_path, monkeypatch):
    from vidlore.clipstudio import llm as L

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    prompts = []

    def complete_ex(**kwargs):
        prompts.append(kwargs["messages"][0]["content"][1]["text"])
        return ('{"matches_narration":true,"correct_subject_visible":true,'
                '"wrong_subject_visible":false,"contradicts_narration":false,'
                '"source_title_conflict_resolved":true,"specific_enough":true,'
                '"quality_ok":true,"confidence":0.9,"verdict":"keep",'
                '"reason":"pixels prove the target scene"}', {"served": "test"})

    monkeypatch.setattr(L, "complete_ex", complete_ex)
    warning = "storyboard names Catelyn Stark but title names Arya Stark"
    verdict = V.verify_frame(
        frame, "Look at the lie.", "Petyr Baelish", "character", [], NS(),
        is_specific=True, expected_visual="Petyr faces Catelyn in the brothel",
        scene_query="Littlefinger Catelyn dagger lie", exact_cast_warning=warning)
    assert verdict["source_title_conflict_resolved"] is True
    assert "title describes the whole upload" in prompts[0].lower()
    assert "ACTUAL PIXELS" in prompts[0]

    base = dict(
        src_hash="a", source_id="s", shot_start=0.0, shot_end=2.0,
        beat_text="Look at the lie.", required_entity="Petyr Baelish",
        required_kind="character", expected_visual="Petyr faces Catelyn",
        scene_query="Littlefinger Catelyn dagger lie", visual_policy=P.EXACT,
        faceid_names=[], multiframe=True, image_id="sheet:x", model="vision")
    assert V.verdict_fingerprint(**base, is_specific=True) != V.verdict_fingerprint(
        **base, is_specific=True, exact_cast_warning=warning)
    assert V.verdict_fingerprint(**base, is_specific=False) == V.verdict_fingerprint(
        **base, is_specific=False, exact_cast_warning=warning)


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


def test_exact_keep_that_denies_required_subject_enters_repair_instead_of_stalling_gate(
        tmp_path, monkeypatch):
    """A self-contradictory vision reply is not a successful strict selection.

    This exact shape aired in the 101-beat reproduction: Gemini said ``keep`` for a blade shot
    while also saying Petyr Baelish was not visible.  Accepting the headline verdict stopped the
    repair ladder, then the publication gate rejected the same explicit negative fact.
    """
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.EXACT)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    contradictory_keep = {
        "verdict": "keep", "matches_narration": True, "specific_enough": True,
        "correct_subject_visible": False, "wrong_subject_visible": False,
        "contradicts_narration": False, "quality_ok": True, "era_ok": True,
        "confidence": 0.8,
    }
    monkeypatch.setattr(V, "verify_frame", lambda *a, **k: dict(contradictory_keep))

    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)

    selection = proj.selections[0]
    assert summary["failed"] == 1
    assert selection.verifier["verdict"] == "replace"
    assert "required subject" in selection.verifier["contract_rejected"]
    assert V.FLAG_EXACT_MISSING in selection.flag_reasons


def test_fresh_strict_answer_drops_stale_selection_transition_state(tmp_path, monkeypatch):
    """A scoped strict retry starts from its current vision answer, not old downgrade labels."""
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.EXACT)
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    polluted_keep = {
        **_POSITIVE_EXACT,
        "verdict": "keep",
        "contradicts_narration": False,
        "downgraded": "exact→contextual",
        "relaxed": "stale non-exact decision",
        "relevance_class": "contextual_fallback",
        "contract_rejected": "stale contract rejection",
    }
    monkeypatch.setattr(V, "verify_frame", lambda *a, **k: dict(polluted_keep))

    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)

    verifier = proj.selections[0].verifier
    assert summary["failed"] == 0
    assert not (set(V._VERDICT_TRANSITION_FIELDS) & set(verifier))
    assert verifier["selection_evidence"]["is_specific"] is True


def test_warned_keep_missing_resolution_is_technical_not_content_softening(
        tmp_path, monkeypatch):
    proj, segs, shot, cfg = _verify_fixture(tmp_path, P.EXACT)
    seg = segs[0]
    seg.text = "Now look at the shape of the lie he tells her."
    seg.required_entity = "Petyr Baelish"
    seg.expected_visual = "Littlefinger faces Catelyn before telling the dagger lie"
    seg.scene_query = "Game of Thrones Littlefinger brothel Catelyn dagger lie"
    proj.sources[0].title = '"Why Littlefinger?" Arya Stark'
    proj.meta["analysis"]["characters"] = [
        {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
        {"name": "Catelyn Stark", "actor": "Michelle Fairley"},
    ]
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shot)
    malformed_keep = {
        "verdict": "keep", "matches_narration": True, "specific_enough": True,
        "correct_subject_visible": True, "wrong_subject_visible": False,
        "contradicts_narration": False, "quality_ok": True, "confidence": 0.8,
        # source_title_conflict_resolved is intentionally missing.
    }
    monkeypatch.setattr(V, "verify_frame", lambda *a, **k: dict(malformed_keep))

    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)

    selection = proj.selections[0]
    assert summary["failed"] == 1
    assert selection.verifier["status"] == "error"
    assert "field is missing or malformed" in selection.verifier["reason"]
    assert "verdict" not in selection.verifier
    assert V.FLAG_VERIFIER_UNVERIFIED in selection.flag_reasons


def test_strict_promotion_refuses_keep_that_denies_required_subject(tmp_path, monkeypatch):
    """The same contract applies to alternates; beat 7's bad window entered via promotion."""
    from vidlore.clipstudio.models import ClipCandidate, Shot

    proj, segs, primary, cfg = _verify_fixture(tmp_path, P.EXACT)
    alt_frame = tmp_path / "alt.jpg"
    alt_frame.write_bytes(b"\xff\xd8\xffalt")
    alternate = Shot(source_id="s1", index=1, start=3.0, end=5.0,
                     keyframe_path=str(alt_frame))
    proj.selections[0].alternates = [ClipCandidate(
        segment_index=0, source_id="s1", shot_index=1, score=0.9,
        in_point=3.0, out_point=5.0)]
    shots = {("s1", 0): primary, ("s1", 1): alternate}
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_EXACT_CONTEXTUAL_DOWNGRADE", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_GENERIC_FILLER_DOWNGRADE", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shots.get((sid, ix)))

    def judge(frame, *_a, **_k):
        if str(frame) == str(primary.keyframe_path):
            return {**_EXPLICIT_MISMATCH, "contradicts_narration": False}
        return {
            "verdict": "keep", "matches_narration": True, "specific_enough": True,
            "correct_subject_visible": False, "wrong_subject_visible": False,
            "contradicts_narration": False, "quality_ok": True, "era_ok": True,
            "confidence": 0.8,
        }

    monkeypatch.setattr(V, "verify_frame", judge)
    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None)

    assert summary["failed"] == 1
    assert proj.selections[0].shot_index == 0, "internally contradictory alternate must not promote"
    assert proj.selections[0].verifier["verdict"] == "replace"


def test_contradictory_primary_keep_still_prefetches_strict_repair(
        tmp_path, monkeypatch):
    """The new keep contract must feed phase-2 scheduling as well as the serial decision."""
    from vidlore.clipstudio.models import ClipCandidate, Shot

    proj, segs, primary, cfg = _verify_fixture(tmp_path, P.EXACT)
    alt_frame = tmp_path / "alt-prefetch.jpg"
    alt_frame.write_bytes(b"\xff\xd8\xffalt")
    alternate = Shot(source_id="s1", index=1, start=3.0, end=5.0,
                     keyframe_path=str(alt_frame))
    proj.selections[0].alternates = [ClipCandidate(
        segment_index=0, source_id="s1", shot_index=1, score=0.9,
        in_point=3.0, out_point=5.0)]
    shots = {("s1", 0): primary, ("s1", 1): alternate}
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "2")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_STRICT_NEIGHBORHOOD", "0")
    monkeypatch.setattr(V, "_shot_lookup", lambda _p: lambda sid, ix: shots.get((sid, ix)))
    calls = []

    def judge(frame, *_a, **_k):
        calls.append((str(frame), threading.current_thread() is threading.main_thread()))
        visible = str(frame) == str(alternate.keyframe_path)
        return {
            "verdict": "keep", "matches_narration": True, "specific_enough": True,
            "correct_subject_visible": visible, "wrong_subject_visible": False,
            "contradicts_narration": False, "quality_ok": True, "era_ok": True,
            "confidence": 0.8,
        }

    monkeypatch.setattr(V, "verify_frame", judge)
    summary = V.verify_and_repair(
        proj, segs, cfg, NS(anthropic_model="m", anthropic_key="k"), progress=None,
        materialize_promotions=False, persist_project=False)

    alt_calls = [on_main for frame, on_main in calls if frame == str(alternate.keyframe_path)]
    assert alt_calls == [False], "strict repair must be warmed in the worker pool, not paid serially"
    assert summary["replaced"] == 1 and proj.selections[0].shot_index == 1


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


@pytest.mark.parametrize("binding_reason", [
    "verifier_evidence_mismatch",
    "verifier_evidence_absent",
    "verifier_evidence_schema_mismatch",
    "verifier_evidence_model_mismatch",
    "verifier_evidence_window_not_sampled",
])
def test_stale_lenient_negative_is_a_technical_reverify_not_a_content_mismatch(
        tmp_path, monkeypatch, binding_reason):
    proj, segs, _shot, _cfg = _verify_fixture(tmp_path, P.CHARACTER)
    sel = proj.selections[0]
    sel.verifier = {
        "status": "ok", "verdict": "keep", "matches_narration": False,
        "specific_enough": False, "correct_subject_visible": True,
        "wrong_subject_visible": False, "quality_ok": True,
    }
    monkeypatch.setattr(
        R, "selection_verifier_evidence_reason",
        lambda *_a, **_k: binding_reason)
    stale = R.evaluate_selection_relevance(proj, segs)
    assert stale["status"] == "blocked"
    assert stale["blockers"][0]["reasons"] == [binding_reason]

    # Once evidence is current, the same explicit negative facts remain hard blockers. The change
    # routes a changed question to re-verification; it does not accept a mismatch or lower a floor.
    monkeypatch.setattr(R, "selection_verifier_evidence_reason", lambda *_a, **_k: "")
    current = R.evaluate_selection_relevance(proj, segs)
    assert "matches_narration_false" in current["blockers"][0]["reasons"]
    assert "specific_enough_false" in current["blockers"][0]["reasons"]


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
