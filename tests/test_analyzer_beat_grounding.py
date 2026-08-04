"""Beat-local analyzer grounding regressions from the 101-beat scale render.

These tests stop the per-beat analyzer from borrowing a vivid exact scene from whole-story context
for generic narration.  They deliberately do not solve cross-source quote timing: an exact event
that the narration really names stays exact and must be proved (or blocked) downstream.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio import analyze as A
from vidlore.clipstudio import policy as P
from vidlore.clipstudio.models import ScriptSegment


def _apply(text: str, **directive) -> ScriptSegment:
    beat = ScriptSegment(index=0, text=text, expected_visual=text)
    payload = {
        "expected_visual": "",
        "scene_query": "",
        "quote": "",
        "required_entity": "",
        "required_kind": "",
        "visual_policy": P.EXACT,
        "specific": True,
    }
    payload.update(directive)
    A._apply_beat_direction(beat, payload)
    return beat


@pytest.mark.parametrize(
    "text, directive",
    [
        (
            "and those people either died or had no reason to be believed.",
            {
                "expected_visual": "Ser Dontos Hollard shot dead by arrows on Littlefinger's order",
                "scene_query": "Game of Thrones Littlefinger orders Dontos killed",
                "quote": "He's seen too much.",
                "required_entity": "Ser Dontos",
                "required_kind": "character",
            },
        ),
        (
            "Verification is in motion. Someone is checking.",
            {
                "expected_visual": "Varys listens to a child spy whisper in his ear",
                "scene_query": "Game of Thrones Varys little bird spy whispers information",
                "required_entity": "Varys",
                "required_kind": "character",
            },
        ),
        (
            "That is not a rarer mind.",
            {
                "expected_visual": "Littlefinger's terrified tearful eyes during his trial",
                "scene_query": "Game of Thrones Littlefinger terrified eyes close-up trial",
                "required_entity": "Petyr Baelish",
                "required_kind": "character",
            },
        ),
        (
            "It is a rarer willingness.",
            {
                "expected_visual": "Littlefinger pushes Lysa Arryn through the Moon Door",
                "scene_query": "Game of Thrones Littlefinger pushes Lysa Arryn Moon Door",
                "quote": "I did warn you not to trust me.",
                "required_entity": "Petyr Baelish",
                "required_kind": "character",
            },
        ),
    ],
)
def test_zero_overlap_global_storyboards_downgrade_to_generic_and_clear_exact_fields(
        text, directive):
    beat = _apply(text, **directive)

    assert beat.visual_policy == P.FILLER
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    assert beat.required_entity == beat.required_kind == ""
    assert beat._analyzer_grounding_guard["branch"] == "ungrounded_exact_downgrade"
    assert beat._analyzer_grounding_guard["to_policy"] == P.FILLER


@pytest.mark.parametrize(
    "text, directive",
    [
        (
            "to the master-at-arms of the Red Keep.",
            {
                "expected_visual": "The master-at-arms examining the dagger",
                "scene_query": "Game of Thrones master-at-arms Aron Santagar dagger",
                "quote": "It's Valyrian steel.",
                "required_entity": "Aron Santagar",
                "required_kind": "character",
            },
        ),
        (
            "He never had to persuade Olenna Tyrell of anything.",
            {
                "expected_visual": "Olenna holds a poison cup and confesses to Jaime",
                "scene_query": "Game of Thrones Olenna confession poison cup Highgarden",
                "quote": "Tell Cersei. I want her to know it was me.",
                "required_entity": "Olenna Tyrell",
                "required_kind": "character",
            },
        ),
    ],
)
def test_named_subject_without_a_narrated_exact_moment_becomes_character_specific(
        text, directive):
    beat = _apply(text, **directive)

    assert beat.visual_policy == P.CHARACTER
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    # The narration names the role/person, so subject correctness remains required.
    assert beat.required_entity == directive["required_entity"]
    assert beat.required_kind == directive["required_kind"]
    assert beat._analyzer_grounding_guard["branch"] == "ungrounded_exact_downgrade"
    assert beat._analyzer_grounding_guard["to_policy"] == P.CHARACTER


@pytest.mark.parametrize(
    "text, directive",
    [
        (
            "That question is exactly how Varys learns she is in the city.",
            {
                "expected_visual": "A child whispers to Varys and his eyes widen",
                "scene_query": "Game of Thrones Varys learns Catelyn is in the city",
                "quote": "Lady Stark is here, in King's Landing.",
                "required_entity": "Varys",
                "required_kind": "character",
            },
        ),
        (
            "And it tells us again years later when Olenna is holding a cup of poison of her own.",
            {
                "expected_visual": "Olenna holds the poison cup Jaime gives her at Highgarden",
                "scene_query": "Game of Thrones Olenna holding poison cup Highgarden dies",
                "quote": "I want her to know it was me.",
                "required_entity": "Olenna Tyrell",
                "required_kind": "character",
            },
        ),
    ],
)
def test_narration_grounded_exact_event_is_not_demoted_to_hide_a_timing_failure(text, directive):
    beat = _apply(text, **directive)

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == directive["expected_visual"]
    assert beat.scene_query == directive["scene_query"]
    assert beat.quote == directive["quote"]
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


def test_direct_exact_action_indirect_scene_pointer_and_narrated_dialogue_stay_exact():
    direct = _apply(
        "Joffrey collapses at the wedding feast, clawing at his throat.",
        expected_visual="Joffrey choking at the Purple Wedding feast",
        scene_query="Game of Thrones Joffrey choking Purple Wedding",
        required_entity="Joffrey Baratheon",
        required_kind="character",
    )
    indirect = _apply(
        "There is a real tragedy sitting inside that scene.",
        expected_visual="Ned realizes the gold cloaks have betrayed him",
        scene_query="Game of Thrones Ned betrayed throne room",
        required_entity="Ned Stark",
        required_kind="character",
    )
    dialogue = _apply(
        "He says, 'Chaos is a ladder.'",
        expected_visual="Petyr Baelish delivers the chaos speech",
        scene_query="Game of Thrones Littlefinger chaos is a ladder",
        quote="Chaos is a ladder.",
        required_entity="Petyr Baelish",
        required_kind="character",
    )
    conservative = _apply(
        "The trap was clever.",
        expected_visual="The concealed trap mechanism beneath the floor",
        scene_query="Example Show concealed trap mechanism",
        required_entity="the trap",
        required_kind="object",
    )

    for beat in (direct, indirect, dialogue, conservative):
        assert beat.visual_policy == P.EXACT
        assert beat.is_specific_claim is True
        assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


def test_grounding_branches_persist_in_analysis_metadata_and_round_trip():
    downgraded = _apply(
        "It is a rarer willingness.",
        expected_visual="Littlefinger pushes Lysa through the Moon Door",
        scene_query="Game of Thrones Littlefinger Moon Door",
        quote="I did warn you not to trust me.",
        required_entity="Petyr Baelish",
        required_kind="character",
    )
    grounded = _apply(
        "Joffrey collapses at the wedding feast.",
        expected_visual="Joffrey choking at the wedding feast",
        scene_query="Game of Thrones Joffrey choking wedding",
        required_entity="Joffrey Baratheon",
        required_kind="character",
    )
    grounded.index = 1
    analysis = A.ScriptAnalysis(movie_title="Game of Thrones")

    counts = A._record_beat_grounding_audit(analysis, [downgraded, grounded])

    assert counts == {
        "exact_directives": 2,
        "grounded_exact": 1,
        "downgraded": 1,
        "to_character_specific": 0,
        "to_generic_filler": 1,
    }
    restored = A.ScriptAnalysis.from_dict(analysis.to_dict())
    assert restored.beat_grounding_audit == analysis.beat_grounding_audit
    assert restored.beat_grounding_audit["beats"]["0"]["branch"] == \
        "ungrounded_exact_downgrade"
    assert restored.beat_grounding_audit["beats"]["1"]["branch"] == "grounded_exact"
