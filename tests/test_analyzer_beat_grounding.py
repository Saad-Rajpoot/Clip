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


def test_short_common_quote_without_literal_narration_is_sanitized_only():
    """The measured ``Very well`` trap must not become an unrelated ASR scene lock."""
    expected = "Tywin Lannister grants Tyrion's demand for a trial by combat without argument"
    query = "Game of Thrones Tywin grants trial by combat Tyrion very well"
    beat = _apply(
        "He does not argue, and he does not refuse.",
        expected_visual=expected,
        scene_query=query,
        quote="Very well.",
        required_entity="Tywin Lannister",
        required_kind="character",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == expected
    assert beat.scene_query == "Game of Thrones Tywin grants trial by combat Tyrion"
    assert beat.required_entity == "Tywin Lannister"
    assert beat.required_kind == "character"
    assert beat.quote == ""
    marker = beat._analyzer_grounding_guard
    assert marker["branch"] == "grounded_exact_sanitized"
    assert marker["reason"] == "short_common_quote_not_literally_narrated"
    assert marker["sanitized_fields"] == ["scene_query", "quote"]
    assert marker["sanitized_values"] == {
        "scene_query": "Game of Thrones Tywin grants trial by combat Tyrion",
        "quote": "",
    }


@pytest.mark.parametrize(("text", "quote"), [
    ("Tywin answers, 'Very well.'", "Very well."),
    ("Tyrion fires while Tywin says nothing.", "You shot me."),
])
def test_literal_or_three_token_quote_remains_for_downstream_proof(text, quote):
    beat = _apply(
        text,
        expected_visual="Tyrion and Tywin in the promised exact scene",
        scene_query="Game of Thrones Tyrion Tywin exact scene",
        quote=quote,
        required_entity="Tywin Lannister",
        required_kind="character",
    )

    assert beat.quote == quote
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


def test_cross_scene_enumeration_is_retyped_as_montage_without_narrowing_contract():
    """Beat 134 keeps all three scenes while existing policy resolves montage to CHARACTER."""
    narration = (
        "Jaime opening the cell, Varys moving him through the city, "
        "Shae in his father's bed,")
    query = "Game of Thrones Tyrion finds Shae in Tywin's bed 'It's you'"

    beat = _apply(
        narration,
        expected_visual=narration,
        scene_query=query,
        required_entity="Shae",
        required_kind="character",
        shot_intent="emotional_closeup",
    )

    assert beat.visual_policy == P.EXACT
    assert P.policy_of(beat) == P.CHARACTER
    assert beat.is_specific_claim is True
    assert beat.expected_visual == narration
    assert beat.scene_query == query
    assert beat.required_entity == "Shae"
    assert beat.required_kind == "montage"
    assert beat.shot_intent == "montage"
    marker = beat._analyzer_grounding_guard
    assert marker["branch"] == "multi_event_montage_retyped"
    assert marker["reason"] == "multi_event_montage_retyped"
    assert marker["to_policy"] == P.CHARACTER
    assert marker["sanitized_fields"] == ["required_kind", "shot_intent"]
    assert marker["sanitized_values"] == {
        "required_kind": "montage", "shot_intent": "montage"}
    assert marker["multi_event_clause_count"] == 3
    assert marker["multi_event_location_groups"] == [
        "confinement", "urban", "sleeping_quarters"]
    assert marker["multi_event_target_clause"] == "Shae in his father's bed"
    assert marker["multi_event_original_required_kind"] == "character"
    assert marker["multi_event_original_shot_intent"] == "emotional_closeup"


@pytest.mark.parametrize(
    ("narration", "query"),
    [
        # Two clauses can be a single ordinary scene; the measured defect needs three named events.
        ("Tyrion faces Tywin, Bronn watches him",
         "Game of Thrones Tyrion faces Tywin"),
        # A query naming another enumerated subject may genuinely target their combined scene.
        ("Tyrion faces Tywin, Bronn watches him, Jaime enters the room",
         "Game of Thrones Tyrion Tywin Bronn confrontation"),
        # Distinct named subjects are not cross-scene proof: all three share one exact encounter.
        ("Arya draws her sword, Joffrey threatens Mycah, Sansa watches the fight",
         "Game of Thrones Arya draws sword against Joffrey Mycah"),
    ],
)
def test_ordinary_or_same_scene_multi_subject_exact_contract_is_not_retyped(
        narration, query):
    beat = _apply(
        narration,
        expected_visual=narration,
        scene_query=query,
        required_entity="Tyrion",
        required_kind="character",
    )

    assert beat.visual_policy == P.EXACT
    assert P.policy_of(beat) == P.EXACT
    assert beat.expected_visual == narration
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


@pytest.mark.parametrize("persisted_policy", [P.EXACT, P.CHARACTER])
def test_schema10_migrates_cached_beat134_montage_even_after_policy_softening(
        persisted_policy):
    narration = (
        "Jaime opening the cell, Varys moving him through the city, "
        "Shae in his father's bed,")
    beat = ScriptSegment(
        index=134,
        text=narration,
        expected_visual=narration,
        scene_query="Game of Thrones Tyrion finds Shae in Tywin's bed 'It's you'",
        quote="",
        required_entity="Shae",
        required_kind="character",
        shot_intent="emotional_closeup",
        visual_policy=persisted_policy,
        is_specific_claim=True,
    )
    prior_marker = {
        "branch": "grounded_exact_sanitized",
        "reason": "short_common_quote_not_literally_narrated",
        "from_policy": P.EXACT,
        "shared_terms": ["bed", "shae"],
        "entity_grounded": True,
        "guard_schema": 9,
        "sanitized_fields": ["expected_visual", "quote"],
        "sanitized_values": {"expected_visual": narration, "quote": ""},
        "cached_revalidation_schema": 9,
    }
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 9,
        "counts": {},
        "beats": {"134": prior_marker},
    })

    first = A.revalidate_cached_directions([beat], analysis)

    assert first["changed_indices"] == [134]
    assert first["exact_revalidated"] == 1
    assert first["changes"]["134"]["changed_fields"] == [
        "required_kind", "shot_intent"]
    assert beat.expected_visual == narration
    assert beat.visual_policy == persisted_policy
    assert P.policy_of(beat) == P.CHARACTER
    assert beat.required_kind == beat.shot_intent == "montage"
    marker = analysis.beat_grounding_audit["beats"]["134"]
    assert marker["branch"] == "multi_event_montage_retyped"
    assert marker["reason"] == "multi_event_montage_retyped"
    assert marker["sanitization_reasons"] == [
        "short_common_quote_not_literally_narrated"]
    assert marker["sanitized_values"] == {
        "expected_visual": narration,
        "quote": "",
        "required_kind": "montage",
        "shot_intent": "montage",
    }
    assert analysis.beat_grounding_audit["counts"]["short_common_quote_sanitized"] == 1
    assert analysis.beat_grounding_audit["counts"]["multi_event_montage_retyped"] == 1
    assert analysis.beat_grounding_audit["schema"] == 10

    loaded = ScriptSegment.from_dict(beat.to_dict())
    loaded_analysis = A.ScriptAnalysis.from_dict(analysis.to_dict())
    second = A.revalidate_cached_directions([loaded], loaded_analysis)

    assert second["changed_count"] == 0
    assert second["exact_revalidated"] == 0
    assert loaded.expected_visual == narration
    assert loaded.required_kind == loaded.shot_intent == "montage"


def test_schema9_cached_revalidation_clears_short_quote_and_auto_breakout_idempotently():
    """Existing schema-8 portal jobs receive the guard without re-running the LLM analyzer."""
    beats = [
        ScriptSegment(
            index=index,
            text=text,
            expected_visual=expected,
            scene_query="Game of Thrones Tywin grants trial by combat Tyrion very well",
            quote="Very well.",
            required_entity="Tywin Lannister",
            required_kind="character",
            visual_policy=P.EXACT,
            is_specific_claim=True,
            breakout_candidate=True,
        )
        for index, text, expected in (
            (87,
             "In a single sentence, Tyrion transfers the decision from Tywin's panel to a fight.",
             "Tywin Lannister, irritated, saying 'Very well.'"),
            (90,
             "He does not argue, and he does not refuse.",
             "Tywin grants Tyrion's demand for trial by combat without argument"),
        )
    ]
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 8,
        "beats": {
            str(beat.index): {
                "branch": "grounded_exact",
                "reason": "conservative_no_mismatch_proven",
                "guard_schema": 8,
                "cached_revalidation_schema": 8,
            }
            for beat in beats
        },
        "breakout_provenance": {"87": "quote_derived", "90": "quote_derived"},
    })

    first = A.revalidate_cached_directions(beats, analysis)

    assert first["changed_indices"] == [87, 90]
    assert all(beat.quote == "" for beat in beats)
    assert all(beat.breakout_candidate is False for beat in beats)
    assert all(beat.visual_policy == P.EXACT for beat in beats)
    assert beats[0].expected_visual == beats[0].text
    assert beats[1].expected_visual == \
        "Tywin grants Tyrion's demand for trial by combat without argument"
    assert all(beat.scene_query == "Game of Thrones Tywin grants trial by combat Tyrion"
               for beat in beats)
    assert beats[0]._analyzer_grounding_guard["sanitized_fields"] == [
        "expected_visual", "scene_query", "quote"]
    assert beats[0]._analyzer_grounding_guard["sanitized_values"] == {
        "expected_visual": beats[0].text,
        "scene_query": "Game of Thrones Tywin grants trial by combat Tyrion",
        "quote": "",
    }
    assert beats[1]._analyzer_grounding_guard["sanitized_fields"] == [
        "scene_query", "quote"]
    assert analysis.beat_grounding_audit["schema"] == 10
    assert analysis.beat_grounding_audit["counts"]["short_common_quote_sanitized"] == 2
    assert analysis.beat_grounding_audit["breakout_provenance"] == {
        "87": "quote_derived_cleared", "90": "quote_derived_cleared"}

    resumed_beats = [ScriptSegment.from_dict(beat.to_dict()) for beat in beats]
    resumed_analysis = A.ScriptAnalysis.from_dict(analysis.to_dict())
    second = A.revalidate_cached_directions(resumed_beats, resumed_analysis)

    assert second["changed_count"] == 0
    assert second["exact_revalidated"] == 0
    assert resumed_analysis.beat_grounding_audit["schema"] == 10
    assert resumed_analysis.beat_grounding_audit["counts"]["short_common_quote_sanitized"] == 2


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
    "text, directive, grounded_role",
    [
        (
            "the master-at-arms of the Red Keep.",
            {
                "expected_visual": "Close up of the master-at-arms examining the dagger",
                "scene_query": "Game of Thrones Rodrik master-at-arms dagger question",
                "required_entity": "master-at-arms",
                "required_kind": "character",
            },
            "master-at-arms",
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
            "",
        ),
    ],
)
def test_named_subject_without_a_narrated_exact_moment_becomes_character_specific(
        text, directive, grounded_role):
    beat = _apply(text, **directive)

    assert beat.visual_policy == P.CHARACTER
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == grounded_role
    assert beat.quote == ""
    # The narration names the role/person, so subject correctness remains required.
    assert beat.required_entity == (grounded_role or directive["required_entity"])
    assert beat.required_kind == directive["required_kind"]
    assert beat._analyzer_grounding_guard["branch"] == "ungrounded_exact_downgrade"
    assert beat._analyzer_grounding_guard["to_policy"] == P.CHARACTER
    if grounded_role:
        assert beat._analyzer_grounding_guard["grounded_subject_role"] == grounded_role


def test_measured_beat32_verb_less_role_cannot_inherit_analyzer_guessed_identity():
    text = "to the master-at-arms of the Red Keep."
    beat = _apply(
        text,
        expected_visual="The master-at-arms examining the dagger",
        scene_query="Game of Thrones master-at-arms Aron Santagar dagger",
        quote="It's Valyrian steel.",
        required_entity="Aron Santagar",
        required_kind="character",
    )

    assert beat.visual_policy == P.FILLER
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    assert beat.required_entity == beat.required_kind == ""
    marker = beat._analyzer_grounding_guard
    assert marker["branch"] == "ungrounded_exact_downgrade"
    assert marker["reason"] == "verb_less_role_has_analyzer_guessed_identity"
    assert marker["to_policy"] == P.FILLER
    assert marker["grounded_subject_role"] == "master-at-arms"
    assert marker["analyzer_guessed_required_entity"] == "Aron Santagar"


def test_generic_the_one_comparison_cannot_inherit_an_unrelated_exact_scene_or_quote():
    text = ("The one who was always three moves ahead of people who were physically stronger, "
            "better born, and better protected.")
    beat = _apply(
        text,
        expected_visual="Littlefinger holds a knife to Ned Stark's throat, betraying him",
        scene_query="Game of Thrones Littlefinger betrays Ned Stark knife",
        quote="I did warn you not to trust me.",
        required_entity="Petyr Baelish",
        required_kind="character",
    )

    assert beat.visual_policy == P.FILLER
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    assert beat.required_entity == beat.required_kind == ""
    assert beat._analyzer_grounding_guard["reason"] == \
        "generic_relative_comparison_has_no_grounded_entity_or_event"


@pytest.mark.parametrize(
    "text, expected_visual, required_entity",
    [
        (
            "The one who killed Joffrey was stronger than either guard: Petyr Baelish.",
            "Petyr Baelish kills Joffrey in front of two guards",
            "Petyr Baelish",
        ),
        (
            "The one who survived the Red Wedding was better protected afterwards.",
            "A survivor escapes the Red Wedding",
            "Red Wedding",
        ),
    ],
)
def test_the_one_comparison_keeps_a_locally_grounded_entity_or_event_exact(
        text, expected_visual, required_entity):
    beat = _apply(
        text,
        expected_visual=expected_visual,
        scene_query=f"Example Show {expected_visual}",
        required_entity=required_entity,
        required_kind="character",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == expected_visual
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


def test_bare_named_event_keeps_event_exact_but_removes_invented_composite_contract():
    text = "The obvious objection arrives early, so let us take it early. The Purple Wedding."
    beat = _apply(
        text,
        expected_visual=("Establishing shot of the Purple Wedding feast in the throne room, "
                         "Joffrey and Margaery raising goblets"),
        scene_query="Game of Thrones Purple Wedding feast Joffrey Margaery goblets",
        quote="He's choking!",
        required_entity="Purple Wedding feast",
        required_kind="event",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == "The Purple Wedding"
    assert beat.scene_query == "The Purple Wedding"
    assert beat.required_entity == "The Purple Wedding"
    assert beat.required_kind == "event"
    assert beat.quote == ""
    marker = beat._analyzer_grounding_guard
    assert marker["branch"] == "grounded_exact_sanitized"
    assert marker["reason"] == "bare_named_event_staging_removed"
    assert marker["grounded_event"] == "The Purple Wedding"
    assert marker["sanitized_fields"] == [
        "expected_visual", "scene_query", "required_entity", "quote"]


def test_named_event_guard_does_not_strip_a_narrated_physical_event_or_non_event_title():
    physical = _apply(
        "At the Purple Wedding, Joffrey raises his goblet.",
        expected_visual="Joffrey raises his goblet at the Purple Wedding",
        scene_query="Game of Thrones Joffrey goblet Purple Wedding",
        required_entity="Purple Wedding",
        required_kind="event",
    )
    non_event = _apply(
        "The comparison ends here. The Purple Wedding.",
        expected_visual="Margaery at the Purple Wedding",
        scene_query="Game of Thrones Margaery Purple Wedding",
        required_entity="Margaery Tyrell",
        required_kind="character",
    )

    for beat in (physical, non_event):
        assert beat.visual_policy == P.EXACT
        assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


@pytest.mark.parametrize(
    "text, directive",
    [
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


def test_shared_abstract_term_does_not_ground_an_invented_subject_for_vague_narration():
    text = "Verification is in motion. Someone is checking."
    beat = _apply(
        text,
        expected_visual="Ser Rodrik showing the dagger, emphasizing the act of verification",
        scene_query="Game of Thrones Ser Rodrik master-at-arms dagger question",
        required_entity="Ser Rodrik Cassel",
        required_kind="character",
    )

    assert beat.visual_policy == P.FILLER
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    assert beat.required_entity == beat.required_kind == ""
    assert beat._analyzer_grounding_guard["reason"] == \
        "vague_subject_has_no_grounded_subject"
    # ``verification`` overlaps, but it is an abstract relation—not a grounded subject.
    assert beat._analyzer_grounding_guard["shared_terms"] == ["verification"]


@pytest.mark.parametrize(
    "text, expected_visual, scene_query, required_entity, required_kind",
    [
        (
            "There is no ledger.",
            "Close-up of the dagger's blank hilt with no ledger or names",
            "Game of Thrones dagger brothel no ledger",
            "the dagger",
            "object",
        ),
        (
            "There are no witnesses named.",
            "Wide shot of the brothel with only Catelyn and Littlefinger present",
            "Game of Thrones brothel interior Catelyn Littlefinger alone",
            "the brothel",
            "location",
        ),
    ],
)
def test_pure_negative_existence_cannot_inherit_an_exact_scene_from_essay_context(
        text, expected_visual, scene_query, required_entity, required_kind):
    beat = _apply(
        text,
        expected_visual=expected_visual,
        scene_query=scene_query,
        required_entity=required_entity,
        required_kind=required_kind,
    )

    assert beat.visual_policy == P.ABSTRACT
    assert beat.is_specific_claim is False
    assert beat.expected_visual == text
    assert beat.scene_query == beat.quote == ""
    assert beat.required_entity == beat.required_kind == ""
    marker = beat._analyzer_grounding_guard
    assert marker["reason"] == "negative_existence_is_not_an_observable_exact_scene"
    assert marker["to_policy"] == P.ABSTRACT


def test_embedded_negative_clause_with_a_real_subject_is_not_mass_downgraded():
    beat = _apply(
        "It was the fact that in Westeros there was no way to check anything he said.",
        expected_visual="Littlefinger speaks while nobody can verify his story",
        scene_query="Game of Thrones Littlefinger speaking",
        required_entity="Petyr Baelish",
        required_kind="character",
    )

    assert beat.visual_policy == P.EXACT
    assert beat._analyzer_grounding_guard["reason"] != \
        "negative_existence_is_not_an_observable_exact_scene"


def test_physical_event_with_record_intent_keeps_action_but_clears_borrowed_quote():
    text = ("And it tells us again years later when Olenna is holding a cup of poison of her "
            "own wants it on the record.")
    directive = {
        "expected_visual": "Olenna holding the poison cup with Jaime present",
        "scene_query": "Game of Thrones Olenna drinks poison Jaime Highgarden",
        "quote": "Tell Cersei. I want her to know it was me.",
        "required_entity": "Olenna Tyrell",
        "required_kind": "character",
    }

    beat = _apply(text, **directive)

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == directive["expected_visual"]
    assert beat.scene_query == directive["scene_query"]
    assert beat.required_entity == directive["required_entity"]
    assert beat.quote == ""
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact_sanitized"
    assert beat._analyzer_grounding_guard["reason"] == \
        "record_intent_is_not_verbatim_dialogue"
    assert beat._analyzer_grounding_guard["sanitized_fields"] == ["quote"]


def test_information_event_stays_exact_but_invented_delivery_staging_is_removed():
    text = "That question is exactly how Varys learns she is in the city."
    directive = {
        "expected_visual": "A child whispers to Varys and his eyes widen",
        "scene_query": "Game of Thrones Varys learns Catelyn is in the city",
        "quote": "Lady Stark is here, in King's Landing.",
        "required_entity": "Varys",
        "required_kind": "character",
    }

    beat = _apply(text, **directive)

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    # The real narrated event/query survives. Only the analyzer-invented mechanism and its
    # unsupported line are removed from the hard conjunction.
    assert beat.expected_visual == text
    assert beat.scene_query == directive["scene_query"]
    assert beat.quote == ""
    assert beat.required_entity == "Varys"
    assert beat._analyzer_grounding_guard["branch"] == "grounded_exact_sanitized"
    assert beat._analyzer_grounding_guard["reason"] == \
        "unsupported_information_staging_removed"
    assert beat._analyzer_grounding_guard["sanitized_fields"] == ["expected_visual", "quote"]


def test_measured_dagger_beat_removes_invented_camera_composition_not_exact_contract():
    text = "the weapon they left behind is the only thing she has."
    query = "Game of Thrones catspaw Valyrian steel dagger left behind"

    beat = _apply(
        text,
        expected_visual=("Close-up of the Valyrian steel dagger lying on the floor or in "
                         "Catelyn's hands after the attack"),
        scene_query=query,
        required_entity="Valyrian steel dagger",
        required_kind="object",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == (
        "Exact scene: Game of Thrones catspaw Valyrian steel dagger left behind. "
        "Also show: Catelyn.")
    assert beat.scene_query == query
    assert beat.required_entity == "Valyrian steel dagger"
    assert beat.required_kind == "object"
    marker = beat._analyzer_grounding_guard
    assert marker["branch"] == "grounded_exact_sanitized"
    assert marker["reason"] == "unsupported_camera_composition_removed"
    assert marker["camera_cues"] == ["close_up"]
    assert marker["sanitized_fields"] == ["expected_visual"]
    assert "Catelyn's hands" in marker["camera_original_values"]["expected_visual"]


def test_measured_brothel_beat_removes_camera_words_but_keeps_semantic_query():
    text = "brothel because it could not be tested."

    beat = _apply(
        text,
        expected_visual=("Wide shot pulling back from the brothel, the private chamber where no "
                         "one can contest the story."),
        scene_query="Game of Thrones brothel wide shot interior Catelyn Littlefinger",
        required_entity="the brothel",
        required_kind="location",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == \
        "Exact scene: Game of Thrones brothel interior Catelyn Littlefinger."
    assert beat.scene_query == "Game of Thrones brothel interior Catelyn Littlefinger"
    assert beat.required_entity == "the brothel"
    assert beat.required_kind == "location"
    marker = beat._analyzer_grounding_guard
    assert marker["reason"] == "unsupported_camera_composition_removed"
    assert marker["camera_cues"] == ["pull_back", "wide_shot"]
    assert marker["sanitized_fields"] == ["expected_visual", "scene_query"]


def test_camera_guard_preserves_authored_cinematography_and_ordinary_wide_prose():
    authored = _apply(
        "The wide shot pulls back from the brothel as Catelyn enters.",
        expected_visual="Wide shot pulling back from the brothel as Catelyn enters",
        scene_query="Example Show wide shot pulls back brothel Catelyn",
        required_entity="the brothel",
        required_kind="location",
    )
    lexical_control = _apply(
        "The wide gate opens onto the courtyard.",
        expected_visual="A wide gate opens onto the courtyard",
        scene_query="Example Show wide gate opens courtyard",
        required_entity="the courtyard gate",
        required_kind="object",
    )

    assert authored.expected_visual == \
        "Wide shot pulling back from the brothel as Catelyn enters"
    assert authored.scene_query == "Example Show wide shot pulls back brothel Catelyn"
    assert authored._analyzer_grounding_guard["branch"] == "grounded_exact"
    assert lexical_control.expected_visual == "A wide gate opens onto the courtyard"
    assert lexical_control._analyzer_grounding_guard["branch"] == "grounded_exact"


def test_query_only_camera_cleanup_keeps_grounded_expected_visual():
    beat = _apply(
        "Catelyn enters the brothel.",
        expected_visual="Catelyn enters the brothel private chamber",
        scene_query="Example Show Catelyn wide shot brothel entrance",
        required_entity="the brothel",
        required_kind="location",
    )

    assert beat.expected_visual == "Catelyn enters the brothel private chamber"
    assert beat.scene_query == "Example Show Catelyn brothel entrance"
    assert beat._analyzer_grounding_guard["sanitized_fields"] == ["scene_query"]


def test_stacked_aerial_wide_camera_phrase_is_fully_removed_without_mangling_place_name():
    beat = _apply(
        "The story returns to King's Landing.",
        expected_visual="Aerial wide shot of King's Landing at dusk",
        scene_query="Game of Thrones King's Landing aerial",
        required_entity="King's Landing",
        required_kind="location",
    )

    assert "aerial" not in beat.expected_visual.lower()
    assert "aerial" not in beat.scene_query.lower()
    assert "King's Landing" in beat.expected_visual
    assert "King Landing" not in beat.expected_visual
    assert set(beat._analyzer_grounding_guard["camera_cues"]) >= {
        "aerial_view", "compound_shot", "wide_shot"}


def test_camera_cleanup_does_not_promote_an_invented_capitalized_place():
    beat = _apply(
        "That is not a background detail of the setting.",
        expected_visual="Aerial wide shot of King's Landing, Red Keep and cityscape",
        scene_query="Game of Thrones King's Landing aerial wide shot",
        required_entity="King's Landing",
        required_kind="location",
    )

    assert beat.scene_query == "Game of Thrones King's Landing"
    assert beat.expected_visual == "Exact scene: Game of Thrones King's Landing."
    assert "Red Keep" not in beat.expected_visual

    ship = _apply(
        "A ship waiting in the bay.",
        expected_visual=("Wide shot of Littlefinger's ship anchored in the bay, "
                         "King's Landing on the shore"),
        scene_query="Game of Thrones Littlefinger ship waiting bay Purple Wedding",
        required_entity="Littlefinger's ship",
        required_kind="object",
    )
    assert ship.expected_visual == (
        "Exact scene: Game of Thrones Littlefinger ship waiting bay Purple Wedding.")
    assert "King's Landing" not in ship.expected_visual


def test_camera_cleanup_preserves_narration_authored_action_and_location():
    beat = _apply(
        "The dagger lies on the floor.",
        expected_visual="Close-up of the dagger lying on the floor",
        scene_query="Example Show dagger",
        required_entity="the dagger",
        required_kind="object",
    )

    assert beat.expected_visual == (
        "Exact scene: Example Show dagger. "
        "Narrated action/location: The dagger lies on the floor.")
    assert beat.scene_query == "Example Show dagger"


def test_adjacent_quote_copy_is_removed_only_from_distinct_grounded_physical_event():
    cup = _apply(
        "And it tells us again years later when Olenna is holding a cup of poison of her own.",
        expected_visual="Olenna Tyrell holding a cup of poison after Jaime offers it",
        scene_query="Game of Thrones Olenna holding poison cup Highgarden dies",
        quote="I want her to know it was me.",
        required_entity="Olenna Tyrell",
        required_kind="character",
    )
    record = _apply(
        "wants it on the record.",
        expected_visual="Olenna says I want her to know it was me",
        scene_query="Game of Thrones Olenna confession",
        quote="I want her to know it was me.",
        required_entity="Olenna Tyrell",
        required_kind="character",
    )
    authored = _apply(
        "She wants Cersei to know the hand was hers.",
        expected_visual="Olenna declares responsibility for Joffrey's murder",
        scene_query="Game of Thrones Olenna tells Jaime Cersei know",
        quote="I want her to know it was me.",
        required_entity="Olenna Tyrell",
        required_kind="character",
    )
    cup.index, record.index, authored.index = 61, 62, 63

    assert A._sanitize_adjacent_quote_borrowing([cup, record, authored]) == 1

    assert cup.visual_policy == P.EXACT
    assert cup.expected_visual == "Olenna Tyrell holding a cup of poison after Jaime offers it"
    assert cup.scene_query == "Game of Thrones Olenna holding poison cup Highgarden dies"
    assert cup.quote == ""
    assert cup._analyzer_grounding_guard["branch"] == "grounded_exact_sanitized"
    assert cup._analyzer_grounding_guard["sanitized_fields"] == ["quote"]
    assert cup._analyzer_grounding_guard["quote_support_beat"] == 63
    # The dialogue-focused neighbouring beats retain the real line.
    assert record.quote == authored.quote == "I want her to know it was me."
    audit = A.ScriptAnalysis()
    counts = A._record_beat_grounding_audit(audit, [cup, record, authored])
    assert counts["sanitized_exact"] == 1
    assert counts["adjacent_quote_copy_sanitized"] == 1
    assert counts["information_staging_sanitized"] == 0


def test_literal_pronoun_action_and_locally_authored_dialogue_are_not_sanitized():
    table = _apply(
        "She puts it on the table and asks one question.",
        expected_visual="Catelyn places the dagger on the table in front of Littlefinger",
        scene_query="Game of Thrones Catelyn dagger table Littlefinger brothel",
        required_entity="Catelyn Stark",
        required_kind="character",
    )
    action_with_line = _apply(
        "He says, 'Not today,' while holding the sword.",
        expected_visual="The swordsman holds his sword and says not today",
        scene_query="Example Show swordsman not today sword",
        quote="Not today.",
        required_entity="the swordsman",
        required_kind="character",
    )
    duplicate = _apply(
        "The line is remembered afterwards.",
        expected_visual="The swordsman remembering the earlier exchange",
        scene_query="Example Show swordsman remembers",
        quote="Not today.",
        required_entity="the swordsman",
        required_kind="character",
    )
    table.index, action_with_line.index, duplicate.index = 4, 5, 6

    assert A._sanitize_adjacent_quote_borrowing([table, action_with_line, duplicate]) == 0
    assert table.visual_policy == P.EXACT
    assert table.expected_visual.startswith("Catelyn places the dagger")
    assert table._analyzer_grounding_guard["branch"] == "grounded_exact"
    assert action_with_line.quote == "Not today."


def test_whole_script_quote_owner_removes_only_distant_unrelated_visual_copies():
    aim = _apply(
        "Everyone remembers how it ended, the crossbow and the privy.",
        expected_visual="Tyrion holding a crossbow aiming at Tywin on the privy",
        scene_query="Tyrion kills Tywin crossbow privy",
        quote="You're no son of mine.", required_entity="Tyrion and Tywin",
        required_kind="scene", shot_intent="action")
    speech = _apply(
        "because of what he chooses to say to his son in the last conversation of his life.",
        expected_visual="Tywin saying 'You're no son of mine' on the privy",
        scene_query="Tywin says you're no son of mine",
        quote="You're no son of mine.", required_entity="Tywin",
        required_kind="character", shot_intent="emotional_closeup")
    death = _apply(
        "Tywin dies from Tyrion's crossbow.",
        expected_visual="Tywin being shot by Tyrion's crossbow, dying",
        scene_query="Tywin shot by Tyrion crossbow death",
        quote="You're no son of mine.", required_entity="Tywin",
        required_kind="character", shot_intent="action")
    aim.index, speech.index, death.index = 4, 79, 146

    assert A._sanitize_adjacent_quote_borrowing([aim, speech, death]) == 2
    assert aim.quote == death.quote == ""
    assert speech.quote == "You're no son of mine."
    assert aim.expected_visual.startswith("Tyrion holding")
    assert death.expected_visual.startswith("Tywin being shot")
    assert aim._analyzer_grounding_guard["original_quote"] == "You're no son of mine."
    assert death._analyzer_grounding_guard["quote_owner_index"] == 79
    assert A._sanitize_adjacent_quote_borrowing([aim, speech, death]) == 0

    audit = A.ScriptAnalysis()
    counts = A._record_beat_grounding_audit(audit, [aim, speech, death])
    assert counts["whole_script_quote_copy_sanitized"] == 2
    assert audit.beat_grounding_audit["quote_ownership"] == [{
        "quote": "You're no son of mine.",
        "members": [4, 79, 146],
        "owner_index": 79,
        "owner_reason": "literal_line_in_expected_visual",
        "sanitized_indices": [4, 146],
        "status": "resolved",
    }]


def test_whole_script_action_owner_beats_aftermath_but_ambiguous_copies_fail_closed():
    aftermath = _apply(
        "The son stands over his dying father.",
        expected_visual="Tyrion standing over Tywin after shooting him",
        scene_query="Tyrion standing over Tywin", quote="You shot me.",
        required_entity="Tyrion and Tywin", required_kind="scene",
        shot_intent="emotional_closeup")
    trigger = _apply(
        "Tyrion fired the crossbow.",
        expected_visual="Tyrion pulls the crossbow trigger and shoots Tywin",
        scene_query="Tyrion shoots Tywin", quote="You shot me.",
        required_entity="Tyrion", required_kind="character", shot_intent="action")
    aftermath.index, trigger.index = 5, 142
    assert A._sanitize_adjacent_quote_borrowing([aftermath, trigger]) == 1
    assert aftermath.quote == ""
    assert trigger.quote == "You shot me."
    assert aftermath._analyzer_grounding_guard["quote_owner_index"] == 142

    first = _apply(
        "He agrees behind the closed door.", expected_visual="Tywin says 'It is done'",
        scene_query="Tywin agrees", quote="It is done.", required_entity="Tywin",
        required_kind="character")
    second = _apply(
        "Tywin agrees immediately.", expected_visual="Tywin says 'It is done' immediately",
        scene_query="Tywin agrees", quote="It is done.", required_entity="Tywin",
        required_kind="character")
    first.index, second.index = 14, 20
    assert A._sanitize_adjacent_quote_borrowing([first, second]) == 0
    assert first.quote == second.quote == "It is done."
    group = first._analyzer_grounding_guard["quote_ownership_group"]
    assert group["status"] == "ambiguous_preserved"
    assert group["owner_index"] is None


def test_distant_repeat_of_same_spoken_event_keeps_both_hard_quote_contracts():
    direct = _apply(
        "Tyrion stands and demands trial by combat.",
        expected_visual="Tyrion shouts 'I demand a trial by combat'",
        scene_query="Tyrion trial by combat", quote="I demand a trial by combat!",
        required_entity="Tyrion", required_kind="character", shot_intent="action")
    revisit = _apply(
        "He lost the verdict he arranged.",
        expected_visual="Tyrion demands a trial by combat in the throne room",
        scene_query="Tyrion trial by combat court", quote="I demand a trial by combat.",
        required_entity="Tyrion", required_kind="character", shot_intent="action")
    direct.index, revisit.index = 83, 172

    assert A._sanitize_adjacent_quote_borrowing([direct, revisit]) == 0
    assert direct.quote and revisit.quote
    group = direct._analyzer_grounding_guard["quote_ownership_group"]
    assert group["owner_index"] == 83
    assert group["status"] == "co_temporal_duplicates_preserved"


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
    finite_action = _apply(
        "A king dies at his own feast in front of hundreds of people.",
        expected_visual="Joffrey choking and dying at the Purple Wedding feast",
        scene_query="Game of Thrones Joffrey choking Purple Wedding",
        required_entity="Joffrey Baratheon",
        required_kind="character",
    )

    for beat in (direct, indirect, dialogue, conservative, finite_action):
        assert beat.visual_policy == P.EXACT
        assert beat.is_specific_claim is True
        assert beat._analyzer_grounding_guard["branch"] == "grounded_exact"


@pytest.mark.parametrize(
    "text, expected_visual, entity",
    [
        ("The Hound abandons Arya.", "The Hound abandons Arya on the road", "The Hound"),
        ("A guard blocks the gate.", "A guard physically blocks the gate", "guard"),
        ("The king executes the prisoner.", "The king executes the prisoner", "the king"),
        ("The queen embraces her child.", "The queen embraces her child", "the queen"),
    ],
)
def test_determiner_led_finite_actions_outside_small_verb_lexicon_stay_exact(
        text, expected_visual, entity):
    beat = _apply(
        text,
        expected_visual=expected_visual,
        scene_query=f"Example Show {expected_visual}",
        required_entity=entity,
        required_kind="character",
    )

    assert beat.visual_policy == P.EXACT
    assert beat.is_specific_claim is True
    assert beat.expected_visual == expected_visual
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
    sanitized = _apply(
        "Varys learns Catelyn is in the city.",
        expected_visual="A child whispers the news into Varys's ear",
        scene_query="Game of Thrones Varys learns Catelyn is in city",
        quote="Lady Stark is here, in King's Landing.",
        required_entity="Varys",
        required_kind="character",
    )
    sanitized.index = 2
    analysis = A.ScriptAnalysis(movie_title="Game of Thrones")

    counts = A._record_beat_grounding_audit(analysis, [downgraded, grounded, sanitized])

    assert counts == {
        "exact_directives": 3,
        "grounded_exact": 2,
        "sanitized_exact": 1,
        "information_staging_sanitized": 1,
        "record_intent_quote_sanitized": 0,
        "short_common_quote_sanitized": 0,
        "multi_event_montage_retyped": 0,
        "bare_named_event_sanitized": 0,
        "unsupported_camera_composition_sanitized": 0,
        "adjacent_quote_copy_sanitized": 0,
        "whole_script_quote_copy_sanitized": 0,
        "nominal_role_contract_narrowed": 0,
        "nominal_guessed_identity_cleared": 0,
        "negative_existence_downgraded": 0,
        "downgraded": 1,
        "to_character_specific": 0,
        "to_generic_filler": 1,
        "to_abstract_effect": 0,
    }
    assert analysis.beat_grounding_audit["schema"] == 10
    restored = A.ScriptAnalysis.from_dict(analysis.to_dict())
    assert restored.beat_grounding_audit == analysis.beat_grounding_audit
    assert restored.beat_grounding_audit["beats"]["0"]["branch"] == \
        "ungrounded_exact_downgrade"
    assert restored.beat_grounding_audit["beats"]["1"]["branch"] == "grounded_exact"
    assert restored.beat_grounding_audit["beats"]["2"]["branch"] == \
        "grounded_exact_sanitized"


def test_cached_direction_revalidation_is_scoped_audited_and_idempotent_for_101_beats():
    beats = [
        ScriptSegment(
            index=i,
            text=f"Generic contextual narration {i}.",
            expected_visual=f"Generic contextual narration {i}.",
            visual_policy=P.FILLER,
        )
        for i in range(101)
    ]
    beats[31] = ScriptSegment(
        index=31,
        text="the master-at-arms of the Red Keep.",
        expected_visual="Close up of the master-at-arms examining the dagger",
        scene_query="Game of Thrones Rodrik master-at-arms dagger question",
        required_entity="master-at-arms",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    beats[32] = ScriptSegment(
        index=32,
        text="Verification is in motion. Someone is checking.",
        expected_visual="Ser Rodrik showing the dagger, emphasizing verification",
        scene_query="Game of Thrones Rodrik master-at-arms dagger question",
        required_entity="Ser Rodrik Cassel",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    beats[59] = ScriptSegment(
        index=59,
        text=("And it tells us again years later when Olenna is holding a cup of poison of her "
              "own wants it on the record."),
        expected_visual="Olenna holding the poison cup with Jaime present",
        scene_query="Game of Thrones Olenna drinks poison Jaime Highgarden",
        quote="Tell Cersei. I want her to know it was me.",
        required_entity="Olenna Tyrell",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
        breakout_candidate=True,
    )
    # A grounded exact control proves the helper is not a global exact->soft policy rewrite.
    beats[70] = ScriptSegment(
        index=70,
        text="Joffrey collapses at the wedding feast, clawing at his throat.",
        expected_visual="Joffrey choking at the Purple Wedding feast",
        scene_query="Game of Thrones Joffrey choking Purple Wedding",
        required_entity="Joffrey Baratheon",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    analysis = A.ScriptAnalysis(movie_title="Game of Thrones")

    first = A.revalidate_cached_directions(beats, analysis)

    assert first["scanned"] == 101
    assert first["exact_revalidated"] == 4
    assert first["changed_count"] == 3
    assert first["changed_indices"] == [31, 32, 59]
    assert beats[31].visual_policy == P.CHARACTER
    assert beats[31].required_entity == "master-at-arms"
    assert beats[32].visual_policy == P.FILLER
    assert beats[32].required_entity == ""
    assert first["changes"]["59"]["changed_fields"] == ["quote", "breakout_candidate"]
    assert beats[59].visual_policy == P.EXACT
    assert beats[59].quote == ""
    assert beats[59].breakout_candidate is False
    assert analysis.beat_grounding_audit["breakout_provenance"]["59"] == \
        "quote_derived_cleared"
    assert beats[70].visual_policy == P.EXACT
    assert analysis.beat_grounding_audit["cached_revalidation"]["changed_indices"] == \
        [31, 32, 59]
    saved_analysis = analysis.to_dict()
    saved_audit = saved_analysis["beat_grounding_audit"]
    # Exercise the real resume seam: ScriptSegment serialization drops process-local attributes,
    # so the second pass must restore guard markers from ScriptAnalysis.
    reloaded_beats = [ScriptSegment.from_dict(beat.to_dict()) for beat in beats]
    reloaded_analysis = A.ScriptAnalysis.from_dict(saved_analysis)

    second = A.revalidate_cached_directions(reloaded_beats, reloaded_analysis)

    assert second["changed_count"] == 0
    assert second["changed_indices"] == []
    assert second["exact_revalidated"] == 0
    resumed_audit = reloaded_analysis.to_dict()["beat_grounding_audit"]
    # Current-pass counters are truthful, while the first material diff and guard records survive.
    assert resumed_audit["cached_revalidation"]["changed_count"] == 0
    assert resumed_audit["cached_revalidation"]["exact_revalidated"] == 0
    assert resumed_audit["last_material_revalidation"]["changed_indices"] == [31, 32, 59]
    assert resumed_audit["beats"] == saved_audit["beats"]
    assert resumed_audit["breakout_provenance"] == saved_audit["breakout_provenance"]

    reloaded_again = [ScriptSegment.from_dict(beat.to_dict()) for beat in reloaded_beats]
    analysis_again = A.ScriptAnalysis.from_dict(reloaded_analysis.to_dict())
    third = A.revalidate_cached_directions(reloaded_again, analysis_again)
    assert third["changed_count"] == 0
    assert analysis_again.to_dict()["beat_grounding_audit"] == resumed_audit


@pytest.mark.parametrize("explicit_provenance", ["manual", "authored"])
def test_cached_revalidation_preserves_explicit_breakout_but_clears_quote_derived_flag(
        explicit_provenance):
    def _cached(index: int) -> ScriptSegment:
        return ScriptSegment(
            index=index,
            text=("Olenna is holding a cup of poison of her own and wants it on the "
                  "record."),
            expected_visual="Olenna holding the poison cup with Jaime present",
            scene_query="Game of Thrones Olenna drinks poison Jaime Highgarden",
            quote="Tell Cersei. I want her to know it was me.",
            required_entity="Olenna Tyrell",
            required_kind="character",
            visual_policy=P.EXACT,
            is_specific_claim=True,
            breakout_candidate=True,
        )

    automatic, manual = _cached(0), _cached(1)
    analysis = A.ScriptAnalysis(
        beat_grounding_audit={"breakout_provenance": {"1": explicit_provenance}})

    report = A.revalidate_cached_directions([automatic, manual], analysis)

    assert automatic.quote == manual.quote == ""
    assert automatic.breakout_candidate is False
    assert manual.breakout_candidate is True
    assert report["changes"]["0"]["changed_fields"] == ["quote", "breakout_candidate"]
    assert report["changes"]["1"]["changed_fields"] == ["quote"]
    assert analysis.beat_grounding_audit["breakout_provenance"] == {
        "0": "quote_derived_cleared",
        "1": explicit_provenance,
    }
    assert report["changes"]["0"]["guard"]["breakout_candidate_action"] == \
        "cleared_quote_derived_breakout"
    assert report["changes"]["1"]["guard"]["breakout_candidate_action"] == \
        "preserved_explicit_breakout"


def test_grounding_audit_records_breakout_origin_before_policy_finalization():
    automatic = ScriptSegment(index=0, text="A quoted beat", quote="A real line")
    explicit = ScriptSegment(
        index=1, text="An editor-authored breakout", quote="Another real line",
        breakout_candidate=True)
    analysis = A.ScriptAnalysis()

    A._record_beat_grounding_audit(analysis, [automatic, explicit])

    assert analysis.beat_grounding_audit["breakout_provenance"] == {
        "0": "quote_derived",
        "1": "explicit",
    }


def test_resume_preserves_fresh_sanitizer_reason_when_post_state_changes_nothing():
    beat = _apply(
        "Olenna is holding a cup of poison of her own and wants it on the record.",
        expected_visual="Olenna holding the poison cup with Jaime present",
        scene_query="Game of Thrones Olenna drinks poison Jaime Highgarden",
        quote="Tell Cersei. I want her to know it was me.",
        required_entity="Olenna Tyrell",
        required_kind="character",
    )
    analysis = A.ScriptAnalysis(movie_title="Game of Thrones")
    A._record_beat_grounding_audit(analysis, [beat])
    assert analysis.beat_grounding_audit["beats"]["0"]["reason"] == \
        "record_intent_is_not_verbatim_dialogue"

    loaded = ScriptSegment.from_dict(beat.to_dict())
    loaded_analysis = A.ScriptAnalysis.from_dict(analysis.to_dict())
    first = A.revalidate_cached_directions([loaded], loaded_analysis)

    assert first["changed_count"] == 0
    assert first["exact_revalidated"] == 0
    assert first["preserved_sanitized_provenance"] == 1
    marker = loaded_analysis.beat_grounding_audit["beats"]["0"]
    assert marker["branch"] == "grounded_exact_sanitized"
    assert marker["reason"] == "record_intent_is_not_verbatim_dialogue"
    assert marker["sanitized_fields"] == ["quote"]
    assert marker["cached_revalidation_schema"] == 10
    saved_audit = loaded_analysis.to_dict()["beat_grounding_audit"]

    loaded_again = ScriptSegment.from_dict(loaded.to_dict())
    analysis_again = A.ScriptAnalysis.from_dict(loaded_analysis.to_dict())
    second = A.revalidate_cached_directions([loaded_again], analysis_again)

    assert second["changed_count"] == 0
    assert analysis_again.to_dict()["beat_grounding_audit"] == saved_audit


def test_schema5_migrates_measured_comparison_event_and_cached_nominal_role_contracts():
    comparison = ScriptSegment(
        index=15,
        text=("The one who was always three moves ahead of people who were physically stronger, "
              "better born, and better protected."),
        expected_visual="Littlefinger holds a knife to Ned Stark's throat, betraying him",
        scene_query="Game of Thrones Littlefinger betrays Ned Stark knife",
        quote="I did warn you not to trust me.",
        required_entity="Petyr Baelish",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
        breakout_candidate=True,
    )
    nominal = ScriptSegment(
        index=32,
        text="to the master-at-arms of the Red Keep.",
        expected_visual="to the master-at-arms of the Red Keep.",
        scene_query="",
        quote="",
        required_entity="Aron Santagar",
        required_kind="character",
        visual_policy=P.CHARACTER,
        is_specific_claim=False,
    )
    event = ScriptSegment(
        index=54,
        text="The obvious objection arrives early, so let us take it early. The Purple Wedding.",
        expected_visual=("Establishing shot of the Purple Wedding feast in the throne room, "
                         "Joffrey and Margaery raising goblets"),
        scene_query="Game of Thrones Purple Wedding feast Joffrey Margaery goblets",
        quote="He's choking!",
        required_entity="Purple Wedding feast",
        required_kind="event",
        visual_policy=P.EXACT,
        is_specific_claim=True,
        breakout_candidate=True,
    )
    grounded_control = ScriptSegment(
        index=70,
        text="Joffrey collapses at the wedding feast, clawing at his throat.",
        expected_visual="Joffrey choking at the Purple Wedding feast",
        scene_query="Game of Thrones Joffrey choking Purple Wedding",
        required_entity="Joffrey Baratheon",
        required_kind="character",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    # Same surface shape, but no persisted downgrade provenance: a manual character contract is
    # not schema-migration authority and must remain byte-for-byte unchanged.
    manual_control = ScriptSegment(
        index=90,
        text="to the commander of the city watch.",
        expected_visual="to the commander of the city watch.",
        scene_query="",
        required_entity="Janos Slynt",
        required_kind="character",
        visual_policy=P.CHARACTER,
        is_specific_claim=False,
    )
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 3,
        "beats": {
            "15": {"branch": "grounded_exact", "reason": "named_subject_and_scene_action",
                   "cached_revalidation_schema": 3},
            "32": {"branch": "ungrounded_exact_downgrade",
                   "reason": "named_subject_without_exact_action",
                   "to_policy": P.CHARACTER, "cached_revalidation_schema": 3},
            "54": {"branch": "grounded_exact", "reason": "named_subject_and_scene_action",
                   "cached_revalidation_schema": 3},
            "70": {"branch": "grounded_exact", "reason": "named_subject_and_scene_action",
                   "cached_revalidation_schema": 3},
        },
        "breakout_provenance": {"15": "quote_derived", "54": "quote_derived"},
    })
    rows = [comparison, nominal, event, grounded_control, manual_control]

    first = A.revalidate_cached_directions(rows, analysis)

    assert first["exact_revalidated"] == 3
    assert first["cached_nominal_roles_migrated"] == 1
    assert first["cached_nominal_role_promises_cleared"] == 1
    assert first["changed_indices"] == [15, 32, 54]
    assert comparison.visual_policy == P.FILLER
    assert comparison.quote == "" and comparison.breakout_candidate is False
    assert nominal.visual_policy == P.FILLER
    assert nominal.required_entity == nominal.required_kind == nominal.scene_query == ""
    assert first["changes"]["32"]["changed_fields"] == [
        "visual_policy", "required_entity", "required_kind"]
    assert first["changes"]["32"]["guard"]["schema_migration"] == \
        "nominal_guessed_identity_contract_v5"
    assert event.visual_policy == P.EXACT
    assert event.expected_visual == event.scene_query == event.required_entity == \
        "The Purple Wedding"
    assert event.quote == "" and event.breakout_candidate is False
    assert grounded_control.visual_policy == P.EXACT
    assert manual_control.required_entity == "Janos Slynt"
    assert manual_control.scene_query == ""
    assert analysis.beat_grounding_audit["schema"] == 10

    saved = analysis.to_dict()
    reloaded = [ScriptSegment.from_dict(row.to_dict()) for row in rows]
    second = A.revalidate_cached_directions(
        reloaded, A.ScriptAnalysis.from_dict(saved))

    assert second["changed_count"] == 0
    assert second["exact_revalidated"] == 0
    assert second["cached_nominal_roles_migrated"] == 0
    assert second["cached_nominal_role_promises_cleared"] == 0


def test_schema5_corrects_cached_schema4_role_migration_and_preserves_provenance():
    beat = ScriptSegment(
        index=32,
        text="to the master-at-arms of the Red Keep.",
        expected_visual="to the master-at-arms of the Red Keep.",
        scene_query="master-at-arms",
        required_entity="master-at-arms",
        required_kind="character",
        visual_policy=P.CHARACTER,
        is_specific_claim=False,
    )
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 4,
        "beats": {
            "32": {
                "branch": "ungrounded_exact_downgrade",
                "reason": "named_subject_without_exact_action",
                "to_policy": P.CHARACTER,
                "grounded_subject_role": "master-at-arms",
                "required_entity_replaced": "Aron Santagar",
                "cached_revalidation_schema": 4,
                "schema_migration": "nominal_role_contract_v4",
            },
        },
    })

    first = A.revalidate_cached_directions([beat], analysis)

    assert first["changed_indices"] == [32]
    assert first["cached_nominal_role_promises_cleared"] == 1
    assert first["changes"]["32"]["changed_fields"] == [
        "visual_policy", "scene_query", "required_entity", "required_kind"]
    assert beat.visual_policy == P.FILLER
    assert beat.scene_query == beat.required_entity == beat.required_kind == ""
    marker = analysis.beat_grounding_audit["beats"]["32"]
    assert marker["reason"] == "verb_less_role_has_analyzer_guessed_identity"
    assert marker["analyzer_guessed_required_entity"] == "Aron Santagar"
    assert marker["schema_migration"] == "nominal_guessed_identity_contract_v5"
    assert marker["previous_schema_migration"] == "nominal_role_contract_v4"
    assert marker["cached_revalidation_schema"] == 10

    saved = analysis.to_dict()
    loaded = ScriptSegment.from_dict(beat.to_dict())
    loaded_analysis = A.ScriptAnalysis.from_dict(saved)
    second = A.revalidate_cached_directions([loaded], loaded_analysis)

    assert second["changed_count"] == 0
    assert second["cached_nominal_role_promises_cleared"] == 0
    resumed = loaded_analysis.to_dict()["beat_grounding_audit"]
    assert resumed["beats"] == saved["beat_grounding_audit"]["beats"]
    assert resumed["cached_revalidation"]["changed_indices"] == []
    assert resumed["last_material_revalidation"]["changed_indices"] == [32]


def test_schema8_migrates_cached_camera_contracts_once_and_preserves_material_diff():
    dagger = ScriptSegment(
        index=3,
        text="the weapon they left behind is the only thing she has.",
        expected_visual=("Close-up of the Valyrian steel dagger lying on the floor or in "
                         "Catelyn's hands after the attack"),
        scene_query="Game of Thrones catspaw Valyrian steel dagger left behind",
        required_entity="Valyrian steel dagger",
        required_kind="object",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    brothel = ScriptSegment(
        index=51,
        text="brothel because it could not be tested.",
        expected_visual=("Wide shot pulling back from the brothel, the private chamber where no "
                         "one can contest the story."),
        scene_query="Game of Thrones brothel wide shot interior Catelyn Littlefinger",
        required_entity="the brothel",
        required_kind="location",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 5,
        "beats": {
            "3": {"branch": "grounded_exact", "reason": "conservative_no_mismatch_proven",
                  "cached_revalidation_schema": 5},
            "51": {"branch": "grounded_exact", "reason": "named_subject_and_scene_action",
                   "cached_revalidation_schema": 5},
        },
    })

    first = A.revalidate_cached_directions([dagger, brothel], analysis)

    assert first["exact_revalidated"] == 2
    assert first["changed_indices"] == [3, 51]
    assert first["changes"]["3"]["changed_fields"] == ["expected_visual"]
    assert first["changes"]["51"]["changed_fields"] == [
        "expected_visual", "scene_query"]
    assert dagger.expected_visual == (
        "Exact scene: Game of Thrones catspaw Valyrian steel dagger left behind. "
        "Also show: Catelyn.")
    assert dagger.scene_query == "Game of Thrones catspaw Valyrian steel dagger left behind"
    assert brothel.expected_visual == \
        "Exact scene: Game of Thrones brothel interior Catelyn Littlefinger."
    assert brothel.scene_query == "Game of Thrones brothel interior Catelyn Littlefinger"
    for beat in (dagger, brothel):
        assert beat.visual_policy == P.EXACT
        assert beat.is_specific_claim is True
        assert beat._analyzer_grounding_guard["guard_schema"] == 10
        assert beat._analyzer_grounding_guard["cached_revalidation_schema"] == 10
    assert analysis.beat_grounding_audit["counts"][
        "unsupported_camera_composition_sanitized"] == 2

    saved = analysis.to_dict()
    loaded = [ScriptSegment.from_dict(row.to_dict()) for row in (dagger, brothel)]
    loaded_analysis = A.ScriptAnalysis.from_dict(saved)
    second = A.revalidate_cached_directions(loaded, loaded_analysis)

    assert second["changed_count"] == 0
    assert second["exact_revalidated"] == 0
    assert second["preserved_sanitized_provenance"] == 2
    assert loaded_analysis.beat_grounding_audit[
        "last_material_revalidation"]["changed_indices"] == [3, 51]


def test_schema8_preserves_effective_schema6_sanitizer_but_migrates_negative_existence():
    camera = ScriptSegment(
        index=3,
        text="the weapon they left behind is the only thing she has.",
        expected_visual=("Exact scene: Game of Thrones catspaw Valyrian steel dagger left "
                         "behind. Also show: Catelyn."),
        scene_query="Game of Thrones catspaw Valyrian steel dagger left behind",
        required_entity="Valyrian steel dagger",
        required_kind="object",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    negative = ScriptSegment(
        index=41,
        text="There are no witnesses named.",
        expected_visual="Wide shot of the brothel with only Catelyn and Littlefinger present",
        scene_query="Game of Thrones brothel interior Catelyn Littlefinger alone",
        required_entity="the brothel",
        required_kind="location",
        visual_policy=P.EXACT,
        is_specific_claim=True,
    )
    camera_marker = {
        "branch": "grounded_exact_sanitized",
        "reason": "unsupported_camera_composition_removed",
        "sanitized_fields": ["expected_visual"],
        "sanitized_values": {"expected_visual": camera.expected_visual},
        "guard_schema": 6,
        "cached_revalidation_schema": 6,
    }
    negative_marker = {
        "branch": "grounded_exact_sanitized",
        "reason": "unsupported_camera_composition_removed",
        "sanitized_fields": ["expected_visual"],
        "sanitized_values": {"expected_visual": negative.expected_visual},
        "guard_schema": 6,
        "cached_revalidation_schema": 6,
    }
    analysis = A.ScriptAnalysis(beat_grounding_audit={
        "schema": 6,
        "beats": {"3": camera_marker, "41": negative_marker},
    })

    result = A.revalidate_cached_directions([camera, negative], analysis)

    assert result["changed_indices"] == [41]
    assert result["preserved_sanitized_provenance"] == 1
    assert camera.visual_policy == P.EXACT
    assert camera._analyzer_grounding_guard["reason"] == \
        "unsupported_camera_composition_removed"
    assert camera._analyzer_grounding_guard["guard_schema"] == 10
    assert camera._analyzer_grounding_guard["cached_revalidation_schema"] == 10
    assert negative.visual_policy == P.ABSTRACT
    assert negative.scene_query == negative.required_entity == negative.required_kind == ""
    assert negative._analyzer_grounding_guard["reason"] == \
        "negative_existence_is_not_an_observable_exact_scene"
    assert negative._analyzer_grounding_guard["cached_revalidation_schema"] == 10
