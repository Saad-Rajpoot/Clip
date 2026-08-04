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
            "the master-at-arms of the Red Keep.",
            {
                "expected_visual": "Close up of the master-at-arms examining the dagger",
                "scene_query": "Game of Thrones Rodrik master-at-arms dagger question",
                "required_entity": "master-at-arms",
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
        "adjacent_quote_copy_sanitized": 0,
        "downgraded": 1,
        "to_character_specific": 0,
        "to_generic_filler": 1,
    }
    assert analysis.beat_grounding_audit["schema"] == 3
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
    assert marker["cached_revalidation_schema"] == 3
    saved_audit = loaded_analysis.to_dict()["beat_grounding_audit"]

    loaded_again = ScriptSegment.from_dict(loaded.to_dict())
    analysis_again = A.ScriptAnalysis.from_dict(loaded_analysis.to_dict())
    second = A.revalidate_cached_directions([loaded_again], analysis_again)

    assert second["changed_count"] == 0
    assert analysis_again.to_dict()["beat_grounding_audit"] == saved_audit
