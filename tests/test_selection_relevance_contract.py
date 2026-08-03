"""Fail-closed pre-render semantic contract for resume/rerender/verify-disabled paths."""
from __future__ import annotations

import inspect
import json

import pytest

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import index as IX
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo
from vidlore.clipstudio.verify import NonRetryableBuildError


GOOD = {
    "status": "ok", "verdict": "keep", "matches_narration": True,
    "specific_enough": True, "correct_subject_visible": True,
    "wrong_subject_visible": False, "contradicts_narration": False,
    "quality_ok": True,
}


def _stamp_current_asr_provenance(proj, sid):
    meta = {
        "schema": IX.INDEX_SCHEMA,
        "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, load_clip_config()),
    }
    (proj.index_dir / f"{sid}.index.meta.json").write_text(json.dumps(meta))


def _fixture(tmp_path, *, policy=P.EXACT, verifier=None, text="Jaime rides Ned down.",
             entity="Jaime Lannister", kind="character", title="Game of Thrones Jaime vs Ned",
             quote="", signals=None):
    proj = ClipProject(name="semantic", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"video")
    proj.sources = [SourceVideo(id="s1", url="u", title=title, permission="owner",
                                status="ok", local_path=str(media))]
    frame = tmp_path / "shot_0000.jpg"
    frame.write_bytes(b"representative-frame")
    shot = Shot(source_id="s1", index=0, start=0.0, end=2.0,
                keyframe_path=str(frame))
    proj.shots_path("s1").write_text(json.dumps([shot.to_dict()]))
    proj.meta["analysis"] = {"video_type": "multi_scene", "characters": [], "actors": []}
    _stamp_current_asr_provenance(proj, "s1")
    seg = ScriptSegment(index=0, text=text, required_entity=entity, required_kind=kind,
                        visual_policy=policy, is_specific_claim=(policy == P.EXACT), quote=quote)
    verdict = dict(GOOD if verifier is None else verifier)
    sel = ClipSelection(segment_index=0, source_id="s1", shot_index=0,
                        in_point=0.0, out_point=2.0, confidence=0.8,
                        verifier=verdict, signals=dict(signals or {}))
    V.bind_selection_verifier_evidence(
        proj, sel, seg, verdict, shot=shot, model="vision-test",
        is_specific=P.verify_strict(seg), multiframe=True, faceid_names=[],
        era=V._project_beat_era(proj, seg), must_see=P.deictic_target(seg))
    proj.selections = [sel]
    return proj, seg, sel


def _add_indexed_source(proj, root, sid, *, title, words, transcripts=None):
    """Add one tiny indexed source whose ASR/source-audio eligibility is controlled by the test."""
    media = root / f"{sid}.mp4"
    media.write_bytes(b"video")
    proj.sources.append(SourceVideo(
        id=sid, url=f"u-{sid}", title=title, permission="owner",
        status="ok", local_path=str(media)))
    rows = []
    for i, transcript in enumerate(transcripts or [""]):
        frame = root / f"{sid}_{i}.jpg"
        frame.write_bytes(f"frame-{sid}-{i}".encode())
        rows.append(Shot(
            source_id=sid, index=i, start=float(i), end=float(i + 1),
            keyframe_path=str(frame), transcript=transcript).to_dict())
    proj.shots_path(sid).write_text(json.dumps(rows))
    (proj.index_dir / f"{sid}.words.json").write_text(json.dumps(words))
    _stamp_current_asr_provenance(proj, sid)


def _block_reasons(proj, seg):
    return R.evaluate_selection_relevance(proj, [seg])["blockers"][0]["reasons"]


def _strict_still_meta(still, *, source="source-frame", exact=True, evidence=None):
    ev = dict(GOOD if evidence is None else evidence)
    return {
        "source": source,
        "relevance_class": "exact_scene" if exact else "contextual_fallback",
        "still_verified": True,
        "still_semantic_verified": True,
        "still_verifier": ev,
        "still_image_sha256": R.image_sha256(still),
        "exact_still_verified": bool(exact),
        "exact_still_verifier": (ev if exact else {}),
    }


def test_exact_positive_selection_passes_and_generic_negative_is_untouched(tmp_path):
    proj, seg, _ = _fixture(tmp_path / "exact")
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass" and audit["strict_checked"] == 1

    generic, gseg, gsel = _fixture(tmp_path / "generic", policy=P.FILLER,
                                   verifier={"status": "error", "verdict": "replace"},
                                   entity="", kind="")
    gseg.is_specific_claim = False
    ga = R.evaluate_selection_relevance(generic, [gseg])
    assert ga["status"] == "pass" and ga["strict_checked"] == 0
    assert ga["generic_or_abstract_skipped"] == 1


@pytest.mark.parametrize(("patch", "reason"), [
    ({"status": "error"}, "verifier_error"),
    ({"verdict": "replace"}, "verdict_replace"),
    ({"matches_narration": False}, "matches_narration_false"),
    ({"specific_enough": False}, "specific_enough_false"),
    ({"wrong_subject_visible": True}, "wrong_subject_visible_true"),
    ({"contradicts_narration": True}, "contradicts_narration_true"),
    ({"quality_ok": False}, "quality_ok_false"),
    ({"correct_subject_visible": False}, "correct_subject_visible_false"),
    ({"era_ok": False}, "era_ok_false"),
])
def test_every_explicit_negative_blocks_exact_publication(tmp_path, patch, reason):
    verdict = {**GOOD, **patch}
    proj, seg, _ = _fixture(tmp_path, verifier=verdict)
    assert reason in _block_reasons(proj, seg)


def test_absent_selection_and_absent_verifier_fields_fail_closed(tmp_path):
    proj, seg, _ = _fixture(tmp_path / "missing_selection")
    proj.selections = []
    assert "selection_absent" in _block_reasons(proj, seg)

    proj2, seg2, _ = _fixture(
        tmp_path / "no_verify", verifier={"status": "ok", "verdict": "keep"})
    reasons = _block_reasons(proj2, seg2)
    assert "matches_narration_absent" in reasons
    assert "specific_enough_absent" in reasons
    assert "quality_ok_absent" in reasons
    assert "wrong_subject_visible_absent" in reasons


def test_legacy_positive_verdict_without_selection_binding_fails_closed(tmp_path):
    proj, seg, sel = _fixture(tmp_path)
    sel.verifier.pop("selection_evidence")
    assert "verifier_evidence_absent" in _block_reasons(proj, seg)


def test_exact_authored_quote_requires_strong_timed_asr_at_selected_window(tmp_path):
    quote = "You don't think I'd let you marry that beast, do you?"
    proj, seg, sel = _fixture(
        tmp_path, text=f'"{quote}" That single line is the whole confession.', quote=quote,
        entity="Olenna Tyrell", signals={"dialogue": 0.965, "moment_lock": 0.965})
    # Timed source ASR independently proves the authored words occur inside the aired trim.
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.20, 0.35, "You"], [0.35, 0.50, "don't"], [0.50, 0.65, "think"],
        [0.65, 0.80, "I'd"], [0.80, 0.95, "let"], [0.95, 1.10, "you"],
        [1.10, 1.22, "marry"], [1.22, 1.34, "that"], [1.34, 1.50, "beast"],
        [1.50, 1.62, "do"], [1.62, 1.74, "you"],
    ]))
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass"
    assert audit["checked"][0]["quote_evidence"]["dialogue_signal"] == 0.965
    assert audit["checked"][0]["quote_evidence"]["branch"] == "verbatim"
    assert audit["checked"][0]["quote_evidence"]["verbatim_required"] is True
    assert audit["quote_branch_counts"] == {
        "verbatim": 1, "paraphrase": 0, "indeterminate": 0}

    # The audited false pass: right face, wrong season/scene, and no authored dialogue evidence.
    sel.signals["dialogue"] = 0.0
    assert "exact_quote_dialogue_signal_below_floor" in _block_reasons(proj, seg)

    # A phrase elsewhere in the same source is not evidence for this selected trim.
    sel.signals["dialogue"] = 0.965
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [10.0 + i * .1, 10.1 + i * .1, w]
        for i, w in enumerate("You don't think I'd let you marry that beast do you".split())
    ]))
    assert "exact_quote_timed_asr_outside_selected_window" in _block_reasons(proj, seg)


def _containment_quote_fixture(tmp_path):
    quote = "You don't think I'd let you marry that beast, do you?"
    proj, seg, sel = _fixture(
        tmp_path, text=f'"{quote}" That single line is the whole confession.', quote=quote,
        entity="Olenna Tyrell", signals={"dialogue": 0.965, "moment_lock": 0.965})
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.20, 0.35, "You"], [0.35, 0.50, "don't"], [0.50, 0.65, "think"],
        [0.65, 0.80, "I'd"], [0.80, 0.95, "let"], [0.95, 1.10, "you"],
        [1.10, 1.22, "marry"], [1.22, 1.34, "that"], [1.34, 1.50, "beast"],
        [1.50, 1.62, "do"], [1.62, 1.74, "you"],
    ]))
    contract = R._quote_pool_branches(proj, [seg])[0]
    assert contract["branch"] == "verbatim"
    return proj, seg, sel, contract


@pytest.mark.parametrize(("selected_window", "old_interval_gap"), [
    ((1.70, 3.70), 0.0),   # only 40 ms overlaps; the quote starts 1.5 s before the trim
    ((2.20, 4.20), 0.46),  # the entire quote is outside, but the old 0.75 s gap test passed
])
def test_quote_span_must_be_contained_not_merely_near_or_touching(
        tmp_path, selected_window, old_interval_gap):
    proj, seg, sel, contract = _containment_quote_fixture(tmp_path)
    sel.in_point, sel.out_point = selected_window

    ok, reason, detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=contract)

    assert ok is False
    assert reason == "exact_quote_timed_asr_outside_selected_window"
    assert detail["timed_asr_span"] == [0.2, 1.74]
    assert detail["selected_window"] == list(selected_window)
    assert detail["window_gap_sec"] == old_interval_gap
    assert detail["window_tolerance_sec"] == R.QUOTE_WINDOW_TOLERANCE_SEC
    assert detail["start_containment_margin_sec"] < 0.0


def test_quote_span_fully_inside_tolerated_window_boundary_passes(tmp_path):
    proj, seg, sel, contract = _containment_quote_fixture(tmp_path)
    # Expanded by ±0.75, this 0.90–1.10 selection contains the complete 0.20–1.74 ASR span.
    sel.in_point, sel.out_point = 0.90, 1.10

    ok, reason, detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=contract)

    assert ok is True and reason == ""
    assert detail["quote_start_vs_window_start_sec"] == -0.7
    assert detail["quote_end_vs_window_end_sec"] == 0.64
    assert detail["start_containment_margin_sec"] == 0.05
    assert detail["end_containment_margin_sec"] == 0.11


def test_character_policy_quote_still_gets_complete_pool_branch(tmp_path):
    """A montage may be CHARACTER while retaining an authored quote; typing cannot disappear."""
    quote = "Chaos isn't a pit. Chaos is a ladder."
    proj, seg, _sel = _fixture(
        tmp_path, policy=P.CHARACTER,
        text="She makes the identification three separate times.", quote=quote,
        entity="Varys, Petyr Baelish, Tyrion Lannister", kind="montage")
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word] for i, word in enumerate(words)
    ]))

    audit = R.evaluate_selection_relevance(proj, [seg])

    assert P.policy_of(seg) == P.CHARACTER
    assert audit["checked"][0]["quote_evidence"]["branch"] == "verbatim"
    assert "exact_quote_dialogue_signal_below_floor" in audit["blockers"][0]["reasons"]
    assert audit["quote_branch_counts"] == {
        "verbatim": 1, "paraphrase": 0, "indeterminate": 0}


def test_pool_absent_quote_is_paraphrase_and_skips_only_verbatim_floor(tmp_path):
    quote = "Who does this belong to?"
    proj, seg, _sel = _fixture(
        tmp_path, text=quote, quote=quote, signals={"dialogue": 0.0})
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1, 0.2, "A"], [0.2, 0.3, "different"], [0.3, 0.4, "line"],
    ]))
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass", "ordinary positive semantic evidence still decides the beat"
    ev = audit["checked"][0]["quote_evidence"]
    assert ev["branch"] == "paraphrase" and ev["verbatim_required"] is False
    assert ev["pool_match"] is None
    assert ev["dialogue_signal"] == 0.0


def test_quote_found_elsewhere_in_eligible_pool_retains_verbatim_floor(tmp_path):
    quote = "I did warn you not to trust me."
    proj, seg, _sel = _fixture(
        tmp_path, text="The betrayal lands.", quote=quote, signals={"dialogue": 0.0})
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1, 0.2, "unrelated"], [0.2, 0.3, "selected"], [0.3, 0.4, "audio"],
    ]))
    words = [[2.0 + i * .1, 2.1 + i * .1, w]
             for i, w in enumerate("I did warn you not to trust me".split())]
    _add_indexed_source(
        proj, tmp_path, "s2", title="Game of Thrones betrayal scene", words=words)
    audit = R.evaluate_selection_relevance(proj, [seg])
    entry = audit["blockers"][0]
    assert "exact_quote_dialogue_signal_below_floor" in entry["reasons"]
    assert entry["quote_evidence"]["branch"] == "verbatim"
    assert entry["quote_evidence"]["pool_match"]["source_id"] == "s2"


def test_commentary_pool_hit_does_not_turn_a_paraphrase_into_verbatim(tmp_path):
    """Beat-8 regression: wall-to-wall Season-7 News narration fuzzy-matched the authored hint."""
    quote = "It belongs to Tyrion Lannister."
    proj, seg, _sel = _fixture(
        tmp_path, text="That name is Tyrion Lannister.", quote=quote,
        signals={"dialogue": 0.0})
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1, 0.2, "unrelated"], [0.2, 0.3, "selected"], [0.3, 0.4, "audio"],
    ]))
    words = [[2.0 + i * .1, 2.1 + i * .1, w]
             for i, w in enumerate("It belongs to Tyrion Lannister".split())]
    rich = ["the narrator keeps explaining this story every single second"] * 8
    _add_indexed_source(
        proj, tmp_path, "news", title="Game of Thrones Season 7 News", words=words,
        transcripts=rich)
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass"
    ev = audit["checked"][0]["quote_evidence"]
    assert ev["branch"] == "paraphrase"
    assert ev["commentary_sources_excluded"] == 1


def test_pool_quote_scan_rejects_one_name_function_word_false_match(tmp_path):
    quote = "Lady Stark is here in King's Landing."
    proj, seg, _sel = _fixture(
        tmp_path, text="Varys learns she is in the city.", quote=quote,
        signals={"dialogue": 0.0})
    # This measured ASR phrase used to score 0.727 from one shared name plus a function-word
    # skeleton.  The whole-pool locator runs at its 0.72 discovery bar, but substantive phrase
    # evidence must still keep this paraphrase from becoming a verbatim promise.
    words = [[i * .1, (i + 1) * .1, w] for i, w in enumerate(
        "Lord Edward Stark is here in named Protector of the Realm".split())]
    (proj.index_dir / "s1.words.json").write_text(json.dumps(words))
    assert R._quote_pool_branches(proj, [seg])[0]["branch"] == "paraphrase"


def test_pool_scan_types_hotword_repaired_real_quote_without_lowering_floor(tmp_path):
    quote = "Tell Cersei. I want her to know it was me."
    proj, seg, _sel = _fixture(
        tmp_path, text="Olenna confesses before dying.", quote=quote,
        signals={"dialogue": 0.77})
    words = [[i * .1, (i + 1) * .1, w] for i, w in enumerate(
        "Tell Cersei I wanted to know he was me".split())]
    (proj.index_dir / "s1.words.json").write_text(json.dumps(words))

    entry = R.evaluate_selection_relevance(proj, [seg])["blockers"][0]
    ev = entry["quote_evidence"]
    assert ev["branch"] == "verbatim"
    assert ev["scan_ratio_floor"] == 0.78
    assert ev["selected_window_ratio_floor"] == 0.78
    assert "exact_quote_dialogue_signal_below_floor" in entry["reasons"]


@pytest.mark.parametrize(("metadata", "provenance_reason"), [
    (None, "index_metadata_missing"),
    ({"asr_prompt_fingerprint": "stale"}, "asr_prompt_fingerprint_mismatch"),
    ("not-json", "index_metadata_unreadable"),
])
def test_stale_or_missing_pool_asr_provenance_is_indeterminate(
        tmp_path, metadata, provenance_reason):
    """A matching word cache cannot type its own quote unless its ASR inputs are current."""
    quote = "Chaos isn't a pit. Chaos is a ladder."
    proj, seg, _sel = _fixture(
        tmp_path, text="Baelish explains his philosophy.", quote=quote,
        signals={"dialogue": 1.0})
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word] for i, word in enumerate(words)
    ]))
    meta_path = proj.index_dir / "s1.index.meta.json"
    if metadata is None:
        meta_path.unlink()
    elif isinstance(metadata, str):
        meta_path.write_text(metadata)
    else:
        meta_path.write_text(json.dumps(metadata))

    audit = R.evaluate_selection_relevance(proj, [seg])

    entry = audit["blockers"][0]
    evidence = entry["quote_evidence"]
    assert evidence["branch"] == "indeterminate"
    assert evidence["pool_match"] is None
    assert evidence["asr_provenance_invalid_source_count"] == 1
    assert evidence["asr_provenance_invalid_sources"] == [{
        "source_id": "s1",
        "reason": provenance_reason,
        "actual_asr_prompt_fingerprint": "stale" if metadata == {
            "asr_prompt_fingerprint": "stale"} else "",
    }]
    assert "exact_quote_pool_classification_indeterminate" in entry["reasons"]


def test_project_asr_identity_change_invalidates_previously_current_pool_cache(tmp_path):
    quote = "Tell Cersei. I want her to know it was me."
    proj, seg, _sel = _fixture(
        tmp_path, text="Olenna confesses before dying.", quote=quote,
        signals={"dialogue": 1.0})
    words = "Tell Cersei I want her to know it was me".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word] for i, word in enumerate(words)
    ]))
    old_fingerprint = json.loads(
        (proj.index_dir / "s1.index.meta.json").read_text())["asr_prompt_fingerprint"]
    proj.meta["analysis"]["characters"] = [{"name": "Cersei Lannister"}]

    evidence = R._quote_pool_branches(proj, [seg])[0]

    assert evidence["branch"] == "indeterminate"
    assert evidence["asr_prompt_fingerprint_expected"] != old_fingerprint
    assert evidence["asr_provenance_invalid_sources"][0]["reason"] == \
        "asr_prompt_fingerprint_mismatch"


def test_current_metadata_cannot_certify_a_corrupt_words_payload(tmp_path):
    quote = "Tell Cersei. I want her to know it was me."
    proj, seg, _sel = _fixture(
        tmp_path, text="Olenna confesses before dying.", quote=quote,
        signals={"dialogue": 1.0})
    (proj.index_dir / "s1.words.json").write_text("{corrupt")

    evidence = R._quote_pool_branches(proj, [seg])[0]

    assert evidence["branch"] == "indeterminate"
    assert evidence["asr_provenance_invalid_sources"] == [{
        "source_id": "s1",
        "reason": "words_cache_invalid_or_missing",
        "actual_asr_prompt_fingerprint": "",
    }]


@pytest.mark.parametrize("shots_payload", [None, "{corrupt", "[]"])
def test_missing_corrupt_or_empty_shots_make_quote_typing_indeterminate(
        tmp_path, shots_payload):
    quote = "Chaos isn't a pit. Chaos is a ladder."
    proj, seg, _sel = _fixture(
        tmp_path, text="Baelish explains his philosophy.", quote=quote,
        signals={"dialogue": 1.0})
    words = "Chaos isn't a pit Chaos is a ladder".split()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word] for i, word in enumerate(words)
    ]))
    shots_path = proj.shots_path("s1")
    if shots_payload is None:
        shots_path.unlink()
    else:
        shots_path.write_text(shots_payload)

    evidence = R._quote_pool_branches(proj, [seg])[0]

    assert evidence["branch"] == "indeterminate"
    assert evidence["pool_match"] is None
    assert evidence["asr_provenance_invalid_sources"] == [{
        "source_id": "s1",
        "reason": "shots_cache_invalid_or_missing",
        "actual_asr_prompt_fingerprint": "",
    }]


def test_title_rejected_commentary_needs_no_shot_cache_to_be_excluded(tmp_path):
    quote = "Chaos isn't a pit. Chaos is a ladder."
    proj, seg, _sel = _fixture(
        tmp_path, title="Why Chaos Is A Ladder - Video Essay",
        text="Baelish explains his philosophy.", quote=quote,
        signals={"dialogue": 1.0})
    proj.shots_path("s1").unlink()
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, word]
        for i, word in enumerate("Chaos isn't a pit Chaos is a ladder".split())
    ]))
    _add_indexed_source(
        proj, tmp_path, "clean", title="Game of Thrones scene",
        words=[[0.0, 0.2, "unrelated"]])

    evidence = R._quote_pool_branches(proj, [seg])[0]

    assert evidence["branch"] == "paraphrase"
    assert evidence["commentary_sources_excluded"] == 1
    assert evidence["asr_provenance_invalid_source_count"] == 0


def test_verbatim_pool_hit_cannot_let_stale_selected_asr_pass(tmp_path):
    """A current second source may type the quote, but selected-window proof is source-bound."""
    quote = "I did warn you not to trust me."
    proj, seg, _sel = _fixture(
        tmp_path, text="The betrayal lands.", quote=quote,
        signals={"dialogue": 1.0})
    matching = [[0.1 + i * .1, 0.2 + i * .1, word]
                for i, word in enumerate("I did warn you not to trust me".split())]
    (proj.index_dir / "s1.words.json").write_text(json.dumps(matching))
    stale = json.loads((proj.index_dir / "s1.index.meta.json").read_text())
    stale["asr_prompt_fingerprint"] = "stale-selected-source"
    (proj.index_dir / "s1.index.meta.json").write_text(json.dumps(stale))
    _add_indexed_source(
        proj, tmp_path, "s2", title="Game of Thrones betrayal scene", words=matching)

    audit = R.evaluate_selection_relevance(proj, [seg])

    entry = audit["blockers"][0]
    evidence = entry["quote_evidence"]
    assert evidence["branch"] == "verbatim"
    assert evidence["pool_match"]["source_id"] == "s2"
    assert evidence["selected_asr_provenance_status"] == \
        "asr_prompt_fingerprint_mismatch"
    assert "exact_quote_selected_asr_provenance_invalid" in entry["reasons"]


def test_no_dialogue_eligible_index_is_indeterminate_and_fails_closed(tmp_path):
    quote = "I did warn you not to trust me."
    proj, seg, _sel = _fixture(
        tmp_path, text="The betrayal lands.", quote=quote, signals={"dialogue": 0.0})
    audit = R.evaluate_selection_relevance(proj, [seg])
    entry = audit["blockers"][0]
    assert entry["quote_evidence"]["branch"] == "indeterminate"
    assert "exact_quote_pool_classification_indeterminate" in entry["reasons"]
    assert audit["quote_branch_counts"]["indeterminate"] == 1

    # Even a strong selected-source ASR hit cannot type its own quote after the pool has declared
    # that source ineligible/unknown.  Indeterminate is a contract failure, not a fallback path.
    _sel = proj.selections[0]
    _sel.signals["dialogue"] = 0.99
    proj.sources[0].title = "Why Game of Thrones betrayal was secretly genius — analysis"
    (proj.index_dir / "s1.words.json").write_text(json.dumps([
        [0.1 + i * .1, 0.2 + i * .1, w]
        for i, w in enumerate("I did warn you not to trust me".split())
    ]))
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert "exact_quote_pool_classification_indeterminate" in audit["blockers"][0]["reasons"]


def test_strict_exact_still_is_allowed_when_authored_quote_has_no_moving_asr(tmp_path):
    quote = "You don't think I'd let you marry that beast, do you?"
    proj, seg, sel = _fixture(
        tmp_path, text=quote, quote=quote, entity="Olenna Tyrell", signals={"dialogue": 0.0})
    still = tmp_path / "exact-confession-frame.jpg"
    still.write_bytes(b"strict-exact-confession-pixels")
    sel.image_path = str(still)
    sel.image_meta = _strict_still_meta(still, source="web-exact-scene")
    assert R.evaluate_selection_relevance(proj, [seg])["status"] == "pass"


@pytest.mark.parametrize("mode", ["lenient_question", "downgraded", "contextual_class"])
def test_exact_moving_publication_rejects_lenient_or_contextual_evidence(tmp_path, mode):
    proj, seg, sel = _fixture(tmp_path)
    if mode == "lenient_question":
        shot = V._shot_lookup(proj)("s1", 0)
        V.bind_selection_verifier_evidence(
            proj, sel, seg, sel.verifier, shot=shot, model="vision-test",
            is_specific=False, multiframe=True, faceid_names=[],
            era=V._project_beat_era(proj, seg), must_see="")
        expected = "exact_verifier_evidence_not_strict"
    elif mode == "downgraded":
        sel.verifier["downgraded"] = "exact→venue_contextual"
        expected = "exact_moving_verdict_was_downgraded"
    else:
        sel.verifier["relevance_class"] = "contextual_fallback"
        expected = "exact_moving_relevance_is_contextual"
    assert expected in _block_reasons(proj, seg)


@pytest.mark.parametrize("mutation", ["window", "source_bytes", "faceids", "beat_prompt"])
def test_moving_verifier_binding_rejects_post_verify_selection_or_content_mutation(
        tmp_path, mutation):
    proj, seg, sel = _fixture(tmp_path)
    assert R.evaluate_selection_relevance(proj, [seg])["status"] == "pass"
    if mutation == "window":
        sel.in_point = 0.125
    elif mutation == "source_bytes":
        (tmp_path / "source.mp4").write_bytes(b"different source content after verification")
    elif mutation == "faceids":
        rows = json.loads(proj.shots_path("s1").read_text())
        rows[0]["face_ids"] = ["different actor"]
        proj.shots_path("s1").write_text(json.dumps(rows))
    else:
        seg.expected_visual = "a materially different exact scene prompt"
    reasons = _block_reasons(proj, seg)
    assert "verifier_evidence_mismatch" in reasons


def test_character_specific_is_concrete_but_abstract_stays_lenient(tmp_path):
    bad = {**GOOD, "matches_narration": False}
    proj, seg, _ = _fixture(tmp_path / "character", policy=P.CHARACTER, verifier=bad)
    assert R.evaluate_selection_relevance(proj, [seg])["status"] == "blocked"

    abstract, aseg, _ = _fixture(tmp_path / "abstract", policy=P.ABSTRACT, verifier=bad,
                                 entity="", kind="")
    aseg.is_specific_claim = False
    assert R.evaluate_selection_relevance(abstract, [aseg])["status"] == "pass"


def test_verified_still_suppresses_rejected_video_but_unverified_still_does_not(tmp_path):
    rejected = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(tmp_path / "verified", verifier=rejected)
    still = tmp_path / "verified" / "still.jpg"
    still.write_bytes(b"jpeg")
    sel.image_path = str(still)
    # A provider/source label is not evidence about the image pixels.
    sel.image_meta = {"source": "web-exact-scene", "relevance_class": "exact_scene"}
    assert R.evaluate_selection_relevance(proj, [seg])["status"] == "blocked"

    sel.image_meta = _strict_still_meta(still, source="web-exact-scene")
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass"
    assert audit["checked"][0]["coverage"] == "verified_still:web-exact-scene"

    proj2, seg2, sel2 = _fixture(tmp_path / "unverified", verifier=rejected)
    still2 = tmp_path / "unverified" / "still.jpg"
    still2.write_bytes(b"jpeg")
    sel2.image_path = str(still2)
    sel2.image_meta = {"source": "source-frame-recovery", "still_verified": True}
    reasons = _block_reasons(proj2, seg2)
    assert any(r.startswith("invalid_still:") for r in reasons)


def test_exact_raven_cannot_reuse_old_generic_source_frame(tmp_path):
    """Regression: policy promotion must not turn an old filler sword still into raven proof."""
    rejected = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(
        tmp_path, verifier=rejected, text="A raven takes weeks to reach the capital.",
        entity="raven", kind="animal")
    still = tmp_path / "still.jpg"
    still.write_bytes(b"jpeg")
    sel.image_path = str(still)
    sel.image_meta = {
        "source": "source-frame", "relevance_class": "generic_filler",
        "still_verified": True,
    }
    reasons = _block_reasons(proj, seg)
    assert any("relevance_class" in r for r in reasons)


def test_exact_source_frame_requires_complete_strict_evidence(tmp_path):
    rejected = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(tmp_path, verifier=rejected)
    still = tmp_path / "still.jpg"
    still.write_bytes(b"jpeg")
    sel.image_path = str(still)
    sel.image_meta = {
        "source": "source-frame", "relevance_class": "exact_scene",
        "still_verified": True,
    }
    assert any("strict actual-image" in r for r in _block_reasons(proj, seg))

    sel.image_meta.update({
        "still_semantic_verified": True,
        "still_verifier": dict(GOOD),
        "still_image_sha256": R.image_sha256(still),
        "exact_still_verified": True,
        "exact_still_verifier": dict(GOOD),
    })
    audit = R.evaluate_selection_relevance(proj, [seg])
    assert audit["status"] == "pass"
    assert audit["checked"][0]["coverage"] == "verified_still:source-frame"

    sel.image_meta["exact_still_verifier"]["specific_enough"] = False
    sel.image_meta["still_verifier"]["specific_enough"] = False
    assert any("still specific_enough" in r for r in _block_reasons(proj, seg))


def test_web_exact_label_with_wrong_or_missing_pixel_verdict_never_airs(tmp_path):
    rejected = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(tmp_path, verifier=rejected,
                              text="Watch the chalice itself.", entity="chalice", kind="object")
    still = tmp_path / "scene_img_002.jpg"
    still.write_bytes(b"joffrey-kissing-margaery-no-chalice")
    sel.image_path = str(still)
    sel.image_meta = {"source": "web-exact-scene", "relevance_class": "exact_scene"}
    assert any("strict actual-image" in r for r in _block_reasons(proj, seg))

    bad_image_verdict = {**GOOD, "matches_narration": False, "specific_enough": False,
                         "correct_subject_visible": False}
    sel.image_meta = _strict_still_meta(
        still, source="web-exact-scene", evidence=bad_image_verdict)
    assert any("still matches_narration" in r for r in _block_reasons(proj, seg))


def test_character_concrete_still_needs_bound_positive_pixel_evidence(tmp_path):
    rejected = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(
        tmp_path, policy=P.CHARACTER, verifier=rejected,
        text="The chalice mattered to the conspiracy.", entity="chalice", kind="object")
    # Direct-visual object promotion is covered separately; this fixture isolates CHARACTER stills.
    seg.is_specific_claim = False
    still = tmp_path / "wedding_table_without_clear_cup.jpg"
    still.write_bytes(b"margaery-joffrey-table")
    sel.image_path = str(still)
    sel.image_meta = {"source": "source-frame", "relevance_class": "contextual_fallback",
                      "still_verified": True}
    assert any("strict actual-image" in r for r in _block_reasons(proj, seg))

    sel.image_meta = _strict_still_meta(still, exact=False)
    assert R.evaluate_selection_relevance(proj, [seg])["status"] == "pass"
    still.write_bytes(b"different-pixels-after-verification")
    assert any("actual image bytes" in r for r in _block_reasons(proj, seg))


def test_ledger_never_reports_an_unproved_concrete_still_as_clean(tmp_path):
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio import ledger
    from vidlore.clipstudio.models import FLAG_EXACT_MISSING, FLAG_VERIFIER_FAILED

    bad = {**GOOD, "verdict": "replace", "matches_narration": False}
    proj, seg, sel = _fixture(tmp_path / "exact", verifier=bad)
    still = tmp_path / "exact" / "still.jpg"
    still.write_bytes(b"pixels")
    sel.image_path = str(still)
    sel.image_meta = {"source": "web-exact-scene", "relevance_class": "exact_scene"}
    assert ledger.evaluate_flags(sel, seg, proj.source("s1"), ClipConfig()) == [FLAG_EXACT_MISSING]

    cproj, cseg, csel = _fixture(
        tmp_path / "character", policy=P.CHARACTER, verifier=bad,
        text="The chalice mattered to the conspiracy.", entity="chalice", kind="object")
    cstill = tmp_path / "character" / "still.jpg"
    cstill.write_bytes(b"pixels")
    csel.image_path = str(cstill)
    csel.image_meta = {"source": "source-frame", "relevance_class": "contextual_fallback",
                       "still_verified": True}
    assert ledger.evaluate_flags(
        csel, cseg, cproj.source("s1"), ClipConfig()) == [FLAG_VERIFIER_FAILED]


def test_exact_contextual_source_frame_remains_eligible_for_web_recovery():
    from vidlore.clipstudio import orchestrate as O
    src = inspect.getsource(O._fill_image_fallbacks)
    web_pass = src[src.index("# ---- PASS 2"):src.index("# ---- PASS 3")]
    assert "_strict_img_confirmed" in web_pass
    assert "_verified_still_coverage(sel, seg)[0]" in web_pass
    assert "_strict_policy and _has_img and not _strict_img_confirmed" in web_pass


def test_narrow_deterministic_contradictions_block_legacy_verdicts(tmp_path):
    # No contradicts_narration field: this simulates the legacy prompt used by the audited job.
    legacy_good = {k: v for k, v in GOOD.items() if k != "contradicts_narration"}

    absent, aseg, _ = _fixture(
        tmp_path / "absence", verifier=legacy_good,
        text="and Baelish is not even in the room.", entity="Petyr Baelish")
    assert "deterministic_contradiction" in _block_reasons(absent, aseg)

    death, dseg, _ = _fixture(
        tmp_path / "death", verifier=legacy_good,
        text="Jon Arryn the Hand of the King dies of a sudden illness",
        entity="Jon Arryn's body", title="Game of Thrones - King Joffrey's Death")
    death.meta["analysis"] = {
        "characters": [{"name": "Joffrey Baratheon", "actor": "Jack Gleeson"}]}
    assert "deterministic_contradiction" in _block_reasons(death, dseg)


def test_audit_is_atomic_and_block_is_nonretryable(tmp_path):
    good, gseg, _ = _fixture(tmp_path / "good")
    dest = good.output_dir / R.AUDIT_FILENAME
    result = R.assert_selection_relevance(good, [gseg], dest)
    assert result["status"] == "pass"
    assert json.loads(dest.read_text())["status"] == "pass"
    assert not dest.with_suffix(".json.tmp").exists()

    bad, bseg, _ = _fixture(
        tmp_path / "bad", verifier={**GOOD, "specific_enough": False})
    bdest = bad.output_dir / R.AUDIT_FILENAME
    with pytest.raises(NonRetryableBuildError) as exc:
        R.assert_selection_relevance(bad, [bseg], bdest)
    assert exc.value.kind == "selection_relevance"
    persisted = json.loads(bdest.read_text())
    assert persisted["status"] == "blocked" and persisted["blocked_count"] == 1
    assert not bdest.with_suffix(".json.tmp").exists()


def test_build_wires_contract_before_caption_narration_or_breakouts():
    from vidlore.clipstudio import build as B
    src = inspect.getsource(B.build_video)
    gate = src.index("_assert_selection_relevance(")
    assert gate < src.index("_resolve_cap_style")
    assert gate < src.index("# 1) engine Script")
    assert gate < src.index("# 2) narration")
    assert gate < src.index("# 2b) REAL-AUDIO BREAKOUTS")
