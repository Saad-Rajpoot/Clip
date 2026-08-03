"""Fail-closed pre-render semantic contract for resume/rerender/verify-disabled paths."""
from __future__ import annotations

import inspect
import json

import pytest

from vidlore.clipstudio import policy as P
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo
from vidlore.clipstudio.verify import NonRetryableBuildError


GOOD = {
    "status": "ok", "verdict": "keep", "matches_narration": True,
    "specific_enough": True, "correct_subject_visible": True,
    "wrong_subject_visible": False, "contradicts_narration": False,
    "quality_ok": True,
}


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
