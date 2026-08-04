"""Focused publication regressions for indexed source-frame stills.

These tests keep candidate discovery out of scope.  They exercise the two boundaries that matter
at publication time: semantic coverage must describe the exact native pixels that can air, and a
persisted native extraction must remain immutably tied to its indexed owner.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from vidlore.clipstudio import build as B
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.verify import NonRetryableBuildError


STRICT_KEEP = {
    "status": "ok",
    "verdict": "keep",
    "matches_narration": True,
    "specific_enough": True,
    "correct_subject_visible": True,
    "wrong_subject_visible": False,
    "contradicts_narration": False,
    "quality_ok": True,
    "era_ok": True,
}


def _jpg(path: Path, size: tuple[int, int], colour=(50, 90, 130)) -> Path:
    Image.new("RGB", size, colour).save(path, quality=90)
    return path


def _segment(*, scene_query="Game of Thrones S03E09 Red Wedding"):
    return NS(
        index=12,
        text="The master-at-arms stands in the court.",
        visual_policy=P.EXACT,
        is_specific_claim=True,
        required_entity="Aron Santagar",
        required_kind="character",
        expected_visual="Aron Santagar in the Red Keep court",
        scene_query=scene_query,
        quote="",
    )


def _project(tmp_path: Path, *, title="Game of Thrones S03E09 Court"):
    index_dir = tmp_path / "index"
    output_dir = tmp_path / "output"
    index_dir.mkdir(parents=True)
    output_dir.mkdir()
    media = tmp_path / "owner.mp4"
    media.write_bytes(b"immutable indexed owner source")
    source = NS(id="owner", title=title, local_path=str(media))
    return NS(
        index_dir=index_dir,
        output_dir=output_dir,
        meta={
            "analysis": {
                "video_type": "single_scene",
                "episode_hint": "S04E01",
                "episode_hint_verified": True,
                "anchor_scenes": [],
                "characters": [],
                "actors": [],
            }
        },
        source=lambda source_id: source if source_id == source.id else None,
    )


def _owned_selection(image: Path, **extra_meta):
    evidence = dict(STRICT_KEEP)
    meta = {
        "source": "source-frame-recovery",
        "src": "owner",
        "shot": 7,
        "relevance_class": "exact_scene",
        "still_verified": True,
        "still_semantic_verified": True,
        "still_verifier": evidence,
        "still_image_sha256": R.image_sha256(image),
        "exact_still_verified": True,
        "exact_still_verifier": evidence,
        **extra_meta,
    }
    return NS(
        segment_index=12,
        source_id="rejected-moving-source",
        shot_index=99,
        in_point=90.0,
        out_point=94.0,
        image_path=str(image),
        image_meta=meta,
    )


def test_lowres_owned_source_frame_is_invalid_until_native_materialization(tmp_path):
    """A positive thumbnail verdict cannot authorize separately materialized HD pixels."""
    proj = _project(tmp_path)
    thumbnail = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    ok, reason = R.verified_still_coverage(
        _owned_selection(thumbnail), _segment(), proj=proj)

    assert ok is False
    assert reason == (
        "owned source-frame still is 512x288; native materialization "
        "and fresh semantic verification are required"
    )


def test_fullres_exact_verified_owned_still_passes_publication_contract(tmp_path):
    """The exact HD bytes may cover the beat once the strict verdict is hash-bound to them."""
    proj = _project(tmp_path)
    native = _jpg(tmp_path / "shot_0007.native.jpg", (1920, 1080))
    (proj.index_dir / "owner.shots.json").write_text(json.dumps([{
        "source_id": "owner", "index": 7, "start": 10.0, "end": 14.0,
        "keyframe_path": str(native),
    }]), encoding="utf-8")

    assert R.verified_still_coverage(
        _owned_selection(native), _segment(), proj=proj) == (True, "source-frame-recovery")


def test_owned_still_title_cast_warning_requires_pixel_resolution_not_title_rejection(tmp_path):
    proj = _project(tmp_path, title='"Why Littlefinger?" Arya Stark')
    proj.meta["analysis"]["characters"] = [
        {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
        {"name": "Catelyn Stark", "actor": None},
    ]
    seg = _segment(scene_query="Littlefinger tells Catelyn the dagger lie in the brothel")
    seg.required_entity = "Petyr Baelish"
    seg.expected_visual = "Littlefinger faces Catelyn while telling the dagger lie"
    native = _jpg(tmp_path / "cast-warning.native.jpg", (1920, 1080))
    (proj.index_dir / "owner.shots.json").write_text(json.dumps([{
        "source_id": "owner", "index": 7, "start": 10.0, "end": 14.0,
        "keyframe_path": str(native),
    }]), encoding="utf-8")
    sel = _owned_selection(native)

    ok, reason = R.verified_still_coverage(sel, seg, proj=proj)
    assert ok is False
    assert "unresolved exact-scene cast warning" in reason

    # The title is upload-level evidence only. A focused strict verdict on these exact native
    # pixels may resolve it (e.g. a correct window inside a compilation).
    sel.image_meta["still_verifier"]["source_title_conflict_resolved"] = True
    sel.image_meta["exact_still_verifier"]["source_title_conflict_resolved"] = True
    assert R.verified_still_coverage(sel, seg, proj=proj) == (
        True, "source-frame-recovery")


def test_standalone_build_preserve_rejects_unresolved_title_cast_warning(
        monkeypatch, tmp_path):
    proj = _project(tmp_path, title='"Why Littlefinger?" Arya Stark')
    proj.meta["analysis"]["characters"] = [
        {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
        {"name": "Catelyn Stark", "actor": None},
    ]
    seg = _segment(scene_query="Littlefinger tells Catelyn the dagger lie in the brothel")
    seg.required_entity = "Petyr Baelish"
    seg.expected_visual = "Littlefinger faces Catelyn while telling the dagger lie"
    native = _jpg(tmp_path / "build-cast-warning.native.jpg", (1920, 1080))
    (proj.index_dir / "owner.shots.json").write_text(json.dumps([{
        "source_id": "owner", "index": 7, "start": 10.0, "end": 14.0,
        "keyframe_path": str(native),
    }]), encoding="utf-8")
    sel = _owned_selection(native)
    monkeypatch.setattr(B, "_probe_image_owner_source", lambda _path: (1920, 1080))

    with pytest.raises(NonRetryableBuildError, match="unresolved source-title cast warning"):
        B._rescue_still_fullres(
            proj, sel, sel.image_path, lambda _msg: None, seg=seg, eng=None)

    sel.image_meta["still_verifier"]["source_title_conflict_resolved"] = True
    sel.image_meta["exact_still_verifier"]["source_title_conflict_resolved"] = True
    rescue = B._rescue_still_fullres(
        proj, sel, sel.image_path, lambda _msg: None, seg=seg, eng=None)
    assert rescue["semantic_binding_preserved"] is True


def test_unproven_separately_extracted_hd_still_is_blocked_before_build(tmp_path):
    """HD dimensions and a legacy SHA cannot prove source ownership or the current question."""
    proj = _project(tmp_path)
    native = _jpg(tmp_path / "unbound.native.jpg", (1920, 1080))

    ok, reason = R.verified_still_coverage(
        _owned_selection(native), _segment(), proj=proj)

    assert ok is False
    assert "indexed provenance is invalid" in reason


def test_owned_still_rejects_wrong_beat_local_era_even_when_global_era_matches_owner(tmp_path):
    """A verified global S04 hint must not override this beat's explicit S03 requirement."""
    proj = _project(tmp_path, title="Game of Thrones S04E01 Red Keep Court")
    native = _jpg(tmp_path / "wrong_era.native.jpg", (1920, 1080))
    seg = _segment(scene_query="Game of Thrones S03E09 Red Wedding court")

    assert V._project_beat_era(proj, seg) == "season 3"
    ok, reason = R.verified_still_coverage(_owned_selection(native), seg, proj=proj)
    assert ok is False
    assert reason == "owned source-frame still declares the wrong era for season 3"


def _persisted_native_fixture(tmp_path: Path):
    proj = _project(tmp_path)
    indexed = _jpg(tmp_path / "indexed_thumbnail.jpg", (512, 288), (20, 40, 60))
    native = _jpg(tmp_path / "native_verified.jpg", (1920, 1080), (80, 100, 120))
    (proj.index_dir / "owner.shots.json").write_text(
        json.dumps([{
            "source_id": "owner",
            "index": 7,
            "start": 10.0,
            "end": 14.0,
            "keyframe_path": str(indexed),
        }]),
        encoding="utf-8",
    )
    source = proj.source("owner")
    sel = _owned_selection(
        native,
        native_semantic_materialized=True,
        native_indexed_keyframe_sha256=B._image_file_sha256(indexed),
        native_owner_source_content_fingerprint=V._file_fingerprint(source.local_path),
        native_owner_time=12.0,
        native_semantic_question_fingerprint="question-fingerprint",
        native_semantic_model="gemini:gemini-test:apikey",
    )
    return proj, sel


def test_resolver_accepts_complete_persisted_native_provenance(tmp_path):
    proj, sel = _persisted_native_fixture(tmp_path)

    owner = B._resolve_indexed_still_owner(proj, sel)

    assert owner is not None
    assert owner["source_id"] == "owner"
    assert owner["shot_index"] == 7
    assert owner["time"] == 12.0


def _bind_current_native_question(proj, sel, seg):
    owner = B._resolve_indexed_still_owner(proj, sel)
    model = "gemini:gemini-test:apikey"
    verdict = dict(STRICT_KEEP, vision_served_by=model)
    image_hash = R.image_sha256(sel.image_path)
    question_fp, source_fp = B._semantic_still_question_fingerprint(
        proj, seg, owner, image_sha256=image_hash, model=model)
    sel.image_meta.update({
        "still_verifier": verdict,
        "exact_still_verifier": verdict,
        "native_semantic_model": model,
        "native_semantic_question_fingerprint": question_fp,
        "native_owner_source_content_fingerprint": source_fp,
    })


def test_native_rescue_revalidates_current_question_before_preserving(monkeypatch, tmp_path):
    proj, sel = _persisted_native_fixture(tmp_path)
    seg = _segment()
    _bind_current_native_question(proj, sel, seg)
    monkeypatch.setattr(B, "_probe_image_owner_source", lambda _path: (1920, 1080))

    rescue = B._rescue_still_fullres(
        proj, sel, sel.image_path, lambda _msg: None, seg=seg, eng=None)

    assert rescue["preserved_original"] is True
    assert rescue["semantic_binding_preserved"] is True


def test_persisted_native_binding_requires_resolved_title_cast_warning(tmp_path):
    proj, sel = _persisted_native_fixture(tmp_path)
    proj.source("owner").title = '"Why Littlefinger?" Arya Stark'
    proj.meta["analysis"]["characters"] = [
        {"name": "Petyr Baelish", "actor": "Aidan Gillen"},
        {"name": "Catelyn Stark", "actor": None},
    ]
    seg = _segment(scene_query="Littlefinger tells Catelyn the dagger lie in the brothel")
    seg.required_entity = "Petyr Baelish"
    seg.expected_visual = "Littlefinger faces Catelyn while telling the dagger lie"
    _bind_current_native_question(proj, sel, seg)
    owner = B._resolve_indexed_still_owner(proj, sel)
    image_hash = R.image_sha256(sel.image_path)
    source_fp = V._file_fingerprint(proj.source("owner").local_path)

    sel.image_meta["still_verifier"]["source_title_conflict_resolved"] = False
    reason = B._persisted_native_still_semantic_reason(
        proj, seg, owner, sel.image_meta, image_sha256=image_hash,
        source_fingerprint=source_fp)
    assert "source-title cast warning is not resolved" in reason

    sel.image_meta["still_verifier"]["source_title_conflict_resolved"] = True
    assert B._persisted_native_still_semantic_reason(
        proj, seg, owner, sel.image_meta, image_sha256=image_hash,
        source_fingerprint=source_fp) == ""


def test_native_rescue_rejects_forged_current_question_binding(monkeypatch, tmp_path):
    proj, sel = _persisted_native_fixture(tmp_path)
    seg = _segment()
    _bind_current_native_question(proj, sel, seg)
    sel.image_meta["native_semantic_question_fingerprint"] = "forged-question"
    monkeypatch.setattr(B, "_probe_image_owner_source", lambda _path: (1920, 1080))

    with pytest.raises(NonRetryableBuildError, match="beat-question fingerprint changed"):
        B._rescue_still_fullres(
            proj, sel, sel.image_path, lambda _msg: None, seg=seg, eng=None)


def test_native_refresh_replaces_stale_question_with_fresh_strict_verdict(
        monkeypatch, tmp_path):
    proj, sel = _persisted_native_fixture(tmp_path)
    seg = _segment()
    sel.image_meta["native_semantic_question_fingerprint"] = "stale-question"
    fresh_verdict = dict(STRICT_KEEP, vision_served_by="vision-live")
    calls = []

    def fake_strict(_proj, _sel, _seg, _owner, image_path, _eng, **kwargs):
        calls.append((Path(image_path), kwargs))
        return {
            "verdict": fresh_verdict,
            "model": "vision-live",
            "question_fingerprint": "fresh-question",
            "source_content_fingerprint": V._file_fingerprint(
                proj.source("owner").local_path),
            "image_sha256": R.image_sha256(image_path),
            "strict_reason": "",
        }

    monkeypatch.setattr(B, "_probe_image_owner_source", lambda _path: (1920, 1080))
    monkeypatch.setattr(B, "_strictly_verify_native_still", fake_strict)

    rescue = B._rescue_still_fullres(
        proj, sel, sel.image_path, lambda _msg: None, seg=seg, eng=NS(),
        allow_semantic_reject=True, refresh_semantic_verdict=True)

    assert len(calls) == 1
    assert calls[0][0] == Path(sel.image_path)
    assert calls[0][1]["allow_content_reject"] is True
    assert rescue["semantic_question_fingerprint"] == "fresh-question"
    assert rescue["semantic_verifier"] == fresh_verdict


@pytest.mark.parametrize(("field", "bad_value"), [
    ("native_semantic_materialized", False),
    ("native_indexed_keyframe_sha256", ""),
    ("native_indexed_keyframe_sha256", "forged-keyframe-hash"),
    ("native_owner_source_content_fingerprint", ""),
    ("native_owner_source_content_fingerprint", "forged-source-fingerprint"),
    ("native_owner_time", None),
    ("native_owner_time", 12.01),
    ("still_image_sha256", ""),
    ("still_image_sha256", "forged-native-image-hash"),
    ("native_semantic_question_fingerprint", ""),
    ("native_semantic_model", "none"),
])
def test_resolver_rejects_missing_or_forged_native_provenance(
        tmp_path, field, bad_value):
    proj, sel = _persisted_native_fixture(tmp_path)
    sel.image_meta[field] = bad_value

    with pytest.raises(
            NonRetryableBuildError,
            match="declared verified still does not match indexed keyframe"):
        B._resolve_indexed_still_owner(proj, sel)
