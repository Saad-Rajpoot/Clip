import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from vidlore.clipstudio import build as B
from vidlore.clipstudio import scene_lineage as SL
from vidlore.clipstudio.verify import NonRetryableBuildError, VisionBackendError


def _jpg(path: Path, size=(1280, 720), colour=(80, 110, 140)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, quality=90)
    return path


def _project(tmp_path: Path, sources: dict):
    index = tmp_path / "index"
    output = tmp_path / "output"
    index.mkdir()
    output.mkdir()
    return NS(index_dir=index, output_dir=output, source=lambda sid: sources.get(sid))


def _source(tmp_path: Path, sid: str):
    path = tmp_path / f"{sid}.mp4"
    path.write_bytes(b"actual source bytes")
    return NS(id=sid, local_path=str(path), width=320, height=180)


def _selection(img: Path, **meta):
    return NS(segment_index=34, source_id="wrong-moving-selection", shot_index=91,
              in_point=99.0, out_point=101.0, image_path=str(img), image_meta=meta)


def _strict_segment():
    return NS(index=34, text="Varys appears in the city.", visual_policy="character_specific",
              required_entity="Varys", required_kind="character", expected_visual="Varys",
              scene_query="", quote="", is_specific_claim=False)


def _strict_keep_verdict():
    return {"status": "ok", "verdict": "keep", "matches_narration": True,
            "specific_enough": True, "quality_ok": True,
            "correct_subject_visible": True, "wrong_subject_visible": False,
            "contradicts_narration": False, "era_ok": True, "confidence": 0.9,
            "vision_served_by": "gemini:gemini-test:apikey"}


def _index_shot(proj, keyframe: Path, *, shot=7, start=10.0, end=14.0):
    (proj.index_dir / "varys.shots.json").write_text(json.dumps([
        {"source_id": "varys", "index": shot, "start": start, "end": end,
         "keyframe_path": str(keyframe)},
    ]))


def test_source_frame_rescue_uses_image_meta_owner_and_indexed_midpoint(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    wrong = _source(tmp_path, "wrong-moving-selection")
    proj = _project(tmp_path, {"varys": intended, "wrong-moving-selection": wrong})
    original = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, original)
    sel = _selection(original, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    seen = {}

    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        seen["argv"] = list(argv)
        _jpg(Path(argv[-1]), (1920, 1080), (120, 90, 60))
        return NS(returncode=0)

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    rescue = B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)

    assert rescue["actual_source_id"] == "varys"
    assert rescue["actual_shot_index"] == 7
    assert rescue["actual_time"] == 12.0
    assert rescue["source_width"] == 1920 and rescue["source_height"] == 1080
    assert seen["argv"][seen["argv"].index("-i") + 1] == str(Path(intended.local_path).resolve())
    assert seen["argv"][seen["argv"].index("-ss") + 1] == "12.000"
    assert "99.000" not in seen["argv"]
    assert str(Path(wrong.local_path).resolve()) not in seen["argv"]

    root = B._verified_image_lineage_root(proj, sel, rescue, 34)
    assert root["expected_image_source_id"] == root["actual_image_source_id"] == "varys"
    assert root["expected_image_shot_index"] == root["actual_image_shot_index"] == 7
    assert root["actual_image_time"] == 12.0
    assert root["source_native_height"] == 1080
    assert root["image_width"] == 1920 and root["image_height"] == 1080
    assert root["validated"] is True and root["root_binding"]


def test_sha_bound_semantic_still_airs_exact_judged_bytes_without_reextract(
        monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    wrong = _source(tmp_path, "wrong-moving-selection")
    proj = _project(tmp_path, {"varys": intended, "wrong-moving-selection": wrong})
    judged = _jpg(tmp_path / "semantically_judged.jpg", (1920, 1080), (31, 73, 119))
    _index_shot(proj, judged)
    judged_hash = B._image_file_sha256(judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=judged_hash)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})
    monkeypatch.setattr(
        B.subprocess, "run",
        lambda *_a, **_k: pytest.fail("SHA-bound judged still must never be re-extracted"))

    rescue = B._rescue_still_fullres(proj, sel, str(judged), lambda _m: None)
    assert Path(rescue["path"]).resolve() == judged.resolve()
    assert rescue["preserved_original"] is True
    assert rescue["semantic_binding_preserved"] is True
    assert rescue["file_sha256"] == rescue["semantic_image_sha256"] == judged_hash
    assert rescue["actual_source_id"] == "varys" and rescue["actual_shot_index"] == 7
    assert rescue["actual_time"] == 12.0

    root = B._verified_image_lineage_root(proj, sel, rescue, 34)
    assert root["semantic_binding_preserved"] is True
    assert root["semantic_image_sha256"] == root["image_sha256"] == judged_hash
    # Mutating the moving selection owner cannot redirect these source-frame bytes.
    assert root["actual_image_source_id"] == root["expected_image_source_id"] == "varys"


def test_lowres_sha_bound_semantic_still_is_native_extracted_then_freshly_verified(
        monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    proj.meta = {"analysis": {"episode_hint": "S01E03"}}
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288), (31, 73, 119))
    _index_shot(proj, judged)
    judged_hash = B._image_file_sha256(judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=judged_hash)
    seg = _strict_segment()
    eng = NS(anthropic_model="claude-test")
    seen = {}
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080), (31, 73, 119))
        return NS(returncode=0)

    def fake_verify(path, *_args, **_kwargs):
        seen["verified_path"] = str(path)
        return _strict_keep_verdict()

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr("vidlore.clipstudio.verify.verify_frame", fake_verify)
    rescue = B._rescue_still_fullres(
        proj, sel, str(judged), lambda _m: None, seg=seg, eng=eng)

    assert rescue["semantic_rematerialized"] is True
    assert rescue["image_width"] == 1920 and rescue["image_height"] == 1080
    assert Path(rescue["path"]).resolve() != judged.resolve()
    assert seen["verified_path"] == rescue["path"]
    assert rescue["semantic_image_sha256"] == B._image_file_sha256(rescue["path"])
    assert rescue["semantic_image_sha256"] != judged_hash
    root = B._verified_image_lineage_root(proj, sel, rescue, 34, seg=seg)
    assert root["validated"] is True
    assert root["semantic_rematerialized"] is True
    assert root["semantic_image_sha256"] == root["image_sha256"]


@pytest.mark.parametrize("verdict", [
    None,
    {"status": "error", "verdict": "keep"},
    {"status": False, "verdict": "keep"},
    {"status": "ok", "verdict": "keep"},
    {**_strict_keep_verdict(), "confidence": "high"},
    {k: v for k, v in _strict_keep_verdict().items() if k != "vision_served_by"},
])
def test_lowres_semantic_native_inconclusive_reverification_is_retryable(
        monkeypatch, tmp_path, verdict):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    proj.meta = {"analysis": {}}
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=B._image_file_sha256(judged))
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080))
        return NS(returncode=0)

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr("vidlore.clipstudio.verify.verify_frame",
                        lambda *_a, **_k: verdict)
    with pytest.raises(VisionBackendError, match="native still verifier"):
        B._rescue_still_fullres(
            proj, sel, str(judged), lambda _m: None,
            seg=_strict_segment(), eng=NS(anthropic_model="claude-test"))


def test_lowres_semantic_native_explicit_rejection_is_nonretryable(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    proj.meta = {"analysis": {}}
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=B._image_file_sha256(judged))
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080))
        return NS(returncode=0)

    rejection = {"status": "ok", "verdict": "replace", "matches_narration": False,
                 "specific_enough": False, "quality_ok": True,
                 "correct_subject_visible": False, "wrong_subject_visible": True,
                 "contradicts_narration": False, "era_ok": True}
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr("vidlore.clipstudio.verify.verify_frame",
                        lambda *_a, **_k: rejection)
    with pytest.raises(NonRetryableBuildError, match="image semantic gate"):
        B._rescue_still_fullres(
            proj, sel, str(judged), lambda _m: None,
            seg=_strict_segment(), eng=NS(anthropic_model="claude-test"))


def _native_rematerialized_record(tmp_path: Path) -> dict:
    aired = _jpg(tmp_path / "aired_native.jpg", (1920, 1080))
    return {
        "kind": "verified_image", "final_scene": 34, "original_beat": 34,
        "owner_beat": 34, "clip": "beat_034.mp4", "file": str(aired),
        "validated": True, "root_binding": "binding",
        "image_owner_kind": "source_frame",
        "image_sha256": B._image_file_sha256(aired),
        "image_width": 1920, "image_height": 1080,
        "expected_image_source_id": "varys", "actual_image_source_id": "varys",
        "expected_image_shot_index": 7, "actual_image_shot_index": 7,
        "actual_image_time": 12.0,
        "source_native_width": 1920, "source_native_height": 1080,
        "preserved_original": False, "semantic_binding_preserved": True,
        "semantic_rematerialized": True,
        "semantic_image_sha256": B._image_file_sha256(aired),
        "semantic_model": "gemini:gemini-test:apikey",
        "semantic_question_fingerprint": "question-fingerprint",
        "semantic_source_content_fingerprint": "source-content-fingerprint",
    }


def test_scene_lineage_assert_accepts_complete_native_semantic_rematerialization(tmp_path):
    rec = _native_rematerialized_record(tmp_path)
    audit = SL.assert_scene_lineage([rec], tmp_path / "lineage.json")
    assert audit["passed"] is True and audit["failures"] == []


@pytest.mark.parametrize("field,value", [
    ("semantic_image_sha256", "forged"),
    ("semantic_model", ""),
    ("semantic_question_fingerprint", ""),
    ("semantic_source_content_fingerprint", ""),
    ("semantic_binding_preserved", False),
    ("preserved_original", True),
])
def test_scene_lineage_assert_rejects_incomplete_or_forged_native_semantic_proof(
        tmp_path, field, value):
    rec = _native_rematerialized_record(tmp_path)
    rec[field] = value
    audit_path = tmp_path / f"lineage_{field}.json"
    with pytest.raises(NonRetryableBuildError, match="scene-lineage gate"):
        SL.assert_scene_lineage([rec], audit_path)
    persisted = json.loads(audit_path.read_text())
    assert persisted["passed"] is False and persisted["failures"]


def test_source_rewrite_during_native_extract_fails_closed(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    original = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, original)
    sel = _selection(original, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def replacing_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080))
        Path(intended.local_path).write_bytes(b"replacement source bytes with different content")
        return NS(returncode=0)

    monkeypatch.setattr(B.subprocess, "run", replacing_run)
    with pytest.raises(NonRetryableBuildError, match="changed while"):
        B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)


def test_native_pixels_replaced_while_verifier_runs_fail_closed(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    proj.meta = {"analysis": {}}
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=B._image_file_sha256(judged))
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080), (10, 20, 30))
        return NS(returncode=0)

    def replacing_verify(path, *_args, **_kwargs):
        _jpg(Path(path), (1920, 1080), (200, 180, 160))
        return _strict_keep_verdict()

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr("vidlore.clipstudio.verify.verify_frame", replacing_verify)
    with pytest.raises(NonRetryableBuildError, match="pixels changed"):
        B._rescue_still_fullres(
            proj, sel, str(judged), lambda _m: None,
            seg=_strict_segment(), eng=NS(anthropic_model="claude-test"))


@pytest.mark.parametrize("claimed", ["", "not-the-declared-sha"])
def test_semantic_claim_missing_or_mismatched_sha_never_falls_into_unverified_extract(
        monkeypatch, tmp_path, claimed):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=claimed)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})
    monkeypatch.setattr(
        B.subprocess, "run",
        lambda *_a, **_k: pytest.fail("invalid semantic SHA must fail before extraction"))
    with pytest.raises(Exception, match="semantic still SHA"):
        B._rescue_still_fullres(
            proj, sel, str(judged), lambda _m: None,
            seg=_strict_segment(), eng=NS(anthropic_model="claude-test"))


@pytest.mark.parametrize("mutation", ["source", "question"])
def test_preflight_native_semantic_rescue_rejects_identity_change_before_air(
        monkeypatch, tmp_path, mutation):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    proj.meta = {"analysis": {}}
    judged = _jpg(tmp_path / "shot_0007.jpg", (512, 288))
    _index_shot(proj, judged)
    sel = _selection(judged, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True, still_semantic_verified=True,
                     still_image_sha256=B._image_file_sha256(judged))
    seg = _strict_segment()
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def fake_run(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1920, 1080))
        return NS(returncode=0)

    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr("vidlore.clipstudio.verify.verify_frame",
                        lambda *_a, **_k: _strict_keep_verdict())
    rescue = B._rescue_still_fullres(
        proj, sel, str(judged), lambda _m: None,
        seg=seg, eng=NS(anthropic_model="claude-test"))
    if mutation == "source":
        Path(intended.local_path).write_bytes(b"changed owner source bytes")
    else:
        seg.text = "A materially different narration question."
    with pytest.raises(Exception, match="image-lineage gate"):
        B._verified_image_lineage_root(proj, sel, rescue, 34, seg=seg)


def test_metadata_free_verified_source_frame_preserves_file_never_selection(monkeypatch, tmp_path):
    wrong = _source(tmp_path, "wrong-moving-selection")
    proj = _project(tmp_path, {"wrong-moving-selection": wrong})
    original = _jpg(tmp_path / "verified_legacy.jpg", (1280, 720))
    sel = _selection(original, source="source-frame-recovery", still_verified=True)

    def no_extract(*_args, **_kwargs):
        raise AssertionError("metadata-free still must not open the moving selection")

    monkeypatch.setattr(B.subprocess, "run", no_extract)
    rescue = B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)
    assert Path(rescue["path"]) == original
    assert rescue["ownership_kind"] == "verified_file"
    assert rescue["actual_source_id"] == "" and rescue["actual_time"] is None
    assert rescue["preserved_original"] is True
    root = B._verified_image_lineage_root(proj, sel, rescue, 34)
    assert root["image_owner_kind"] == "verified_file"
    assert root["image_sha256"] == B._image_file_sha256(original)


@pytest.mark.parametrize("meta", [
    {"source": "source-frame-recovery", "src": "varys", "shot": 999,
     "still_verified": True},
    {"source": "source-frame-recovery", "src": "varys", "still_verified": True},
    {"source": "source-frame-recovery", "shot": 7, "still_verified": True},
])
def test_wrong_or_partial_image_owner_metadata_fails_without_selection_fallback(
        monkeypatch, tmp_path, meta):
    intended = _source(tmp_path, "varys")
    wrong = _source(tmp_path, "wrong-moving-selection")
    proj = _project(tmp_path, {"varys": intended, "wrong-moving-selection": wrong})
    original = _jpg(tmp_path / "still.jpg")
    _index_shot(proj, original)
    sel = _selection(original, **meta)
    monkeypatch.setattr(B.subprocess, "run",
                        lambda *_a, **_k: pytest.fail("wrong metadata must not extract any source"))
    with pytest.raises(Exception, match="image-lineage gate"):
        B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)


def test_valid_looking_but_wrong_shot_metadata_cannot_redirect_verified_image(
        monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    declared = _jpg(tmp_path / "verified_other_frame.jpg", (1280, 720), (10, 20, 30))
    indexed = _jpg(tmp_path / "shot_0007.jpg", (512, 288), (200, 180, 160))
    _index_shot(proj, indexed)
    sel = _selection(declared, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    monkeypatch.setattr(B.subprocess, "run",
                        lambda *_a, **_k: pytest.fail("mismatched keyframe must not extract"))
    with pytest.raises(Exception, match="does not match indexed keyframe"):
        B._rescue_still_fullres(proj, sel, str(declared), lambda _m: None)


def test_lineage_rejects_forged_rescue_owner(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    original = _jpg(tmp_path / "still.jpg")
    _index_shot(proj, original)
    sel = _selection(original, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})
    forged = {"path": str(original), "ownership_kind": "source_frame",
              "actual_source_id": "olenna", "actual_shot_index": 2, "actual_time": 12.0,
              "source_width": 1920, "source_height": 1080,
              "image_width": 1280, "image_height": 720,
              "file_sha256": B._image_file_sha256(original)}
    with pytest.raises(Exception, match="expected owner"):
        B._verified_image_lineage_root(proj, sel, forged, 34)


@pytest.mark.parametrize("dims", [(640, 360), (640, 720), (1279, 720)])
def test_source_frame_native_bytes_below_1280x720_fail_before_extract(
        monkeypatch, tmp_path, dims):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    original = _jpg(tmp_path / "still.jpg", (512, 288))
    _index_shot(proj, original)
    sel = _selection(original, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": dims[0], "height": dims[1]})
    monkeypatch.setattr(B.subprocess, "run",
                        lambda *_a, **_k: pytest.fail("SD owner must fail before extraction"))
    with pytest.raises(Exception, match="native-resolution gate"):
        B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)


def test_source_frame_extraction_must_itself_retain_1280x720(monkeypatch, tmp_path):
    intended = _source(tmp_path, "varys")
    proj = _project(tmp_path, {"varys": intended})
    original = _jpg(tmp_path / "still.jpg", (512, 288))
    _index_shot(proj, original)
    sel = _selection(original, source="source-frame-recovery", src="varys", shot=7,
                     still_verified=True)
    monkeypatch.setattr("vidlore.clipstudio.ingest.probe",
                        lambda _p: {"width": 1920, "height": 1080})

    def lowres_extract(argv, **_kwargs):
        _jpg(Path(argv[-1]), (1024, 768))
        return NS(returncode=0)

    monkeypatch.setattr(B.subprocess, "run", lowres_extract)
    with pytest.raises(Exception, match="1280x720-or-better"):
        B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)


@pytest.mark.parametrize("size,passed", [
    ((1280, 720), True), ((720, 1280), True), ((1279, 720), False),
    ((1280, 719), False), ((400, 1000), False),
])
def test_web_still_requires_real_1280x720_pixels(tmp_path, size, passed):
    proj = _project(tmp_path, {})
    original = _jpg(tmp_path / f"web_{size[0]}x{size[1]}.jpg", size)
    sel = _selection(original, source="web-exact-scene", relevance_class="exact_scene")
    if passed:
        rescue = B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)
        assert rescue["image_width"] == size[0] and rescue["image_height"] == size[1]
    else:
        with pytest.raises(Exception, match="native-resolution gate"):
            B._rescue_still_fullres(proj, sel, str(original), lambda _m: None)


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_declared_missing_or_empty_image_fails_closed(kind, tmp_path):
    path = tmp_path / f"{kind}.jpg"
    if kind == "empty":
        path.write_bytes(b"")
    sel = _selection(path, source="web-exact-scene")
    with pytest.raises(Exception, match="refusing to fall through to moving footage"):
        B._require_declared_image_path(sel)


def test_build_registers_actual_image_root_and_persists_dimensions():
    src = Path(B.__file__).read_text()
    branch = src[src.index("# A declared image is a semantic replacement"):
                 src.index("# VERIFIED-SELECTION LOCK")]
    assert "_require_declared_image_path(sel)" in branch
    assert "_rescue_still_fullres" in branch
    assert "_verified_image_lineage_root" in branch
    assert "_lineage_register(_img, _img_root)" in branch
    assert "_selection_root(sel" not in branch
    assert "or _img" not in branch
    assert "_persist_image_lineage_audit" in branch


def test_image_audit_records_native_and_pixel_dimensions(tmp_path):
    proj = _project(tmp_path, {})
    entry = {"final_scene": 34, "actual_image_source_id": "varys",
             "actual_image_shot_index": 7, "actual_image_time": 12.0,
             "source_native_width": 1920, "source_native_height": 1080,
             "image_width": 1920, "image_height": 1080, "validated": True}
    path = B._persist_image_lineage_audit(proj, [entry], [])
    data = json.loads(path.read_text())
    assert data["passed"] is True and data["entries"] == [entry]
    assert data["minimum_source_video_height"] == 720
    assert data["minimum_still_short_edge"] == 720
    assert data["minimum_still_long_edge"] == 1280
