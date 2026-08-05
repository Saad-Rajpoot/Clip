"""Cross-copy quote recovery: local alignment plus fail-closed publication binding."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from vidlore.clipstudio import audio_align as A
from vidlore.clipstudio import index as IX
from vidlore.clipstudio import ledger as L
from vidlore.clipstudio import orchestrate as O
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import ClipConfig, load_clip_config
from vidlore.clipstudio.models import (
    ClipCandidate, ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo,
)


QUOTE = "He's choking!"
CONFIRMATION_ARTIFACT_KEY = "a" * 64


def _confirmation(proj, *, source_id="dirty"):
    decoder = R._quote_confirmation_decoder_fingerprint(load_clip_config())
    return {
        "schema_version": R.QUOTE_CONFIRMATION_SCHEMA,
        "algorithm": R.QUOTE_CONFIRMATION_ALGORITHM,
        "status": "confirmed",
        "artifact_key": CONFIRMATION_ARTIFACT_KEY,
        "decoder_fingerprint": decoder,
        "prompted_span": [1.0, 2.0, 1.0],
        "confirmed_span": [1.0, 2.0, 1.0],
        "timed_asr_ratio": 1.0,
        "match_method": "exact_contiguous_timed_asr+unprompted_confirmation",
        "source_id": source_id,
    }


def _confirmed_revalidation(proj, src, _quote, _span, _cfg, **_kwargs):
    return _confirmation(proj, source_id=src.id)


@pytest.fixture(autouse=True)
def _independent_quote_confirmation(monkeypatch):
    """Fixture media are byte sentinels; keep the independent proof explicit in every path."""
    monkeypatch.setattr(
        R, "_confirm_prompted_quote_span_unprompted",
        _confirmed_revalidation)


def _stamp(proj, sid):
    (proj.index_dir / f"{sid}.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA,
        "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, load_clip_config()),
    }))


def _project(tmp_path):
    proj = ClipProject(name="audio-transfer", root=str(tmp_path))
    proj.ensure_dirs()
    target_media = tmp_path / "clean_hd.mp4"
    target_media.write_bytes(b"clean-hd-media-bytes")
    reference_media = tmp_path / "dirty_sd.mp4"
    reference_media.write_bytes(b"dirty-sd-media-bytes")
    proj.sources = [
        SourceVideo(id="clean", url="u-clean", title="Purple Wedding clean scene",
                    permission="owner", status="ok", local_path=str(target_media),
                    duration=10.0, width=1920, height=1080),
        SourceVideo(id="dirty", url="u-dirty", title="Purple Wedding burned subtitles",
                    permission="owner", status="ok", local_path=str(reference_media),
                    duration=10.0, width=640, height=360),
    ]
    for sid, spans in {
            "clean": [(0.0, 2.0), (3.5, 5.5)],
            "dirty": [(0.5, 2.5)],
    }.items():
        rows = []
        for index, (start, end) in enumerate(spans):
            frame = tmp_path / f"{sid}_{index}.jpg"
            frame.write_bytes(f"frame-{sid}-{index}".encode())
            rows.append(Shot(source_id=sid, index=index, start=start, end=end,
                             keyframe_path=str(frame), quality=.9).to_dict())
        proj.shots_path(sid).write_text(json.dumps(rows))
    # The reference has the authoritative words; current ASR on the clean copy missed them.
    (proj.index_dir / "dirty.words.json").write_text(json.dumps([
        [1.0, 1.35, "He's"], [1.35, 2.0, "choking"],
    ]))
    (proj.index_dir / "clean.words.json").write_text(json.dumps([
        [0.1, 0.4, "unrelated"], [0.4, 0.7, "dialogue"],
    ]))
    _stamp(proj, "dirty")
    _stamp(proj, "clean")
    seg = ScriptSegment(
        index=0, text="Joffrey suddenly cannot breathe.", expected_visual="Joffrey choking",
        required_entity="Joffrey Baratheon", required_kind="character",
        scene_query="Joffrey Purple Wedding choking", quote=QUOTE,
        visual_policy=P.EXACT, is_specific_claim=True, est_duration=2.0)
    sel = ClipSelection(
        segment_index=0, source_id="clean", shot_index=0, in_point=0.0, out_point=2.0,
        confidence=.8, signals={"dialogue": .98})
    proj.segments = [seg]
    proj.selections = [sel]
    # ASR provenance is bound to authored quote hints as well as model/cast options.
    _stamp(proj, "dirty")
    _stamp(proj, "clean")
    return proj, seg, sel


def _contract(proj):
    confirmation = _confirmation(proj)
    match = {
        "source_id": "dirty", "source_title": "dirty",
        "timed_asr_span": [1.0, 2.0], "timed_asr_ratio": 1.0,
        "prompted_asr_span": [1.0, 2.0, 1.0],
        "unprompted_confirmation": confirmation,
    }
    return {
        "authored_quote": QUOTE,
        "branch": "verbatim",
        "verbatim_required": True,
        "requires_exact_contiguous_match": True,
        "confirmation_decoder_fingerprint_expected":
            confirmation["decoder_fingerprint"],
        "asr_prompt_fingerprint_expected": IX.asr_semantic_fingerprint(
            proj, load_clip_config()),
        "pool_match": dict(match),
        "pool_matches": [dict(match)],
    }


def _alignment(*, correlation=.98, runner=.10, margin=.88):
    return {
        "status": "matched", "reason": "",
        "schema_version": A.AUDIO_QUOTE_TRANSFER_SCHEMA,
        "algorithm": A.AUDIO_QUOTE_TRANSFER_ALGORITHM,
        "sample_rate_hz": A.AUDIO_QUOTE_TRANSFER_SAMPLE_RATE,
        "reference_quote_span": [1.0, 2.0],
        "reference_extract_window": [0.6, 2.4],
        "target_search_window": [0.0, 10.0],
        "target_quote_span": [4.0, 5.0],
        "correlation": correlation,
        "runner_up_correlation": runner,
        "uniqueness_margin": margin,
        "minimum_correlation": A.AUDIO_QUOTE_TRANSFER_MIN_CORRELATION,
        "minimum_uniqueness_margin": A.AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN,
    }


def _evidence(proj, *, alignment=None):
    return A.make_transfer_evidence(
        authored_quote=QUOTE, reference_source_id="dirty",
        reference_source_content_fingerprint=V._file_fingerprint(
            proj.source("dirty").local_path),
        reference_asr_ratio=1.0,
        reference_quote_confirmation_artifact_key=CONFIRMATION_ARTIFACT_KEY,
        reference_quote_confirmation_decoder_fingerprint=
            R._quote_confirmation_decoder_fingerprint(load_clip_config()),
        target_source_id="clean",
        target_source_content_fingerprint=V._file_fingerprint(
            proj.source("clean").local_path),
        target_selected_window=[3.5, 5.5], alignment=alignment or _alignment())


def test_pcm_alignment_finds_one_scaled_noisy_copy():
    rng = np.random.default_rng(713)
    rate = 4000
    time = np.arange(rate * 2, dtype=np.float64) / rate
    reference = (
        .34 * np.sin(2 * np.pi * (180 * time + 75 * time * time))
        + .19 * np.sin(2 * np.pi * 431 * time)
        + .04 * rng.standard_normal(time.size)).astype(np.float32)
    target = (.01 * rng.standard_normal(rate * 8)).astype(np.float32)
    expected = rate * 3 + 317
    target[expected:expected + reference.size] += .72 * reference

    result = A.normalized_pcm_alignment(reference, target, sample_rate=rate)

    assert result["status"] == "matched"
    assert result["best_start_sample"] == expected
    assert result["correlation"] > .98
    assert result["uniqueness_margin"] > .70


def test_pcm_alignment_rejects_low_and_nonunique_matches():
    rng = np.random.default_rng(911)
    rate = 4000
    reference = rng.standard_normal(rate).astype(np.float32)
    unrelated = rng.standard_normal(rate * 5).astype(np.float32)
    low = A.normalized_pcm_alignment(reference, unrelated, sample_rate=rate)
    assert low["status"] == "rejected"
    assert low["reason"] == "correlation_below_floor"

    repeated = (.01 * rng.standard_normal(rate * 7)).astype(np.float32)
    repeated[rate:rate * 2] += reference
    repeated[rate * 5:rate * 6] += reference
    nonunique = A.normalized_pcm_alignment(reference, repeated, sample_rate=rate)
    assert nonunique["status"] == "rejected"
    assert nonunique["reason"] == "alignment_not_unique"


def test_pcm_alignment_rejects_exactly_back_to_back_duplicate_occurrences():
    """A second occurrence one template away is distinct, not part of the first peak's lobe."""
    rng = np.random.default_rng(1201)
    rate = 4000
    reference = rng.standard_normal(rate).astype(np.float32)
    target = (.005 * rng.standard_normal(rate * 5)).astype(np.float32)
    first = rate
    second = first + reference.size
    target[first:first + reference.size] += reference
    target[second:second + reference.size] += reference

    result = A.normalized_pcm_alignment(reference, target, sample_rate=rate)

    assert result["status"] == "rejected"
    assert result["reason"] == "alignment_not_unique"
    assert result["same_peak_tolerance_samples"] < reference.size
    assert result["runner_up_correlation"] > .99


def test_batch_transfer_decodes_long_target_once_for_multiple_references():
    target = np.zeros(A.AUDIO_QUOTE_TRANSFER_SAMPLE_RATE * 10, dtype=np.float32)
    short = np.ones(A.AUDIO_QUOTE_TRANSFER_SAMPLE_RATE * 2, dtype=np.float32)
    decode_paths = []

    def decode(path, _start, _end, *, sample_rate):
        assert sample_rate == A.AUDIO_QUOTE_TRANSFER_SAMPLE_RATE
        decode_paths.append(str(path))
        return (target.copy(), "") if str(path) == "target.mp4" else (short.copy(), "")

    aligned = {
        "status": "matched", "reason": "", "best_start_sample": 4000,
        "schema_version": A.AUDIO_QUOTE_TRANSFER_SCHEMA,
        "algorithm": A.AUDIO_QUOTE_TRANSFER_ALGORITHM,
        "sample_rate_hz": A.AUDIO_QUOTE_TRANSFER_SAMPLE_RATE,
        "correlation": .97, "runner_up_correlation": .2, "uniqueness_margin": .77,
    }
    references = [(f"ref-{i}.mp4", [1.0, 2.0]) for i in range(3)]
    with mock.patch.object(A, "_pcm_from_media", side_effect=decode), \
            mock.patch.object(A, "normalized_pcm_alignment", return_value=aligned):
        results = A.transfer_quote_spans(
            references, "target.mp4", target_search_window=[0.0, 10.0])

    assert len(results) == 3 and all(row["status"] == "matched" for row in results)
    assert decode_paths.count("target.mp4") == 1
    assert sorted(path for path in decode_paths if path != "target.mp4") == \
        ["ref-0.mp4", "ref-1.mp4", "ref-2.mp4"]


def test_contract_accepts_strong_bound_transfer_when_current_target_asr_missed(tmp_path):
    proj, seg, sel = _project(tmp_path)
    sel.in_point, sel.out_point, sel.shot_index = 3.5, 5.5, 1
    sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL] = _evidence(proj)

    ok, reason, detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=_contract(proj))

    assert ok is True and reason == ""
    assert detail["quote_location_method"] == "audio_transfer"
    assert detail["audio_transfer"]["correlation"] == .98
    assert detail["dialogue_signal"] == .98, "the existing dialogue floor remains in force"


def test_transfer_evidence_can_only_rebind_its_selected_window_with_fresh_fingerprint(tmp_path):
    proj, _seg, _sel = _project(tmp_path)
    old = _evidence(proj)

    rebound = A.rebind_transfer_evidence_window(old, [3.6, 5.5])

    assert rebound["target_selected_window"] == [3.6, 5.5]
    assert rebound["binding_fingerprint"] != old["binding_fingerprint"]
    assert A.transfer_evidence_shape_reason(rebound) == ""
    fabricated = copy.deepcopy(old)
    fabricated["correlation"] = 1.0
    assert A.rebind_transfer_evidence_window(fabricated, [3.6, 5.5]) == {}


def test_preconfirmation_transfer_records_and_malformed_confirmation_keys_fail_closed(tmp_path):
    proj, _seg, _sel = _project(tmp_path)
    current = _evidence(proj)

    legacy = copy.deepcopy(current)
    legacy["schema_version"] = 2
    legacy.pop("reference_quote_confirmation_artifact_key")
    legacy.pop("reference_quote_confirmation_decoder_fingerprint")
    assert A.transfer_evidence_shape_reason(legacy) == "evidence_schema_mismatch"

    malformed = copy.deepcopy(current)
    malformed["reference_quote_confirmation_artifact_key"] = "not-a-sha256"
    malformed["binding_fingerprint"] = A.evidence_binding_fingerprint(malformed)
    assert A.transfer_evidence_shape_reason(malformed) == \
        "reference_quote_confirmation_artifact_key_malformed"


def test_ledger_preserves_valid_transfer_proof_outside_numeric_signals(tmp_path):
    proj, seg, sel = _project(tmp_path)
    sel.in_point, sel.out_point, sel.shot_index = 3.5, 5.5, 1
    evidence = _evidence(proj)
    sel.signals = {
        "dialogue": .98,
        "quote_audio_transfer": True,
        A.AUDIO_QUOTE_TRANSFER_SIGNAL: evidence,
    }

    L.write_ledger(proj, [seg])

    record = json.loads(proj.ledger_path.read_text().strip())
    assert record["signals"] == {"dialogue": .98, "quote_audio_transfer": 1.0}
    assert record[A.AUDIO_QUOTE_TRANSFER_SIGNAL] == evidence


def _persist_direct_confirmation(proj, sel, seg):
    cfg = load_clip_config()
    prompted = [1.0, 2.0, 1.0]
    binding, reason = R._quote_confirmation_binding(
        proj, proj.source(sel.source_id), seg.quote, prompted, cfg,
        exact_contiguous_required=True)
    assert reason == "" and binding is not None
    artifact = {
        "schema_version": R.QUOTE_CONFIRMATION_SCHEMA,
        "algorithm": R.QUOTE_CONFIRMATION_ALGORITHM,
        "binding": binding,
        "status": "confirmed",
        "reason": "",
        "decoder_fingerprint": binding["decoder_fingerprint"],
        "decode_window": binding["decode_window"],
        "prompted_span": binding["prompted_span"],
        "timed_words": [[1.0, 1.35, "He's"], [1.35, 2.0, "choking"]],
        "segment_confidence": [{
            "no_speech_prob": .05, "avg_logprob": -.1, "accepted": True,
        }],
        "confirmed_span": [1.0, 2.0, 1.0],
        "timed_asr_ratio": 1.0,
        "match_method": "exact_contiguous_timed_asr+unprompted_confirmation",
        "decoder_used": "primary",
        "primary_model": str(cfg.whisper_model),
        "rescue_model": R.QUOTE_CONFIRMATION_RESCUE_MODEL,
        "primary_decode_status": "ok",
        "primary_decode_reason": "",
        "primary_phrase_matched": True,
        "rescue_attempted": False,
        "rescue_decode_status": "",
        "rescue_decode_reason": "",
        "rescue_phrase_matched": False,
    }
    artifact["result_content_sha256"] = R._quote_confirmation_result_sha256(artifact)
    path = R._quote_confirmation_artifact_path(proj, binding)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    enriched = {
        **artifact,
        "artifact_key": binding["binding_fingerprint"],
        "artifact_path": R._quote_confirmation_artifact_display_path(proj, path),
    }
    return R._quote_confirmation_summary(enriched), path


def test_ledger_preserves_valid_direct_confirmation_outside_numeric_signals(tmp_path):
    proj, seg, sel = _project(tmp_path)
    sel.source_id = "dirty"
    sel.in_point, sel.out_point = .5, 2.5
    summary, _path = _persist_direct_confirmation(proj, sel, seg)
    sel.signals = {"dialogue": 1.0, "quote_unprompted_confirmation": summary}

    L.write_ledger(proj, [seg])

    record = json.loads(proj.ledger_path.read_text().strip())
    assert record["signals"] == {"dialogue": 1.0}
    assert record["quote_unprompted_confirmation"] == summary


@pytest.mark.parametrize("mutation", [
    "summary", "result_hash", "source", "quote", "path_traversal", "missing_artifact",
])
def test_ledger_rejects_stale_or_fabricated_direct_confirmation(
        tmp_path, mutation):
    proj, seg, sel = _project(tmp_path)
    sel.source_id = "dirty"
    sel.in_point, sel.out_point = .5, 2.5
    summary, path = _persist_direct_confirmation(proj, sel, seg)
    summary = copy.deepcopy(summary)
    if mutation == "summary":
        summary["timed_asr_ratio"] = .99
    elif mutation == "result_hash":
        raw = json.loads(path.read_text())
        raw["result_content_sha256"] = "0" * 64
        path.write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "source":
        sel.source_id = "clean"
    elif mutation == "quote":
        seg.quote = "Different authored quote."
    elif mutation == "path_traversal":
        summary["artifact_path"] = "../project.json"
    elif mutation == "missing_artifact":
        path.unlink()
    sel.signals = {"quote_unprompted_confirmation": summary}

    with pytest.raises(ValueError, match="quote_unprompted_confirmation"):
        L.write_ledger(proj, [seg])


def test_ledger_rejects_malformed_or_unrecognized_structured_signal_evidence(tmp_path):
    proj, seg, sel = _project(tmp_path)

    sel.signals = {A.AUDIO_QUOTE_TRANSFER_SIGNAL: {"fabricated": True}}
    with pytest.raises(ValueError, match="evidence_schema_mismatch"):
        L.write_ledger(proj, [seg])

    # The named, shape-validated quote proof is the only structured exception.  An arbitrary
    # object or list remains a hard QC error instead of being discarded or treated as truthy.
    for unknown in ({"fabricated": True}, ["fabricated"]):
        sel.signals = {"unknown_structured_evidence": unknown}
        with pytest.raises(TypeError):
            L.write_ledger(proj, [seg])


def test_contract_rejects_stale_low_nonunique_and_fabricated_transfer_evidence(tmp_path):
    proj, seg, sel = _project(tmp_path)
    sel.in_point, sel.out_point, sel.shot_index = 3.5, 5.5, 1
    contract = _contract(proj)

    stale = _evidence(proj)
    proj.source("clean").local_path += ".replaced"
    sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL] = stale
    ok, reason, detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=contract)
    assert ok is False and reason == "exact_quote_audio_transfer_evidence_invalid"
    assert detail["audio_transfer_status"] == "target_source_content_fingerprint_mismatch"
    proj.source("clean").local_path = str(tmp_path / "clean_hd.mp4")

    for alignment, expected in [
        (_alignment(correlation=.89, runner=.10, margin=.79),
         "alignment_correlation_below_floor"),
        (_alignment(correlation=.95, runner=.90, margin=.05),
         "alignment_uniqueness_below_floor"),
    ]:
        sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL] = _evidence(proj, alignment=alignment)
        ok, reason, detail = R.exact_quote_dialogue_evidence(
            proj, sel, seg, quote_contract=contract)
        assert ok is False and reason == "exact_quote_audio_transfer_evidence_invalid"
        assert detail["audio_transfer_status"] == expected

    fabricated = copy.deepcopy(_evidence(proj))
    fabricated["correlation"] = 1.0  # persisted record edited without its deterministic binding
    sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL] = fabricated
    ok, reason, detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=contract)
    assert ok is False and reason == "exact_quote_audio_transfer_evidence_invalid"
    assert detail["audio_transfer_status"] == "evidence_binding_fingerprint_mismatch"


def test_contract_does_not_use_transfer_to_bypass_dialogue_floor_or_quote_typing(tmp_path):
    proj, seg, sel = _project(tmp_path)
    sel.in_point, sel.out_point, sel.shot_index = 3.5, 5.5, 1
    sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL] = _evidence(proj)
    sel.signals["dialogue"] = R.QUOTE_DIALOGUE_FLOOR - .01
    ok, reason, _detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=_contract(proj))
    assert ok is False and reason == "exact_quote_dialogue_signal_below_floor"

    sel.signals["dialogue"] = .98
    unknown = {**_contract(proj), "branch": "indeterminate"}
    ok, reason, _detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=unknown)
    assert ok is False and reason == "exact_quote_pool_classification_indeterminate"


def test_quote_recovery_transfers_sd_reference_only_into_existing_hd_bench(tmp_path):
    proj, seg, old = _project(tmp_path)
    contract = _contract(proj)

    def dimensions(path):
        return ({"width": 640, "height": 360} if path.name == "dirty_sd.mp4"
                else {"width": 1920, "height": 1080})

    with mock.patch.object(R, "_quote_pool_branches", return_value={0: contract}), \
            mock.patch.object(R, "_confirm_prompted_quote_span_unprompted",
                              side_effect=_confirmed_revalidation), \
            mock.patch("vidlore.clipstudio.ingest.probe", side_effect=dimensions), \
            mock.patch.object(A, "transfer_quote_spans", return_value=[_alignment()]):
        built, audit = O._quote_window_recovery_selections(
            proj, [seg], ClipConfig(), {0})

    assert set(built) == {0}
    selected = built[0]
    assert selected.source_id == "clean"
    assert selected.shot_index == 1
    assert selected.in_point <= 4.0 and selected.out_point >= 5.0
    evidence = selected.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL]
    assert evidence["reference_source_id"] == "dirty"
    assert evidence["target_source_id"] == "clean"
    assert evidence["target_selected_window"] == [selected.in_point, selected.out_point]
    row = next(row for row in audit["beats"][0]["candidates"]
               if row["source_id"] == "clean")
    assert row["status"] == "candidate"
    assert row["quote_location_method"] == "cross_copy_pcm"
    assert audit["beats"][0]["candidate_bench_cap"] == 12
    assert audit["beats"][0]["audio_transfer_target_source_cap"] == 12


def test_quote_recovery_rejects_a_legacy_prompt_only_pcm_reference(tmp_path):
    proj, seg, _old = _project(tmp_path)
    contract = _contract(proj)
    contract["pool_match"].pop("unprompted_confirmation")
    contract["pool_matches"][0].pop("unprompted_confirmation")

    def dimensions(path):
        return ({"width": 640, "height": 360} if path.name == "dirty_sd.mp4"
                else {"width": 1920, "height": 1080})

    with mock.patch.object(R, "_quote_pool_branches", return_value={0: contract}), \
            mock.patch("vidlore.clipstudio.ingest.probe", side_effect=dimensions), \
            mock.patch.object(A, "transfer_quote_spans", return_value=[]):
        built, audit = O._quote_window_recovery_selections(
            proj, [seg], ClipConfig(), {0})

    assert built == {}
    assert audit["beats"][0]["audio_transfer_reference_count"] == 0
    assert audit["beats"][0]["audio_transfer_reference_rejections"] == [{
        "source_id": "dirty",
        "reason": "unprompted_confirmation_absent_or_not_confirmed",
    }]


def test_short_quote_fuzzy_only_target_still_reaches_strict_pcm_transfer(tmp_path):
    proj, seg, _old = _project(tmp_path)
    # Fuzzy phrase retrieval can assemble these two quote tokens across an unrelated word, but the
    # v10 short-quote contract correctly refuses to call that verbatim timed ASR.
    (proj.index_dir / "clean.words.json").write_text(json.dumps([
        [1.0, 1.3, "He's"], [1.3, 1.5, "unrelated"], [1.5, 1.8, "choking"],
    ]))
    assert IX.find_quote_span(IX.load_words(proj, "clean"), QUOTE,
                              min_ratio=R.QUOTE_DIALOGUE_FLOOR)
    assert R._exact_contiguous_quote_span(IX.load_words(proj, "clean"), QUOTE) is None

    def dimensions(path):
        return ({"width": 640, "height": 360} if path.name == "dirty_sd.mp4"
                else {"width": 1920, "height": 1080})

    with mock.patch.object(R, "_quote_pool_branches", return_value={0: _contract(proj)}), \
            mock.patch.object(R, "_confirm_prompted_quote_span_unprompted",
                              side_effect=_confirmed_revalidation), \
            mock.patch("vidlore.clipstudio.ingest.probe", side_effect=dimensions), \
            mock.patch.object(A, "transfer_quote_spans", return_value=[_alignment()]):
        built, audit = O._quote_window_recovery_selections(
            proj, [seg], ClipConfig(), {0})

    assert built[0].source_id == "clean"
    beat = audit["beats"][0]
    assert "clean" not in beat["audio_transfer_direct_asr_duplicates_filtered"]
    assert not any(row.get("source_id") == "clean" and
                   row.get("reason") == "target_already_has_timed_asr_quote"
                   for row in beat["audio_transfer_pre_cap_excluded"])


def test_quote_locked_verifier_skips_out_of_phrase_neighbor_and_rebinds_contained_one(
        tmp_path, monkeypatch):
    proj, seg, sel = _project(tmp_path)
    sel.shot_index, sel.in_point, sel.out_point = 1, 3.5, 5.5
    sel.signals = {
        "dialogue": 1.0,
        "quote_pool_exact": True,
        A.AUDIO_QUOTE_TRANSFER_SIGNAL: _evidence(proj),
    }
    contained_frame = tmp_path / "clean_2.jpg"
    contained_frame.write_bytes(b"contained quote action frame")
    clean_shots = json.loads(proj.shots_path("clean").read_text())
    clean_shots.append(Shot(
        source_id="clean", index=2, start=3.6, end=5.5,
        keyframe_path=str(contained_frame), quality=.9).to_dict())
    proj.shots_path("clean").write_text(json.dumps(clean_shots))
    bad = ClipCandidate(
        segment_index=0, source_id="clean", shot_index=0, score=.99,
        in_point=0.0, out_point=2.0, signals={"dialogue": 0.0})
    contained = ClipCandidate(
        segment_index=0, source_id="clean", shot_index=2, score=.90,
        in_point=3.6, out_point=5.5, signals={"dialogue": 0.0})
    monkeypatch.setattr(V, "_strict_scene_neighborhood_candidates",
                        lambda *_a, **_k: [bad, contained])
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "1")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET", "0")
    monkeypatch.setenv("VIDLORE_CLIPSTUDIO_WINDOW_QC", "0")
    calls = []

    def verdict(path, *_args, **_kwargs):
        calls.append(str(path))
        keep = str(path).endswith("clean_2.jpg")
        return {
            "verdict": "keep" if keep else "replace",
            "matches_narration": keep,
            "correct_subject_visible": keep,
            "wrong_subject_visible": False,
            "contradicts_narration": False,
            "era_ok": True,
            "specific_enough": keep,
            "quality_ok": True,
            "confidence": .95,
            "reason": "exact action" if keep else "reaction only",
        }

    monkeypatch.setattr(V, "verify_frame", verdict)
    summary = V.verify_and_repair(
        proj, [seg], ClipConfig(),
        SimpleNamespace(anthropic_model="vision", anthropic_key="key"),
        materialize_promotions=False, persist_project=False)

    assert summary["replaced"] == 1
    assert len(calls) == 2, "the out-of-quote neighbor must be rejected before vision"
    assert (sel.shot_index, sel.in_point, sel.out_point) == (2, 3.6, 5.5)
    assert sel.signals["quote_pool_exact"] is True
    assert sel.signals["dialogue"] == 1.0
    rebound = sel.signals[A.AUDIO_QUOTE_TRANSFER_SIGNAL]
    assert rebound["target_selected_window"] == [3.6, 5.5]
    ok, reason, _detail = R.exact_quote_dialogue_evidence(
        proj, sel, seg, quote_contract=_contract(proj))
    assert ok is True and reason == ""


def test_direct_asr_duplicates_are_filtered_before_audio_target_cap(tmp_path):
    proj, seg, old = _project(tmp_path)
    direct_ids = ["dirty"]
    for number in range(1, 12):
        sid = f"dirty_{number:02d}"
        direct_ids.append(sid)
        media = tmp_path / f"{sid}.mp4"
        media.write_bytes(f"sd-{sid}".encode())
        proj.sources.append(SourceVideo(
            id=sid, url=f"u-{sid}", title=f"SD direct quote {sid}", permission="owner",
            status="ok", local_path=str(media), duration=10.0, width=640, height=360))
        frame = tmp_path / f"{sid}.jpg"
        frame.write_bytes(f"frame-{sid}".encode())
        proj.shots_path(sid).write_text(json.dumps([
            Shot(source_id=sid, index=0, start=.5, end=2.5,
                 keyframe_path=str(frame), quality=.8).to_dict(),
        ]))
        (proj.index_dir / f"{sid}.words.json").write_text(json.dumps([
            [1.0, 1.35, "He's"], [1.35, 2.0, "choking"],
        ]))
        _stamp(proj, sid)

    # All twelve leading bench sources already have direct whole-pool ASR. The only transfer target
    # is the clean copy at position thirteen; duplicates must not spend its bounded slot.
    old.source_id = direct_ids[0]
    old.shot_index, old.in_point, old.out_point = 0, .5, 2.5
    old.alternates = [ClipCandidate(
        segment_index=0, source_id=sid, shot_index=0, in_point=.5, out_point=2.5)
        for sid in direct_ids[1:]]
    # A second twelve-entry prefix is deterministically dead before any PCM work: eleven missing
    # sources and one local source whose ASR generation is stale. Neither group may spend a target
    # opportunity either.
    dead_ids = [f"missing_{number:02d}" for number in range(11)] + ["stale_asr"]
    stale_media = tmp_path / "stale_asr.mp4"
    stale_media.write_bytes(b"stale-asr-media")
    proj.sources.append(SourceVideo(
        id="stale_asr", url="u-stale", title="stale ASR clean copy", permission="owner",
        status="ok", local_path=str(stale_media), duration=10.0, width=1920, height=1080))
    (proj.index_dir / "stale_asr.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": True,
        "asr_prompt_fingerprint": "stale-generation",
    }))
    old.deep_alternates = [ClipCandidate(
        segment_index=0, source_id=sid, shot_index=0, in_point=0.0, out_point=2.0)
        for sid in dead_ids]
    old.deep_alternates.append(ClipCandidate(
        segment_index=0, source_id="clean", shot_index=1, in_point=3.5, out_point=5.5))
    contract = _contract(proj)
    contract["pool_matches"] = [{
        "source_id": sid, "source_title": sid,
        "timed_asr_span": [1.0, 2.0], "timed_asr_ratio": 1.0,
        "prompted_asr_span": [1.0, 2.0, 1.0],
        "unprompted_confirmation": _confirmation(proj, source_id=sid),
    } for sid in direct_ids]
    contract["pool_match_count"] = len(direct_ids)

    def dimensions(path):
        return ({"width": 640, "height": 360} if path.stem.startswith("dirty")
                else {"width": 1920, "height": 1080})

    def batch(references, _target_path, *, target_search_window):
        assert target_search_window == [0.0, 10.0]
        return [_alignment() for _reference in references]

    with mock.patch.object(R, "_quote_pool_branches", return_value={0: contract}), \
            mock.patch.object(R, "_confirm_prompted_quote_span_unprompted",
                              side_effect=_confirmed_revalidation), \
            mock.patch("vidlore.clipstudio.ingest.probe", side_effect=dimensions), \
            mock.patch.object(A, "transfer_quote_spans", side_effect=batch):
        built, audit = O._quote_window_recovery_selections(
            proj, [seg], ClipConfig(), {0})

    beat = audit["beats"][0]
    assert built[0].source_id == "clean"
    assert beat["audio_transfer_bench_source_count"] == 25
    assert beat["audio_transfer_target_source_count"] == 1
    assert beat["audio_transfer_target_source_overflow"] == 0
    assert set(beat["audio_transfer_direct_asr_duplicates_filtered"]) == set(direct_ids)
    excluded = {row["source_id"]: row["reason"]
                for row in beat["audio_transfer_pre_cap_excluded"]}
    assert set(excluded) == set(dead_ids)
    assert all(excluded[sid] == "source_unavailable" for sid in dead_ids[:-1])
    assert excluded["stale_asr"].startswith("target_asr_provenance_invalid:")
