"""Fail-closed provenance contracts for ClipStudio scene footage.

The matcher/verifier chooses one window per narration beat.  Any later visual
derivative may crop, grade, zoom, or freeze that window, but it must not quietly
change the owner, source, or selected window.  These helpers intentionally use
construction provenance as the primary authority; decoded-frame canaries in the
renderer provide an independent second line of defence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


_ORDINARY_KINDS = {"selection_video", "selection_derivative"}
_SPECIAL_KINDS = {"verified_image", "breakout"}


def selection_binding(original_beat: int, source_id: str, in_point: float,
                      out_point: float, verifier: dict | None = None) -> str:
    """Stable identity of the verified selection a derivative is allowed to use."""
    vd = verifier or {}
    payload = {
        "original_beat": int(original_beat),
        "source_id": str(source_id or ""),
        "in": round(float(in_point or 0.0), 3),
        "out": round(float(out_point or 0.0), 3),
        "verdict": str(vd.get("verdict") or ""),
        "status": str(vd.get("status") or ""),
        "served_by": str(vd.get("vision_served_by") or ""),
        "relevance_class": str(vd.get("relevance_class") or ""),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_scene_lineage(entries: list[dict]) -> list[dict]:
    """Return every lineage violation; an empty result is a publishable contract.

    Required facts are deliberately redundant.  A filename such as
    ``beat_034_0.mp4`` is not provenance, and neither is an audit label saying
    ``window[0]``.  The owner, selected tuple, and immutable selection binding
    must all agree.
    """
    problems: list[dict] = []
    for pos, rec in enumerate(entries or []):
        kind = str(rec.get("kind") or "")
        base = {
            "entry": pos,
            "final_scene": rec.get("final_scene"),
            "original_beat": rec.get("original_beat"),
            "clip": rec.get("clip"),
            "file": rec.get("file"),
        }

        def bad(reason: str) -> None:
            problems.append({**base, "reason": reason})

        if kind not in (_ORDINARY_KINDS | _SPECIAL_KINDS):
            bad(f"unknown/untracked lineage kind {kind!r}")
            continue
        try:
            owner = int(rec.get("owner_beat"))
            original = int(rec.get("original_beat"))
        except (TypeError, ValueError):
            bad("missing/non-integer original beat or root owner")
            continue
        if owner != original:
            bad(f"root belongs to beat {owner}, not this beat {original}")

        path = str(rec.get("file") or "")
        if not path or not Path(path).exists() or Path(path).stat().st_size <= 0:
            bad(f"aired clip is missing/empty ({path or '?'})")

        if kind == "breakout":
            if rec.get("validated") is not True:
                bad("breakout root is not validated")
            continue
        if kind == "verified_image":
            if rec.get("validated") is not True:
                bad("image root is not verifier-validated")
            if not rec.get("root_binding"):
                bad("verified image has no immutable root binding")
            owner_kind = str(rec.get("image_owner_kind") or "")
            if owner_kind not in {"source_frame", "verified_file"}:
                bad(f"verified image has unknown owner kind {owner_kind!r}")
            if not str(rec.get("image_sha256") or ""):
                bad("verified image has no immutable source-image hash")
            semantic_bound = rec.get("semantic_binding_preserved") is True
            rematerialized = rec.get("semantic_rematerialized") is True
            if rematerialized and not semantic_bound:
                bad("native semantic still is not marked as immutably bound")
            if semantic_bound:
                semantic_hash = str(rec.get("semantic_image_sha256") or "")
                if not semantic_hash or semantic_hash != str(rec.get("image_sha256") or ""):
                    bad("strict semantic still hash does not match the exact aired bytes")
                if rematerialized:
                    if owner_kind != "source_frame":
                        bad("native semantic rematerialization has no indexed source-frame owner")
                    if rec.get("preserved_original") is not False:
                        bad("native semantic rematerialization incorrectly claims original bytes")
                    model = str(rec.get("semantic_model") or "")
                    question_fp = str(rec.get("semantic_question_fingerprint") or "")
                    source_fp = str(rec.get("semantic_source_content_fingerprint") or "")
                    if not model or model == "none":
                        bad("native semantic rematerialization has no verifier model identity")
                    if not question_fp:
                        bad("native semantic rematerialization has no question fingerprint")
                    if not source_fp:
                        bad("native semantic rematerialization has no source-content fingerprint")
                elif rec.get("preserved_original") is not True:
                    bad("strict semantic still hash does not match the exact preserved aired bytes")
            try:
                iw, ih = int(rec.get("image_width") or 0), int(rec.get("image_height") or 0)
                short, long = sorted((iw, ih))
                if short < 720 or long < 1280:
                    bad(f"verified image is not publishable native resolution ({iw}x{ih})")
            except (TypeError, ValueError, OverflowError):
                bad("verified image dimensions are malformed")
            if owner_kind == "source_frame":
                expected_sid = str(rec.get("expected_image_source_id") or "")
                actual_sid = str(rec.get("actual_image_source_id") or "")
                expected_shot = rec.get("expected_image_shot_index")
                actual_shot = rec.get("actual_image_shot_index")
                try:
                    same_shot = int(expected_shot) == int(actual_shot)
                except (TypeError, ValueError, OverflowError):
                    same_shot = False
                if not expected_sid or actual_sid != expected_sid or not same_shot:
                    bad(f"verified image owner changed from {expected_sid!r} shot "
                        f"{expected_shot!r} to {actual_sid!r} shot {actual_shot!r}")
                try:
                    sw = int(rec.get("source_native_width") or 0)
                    sh = int(rec.get("source_native_height") or 0)
                    short, long = sorted((sw, sh))
                    if short < 720 or long < 1280:
                        bad("source-frame image owner is below native 1280x720 or unprobeable")
                except (TypeError, ValueError, OverflowError):
                    bad("source-frame owner native dimensions are malformed")
                try:
                    if float(rec.get("actual_image_time")) < 0:
                        bad("source-frame image time is missing/malformed")
                except (TypeError, ValueError, OverflowError):
                    bad("source-frame image time is missing/malformed")
            elif owner_kind == "verified_file":
                if rec.get("preserved_original") is not True:
                    bad("metadata-free/web verified image was not preserved byte-for-byte")
                if rec.get("actual_image_source_id") not in (None, "") \
                        or rec.get("actual_image_shot_index") is not None:
                    bad("verified-file image invented a source-frame owner")
            continue

        # Ordinary moving footage has no sanctioned alternate/walk path.  A
        # separately verified candidate must first become the selection itself;
        # only then does it receive a new binding and become airable.
        via = str(rec.get("via") or "")
        if via not in ("selection", "selection_derivative"):
            bad(f"ordinary footage arrived via unverified path {via!r}")
        selected_sid = str(rec.get("selected_source_id") or "")
        actual_sid = str(rec.get("actual_source_id") or "")
        if not selected_sid or actual_sid != selected_sid:
            bad(f"source changed from selected {selected_sid!r} to {actual_sid!r}")
        source_path = str(rec.get("selection_source_path") or "")
        if (not source_path or not Path(source_path).is_file()
                or Path(source_path).stat().st_size <= 0):
            bad("verified selection source bytes are missing/empty")
        selected_window = rec.get("selected_window") or []
        actual_window = rec.get("actual_window") or []
        if len(selected_window) < 3 or len(actual_window) < 3:
            bad("selected/actual source window is missing")
        else:
            try:
                if (str(actual_window[0]) != str(selected_window[0])
                        or abs(float(actual_window[1]) - float(selected_window[1])) > 0.051
                        or abs(float(actual_window[2]) - float(selected_window[2])) > 0.051):
                    bad(f"actual window {actual_window!r} is not the verified selection "
                        f"{selected_window!r}")
            except (TypeError, ValueError):
                bad("selected/actual source window is malformed")
        root = str(rec.get("root_binding") or "")
        expected = str(rec.get("selection_binding") or "")
        if not expected or root != expected:
            bad("immutable root binding does not match this beat's verified selection")
        if rec.get("validated") is not True:
            bad("ordinary scene derivative was not validated")
    return problems


def assert_scene_lineage(entries: list[dict], audit_path: Path) -> dict:
    """Persist the complete contract and raise on *any* mismatch or audit-write failure."""
    problems = validate_scene_lineage(entries)
    payload = {
        "schema": "scene_lineage/1",
        "passed": not problems,
        "entries": entries,
        "failures": problems,
    }
    # Audit persistence is part of the invariant.  A render whose provenance
    # cannot be recorded must not silently look publishable.
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = audit_path.with_name(audit_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, audit_path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    if problems:
        from .verify import NonRetryableBuildError
        raise NonRetryableBuildError(
            f"scene-lineage gate: {len(problems)} clip(s) do not derive from their own "
            f"verified selection (first: {problems[0]['reason']}); see {audit_path.name}",
            kind="scene_lineage")
    return payload
