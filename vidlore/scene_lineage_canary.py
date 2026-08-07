"""Decoded-frame canaries for the assembly scene-lineage contract.

The construction manifest is the primary authority: every encode-plan row must
name the exact file it is allowed to read.  Small, centre-cropped perceptual
fingerprints then independently prove that (a) the encoder did not silently
produce another beat and (b) the conformed concat kept those beats in order.

This module is deliberately renderer-agnostic and has no ClipStudio imports.
The contract is opt-in at :func:`vidlore.assemble.assemble`; once supplied it is
strict and fail-closed, including audit persistence and frame extraction.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .ffmpeg_tool import ffmpeg_exe


_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".ts"}
# Expected-moment cadence for the decoded fingerprint banks: never leave more than this much of a
# scene unsampled, bounded so a long scene cannot make the canary itself the slow stage.
_BANK_MAX_GAP_SEC = 0.40
_BANK_MAX_FRAMES = 24
_FEATURE_W = 17
_FEATURE_H = 16
_FEATURE_BYTES = _FEATURE_W * _FEATURE_H * 3
_AUTHORIZED_SOURCE_COMPARE_FILTERS = {
    "crop=iw*0.840:ih*0.840:0:0",
    "crop=iw*0.840:ih*0.840:0:ih*0.160",
    "crop=iw*0.840:ih*0.840:iw*0.160:0",
    "crop=iw*0.840:ih*0.840:iw*0.160:ih*0.160",
}

# AUTHORIZED TRANSFORM SCHEMA
#
# Lineage compares a delivered frame against its source window, and the pipeline is allowed to have
# punched a corner crop into that frame on the way. So the comparison has to apply the SAME crop —
# which means the crop has to be declared, and a declared transform is an input an attacker (or a
# bug) could use to make foreign pixels compare equal. The original defence was a set of four
# literal filter strings: airtight, and brittle in a way that matters — an identical crop written
# `iw*0.84` instead of `iw*0.840` is the same geometry and was refused, so any downstream change to
# how the filter is formatted silently turned a provable frame into a blocked render.
#
# Generalised WITHOUT widening what is permitted: the declaration is now PARSED into canonical
# geometry and checked against this schema, rather than string-matched. What survives is exactly
# what survived before — a pure corner crop, at an authorized fraction, anchored to a real corner:
#   * only `crop`; no filter graph, no chaining, no second filter, no arbitrary expression
#   * the fraction must be one this pipeline actually produces (_AUTHORIZED_CROP_FRACTIONS)
#   * width and height fractions must be equal (a corner bug crop is square in fraction terms)
#   * the offsets must be exactly 0 or exactly the complement (1 - fraction) on their own axis
# Anything else — a different fraction, a shifted origin, a scale, an eq, a drawbox, a second
# clause, or a filter naming a file — parses to None and is rejected exactly as before.
_AUTHORIZED_CROP_FRACTIONS = (0.840,)
_CROP_RX = re.compile(
    r"^crop=iw\*(?P<w>[0-9]*\.?[0-9]+):ih\*(?P<h>[0-9]*\.?[0-9]+)"
    r":(?P<x>0|iw\*[0-9]*\.?[0-9]+):(?P<y>0|ih\*[0-9]*\.?[0-9]+)$")


def canonical_source_compare_transform(filter_expr: str):
    """Canonical geometry for an authorized corner crop, or None if it is not one.

    None means "refuse" at every call site. This never executes the expression; it only decides
    whether the declaration describes a transform the pipeline is allowed to have applied."""
    text = str(filter_expr or "").strip()
    if not text:
        return None
    m = _CROP_RX.match(text)
    if not m:
        return None
    try:
        w = float(m.group("w"))
        h = float(m.group("h"))
    except (TypeError, ValueError):
        return None
    frac = next((f for f in _AUTHORIZED_CROP_FRACTIONS if abs(w - f) <= 1e-6), None)
    if frac is None or abs(w - h) > 1e-6:
        return None
    comp = round(1.0 - frac, 6)

    def _axis(raw: str, unit: str):
        if raw == "0":
            return 0.0
        try:
            val = float(raw.split("*", 1)[1])
        except (IndexError, TypeError, ValueError):
            return None
        return val if abs(val - comp) <= 1e-6 else None

    x = _axis(m.group("x"), "iw")
    y = _axis(m.group("y"), "ih")
    if x is None or y is None:
        return None
    return {"schema": "source_compare_transform/1", "kind": "corner_crop",
            "w_frac": round(frac, 6), "h_frac": round(frac, 6),
            "x_frac": round(x, 6), "y_frac": round(y, 6)}


def source_compare_filter_authorized(filter_expr: str) -> bool:
    """True only for a declaration this pipeline is allowed to have applied."""
    return canonical_source_compare_transform(filter_expr) is not None


class SceneLineageError(RuntimeError):
    """An assembled frame cannot be proved to belong to its planned beat."""


def new_audit(output: Path) -> dict:
    return {
        "schema": "assemble_scene_lineage/1",
        "status": "running",
        "stage": "initializing",
        "output": str(Path(output).resolve()),
        "binding": [],
        "encoded_segments": [],
        "timeline_order": [],
        "failures": [],
    }


def write_audit(path: Path, payload: dict) -> None:
    """Atomically persist the canary record; an unwritable audit is a failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        # os.replace removes the temp file on success.  On failure, best-effort
        # cleanup never masks the original persistence exception.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def fail_audit(path: Path, payload: dict, stage: str,
               failures: list[dict]) -> None:
    payload["status"] = "failed"
    payload["stage"] = stage
    payload["failures"] = list(payload.get("failures") or []) + list(failures)
    write_audit(path, payload)
    first = failures[0].get("reason", "unknown lineage failure") if failures else "unknown"
    raise SceneLineageError(
        f"scene-lineage canary failed at {stage}: {len(failures)} violation(s); "
        f"first: {first}; see {Path(path).name}")


def _first(row: dict, *names: str):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _as_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _path(value) -> Path | None:
    if value in (None, ""):
        return None
    try:
        return Path(str(value)).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: Path) -> str:
    """Stream a media identity without loading a full clip into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_hint(value) -> tuple[int | None, int | None]:
    if isinstance(value, tuple) and len(value) >= 2:
        return _as_int(value[0]), _as_int(value[1])
    m = re.fullmatch(r"\s*(-?\d+)\s*[:/,]\s*(\d+)\s*", str(value))
    if m:
        return int(m.group(1)), int(m.group(2))
    return _as_int(value), None


def _expectation_rows(raw: Any) -> list[dict]:
    """Normalize list, ``{entries:[...]}``, and scene-keyed contracts.

    ClipStudio's build manifest uses ``final_scene``/``clip``/``file``;
    generic callers may use ``scene_index``/``beat``/``input_path``.  Both
    become one exact ``(scene, beat, path, kind)`` ownership tuple.
    """
    rows: list[dict] = []

    def add(value, scene_hint=None, beat_hint=None, parent=None) -> None:
        if not isinstance(value, dict):
            value = {"input_path": value}
        merged = dict(parent or {})
        merged.update(value)
        merged.pop("beats", None)
        scene = _as_int(_first(
            merged, "scene_index", "final_scene", "scene", "original_beat",
            "owner_beat"), scene_hint)
        beat = _as_int(_first(merged, "beat", "beat_index", "clip", "m"), beat_hint)
        beat = 0 if beat is None else beat
        input_path = _path(_first(
            merged, "encoded_input_path", "input_path", "path", "file", "clip_path"))
        kind = str(_first(merged, "kind", "lineage_kind", "source_kind") or "")
        media_kind = str(_first(merged, "media_kind") or "").lower()
        rows.append({
            "scene": scene,
            "beat": beat,
            "input_path": input_path,
            "kind": kind,
            "media_kind": media_kind,
            "raw": merged,
        })

    if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        raw = raw["entries"]
    elif isinstance(raw, dict) and isinstance(raw.get("lineage_expectations"), list):
        raw = raw["lineage_expectations"]

    if isinstance(raw, (list, tuple)):
        for value in raw:
            add(value)
        return rows

    if isinstance(raw, dict):
        # A single record is also valid.
        if any(k in raw for k in (
                "input_path", "encoded_input_path", "path", "file", "clip_path")):
            add(raw)
            return rows
        for key, value in raw.items():
            scene_hint, beat_hint = _key_hint(key)
            if isinstance(value, dict) and isinstance(value.get("beats"), (list, tuple)):
                parent = dict(value)
                beats = parent.pop("beats")
                for pos, child in enumerate(beats):
                    add(child, scene_hint, pos, parent)
            elif isinstance(value, (list, tuple)):
                for pos, child in enumerate(value):
                    add(child, scene_hint, pos)
            else:
                add(value, scene_hint, beat_hint)
        return rows

    return rows


def bind_encode_plan(encode_plan: list[dict], raw: Any) -> tuple[list[dict], list[dict]]:
    """Bind every planned beat to an exact declared input file and kind.

    This is the primary invariant.  Perceptual hashes are evidence of decoded
    continuity, never a substitute for provenance/path ownership.
    """
    expected = _expectation_rows(raw)
    failures: list[dict] = []
    lookup: dict[tuple[int, int], dict] = {}
    duplicate_keys: set[tuple[int, int]] = set()
    for row in expected:
        key = (row.get("scene"), row.get("beat"))
        if key[0] is None:
            failures.append({"stage": "binding", "reason": "expectation has no scene owner"})
            continue
        if key in lookup:
            duplicate_keys.add(key)
            failures.append({
                "stage": "binding", "scene": key[0], "beat": key[1],
                "reason": "duplicate lineage expectation",
            })
            continue
        lookup[key] = row

    audit_rows: list[dict] = []
    # Several beats may point into one source file.  Cache only the small
    # decoded bank for an exact source window; never hash a multi-GB source once
    # per beat.
    source_bank_cache: dict[tuple[str, float, float, str], list[dict]] = {}
    consumed: set[tuple[int, int]] = set()
    for plan_row in encode_plan:
        scene = _as_int(plan_row.get("scene_index"), _as_int(plan_row.get("j")))
        beat = _as_int(plan_row.get("m"), 0)
        key = (scene, beat)
        exp = lookup.get(key)
        actual = _path(plan_row.get("mg_clip") or getattr(plan_row.get("item"), "path", None))
        record = {
            "bi": _as_int(plan_row.get("bi")),
            "scene": scene,
            "beat": beat,
            "kind": exp.get("kind") if exp else "",
            "expected_input": str(exp.get("input_path")) if exp and exp.get("input_path") else None,
            "planned_input": str(actual) if actual else None,
            "passed": True,
        }
        reasons: list[str] = []
        if exp is None:
            reasons.append("no lineage expectation for planned beat")
        else:
            consumed.add(key)
            expected_path = exp.get("input_path")
            if not exp.get("kind"):
                reasons.append("lineage expectation has no kind")
            if expected_path is None:
                reasons.append("lineage expectation has no input path")
            if actual is None:
                reasons.append("encode plan has no readable input path")
            if expected_path is not None and actual is not None and expected_path != actual:
                reasons.append("planned input path differs from declared lineage input")
            if expected_path is not None and (
                    not expected_path.exists() or expected_path.stat().st_size <= 0):
                reasons.append("declared lineage input is missing or empty")
            if actual is not None and (not actual.exists() or actual.stat().st_size <= 0):
                reasons.append("planned encode input is missing or empty")
            mk = exp.get("media_kind")
            if mk in {"video", "image"} and actual is not None:
                actual_kind = "video" if actual.suffix.lower() in _VIDEO_EXTS else "image"
                if mk != actual_kind:
                    reasons.append(
                        f"declared media kind {mk!r} differs from planned {actual_kind!r}")
            if not reasons:
                try:
                    # Freeze both byte identity and decoded visual identity NOW,
                    # before the parallel encoder can read or overwrite anything.
                    # Re-reading the same path after encode is not independent
                    # evidence: a foreign replacement would otherwise be both the
                    # encoded input and the later "expected" sample.
                    src_start = (float(plan_row.get("mg_off") or 0.0)
                                 if plan_row.get("mg_clip") else 0.0)
                    src_window = (float(plan_row.get("bd") or 0.0)
                                  + float(plan_row.get("pad") or 0.0))
                    bound_times = _uniform_times(
                        actual, 5, start=src_start, duration=src_window)
                    bound_features = _features_at_times(actual, bound_times)
                    bound_sha = _sha256(actual)
                    raw_row = exp.get("raw") or {}
                    source_path = _path(raw_row.get("selection_source_path"))
                    source_comparison = None
                    if raw_row.get("selected_source_id"):
                        selected_window = raw_row.get("selected_window") or []
                        if source_path is None or not source_path.is_file() \
                                or source_path.stat().st_size <= 0:
                            raise SceneLineageError(
                                "verified selection source bytes are missing/empty")
                        if len(selected_window) < 3:
                            raise SceneLineageError(
                                "verified selection source window is incomplete")
                        win_start = float(selected_window[1])
                        win_end = float(selected_window[2])
                        if win_end <= win_start:
                            raise SceneLineageError(
                                "verified selection source window is non-positive")
                        source_filter = str(raw_row.get(
                            "selection_source_compare_filter") or "").strip()
                        if source_filter and not source_compare_filter_authorized(
                                source_filter):
                            raise SceneLineageError(
                                "selection source comparison declares an unauthorized filter")
                        cache_key = (str(source_path), round(win_start, 4),
                                     round(win_end, 4), source_filter)
                        source_features = source_bank_cache.get(cache_key)
                        if source_features is None:
                            source_features = _features_at_times(
                                source_path,
                                _uniform_times(source_path, 5, start=win_start,
                                               duration=win_end - win_start),
                                filter_prefix=source_filter,
                            )
                            source_bank_cache[cache_key] = source_features
                        source_comparison = _compare_bank(bound_features, source_features)
                        if source_filter:
                            source_comparison["authorized_source_filter"] = source_filter
                        if not source_comparison["passed"]:
                            raise SceneLineageError(
                                "planned derivative visually mismatches its selected source window")
                    record["bound_input_sha256"] = bound_sha
                    record["bound_sample_count"] = len(bound_features)
                    if source_comparison is not None:
                        record["selected_source_comparison"] = source_comparison
                    plan_row["_lineage"] = {
                        "kind": exp["kind"],
                        "input_path": expected_path,
                        "input_sha256": bound_sha,
                        "bound_features": bound_features,
                        "media_kind": mk,
                        "raw": raw_row,
                    }
                except Exception as exc:  # noqa: BLE001 — inability to freeze proof blocks
                    reasons.append(f"cannot bind immutable input identity: {exc}")
        if reasons:
            record["passed"] = False
            record["reasons"] = reasons
            for reason in reasons:
                failures.append({
                    "stage": "binding", "bi": record["bi"], "scene": scene,
                    "beat": beat, "reason": reason,
                })
        audit_rows.append(record)

    for key, exp in lookup.items():
        if key not in consumed and key not in duplicate_keys:
            failures.append({
                "stage": "binding", "scene": key[0], "beat": key[1],
                "reason": "lineage expectation does not map to an aired encode-plan beat",
            })
            audit_rows.append({
                "bi": None, "scene": key[0], "beat": key[1],
                "kind": exp.get("kind"),
                "expected_input": str(exp.get("input_path")) if exp.get("input_path") else None,
                "planned_input": None, "passed": False,
                "reasons": ["lineage expectation does not map to an aired encode-plan beat"],
            })
    if not encode_plan:
        failures.append({"stage": "binding", "reason": "encode plan is empty"})
    if not expected:
        failures.append({"stage": "binding", "reason": "lineage contract is empty"})
    return audit_rows, failures


def _probe_duration(path: Path) -> float:
    try:
        proc = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _feature(raw: bytes) -> dict:
    if len(raw) != _FEATURE_BYTES:
        raise SceneLineageError(
            f"lineage frame has {len(raw)} bytes; expected {_FEATURE_BYTES}")
    rgb = list(raw)
    pixels = len(rgb) // 3
    means = [sum(rgb[c::3]) / pixels for c in range(3)]
    grey: list[float] = []
    for i in range(0, len(rgb), 3):
        grey.append(0.299 * rgb[i] + 0.587 * rgb[i + 1] + 0.114 * rgb[i + 2])
    mean_luma = sum(grey) / len(grey)
    std = math.sqrt(sum((x - mean_luma) ** 2 for x in grey) / len(grey))
    bits = 0
    n = 0
    for y in range(_FEATURE_H):
        row = y * _FEATURE_W
        for x in range(_FEATURE_W - 1):
            if grey[row + x + 1] >= grey[row + x]:
                bits |= 1 << n
            n += 1
    return {
        "dhash": f"{bits:064x}",
        "mean_rgb": [round(v, 2) for v in means],
        "mean_luma": round(mean_luma, 2),
        "luma_std": round(std, 2),
    }


def _fingerprint_filter(prefix: str = "") -> str:
    # Crop away borders, letterbox edges, and corner bugs before making the
    # small perceptual signature.  Encoding/grade/camera drift survive this;
    # an unrelated character or location normally does not.
    return (
        f"{prefix}crop="
        "w='max(4,trunc(iw*0.74/2)*2)':"
        "h='max(4,trunc(ih*0.74/2)*2)':"
        "x='(iw-ow)/2':y='(ih-oh)/2',"
        "format=rgb24,scale=17:16:flags=area"
    )


def _features_at_times(path: Path, times: list[float],
                       *, filter_prefix: str = "") -> list[dict]:
    """Seek several timestamps in one ffmpeg process and fingerprint them."""
    path = Path(path)
    if not times:
        return []
    args: list[str] = [str(ffmpeg_exe()), "-hide_banner", "-loglevel", "error"]
    is_video = path.suffix.lower() in _VIDEO_EXTS
    for t in times:
        if is_video:
            args += ["-ss", f"{max(0.0, float(t)):.6f}"]
        args += ["-i", str(path)]
    chains = []
    labels = []
    for pos in range(len(times)):
        label = f"f{pos}"
        prefix = f"{filter_prefix}," if filter_prefix else ""
        chains.append(
            f"[{pos}:v]trim=end_frame=1,setpts=PTS-STARTPTS,"
            f"{_fingerprint_filter(prefix)}[{label}]")
        labels.append(f"[{label}]")
    chains.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[out]")
    args += [
        "-filter_complex", ";".join(chains), "-map", "[out]", "-an",
        "-fps_mode", "passthrough", "-frames:v", str(len(times)),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=90)
    except Exception as exc:  # noqa: BLE001
        raise SceneLineageError(f"cannot decode lineage frames from {path}: {exc}") from exc
    expected = len(times) * _FEATURE_BYTES
    if proc.returncode != 0 or len(proc.stdout) != expected:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-500:]
        raise SceneLineageError(
            f"cannot decode {len(times)} lineage frame(s) from {path.name}: "
            f"rc={proc.returncode}, bytes={len(proc.stdout)}/{expected}; {err}")
    out = []
    for pos, t in enumerate(times):
        feat = _feature(proc.stdout[pos * _FEATURE_BYTES:(pos + 1) * _FEATURE_BYTES])
        feat["time_s"] = round(float(t), 4)
        out.append(feat)
    return out


def _features_by_frames(path: Path, frames: list[int]) -> dict[int, dict]:
    """Decode exact frame numbers in one sequential pass (timeline order check)."""
    ordered = sorted(set(max(0, int(n)) for n in frames))
    if not ordered:
        return {}
    expr = "+".join(f"eq(n\\,{n})" for n in ordered)
    vf = f"select='{expr}',{_fingerprint_filter()}"
    args = [
        str(ffmpeg_exe()), "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", vf, "-an", "-fps_mode", "passthrough",
        "-frames:v", str(len(ordered)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        raise SceneLineageError(f"cannot decode timeline canaries from {path}: {exc}") from exc
    expected = len(ordered) * _FEATURE_BYTES
    if proc.returncode != 0 or len(proc.stdout) != expected:
        err = (proc.stderr or b"").decode("utf-8", "replace")[-500:]
        raise SceneLineageError(
            f"cannot decode timeline canaries from {Path(path).name}: "
            f"rc={proc.returncode}, bytes={len(proc.stdout)}/{expected}; {err}")
    out: dict[int, dict] = {}
    for pos, frame in enumerate(ordered):
        feat = _feature(proc.stdout[pos * _FEATURE_BYTES:(pos + 1) * _FEATURE_BYTES])
        feat["frame"] = frame
        out[frame] = feat
    return out


def _dist(actual: dict, expected: dict) -> dict:
    # ``int.bit_count`` is unavailable in the Python 3.7-compatible desktop
    # runtime; the signatures are only 256 bits, so the portable spelling is
    # effectively free here.
    h = bin(int(actual["dhash"], 16) ^ int(expected["dhash"], 16)).count("1") / 256.0
    color = math.sqrt(sum(
        (float(actual["mean_rgb"][i]) - float(expected["mean_rgb"][i])) ** 2
        for i in range(3))) / (255.0 * math.sqrt(3.0))
    luma = abs(float(actual["mean_luma"]) - float(expected["mean_luma"])) / 255.0
    low_texture = max(float(actual["luma_std"]), float(expected["luma_std"])) < 7.0
    # Deliberately conservative: normal H.264 loss, modest grades and a
    # Ken-Burns crop do not trip this.  Random scene content trends toward
    # 0.5 dHash distance; flat/near-flat frames need colour as a second axis.
    gross = (
        h > 0.52
        or (h > 0.43 and color > 0.075)
        or color > 0.36
        or (low_texture and (color > 0.16 or luma > 0.18))
    )
    return {
        "dhash_distance": round(h, 4),
        "color_distance": round(color, 4),
        "luma_distance": round(luma, 4),
        "gross_mismatch": bool(gross),
    }


def _compare_bank(actual: list[dict], expected: list[dict]) -> dict:
    if not actual or not expected:
        return {
            "passed": False, "samples": [],
            "reason": "decoded fingerprint bank is empty",
        }
    samples = []
    mismatch_count = 0
    for af in actual:
        choices = [(_dist(af, ef), ef) for ef in expected]
        best, matched = min(
            choices,
            key=lambda pair: pair[0]["dhash_distance"]
            + 0.35 * pair[0]["color_distance"],
        )
        # A sample is gross only when EVERY expected moment rejects it.  This
        # prevents motion within the correct shot from looking like a swap.
        gross = all(distance[0]["gross_mismatch"] for distance in choices)
        mismatch_count += int(gross)
        samples.append({
            "actual": af,
            "nearest_expected": matched,
            **best,
            "gross_mismatch": gross,
        })
    needed = 1 if len(actual) == 1 else math.ceil(len(actual) * 2 / 3)
    passed = mismatch_count < needed
    return {
        "passed": passed,
        "samples": samples,
        "gross_mismatches": mismatch_count,
        "failure_threshold": needed,
        "reason": None if passed else (
            f"{mismatch_count}/{len(actual)} decoded samples grossly mismatch planned input"),
    }


def _uniform_times(path: Path, count: int, *, start: float = 0.0,
                   duration: float | None = None) -> list[float]:
    if Path(path).suffix.lower() not in _VIDEO_EXTS:
        return [0.0]
    total = _probe_duration(Path(path))
    if total <= 0:
        raise SceneLineageError(f"cannot determine duration of lineage input {path}")
    start = min(max(0.0, float(start)), max(0.0, total - 1 / 30))
    available = max(1 / 30, total - start)
    span = min(available, max(1 / 30, float(duration))) if duration else available
    if count <= 1:
        fractions = [0.5]
    elif count < 5:
        fractions = [0.20, 0.50, 0.80]
    else:
        # Density follows the span. Five fixed fractions leave ~0.9s between expected moments on
        # an ordinary scene, and "a sample is gross only when EVERY expected moment rejects it"
        # can only mean something when those moments actually cover the scene. Job 0ca9dc4c2f's
        # scene 142 aired precisely its planned bytes and still failed the timeline canary: the
        # decoded frame sat at 1.2s and the nearest expected moment was 1.4s, far enough apart in
        # a fast push-in to read as foreign footage (measured: dhash 0.445 against the 5-frame
        # bank, 0.004 against the same segment sampled densely).
        #
        # This is a compensating fix, not the root cause. The canary derives each scene's timeline
        # position by accumulating round(beat_dur * fps), which drifts from the real concat (~7
        # frames by scene 142 here), so it compares the right footage at slightly the wrong time.
        # Deriving offsets from the actual concat entries would remove the drift itself.
        #
        # Measured on that render: densifying removed the false positive and changed nothing about
        # which genuinely foreign segments are caught.
        span_samples = math.ceil(span / _BANK_MAX_GAP_SEC) if span > 0 else int(count)
        n = max(5, min(_BANK_MAX_FRAMES, max(int(count), span_samples)))
        fractions = ([0.12, 0.31, 0.50, 0.69, 0.88] if n == 5
                     else [(i + 0.5) / n for i in range(n)])
    # A HELD TAIL REPEATS THE WINDOW'S LAST FRAME — SO SAMPLE IT.
    #
    # Every fraction above stops at 0.88 of the span, so the window's final frame is never in the
    # bank. That frame is exactly the one a hold repeats: when a beat needs more time than its
    # window holds, the derivative plays the window and then freezes on its last frame. Sampling
    # the derivative uniformly then returns that held image many times, none of which any expected
    # moment can match, and a correct render fails.
    #
    # Measured on job 229233891e, which died at this gate with 4 violations. Beat 7: a 1.63s window
    # filling 5.50s; the held frame's true source time is 189.76s and the bank's last sample was
    # 189.54s. A fine scan of the source put that frame INSIDE the window (dhash 0.0117, colour
    # 0.0041) — the hold was correct and the bank simply could not see it. Adding the final frame
    # took all four scenes from gross 10/14, 16/19, 10/14, 10/14 to ZERO.
    #
    # This ADDS a moment; it removes none, so nothing that was caught before can now slip past. I
    # first tried the other repair — collapsing the repeated samples — and the control measured it
    # honestly: it fixed the false positive but halved foreign detection (4/7 -> 2/7), because that
    # sensitivity came from the repeats themselves. That version was discarded.
    ceil = max(0.0, total - 1 / 30)
    times = [min(ceil, start + max(0.0, span - 1 / 30) * f) for f in fractions]
    # Stay a real margin clear of the FILE's end, not just of `ceil`. Measured: a window ending at
    # 152.533s in a 152.6s source put this sample at 152.523s, which ffmpeg could not decode at all
    # (bytes=0/816) and which failed the whole bind. `ceil` (total - 1/30) was not enough room.
    _end = min(start + span - 0.01, max(0.0, total - 0.12))
    if duration and _end > (times[-1] if times else start):
        # As close to the window's true end as the container allows. The 1/30 back-off used above
        # is a whole frame at 30fps and lands 0.02s short of the held frame on the measured case
        # (bank 189.74 vs held 189.76), which is enough to miss it entirely.
        #
        # Only for a window strictly INSIDE a longer source — the held-tail case. Asking for a
        # frame at the very end of the file itself is how this first broke eight canary tests on
        # short synthetic clips: the decoder returned one frame short and every bind failed.
        times.append(max(start, _end))
    return sorted(set(round(t, 3) for t in times))


def verify_encoded_plan(encode_plan: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Prove each decoded ``seg`` still resembles its bound planned input."""
    rows: list[dict] = []
    failures: list[dict] = []
    banks: dict[int, list[dict]] = {}
    for p in encode_plan:
        bi = _as_int(p.get("bi"))
        scene = _as_int(p.get("scene_index"), _as_int(p.get("j")))
        beat = _as_int(p.get("m"), 0)
        row = {"bi": bi, "scene": scene, "beat": beat, "passed": False}
        try:
            if p.get("_lineage_emergency_slate"):
                raise SceneLineageError("emergency slate replaced the planned scene")
            lineage = p.get("_lineage") or {}
            input_path = Path(lineage["input_path"])
            seg = Path(p["seg"])
            if not seg.exists() or seg.stat().st_size <= 0:
                raise SceneLineageError("encoded segment is missing or empty")
            bound_sha = str(lineage.get("input_sha256") or "")
            current_sha = _sha256(input_path)
            if not bound_sha or current_sha != bound_sha:
                raise SceneLineageError(
                    "planned input bytes changed after lineage binding")
            expected = list(lineage.get("bound_features") or [])
            if not expected:
                raise SceneLineageError("bind-time decoded fingerprint bank is absent")
            encoded = _features_at_times(seg, _uniform_times(seg, 5))
            comparison = _compare_bank(encoded, expected)
            banks[bi] = encoded
            row.update({
                "kind": lineage.get("kind"),
                "input": str(input_path), "segment": str(seg),
                "bound_input_sha256": bound_sha,
                "verified_input_sha256": current_sha,
                "comparison": comparison, "passed": bool(comparison["passed"]),
            })
            if not comparison["passed"]:
                failures.append({
                    "stage": "encoded_segments", "bi": bi, "scene": scene, "beat": beat,
                    "reason": comparison.get("reason") or "encoded segment visually mismatches input",
                })
        except Exception as exc:  # noqa: BLE001 — strict contract records and blocks
            row["error"] = str(exc)
            failures.append({
                "stage": "encoded_segments", "bi": bi, "scene": scene, "beat": beat,
                "reason": str(exc),
            })
        rows.append(row)
    return rows, failures, banks


def _sample_local_frames(duration_frames: int, left_blend_frames: int) -> list[int]:
    left = max(2, int(left_blend_frames) + 2)
    right = int(duration_frames) - 3
    if right < left:
        return []
    span = right - left
    if span < 4:
        return [left + span // 2]
    return sorted(set((left + round(span * 0.34), left + round(span * 0.68))))


def verify_timeline_order(video: Path, encode_plan: list[dict], beat_durs: list[float],
                          trans_tails: dict, encoded_banks: dict,
                          fps: int = 30) -> tuple[list[dict], list[dict]]:
    """Verify safe interior frames follow encode-plan order after concat/conform.

    The first frames of an incoming transitioned beat are intentionally excluded:
    pairwise xfade makes them a legal blend of two owners.  All selected canaries
    are strictly inside the beat's unblended body.
    """
    requests: list[tuple[dict, int, int]] = []
    start_frame = 0
    for pos, p in enumerate(encode_plan):
        dur = beat_durs[pos] if pos < len(beat_durs) else p.get("bd", 0.0)
        dur_frames = max(1, int(round(float(dur) * fps)))
        prev_xf = trans_tails.get(pos - 1, (0.0, ""))[0] if pos > 0 else 0.0
        left_blend = max(0, int(round(float(prev_xf) * fps)))
        for local in _sample_local_frames(dur_frames, left_blend):
            requests.append((p, local, start_frame + local))
        start_frame += dur_frames
    rows_by_bi: dict[int, dict] = {}
    for p in encode_plan:
        bi = _as_int(p.get("bi"))
        rows_by_bi[bi] = {
            "bi": bi,
            "scene": _as_int(p.get("scene_index"), _as_int(p.get("j"))),
            "beat": _as_int(p.get("m"), 0),
            "samples": [], "passed": True,
        }
    failures: list[dict] = []
    try:
        actual_by_frame = _features_by_frames(Path(video), [r[2] for r in requests])
    except Exception as exc:  # noqa: BLE001
        reason = str(exc)
        return list(rows_by_bi.values()), [{"stage": "timeline_order", "reason": reason}]

    for p, local, global_frame in requests:
        bi = _as_int(p.get("bi"))
        row = rows_by_bi[bi]
        actual = actual_by_frame.get(global_frame)
        bank = encoded_banks.get(bi) or []
        comparison = _compare_bank([actual] if actual else [], bank)
        sample = {
            "local_frame": local,
            "timeline_frame": global_frame,
            "comparison": comparison,
        }
        row["samples"].append(sample)
        if not comparison["passed"]:
            row["passed"] = False
            failures.append({
                "stage": "timeline_order", "bi": bi, "scene": row["scene"],
                "beat": row["beat"],
                "reason": "conformed timeline frame does not match its encode-plan beat",
                "timeline_frame": global_frame,
            })
    for row in rows_by_bi.values():
        if not row["samples"]:
            row["passed"] = False
            row["error"] = "beat has no unblended interior frame to verify"
            failures.append({
                "stage": "timeline_order", "bi": row["bi"],
                "scene": row["scene"], "beat": row["beat"],
                "reason": row["error"],
            })
    return list(rows_by_bi.values()), failures


def verify_delivered_output(video: Path, audit_path: Path, *,
                            stage: str = "delivered_output") -> list[dict]:
    """Re-run recorded timeline canaries on the artifact that will be delivered.

    Assembly's first timeline check necessarily precedes later overlay, caption,
    black-repair and mux passes.  Those passes are not allowed to reorder or
    substitute a scene.  This function consumes only the atomically persisted
    bind/encode evidence, so a post-pass cannot redefine its own expected frames.
    """
    audit_path = Path(audit_path)
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SceneLineageError(
            f"cannot read persisted scene-lineage evidence {audit_path}: {exc}") from exc
    if payload.get("status") != "passed":
        raise SceneLineageError(
            f"scene-lineage evidence is not in a passed state ({payload.get('status')!r})")
    video = Path(video)
    if not video.is_file() or video.stat().st_size <= 0:
        fail_audit(audit_path, payload, stage, [{
            "stage": stage, "reason": "delivered artifact is missing or empty",
            "artifact": str(video),
        }])

    banks: dict[int, list[dict]] = {}
    for row in payload.get("encoded_segments") or []:
        bi = _as_int(row.get("bi"))
        samples = ((row.get("comparison") or {}).get("samples") or [])
        bank = [dict(s.get("actual") or {}) for s in samples if s.get("actual")]
        if bi is not None and bank:
            banks[bi] = bank

    requests: list[tuple[int, int, dict]] = []
    failures: list[dict] = []
    for row in payload.get("timeline_order") or []:
        bi = _as_int(row.get("bi"))
        samples = list(row.get("samples") or [])
        if bi is None or bi not in banks:
            failures.append({
                "stage": stage, "bi": bi,
                "reason": "persisted encoded fingerprint bank is absent",
            })
            continue
        if not samples:
            failures.append({
                "stage": stage, "bi": bi,
                "reason": "persisted timeline has no sample for this beat",
            })
            continue
        for sample in samples:
            frame = _as_int(sample.get("timeline_frame"))
            if frame is None:
                failures.append({
                    "stage": stage, "bi": bi,
                    "reason": "persisted timeline sample has no frame number",
                })
            else:
                requests.append((bi, frame, row))
    if not requests:
        failures.append({"stage": stage, "reason": "no delivered timeline canaries were recorded"})

    rows: dict[int, dict] = {}
    if not failures:
        try:
            actual_by_frame = _features_by_frames(video, [r[1] for r in requests])
        except Exception as exc:  # noqa: BLE001
            failures.append({"stage": stage, "reason": str(exc)})
            actual_by_frame = {}
        for bi, frame, old_row in requests:
            row = rows.setdefault(bi, {
                "bi": bi, "scene": old_row.get("scene"), "beat": old_row.get("beat"),
                "samples": [], "passed": True,
            })
            actual = actual_by_frame.get(frame)
            comparison = _compare_bank([actual] if actual else [], banks.get(bi) or [])
            row["samples"].append({"timeline_frame": frame, "comparison": comparison})
            if not comparison["passed"]:
                row["passed"] = False
                failures.append({
                    "stage": stage, "bi": bi, "scene": row["scene"], "beat": row["beat"],
                    "timeline_frame": frame,
                    "reason": "delivered artifact frame does not match its bound scene",
                })
    delivered = {
        "artifact": str(video.resolve()),
        "artifact_sha256": (_sha256(video) if video.is_file() else ""),
        "rows": list(rows.values()),
        "passed": not failures,
    }
    payload.setdefault("delivered_checks", {})[stage] = delivered
    if failures:
        fail_audit(audit_path, payload, stage, failures)
    payload["status"] = "passed"
    payload["stage"] = stage
    payload["delivered_artifact"] = str(video.resolve())
    payload["delivered_artifact_sha256"] = delivered["artifact_sha256"]
    write_audit(audit_path, payload)
    return list(rows.values())
