"""Strict local audio alignment for cross-copy quote recovery.

Some clean scene uploads have usable 1080p pixels but imperfect ASR, while a burned-subtitle or
SD copy of the *same* scene has the authoritative timed words.  This module can locate the audio
from that authoritative interval in another local copy.  It does not discover or rank footage and
it never decides publication: callers still apply native-resolution, window-QC, vision and
relevance gates.

The proof is deliberately self-contained and stale-bound.  It records the exact source-content
fingerprints, reference/target spans, final selected window, correlation, runner-up and algorithm
generation.  ``relevance_contract`` recomputes every binding before accepting it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess


AUDIO_QUOTE_TRANSFER_SCHEMA = 2
AUDIO_QUOTE_TRANSFER_ALGORITHM = "pcm-pearson-ncc-fft-v2"
AUDIO_QUOTE_TRANSFER_SIGNAL = "quote_audio_transfer_evidence"
AUDIO_QUOTE_TRANSFER_SAMPLE_RATE = 4000
AUDIO_QUOTE_TRANSFER_MIN_CORRELATION = 0.90
AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN = 0.08
AUDIO_QUOTE_TRANSFER_REFERENCE_PAD_SEC = 0.40
AUDIO_QUOTE_TRANSFER_MAX_REFERENCE_SEC = 16.0
AUDIO_QUOTE_TRANSFER_MAX_SEARCH_SEC = 900.0


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _span(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = _finite_float(value[0]), _finite_float(value[1])
    if start is None or end is None or not (end > start >= 0.0):
        return None
    return start, end


def _pcm_from_media(path: Path | str, start: float, end: float, *, sample_rate: int) -> tuple:
    """Decode one exact mono float32 interval with the project-local ffmpeg dependency."""
    import numpy as np
    from vidlore.ffmpeg_tool import ffmpeg_exe

    media = Path(path)
    try:
        ffmpeg = ffmpeg_exe()
    except Exception:
        ffmpeg = ""
    if not ffmpeg or not Path(ffmpeg).is_file() or not media.is_file() \
            or not (end > start >= 0.0):
        return np.empty(0, dtype=np.float32), "decoder_or_media_unavailable"
    duration = end - start
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(media),
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-vn", "-ac", "1",
        "-ar", str(int(sample_rate)), "-f", "f32le", "pipe:1",
    ]
    try:
        proc = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=max(30.0, min(180.0, duration * 1.5 + 20.0)))
    except (OSError, subprocess.SubprocessError) as exc:
        return np.empty(0, dtype=np.float32), f"decode_error:{type(exc).__name__}"
    if proc.returncode != 0:
        return np.empty(0, dtype=np.float32), "decode_failed"
    pcm = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if pcm.size < max(32, int(sample_rate * 0.10)):
        return np.empty(0, dtype=np.float32), "decoded_audio_too_short"
    if not np.isfinite(pcm).all():
        return np.empty(0, dtype=np.float32), "decoded_audio_nonfinite"
    return pcm, ""


def _preemphasis(pcm):
    """Promote PCM to stable float64; Pearson centering below removes the DC component.

    Deliberately do not differentiate/pre-emphasize here.  Cross-upload AAC compression preserves
    the broad waveform more faithfully than its highest frequencies (the real beat-55 copies score
    0.918 on raw Pearson NCC but only 0.845 after pre-emphasis).  The high acceptance floor and
    independent uniqueness margin provide the false-match protection.
    """
    import numpy as np

    values = np.asarray(pcm, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        return np.empty(0, dtype=np.float64)
    return values


def _ncc_scores(reference, target, *, chunk_samples: int):
    """Return absolute sliding Pearson-NCC scores using bounded-memory numpy FFT chunks."""
    import numpy as np

    ref = _preemphasis(reference)
    hay = _preemphasis(target)
    width = int(ref.size)
    if width < 32 or hay.size < width:
        return np.empty(0, dtype=np.float32), "audio_interval_too_short"
    ref = ref - float(ref.mean())
    ref_energy = float(np.dot(ref, ref))
    if not math.isfinite(ref_energy) or ref_energy <= 1e-8:
        return np.empty(0, dtype=np.float32), "reference_audio_near_silent"

    positions = int(hay.size - width + 1)
    scores = np.full(positions, -np.inf, dtype=np.float32)
    starts_per_chunk = max(width, int(chunk_samples))
    for base in range(0, positions, starts_per_chunk):
        take = min(starts_per_chunk, positions - base)
        block = hay[base:base + take + width - 1]
        fft_size = 1 << int(block.size + width - 1).bit_length()
        # convolution(block, reverse(ref))[width-1:] is the sliding dot product.
        dot_full = np.fft.irfft(
            np.fft.rfft(block, fft_size) * np.fft.rfft(ref[::-1], fft_size), fft_size)
        dots = dot_full[width - 1:width - 1 + take]
        prefix = np.concatenate(([0.0], np.cumsum(block, dtype=np.float64)))
        prefix2 = np.concatenate(([0.0], np.cumsum(block * block, dtype=np.float64)))
        sums = prefix[width:width + take] - prefix[:take]
        sums2 = prefix2[width:width + take] - prefix2[:take]
        variance_energy = np.maximum(sums2 - (sums * sums / width), 0.0)
        denominator = np.sqrt(variance_energy * ref_energy)
        valid = denominator > 1e-9
        chunk_score = np.full(take, -np.inf, dtype=np.float64)
        chunk_score[valid] = np.abs(dots[valid] / denominator[valid])
        chunk_score[~np.isfinite(chunk_score)] = -np.inf
        scores[base:base + take] = chunk_score.astype(np.float32)
    if not np.isfinite(scores).any():
        return np.empty(0, dtype=np.float32), "target_audio_near_silent"
    return scores, ""


def normalized_pcm_alignment(reference_pcm, target_pcm, *, sample_rate: int = 4000) -> dict:
    """Align one PCM template within a PCM target and report a strict uniqueness measurement.

    This array-level entry point keeps the numerical primitive directly testable without codecs.
    Acceptance is reported but not hidden: callers persist the peak, runner-up and margin so a
    later publication check can enforce the current constants independently.
    """
    import numpy as np

    try:
        rate = int(sample_rate)
    except (TypeError, ValueError):
        rate = 0
    if rate < 1000:
        return {"status": "rejected", "reason": "invalid_sample_rate"}
    ref = np.asarray(reference_pcm, dtype=np.float32).reshape(-1)
    target = np.asarray(target_pcm, dtype=np.float32).reshape(-1)
    if (ref.size < max(32, int(rate * 0.10)) or target.size < ref.size
            or not np.isfinite(ref).all() or not np.isfinite(target).all()):
        return {"status": "rejected", "reason": "invalid_pcm"}
    scores, error = _ncc_scores(ref, target, chunk_samples=rate * 120)
    if error:
        return {"status": "rejected", "reason": error}

    best_index = int(np.argmax(scores))
    correlation = float(scores[best_index])
    # Neighbouring sample offsets on the SAME physical occurrence form one correlation lobe. Mask
    # only that local tolerance—not a whole template duration. The old template-wide mask hid a
    # second, genuinely separate occurrence placed immediately back-to-back. Keep the tolerance
    # proportional for short phrases and capped at 200 ms for long dialogue.
    exclusion = min(
        max(int(rate * 0.025), int(ref.size * 0.10)),
        max(1, int(ref.size) - 1),
        int(rate * 0.20),
    )
    lo, hi = max(0, best_index - exclusion), min(scores.size, best_index + exclusion + 1)
    scores[lo:hi] = -np.inf
    finite = scores[np.isfinite(scores)]
    runner_up = float(finite.max()) if finite.size else 0.0
    margin = max(0.0, correlation - runner_up)
    reason = ""
    if correlation < AUDIO_QUOTE_TRANSFER_MIN_CORRELATION:
        reason = "correlation_below_floor"
    elif margin < AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN:
        reason = "alignment_not_unique"
    return {
        "status": "matched" if not reason else "rejected",
        "reason": reason,
        "algorithm": AUDIO_QUOTE_TRANSFER_ALGORITHM,
        "schema_version": AUDIO_QUOTE_TRANSFER_SCHEMA,
        "sample_rate_hz": rate,
        "template_samples": int(ref.size),
        "target_samples": int(target.size),
        "best_start_sample": best_index,
        "same_peak_tolerance_samples": exclusion,
        "same_peak_tolerance_sec": round(exclusion / float(rate), 6),
        "correlation": round(correlation, 6),
        "runner_up_correlation": round(runner_up, 6),
        "uniqueness_margin": round(margin, 6),
        "minimum_correlation": AUDIO_QUOTE_TRANSFER_MIN_CORRELATION,
        "minimum_uniqueness_margin": AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN,
    }


def _transfer_quote_span_against_pcm(reference_path: Path | str, reference_quote_span,
                                     target_pcm, search_span: tuple[float, float]) -> dict:
    """Align one short reference against an already-decoded target search interval."""
    reference_span = _span(reference_quote_span)
    if reference_span is None:
        return {"status": "rejected", "reason": "invalid_span"}
    q0, q1 = reference_span
    t0, t1 = search_span
    quote_duration = q1 - q0
    if quote_duration < 0.20:
        return {"status": "rejected", "reason": "reference_quote_too_short"}
    ref_start = max(0.0, q0 - AUDIO_QUOTE_TRANSFER_REFERENCE_PAD_SEC)
    ref_end = q1 + AUDIO_QUOTE_TRANSFER_REFERENCE_PAD_SEC
    if ref_end - ref_start > AUDIO_QUOTE_TRANSFER_MAX_REFERENCE_SEC:
        return {"status": "rejected", "reason": "reference_interval_too_long"}
    ref_pcm, ref_error = _pcm_from_media(
        reference_path, ref_start, ref_end, sample_rate=AUDIO_QUOTE_TRANSFER_SAMPLE_RATE)
    if ref_error:
        return {"status": "rejected", "reason": f"reference_{ref_error}"}

    result = normalized_pcm_alignment(
        ref_pcm, target_pcm, sample_rate=AUDIO_QUOTE_TRANSFER_SAMPLE_RATE)
    result.update({
        "reference_quote_span": [round(q0, 6), round(q1, 6)],
        "reference_extract_window": [round(ref_start, 6), round(ref_end, 6)],
        "target_search_window": [round(t0, 6), round(t1, 6)],
    })
    if result.get("status") != "matched":
        return result
    matched_extract_start = t0 + float(result["best_start_sample"]) / float(
        AUDIO_QUOTE_TRANSFER_SAMPLE_RATE)
    target_q0 = matched_extract_start + (q0 - ref_start)
    target_q1 = target_q0 + quote_duration
    if not (target_q0 >= t0 and target_q1 <= t1):
        result.update({"status": "rejected", "reason": "mapped_quote_outside_search_window"})
        return result
    result["target_quote_span"] = [round(target_q0, 6), round(target_q1, 6)]
    return result


def transfer_quote_spans(references, target_path: Path | str, *, target_search_window) -> list[dict]:
    """Align several references while decoding the long target interval exactly once.

    ``references`` is an ordered iterable of ``(media_path, quote_span)`` pairs and results preserve
    that order. A recovery beat currently admits at most three references and twelve existing target
    sources, so this turns the previous worst case of 36 full 900-second decodes into at most twelve;
    the remaining decodes are only the short reference templates.
    """
    pairs = list(references or [])
    if not pairs:
        return []
    search_span = _span(target_search_window)
    if search_span is None:
        return [{"status": "rejected", "reason": "invalid_span"} for _item in pairs]
    t0, t1 = search_span
    if t1 - t0 > AUDIO_QUOTE_TRANSFER_MAX_SEARCH_SEC:
        return [{"status": "rejected", "reason": "target_search_interval_too_long"}
                for _item in pairs]
    target_pcm, target_error = _pcm_from_media(
        target_path, t0, t1, sample_rate=AUDIO_QUOTE_TRANSFER_SAMPLE_RATE)
    if target_error:
        return [{"status": "rejected", "reason": f"target_{target_error}"}
                for _item in pairs]
    results = []
    for item in pairs:
        try:
            reference_path, reference_quote_span = item
        except (TypeError, ValueError):
            results.append({"status": "rejected", "reason": "invalid_reference_request"})
            continue
        results.append(_transfer_quote_span_against_pcm(
            reference_path, reference_quote_span, target_pcm, search_span))
    return results


def transfer_quote_span(reference_path: Path | str, target_path: Path | str,
                        reference_quote_span, *, target_search_window) -> dict:
    """Locate one timed reference quote in a second local media file using decoded PCM."""
    results = transfer_quote_spans(
        [(reference_path, reference_quote_span)], target_path,
        target_search_window=target_search_window)
    return results[0] if results else {"status": "rejected", "reason": "invalid_reference_request"}


def _binding_payload(evidence: dict) -> dict:
    return {key: evidence[key] for key in sorted(evidence) if key != "binding_fingerprint"}


def evidence_binding_fingerprint(evidence: dict) -> str:
    """Tamper-evident deterministic identity for a transfer evidence record."""
    try:
        payload = json.dumps(
            _binding_payload(evidence), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, KeyError):
        return ""
    return hashlib.sha256(
        (f"quote-audio-transfer-binding-v{AUDIO_QUOTE_TRANSFER_SCHEMA}\x1f" + payload)
        .encode("utf-8", "replace")) \
        .hexdigest()


def make_transfer_evidence(*, authored_quote: str, reference_source_id: str,
                           reference_source_content_fingerprint: str,
                           reference_asr_ratio: float, target_source_id: str,
                           target_source_content_fingerprint: str,
                           target_selected_window, alignment: dict) -> dict:
    """Create the immutable JSON record consumed by the relevance contract."""
    selected = _span(target_selected_window)
    ref_span = _span((alignment or {}).get("reference_quote_span"))
    ref_extract = _span((alignment or {}).get("reference_extract_window"))
    target_span = _span((alignment or {}).get("target_quote_span"))
    target_search = _span((alignment or {}).get("target_search_window"))
    correlation = _finite_float((alignment or {}).get("correlation"))
    runner_up = _finite_float((alignment or {}).get("runner_up_correlation"))
    margin = _finite_float((alignment or {}).get("uniqueness_margin"))
    ratio = _finite_float(reference_asr_ratio)
    if (not str(authored_quote or "").strip() or not reference_source_id or not target_source_id
            or reference_source_id == target_source_id
            or not reference_source_content_fingerprint
            or not target_source_content_fingerprint or selected is None or ref_span is None
            or ref_extract is None or target_span is None or target_search is None
            or None in (correlation, runner_up, margin, ratio)
            or (alignment or {}).get("status") != "matched"
            or (alignment or {}).get("algorithm") != AUDIO_QUOTE_TRANSFER_ALGORITHM
            or int((alignment or {}).get("schema_version", 0) or 0)
            != AUDIO_QUOTE_TRANSFER_SCHEMA):
        return {}
    quote = " ".join(str(authored_quote).strip().split())
    evidence = {
        "schema_version": AUDIO_QUOTE_TRANSFER_SCHEMA,
        "algorithm": AUDIO_QUOTE_TRANSFER_ALGORITHM,
        "sample_rate_hz": AUDIO_QUOTE_TRANSFER_SAMPLE_RATE,
        "authored_quote": quote,
        "authored_quote_sha256": hashlib.sha256(quote.encode("utf-8", "replace")).hexdigest(),
        "reference_source_id": str(reference_source_id),
        "reference_source_content_fingerprint": str(reference_source_content_fingerprint),
        "reference_quote_span": [round(ref_span[0], 6), round(ref_span[1], 6)],
        "reference_extract_window": [round(ref_extract[0], 6), round(ref_extract[1], 6)],
        "reference_asr_ratio": round(float(ratio), 6),
        "target_source_id": str(target_source_id),
        "target_source_content_fingerprint": str(target_source_content_fingerprint),
        "target_quote_span": [round(target_span[0], 6), round(target_span[1], 6)],
        "target_search_window": [round(target_search[0], 6), round(target_search[1], 6)],
        "target_selected_window": [round(selected[0], 6), round(selected[1], 6)],
        "correlation": round(float(correlation), 6),
        "runner_up_correlation": round(float(runner_up), 6),
        "uniqueness_margin": round(float(margin), 6),
        "minimum_correlation": AUDIO_QUOTE_TRANSFER_MIN_CORRELATION,
        "minimum_uniqueness_margin": AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN,
    }
    evidence["binding_fingerprint"] = evidence_binding_fingerprint(evidence)
    return evidence


def transfer_evidence_shape_reason(evidence: dict) -> str:
    """Return a stable fail-closed reason for malformed or fabricated transfer evidence."""
    if not isinstance(evidence, dict):
        return "evidence_not_an_object"
    if evidence.get("schema_version") != AUDIO_QUOTE_TRANSFER_SCHEMA:
        return "evidence_schema_mismatch"
    if evidence.get("algorithm") != AUDIO_QUOTE_TRANSFER_ALGORITHM:
        return "evidence_algorithm_mismatch"
    if evidence.get("sample_rate_hz") != AUDIO_QUOTE_TRANSFER_SAMPLE_RATE:
        return "evidence_sample_rate_mismatch"
    for name in (
            "authored_quote", "authored_quote_sha256", "reference_source_id",
            "reference_source_content_fingerprint", "target_source_id",
            "target_source_content_fingerprint", "binding_fingerprint"):
        if not str(evidence.get(name, "") or "").strip():
            return f"{name}_missing"
    if evidence.get("reference_source_id") == evidence.get("target_source_id"):
        return "reference_and_target_source_identical"
    for name in (
            "reference_quote_span", "reference_extract_window", "target_quote_span",
            "target_search_window", "target_selected_window"):
        if _span(evidence.get(name)) is None:
            return f"{name}_invalid"
    for name in (
            "reference_asr_ratio", "correlation", "runner_up_correlation",
            "uniqueness_margin", "minimum_correlation", "minimum_uniqueness_margin"):
        if _finite_float(evidence.get(name)) is None:
            return f"{name}_invalid"
    if float(evidence["minimum_correlation"]) != AUDIO_QUOTE_TRANSFER_MIN_CORRELATION:
        return "evidence_correlation_floor_mismatch"
    if float(evidence["minimum_uniqueness_margin"]) \
            != AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN:
        return "evidence_uniqueness_floor_mismatch"
    quote = " ".join(str(evidence["authored_quote"]).strip().split())
    expected_quote_hash = hashlib.sha256(quote.encode("utf-8", "replace")).hexdigest()
    if evidence.get("authored_quote_sha256") != expected_quote_hash:
        return "authored_quote_hash_mismatch"
    expected_binding = evidence_binding_fingerprint(evidence)
    if not expected_binding or evidence.get("binding_fingerprint") != expected_binding:
        return "evidence_binding_fingerprint_mismatch"
    return ""
