"""Fail-closed semantic publication contract for persisted ClipStudio selections.

The verifier normally enforces this while matching, but build/rerender/resume can consume an old
``project.json`` without running verification again. This module re-derives the publication answer
from the persisted facts before any expensive render work and writes the answer atomically.

It never ranks or changes a selection. Generic/abstract filler keeps its existing lenient policy;
only exact and concrete character-specific beats require positive semantic evidence.
"""
from __future__ import annotations

import json
import hashlib
import re
import uuid
from pathlib import Path

from . import policy as _policy
from .models import SOURCE_OK
from .verify import (
    _cast_warning_resolution_reason,
    _contradiction_reason,
    _project_char2actor,
    _source_title_exact_cast_conflict,
    effective_deictic_target,
    selection_verifier_evidence_reason,
)

# v9 also requires a claimed source-title cast-warning resolution to identify the expected
# co-character in its pixel evidence.  Old audits accepted a bare boolean even when the verdict's
# own reason named nobody expected, so cached semantic recovery must not inherit that false proof.
# v10 makes short/common quote typing fail closed: fuzzy or non-contiguous token overlap can no
# longer turn an analyzer paraphrase into a hard promise of verbatim show dialogue. v11 requires
# every prompted-ASR quote hit to survive an independently decoded, unprompted narrow audio window
# before it can type a quote or prove a selected/breakout window.
SCHEMA_VERSION = 11
AUDIT_FILENAME = "selection_relevance_audit.json"
QUOTE_DIALOGUE_FLOOR = 0.78
QUOTE_WINDOW_TOLERANCE_SEC = 0.75
SHORT_QUOTE_EXACT_MAX_TOKENS = 3
SHORT_QUOTE_MAX_INTERWORD_GAP_SEC = 1.0
QUOTE_CONFIRMATION_SCHEMA = 1
QUOTE_CONFIRMATION_ALGORITHM = "faster-whisper-unprompted-window-v1"
QUOTE_CONFIRMATION_SAMPLE_RATE = 16000
QUOTE_CONFIRMATION_PAD_SEC = 2.0
QUOTE_CONFIRMATION_MAX_WINDOW_SEC = 30.0
QUOTE_CONFIRMATION_MAX_HINT_GAP_SEC = 2.0
QUOTE_CONFIRMATION_MAX_NO_SPEECH = 0.60
QUOTE_CONFIRMATION_MIN_AVG_LOGPROB = -1.0
QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM = 32

# A lenient verifier is allowed to say that the selected bytes are usable context while also
# recording that they do not prove the authored exact moment.  That is a completed CONTENT
# judgment, not missing verifier evidence.  Keep the vocabulary here beside the publication
# contract that emits it so recovery, self-heal, and audit tooling cannot drift apart.
DELIBERATE_EXACT_DOWNGRADE_REASONS = frozenset({
    "exact_verifier_evidence_not_strict",
    "exact_moving_verdict_was_downgraded",
    "exact_moving_relevance_is_contextual",
})
_REQUIRED_DELIBERATE_EXACT_DOWNGRADE_REASONS = frozenset({
    "exact_verifier_evidence_not_strict",
    "exact_moving_verdict_was_downgraded",
})
_CONCLUSIVE_CONTENT_REASONS = frozenset({
    "verdict_replace",
    "matches_narration_false",
    "specific_enough_false",
    "quality_ok_false",
    "wrong_subject_visible_true",
    "correct_subject_visible_false",
    "target_visible_false",
    "contradicts_narration_true",
    "era_ok_false",
    "deterministic_contradiction",
    "exact_source_title_cast_conflict_unresolved",
    # A located real quote still takes the quote-window recovery rung and can never be softened,
    # but its failed selected window is conclusive content evidence rather than a verifier outage.
    "exact_quote_dialogue_signal_below_floor",
    "exact_quote_timed_asr_span_absent",
    "exact_quote_timed_asr_outside_selected_window",
    "exact_quote_unprompted_confirmation_rejected",
})


def completed_deliberate_exact_downgrade(entry: dict) -> bool:
    """Whether an audit blocker is a fresh, bound completed judgment of an exact beat.

    The strict gate must continue to block this selection: lenient evidence cannot prove an exact
    promise.  Recovery typing is the only thing that changes.  A completed exact→contextual/generic
    judgment belongs in the content lane, where strict recovery and an independently bound gap
    review can act on it; it must not be re-asked as though its verifier evidence were absent.

    The ordinary shape is a bound lenient ``is_specific=false`` KEEP.  A strict verifier answer can
    also be relabelled contextual by the same repair pass after a deterministic content guard (for
    example, an unresolved source-title cast conflict) rejects it.  That mixed-looking record is
    still a conclusive CONTENT result when it has current strict evidence and an explicit content
    reason; treating its downgrade marker as a backend/schema fault starves exact-footage recovery.

    This predicate remains fail-closed.  A complete immutable evidence identity is required, every
    reason must be either a known downgrade or known content rejection, and the strict-evidence
    shape additionally requires an affirmative content rejection.  Unknown/binding/ASR provenance
    reasons remain technical, so this can never authorize publication or specificity loss.
    """
    if not isinstance(entry, dict):
        return False
    reasons = {str(reason) for reason in (entry.get("reasons") or []) if str(reason)}
    allowed = DELIBERATE_EXACT_DOWNGRADE_REASONS | _CONCLUSIVE_CONTENT_REASONS
    if not reasons or not reasons.issubset(allowed):
        return False

    verifier = entry.get("verifier") or {}
    if not isinstance(verifier, dict):
        return False
    if verifier.get("status") != "ok" or verifier.get("verdict") != "keep":
        return False
    evidence = verifier.get("selection_evidence") or {}
    if not isinstance(evidence, dict):
        return False
    is_specific = evidence.get("is_specific")
    lenient_shape = (
        is_specific is False
        and _REQUIRED_DELIBERATE_EXACT_DOWNGRADE_REASONS.issubset(reasons)
    )
    strict_content_shape = (
        is_specific is True
        and "exact_moving_verdict_was_downgraded" in reasons
        and bool(reasons & _CONCLUSIVE_CONTENT_REASONS)
    )
    if not (lenient_shape or strict_content_shape):
        return False
    if evidence.get("multiframe") is not True:
        return False
    for field in (
            "fingerprint", "question_fingerprint", "source_content_fingerprint",
            "source_id", "image_id", "model"):
        if not str(evidence.get(field, "") or "").strip():
            return False
    try:
        if int(evidence.get("schema_version", 0) or 0) <= 0:
            return False
        selection_in = float(evidence.get("selection_in"))
        selection_out = float(evidence.get("selection_out"))
    except (TypeError, ValueError):
        return False
    return selection_out > selection_in >= 0.0


def _strict_semantic_beat(seg) -> bool:
    """Exact scenes and concrete character beats need affirmative persisted verification."""
    return _policy.policy_of(seg) in (_policy.EXACT, _policy.CHARACTER)


def image_sha256(path: Path | str) -> str:
    """Content identity of the exact still bytes judged by the semantic verifier."""
    try:
        p = Path(path)
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    except Exception:
        return ""


def strict_still_evidence_reason(evidence: dict, seg) -> str:
    """Return why actual-image semantic evidence is insufficient, or ``""`` when publishable."""
    if not isinstance(evidence, dict):
        return "strict still verifier evidence is absent"
    if evidence.get("status") != "ok":
        return "still verifier status is not ok"
    if evidence.get("verdict") != "keep":
        return "still verifier verdict is not keep"
    for field in ("matches_narration", "specific_enough", "quality_ok"):
        if evidence.get(field) is not True:
            return f"still {field} is not affirmatively true"
    if evidence.get("wrong_subject_visible") is not False:
        return "still wrong_subject_visible is not affirmatively false"
    if evidence.get("contradicts_narration") is not False:
        return "still contradiction status is absent or true"
    if evidence.get("era_ok") is False:
        return "still era_ok is false"
    if (getattr(seg, "required_entity", "") or "").strip() \
            and evidence.get("correct_subject_visible") is not True:
        return "still required subject is not affirmatively visible"
    try:
        must_see = effective_deictic_target(seg)
    except Exception:
        must_see = ""
    if must_see and evidence.get("target_visible") is not True:
        return "still instructed-look target is not affirmatively visible"
    return ""


def verified_still_coverage(sel, seg, *, proj=None) -> tuple[bool, str]:
    """Whether the exact image that will air has complete positive semantic evidence.

    Source labels, CLIP similarity, Face-ID and the legacy ``still_verified`` boolean are candidate
    filters, not proof of what the image depicts. Both EXACT and concrete CHARACTER stills must be
    bound by sha256 to a strict actual-image verdict carrying the same positive facts required of
    moving footage.
    """
    image_path = str(getattr(sel, "image_path", "") or "")
    if not image_path:
        return False, ""
    p = Path(image_path).expanduser()
    if not p.is_file() or p.stat().st_size <= 0:
        return False, "image_path is missing or empty"
    meta = getattr(sel, "image_meta", {}) or {}
    evidence = meta.get("still_verifier") or meta.get("exact_still_verifier") or {}
    source = str(meta.get("source", "") or "")
    relevance_class = str(meta.get("relevance_class", "") or "")
    policy = _policy.policy_of(seg)

    if source not in ("source-frame", "source-frame-recovery", "web-exact-scene"):
        return False, f"untrusted still source {source or 'missing'}"
    if policy == _policy.EXACT and relevance_class != "exact_scene":
        return False, f"exact still relevance_class is {relevance_class or 'absent'}, not exact_scene"
    if policy != _policy.EXACT and relevance_class in ("generic_filler", "unverified_fallback", ""):
        return False, f"concrete still relevance_class is {relevance_class or 'absent'}"

    # Indexed source-frame stills are CLIP thumbnails (normally 512x288), not publication pixels.
    # A semantic verdict bound to those bytes cannot silently transfer to a separately encoded HD
    # JPEG. Force the scoped recovery lane to materialize and freshly judge the exact native owner
    # before this image can suppress its moving selection. Metadata-free/web stills retain their
    # existing semantic-only treatment here; the independent build native-resolution gate remains
    # authoritative for them.
    if source in ("source-frame", "source-frame-recovery") \
            and str(meta.get("src", "") or "") and meta.get("shot") is not None:
        try:
            from PIL import Image
            with Image.open(p) as image:
                iw, ih = image.size
        except Exception:
            return False, "owned source-frame still dimensions are unreadable"
        short, long = sorted((int(iw), int(ih)))
        if short < 720 or long < 1280:
            return False, (f"owned source-frame still is {iw}x{ih}; native materialization "
                           "and fresh semantic verification are required")
        if proj is not None:
            owner_id = str(meta.get("src", "") or "")
            owner = proj.source(owner_id)
            if owner is None:
                return False, f"owned source-frame still source {owner_id!r} is absent"
            try:
                from .era import title_era_conflicts
                from .verify import _project_beat_era
                beat_era = _project_beat_era(proj, seg)
                owner_title = ((getattr(owner, "title", "") or "") + " " + owner_id)
                if title_era_conflicts(beat_era, owner_title):
                    return False, (f"owned source-frame still declares the wrong era for "
                                   f"{beat_era or 'this beat'}")
                cast_reason = _source_title_exact_cast_conflict(
                    seg, owner_title, _project_char2actor(proj))
                if cast_reason and _cast_warning_resolution_reason(
                        evidence, seg, _project_char2actor(proj)):
                    return False, ("owned source-frame still has an unresolved exact-scene cast "
                                   f"warning: {cast_reason}")
            except Exception:
                return False, "owned source-frame still era provenance is unprovable"
            # Resolve exact indexed ownership now, not hours later in build.  A native image is
            # valid either because it is the original indexed keyframe itself, or because the
            # rematerialization lane persisted and we can recompute every owner/question binding.
            # Legacy booleans plus an image SHA do not prove where separately extracted HD pixels
            # came from or which semantic question judged them.
            try:
                from .build import (
                    _persisted_native_still_semantic_reason,
                    _resolve_indexed_still_owner,
                    _still_owner_source_fingerprint,
                )
                indexed_owner = _resolve_indexed_still_owner(proj, sel)
                if indexed_owner is None:
                    return False, "owned source-frame still indexed provenance is absent"
                if meta.get("native_semantic_materialized") is True:
                    native_reason = _persisted_native_still_semantic_reason(
                        proj, seg, indexed_owner, meta,
                        image_sha256=image_sha256(p),
                        source_fingerprint=_still_owner_source_fingerprint(indexed_owner))
                    if native_reason:
                        return False, f"owned native semantic binding is stale: {native_reason}"
            except Exception as exc:
                return False, f"owned source-frame still indexed provenance is invalid: {exc}"

    proven = meta.get("still_semantic_verified") is True
    if policy == _policy.EXACT:
        proven = proven and meta.get("exact_still_verified") is True
    if not proven:
        return False, "still lacks strict actual-image semantic verification"
    reason = strict_still_evidence_reason(evidence, seg)
    if reason:
        return False, reason
    claimed_hash = str(meta.get("still_image_sha256", "") or "")
    actual_hash = image_sha256(p)
    if not claimed_hash or not actual_hash or claimed_hash != actual_hash:
        return False, "still verifier evidence is not bound to the actual image bytes"
    return True, source


def _selection_source_title(proj, sel) -> str:
    src = proj.source(getattr(sel, "source_id", "") or "") if sel is not None else None
    return ((getattr(src, "title", "") or "") + " " +
            (getattr(sel, "source_id", "") or ""))


def _source_asr_provenance(proj, source_id: str, expected_fingerprint: str) \
        -> tuple[bool, str, str]:
    """Validate that a timed-word cache belongs to the current ASR semantics.

    Word timings are evidence, not an opportunistic cache.  A vocabulary, model, decoder-version
    or rescue-policy change can change both the words and their timestamps.  Missing/unreadable
    metadata therefore means UNKNOWN; it must never silently type an authored phrase as verbatim
    (or let a selected window prove that phrase).
    """
    if not expected_fingerprint:
        return False, "", "expected_fingerprint_unavailable"
    meta_path = proj.index_dir / f"{source_id}.index.meta.json"
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "", "index_metadata_missing"
    except Exception:
        return False, "", "index_metadata_unreadable"
    if not isinstance(raw, dict):
        return False, "", "index_metadata_invalid"
    actual = str(raw.get("asr_prompt_fingerprint", "") or "")
    if not actual:
        return False, "", "asr_prompt_fingerprint_missing"
    if actual != expected_fingerprint:
        return False, actual, "asr_prompt_fingerprint_mismatch"
    if raw.get("words") is not True:
        return False, actual, "word_evidence_not_certified"
    try:
        from .index import INDEX_SCHEMA
        if int(raw.get("schema", 0) or 0) < INDEX_SCHEMA:
            return False, actual, "word_evidence_schema_stale"
    except Exception:
        return False, actual, "word_evidence_schema_invalid"
    return True, actual, ""


def _normalized_quote_tokens(value: str, *, index_module=None) -> list[str]:
    """Return deterministic lexical tokens for exact short-quote evidence.

    Curly and straight apostrophes are formatting variants, but apostrophe presence is lexical:
    ``we'll`` must never be collapsed into ``well``.  Token boundaries are equally strict, so
    ``it's for you`` cannot prove ``it's you``.
    """
    if index_module is None:
        from . import index as index_module
    tokens = []
    for raw in re.findall(r"[\w]+(?:['’][\w]+)*", str(value or "")):
        token = str(index_module._norm_tok(raw.replace("’", "'")) or "")
        if token:
            tokens.append(token)
    return tokens


def _quote_requires_exact_contiguous_match(quote: str, *, index_module=None) -> bool:
    """Whether fuzzy phrase overlap is too ambiguous to type a quote as verbatim.

    Two/three-word lines are common enough that a high SequenceMatcher score is not identity
    evidence.  Longer dialogue retains the existing ASR-garble tolerance; measured real lines such
    as ``I will be your champion`` need it for minor ASR errors.
    """
    if index_module is None:
        from . import index as index_module
    tokens = _normalized_quote_tokens(quote, index_module=index_module)
    return len(tokens) <= SHORT_QUOTE_EXACT_MAX_TOKENS


def _exact_contiguous_quote_spans(words, quote: str, *, index_module=None) -> list[tuple]:
    """Locate every distinct exact normalized adjacent-token phrase in timed ASR.

    Short/common authored phrases need exact adjacency, but a prompted retrieval stream may repeat
    a hallucinated tight occurrence before/after the real utterance. Returning only the globally
    tightest occurrence would let that false candidate hide the real one from confirmation.
    """
    if index_module is None:
        from . import index as index_module
    wanted = _normalized_quote_tokens(quote, index_module=index_module)
    if len(wanted) < 2 or not words:
        return []
    stream = []
    for row in words:
        try:
            start, end, raw = float(row[0]), float(row[1]), str(row[2])
        except (IndexError, TypeError, ValueError):
            continue
        if not (end >= start >= 0.0):
            continue
        for token in _normalized_quote_tokens(raw, index_module=index_module):
            stream.append((token, start, end))
    size = len(wanted)
    found: list[tuple] = []
    for offset in range(0, len(stream) - size + 1):
        window = stream[offset:offset + size]
        if [row[0] for row in window] != wanted:
            continue
        if any(float(right[1]) - float(left[2]) > SHORT_QUOTE_MAX_INTERWORD_GAP_SEC
               for left, right in zip(window, window[1:])):
            continue
        found.append((float(window[0][1]), float(window[-1][2]), 1.0))
    unique = list({(round(row[0], 3), round(row[1], 3), 1.0) for row in found})
    return sorted(unique, key=lambda row: (row[1] - row[0], row[0]))


def _exact_contiguous_quote_span(words, quote: str, *, index_module=None):
    """Compatibility locator returning the tightest then earliest exact occurrence."""
    spans = _exact_contiguous_quote_spans(
        words, quote, index_module=index_module)
    return spans[0] if spans else None


def _prompted_quote_candidate_spans(
        words, quote: str, *, exact_contiguous_required: bool,
        index_module=None) -> list[tuple]:
    """Enumerate distinct retrieval occurrences; every returned hint still needs confirmation."""
    if index_module is None:
        from . import index as index_module
    if exact_contiguous_required:
        return _exact_contiguous_quote_spans(
            words, quote, index_module=index_module)
    return list(index_module.find_quote_span(
        words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR, all_matches=True) or [])


def _quote_confirmation_decoder_fingerprint(cfg) -> str:
    """Semantic identity of the independent no-prompt narrow decoder."""
    from . import index as _index

    payload = {
        "schema_version": QUOTE_CONFIRMATION_SCHEMA,
        "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
        "model": str(getattr(cfg, "whisper_model", "") or ""),
        "compute": str(getattr(cfg, "whisper_compute", "") or ""),
        "faster_whisper": str(_index._faster_whisper_version() or "unknown"),
        "sample_rate_hz": QUOTE_CONFIRMATION_SAMPLE_RATE,
        "word_timestamps": True,
        "vad_filter": False,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "hotwords": None,
        "maximum_no_speech_probability": QUOTE_CONFIRMATION_MAX_NO_SPEECH,
        "minimum_average_log_probability": QUOTE_CONFIRMATION_MIN_AVG_LOGPROB,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _quote_confirmation_locator(words, quote: str, *, exact_contiguous_required: bool,
                                index_module=None):
    """Apply the unchanged branch-specific phrase locator to unprompted words."""
    if index_module is None:
        from . import index as index_module
    if exact_contiguous_required:
        return _exact_contiguous_quote_span(words, quote, index_module=index_module)
    return index_module.find_quote_span(
        words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)


def _confirmation_span_near_hint(confirmed, prompted) -> bool:
    try:
        c0, c1 = float(confirmed[0]), float(confirmed[1])
        p0, p1 = float(prompted[0]), float(prompted[1])
    except (IndexError, TypeError, ValueError):
        return False
    if not (c1 > c0 >= 0.0 and p1 > p0 >= 0.0):
        return False
    gap = max(0.0, p0 - c1, c0 - p1)
    return gap <= QUOTE_CONFIRMATION_MAX_HINT_GAP_SEC


def _quote_confirmation_binding(
        proj, src, quote: str, prompted_span, cfg, *,
        exact_contiguous_required: bool) -> tuple[dict | None, str]:
    """Bind one narrow confirmation request to immutable source/decoder/locator inputs."""
    try:
        p0, p1 = float(prompted_span[0]), float(prompted_span[1])
        prompted_ratio = float(prompted_span[2])
    except (IndexError, TypeError, ValueError):
        return None, "prompted_span_invalid"
    if not (p1 > p0 >= 0.0 and 0.0 <= prompted_ratio <= 1.000001):
        return None, "prompted_span_invalid"
    raw_quote = " ".join(str(quote or "").strip().split())
    if len(_normalized_quote_tokens(raw_quote)) < 2:
        return None, "authored_quote_invalid"

    source_id = str(getattr(src, "id", "") or "")
    source_path = str(getattr(src, "local_path", "") or "")
    if not source_id or not source_path or not Path(source_path).is_file():
        return None, "source_media_unavailable"
    from .verify import _file_fingerprint
    source_fingerprint = str(_file_fingerprint(source_path) or "")
    if source_fingerprint in ("", "missing", "unreadable"):
        return None, "source_content_fingerprint_unavailable"

    source_end = float(getattr(src, "duration", 0.0) or 0.0)
    if source_end <= 0.0:
        try:
            from . import index as _index
            source_end = max(
                (float(getattr(shot, "end", 0.0) or 0.0)
                 for shot in _index.load_shots(proj, source_id)), default=0.0)
        except Exception:
            source_end = 0.0
    lo = max(0.0, p0 - QUOTE_CONFIRMATION_PAD_SEC)
    hi = p1 + QUOTE_CONFIRMATION_PAD_SEC
    if source_end > 0.0:
        hi = min(source_end, hi)
    if not hi > lo:
        return None, "confirmation_window_invalid"
    if hi - lo > QUOTE_CONFIRMATION_MAX_WINDOW_SEC + 1e-9:
        return None, "prompted_span_too_wide"

    quote_hash = hashlib.sha256(raw_quote.encode("utf-8", "replace")).hexdigest()
    locator_policy = ("exact_contiguous_short_common" if exact_contiguous_required
                      else "fuzzy_distinctive_phrase")
    binding = {
        "schema_version": QUOTE_CONFIRMATION_SCHEMA,
        "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
        "source_id": source_id,
        "source_content_fingerprint": source_fingerprint,
        "authored_quote": raw_quote,
        "authored_quote_sha256": quote_hash,
        "prompted_span": [round(p0, 3), round(p1, 3), round(prompted_ratio, 3)],
        "decode_window": [round(lo, 3), round(hi, 3)],
        "locator_policy": locator_policy,
        "requires_exact_contiguous_match": bool(exact_contiguous_required),
        "ratio_floor": QUOTE_DIALOGUE_FLOOR,
        "maximum_hint_gap_sec": QUOTE_CONFIRMATION_MAX_HINT_GAP_SEC,
        "decoder_fingerprint": _quote_confirmation_decoder_fingerprint(cfg),
    }
    raw = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    binding["binding_fingerprint"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return binding, ""


def _quote_confirmation_artifact_path(proj, binding: dict) -> Path:
    sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(binding.get("source_id", "") or ""))
    key = str(binding.get("binding_fingerprint", "") or "")
    return Path(proj.index_dir) / "quote_confirmations" / sid / f"{key}.json"


def _quote_confirmation_artifact_display_path(proj, path: Path) -> str:
    try:
        return str(path.relative_to(Path(proj.root_path)))
    except Exception:
        return str(path)


def _quote_confirmation_result_sha256(artifact: dict) -> str:
    """Bind persisted decoder output separately from the request/cache-path identity."""
    try:
        payload = {
            key: artifact[key]
            for key in sorted(artifact)
            if key != "result_content_sha256"
        }
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False)
    except (KeyError, TypeError, ValueError):
        return ""
    return hashlib.sha256(
        ("quote-confirmation-result-v1\x1f" + raw).encode("utf-8", "replace")) \
        .hexdigest()


def _validated_quote_confirmation_artifact(path: Path, binding: dict) -> dict | None:
    """Load and recompute one content-bound confirmation result; malformed means cache miss."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict) or raw.get("binding") != binding:
        return None
    if raw.get("schema_version") != QUOTE_CONFIRMATION_SCHEMA \
            or raw.get("algorithm") != QUOTE_CONFIRMATION_ALGORITHM:
        return None
    result_sha256 = str(raw.get("result_content_sha256", "") or "")
    if (len(result_sha256) != 64
            or any(char not in "0123456789abcdef" for char in result_sha256)
            or result_sha256 != _quote_confirmation_result_sha256(raw)):
        return None
    status = str(raw.get("status", "") or "")
    if status not in ("confirmed", "rejected"):
        return None

    confidence = raw.get("segment_confidence")
    if not isinstance(confidence, list) or not confidence:
        return None
    for row in confidence:
        if not isinstance(row, dict) or type(row.get("accepted")) is not bool:
            return None
        try:
            no_speech = float(row["no_speech_prob"])
            avg_logprob = float(row["avg_logprob"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (no_speech == no_speech and avg_logprob == avg_logprob
                and abs(no_speech) != float("inf") and abs(avg_logprob) != float("inf")):
            return None
        accepted = (no_speech <= QUOTE_CONFIRMATION_MAX_NO_SPEECH
                    and avg_logprob >= QUOTE_CONFIRMATION_MIN_AVG_LOGPROB)
        if row["accepted"] != bool(accepted):
            return None
        if not accepted:
            # Live decoding treats one uncertain segment as inconclusive. Cached results must not
            # silently use a weaker confidence rule merely because their ``False`` flag is honest.
            return None

    words = raw.get("timed_words")
    if not isinstance(words, list) or not words:
        return None
    try:
        lo, hi = [float(value) for value in binding["decode_window"]]
    except (KeyError, TypeError, ValueError):
        return None
    cleaned = []
    for row in words:
        try:
            start, end, word = float(row[0]), float(row[1]), str(row[2])
        except (IndexError, TypeError, ValueError):
            return None
        if (not word.strip() or not end > start >= lo - 0.25 or end > hi + 0.25
                or start != start or end != end
                or abs(start) == float("inf") or abs(end) == float("inf")):
            return None
        cleaned.append([start, end, word])
    if cleaned != sorted(cleaned, key=lambda row: (row[0], row[1])):
        return None

    span = _quote_confirmation_locator(
        cleaned, str(binding.get("authored_quote", "") or ""),
        exact_contiguous_required=bool(binding.get("requires_exact_contiguous_match")))
    if span is not None and not _confirmation_span_near_hint(
            span, binding.get("prompted_span") or []):
        span = None
    expected_status = "confirmed" if span is not None else "rejected"
    if status != expected_status:
        return None
    if span is not None:
        saved = raw.get("confirmed_span")
        try:
            if any(abs(float(saved[i]) - float(span[i])) > 0.001 for i in range(3)):
                return None
        except (IndexError, TypeError, ValueError):
            return None
    elif raw.get("confirmed_span") not in (None, []):
        return None
    return raw


def _quote_confirmation_summary(evidence: dict) -> dict:
    """Small audit-safe view; the bound artifact retains words/confidence for revalidation."""
    return {
        key: evidence.get(key)
        for key in (
            "schema_version", "algorithm", "status", "reason", "artifact_key",
            "artifact_path", "decoder_fingerprint", "decode_window", "prompted_span",
            "confirmed_span", "timed_asr_ratio", "match_method", "result_content_sha256")
        if key in evidence
    }


def _confirm_prompted_quote_span_unprompted(
        proj, src, quote: str, prompted_span, cfg, *,
        exact_contiguous_required: bool) -> dict:
    """Confirm one prompted-ASR retrieval hit with an immutable narrow no-prompt decode."""
    binding, binding_reason = _quote_confirmation_binding(
        proj, src, quote, prompted_span, cfg,
        exact_contiguous_required=exact_contiguous_required)
    if binding is None:
        return {
            "schema_version": QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
            "status": "inconclusive",
            "reason": binding_reason,
        }
    artifact_path = _quote_confirmation_artifact_path(proj, binding)
    cached = _validated_quote_confirmation_artifact(artifact_path, binding)
    if cached is not None:
        return {
            **cached,
            "artifact_key": binding["binding_fingerprint"],
            "artifact_path": _quote_confirmation_artifact_display_path(proj, artifact_path),
            "cache_hit": True,
        }

    from . import index as _index
    decode = _index.transcribe_unprompted_window(
        getattr(src, "local_path", "") or "", cfg,
        binding["decode_window"][0], binding["decode_window"][1],
        sample_rate=QUOTE_CONFIRMATION_SAMPLE_RATE,
        max_no_speech=QUOTE_CONFIRMATION_MAX_NO_SPEECH,
        min_avg_logprob=QUOTE_CONFIRMATION_MIN_AVG_LOGPROB)
    if str(decode.get("status", "") or "") != "ok":
        return {
            "schema_version": QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
            "status": "inconclusive",
            "reason": str(decode.get("reason", "") or "unprompted_decode_inconclusive"),
            "artifact_key": binding["binding_fingerprint"],
            "artifact_path": _quote_confirmation_artifact_display_path(proj, artifact_path),
            "decoder_fingerprint": binding["decoder_fingerprint"],
            "decode_window": binding["decode_window"],
            "prompted_span": binding["prompted_span"],
        }
    if decode.get("decode_window") != binding["decode_window"]:
        return {
            "schema_version": QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
            "status": "inconclusive",
            "reason": "unprompted_decode_window_mismatch",
            "artifact_key": binding["binding_fingerprint"],
        }

    words = list(decode.get("timed_words") or [])
    span = _quote_confirmation_locator(
        words, str(binding["authored_quote"]),
        exact_contiguous_required=exact_contiguous_required, index_module=_index)
    if span is not None and not _confirmation_span_near_hint(span, binding["prompted_span"]):
        span = None
    status = "confirmed" if span is not None else "rejected"
    method = ("exact_contiguous_timed_asr+unprompted_confirmation"
              if exact_contiguous_required
              else "fuzzy_phrase_timed_asr+unprompted_confirmation")
    artifact = {
        "schema_version": QUOTE_CONFIRMATION_SCHEMA,
        "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
        "binding": binding,
        "status": status,
        "reason": "" if span is not None else "unprompted_phrase_not_found_near_hint",
        "decoder_fingerprint": binding["decoder_fingerprint"],
        "decode_window": binding["decode_window"],
        "prompted_span": binding["prompted_span"],
        "timed_words": words,
        "segment_confidence": list(decode.get("segment_confidence") or []),
        "confirmed_span": ([round(float(span[0]), 3), round(float(span[1]), 3),
                            round(float(span[2]), 3)] if span is not None else None),
        "timed_asr_ratio": round(float(span[2]), 3) if span is not None else 0.0,
        "match_method": method,
    }
    artifact["result_content_sha256"] = _quote_confirmation_result_sha256(artifact)
    # Revalidate the exact bytes before publishing the cache entry. A malformed decoder result is
    # uncertainty, not a cacheable negative observation.
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    # A process can evaluate several projects/beats concurrently.  A PID-only name lets two
    # confirmation writers in that process overwrite or unlink each other's candidate artifact.
    # Give every atomic publication its own path; validation below still decides whether the bytes
    # are admissible before the final replace.
    tmp = artifact_path.with_name(
        artifact_path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(artifact, sort_keys=True, indent=1), encoding="utf-8")
        if _validated_quote_confirmation_artifact(tmp, binding) is None:
            return {
                "schema_version": QUOTE_CONFIRMATION_SCHEMA,
                "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
                "status": "inconclusive",
                "reason": "confirmation_artifact_self_validation_failed",
                "artifact_key": binding["binding_fingerprint"],
            }
        tmp.replace(artifact_path)
    except OSError as exc:
        return {
            "schema_version": QUOTE_CONFIRMATION_SCHEMA,
            "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
            "status": "inconclusive",
            "reason": f"confirmation_cache_write_error:{type(exc).__name__}",
            "artifact_key": binding["binding_fingerprint"],
        }
    finally:
        tmp.unlink(missing_ok=True)
    return {
        **artifact,
        "artifact_key": binding["binding_fingerprint"],
        "artifact_path": _quote_confirmation_artifact_display_path(proj, artifact_path),
        "cache_hit": False,
    }


def _quote_pool_branches(proj, segments, *, cfg=None) -> dict[int, dict]:
    """Classify authored strict-beat quotes against the complete usable dialogue index.

    The analyzer's ``quote`` field is not a type guarantee: it contains real show dialogue,
    paraphrases, and sometimes the essayist's own narration.  A selected-window ASR floor is only
    meaningful for the first class.  Search every indexed source that is eligible to supply the
    show's own audio *before* applying that floor.  ``_breakout_src_ok`` is the existing audio-trust
    boundary: it excludes reaction/essay narration and wall-to-wall commentary, which otherwise
    makes a fuzzy phrase hit (observed on beat 8's Season-7 News source) look like show dialogue.

    The map is recomputed once per contract evaluation.  Recovery can add sources between
    evaluations, so a process-global cache would make an old ``paraphrase`` decision stale.
    """
    quoted = [s for s in (segments or [])
              if _strict_semantic_beat(s)
              and str(getattr(s, "quote", "") or "").strip()]
    if not quoted:
        return {}

    from . import index as _index
    if cfg is None:
        from .config import load_clip_config
        cfg = load_clip_config()
    try:
        expected_asr_fingerprint = str(_index.asr_semantic_fingerprint(proj, cfg) or "")
    except Exception:
        # Fingerprint computation is part of the evidence chain.  Treat its failure exactly like
        # missing source metadata: the quote's type is indeterminate, never a permissive fallback.
        expected_asr_fingerprint = ""
    # Lazy import avoids a module-initialisation cycle: build imports this contract only at runtime.
    from .build import _breakout_src_ok, _ESSAYISH_RX
    try:
        from .discover import _REACTION_TITLE
    except Exception:
        _REACTION_TITLE = None

    streams: list[tuple[object, list[tuple[str, list]]]] = []
    indexed = 0
    rejected_commentary = 0
    invalid_provenance: list[dict] = []
    seen_source_ids: set[str] = set()
    for src in (getattr(proj, "sources", None) or []):
        sid = str(getattr(src, "id", "") or "")
        if (not sid or sid in seen_source_ids
                or str(getattr(src, "status", "") or "") != SOURCE_OK):
            continue
        seen_source_ids.add(sid)
        title = str(getattr(src, "title", "") or "")
        # Title-only exclusions are independently knowable even when their indexes are damaged;
        # they are commentary/non-show audio and never belong in the dialogue evidence universe.
        if _ESSAYISH_RX.search(title) or (_REACTION_TITLE is not None
                                          and _REACTION_TITLE.search(title)):
            rejected_commentary += 1
            continue
        indexed += 1
        try:
            words_valid, words = _index._load_words_result(proj, sid)
        except Exception:
            words_valid, words = False, []
        try:
            shots = _index.load_shots(proj, sid)
            shots_valid = bool(shots)
        except Exception:
            shots, shots_valid = [], False
        if not shots_valid:
            # Speech coverage is the second half of the commentary gate. Missing/corrupt/empty
            # shots make that eligibility unknowable, so the pool result must be indeterminate.
            invalid_provenance.append({
                "source_id": sid,
                "reason": "shots_cache_invalid_or_missing",
                "actual_asr_prompt_fingerprint": "",
            })
            continue
        if not words_valid:
            invalid_provenance.append({
                "source_id": sid,
                "reason": "words_cache_invalid_or_missing",
                "actual_asr_prompt_fingerprint": "",
            })
            continue
        provenance_ok, actual_fingerprint, provenance_reason = _source_asr_provenance(
            proj, sid, expected_asr_fingerprint)
        if not provenance_ok:
            invalid_provenance.append({
                "source_id": sid,
                "reason": provenance_reason,
                "actual_asr_prompt_fingerprint": actual_fingerprint,
            })
            continue
        try:
            dialogue_eligible = bool(_breakout_src_ok(src, shots))
        except Exception:
            invalid_provenance.append({
                "source_id": sid,
                "reason": "dialogue_eligibility_unverifiable",
                "actual_asr_prompt_fingerprint": actual_fingerprint,
            })
            continue
        if not dialogue_eligible:
            rejected_commentary += 1
            continue
        try:
            retrieval_valid, retrieval_streams, retrieval_reason, retrieval_complete = \
                _index._load_quote_retrieval_streams_result(
                    proj, src, cfg, require_complete=True)
        except Exception:
            retrieval_valid, retrieval_streams, retrieval_complete = False, [], False
            retrieval_reason = "quote_retrieval_validation_error"
        if not retrieval_valid or not retrieval_complete:
            # Clean general ASR can miss real dialogue. Absence is only conclusive after the
            # separate authored-prompt retrieval stream is complete for every eligible source.
            invalid_provenance.append({
                "source_id": sid,
                "reason": retrieval_reason or "quote_retrieval_cache_invalid_or_missing",
                "actual_asr_prompt_fingerprint": actual_fingerprint,
            })
            continue
        # An empty cache with current provenance is a valid observation of silence.  Retain it in
        # the scanned set so an entirely silent but fully-current pool can classify a quote as a
        # paraphrase rather than pretending the index is absent.
        source_streams = [("general_names_only_asr", words)]
        source_streams.extend(
            (f"authored_prompt_retrieval_chunk_{chunk_index}", stream["words"])
            for chunk_index, stream in enumerate(retrieval_streams))
        streams.append((src, source_streams))

    by_quote: dict[str, dict] = {}
    out: dict[int, dict] = {}
    for seg in quoted:
        quote = str(getattr(seg, "quote", "") or "").strip()
        key = " ".join(quote.lower().split())
        branch = by_quote.get(key)
        if branch is None:
            best: dict | None = None
            matches: list[dict] = []
            confirmation_attempts: list[dict] = []
            confirmation_inconclusive: list[dict] = []
            prompted_pool_hit_count = 0
            general_pool_hit_count = 0
            retrieval_candidate_hit_count = 0
            confirmation_rejected_count = 0
            fuzzy_only_best = None
            fuzzy_only_count = 0
            exact_contiguous_required = _quote_requires_exact_contiguous_match(
                quote, index_module=_index)
            seen_candidates: set[tuple] = set()
            truncated_candidate_streams: list[dict] = []
            locator_errors: list[dict] = []
            for src, source_streams in streams:
                for stream_kind, words in source_streams:
                    try:
                        candidate_spans = _prompted_quote_candidate_spans(
                            words, quote,
                            exact_contiguous_required=exact_contiguous_required,
                            index_module=_index)
                        if exact_contiguous_required and not candidate_spans:
                            fuzzy_only = _index.find_quote_span(
                                words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)
                            if fuzzy_only:
                                fuzzy_only_count += 1
                                if (fuzzy_only_best is None
                                        or float(fuzzy_only[2])
                                        > float(fuzzy_only_best[1][2])):
                                    fuzzy_only_best = (src, fuzzy_only)
                    except Exception as exc:
                        # A locator failure is missing evidence, not evidence that the authored
                        # phrase is absent.  Preserve it in the contract so the publication branch
                        # fails closed instead of silently becoming ``paraphrase``.
                        locator_errors.append({
                            "source_id": str(getattr(src, "id", "") or ""),
                            "retrieval_stream": stream_kind,
                            "reason": f"quote_locator_error:{type(exc).__name__}",
                        })
                        candidate_spans = []
                    if len(candidate_spans) > QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM:
                        truncated_candidate_streams.append({
                            "source_id": str(getattr(src, "id", "") or ""),
                            "retrieval_stream": stream_kind,
                            "candidate_count": len(candidate_spans),
                            "attempted_count": QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM,
                        })
                        candidate_spans = candidate_spans[
                            :QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM]
                    for span in candidate_spans:
                        candidate_key = (
                            str(getattr(src, "id", "") or ""),
                            round(float(span[0]), 3), round(float(span[1]), 3),
                            round(float(span[2]), 3))
                        if candidate_key in seen_candidates:
                            continue
                        seen_candidates.add(candidate_key)
                        retrieval_candidate_hit_count += 1
                        if stream_kind.startswith("authored_prompt_retrieval_chunk_"):
                            prompted_pool_hit_count += 1
                        else:
                            general_pool_hit_count += 1
                        try:
                            confirmation = _confirm_prompted_quote_span_unprompted(
                                proj, src, quote, span, cfg,
                                exact_contiguous_required=exact_contiguous_required)
                        except Exception as exc:
                            confirmation = {
                                "schema_version": QUOTE_CONFIRMATION_SCHEMA,
                                "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
                                "status": "inconclusive",
                                "reason": f"confirmation_internal_error:{type(exc).__name__}",
                            }
                        confirmation_summary = _quote_confirmation_summary(confirmation)
                        confirmation_summary.update({
                            "source_id": str(getattr(src, "id", "") or ""),
                            "source_title": str(getattr(src, "title", "") or ""),
                            "retrieval_stream": stream_kind,
                        })
                        confirmation_attempts.append(confirmation_summary)
                        confirmation_status = str(confirmation.get("status", "") or "")
                        if confirmation_status == "rejected":
                            confirmation_rejected_count += 1
                            continue
                        if confirmation_status != "confirmed":
                            confirmation_inconclusive.append(confirmation_summary)
                            continue
                        confirmed_span = confirmation.get("confirmed_span")
                        try:
                            confirmed_span = (
                                float(confirmed_span[0]), float(confirmed_span[1]),
                                float(confirmed_span[2]))
                        except (IndexError, TypeError, ValueError):
                            malformed = dict(confirmation_summary)
                            malformed.update({
                                "status": "inconclusive",
                                "reason": "confirmed_span_invalid",
                            })
                            confirmation_inconclusive.append(malformed)
                            continue
                        match = {
                            "source_id": str(getattr(src, "id", "") or ""),
                            "source_title": str(getattr(src, "title", "") or ""),
                            "timed_asr_span": [
                                round(confirmed_span[0], 3), round(confirmed_span[1], 3)],
                            "timed_asr_ratio": round(confirmed_span[2], 3),
                            "prompted_asr_span": [
                                round(float(span[0]), 3), round(float(span[1]), 3),
                                round(float(span[2]), 3)],
                            "match_method": str(confirmation.get("match_method", "") or ""),
                            "retrieval_stream": stream_kind,
                            "unprompted_confirmation": confirmation_summary,
                        }
                        matches.append(match)
                        # The strongest independently confirmed phrase wins. Equal ratios preserve
                        # stable source/occurrence order; all confirmed windows remain in matches.
                        if (best is None or float(match["timed_asr_ratio"])
                                > float(best["timed_asr_ratio"])):
                            best = match
            if best is not None:
                kind = "verbatim"
                branch_reason = ("short_common_quote_exact_contiguous_timed_asr_"
                                 "unprompted_confirmed"
                                 if exact_contiguous_required
                                 else "distinctive_quote_fuzzy_phrase_timed_asr_"
                                      "unprompted_confirmed")
            elif (invalid_provenance or not streams or locator_errors
                  or confirmation_inconclusive or truncated_candidate_streams):
                kind = "indeterminate"
                if locator_errors:
                    branch_reason = "quote_locator_error"
                elif confirmation_inconclusive:
                    branch_reason = "unprompted_quote_confirmation_inconclusive"
                elif truncated_candidate_streams:
                    branch_reason = "quote_retrieval_occurrence_bound_exhausted"
                else:
                    branch_reason = "usable_dialogue_pool_incomplete"
            else:
                kind = "paraphrase"
                if retrieval_candidate_hit_count and confirmation_rejected_count:
                    branch_reason = "prompted_asr_hits_rejected_by_unprompted_confirmation"
                elif exact_contiguous_required and fuzzy_only_count:
                    branch_reason = (
                        "short_common_quote_fuzzy_only_no_exact_contiguous_timed_asr_match")
                elif exact_contiguous_required:
                    branch_reason = (
                        "short_common_quote_no_exact_contiguous_timed_asr_match")
                else:
                    branch_reason = "no_qualifying_timed_asr_phrase_match"
            match = best
            fuzzy_only_match = None
            if fuzzy_only_best is not None:
                fuzzy_src, fuzzy_span = fuzzy_only_best
                fuzzy_only_match = {
                    "source_id": str(getattr(fuzzy_src, "id", "") or ""),
                    "source_title": str(getattr(fuzzy_src, "title", "") or ""),
                    "timed_asr_span": [
                        round(float(fuzzy_span[0]), 3), round(float(fuzzy_span[1]), 3)],
                    "timed_asr_ratio": round(float(fuzzy_span[2]), 3),
                    "rejected_reason": "short_common_quote_requires_exact_contiguous_match",
                }
            branch = {
                "authored_quote": quote,
                "branch": kind,
                "branch_reason": branch_reason,
                "verbatim_required": kind != "paraphrase",
                "quote_match_policy": ("exact_contiguous_short_common"
                                       if exact_contiguous_required
                                       else "fuzzy_distinctive_phrase"),
                "requires_exact_contiguous_match": exact_contiguous_required,
                "scan_ratio_floor": QUOTE_DIALOGUE_FLOOR,
                "selected_window_ratio_floor": QUOTE_DIALOGUE_FLOOR,
                "pool_sources_indexed": indexed,
                "dialogue_eligible_sources_scanned": len(streams),
                "commentary_sources_excluded": rejected_commentary,
                "asr_prompt_fingerprint_expected": expected_asr_fingerprint,
                "quote_retrieval_fingerprint_expected": (
                    _index._quote_retrieval_fingerprint(proj, cfg)),
                "asr_provenance_invalid_source_count": len(invalid_provenance),
                "asr_provenance_invalid_sources": invalid_provenance,
                "confirmation_decoder_fingerprint_expected": (
                    _quote_confirmation_decoder_fingerprint(cfg)),
                "prompted_pool_hit_count": prompted_pool_hit_count,
                "general_pool_hit_count": general_pool_hit_count,
                "retrieval_candidate_hit_count": retrieval_candidate_hit_count,
                "retrieval_occurrence_bound": QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM,
                "retrieval_truncated_stream_count": len(truncated_candidate_streams),
                "retrieval_truncated_streams": truncated_candidate_streams,
                "quote_locator_error_count": len(locator_errors),
                "quote_locator_errors": locator_errors,
                "unprompted_confirmation_attempt_count": len(confirmation_attempts),
                "unprompted_confirmation_confirmed_count": len(matches),
                "unprompted_confirmation_rejected_count": confirmation_rejected_count,
                "unprompted_confirmation_inconclusive_count": len(
                    confirmation_inconclusive),
                "unprompted_confirmation_inconclusive": confirmation_inconclusive,
                "unprompted_confirmation_attempts": confirmation_attempts,
                "pool_match": match,
                "rejected_fuzzy_pool_match_count": fuzzy_only_count,
                "rejected_fuzzy_pool_match": fuzzy_only_match,
                # Complete, occurrence-deduplicated whole-pool location evidence. This does not alter
                # quote typing or ranking.  A bounded scoped recovery rung may try these exact
                # windows before buying another URL; every candidate still faces native-HD,
                # selected-window, strict-vision and lineage gates.
                "pool_match_count": len(matches),
                "pool_matches": matches,
            }
            by_quote[key] = branch
        # Each entry owns its dict so selected-window evidence can be appended without leaking to a
        # neighbouring beat that happens to carry the same authored phrase.
        out[int(getattr(seg, "index", -1))] = json.loads(json.dumps(branch))
    return out


def _quote_pool_input_key(proj, segments, cfg) -> tuple:
    """Identity of every input that can change whole-pool quote classification."""
    from . import index as _index

    def _stat(path) -> tuple[int, int]:
        try:
            value = Path(path).stat()
            return int(value.st_size), int(value.st_mtime_ns)
        except Exception:                                # noqa: BLE001 — absence is state too
            return 0, 0

    sources = []
    for source in sorted((getattr(proj, "sources", None) or []),
                         key=lambda item: str(getattr(item, "id", "") or "")):
        sid = str(getattr(source, "id", "") or "")
        sources.append((
            sid,
            str(getattr(source, "status", "") or ""),
            str(getattr(source, "title", "") or ""),
            _stat(getattr(source, "local_path", "") or ""),
            _stat(proj.shots_path(sid)),
            _stat(Path(proj.index_dir) / f"{sid}.words.json"),
            _stat(Path(proj.index_dir) / f"{sid}.quote_retrieval.json"),
            _stat(Path(proj.index_dir) / f"{sid}.index.meta.json"),
        ))
    try:
        asr_generation = str(_index.asr_semantic_fingerprint(proj, cfg) or "")
    except Exception:                                   # noqa: BLE001 — same fail-closed generation
        asr_generation = ""
    try:
        retrieval_generation = str(_index._quote_retrieval_fingerprint(proj, cfg) or "")
    except Exception:
        retrieval_generation = ""
    beat_state = tuple(
        (int(getattr(seg, "index", -1)),
         str(getattr(seg, "quote", "") or "").strip(),
         str(getattr(seg, "visual_policy", "") or ""),
         _policy.policy_of(seg))
        for seg in (segments or []))
    return tuple(sources), asr_generation, retrieval_generation, beat_state


class _RequestQuotePoolClassificationCache:
    """Opaque, one-request memo for internally-built quote branches.

    Callers can request reuse but cannot supply their own ``paraphrase`` facts.  Every hit is bound
    to source/index provenance plus authored quote/resolved-policy state; a changed generation is a
    miss.  The object is deliberately created inside one recovery request and never persisted.
    """
    __slots__ = ("__key", "__contracts")

    def __init__(self):
        self.__key = None
        self.__contracts = None

    def contracts_for(self, proj, segments, *, cfg=None) -> dict[int, dict]:
        if cfg is None:
            from .config import load_clip_config
            cfg = load_clip_config()
        key = _quote_pool_input_key(proj, segments, cfg)
        if self.__contracts is None or self.__key != key:
            self.__contracts = _quote_pool_branches(proj, segments, cfg=cfg)
            self.__key = key
        # Never expose the memoized branch dict itself.  The evaluator may enrich beat-local quote
        # evidence, and an external caller holding a mutable reference must not be able to rewrite a
        # real quote into a cached paraphrase.
        return {int(idx): json.loads(json.dumps(branch))
                for idx, branch in self.__contracts.items()}

    def invalidate(self) -> None:
        self.__key = None
        self.__contracts = None


def _exact_quote_audio_transfer_evidence(
        proj, sel, authored_quote: str, quote_contract: dict, evidence: dict, *, cfg=None) \
        -> tuple[bool, str, dict]:
    """Validate a cross-copy PCM quote location against current immutable inputs.

    The authoritative reference remains the whole-pool timed-ASR match.  Audio alignment merely
    transfers that already-typed phrase into a clean copy whose *current* ASR missed it.  Every
    byte/time/score binding is checked again here; a target cannot type its own quote, and changing
    either media file or the final trim invalidates the proof.
    """
    import math as _math_transfer

    from . import audio_align as _audio_transfer
    from . import verify as _verify_transfer

    detail = dict(evidence or {}) if isinstance(evidence, dict) else {}
    shape_reason = _audio_transfer.transfer_evidence_shape_reason(evidence)
    if shape_reason:
        return False, shape_reason, detail
    # This path is never a substitute for whole-pool quote typing.
    if str((quote_contract or {}).get("branch", "") or "") != "verbatim":
        return False, "quote_contract_not_verbatim", detail

    quote = " ".join(str(authored_quote or "").strip().split())
    if str(evidence.get("authored_quote", "") or "") != quote:
        return False, "authored_quote_mismatch", detail
    target_sid = str(getattr(sel, "source_id", "") or "")
    reference_sid = str(evidence.get("reference_source_id", "") or "")
    if str(evidence.get("target_source_id", "") or "") != target_sid:
        return False, "target_source_mismatch", detail

    def _pair(value):
        try:
            start, end = float(value[0]), float(value[1])
        except (IndexError, TypeError, ValueError):
            return None
        if not (_math_transfer.isfinite(start) and _math_transfer.isfinite(end)
                and end > start >= 0.0):
            return None
        return start, end

    reference_span = _pair(evidence.get("reference_quote_span"))
    reference_extract = _pair(evidence.get("reference_extract_window"))
    target_span = _pair(evidence.get("target_quote_span"))
    target_search = _pair(evidence.get("target_search_window"))
    selected_bound = _pair(evidence.get("target_selected_window"))
    if None in (reference_span, reference_extract, target_span, target_search, selected_bound):
        return False, "evidence_window_invalid", detail
    assert reference_span is not None and reference_extract is not None
    assert target_span is not None and target_search is not None and selected_bound is not None
    if not (reference_extract[0] <= reference_span[0] < reference_span[1]
            <= reference_extract[1]):
        return False, "reference_quote_outside_extract", detail
    if not (target_search[0] <= target_span[0] < target_span[1] <= target_search[1]):
        return False, "target_quote_outside_search", detail
    # Direct PCM correlation preserves duration; a record claiming time-warped words was not made
    # by this algorithm generation.
    if abs((reference_span[1] - reference_span[0])
           - (target_span[1] - target_span[0])) > 0.002:
        return False, "transferred_quote_duration_mismatch", detail

    selection_window = (
        float(getattr(sel, "in_point", 0.0) or 0.0),
        float(getattr(sel, "out_point", 0.0) or 0.0),
    )
    if (not (selection_window[1] > selection_window[0] >= 0.0)
            or max(abs(selection_window[0] - selected_bound[0]),
                   abs(selection_window[1] - selected_bound[1])) > 0.001):
        return False, "target_selected_window_mismatch", detail
    tolerance = float(QUOTE_WINDOW_TOLERANCE_SEC)
    if not (target_span[0] >= selection_window[0] - tolerance
            and target_span[1] <= selection_window[1] + tolerance):
        return False, "transferred_quote_outside_selected_window", detail

    raw_matches = list((quote_contract or {}).get("pool_matches") or [])
    if not raw_matches and isinstance((quote_contract or {}).get("pool_match"), dict):
        raw_matches = [quote_contract["pool_match"]]
    authoritative_match = None
    for match in raw_matches:
        if str((match or {}).get("source_id", "") or "") != reference_sid:
            continue
        pool_span = _pair((match or {}).get("timed_asr_span"))
        try:
            pool_ratio = float((match or {}).get("timed_asr_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if (pool_span is not None
                and max(abs(pool_span[0] - reference_span[0]),
                        abs(pool_span[1] - reference_span[1])) <= 0.001
                and abs(pool_ratio - float(evidence["reference_asr_ratio"])) <= 0.001
                and pool_ratio >= QUOTE_DIALOGUE_FLOOR):
            authoritative_match = match
            break
    if authoritative_match is None:
        return False, "reference_match_not_authoritative_in_current_contract", detail
    confirmation = authoritative_match.get("unprompted_confirmation")
    if not isinstance(confirmation, dict) or confirmation.get("status") != "confirmed":
        return False, "reference_unprompted_confirmation_absent", detail
    expected_confirmation_decoder = str((quote_contract or {}).get(
        "confirmation_decoder_fingerprint_expected", "") or "").lower()
    confirmation_key = str(confirmation.get("artifact_key", "") or "").lower()
    confirmation_decoder = str(
        confirmation.get("decoder_fingerprint", "") or "").lower()
    if (len(confirmation_key) != 64
            or any(ch not in "0123456789abcdef" for ch in confirmation_key)):
        return False, "reference_unprompted_confirmation_artifact_invalid", detail
    if (len(confirmation_decoder) != 64
            or confirmation_decoder != expected_confirmation_decoder):
        return False, "reference_unprompted_confirmation_decoder_invalid", detail
    if str(evidence.get("reference_quote_confirmation_artifact_key", "") or "").lower() \
            != confirmation_key:
        return False, "reference_unprompted_confirmation_artifact_mismatch", detail
    if str(evidence.get(
            "reference_quote_confirmation_decoder_fingerprint", "") or "").lower() \
            != confirmation_decoder:
        return False, "reference_unprompted_confirmation_decoder_mismatch", detail
    if cfg is None:
        from .config import load_clip_config
        cfg = load_clip_config()
    reference_source_for_confirmation = proj.source(reference_sid)
    prompted_span = authoritative_match.get("prompted_asr_span")
    try:
        revalidated_confirmation = _confirm_prompted_quote_span_unprompted(
            proj, reference_source_for_confirmation, quote, prompted_span, cfg,
            exact_contiguous_required=bool((quote_contract or {}).get(
                "requires_exact_contiguous_match")))
    except Exception as exc:
        return False, f"reference_unprompted_confirmation_error:{type(exc).__name__}", detail
    if revalidated_confirmation.get("status") != "confirmed":
        return False, "reference_unprompted_confirmation_no_longer_current", detail
    if str(revalidated_confirmation.get("artifact_key", "") or "").lower() != confirmation_key:
        return False, "reference_unprompted_confirmation_binding_changed", detail
    revalidated_span = _pair(revalidated_confirmation.get("confirmed_span"))
    if (revalidated_span is None
            or max(abs(revalidated_span[0] - reference_span[0]),
                   abs(revalidated_span[1] - reference_span[1])) > 0.001):
        return False, "reference_unprompted_confirmation_span_changed", detail

    expected_asr = str((quote_contract or {}).get(
        "asr_prompt_fingerprint_expected", "") or "")
    ref_asr_ok, _actual_ref_asr, ref_asr_reason = _source_asr_provenance(
        proj, reference_sid, expected_asr)
    if not ref_asr_ok:
        return False, f"reference_asr_provenance_{ref_asr_reason}", detail

    reference_source = proj.source(reference_sid)
    target_source = proj.source(target_sid)
    if reference_source is None or target_source is None:
        return False, "bound_source_missing", detail
    reference_fp = _verify_transfer._file_fingerprint(
        getattr(reference_source, "local_path", "") or "")
    target_fp = _verify_transfer._file_fingerprint(
        getattr(target_source, "local_path", "") or "")
    if reference_fp in ("", "missing", "unreadable") \
            or reference_fp != evidence.get("reference_source_content_fingerprint"):
        return False, "reference_source_content_fingerprint_mismatch", detail
    if target_fp in ("", "missing", "unreadable") \
            or target_fp != evidence.get("target_source_content_fingerprint"):
        return False, "target_source_content_fingerprint_mismatch", detail

    try:
        correlation = float(evidence["correlation"])
        runner_up = float(evidence["runner_up_correlation"])
        uniqueness = float(evidence["uniqueness_margin"])
    except (KeyError, TypeError, ValueError):
        return False, "alignment_scores_invalid", detail
    if not all(_math_transfer.isfinite(value)
               for value in (correlation, runner_up, uniqueness)):
        return False, "alignment_scores_nonfinite", detail
    if not (0.0 <= runner_up <= correlation <= 1.000001):
        return False, "alignment_score_order_invalid", detail
    if abs((correlation - runner_up) - uniqueness) > 0.00001:
        return False, "alignment_uniqueness_margin_inconsistent", detail
    if correlation < _audio_transfer.AUDIO_QUOTE_TRANSFER_MIN_CORRELATION:
        return False, "alignment_correlation_below_floor", detail
    if uniqueness < _audio_transfer.AUDIO_QUOTE_TRANSFER_MIN_UNIQUENESS_MARGIN:
        return False, "alignment_uniqueness_below_floor", detail
    return True, "", detail


def exact_quote_dialogue_evidence(proj, sel, seg, *, quote_contract=None, cfg=None) \
        -> tuple[bool, str, dict]:
    """Prove an authored quote is spoken at the selected source window.

    Face-ID or a visually plausible close-up cannot identify a dialogue moment.  Require both the
    matcher's strong dialogue score and an independently re-located phrase in the source's timed ASR
    within the aired trim (allowing only a small ASR/editorial boundary tolerance).  A strict exact
    still is handled before this moving-footage branch and therefore remains the explicit fallback.
    """
    quote = str(getattr(seg, "quote", "") or "").strip()
    if not quote:
        return True, "", {}
    signals = getattr(sel, "signals", {}) or {}
    try:
        dialogue = float(signals.get("dialogue", 0.0) or 0.0)
    except (TypeError, ValueError):
        dialogue = 0.0
    detail = dict(quote_contract or {})
    detail.setdefault("authored_quote", quote)
    detail["dialogue_signal"] = round(dialogue, 3)
    # No source in the complete usable dialogue pool contains this authored phrase.  It is an
    # analyzer paraphrase/narration hint, not a promise that a character says those words.  Only the
    # verbatim floor is skipped; the ordinary strict visual verifier below remains authoritative.
    if detail.get("branch") == "paraphrase":
        return True, "", detail
    # A selected source/window is not allowed to type its own authored quote.  When the complete
    # usable dialogue pool could not be classified, trusting that same selected ASR would recreate
    # the original circular contract and could even admit commentary audio excluded by the pool
    # gate.  Unknown therefore remains a hard, separately-auditable failure.
    if detail.get("branch") != "verbatim":
        return False, "exact_quote_pool_classification_indeterminate", detail
    if dialogue < QUOTE_DIALOGUE_FLOOR:
        return False, "exact_quote_dialogue_signal_below_floor", detail
    if cfg is None:
        from .config import load_clip_config
        cfg = load_clip_config()
    source_id = str(getattr(sel, "source_id", "") or "")
    expected_fingerprint = str(detail.get("asr_prompt_fingerprint_expected", "") or "")
    provenance_ok, actual_fingerprint, provenance_reason = _source_asr_provenance(
        proj, source_id, expected_fingerprint)
    detail.update({
        "selected_asr_prompt_fingerprint": actual_fingerprint,
        "selected_asr_provenance_status": "current" if provenance_ok else provenance_reason,
    })
    if not provenance_ok:
        return False, "exact_quote_selected_asr_provenance_invalid", detail
    w0 = float(getattr(sel, "in_point", 0.0) or 0.0)
    w1 = float(getattr(sel, "out_point", 0.0) or 0.0)
    candidate_spans: list[tuple] = []
    exact_contiguous_required = bool(detail.get("requires_exact_contiguous_match", False))
    try:
        from . import index as _index
        words = _index.load_words(proj, source_id)
        # Search the selected neighborhood, not the source-global strongest occurrence. Repeated
        # dialogue can have an earlier equal/better occurrence while this trim contains another
        # valid one. The independent decode is then bound to this local prompted hint.
        selected_words = []
        for row in words:
            try:
                start, end = float(row[0]), float(row[1])
            except (IndexError, TypeError, ValueError):
                continue
            if (end >= w0 - QUOTE_WINDOW_TOLERANCE_SEC
                    and start <= w1 + QUOTE_WINDOW_TOLERANCE_SEC):
                selected_words.append(row)
        exact_contiguous_required = bool(detail.get(
            "requires_exact_contiguous_match",
            _quote_requires_exact_contiguous_match(quote, index_module=_index)))
        candidate_spans.extend(_prompted_quote_candidate_spans(
            selected_words, quote,
            exact_contiguous_required=exact_contiguous_required,
            index_module=_index))
        detail["quote_location_method"] = (
            "exact_contiguous_timed_asr_candidates" if exact_contiguous_required
            else "fuzzy_phrase_timed_asr_candidates")
    except Exception:
        candidate_spans = []
    # The clean general stream intentionally has no authored sentence prompt and can miss a real
    # line. Add every independently confirmed whole-pool retrieval candidate for this exact source;
    # the confirmer is called again below to revalidate its immutable artifact under active cfg.
    for pool_match in (detail.get("pool_matches") or []):
        if str(pool_match.get("source_id", "") or "") != source_id:
            continue
        prompted = pool_match.get("prompted_asr_span")
        try:
            candidate_spans.append((
                float(prompted[0]), float(prompted[1]), float(prompted[2])))
        except (IndexError, TypeError, ValueError):
            continue
    unique_candidates = []
    seen_candidate_spans = set()
    for candidate in candidate_spans:
        try:
            key = tuple(round(float(value), 3) for value in candidate)
        except (TypeError, ValueError):
            continue
        if key not in seen_candidate_spans:
            seen_candidate_spans.add(key)
            unique_candidates.append(candidate)
    if not unique_candidates:
        transfer = signals.get("quote_audio_transfer_evidence")
        if transfer is not None:
            transfer_ok, transfer_reason, transfer_detail = \
                _exact_quote_audio_transfer_evidence(
                    proj, sel, quote, detail, transfer, cfg=cfg)
            detail["audio_transfer"] = transfer_detail
            detail["quote_location_method"] = "audio_transfer"
            if transfer_ok:
                return True, "", detail
            detail["audio_transfer_status"] = transfer_reason
            return False, "exact_quote_audio_transfer_evidence_invalid", detail
        return False, "exact_quote_timed_asr_span_absent", detail
    src = proj.source(source_id)
    confirmation_attempts = []
    confirmed_choice = None
    confirmed_outside = False
    for span in unique_candidates[:QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM]:
        try:
            confirmation = _confirm_prompted_quote_span_unprompted(
                proj, src, quote, span, cfg,
                exact_contiguous_required=exact_contiguous_required)
        except Exception as exc:
            confirmation = {
                "schema_version": QUOTE_CONFIRMATION_SCHEMA,
                "algorithm": QUOTE_CONFIRMATION_ALGORITHM,
                "status": "inconclusive",
                "reason": f"confirmation_internal_error:{type(exc).__name__}",
            }
        summary = _quote_confirmation_summary(confirmation)
        confirmation_attempts.append(summary)
        if str(confirmation.get("status", "") or "") != "confirmed":
            continue
        confirmed_span = confirmation.get("confirmed_span")
        try:
            q0, q1, ratio = (
                float(confirmed_span[0]), float(confirmed_span[1]),
                float(confirmed_span[2]))
        except (IndexError, TypeError, ValueError):
            continue
        contained_here = (
            q1 >= q0 >= 0.0
            and q0 >= w0 - QUOTE_WINDOW_TOLERANCE_SEC
            and q1 <= w1 + QUOTE_WINDOW_TOLERANCE_SEC)
        if contained_here:
            confirmed_choice = (q0, q1, ratio, confirmation)
            break
        confirmed_outside = True
    detail["unprompted_confirmation_attempts"] = confirmation_attempts
    if confirmed_choice is None:
        # PCM transfer is independent acoustic evidence and remains a valid alternate route when
        # it is bound to an independently confirmed whole-pool reference quote.
        transfer = signals.get("quote_audio_transfer_evidence")
        if transfer is not None:
            transfer_ok, transfer_reason, transfer_detail = \
                _exact_quote_audio_transfer_evidence(
                    proj, sel, quote, detail, transfer, cfg=cfg)
            detail["audio_transfer"] = transfer_detail
            detail["quote_location_method"] = "audio_transfer"
            if transfer_ok:
                return True, "", detail
            detail["audio_transfer_status"] = transfer_reason
        statuses = {str(row.get("status", "") or "") for row in confirmation_attempts}
        if "inconclusive" in statuses or len(unique_candidates) \
                > QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM:
            return False, "exact_quote_unprompted_confirmation_inconclusive", detail
        if confirmed_outside:
            return False, "exact_quote_timed_asr_outside_selected_window", detail
        if "rejected" in statuses:
            return False, "exact_quote_unprompted_confirmation_rejected", detail
        return False, "exact_quote_unprompted_confirmation_inconclusive", detail
    q0, q1, ratio, confirmation = confirmed_choice
    detail["unprompted_confirmation"] = _quote_confirmation_summary(confirmation)
    detail["quote_location_method"] = str(
        confirmation.get("match_method", "") or "unprompted_confirmation")
    gap = max(0.0, w0 - q1, q0 - w1)
    start_delta = q0 - w0
    end_delta = q1 - w1
    start_margin = q0 - (w0 - QUOTE_WINDOW_TOLERANCE_SEC)
    end_margin = (w1 + QUOTE_WINDOW_TOLERANCE_SEC) - q1
    detail.update({
        "timed_asr_span": [round(q0, 3), round(q1, 3)],
        "timed_asr_ratio": round(ratio, 3),
        "selected_window": [round(w0, 3), round(w1, 3)],
        "window_gap_sec": round(gap, 3),
        "window_tolerance_sec": QUOTE_WINDOW_TOLERANCE_SEC,
        "quote_start_vs_window_start_sec": round(start_delta, 3),
        "quote_end_vs_window_end_sec": round(end_delta, 3),
        "start_containment_margin_sec": round(start_margin, 3),
        "end_containment_margin_sec": round(end_margin, 3),
    })
    contained = (q1 >= q0 >= 0.0
                 and q0 >= w0 - QUOTE_WINDOW_TOLERANCE_SEC
                 and q1 <= w1 + QUOTE_WINDOW_TOLERANCE_SEC)
    if not (w1 > w0 >= 0.0) or not contained:
        return False, "exact_quote_timed_asr_outside_selected_window", detail
    return True, "", detail


def evaluate_selection_relevance(proj, segments, *, cfg=None,
                                 quote_pool_cache=None) -> dict:
    """Return the complete semantic publication audit without mutating project or selections.

    ``quote_pool_cache`` may be the opaque request-local cache above. Recovery invokes this audit
    several times while only selections/verifier evidence change; the cache avoids rescanning every
    source for every audit while still constructing and validating every branch internally. Ordinary
    callers omit it and retain the fail-closed, freshly-computed behaviour.
    """
    by_idx = {int(getattr(s, "segment_index", -1)): s for s in (proj.selections or [])}
    char2actor = _project_char2actor(proj)
    if quote_pool_cache is None:
        quote_contracts = _quote_pool_branches(proj, segments, cfg=cfg)
    elif type(quote_pool_cache) is _RequestQuotePoolClassificationCache:
        quote_contracts = quote_pool_cache.contracts_for(proj, segments, cfg=cfg)
    else:
        raise TypeError("quote_pool_cache must be a request-local classification cache")
    checked, blockers = [], []
    skipped_generic = 0

    for seg in segments or []:
        policy = _policy.policy_of(seg)
        if not _strict_semantic_beat(seg):
            skipped_generic += 1
            continue
        idx = int(getattr(seg, "index", -1))
        sel = by_idx.get(idx)
        reasons: list[str] = []
        coverage = "moving_video"
        quote_evidence: dict = dict(quote_contracts.get(idx) or {})

        if sel is None:
            reasons.append("selection_absent")
            verifier = {}
            source_id = ""
        else:
            verifier = dict(getattr(sel, "verifier", {}) or {})
            source_id = str(getattr(sel, "source_id", "") or "")
            still_ok, still_why = verified_still_coverage(sel, seg, proj=proj)
            if still_ok:
                coverage = f"verified_still:{still_why}"
            else:
                if getattr(sel, "image_path", ""):
                    reasons.append(f"invalid_still:{still_why}")
                if not source_id:
                    reasons.append("moving_source_absent")
                status = str(verifier.get("status", "") or "")
                verdict = str(verifier.get("verdict", "") or "")
                if status != "ok":
                    reasons.append(f"verifier_{status or 'absent'}")

                # Any JSON verdict—positive OR negative—is meaningful only for the exact current
                # source/shot/window and prompt. A prompt-version change makes the old false fields
                # technical stale evidence, not a fresh content rejection. Keep the gate blocked on
                # the binding reason alone so scoped re-verification asks the corrected question;
                # never route a stale `specific_enough=false` into new-footage acquisition.
                evidence_reason = selection_verifier_evidence_reason(proj, sel, seg, verifier)
                if evidence_reason:
                    reasons.append(evidence_reason)
                # Every binding failure means these facts answered a different or unprovable
                # question (old model/schema, different bytes/window/prompt, missing sampled
                # pixels, or legacy evidence with no binding at all).  Keep the technical reason
                # as a hard blocker and re-ask the same selection; do not also treat stale false
                # fields as a current semantic rejection that diverts it into acquisition.
                stale_question = bool(evidence_reason)
                if not stale_question:
                    if verdict != "keep":
                        reasons.append(f"verdict_{verdict or 'absent'}")
                    # EXACT means exact: a lenient/contextual question cannot prove an exact scene,
                    # and a later relabel/downgrade must not erase that fact from the gate.
                    evidence = verifier.get("selection_evidence") or {}
                    if policy == _policy.EXACT:
                        if evidence.get("is_specific") is not True:
                            reasons.append("exact_verifier_evidence_not_strict")
                        if verifier.get("downgraded"):
                            reasons.append("exact_moving_verdict_was_downgraded")
                        if str(verifier.get("relevance_class", "") or "") == "contextual_fallback":
                            reasons.append("exact_moving_relevance_is_contextual")
                # The montage guard may demote a multi-subject beat from EXACT to CHARACTER, but a
                # real authored quote remains a timed dialogue promise.  Apply the same pool typing
                # and selected-window floor to every strict beat; only an affirmative paraphrase
                # branch skips it.  Otherwise CHARACTER would merely record "verbatim" in the
                # audit while still allowing an unrelated, silent right-subject window to pass.
                if str(getattr(seg, "quote", "") or "").strip():
                    quote_ok, quote_why, quote_evidence = exact_quote_dialogue_evidence(
                        proj, sel, seg, quote_contract=quote_evidence, cfg=cfg)
                    if not quote_ok:
                        reasons.append(quote_why)

                if not stale_question:
                    # These fields remain strict. Missing values are UNKNOWN, never a pass; only a
                    # stale answer is withheld until the same footage is judged under this prompt.
                    for field in ("matches_narration", "specific_enough", "quality_ok"):
                        value = verifier.get(field)
                        if value is not True:
                            reasons.append(f"{field}_{'false' if value is False else 'absent'}")
                    wrong = verifier.get("wrong_subject_visible")
                    if wrong is not False:
                        reasons.append(
                            f"wrong_subject_visible_{'true' if wrong is True else 'absent'}")

                    # A named required subject must be positively present. The object verifier
                    # defines this as scene/context when the prop itself is too small to resolve.
                    if (getattr(seg, "required_entity", "") or "").strip():
                        visible = verifier.get("correct_subject_visible")
                        if visible is not True:
                            reasons.append(
                                f"correct_subject_visible_{'false' if visible is False else 'absent'}")
                    if verifier.get("contradicts_narration") is True:
                        reasons.append("contradicts_narration_true")
                    if verifier.get("era_ok") is False:
                        reasons.append("era_ok_false")
                    try:
                        must_see = effective_deictic_target(seg)
                    except Exception:
                        must_see = ""
                    if must_see and verifier.get("target_visible") is not True:
                        reasons.append(
                            f"target_visible_{'false' if verifier.get('target_visible') is False else 'absent'}")

                    deterministic = _contradiction_reason(
                        seg, verifier, _selection_source_title(proj, sel), char2actor)
                    if deterministic and "contradicts_narration_true" not in reasons:
                        reasons.append("deterministic_contradiction")
                        verifier = {**verifier, "contract_contradiction_reason": deterministic}
                    cast_warning = (_source_title_exact_cast_conflict(
                        seg, _selection_source_title(proj, sel), char2actor)
                        if policy == _policy.EXACT else "")
                    if (cast_warning and _cast_warning_resolution_reason(
                            verifier, seg, char2actor)):
                        reasons.append("exact_source_title_cast_conflict_unresolved")
                        verifier = {**verifier, "source_title_cast_warning": cast_warning}

        # A verified still suppresses the moving selection entirely, so its old moving-video
        # verdict is audit context rather than a publication blocker.
        if coverage.startswith("verified_still:"):
            reasons = []

        entry = {
            "segment_index": idx,
            "visual_policy": policy,
            "coverage": coverage,
            "source_id": source_id,
            "status": "blocked" if reasons else "pass",
            "reasons": reasons,
            "quote_evidence": quote_evidence,
            "verifier": {
                k: verifier.get(k) for k in (
                    "status", "verdict", "matches_narration", "specific_enough",
                    "correct_subject_visible", "wrong_subject_visible",
                    "contradicts_narration", "quality_ok", "era_ok", "target_visible",
                    "source_title_conflict_resolved", "source_title_cast_warning",
                    "contract_contradiction_reason", "selection_evidence") if k in verifier
            },
        }
        checked.append(entry)
        if reasons:
            blockers.append(entry)

    quote_branch_counts = {"verbatim": 0, "paraphrase": 0, "indeterminate": 0}
    for detail in quote_contracts.values():
        branch = str(detail.get("branch", "") or "")
        if branch in quote_branch_counts:
            quote_branch_counts[branch] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "pass",
        "project": str(getattr(proj, "name", "") or ""),
        "selection_count": len(proj.selections or []),
        "strict_checked": len(checked),
        "generic_or_abstract_skipped": skipped_generic,
        "blocked_count": len(blockers),
        "quote_branch_counts": quote_branch_counts,
        "checked": checked,
        "blockers": blockers,
    }


def write_selection_relevance_audit(path: Path | str, audit: dict) -> Path:
    """Persist a complete audit atomically; an interrupted write never masquerades as a pass."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Contract checks can run concurrently for portal monitoring and publication.  Each writer must
    # own its temporary file so one cannot replace or clean up another writer's in-flight bytes.
    tmp = dest.with_name(dest.name + f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def assert_selection_relevance(proj, segments, audit_path: Path | str | None = None, *, cfg=None) \
        -> dict:
    """Persist the audit and release-block any exact/concrete semantic uncertainty."""
    # A specificity downgrade is evidence about one exact indexed pool, not a permanent rewrite of
    # the authored beat.  When footage/indexes change, revive the full original segment + selection
    # before this strict assertion runs so the new pool gets a chance to satisfy the exact promise.
    # Keep ``evaluate_selection_relevance`` itself read-only for audit/replay callers.
    try:
        from .selfheal import restore_stale_selection_relevance_softenings
        restore_stale_selection_relevance_softenings(proj, segments, cfg=cfg)
    except Exception as exc:
        raise RuntimeError(
            f"could not restore stale pool-bound semantic softening before publication: {exc}"
        ) from exc
    audit = evaluate_selection_relevance(proj, segments, cfg=cfg)
    dest = Path(audit_path) if audit_path is not None else proj.output_dir / AUDIT_FILENAME
    try:
        write_selection_relevance_audit(dest, audit)
    except Exception as exc:
        from .verify import NonRetryableBuildError
        raise NonRetryableBuildError(
            f"selection relevance audit could not be persisted atomically at {dest}: {exc}",
            kind="selection_relevance_audit") from exc
    if audit["blockers"]:
        from .verify import NonRetryableBuildError
        sample = [e["segment_index"] for e in audit["blockers"][:8]]
        raise NonRetryableBuildError(
            f"selection relevance gate: {audit['blocked_count']} exact/concrete beat(s) lack "
            f"positive semantic verification — scene(s) {sample}. See {dest.name}; re-verify or "
            f"recover those beats before rendering.",
            kind="selection_relevance")
    return audit
