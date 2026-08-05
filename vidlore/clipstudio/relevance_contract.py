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
SCHEMA_VERSION = 9
AUDIT_FILENAME = "selection_relevance_audit.json"
QUOTE_DIALOGUE_FLOOR = 0.78
QUOTE_WINDOW_TOLERANCE_SEC = 0.75

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

    streams: list[tuple[object, list]] = []
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
        # An empty cache with current provenance is a valid observation of silence.  Retain it in
        # the scanned set so an entirely silent but fully-current pool can classify a quote as a
        # paraphrase rather than pretending the index is absent.
        streams.append((src, words))

    by_quote: dict[str, dict] = {}
    out: dict[int, dict] = {}
    for seg in quoted:
        quote = str(getattr(seg, "quote", "") or "").strip()
        key = " ".join(quote.lower().split())
        branch = by_quote.get(key)
        if branch is None:
            best = None
            matches: list[dict] = []
            for src, words in streams:
                try:
                    span = _index.find_quote_span(
                        words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)
                except Exception:
                    span = None
                if span:
                    match = {
                        "source_id": str(getattr(src, "id", "") or ""),
                        "source_title": str(getattr(src, "title", "") or ""),
                        "timed_asr_span": [
                            round(float(span[0]), 3), round(float(span[1]), 3)],
                        "timed_asr_ratio": round(float(span[2]), 3),
                    }
                    matches.append(match)
                    # Preserve the historical ``pool_match`` choice exactly: first source at the
                    # strongest phrase ratio wins.  Recovery consumes ``pool_matches`` below so an
                    # arbitrary equal-ratio first source (often a 360p copy) is evidence of
                    # existence, not an accidental recommendation.
                    if best is None or float(span[2]) > float(best[1][2]):
                        best = (src, span)
            if best is not None:
                kind = "verbatim"
            elif invalid_provenance or not streams:
                kind = "indeterminate"
            else:
                kind = "paraphrase"
            match = None
            if best is not None:
                src, span = best
                match = {
                    "source_id": str(getattr(src, "id", "") or ""),
                    "source_title": str(getattr(src, "title", "") or ""),
                    "timed_asr_span": [round(float(span[0]), 3), round(float(span[1]), 3)],
                    "timed_asr_ratio": round(float(span[2]), 3),
                }
            branch = {
                "authored_quote": quote,
                "branch": kind,
                "verbatim_required": kind != "paraphrase",
                "scan_ratio_floor": QUOTE_DIALOGUE_FLOOR,
                "selected_window_ratio_floor": QUOTE_DIALOGUE_FLOOR,
                "pool_sources_indexed": indexed,
                "dialogue_eligible_sources_scanned": len(streams),
                "commentary_sources_excluded": rejected_commentary,
                "asr_prompt_fingerprint_expected": expected_asr_fingerprint,
                "asr_provenance_invalid_source_count": len(invalid_provenance),
                "asr_provenance_invalid_sources": invalid_provenance,
                "pool_match": match,
                # Complete, source-deduplicated whole-pool location evidence.  This does not alter
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
            _stat(Path(proj.index_dir) / f"{sid}.index.meta.json"),
        ))
    try:
        asr_generation = str(_index.asr_semantic_fingerprint(proj, cfg) or "")
    except Exception:                                   # noqa: BLE001 — same fail-closed generation
        asr_generation = ""
    beat_state = tuple(
        (int(getattr(seg, "index", -1)),
         str(getattr(seg, "quote", "") or "").strip(),
         str(getattr(seg, "visual_policy", "") or ""),
         _policy.policy_of(seg))
        for seg in (segments or []))
    return tuple(sources), asr_generation, beat_state


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
        proj, sel, authored_quote: str, quote_contract: dict, evidence: dict) \
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


def exact_quote_dialogue_evidence(proj, sel, seg, *, quote_contract=None) \
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
    try:
        from . import index as _index
        words = _index.load_words(proj, source_id)
        span = _index.find_quote_span(words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)
    except Exception:
        span = None
    if not span:
        transfer = signals.get("quote_audio_transfer_evidence")
        if transfer is not None:
            transfer_ok, transfer_reason, transfer_detail = \
                _exact_quote_audio_transfer_evidence(
                    proj, sel, quote, detail, transfer)
            detail["audio_transfer"] = transfer_detail
            detail["quote_location_method"] = "audio_transfer"
            if transfer_ok:
                return True, "", detail
            detail["audio_transfer_status"] = transfer_reason
            return False, "exact_quote_audio_transfer_evidence_invalid", detail
        return False, "exact_quote_timed_asr_span_absent", detail
    q0, q1, ratio = float(span[0]), float(span[1]), float(span[2])
    w0 = float(getattr(sel, "in_point", 0.0) or 0.0)
    w1 = float(getattr(sel, "out_point", 0.0) or 0.0)
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
                        proj, sel, seg, quote_contract=quote_evidence)
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
    tmp = dest.with_suffix(dest.suffix + ".tmp")
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
        restore_stale_selection_relevance_softenings(proj, segments)
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
