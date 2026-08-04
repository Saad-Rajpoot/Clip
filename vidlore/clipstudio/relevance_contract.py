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
from .verify import _contradiction_reason, selection_verifier_evidence_reason

SCHEMA_VERSION = 6
AUDIT_FILENAME = "selection_relevance_audit.json"
QUOTE_DIALOGUE_FLOOR = 0.78
QUOTE_WINDOW_TOLERANCE_SEC = 0.75


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
        must_see = _policy.deictic_target(seg)
    except Exception:
        must_see = ""
    if must_see and evidence.get("target_visible") is not True:
        return "still instructed-look target is not affirmatively visible"
    return ""


def verified_still_coverage(sel, seg) -> tuple[bool, str]:
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
    source = str(meta.get("source", "") or "")
    relevance_class = str(meta.get("relevance_class", "") or "")
    policy = _policy.policy_of(seg)

    if source not in ("source-frame", "source-frame-recovery", "web-exact-scene"):
        return False, f"untrusted still source {source or 'missing'}"
    if policy == _policy.EXACT and relevance_class != "exact_scene":
        return False, f"exact still relevance_class is {relevance_class or 'absent'}, not exact_scene"
    if policy != _policy.EXACT and relevance_class in ("generic_filler", "unverified_fallback", ""):
        return False, f"concrete still relevance_class is {relevance_class or 'absent'}"

    evidence = meta.get("still_verifier") or meta.get("exact_still_verifier") or {}
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
            for src, words in streams:
                try:
                    span = _index.find_quote_span(
                        words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)
                except Exception:
                    span = None
                if span and (best is None or float(span[2]) > float(best[1][2])):
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
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis", {}) or {}
    char2actor = {
        str(c.get("name", "")).strip().lower(): str(c.get("actor", "")).strip()
        for c in (analysis.get("characters") or [])
        if isinstance(c, dict) and c.get("name") and c.get("actor")
    }
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
            still_ok, still_why = verified_still_coverage(sel, seg)
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
                if verdict != "keep":
                    reasons.append(f"verdict_{verdict or 'absent'}")

                # A positive JSON verdict is meaningful only for the exact source/shot/window and
                # prompt pixels it judged.  Resume and late project edits used to retain `keep`
                # while changing the selection underneath it; recompute the persisted binding from
                # current source bytes, shot index/bounds, trim, beat prompt, model and frame/sheet.
                evidence_reason = selection_verifier_evidence_reason(proj, sel, seg, verifier)
                if evidence_reason:
                    reasons.append(evidence_reason)

                # EXACT means exact: a lenient/contextual question cannot prove an exact scene, and
                # a later relabel/downgrade must not erase that fact from the publication gate.
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

                # These fields are part of the longstanding verifier JSON contract. Missing values
                # are UNKNOWN, never a pass: this is what closes verify-disabled and stale resume.
                for field in ("matches_narration", "specific_enough", "quality_ok"):
                    value = verifier.get(field)
                    if value is not True:
                        reasons.append(f"{field}_{'false' if value is False else 'absent'}")
                wrong = verifier.get("wrong_subject_visible")
                if wrong is not False:
                    reasons.append(f"wrong_subject_visible_{'true' if wrong is True else 'absent'}")

                # A named required subject must be positively present. The object verifier defines
                # this as the correct scene/context when the prop itself is too small to resolve.
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
                    must_see = _policy.deictic_target(seg)
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
