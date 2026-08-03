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
from .verify import _contradiction_reason, selection_verifier_evidence_reason

SCHEMA_VERSION = 3
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


def exact_quote_dialogue_evidence(proj, sel, seg) -> tuple[bool, str, dict]:
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
    detail = {"authored_quote": quote, "dialogue_signal": round(dialogue, 3)}
    if dialogue < QUOTE_DIALOGUE_FLOOR:
        return False, "exact_quote_dialogue_signal_below_floor", detail
    try:
        from . import index as _index
        words = _index.load_words(proj, str(getattr(sel, "source_id", "") or ""))
        span = _index.find_quote_span(words, quote, min_ratio=QUOTE_DIALOGUE_FLOOR)
    except Exception:
        span = None
    if not span:
        return False, "exact_quote_timed_asr_span_absent", detail
    q0, q1, ratio = float(span[0]), float(span[1]), float(span[2])
    w0 = float(getattr(sel, "in_point", 0.0) or 0.0)
    w1 = float(getattr(sel, "out_point", 0.0) or 0.0)
    gap = max(0.0, w0 - q1, q0 - w1)
    detail.update({
        "timed_asr_span": [round(q0, 3), round(q1, 3)],
        "timed_asr_ratio": round(ratio, 3),
        "selected_window": [round(w0, 3), round(w1, 3)],
        "window_gap_sec": round(gap, 3),
    })
    if not (w1 > w0 >= 0.0) or gap > QUOTE_WINDOW_TOLERANCE_SEC:
        return False, "exact_quote_timed_asr_outside_selected_window", detail
    return True, "", detail


def evaluate_selection_relevance(proj, segments) -> dict:
    """Return the complete semantic publication audit without mutating project or selections."""
    by_idx = {int(getattr(s, "segment_index", -1)): s for s in (proj.selections or [])}
    analysis = (getattr(proj, "meta", {}) or {}).get("analysis", {}) or {}
    char2actor = {
        str(c.get("name", "")).strip().lower(): str(c.get("actor", "")).strip()
        for c in (analysis.get("characters") or [])
        if isinstance(c, dict) and c.get("name") and c.get("actor")
    }
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
        quote_evidence: dict = {}

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
                    quote_ok, quote_why, quote_evidence = exact_quote_dialogue_evidence(
                        proj, sel, seg)
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

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "pass",
        "project": str(getattr(proj, "name", "") or ""),
        "selection_count": len(proj.selections or []),
        "strict_checked": len(checked),
        "generic_or_abstract_skipped": skipped_generic,
        "blocked_count": len(blockers),
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


def assert_selection_relevance(proj, segments, audit_path: Path | str | None = None) -> dict:
    """Persist the audit and release-block any exact/concrete semantic uncertainty."""
    audit = evaluate_selection_relevance(proj, segments)
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
