"""Compliance ledger for ClipStudio.

One JSONL record per selected clip: source URL, in/out, duration, reuse count, confidence,
the signals that fired, the user's permission assertion, and any review flags. Plus a
`review_queue.json` of everything a human must look at, and a human-readable summary.

IMPORTANT: this ledger TRACKS and FLAGS. It never asserts guaranteed fair use or guaranteed
copyright safety — that determination is the user's, per source (see DESIGN.md §0, §5).
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    ClipProject, ClipSelection, ScriptSegment, SourceVideo,
    FLAG_LOW_CONFIDENCE, FLAG_IDENTITY, FLAG_SPECIFIC_CLAIM,
    FLAG_OVERUSED_SOURCE, FLAG_NO_CANDIDATE, FLAG_VERIFIER_FAILED,
    FLAG_VERIFIER_UNVERIFIED, FLAG_EXACT_MISSING,
)
from .config import ClipConfig
from . import policy as _policy

DISCLAIMER = (
    "ClipStudio records provenance and flags risk. It does NOT certify fair use or copyright "
    "safety. Each source's permission is the user's own assertion. Verify rights before publishing."
)


def evaluate_flags(
    sel: ClipSelection,
    segment: ScriptSegment,
    source: Optional[SourceVideo],
    cfg: ClipConfig,
) -> list[str]:
    """Decide which manual-review flags apply to a selection. Pure; sets nothing."""
    flags: list[str] = []
    # IMAGE-FALLBACK COVERAGE: a beat with no video clip but a real image still IS covered — it must
    # NOT be marked no_candidate. source-frame / web-exact-scene = valid coverage; a
    # source-frame-recovery covers an exact beat VISUALLY but it stays exact_scene_missing (review).
    _img_src = (getattr(sel, "image_meta", {}) or {}).get("source", "")
    _has_img = bool(getattr(sel, "image_path", "")) and bool(_img_src)
    if _has_img:        # an image still AIRS for this beat (build renders image_path over the clip)
        if _img_src == "source-frame-recovery" and _policy.is_exact(segment):
            return [FLAG_EXACT_MISSING]
        return []
    if not sel.source_id or sel.confidence <= 0.0:
        return [FLAG_NO_CANDIDATE]

    # The AI verifier is the PRIMARY gate: a clip it KEPT (subject visible) is trusted and not
    # re-flagged for being a "specific" line or a modest match score.
    v = sel.verifier or {}
    status = v.get("status", "")
    verdict = v.get("verdict", "")
    verifier_kept = status == "ok" and verdict == "keep"
    verifier_failed = status == "ok" and verdict == "replace"   # rejected + no passing alternate
    verifier_errored = status == "error"                        # transient API error — unconfirmed
    verifier_present = status in ("ok", "error")                # the verifier pass actually ran

    if verifier_failed:
        flags.append(FLAG_VERIFIER_FAILED)
    elif verifier_errored:
        flags.append(FLAG_VERIFIER_UNVERIFIED)

    # A specific actor/character is required but neither Face-ID/OCR nor the verifier confirmed it.
    if segment.required_kind in ("actor", "character") and segment.required_entity \
            and not sel.signals.get("faceid", 0.0) \
            and not (verifier_kept and v.get("correct_subject_visible")):
        flags.append(FLAG_IDENTITY)

    if sel.reuse_count >= cfg.max_reuse_per_source:
        flags.append(FLAG_OVERUSED_SOURCE)

    # EXACT-SCENE MISSING (req. 9): an exact_scene beat the verifier could not satisfy — or that has
    # no real footage at all — goes to MANUAL REVIEW (never silent weak filler). Re-derived here so
    # apply_flags' overwrite preserves the flag verify/image-fallback already set. A real source-frame
    # still of the exact scene is acceptable coverage (not "missing").
    _has_real_still = bool(getattr(sel, "image_path", "")) \
        and (getattr(sel, "image_meta", {}) or {}).get("source") == "source-frame"
    if _policy.is_exact(segment) and (verifier_failed or not sel.source_id) and not _has_real_still:
        flags.append(FLAG_EXACT_MISSING)

    # Confidence + specific-claim heuristics are a FALLBACK only when the verifier never ran
    # (e.g. no LLM key). With a verifier verdict present, it is the authority.
    if not verifier_present:
        if sel.confidence < cfg.min_confidence and FLAG_LOW_CONFIDENCE not in flags:
            flags.append(FLAG_LOW_CONFIDENCE)
        if segment.is_specific_claim and FLAG_SPECIFIC_CLAIM not in flags:
            flags.append(FLAG_SPECIFIC_CLAIM)
    return flags


def apply_flags(
    selections: list[ClipSelection],
    segments: list[ScriptSegment],
    proj: ClipProject,
    cfg: ClipConfig,
) -> None:
    """Annotate selections in place with flags + reasons."""
    by_idx = {s.index: s for s in segments}
    for sel in selections:
        seg = by_idx.get(sel.segment_index)
        if seg is None:
            continue
        reasons = evaluate_flags(sel, seg, proj.source(sel.source_id), cfg)
        sel.flag_reasons = reasons
        sel.flagged = bool(reasons)


def write_ledger(proj: ClipProject, segments: list[ScriptSegment]) -> Path:
    """(Re)write the full JSONL ledger from the project's selections."""
    by_idx = {s.index: s for s in segments}
    lines = []
    for sel in sorted(proj.selections, key=lambda s: s.segment_index):
        src = proj.source(sel.source_id)
        seg = by_idx.get(sel.segment_index)
        # HONEST relevance class (req. 4) — one of: exact_scene (verifier-kept moving footage on an
        # exact beat, or a validated web-exact-scene still) · contextual_fallback (right character/
        # scene/era but the exact moment is unconfirmed) · generic_filler (thematic) · exact_scene_
        # missing (an exact beat with no confirmed footage) · unverified.
        _imeta = getattr(sel, "image_meta", {}) or {}
        _rclass = _imeta.get("relevance_class", "")
        if not _rclass:
            _v = getattr(sel, "verifier", {}) or {}
            _kept = _v.get("verdict") == "keep" and _v.get("status") == "ok"
            _pol = (_policy.policy_of(seg) if seg else "")
            _flags = sel.flag_reasons or []
            if "exact_scene_missing" in _flags:
                _rclass = "exact_scene_missing"
            elif sel.source_id and _kept:
                # an exact beat whose EXACT moment was unconfirmed but was kept as a relevant
                # same-scene clip (the exact→contextual downgrade) is HONESTLY a contextual fallback,
                # never labelled exact_scene.
                if _v.get("downgraded") or _v.get("relevance_class") == "contextual_fallback":
                    _rclass = "contextual_fallback"
                else:
                    _rclass = "exact_scene" if _pol == "exact_scene" else "contextual_fallback"
            elif "verifier_failed" in _flags:
                _rclass = "exact_scene_missing" if _pol == "exact_scene" else "contextual_fallback"
            elif sel.source_id:
                _rclass = "unverified"
            else:
                _rclass = "none"
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "segment_index": sel.segment_index,
            "narration": (seg.text if seg else ""),
            "visual_policy": (_policy.policy_of(seg) if seg else ""),
            "source_id": sel.source_id,
            "source_url": (src.url if src else sel.source_url),
            "source_title": (src.title if src else ""),
            "permission": (src.permission if src else "unverified"),
            "permission_note": (src.permission_note if src else ""),
            "in_point": round(sel.in_point, 3),
            "out_point": round(sel.out_point, 3),
            "duration": round(sel.duration, 3),
            "clip_path": sel.clip_path,
            "image_path": getattr(sel, "image_path", ""),
            "image_source": (getattr(sel, "image_meta", {}) or {}).get("source", ""),
            "relevance_class": _rclass,
            "reuse_count": sel.reuse_count,
            "confidence": round(sel.confidence, 4),
            "signals": {k: round(float(v), 4) for k, v in sel.signals.items()},
            "flagged": sel.flagged,
            "flag_reasons": sel.flag_reasons,
            "approved": sel.approved,
        }
        lines.append(json.dumps(rec))
    proj.ledger_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return proj.ledger_path


def write_review_queue(proj: ClipProject, segments: list[ScriptSegment]) -> Path:
    """Write the subset of selections a human must review."""
    by_idx = {s.index: s for s in segments}
    queue = []
    for sel in sorted(proj.selections, key=lambda s: s.segment_index):
        if not sel.flagged or sel.approved:
            continue
        seg = by_idx.get(sel.segment_index)
        src = proj.source(sel.source_id)
        queue.append({
            "segment_index": sel.segment_index,
            "narration": (seg.text if seg else ""),
            "reasons": sel.flag_reasons,
            "confidence": round(sel.confidence, 4),
            "source_id": sel.source_id,
            "source_url": (src.url if src else sel.source_url),
            "in_point": round(sel.in_point, 3),
            "out_point": round(sel.out_point, 3),
            "clip_path": sel.clip_path,
            "alternates": [a.to_dict() for a in sel.alternates],
        })
    payload = {"disclaimer": DISCLAIMER, "count": len(queue), "items": queue}
    proj.review_queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return proj.review_queue_path


def summary(proj: ClipProject) -> dict:
    """Aggregate compliance/quality stats for the project."""
    sels = proj.selections
    n = len(sels)
    confs = [s.confidence for s in sels if s.source_id]
    flagged = [s for s in sels if s.flagged and not s.approved]
    src_counts = Counter(s.source_id for s in sels if s.source_id)
    reason_counts: Counter = Counter()
    for s in flagged:
        reason_counts.update(s.flag_reasons)
    total_dur = sum(s.duration for s in sels)
    dominant = src_counts.most_common(1)[0] if src_counts else ("", 0)
    return {
        "disclaimer": DISCLAIMER,
        "segments": n,
        "with_clip": sum(1 for s in sels if s.source_id),
        "flagged_for_review": len(flagged),
        "flag_breakdown": dict(reason_counts),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "min_confidence": round(min(confs), 4) if confs else 0.0,
        "sources_used": len(src_counts),
        "source_reuse": dict(src_counts),
        "dominant_source": {"source_id": dominant[0], "clips": dominant[1],
                            "share": round(dominant[1] / n, 3) if n else 0.0},
        "total_clip_seconds": round(total_dur, 1),
        "permission_unverified_sources": [s.id for s in proj.sources
                                          if s.permission == "unverified"],
    }


def finalize(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig) -> dict:
    """One call after matching: flag, write ledger + queue, return summary."""
    apply_flags(proj.selections, segments, proj, cfg)
    write_ledger(proj, segments)
    write_review_queue(proj, segments)
    return summary(proj)
