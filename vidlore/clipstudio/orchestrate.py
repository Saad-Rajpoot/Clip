"""The end-to-end pipeline: ingest → index → segment → match → cut → ledger → build.

One call (`produce`) runs the whole thing, saving the project + ledger + review queue after each
stage so a run is resumable and every intermediate decision is inspectable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import ClipProject, SOURCE_OK, SOURCE_BLOCKED, SOURCE_FAILED
from .config import ClipConfig, load_clip_config, engine_config
from .ingest import ingest_sources, SourceSpec
from .index import (asr_pool_cache_audit, asr_pool_current,
                    asr_semantic_fingerprint, index_all)
from .segment import segment_script, enrich_with_llm
from .match import match_segments
from .cut import cut_all
from . import ledger
from .build import build_video


class PipelineError(RuntimeError):
    """A user-actionable pipeline failure (no sources / no script / nothing discovered).

    Raised instead of SystemExit: SystemExit derives from BaseException, so the web portal's
    worker thread (`except Exception`) never saw it and the job spun forever. The CLI converts
    it back to a non-zero exit at its boundary."""


# A stage that "must never break a render" is wrapped in a fail-open catch. That is right for
# ENVIRONMENTAL faults (network, provider, disk) — but it also swallowed a plain NameError in the
# bounded-recovery stage for MONTHS: `recovery: skipped (NameError: name 'os' is not defined)`
# reads exactly like a benign skip, so the whole R4-5 subsystem was dead on every render and no
# one could tell from the log. Programming errors are not environmental: they can never be
# resolved by retrying, and they mean the stage did NOT run. Say so loudly and record it, while
# keeping the fail-open behaviour (a render must still finish).
_BUG_EXC = (NameError, UnboundLocalError, AttributeError, TypeError, ImportError)


def _log_stage_skip(log, proj, stage: str, exc: Exception) -> None:
    """Log a fail-open stage skip; a programming error gets a loud, greppable BUG line and is
    recorded in proj.meta['stage_bugs'] so audits and the portal can surface it."""
    _n = type(exc).__name__
    if isinstance(exc, _BUG_EXC):
        log(f"⚠ BUG — {stage} did NOT run: {_n}: {exc}. This is a CODE fault, not an "
            f"environmental one; retrying cannot fix it and the stage's protection is absent "
            f"from this render.")
        try:
            proj.meta.setdefault("stage_bugs", []).append({"stage": stage, "error": f"{_n}: {exc}"[:200]})
        except Exception:                                # noqa: BLE001 — never break a render
            pass
    else:
        log(f"{stage}: skipped ({_n}: {str(exc)[:80]})")


def _scaled_source_budget(max_sources: int, n_beats: int, video_type: str = "") -> int:
    """Footage budget scaled to the SCRIPT length.

    A long multi-scene essay references dozens of DISTINCT scenes; with a small budget only a few
    generic clips download and most beats fall back to re-using the same handful of footage (observed:
    a 181-beat Daenerys essay aired 14 generic compilations → 37% of beats verifier-flagged as
    wrong-scene). Scale to ~1 source per 5 beats (capped), but NEVER below the user's explicit
    request.

    LONG-FORM FLOOR: a long video (≈10+ min, 100+ beats) needs real footage BREADTH even when it is a
    single-scene deep-dive — otherwise the matcher re-airs one clip dozens of times (observed: a
    211-beat single-scene essay used its top source 57× → 99% of beats were the same scene). So long
    videos get a ≥20-source floor (scaling gently with length, capped), subject to what discovery can
    actually find. A SHORT single-scene clip still just wants its own raw footage (no floor).

    BUDGET RAISE (2026-07-26). The old floor gave a 272-beat single-scene essay only 31 sources, and
    the frame-by-frame audit of that render showed exactly the failure this docstring predicts: 78% of
    beats were tagged exact_scene but only 5% actually showed the described moment, and 46% of
    delivered scenes were near-duplicates of another scene. Both are pool-starvation symptoms — the
    matcher cannot pick a distinct exact shot that was never downloaded. Budgets are now ~1 source per
    4 beats with materially higher caps, and VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_MULT scales the whole
    thing (e.g. 1.5 for a "max footage" run) for operators who will trade render time for relevance."""
    import os as _os_sb
    base = max(4, int(max_sources or 4))
    n = int(n_beats or 0)
    # long-form floor: steeper slope (per 8 beats, was per 15) and a 56 cap (was 32)
    longform_floor = min(56, 24 + (n - 100) // 8) if n >= 100 else 0
    if (video_type or "").strip().lower() == "single_scene":
        # short deep-dive → raw scene only; LONG deep-dive → still needs angle/B-roll variety
        out = max(base, longform_floor)
    else:
        scaled = min(72, (n + 3) // 4)          # ~1 per 4 beats (was 1 per 5, cap 48)
        out = max(base, scaled, longform_floor)
    try:
        _mult = float(_os_sb.environ.get("VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_MULT", "1") or 1)
    except (TypeError, ValueError):
        _mult = 1.0
    if _mult and _mult != 1.0:
        out = int(round(out * max(0.1, _mult)))
    # never below the operator's explicit request; hard ceiling keeps a runaway script from
    # queueing hundreds of downloads.
    return max(base, min(out, int(_os_sb.environ.get("VIDLORE_CLIPSTUDIO_SOURCE_BUDGET_CAP", "96") or 96)))


# ---------------------------------------------------------------------------
# STAGE CHECKPOINTS (resume). The whole point: a render that dies (or content-blocks) at stage 9
# after 2+ hours must NOT redo analyze→discover→download→index→match→verify→recover from scratch.
# All of that state is already persisted in project.json (selections carry beat_windows / verifier
# verdicts / image_path; sources carry local_path; meta carries analysis+candidates), so a resume
# only needs to (a) know which stages already finished and (b) confirm their inputs are unchanged.
# Each stage records a *signature of its inputs*; a resume skips a stage iff that signature still
# matches AND the stage's persisted artifact is present. Because every stage folds the upstream
# signature into its own, changing anything (a new script, a bigger budget) re-runs that stage and
# everything after it — never a stale mix.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
from datetime import datetime as _dt, timezone as _tz

_PIPELINE_CKPT_VERSION = 2   # bump when a stage's semantics change so old checkpoints are ignored

# Narrow semantic versions for stages whose code-level content gates are not otherwise represented
# by their data/config inputs.  Keep these separate from _PIPELINE_CKPT_VERSION: changing a source
# admission rule must replay backfill + match, but it must not throw away valid downloads/indexes.
_MATCH_GATE_VERSION = "gatev5-typed-quotes-entity-safe-era"
_VERIFY_GATE_VERSION = "verifyv2-exact-reaction-context"
_BACKFILL_SIGNATURE_VERSION = "backfillv6-native-hd-actual-bytes"


def _sig(*parts) -> str:
    """Stable short signature of a stage's inputs (order-sensitive)."""
    h = _hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


def _seg_sig(segs) -> str:
    """Signature of the beats that actually drive footage selection — text + the classified visual
    policy + the entity/scene/quote anchors. A change here must re-match."""
    return _sig(tuple((
        s.index, s.text, s.visual_policy, s.required_entity, s.required_kind,
        s.scene_query, s.quote, getattr(s, "expected_visual", ""),
        bool(getattr(s, "is_specific_claim", False)),
        bool(getattr(s, "breakout_candidate", False)),
    ) for s in (segs or [])))


def _ckpt(proj) -> dict:
    pl = proj.meta.get("pipeline")
    if not isinstance(pl, dict) or pl.get("version") != _PIPELINE_CKPT_VERSION:
        pl = {"version": _PIPELINE_CKPT_VERSION, "stages": {}}
        proj.meta["pipeline"] = pl
    return pl


def _stage_done(proj, name: str, sig: str) -> None:
    """Mark a stage complete with the signature of its inputs, and flush the manifest so a crash in
    the NEXT stage still finds this one done. This also closes the historical gap where match wrote
    no save (a crash after match, before cut, lost every selection)."""
    _ckpt(proj)["stages"][name] = {
        "sig": sig, "at": _dt.now(_tz.utc).isoformat(timespec="seconds"), "status": "done"}
    proj.save()


def _stage_skip(proj, name: str, sig: str, resume: bool, artifact_ok: bool = True) -> bool:
    """A stage is skippable iff we are resuming, its recorded input-signature still matches, and its
    persisted output is actually present on disk / in the manifest."""
    if not resume:
        return False
    rec = _ckpt(proj)["stages"].get(name)
    return bool(rec and rec.get("status") == "done" and rec.get("sig") == sig and artifact_ok)


def _rebind_completed_footage_stages(proj, signatures: dict[str, str], *, reason: str) -> bool:
    """Bind completed match/cut/verify/recover artifacts to a conclusive scoped-recovery pool.

    Semantic recovery is intentionally transactional: it may add/index sources, rematch only the
    blocked page, re-cut every accepted changed selection, and restore every out-of-scope selection.
    Once that page completes, the persisted selection/clip/verifier state is the authoritative
    reconciled result for the enlarged pool.  Leaving the older pool signatures on those artifacts
    makes Resume treat the recovery source as an external mutation, rebuild Face-ID, globally rerun
    match/cut/verify, and destroy the audited deferred-page cursor.

    Rebinding is fail-closed. Every named stage must already be completed; callers must separately
    prove the enlarged ASR/index pool is current. Preserve ``at`` because it records when the global
    stage actually ran (offline reproduction filters sources by that timestamp); ``rebound_at`` says
    when a later scoped transaction made its artifacts current for a newer pool.
    """
    stages = _ckpt(proj)["stages"]
    wanted = {str(name): str(sig) for name, sig in (signatures or {}).items() if name and sig}
    if not wanted or any(
            not isinstance(stages.get(name), dict)
            or stages[name].get("status") != "done"
            for name in wanted):
        return False
    rebound_at = _dt.now(_tz.utc).isoformat(timespec="seconds")
    for name, sig in wanted.items():
        rec = dict(stages[name])
        rec["sig"] = sig
        rec["rebound_at"] = rebound_at
        rec["rebound_reason"] = str(reason or "scoped_recovery")
        stages[name] = rec
    proj.save()
    return True


def _footage_stage_signatures(download_sig: str, sources, *, force_index: bool,
                              segments, verify: bool,
                              asr_signature: str) -> tuple[str, str, str, str, str]:
    """Sign the pool actually consumed by match and every dependent footage stage.

    Pre-match anchor/backfill passes may add sources after the download checkpoint.  Computing these
    signatures only before those passes records a pre-mutation signature beside post-mutation
    selections, so every Resume needlessly rematches and re-runs backfill.  Source ids are canonical
    because duplicate records with the same id resolve to one searchable index namespace.
    """
    # A source id names an index namespace, but it does not prove the bytes under that id stayed
    # unchanged.  HD recovery can replace a file in place; signing ids alone then let Resume skip
    # index+match+cut and pair stale SD clips with newly-HD source metadata. Keep the first record
    # for each canonical id (matching ``proj.source`` semantics) and bind every downstream stage to
    # path/stat/checksum identity. Duplicate manifest rows remain inert.
    canonical_sources = {}
    for source in (sources or []):
        sid = str(getattr(source, "id", "") or "")
        if sid:
            canonical_sources.setdefault(sid, source)

    def _source_byte_identity(sid, source):
        path = str(getattr(source, "local_path", "") or "")
        try:
            stat = Path(path).stat() if path else None
            size = int(stat.st_size) if stat is not None else 0
            mtime_ns = int(stat.st_mtime_ns) if stat is not None else 0
        except OSError:
            size = mtime_ns = 0
        return (sid, path, size, mtime_ns,
                str(getattr(source, "checksum", "") or ""))

    source_identity = tuple(
        _source_byte_identity(sid, canonical_sources[sid])
        for sid in sorted(canonical_sources))
    # Match consumes persisted shot transcripts, so the ASR decoder/model/prompt identity is an
    # index input even when the downloaded files themselves have not changed.  Without this edge,
    # a fully checkpointed Resume can skip index + match and silently reuse words decoded under an
    # older vocabulary (or even another Whisper build).
    sig_index = _sig(download_sig, source_identity, bool(force_index), str(asr_signature or ""))
    sig_match = _sig(sig_index, _seg_sig(segments), _MATCH_GATE_VERSION)
    sig_cut = _sig(sig_match)
    sig_verify = _sig(sig_cut, bool(verify), _VERIFY_GATE_VERSION)
    return sig_index, sig_match, sig_cut, sig_verify, _sig(sig_verify)


def _semantic_fingerprint(value) -> str:
    """Hash dataclass/dict-like semantic inputs without object ids or map-order noise."""
    import dataclasses as _dc_sig
    import json as _json_sig

    def _plain(obj):
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if _dc_sig.is_dataclass(obj) and not isinstance(obj, type):
            return _plain(_dc_sig.asdict(obj))
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return _plain(obj.to_dict())
        if isinstance(obj, dict):
            return {str(k): _plain(v) for k, v in sorted(
                obj.items(), key=lambda item: str(item[0]))}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            rows = [_plain(v) for v in obj]
            return sorted(rows, key=lambda row: _json_sig.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        if hasattr(obj, "__dict__"):
            return _plain({k: v for k, v in vars(obj).items()
                           if not str(k).startswith("_") and not callable(v)})
        # Unknown runtime handles (models, locks, clients) are deliberately represented only by
        # type.  Backfill's persisted inputs are plain config/analysis values in production; this
        # fallback keeps a diagnostic object from leaking a process-specific memory address.
        return {"type": f"{type(obj).__module__}.{type(obj).__qualname__}"}

    raw = _json_sig.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


_BACKFILL_SEMANTIC_ENV_KEYS = (
    # Source-pool admission/rejection switches read directly by match._load_pool.
    "VIDLORE_CLIPSTUDIO_BANNED_SOURCES",
    "VIDLORE_CLIPSTUDIO_OCR_GATE",
    "VIDLORE_CLIPSTUDIO_WATERMARK_MODE",
    "VIDLORE_CLIPSTUDIO_NONSHOW_GATE",
    "VIDLORE_CLIPSTUDIO_WRONGSHOW_GATE",
    "VIDLORE_CLIPSTUDIO_FACE_FOOTAGE_GATE",
    "VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE",
    "VIDLORE_CLIPSTUDIO_SUBBED_SOURCE_MAX_FRAC",
    "VIDLORE_CLIPSTUDIO_PROMO_OVERLAY_GATE",
    "VIDLORE_CLIPSTUDIO_GRAPHICS_GATE",
    "VIDLORE_CLIPSTUDIO_NUMERAL_GATE",
    "VIDLORE_CLIPSTUDIO_NUMERAL_SRC_FRAC",
    "VIDLORE_CLIPSTUDIO_STATIC_GATE",
    "VIDLORE_CLIPSTUDIO_CURSOR_SRC_GATE",
    # Shot-yield gates used to decide whether a fetched replacement can actually air.
    "VIDLORE_CLIPSTUDIO_TEXT_GATE",
    "VIDLORE_CLIPSTUDIO_SUBBAND_GATE",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_GATE",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_AVG",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_HI",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_HI_HARD",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_MIN",
    "VIDLORE_CLIPSTUDIO_UNREADABLE_BLACKFRAC",
    # Discovery/index semantics that are not represented by ClipConfig fields.
    "VIDLORE_CLIPSTUDIO_ANGLE_VARIANTS",
    "VIDLORE_CLIPSTUDIO_HD_PROBE",
    "VIDLORE_CLIPSTUDIO_KEYSCENE_COVERAGE",
    "VIDLORE_CLIPSTUDIO_QUERY_CAP",
    "VIDLORE_CLIPSTUDIO_QUERY_CAP_MAX",
    "VIDLORE_CLIPSTUDIO_SUB_VERIFY",
    "VIDLORE_CLIPSTUDIO_FLAGS_FAST",
    "VIDLORE_CLIPSTUDIO_GRAPHIC_MAX",
    "VIDLORE_CLIPSTUDIO_GRAPHIC_SOFT",
    "VIDLORE_CLIPSTUDIO_KF_PREEXTRACT",
    "VIDLORE_CLIPSTUDIO_MULTIFRAME_FLAGS",
    "VIDLORE_CLIPSTUDIO_STATIC_PAIR_DIFF",
    # Download-quality behavior that can turn the same discovered upload into different media.
    "VIDLORE_HD_DOWNLOAD",
    "VIDLORE_HD_403_SWEEP",
    "VIDLORE_HD_403_SWEEP2",
    "VIDLORE_HD_COOKIES_BROWSER",
    "VIDLORE_HD_COOKIES_FILE",
    "VIDLORE_HD_REMOTE_COMPONENTS",
)


def _backfill_semantic_environment() -> tuple[tuple[str, str], ...]:
    """Raw direct env inputs, excluding performance-only worker/timing knobs.

    An unset value is explicit in the tuple.  Therefore unset->override and override->unset both
    invalidate the checkpoint, while changing MAX_CPU/worker counts does not replay a completed
    search.  Code-default changes are covered by the version literal in the parent signature.
    """
    import os as _os_bf_sig
    return tuple((key, _os_bf_sig.environ.get(key, "<unset>"))
                 for key in _BACKFILL_SEMANTIC_ENV_KEYS)


_BACKFILL_SEMANTIC_CFG_FIELDS = (
    # Source admission/index semantics used by this pass.  Keep this an explicit allow-list:
    # ClipConfig also contains voice/render pacing plus worker/retry knobs, and hashing the whole
    # dataclass made a MAX_CPU-only speed change replay a completed web search.
    "watermark_mode",
    # These four change persisted shot boundaries/ASR and therefore the searchable index built for
    # every admitted replacement.  They are content semantics, not worker or render timing knobs.
    "target_clip_sec",
    "min_clip_sec",
    "whisper_model",
    "whisper_compute",
    "scene_threshold",
    "min_shot_sec",
    "detect_faces",
    "detect_ocr",
    "dup_hamming",
    # Discovery/download content decisions.  Retry counts and concurrency are deliberately absent;
    # a technical attempt never receives a completed checkpoint, so they affect speed/reliability,
    # not the meaning of a conclusive result.
    "max_height",
    "discover_target",
    "discover_per_query",
    "discover_min_sec",
    "discover_max_sec",
    "discover_min_height",
    "discover_prefer_height",
    "discover_max_per_channel",
    "discover_coverage_extra",
    "discover_resolve_quality",
    "discover_resolve_limit",
)


def _backfill_semantic_config(cfg) -> tuple[tuple[str, object], ...]:
    """Only ClipConfig values that can change the sources/shots a completed pass admits."""
    return tuple((field, getattr(cfg, field, None))
                 for field in _BACKFILL_SEMANTIC_CFG_FIELDS)


def _backfill_input_signature(download_sig: str, segments, *, policy: str, max_sources: int,
                              show_title: str, enabled: bool, rounds: int, cfg,
                              analysis) -> str:
    """Stable signature of every semantic input to the clean-copy backfill pass.

    The source pool is intentionally absent: targeted recovery may add sources after match, and
    that must not replay the same global clean-copy search forever.  Script analysis and ClipConfig
    are included because they control discovery, download, indexing, and pool-gate decisions; a
    completed audit made under different semantics is not reusable.
    """
    return _sig(
        str(download_sig or ""), _seg_sig(segments), str(policy or ""), int(max_sources),
        str(show_title or ""), bool(enabled), max(1, int(rounds)),
        _semantic_fingerprint(_backfill_semantic_config(cfg)),
        _semantic_fingerprint(analysis),
        _semantic_fingerprint(_backfill_semantic_environment()),
        _BACKFILL_SIGNATURE_VERSION,
    )


def _footage_stages_required(*, skip_match: bool, skip_verify: bool, skip_recover: bool,
                             backfill_enabled: bool, skip_backfill: bool) -> bool:
    """Whether footage-stage context is needed, including a retryable backfill attempt."""
    downstream_cached = bool(skip_match and skip_verify and skip_recover)
    backfill_pending = bool(backfill_enabled and not skip_backfill)
    return bool(not downstream_cached or backfill_pending)


def _backfill_audit_complete_for(proj, input_sig: str) -> bool:
    """True only for a conclusive backfill invocation bound to these exact inputs."""
    audit = (getattr(proj, "meta", {}) or {}).get("backfill_audit") or {}
    return bool(
        audit.get("schema_version") == 2
        and audit.get("status") == "complete"
        and str(audit.get("input_sig", "") or "") == str(input_sig or "")
    )


def _run_backfill_invocation(proj, input_sig: str, runner, *, log) -> bool:
    """Run one completion-bound backfill attempt without blessing stale audit state.

    The improvement pass remains fail-open for the current render, but its checkpoint is
    fail-closed: an unexpected exception can never reuse a ``complete`` audit left by an older
    signature.  ``runner`` owns the detailed audit on normal return.
    """
    import copy as _copy_bf_outer

    # Known transport/index exits are rolled back by the pass itself.  This outer snapshot closes
    # the remaining programming-error window: a crash after download/index but before the title or
    # shot-yield screen must not leave an unscreened source available to the same render's matcher.
    meta_before = _copy_bf_outer.deepcopy(getattr(proj, "meta", {}) or {})
    source_before = [
        (source, _copy_bf_outer.deepcopy(vars(source)))
        for source in (getattr(proj, "sources", None) or [])
    ]
    source_ids_before = {
        str(getattr(source, "id", "") or "") for source, _state in source_before
    }

    running = {"schema_version": 2, "status": "running", "reason": "",
               "input_sig": str(input_sig or ""), "rounds": [], "admitted": 0}
    proj.meta["backfill_audit"] = running
    proj.save()
    try:
        runner()
    except Exception as exc:                            # noqa: BLE001 — improvement stays fail-open
        audit = (proj.meta or {}).get("backfill_audit") or {}
        if str(audit.get("input_sig", "") or "") != str(input_sig or ""):
            audit = dict(running)
        else:
            audit = dict(audit)
        current_sources = list(getattr(proj, "sources", None) or [])
        new_ids = {
            str(getattr(source, "id", "") or "") for source in current_sources
            if str(getattr(source, "id", "") or "") not in source_ids_before
        }
        try:
            from .index import purge_source_index as _purge_backfill_outer
            for sid in sorted(new_ids):
                if sid:
                    _purge_backfill_outer(proj, sid)
        except Exception:                              # noqa: BLE001 — manifest restore is primary
            pass
        for source, state in source_before:
            vars(source).clear()
            vars(source).update(_copy_bf_outer.deepcopy(state))
        proj.sources = [source for source, _state in source_before]
        proj.meta = _copy_bf_outer.deepcopy(meta_before)
        audit.update({
            "schema_version": 2,
            "status": "incomplete",
            "reason": f"unexpected_failure:{type(exc).__name__}",
            "input_sig": str(input_sig or ""),
        })
        proj.meta["backfill_audit"] = audit
        proj.save()
        log(f"5b/9 · backfill: skipped ({str(exc)[:80]})")
        return False
    return _backfill_audit_complete_for(proj, input_sig)


def _run_preassemble_selfheal(proj, segs, cfg, analysis, *, policy: str,
                              pre: str, faceid_obj=None, refs=None, roster=None,
                              log) -> str | None:
    """Run structural-gate self-heal without laundering technical acquisition into content.

    Most legacy self-heal faults remain fail-open so the authoritative gate can speak.  A typed
    acquisition failure is different: the recovery pool was never exhaustively searched, so the
    old catch-and-continue path converted infrastructure uncertainty into a non-retryable footage
    verdict.  Remove the just-written recovery checkpoint and propagate the typed error; Resume
    must retry recovery after network/download/index health is restored.
    """
    if not pre:
        return pre
    from . import selfheal as _selfheal
    try:
        return _selfheal.run(
            proj, segs, cfg, analysis, policy=policy, faceid_obj=faceid_obj,
            refs=refs, roster=roster, log=log)
    except _selfheal.InconclusiveAcquisitionError:
        _ckpt(proj)["stages"].pop("recover", None)
        proj.save()
        raise
    except Exception as exc:                            # noqa: BLE001 — legacy fail-open boundary
        _log_stage_skip(log, proj, "self-heal", exc)
        return pre


def produce(project_dir, *, script_path: Optional[str] = None, script_text: Optional[str] = None,
            source_specs: Optional[list[SourceSpec]] = None, name: str = "",
            cfg: Optional[ClipConfig] = None, theme: str = "history", voice: str = "",
            title: str = "", captions: Optional[bool] = None, caption_style: str = "",
            use_tts: bool = True, enrich: bool = True, voiceover: Optional[str] = None,
            do_build: bool = True, force_index: bool = False, progress=None) -> dict:
    cfg = cfg or load_clip_config()
    eng = engine_config()
    project_dir = Path(project_dir)

    def log(m):
        if progress:
            progress(m)

    if (project_dir / "project.json").exists():
        proj = ClipProject.load(project_dir)
        if name:
            proj.name = name
    else:
        proj = ClipProject(name=name or project_dir.name, root=str(project_dir))
    proj.ensure_dirs()

    # 1 — ingest (permission-gated)
    if source_specs:
        log("stage 1/6 · ingest")
        ingest_sources(proj, source_specs, cfg, progress=progress)
    blocked = [s.id for s in proj.sources if s.is_blocked]
    if blocked:
        log(f"  ⚠ blocked (permission unverified): {blocked}")
    # a source must be downloaded ('ok') AND permitted — a permitted source whose download
    # failed would pass an is_blocked-only gate and the pipeline would render black placeholders
    usable = [s for s in proj.sources if not s.is_blocked and s.status == SOURCE_OK]
    if not usable:
        raise PipelineError("no usable sources — every source is blocked or failed. "
                            "Set a permission (owner|licensed|public_domain|cc|fair_use_claim).")

    # 2 — index
    log("stage 2/6 · index")
    index_all(proj, cfg, force=force_index, progress=progress)

    # 3 — segment
    log("stage 3/6 · segment")
    if script_text is None and script_path:
        script_text = Path(script_path).read_text(encoding="utf-8")
        proj.script_path = str(script_path)
    if not (script_text or "").strip():
        raise PipelineError("no narration script provided (--script).")
    segs = segment_script(script_text, cfg)
    from . import llm as _llm
    if enrich and _llm.has_llm(eng):     # llm.has_llm counts Gemini too (engine has_llm is
        enrich_with_llm(segs, eng, progress=progress)     # Anthropic-only — Gemini-only setups skipped enrichment)
    from . import policy as _policy
    _policy.finalize_beats(segs)         # unified per-beat visual policy (all stages obey it)
    proj.segments = segs
    proj.save()

    # 4 — match
    log("stage 4/6 · match")
    match_segments(proj, segs, cfg, progress=progress)

    # 5 — cut
    log("stage 5/6 · cut")
    cut_all(proj, cfg, progress=progress)

    # 6 — compliance ledger + human review surface
    summ = ledger.finalize(proj, segs, cfg)
    from . import review as _review
    review_path = _review.write_review(proj, segs)
    proj.save()
    log(f"  ledger · {summ['flagged_for_review']}/{summ['segments']} flagged for review · "
        f"mean confidence {summ['mean_confidence']}")
    log(f"  review · {review_path}")

    out = None
    if do_build:
        log("stage 6/6 · build")
        out = build_video(proj, segs, cfg, voice=voice, captions=captions,
                          caption_style=caption_style,
                          title=title or proj.name, theme_name=theme, use_tts=use_tts,
                          voiceover=voiceover, progress=progress)

    return {
        "project": str(proj.root),
        "summary": summ,
        "output": (str(out) if out else None),
        "manifest": str(proj.manifest_path),
        "ledger": str(proj.ledger_path),
        "review_queue": str(proj.review_queue_path),
        "review_html": str(review_path),
    }


def _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log, *, eng_cfg=None,
                          fail_on_web_technical: bool = False) -> int:
    """For every WEAK beat (no source / low confidence / confirmed wrong character / a repeated
    filler shot) attach a relevant still as a Ken-Burns image_path. PREFERENCE ORDER:
      1) a REAL source-footage frame (a shot keyframe) drawn from the matcher's own ranked
         alternates — 100% real, era-correct, no AI/watermark risk, and a DISTINCT composition
         that breaks up footage repetition on filler beats;
      2) only if the footage pool truly has nothing relevant (and web fallback is enabled) a
         web-searched, hard-validated still.
    Exact-scene priority is preserved: only WEAK beats are touched, and the source-still is chosen
    from the matcher's relevance-ranked alternates, so a named character/scene still resolves to
    that character/scene's footage."""
    import os
    from . import image_fallback as _imgfb
    from . import index as _index
    from . import policy as _policy
    from .relevance_contract import (
        verified_still_coverage as _verified_still_coverage,
        strict_still_evidence_reason as _strict_still_evidence_reason,
        image_sha256 as _still_image_sha256,
    )
    from .models import FLAG_EXACT_MISSING, SOURCE_OK, ClipSelection
    sel_by_idx = {s.segment_index: s for s in proj.selections}
    char2actor = analysis.char_to_actor() if analysis is not None else {}
    low = float(os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_CONF_FLOOR", "0.55"))
    cap = int(os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_MAX_BEATS", "24") or 24)
    # web image fallback is OFF unless explicitly enabled — and even then ONLY a strictly-validated,
    # real LIVE-ACTION exact-scene still for exact/character beats (never AI, never generic filler).
    web_fallback_on = os.environ.get("VIDLORE_CLIPSTUDIO_WEB_IMAGE_FALLBACK", "1").strip() \
        not in ("0", "false", "no")
    img_dir = proj.root_path / "scene_images"

    # SEMANTIC recovery-still verification (Gap 2): a recovery still is a real source keyframe, but it
    # must still be SEMANTICALLY correct — never a wrong character / wrong show / wrong era / an action
    # contradicting the narration. Candidate ranking may be contextual, but publication evidence is
    # STRICT on the actual keyframe: matches+specific+quality must be positive, required subject must
    # be present, and wrong/contradiction must be negative. Honest labeling is not proof.
    # Verification runs only when an LLM is available; without it the publication gate blocks.
    from . import verify as _verify_mod
    from . import llm as _llm_mod
    _still_verify_on = (eng_cfg is not None and _llm_mod.has_llm(eng_cfg)
                        and os.environ.get("VIDLORE_CLIPSTUDIO_VERIFY_STILLS", "1").strip()
                        not in ("0", "false", "no"))
    _vtype_sv = (proj.meta.get("analysis", {}) or {}).get("video_type", "")
    _global_era_sv = str((proj.meta.get("analysis", {}) or {}).get("episode_hint", "") or "")
    from . import era as _era_sv
    _anchor_eras_sv = _era_sv.anchor_token_eras(
        type("A", (), {"anchor_scenes": (proj.meta.get("analysis", {}) or {}).get("anchor_scenes"),
                       "movie_title": (proj.meta.get("analysis", {}) or {}).get("movie_title", ""),
                       "characters": (proj.meta.get("analysis", {}) or {}).get("characters"),
                       "actors": (proj.meta.get("analysis", {}) or {}).get("actors")})())

    from . import index as _index_sv

    # PER-RENDER MEMOS (perf only — same data, loaded once instead of per candidate):
    #   _shots_cache    source_id -> list[Shot]  (shots.json parsed once per source)
    #   _embeds_cache   source_id -> persisted embeds matrix (or None when absent)
    #   _rel_memo       (source_id, shot_index, text) -> CLIP relevance score
    _shots_cache: dict = {}
    _embeds_cache: dict = {}
    _rel_memo: dict = {}

    def _shots_of(sid):
        if sid not in _shots_cache:
            try:
                _shots_cache[sid] = _index_sv.load_shots(proj, sid)
            except Exception:
                _shots_cache[sid] = []
        return _shots_cache[sid]

    def _embeds_of(sid):
        # (matrix, manifest row_map), certified for the current schema/model/dim/rows —
        # (None, None) for every legacy/mismatched index, which _shot_relevance treats as
        # "use the live embedding path" (see index.load_embeds_verified)
        if sid not in _embeds_cache:
            try:
                _embeds_cache[sid] = _index_sv.load_embeds_verified(proj, sid)
            except Exception:
                _embeds_cache[sid] = (None, None)
        return _embeds_cache[sid]

    def _shot_face_ids(sid, sidx):
        """The candidate SHOT's actual persisted face detections (NOT the beat's requested entities —
        those are script asks, not faces found in this frame). Empty list when unavailable."""
        try:
            for _sh in _shots_of(sid):
                if getattr(_sh, "index", None) == sidx:
                    return list(getattr(_sh, "face_ids", None) or [])
        except Exception:
            pass
        return []

    def _still_verdict(kf_path, seg, sid, sidx):
        """STRICT actual-image verdict: 'ok' | 'reject' | 'unverified' | 'disabled'.
        Passes the SHOT's real face_ids. A transport error / timeout / malformed response (while the
        verifier IS configured) → 'unverified' (NOT silently 'ok'). A label, CLIP score, source title,
        or deterministic Face-ID check can never manufacture this positive verdict. Retries once on
        a transient error."""
        if not _still_verify_on:
            return "disabled"
        if not kf_path:
            return "unverified"
        return _still_verdict_call(kf_path, seg, sid, sidx)

    import re
    _movie_toks = set(re.findall(r"[a-z0-9]+", str(
        (proj.meta.get("analysis", {}) or {}).get("movie_title", "") or "").lower()))

    def _still_deterministic_ok(sid, sidx, score, seg) -> bool:
        """When NO vision model is available, an exact/character recovery still may be installed ONLY
        if it passes the module-level DETERMINISTIC gate (same-show via meaningful title identity +
        explicit beat-local era vs the source's declared season + CLIP relevance + Face-ID for a
        named character). No arbitrary 'unverified_fallback' that could be a contradictory frame."""
        try:
            src = proj.source(sid)
            title = getattr(src, "title", "") or ""
            _kind = (getattr(seg, "required_kind", "") or "").lower()
            faces = _shot_face_ids(sid, sidx) if _kind in ("character", "actor") else []
            ok, reason = _deterministic_still_ok(
                source_title=title, score=score, seg=seg, faces=faces, movie_toks=_movie_toks,
                global_era=_global_era_sv, single_scene=(_vtype_sv == "single_scene"),
                min_clip=float(os.environ.get("VIDLORE_CLIPSTUDIO_DET_STILL_MIN_CLIP", "0.30") or 0.30),
                char2actor=char2actor, anchor_eras=_anchor_eras_sv)
            if not ok:
                log(f"image-fallback: beat {seg.index} — deterministic still rejected ({reason})")
            return ok
        except Exception:
            return False

    # STILL-VERDICT CACHE — the same verdict_cache.json and the same full-fingerprint doctrine as
    # verify.py's rungs: a still verdict is reusable ONLY when the complete question (source content
    # hash, shot bounds, judged frame identity, every prompt field, the strictness/question
    # variant, real vision model, prompt/sheet versions) is byte-identical. Reuse is NEVER keyed on
    # the image path alone — the same frame judged for a different beat is a different question.
    # Only successful schema-valid verdicts are stored ('unverified' transport outcomes never are),
    # so the 2-attempt retry and the vision-outage backoff behavior are unchanged.
    _still_vcache = _verify_mod._load_verdict_cache(proj)
    _still_evidence: dict[tuple[str, int], dict] = {}
    try:
        _vmodel_sv = _llm_mod.vision_config(eng_cfg)
    except Exception:
        _vmodel_sv = str(getattr(eng_cfg, "anthropic_model", "") or "")
    _srch_cache: dict = {}

    def _src_hash_sv(sid):
        if sid not in _srch_cache:
            _src_o = proj.source(sid)
            _srch_cache[sid] = _verify_mod._file_fingerprint(
                getattr(_src_o, "local_path", "") or "")
        return _srch_cache[sid]

    def _shot_obj_sv(sid, sidx):
        for _sh in _shots_of(sid):
            if getattr(_sh, "index", None) == sidx:
                return _sh
        return None

    def _still_fp(kf_path, seg, sid, sidx, faces, model_id: str = ""):
        try:
            _sh = _shot_obj_sv(sid, sidx)
            if _sh is None:
                return ""
            return _verify_mod.verdict_fingerprint(
                src_hash=_src_hash_sv(sid), source_id=sid or "",
                shot_start=getattr(_sh, "start", 0.0), shot_end=getattr(_sh, "end", 0.0),
                beat_text=getattr(seg, "text", ""),
                required_entity=getattr(seg, "required_entity", ""),
                required_kind=getattr(seg, "required_kind", ""),
                expected_visual=getattr(seg, "expected_visual", "") or "",
                scene_query=getattr(seg, "scene_query", "") or "",
                era=_verify_mod._beat_era(seg, _global_era_sv, _vtype_sv == "single_scene",
                                          anchor_eras=_anchor_eras_sv),
                visual_policy=_policy.policy_of(seg), is_specific=True,
                faceid_names=list(faces or []), multiframe=False,
                image_id=f"kf:{_verify_mod._file_fingerprint(kf_path)}",
                model=(model_id or _vmodel_sv), venue_fallback=False,
                must_see=_verify_mod.effective_deictic_target(seg))
        except Exception:
            return ""                                    # no key → baseline uncached call

    def _still_verdict_call(kf_path, seg, sid, sidx):
        from . import perf_metrics as _pm_sv
        faces = _shot_face_ids(sid, sidx)
        _fp = _still_fp(kf_path, seg, sid, sidx, faces)
        _hit = _still_vcache.get(_fp) if _fp else None
        _required_sv = getattr(seg, "required_entity", "") or ""
        _must_see_sv = _verify_mod.effective_deictic_target(seg)
        if _hit is not None and _verify_mod._verdict_schema_ok(
                _hit, required_entity=_required_sv, must_see=_must_see_sv) \
                and _verify_mod._hit_provider_ok(_hit, _vmodel_sv):
            _pm_sv.incr("still.verdict.cache_hit")
            _ev = {**dict(_hit), "status": "ok"}
            _still_evidence[(str(kf_path), int(seg.index))] = _ev
            return "reject" if _strict_still_evidence_reason(_ev, seg) else "ok"
        for _attempt in (1, 2):
            try:
                _pm_sv.incr("still.verdict.call")
                v = _verify_mod.verify_frame(
                    str(kf_path), seg.text, getattr(seg, "required_entity", ""),
                    getattr(seg, "required_kind", ""), faces, eng_cfg,
                    getattr(eng_cfg, "anthropic_model", ""), is_specific=True,
                    expected_visual=getattr(seg, "expected_visual", "") or "",
                    scene_query=getattr(seg, "scene_query", "") or "",
                    era_hint=_verify_mod._beat_era(seg, _global_era_sv, _vtype_sv == "single_scene",
                                                   anchor_eras=_anchor_eras_sv),
                    venue_fallback=False,
                    must_see=_verify_mod.effective_deictic_target(seg))
                if v is None:
                    continue                              # transport error → retry, then unverified
                v = {**v, "status": "ok"}
                _still_evidence[(str(kf_path), int(seg.index))] = dict(v)
                if _fp and _verify_mod._verdict_schema_ok(
                        {**v, "status": "ok"}, required_entity=_required_sv,
                        must_see=_must_see_sv):
                    _sb = str(v.get("vision_served_by") or "")
                    _fp_store = _fp if (not _sb or _sb == _vmodel_sv) else \
                        _still_fp(kf_path, seg, sid, sidx, faces, model_id=_sb)
                    if _fp_store:
                        _still_vcache[_fp_store] = dict(v)
                        _verify_mod._save_verdict_cache(proj, _still_vcache)   # atomic
                return "reject" if _strict_still_evidence_reason(v, seg) else "ok"
            except Exception:
                continue
        return "unverified"                               # no real verdict after retries

    # Index every pooled shot — REAL downloaded source-video keyframes are the PREFERRED (and primary)
    # image source. No NOW-photos / recent-actor photos / AI images anywhere in this path.
    from . import discover as _discover
    shots_by_key: dict = {}
    shots_lowres_by_key: dict = {}
    _skipped_src = 0
    _skipped_wm = 0
    _skipped_lowres = 0
    # STILL-POOL PURITY: clips from a watermarked source get a punch-in CROP at cut time, but a
    # STILL is the raw keyframe — the channel bug airs frozen for seconds (observed: a 'BLACK
    # TRVLLS' logo on Tywin stills). And a sub-SD keyframe blown up to the 1080p canvas is a soft
    # blur held on screen (observed: 6 aired 360p stills, all judged degraded). Both stay OUT of
    # the still pool; their MOVING clips remain available (cropped / detail-enhanced) so relevance
    # is untouched.
    _corner_on = os.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no")
    try:
        _still_min_h = int(os.environ.get("VIDLORE_CLIPSTUDIO_STILL_MIN_SRC_HEIGHT", "480") or 480)
    except (TypeError, ValueError):
        _still_min_h = 480
    from .match import _source_is_watermarked as _src_wm, _source_corner_logo as _src_logo
    from .match import banned_source_ids as _banned_ids
    # BANNED sources (fan-film / AI-recreation) are excluded from the STILL pool too — a ban that
    # only covered match would still let the same non-authentic upload air as a frozen Ken-Burns
    # still, which is worse (it sits on screen for seconds).
    _banned_sv = _banned_ids(proj)
    _skipped_banned = 0
    for s in proj.sources:
        if s.status != SOURCE_OK:
            continue
        if s.id in _banned_sv:
            _skipped_banned += 1
            continue
        # A reaction / review / interview / essay / non-show upload may have been downloaded+indexed
        # (e.g. as a dialogue-verify backup) yet gated out of CLIP selection. Its keyframes are
        # reactor facecams, channel logos, 'GAME OF THRONES' title cards and burned-name overlays —
        # never clean scene stills. Exclude such sources from the image-recovery pool so an
        # exact-scene-missing beat can never pull a reactor frame as a "source-frame" still.
        if _discover.is_unwanted_source_title(getattr(s, "title", "") or ""):
            _skipped_src += 1
            continue
        if s.id in _shots_cache:                       # perf memo only — flow identical to baseline
            _shots = _shots_cache[s.id]
        else:
            try:
                _shots = _index.load_shots(proj, s.id)
            except Exception:
                continue                               # exactly the baseline skip (no counters touched)
            _shots_cache[s.id] = _shots
        if _src_wm(_shots) or (_corner_on and _src_logo(_shots)):
            _skipped_wm += 1
            continue
        if _still_min_h and 0 < int(getattr(s, "height", 0) or 0) < _still_min_h:
            # kept OUT of the normal still pool, but retained as a LAST-RESORT pool for
            # exact/character beats the HD pool can't cover (observed: every face-confirmed
            # Tywin shot lived in 360p uploads, so the HD-only pool starved the whole fallback
            # chain and the render release-blocked). Watermarked / reaction-essay sources are
            # CONTENT exclusions and stay out of both pools.
            _skipped_lowres += 1
            for sh in _shots:
                shots_lowres_by_key[(sh.source_id, sh.index)] = sh
            continue
        for sh in _shots:
            shots_by_key[(sh.source_id, sh.index)] = sh
    if _skipped_banned and log:
        log(f"  [image-pool] excluded {_skipped_banned} BANNED source(s) "
            f"(fan-film / AI-recreation) from the still pool")
    if _skipped_src and log:
        log(f"  [image-pool] excluded {_skipped_src} reaction/essay/non-show source(s) "
            f"from the still-recovery pool")
    if (_skipped_wm or _skipped_lowres) and log:
        log(f"  [image-pool] excluded {_skipped_wm} watermarked + {_skipped_lowres} sub-{_still_min_h}p "
            f"source(s) from the still pool (their moving clips stay available)")

    def _phash_of(sel):
        sh = shots_by_key.get((getattr(sel, "source_id", ""), getattr(sel, "shot_index", -1)))
        return (getattr(sh, "phash", "") or "") if sh else ""

    def _verifier_kept(sel):
        v = (getattr(sel, "verifier", None) or {})
        return v.get("status") == "ok" and v.get("verdict") == "keep"

    still_cap = min(cap, max(1, int(len(segs) * float(
        os.environ.get("VIDLORE_CLIPSTUDIO_SOURCE_STILL_FRACTION", "0.30")))))
    sim_thresh = int(os.environ.get("VIDLORE_CLIPSTUDIO_REPEAT_HAMMING", "12"))
    src_filled = web_filled = 0
    used_keys: set = set()
    used_phash: set = set()
    aired_shots: set = set()
    aired_phash: set = set()
    # Strict beats whose source-frame candidates produced ONLY technical non-verdicts. Keep this
    # separate from explicit semantic rejects: an outage is retryable, while a real reject is a
    # conclusive content result that may legitimately proceed to the audited gap ladder.
    _strict_unverified_only: dict[int, dict] = {}

    # ---- PASS 1 — SOURCE-FRAME STILLS (preferred): real downloaded keyframes (Ken-Burns) for beats
    # that are abstract (prefer a still/effect), a FILLER repeat (variety), weak/unconfirmed, an
    # exact beat whose footage FAILED (recovery), or that have no clip at all (pool-pick). Confirmed-
    # good footage keeps moving. Capped so the video stays mostly motion.
    for seg in segs:
        sel = sel_by_idx.get(seg.index)
        pol = _policy.policy_of(seg)
        has_src = bool(sel and sel.source_id)
        ph = _phash_of(sel) if has_src else ""
        repeat = has_src and ((sel.source_id, sel.shot_index) in aired_shots
                              or (ph and _imgfb._min_hamming(ph, aired_phash) <= sim_thresh))
        weak = has_src and not _verifier_kept(sel) and (
            (sel.confidence or 0.0) < low
            or FLAG_EXACT_MISSING in (sel.flag_reasons or [])
            or "verifier_failed" in (sel.flag_reasons or []))
        no_clip = sel is None or not has_src
        # A beat that tells the viewer to LOOK at something and never got footage showing it is
        # better served by a still OF THAT MOMENT than by moving footage of the wrong scene: the
        # owner's rule is "exact scene ka screenshot bhi use kar sakta ho". The verifier records the
        # miss (see the look gate in verify), so this reads the flag rather than re-deciding.
        look_missed = bool(sel is not None
                           and "look_target_missing" in (sel.flag_reasons or []))
        want_still = (no_clip or pol == _policy.ABSTRACT
                      or (pol == _policy.FILLER and repeat) or weak or look_missed)
        # ESSENTIAL fills (a beat with NO clip, or an exact-scene recovery) must always proceed — the
        # still_cap only throttles OPTIONAL repeat-breaking / variety stills, not coverage (req. 5).
        # a verifier-REJECTED beat's still is COVERAGE, not variety — the release gate blocks
        # ANY policy's rejected beat, so none of them may be starved by the variety cap
        # (observed: an abstract outro beat hit `cap 24` late in the video, got no still, and
        # release-blocked the render)
        essential = no_clip or weak
        within_cap = essential or src_filled < still_cap
        if want_still and within_cap and not (sel and getattr(sel, "image_path", "")):
            # Collect several RANKED still candidates, then (for exact/character beats) install the
            # first one that passes SEMANTIC verification. A verdict of 'reject' (wrong character/era/
            # contradictory) or 'unverified' (transport error / no verdict) is NOT installed — we try
            # the next candidate; if none pass, the beat is left flagged (PASS 3 → web-exact / review),
            # and build will NOT air a rejected still or the rejected moving clip (freeze fallback).
            _cands = []
            _seen_keys = set(used_keys)
            if sel is not None:
                _s0 = _imgfb.pick_source_still(sel, shots_by_key, _seen_keys, used_phash,
                                               distinct_from=aired_phash)
                if _s0:
                    _cands.append(_s0); _seen_keys = _seen_keys | {(_s0[1], _s0[2])}
            # FACE-AWARE pool pick for a named-character beat: without want_faces the picker fed
            # 3 no-face candidates and every one was (correctly) rejected by the still checks —
            # the beat then release-blocked despite the pool being FULL of the right character.
            _wf = (entity_name_variants(getattr(seg, "required_entity", ""), char2actor)
                   if (getattr(seg, "required_kind", "") or "").lower() in ("character", "actor")
                   else None)
            # candidate breadth is decisive: measured on a blocked render, the first ACCEPTED
            # frame for one starved beat was candidate #4 — a cap of 3 release-blocked it every
            # run. Each candidate is still individually vision/deterministically verified.
            _cand_n = max(3, int(os.environ.get("VIDLORE_CLIPSTUDIO_STILL_CANDIDATES", "6") or 6))
            for _ in range(_cand_n):                   # ranked pool candidates
                _sp = _imgfb.pick_pool_still(seg, shots_by_key, _seen_keys, used_phash,
                                             distinct_from=aired_phash, want_faces=_wf,
                                             embeds_of=_embeds_of, rel_memo=_rel_memo)
                if not _sp:
                    break
                _cands.append(_sp); _seen_keys = _seen_keys | {(_sp[1], _sp[2])}
            still = None
            _still_verified = False
            if pol in (_policy.EXACT, _policy.CHARACTER):
                _rej = _unv = 0
                _disabled = False
                for _c in _cands:
                    _vd = _still_verdict(_c[0], seg, _c[1], _c[2])
                    if _vd == "ok":
                        still, _still_verified = _c, True
                        break
                    if _vd == "disabled":                # no verifier configured at all
                        _disabled = True
                        continue
                    _rej += (_vd == "reject"); _unv += (_vd == "unverified")
                if still is None and _disabled and _cands:
                    # NO vision model: do NOT install an arbitrary 'unverified_fallback' (it could be a
                    # contradictory frame). Install ONLY a candidate that passes DETERMINISTIC checks
                    # (same-show + era + CLIP relevance + actual Face-ID) — labelled contextual_fallback
                    # (deterministically verified); else leave unresolved for review.
                    _det = next((c for c in _cands if _still_deterministic_ok(c[1], c[2], c[3], seg)), None)
                    if _det is not None:
                        still, _still_verified = _det, True
                    else:
                        log(f"image-fallback: beat {seg.index} — no vision model and no candidate "
                            f"passed the deterministic same-show/era/CLIP/Face-ID checks → left "
                            f"unresolved (no arbitrary unverified still installed)")
                _lowres_pick = False
                _lr_cands = []
                if still is None and shots_lowres_by_key:
                    # LAST-RESORT sub-HD retry: this project's only face-confirmed shots of the
                    # required character may live entirely in sub-480p uploads (observed: ALL 73
                    # Tywin-confirmed shots were 360p, so the HD-only pool starved every fallback
                    # and the render release-blocked). A soft-but-CORRECT still of the right
                    # person beats a dead render — relevance-first, same doctrine as match's
                    # '360p exact beats irrelevant 1080p'. The SAME semantic verification
                    # applies (vision verdict, else the deterministic same-show/era/CLIP/Face-ID
                    # gate); watermarked and reaction/essay sources stay excluded from this pool.
                    _lr_seen = set(used_keys)
                    _lr_cands = []
                    for _ in range(_cand_n):
                        _lp = _imgfb.pick_pool_still(seg, shots_lowres_by_key, _lr_seen,
                                                     used_phash, distinct_from=aired_phash,
                                                     want_faces=_wf, embeds_of=_embeds_of,
                                                     rel_memo=_rel_memo)
                        if not _lp:
                            break
                        _lr_cands.append(_lp); _lr_seen = _lr_seen | {(_lp[1], _lp[2])}
                    for _c in _lr_cands:
                        _vd = _still_verdict(_c[0], seg, _c[1], _c[2])
                        if _vd == "ok" or (_vd == "disabled"
                                           and _still_deterministic_ok(_c[1], _c[2], _c[3], seg)):
                            still, _still_verified, _lowres_pick = _c, True, True
                            log(f"image-fallback: beat {seg.index} — LAST-RESORT sub-HD still "
                                f"installed (semantically verified; the only confirmed footage "
                                f"of the required subject is low-res)")
                            break
                        _rej += (_vd == "reject"); _unv += (_vd == "unverified")
                        log(f"image-fallback: beat {seg.index} — last-resort candidate "
                            f"src={str(_c[1])[:28]}#{_c[2]} verdict={_vd} (not installed)")
                    if still is None and not _lr_cands:
                        log(f"image-fallback: beat {seg.index} — last-resort pool offered no "
                            f"candidate (face/quality/CLIP floors)")
                if still is None and _unv > 0 and _rej == 0 and (_cands or _lr_cands):
                    # VISION-OUTAGE signature: every judged candidate came back 'unverified'
                    # (transport failures — NOT semantic rejections; a mixed batch with any real
                    # 'reject' means the API works and the verdicts stand). A 2-minute API blip
                    # starved a beat whose frames verified 'keep' an hour earlier — wait the blip
                    # out ONCE and re-judge the top candidates; still fail-closed if the outage
                    # persists.
                    import time as _time_ov
                    _time_ov.sleep(float(os.environ.get(
                        "VIDLORE_CLIPSTUDIO_VISION_RETRY_SEC", "25") or 25))
                    for _c in (list(_cands) + list(_lr_cands))[:3]:
                        _vd = _still_verdict(_c[0], seg, _c[1], _c[2])
                        if _vd == "ok":
                            still, _still_verified = _c, True
                            _lowres_pick = _c in _lr_cands
                            log(f"image-fallback: beat {seg.index} — vision recovered after an "
                                f"outage window; still installed on backoff retry")
                            break
                        _rej += (_vd == "reject")
                        _unv += (_vd == "unverified")
                if still is None and _unv > 0 and _rej == 0:
                    _strict_unverified_only[int(seg.index)] = {
                        "unverified": int(_unv),
                        "source_candidates": len(_cands),
                        "lowres_candidates": len(_lr_cands),
                    }
                if still is None and _cands:
                    log(f"image-fallback: beat {seg.index} — {len(_cands)} recovery still(s) not "
                        f"installed ({_rej} rejected wrong-char/era, {_unv} unverified) → left for "
                        f"web-exact/review (no wrong still airs)")
            else:
                still, _still_verified = (_cands[0] if _cands else None), True   # filler: CLIP-ranked ok
                _lowres_pick = False
            if still:
                kf, sid, sidx, score, sph = still
                if sel is None:
                    sel = ClipSelection(segment_index=seg.index, source_id="", shot_index=-1,
                                        in_point=0, out_point=0, confidence=0.0)
                    proj.selections.append(sel)
                    sel_by_idx[seg.index] = sel
                sel.image_path = str(kf)
                _src = "source-frame-recovery" if (pol == _policy.EXACT and (weak or no_clip)) \
                    else "source-frame"
                # HONEST relevance class for the audit (req. 4): a real source keyframe of the right
                # subject is a CONTEXTUAL fallback for an exact/character beat (correct character/
                # scene/era, but the exact moment is not verifier-confirmed), and thematic FILLER for
                # a generic/abstract beat. Never labelled 'exact' — only verifier-kept moving footage
                # or a validated web-exact-scene still earns that.
                _sev = dict(_still_evidence.get((str(kf), int(seg.index)), {}) or {})
                _semantic_ok = bool(_sev and not _strict_still_evidence_reason(_sev, seg))
                if pol == _policy.EXACT and _semantic_ok:
                    _rclass = "exact_scene"
                elif pol in (_policy.EXACT, _policy.CHARACTER):
                    _rclass = "contextual_fallback" if _still_verified else "unverified_fallback"
                else:
                    _rclass = "generic_filler"
                sel.image_meta = {"source": _src, "score": round(float(score), 3),
                                  "src": sid, "shot": sidx, "relevance_class": _rclass,
                                  "still_verified": bool(_still_verified),
                                  "still_semantic_verified": bool(_semantic_ok),
                                  "still_verifier": _sev,
                                  "still_image_sha256": (_still_image_sha256(kf)
                                                         if _semantic_ok else ""),
                                  "exact_still_verified": bool(
                                      pol == _policy.EXACT and _semantic_ok),
                                  "exact_still_verifier": (_sev if pol == _policy.EXACT else {}),
                                  "lowres_still": bool(_lowres_pick),
                                  "exact_scene_missing": bool(
                                      pol == _policy.EXACT and not _semantic_ok)}
                used_keys.add((sid, sidx))
                if sph:
                    used_phash.add(sph)
                src_filled += 1
                continue
        if has_src:                                    # airs its moving footage → record what's shown
            aired_shots.add((sel.source_id, sel.shot_index))
            if ph:
                aired_phash.add(ph)

    # ---- PASS 2 — WEB-EXACT-SCENE (last resort, ONLY exact/character beats still uncovered): a real
    # LIVE-ACTION still, strictly validated (CLIP relevance + Face-ID + no watermark/text/collage/
    # poster, AI sources rejected). NEVER for generic_filler/abstract (those use source frames only).
    seen_hashes: set = set()
    if web_fallback_on:
        for seg in segs:
            sel = sel_by_idx.get(seg.index)
            _isrc = (getattr(sel, "image_meta", {}) or {}).get("source", "") if sel else ""
            _has_img = bool(sel and getattr(sel, "image_path", ""))
            _strict_policy = _policy.policy_of(seg) in (_policy.EXACT, _policy.CHARACTER)
            _strict_img_confirmed = bool(
                _has_img and _strict_policy and _verified_still_coverage(sel, seg)[0])
            if sel is not None and sel.source_id and not _has_img:
                continue                               # airs real moving footage — leave it
            # An EXACT beat covered by ANY merely-contextual source frame still TRIES web. The old
            # source-name shortcut treated `source-frame` as confirmed while the publication gate
            # correctly rejected it, creating a deterministic recovery/build dead end. Only a
            # validated web exact-scene or future strict exact-still evidence may suppress recovery.
            if _has_img:
                if _strict_policy:
                    if _strict_img_confirmed:
                        continue
                elif _isrc != "source-frame-recovery":
                    continue                           # concrete contextual still is confirmed
            if not _policy.allows_web_image(seg):      # filler/abstract → no random web decoration
                continue
            # A web-exact-scene still for an EXACT beat with NO real footage and NO confirmed still is
            # ESSENTIAL COVERAGE, not decorative variety — it must bypass the still cap. Without this
            # the cap (already exceeded by essential source-frame stills) breaks the whole web pass on
            # its first iteration, so footage-gap beats (barely-filmed backstory characters) get NO
            # real still and the render re-airs verifier-REJECTED footage on them instead. Only the
            # OPTIONAL web fills (a beat that already has some coverage) stay capped.
            _essential_web = (
                (_strict_policy and _has_img and not _strict_img_confirmed)
                or (_policy.is_exact(seg) and not _has_img
                    and not (sel is not None and getattr(sel, "source_id", ""))))
            if not _essential_web and (src_filled + web_filled) >= cap:
                continue
            try:
                res = _imgfb.fetch_scene_image(seg, analysis, img_dir, faceid_obj=faceid_obj,
                                               refs=refs, char2actor=char2actor, log=log,
                                               seen_hashes=seen_hashes, eng_cfg=eng_cfg,
                                               raise_on_technical=fail_on_web_technical)
            except _imgfb.SceneImageTechnicalError:
                if fail_on_web_technical:
                    raise
                log(f"image-fallback: beat {seg.index} web acquisition was technically "
                    "inconclusive")
                res = None
            except Exception as e:
                if fail_on_web_technical:
                    raise _imgfb.SceneImageTechnicalError(
                        f"unexpected web-image failure for beat {seg.index}: "
                        f"{type(e).__name__}: {e}") from e
                log(f"image-fallback: beat {seg.index} web error ({type(e).__name__})")
                res = None
            if not res:
                continue
            if sel is None:
                sel = ClipSelection(segment_index=seg.index, source_id="", shot_index=-1,
                                    in_point=0, out_point=0, confidence=0.0)
                proj.selections.append(sel)
                sel_by_idx[seg.index] = sel
            sel.image_path = str(res["path"])
            _web_ev = dict(res.get("strict_verifier") or {})
            _web_semantic = bool(_web_ev and not _strict_still_evidence_reason(_web_ev, seg))
            # Candidate-source/CLIP/Face-ID gates are not publication proof. Persist the strict
            # verdict on the actual downloaded image and bind it to those exact bytes.
            sel.image_meta = {"source": "web-exact-scene", "score": res.get("score"),
                              "clip": res.get("clip"), "face": res.get("face"), "query": res.get("query"),
                              "relevance_class": ("exact_scene" if _policy.is_exact(seg)
                                                  else "contextual_fallback"),
                              "still_verified": bool(_web_semantic),
                              "still_semantic_verified": bool(_web_semantic),
                              "still_verifier": _web_ev,
                              "still_image_sha256": str(res.get("image_sha256") or ""),
                              "exact_still_verified": bool(
                                  _policy.is_exact(seg) and _web_semantic),
                              "exact_still_verifier": (_web_ev if _policy.is_exact(seg) else {})}
            web_filled += 1

    # A strict beat with only transport/malformed non-verdicts has NOT exhausted image recovery.
    # Give a successfully verified web result the final word; otherwise stop before PASS 3 mutates
    # flags and before the caller can run specificity softening or write an exhaustion marker.
    _strict_technical_pending = []
    for _idx, _facts in sorted(_strict_unverified_only.items()):
        _sel = sel_by_idx.get(_idx)
        _seg = next((s for s in segs if int(getattr(s, "index", -1)) == _idx), None)
        _covered = False
        if _seg is not None and _sel is not None and getattr(_sel, "image_path", ""):
            try:
                _covered = bool(_verified_still_coverage(_sel, _seg)[0])
            except Exception:                            # noqa: BLE001 — absence remains inconclusive
                _covered = False
        if not _covered:
            _strict_technical_pending.append((_idx, _facts))
    if _strict_technical_pending:
        detail = ", ".join(
            f"{idx}({facts['unverified']} unverified)" for idx, facts in _strict_technical_pending)
        raise PipelineError(
            "strict image fallback verifier was technically inconclusive for beat(s) " + detail)

    # ---- PASS 3 — exact_scene still uncovered/unconfirmed → MANUAL REVIEW (never silent weak filler
    # or AI). A source-frame-recovery still covers it visually but stays flagged; a validated
    # web-exact-scene still (or verifier-kept footage) counts as confirmed coverage.
    exact_missing = 0
    for seg in segs:
        if not _policy.is_exact(seg):
            continue
        sel = sel_by_idx.get(seg.index)
        has_img = bool(sel and getattr(sel, "image_path", ""))
        # If an image overlays the clip, audit the image actually shown. A keep verdict for the
        # hidden moving selection cannot make a contextual exact still publishable.
        confirmed = (_verified_still_coverage(sel, seg)[0] if has_img
                     else bool(sel is not None and _verifier_kept(sel)))
        if confirmed:
            continue
        if sel is None:
            sel = ClipSelection(segment_index=seg.index, source_id="", shot_index=-1,
                                in_point=0, out_point=0, confidence=0.0)
            proj.selections.append(sel)
            sel_by_idx[seg.index] = sel
        if FLAG_EXACT_MISSING not in sel.flag_reasons:
            sel.flag_reasons.append(FLAG_EXACT_MISSING)
            sel.flagged = True
            exact_missing += 1

    log(f"8b/9 · still pass: {src_filled} source-frame still(s) (cap {still_cap}) · "
        f"{web_filled} web-exact-scene still(s) · {exact_missing} exact-scene flagged for review "
        f"(no AI / NOW / generic-web images)")
    return src_filled + web_filled


def _purge_unwanted_sources(proj, log) -> int:
    """Belt-and-suspenders for EXISTING projects. A reaction / essay / non-show source that was
    downloaded + indexed BEFORE the discovery title-gate was fixed persists in proj.sources (the
    download merge preserves prior sources; indexing resume-skips them) with status='ok', and
    re-running discovery does NOT re-evaluate it. So on EVERY run we re-apply is_unwanted_source_title()
    to already-stored sources and BLOCK the junk ones. A blocked source (status != SOURCE_OK) is then
    excluded UNCONDITIONALLY by every downstream pool — match (_load_pool), still-recovery (shots_by_key),
    and indexing — even if the env title-gates (VIDLORE_CLIPSTUDIO_NONSHOW_GATE) are disabled. Idempotent."""
    from . import discover as _discover
    from .models import SOURCE_OK, SOURCE_BLOCKED
    n = 0
    for s in proj.sources:
        if s.status == SOURCE_OK and _discover.is_unwanted_source_title(getattr(s, "title", "") or ""):
            s.status = SOURCE_BLOCKED
            n += 1
            if log:
                log(f"  [purge] blocked cached reaction/essay/non-show source: {(s.title or '')[:54]!r}")
    return n


def recovery_query(seg) -> str:
    """The same authored text hierarchy that makes a beat retrievable by matching/discovery."""
    if seg is None:
        return ""
    return next((str(getattr(seg, field, "") or "").strip()
                 for field in ("scene_query", "required_entity", "expected_visual", "text")
                 if str(getattr(seg, field, "") or "").strip()), "")


def recovery_pick(unresolved: list, seg_by_idx: dict, policy_mod, tried: set,
                  max_beats: int) -> list:
    """Which unresolved beats get this bounded recovery round's slots.

    Module level so the rotation is testable without a project — the ordering IS the fix (see the
    call site for the measurement).

      1. Beats with no effective matcher/discovery text at all cannot be rediscovered; they must
         not burn a slot. ``expected_visual`` and narration text are real fallback queries.
      2. Gate-vulnerable classes first: CHARACTER, then FILLER/ABSTRACT, then EXACT.
      3. Within a class, a beat that has never been searched outranks one already tried and not
         recovered. Re-attempts are kept, just last: a later round has a larger pool.
      4. Script order breaks the remaining ties, so the result is still fully deterministic.
    """
    def _rank(i):
        s = seg_by_idx.get(i)
        p = policy_mod.policy_of(s) if s is not None else policy_mod.FILLER
        return ({policy_mod.CHARACTER: 0, policy_mod.FILLER: 1, policy_mod.ABSTRACT: 1,
                 policy_mod.EXACT: 2}.get(p, 1), 1 if i in (tried or set()) else 0, i)

    have_query = [i for i in unresolved if recovery_query(seg_by_idx.get(i))]
    return sorted(have_query, key=_rank)[:max_beats]


def _write_recovery_audit(proj, audit: dict,
                          filename: str = "recovery_audit.json") -> None:
    import json as _json
    try:
        (proj.output_dir / filename).write_text(_json.dumps(audit, indent=1), encoding="utf-8")
    except Exception:
        pass


_SEMANTIC_RECOVERY_PAGE_SCHEMA = 1


def _verifier_summary_error(summary) -> str:
    """Why a scoped verifier pass is technically inconclusive, else ``""``.

    The strict contract can only judge evidence that was actually produced. A partial batch with
    even one transport error, an unavailable provider, or an open breaker is not a negative content
    verdict and must never authorize fallback, specificity loss, or a completed recovery page.
    """
    if not isinstance(summary, dict):
        return "verifier_summary_missing"
    available = summary.get("available")
    if available is not True:
        return "verifier_available_false" if available is False else "verifier_available_missing"
    errored = summary.get("errored")
    if isinstance(errored, bool) or not isinstance(errored, int):
        return "verifier_summary_errored_missing_or_malformed"
    if errored != 0:
        return f"verifier_errored:{errored}"
    verifier_down = summary.get("verifier_down")
    if verifier_down is not False:
        if verifier_down is True:
            return "verifier_down"
        return "verifier_down_missing_or_malformed"
    return ""


def _verifier_summary_is_global_outage(summary) -> bool:
    """Whether a scoped verifier summary says the backend/batch itself is unusable.

    A well-formed available batch with an explicit closed breaker may contain isolated per-beat
    errors; those can be partitioned. Missing/malformed liveness, an unavailable provider, or an
    open breaker means recovery would immediately spend more work against a dead dependency.
    """
    if not isinstance(summary, dict):
        return True
    if summary.get("available") is not True:
        return True
    if summary.get("verifier_down") is not False:
        return True
    errored = summary.get("errored")
    return isinstance(errored, bool) or not isinstance(errored, int)


def _load_conclusive_semantic_recovery_page(path: Path, *, request_id: str,
                                             requested_scope: set[int]) -> dict:
    """Load one freshly-requested semantic recovery page, or fail closed.

    The audit is the hand-off between the bounded recovery helper and the image/softening stages.
    A helper return value cannot prove that its page actually ran: the helper deliberately catches
    environmental failures, and a stale audit may still be present from an earlier Resume.  Require
    a per-call nonce, exact scope/partition, an explicit completion bit, and an error-free current
    pool attempt before any downstream loss of specificity is allowed.
    """
    import json as _json_page

    try:
        raw = _json_page.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                            # noqa: BLE001 — missing/corrupt is retryable
        raise PipelineError(
            f"semantic recovery page audit is missing or corrupt: {type(exc).__name__}: {exc}") \
            from exc
    if not isinstance(raw, dict):
        raise PipelineError("semantic recovery page audit is not an object")

    def _indices(name: str) -> list[int]:
        value = raw.get(name)
        if not isinstance(value, list) or any(
                isinstance(i, bool) or not isinstance(i, int) for i in value):
            raise PipelineError(f"semantic recovery page audit has invalid {name}")
        if len(value) != len(set(value)):
            raise PipelineError(f"semantic recovery page audit has duplicate {name}")
        return value

    expected = sorted({int(i) for i in requested_scope})
    if raw.get("schema_version") != _SEMANTIC_RECOVERY_PAGE_SCHEMA:
        raise PipelineError("semantic recovery page audit schema mismatch")
    if raw.get("request_id") != request_id:
        raise PipelineError("semantic recovery page audit request mismatch")
    if sorted(_indices("requested_scope")) != expected:
        raise PipelineError("semantic recovery page audit scope mismatch")

    page_scope = _indices("page_scope")
    deferred = _indices("deferred")
    if set(page_scope) & set(deferred) or sorted(page_scope + deferred) != expected:
        raise PipelineError("semantic recovery page audit partition mismatch")
    deferred_retriable = _indices("deferred_retriable")
    if not set(deferred_retriable).issubset(set(deferred)):
        raise PipelineError("semantic recovery page audit retriable-deferred mismatch")

    current = raw.get("current_pool_rematch")
    if page_scope:
        if not isinstance(current, dict):
            raise PipelineError("semantic recovery page audit lacks current-pool result")
        if "error" not in current or not isinstance(current.get("error"), str):
            raise PipelineError("semantic recovery page audit lacks explicit current-pool status")
        attempted = current.get("attempted")
        if (not isinstance(attempted, list)
                or any(isinstance(i, bool) or not isinstance(i, int) for i in attempted)
                or sorted(attempted) != sorted(page_scope)):
            raise PipelineError("semantic recovery page audit attempted-scope mismatch")
    if isinstance(current, dict) and str(current.get("error") or "").strip():
        raise PipelineError(
            f"semantic recovery current-pool attempt was inconclusive: {current['error']}")
    if "page_error" not in raw or not isinstance(raw.get("page_error"), str):
        raise PipelineError("semantic recovery page audit lacks explicit error status")
    if str(raw["page_error"] or "").strip():
        raise PipelineError(f"semantic recovery page was inconclusive: {raw['page_error']}")
    if raw.get("page_completed") is not True:
        raise PipelineError("semantic recovery page did not record conclusive completion")
    return raw


def _quote_window_recovery_selections(proj, segs, cfg, scope, *, quote_pool_cache=None) \
        -> tuple[dict[int, object], dict]:
    """Build scoped selections directly from authoritative whole-pool quote locations.

    This is a recovery candidate builder, not a new matcher or a permissive acceptance path.  It is
    deliberately called only for blockers selected into one bounded semantic page.  A real authored
    quote has already been located by ``find_quote_span`` across the complete dialogue-eligible pool;
    rebuilding the whole project matcher merely to rediscover that source/time lets diversity and
    anti-reuse choose a neighbouring shot again.  Instead, put the full located phrase into a small
    exact-window bench.  The ordinary strict verifier and relevance contract decide whether those
    pixels actually depict the beat, and ``_commit_scoped_recovery`` owns the eventual recut/rollback.

    Only natively-HD, locally probeable sources become candidates.  A sole SD quote hit is useful
    absence evidence, but accepting it here would only move the same beat to the later native-HD
    publication failure instead of letting acquisition seek a publishable copy.
    """
    import copy as _copy_quote

    from . import audio_align as _audio_quote
    from . import index as _index_quote
    from . import match as _match_quote
    from . import relevance_contract as _rel_quote
    from . import verify as _verify_quote
    from .ingest import probe as _probe_quote
    from .models import ClipCandidate as _ClipCandidate_quote
    from .models import ClipSelection as _ClipSelection_quote
    from .quality_contract import native_video_ok as _native_video_ok_quote

    if quote_pool_cache is None:
        contracts = _rel_quote._quote_pool_branches(proj, segs, cfg=cfg)
    elif type(quote_pool_cache) is _rel_quote._RequestQuotePoolClassificationCache:
        contracts = quote_pool_cache.contracts_for(proj, segs, cfg=cfg)
    else:
        # Do not let a caller-authored ``paraphrase`` or fabricated span enter this evidence path.
        raise TypeError("quote_pool_cache must be a request-local classification cache")

    seg_by_idx = {int(getattr(seg, "index", -1)): seg for seg in (segs or [])}
    sel_by_idx = {int(getattr(sel, "segment_index", -1)): sel
                  for sel in (getattr(proj, "selections", None) or [])}
    dim_cache: dict[str, dict] = {}
    built: dict[int, object] = {}
    result = {
        "attempted": [],
        "recovered": [],
        "still_unresolved": [],
        "error": "",
        "beats": [],
    }

    def _window(seg, src, shots, q0: float, q1: float) -> tuple[float, float] | None:
        if not (q1 > q0 >= 0.0):
            return None
        qdur = q1 - q0
        try:
            narration_need = float(getattr(seg, "est_duration", 0.0) or 0.0) + 0.6
            normal_need = min(float(getattr(cfg, "max_clip_sec", qdur) or qdur),
                              max(float(getattr(cfg, "min_clip_sec", 0.0) or 0.0),
                                  narration_need))
        except (TypeError, ValueError):
            normal_need = qdur
        # Never truncate a long spoken phrase to a short narration beat (the measured beat-13 bug).
        need = max(qdur, normal_need)
        extra = max(0.0, need - qdur)
        start = max(0.0, q0 - extra / 2.0)
        end = q1 + (extra - (q0 - start))
        source_end = max(
            float(getattr(src, "duration", 0.0) or 0.0),
            max((float(getattr(shot, "end", 0.0) or 0.0) for shot in shots), default=0.0))
        if source_end > 0.0 and end > source_end:
            shift = end - source_end
            end = source_end
            start = max(0.0, start - shift)
        if start > q0 or end < q1:
            return None
        return round(start, 3), round(end, 3)

    for idx in sorted({int(i) for i in (scope or set())}):
        seg = seg_by_idx.get(idx)
        branch = dict(contracts.get(idx) or {})
        beat_row = {
            "segment_index": idx,
            "branch": str(branch.get("branch", "") or ""),
            "authored_quote": str(branch.get("authored_quote", "") or ""),
            "status": "not_verbatim",
            "candidates": [],
        }
        if seg is None or branch.get("branch") != "verbatim":
            result["beats"].append(beat_row)
            continue

        raw_matches = list(branch.get("pool_matches") or [])
        if not raw_matches and isinstance(branch.get("pool_match"), dict):
            raw_matches = [branch["pool_match"]]          # old contract/audit compatibility
        candidates = []
        candidate_rows: dict[tuple, dict] = {}
        candidate_shots: dict[tuple, object] = {}
        seen_direct_occurrences: set[tuple[str, float, float]] = set()
        # This remains source-level deliberately: a source with a direct timed-ASR hit must not be
        # searched again through the weaker cross-copy PCM rung. Direct recovery itself, however,
        # must try every distinct confirmed occurrence in that source.
        seen_sources: set[str] = set()

        def _target_quote_span(words):
            """Use the same branch-specific locator that typed the whole-pool quote.

            A fuzzy/non-contiguous short phrase is explicitly *not* verbatim evidence.  Treating
            it as a direct target hit here prevented the stronger cross-copy PCM locator from ever
            running (measured beat 5: ``You``/``shot`` and a much later ``me``).
            """
            quote = str(getattr(seg, "quote", "") or "")
            if bool(branch.get("requires_exact_contiguous_match")):
                return _rel_quote._exact_contiguous_quote_span(
                    words, quote, index_module=_index_quote)
            return _index_quote.find_quote_span(
                words, quote, min_ratio=float(_rel_quote.QUOTE_DIALOGUE_FLOOR))

        def _candidate_from_span(sid: str, q0: float, q1: float, ratio: float,
                                 row: dict, *, transfer_context: dict | None = None) -> bool:
            """Apply the unchanged HD/window gates to one ASR or PCM-located phrase."""
            src = proj.source(sid)
            path = str(getattr(src, "local_path", "") or "") if src is not None else ""
            if (src is None or str(getattr(src, "status", "") or "") != SOURCE_OK
                    or not path or not Path(path).is_file()):
                row["status"] = "source_unavailable"
                return False
            if path not in dim_cache:
                try:
                    dim_cache[path] = dict(_probe_quote(Path(path)) or {})
                except Exception:
                    dim_cache[path] = {}
            dims = dim_cache[path]
            row["native_width"] = int(dims.get("width") or 0)
            row["native_height"] = int(dims.get("height") or 0)
            row["native_hd"] = bool(_native_video_ok_quote(dims))
            if not row["native_hd"]:
                row["status"] = "skipped_non_hd_or_unprobeable"
                return False
            try:
                shots = list(_index_quote.load_shots(proj, sid) or [])
            except Exception:
                shots = []
            overlaps = [shot for shot in shots
                        if float(getattr(shot, "end", 0.0) or 0.0) >= q0
                        and float(getattr(shot, "start", 0.0) or 0.0) <= q1]
            if not overlaps:
                row["status"] = "quote_span_has_no_indexed_shot"
                return False

            def _anchor_rank(shot):
                start = float(getattr(shot, "start", 0.0) or 0.0)
                end = float(getattr(shot, "end", 0.0) or 0.0)
                overlap = max(0.0, min(end, q1) - max(start, q0))
                midpoint = (q0 + q1) / 2.0
                contains_midpoint = 1 if start <= midpoint <= end else 0
                return overlap, contains_midpoint, \
                    float(getattr(shot, "quality", 0.0) or 0.0), \
                    -int(getattr(shot, "index", 0) or 0)

            anchor = max(overlaps, key=_anchor_rank)
            selected_window = _window(seg, src, shots, q0, q1)
            if selected_window is None:
                row["status"] = "quote_span_outside_source"
                return False
            transfer_corr = (float((transfer_context or {}).get(
                "alignment", {}).get("correlation", 0.0) or 0.0)
                if transfer_context else 1.0)
            strength = min(ratio, transfer_corr) if transfer_context else ratio
            signals = {
                "dialogue": round(ratio, 3),
                "moment_lock": 1.0,
                "moment_ratio": round(strength, 3),
                "quote_pool_exact": True,
                "quality": round(float(getattr(anchor, "quality", 0.0) or 0.0), 3),
                "native_width": row["native_width"],
                "native_height": row["native_height"],
            }
            cand = _ClipCandidate_quote(
                segment_index=idx, source_id=sid,
                shot_index=int(getattr(anchor, "index", -1)), score=round(strength, 4),
                in_point=selected_window[0], out_point=selected_window[1], signals=signals)
            try:
                action, reason, _meta = _match_quote.validate_candidate_window(
                    cand, anchor, shots, cfg, seg)
            except Exception as exc:
                row["status"] = f"window_qc_error:{type(exc).__name__}"
                return False
            tolerance = float(_rel_quote.QUOTE_WINDOW_TOLERANCE_SEC)
            contained = (q0 >= float(cand.in_point) - tolerance
                         and q1 <= float(cand.out_point) + tolerance)
            if action == "rejected" or not contained:
                row["status"] = ("window_qc_rejected" if action == "rejected"
                                 else "window_qc_lost_quote_containment")
                row["window_qc_reason"] = str(reason or "")
                return False
            if transfer_context:
                evidence = _audio_quote.make_transfer_evidence(
                    authored_quote=str(getattr(seg, "quote", "") or ""),
                    reference_source_id=transfer_context["reference_source_id"],
                    reference_source_content_fingerprint=
                    transfer_context["reference_source_content_fingerprint"],
                    reference_asr_ratio=ratio,
                    reference_quote_confirmation_artifact_key=
                    transfer_context["reference_quote_confirmation_artifact_key"],
                    reference_quote_confirmation_decoder_fingerprint=
                    transfer_context["reference_quote_confirmation_decoder_fingerprint"],
                    target_source_id=sid,
                    target_source_content_fingerprint=
                    transfer_context["target_source_content_fingerprint"],
                    target_selected_window=[cand.in_point, cand.out_point],
                    alignment=transfer_context["alignment"])
                if not evidence:
                    row["status"] = "audio_transfer_evidence_not_bound"
                    return False
                cand.signals[_audio_quote.AUDIO_QUOTE_TRANSFER_SIGNAL] = evidence
                cand.signals["quote_audio_transfer"] = True
                cand.signals["quote_audio_transfer_correlation"] = round(transfer_corr, 6)
                row["audio_transfer_evidence"] = evidence
            row.update({
                "status": "candidate",
                "anchor_shot_index": int(getattr(anchor, "index", -1)),
                "selected_window": [round(float(cand.in_point), 3),
                                    round(float(cand.out_point), 3)],
                "window_qc": str(action or "ok"),
                "window_qc_reason": str(reason or ""),
            })
            key = (cand.source_id, cand.shot_index, cand.in_point, cand.out_point)
            candidates.append(cand)
            candidate_rows[key] = row
            candidate_shots[key] = anchor
            return True

        for match in raw_matches:
            sid = str((match or {}).get("source_id", "") or "")
            row = {
                "source_id": sid,
                "source_title": str((match or {}).get("source_title", "") or ""),
                "timed_asr_span": list((match or {}).get("timed_asr_span") or []),
                "timed_asr_ratio": (match or {}).get("timed_asr_ratio"),
                "status": "invalid_pool_match",
            }
            beat_row["candidates"].append(row)
            if not sid:
                row["status"] = "missing_source_id"
                continue
            try:
                q0, q1 = (float(row["timed_asr_span"][0]),
                          float(row["timed_asr_span"][1]))
                ratio = float(row["timed_asr_ratio"] or 0.0)
            except (IndexError, TypeError, ValueError):
                continue
            occurrence = (sid, round(q0, 3), round(q1, 3))
            if occurrence in seen_direct_occurrences:
                row["status"] = "duplicate_confirmed_occurrence"
                continue
            seen_direct_occurrences.add(occurrence)
            if ratio < float(_rel_quote.QUOTE_DIALOGUE_FLOOR):
                row["status"] = "below_quote_floor"
                continue
            seen_sources.add(sid)
            row["quote_location_method"] = "timed_asr"
            _candidate_from_span(sid, q0, q1, ratio, row)

        # ASR can miss a short line in one upload even when another copy has authoritative timed
        # words (measured beat 55: clean 1080p scene vs burned-subtitle/SD reference).  Search no
        # new footage: only the already-selected source and its existing alternate/deep source
        # bench are eligible targets, and the same 12-candidate publication cap below still owns
        # the result.  PCM evidence only locates time; every ordinary gate remains in force.
        old_selection = sel_by_idx.get(idx)
        existing_bench = []
        if old_selection is not None:
            existing_bench.append(old_selection)
            existing_bench.extend(list(getattr(old_selection, "alternates", None) or []))
            existing_bench.extend(list(getattr(old_selection, "deep_alternates", None) or []))

        def _bench_sid(item) -> str:
            if isinstance(item, dict):
                return str(item.get("source_id", "") or "")
            return str(getattr(item, "source_id", "") or "")

        target_source_ids = []
        target_seen = set()
        for bench_item in existing_bench:
            target_sid = _bench_sid(bench_item)
            if target_sid and target_sid not in target_seen:
                target_source_ids.append(target_sid)
                target_seen.add(target_sid)
        # Whole-pool ASR sources were already attempted directly above. Remove them before—not
        # after—the bounded target slice, otherwise twelve duplicate bench IDs can consume all
        # twelve slots and starve a valid later clean-copy target without ever being aligned.
        beat_row["audio_transfer_bench_source_count"] = len(target_source_ids)
        duplicate_target_ids = [sid for sid in target_source_ids if sid in seen_sources]
        target_source_ids = [sid for sid in target_source_ids if sid not in seen_sources]
        beat_row["audio_transfer_direct_asr_duplicates_filtered"] = duplicate_target_ids
        pre_cap_excluded = []
        target_words_cache = {}
        pre_cap_eligible = []
        expected_asr_fingerprint = str(
            branch.get("asr_prompt_fingerprint_expected", "") or "")
        for target_sid in target_source_ids:
            target_source = proj.source(target_sid)
            target_path = str(getattr(target_source, "local_path", "") or "") \
                if target_source is not None else ""
            exclusion_reason = ""
            if (target_source is None
                    or str(getattr(target_source, "status", "") or "") != SOURCE_OK
                    or not target_path or not Path(target_path).is_file()):
                exclusion_reason = "source_unavailable"
            else:
                target_asr_ok, _actual, target_asr_reason = \
                    _rel_quote._source_asr_provenance(
                        proj, target_sid, expected_asr_fingerprint)
                if not target_asr_ok:
                    exclusion_reason = f"target_asr_provenance_invalid:{target_asr_reason}"
                else:
                    try:
                        target_words = _index_quote.load_words(proj, target_sid)
                        target_words_cache[target_sid] = target_words
                        target_asr_span = _target_quote_span(target_words)
                    except Exception:
                        target_asr_span = None
                    if target_asr_span:
                        exclusion_reason = "target_already_has_timed_asr_quote"
            if exclusion_reason:
                pre_cap_excluded.append({
                    "source_id": target_sid, "reason": exclusion_reason,
                })
            else:
                pre_cap_eligible.append(target_sid)
        target_source_ids = pre_cap_eligible
        beat_row["audio_transfer_pre_cap_excluded"] = pre_cap_excluded
        _audio_target_source_cap = 12
        beat_row["audio_transfer_target_source_count"] = len(target_source_ids)
        beat_row["audio_transfer_target_source_cap"] = _audio_target_source_cap
        beat_row["audio_transfer_target_source_overflow"] = max(
            0, len(target_source_ids) - _audio_target_source_cap)
        target_source_ids = target_source_ids[:_audio_target_source_cap]

        reference_rejections = []

        def _reject_reference(sid: str, reason: str):
            reference_rejections.append({
                "source_id": str(sid or ""),
                "reason": str(reason or "reference_not_authoritative"),
            })
            return None

        def _reference_record(match):
            sid = str((match or {}).get("source_id", "") or "")
            try:
                q0, q1 = (float((match or {}).get("timed_asr_span", [])[0]),
                          float((match or {}).get("timed_asr_span", [])[1]))
                ratio = float((match or {}).get("timed_asr_ratio", 0.0) or 0.0)
            except (IndexError, TypeError, ValueError):
                return _reject_reference(sid, "confirmed_span_malformed")
            src = proj.source(sid)
            path = str(getattr(src, "local_path", "") or "") if src is not None else ""
            if (not sid or src is None or str(getattr(src, "status", "") or "") != SOURCE_OK
                    or not Path(path).is_file()
                    or ratio < float(_rel_quote.QUOTE_DIALOGUE_FLOOR)):
                return _reject_reference(sid, "source_or_ratio_ineligible")
            provenance_ok, _actual, _reason = _rel_quote._source_asr_provenance(
                proj, sid, str(branch.get("asr_prompt_fingerprint_expected", "") or ""))
            if not provenance_ok:
                return _reject_reference(sid, f"prompted_asr_provenance_invalid:{_reason}")

            # A prompted ASR hit is not an authoritative PCM template by itself.  Revalidate the
            # exact immutable no-prompt artifact here, immediately before decoding its audio.  This
            # also makes hand-authored/legacy branch dictionaries fail closed instead of silently
            # manufacturing a cross-copy quote proof from the old prompted word cache.
            confirmation = (match or {}).get("unprompted_confirmation")
            if not isinstance(confirmation, dict) \
                    or str(confirmation.get("status", "") or "") != "confirmed":
                return _reject_reference(sid, "unprompted_confirmation_absent_or_not_confirmed")
            artifact_key = str(confirmation.get("artifact_key", "") or "").strip().lower()
            decoder_fingerprint = str(
                confirmation.get("decoder_fingerprint", "") or "").strip().lower()
            expected_decoder = str(branch.get(
                "confirmation_decoder_fingerprint_expected", "") or "").strip().lower()

            def _is_sha256(value):
                return len(value) == 64 \
                    and all(c in "0123456789abcdef" for c in value)
            if not _is_sha256(artifact_key):
                return _reject_reference(sid, "unprompted_confirmation_artifact_key_malformed")
            if not _is_sha256(decoder_fingerprint) or decoder_fingerprint != expected_decoder:
                return _reject_reference(sid, "unprompted_confirmation_decoder_mismatch")
            prompted_span = (match or {}).get("prompted_asr_span")
            try:
                current_confirmation = _rel_quote._confirm_prompted_quote_span_unprompted(
                    proj, src, str(getattr(seg, "quote", "") or ""), prompted_span, cfg,
                    exact_contiguous_required=bool(
                        branch.get("requires_exact_contiguous_match")))
            except Exception as exc:
                return _reject_reference(
                    sid, f"unprompted_confirmation_revalidation_error:{type(exc).__name__}")
            if str(current_confirmation.get("status", "") or "") != "confirmed":
                return _reject_reference(
                    sid, "unprompted_confirmation_no_longer_confirmed")
            if str(current_confirmation.get("artifact_key", "") or "") != artifact_key \
                    or str(current_confirmation.get("decoder_fingerprint", "") or "") \
                    != decoder_fingerprint:
                return _reject_reference(sid, "unprompted_confirmation_binding_changed")
            current_span = current_confirmation.get("confirmed_span")
            try:
                c0, c1, cratio = (float(current_span[0]), float(current_span[1]),
                                  float(current_span[2]))
            except (IndexError, TypeError, ValueError):
                return _reject_reference(sid, "unprompted_confirmation_span_malformed")
            if max(abs(c0 - q0), abs(c1 - q1), abs(cratio - ratio)) > 0.001:
                return _reject_reference(sid, "unprompted_confirmation_span_changed")
            fingerprint = _verify_quote._file_fingerprint(path)
            if fingerprint in ("", "missing", "unreadable"):
                return _reject_reference(sid, "source_content_fingerprint_unavailable")
            return {
                "source_id": sid, "path": path, "quote_span": [q0, q1], "ratio": ratio,
                "source_content_fingerprint": fingerprint,
                "quote_confirmation_artifact_key": artifact_key,
                "quote_confirmation_decoder_fingerprint": decoder_fingerprint,
            }

        reference_records = [record for record in (
            _reference_record(match) for match in raw_matches) if record is not None]
        # Strongest ASR copies first; three independent references are ample and bound worst-case
        # local decoding to 12 targets × 3 short templates without expanding the candidate bench.
        reference_records.sort(key=lambda item: (item["ratio"], item["source_id"]), reverse=True)
        _audio_reference_cap = 3
        beat_row["audio_transfer_reference_count"] = len(reference_records)
        beat_row["audio_transfer_reference_cap"] = _audio_reference_cap
        beat_row["audio_transfer_reference_rejections"] = reference_rejections
        reference_records = reference_records[:_audio_reference_cap]

        for target_sid in target_source_ids:
            if not target_sid or target_sid in seen_sources:
                continue
            seen_sources.add(target_sid)
            target_row = {
                "source_id": target_sid,
                "source_title": str(getattr(proj.source(target_sid), "title", "") or ""),
                "status": "audio_transfer_not_attempted",
                "quote_location_method": "cross_copy_pcm",
                "reference_attempts": [],
            }
            beat_row["candidates"].append(target_row)
            target_source = proj.source(target_sid)
            target_path = str(getattr(target_source, "local_path", "") or "") \
                if target_source is not None else ""
            if (target_source is None
                    or str(getattr(target_source, "status", "") or "") != SOURCE_OK
                    or not Path(target_path).is_file()):
                target_row["status"] = "source_unavailable"
                continue
            target_asr_ok, _actual, target_asr_reason = \
                _rel_quote._source_asr_provenance(
                    proj, target_sid, expected_asr_fingerprint)
            if not target_asr_ok:
                target_row["status"] = f"target_asr_provenance_invalid:{target_asr_reason}"
                continue
            try:
                target_words = target_words_cache.get(target_sid)
                if target_words is None:
                    target_words = _index_quote.load_words(proj, target_sid)
                target_asr_span = _target_quote_span(target_words)
            except Exception:
                target_asr_span = None
            if target_asr_span:
                # A current timed-ASR phrase belongs in the ordinary whole-pool path, never a
                # weaker-looking transfer record.  A fresh contract scan will expose it directly.
                target_row["status"] = "target_already_has_timed_asr_quote"
                continue
            try:
                target_shots = list(_index_quote.load_shots(proj, target_sid) or [])
            except Exception:
                target_shots = []
            target_end = max(
                float(getattr(target_source, "duration", 0.0) or 0.0),
                max((float(getattr(shot, "end", 0.0) or 0.0)
                     for shot in target_shots), default=0.0))
            if target_end <= 0.0:
                target_row["status"] = "target_duration_unavailable"
                continue
            target_fp = _verify_quote._file_fingerprint(target_path)
            if target_fp in ("", "missing", "unreadable"):
                target_row["status"] = "target_content_fingerprint_unavailable"
                continue

            accepted_alignments = []
            target_references = [reference for reference in reference_records
                                 if reference["source_id"] != target_sid]
            alignments = _audio_quote.transfer_quote_spans(
                [(reference["path"], reference["quote_span"])
                 for reference in target_references],
                target_path, target_search_window=[0.0, target_end])
            for reference_index, reference in enumerate(target_references):
                alignment = (alignments[reference_index]
                             if reference_index < len(alignments)
                             else {"status": "rejected", "reason": "alignment_result_missing"})
                attempt = {
                    "reference_source_id": reference["source_id"],
                    "status": str(alignment.get("status", "") or ""),
                    "reason": str(alignment.get("reason", "") or ""),
                    "correlation": alignment.get("correlation"),
                    "uniqueness_margin": alignment.get("uniqueness_margin"),
                }
                target_row["reference_attempts"].append(attempt)
                if alignment.get("status") == "matched":
                    accepted_alignments.append((alignment, reference))
            if not accepted_alignments:
                target_row["status"] = "audio_transfer_no_strict_unique_match"
                continue
            accepted_alignments.sort(
                key=lambda pair: (float(pair[0].get("correlation", 0.0) or 0.0),
                                  float(pair[0].get("uniqueness_margin", 0.0) or 0.0),
                                  float(pair[1]["ratio"])), reverse=True)
            alignment, reference = accepted_alignments[0]
            try:
                target_q0, target_q1 = (float(alignment["target_quote_span"][0]),
                                        float(alignment["target_quote_span"][1]))
            except (IndexError, KeyError, TypeError, ValueError):
                target_row["status"] = "audio_transfer_target_span_invalid"
                continue
            target_row["reference_source_id"] = reference["source_id"]
            target_row["timed_asr_span"] = list(reference["quote_span"])
            target_row["timed_asr_ratio"] = reference["ratio"]
            target_row["transferred_quote_span"] = [target_q0, target_q1]
            _candidate_from_span(
                target_sid, target_q0, target_q1, reference["ratio"], target_row,
                transfer_context={
                    "alignment": alignment,
                    "reference_source_id": reference["source_id"],
                    "reference_source_content_fingerprint":
                    reference["source_content_fingerprint"],
                    "reference_quote_confirmation_artifact_key":
                    reference["quote_confirmation_artifact_key"],
                    "reference_quote_confirmation_decoder_fingerprint":
                    reference["quote_confirmation_decoder_fingerprint"],
                    "target_source_content_fingerprint": target_fp,
                })

        if not candidates:
            beat_row["status"] = "no_publishable_exact_window"
            result["beats"].append(beat_row)
            continue

        # Quality/phrase strength order within a scene-affinity tier.  The existing stable source
        # affinity helper then puts a source whose title names the authored scene first; neither sort
        # is an acceptance decision and every candidate is judged below.
        candidates.sort(key=lambda cand: (
            float((cand.signals or {}).get("moment_ratio", 0.0) or 0.0),
            int((cand.signals or {}).get("native_width", 0) or 0)
            * int((cand.signals or {}).get("native_height", 0) or 0),
            float((cand.signals or {}).get("quality", 0.0) or 0.0),
            cand.source_id), reverse=True)
        candidates = _verify_quote._scene_affinity_order(
            candidates, seg, proj,
            str(getattr(sel_by_idx.get(idx), "source_id", "") or ""))
        _candidate_bench_cap = 12
        beat_row["publishable_candidate_count"] = len(candidates)
        beat_row["candidate_bench_cap"] = _candidate_bench_cap
        beat_row["candidate_overflow"] = max(0, len(candidates) - _candidate_bench_cap)
        for overflow in candidates[_candidate_bench_cap:]:
            okey = (overflow.source_id, overflow.shot_index,
                    overflow.in_point, overflow.out_point)
            candidate_rows[okey]["status"] = "candidate_overflow_not_attempted"
        candidates = candidates[:_candidate_bench_cap]
        for rank, cand in enumerate(candidates, 1):
            key = (cand.source_id, cand.shot_index, cand.in_point, cand.out_point)
            candidate_rows[key]["rank"] = rank

        primary = candidates[0]
        old = sel_by_idx.get(idx)
        if old is None:
            new_sel = _ClipSelection_quote(
                segment_index=idx, source_id=primary.source_id,
                shot_index=primary.shot_index, in_point=primary.in_point,
                out_point=primary.out_point, confidence=primary.score)
        else:
            new_sel = _copy_quote.deepcopy(old)
        pkey = (primary.source_id, primary.shot_index, primary.in_point, primary.out_point)
        pshot = candidate_shots[pkey]
        new_sel.source_id = primary.source_id
        new_sel.shot_index = primary.shot_index
        new_sel.in_point = primary.in_point
        new_sel.out_point = primary.out_point
        new_sel.confidence = primary.score
        new_sel.signals = dict(primary.signals or {})
        new_sel.identity = ((getattr(pshot, "face_ids", None) or [""])[0] or "")
        new_sel.identity_score = 0.0
        # This object is a fresh recovery hypothesis.  Failure flags copied from the rejected
        # selection describe the old pixels and must not survive into verifier prediction/self-heal;
        # the verifier and publication contract will add current reasons if this window also fails.
        new_sel.flagged = False
        new_sel.flag_reasons = []
        new_sel.verifier = {}                              # exact new bytes require a fresh judgment
        # Only the primary has been selected for verification.  ASR-identical alternates can be a
        # recap or another scene (the measured beat-84 case), so putting them in beat_windows would
        # let unverified pixels air.  verify._try_promote rewrites this sole entry to the first
        # strictly accepted alternate when promotion succeeds.
        new_sel.beat_windows = [[primary.source_id, round(primary.in_point, 3),
                                 round(primary.out_point, 3)]]
        new_sel.alternates = list(candidates[1:])
        new_sel.deep_alternates = []
        new_sel.source_url = (getattr(proj.source(primary.source_id), "url", "") or "")
        new_sel.image_path = ""
        new_sel.image_meta = {}
        new_sel.clip_path = ""                            # materialized only by scoped commit
        new_sel.approved = False
        new_sel.legibility_grade = ""
        built[idx] = new_sel
        result["attempted"].append(idx)
        beat_row["status"] = "candidate_ready"
        beat_row["primary"] = {
            "source_id": primary.source_id,
            "shot_index": primary.shot_index,
            "selected_window": [primary.in_point, primary.out_point],
        }
        result["beats"].append(beat_row)

    return built, result


def _commit_scoped_recovery(proj, cfg, snapshot: dict, rematched: dict,
                            recovered: set[int], *, log=print) -> bool:
    """Atomically retain and cut only strictly recovered scoped selections.

    ``match_segments`` rebuilds the whole selection list.  Recovery is allowed to retain only
    scoped entries that passed the publication contract; every snapshot entry outside that set is
    restored byte-for-byte, and arbitrary new rematch entries are discarded.  Clip filenames are
    keyed only by beat index, so cutting before reconciliation can overwrite a non-target clip and
    recreate the selection/clip divergence this pipeline's lineage gate was added to catch.

    The final recovered clips are therefore cut only after reconciliation.  Existing target clip
    bytes are backed up, and any partial cut failure rolls both metadata and bytes back.  A recovery
    is never reported unless every retained selection has a fresh clip from its own source window.
    """
    import shutil as _shutil_commit
    import tempfile as _tempfile_commit
    from .cut import cut_selection as _cut_one_commit

    accepted = {int(i) for i in (recovered or set())}
    if any(i not in rematched for i in accepted):
        log("recovery: scoped commit refused — a recovered selection is absent from rematch")
        proj.selections = [snapshot[i] for i in sorted(snapshot)]
        try:
            proj.save()
        except Exception:                                # noqa: BLE001 — caller/final gate stops
            pass
        return False

    final = [rematched[i] if i in accepted else snapshot[i] for i in sorted(snapshot)]
    final.extend(rematched[i] for i in sorted(accepted - set(snapshot)))
    if not accepted:
        proj.selections = final
        try:
            proj.save()                                  # verifier may have persisted its rematch
            return True
        except Exception as exc:                         # noqa: BLE001 — disk state is not proven
            log(f"recovery: scoped metadata restore failed ({type(exc).__name__}: {exc})")
            return False

    backed_up: dict[int, Path] = {}
    clip_paths = {i: proj.clips_dir / f"seg_{i:03d}.mp4" for i in accepted}
    original_exists: Optional[dict[int, bool]] = None
    backup_dir: Optional[Path] = None
    try:
        # Allocation belongs inside the transaction too.  At entry the exploratory matcher may
        # already have replaced ``proj.selections``; if mkdir/mkdtemp fails before the old guard,
        # those uncommitted metadata could remain paired with the original clip bytes.
        original_exists = {i: p.is_file() for i, p in clip_paths.items()}
        proj.output_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = Path(_tempfile_commit.mkdtemp(
            prefix=".scoped_recovery_", dir=str(proj.output_dir)))
        for idx, clip in clip_paths.items():
            if clip.is_file():
                dest = backup_dir / clip.name
                _shutil_commit.copy2(clip, dest)
                backed_up[idx] = dest

        proj.selections = final
        by_idx = {int(s.segment_index): s for s in final}
        for idx in sorted(accepted):
            made = _cut_one_commit(proj, by_idx[idx], cfg, resume=False)
            if made is None or not Path(made).is_file() or Path(made).stat().st_size <= 0:
                raise RuntimeError(f"cut failed for recovered beat {idx}")
            by_idx[idx].clip_path = str(made)
        proj.save()
        return True
    except Exception as exc:                              # noqa: BLE001 — rollback is fail-closed
        for idx, clip in clip_paths.items():
            try:
                if idx in backed_up and backed_up[idx].is_file():
                    _shutil_commit.copy2(backed_up[idx], clip)
                elif original_exists is not None and not original_exists.get(idx, False):
                    clip.unlink(missing_ok=True)
            except Exception:                            # noqa: BLE001 — lineage gate remains final
                pass
        proj.selections = [snapshot[i] for i in sorted(snapshot)]
        try:
            proj.save()
        except Exception:                                # noqa: BLE001
            pass
        log(f"recovery: scoped commit rolled back ({type(exc).__name__}: {exc})")
        return False
    finally:
        if backup_dir is not None:
            _shutil_commit.rmtree(backup_dir, ignore_errors=True)


# Generic title words that carry NO show identity — they must never manufacture a same-show match
# (a shared 'of' / 'the' / 'season' can't make 'The Last of Us' the same show as 'Game of Thrones').
_TITLE_STOP = set(
    "the a an of to in on at for and or but with from into as it its this that is are was were be "
    "s01 s02 season episode ep part pt scene clip full hd 4k official trailer teaser recap explained "
    "breakdown analysis review reaction best moments compilation supercut vs versus ft feat featuring "
    "movie series show tv hbo max netflix subscribe watch online".split())


_EP_CODE_RX = None  # lazy-compiled below


def _norm_title_toks(title: str) -> set:
    """Meaningful (>2-char, non-generic) tokens of a source title — the show-identity fingerprint.
    Season/episode codes (s03, s03e10, e10, 3x10) are stripped too — they are NOT show identity and
    would otherwise survive as 'meaningful' tokens and pollute the same-show comparison."""
    import re as _re
    global _EP_CODE_RX
    if _EP_CODE_RX is None:
        _EP_CODE_RX = _re.compile(r"^(s\d{1,2}(e\d{1,2})?|e\d{1,2}|\d{1,2}x\d{1,2})$")
    return {w for w in _re.findall(r"[a-z0-9]+", (title or "").lower())
            if w not in _TITLE_STOP and len(w) > 2 and not _EP_CODE_RX.match(w)}


def _title_season(title: str) -> str:
    """The season a source TITLE declares (S3E10 / 'season 3' / 'season three' / 3x10) as
    'season N', else ''."""
    from . import era as _era
    return _era.as_era(title or "")


_NAME_ARTICLES = {"the", "a", "an", "of", "and"}


def entity_name_variants(entity: str, char2actor: dict | None = None) -> list:
    """Accepted NAME token-sets for a required entity: the entity itself plus its analysis-roster
    aliases BOTH ways (a beat naming the CHARACTER accepts the ACTOR's Face-ID name and vice
    versa — Face-ID references are built from actor names, so 'joffrey baratheon' must accept a
    'jack gleeson' face). Each variant keeps the ALL-distinctive-tokens rule (Jon Snow vs Jon
    Arryn can never cross-match). Returns a list of lower-cased token sets."""
    import re as _re

    def _toks(s):
        return {w for w in _re.findall(r"[a-z0-9]+", str(s or "").lower())
                if len(w) > 2 and w not in _NAME_ARTICLES}

    ent_toks = _toks(entity)
    variants = [ent_toks] if ent_toks else []
    for ch, ac in (char2actor or {}).items():
        cht, act = _toks(ch), _toks(ac)
        if not (cht and act):
            continue
        if cht <= ent_toks:                     # beat names this CHARACTER → its actor is the same person
            variants.append(act)
        if act <= ent_toks:                     # beat names the ACTOR → their character is the same person
            variants.append(cht)
    return variants


def _deterministic_still_ok(*, source_title, score, seg, faces, movie_toks, global_era,
                            single_scene, min_clip=0.30, char2actor=None,
                            global_verified=False, anchor_eras=None):
    """R4-6 — the PURE deterministic gate for installing a recovery still when NO vision model is
    available. Every claim is EVALUATED (never assumed): a still is accepted only when ALL of
      (1) SAME-SHOW via meaningful normalized title identity (generic/stopwords removed — 'Game of
          Thrones' shares no meaningful token with 'The Last of Us'),
      (2) explicit BEAT-LOCAL season/era vs the SOURCE's declared season (a S5 clip can't cover a
          S3 beat),
      (3) CLIP relevance above a floor, and
      (4) for a named character/actor, that entity actually present in the shot's Face-ID
    pass. Returns (ok: bool, reason: str)."""
    import re as _re
    from .verify import _beat_era
    # (1) same-show — meaningful token identity, not any incidental shared common noun. A SINGLE
    # shared generic token ('dragon' in 'House of the Dragon' vs 'Dragon Ball'; 'game' in 'Game of
    # Thrones' vs 'The Game') is too weak. Require the BULK of the movie's distinctive fingerprint:
    # both tokens of a two-token title, else the one distinctive token of a single-token title.
    mt = {w for w in (movie_toks or set()) if w not in _TITLE_STOP and len(w) > 2
          and not _re.match(r"^(s\d{1,2}(e\d{1,2})?|e\d{1,2}|\d{1,2}x\d{1,2})$", w)}
    tt = _norm_title_toks(source_title)
    if mt:
        if not tt:
            return False, "source title has no meaningful tokens (cannot confirm same show)"
        _shared = mt & tt
        _need = 1 if len(mt) == 1 else 2
        if len(_shared) < _need:
            return False, (f"different / unconfirmed show (title {sorted(tt)} shares only "
                           f"{sorted(_shared)} with movie {sorted(mt)}; need >= {_need})")
    # (2) explicit era: compare the beat's own era to the source title's declared season —
    # CANONICALLY (era strings arrive in mixed formats: 'S04E01' vs 'season 4' is the SAME era).
    # An UNCORROBORATED global hint yields no beat era, so nothing is rejected on era grounds here:
    # calling a source "wrong era" on the strength of a guess is how the correct episode's own
    # upload scored zero shots. The same-show / CLIP / Face-ID gates below still apply.
    from .verify import _era_conflict
    era_beat = _beat_era(seg, global_era, single_scene, global_verified=global_verified,
                         anchor_eras=anchor_eras)
    era_src = _title_season(source_title)
    if _era_conflict(era_beat, era_src):
        return False, f"wrong era (beat {era_beat} vs source {era_src})"
    # (3) CLIP relevance floor
    if float(score or 0.0) < float(min_clip):
        return False, f"CLIP relevance {float(score or 0.0):.2f} < floor {float(min_clip):.2f}"
    # (4) named character must be Face-ID-present in the shot. Match on WHOLE-WORD tokens (not
    # substring — 'the' in 'theon' or 'jon' in 'jonas' must not count) and drop leading articles
    # ('The Hound' -> distinctive token 'hound', not 'the'). Require ALL distinctive tokens of ONE
    # accepted NAME present, so a shared given name or surname alone (Jon Snow vs Jon Arryn; Tywin
    # vs Cersei Lannister) can NEVER satisfy a different character. Face-ID identities are ACTOR
    # names while beats usually name CHARACTERS — a perfect Joffrey frame carries face 'jack
    # gleeson', so the accepted names include the roster (char2actor) mapping BOTH ways.
    _ent = (getattr(seg, "required_entity", "") or "").strip().lower()
    _kind = (getattr(seg, "required_kind", "") or "").lower()
    if _ent and _kind in ("character", "actor"):
        _face_toks = set(_re.findall(r"[a-z0-9]+", " ".join(faces or []).lower()))
        _variants = entity_name_variants(_ent, char2actor)
        _matched = any(v and all(t in _face_toks for t in v) for v in _variants)
        if not _matched:
            return False, (f"required character '{_ent}' not Face-ID-confirmed "
                           f"(need all tokens of one of {[sorted(v) for v in _variants[:4]]} "
                           f"in {sorted(_face_toks)[:6]})")
    return True, f"same-show + era({era_beat or 'any'}) + CLIP + Face-ID all pass"


def _beat_is_unresolved(sel, seg, _policy) -> bool:
    """A beat the build-stage rejected-footage gate could RELEASE-BLOCK, i.e. one recovery must
    try to fix. Two classes:
      • EXACT beats — no selection, a verifier REJECTION (replace), or no source at all, and no
        real still (a source-frame / web-exact-scene still IS coverage).
      • EVERY OTHER policy — a verifier REJECTION with no still. The gate does NOT exempt
        non-exact beats: a rejected character/filler clip whose editorial hold later fails the
        R4-3/R4-4 validity checks release-blocks the finished render exactly like an exact one
        (observed: 7 character_specific beats FATALed a 4½-hour render while recovery reported
        zero unresolved, because this check filtered to exact_scene only). Recovery and the gate
        must see the SAME set — the gate's own block message says 'rediscovery needed'."""
    if seg is None:
        return False
    exact = _policy.is_exact(seg)
    if sel is None:
        return exact                      # a discovery gap: only exact beats demand footage here
    if getattr(sel, "image_path", ""):
        # Judge the image that will actually air, never its source label. Concrete/exact stills
        # require strict actual-image evidence bound to the bytes; legacy `still_verified` and
        # `web-exact-scene` labels are candidates, not coverage.
        if _policy.policy_of(seg) not in (_policy.EXACT, _policy.CHARACTER):
            return False
        from .relevance_contract import verified_still_coverage
        if verified_still_coverage(sel, seg)[0]:
            return False
    v = getattr(sel, "verifier", {}) or {}
    verifier_failed = v.get("status") == "ok" and v.get("verdict") == "replace"
    if exact:
        return bool(verifier_failed or not getattr(sel, "source_id", ""))
    return bool(verifier_failed)


def _ensure_anchor_coverage(proj, analysis, cfg, *, policy: str, roster=None, log=print) -> int:
    """single_scene renders: guarantee the ANCHOR scene has dedicated sources in the pool.

    Coverage test is deterministic: a source counts when its title matches >=2 anchor content
    tokens (prefix-tolerant, movie-title tokens excluded); episode-coded titles matching the
    anchor's episode count double. Below VIDLORE_CLIPSTUDIO_ANCHOR_MIN_SOURCES (2), fetch the
    anchor's own uploads via DIRECT ytsearch (anchor query + episode-code variants), download,
    index. Fail-open everywhere. Env kill: VIDLORE_CLIPSTUDIO_ANCHOR_COVERAGE=0."""
    import os as _os_ac
    import re as _re_ac
    if _os_ac.environ.get("VIDLORE_CLIPSTUDIO_ANCHOR_COVERAGE", "1").strip() \
            in ("0", "false", "no"):
        return 0
    a = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis or {})
    if (a.get("video_type") or "") != "single_scene":
        return 0
    anchors = a.get("anchor_scenes") or []
    if not anchors:
        return 0
    anchor = anchors[0] if isinstance(anchors[0], dict) else {}
    aq = " ".join(str(anchor.get(k, "") or "") for k in ("name", "query"))
    from .era import parse_episode
    epc = parse_episode(str(anchor.get("episode", "") or "") or aq)
    movie = str(a.get("movie_title", "") or "")
    _mv = {w for w in _re_ac.findall(r"[a-z']+", movie.lower()) if len(w) > 2}
    toks = {w for w in _re_ac.findall(r"[a-z']+", aq.lower())
            if len(w) > 3 and w not in _mv and w not in ("scene", "episode", "season", "game",
                                                         "thrones")}
    if len(toks) < 2:
        return 0

    def _hits(title: str) -> int:
        tw = set(_re_ac.findall(r"[a-z']+", (title or "").lower()))
        n = sum(1 for w in toks
                if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2) for t in tw))
        if epc and parse_episode(title or "") == epc:
            n += 2                                       # right episode code is strong identity
        return n

    covered = [s for s in proj.sources
               if getattr(s, "status", "") == SOURCE_OK and _hits(s.title or "") >= 3]
    need = max(1, int(_os_ac.environ.get("VIDLORE_CLIPSTUDIO_ANCHOR_MIN_SOURCES", "2") or 2))
    if len(covered) >= need:
        log(f"5a2/9 · anchor coverage OK — {len(covered)} dedicated source(s) for the anchor "
            f"scene ({sorted(toks)[:4]}…)")
        return 0
    log(f"5a2/9 · anchor coverage LOW ({len(covered)}/{need}) — fetching the anchor scene's "
        f"own uploads directly")
    from . import selfheal as _sh
    from .download import download_candidates
    from . import index as _I
    queries = [f"{movie} {anchor.get('query') or anchor.get('name')}"]
    if epc:
        queries.append(f"{movie} {anchor.get('name', '')} scene "
                       f"S{epc[0]:02d}E{epc[1]:02d}")
        queries.append(f"{movie} {anchor.get('name', '')} {epc[0]}x{epc[1]:02d}")
    cands = _sh._yt_search_candidates(queries[:3], per_query=6, log=log)
    have = {(s.url or "").strip() for s in proj.sources}
    fresh = [c for c in cands if (c.url or "").strip() not in have and _hits(c.title) >= 2]
    fresh.sort(key=lambda c: _hits(c.title), reverse=True)
    fresh = fresh[:max(1, need + 1 - len(covered))]
    if not fresh:
        log("5a2/9 · anchor coverage: no dedicated upload found by direct search")
        return 0
    for c in fresh:
        log(f"5a2/9 · anchor fetch: {(c.title or '')[:76]!r}")
    try:
        download_candidates(proj, fresh, cfg, policy=policy, progress=None)
    except Exception as e:                               # noqa: BLE001
        log(f"5a2/9 · anchor download failed ({str(e)[:60]})")
        return 0
    n = 0
    for sv in proj.sources:
        if sv.url in {c.url for c in fresh} and sv.status == SOURCE_OK and sv.local_path:
            try:
                _I.index_source(proj, sv, cfg, roster=roster, progress=None)
                n += 1
            except Exception as e:                       # noqa: BLE001
                log(f"5a2/9 · anchor index failed for {sv.id} ({str(e)[:50]})")
    proj.save()
    log(f"5a2/9 · anchor coverage: +{n} dedicated source(s) indexed")
    return n


class _IndexPrewarmer:
    """SPEED (OPT-8): one worker THREAD pre-builds per-source index artifacts while the paced
    download window is otherwise dead wait (measured: the 6s-per-file 403-protection gap makes
    ~100% of download wall time deliberate sleep, and the backfill rounds then re-index
    serially). Model thread-safety invariant intact: models are used by exactly ONE thread at
    a time — the worker owns them during the download stage (the main thread only waits on
    yt-dlp subprocesses there) and close() JOINS it before index_all touches them again.
    index_all then runs UNCHANGED and cache-hits the prewarmed artifacts through the same
    battle-tested resume path, so every decision is identical whether or not a source was
    prewarmed; a prewarm failure just means that source indexes fresh, serially, as today.
    download_candidates only hands over sources whose media file is FINAL (403-swept files are
    held back until their sweep resolves, and a sweep replacement purges any stale artifacts).
    VIDLORE_CLIPSTUDIO_INDEX_OVERLAP=0 disables."""

    def __init__(self, proj, cfg, *, references, faceid, roster, log=None):
        import queue as _queue
        import threading as _threading
        self._q: "_queue.Queue" = _queue.Queue()
        self._STOP = object()
        self.count = 0
        self._log = log
        self._t = _threading.Thread(target=self._run, daemon=True,
                                    args=(proj, cfg, references, faceid, roster))
        self._t.start()

    def submit(self, sv):
        self._q.put(sv)

    def _run(self, proj, cfg, references, faceid, roster):
        from .index import index_source as _ix_src
        while True:
            sv = self._q.get()
            if sv is self._STOP:
                return
            try:
                _ix_src(proj, sv, cfg, references=references, faceid=faceid,
                        roster=roster, progress=None)
                self.count += 1
            except Exception:                            # noqa: BLE001 — partial artifacts
                pass                                     # fail the cache check → fresh re-index

    def close(self):
        """Drain + join. MUST complete before any other thread touches the models."""
        self._q.put(self._STOP)
        self._t.join()


def _index_overlap_on() -> bool:
    import os as _os_ov
    return _os_ov.environ.get("VIDLORE_CLIPSTUDIO_INDEX_OVERLAP", "1").strip().lower() \
        not in ("0", "false", "no")


def _backfill_rejected_sources(proj, segs, analysis, cfg, *, refs, faceid_obj, roster,
                               policy, max_sources, show_title, log,
                               input_sig: str = "") -> int:
    """Replace footage the pool gates threw out, BEFORE match ever runs.

    The gates themselves are right — a burned-caption re-upload, a promo-card compilation, a screener
    with a timecode burned into the frame are all genuinely unusable. The defect is what happens
    next: discovery spends its budget once, then `_load_pool` silently drops those sources at MATCH
    time, and nothing goes looking for a clean copy. Measured on job 69d80e9dd4_v4 — 11 of 84 ok
    sources dropped (7 subtitled copies, 2 promo cards, 1 screener, 1 talking-head), and among them
    the single most on-topic upload in the pool ("The Trial of Petyr Baelish", 1080p, 23 min, 215
    shots) plus the only clip of the dagger handover. 207 beats then asked for the trial and got
    scene packs of adjacent scenes: right character, wrong scene, 135 times.

    So: derive the rejections early (a `_load_pool` pass persists them), search for a CLEAN copy of
    each lost upload using its own title as the query, download and index whatever is new, and
    repeat while progress is being made. Every candidate goes through the same gates on the next
    round, so a replacement that is itself subtitled simply gets rejected too and costs one round.

    Returns the number of new sources admitted. Kill switch: VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED=0.
    """
    import os as _os_b
    import re as _re_b
    if _os_b.environ.get("VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", "1").strip() in ("0", "false", "no"):
        return 0
    try:
        rounds = int(_os_b.environ.get("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "2") or 2)
    except (TypeError, ValueError):
        rounds = 2

    from . import match as _M
    from .discover import discover_sources
    from .download import download_candidates
    from .index import index_all as _index_all

    audit = {"schema_version": 2, "status": "running", "reason": "",
             "input_sig": str(input_sig or ""), "rounds": [], "admitted": 0}
    tried_titles: set = set()
    admitted_total = 0
    terminal_reason = "configured_rounds_completed"

    def _finish(status: str, reason: str) -> int:
        audit["status"] = str(status)
        audit["reason"] = str(reason)
        audit["admitted"] = admitted_total
        try:
            proj.meta["backfill_audit"] = audit
            proj.save()
        except Exception:                                # noqa: BLE001 — audit cannot mask result
            pass
        return admitted_total

    for rnd in range(max(1, rounds)):
        # a pool pass re-derives and persists this round's source-level rejections
        try:
            _M._load_pool(proj, cfg, progress=None, show_title=show_title)
        except Exception as e:                                   # noqa: BLE001
            log(f"5b/9 · backfill: pool probe failed ({str(e)[:70]}) — skipping")
            return _finish("incomplete", f"pool_probe_failed:{type(e).__name__}")
        # Only QUALITY rejects deserve a replacement: the footage was right and the copy was not.
        # A CONTENT reject (interview, reaction, wrong show, fan art, still-image) is material we
        # never wanted, and searching for "a cleaner copy" of it just spends the budget pulling in
        # more of the same — measured on the first live run, which burned a search on
        # "Arya and Bran Stark actors on growing up on the set".
        _why = (proj.meta or {}).get("auto_rejected_reasons") or {}
        _replaceable = {"subtitled_copy", "watermarked", "promo_overlay", "numeral_overlay",
                        "screen_recording", "sub_native_hd"}
        rejected = [s for s in proj.sources
                    if s.id in set((proj.meta or {}).get("auto_rejected_sources") or [])
                    and _why.get(s.id, "subtitled_copy") in _replaceable]
        fresh = [s for s in rejected if (s.title or "").strip()
                 and (s.title or "").strip().lower() not in tried_titles]
        if not fresh:
            if rnd == 0:
                log("5b/9 · backfill: no gate-rejected source to replace")
            terminal_reason = "no_untried_quality_rejections"
            break

        # A replacement indexed WITHOUT Face-ID is structurally unable to win the beats it was
        # fetched for: `faceid` carries w_face (0.30) of the score, so its shots start a third of a
        # point behind every incumbent that has cast data.  Check only after proving a replacement
        # is actually needed: an empty rejected set is already a conclusive completed search and
        # must not become a permanently retrying technical failure on Face-ID-less projects.
        if faceid_obj is None or not refs:
            log("5b/9 · backfill: WARNING — no Face-ID references available; replacements would "
                "be indexed without cast data and could not compete. Skipping.")
            return _finish("incomplete", "faceid_references_unavailable")

        # the rejected upload's own title IS the description of the footage we lost; strip the
        # channel furniture that would otherwise pull the search back to more re-uploads
        def _q(title: str) -> str:
            t = _re_b.sub(r"\s*[|\-–—]\s*(4k|hd|1080p|720p|full\s*scene|reaction|explained|"
                          r"subtitles?|eng(lish)?\s*sub\w*).*$", "", title, flags=_re_b.I)
            t = _re_b.sub(r"[\[(][^\])]*[\])]", " ", t)
            return _re_b.sub(r"\s+", " ", t).strip()[:90]

        queries = [q for q in (_q(s.title or "") for s in fresh) if len(q) > 8]
        for s in fresh:
            tried_titles.add((s.title or "").strip().lower())
        log(f"5b/9 · backfill round {rnd + 1}: {len(fresh)} gate-rejected source(s) → "
            f"searching for clean copies")

        have = {(getattr(s, "url", "") or "").strip() for s in proj.sources if getattr(s, "url", "")}
        try:
            cands = discover_sources(analysis, cfg, segments=segs, progress=None,
                                     extra_queries=queries) or []
        except Exception as e:                                   # noqa: BLE001
            log(f"5b/9 · backfill: discovery failed ({str(e)[:70]})")
            return _finish("incomplete", f"discovery_failed:{type(e).__name__}")
        new = [c for c in cands if (getattr(c, "url", "") or "").strip()
               and (getattr(c, "url", "") or "").strip() not in have][:max(4, max_sources)]
        audit["rounds"].append({"round": rnd + 1,
                                "rejected": [s.id for s in fresh],
                                "queries": queries,
                                "candidates": len(cands), "new": len(new)})
        if not new:
            log("5b/9 · backfill: no NEW candidate found — the pool keeps what it has")
            terminal_reason = "no_new_candidate"
            break

        source_snapshot = list(proj.sources)
        source_ids_before = {str(getattr(s, "id", "") or "") for s in source_snapshot}
        before = {s.id for s in source_snapshot if s.status == SOURCE_OK}
        wanted_urls = {(getattr(c, "url", "") or "").strip() for c in new
                       if (getattr(c, "url", "") or "").strip()}

        def _rollback_download_attempt() -> None:
            attempted = [s for s in proj.sources
                         if (getattr(s, "url", "") or "").strip() in wanted_urls]
            attempted_new_ids = {
                str(getattr(s, "id", "") or "") for s in attempted
                if str(getattr(s, "id", "") or "") not in source_ids_before
            }
            from .index import purge_source_index as _purge_backfill_index
            for sid in sorted(attempted_new_ids):
                if sid:
                    try:
                        _purge_backfill_index(proj, sid)
                    except Exception:                    # noqa: BLE001 — manifest rollback is primary
                        pass
            proj.sources = source_snapshot

        _pw = None
        if _index_overlap_on():
            try:
                _pw = _IndexPrewarmer(proj, cfg, references=refs, faceid=faceid_obj,
                                      roster=roster, log=log)
            except Exception:                                    # noqa: BLE001
                _pw = None
        _download_exc = None
        try:
            download_candidates(proj, new, cfg, policy=policy, limit=len(new), progress=None,
                                on_ready=(_pw.submit if _pw is not None else None))
        except Exception as e:                                   # noqa: BLE001
            _download_exc = e
        finally:
            if _pw is not None:
                _pw.close()
        if _download_exc is not None:
            _rollback_download_attempt()
            log(f"5b/9 · backfill: download failed ({str(_download_exc)[:70]})")
            return _finish("incomplete", f"download_failed:{type(_download_exc).__name__}")
        # The downloader reports ordinary per-source failures in the manifest instead of raising.
        # Treat those rows exactly like a thrown transport error: even one failed sibling means the
        # searched pool is incomplete, so a later Resume must retry rather than cache a false
        # "nothing usable exists" result.  Match by URL (the downloader's candidate id is private
        # and title-derived); every URL here was absent from ``have`` immediately before the call.
        outcomes = [s for s in proj.sources
                    if (getattr(s, "url", "") or "").strip() in wanted_urls]
        failed = [s for s in outcomes if getattr(s, "status", "") == SOURCE_FAILED]
        if failed:
            audit["rounds"][-1]["download_failures"] = [
                {"id": getattr(s, "id", ""), "url": getattr(s, "url", ""),
                 "error": str(getattr(s, "error", "") or "download_failed")[:240]}
                for s in failed
            ]
            # A partial batch is one incomplete transaction: a successful sibling has not yet
            # passed the backfill-only title/yield screens below, so retaining it here could let an
            # unscreened anthology/BTS source enter match.  Roll every attempted row back to the
            # pre-download manifest and purge prewarmed indexes; media bytes may remain as a safe
            # resumable download, but no URL enters ``have`` until the whole batch is conclusive.
            _rollback_download_attempt()
            detail = ",".join(str(getattr(s, "id", "") or "?") for s in failed[:4])
            log(f"5b/9 · backfill: {len(failed)} download(s) technically failed "
                f"({detail}) — leaving checkpoint retryable")
            return _finish("incomplete", f"download_failed_status:{len(failed)}")
        newly = [s for s in proj.sources if s.status == SOURCE_OK and s.id not in before]
        if not newly:
            # Policy exclusions and checksum duplicates are content-conclusive.  An absent or
            # unknown outcome is technical uncertainty and must never earn a completed checkpoint.
            if outcomes and all(getattr(s, "status", "") in (SOURCE_BLOCKED, "duplicate")
                                for s in outcomes):
                log(f"5b/9 · backfill: all candidates excluded by policy/duplicate "
                    f"under policy={policy}")
                terminal_reason = "download_conclusive_policy_or_duplicate"
                break
            log(f"5b/9 · backfill: downloader produced no usable or conclusive outcome "
                f"under policy={policy} — leaving checkpoint retryable")
            return _finish("incomplete", "download_missing_or_unknown_outcome")
        try:
            _index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster,
                       force=False, progress=None)
        except Exception as e:                                   # noqa: BLE001
            # These rows have not passed the title/yield screen, and retaining their URLs would
            # make the next Resume filter them out of discovery as already present.  That turns a
            # technical index failure into a conclusive ``no_new_candidate`` checkpoint.  Roll the
            # whole unscreened batch (and any prewarmed/partial indexes) back so the same clean-copy
            # candidates remain genuinely retryable.
            _rollback_download_attempt()
            log(f"5b/9 · backfill: indexing failed ({str(e)[:70]})")
            return _finish("incomplete", f"indexing_failed:{type(e).__name__}")
        # A replacement only counts once it can actually AIR. Clearing the source-level gates is not
        # enough: a fetched copy of the exact scene we lost came back as another screener with
        # "FOR INTERNAL VIEWING ONLY" burned in, cleared every source gate, and then lost all 11 of
        # its shots to the shot-level text gate. Measure the yield and say so.
        # A usable yield proves a source CAN air, never that it SHOULD. Measured on job benjen_v2,
        # where this pass admitted 26 sources into a BENJEN STARK essay and among them:
        #   "Cersei and Jaime Lannister - All Scenes Part 3/8"   162 usable shots
        #   "Cersei and Jaime Lannister - All Scenes Part 4/8"   150 usable shots
        #   "S07E07 - Behind the Scene - Dragon Pit Meeting"     182 usable shots  (BTS!)
        #   "A tale of Benjen Stark - A Game of Thrones fanfiction"
        # The Lannister compilations then fed the breakout miner and one of them aired a Season-1
        # Cersei/Ned conversation inside the essay. The pass had searched with the REJECTED source's
        # own title ("Cersei and Jaime Lannister - The Sept Scene"), which is only a good query when
        # the rejected upload was genuinely wanted — that one was already marginal, so asking for a
        # cleaner copy of it bought a 13-minute compilation of a different story.
        #
        # So apply discovery's OWN title rules to what comes back. They already encode "not scene
        # footage": behind-the-scenes, fanfiction, reaction, talking-head, non-show.
        from .discover import _REJECT_TITLE, _NONSHOW_TITLE, _REACTION_TITLE
        import re as _re_bf

        def _title_ok(title: str) -> str:
            t = title or ""
            if _REJECT_TITLE.search(t):
                return "making-of / talking-head / promo title"
            if _NONSHOW_TITLE.search(t):
                return "non-show title"
            if _REACTION_TITLE.search(t):
                return "reaction title"
            # a multi-part "All Scenes Part n/N" character compilation is an anthology of a DIFFERENT
            # story; it passes every per-shot gate and hands the breakout miner hours of off-topic
            # dialogue. The original discovery rules do not cover it because it is genuine footage.
            if _re_bf.search(r"\ball\s+scenes?\b|\bpart\s*\d+\s*/\s*\d+", t, _re_bf.I):
                return "multi-part character compilation"
            if _re_bf.search(r"\bfan\s*fiction\b|\bfanfic", t, _re_bf.I):
                return "fanfiction"
            # _REJECT_TITLE covers "how it was filmed" and stunt rehearsals but not the plain
            # "Behind the Scene(s)" phrasing — measured: "S07E07 - Behind the Scene - Dragon Pit
            # Meeting" was admitted with 182 usable shots and aired on a beat.
            if _re_bf.search(r"behind[\s\-]+the[\s\-]+scenes?\b|\bb-?roll\b|\bbloopers?\b|"
                             r"\bgag\s*reel\b", t, _re_bf.I):
                return "behind-the-scenes"
            return ""

        from .quality_contract import native_video_ok as _backfill_native_ok
        from .quality_contract import probe_native_video_info as _probe_backfill_native
        kept, dead = [], []
        dead_reasons: dict[str, str] = {}
        for s in newly:
            _why_bad = _title_ok(s.title or "")
            if _why_bad:
                dead.append((s, 0, 0))
                dead_reasons[s.id] = _why_bad
                log(f"5b/9 · backfill: rejected {(s.title or s.id)[:44]!r} — {_why_bad}")
                continue
            # `usable_shot_yield` measures frame/content gates, not native raster. A newly fetched
            # 360p copy could therefore be counted as "admitted" in the last configured round even
            # though the next match pass and final build must reject it. Probe the actual bytes here
            # and make the audit honest in the same invocation.
            _native_path = str(getattr(s, "local_path", "") or "")
            _native_info = dict(_probe_backfill_native(_native_path) or {})
            if not _native_info.get("width") or not _native_info.get("height"):
                # An unavailable decoder/probe is technical uncertainty, not a conclusive bad
                # copy. Roll back the unscreened batch so this URL remains retryable on Resume.
                _rollback_download_attempt()
                log(f"5b/9 · backfill: native-resolution probe unavailable for "
                    f"{(s.title or s.id)[:44]!r} — leaving checkpoint retryable")
                return _finish("incomplete", f"native_probe_unavailable:{s.id}")
            if not _backfill_native_ok(_native_info):
                try:
                    _nw = int(_native_info.get("width") or 0)
                    _nh = int(_native_info.get("height") or 0)
                except (TypeError, ValueError):
                    _nw = _nh = 0
                _native_reason = (f"sub-native-HD actual bytes {_nw}x{_nh}"
                                  if _nw and _nh else "native resolution unprobeable")
                dead.append((s, 0, 0))
                dead_reasons[s.id] = _native_reason
                log(f"5b/9 · backfill: rejected {(s.title or s.id)[:44]!r} — "
                    f"{_native_reason}; publication requires 1280x720")
                continue
            try:
                ok, tot = _M.usable_shot_yield(proj, s.id, cfg)
            except Exception as e:                               # noqa: BLE001
                # This is technical uncertainty, not evidence of one airable shot.  Roll the whole
                # unscreened batch (including its indexes) so Resume can retry the same candidates;
                # the completed checkpoint must never cache a fabricated 1/1 yield.
                _rollback_download_attempt()
                log(f"5b/9 · backfill: shot-yield measurement failed "
                    f"({type(e).__name__}: {str(e)[:60]})")
                return _finish("incomplete", f"yield_measurement_failed:{type(e).__name__}")
            if not ok:
                dead_reasons[s.id] = "no usable indexed shots"
            (kept if ok else dead).append((s, ok, tot))
        admitted_total += len(kept)
        audit["rounds"][-1]["downloaded"] = [
            {"id": s.id, "title": (s.title or "")[:70], "usable_shots": ok, "shots": tot,
             "rejection_reason": dead_reasons.get(s.id, "")}
            for s, ok, tot in kept + dead]
        for s, ok, tot in dead:
            proj.meta.setdefault("banned_sources", [])
            if s.id not in proj.meta["banned_sources"]:
                proj.meta["banned_sources"].append(s.id)
            _dead_reason = dead_reasons.get(s.id, "burned text / graphics / too dark")
            log(f"5b/9 · backfill: {(s.title or s.id)[:44]!r} has 0 usable shots of {tot} "
                f"({_dead_reason}) — banned, not a replacement")
        if kept:
            log(f"5b/9 · backfill: +{len(kept)} usable source(s) indexed — "
                + ", ".join(f"{(s.title or s.id)[:36]} ({ok}/{tot} shots)" for s, ok, tot in kept[:3]))
        elif dead:
            log("5b/9 · backfill: every replacement this round was unusable — pool unchanged")

    return _finish("complete", terminal_reason)


def _recover_unresolved_beats(proj, segs, analysis, cfg, eng, *, faceid_obj, refs, roster,
                              policy, log, only_indices=None,
                              audit_filename: str = "recovery_audit.json",
                              audit_request_id: str = "", quote_pool_cache=None) -> int:
    """R4-5 BOUNDED AUTONOMOUS RECOVERY. For EXACT beats still unresolved after match + verify, run
    ONE bounded round of targeted rediscovery → download → index → rematch → (re-cut) → reverify
    BEFORE the render can fall back to a deterministic still / editorial hold / release-block.

    Invariants:
      • Already-resolved beats are PRESERVED byte-for-byte (snapshot + restore + final re-cut) —
        recovery can only IMPROVE unresolved beats, never disturb a good one.
      • A recovered pick is accepted ONLY if it VERIFIES. The matcher ranks candidates by relevance
        and the verifier is the gate, so an irrelevant HD clip that fails verify is never installed —
        an EXACT (even low-res) clip that verifies is preferred over an irrelevant HD one.
      • Fail-closed: any error, no-LLM, or empty discovery leaves the beats unresolved for the
        downstream still/hold/block to handle honestly. Cheap no-op when there is nothing to recover.
      • ``only_indices`` lets the final semantic contract request a bounded retry for ONLY its
        blockers. Match may score the full project internally, but snapshot reconciliation restores
        every non-target selection byte-for-byte, so the retry cannot churn already-approved beats.
      • Every attempt (queries, candidates, new sources, per-beat outcome) is audited to
        ``audit_filename`` (the contract retry uses a separate file, preserving round one).
    Returns the number of beats recovered with real verified footage."""
    import os as _os
    import copy as _copy
    import dataclasses as _dc
    from . import policy as _policy

    max_beats = int(_os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS", "8") or 8)
    max_sources = int(_os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY_MAX_SOURCES", "4") or 4)

    seg_by_idx = {s.index: s for s in segs}
    sel_by_idx = {s.segment_index: s for s in proj.selections}
    _scope = set(only_indices) if only_indices is not None else None
    # ``only_indices`` comes from the strict publication contract, whose blocker vocabulary is
    # deliberately wider than the legacy rejected-footage predicate below.  Re-filtering that
    # scope through ``_beat_is_unresolved`` silently discarded real quote-window failures and
    # strict/contextual evidence failures (e.g. the dedicated "Chaos is a ladder" source was on
    # disk, but its wrong selected window never entered recovery).  The caller has already proved
    # every scoped index is blocked, so preserve the whole scope.  The legacy call still uses its
    # original predicate.
    unresolved = [s.index for s in segs
                  if (_scope is not None and s.index in _scope)
                  or (_scope is None and _beat_is_unresolved(sel_by_idx.get(s.index), s, _policy))]
    audit = {
        "schema_version": _SEMANTIC_RECOVERY_PAGE_SCHEMA,
        "request_id": str(audit_request_id or ""),
        "requested_scope": sorted(_scope) if _scope is not None else [],
        "page_scope": [],
        "page_completed": False,
        "page_error": "",
        "unresolved_before": list(unresolved),
        "caps": {"beats": max_beats, "sources": max_sources},
        "attempts": [], "new_sources": [], "download_failures": [], "index_results": [],
        "recovered": [], "still_unresolved": [],
        "current_pool_rematch": {},
    }

    def _finish_page(result: int, *, completed: bool, error: str = "") -> int:
        current_error = ""
        current = audit.get("current_pool_rematch")
        if isinstance(current, dict):
            current_error = str(current.get("error") or "").strip()
        final_error = str(error or current_error).strip()
        audit["page_error"] = final_error
        audit["page_completed"] = bool(completed and not final_error)
        _write_recovery_audit(proj, audit, audit_filename)
        return result

    if _os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY", "1").strip() in ("0", "false", "no"):
        return _finish_page(0, completed=False, error="bounded recovery is disabled")
    if _scope is not None and set(unresolved) != _scope:
        missing = sorted(_scope - set(unresolved))
        return _finish_page(
            0, completed=False,
            error=f"requested semantic scope has no matching segment(s): {missing}")
    if not unresolved:
        return _finish_page(0, completed=(_scope is None))

    unresolved_before = list(unresolved)
    current_pool_recovered: set[int] = set()
    _tried = {int(i) for i in (proj.meta.get("recovery_attempted") or [])
              if str(i).lstrip("-").isdigit()}
    # The per-round cap applies to local rematch/verification as well as network acquisition.
    # Otherwise a 29-beat strict scope silently turns an 8-beat bounded recovery into 29 fresh
    # vision decisions and two full-project cuts.  Deferred beats stay explicit in the audit and
    # rotate into a later Resume; they are never mislabeled as attempted or recovered.
    if _scope is not None:
        # Semantic rotation is generation-scoped by `_retry_selection_relevance`: a changed source/
        # selection fingerprint starts a new finite walk, and an unchanged Resume passes only the
        # prior tail. Do not let the legacy project's process-wide `recovery_attempted` history
        # reorder a fresh semantic generation (otherwise [0,1,2] with cap=2 can start [2,0]).
        unresolved = recovery_pick(unresolved, seg_by_idx, _policy, set(), max_beats)
    audit["page_scope"] = list(unresolved)
    audit["deferred"] = [i for i in unresolved_before if i not in set(unresolved)]
    audit["deferred_retriable"] = [
        i for i in audit["deferred"]
        if recovery_query(seg_by_idx.get(i))]
    if not unresolved:
        audit["still_unresolved"] = list(unresolved_before)
        # Every scoped beat was explicitly classified as non-retrievable (no query material).
        # That is a conclusive page result, not a helper crash; downstream still/abstract handling
        # may proceed for those beats while capped retriable beats remain in deferred_retriable.
        return _finish_page(0, completed=True)

    # A source acquired by the earlier self-heal pass is downloaded and indexed immediately, but
    # only the beat that triggered that acquisition searches it.  The final recovery used to ask
    # discovery for a NEW URL first and returned early when the right URL was already present.  On
    # the 101-beat run this left the dedicated Littlefinger-death and Chaos-monologue uploads idle
    # in the indexed pool while the affected beats stayed on known-wrong windows.  Before spending
    # network budget, rematch the strict blockers against the pool that ACTUALLY exists now.  Match
    # may rebuild every selection internally for its diversity constraints; snapshot reconciliation
    # ensures only contract-cleared scoped beats survive, and a final cut re-establishes clip lineage.
    if _scope is not None:
        _page_scope = list(unresolved)
        current_error = ""
        from . import verify as _verify_pool
        from . import relevance_contract as _rel_pool

        def _run_scoped_verify(_indices):
            import os as _os_pool
            _pool_workers_unset = "VIDLORE_CLIPSTUDIO_VERIFY_WORKERS" not in _os_pool.environ
            if _pool_workers_unset:
                _os_pool.environ["VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"] = "4"
            try:
                return _verify_pool.verify_and_repair(
                    proj, segs, cfg, eng, only_indices=set(_indices), progress=None,
                    materialize_promotions=False, persist_project=False,
                    strict_pool_recovery=True)
            finally:
                if _pool_workers_unset:
                    _os_pool.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", None)

        # REAL-QUOTE WINDOW RUNG.  Whole-pool ASR has already proved the phrase's source/time;
        # consume that exact evidence before a global rematch lets diversity choose a neighbouring
        # instant again.  This transaction is scoped and contract-positive just like the broad rung.
        quote_snapshot = {s.segment_index: _copy.deepcopy(s) for s in proj.selections}
        quote_audit = {"attempted": [], "recovered": [], "still_unresolved": [],
                       "error": "", "beats": []}
        quote_recovered: set[int] = set()
        try:
            quote_selections, quote_audit = _quote_window_recovery_selections(
                proj, segs, cfg, set(unresolved), quote_pool_cache=quote_pool_cache)
            if quote_selections:
                proj.selections = [quote_selections.get(i, quote_snapshot[i])
                                   for i in sorted(quote_snapshot)]
                proj.selections.extend(quote_selections[i]
                                       for i in sorted(set(quote_selections) - set(quote_snapshot)))
                quote_summary = _run_scoped_verify(set(quote_selections))
                quote_error = _verifier_summary_error(quote_summary)
                if quote_error:
                    raise RuntimeError(f"quote_window_{quote_error}")
                quote_contract = _rel_pool.evaluate_selection_relevance(
                    proj, segs, cfg=cfg, quote_pool_cache=quote_pool_cache)
                quote_blocked = {int(e.get("segment_index", -1))
                                 for e in (quote_contract.get("blockers") or [])}
                quote_recovered = set(quote_selections) - quote_blocked
                by_block = {int(e.get("segment_index", -1)): e
                            for e in (quote_contract.get("blockers") or [])}
                for row in quote_audit.get("beats") or []:
                    idx = int(row.get("segment_index", -1))
                    if idx in quote_selections:
                        row["strict_result"] = ("recovered" if idx in quote_recovered else "blocked")
                        row["strict_reasons"] = list((by_block.get(idx) or {}).get("reasons") or [])
            quote_rematched = {s.segment_index: s for s in proj.selections}
            if not _commit_scoped_recovery(
                    proj, cfg, quote_snapshot, quote_rematched, quote_recovered, log=log):
                raise RuntimeError("quote_window_scoped_cut_or_lineage_commit_failed")
        except Exception as e:                           # noqa: BLE001 — restore and fail closed
            current_error = f"{type(e).__name__}: {e}"
            quote_audit["error"] = current_error
            _commit_scoped_recovery(
                proj, cfg, quote_snapshot,
                {s.segment_index: s for s in proj.selections}, set(), log=log)
            quote_recovered.clear()
        quote_audit["recovered"] = sorted(quote_recovered)
        quote_audit["still_unresolved"] = [i for i in _page_scope if i not in quote_recovered]
        audit["current_pool_quote_windows"] = quote_audit
        current_pool_recovered.update(quote_recovered)
        unresolved = [i for i in unresolved if i not in quote_recovered]
        if quote_recovered:
            log(f"recovery: exact whole-pool quote windows recovered {len(quote_recovered)} "
                f"strict blocker(s) {sorted(quote_recovered)} before global rematch/acquisition")

        broad_attempted = list(unresolved)
        if unresolved and not current_error:
            current_snapshot = {s.segment_index: _copy.deepcopy(s) for s in proj.selections}
            broad_recovered: set[int] = set()
            try:
                match_segments(proj, segs, cfg, analysis=analysis, progress=None)
                # The global matcher must see every beat to preserve its diversity constraints, but
                # this transaction can commit only ``unresolved``.  Restore every unscoped snapshot
                # row BEFORE scoped verification so verify's reuse ledger describes the prospective
                # committed project.  Otherwise temporary matcher churn on beats that are about to
                # be rolled back can consume a shot's reuse cap and hide a strict-passing recovery.
                # This is in-memory reconciliation only; `_commit_scoped_recovery` remains the sole
                # metadata/clip commit authority.
                _scope_set = set(unresolved)
                _global_rematch = {s.segment_index: s for s in proj.selections}
                proj.selections = [
                    (_global_rematch[idx] if idx in _scope_set and idx in _global_rematch
                     else _copy.deepcopy(current_snapshot[idx]))
                    for idx in sorted(current_snapshot)
                ]
                proj.selections.extend(
                    _global_rematch[idx]
                    for idx in sorted(_scope_set - set(current_snapshot))
                    if idx in _global_rematch)
                _pool_verify_result = _run_scoped_verify(set(unresolved))
                _pool_verify_error = _verifier_summary_error(_pool_verify_result)
                if _pool_verify_error:
                    raise RuntimeError(f"current_pool_{_pool_verify_error}")
                strict_after = _rel_pool.evaluate_selection_relevance(
                    proj, segs, cfg=cfg, quote_pool_cache=quote_pool_cache)
                blocked_after = {int(e.get("segment_index", -1))
                                 for e in (strict_after.get("blockers") or [])}
                broad_recovered = set(unresolved) - blocked_after
            except Exception as e:                       # noqa: BLE001 — restore and fail closed
                current_error = f"{type(e).__name__}: {e}"
                log(f"recovery: current-pool scoped rematch errored ({current_error[:120]}) — "
                    "no broad-rematch selection accepted")
                broad_recovered.clear()
            rematched = {s.segment_index: s for s in proj.selections}
            if not _commit_scoped_recovery(
                    proj, cfg, current_snapshot, rematched, broad_recovered, log=log):
                current_error = current_error or "scoped_cut_or_lineage_commit_failed"
                broad_recovered.clear()
            current_pool_recovered.update(broad_recovered)

        audit["current_pool_rematch"] = {
            # Compatibility/conclusive-page invariant: this remains the exact page scope even when
            # its quote subset was recovered by the narrower rung before the broad rematch.
            "attempted": list(_page_scope),
            "broad_attempted": broad_attempted,
            "recovered": sorted(current_pool_recovered),
            "still_unresolved": [i for i in unresolved_before
                                 if i not in current_pool_recovered],
            "deferred": list(audit["deferred"]),
            "error": current_error,
        }
        if current_error:
            # Matching/verifier infrastructure did not complete, so this is not an exhausted
            # content page. The scoped commit above has already restored exploratory metadata.
            audit["recovered"] = sorted(current_pool_recovered)
            return _finish_page(len(current_pool_recovered), completed=False, error=current_error)
        if current_pool_recovered:
            log(f"recovery: current indexed pool recovered {len(current_pool_recovered)} "
                f"strict blocker(s) {sorted(current_pool_recovered)} before rediscovery")
        unresolved = [i for i in unresolved if i not in current_pool_recovered]
        if not unresolved:
            audit["recovered"] = sorted(current_pool_recovered)
            audit["still_unresolved"] = [i for i in unresolved_before
                                         if i not in current_pool_recovered]
            return _finish_page(len(current_pool_recovered), completed=True)
    # GATE-VULNERABLE beats first when the cap trims. Evidence across four blocked renders:
    # every blocker was a CHARACTER beat (their stills need a verified face — the hardest
    # coverage) plus one capped abstract beat; abstract/filler rejected beats take a lenient
    # CLIP-ranked still, and exact beats keep the deterministic-still fallback downstream. So:
    # character first, then filler/abstract, exact last; script order within each class. Beats
    # with NO effective query material across scene_query/entity/expected_visual/narration cannot
    # be rediscovered and must not burn a slot; stills cover those truly empty authored beats.
    # ROTATION. The rank below is deterministic, so every round re-selected the SAME head of the
    # same list. MEASURED on job 409e284b60: 32 beats unresolved, cap 8; round 1 took
    # [90,110,166,76,89,91,12,13] and round 2 took [90,110,76,79,89,12,13,19] — six of eight were
    # re-attempts, and round 2 reported `candidates_found 21, new_candidates 0` because re-issuing
    # a query YouTube already answered returns the sources we already downloaded. Meanwhile beats
    # 22/60/75/94/115/122/140/144/168 — "Arya kills the Night King", "the Children make the Night
    # King", "the wight torso in the Dragonpit" — were never searched for even once.
    # A beat already tried and not recovered therefore goes LAST within its class rather than
    # being dropped: a later round has a bigger pool, so it may still be worth a retry, but only
    # after every beat that has had no turn at all. Same cap, same cost, strictly wider coverage.
    if _scope is None:
        unresolved = recovery_pick(unresolved, seg_by_idx, _policy, _tried, max_beats)
    # LOOK-MISS ACQUISITION — "watch the chalice" beats whose target nothing on disk shows are
    # KEPT (usable footage, never gambled) but flagged; when the recovery round has spare slots,
    # spend up to 3 on fetching footage that actually shows the named thing ("the footage exists
    # on the internet" is the owner's standing complaint). Snapshot semantics protect the kept
    # pick: a look-miss beat only changes if rediscovery positively resolves it at the strict
    # bar; otherwise the snapshot is restored. Env: VIDLORE_CLIPSTUDIO_LOOK_RECOVERY=0 disables.
    _look_aug: dict[int, str] = {}
    if _os.environ.get("VIDLORE_CLIPSTUDIO_LOOK_RECOVERY", "1").strip() \
            not in ("0", "false", "no"):
        _lm = [s.segment_index for s in proj.selections
               if "look_target_missing" in (getattr(s, "flag_reasons", None) or [])
               and s.segment_index not in unresolved
               and (_scope is None or s.segment_index in _scope)
               and s.segment_index in seg_by_idx]
        _room = max(0, min(max_beats - len(unresolved), 3))
        for i in _lm[:_room]:
            _tgt_r = _policy.deictic_target(seg_by_idx[i])
            if _tgt_r:
                unresolved.append(i)
                _look_aug[i] = _tgt_r
        if _look_aug:
            log(f"recovery: +{len(_look_aug)} look-miss beat(s) added for targeted "
                f"acquisition {sorted(_look_aug)} (targets: "
                f"{', '.join(repr(t) for t in _look_aug.values())})")
    if not unresolved:
        return _finish_page(0, completed=True)
    #  Recorded BEFORE the attempt: a round that dies mid-way must still count as this beat's
    #  turn, otherwise a beat whose search reliably crashes monopolises the cap forever.
    _fresh = [i for i in unresolved if i not in _tried]
    # This process-wide marker belongs to the legacy recovery rotation. Semantic pages have their
    # own nonce-bound completion marker and must not record exhaustion before the page succeeds.
    if _scope is None:
        proj.meta["recovery_attempted"] = sorted(_tried | set(unresolved))
    audit["attempted_before"] = sorted(_tried)
    audit["first_attempt_this_round"] = _fresh
    log(f"recovery: {len(unresolved)} unresolved beat(s) → bounded rediscovery {unresolved} "
        f"({len(_fresh)} never searched before)")

    # A round is search → download → index → rematch → scoped reverify, and every one of those was
    # passed progress=None. Measured on job 218acdfe10: fifteen rounds, 216 minutes, and 841-second
    # stretches with NOT ONE line of log — indistinguishable from a wedged render, which is exactly
    # what it was reported as. The rounds are buying real footage (16 beats recovered), so the
    # answer is not to cut them; it is to stop making them invisible, and to make the next
    # measurement of WHERE those 14 minutes go a matter of reading the log instead of guessing.
    import time as _time_r
    _t_round = _time_r.time()
    _stage_t: dict = {}

    def _rlog(stage: str):
        _stage_t[stage] = _time_r.time()
        return lambda m: log(f"recovery/{stage}: {m}")

    def _rdone(stage: str) -> None:
        log(f"recovery/{stage}: done in {_time_r.time() - _stage_t.get(stage, _t_round):.0f}s "
            f"({_time_r.time() - _t_round:.0f}s into this round)")

    # Snapshot EVERY current selection; the final selection list is snapshot ∪ (recovered new picks).
    snapshot = {s.segment_index: _copy.deepcopy(s) for s in proj.selections}
    # Discovery's query builder consumes ``scene_query``/entity/keywords, while recovery eligibility
    # intentionally also admits expected_visual- and narration-only beats. Give discovery a shallow
    # recovery-only copy whose scene_query is the exact effective query we audit and title-rank.
    # Never mutate the authored/project segments just to adapt one recovery attempt.
    unresolved_segs = []
    for _original_seg in (seg_by_idx[i] for i in unresolved if i in seg_by_idx):
        _recovery_seg = _copy.copy(_original_seg)
        _effective_query = recovery_query(_original_seg)
        if not (getattr(_recovery_seg, "scene_query", "") or "").strip():
            _recovery_seg.scene_query = _effective_query
        # look-miss beats search WITH their named target
        # ("... Joffrey drinks chalice close-up") on the same disposable copy.
        if _original_seg.index in _look_aug:
            _recovery_seg.scene_query = (
                f"{(_recovery_seg.scene_query or '').strip()} "
                f"{_look_aug[_original_seg.index]} close-up").strip()
        unresolved_segs.append(_recovery_seg)
    _effective_queries = [
        (getattr(s, "scene_query", "") or "").strip() for s in unresolved_segs]
    recovered: set = set()
    round_error = ""

    try:
        from .discover import discover_sources, _STOPQ as _R_STOP
        from .download import download_candidates
        # discover_target must be WIDER than max_sources here. The global selection pass (per-channel
        # caps, anchor/entity coverage) re-ranks everything, so with target==4 the four survivors are
        # the same top anchor uploads the project already has — measured: rediscovery for five
        # unresolved beats found 11 candidates and 0 new, twice in a row, while the per-beat queries'
        # actual results (the scene uploads the beats need) were cut from the selection. Widen the
        # target so per-beat results survive, then PREFER the new urls whose title matches an
        # unresolved beat's scene tokens (>=2, same rule as anchor/key-scene coverage) — those are
        # the uploads the recovery exists to fetch.
        cfg_r = _dc.replace(cfg, discover_target=max(12, 3 * max_sources))
        have_urls = {(getattr(s, "url", "") or "").strip() for s in proj.sources if getattr(s, "url", "")}
        cands = discover_sources(
            analysis, cfg_r, segments=unresolved_segs, progress=_rlog("search"),
            extra_queries=_effective_queries, required_queries=_effective_queries) or []
        _rdone("search")
        import re as _re_r
        _r_mv = {w for w in _re_r.findall(r"\w+", (getattr(analysis, "movie_title", "") or "").lower())
                 if len(w) > 2}

        def _beat_hits(c) -> int:
            tw = set(_re_r.findall(r"\w+", (getattr(c, "title", "") or "").lower()))
            best = 0
            for _query in _effective_queries:
                toks = {w for w in _re_r.findall(
                            r"\w+", _query.lower())
                        if len(w) > 2 and w not in _R_STOP and w not in _r_mv}
                hits = sum(1 for w in toks
                           if any(t == w or (t.startswith(w) and len(t) - len(w) <= 2)
                                  for t in tw))
                best = max(best, hits)
            return best
        _new_all = [c for c in cands
                    if (getattr(c, "url", "") or "").strip() and
                    (getattr(c, "url", "") or "").strip() not in have_urls]
        _new_all.sort(key=_beat_hits, reverse=True)
        new_cands = ([c for c in _new_all if _beat_hits(c) >= 2]
                     or _new_all)[:max_sources]
        audit["attempts"].append({
            "queries": list(_effective_queries),
            "candidates_found": len(cands), "new_candidates": len(new_cands)})
        if not new_cands:
            log("recovery: targeted rediscovery found no NEW source — leaving beats for downstream fallback")
            audit["recovered"] = sorted(current_pool_recovered)
            audit["still_unresolved"] = [i for i in unresolved_before
                                         if i not in current_pool_recovered]
            return _finish_page(len(current_pool_recovered), completed=True)

        # Source acquisition is part of this page's transaction too. A technical failure must not
        # leave a failed/zero-index row whose URL suppresses the same candidate on Resume.
        _sources_before_acquire = _copy.deepcopy(proj.sources)
        _source_ids_before = {s.id for s in _sources_before_acquire}

        def _rollback_acquired_sources(source_ids) -> None:
            from .index import purge_source_index as _purge_recovery_index
            for _sid in source_ids:
                try:
                    _purge_recovery_index(proj, _sid)
                except Exception:                       # noqa: BLE001 — manifest rollback is primary
                    pass
            proj.sources = list(_sources_before_acquire)
            proj.save()

        before_ok = {s.id for s in proj.sources if s.status == SOURCE_OK}
        try:
            download_candidates(
                proj, new_cands, cfg_r, policy=policy, limit=len(new_cands),
                progress=_rlog("download"))
        except Exception:
            _rollback_acquired_sources({s.id for s in proj.sources
                                        if s.id not in _source_ids_before})
            raise
        _rdone("download")
        from .models import SOURCE_FAILED as _SOURCE_FAILED_RECOVERY
        _attempted_new = [s for s in proj.sources if s.id not in _source_ids_before]
        newly = [s for s in proj.sources if s.status == SOURCE_OK and s.id not in before_ok]
        audit["new_sources"] = [{"id": s.id, "title": (getattr(s, "title", "") or "")[:70],
                                 "height": getattr(s, "height", 0)} for s in newly]
        _technical_downloads = [s for s in _attempted_new
                                if s.status == _SOURCE_FAILED_RECOVERY
                                and str(getattr(s, "error", "") or "").strip()]
        if _technical_downloads:
            audit["download_failures"] = [
                {"id": s.id, "url": s.url, "error": str(s.error)[:240]}
                for s in _technical_downloads]
            _rollback_acquired_sources({s.id for s in _attempted_new})
            raise RuntimeError(
                f"targeted downloads were technically inconclusive for "
                f"{len(_technical_downloads)} candidate(s)")
        if not newly:
            log(f"recovery: no NEW source downloaded under policy={policy} — leaving beats unresolved")
            audit["recovered"] = sorted(current_pool_recovered)
            audit["still_unresolved"] = [i for i in unresolved_before
                                         if i not in current_pool_recovered]
            return _finish_page(len(current_pool_recovered), completed=True)

        # Index the new sources, rematch, then reverify directly from the selected SOURCE windows.
        # Never cut the exploratory full-project rematch: deterministic seg_NNN filenames would
        # overwrite clips belonging to the snapshot before we know which scoped picks are valid.
        # proj.selections is fully rebuilt here;
        # it is reconciled against the snapshot below so only recovered beats survive the change.
        # The re-verify is restricted to the recovered beats ONLY (only_indices) — re-verifying all
        # 229 beats here just to re-check ~8 recovered ones cost hours; every other beat is restored
        # from the snapshot anyway, so its verdict does not matter.
        try:
            _indexed = index_all(proj, cfg_r, references=refs, faceid=faceid_obj, roster=roster,
                                 force=False, progress=_rlog("index"))
        except Exception:
            _rollback_acquired_sources({s.id for s in _attempted_new})
            raise
        _indexed = _indexed if isinstance(_indexed, dict) else {}
        audit["index_results"] = [
            {"id": s.id, "shots": len(_indexed.get(s.id) or [])} for s in newly]
        _unindexed_new = [s for s in newly if not _indexed.get(s.id)]
        if _unindexed_new:
            _failed_ids = {s.id for s in _attempted_new}
            _rollback_acquired_sources(_failed_ids)
            raise RuntimeError(
                "targeted indexing produced no searchable shots for new source(s): "
                + ", ".join(sorted(s.id for s in _unindexed_new)))
        _rdone("index")
        _m_log = _rlog("rematch")
        match_segments(proj, segs, cfg_r, analysis=analysis, progress=_m_log)
        _rdone("rematch")
        from . import verify as _verify_r
        # same scoped 4-worker prefetch as the main verify stage (set + restore, never leaked)
        import os as _os_vwr
        _vwr_unset = "VIDLORE_CLIPSTUDIO_VERIFY_WORKERS" not in _os_vwr.environ
        if _vwr_unset:
            _os_vwr.environ["VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"] = "4"
        try:
            _new_verify_result = _verify_r.verify_and_repair(
                proj, segs, cfg, eng, only_indices=set(unresolved),
                progress=_rlog("reverify"), materialize_promotions=False,
                persist_project=False,
                strict_pool_recovery=True)
        finally:
            if _vwr_unset:
                _os_vwr.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", None)
        _rdone("reverify")
        _new_verify_error = _verifier_summary_error(_new_verify_result)
        if _new_verify_error:
            # SAME DISTINCTION AS e1e5802, AT THE OTHER SITE THAT NEEDED IT.
            #
            # A batch with an error is not a content verdict — right, and unchanged for anything
            # that would AUTHORIZE something. But it is not automatically a broken system either,
            # and treating it as one killed job 229233891e on `new_source_verifier_errored:1`:
            # exactly one freshly acquired source could not be judged, while the rest of the same
            # batch was judged normally. That is a fact about that source's frames.
            #
            # So an outage still raises, and it is decided by evidence from THIS batch: if enough
            # of it produced real verdicts, the verifier plainly worked. The unjudged source
            # authorizes nothing — its beat stays unresolved and the publication contract keeps
            # blocking it — the page simply stops being fatal.
            _errored = _new_verify_result.get("errored") \
                if isinstance(_new_verify_result, dict) else None
            _judged_ok = _new_verify_result.get("verified") \
                if isinstance(_new_verify_result, dict) else None
            if not (isinstance(_errored, int) and not isinstance(_errored, bool)
                    and isinstance(_judged_ok, int) and not isinstance(_judged_ok, bool)
                    and _judged_ok >= max(3, _errored * 3)):
                raise RuntimeError(f"new_source_{_new_verify_error}")
            log(f"semantic-recovery: {_errored} newly acquired source(s) could not be judged while "
                f"{_judged_ok} were judged normally in the SAME batch — their frames are treated as "
                f"unjudgeable, not the verifier as broken; those beats stay unresolved")

        new_sel_by_idx = {s.segment_index: s for s in proj.selections}
        if _scope is not None:
            # Acceptance in the semantic-recovery path is the SAME strict contract that invoked
            # it, not the weaker legacy verifier predicate.  This keeps a visually plausible but
            # dialogue-wrong quote window from being recorded/retained as recovered.
            from . import relevance_contract as _rel_new
            strict_new = _rel_new.evaluate_selection_relevance(proj, segs, cfg=cfg)
            strict_blocked = {int(e.get("segment_index", -1))
                              for e in (strict_new.get("blockers") or [])}
            recovered.update(i for i in unresolved if i not in strict_blocked)
        else:
            for i in unresolved:
                if not _beat_is_unresolved(new_sel_by_idx.get(i), seg_by_idx.get(i), _policy):
                    recovered.add(i)
    except Exception as e:
        round_error = f"{type(e).__name__}: {e}"
        log(f"recovery: bounded round errored ({type(e).__name__}: {e}) — leaving beats unresolved")

    # Reconcile and CUT only contract-cleared scoped picks. The transactional helper restores both
    # selections and prior clip bytes on any partial cut; arbitrary rematched indices never survive.
    new_sel_by_idx = {s.segment_index: s for s in proj.selections}
    if not _commit_scoped_recovery(proj, cfg, snapshot, new_sel_by_idx, recovered, log=log):
        recovered.clear()
        round_error = round_error or "scoped_cut_or_lineage_commit_failed"

    all_recovered = current_pool_recovered | recovered
    audit["recovered"] = sorted(all_recovered)
    audit["still_unresolved"] = [i for i in unresolved_before if i not in all_recovered]
    _finish_page(len(all_recovered), completed=not bool(round_error), error=round_error)
    if all_recovered:
        log(f"recovery: ✓ recovered {len(all_recovered)} beat(s) with real verified footage "
            f"{sorted(all_recovered)}")
    if audit["still_unresolved"]:
        log(f"recovery: {len(audit['still_unresolved'])} beat(s) still unresolved → deterministic still / "
            f"editorial hold / honest release-block downstream {audit['still_unresolved']}")
    return len(all_recovered)


def _semantic_recovery_pool_fingerprint(proj) -> str:
    """Identity of footage/index bytes available to a semantic recovery page."""
    import hashlib as _hashlib_pool
    import json as _json_pool

    def _stat(path) -> list[int]:
        try:
            st = Path(str(path or "")).stat()
            return [int(st.st_size), int(st.st_mtime_ns)]
        except Exception:                                # noqa: BLE001 — absence is pool state too
            return [0, 0]

    rows = []
    for source in sorted((proj.sources or []),
                         key=lambda item: str(getattr(item, "id", "") or "")):
        sid = str(getattr(source, "id", "") or "")
        rows.append({
            "id": sid,
            "status": str(getattr(source, "status", "") or ""),
            "media": _stat(getattr(source, "local_path", "") or ""),
            "shots": _stat(proj.shots_path(sid)),
            "words": _stat(Path(proj.index_dir) / f"{sid}.words.json"),
        })
    raw = _json_pool.dumps(rows, sort_keys=True, separators=(",", ":"))
    return _hashlib_pool.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _selection_relevance_retry_fingerprint(proj, segs, audit: dict) -> str:
    """State fingerprint that makes the final semantic retry bounded across Resume.

    A retry is useful again only when its actual inputs changed: blocker facts, selection window/
    still/verifier state, beat contract, or available source bytes. Recording the *post-attempt*
    fingerprint means an unchanged content failure immediately returns to the authoritative gate on
    Resume instead of repeatedly downloading/re-verifying the same material.
    """
    import hashlib as _hashlib_sr
    import json as _json_sr

    blocked = {int(e.get("segment_index", -1)) for e in (audit.get("blockers") or [])}

    def _stat(path):
        try:
            p = Path(str(path or ""))
            st = p.stat()
            return [str(p), int(st.st_size), int(st.st_mtime_ns)]
        except Exception:                                # noqa: BLE001 — missing is meaningful state
            return [str(path or ""), 0, 0]

    sels = []
    for s in sorted((x for x in (proj.selections or [])
                     if int(getattr(x, "segment_index", -1)) in blocked),
                    key=lambda x: int(getattr(x, "segment_index", -1))):
        sels.append({
            "index": int(getattr(s, "segment_index", -1)),
            "source_id": str(getattr(s, "source_id", "") or ""),
            "shot_index": int(getattr(s, "shot_index", -1)),
            "in": round(float(getattr(s, "in_point", 0.0) or 0.0), 3),
            "out": round(float(getattr(s, "out_point", 0.0) or 0.0), 3),
            "image": _stat(getattr(s, "image_path", "")),
            "image_meta": getattr(s, "image_meta", {}) or {},
            "verifier": getattr(s, "verifier", {}) or {},
        })
    seg_state = [{
        "index": int(getattr(s, "index", -1)),
        "text": str(getattr(s, "text", "") or ""),
        "policy": str(getattr(s, "visual_policy", "") or ""),
        "entity": str(getattr(s, "required_entity", "") or ""),
        "kind": str(getattr(s, "required_kind", "") or ""),
        "scene_query": str(getattr(s, "scene_query", "") or ""),
    } for s in (segs or []) if int(getattr(s, "index", -1)) in blocked]
    src_state = [{
        "id": str(getattr(s, "id", "") or ""),
        "status": str(getattr(s, "status", "") or ""),
        "file": _stat(getattr(s, "local_path", "")),
    } for s in sorted((proj.sources or []), key=lambda x: str(getattr(x, "id", "") or ""))]
    payload = {
        "schema": int(audit.get("schema_version", 0) or 0),
        "blockers": audit.get("blockers") or [],
        "selections": sels,
        "segments": seg_state,
        "sources": src_state,
        # An exhausted generation is invalidated by index-only repair/growth too. Source media may
        # be byte-identical while fresh shot boundaries or ASR words make a strict rematch solvable.
        "pool_fingerprint": _semantic_recovery_pool_fingerprint(proj),
        # Adding or revising a viewer-confirmed gap review must make a previously exhausted retry
        # runnable again.  The review itself is content evidence, not an out-of-band escape hatch.
        "gap_review": (getattr(proj, "meta", {}) or {}).get(
            "selection_relevance_gap_review") or {},
    }
    raw = _json_sr.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _hashlib_sr.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _unverifiable_relevance_indices(audit: dict) -> set[int]:
    """Strict blockers where a scoped re-verification can add missing evidence.

    Explicit semantic negatives are deliberately excluded: asking the identical cached question
    again is neither recovery nor evidence. Those go straight to new-footage recovery.
    """
    prefixes = (
        "verifier_absent", "verifier_error", "verifier_breaker", "verifier_unavailable",
        # A native still may have been the reason the moving selection was never asked the
        # publication question (or was asked an older question).  Once that still is rejected we
        # deliberately mark the moving verdict stale; it must receive one fresh scoped judgment
        # before acquisition or the specificity ladder can treat it as a content failure.
        "verifier_stale_native_still_conflict",
        "verdict_absent", "matches_narration_absent", "specific_enough_absent",
        "quality_ok_absent", "wrong_subject_visible_absent",
        "correct_subject_visible_absent", "target_visible_absent",
        "verifier_evidence_absent", "verifier_evidence_schema_mismatch",
        "verifier_evidence_model_mismatch", "verifier_evidence_mismatch",
        "verifier_evidence_unrecomputable", "verifier_evidence_window_not_sampled",
        # These are provenance defects, not semantic rejections.  The selected bytes may be the
        # exact scene (for example, an earlier lenient/contextual verifier judged them), but they
        # have never been asked the strict publication question.  Re-verify that same window once
        # before spending acquisition budget or declaring a footage gap.
        "exact_verifier_evidence_not_strict", "exact_moving_verdict_was_downgraded",
        "exact_moving_relevance_is_contextual",
    )
    cannot_reverify_in_place = (
        # There are no moving bytes to ask about.  These beats need selection recovery/stills,
        # not a verifier call that can never bind evidence to an absent source.
        "selection_absent", "moving_source_absent",
        "verdict_replace", "matches_narration_false", "specific_enough_false",
        "quality_ok_false", "wrong_subject_visible_true", "correct_subject_visible_false",
        "target_visible_false", "contradicts_narration_true", "era_ok_false",
        "deterministic_contradiction",
    )
    from .relevance_contract import completed_deliberate_exact_downgrade
    return {
        int(e.get("segment_index", -1))
        for e in (audit.get("blockers") or [])
        if not completed_deliberate_exact_downgrade(e)
        and any(str(r).startswith(prefixes) for r in (e.get("reasons") or []))
        and not any(str(r).startswith(cannot_reverify_in_place)
                    for r in (e.get("reasons") or []))
    }


_PERSISTENT_VERIFIER_TECHNICAL_PREFIXES = (
    "verifier_absent", "verifier_error", "verifier_breaker", "verifier_unavailable",
    "verdict_absent", "matches_narration_absent", "specific_enough_absent",
    "quality_ok_absent", "wrong_subject_visible_absent",
    "correct_subject_visible_absent", "target_visible_absent",
    "verifier_evidence_absent", "verifier_evidence_schema_mismatch",
    "verifier_evidence_model_mismatch", "verifier_evidence_mismatch",
    "verifier_evidence_unrecomputable", "verifier_evidence_window_not_sampled",
    "exact_verifier_evidence_not_strict", "exact_moving_verdict_was_downgraded",
    "exact_moving_relevance_is_contextual",
    # Quote location is a content requirement, but an uncertified/missing ASR generation is not
    # evidence that the pool lacks the line.  It must be repaired, never acquired around/softened.
    "exact_quote_pool_classification_indeterminate",
    "exact_quote_selected_asr_provenance_invalid",
    "exact_quote_unprompted_confirmation_inconclusive",
    "exact_reaction_context_asr_provenance_invalid",
)


def _persistent_verifier_technical_indices(audit: dict) -> set[int]:
    """Return blockers whose current facts are technically inconclusive, not content rejects.

    A missing moving selection is intentionally a content/footage recovery problem even though
    its verifier evidence is necessarily unrecomputable.  Everything else in this set has actual
    selected bytes but lacks a current, schema-complete, provenance-bound judgment.  Such beats
    stay blocked, while independent conclusive blockers may continue through bounded recovery.
    """
    from .relevance_contract import completed_deliberate_exact_downgrade
    out: set[int] = set()
    for entry in (audit.get("blockers") or []):
        reasons = [str(reason) for reason in (entry.get("reasons") or [])]
        if any(reason.startswith(("selection_absent", "moving_source_absent"))
               for reason in reasons):
            continue
        # The strict gate still rejects these bytes, but a fresh bound lenient KEEP has already
        # answered its verifier question.  It is an exact-footage/content shortfall, not a
        # persistent schema/backend/binding fault.  Mixed real technical reasons fail the shared
        # predicate and remain in this lane.
        if completed_deliberate_exact_downgrade(entry):
            continue
        if any(reason.startswith(_PERSISTENT_VERIFIER_TECHNICAL_PREFIXES)
               for reason in reasons):
            out.add(int(entry.get("segment_index", -1)))
    return out


def _selection_relevance_audit_without(audit: dict, excluded: set[int]) -> dict:
    """Copy an audit with selected blocker indices removed from its decision surface."""
    excluded = {int(i) for i in excluded}
    blockers = [
        entry for entry in (audit.get("blockers") or [])
        if int(entry.get("segment_index", -1)) not in excluded
    ]
    return {
        **audit,
        "blockers": blockers,
        "blocked_count": len(blockers),
        "status": "blocked" if blockers else "pass",
    }


def _strict_still_reply_schema_error(verdict, seg) -> str:
    """Validate a legacy-still reply before it can mutate persisted evidence.

    An explicit replace is a conclusive content judgment.  A keep is stronger: every boolean the
    publication contract consumes must be explicitly present and typed, including conditional
    subject/deictic facts.  Partial JSON and error wrappers are technical non-verdicts, not proof
    that a still failed semantically.
    """
    if not isinstance(verdict, dict):
        return "reply is not an object"
    raw_status = verdict.get("status", "")
    if not isinstance(raw_status, str):
        return "status is malformed"
    status = raw_status.strip().lower()
    if status not in ("", "ok"):
        return f"status={status or '<empty>'}"
    value = verdict.get("verdict")
    if value not in ("keep", "replace"):
        return f"verdict is unrecognized ({value!r})"
    if value == "replace":
        return ""
    for field in ("matches_narration", "specific_enough", "quality_ok",
                  "wrong_subject_visible", "contradicts_narration"):
        if not isinstance(verdict.get(field), bool):
            return f"keep is missing/malformed required boolean {field!r}"
    if str(getattr(seg, "required_entity", "") or "").strip() \
            and not isinstance(verdict.get("correct_subject_visible"), bool):
        return "keep is missing/malformed required boolean 'correct_subject_visible'"
    try:
        from . import policy as _policy_still_schema
        from .verify import effective_deictic_target
        target = effective_deictic_target(seg)
    except Exception:                                    # noqa: BLE001 — schema must fail closed
        target = ""
    if target and not isinstance(verdict.get("target_visible"), bool):
        return "keep is missing/malformed required boolean 'target_visible'"
    return ""


def _invalid_still_recovery_indices(audit: dict) -> set[int]:
    """All invalid stills, independent of the moving verifier's technical/content lane.

    A stale moving proof and an invalid still are two separate facts.  Excluding technical moving
    beats here stranded the bad image forever because scoped moving re-verification cannot repair
    or remove image bytes.
    """
    return {
        int(entry.get("segment_index", -1))
        for entry in (audit.get("blockers") or [])
        if any(str(reason).startswith("invalid_still:")
               for reason in (entry.get("reasons") or []))
    }


def _owned_native_still_recovery_candidate(meta: dict, invalid_reasons) -> bool:
    """Whether an invalid still has enough indexed ownership to use the native refresh lane."""
    return bool(
        str((meta or {}).get("source", "") or "")
        in ("source-frame", "source-frame-recovery")
        and str((meta or {}).get("src", "") or "")
        and (meta or {}).get("shot") is not None
        and list(invalid_reasons or []))


def _raise_semantic_recovery_failure(original_error, recovery_error, log) -> None:
    """Preserve typed repair failures; only unknown plumbing faults restore the original gate."""
    from .verify import NonRetryableBuildError, VisionBackendError

    if isinstance(recovery_error, (NonRetryableBuildError, VisionBackendError, PipelineError)):
        raise recovery_error
    log(f"semantic-recovery: technical failure ({type(recovery_error).__name__}: "
        f"{str(recovery_error)[:100]}); original publication block preserved")
    raise original_error


def _retry_selection_relevance(proj, segs, cfg, analysis, eng, *, faceid_obj, refs,
                               roster, policy, log) -> dict:
    """One scoped, strictly-positive repair chance after the publication contract blocks.

    This is intentionally NOT the legacy self-heal path: that path may demote an unreachable exact
    beat to character/abstract. Here policies and requirements are immutable. Missing verifier facts
    get one scoped re-check; remaining blockers get one bounded acquisition/rematch/reverify round;
    then only those blockers retry exact image recovery. The same semantic contract decides the
    result, so failure remains a NonRetryable content stop.
    """
    from . import relevance_contract as _R_sr

    # Request-local only: one strict recovery calls the publication audit repeatedly while verifier
    # and selection facts change, but whole-pool quote typing depends on neither.  Reuse that costly
    # source×quote scan while its actual inputs are unchanged.  The opaque cache builds its own
    # branches and validates source/index provenance plus quote/resolved-policy state on every use;
    # callers cannot inject a permissive ``paraphrase`` classification.
    _quote_pool_cache_sr = _R_sr._RequestQuotePoolClassificationCache()

    # Authored retrieval artifacts are generation-bound to every project quote.  A reviewed beat's
    # provenance can therefore become stale when an unrelated analyzer quote is removed even though
    # its beat, footage pools, and conclusive paraphrase result did not change. Refresh only that
    # narrow global-generation binding before deriving the bounded-retry fingerprint; the helper
    # fails closed on every semantic/evidence change. Persisting here makes the refresh auditable and
    # gives it a new retry generation instead of letting an old exhausted marker fast-return first.
    from . import selfheal as _selfheal_review_refresh_sr
    _review_refresh = \
        _selfheal_review_refresh_sr.refresh_selection_relevance_gap_review_quote_bindings(
            proj, cfg=cfg)
    if _review_refresh.get("refreshed"):
        proj.save()
        log("semantic-gap: refreshed unchanged paraphrase review quote binding for beat(s) "
            f"{_review_refresh.get('reviewed_beats', [])} after unrelated retrieval generation "
            f"{str(_review_refresh.get('previous_generation_fingerprint', ''))[:12]}→"
            f"{str(_review_refresh.get('current_generation_fingerprint', ''))[:12]}")

    audit_path = proj.output_dir / _R_sr.AUDIT_FILENAME
    audit = _R_sr.evaluate_selection_relevance(
        proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)
    _R_sr.write_selection_relevance_audit(audit_path, audit)
    if not audit.get("blockers"):
        return audit

    # Exhaustion is a CONTENT-generation concept. A current verifier/schema/provenance fault is a
    # retryable technical blocker and must never be hidden by the marker that bounds downloads.
    _initial_technical = _persistent_verifier_technical_indices(audit)
    _initial_content_audit = _selection_relevance_audit_without(audit, _initial_technical)
    initial_fp = _selection_relevance_retry_fingerprint(proj, segs, _initial_content_audit)
    previous = (getattr(proj, "meta", {}) or {}).get("selection_relevance_recovery") or {}
    _same_recovery_generation = previous.get("post_fingerprint") == initial_fp
    _current_initial_blockers = {
        int(e.get("segment_index", -1))
        for e in (_initial_content_audit.get("blockers") or [])}
    _pending_deferred = [int(i) for i in (previous.get("deferred") or [])
                         if str(i).lstrip("-").isdigit()
                         and int(i) in _current_initial_blockers]
    _content_generation_exhausted = _same_recovery_generation and not _pending_deferred
    # A viewer may bind a completed, hash-checked gap audit *after* an earlier strict generation
    # exhausted.  That review does not change the content fingerprint, so returning solely on the
    # old marker would make the newly authorized specificity ladder unreachable forever.  Check
    # current authorization before the fast return; it still grants no acquisition bypass or
    # semantic pass on its own—the ladder and unchanged final publication assertion remain below.
    _entry_exhausted_gap_authorizations = {}
    if _content_generation_exhausted:
        from . import selfheal as _selfheal_entry_gap_sr
        _entry_exhausted_gap_authorizations = \
            _selfheal_entry_gap_sr.reviewed_exhausted_gap_authorizations(
                proj, _initial_content_audit, cfg=cfg)
    if (_content_generation_exhausted and not _initial_technical
            and not _entry_exhausted_gap_authorizations):
        log(f"semantic-recovery: unchanged content failure already exhausted for "
            f"{_initial_content_audit['blocked_count']} beat(s) — skipping duplicate "
            "download/verification")
        return audit
    if _content_generation_exhausted and _entry_exhausted_gap_authorizations:
        log("semantic-recovery: strict generation is already exhausted, but current bound gap "
            "authorization is pending for beat(s) "
            f"{sorted(_entry_exhausted_gap_authorizations)}; continuing directly to the "
            "specificity ladder")
    if _same_recovery_generation and _pending_deferred:
        log(f"semantic-recovery: continuing {len(_pending_deferred)} audited deferred blocker(s) "
            f"from the prior bounded round {_pending_deferred}")
    elif _content_generation_exhausted and _initial_technical:
        log("semantic-recovery: content lane is unchanged/exhausted; retrying only the "
            f"technically inconclusive beat(s) {sorted(_initial_technical)}")

    before = sorted(_current_initial_blockers)
    _all_initial_blockers = sorted(
        int(e["segment_index"]) for e in (audit.get("blockers") or []))
    log(f"semantic-recovery: publication contract blocked {len(_all_initial_blockers)} "
        f"beat(s) {_all_initial_blockers}; content lane={before}, "
        f"technical lane={sorted(_initial_technical)} (no policy downgrade)")
    backend_down = False

    # Legacy source/web stills may be genuinely correct but predate persisted actual-image evidence.
    # Judge those exact bytes once before buying new footage. Persist NEGATIVE verdicts too, so the
    # audit says what the image showed rather than merely "old metadata"; only a complete positive
    # verdict flips the semantic flag.
    image_blockers = _invalid_still_recovery_indices(audit)
    if _content_generation_exhausted:
        image_blockers.clear()
    if image_blockers:
        from . import verify as _verify_img_sr
        from . import policy as _policy_img_sr
        from .verify import NonRetryableBuildError as _StillContentReject
        by_seg = {int(getattr(s, "index", -1)): s for s in segs}
        by_sel = {int(getattr(s, "segment_index", -1)): s for s in (proj.selections or [])}
        blocker_by_idx = {
            int(entry.get("segment_index", -1)): entry
            for entry in (audit.get("blockers") or [])}
        native_rows = []

        def _invalidate_conflicting_still(_sel, _idx: int, _reason: str) -> None:
            """Detach rejected image bytes and force one fresh judgment of the moving selection."""
            from .selfheal import _discard_invalid_still
            _discard_invalid_still(_sel, _reason, log)
            moving = dict(getattr(_sel, "verifier", {}) or {})
            moving["status"] = "stale_native_still_conflict"
            moving["native_still_conflict"] = _reason
            moving.pop("reused", None)
            _sel.verifier = moving

        for idx in sorted(image_blockers):
            seg, sel = by_seg.get(idx), by_sel.get(idx)
            path = str(getattr(sel, "image_path", "") or "") if sel is not None else ""
            if seg is None or not path or not Path(path).is_file():
                continue
            meta = dict(getattr(sel, "image_meta", {}) or {})
            invalid_reasons = list((blocker_by_idx.get(idx) or {}).get("reasons") or [])
            owned_native_needed = _owned_native_still_recovery_candidate(
                meta, invalid_reasons)
            if owned_native_needed:
                try:
                    from .build import _rescue_still_fullres
                    rescue = _rescue_still_fullres(
                        proj, sel, path, log, seg=seg, eng=eng,
                        allow_semantic_reject=True, refresh_semantic_verdict=True)
                except _verify_img_sr.VisionBackendError as exc:
                    raise PipelineError(
                        f"semantic recovery native-still verifier was technically unavailable "
                        f"for beat {idx}: {exc}") from exc
                except _StillContentReject as exc:
                    # Wrong-era/content and genuinely sub-HD owners are candidate failures: detach
                    # them and judge the moving selection.  Broken ownership/hash provenance is an
                    # integrity failure, not a candidate verdict, and must remain a loud stop.
                    if str(getattr(exc, "kind", "") or "") not in (
                            "selection_relevance", "native_resolution"):
                        raise
                    reason = str(exc)
                    native_rows.append({
                        "segment_index": idx, "status": "rejected_before_verify",
                        "declared_path": path, "owner_source_id": str(meta.get("src") or ""),
                        "owner_shot_index": meta.get("shot"), "reason": reason,
                    })
                    _invalidate_conflicting_still(sel, idx, reason)
                    log(f"semantic-recovery: beat {idx} owned still rejected before native "
                        f"publication ({reason}); moving selection will be freshly reverified")
                    continue

                verdict = dict(rescue.get("semantic_verifier") or {})
                why = str(rescue.get("semantic_strict_reason") or "")
                native_rows.append({
                    "segment_index": idx,
                    "status": "rejected" if why else "passed",
                    "declared_path": path, "native_path": str(rescue.get("path") or ""),
                    "owner_source_id": str(rescue.get("actual_source_id") or ""),
                    "owner_shot_index": rescue.get("actual_shot_index"),
                    "owner_time": rescue.get("actual_time"),
                    "native_dimensions": [int(rescue.get("image_width") or 0),
                                          int(rescue.get("image_height") or 0)],
                    "native_image_sha256": str(rescue.get("file_sha256") or ""),
                    "vision_served_by": str(rescue.get("semantic_model") or ""),
                    "reason": why,
                })
                if why:
                    _invalidate_conflicting_still(sel, idx, why)
                    log(f"semantic-recovery: beat {idx} native still failed — {why}; moving "
                        "selection will be freshly reverified")
                    continue

                exact = _policy_img_sr.is_exact(seg)
                sel.image_path = str(rescue["path"])
                meta.update({
                    "still_verification_attempted": True,
                    "still_verified": True,
                    "still_semantic_verified": True,
                    "still_verifier": verdict,
                    "still_image_sha256": str(rescue["file_sha256"]),
                    "exact_still_verified": bool(exact),
                    "exact_still_verifier": (verdict if exact else {}),
                    "relevance_class": ("exact_scene" if exact else "contextual_fallback"),
                    "native_semantic_materialized": True,
                    "native_indexed_keyframe_sha256": str(
                        rescue.get("indexed_keyframe_sha256") or ""),
                    "native_owner_source_content_fingerprint": str(
                        rescue.get("owner_source_content_fingerprint") or ""),
                    "native_owner_time": rescue.get("owner_time"),
                    "native_semantic_question_fingerprint": str(
                        rescue.get("semantic_question_fingerprint") or ""),
                    "native_semantic_model": str(rescue.get("semantic_model") or ""),
                })
                sel.image_meta = meta
                log(f"semantic-recovery: beat {idx} native still passed on exact "
                    f"{rescue.get('image_width')}x{rescue.get('image_height')} bytes")
                continue

            try:
                verdict = _verify_img_sr.verify_frame(
                    path, getattr(seg, "text", "") or "",
                    getattr(seg, "required_entity", "") or "",
                    getattr(seg, "required_kind", "") or "", [], eng,
                    getattr(eng, "anthropic_model", ""), is_specific=True,
                    expected_visual=getattr(seg, "expected_visual", "") or "",
                    scene_query=getattr(seg, "scene_query", "") or "",
                    era_hint=_verify_img_sr._project_beat_era(proj, seg),
                    venue_fallback=False,
                    must_see=_verify_img_sr.effective_deictic_target(seg))
            except Exception as exc:                     # noqa: BLE001 — retryable technical stop
                raise PipelineError(
                    f"semantic recovery legacy-still verifier failed for beat {idx}: "
                    f"{type(exc).__name__}: {exc}") from exc
            schema_error = _strict_still_reply_schema_error(verdict, seg)
            if schema_error:
                raise PipelineError(
                    f"semantic recovery legacy-still verifier was malformed/incomplete for "
                    f"beat {idx}: {schema_error}")
            verdict = dict(verdict)
            if not str(verdict.get("status", "") or "").strip():
                verdict["status"] = "ok"
            why = _R_sr.strict_still_evidence_reason(verdict, seg)
            meta = dict(getattr(sel, "image_meta", {}) or {})
            exact = _policy_img_sr.is_exact(seg)
            meta.update({
                "still_verification_attempted": True,
                "still_verified": not bool(why),
                "still_semantic_verified": not bool(why),
                "still_verifier": verdict,
                "still_image_sha256": _R_sr.image_sha256(path),
                "exact_still_verified": bool(exact and not why),
                "exact_still_verifier": (verdict if exact else {}),
            })
            if not why:
                meta["relevance_class"] = ("exact_scene" if exact else "contextual_fallback")
            sel.image_meta = meta
            log(f"semantic-recovery: beat {idx} legacy still actual-image reverify "
                f"{'passed' if not why else 'failed — ' + why}")
        if native_rows:
            import json as _json_native_still
            payload = {
                "schema": "native_still_reverification/1",
                "count": len(native_rows),
                "passed": sum(row.get("status") == "passed" for row in native_rows),
                "rejected": sum(row.get("status") != "passed" for row in native_rows),
                "rows": native_rows,
            }
            proj.meta["native_still_reverification"] = payload
            native_dest = proj.output_dir / "native_still_reverification_audit.json"
            native_tmp = native_dest.with_suffix(native_dest.suffix + ".tmp")
            native_tmp.write_text(
                _json_native_still.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            native_tmp.replace(native_dest)
        proj.save()
        audit = _R_sr.evaluate_selection_relevance(
            proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)

    # Missing/old evidence can be repaired without changing selection order. Explicit `false` facts
    # never enter this branch; they require genuinely different footage.
    unverified = _unverifiable_relevance_indices(audit)
    _scoped_verify_error = ""
    if unverified:
        try:
            from . import verify as _verify_sr
            result = _verify_sr.verify_and_repair(
                proj, segs, cfg, eng, only_indices=unverified, progress=None)
            _scoped_verify_error = _verifier_summary_error(result)
            if _scoped_verify_error:
                if _verifier_summary_is_global_outage(result):
                    raise PipelineError(
                        "semantic scoped re-verification was globally inconclusive: "
                        f"{_scoped_verify_error}")
                log("semantic-recovery: scoped reverify had isolated technical error(s) for "
                    f"{sorted(unverified)} ({_scoped_verify_error}); continuing independent "
                    "content blockers through bounded recovery")
            else:
                log(f"semantic-recovery: scoped reverify completed for {sorted(unverified)}")
            proj.save()
        except PipelineError:
            raise
        except _verify_sr.VisionBackendError as exc:
            raise PipelineError(
                "semantic scoped re-verification backend failed globally: "
                f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:                         # noqa: BLE001 — retryable technical stop
            raise PipelineError(
                "semantic scoped re-verification failed before a partitionable summary: "
                f"{type(exc).__name__}: {exc}") from exc

    audit = _R_sr.evaluate_selection_relevance(
        proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)
    _persistent_technical = _persistent_verifier_technical_indices(audit)
    if unverified:
        _still_unverifiable = sorted(
            set(unverified) & _persistent_technical)
        if _still_unverifiable:
            _remaining_facts = {
                int(_entry.get("segment_index", -1)): list(_entry.get("reasons") or [])
                for _entry in (audit.get("blockers") or [])
                if int(_entry.get("segment_index", -1)) in set(_still_unverifiable)
            }
            # A nominally successful verifier batch can still contain a fresh malformed KEEP. It
            # is correctly absent from the verdict cache, but it is not content evidence and must
            # not send the beat into acquisition/softening as though footage had been rejected.
            log("semantic-recovery: scoped re-verification remained technically inconclusive for "
                f"beat(s) {_still_unverifiable}: missing/schema evidence {_remaining_facts}; "
                "excluding them from content exhaustion and softening")

    # Recompute the bounded content generation after scoped reverify. A technical beat can become a
    # conclusive reject here; that newly typed content blocker earns the same one bounded recovery
    # page as any other. Conversely, persistent technical beats are absent from the fingerprint.
    _content_audit = _selection_relevance_audit_without(audit, _persistent_technical)
    _current_content_blockers = {
        int(e.get("segment_index", -1)) for e in (_content_audit.get("blockers") or [])}
    # A hash-bound, current-pool strict-exhaustion audit is stronger than another identical
    # discovery/rematch page.  Keep those beats in the publication audit and in ``before``; remove
    # them only from duplicate strict acquisition/image fallback so the audited specificity ladder
    # below can settle them.  Ordinary viewer-confirmed gaps (and every stale/tampered/technical
    # record) remain in the normal acquire-first path.
    from . import selfheal as _selfheal_gap_auth_sr
    _exhausted_gap_authorizations = \
        _selfheal_gap_auth_sr.reviewed_exhausted_gap_authorizations(
            proj, _content_audit, cfg=cfg)
    _exhausted_gap_indices = set(_exhausted_gap_authorizations)
    before = sorted(set(before) | _current_content_blockers)
    _content_fp = _selection_relevance_retry_fingerprint(proj, segs, _content_audit)
    _same_recovery_generation = previous.get("post_fingerprint") == _content_fp
    _pending_deferred = [int(i) for i in (previous.get("deferred") or [])
                         if str(i).lstrip("-").isdigit()
                         and int(i) in _current_content_blockers]
    _content_generation_exhausted = _same_recovery_generation and not _pending_deferred
    if _content_generation_exhausted and _current_content_blockers:
        log("semantic-recovery: current content generation is already exhausted for beat(s) "
            f"{sorted(_current_content_blockers)}; technical beats will still fail closed")
    blockers = (set() if _content_generation_exhausted else
                set(_current_content_blockers) - _exhausted_gap_indices)
    if _exhausted_gap_indices and not _content_generation_exhausted:
        log("semantic-recovery: audited strict-exhaustion gap beat(s) "
            f"{sorted(_exhausted_gap_indices)} bypass duplicate acquisition; "
            "the specificity ladder and final publication gate still decide")
    _recovery_deferred: list[int] = []
    _page_pool_changed = False
    _completed_page_scope: set[int] = set()
    _semantic_page_completed = False
    if blockers and not backend_down:
        # One unchanged generation walks its capped scope exactly once. The first round receives all
        # blockers and persists the cap tail; the next Resume receives ONLY that tail. Passing the
        # full blocker set again would let `recovery_pick` refill spare slots with already-attempted
        # beats, producing an endless [head+tail] cycle. A changed source/selection fingerprint starts
        # a new generation and may legitimately reconsider the full scope.
        _recovery_scope = (set(_pending_deferred) & blockers
                           if _same_recovery_generation else set(blockers))
        if _recovery_scope:
            import uuid as _uuid_sr_page
            _page_request_id = _uuid_sr_page.uuid4().hex
            _pool_before_page = _semantic_recovery_pool_fingerprint(proj)
            try:
                _recover_unresolved_beats(
                    proj, segs, analysis, cfg, eng, faceid_obj=faceid_obj, refs=refs,
                    roster=roster, policy=policy, log=log, only_indices=_recovery_scope,
                    audit_filename="semantic_recovery_audit.json",
                    audit_request_id=_page_request_id,
                    quote_pool_cache=_quote_pool_cache_sr)
            except Exception as exc:                     # noqa: BLE001 — retryable technical stop
                raise PipelineError(
                    f"semantic recovery helper failed before conclusive page completion: "
                    f"{type(exc).__name__}: {exc}") from exc
            try:
                _ra = _load_conclusive_semantic_recovery_page(
                    proj.output_dir / "semantic_recovery_audit.json",
                    request_id=_page_request_id, requested_scope=_recovery_scope)
            except PipelineError:
                # Do not write selection_relevance_recovery, and do not let an empty/stale audit
                # turn an unattempted strict page into permission for stills or specificity loss.
                raise
            _semantic_page_completed = True
            _recovery_deferred = sorted({
                int(i) for i in (_ra.get("deferred_retriable") or [])
                if str(i).lstrip("-").isdigit()})
            _completed_page_scope = {int(i) for i in (_ra.get("page_scope") or [])}
            _page_pool_changed = (
                _semantic_recovery_pool_fingerprint(proj) != _pool_before_page)
            if _page_pool_changed:
                # A semantic page can add/index footage while an earlier genuine-gap beat is
                # still carrying a pool-bound exact→character→abstract softening.  Restore
                # that stale promise *inside this recovery generation*, before deriving blockers
                # or pagination state.  Leaving restoration to the later publication assertion
                # made the page persist a 28-beat fingerprint/deferred tail even though that
                # assertion immediately revived a 29th strict beat; Resume then mistook the same
                # pool for a fresh generation and restarted at its head.
                try:
                    from . import selfheal as _selfheal_restore_sr
                    _restored_softenings = \
                        _selfheal_restore_sr.restore_stale_selection_relevance_softenings(
                            proj, segs, cfg=cfg, log=log)
                except Exception as exc:                 # noqa: BLE001 — retryable state repair
                    raise PipelineError(
                        "semantic recovery could not restore stale pool-bound softening "
                        f"after page growth: {type(exc).__name__}: {exc}") from exc
                if _restored_softenings.get("restored"):
                    log("semantic-recovery: page changed the indexed pool; restored stale "
                        "authored promise(s) "
                        f"{_restored_softenings['restored']} before pagination bookkeeping")
        proj.save()

        # Re-evaluate before the still pass: recovered moving footage needs no image, and passing a
        # subset keeps the second still pass from decorating/changing any already-approved beat.
        audit = _R_sr.evaluate_selection_relevance(
            proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)
        _technical_after_page = _persistent_verifier_technical_indices(audit)
        _content_after_page = _selection_relevance_audit_without(
            audit, _technical_after_page)
        _all_blockers_after_page = {
            int(e["segment_index"]) for e in (_content_after_page.get("blockers") or [])}
        # Revalidate against the post-page pool.  A newly downloaded/indexed source invalidates the
        # absence proof, so that beat immediately returns to strict recovery instead of spending a
        # stale downgrade authorization.  An unchanged pool keeps it out of exact image fallback.
        _exhausted_after_page = \
            _selfheal_gap_auth_sr.reviewed_exhausted_gap_authorizations(
                proj, _content_after_page, cfg=cfg)
        blockers = _all_blockers_after_page - set(_exhausted_after_page)
        if _page_pool_changed:
            # The page's own scoped beats were rematched after its downloads/indexing. Every prior
            # blocker outside that page was transactionally restored, so it has never seen the new
            # pool. Re-open those retrievable beats exactly once under the new post-page generation;
            # queryless beats cannot benefit from another matcher page and do not rotate.
            by_idx = {int(getattr(s, "index", -1)): s for s in segs}
            retriable = {idx for idx in blockers if recovery_query(by_idx.get(idx))}
            reconsider = retriable - _completed_page_scope
            _recovery_deferred = sorted(
                (set(_recovery_deferred) & blockers) | reconsider)
            if reconsider:
                log(f"semantic-recovery: page expanded the indexed pool; deferring prior "
                    f"out-of-page blocker(s) {sorted(reconsider)} for one strict pass against "
                    "the new generation")
        _recovery_deferred = sorted(set(_recovery_deferred) & blockers)
        # Pagination is one generation-wide strict walk. If any retrievable tail remains, none of
        # the current blockers may lose specificity yet: a later page can add footage that solves a
        # previously attempted head beat. Only a fully exhausted/stable walk releases the whole
        # blocker set to image fallback and the audited softening ladder.
        _waiting_for_strict_page = set(blockers) if _recovery_deferred else set()
        _strict_page_exhausted = blockers - _waiting_for_strict_page
        if _waiting_for_strict_page:
            log(f"semantic-recovery: strict pagination still has deferred page(s) "
                f"{_recovery_deferred}; preserving exact state for all current blockers "
                f"{sorted(_waiting_for_strict_page)} until the generation-wide walk completes")
        if _strict_page_exhausted:
            import os as _os_sr
            if _os_sr.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip().lower() \
                    not in ("0", "false", "no"):
                target_segs = [s for s in segs
                               if int(getattr(s, "index", -1)) in _strict_page_exhausted]
                try:
                    _fill_image_fallbacks(
                        proj, target_segs, analysis, faceid_obj, refs, log, eng_cfg=eng,
                        fail_on_web_technical=True)
                except Exception as exc:                 # noqa: BLE001 — retryable technical stop
                    raise PipelineError(
                        f"semantic recovery exact-image fallback failed before completion: "
                        f"{type(exc).__name__}: {exc}") from exc
                proj.save()

    # Genuine, audited footage gaps are different from wrong picks and technical evidence faults.
    # Strict-positive recovery above gets the first and final chance to preserve specificity. Only
    # after it is exhausted may the existing exact→character→abstract ladder run. Located real
    # quotes remain excluded unless a current schema-3 audit proves their exact footage exists only
    # below the native-HD floor; binding/schema/backend failures are always excluded. The unchanged
    # publication contract is immediately re-evaluated below; softening never bypasses it.
    pre_soften = _R_sr.evaluate_selection_relevance(
        proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)
    _pre_soften_technical = _persistent_verifier_technical_indices(pre_soften)
    _pre_soften_for_ladder = _selection_relevance_audit_without(
        pre_soften, _pre_soften_technical)
    if _recovery_deferred:
        # Keep this generation-wide too. A deferred tail may acquire footage for an already-tried
        # head, so softening even that head before the tail runs would erase the very blocker that
        # needs to be reconsidered against the expanded pool.
        _ladder_blockers = []
        _pre_soften_for_ladder = {
            **_pre_soften_for_ladder,
            "blockers": _ladder_blockers,
            "blocked_count": len(_ladder_blockers),
            "status": "blocked" if _ladder_blockers else "pass",
        }
    elif _content_generation_exhausted:
        # The old strict marker suppresses duplicate recovery, not a newly bound completed gap
        # audit.  Retain exactly the currently authorized rows; every other exhausted blocker stays
        # frozen and the final publication assertion continues to block it.
        _pending_authorized = _selfheal_gap_auth_sr.reviewed_exhausted_gap_authorizations(
            proj, _pre_soften_for_ladder, cfg=cfg)
        _ladder_blockers = [
            entry for entry in (_pre_soften_for_ladder.get("blockers") or [])
            if int(entry.get("segment_index", -1)) in _pending_authorized
        ]
        _pre_soften_for_ladder = {
            **_pre_soften_for_ladder,
            "blockers": _ladder_blockers,
            "blocked_count": len(_ladder_blockers),
            "status": "blocked" if _ladder_blockers else "pass",
        }
    _gap_softening = None
    if _pre_soften_for_ladder.get("blockers"):
        import os as _os_gap
        if (_os_gap.environ.get("VIDLORE_CLIPSTUDIO_SELFHEAL", "1").strip().lower()
                not in ("0", "false", "no")
                and _os_gap.environ.get("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN", "1")
                .strip().lower() not in ("0", "false", "no")):
            from .verify import NonRetryableBuildError as _GapNonRetryable
            try:
                from . import selfheal as _selfheal_gap
                _gap_softening = _selfheal_gap.heal_selection_relevance_gaps(
                    proj, segs, cfg, _pre_soften_for_ladder,
                    policy=policy, eng=eng, log=log)
                # The ladder may change exact→character→abstract specificity.  Never carry the
                # pre-softening quote branch snapshot across that semantic boundary, even if a
                # future policy implementation happens to resolve to the same label.
                _quote_pool_cache_sr.invalidate()
                if _gap_softening.get("candidate_count"):
                    log(f"semantic-gap: specificity ladder softened "
                        f"{_gap_softening.get('softened_count', 0)}/"
                        f"{_gap_softening.get('candidate_count', 0)} exhausted gap beat(s)")
            except _GapNonRetryable:
                # A scene-lineage/native-owner invariant is deterministic content/build failure;
                # relabelling it PipelineError would make unchanged Resume attempts loop forever.
                raise
            except Exception as exc:                     # noqa: BLE001 — retryable technical stop
                # The ladder owns a transaction that restores the exact beat/selection state before
                # re-raising. Do not convert that technical failure into a content exhaustion marker:
                # Resume must get another chance once the verifier/cache/disk fault is repaired.
                raise PipelineError(
                    f"semantic gap specificity ladder failed before completion: "
                    f"{type(exc).__name__}: {exc}") from exc

    final = _R_sr.evaluate_selection_relevance(
        proj, segs, cfg=cfg, quote_pool_cache=_quote_pool_cache_sr)
    _R_sr.write_selection_relevance_audit(audit_path, final)
    _final_technical = _persistent_verifier_technical_indices(final)
    _final_content = _selection_relevance_audit_without(final, _final_technical)
    after = sorted(int(e["segment_index"]) for e in (_final_content.get("blockers") or []))
    _recovery_deferred = [i for i in _recovery_deferred if i in set(after)]
    _technical_details = [
        {
            "segment_index": int(entry.get("segment_index", -1)),
            "reasons": list(entry.get("reasons") or []),
        }
        for entry in (final.get("blockers") or [])
        if int(entry.get("segment_index", -1)) in _final_technical
    ]
    # Pure technical failure is not content exhaustion and creates no recovery marker. In a mixed
    # run, the marker records only the content lane while retaining the technical lane separately
    # for diagnosis; its before/after/fingerprint can therefore never suppress verifier retry.
    if before or after or previous:
        _marker = (dict(previous) if _content_generation_exhausted else {})
        _marker.update({
            "schema_version": _R_sr.SCHEMA_VERSION,
            "before": ((list(_marker.get("before") or []))
                       if _content_generation_exhausted and not before else before),
            "after": after,
            "post_fingerprint": _selection_relevance_retry_fingerprint(
                proj, segs, _final_content),
            "gap_softening": (((_gap_softening or _marker.get("gap_softening") or {}))
                              if _content_generation_exhausted else (_gap_softening or {})),
            "deferred": (_marker.get("deferred") or [])
                        if _content_generation_exhausted else _recovery_deferred,
            "pool_fingerprint": _semantic_recovery_pool_fingerprint(proj),
            "pool_changed_during_page": (
                bool(_marker.get("pool_changed_during_page"))
                if _content_generation_exhausted else _page_pool_changed),
            "completed_page_scope": (
                list(_marker.get("completed_page_scope") or [])
                if _content_generation_exhausted else sorted(_completed_page_scope)),
            "technical_blockers": _technical_details,
        })
        # The page receipt is part of the same atomic project save as its cursor.  If the process
        # dies after this save but before checkpoint rebinding, Resume can validate/adopt the exact
        # completed state without resetting either its repeat detector or its finite page budget.
        if _semantic_page_completed:
            _marker["pagination_receipt"] = _advance_semantic_pagination_receipt(
                previous if _same_recovery_generation else None, _marker)
        elif not _recovery_deferred:
            # An exhausted marker needs no executable cursor.  In particular, an authorized gap
            # ladder can change the final fingerprint without spending another strict page; do not
            # carry a receipt bound to the old fingerprint into that terminal state.
            _marker.pop("pagination_receipt", None)
        proj.meta["selection_relevance_recovery"] = _marker
    proj.save()
    if after and _recovery_deferred:
        log(f"semantic-recovery: {len(after)} content beat(s) still fail positive semantic "
            f"evidence {after}; {len(_recovery_deferred)} audited deferred beat(s) remain queued")
    elif after:
        log(f"semantic-recovery: {len(after)} content beat(s) still fail positive semantic "
            f"evidence {after}; stopping before render")
    elif not _technical_details:
        log("semantic-recovery: publication contract CLEAR after scoped strict recovery")
    if _technical_details:
        _technical_indices = sorted(_final_technical)
        _technical_reasons = {
            int(entry["segment_index"]): list(entry.get("reasons") or [])
            for entry in _technical_details
        }
        _summary = (f"; verifier summary/error: {_scoped_verify_error}"
                    if _scoped_verify_error else "")
        # A BACKEND THAT IS DOWN AND A FRAME THE BACKEND WON'T LOOK AT ARE NOT THE SAME THING.
        #
        # Measured on job 229233891e: 145 of 146 beats verified normally, and beat 36 alone came
        # back with no verdict at all — every time, for hours. Diagnosed by hand: the vision probe
        # passed, Gemini answered ordinary text, four other keyframes from the same job were judged
        # fine, and the file itself was a clean 512x288 JPEG. Only that one image was refused, by
        # Gemini and by the Claude fallback both. That is a fact about the picture, not a broken
        # system — and treating it as a broken system threw away a 2h20m render at the last gate.
        #
        # An outage must still be fatal: during one, "no verdict" carries no information and
        # accepting anything would ship unverified footage. So the two are told apart by asking the
        # backend, right here, with the pipeline's own health probe. Healthy backend + a beat that
        # will not resolve = an unjudgeable frame; the beat stays UNVERIFIED and the publication
        # contract goes on blocking it exactly as before, which is the whole point — nothing is
        # accepted, the render simply stops dying for it.
        # Whether the backend is up is already answered by THIS pass: if other beats came back
        # with real verdicts moments ago, the verifier worked. That evidence is free, deterministic
        # and taken from the same backend, the same moment and the same kind of image — a live
        # probe would be none of those, and it made the outcome depend on network state (three
        # existing tests started passing or failing by suite ordering alone).
        _judged = 0
        for _row in (final.get("checked") or []):
            if str(((_row.get("verifier") or {}) if isinstance(_row, dict) else {})
                   .get("verdict") or ""):
                _judged += 1
        _healthy = _judged >= max(3, len(_technical_indices) * 3)
        if not _healthy:
            _inconclusive = PipelineError(
                "semantic scoped re-verification remained technically inconclusive for "
                f"beat(s) {_technical_indices}: missing/schema evidence {_technical_reasons}"
                f"{_summary}; only {_judged} beat(s) in this pass produced any verdict at all, so "
                "the verifier itself is not proven healthy — refusing to read an outage as a "
                "content result; independent content recovery was saved")
            # Job 0ca9dc4c2f died here THREE times. The message, the audit rows and block-mode
            # behaviour are unchanged; all that is added is the error's identity.
            #
            # The bar above asks for a health PROOF, and the absence of a proof is not proof of an
            # outage. When this pass obtained NO verdict at all, that IS positive outage evidence
            # and the error stays untyped — infrastructure, fatal in every mode, restore the
            # backend and resume. But when the backend demonstrably answered and merely answered
            # for too few beats to clear the ratio, the residual beats are the content fact "a
            # verdict we could not obtain" — which this same pipeline already ships as a marked
            # draft at the unverified-exact gate, and which the sibling arm of this very fork hands
            # to assert_selection_relevance as a kind="selection_relevance" content stop. This arm
            # was the one that killed the render instead. Nothing is accepted: the blockers stay in
            # the audit, the beats stay UNVERIFIED, and block mode still raises.
            if _judged > 0:
                _inconclusive.kind = "selection_relevance"
            raise _inconclusive
        log(f"semantic-recovery: beat(s) {_technical_indices} could not be judged while "
            f"{_judged} other beat(s) in the SAME pass were judged normally — treating their "
            f"frames as unjudgeable, not the system as broken. They stay UNVERIFIED and the "
            f"publication contract still blocks them: {_technical_reasons}{_summary}")
    return final


_SEMANTIC_PAGINATION_RECEIPT_SCHEMA = 1
_SEMANTIC_PAGINATION_RECEIPT_KEY = "pagination_receipt"


def _semantic_cursor_int_list(marker: dict, field: str, *, require_nonempty: bool = False) \
        -> list[int]:
    """Read one cursor index list without accepting bools, strings, or missing/falsey fields."""
    if field not in marker:
        raise PipelineError(f"semantic recovery cursor is missing {field!r}")
    value = marker[field]
    if (not isinstance(value, list)
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in value)
            or len(value) != len(set(value))
            or (require_nonempty and not value)):
        raise PipelineError(f"semantic recovery cursor field {field!r} is malformed")
    return sorted(value)


def _semantic_recovery_marker_fields(marker) -> dict:
    """Validate every core field consumed by pagination before it can drive work or rebinding."""
    from . import relevance_contract as _R_cursor

    if not isinstance(marker, dict) or not marker:
        raise PipelineError("semantic recovery cursor is missing or malformed")
    schema = marker.get("schema_version")
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != int(_R_cursor.SCHEMA_VERSION)):
        raise PipelineError("semantic recovery cursor schema is missing or malformed")
    before = _semantic_cursor_int_list(marker, "before", require_nonempty=True)
    after = _semantic_cursor_int_list(marker, "after")
    deferred = _semantic_cursor_int_list(marker, "deferred")
    completed = _semantic_cursor_int_list(marker, "completed_page_scope")
    post_fp = marker.get("post_fingerprint")
    pool_fp = marker.get("pool_fingerprint")
    if not isinstance(post_fp, str) or not post_fp.strip():
        raise PipelineError("semantic recovery cursor post fingerprint is missing or malformed")
    if not isinstance(pool_fp, str) or not pool_fp.strip():
        raise PipelineError("semantic recovery cursor pool fingerprint is missing or malformed")
    if not isinstance(marker.get("pool_changed_during_page"), bool):
        raise PipelineError("semantic recovery cursor pool-change receipt is missing or malformed")
    if not isinstance(marker.get("technical_blockers"), list):
        raise PipelineError("semantic recovery cursor technical-blocker receipt is missing or malformed")
    if not set(deferred).issubset(set(after)):
        raise PipelineError("semantic recovery cursor contains a deferred non-blocker")
    if set(completed) & set(deferred):
        raise PipelineError("semantic recovery completed and deferred scopes overlap")
    return {
        "before": before, "after": after, "deferred": deferred, "completed": completed,
        "post_fingerprint": post_fp, "pool_fingerprint": pool_fp,
    }


def _semantic_pagination_state_digest(marker: dict) -> str:
    """Digest the progress-bearing state; receipt/log fields deliberately cannot create progress."""
    import hashlib as _hashlib_page_state
    import json as _json_page_state

    fields = _semantic_recovery_marker_fields(marker)
    payload = {
        "post_fingerprint": fields["post_fingerprint"],
        "deferred": fields["deferred"],
        "pool_fingerprint": fields["pool_fingerprint"],
    }
    raw = _json_page_state.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _hashlib_page_state.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _semantic_pagination_max_pages(marker: dict) -> int:
    """Derive the finite walk ceiling once; the persisted value wins on every later process."""
    import math as _math_pages
    import os as _os_pages

    fields = _semantic_recovery_marker_fields(marker)
    try:
        page_cap = max(1, int(_os_pages.environ.get(
            "VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS", "8") or 8))
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            "VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS is not an integer") from exc
    generation_size = max(
        len(fields["before"]), len(fields["after"]), len(fields["deferred"]), 1)
    default_max = max(4, 2 * int(_math_pages.ceil(generation_size / page_cap)) + 4)
    try:
        return max(1, int(_os_pages.environ.get(
            "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY_MAX_PAGES", str(default_max))
            or default_max))
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            "VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY_MAX_PAGES is not an integer") from exc


def _seed_semantic_pagination_receipt(marker: dict) -> dict:
    """Adopt one valid pre-receipt (schema-9) marker without resetting its completed first page."""
    fields = _semantic_recovery_marker_fields(marker)
    digest = _semantic_pagination_state_digest(marker)
    max_pages = _semantic_pagination_max_pages(marker)
    terminal = ""
    if fields["deferred"] and max_pages <= 1:
        terminal = (f"finite_guard:{max_pages}:"
                    + ",".join(str(i) for i in fields["deferred"]))
    return {
        "schema_version": _SEMANTIC_PAGINATION_RECEIPT_SCHEMA,
        "generation_fingerprint": fields["post_fingerprint"],
        "generation_pool_fingerprint": fields["pool_fingerprint"],
        "cursor_fingerprint": fields["post_fingerprint"],
        "cursor_pool_fingerprint": fields["pool_fingerprint"],
        "page_count": 1,
        "max_pages": max_pages,
        "attempted_state_digests": [digest],
        "terminal_reason": terminal,
    }


def _validate_semantic_pagination_receipt(marker: dict) -> dict:
    """Validate the persisted cross-process budget and repeat detector against this cursor."""
    fields = _semantic_recovery_marker_fields(marker)
    if _SEMANTIC_PAGINATION_RECEIPT_KEY not in marker:
        raise PipelineError("semantic recovery pagination receipt is missing")
    receipt = marker[_SEMANTIC_PAGINATION_RECEIPT_KEY]
    if not isinstance(receipt, dict) or not receipt:
        raise PipelineError("semantic recovery pagination receipt is malformed")
    schema = receipt.get("schema_version")
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != _SEMANTIC_PAGINATION_RECEIPT_SCHEMA):
        raise PipelineError("semantic recovery pagination receipt schema is malformed")
    for field in ("generation_fingerprint", "generation_pool_fingerprint",
                  "cursor_fingerprint", "cursor_pool_fingerprint"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise PipelineError(
                f"semantic recovery pagination receipt field {field!r} is malformed")
    if (receipt["cursor_fingerprint"] != fields["post_fingerprint"]
            or receipt["cursor_pool_fingerprint"] != fields["pool_fingerprint"]):
        raise PipelineError("semantic recovery pagination receipt is not bound to the current cursor")
    page_count = receipt.get("page_count")
    max_pages = receipt.get("max_pages")
    if (isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1
            or isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1
            or page_count > max_pages):
        raise PipelineError("semantic recovery pagination page budget is malformed")
    digests = receipt.get("attempted_state_digests")
    if (not isinstance(digests, list) or not digests
            or any(not isinstance(item, str) or not item.strip() for item in digests)
            or len(digests) != len(set(digests)) or len(digests) > page_count):
        raise PipelineError("semantic recovery pagination attempted-state receipt is malformed")
    current_digest = _semantic_pagination_state_digest(marker)
    if current_digest not in digests:
        raise PipelineError("semantic recovery pagination receipt does not include current state")
    terminal = receipt.get("terminal_reason")
    if not isinstance(terminal, str):
        raise PipelineError("semantic recovery pagination terminal receipt is malformed")
    if terminal and not (terminal.startswith("no_progress:")
                         or terminal.startswith("finite_guard:")):
        raise PipelineError("semantic recovery pagination terminal reason is unrecognized")
    return dict(receipt)


def _advance_semantic_pagination_receipt(previous_marker, marker: dict) -> dict:
    """Atomically advance page count/state with the new marker written by a conclusive page."""
    fields = _semantic_recovery_marker_fields(marker)
    digest = _semantic_pagination_state_digest(marker)
    if previous_marker is None:
        return _seed_semantic_pagination_receipt(marker)

    _semantic_recovery_marker_fields(previous_marker)
    if _SEMANTIC_PAGINATION_RECEIPT_KEY in previous_marker:
        previous = _validate_semantic_pagination_receipt(previous_marker)
    else:
        # One-time compatibility adoption for the live schema-9 marker created before receipts.
        previous = _seed_semantic_pagination_receipt(previous_marker)
    count = int(previous["page_count"]) + 1
    max_pages = int(previous["max_pages"])
    if count > max_pages:
        raise PipelineError(
            f"semantic recovery pagination refused page {count} beyond persisted {max_pages}-page guard")
    digests = list(previous["attempted_state_digests"])
    terminal = ""
    if digest in digests:
        terminal = "no_progress:" + ",".join(str(i) for i in fields["deferred"])
    else:
        digests.append(digest)
    if fields["deferred"] and count >= max_pages and not terminal:
        terminal = (f"finite_guard:{max_pages}:"
                    + ",".join(str(i) for i in fields["deferred"]))
    return {
        **previous,
        "cursor_fingerprint": fields["post_fingerprint"],
        "cursor_pool_fingerprint": fields["pool_fingerprint"],
        "page_count": count,
        "attempted_state_digests": digests,
        "terminal_reason": terminal,
    }


def _semantic_recovery_cursor(proj, segs, audit: dict, *, allow_absent: bool,
                              allow_stale: bool, adopt_legacy: bool) -> dict:
    """Classify and validate a persisted cursor against the exact current semantic facts."""
    from . import relevance_contract as _R_cursor

    if not isinstance(audit, dict):
        raise PipelineError("semantic recovery page returned no current relevance audit")
    all_blockers = list(audit.get("blockers") or [])
    if not all_blockers:
        return {"status": "clear", "deferred": [], "audit": audit}
    technical = _persistent_verifier_technical_indices(audit)
    content_audit = _selection_relevance_audit_without(audit, technical)
    blockers = sorted({
        int(entry.get("segment_index", -1))
        for entry in (content_audit.get("blockers") or [])
    })
    if not blockers:
        # Technical verifier facts still need `_retry_selection_relevance`; they simply do not own
        # an executable content-pagination cursor and therefore cannot consume/reset its budget.
        return {"status": "unpaged", "deferred": [], "audit": content_audit}

    meta = getattr(proj, "meta", {}) or {}
    if "selection_relevance_recovery" not in meta:
        if allow_absent:
            return {"status": "absent", "deferred": [], "audit": content_audit}
        raise PipelineError("semantic recovery page did not persist its cursor")
    marker = meta["selection_relevance_recovery"]
    # A relevance-contract schema bump deliberately invalidates every old exhaustion decision.
    # During the initial preflight walk, a complete positive older schema is stale state: run a
    # fresh bounded page and replace it with a current cursor.  Missing/falsey/malformed schemas,
    # future schemas, and every post-page validation remain hard technical failures; they cannot
    # reset a finite recovery budget or authorize publication.
    current_schema = int(_R_cursor.SCHEMA_VERSION)
    marker_schema = marker.get("schema_version") if isinstance(marker, dict) else None
    if (allow_stale and isinstance(marker_schema, int)
            and not isinstance(marker_schema, bool)
            and 0 < marker_schema < current_schema):
        # A stale cursor is being DISCARDED — that is what "stale" means here, and the fresh
        # bounded page below replaces it. So validate only what an older schema is guaranteed to
        # carry.
        #
        # The previous form stamped the CURRENT schema onto the old marker and then ran the current
        # validator over it, which demands fields (`deferred`, `completed_page_scope`,
        # `pool_fingerprint`, `gap_softening`) added long after that marker was written. Every
        # cursor older than those additions therefore RAISED instead of being discarded, and the
        # render died. MEASURED: the short end-to-end test crashed with "semantic recovery cursor is
        # missing 'deferred'" against a real on-disk marker
        # {"schema_version": 3, "before": [2, 5], "after": [2, 5], "post_fingerprint": "..."} while
        # the current schema is 13 — a cursor left behind by an earlier build permanently breaking
        # the job. Validating what is about to be thrown away is what turned recoverable state into
        # a fatal error.
        #
        # `before` is the one list every schema of this cursor has carried, and a non-empty `before`
        # is exactly the "complete positive older schema" this branch exists for. A corrupt one
        # still raises, so a damaged marker is not silently accepted either.
        _semantic_cursor_int_list(marker, "before", require_nonempty=True)
        return {"status": "stale", "deferred": [], "audit": content_audit}
    fields = _semantic_recovery_marker_fields(marker)
    current_pool = _semantic_recovery_pool_fingerprint(proj)
    current_content = _selection_relevance_retry_fingerprint(proj, segs, content_audit)
    if (fields["pool_fingerprint"] != current_pool
            or fields["post_fingerprint"] != current_content):
        if allow_stale:
            return {"status": "stale", "deferred": [], "audit": content_audit}
        raise PipelineError("semantic recovery page cursor is stale against current content/pool")
    if fields["after"] != blockers:
        raise PipelineError(
            "semantic recovery cursor does not match current content blockers")

    receipt = None
    if _SEMANTIC_PAGINATION_RECEIPT_KEY in marker:
        receipt = _validate_semantic_pagination_receipt(marker)
    elif fields["deferred"]:
        if not adopt_legacy:
            raise PipelineError("semantic recovery page omitted its pagination receipt")
        receipt = _seed_semantic_pagination_receipt(marker)
        marker[_SEMANTIC_PAGINATION_RECEIPT_KEY] = receipt
        proj.meta["selection_relevance_recovery"] = marker
        proj.save()
    return {
        "status": "current", "deferred": fields["deferred"], "audit": content_audit,
        "marker": marker, "receipt": receipt,
    }


def _pagination_terminal_error(cursor: dict):
    """Exhaustion of the repair walk, typed as the CONTENT stop it actually is.

    Both terminal reasons say the same thing: beats still fail the semantic publication contract
    and this machinery has no move left. That is exactly the situation review mode exists for — yet
    a bare PipelineError carries no `kind`, so the review escape in `_produce_auto` (which keys on
    kind == "selection_relevance") let it through as if it were a plumbing fault and killed an
    8-hour render that had a deliverable draft sitting one stage away. Tag it. Nothing softens:
    strict mode still raises, the blocked beats stay blocked and audited, and the review build is
    still named REVIEW_DRAFT. The neighbouring guards for a malformed/unrecognised receipt stay
    untyped on purpose — those are technical faults and must fail hard in every mode.
    """
    receipt = cursor.get("receipt") or {}
    reason = str(receipt.get("terminal_reason", "") or "")
    deferred = list(cursor.get("deferred") or [])
    err = None
    if reason.startswith("no_progress:"):
        err = PipelineError(
            "semantic recovery pagination made no forward progress: repeated current "
            f"deferred scope {deferred}")
    elif reason.startswith("finite_guard:"):
        err = PipelineError(
            f"semantic recovery pagination reached its finite {receipt.get('max_pages')}-page "
            f"guard with current deferred scope {deferred}")
    if err is not None:
        err.kind = "selection_relevance"
    return err


def _pending_semantic_recovery_page(proj, segs, audit: dict) \
        -> tuple[list[int], tuple[str, tuple[int, ...], str]]:
    """Compatibility wrapper returning a fully validated post-page cursor."""
    cursor = _semantic_recovery_cursor(
        proj, segs, audit, allow_absent=False, allow_stale=False, adopt_legacy=False)
    if cursor["status"] in ("clear", "unpaged"):
        return [], ("", (), "")
    marker = cursor["marker"]
    return list(cursor["deferred"]), (
        str(marker["post_fingerprint"]), tuple(cursor["deferred"]),
        str(marker["pool_fingerprint"]))


def _drain_semantic_recovery_pages(proj, segs, recover_page, *, initial_audit=None,
                                   rebind_page=None, log=print) -> dict:
    """Drain a hash-bound finite walk; its receipt remains authoritative across process reloads."""
    if initial_audit is None:
        from . import relevance_contract as _R_initial_page
        initial_audit = _R_initial_page.evaluate_selection_relevance(proj, segs)
    last_audit = initial_audit
    initial = _semantic_recovery_cursor(
        proj, segs, initial_audit,
        allow_absent=True, allow_stale=True, adopt_legacy=True)
    if initial["status"] == "clear":
        return initial_audit
    if initial["status"] == "current":
        # A prior process may have died after atomically saving its page but before this repair.
        # Rebind only after the cursor and receipt prove which exact pool/artifacts were completed.
        if rebind_page is not None:
            rebind_page()
        terminal = _pagination_terminal_error(initial)
        if terminal is not None:
            raise terminal
        if not initial["deferred"]:
            return initial_audit

    while True:
        last_audit = recover_page()
        current = _semantic_recovery_cursor(
            proj, segs, last_audit,
            allow_absent=False, allow_stale=False, adopt_legacy=False)
        # Checkpoint signatures are writes. They may never bless a malformed, stale, cap-reset, or
        # repeated cursor, but a validated terminal page still needs its completed artifacts bound.
        if rebind_page is not None:
            rebind_page()
        if current["status"] == "clear" or not current["deferred"]:
            return last_audit
        terminal = _pagination_terminal_error(current)
        if terminal is not None:
            raise terminal
        receipt = current["receipt"]
        log(f"semantic-recovery: page {receipt['page_count']} complete; continuing "
            f"{len(current['deferred'])} audited deferred beat(s) inside this render "
            f"{current['deferred']}")


def produce_auto_resilient(project_dir, **kw):
    """produce_auto wrapped by the INCIDENT ADVISOR: on an unexpected technical failure the
    reasoning LLM (DeepSeek primary) triages the error with full context and may order a
    bounded retry FROM STAGE CHECKPOINTS (resume=True — completed stages are never redone).
    Content failures keep their own machinery: NonRetryableBuildError (footage feasibility)
    and VisionBackendError (billing/outage) are never retried here — retrying cannot fix
    them and the advisor's menu exists precisely to avoid such waste. Cap:
    VIDLORE_CLIPSTUDIO_INCIDENT_MAX (2) interventions per render; kill switch:
    VIDLORE_CLIPSTUDIO_INCIDENT_ADVISOR=0 (then this is produce_auto verbatim)."""
    import os as _os
    import time as _time
    from . import incident as _inc
    from .verify import NonRetryableBuildError, VisionBackendError
    progress = kw.get("progress")

    def _log(m):
        (progress or print)(m)

    attempt = 0
    while True:
        try:
            return produce_auto(project_dir, **kw)
        except (NonRetryableBuildError, VisionBackendError):
            raise                                        # content/billing — their own machinery
        except KeyboardInterrupt:
            raise
        except Exception as e:                           # noqa: BLE001 — unexpected technical failure
            proj = None
            try:
                proj = ClipProject.load(str(project_dir))
            except Exception:                            # noqa: BLE001
                pass
            _max = max(0, int(_os.environ.get("VIDLORE_CLIPSTUDIO_INCIDENT_MAX", "2") or 2))
            used = _inc.interventions_used(proj) if proj is not None else attempt
            if used >= _max:
                _log(f"incident-advisor: intervention cap ({_max}) reached — aborting")
                raise
            v = _inc.advise("produce_auto", e, proj=proj, log=_log)
            if v["action"] not in ("retry", "retry_after_wait"):
                raise
            if v["action"] == "retry_after_wait" and v.get("wait_s"):
                _log(f"incident-advisor: waiting {v['wait_s']}s before the checkpoint resume")
                _time.sleep(v["wait_s"])
            attempt += 1
            kw["resume"] = True
            _log(f"incident-advisor: retrying produce from stage checkpoints "
                 f"(intervention {used + 1}/{_max})")


def _persist_cost(project_dir, *, partial: bool, log=None) -> dict:
    """Write this job's spend to output/cost_report.json. Best-effort and NEVER raises: it runs
    from a finally, so an accounting hiccup must not replace the render's real exception.

    `partial=True` marks a run that ended on a raise (gate block, vision outage, crash). Those runs
    used to record NOTHING — the whole cost block sat after build — so the most expensive failure
    mode in the pipeline (verify + self-heal, then abort) was invisible in every cost report."""
    from pathlib import Path as _P
    try:
        from . import llm as _llm_c
        cost = _llm_c.usage_summary()
        if not cost.get("calls"):
            return cost
        cost["partial"] = bool(partial)
        out = _P(project_dir) / "output"
        out.mkdir(parents=True, exist_ok=True)
        import json as _json_c
        (out / "cost_report.json").write_text(_json_c.dumps(cost, indent=1), encoding="utf-8")
        from . import perf_metrics as _pm_c
        _pm_c.write_report(out / "perf_report.json")     # un-gated: the counters are free to dump
        if partial and log:
            log(f"cost: ~${cost['usd']:.2f} over {cost['calls']} call(s) spent before this render "
                f"stopped — recorded to cost_report.json (partial)")
        return cost
    except Exception:                                    # noqa: BLE001 — never mask the real error
        return {}


_LAST_ACCOUNTED = [""]      # project dir whose spend is currently accumulating in llm._USAGE


def produce_auto(project_dir, **kw) -> dict:
    """Per-job accounting scope around the pipeline.

    The portal is a long-lived process and `_USAGE` is module state, so without a reset every
    job's cost_report also contained every previous job's spend. And because the cost dump lived
    at the very END of the pipeline, any raise (release-block, vision outage) threw away the
    record of everything already spent. Both are fixed here, in a wrapper, so no raise path can
    skip accounting — and the finally never masks the original exception.

    The reset is deliberately NOT unconditional: a resume of the SAME project (the incident
    advisor's checkpoint retry, the portal's /retry button) must keep accumulating, because the
    true price of a render includes the attempt that failed — that abort→retry cycle is the
    single most expensive pattern the cost audit found. A different project always resets."""
    _same_job = (str(project_dir) == _LAST_ACCOUNTED[0]) and bool(kw.get("resume"))
    from . import llm as _llm_w
    if not _same_job:
        _llm_w.reset_usage()
    _LAST_ACCOUNTED[0] = str(project_dir)
    # PREVENT the sleep the pipeline has only ever REPORTED. Job 218acdfe10's resume slept
    # 17+17+17+17+50 minutes — 118 of 420 — every one inside verify, the stage that waits on remote
    # vision answers and so looks perfectly idle to the OS. Held here rather than inside
    # _produce_auto so that every exit path, raise included, releases it: the portal is a long-lived
    # process and a leaked assertion would keep the machine awake long after the render ended.
    from .keep_awake import KeepAwake as _KeepAwake
    _awake = _KeepAwake().start(log=kw.get("progress"))
    _ok = False
    try:
        r = _produce_auto(project_dir, **kw)
        _ok = True
        return r
    finally:
        _awake.stop()
        if not _ok:                                      # the success path records it itself
            _persist_cost(project_dir, partial=True, log=kw.get("progress"))


def _log_time_breakdown(log, pm, slept_s: float = 0.0, top: int = 8) -> dict:
    """Print where a render's wall-clock went, in the log, at the end of every render.

    perf_report.json has carried per-stage durations all along, and nobody opens a JSON while
    watching a render crawl — so "why is this taking so long" has been answered by hand-parsing
    build.log three separate times, each one rediscovering the same two facts: an idle-slept
    machine, and one rung asking its questions serially. Print it instead.

    Strictly observational. The durations are the ones perf_metrics already recorded; the slept
    seconds are the ones the sleep watcher already measured. It cannot move a decision or an output
    byte, and it never raises: a report that breaks a render is worse than no report.

    Returns the aggregated {stage: seconds} so callers (and tests) can assert on it.
    """
    rows: dict = {}
    try:
        pm.stage("done")                          # close the final open stage before reading
        for r in (pm.snapshot() or {}).get("stages", []):
            name = str(r.get("stage", "?"))
            rows[name] = rows.get(name, 0.0) + float(r.get("dur_s", 0.0) or 0.0)
        total = sum(rows.values())
        if total <= 0:
            return rows
        log(f"time: where this render went — {total / 60:.0f} min of pipeline"
            + (f", of which {slept_s / 60:.0f} min the machine was ASLEEP" if slept_s > 60 else ""))
        for name, secs in sorted(rows.items(), key=lambda kv: -kv[1])[:top]:
            if secs >= 1.0:
                log(f"time:   {secs / 60:6.1f} min  {100 * secs / total:4.0f}%  {name}")
        if slept_s > 60:
            log("time:   that sleep is INSIDE those stages — a power assertion is held for the "
                "length of a render (VIDLORE_CLIPSTUDIO_KEEP_AWAKE=0 disables it), so this line "
                "should normally not appear at all")
    except Exception as exc:                      # noqa: BLE001 — a report, never a gate
        try:
            log(f"time: breakdown unavailable ({type(exc).__name__})")
        except Exception:                         # noqa: BLE001
            pass
    return rows


def _produce_auto(project_dir, *, topic: str = "", script_path: Optional[str] = None,
                  script_text: Optional[str] = None, movie_hint: str = "",
                  policy: str = "block", max_sources: int = 6, cfg: Optional[ClipConfig] = None,
                  theme: str = "history", voice: str = "", title: str = "",
                  captions: Optional[bool] = None, caption_style: str = "",
                  voiceover: Optional[str] = None, voice_provider: str = "", voice_preset: str = "",
                  use_tts: bool = True, verify: bool = True, do_build: bool = True,
                  force_index: bool = False, resume: bool = False, progress=None) -> dict:
    """FULL AUTO: topic + script in → finished video out. Discovers + downloads its own sources.
    analyze → discover → download(policy) → Face-ID refs → deep index → match → cut → AI verify
    → ledger + QC report → assemble.

    resume=True reuses an existing project dir and SKIPS every stage that already finished with the
    same inputs (checkpoints in proj.meta['pipeline']), so a render that died — or content-blocked —
    at assembly continues from where it stopped instead of redoing hours of index/match/verify. The
    final assemble always re-runs (a killed render leaves no usable output)."""
    from .analyze import analyze_script, ScriptAnalysis, revalidate_cached_directions
    from .discover import discover_sources
    from .download import download_candidates
    from . import faceid as _faceid
    from . import verify as _verify
    from . import review as _review
    from . import llm as _llm

    cfg = cfg or load_clip_config()
    eng = engine_config()
    project_dir = Path(project_dir)

    from . import perf_metrics as _pm_stage

    def log(m):
        # decision-neutral stage-duration marks, driven by the existing "N/9 ·" progress lines
        try:
            _s = str(m)
            if "·" in _s and _s.split("/")[0].strip().rstrip("ab").isdigit():
                _stage_name = _s.split("·", 1)[1].strip()[:48]
                _pm_stage.stage(_stage_name)
                # cost attribution rides the SAME stage marks — this is what makes
                # cost_report's per-stage breakdown real instead of an empty {}
                from . import llm as _llm_stage
                _llm_stage.set_stage(_stage_name)
        except Exception:
            pass
        if progress:
            progress(m)

    if (project_dir / "project.json").exists():
        proj = ClipProject.load(project_dir)
    else:
        proj = ClipProject(name=project_dir.name, root=str(project_dir))
    proj.ensure_dirs()

    # SLEEP DETECTION (log/meta only — zero effect on any decision): a clamshell/idle sleep
    # mid-render froze the process for 85 min on a measured run and the stall was
    # indistinguishable from slow code in every timing report. CLOCK_UPTIME_RAW ticks only
    # while the machine is awake; its drift against wall time IS the slept time.
    import threading as _thr_sl
    import time as _time_sl
    _sleep_state = {"stop": False, "total": 0.0}

    def _sleep_watch():
        try:
            up0, w0 = _time_sl.clock_gettime(_time_sl.CLOCK_UPTIME_RAW), _time_sl.time()
        except (AttributeError, OSError):
            return                               # clock unavailable → no watcher, no harm
        while not _sleep_state["stop"]:
            _time_sl.sleep(30)
            try:
                up1, w1 = _time_sl.clock_gettime(_time_sl.CLOCK_UPTIME_RAW), _time_sl.time()
            except OSError:
                return
            drift = (w1 - w0) - (up1 - up0)
            if drift > 60:
                _sleep_state["total"] += drift
                log(f"⏸ system SLEPT ~{drift / 60:.0f} min mid-render (lid closed / idle "
                    f"sleep) — wall-clock timings include this; the render itself is fine")
                try:
                    proj.meta["sleep_seconds"] = round(
                        float(proj.meta.get("sleep_seconds", 0.0)) + drift, 1)
                except Exception:
                    pass
            up0, w0 = up1, w1

    _thr_sl.Thread(target=_sleep_watch, daemon=True).start()

    # VISUAL-FILTER REQUIREMENT (fail-CLOSED). The animated / video-game / toy footage gate
    # (_source_is_nonphotographic -> _photographic_ok, verified to correctly flag Telltale 'Game of
    # Thrones' game cut-scenes as art) is CLIP-based. Without the CLIP model it SILENTLY fails open,
    # so a machine with no CLIP — a fresh Windows box has an EMPTY ~/.cache/vidlore_clip and there is
    # no auto-download — shipped a render FULL of cartoonish game cut-scenes. Refuse up front rather
    # than ship footage we cannot visually verify. Escape hatch: VIDLORE_CLIPSTUDIO_ALLOW_NO_CLIP=1.
    import os as _oscl
    from .index import clip_available as _clip_ok
    if not _clip_ok() and _oscl.environ.get("VIDLORE_CLIPSTUDIO_ALLOW_NO_CLIP", "").strip() \
            not in ("1", "true", "yes"):
        raise PipelineError(
            "CLIP visual model not loaded — the animated / video-game / toy footage filter cannot run, "
            "so this render would silently include cartoon/game footage (e.g. Telltale 'Game of Thrones' "
            "cut-scenes). Put the CLIP models (clip_vision.onnx, clip_text.onnx, tokenizer.json) in "
            "~/.cache/vidlore_clip (or set VIDLORE_CLIP_DIR to their folder), then retry. To render anyway "
            "with degraded footage filtering, set VIDLORE_CLIPSTUDIO_ALLOW_NO_CLIP=1.")

    # purge any cached reaction/essay/non-show sources left over from before the title-gate fix
    _purged = _purge_unwanted_sources(proj, log)
    if _purged:
        log(f"  [purge] blocked {_purged} cached unwanted source(s) before render")
        proj.save()

    if resume:
        log("resume: reusing project — completed stages will be skipped where inputs are unchanged")

    # 1 — analyze
    log("1/9 · analyze script")
    if script_text is None and script_path:
        script_text = Path(script_path).read_text(encoding="utf-8")
        proj.script_path = str(script_path)
    if not (script_text or "").strip():
        raise PipelineError("no narration script provided.")
    from . import policy as _policy
    _sig_analyze = _sig(script_text, topic, movie_hint)
    # TRUSTED RESUME (portal-restart recovery): the raw script text lives ONLY in the portal's
    # in-memory job registry — a restarted portal (or a CLI resume of a portal job) can never
    # reproduce the analyze signature even though the complete analyze ARTIFACT (analysis +
    # segments) is cached on disk. With the env set, a resume adopts the RECORDED signature when
    # that artifact is complete, so analyze skips exactly as the in-process retry would and every
    # downstream signature reproduces identically. The script is NOT re-validated against the
    # cached beats — only use on a project whose cached analysis is known-good; the log is loud.
    import os as _os_tr
    if resume and _os_tr.environ.get("VIDLORE_CLIPSTUDIO_RESUME_TRUST_CACHED", "0").strip() \
            in ("1", "true", "yes"):
        try:
            _rec_sig = str((((proj.meta.get("pipeline") or {}).get("stages") or {})
                            .get("analyze") or {}).get("sig") or "")
        except Exception:
            _rec_sig = ""
        if _rec_sig and _rec_sig != _sig_analyze \
                and proj.segments and proj.meta.get("analysis"):
            log(f"resume: TRUSTED-CACHE — adopting the recorded analyze signature ({_rec_sig}); "
                f"the provided script text was NOT re-validated against the cached beats")
            _sig_analyze = _rec_sig
    if _stage_skip(proj, "analyze", _sig_analyze, resume,
                   artifact_ok=bool(proj.segments and proj.meta.get("analysis"))):
        analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
        segs = proj.segments
        # Analyzer directives are cached model output, but the deterministic grounding guards are
        # code.  Re-apply today's guards on Resume so an older exact-scene storyboard cannot evade
        # a newly fixed contract merely because its analyze checkpoint is valid.  Persist even a
        # no-op pass: it records the guard schema/provenance that makes the next Resume idempotent.
        _cached_guard = revalidate_cached_directions(segs, analysis)
        _tally = _policy.finalize_beats(segs)          # idempotent re-classify (cheap; for the tally)
        proj.segments = segs
        proj.meta["analysis"] = analysis.to_dict()
        proj.save()
        log("  cached analyzer guards → "
            f"checked:{_cached_guard.get('exact_revalidated', 0)} · "
            f"changed:{_cached_guard.get('changed_count', 0)} "
            f"{_cached_guard.get('changed_indices', [])}")
        log(f"  ↻ skipped (resume) — movie={analysis.movie_title!r} · {len(segs)} beats cached")
    else:
        analysis, segs = analyze_script(script_text, topic=topic, movie_hint=movie_hint,
                                        eng_cfg=eng, cfg=cfg, progress=progress)
        proj.segments = segs
        # unified per-beat VISUAL POLICY — classify every beat once; all downstream stages obey it
        _tally = _policy.finalize_beats(segs)
        proj.meta["analysis"] = analysis.to_dict()
        _stage_done(proj, "analyze", _sig_analyze)
        log(f"  movie={analysis.movie_title!r} · {len(analysis.actors)} actors · {len(segs)} beats")
        log(f"  beat policy → " + " · ".join(f"{k}:{v}" for k, v in sorted(_tally.items())))

    # PREFLIGHT VISION HEALTH — fail in SECONDS, not after ~1.5h of discover/download/index/match,
    # when footage verification is impossible up front. Footage QC is a hard requirement (an exact
    # beat that can't be verified release-blocks), so a dead vision backend dooms the render before
    # it starts. Only a HARD outage (billing/auth) aborts here; a transient blip proceeds and the
    # per-beat breaker handles it. Skipped when verify is off or when resuming past a good verify.
    # Only on a FRESH render (not a resume — the resume path's own verify re-run + the post-verify
    # outage detector cover that case; the skip signatures aren't computed until after download).
    import os as _os_pf0
    if verify and not resume and _os_pf0.environ.get(
            "VIDLORE_CLIPSTUDIO_VISION_PREFLIGHT", "1").strip() not in ("0", "false", "no"):
        try:
            _pf_ok, _pf_kind = _llm.vision_probe(eng)
        except Exception:
            _pf_ok, _pf_kind = True, "ok"                 # never let the probe itself kill a render
        if not _pf_ok and _pf_kind in ("billing", "auth"):
            _pf_hint = ("the vision API is OUT OF CREDITS — top up billing (Gemini AI Studio or "
                        "Anthropic), then run again" if _pf_kind == "billing" else
                        "the vision API key is invalid/unauthorized — fix it in .env, then run again")
            log(f"⛔ VISION PREFLIGHT FAILED ({_pf_kind}) — {_pf_hint}. Stopping now (before "
                f"downloading/indexing) so no time is wasted on a render that cannot be verified.")
            from .verify import VisionBackendError
            raise VisionBackendError(
                f"vision backend unavailable ({_pf_kind}): {_pf_hint}.", kind=_pf_kind)

    # 2 — discover. The ranked head of the returned list = the requested download budget;
    # entity-coverage + anchor force-includes are APPENDED past it by discover, so the
    # download cap below covers the WHOLE returned list — any prefix cap would cut exactly
    # the must-download items. cfg is copied, never mutated (callers may share it across runs).
    # Deliberate: max_sources (the user's budget) overrides cfg.discover_target here — the env
    # knob still governs direct discover_sources() callers.
    log("2/9 · discover sources")
    import dataclasses as _dc
    from .discover import SourceCandidate as _SourceCandidate
    # scale the footage budget to the script: a long multi-scene essay needs far MORE distinct
    # sources than the user's default, or most beats re-use a few generic clips (wrong-scene).
    _budget = _scaled_source_budget(max_sources, len(segs), getattr(analysis, "video_type", ""))
    if _budget > max_sources:
        log(f"  long script ({len(segs)} beats) → scaling footage budget {max_sources} → {_budget} sources")
    cfg = _dc.replace(cfg, discover_target=max(4, _budget))
    _sig_discover = _sig(_sig_analyze, _budget)
    if _stage_skip(proj, "discover", _sig_discover, resume,
                   artifact_ok=bool(proj.meta.get("candidates"))):
        candidates = [_SourceCandidate.from_dict(d) for d in proj.meta.get("candidates", [])]
        log(f"  ↻ skipped (resume) — {len(candidates)} discovered source(s) cached")
    else:
        candidates = discover_sources(analysis, cfg, segments=segs, progress=progress)
        proj.meta["candidates"] = [c.to_dict() for c in candidates]
        if not candidates:
            raise PipelineError("source discovery found nothing — check connectivity / movie name.")
        _stage_done(proj, "discover", _sig_discover)

    # 3 — download (permission-policy gated). Download the FULL discovered set: discover's own
    # construction bounds it (head ≤ max_sources, coverage ≤ coverage_extra, anchors ≤ 4/scene),
    # and download applies the limit as a PREFIX slice — any smaller cap would slice off the
    # anchor force-includes appended last.
    dl_limit = len(candidates)
    log(f"3/9 · download (policy={policy}, budget={_budget}+coverage)")
    _sig_download = _sig(_sig_discover, policy)
    _usable0 = [s for s in proj.sources if s.status == "ok"]
    _refs_pre, _faceid_pre = None, None      # set when the index-overlap prewarmer builds refs early
    if _stage_skip(proj, "download", _sig_download, resume, artifact_ok=bool(_usable0)):
        usable = _usable0
        log(f"  ↻ skipped (resume) — {len(usable)} source(s) already downloaded")
    else:
        # SPEED (OPT-8): the download wall is ~100% deliberate 6s 403-pacing gaps — index each
        # completed source in a single prewarm worker during that dead wait. Face-ID refs are
        # built EARLY for it (they depend only on analysis identities, never on sources); the
        # regular stage 4 below reuses them. Worker is JOINED before anything else touches the
        # models. VIDLORE_CLIPSTUDIO_INDEX_OVERLAP=0 restores the serial order.
        _pw_main, _refs_pre, _faceid_pre = None, None, None
        if _index_overlap_on():
            try:
                log("4/9 · build Face-ID references (early — feeds the index prewarmer)")
                if _faceid.available():
                    _faceid_pre = _faceid.FaceID()
                    _refs_pre = _faceid.build_references(analysis.reference_identities(),
                                                         proj.index_dir, _faceid_pre,
                                                         progress=progress)
                else:
                    _refs_pre = {}
                _pw_main = _IndexPrewarmer(proj, cfg, references=_refs_pre, faceid=_faceid_pre,
                                           roster=analysis.actors, log=log)
            except Exception:                            # noqa: BLE001 — overlap is optional
                _pw_main, _refs_pre, _faceid_pre = None, None, None
        try:
            download_candidates(proj, candidates, cfg, policy=policy, limit=dl_limit,
                                progress=progress,
                                on_ready=(_pw_main.submit if _pw_main is not None else None))
        finally:
            if _pw_main is not None:
                _pw_main.close()
                if _pw_main.count:
                    log(f"5/9 · index prewarm — {_pw_main.count} source(s) indexed during the "
                        f"download window")
        usable = [s for s in proj.sources if s.status == "ok"]
        if not usable:
            raise PipelineError(f"no usable sources under policy={policy}. Use --policy approved_testing "
                             "(your explicit testing approval) or open_only (public-domain).")
        _stage_done(proj, "download", _sig_download)
    log(f"  {len(usable)} sources downloaded")

    # Downstream signatures — computable now that the source set is known. Each folds in the upstream
    # signature so any earlier change cascades a re-run. These also let us skip the whole
    # index/Face-ID/match/verify block when everything after download is already cached (the common
    # case for a content-blocked resume: jump straight to assemble).
    # MATCH_GATE_VERSION is folded into the helper: a resume must NOT replay selections chosen
    # before a new footage gate existed.  These are refreshed again after every pre-match source
    # mutator so checkpoints describe the pool actually matched, not its pre-backfill ancestor.
    _asr_signature = asr_semantic_fingerprint(proj, cfg)
    _sig_index, _sig_match, _sig_cut, _sig_verify, _sig_recover = \
        _footage_stage_signatures(
            _sig_download, usable, force_index=force_index, segments=segs, verify=verify,
            asr_signature=_asr_signature)
    _sel_complete = (bool(proj.selections)
                     and {s.segment_index for s in proj.selections} >= {s.index for s in segs}
                     and asr_pool_current(proj, cfg, usable))
    _skip_match = _stage_skip(proj, "match", _sig_match, resume, artifact_ok=_sel_complete)
    _skip_verify = _stage_skip(proj, "verify", _sig_verify, resume, artifact_ok=_sel_complete)
    _skip_recover = _stage_skip(proj, "recover", _sig_recover, resume, artifact_ok=_sel_complete)
    import os as _os_bf_stage
    _backfill_enabled = _os_bf_stage.environ.get(
        "VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED", "1").strip().lower() \
        not in ("0", "false", "no")
    try:
        _backfill_rounds = int(_os_bf_stage.environ.get(
            "VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "2") or 2)
    except (TypeError, ValueError):
        _backfill_rounds = 2
    _sig_backfill = _backfill_input_signature(
        _sig_download, segs, policy=policy, max_sources=max_sources,
        show_title=movie_hint or "", enabled=_backfill_enabled,
        rounds=max(1, _backfill_rounds), cfg=cfg, analysis=analysis)
    _backfill_artifact_ok = (not _backfill_enabled) or \
        _backfill_audit_complete_for(proj, _sig_backfill)
    _skip_backfill = _stage_skip(
        proj, "backfill", _sig_backfill, resume,
        artifact_ok=_backfill_artifact_ok)
    # Face-ID refs feed indexing (shot identities) + recovery re-indexing; match consumes persisted
    # shot identities, not the refs object. So they're only worth building when a footage stage will
    # actually run. A fully-cached resume (→ assemble only) skips this and the model load with it.
    # Backfill is a first-class resumable stage too.  A transient backfill failure must retry even
    # when the same failed run went on to checkpoint match/verify/recover; omitting it here made the
    # removed backfill checkpoint unreachable on the next Resume.
    _downstream_footage_pending = not (
        _skip_match and _skip_verify and _skip_recover)
    _need_footage_stages = _footage_stages_required(
        skip_match=_skip_match, skip_verify=_skip_verify, skip_recover=_skip_recover,
        backfill_enabled=_backfill_enabled, skip_backfill=_skip_backfill)

    # 4 — Face-ID references
    faceid_obj, refs = None, {}
    # OCR identity remains actor-keyed: character names in subtitles/recaps must never count as
    # proof that the character is visible. Indexing derives a separate actors+characters vocabulary
    # from project analysis exclusively for ASR proper-name decoding.
    roster = analysis.actors
    if _need_footage_stages:
        log("4/9 · build Face-ID references")
        if _refs_pre is not None:
            faceid_obj, refs = _faceid_pre, _refs_pre
            log("  ↻ reused (built early for the index prewarmer)")
        elif _faceid.available():
            faceid_obj = _faceid.FaceID()
            refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                            faceid_obj, progress=progress)
        else:
            # NOT silent. Without Face-ID the pipeline still renders, but identity recognition is
            # gone: no wrong-character rejection, no actor↔character roster matching, and the
            # rejected-footage gate loses its strongest evidence. On a fresh machine the cause is
            # always the same — the model files were never copied (they are gitignored, ~39 MB).
            log(f"⚠ 4/9 · Face-ID UNAVAILABLE — yunet.onnx / sface.onnx not found in "
                f"{_faceid.MODELS_DIR}. Identity recognition is OFF for this render: "
                f"wrong-character footage can no longer be rejected. Copy both models there "
                f"(or set VIDLORE_CLIPSTUDIO_MODELS) to restore it.")
    else:
        log("4/9 · Face-ID references — ↻ skipped (resume, all footage stages cached)")

    # 5 — deep index
    if _downstream_footage_pending:
        log("5/9 · deep index (ASR + scenes + CLIP + Face-ID + OCR + quality + dedup)")
        index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster,
                  force=(force_index and not resume), progress=progress)
    elif _need_footage_stages:
        # Backfill can be the only retryable stage after match/verify/recover were checkpointed.
        # Existing sources were already indexed for those stages; re-indexing the whole pool before
        # merely retrying discovery is wasted work.  Any newly downloaded replacement is indexed
        # transactionally inside `_backfill_rejected_sources` before it can affect match.
        log("5/9 · deep index — ↻ skipped (resume, cached pool; backfill retry only)")
    else:
        log("5/9 · deep index — ↻ skipped (resume, all footage stages cached)")

    # 5a2 — ANCHOR COVERAGE (single_scene): the whole essay orbits ONE scene, yet the global
    # discovery selection can spend every slot on look-alike uploads of OTHER scenes (measured:
    # an S8E4 night-courtyard essay shipped with S8E2 daytime-yard footage under its anchor
    # narration — 44 wrong-scene beats — because no dedicated anchor upload survived selection).
    # Deterministic check + direct ytsearch top-up BEFORE match, so the pool is right from the
    # first pick rather than healed afterwards.
    if _need_footage_stages and not _skip_match:
        try:
            _ensure_anchor_coverage(
                proj, analysis, cfg, policy=policy, roster=roster, log=log)
        except Exception as e:                                   # noqa: BLE001
            log(f"5a2/9 · anchor coverage: skipped ({str(e)[:80]})")

    # 5b — backfill footage the pool gates will reject. Runs BEFORE match, because at match time the
    # discovery budget is already spent and a dropped source just leaves a hole nothing fills.
    if _need_footage_stages and (
            not _skip_match or (_backfill_enabled and not _skip_backfill)):
        if _skip_backfill:
            log("5b/9 · backfill — ↻ skipped (resume, completed clean-copy search cached)")
        elif not _backfill_enabled:
            log("5b/9 · backfill — disabled by operator configuration")
            _stage_done(proj, "backfill", _sig_backfill)
        else:
            _backfill_complete = _run_backfill_invocation(
                proj, _sig_backfill,
                lambda: _backfill_rejected_sources(
                    proj, segs, analysis, cfg, refs=refs, faceid_obj=faceid_obj, roster=roster,
                    policy=policy, max_sources=max_sources, show_title=movie_hint or "", log=log,
                    input_sig=_sig_backfill),
                log=log)
            if _backfill_complete:
                _stage_done(proj, "backfill", _sig_backfill)
            else:
                # A transient discovery/download/index failure is retryable.  Never bless it as a
                # completed quality search merely to make Resume faster.
                _ckpt(proj)["stages"].pop("backfill", None)
                proj.save()

        # Anchor/backfill are source-pool mutators.  Refresh every downstream signature before
        # recording match/cut/verify/recover checkpoints so they describe the pool actually used.
        usable = [s for s in proj.sources if s.status == "ok"]
        # Once this run observed stale/missing ASR, keep downstream checkpoints invalid even after
        # targeted indexing repairs the files: refreshed transcripts can change the right window,
        # so match/verify/recover must consume them once before a later Resume may trust the cache.
        # A newly-added anchor/backfill source can still turn an initially-current pool stale here.
        _sel_complete = bool(_sel_complete and asr_pool_current(proj, cfg, usable))
        _sig_index, _sig_match, _sig_cut, _sig_verify, _sig_recover = \
            _footage_stage_signatures(
                _sig_download, usable, force_index=force_index, segments=segs, verify=verify,
                asr_signature=_asr_signature)
        # A pending backfill can be the sole reason we reached this block while the old downstream
        # checkpoints were initially valid.  If it admitted a source, the refreshed signatures
        # must invalidate those checkpoints now; if it made no pool change, the cached work remains
        # reusable for this attempt.
        _skip_match = _stage_skip(
            proj, "match", _sig_match, resume, artifact_ok=_sel_complete)
        _skip_verify = _stage_skip(
            proj, "verify", _sig_verify, resume, artifact_ok=_sel_complete)
        _skip_recover = _stage_skip(
            proj, "recover", _sig_recover, resume, artifact_ok=_sel_complete)

    # Anchor coverage and backfill can add sources after the main index_all audit. Their individual
    # index calls preserve useful visual artifacts on ASR failure, but matching must not consume or
    # checkpoint an incomplete transcript pool. Stop here, before any downstream stage is blessed.
    if _need_footage_stages and not _skip_match \
            and not asr_pool_current(proj, cfg, usable):
        _asr_audit = asr_pool_cache_audit(proj, cfg, usable)
        _asr_bad = [row.get("source_id", "") for row in _asr_audit.get("invalid", [])[:8]]
        raise PipelineError(
            f"post-acquisition ASR evidence incomplete for "
            f"{len(_asr_audit.get('invalid', []))}/{_asr_audit.get('source_count', 0)} "
            f"usable source(s): {_asr_bad}; match checkpoints were not written")

    # 6 — match
    if _skip_match:
        log("6/9 · match — ↻ skipped (resume, selections cached)")
    else:
        log("6/9 · match")
        match_segments(proj, segs, cfg, analysis=analysis, progress=progress)
        _stage_done(proj, "match", _sig_match)         # closes the historic match→cut save gap

    # 7 — cut
    if _skip_match and _stage_skip(proj, "cut", _sig_cut, resume, artifact_ok=_sel_complete):
        log("7/9 · cut — ↻ skipped (resume, clips cached)")
    else:
        log("7/9 · cut")
        # Existing seg_NNN files are reusable only when the persisted match itself was reused.
        # If match ran in this attempt, its deterministic segment indices may now name different
        # source windows; passing the job-level Resume flag here silently returned those stale files
        # without cutting the new selections (the 101-beat reproduction completed this stage in
        # 0.439s).  A partial cut may still resume cheaply when match was genuinely skipped.
        cut_all(proj, cfg, resume=bool(resume and _skip_match), progress=progress)
        _stage_done(proj, "cut", _sig_cut)

    # 8 — AI verify + repair
    _verify_down = False
    if verify and _skip_verify:
        log("8/9 · AI verify + repair — ↻ skipped (resume, verdicts cached)")
    elif verify:
        log("8/9 · AI verify + repair")
        # The proven 4-worker verdict prefetch was only enabled by the portal and CLI entrypoints;
        # direct produce_auto callers (tools scripts, tests driving real renders) ran verify fully
        # serial — measured 28 min on a render whose warm prefetch cost is minutes. Scoped set +
        # restore (not a bare setdefault) so the env never leaks into later direct
        # verify_and_repair callers in the same process (the outage suite's serial call-count
        # contract depends on workers=1).
        import os as _os_vw
        _vw_unset = "VIDLORE_CLIPSTUDIO_VERIFY_WORKERS" not in _os_vw.environ
        if _vw_unset:
            _os_vw.environ["VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"] = "4"
        try:
            _vres = _verify.verify_and_repair(proj, segs, cfg, eng, progress=progress) or {}
        finally:
            if _vw_unset:
                _os_vw.environ.pop("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", None)
        # VISION-BACKEND OUTAGE (billing / bad key / persistent down): the whole stage errored, so
        # every exact beat is unresolved. Do NOT checkpoint this stage 'done' — that made a later
        # Resume SKIP verify and re-hit the block forever. Do NOT grind hours of image-fallback
        # against a dead API (measured: ~6.8h of 'unverified' stills). Fail FAST + retryable with an
        # actionable message; once the backend is restored, Resume re-runs verify and completes.
        if _vres.get("verifier_down"):
            _verify_down = True
            _err = int(_vres.get("errored", 0))
            _att = int(_vres.get("attempted", 0)) or 1
            _kind = "down"
            try:
                _ok_probe, _kind = _llm.vision_probe(eng)   # classify billing vs auth vs blip
                if _ok_probe:
                    _kind = "down"
            except Exception:
                _kind = "down"
            _hint = {"billing": "the vision API is OUT OF CREDITS — top up billing (Gemini AI "
                                 "Studio or Anthropic), then click Resume",
                     "auth": "the vision API key is invalid/unauthorized — fix the key in .env, "
                             "then click Resume",
                     "down": "the vision backend is unreachable — check your connection, then "
                             "click Resume"}.get(_kind, "restore the vision backend, then Resume")
            # NEVER checkpoint an errored verify (so Resume re-verifies once the backend is back),
            # and NEVER grind the image-fallback stage against a dead API (raising below / the
            # _verify_down guard on stage 8a/8b both skip it — hours → seconds).
            _review = _os_pf0.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block") \
                .strip().lower() == "warn"
            if _review:
                # explicit review-draft: the user asked for a watchable draft anyway. Proceed to
                # assembly, which in warn mode airs the CLIP/Face-ID/transcript-matched footage
                # LOUDLY flagged 'not vision-verified · not for publication'. Grind still skipped.
                log(f"⚠ VISION BACKEND UNAVAILABLE ({_kind}) — {_err}/{_att} beats unverified; "
                    f"REVIEW DRAFT mode airs matched footage flagged (not for publication). "
                    f"{_hint} for a verified build.")
            else:
                log(f"⛔ VISION BACKEND UNAVAILABLE ({_kind}) — {_err}/{_att} beats could not be "
                    f"verified. {_hint}. Verify was NOT checkpointed, so Resume re-verifies "
                    f"(no re-download / re-index / re-match).")
                from .verify import VisionBackendError
                raise VisionBackendError(
                    f"vision backend unavailable ({_kind}): {_err}/{_att} exact-scene beats could "
                    f"not be verified. {_hint}.", kind=_kind)
        elif int(_vres.get("materialization_errors", 0) or 0) > 0:
            # Vision reached a positive alternate, but its deterministic clip could not be cut and
            # was transactionally rolled back. This is disk/ffmpeg infrastructure uncertainty, not
            # a semantic rejection and not a completed verify stage. Leave the checkpoint absent so
            # Resume retries from the restored selection/clip instead of silently keeping stale or
            # partial bytes under promoted metadata.
            _mat_err = int(_vres.get("materialization_errors", 0) or 0)
            log(f"⛔ VERIFY PROMOTION MATERIALIZATION FAILED — {_mat_err} promotion(s) rolled "
                "back; verify was NOT checkpointed. Fix ffmpeg/disk and Resume.")
            raise PipelineError(
                f"verify promotion materialization failed for {_mat_err} beat(s); "
                "selection and clip were rolled back, Resume will retry")
        else:
            _stage_done(proj, "verify", _sig_verify)

    # 8a/8b — BOUNDED RECOVERY + IMAGE FALLBACK. Grouped under ONE 'recover' checkpoint: on resume
    # with unchanged inputs both are skipped (their outcomes — recovered footage, installed stills —
    # are already persisted on the selections). This is the expensive tail (rediscovery + web-image
    # verification) that a content-blocked render should never repeat just to re-block at assembly.
    import os as _os
    if _verify_down:
        # the vision backend is dead — recovery re-verification and web-still verdicts would ALL
        # come back 'unverified' (measured: ~6.8h of doomed stills). Skip the whole grind; the
        # matched footage stands and the warn-mode assembly airs it flagged.
        log("8a/8b · recovery + image fallback — ⏭ skipped (vision backend down; would only "
            "produce unverified stills)")
    elif _skip_recover:
        log("8a/8b · recovery + image fallback — ↻ skipped (resume, outcomes cached)")
    else:
        # 8a — BOUNDED AUTONOMOUS RECOVERY (R4-5): for exact beats the verifier could NOT satisfy, try
        # ONE targeted rediscovery→download→index→rematch→re-cut→reverify round BEFORE falling back to a
        # still/hold/block. Prefers an exact clip that verifies over an irrelevant HD one; preserves
        # every already-resolved beat; fail-closed. The deterministic still (8b) and the build-stage
        # editorial hold / honest release-block remain the downstream last resorts.
        _scoped_recovery_conclusive = False
        try:
            _recover_unresolved_beats(proj, segs, analysis, cfg, eng, faceid_obj=faceid_obj, refs=refs,
                                      roster=roster, policy=policy, log=log)
            _scoped_recovery_conclusive = True
        except Exception as _e:
            _log_stage_skip(log, proj, "recovery", _e)

        # 8b — EXACT-SCENE IMAGE FALLBACK: beats that still have no relevant footage (no source,
        # low confidence, or a confirmed wrong-character pick) get a face/CLIP-verified still from
        # the open web (Bing/DDG/Wikimedia). A wrong image is worse than generic footage, so the
        # fetcher only accepts what it can verify; otherwise the beat keeps its clip.
        if _os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip() not in ("0", "false", "no"):
            try:
                _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log, eng_cfg=eng)
            except Exception as _e:
                _log_stage_skip(log, proj, "image-fallback", _e)
        # Scoped recovery can add an indexed source after global match/cut/verify.  Its transaction
        # rematches only blocked beats, materializes accepted changes, and restores every other
        # selection. Rebind those completed artifacts to the enlarged current pool before writing
        # recover's checkpoint; otherwise the next Resume sees the source as an external mutation,
        # globally reruns all footage stages, and loses semantic pagination progress. A failed or
        # ASR-incomplete recovery never earns the rebind.
        if _scoped_recovery_conclusive:
            _usable_after_recovery = [s for s in proj.sources if s.status == "ok"]
            if asr_pool_current(proj, cfg, _usable_after_recovery):
                (_sig_index, _sig_match, _sig_cut, _sig_verify,
                 _sig_recover) = _footage_stage_signatures(
                    _sig_download, _usable_after_recovery, force_index=force_index,
                    segments=segs, verify=verify, asr_signature=_asr_signature)
                _rebind_completed_footage_stages(
                    proj,
                    {"match": _sig_match, "cut": _sig_cut,
                     **({"verify": _sig_verify} if verify else {})},
                    reason="conclusive_scoped_recovery_pool")
        _stage_done(proj, "recover", _sig_recover)

    def _refresh_qc():
        """Rewrite ledger/review from the current selections after any repair mutation."""
        _summ = ledger.finalize(proj, segs, cfg)
        _review_path = _review.write_review(proj, segs)
        proj.save()
        log(f"  QC · {_summ['flagged_for_review']}/{_summ['segments']} flagged · "
            f"mean conf {_summ['mean_confidence']}")
        return _summ, _review_path

    def _refresh_qc_without_masking_active_error():
        """Refresh final-state QC, but never replace an already-raised typed pipeline error."""
        import sys as _sys_qc
        _active_error = _sys_qc.exc_info()[0] is not None
        try:
            return _refresh_qc()
        except Exception as _qc_error:
            if not _active_error:
                raise
            log(f"QC refresh failed while preserving active pipeline error: "
                f"{type(_qc_error).__name__}: {str(_qc_error)[:100]}")
            return None

    def _recover_selection_relevance_or_raise(_original_error):
        """Drain bounded strict-semantic pages, then prove the current state passes.

        This belongs only to the pre-assembly semantic preflight. Technical failures remain
        retryable/fail-closed, recovery may never demote policy, and a claimed clear result is
        checked again against persisted state. build_video's later independent assertion never
        recovers in the same attempt: if legacy repair regresses relevance, it hard-fails and Resume
        can begin the next bounded generation.
        """
        import os as _os_sr
        from .verify import NonRetryableBuildError as _NRBE_sr

        if (not isinstance(_original_error, _NRBE_sr)
                or getattr(_original_error, "kind", "") != "selection_relevance"
                or _os_sr.environ.get("VIDLORE_CLIPSTUDIO_SEMANTIC_RECOVERY", "1")
                .strip().lower() in ("0", "false", "no")):
            raise _original_error

        # A completed semantic page is the same scoped transaction as stage 8a, only later in the
        # pipeline. It may have added an unusable-but-indexed source while retaining an audited
        # deferred tail. Bind the already-reconciled artifacts to that pool so Resume continues the
        # tail directly instead of rebuilding Face-ID/global match/cut/verify and restarting at the
        # first page. Never bless an incomplete ASR pool or an absent prior stage checkpoint.
        def _rebind_semantic_page():
            _usable_after_semantic = [s for s in proj.sources if s.status == "ok"]
            if asr_pool_current(proj, cfg, _usable_after_semantic):
                (_sr_sig_index, _sr_sig_match, _sr_sig_cut, _sr_sig_verify,
                 _sr_sig_recover) = _footage_stage_signatures(
                    _sig_download, _usable_after_semantic, force_index=force_index,
                    segments=segs, verify=verify, asr_signature=_asr_signature)
                _rebind_completed_footage_stages(
                    proj,
                    {"match": _sr_sig_match, "cut": _sr_sig_cut,
                     **({"verify": _sr_sig_verify} if verify else {}),
                     "recover": _sr_sig_recover},
                    reason="conclusive_semantic_recovery_pool")

        try:
            # The assertion that invoked recovery atomically wrote this exact viewer-facing audit.
            # Reuse it to validate any persisted cursor *before* spending the next page.  A mocked
            # or legacy caller may lack the file; only that absence falls back to a fresh read-only
            # evaluation. Corrupt JSON is a technical fault, never permission to reset the walk.
            import json as _json_sr_cursor
            _sr_audit_path = proj.output_dir / "selection_relevance_audit.json"
            if _sr_audit_path.is_file():
                try:
                    _sr_initial_audit = _json_sr_cursor.loads(
                        _sr_audit_path.read_text(encoding="utf-8"))
                except Exception as _cursor_audit_error:
                    raise PipelineError(
                        "semantic recovery could not read the preflight relevance audit: "
                        f"{type(_cursor_audit_error).__name__}: {_cursor_audit_error}") \
                        from _cursor_audit_error
            else:
                from . import relevance_contract as _R_sr_initial
                _sr_initial_audit = _R_sr_initial.evaluate_selection_relevance(
                    proj, segs, cfg=cfg)
            _drain_semantic_recovery_pages(
                proj, segs,
                lambda: _retry_selection_relevance(
                    proj, segs, cfg, analysis, eng, faceid_obj=faceid_obj, refs=refs,
                    roster=roster, policy=policy, log=log),
                initial_audit=_sr_initial_audit,
                rebind_page=_rebind_semantic_page, log=log)
        except Exception as _se:                     # technical repair fault cannot bypass gate
            _raise_semantic_recovery_failure(_original_error, _se, log)

        # Never trust the retry helper's summary as an authorization. Re-run the exact publication
        # assertion against the now-persisted project after the audited cursor is empty, whether the
        # helper reported blockers or a clear page. This is also what build_video independently does
        # before any encoding.
        from .relevance_contract import assert_selection_relevance as _assert_sr_recovered
        return _assert_sr_recovered(
            proj, segs, proj.output_dir / "selection_relevance_audit.json", cfg=cfg)

    # STRICT SEMANTIC PREFLIGHT — this must precede the older rejected-footage predictor on every
    # build. A pagination marker is only a progress hint and can legitimately be absent or stale
    # after match/cut reruns; using it to decide ordering let legacy self-heal deadlock a newly
    # restored exact beat before semantic retry was reached. This assertion is the same contract
    # build_video starts with, runs before encoding, and is followed by that independent in-build
    # assertion after any legacy repair mutates selections.
    if do_build:
        from .relevance_contract import assert_selection_relevance as _assert_sr_preflight
        try:
            try:
                _assert_sr_preflight(
                    proj, segs, proj.output_dir / "selection_relevance_audit.json", cfg=cfg)
            except RuntimeError as _semantic_block:
                try:
                    _recover_selection_relevance_or_raise(_semantic_block)
                except RuntimeError as _semantic_final:
                    _review_mode = _os.environ.get(
                        "VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower() == "warn"
                    if (_review_mode
                            and getattr(_semantic_final, "kind", "") == "selection_relevance"):
                        log("⚠ semantic preflight (mode=warn, REVIEW BUILD — not for publication) "
                            f"— {_semantic_final} Continuing only because review mode was explicitly "
                            "requested; the delivered filename will be marked REVIEW_DRAFT.")
                    else:
                        raise
        finally:
            # Recovery may rematch/re-cut even when the strict re-assertion still blocks. Keep the
            # terminal failure's ledger/review aligned with that final persisted state.
            _qc = _refresh_qc_without_masking_active_error()
            if _qc is not None:
                summ, review_path = _qc
    else:
        # Preserve audit-only/no-build behavior: one QC snapshot after normal recovery stages.
        summ, review_path = _refresh_qc()

    # PRE-ASSEMBLE FEASIBILITY GATE (fail-fast). The rejected-footage release gate lives at the END of
    # assembly, so on a footage-gap render it fired only AFTER ~20 min of per-beat re-encoding — all of
    # it thrown away. That verdict is deterministic from the (now frozen) selections, so surface it HERE
    # and abort before assembly. Structural-only + fail-open: it re-raises the SAME NonRetryableBuildError
    # the build gate would, but never blocks a render the build gate would let through (see build.py).
    if do_build:
        try:
            from .build import preassemble_release_block_reason
            _pre = preassemble_release_block_reason(proj, segs, analysis)
        except Exception as _e:                          # never let the fast-path itself break a render
            _pre = None
            _log_stage_skip(log, proj, "pre-assemble gate", _e)
        # SELF-HEALING FOOTAGE RECOVERY — the automated gate-unblock playbook (verified stills
        # at the venue bar → targeted LLM-queried acquisition → retry), bounded rounds, never
        # weakens the gate. Manual on job olenna_v2_allfixes it took 17 blocked beats to 0;
        # this makes that loop the pipeline's own behavior. VIDLORE_CLIPSTUDIO_SELFHEAL=0 off.
        if _pre:
            try:
                _pre = _run_preassemble_selfheal(
                    proj, segs, cfg, analysis, policy=policy, pre=_pre,
                    faceid_obj=faceid_obj, refs=refs, roster=roster, log=log)
            finally:
                # The fail-fast error must ship final-state QC too, even when self-heal raises or
                # returns an unresolved blocker.
                _qc = _refresh_qc_without_masking_active_error()
                if _qc is not None:
                    summ, review_path = _qc
        if _pre:
            _mode = _os.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower()
            if _mode == "warn":
                log(f"⚠ pre-assemble feasibility (mode=warn) — {_pre} Building a REVIEW draft anyway.")
            else:
                log(f"⛔ pre-assemble feasibility — {_pre} Aborting BEFORE assembly (fail-fast; the "
                    f"final render would release-block after ~20 min of wasted encoding). Add footage "
                    f"for these scenes or set RELEASE_BLOCK_MODE=warn for a review draft.")
                from .verify import NonRetryableBuildError
                # Typed for the same reason as the branch above it: warn mode already builds a
                # draft here, so the driver must be able to recognise this as a content verdict and
                # offer one. It is the SOUND subset of the rejected-footage gate, predicted early.
                raise NonRetryableBuildError(_pre, kind="preassemble_feasibility")

    out = None
    if do_build:
        log("9/9 · assemble final video")
        try:
            out = build_video(proj, segs, cfg, voice=voice, captions=captions,
                              caption_style=caption_style,
                              title=title or analysis.movie_title or proj.name,
                              theme_name=theme, voiceover=voiceover, voice_provider=voice_provider,
                              voice_preset=voice_preset, use_tts=use_tts, progress=progress)
        except RuntimeError as _be:
            # The AUTHORITATIVE in-build release gate sees hold-cap failures the pre-gate
            # predictor cannot (it is a documented subset). Heal the audit's unresolved list
            # once and rebuild — same bounded machinery, gate stays the final word.
            import os as _os_bh
            from .verify import NonRetryableBuildError as _NRBE
            if (isinstance(_be, _NRBE) and getattr(_be, "kind", "") == "rejected_footage"
                    and _os_bh.environ.get("VIDLORE_CLIPSTUDIO_SELFHEAL", "1").strip().lower()
                    not in ("0", "false", "no")
                    and _os_bh.environ.get("VIDLORE_CLIPSTUDIO_SELFHEAL_BUILD_RETRY", "1")
                    .strip().lower() not in ("0", "false", "no")):
                from . import selfheal as _selfheal_b
                _idxs = _selfheal_b.blocked_indexes(proj)
                log(f"self-heal: in-build release gate blocked {len(_idxs)} beat(s) — healing "
                    f"and rebuilding once")
                try:
                    _n = _selfheal_b.heal_blocked_beats(
                        proj, segs, cfg, blocked=_idxs, policy=policy,
                        faceid_obj=faceid_obj, refs=refs, roster=roster, log=log)
                except Exception as _he:          # noqa: BLE001 — see below; never BaseException
                    # The repair pass is OPTIONAL (env SELFHEAL_BUILD_RETRY=0 turns it off), but an
                    # exception escaping it REPLACED the render's verdict: a content stop the portal
                    # would have delivered as a draft became whatever heal happened to raise, and
                    # the job ended with no file. Keep the original verdict and its identity; the
                    # heal fault rides along as __cause__ so it is diagnosable, and is shouted the
                    # same way a dead stage is — a repair pass that cannot run is a BUG, not weather.
                    _log_stage_skip("selfheal.in_build_retry", _he, log)
                    raise _be from _he
                finally:
                    _qc = _refresh_qc_without_masking_active_error()
                    if _qc is not None:
                        summ, review_path = _qc
                if not _n:
                    raise
                out = build_video(proj, segs, cfg, voice=voice, captions=captions,
                                  caption_style=caption_style,
                                  title=title or analysis.movie_title or proj.name,
                                  theme_name=theme, voiceover=voiceover,
                                  voice_provider=voice_provider,
                                  voice_preset=voice_preset, use_tts=use_tts, progress=progress)
            else:
                raise

    _pm_stage.write_report(proj.output_dir / "perf_report.json")

    # WHERE THE TIME WENT, in the log, every render. perf_report.json has carried per-stage
    # durations all along and nobody reads a JSON while watching a render crawl — so "why is this
    # taking so long" has been answered by hand-parsing build.log, three times now, each time
    # rediscovering the same two facts (an idle-slept machine, and one serial rung). Print it.
    # Strictly observational: durations already recorded by perf_metrics, plus the slept seconds
    # the watcher above measured. Nothing here can influence a decision or an output byte.
    _log_time_breakdown(log, _pm_stage, float(proj.meta.get("sleep_seconds", 0.0) or 0.0))

    # What this render actually spent. Hundreds of vision calls go into verify alone and none of it
    # used to be recorded, so cost was unanswerable and "run verify twice" was an unpriced decision.
    _cost = {}
    try:
        from . import llm as _llm_cost
        _cost = _llm_cost.usage_summary()
        if _cost.get("calls"):
            _per = " · ".join(
                f"{m.split('/')[-1]}: {d['calls']} calls ${d['usd']:.2f}"
                for m, d in sorted(_cost["models"].items(), key=lambda kv: -kv[1]["usd"]))
            log(f"cost: ~${_cost['usd']:.2f} over {_cost['calls']} LLM/vision call(s) "
                f"({_cost['prompt'] / 1000:.0f}k in / {_cost['completion'] / 1000:.0f}k out) — {_per}")
            log("cost: prices are configurable estimates "
                "(VIDLORE_CLIPSTUDIO_PRICE_<MODEL>_IN/_OUT), not a provider invoice")
        proj.meta["cost"] = _cost
        proj.save()
        import json as _json_cost
        (proj.output_dir / "cost_report.json").write_text(_json_cost.dumps(_cost, indent=1))
    except Exception as e:                                       # noqa: BLE001
        log(f"cost: accounting unavailable ({str(e)[:60]})")

    return {
        "cost": _cost,
        "project": str(proj.root),
        "summary": summ,
        "analysis": analysis.to_dict(),
        "output": (str(out) if out else None),
        "manifest": str(proj.manifest_path),
        "ledger": str(proj.ledger_path),
        "review_queue": str(proj.review_queue_path),
        "review_html": str(review_path),
    }


# `produce_auto` is a thin (project_dir, **kw) accounting wrapper, which would otherwise hide
# the pipeline's real, documented signature from `inspect.signature` — callers, IDEs and the
# resume contract test all read it. Point introspection at the implementation.
produce_auto.__wrapped__ = _produce_auto
