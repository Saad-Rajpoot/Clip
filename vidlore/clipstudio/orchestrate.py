"""The end-to-end pipeline: ingest → index → segment → match → cut → ledger → build.

One call (`produce`) runs the whole thing, saving the project + ledger + review queue after each
stage so a run is resumable and every intermediate decision is inspectable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import ClipProject, SOURCE_OK
from .config import ClipConfig, load_clip_config, engine_config
from .ingest import ingest_sources, SourceSpec
from .index import index_all
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
    actually find. A SHORT single-scene clip still just wants its own raw footage (no floor)."""
    base = max(4, int(max_sources or 4))
    n = int(n_beats or 0)
    longform_floor = min(32, 20 + (n - 100) // 15) if n >= 100 else 0
    if (video_type or "").strip().lower() == "single_scene":
        # short deep-dive → raw scene only; LONG deep-dive → still needs angle/B-roll variety
        return max(base, longform_floor)
    scaled = min(48, (n + 4) // 5)
    return max(base, scaled, longform_floor)


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
    return _sig(tuple((s.index, s.text, s.visual_policy, s.required_entity, s.required_kind,
                       s.scene_query, s.quote) for s in (segs or [])))


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


def _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log, *, eng_cfg=None) -> int:
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
    # contradicting the narration. We verify it with the vision verifier in LENIENT mode (contextual
    # acceptance: a relevant same-character/scene still is fine; only off-topic / wrong-character /
    # wrong-era is rejected). Honest labeling ('contextual_fallback') is NOT a substitute for passing
    # these guards. Verification runs only when an LLM is available; env disables.
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
                       "movie_title": (proj.meta.get("analysis", {}) or {}).get("movie_title", "")})())

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
        if sid not in _embeds_cache:
            try:
                _embeds_cache[sid] = _index_sv.load_embeds(proj, sid)
            except Exception:
                _embeds_cache[sid] = None
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
        """LENIENT semantic verdict on a recovery still: 'ok' | 'reject' | 'unverified' | 'disabled'.
        Passes the SHOT's real face_ids. A transport error / timeout / malformed response (while the
        verifier IS configured) → 'unverified' (NOT silently 'ok'). 'disabled' means NO verifier is
        configured at all — the caller may still install to avoid a black gap, but labels the still
        'unverified_fallback' (never 'contextual_fallback', which implies a real verdict). Retries
        once on a transient error."""
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
    # hash, shot bounds, judged frame identity, every prompt field, the venue_fallback question
    # variant, real vision model, prompt/sheet versions) is byte-identical. Reuse is NEVER keyed on
    # the image path alone — the same frame judged for a different beat is a different question.
    # Only successful schema-valid verdicts are stored ('unverified' transport outcomes never are),
    # so the 2-attempt retry and the vision-outage backoff behavior are unchanged.
    _still_vcache = _verify_mod._load_verdict_cache(proj)
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

    def _still_fp(kf_path, seg, sid, sidx, faces):
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
                visual_policy=_policy.policy_of(seg), is_specific=False,
                faceid_names=list(faces or []), multiframe=False,
                image_id=f"kf:{_verify_mod._file_fingerprint(kf_path)}",
                model=_vmodel_sv, venue_fallback=True)
        except Exception:
            return ""                                    # no key → baseline uncached call

    def _still_verdict_call(kf_path, seg, sid, sidx):
        from . import perf_metrics as _pm_sv
        faces = _shot_face_ids(sid, sidx)
        _fp = _still_fp(kf_path, seg, sid, sidx, faces)
        _hit = _still_vcache.get(_fp) if _fp else None
        if _hit is not None and _verify_mod._verdict_schema_ok(_hit):
            _pm_sv.incr("still.verdict.cache_hit")
            return "reject" if _hit.get("verdict") == "replace" else "ok"
        for _attempt in (1, 2):
            try:
                _pm_sv.incr("still.verdict.call")
                v = _verify_mod.verify_frame(
                    str(kf_path), seg.text, getattr(seg, "required_entity", ""),
                    getattr(seg, "required_kind", ""), faces, eng_cfg,
                    getattr(eng_cfg, "anthropic_model", ""), is_specific=False,
                    expected_visual=getattr(seg, "expected_visual", "") or "",
                    scene_query=getattr(seg, "scene_query", "") or "",
                    era_hint=_verify_mod._beat_era(seg, _global_era_sv, _vtype_sv == "single_scene",
                                                   anchor_eras=_anchor_eras_sv),
                    # every strict layer already refused this beat — ask the still layer's real
                    # question (right scene/venue, no contradiction), not the micro-action's
                    venue_fallback=True)
                if v is None:
                    continue                              # transport error → retry, then unverified
                if _fp and _verify_mod._verdict_schema_ok({**v, "status": "ok"}):
                    _still_vcache[_fp] = dict(v)
                    _verify_mod._save_verdict_cache(proj, _still_vcache)   # atomic tmp+replace
                return "reject" if v.get("verdict") == "replace" else "ok"
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
    for s in proj.sources:
        if s.status != SOURCE_OK:
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
        want_still = no_clip or pol == _policy.ABSTRACT or (pol == _policy.FILLER and repeat) or weak
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
                if pol in (_policy.EXACT, _policy.CHARACTER):
                    _rclass = "contextual_fallback" if _still_verified else "unverified_fallback"
                else:
                    _rclass = "generic_filler"
                sel.image_meta = {"source": _src, "score": round(float(score), 3),
                                  "src": sid, "shot": sidx, "relevance_class": _rclass,
                                  "still_verified": bool(_still_verified),
                                  "lowres_still": bool(_lowres_pick),
                                  "exact_scene_missing": bool(pol == _policy.EXACT and (weak or no_clip))}
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
            if (src_filled + web_filled) >= cap:
                break
            sel = sel_by_idx.get(seg.index)
            _isrc = (getattr(sel, "image_meta", {}) or {}).get("source", "") if sel else ""
            _has_img = bool(sel and getattr(sel, "image_path", ""))
            if sel is not None and sel.source_id and not _has_img:
                continue                               # airs real moving footage — leave it
            # an exact beat covered only by an unconfirmed source-frame-RECOVERY still TRIES web: a
            # validated web-exact-scene is a better exact match and REPLACES the recovery (req. 1).
            if _has_img and _isrc != "source-frame-recovery":
                continue                               # already has a confirmed still
            if not _policy.allows_web_image(seg):      # filler/abstract → no random web decoration
                continue
            try:
                res = _imgfb.fetch_scene_image(seg, analysis, img_dir, faceid_obj=faceid_obj,
                                               refs=refs, char2actor=char2actor, log=log,
                                               seen_hashes=seen_hashes)
            except Exception as e:
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
            # validated real live-action still → tag explicitly (never the raw host / 'web' / 'ai').
            # A CLIP+Face-ID-validated exact-scene web still is confirmed exact coverage.
            sel.image_meta = {"source": "web-exact-scene", "score": res.get("score"),
                              "clip": res.get("clip"), "face": res.get("face"), "query": res.get("query"),
                              "relevance_class": "exact_scene"}
            web_filled += 1

    # ---- PASS 3 — exact_scene still uncovered/unconfirmed → MANUAL REVIEW (never silent weak filler
    # or AI). A source-frame-recovery still covers it visually but stays flagged; a validated
    # web-exact-scene still (or verifier-kept footage) counts as confirmed coverage.
    exact_missing = 0
    for seg in segs:
        if not _policy.is_exact(seg):
            continue
        sel = sel_by_idx.get(seg.index)
        img_src = (getattr(sel, "image_meta", {}) or {}).get("source", "") if sel else ""
        confirmed = (sel is not None and _verifier_kept(sel)) or img_src == "web-exact-scene"
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


def _write_recovery_audit(proj, audit: dict) -> None:
    import json as _json
    try:
        (proj.output_dir / "recovery_audit.json").write_text(_json.dumps(audit, indent=1),
                                                             encoding="utf-8")
    except Exception:
        pass


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
    _im = getattr(sel, "image_meta", {}) or {}
    if getattr(sel, "image_path", ""):
        # exact beats accept only a REAL still as coverage; the gate skips any beat with an
        # image_path, so for non-exact beats any still resolves it
        if not exact or _im.get("source") in ("source-frame", "web-exact-scene"):
            return False
    v = getattr(sel, "verifier", {}) or {}
    verifier_failed = v.get("status") == "ok" and v.get("verdict") == "replace"
    if exact:
        return bool(verifier_failed or not getattr(sel, "source_id", ""))
    return bool(verifier_failed)


def _recover_unresolved_beats(proj, segs, analysis, cfg, eng, *, faceid_obj, refs, roster,
                              policy, log) -> int:
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
      • Every attempt (queries, candidates, new sources, per-beat outcome) is audited to
        recovery_audit.json.
    Returns the number of beats recovered with real verified footage."""
    import os as _os
    import copy as _copy
    import dataclasses as _dc
    from . import policy as _policy

    if _os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY", "1").strip() in ("0", "false", "no"):
        return 0
    max_beats = int(_os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY_MAX_BEATS", "8") or 8)
    max_sources = int(_os.environ.get("VIDLORE_CLIPSTUDIO_RECOVERY_MAX_SOURCES", "4") or 4)

    seg_by_idx = {s.index: s for s in segs}
    sel_by_idx = {s.segment_index: s for s in proj.selections}
    unresolved = [s.index for s in segs
                  if _beat_is_unresolved(sel_by_idx.get(s.index), s, _policy)]
    audit = {"unresolved_before": list(unresolved), "caps": {"beats": max_beats, "sources": max_sources},
             "attempts": [], "new_sources": [], "recovered": [], "still_unresolved": []}
    if not unresolved:
        _write_recovery_audit(proj, audit)
        return 0
    # GATE-VULNERABLE beats first when the cap trims. Evidence across four blocked renders:
    # every blocker was a CHARACTER beat (their stills need a verified face — the hardest
    # coverage) plus one capped abstract beat; abstract/filler rejected beats take a lenient
    # CLIP-ranked still, and exact beats keep the deterministic-still fallback downstream. So:
    # character first, then filler/abstract, exact last; script order within each class. Beats
    # with NO query material (no scene_query/required_entity — e.g. an abstract outro line)
    # can't be rediscovered and must not burn a slot; stills cover them.
    def _rec_rank(i):
        s = seg_by_idx.get(i)
        p = _policy.policy_of(s) if s is not None else _policy.FILLER
        return ({_policy.CHARACTER: 0, _policy.FILLER: 1, _policy.ABSTRACT: 1,
                 _policy.EXACT: 2}.get(p, 1), i)
    unresolved = [i for i in unresolved
                  if ((getattr(seg_by_idx.get(i), "scene_query", "") or "").strip()
                      or (getattr(seg_by_idx.get(i), "required_entity", "") or "").strip())]
    unresolved.sort(key=_rec_rank)
    unresolved = unresolved[:max_beats]
    if not unresolved:
        _write_recovery_audit(proj, audit)
        return 0
    log(f"recovery: {len(unresolved)} unresolved beat(s) → bounded rediscovery {unresolved}")

    # Snapshot EVERY current selection; the final selection list is snapshot ∪ (recovered new picks).
    snapshot = {s.segment_index: _copy.deepcopy(s) for s in proj.selections}
    unresolved_segs = [seg_by_idx[i] for i in unresolved if i in seg_by_idx]
    recovered: set = set()

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
        cands = discover_sources(analysis, cfg_r, segments=unresolved_segs, progress=None) or []
        import re as _re_r
        _r_mv = {w for w in _re_r.findall(r"\w+", (getattr(analysis, "movie_title", "") or "").lower())
                 if len(w) > 2}

        def _beat_hits(c) -> int:
            tw = set(_re_r.findall(r"\w+", (getattr(c, "title", "") or "").lower()))
            best = 0
            for s in unresolved_segs:
                toks = {w for w in _re_r.findall(
                            r"\w+", (getattr(s, "scene_query", "") or "").lower())
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
            "queries": [getattr(s, "scene_query", "") or getattr(s, "required_entity", "")
                        for s in unresolved_segs],
            "candidates_found": len(cands), "new_candidates": len(new_cands)})
        if not new_cands:
            log("recovery: targeted rediscovery found no NEW source — leaving beats for downstream fallback")
            _write_recovery_audit(proj, audit)
            return 0

        before_ok = {s.id for s in proj.sources if s.status == SOURCE_OK}
        download_candidates(proj, new_cands, cfg_r, policy=policy, limit=len(new_cands), progress=None)
        newly = [s for s in proj.sources if s.status == SOURCE_OK and s.id not in before_ok]
        audit["new_sources"] = [{"id": s.id, "title": (getattr(s, "title", "") or "")[:70],
                                 "height": getattr(s, "height", 0)} for s in newly]
        if not newly:
            log(f"recovery: no NEW source downloaded under policy={policy} — leaving beats unresolved")
            _write_recovery_audit(proj, audit)
            return 0

        # index the new sources, rematch, cut, then reverify. proj.selections is fully rebuilt here;
        # it is reconciled against the snapshot below so only recovered beats survive the change.
        # The re-verify is restricted to the recovered beats ONLY (only_indices) — re-verifying all
        # 229 beats here just to re-check ~8 recovered ones cost hours; every other beat is restored
        # from the snapshot anyway, so its verdict does not matter.
        index_all(proj, cfg_r, references=refs, faceid=faceid_obj, roster=roster,
                  force=False, progress=None)
        match_segments(proj, segs, cfg_r, analysis=analysis, progress=None)
        cut_all(proj, cfg_r, progress=None)
        from . import verify as _verify_r
        _verify_r.verify_and_repair(proj, segs, cfg, eng, only_indices=set(unresolved), progress=None)

        new_sel_by_idx = {s.segment_index: s for s in proj.selections}
        for i in unresolved:
            if not _beat_is_unresolved(new_sel_by_idx.get(i), seg_by_idx.get(i), _policy):
                recovered.add(i)
    except Exception as e:
        log(f"recovery: bounded round errored ({type(e).__name__}: {e}) — leaving beats unresolved")

    # Reconcile: recovered beats keep their NEW pick; EVERY other beat is restored from the snapshot
    # (good beats unchanged; un-recovered unresolved beats keep their prior state for the honest
    # downstream still/hold/block). A final re-cut guarantees each beat's on-disk clip matches its
    # final selection — so a re-verify that happened to swap a good beat can never leave a stale clip.
    new_sel_by_idx = {s.segment_index: s for s in proj.selections}
    final = []
    for idx in sorted(set(snapshot) | set(new_sel_by_idx)):
        if idx in recovered and idx in new_sel_by_idx:
            final.append(new_sel_by_idx[idx])
        elif idx in snapshot:
            final.append(snapshot[idx])
        else:
            final.append(new_sel_by_idx[idx])
    proj.selections = final
    try:
        cut_all(proj, cfg, progress=None)
    except Exception as e:
        log(f"recovery: final re-cut errored ({type(e).__name__}: {e})")
    proj.save()

    audit["recovered"] = sorted(recovered)
    audit["still_unresolved"] = [i for i in unresolved if i not in recovered]
    _write_recovery_audit(proj, audit)
    if recovered:
        log(f"recovery: ✓ recovered {len(recovered)} beat(s) with real verified footage {sorted(recovered)}")
    if audit["still_unresolved"]:
        log(f"recovery: {len(audit['still_unresolved'])} beat(s) still unresolved → deterministic still / "
            f"editorial hold / honest release-block downstream {audit['still_unresolved']}")
    return len(recovered)


def produce_auto(project_dir, *, topic: str = "", script_path: Optional[str] = None,
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
    from .analyze import analyze_script, ScriptAnalysis
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
                _pm_stage.stage(_s.split("·", 1)[1].strip()[:48])
        except Exception:
            pass
        if progress:
            progress(m)

    if (project_dir / "project.json").exists():
        proj = ClipProject.load(project_dir)
    else:
        proj = ClipProject(name=project_dir.name, root=str(project_dir))
    proj.ensure_dirs()

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
    if _stage_skip(proj, "analyze", _sig_analyze, resume,
                   artifact_ok=bool(proj.segments and proj.meta.get("analysis"))):
        analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
        segs = proj.segments
        _tally = _policy.finalize_beats(segs)          # idempotent re-classify (cheap; for the tally)
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
    if _stage_skip(proj, "download", _sig_download, resume, artifact_ok=bool(_usable0)):
        usable = _usable0
        log(f"  ↻ skipped (resume) — {len(usable)} source(s) already downloaded")
    else:
        download_candidates(proj, candidates, cfg, policy=policy, limit=dl_limit, progress=progress)
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
    _sig_index = _sig(_sig_download, tuple(sorted(s.id for s in usable)), bool(force_index))
    # MATCH_GATE_VERSION folds pool-gate semantics into the match signature: a resume must NOT
    # replay selections chosen before a new footage gate existed (observed: cached selections
    # kept airing news-CGI/fan-art shots the new graphics gate would refuse — the gate was
    # silently inert on every resumed project). Bump on any pool-gate semantics change.
    _sig_match = _sig(_sig_index, _seg_sig(segs), "gatev2-graphics")
    _sig_cut = _sig(_sig_match)
    _sig_verify = _sig(_sig_cut, bool(verify))
    _sig_recover = _sig(_sig_verify)
    _sel_complete = (bool(proj.selections)
                     and {s.segment_index for s in proj.selections} >= {s.index for s in segs})
    _skip_match = _stage_skip(proj, "match", _sig_match, resume, artifact_ok=_sel_complete)
    _skip_verify = _stage_skip(proj, "verify", _sig_verify, resume, artifact_ok=_sel_complete)
    _skip_recover = _stage_skip(proj, "recover", _sig_recover, resume, artifact_ok=_sel_complete)
    # Face-ID refs feed indexing (shot identities) + recovery re-indexing; match consumes persisted
    # shot identities, not the refs object. So they're only worth building when a footage stage will
    # actually run. A fully-cached resume (→ assemble only) skips this and the model load with it.
    _need_footage_stages = not (_skip_match and _skip_verify and _skip_recover)

    # 4 — Face-ID references
    faceid_obj, refs = None, {}
    roster = analysis.actors
    if _need_footage_stages:
        log("4/9 · build Face-ID references")
        if _faceid.available():
            faceid_obj = _faceid.FaceID()
            refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                            faceid_obj, progress=progress)
    else:
        log("4/9 · Face-ID references — ↻ skipped (resume, all footage stages cached)")

    # 5 — deep index
    if _need_footage_stages:
        log("5/9 · deep index (ASR + scenes + CLIP + Face-ID + OCR + quality + dedup)")
        index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster,
                  force=(force_index and not resume), progress=progress)
    else:
        log("5/9 · deep index — ↻ skipped (resume, all footage stages cached)")

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
        cut_all(proj, cfg, resume=resume, progress=progress)
        _stage_done(proj, "cut", _sig_cut)

    # 8 — AI verify + repair
    _verify_down = False
    if verify and _skip_verify:
        log("8/9 · AI verify + repair — ↻ skipped (resume, verdicts cached)")
    elif verify:
        log("8/9 · AI verify + repair")
        _vres = _verify.verify_and_repair(proj, segs, cfg, eng, progress=progress) or {}
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
        try:
            _recover_unresolved_beats(proj, segs, analysis, cfg, eng, faceid_obj=faceid_obj, refs=refs,
                                      roster=roster, policy=policy, log=log)
        except Exception as _e:
            log(f"recovery: skipped ({type(_e).__name__}: {_e})")

        # 8b — EXACT-SCENE IMAGE FALLBACK: beats that still have no relevant footage (no source,
        # low confidence, or a confirmed wrong-character pick) get a face/CLIP-verified still from
        # the open web (Bing/DDG/Wikimedia). A wrong image is worse than generic footage, so the
        # fetcher only accepts what it can verify; otherwise the beat keeps its clip.
        if _os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip() not in ("0", "false", "no"):
            try:
                _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log, eng_cfg=eng)
            except Exception as _e:
                log(f"image-fallback: skipped ({type(_e).__name__}: {_e})")
        _stage_done(proj, "recover", _sig_recover)

    # ledger + QC report
    summ = ledger.finalize(proj, segs, cfg)
    review_path = _review.write_review(proj, segs)
    proj.save()
    log(f"  QC · {summ['flagged_for_review']}/{summ['segments']} flagged · mean conf {summ['mean_confidence']}")

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
            log(f"pre-assemble gate: skipped ({type(_e).__name__}: {_e})")
        if _pre:
            _mode = _os.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower()
            if _mode == "warn":
                log(f"⚠ pre-assemble feasibility (mode=warn) — {_pre} Building a REVIEW draft anyway.")
            else:
                log(f"⛔ pre-assemble feasibility — {_pre} Aborting BEFORE assembly (fail-fast; the "
                    f"final render would release-block after ~20 min of wasted encoding). Add footage "
                    f"for these scenes or set RELEASE_BLOCK_MODE=warn for a review draft.")
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(_pre)

    out = None
    if do_build:
        log("9/9 · assemble final video")
        out = build_video(proj, segs, cfg, voice=voice, captions=captions,
                          caption_style=caption_style,
                          title=title or analysis.movie_title or proj.name,
                          theme_name=theme, voiceover=voiceover, voice_provider=voice_provider,
                          voice_preset=voice_preset, use_tts=use_tts, progress=progress)

    if _pm_stage.enabled():
        _pm_stage.write_report(proj.output_dir / "perf_report.json")

    return {
        "project": str(proj.root),
        "summary": summ,
        "analysis": analysis.to_dict(),
        "output": (str(out) if out else None),
        "manifest": str(proj.manifest_path),
        "ledger": str(proj.ledger_path),
        "review_queue": str(proj.review_queue_path),
        "review_html": str(review_path),
    }
