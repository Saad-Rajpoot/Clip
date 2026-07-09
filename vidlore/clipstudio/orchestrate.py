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


def produce(project_dir, *, script_path: Optional[str] = None, script_text: Optional[str] = None,
            source_specs: Optional[list[SourceSpec]] = None, name: str = "",
            cfg: Optional[ClipConfig] = None, theme: str = "history", voice: str = "",
            title: str = "", captions: bool = True, use_tts: bool = True, enrich: bool = True,
            voiceover: Optional[str] = None,
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


def _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log) -> int:
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

    # Index every pooled shot — REAL downloaded source-video keyframes are the PREFERRED (and primary)
    # image source. No NOW-photos / recent-actor photos / AI images anywhere in this path.
    from . import discover as _discover
    shots_by_key: dict = {}
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
        if _still_min_h and 0 < int(getattr(s, "height", 0) or 0) < _still_min_h:
            _skipped_lowres += 1
            continue
        try:
            _shots = _index.load_shots(proj, s.id)
        except Exception:
            continue
        if _src_wm(_shots) or (_corner_on and _src_logo(_shots)):
            _skipped_wm += 1
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
        essential = no_clip or (pol == _policy.EXACT and weak)
        within_cap = essential or src_filled < still_cap
        if want_still and within_cap and not (sel and getattr(sel, "image_path", "")):
            still = None
            if sel is not None:
                still = _imgfb.pick_source_still(sel, shots_by_key, used_keys, used_phash,
                                                 distinct_from=aired_phash)
            if still is None:                          # no relevance-ranked alternate → scan the pool
                still = _imgfb.pick_pool_still(seg, shots_by_key, used_keys, used_phash,
                                               distinct_from=aired_phash)
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
                sel.image_meta = {"source": _src, "score": round(float(score), 3),
                                  "src": sid, "shot": sidx}
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
            # validated real live-action still → tag explicitly (never the raw host / 'web' / 'ai')
            sel.image_meta = {"source": "web-exact-scene", "score": res.get("score"),
                              "clip": res.get("clip"), "face": res.get("face"), "query": res.get("query")}
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


def produce_auto(project_dir, *, topic: str = "", script_path: Optional[str] = None,
                 script_text: Optional[str] = None, movie_hint: str = "",
                 policy: str = "block", max_sources: int = 6, cfg: Optional[ClipConfig] = None,
                 theme: str = "history", voice: str = "", title: str = "", captions: bool = True,
                 voiceover: Optional[str] = None, voice_provider: str = "", voice_preset: str = "",
                 use_tts: bool = True, verify: bool = True, do_build: bool = True,
                 force_index: bool = False, progress=None) -> dict:
    """FULL AUTO: topic + script in → finished video out. Discovers + downloads its own sources.
    analyze → discover → download(policy) → Face-ID refs → deep index → match → cut → AI verify
    → ledger + QC report → assemble."""
    from .analyze import analyze_script
    from .discover import discover_sources
    from .download import download_candidates
    from . import faceid as _faceid
    from . import verify as _verify
    from . import review as _review

    cfg = cfg or load_clip_config()
    eng = engine_config()
    project_dir = Path(project_dir)

    def log(m):
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

    # 1 — analyze
    log("1/9 · analyze script")
    if script_text is None and script_path:
        script_text = Path(script_path).read_text(encoding="utf-8")
        proj.script_path = str(script_path)
    if not (script_text or "").strip():
        raise PipelineError("no narration script provided.")
    from . import policy as _policy
    analysis, segs = analyze_script(script_text, topic=topic, movie_hint=movie_hint,
                                    eng_cfg=eng, cfg=cfg, progress=progress)
    proj.segments = segs
    # unified per-beat VISUAL POLICY — classify every beat once; all downstream stages obey it
    _tally = _policy.finalize_beats(segs)
    proj.meta["analysis"] = analysis.to_dict()
    proj.save()
    log(f"  movie={analysis.movie_title!r} · {len(analysis.actors)} actors · {len(segs)} beats")
    log(f"  beat policy → " + " · ".join(f"{k}:{v}" for k, v in sorted(_tally.items())))

    # 2 — discover. The ranked head of the returned list = the requested download budget;
    # entity-coverage + anchor force-includes are APPENDED past it by discover, so the
    # download cap below covers the WHOLE returned list — any prefix cap would cut exactly
    # the must-download items. cfg is copied, never mutated (callers may share it across runs).
    # Deliberate: max_sources (the user's budget) overrides cfg.discover_target here — the env
    # knob still governs direct discover_sources() callers.
    log("2/9 · discover sources")
    import dataclasses as _dc
    # scale the footage budget to the script: a long multi-scene essay needs far MORE distinct
    # sources than the user's default, or most beats re-use a few generic clips (wrong-scene).
    _budget = _scaled_source_budget(max_sources, len(segs), getattr(analysis, "video_type", ""))
    if _budget > max_sources:
        log(f"  long script ({len(segs)} beats) → scaling footage budget {max_sources} → {_budget} sources")
    cfg = _dc.replace(cfg, discover_target=max(4, _budget))
    candidates = discover_sources(analysis, cfg, segments=segs, progress=progress)
    proj.meta["candidates"] = [c.to_dict() for c in candidates]
    proj.save()
    if not candidates:
        raise PipelineError("source discovery found nothing — check connectivity / movie name.")

    # 3 — download (permission-policy gated). Download the FULL discovered set: discover's own
    # construction bounds it (head ≤ max_sources, coverage ≤ coverage_extra, anchors ≤ 4/scene),
    # and download applies the limit as a PREFIX slice — any smaller cap would slice off the
    # anchor force-includes appended last.
    dl_limit = len(candidates)
    log(f"3/9 · download (policy={policy}, budget={_budget}+coverage)")
    download_candidates(proj, candidates, cfg, policy=policy, limit=dl_limit, progress=progress)
    usable = [s for s in proj.sources if s.status == "ok"]
    if not usable:
        raise PipelineError(f"no usable sources under policy={policy}. Use --policy approved_testing "
                         "(your explicit testing approval) or open_only (public-domain).")
    log(f"  {len(usable)} sources downloaded")

    # 4 — Face-ID references
    log("4/9 · build Face-ID references")
    faceid_obj, refs = None, {}
    if _faceid.available():
        faceid_obj = _faceid.FaceID()
        refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                        faceid_obj, progress=progress)
    roster = analysis.actors

    # 5 — deep index
    log("5/9 · deep index (ASR + scenes + CLIP + Face-ID + OCR + quality + dedup)")
    index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster,
              force=force_index, progress=progress)

    # 6 — match
    log("6/9 · match")
    match_segments(proj, segs, cfg, analysis=analysis, progress=progress)

    # 7 — cut
    log("7/9 · cut")
    cut_all(proj, cfg, progress=progress)

    # 8 — AI verify + repair
    if verify:
        log("8/9 · AI verify + repair")
        _verify.verify_and_repair(proj, segs, cfg, eng, progress=progress)

    # 8b — EXACT-SCENE IMAGE FALLBACK: beats that still have no relevant footage (no source,
    # low confidence, or a confirmed wrong-character pick) get a face/CLIP-verified still from
    # the open web (Bing/DDG/Wikimedia). A wrong image is worse than generic footage, so the
    # fetcher only accepts what it can verify; otherwise the beat keeps its clip.
    import os as _os
    if _os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip() not in ("0", "false", "no"):
        try:
            _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log)
        except Exception as _e:
            log(f"image-fallback: skipped ({type(_e).__name__}: {_e})")

    # ledger + QC report
    summ = ledger.finalize(proj, segs, cfg)
    review_path = _review.write_review(proj, segs)
    proj.save()
    log(f"  QC · {summ['flagged_for_review']}/{summ['segments']} flagged · mean conf {summ['mean_confidence']}")

    out = None
    if do_build:
        log("9/9 · assemble final video")
        out = build_video(proj, segs, cfg, voice=voice, captions=captions,
                          title=title or analysis.movie_title or proj.name,
                          theme_name=theme, voiceover=voiceover, voice_provider=voice_provider,
                          voice_preset=voice_preset, use_tts=use_tts, progress=progress)

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
