"""Self-healing footage recovery — the automated version of the manual gate-unblock playbook.

When the pre-assembly feasibility gate (or the in-build release gate) blocks beats because no
valid footage/fallback exists, this module resolves them WITHOUT weakening any gate and
without human intervention, exactly the way the manual recovery did on job olenna_v2_allfixes
(17 blocked beats → 0):

  1. UNFILLABLE beats (the script asks for content our own gates forbid by design — BTS,
     showrunner/interview footage) are softened to an abstract beat: a 2-second meta line must
     not fail a whole video. Deterministic regex classification, honestly logged.
  2. STILL RECOVERY: pool-wide CLIP-ranked candidate keyframes, each verified by the vision
     model at the pipeline's own venue bar (right scene, no contradiction — the exact
     _still_verdict semantics), installed as sel.image_path with the honest
     contextual_fallback labeling. A validated still is air-worthy for the release gate.
  3. TARGETED ACQUISITION: when the pool simply lacks the scene, the LLM (DeepSeek primary)
     writes precise YouTube queries from the beat's own script requirements (deterministic
     fallback when no LLM), discovery/download/index run for exactly that scene, and still
     recovery retries against the new footage — including sampled region frames for
     coarse-shot/dark sources.

Bounded throughout: max rounds, max sources per beat, a no-progress round stops the loop, and every
gate decision stays with the existing gate code. Any failure in specificity softening rolls its
mutations back before the publication contract runs again. Kill switch:
VIDLORE_CLIPSTUDIO_SELFHEAL=0.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from .models import ClipProject, SOURCE_BLOCKED, SOURCE_FAILED, SOURCE_OK

# content the script may ask for that our own source gates refuse BY DESIGN — such a beat can
# never be filled with footage, no matter how many rounds run
_UNFILLABLE_RX = re.compile(
    r"behind[- ]the[- ]scenes|showrunners?|writers'? room|\bbenioff\b|\bd\.?b\.? weiss\b|"
    r"\bweiss\b|camera crew|film(ing)? set|production footage|interview footage|"
    r"documentary crew|press junket", re.I)


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def parse_blocked(reason: str) -> list[int]:
    """Beat indexes out of a gate message ('… scene(s) [10, 11, 116]. …')."""
    m = re.search(r"scene\(s\)\s*\[([0-9,\s]*)\]", reason or "")
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def blocked_indexes(proj) -> list[int]:
    """The authoritative unresolved list from output/rejected_footage_audit.json."""
    try:
        d = json.loads((proj.output_dir / "rejected_footage_audit.json").read_text())
        return sorted({int(e["seg_index"]) for e in (d.get("unresolved_release_block") or [])})
    except Exception:                                    # noqa: BLE001
        return []


def beat_unfillable(seg) -> bool:
    txt = " ".join(str(getattr(seg, f, "") or "") for f in
                   ("text", "expected_visual", "scene_query", "required_entity"))
    return bool(_UNFILLABLE_RX.search(txt))


def _soften_to_abstract(seg, log, *, cause: str = "gate-forbidden content") -> None:
    log(f"self-heal: beat {seg.index} — {cause} "
        f"({(seg.expected_visual or seg.text or '')[:60]!r}) — softened to an abstract beat "
        f"(the audit records that literal specificity was surrendered)")
    seg.visual_policy = "abstract_effect"
    seg.required_entity = ""
    seg.required_kind = ""
    # These fields point retrieval back at the exact moment we just proved unreachable. Keeping
    # them made the abstract rung repeatedly fetch the same rejected scene and also left an authored
    # dialogue promise attached to a beat that no longer claims to show literal dialogue.
    seg.quote = ""
    seg.scene_query = ""
    try:
        seg.is_specific_claim = False
    except Exception:                                    # noqa: BLE001
        pass
    seg.expected_visual = ("Neutral, era-appropriate imagery from the show — a wide or "
                           "atmospheric shot as visual rest under a meta line.")


def _soften_to_character(seg, log, *, surrender_verbatim_quote: bool = False) -> bool:
    """LAST RESORT for an exact beat whose moment is simply not in reach. Returns True if softened.

    The owner's standing rule for this pipeline: if the exact scene is not available, do not insist
    on it — settle for a similar one. This is where "do not insist" has to be implemented, because
    everything upstream has already been tried and the only remaining outcomes are a related shot or
    no video at all.

    Measured, job 6a26707939. Four beats survived acquisition and release-blocked the render; three
    of them (24, 82, 149) describe the Bolton flayed-man banner coming down at Winterfell. The frame
    exists — game_of_thrones_jon_sn_123ebf87 shot 64 is the banners lying in the snow — and CLIP
    cannot see it: it ranks 48/67, 33/67 and 40/67 WITHIN ITS OWN FILE on those three queries, while
    the Stark-banner shot beside it ranks 1/67 on all three. No bench of any sane depth reaches rank
    48, so demanding the exact moment on those beats is demanding something retrieval cannot deliver.

    Softening EXACT to CHARACTER keeps the beat pointed at the right subject and era and lets a
    related frame satisfy it. The still is then labelled `contextual_fallback` by the existing
    honest-class logic, so the audit still says plainly that the exact moment was never found.

    Only ever reached by a beat that would otherwise release-block the whole video, so it cannot
    change any beat that is already working. env VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN=0 to disable."""
    if not _env_on("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN", "1"):
        return False
    from . import policy as _P
    if _P.policy_of(seg) != _P.EXACT:
        return False
    # A PROP IS NOT A PERSON. Softening the policy alone does nothing when the requirement itself is
    # the unfindable thing: beats 82 and 149 demand required_entity='flayed man banner' / 'Bolton
    # banners', and beat 160 a 'tent'. Every still candidate is checked against that entity, so the
    # beat keeps failing at the looser policy exactly as it did at the strict one — measured, all
    # three survived the first softening pass and blocked the render anyway.
    # So on this last resort the OBJECT/LOCATION/SCENE requirement is dropped and the beat falls
    # back to its narration and era. A CHARACTER requirement is NOT dropped: "any Melisandre shot"
    # is an honest fallback, "any shot at all" on a beat about a person is how a wrong-character
    # leak gets in, and that is the one class of error the identity gate exists to stop.
    _kind = (getattr(seg, "required_kind", "") or "").strip().lower()
    _drop = _kind not in ("character", "actor")
    _quote_note = ", surrendering the audited verbatim-dialogue promise" \
        if surrender_verbatim_quote and str(getattr(seg, "quote", "") or "").strip() else ""
    log(f"self-heal: beat {seg.index} — the exact moment is not in reach after acquisition; "
        f"settling for the right subject instead of failing the video "
        f"(exact_scene -> character_specific"
        + (f", dropping the {_kind or 'unnamed'} requirement "
           f"{(getattr(seg, 'required_entity', '') or '')[:40]!r}" if _drop else "")
        + _quote_note + ")")
    seg.visual_policy = _P.CHARACTER
    if _drop:
        seg.required_entity = ""
        seg.required_kind = ""
    # A normal character fallback never erases real dialogue.  The sole exception is a separately
    # validated schema-4 proof that the exact quote/scene exists only in native-SD bytes and no
    # publishable-HD copy exists in the current, fully audited pool.  Clearing the field here makes
    # the surrendered promise explicit before a character still is judged; otherwise still
    # coverage could pass while project.json continued to claim timed on-screen dialogue.
    if surrender_verbatim_quote:
        seg.quote = ""
    try:
        seg.is_specific_claim = False
    except Exception:                                    # noqa: BLE001
        pass
    return True


# ── candidate pool ────────────────────────────────────────────────────────────────────────────

def _clean_pool(proj) -> list[tuple[str, object]]:
    """(source_id, Shot) for every shot that passes the same hard gates match applies."""
    from . import index as I
    from .match import (banned_source_ids, _shot_unreadable, _ocr_is_junk,
                        _ocr_text_heavy, _shot_static_collage, _shot_numeral_overlay)
    banned = banned_source_ids(proj, include_auto=True)
    # An index file is not an authorization record.  Downloads can be blocked/removed after their
    # old ``*.shots.json`` survives, and a stray index may not belong to the manifest at all.  The
    # normal moving-footage pool only admits manifest sources whose current status is SOURCE_OK;
    # the specificity ladder must use the same eligibility boundary or a stronger-ranked blocked
    # source can become an abstract still that the concrete publication contract then skips.
    eligible = {
        str(getattr(src, "id", "") or "")
        for src in (getattr(proj, "sources", None) or [])
        if str(getattr(src, "status", "") or "") == SOURCE_OK
    }
    pool = []
    for f in sorted(Path(proj.index_dir).glob("*.shots.json")):
        sid = f.name.replace(".shots.json", "")
        if sid not in eligible or sid in banned:
            continue
        try:
            shots = I.load_shots(proj, sid)
        except Exception:                                # noqa: BLE001
            continue
        for sh in shots:
            kf = getattr(sh, "keyframe_path", "") or ""
            if not kf or not Path(kf).exists():
                continue
            if (_shot_unreadable(sh) or _ocr_is_junk(sh) or _ocr_text_heavy(sh)
                    or _shot_static_collage(sh) or _shot_numeral_overlay(sh)):
                continue
            la = float(getattr(sh, "luma_avg", -1) or -1)
            if 0 <= la < 10:
                continue
            pool.append((sid, sh))
    return pool


_VV_CACHE = {"root": "", "data": {}}


def _venue_cache(proj) -> dict:
    """The project's verdict cache, loaded once per project — self-heal runs up to 3 rounds per
    pass (and the whole pass twice on a review-draft retry), so re-reading per candidate would
    be pointless IO."""
    root = str(getattr(proj, "root", ""))
    if _VV_CACHE["root"] != root:
        try:
            from .verify import _load_verdict_cache
            _VV_CACHE["data"] = _load_verdict_cache(proj) or {}
        except Exception:                                # noqa: BLE001
            _VV_CACHE["data"] = {}
        _VV_CACHE["root"] = root
    return _VV_CACHE["data"]


def _venue_cache_save(proj) -> None:
    """Persist new venue verdicts by MERGING onto whatever is on disk — verify.py and the still
    layer write the same file, and a blind overwrite from a stale in-memory copy would silently
    drop their entries (and re-buy those verdicts on the next pass)."""
    try:
        from .verify import _load_verdict_cache, _save_verdict_cache
        merged = _load_verdict_cache(proj) or {}
        merged.update(_VV_CACHE.get("data") or {})
        _save_verdict_cache(proj, merged)
        _VV_CACHE["data"] = merged
    except Exception:                                    # noqa: BLE001 — cache is an optimisation
        pass


def _venue_fallback_for(seg) -> bool:
    """Whether the preliminary still search should ask the exact venue question.

    The first specificity-ladder rung has already changed the promise to character-general.  Asking
    its candidates to prove the abandoned exact storyboard made that rung unsatisfiable and sent a
    real Jaime footage gap straight to abstract.  Exact and abstract legacy callers retain their
    existing venue search; only CHARACTER asks the generic subject question.
    """
    try:
        from . import policy as _P
        return _P.policy_of(seg) != _P.CHARACTER
    except Exception:                                  # noqa: BLE001 — uncertainty stays strict
        return True


def _venue_fp(proj, kf_path: str, seg, faces, eng_cfg, model_id: str = "") -> str:
    """Fingerprint of the question _venue_verify ACTUALLY asks.

    Deliberately NOT the still layer's `_still_fp`: that one bakes `era=_beat_era(...)` into the
    key while this call sends no era_hint at all. Sharing keys would let one layer consume the
    other layer's answer to a weaker/stronger question — the exact cross-question reuse the
    whole full-fingerprint doctrine exists to prevent. era/must_see stay empty here because they
    are empty in the prompt."""
    try:
        from . import verify as _V
        from . import policy as _P
        venue_fallback = _venue_fallback_for(seg)
        return _V.verdict_fingerprint(
            src_hash=_src_hash_of(proj, kf_path), source_id=_sid_of_kf(kf_path),
            shot_start=0.0, shot_end=0.0,
            beat_text=getattr(seg, "text", ""),
            required_entity=getattr(seg, "required_entity", "") or "",
            required_kind=getattr(seg, "required_kind", "") or "",
            expected_visual=getattr(seg, "expected_visual", "") or "",
            scene_query=getattr(seg, "scene_query", "") or "",
            era="", visual_policy=_P.policy_of(seg), is_specific=False,
            faceid_names=list(faces or []), multiframe=False,
            image_id=f"kf:{_V._file_fingerprint(kf_path)}",
            model=(model_id or _vision_model(eng_cfg)),
            venue_fallback=venue_fallback, must_see="")
    except Exception:                                    # noqa: BLE001 — no key → uncached call
        return ""


def _vision_model(eng_cfg) -> str:
    try:
        from . import llm as _L
        return _L.vision_config(eng_cfg)
    except Exception:                                    # noqa: BLE001
        return str(getattr(eng_cfg, "anthropic_model", "") or "")


def _sid_of_kf(kf_path: str) -> str:
    """Source id from a keyframe path (…/index/<sid>/keyframes/shot_NNNN.jpg)."""
    try:
        return Path(kf_path).parent.parent.name
    except Exception:                                    # noqa: BLE001
        return ""


def _src_hash_of(proj, kf_path: str) -> str:
    """Content hash of the source the keyframe came from ('' when unresolvable) — the same
    derivation the still layer uses (`_file_fingerprint` of the source's local_path)."""
    try:
        from . import verify as _V
        sv = proj.source(_sid_of_kf(kf_path))
        return _V._file_fingerprint(getattr(sv, "local_path", "") or "") if sv is not None else ""
    except Exception:                                    # noqa: BLE001
        return ""


def _venue_verify(kf_path: str, seg, faces, eng_cfg, *, proj=None, cache=None):
    """One venue-bar vision verdict (right scene/venue, no contradiction) — the still layer's
    real question. Returns the verdict dict or None.

    Served from the project's verdict_cache when the identical question was already answered.
    This was the last uncached vision path in the pipeline: self-heal runs up to 3 rounds per
    render (twice over, on the review-draft retry), the CLIP ranking is deterministic and
    used_paths only grows on INSTALL — so every round re-bought verdicts for the same rejected
    frames. Same doctrine as every other layer: identical question → cached answer; only
    schema-valid successes are stored, so retry/breaker behaviour is untouched."""
    from .verify import verify_frame, _verdict_schema_ok, _hit_provider_ok
    from . import perf_metrics as _pm_vv
    _fp = _venue_fp(proj, kf_path, seg, faces, eng_cfg) if (proj is not None
                                                            and cache is not None) else ""
    if _fp:
        _hit = cache.get(_fp)
        if _hit is not None and _verdict_schema_ok(
                _hit, required_entity=getattr(seg, "required_entity", "") or "") \
                and _hit_provider_ok(_hit, _vision_model(eng_cfg)):
            _pm_vv.incr("selfheal.venue.cache_hit")
            return dict(_hit)                            # copy: callers mutate their verdict
    _pm_vv.incr("selfheal.venue.call")
    venue_fallback = _venue_fallback_for(seg)
    v = verify_frame(str(kf_path), seg.text, getattr(seg, "required_entity", "") or "",
                     getattr(seg, "required_kind", "") or "", list(faces or []), eng_cfg,
                     is_specific=False,
                     expected_visual=getattr(seg, "expected_visual", "") or "",
                     scene_query=getattr(seg, "scene_query", "") or "",
                     venue_fallback=venue_fallback)
    # Never turn an explicit provider/error status into a successful cached judgment.  Raw
    # ``verify_frame`` replies historically omit status, which is the one legacy-success form the
    # schema accepts; an explicitly non-ok status must remain non-ok all the way to the caller.
    if _fp and v is not None and _verdict_schema_ok(
            v, required_entity=getattr(seg, "required_entity", "") or ""):
        cache[_fp] = {k: val for k, val in v.items() if k != "reused"}
    return v


def _install_still(sel, kf_path: str, sid: str, shot_idx: int, rel: float) -> None:
    sel.image_path = str(kf_path)
    sel.image_meta = {"source": "source-frame-recovery", "score": round(float(rel or 0), 3),
                      "src": sid, "shot": int(shot_idx),
                      "relevance_class": "contextual_fallback", "still_verified": True,
                      "lowres_still": False, "exact_scene_missing": True}


def _discard_invalid_still(sel, reason: str, log=print) -> dict:
    """Detach a strict-rejected still without deleting its recoverable bytes."""
    old = {
        "image_path": str(getattr(sel, "image_path", "") or ""),
        "image_meta": dict(getattr(sel, "image_meta", {}) or {}),
        "reason": str(reason or "unverified still"),
    }
    if old["image_path"]:
        log(f"self-heal: beat {getattr(sel, 'segment_index', '?')} — existing still is not "
            f"publication evidence ({old['reason']}); retrying without it")
    sel.image_path = ""
    sel.image_meta = {}
    return old


def _still_verdict_schema_error(verdict, seg=None, *, require_keep_facts: bool = False) -> str:
    """Why a still-verifier reply is not a recognized successful judgment.

    ``verify_frame``'s native success payload has no status key, while wrappers/cache records use
    ``status='ok'``.  Both are legitimate.  Every explicit non-ok status is infrastructure state,
    never a content verdict; validating the *original* object here prevents callers from laundering
    ``{'status': 'error', 'verdict': 'keep'}`` by overwriting status before inspection.
    """
    if not isinstance(verdict, dict):
        return "verifier returned no verdict"
    raw_status = verdict.get("status", "")
    if not isinstance(raw_status, str):
        return "verifier status is malformed"
    status = raw_status.strip().lower()
    if status not in ("", "ok"):
        return f"verifier status is {status!r}, not ok"
    try:
        from .verify import _verdict_schema_ok
        # This helper reports the precise missing keep fact below. Ask the shared cache validator
        # only for the status/verdict/confidence envelope here; cache read/write call sites use its
        # complete-keep default and therefore still evict malformed positive entries.
        valid = _verdict_schema_ok(verdict, complete_keep=False)
    except Exception:                                   # noqa: BLE001 — schema uncertainty fails closed
        valid = False
    if not valid:
        value = verdict.get("verdict")
        return f"verifier reply has unrecognized verdict/schema ({value!r})"
    if require_keep_facts and verdict.get("verdict") == "keep":
        required = (
            "matches_narration", "specific_enough", "quality_ok",
            "wrong_subject_visible", "contradicts_narration",
        )
        for field in required:
            if not isinstance(verdict.get(field), bool):
                return f"verifier keep is missing/malformed required boolean {field!r}"
        if seg is not None and str(getattr(seg, "required_entity", "") or "").strip() \
                and not isinstance(verdict.get("correct_subject_visible"), bool):
            return "verifier keep is missing/malformed required boolean 'correct_subject_visible'"
        try:
            from . import policy as _policy_verdict
            target = _policy_verdict.deictic_target(seg) if seg is not None else ""
        except Exception:                                # noqa: BLE001 — cannot prove schema
            target = ""
        if target and not isinstance(verdict.get("target_visible"), bool):
            return "verifier keep is missing/malformed required boolean 'target_visible'"
    return ""


def _preliminary_still_keep(verdict, seg, *, require_conclusive: bool) -> bool:
    """Policy-correct positive filter for one recovery candidate.

    A CHARACTER rung exists to show its named person.  A model can occasionally emit ``keep`` for
    an atmospheric frame while also saying that person is not visible; do not install that frame
    merely to make the publication-strength pass reject it one call later.
    """
    if _still_verdict_schema_error(
            verdict, seg, require_keep_facts=require_conclusive):
        return False
    if verdict.get("verdict") != "keep":
        return False
    try:
        from . import policy as _P
        character = _P.policy_of(seg) == _P.CHARACTER
    except Exception:                                  # noqa: BLE001
        character = False
    if (character and str(getattr(seg, "required_entity", "") or "").strip()
            and verdict.get("correct_subject_visible") is not True):
        return False
    return True


def still_recover(proj, seg, sel, eng_cfg, *, pool=None, cand_n: int = 8,
                  used_paths=None, log=print, require_conclusive: bool = False) -> bool:
    """Pool-wide venue-bar still recovery for one beat.

    True means a verified still was installed.  False remains the legacy/conclusive outcome for an
    empty candidate set or candidates explicitly rejected by vision.  Specificity-changing callers
    set ``require_conclusive`` so a missing/malformed verdict becomes a typed technical exception
    rather than being mistaken for proof that the footage pool lacks the requested scene.
    """
    from . import image_fallback as IF
    if getattr(sel, "image_path", ""):
        # A path/legacy `still_verified` flag is not proof of its pixels. The 101-beat run installed
        # three contextual exact stills, counted them as healed, then the strict contract correctly
        # rejected all three. Only bound strict evidence may short-circuit this recovery pass.
        try:
            from .relevance_contract import verified_still_coverage
            ok, why = verified_still_coverage(sel, seg)
        except Exception as exc:                         # noqa: BLE001 — fail closed
            ok, why = False, f"still evidence check failed: {type(exc).__name__}"
        if ok:
            return True
        _discard_invalid_still(sel, why, log)
    pool = pool if pool is not None else _clean_pool(proj)
    used_paths = used_paths if used_paths is not None else set()
    try:
        from . import policy as _policy_rank
        character_general = _policy_rank.policy_of(seg) == _policy_rank.CHARACTER
    except Exception:                                  # noqa: BLE001
        character_general = False
    if character_general:
        # The exact storyboard was deliberately surrendered before this rung.  Keeping it in the
        # CLIP query hid ordinary Jaime frames below the same absent street-attack demand that the
        # verifier no longer asks.  Rank by the current promise: named subject + narration only.
        q = " ".join(x for x in (
            getattr(seg, "required_entity", ""), getattr(seg, "text", "")) if x)
    else:
        q = " ".join(x for x in (getattr(seg, "scene_query", ""),
                                 getattr(seg, "expected_visual", "")) if x) or seg.text

    # SPEED: rank via the certified persisted-embed path (_shot_relevance is numerically
    # identical to _clip_relevance — the persisted row IS the vector it would recompute;
    # falls back to the live CLIP pass per candidate when a row is missing/stale). One
    # embeds cache per call; ordering and every downstream decision unchanged.
    from . import index as _ix_sr
    _emb_cache: dict = {}

    def _embeds_of(sid):
        if sid not in _emb_cache:
            try:
                _emb_cache[sid] = _ix_sr.load_embeds_verified(proj, sid)
            except Exception:                            # noqa: BLE001
                _emb_cache[sid] = (None, None)
        return _emb_cache[sid]

    _vv_cache = _venue_cache(proj)
    _rel_memo: dict = {}
    ranked = []
    for sid, sh in pool:
        rel = None
        try:
            rel = IF._shot_relevance(sh, Path(sh.keyframe_path), q,
                                     embeds_of=_embeds_of, rel_memo=_rel_memo)
        except Exception:                                # noqa: BLE001
            rel = None
        ranked.append((rel if rel is not None and rel >= 0 else 0.0, sid, sh))
    ranked.sort(key=lambda c: -c[0])

    # The accept decision is UNCHANGED — walk the ranked list in order, install the first
    # 'keep'. Verdicts are warmed concurrently for wall-clock, but in WAVES rather than all
    # cand_n at once: the winner is usually rank 1-2, so warming the whole window bought
    # verdicts nobody reads (measured ~5-7 discarded per successful heal). A wave stops as
    # soon as it contains a keep, because everything after the winner is unreadable by
    # construction. used_paths is filtered BEFORE the window (a used path never consumed a
    # `tried` slot in the serial walk either) and is static during this call.
    cands = [(rel, sid, sh) for rel, sid, sh in ranked
             if sh.keyframe_path not in used_paths][:cand_n]
    _nw = min(_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS", 4), 6)
    _wave = max(1, _env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WAVE", 4))
    verdicts: dict = {}

    def _warm_wave(batch):
        """Warm one wave; True when it contains a keep (so no later wave is worth paying for)."""
        if len(batch) < 2 or _nw <= 1:
            return False
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=_nw) as _ex:
                _fs = {_ex.submit(_venue_verify, sh.keyframe_path, seg,
                                  getattr(sh, "face_ids", []), eng_cfg,
                                  proj=proj, cache=_vv_cache): sh.keyframe_path
                       for _rel, _sid, sh in batch}
                for _f in _cf.as_completed(_fs):
                    try:
                        verdicts[_fs[_f]] = _f.result()
                    except Exception:                    # noqa: BLE001
                        pass                             # serial re-ask below
        except Exception:                                # noqa: BLE001
            return False
        return any(_preliminary_still_keep(
                       verdicts.get(sh.keyframe_path), seg,
                       require_conclusive=require_conclusive)
                   for _r, _s, sh in batch)

    for _i in range(0, len(cands), _wave):
        if _warm_wave(cands[_i:_i + _wave]):
            break
    for rel, sid, sh in cands:
        v = verdicts.get(sh.keyframe_path)
        if v is None:
            v = _venue_verify(sh.keyframe_path, seg, getattr(sh, "face_ids", []), eng_cfg,
                              proj=proj, cache=_vv_cache)
        # Only a recognized judgment with its ORIGINAL status absent/ok may influence the ladder.
        # A transport/error wrapper carrying a stale/default ``keep`` is still unjudged.  Strict
        # callers receive a typed retryable exception; legacy search simply refuses to install it.
        schema_error = _still_verdict_schema_error(
            v, seg, require_keep_facts=require_conclusive)
        if require_conclusive and schema_error:
            raise InconclusiveStillVerificationError(
                int(getattr(seg, "index", -1)), "specificity-ladder venue verification",
                candidate=str(sh.keyframe_path or ""),
                detail=schema_error)
        if not schema_error and _preliminary_still_keep(
                v, seg, require_conclusive=require_conclusive):
            _install_still(sel, sh.keyframe_path, sid, int(getattr(sh, "index", -1)), rel)
            used_paths.add(sh.keyframe_path)
            log(f"self-heal: beat {seg.index} — verified still installed "
                f"({sid[:38]} shot {getattr(sh, 'index', '?')}, rel {rel:.2f})")
            return True
    return False


def _frame_text_dirty(fp: str) -> bool:
    """Deterministic burned-text screen for a RAW sampled frame — the same class of checks the
    index applies to shots, because region frames bypass the persisted flags entirely.
    Measured necessity: the first live self-heal run installed a frame with giant Italian
    meme text + a cartoon watermark, and another with a French burned subtitle — the vision
    venue bar judges the SCENE and ignores overlay text by design, so text safety must be
    deterministic and happen BEFORE vision."""
    try:
        import numpy as np
        from PIL import Image
        from .index import _flags_from_frames
        im = Image.open(fp).convert("L").resize((640, 360))
        arr = np.asarray(im, dtype="float32")
        flags = _flags_from_frames([arr])
        if int(flags.get("subs_flag", 0) or 0) == 1:
            return True                                  # subtitle band (any script)
    except Exception:                                    # noqa: BLE001
        return True                                      # cannot check → do not air
    try:
        from . import ocr as _ocr
        txt = (_ocr.read_text(fp) or "").strip() if hasattr(_ocr, "read_text") else ""
        words = re.findall(r"[A-Za-z']{2,}", txt)
        if len(words) >= 3 or sum(len(w) for w in words) >= 14:
            return True                                  # readable overlay text never airs
    except Exception:                                    # noqa: BLE001
        pass                                             # OCR unavailable → band check stands
    return False


def _region_frames_recover(proj, seg, sel, src, eng_cfg, *, log=print, n: int = 8) -> bool:
    """Coarse-shot/dark-source fallback: sample frames straight from the FILE across its span,
    venue-verify the brightest few (the arrest-scene trick from the manual recovery)."""
    from .config import ffmpeg_exe
    try:
        import numpy as np
        from PIL import Image
    except Exception:                                    # noqa: BLE001
        return False
    dur = float(getattr(src, "duration", 0) or 0)
    if not (getattr(src, "local_path", None) and dur > 4):
        return False
    kfdir = Path(proj.index_dir) / src.id / "keyframes"
    kfdir.mkdir(parents=True, exist_ok=True)
    # THE FRAME'S OWNING SHOT IS KNOWN — RECORD IT.
    #
    # These frames are sampled at arbitrary timestamps, so they are not themselves detected shots,
    # and this used to install them declaring `shot: -1`. The image-lineage gate then has to prove
    # the still belongs to source X shot -1, which cannot exist, and it correctly refuses — killing
    # a finished 146-beat render at the very last gate (job 229233891e, beat 108, frame
    # selfheal_202.jpg).
    #
    # But a frame at time t is not unowned: it lies inside exactly one indexed shot, and that shot
    # is on disk already. Resolving it turns an unprovable claim into a provable one, which is what
    # the gate was asking for all along. If no shot contains t — an unindexed source — nothing is
    # installed, because a still whose lineage cannot be established must not air.
    try:
        from .index import load_shots as _load_shots_sh
        _shots = _load_shots_sh(proj, src.id) or []
    except Exception:                                    # noqa: BLE001
        _shots = []

    def _owning_shot(_t: float) -> int:
        for _sh in _shots or []:
            try:
                if float(getattr(_sh, "start", 0.0)) <= _t < float(getattr(_sh, "end", 0.0)):
                    return int(getattr(_sh, "index", -1))
            except (TypeError, ValueError):
                continue
        return -1

    cands = []
    for k in range(n):
        t = (k + 0.5) / n * dur
        fp = kfdir / f"selfheal_{int(t)}.jpg"
        try:
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
                            "-i", src.local_path, "-frames:v", "1", "-q:v", "2", str(fp)],
                           capture_output=True, timeout=60)
        except Exception:                                # noqa: BLE001
            continue
        if fp.exists():
            try:
                arr = np.asarray(Image.open(fp).convert("L"), dtype="float32")
                cands.append((float(arr.mean()), str(fp), _owning_shot(t)))
            except Exception:                            # noqa: BLE001
                continue
    cands.sort(reverse=True)
    for luma, fp, _shot in cands[:4]:
        if _shot < 0:
            continue                                     # unprovable lineage — never air it
        if _frame_text_dirty(fp):
            continue                                     # burned subs/meme text/watermark card
        v = _venue_verify(fp, seg, [], eng_cfg, proj=proj, cache=_venue_cache(proj))
        if isinstance(v, dict) and v.get("verdict") == "keep":
            _install_still(sel, fp, src.id, _shot, 0.8)
            log(f"self-heal: beat {seg.index} — region-frame still installed from {src.id[:38]}")
            return True
    return False


# ── targeted acquisition ──────────────────────────────────────────────────────────────────────

def _llm_queries(seg, movie: str) -> list[str]:
    """Precise YouTube search queries for the beat's missing scene — DeepSeek writes them,
    deterministic construction when no LLM is reachable."""
    fallback = [q for q in (
        f"{movie} {getattr(seg, 'scene_query', '') or seg.text} scene".strip(),
        f"{movie} {getattr(seg, 'required_entity', '')} "
        f"{' '.join((getattr(seg, 'expected_visual', '') or '').split()[:6])} scene".strip(),
    ) if len(q.split()) >= 3]
    if not _env_on("VIDLORE_CLIPSTUDIO_SELFHEAL_LLM", "1"):
        return fallback
    try:
        from . import llm
        out, _meta = llm.complete_ex(
            system="You write precise YouTube search queries for finding an exact TV scene. "
                   "Reply ONLY a JSON array of 2 short query strings.",
            messages=[{"role": "user", "content":
                       f"Show: {movie}\nNarration: {seg.text}\n"
                       f"Visual needed: {getattr(seg, 'expected_visual', '')}\n"
                       f"Scene hint: {getattr(seg, 'scene_query', '')}"}],
            max_tokens=120)
        arr = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        qs = [str(x).strip() for x in arr if str(x).strip()][:2]
        return qs + [q for q in fallback if q not in qs]
    except Exception:                                    # noqa: BLE001
        return fallback


def _yt_search_candidates(queries: list[str], *, per_query: int = 6, log=print) -> list:
    """Direct yt-dlp ytsearch (flat) — the method that beat discovery in every manual precision
    pass: search the LLM's exact query, keep titles that look like clean scene uploads
    (title gates applied), 1-15 min long. Returns SourceCandidate-compatible objects."""
    from .discover import SourceCandidate, is_unwanted_source_title
    from . import hd_download as _hd
    # Invoke yt-dlp as a MODULE, the way every other call site does (hd_download, discover):
    # probing for a console-script FILE named `yt-dlp` next to the interpreter found nothing on
    # Windows (it is `yt-dlp.exe` in Scripts\), so footage rescue silently returned no candidates
    # there — and the `return []` was indistinguishable from "the search found nothing".
    hd_py = str(getattr(_hd, "HD_PY", "") or "")
    if not hd_py or not Path(hd_py).exists():
        log("self-heal: yt-search unavailable — no HD python (.hdvenv) found; "
            "footage rescue cannot search YouTube")
        return []
    out = []
    for q in queries[:2]:
        try:
            r = subprocess.run(
                [hd_py, "-m", "yt_dlp",
                 f"ytsearch{per_query}:{q}", "--flat-playlist", "--no-warnings",
                 "--print", "%(id)s\t%(duration)s\t%(title)s"],
                capture_output=True, text=True, timeout=90)
            for line in (r.stdout or "").splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                vid, dur, title = parts
                try:
                    d = float(dur or 0)
                except (TypeError, ValueError):
                    d = 0
                if not (45 <= d <= 900):
                    continue                              # shorts and hour-long reactions out
                if is_unwanted_source_title(title):
                    continue
                out.append(SourceCandidate(
                    id=vid, url=f"https://www.youtube.com/watch?v={vid}",
                    title=title, provider="youtube", query=q))
        except Exception as e:                            # noqa: BLE001
            log(f"self-heal: ytsearch failed for {q[:40]!r} ({str(e)[:60]})")
    return out


class InconclusiveAcquisitionError(RuntimeError):
    """Targeted footage acquisition failed technically, so the pool is not exhausted.

    An empty list from ``acquire_for_beat`` is reserved for a completed search with no candidates
    or candidates conclusively excluded by policy/content.  Discovery transport errors, failed
    downloads, and unreadable/unindexed media use this type so phase 1 retries without spending the
    beat's specificity.
    """

    def __init__(self, beat_index: int, stage: str, *, detail: str = ""):
        bits = [f"beat {beat_index}", f"targeted acquisition {stage}"]
        if detail:
            bits.append(str(detail))
        super().__init__(" — ".join(bits))
        self.beat_index = int(beat_index)
        self.stage = str(stage or "acquisition")
        self.detail = str(detail or "")


def acquire_for_beat(proj, seg, cfg, *, policy: str, faceid_obj=None, refs=None,
                     roster=None, log=print) -> list:
    """Discover→download→index up to SELFHEAL_MAX_SRC new sources for this beat's scene.
    Direct ytsearch results (the precision method) are considered FIRST, the broader
    discovery machinery second. Returns the newly indexed SourceVideo objects."""
    from . import discover as D
    from . import index as I
    from .download import download_candidates
    import dataclasses as _dc
    movie = str(((getattr(proj, "meta", {}) or {}).get("analysis") or {})
                .get("movie_title") or "")
    queries = _llm_queries(seg, movie)
    log(f"self-heal: beat {seg.index} — acquisition queries: {queries[:2]}")
    a_meta = (proj.meta or {}).get("analysis") or {}
    ana = type("A", (), dict(
        movie_title=movie or "the show", year=str(a_meta.get("year", "")),
        topic=seg.text[:80],
        anchor_scenes=[{"name": (getattr(seg, "scene_query", "") or seg.text)[:60],
                        "query": q, "episode": "", "dialogue": []} for q in queries[:2]],
        characters=a_meta.get("characters") or [], actors=a_meta.get("actors") or [],
        events=[], key_scenes=[], locations=a_meta.get("locations") or [],
        visual_keywords=[], episode_hint="", episode_hint_verified=False,
        video_type="multi_scene"))()
    cfg_r = _dc.replace(cfg, discover_target=16)
    cands = _yt_search_candidates(queries, log=log)
    try:
        cands = cands + (D.discover_sources(
            ana, cfg_r, segments=[seg], progress=None,
            # Targeted recovery is not the normal best-effort discovery API.  These authored
            # queries must each receive a real provider answer (including legitimate empty); the
            # discovery layer raises when every provider remains technical after its retries.
            extra_queries=queries, required_queries=queries) or [])
    except Exception as e:                               # noqa: BLE001
        log(f"self-heal: discovery failed ({str(e)[:60]})")
        raise InconclusiveAcquisitionError(
            int(getattr(seg, "index", -1)), "discovery",
            detail=f"{type(e).__name__}: {e}") from e
    have = {(s.url or "").strip() for s in proj.sources}
    qtoks = {w.lower() for q in queries for w in re.findall(r"[a-z']{3,}", q.lower())}
    new = [c for c in cands if (c.url or "").strip() and (c.url or "").strip() not in have]
    new.sort(key=lambda c: sum(1 for w in re.findall(r"[a-z']{3,}", (c.title or "").lower())
                               if w in qtoks), reverse=True)
    new = new[:max(1, _env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_MAX_SRC", 2))]
    if not new:
        log(f"self-heal: beat {seg.index} — no new source found")
        return []
    for c in new:
        log(f"self-heal: beat {seg.index} — fetching {(c.title or '')[:70]!r}")

    # Source rows and their search indexes are one retryable acquisition transaction.  A failed row
    # left in ``proj.sources`` makes the next pass's ``have`` set suppress that same URL forever;
    # a partial index can similarly masquerade as a searched pool.  Restore the exact pre-call
    # manifest and purge only newly-created source indexes on every technical abort.  Downloaded
    # media bytes may remain as an orphaned resumable cache, but no project row authorizes them.
    source_snapshot = [(sv, copy.deepcopy(vars(sv))) for sv in proj.sources]
    preexisting_ids = {str(getattr(sv, "id", "") or "") for sv, _state in source_snapshot}

    def _rollback_acquisition() -> str:
        current_ids = {str(getattr(sv, "id", "") or "") for sv in proj.sources}
        for sid in sorted(current_ids - preexisting_ids):
            if sid:
                try:
                    I.purge_source_index(proj, sid)
                except Exception:                        # noqa: BLE001 — report via retry state
                    pass
        restored = []
        for sv, state in source_snapshot:
            vars(sv).clear()
            vars(sv).update(copy.deepcopy(state))
            restored.append(sv)
        proj.sources = restored
        try:
            save = getattr(proj, "save", None)
            if callable(save):
                save()
            return ""
        except Exception as exc:                         # noqa: BLE001
            return f"; rollback save failed: {type(exc).__name__}: {exc}"

    def _abort(stage: str, detail: str, cause=None):
        rollback_error = _rollback_acquisition()
        err = InconclusiveAcquisitionError(
            int(getattr(seg, "index", -1)), stage,
            detail=f"{detail}{rollback_error}")
        if cause is not None:
            raise err from cause
        raise err

    try:
        download_candidates(proj, new, cfg, policy=policy, progress=None)
    except Exception as e:                               # noqa: BLE001
        log(f"self-heal: download failed ({str(e)[:60]})")
        _abort("download", f"{type(e).__name__}: {e}", e)

    wanted_urls = {(c.url or "").strip() for c in new if (c.url or "").strip()}
    outcomes = [sv for sv in proj.sources
                if (getattr(sv, "url", "") or "").strip() in wanted_urls]
    usable = [sv for sv in outcomes
              if sv.status == SOURCE_OK and bool(getattr(sv, "local_path", ""))]
    failed = [sv for sv in outcomes if sv.status == SOURCE_FAILED]
    if failed:
        detail = ", ".join(
            f"{getattr(sv, 'id', '?')}:{(getattr(sv, 'error', '') or 'download_failed')[:80]}"
            for sv in failed)
        # Partial success is still an incomplete pool: the failed sibling may be the exact source.
        # Never proceed to a no-hit/softening decision after silently ignoring it.
        _abort("download", detail)
    if not usable:
        # An explicit policy block is a conclusive outcome: those candidates are outside the
        # authorized pool. A checksum duplicate is content-conclusive too: its bytes are already
        # represented in the usable pool. Missing/unknown outcomes remain technical.
        if outcomes and all(sv.status in (SOURCE_BLOCKED, "duplicate") for sv in outcomes):
            log(f"self-heal: beat {seg.index} — all new sources excluded by policy/duplicate")
            return []
        _abort("download", "downloader produced no usable source or conclusive policy result")

    fresh = []
    index_errors = []
    for sv in usable:
        try:
            # Late-acquired footage must be indexed with the same identity/name capabilities as
            # the main pool.  Omitting these built a weaker cache (faceid=False, roster=False), so
            # every Resume re-indexed the source and changed the pool fingerprint before healing.
            shots = I.index_source(
                proj, sv, cfg, references=refs, faceid=faceid_obj,
                roster=roster, progress=None)
            if not shots:
                index_errors.append(f"{sv.id}:zero_shots")
                log(f"self-heal: index produced 0 shots for {sv.id}")
                continue
            fresh.append(sv)
        except Exception as e:                           # noqa: BLE001
            index_errors.append(f"{sv.id}:{type(e).__name__}:{str(e)[:80]}")
            log(f"self-heal: index failed for {sv.id} ({str(e)[:60]})")
    if index_errors:
        # Even if a sibling source indexed, declaring the expanded pool exhausted would ignore an
        # acquired source that was never searchable.  Retry the incomplete page as infrastructure.
        _abort("index", ", ".join(index_errors))
    if not fresh:
        _abort("index", "no source produced searchable shots")
    return fresh


# ── the loop ──────────────────────────────────────────────────────────────────────────────────

def _strictly_confirm_concrete_still(proj, seg, sel, eng, *,
                                     require_conclusive: bool = False) -> tuple[bool, str]:
    """Bind an exact/character still decision to the actual bytes under the strict contract.

    ``still_recover`` intentionally uses the cheaper venue/context question to search. That answer
    is a candidate filter, not publication proof. Re-ask the exact bytes with the strict concrete
    question before claiming recovery succeeded; otherwise a lenient contextual still can silently
    bypass the relevance contract before or after policy softening.
    """
    from .verify import NonRetryableBuildError

    path = str(getattr(sel, "image_path", "") or "")
    if not path or not Path(path).is_file():
        if require_conclusive:
            raise InconclusiveStillVerificationError(
                int(getattr(seg, "index", -1)), "strict concrete-still verification",
                candidate=path, detail="installed candidate bytes are missing")
        return False, "character-rung still is missing"
    try:
        from . import verify as V
        from . import policy as P
        from . import relevance_contract as R
        analysis = (getattr(proj, "meta", {}) or {}).get("analysis", {}) or {}
        era = str(analysis.get("episode_hint", "") or "")
        exact = P.policy_of(seg) == P.EXACT
        meta = dict(getattr(sel, "image_meta", {}) or {})
        owned_source_frame = bool(
            str(meta.get("source", "") or "") in ("source-frame", "source-frame-recovery")
            and str(meta.get("src", "") or "") and meta.get("shot") is not None)
        if owned_source_frame:
            # Index keyframes are search thumbnails (normally 512x288).  The real-scale ladder
            # found sixteen correct Jaime candidates, then rejected every one for thumbnail size
            # before build's existing native-owner lane could run.  Materialize the exact indexed
            # midpoint from its source now and freshly judge those native pixels; thumbnail verdicts
            # are never transferred, native-HD is still mandatory, and lineage stays hash-bound.
            from .build import _rescue_still_fullres
            from .verify import VisionBackendError
            try:
                rescue = _rescue_still_fullres(
                    proj, sel, path, lambda _message: None, seg=seg, eng=eng,
                    allow_semantic_reject=True, refresh_semantic_verdict=True)
            except NonRetryableBuildError as exc:
                if str(getattr(exc, "kind", "") or "") in (
                        "selection_relevance", "native_resolution"):
                    return False, str(exc)
                raise
            except VisionBackendError as exc:
                if require_conclusive:
                    raise InconclusiveStillVerificationError(
                        int(getattr(seg, "index", -1)),
                        "native concrete-still verification", candidate=path,
                        detail=str(exc)) from exc
                return False, f"native concrete-still verifier was inconclusive: {exc}"
            why = str(rescue.get("semantic_strict_reason") or "")
            if why:
                return False, why
            verdict = dict(rescue.get("semantic_verifier") or {})
            native_path = str(rescue.get("path") or "")
            native_hash = str(rescue.get("file_sha256") or "")
            if not native_path or not native_hash:
                if require_conclusive:
                    raise InconclusiveStillVerificationError(
                        int(getattr(seg, "index", -1)),
                        "native concrete-still verification", candidate=path,
                        detail="native materialization omitted path/hash provenance")
                return False, "native materialization omitted path/hash provenance"
            sel.image_path = native_path
            meta.update({
                "still_verification_attempted": True,
                "still_verified": True,
                "still_semantic_verified": True,
                "still_verifier": verdict,
                "still_image_sha256": native_hash,
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
            covered, coverage_why = R.verified_still_coverage(sel, seg, proj=proj)
            return (True, "") if covered else (
                False, coverage_why or "native strict coverage absent")
        # The bound gap review has already authorized EXACT -> CHARACTER before this helper is
        # reached.  Requiring the original exact-scene question here made that first rung
        # unsatisfiable by construction: a clean related ship/Ned frame passed the venue filter,
        # then was asked to prove the same missing exact moment that caused the downgrade.  Ask the
        # current policy's question while retaining every positive pixel/subject/quality fact below.
        # EXACT callers remain strict; only the deliberately softened CHARACTER rung is general.
        verdict = V.verify_frame(
            path, getattr(seg, "text", "") or "",
            getattr(seg, "required_entity", "") or "",
            getattr(seg, "required_kind", "") or "", [], eng,
            getattr(eng, "anthropic_model", ""), is_specific=exact,
            expected_visual=getattr(seg, "expected_visual", "") or "",
            scene_query=getattr(seg, "scene_query", "") or "", era_hint=era,
            venue_fallback=False, must_see=P.deictic_target(seg))
        schema_error = _still_verdict_schema_error(
            verdict, seg, require_keep_facts=True)
        if schema_error:
            if require_conclusive:
                raise InconclusiveStillVerificationError(
                    int(getattr(seg, "index", -1)), "strict concrete-still verification",
                    candidate=path, detail=schema_error)
            return False, f"strict character-rung verifier was inconclusive: {schema_error}"
        # Native verify_frame successes omit status; normalize that one legacy-success shape for
        # the persisted publication evidence.  Crucially, explicit non-ok statuses were rejected
        # above and are never overwritten/laundered into ``ok``.
        verdict = dict(verdict)
        if not str(verdict.get("status", "") or "").strip():
            verdict["status"] = "ok"
        why = R.strict_still_evidence_reason(verdict, seg)
        meta.update({
            "still_verification_attempted": True,
            "still_verified": not bool(why),
            "still_semantic_verified": not bool(why),
            "still_verifier": verdict,
            "still_image_sha256": R.image_sha256(path),
            "exact_still_verified": bool(exact and not why),
            "exact_still_verifier": (verdict if exact else {}),
            "relevance_class": ("exact_scene" if exact else "contextual_fallback"),
        })
        sel.image_meta = meta
        if why:
            return False, why
        covered, coverage_why = R.verified_still_coverage(sel, seg, proj=proj)
        return (True, "") if covered else (False, coverage_why or "strict coverage absent")
    except InconclusiveStillVerificationError:
        raise
    except NonRetryableBuildError:
        # Deterministic owner/keyframe/hash corruption is a publication invariant, not a verifier
        # outage. Preserve its non-retryable identity so Resume cannot loop on unchanged bytes.
        raise
    except Exception as exc:                             # noqa: BLE001 — fail closed
        if require_conclusive:
            raise InconclusiveStillVerificationError(
                int(getattr(seg, "index", -1)), "strict concrete-still verification",
                candidate=path,
                detail=f"{type(exc).__name__}: {exc}") from exc
        return False, f"strict character-rung verification failed: {type(exc).__name__}: {exc}"


def _soften_and_retry(proj, seg, sel, eng, pool, used, log, *,
                      surrender_verbatim_quote: bool = False) -> bool:
    """Soften an unreachable exact beat and search the SAME pool once more under the looser bar.

    Deliberately no re-acquisition and no new pool: the point is not to try harder, it is to stop
    asking for something retrieval cannot return. The candidate depth is raised because a
    character-policy still only has to show the right subject, so it is worth looking past the
    handful of frames that already failed the exact test."""
    if not _env_on("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFTEN", "1"):
        return False
    _n = _env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFT_CANDS", 16)
    seg_before = copy.deepcopy(vars(seg))
    sel_before = copy.deepcopy(vars(sel))
    used_before = copy.deepcopy(used)

    def _rollback() -> None:
        vars(seg).clear()
        vars(seg).update(copy.deepcopy(seg_before))
        vars(sel).clear()
        vars(sel).update(copy.deepcopy(sel_before))
        if hasattr(used, "clear") and hasattr(used, "update"):
            used.clear()
            used.update(copy.deepcopy(used_before))

    try:
        character_rejects = []
        if _soften_to_character(
                seg, log, surrender_verbatim_quote=surrender_verbatim_quote):
            # A cheap candidate question can produce a false-positive holding image.  Continue the
            # already-bounded ranked window after the final byte-bound proof rejects it instead of
            # abandoning CHARACTER after one frame.  Rejected paths remain excluded only during
            # this rung and are released before return/fall-through for unrelated beats.
            for _attempt in range(_n):
                if not still_recover(
                        proj, seg, sel, eng, pool=pool, used_paths=used, cand_n=_n, log=log,
                        require_conclusive=True):
                    break
                candidate_path = str(getattr(sel, "image_path", "") or "")
                ok, why = _strictly_confirm_concrete_still(
                    proj, seg, sel, eng, require_conclusive=True)
                if ok:
                    for path in character_rejects:
                        used.discard(path)
                    sel.image_meta["selfheal_rung"] = "character_specific"
                    return True
                _discard_invalid_still(sel, why, log)
                if candidate_path:
                    character_rejects.append(candidate_path)
            for path in character_rejects:
                used.discard(path)
        # SECOND AND FINAL RUNG. Dropping the requirement is not enough when the NARRATION itself
        # names the unfindable thing: beat 24 reads "The flayed man banners brought down to the
        # ground", and the still verifier judges candidates against that sentence, so no frame in
        # the pool can pass however loose the policy is. `_soften_to_abstract` is the pipeline's own
        # designed escape for a beat nothing can satisfy: it rewrites the beat as visual rest and
        # takes era-appropriate atmosphere. Reached only after the character rung also fails.
        _soften_to_abstract(
            seg, log, cause="exact and character-specific recovery both exhausted")
        if still_recover(
                proj, seg, sel, eng, pool=pool, used_paths=used, cand_n=_n, log=log,
                require_conclusive=True):
            sel.image_meta["selfheal_rung"] = "abstract"
            return True
        # A conclusive exhausted ladder is still a normal False result, but it cannot retain any
        # policy, still, evidence, or used-path mutation made while trying the two rungs.
        _rollback()
        return False
    except Exception:
        # This helper is also called by phase 1, outside the post-gap outer transaction.  Own the
        # rollback here so verifier transport/schema failures can never strand a softened beat.
        _rollback()
        raise


def heal_blocked_beats(proj, segments, cfg, *, blocked: list[int], policy: str,
                       allow_acquire: bool = True, faceid_obj=None, refs=None,
                       roster=None, log=print) -> int:
    """One healing pass over `blocked` beat indexes. Returns beats resolved this pass.

    Adding footage/stills is always allowed.  Changing an authored visual promise is not: both
    phase-1 policy-mutation sites below require a current, evidence-bound gap review.  This keeps
    the broad pre-assembly predictor from silently abstracting a beat the viewer never classified,
    or one whose pool changed after that classification.
    """
    from .config import engine_config
    from . import policy as _P
    eng = engine_config()
    segs = {s.index: s for s in segments}
    sels = {s.segment_index: s for s in proj.selections}
    pool = None
    used = {getattr(s, "image_path", "") for s in proj.selections
            if getattr(s, "image_path", "")}
    resolved = 0
    logged_denials: set[tuple[int, str]] = set()

    def _may_soften(seg) -> bool:
        ok, reason = _phase1_softening_authorization(proj, seg, cfg)
        if not ok:
            key = (int(getattr(seg, "index", -1)), reason)
            if key not in logged_denials:
                logged_denials.add(key)
                log(f"self-heal: beat {key[0]} — specificity softening DENIED ({reason}); "
                    "strict authored policy preserved")
        return ok

    def _accept_installed(seg, sel) -> bool:
        if _P.policy_of(seg) not in (_P.EXACT, _P.CHARACTER):
            return True
        ok, why = _strictly_confirm_concrete_still(proj, seg, sel, eng)
        if not ok:
            _discard_invalid_still(sel, why, log)
        return ok

    # Stable-partition independently exhausted beats ahead of every acquisition-capable beat.
    # The authorization is read-only and is still recomputed immediately before mutation below.
    # Without this ordering, an ordinary earlier beat can download/index a new SOURCE_OK file and
    # correctly stale a later beat's whole-pool absence proof before that proof can be consumed.
    # Only beats carrying a strict-exhaustion record need the comparatively expensive semantic +
    # whole-index validation; every other beat retains its original relative order.
    review = (getattr(proj, "meta", {}) or {}).get(
        "selection_relevance_gap_review") or {}
    exhaustion_records = review.get("strict_acquisition_exhaustion") or {}
    priority, ordinary = [], []
    for bidx in blocked:
        seg = segs.get(bidx)
        has_record = isinstance(exhaustion_records, dict) \
            and isinstance(exhaustion_records.get(str(bidx)), dict)
        authorized = bool(seg is not None and has_record
                          and _phase1_reviewed_exhaustion_authorization(proj, seg, cfg)[0])
        (priority if authorized else ordinary).append(bidx)
    healing_order = priority + ordinary
    if priority and healing_order != list(blocked):
        log("self-heal: consuming audited strict-exhaustion beat(s) before "
            f"pool-mutating acquisition: {priority}")

    for bidx in healing_order:
        seg = segs.get(bidx)
        sel = sels.get(bidx)
        if seg is None or sel is None:
            continue
        # Targeted acquisition is an infrastructure transaction with respect to the authored beat.
        # Earlier strict-search steps may detach a stale still or touch ``used`` before acquisition
        # discovers that search/download/index never completed.  Preserve the full in-memory state
        # so that technical uncertainty cannot leave even those incidental mutations behind.
        beat_seg_before = copy.deepcopy(vars(seg))
        beat_sel_before = copy.deepcopy(vars(sel))
        beat_used_before = copy.deepcopy(used)

        def _rollback_beat() -> None:
            vars(seg).clear()
            vars(seg).update(copy.deepcopy(beat_seg_before))
            vars(sel).clear()
            vars(sel).update(copy.deepcopy(beat_sel_before))
            if hasattr(used, "clear") and hasattr(used, "update"):
                used.clear()
                used.update(copy.deepcopy(beat_used_before))

        # Preserve the strict, pre-heal selection as the reversible target even if ordinary search
        # first detaches a stale still or acquisition later changes working fields.
        marker_original = _capture_softening_state(seg, sel)
        if getattr(sel, "image_path", ""):
            try:
                from .relevance_contract import verified_still_coverage
                _covered, _why = verified_still_coverage(sel, seg)
            except Exception as _exc:                    # noqa: BLE001 — fail closed
                _covered, _why = False, f"still evidence check failed: {type(_exc).__name__}"
            if _covered:
                resolved += 1
                continue
            _discard_invalid_still(sel, _why, log)
        if pool is None:
            pool = _clean_pool(proj)
        if beat_unfillable(seg) and _may_soften(seg):
            def _try_unfillable_abstract() -> bool:
                _soften_to_abstract(seg, log)
                ok = still_recover(
                    proj, seg, sel, eng, pool=pool, used_paths=used,
                    cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", 8),
                    log=log, require_conclusive=True)
                if ok:
                    sel.image_meta["selfheal_rung"] = "abstract"
                return ok

            if _phase1_softening_attempt(
                    proj, seg, sel, used, _try_unfillable_abstract,
                    basis="phase1_gate_forbidden_content",
                    marker_original=marker_original, cfg=cfg):
                resolved += 1
                continue
        if still_recover(proj, seg, sel, eng, pool=pool, used_paths=used,
                         cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", 8),
                         log=log):
            if _accept_installed(seg, sel):
                resolved += 1
                continue
        # A current, independently audited exhaustion record means strict acquisition already ran
        # against this exact beat and this exact searchable pool.  Consume that evidence before
        # ``acquire_for_beat`` can add media and make its own authorization stale.  Ordinary
        # unreviewed/stale/incomplete beats keep the acquire-first path below unchanged.
        reviewed_authorization, _review_reason = \
            _phase1_reviewed_exhaustion_details(proj, seg, cfg)
        if reviewed_authorization is not None:
            if _phase1_softening_attempt(
                    proj, seg, sel, used,
                    lambda: _soften_and_retry(
                        proj, seg, sel, eng, pool, used, log,
                        surrender_verbatim_quote=bool(
                            reviewed_authorization.get("surrender_verbatim_quote"))),
                    basis="phase1_bound_strict_acquisition_exhausted",
                    marker_original=marker_original,
                    authorization=reviewed_authorization, cfg=cfg):
                resolved += 1
            else:
                log(f"self-heal: beat {bidx} — audited specificity ladder exhausted; "
                    "strict state restored")
            # Do not re-acquire after consuming a completed exhaustion audit.  Doing so would
            # mutate the pool, invalidate the review, and recreate the stale-authorization loop.
            continue
        if allow_acquire:
            try:
                fresh = acquire_for_beat(
                    proj, seg, cfg, policy=policy, faceid_obj=faceid_obj,
                    refs=refs, roster=roster, log=log)
            except InconclusiveAcquisitionError:
                _rollback_beat()
                raise
            if fresh:
                pool = _clean_pool(proj)                 # refresh with the new shots
                if still_recover(proj, seg, sel, eng, pool=pool, used_paths=used,
                                 cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", 8),
                                 log=log):
                    if _accept_installed(seg, sel):
                        resolved += 1
                        continue
                for sv in fresh:
                    if _region_frames_recover(proj, seg, sel, sv, eng, log=log):
                        if _accept_installed(seg, sel):
                            resolved += 1
                            break
                else:
                    if _may_soften(seg):
                        if _phase1_softening_attempt(
                                proj, seg, sel, used,
                                lambda: _soften_and_retry(
                                    proj, seg, sel, eng, pool, used, log),
                                basis="phase1_acquisition_exhausted",
                                marker_original=marker_original, cfg=cfg):
                            resolved += 1
                            continue
                    log(f"self-heal: beat {bidx} unresolved this pass")
                    continue
                continue
        if _may_soften(seg):
            if _phase1_softening_attempt(
                    proj, seg, sel, used,
                    lambda: _soften_and_retry(proj, seg, sel, eng, pool, used, log),
                    basis="phase1_strict_recovery_exhausted",
                    marker_original=marker_original, cfg=cfg):
                resolved += 1
                continue
        log(f"self-heal: beat {bidx} unresolved this pass")
    # persist this pass's venue verdicts so the NEXT round (and the review-draft retry, which
    # re-runs the whole heal) replays them instead of re-buying identical answers
    _venue_cache_save(proj)
    return resolved


_SEMANTIC_TECHNICAL_REASONS = (
    "selection_absent", "moving_source_absent", "verifier_absent", "verifier_error",
    "verifier_breaker", "verifier_unavailable", "verifier_evidence_absent",
    "verifier_evidence_schema_mismatch", "verifier_evidence_model_mismatch",
    "verifier_evidence_mismatch", "verifier_evidence_unrecomputable",
    "verifier_evidence_window_not_sampled", "matches_narration_absent",
    "specific_enough_absent", "quality_ok_absent", "wrong_subject_visible_absent",
    "correct_subject_visible_absent", "target_visible_absent", "quality_ok_false",
    "exact_quote_pool_classification_indeterminate",
    "exact_quote_unprompted_confirmation_inconclusive",
)


class InconclusiveStillVerificationError(RuntimeError):
    """A specificity-ladder candidate could not receive a conclusive vision verdict.

    ``False`` from the ladder means the available candidates were actually exhausted (there were
    none, or each was explicitly rejected).  This exception means something materially different:
    a candidate existed but the verifier did not judge it, so the caller must retry the technical
    operation without persisting a semantic downgrade or an exhaustion marker.
    """

    def __init__(self, beat_index: int, stage: str, *, candidate: str = "", detail: str = ""):
        bits = [f"beat {beat_index}", str(stage or "still verification")]
        if candidate:
            bits.append(f"candidate {candidate}")
        if detail:
            bits.append(str(detail))
        super().__init__(" — ".join(bits))
        self.beat_index = int(beat_index)
        self.stage = str(stage or "still verification")
        self.candidate = str(candidate or "")
        self.detail = str(detail or "")

# Phase 1 runs from the older rejected-footage predictor, before the selection-relevance gate has
# written its audit.  A bound human gap review proves what the *pool* lacked when it was reviewed;
# it does not turn a current verifier outage, stale evidence record, low-quality clip, or other
# pipeline fault into permission to rewrite the authored beat.  Keep this allow-list deliberately
# narrow: only an explicit, successfully-bound semantic negative can spend that authorization.
_SEMANTIC_NEGATIVE_REASONS = (
    "verdict_replace", "matches_narration_false", "specific_enough_false",
    "correct_subject_visible_false", "wrong_subject_visible_true",
    "target_visible_false", "contradicts_narration_true",
    "deterministic_contradiction", "era_ok_false",
    "exact_source_title_cast_conflict_unresolved",
)


def _phase1_current_gap_evidence(proj, seg, cfg=None, *,
                                 allow_verified_native_gap: bool = False) -> tuple[bool, str]:
    """Type the *current* phase-1 blocker before allowing specificity loss.

    The review binding below answers "did a viewer audit this authored request against this pool?"
    This second check answers the equally important "is the failure we are handling now still an
    ordinary semantic footage gap?"  Reuse the publication contract so quote typing comes from the
    same complete-pool ``find_quote_span`` scan and evidence bindings are recomputed from the exact
    current selection.  Any inability to make that determination is a denial, never a downgrade.
    """
    try:
        from . import relevance_contract as _rel
        audit = _rel.evaluate_selection_relevance(proj, [seg], cfg=cfg)
        idx = int(getattr(seg, "index", -1))
        entry = next(
            (e for e in (audit.get("checked") or [])
             if int(e.get("segment_index", -1)) == idx),
            None,
        )
        if not isinstance(entry, dict):
            return False, "phase1_current_relevance_evidence_absent"

        quote = str(getattr(seg, "quote", "") or "").strip()
        if quote:
            branch = str((entry.get("quote_evidence") or {}).get("branch", "") or "")
            if branch == "verbatim" and not allow_verified_native_gap:
                return False, "phase1_verbatim_quote_promise"
            if branch not in (("paraphrase", "verbatim")
                              if allow_verified_native_gap else ("paraphrase",)):
                # Empty is also indeterminate: every strict quoted beat should have been typed by
                # the complete-pool scan, and phase 1 has no authority to guess if that evidence is
                # absent or malformed.
                return False, "phase1_quote_pool_classification_indeterminate"

        reasons = [str(r) for r in (entry.get("reasons") or []) if str(r)]
        if not reasons:
            # A stale ``verifier_failed`` flag can make the structural predictor call self-heal
            # even though the current semantic contract passes.  That is pipeline state to repair,
            # not evidence that the requested scene is absent.
            return False, "phase1_current_semantic_blocker_absent"

        technical = [
            reason for reason in reasons
            if any(reason.startswith(prefix) for prefix in _SEMANTIC_TECHNICAL_REASONS)
        ]
        if technical:
            return False, f"phase1_technical_or_evidence_blocker:{technical[0]}"
        if _rel.completed_deliberate_exact_downgrade(entry):
            # A bound lenient KEEP is positive evidence for the looser scene and negative evidence
            # for the authored exact promise.  The review still has to be current and pool-bound;
            # this only types the current failure so that authorization may reach that ladder.
            return True, "phase1_confirmed_deliberate_exact_downgrade"
        semantic = [
            reason for reason in reasons
            if reason in _SEMANTIC_NEGATIVE_REASONS
            or (allow_verified_native_gap and reason in _NATIVE_QUOTE_CONTENT_REASONS)
        ]
        nonsemantic = [reason for reason in reasons if reason not in semantic]
        if nonsemantic:
            return False, f"phase1_technical_or_evidence_blocker:{nonsemantic[0]}"
        if not semantic:
            return False, "phase1_explicit_semantic_negative_absent"
        return True, "phase1_confirmed_ordinary_semantic_gap"
    except Exception as exc:                              # noqa: BLE001 — mutation must fail closed
        return False, f"phase1_relevance_evidence_error:{type(exc).__name__}"


def _gap_beat_fingerprint(seg) -> str:
    """Identity of the authored visual promise a human/pool audit actually classified."""
    payload = {
        "index": int(getattr(seg, "index", -1)),
        "text": str(getattr(seg, "text", "") or ""),
        "visual_policy": str(getattr(seg, "visual_policy", "") or ""),
        "required_entity": str(getattr(seg, "required_entity", "") or ""),
        "required_kind": str(getattr(seg, "required_kind", "") or ""),
        "expected_visual": str(getattr(seg, "expected_visual", "") or ""),
        "scene_query": str(getattr(seg, "scene_query", "") or ""),
        "quote": str(getattr(seg, "quote", "") or ""),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _gap_pool_fingerprint(proj, *, include_banned: bool = False) -> tuple[str, int]:
    """Bind a footage-pool decision to source bytes and their searchable indexes.

    Active softenings use the publishable pool (the default), because banning the source that
    supplies a softened still must restore the authored promise.  Absence reviews use the broader
    all-indexed universe through ``_gap_absence_pool_fingerprint``: matcher source gates may only
    *remove* media from the publishable pool, and removing an already-reviewed source cannot make a
    missing scene appear.  Keeping those rows in the absence universe prevents a deterministic
    match rerun from invalidating completed acquisition evidence before self-heal can consume it.
    """
    # Eligibility is part of the pool just as much as bytes are.  An operator/automatic ban can
    # remove a source without touching its media or index timestamps; excluding those ids here in
    # the same way as ``_clean_pool`` makes that policy change invalidate any fallback that depended
    # on the old pool.
    from .match import banned_source_ids
    banned = set() if include_banned else banned_source_ids(proj, include_auto=True)
    rows = []
    for src in sorted((getattr(proj, "sources", None) or []),
                      key=lambda s: str(getattr(s, "id", "") or "")):
        if str(getattr(src, "status", "") or "") != SOURCE_OK:
            continue
        sid = str(getattr(src, "id", "") or "")
        if sid in banned:
            continue

        def _stat(path) -> list:
            try:
                st = Path(path).stat()
                return [int(st.st_size), int(st.st_mtime_ns)]
            except Exception:                            # noqa: BLE001 — absence is part of identity
                return [0, 0]

        # Search consumes more than the shot/word JSON.  In particular the CLIP matrix decides
        # which frame is retrieved, its manifest binds rows to keyframes/model identity, and a
        # changed keyframe can alter both live-embedding fallback and the actual frame a verifier
        # sees.  Stat every referenced keyframe rather than hashing image bytes: size + nanosecond
        # mtime is cheap enough for the preassembly authorization and still makes an index rewrite
        # invalidate a previously completed whole-pool review.
        shots_path = proj.shots_path(sid)
        keyframes = []
        try:
            shot_rows = json.loads(shots_path.read_text(encoding="utf-8"))
            referenced = sorted({
                str(row.get("keyframe_path", "") or "")
                for row in shot_rows if isinstance(row, dict)
                and str(row.get("keyframe_path", "") or "")
            })
            keyframes = [[path, _stat(path)] for path in referenced]
        except Exception:                               # noqa: BLE001 — malformed is distinct state
            keyframes = [["<unreadable-shot-keyframes>", [0, 0]]]

        embeds_path_fn = getattr(proj, "embeds_path", None)
        embeds_path = embeds_path_fn(sid) if callable(embeds_path_fn) else \
            Path(proj.index_dir) / f"{sid}.embeds.npy"
        embeds_manifest = embeds_path.with_name(
            embeds_path.name.replace(".npy", "") + ".manifest.json")

        rows.append({
            "id": sid,
            "checksum": str(getattr(src, "checksum", "") or ""),
            "media": _stat(getattr(src, "local_path", "") or ""),
            "shots": _stat(shots_path),
            "words": _stat(Path(proj.index_dir) / f"{sid}.words.json"),
            "embeds": _stat(embeds_path),
            "embeds_manifest": _stat(embeds_manifest),
            "index_meta": _stat(Path(proj.index_dir) / f"{sid}.index.meta.json"),
            "keyframes": keyframes,
        })
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), len(rows)


def _gap_absence_pool_fingerprint(proj) -> tuple[str, int]:
    """Fingerprint every indexed OK source for a conservative whole-pool absence claim.

    Operator and automatic bans are deliberately ignored here.  An absence claim that covered a
    superset remains true when publication gates later shrink that set; a newly downloaded source
    or any media/index rewrite still changes this fingerprint and invalidates the review.
    """
    return _gap_pool_fingerprint(proj, include_banned=True)


_GAP_ABSENCE_POOL_SCOPE = "all_source_ok_indexed"


def _strict_original_project_view(proj, strict_segment, original: dict) -> tuple[object, object]:
    """Return a non-mutating project/segment view with one softened beat made strict again."""
    binding_proj = copy.copy(proj)
    strict_copy = copy.deepcopy(strict_segment)
    for field in _SOFTENING_SEGMENT_FIELDS:
        if field in original:
            setattr(strict_copy, field, copy.deepcopy(original[field]))
    target_idx = int(getattr(strict_copy, "index", -1))
    rebound_segments = []
    replaced = False
    for candidate in (getattr(proj, "segments", None) or []):
        if int(getattr(candidate, "index", -1)) == target_idx and not replaced:
            rebound_segments.append(strict_copy)
            replaced = True
        else:
            rebound_segments.append(candidate)
    if not replaced:
        rebound_segments.append(strict_copy)
    binding_proj.segments = rebound_segments
    return binding_proj, strict_copy


def _quote_retrieval_pool_binding(proj, cfg=None, *, strict_segment=None,
                                  original: dict | None = None) -> dict:
    """Identity/completeness of the separate authored-prompt retrieval universe.

    ``*.words.json`` no longer owns quote absence: the names-only general transcript and the
    authored-prompt retrieval stream are deliberately separate.  A gap review or active softening
    must therefore bind the latter's decoder/prompt generation *and* every eligible sidecar's exact
    bytes.  Merely adding those files after a review is not permission to reuse the old conclusion.
    """
    from . import index as _index_quote_binding
    binding_proj = proj
    if strict_segment is not None and isinstance(original, dict):
        # Quote softening itself clears ``seg.quote``.  Retrieval provenance must remain bound to the
        # authored promise that was reviewed, not to the deliberately weakened live segment.  A
        # shallow project view plus one copied segment avoids mutating the shared project while the
        # index helper recomputes prompt identity.
        binding_proj, _strict_copy = _strict_original_project_view(
            proj, strict_segment, original)
    if cfg is None:
        from .config import load_clip_config
        cfg = load_clip_config()
    try:
        generation = str(_index_quote_binding._quote_retrieval_fingerprint(
            binding_proj, cfg) or "")
    except Exception:
        generation = ""
    rows = []
    invalid = []
    for src in sorted((getattr(proj, "sources", None) or []),
                      key=lambda item: str(getattr(item, "id", "") or "")):
        if str(getattr(src, "status", "") or "") != SOURCE_OK:
            continue
        sid = str(getattr(src, "id", "") or "")
        try:
            shots = list(_index_quote_binding.load_shots(binding_proj, sid) or [])
            eligible = bool(_index_quote_binding.quote_retrieval_source_eligible(src, shots))
        except Exception:
            shots, eligible = [], True
        if not eligible:
            continue
        try:
            valid, _streams, reason, complete = \
                _index_quote_binding._load_quote_retrieval_streams_result(
                    binding_proj, src, cfg, require_complete=True)
            valid = bool(valid and complete)
        except Exception as exc:
            valid, reason = False, f"quote_retrieval_validation_error:{type(exc).__name__}"
        sidecar = Path(proj.index_dir) / f"{sid}.quote_retrieval.json"
        try:
            artifact_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        except Exception:
            artifact_sha256 = ""
        row = {
            "source_id": sid,
            "current": bool(valid),
            "artifact_sha256": artifact_sha256,
            "reason": "" if valid else str(reason or "quote_retrieval_invalid"),
        }
        rows.append(row)
        if not valid:
            invalid.append({"source_id": sid, "reason": row["reason"]})
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "generation_fingerprint": generation,
        "eligible_source_count": len(rows),
        "sidecar_pool_fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "complete": bool(generation) and not invalid,
        "invalid_sources": invalid,
    }


def _quote_review_branch_bindings(proj, segments, cfg=None) -> dict[str, dict]:
    """Bind quote type to its independent decoder, artifacts, and conclusive results."""
    quoted = [seg for seg in (segments or []) if str(getattr(seg, "quote", "") or "").strip()]
    if not quoted:
        return {}
    try:
        from . import relevance_contract as _rel_quote_binding
        contracts = _rel_quote_binding._quote_pool_branches(proj, quoted, cfg=cfg)
    except Exception:
        contracts = {}
    out = {}
    for seg in quoted:
        idx = int(getattr(seg, "index", -1))
        quote = " ".join(str(getattr(seg, "quote", "") or "").strip().split())
        contract = dict(contracts.get(idx) or {})
        raw_attempts = contract.get("unprompted_confirmation_attempts")
        attempts_shape_valid = isinstance(raw_attempts, list)
        result_identities = []
        artifact_keys = set()
        for raw in raw_attempts if attempts_shape_valid else []:
            if not isinstance(raw, dict):
                result_identities.append({"malformed": True})
                continue
            identity = {
                key: copy.deepcopy(raw.get(key))
                for key in (
                    "source_id", "retrieval_stream", "status", "reason", "artifact_key",
                    "decoder_fingerprint", "prompted_span", "confirmed_span",
                    "timed_asr_ratio", "match_method", "result_content_sha256")
            }
            result_identities.append(identity)
            artifact_key = str(raw.get("artifact_key", "") or "").strip().lower()
            if artifact_key:
                artifact_keys.add(artifact_key)
        result_raw = json.dumps(
            result_identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        def _count(field: str) -> int:
            value = contract.get(field)
            return value if isinstance(value, int) and not isinstance(value, bool) \
                and value >= 0 else -1

        out[str(idx)] = {
            "binding_schema_version": 1,
            "authored_quote_sha256": hashlib.sha256(
                quote.encode("utf-8", "replace")).hexdigest(),
            "branch": str(contract.get("branch", "") or "indeterminate"),
            "branch_reason": str(contract.get("branch_reason", "") or ""),
            "quote_retrieval_fingerprint_expected": str(contract.get(
                "quote_retrieval_fingerprint_expected", "") or ""),
            "confirmation_generation_fingerprint": str(contract.get(
                "confirmation_decoder_fingerprint_expected", "") or "").strip().lower(),
            "confirmation_attempts_shape_valid": attempts_shape_valid,
            "confirmation_attempt_count": _count("unprompted_confirmation_attempt_count"),
            "confirmation_confirmed_count": _count(
                "unprompted_confirmation_confirmed_count"),
            "confirmation_rejected_count": _count(
                "unprompted_confirmation_rejected_count"),
            "confirmation_inconclusive_count": _count(
                "unprompted_confirmation_inconclusive_count"),
            "retrieval_truncated_stream_count": _count(
                "retrieval_truncated_stream_count"),
            "confirmation_artifact_keys": sorted(artifact_keys),
            "confirmation_result_identities": result_identities,
            "confirmation_result_fingerprint": hashlib.sha256(
                result_raw.encode("utf-8", "replace")).hexdigest(),
        }
    return out


def refresh_selection_relevance_gap_review_quote_bindings(proj, *, cfg=None) -> dict:
    """Refresh a still-current ordinary gap review after unrelated quote-prompt churn.

    The authored retrieval sidecars are generation-bound to *every* project quote.  Consequently,
    removing an unrelated analyzer hallucination changes every sidecar generation and used to make
    a reviewed paraphrase gap permanently stale even when the reviewed beat, both footage pools,
    and every quote-classification result were byte-for-byte identical.

    This is deliberately a very narrow provenance migration, not a new gap classification.  It may
    replace only the global retrieval-generation fields, and only when all objective review facts
    and every per-beat paraphrase/result field still match.  Any real quote, indeterminate result,
    incomplete retrieval pool, changed confirmation result, beat drift, pool drift, or strict-
    exhaustion review remains stale and therefore release-blocking.
    """
    unchanged = {"refreshed": False, "reason": "no_refreshable_gap_review"}
    try:
        review = (getattr(proj, "meta", {}) or {}).get(
            "selection_relevance_gap_review") or {}
        if int(review.get("schema_version", 0) or 0) != 2:
            return {**unchanged, "reason": "gap_review_schema_not_refreshable"}
        if str(review.get("method", "") or "") != "actual_frame_and_pool_audit":
            return {**unchanged, "reason": "gap_review_method_not_refreshable"}
        # Strict-exhaustion rows bind a separate content-hashed evidence artifact.  Rewriting any
        # part of those reviews requires that artifact's own validator, never this ordinary-review
        # migration.
        if review.get("strict_acquisition_exhaustion"):
            return {**unchanged, "reason": "strict_exhaustion_review_not_refreshable"}

        raw_confirmed = review.get("confirmed_gap_beats")
        if not isinstance(raw_confirmed, (list, tuple, set)) or not raw_confirmed:
            return {**unchanged, "reason": "confirmed_gap_beats_missing"}
        confirmed = sorted({int(index) for index in raw_confirmed})
        by_seg = {
            int(getattr(seg, "index", -1)): seg
            for seg in (getattr(proj, "segments", None) or [])
        }
        bound_beats = review.get("beat_fingerprints")
        if not isinstance(bound_beats, dict) or any(
                index not in by_seg
                or str(bound_beats.get(str(index), "") or "")
                != _gap_beat_fingerprint(by_seg[index]) for index in confirmed):
            return {**unchanged, "reason": "gap_review_beat_changed"}

        pool_fp, pool_n = _gap_pool_fingerprint(proj)
        absence_fp, absence_n = _gap_absence_pool_fingerprint(proj)
        if (str(review.get("pool_fingerprint", "") or "") != pool_fp
                or review.get("pool_source_count") != pool_n):
            return {**unchanged, "reason": "gap_review_publishable_pool_changed"}
        if (str(review.get("absence_pool_scope", "") or "")
                != _GAP_ABSENCE_POOL_SCOPE
                or str(review.get("absence_pool_fingerprint", "") or "") != absence_fp
                or review.get("absence_pool_source_count") != absence_n):
            return {**unchanged, "reason": "gap_review_absence_pool_changed"}

        quoted = [
            by_seg[index] for index in confirmed
            if str(getattr(by_seg[index], "quote", "") or "").strip()
        ]
        if not quoted:
            return {**unchanged, "reason": "gap_review_has_no_quoted_beats"}

        stored_pool = review.get("quote_retrieval_binding")
        if not isinstance(stored_pool, dict) or stored_pool.get("complete") is not True:
            return {**unchanged, "reason": "stored_quote_retrieval_binding_incomplete"}
        current_pool = _quote_retrieval_pool_binding(proj, cfg)
        if not isinstance(current_pool, dict) or current_pool.get("complete") is not True:
            return {**unchanged, "reason": "current_quote_retrieval_binding_incomplete"}
        if stored_pool.get("invalid_sources") not in ([], ()) \
                or current_pool.get("invalid_sources") not in ([], ()):
            return {**unchanged, "reason": "quote_retrieval_binding_has_invalid_sources"}

        old_generation = str(stored_pool.get("generation_fingerprint", "") or "")
        new_generation = str(current_pool.get("generation_fingerprint", "") or "")
        if old_generation == new_generation:
            return {**unchanged, "reason": "quote_retrieval_generation_is_current"}
        if not re.fullmatch(r"[0-9a-f]{64}", old_generation) \
                or not re.fullmatch(r"[0-9a-f]{64}", new_generation):
            return {**unchanged, "reason": "quote_retrieval_generation_malformed"}
        if not re.fullmatch(
                r"[0-9a-f]{64}", str(stored_pool.get("sidecar_pool_fingerprint", "") or "")) \
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(current_pool.get("sidecar_pool_fingerprint", "") or "")):
            return {**unchanged, "reason": "quote_retrieval_sidecar_pool_fingerprint_malformed"}

        # Generation and the hash of the generation-stamped sidecars are the only global fields
        # allowed to move. Source eligibility/completeness and every future field must match.
        global_generation_fields = {"generation_fingerprint", "sidecar_pool_fingerprint"}
        stored_pool_facts = {
            key: value for key, value in stored_pool.items()
            if key not in global_generation_fields
        }
        current_pool_facts = {
            key: value for key, value in current_pool.items()
            if key not in global_generation_fields
        }
        if stored_pool_facts != current_pool_facts:
            return {**unchanged, "reason": "quote_retrieval_pool_facts_changed"}

        stored_branches = review.get("quote_branch_bindings")
        if not isinstance(stored_branches, dict):
            return {**unchanged, "reason": "stored_quote_branch_bindings_missing"}
        current_branches = _quote_review_branch_bindings(proj, quoted, cfg)
        retrieval_field = "quote_retrieval_fingerprint_expected"
        refreshed_indices = []
        expected_branch_keys = {str(int(getattr(seg, "index", -1))) for seg in quoted}
        if set(stored_branches) != expected_branch_keys \
                or set(current_branches) != expected_branch_keys:
            return {**unchanged, "reason": "quote_branch_binding_scope_changed"}
        for seg in quoted:
            index = int(getattr(seg, "index", -1))
            stored = stored_branches.get(str(index))
            current = current_branches.get(str(index))
            if not isinstance(stored, dict) or not isinstance(current, dict):
                return {**unchanged, "reason": "quote_branch_binding_missing"}
            quote = " ".join(str(getattr(seg, "quote", "") or "").strip().split())
            quote_hash = hashlib.sha256(quote.encode("utf-8", "replace")).hexdigest()
            if stored.get("authored_quote_sha256") != quote_hash \
                    or current.get("authored_quote_sha256") != quote_hash:
                return {**unchanged, "reason": "authored_quote_changed"}
            if stored.get("branch") != "paraphrase" or current.get("branch") != "paraphrase":
                return {**unchanged, "reason": "quote_branch_not_matching_paraphrase"}
            for binding in (stored, current):
                if binding.get("binding_schema_version") != 1:
                    return {**unchanged, "reason": "quote_branch_binding_schema_mismatch"}
                for field in (
                        "confirmation_generation_fingerprint",
                        "confirmation_result_fingerprint"):
                    if not re.fullmatch(
                            r"[0-9a-f]{64}", str(binding.get(field, "") or "")):
                        return {**unchanged, "reason": "quote_branch_fingerprint_malformed"}
                if binding.get("confirmation_attempts_shape_valid") is not True:
                    return {**unchanged, "reason": "quote_confirmation_shape_invalid"}
                for field in (
                        "confirmation_attempt_count", "confirmation_confirmed_count",
                        "confirmation_rejected_count", "confirmation_inconclusive_count",
                        "retrieval_truncated_stream_count"):
                    value = binding.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        return {**unchanged, "reason": "quote_branch_count_malformed"}
                if binding.get("confirmation_confirmed_count") != 0 \
                        or binding.get("confirmation_inconclusive_count") != 0 \
                        or binding.get("retrieval_truncated_stream_count") != 0:
                    return {**unchanged, "reason": "quote_branch_not_conclusive_paraphrase"}
            if stored.get(retrieval_field) != old_generation \
                    or current.get(retrieval_field) != new_generation:
                return {**unchanged, "reason": "quote_branch_generation_mismatch"}
            stored_facts = {
                key: value for key, value in stored.items() if key != retrieval_field
            }
            current_facts = {
                key: value for key, value in current.items() if key != retrieval_field
            }
            if stored_facts != current_facts:
                return {**unchanged, "reason": "quote_branch_results_changed"}
            refreshed_indices.append(index)

        updated = copy.deepcopy(review)
        updated["quote_retrieval_binding"] = copy.deepcopy(current_pool)
        updated["quote_branch_bindings"] = copy.deepcopy(current_branches)
        updated["quote_binding_refresh"] = {
            "schema_version": 1,
            "reason": "unrelated_global_quote_retrieval_generation_changed",
            "previous_generation_fingerprint": old_generation,
            "current_generation_fingerprint": new_generation,
            "reviewed_beats": sorted(refreshed_indices),
        }
        proj.meta["selection_relevance_gap_review"] = updated
        return {
            "refreshed": True,
            "reason": "unrelated_global_quote_retrieval_generation_changed",
            "previous_generation_fingerprint": old_generation,
            "current_generation_fingerprint": new_generation,
            "reviewed_beats": sorted(refreshed_indices),
        }
    except Exception as exc:                             # noqa: BLE001 — migration fails closed
        return {**unchanged, "reason": f"quote_binding_refresh_error:{type(exc).__name__}"}


def _strict_original_quote_branch_binding(proj, seg, original: dict, cfg=None) -> dict:
    """Recompute one active softening's quote contract against its authored strict state."""
    binding_proj, strict_seg = _strict_original_project_view(proj, seg, original)
    idx = int(getattr(strict_seg, "index", -1))
    return _quote_review_branch_bindings(binding_proj, [strict_seg], cfg).get(str(idx)) or {}


def _gap_review_quote_binding_reason(
        proj, seg, review: dict, cfg=None, *, allow_native_verbatim: bool = False) -> str:
    """Return why a quoted review is stale/incomplete; empty means current and conclusive."""
    quote = " ".join(str(getattr(seg, "quote", "") or "").strip().split())
    if not quote:
        return ""
    stored_pool = review.get("quote_retrieval_binding")
    if not isinstance(stored_pool, dict) or stored_pool.get("complete") is not True:
        return "gap_review_quote_retrieval_binding_missing_or_incomplete"
    current_pool = _quote_retrieval_pool_binding(proj, cfg)
    if current_pool.get("complete") is not True:
        return "current_quote_retrieval_pool_incomplete"
    for field in (
            "generation_fingerprint", "eligible_source_count", "sidecar_pool_fingerprint"):
        if stored_pool.get(field) != current_pool.get(field):
            return f"stale_gap_review_quote_retrieval_{field}"
    idx = int(getattr(seg, "index", -1))
    stored_branches = review.get("quote_branch_bindings")
    stored = stored_branches.get(str(idx)) if isinstance(stored_branches, dict) else None
    if not isinstance(stored, dict):
        return "gap_review_quote_branch_binding_missing"
    quote_hash = hashlib.sha256(quote.encode("utf-8", "replace")).hexdigest()
    if str(stored.get("authored_quote_sha256", "") or "") != quote_hash:
        return "stale_gap_review_authored_quote_changed"
    current = _quote_review_branch_bindings(proj, [seg], cfg).get(str(idx)) or {}
    current_branch = str(current.get("branch", "") or "")
    allowed = {"paraphrase", "verbatim"} if allow_native_verbatim else {"paraphrase"}
    if current_branch not in allowed:
        return "current_quote_branch_not_conclusive_paraphrase"
    if stored.get("branch") != current_branch:
        return "stale_gap_review_quote_branch_changed"
    if current.get("quote_retrieval_fingerprint_expected") != current_pool.get(
            "generation_fingerprint"):
        return "current_quote_branch_retrieval_generation_mismatch"
    if stored.get("quote_retrieval_fingerprint_expected") != current.get(
            "quote_retrieval_fingerprint_expected"):
        return "stale_gap_review_quote_branch_retrieval_generation"
    stored_confirmation_generation = str(stored.get(
        "confirmation_generation_fingerprint", "") or "").strip().lower()
    current_confirmation_generation = str(current.get(
        "confirmation_generation_fingerprint", "") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", stored_confirmation_generation):
        return "gap_review_quote_confirmation_generation_missing_or_malformed"
    if not re.fullmatch(r"[0-9a-f]{64}", current_confirmation_generation):
        return "current_quote_confirmation_generation_missing_or_malformed"
    if stored_confirmation_generation != current_confirmation_generation:
        return "stale_gap_review_quote_confirmation_generation"
    confirmation_identity_fields = (
        "confirmation_attempts_shape_valid", "confirmation_attempt_count",
        "confirmation_confirmed_count", "confirmation_rejected_count",
        "confirmation_inconclusive_count", "retrieval_truncated_stream_count",
        "confirmation_artifact_keys", "confirmation_result_identities",
        "confirmation_result_fingerprint",
    )
    if any(stored.get(field) != current.get(field)
           for field in confirmation_identity_fields):
        return "stale_gap_review_quote_confirmation_results"
    if stored != current:
        return "stale_gap_review_quote_branch_binding_changed"
    return ""


_SOFTENING_SCHEMA = 3
_SOFTENING_SEGMENT_FIELDS = (
    "visual_policy", "required_entity", "required_kind", "quote", "scene_query",
    "expected_visual", "is_specific_claim",
)
_SOFTENING_SELECTION_FIELDS = ("image_path", "image_meta")


def _capture_softening_state(seg, sel) -> dict:
    """The complete reversible state of every field either specificity rung may mutate."""
    out = {
        field: copy.deepcopy(getattr(seg, field, False if field == "is_specific_claim" else ""))
        for field in _SOFTENING_SEGMENT_FIELDS
    }
    out.update({
        "image_path": copy.deepcopy(getattr(sel, "image_path", "") or ""),
        "image_meta": copy.deepcopy(getattr(sel, "image_meta", {}) or {}),
    })
    return out


def _restore_softening_state(seg, sel, original: dict) -> None:
    required = set(_SOFTENING_SEGMENT_FIELDS + _SOFTENING_SELECTION_FIELDS)
    missing = sorted(required - set(original or {}))
    if missing:
        raise RuntimeError(
            "semantic softening marker lacks reversible original field(s): "
            + ", ".join(missing))
    for field in _SOFTENING_SEGMENT_FIELDS:
        setattr(seg, field, copy.deepcopy(original[field]))
    for field in _SOFTENING_SELECTION_FIELDS:
        setattr(sel, field, copy.deepcopy(original[field]))


def _softening_new_state(seg, sel) -> dict:
    out = _capture_softening_state(seg, sel)
    out["rung"] = str((getattr(sel, "image_meta", {}) or {}).get("selfheal_rung", "") or "")
    return out


def _softening_row(proj, seg, sel, original: dict, *, phase: str, basis: str,
                   pool_fingerprint: str | None = None,
                   pool_source_count: int | None = None,
                   trigger_reasons=None, quote_branch: str = "",
                   authorization=None, cfg=None) -> dict:
    if pool_fingerprint is None or pool_source_count is None:
        pool_fingerprint, pool_source_count = _gap_pool_fingerprint(proj)
    row = {
        "segment_index": int(getattr(seg, "index", -1)),
        "phase": str(phase or ""),
        "basis": str(basis or ""),
        "status": "softened",
        "active": True,
        "pool_fingerprint": str(pool_fingerprint or ""),
        "pool_source_count": int(pool_source_count or 0),
        "original": copy.deepcopy(original),
        "new": _softening_new_state(seg, sel),
        "trigger_reasons": list(trigger_reasons or []),
        "quote_branch": str(quote_branch or ""),
        "authorization": copy.deepcopy(authorization or {}),
    }
    row["dropped_requirement"] = bool(
        row["original"].get("required_entity") and not row["new"].get("required_entity"))
    row["surrendered_quote"] = bool(
        row["original"].get("quote") and not row["new"].get("quote"))
    if str(row["original"].get("quote", "") or "").strip():
        row["quote_retrieval_binding"] = _quote_retrieval_pool_binding(
            proj, cfg, strict_segment=seg, original=row["original"])
        row["quote_branch_binding"] = _strict_original_quote_branch_binding(
            proj, seg, row["original"], cfg)
        expected_branch = ("verbatim" if bool(
            (authorization or {}).get("surrender_verbatim_quote")) else "paraphrase")
        if row["quote_branch_binding"].get("branch") != expected_branch:
            raise RuntimeError(
                "semantic softening quote branch changed before persistence: "
                f"expected {expected_branch}, got "
                f"{row['quote_branch_binding'].get('branch') or 'missing'}")
        generation = str(row["quote_branch_binding"].get(
            "confirmation_generation_fingerprint", "") or "").strip().lower()
        result_fp = str(row["quote_branch_binding"].get(
            "confirmation_result_fingerprint", "") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", generation) \
                or not re.fullmatch(r"[0-9a-f]{64}", result_fp):
            raise RuntimeError(
                "semantic softening quote confirmation binding is missing or malformed")
        if not row["quote_branch"]:
            row["quote_branch"] = expected_branch
    return row


def _merge_softening_payload(proj, new_rows: list[dict], *, basis: str,
                             existing=None) -> dict:
    """Append reversible rows without losing earlier active phase-1/phase-2 softenings."""
    existing = copy.deepcopy(existing or {})
    if existing and int(existing.get("schema_version", 0) or 0) != _SOFTENING_SCHEMA:
        if existing.get("active", True):
            raise RuntimeError(
                "cannot merge an active legacy semantic-softening marker without full originals")
        existing = {}
    rows = list(existing.get("beats") or [])
    active_by_idx = {
        int(row.get("segment_index", -1)): pos
        for pos, row in enumerate(rows)
        if isinstance(row, dict) and row.get("status") == "softened"
        and row.get("active", True)
    }
    for row in new_rows:
        idx = int(row.get("segment_index", -1))
        if row.get("status") == "softened" and row.get("active", True) \
                and idx in active_by_idx:
            # Preserve the earliest authored promise.  Replacing it with a state that was already
            # softened would make a later pool-change restoration permanently lossy.
            old = rows[active_by_idx[idx]]
            if isinstance(old, dict) and isinstance(old.get("original"), dict):
                row = copy.deepcopy(row)
                row["original"] = copy.deepcopy(old["original"])
                for field in ("quote_retrieval_binding", "quote_branch_binding"):
                    if field in old:
                        row[field] = copy.deepcopy(old[field])
            rows[active_by_idx[idx]] = row
        else:
            rows.append(row)
    active = [
        row for row in rows if isinstance(row, dict)
        and row.get("status") == "softened" and row.get("active", True)
    ]
    pool_fp, pool_n = _gap_pool_fingerprint(proj)
    return {
        "schema_version": _SOFTENING_SCHEMA,
        "active": bool(active),
        "basis": str(basis or existing.get("basis", "") or ""),
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
        # This is the number attempted by the CURRENT invocation.  Counting every historical row
        # made a later two-beat pass report "2/3" merely because an old, inactive restoration was
        # still retained for audit.  Keep total history explicit instead of conflating the two.
        "candidate_count": len(new_rows),
        "history_count": len(rows),
        "candidates": sorted({int(r.get("segment_index", -1)) for r in active}),
        "softened_count": len(active),
        "still_blocked_count": sum(
            1 for r in rows if isinstance(r, dict) and r.get("status") == "still_blocked"),
        "restored_count": sum(
            1 for r in rows if isinstance(r, dict)
            and str(r.get("status", "")).startswith("restored_")),
        "beats": rows,
    }


def _persist_softening_payload(proj, payload: dict) -> None:
    """Atomically persist marker + audit, restoring both if the project save fails."""
    if getattr(proj, "meta", None) is None:
        proj.meta = {}
    old_present = "selection_relevance_gap_softening" in proj.meta
    old_meta = copy.deepcopy(proj.meta.get("selection_relevance_gap_softening"))
    dest = Path(proj.output_dir) / "semantic_gap_softening_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    old_file_present = dest.is_file()
    old_file = dest.read_bytes() if old_file_present else b""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(dest)
        proj.meta["selection_relevance_gap_softening"] = copy.deepcopy(payload)
        save = getattr(proj, "save", None)
        if callable(save):
            save()
    except Exception:
        if old_present:
            proj.meta["selection_relevance_gap_softening"] = old_meta
        else:
            proj.meta.pop("selection_relevance_gap_softening", None)
        try:
            if old_file_present:
                rollback = dest.with_suffix(dest.suffix + ".rollback.tmp")
                rollback.write_bytes(old_file)
                rollback.replace(dest)
            else:
                dest.unlink(missing_ok=True)
        except Exception:                                # noqa: BLE001 — original exception wins
            pass
        raise
    finally:
        tmp.unlink(missing_ok=True)


def _record_phase1_softening(proj, seg, sel, original: dict, *, basis: str,
                             pool_fingerprint: str, pool_source_count: int,
                             authorization=None, cfg=None) -> dict:
    row = _softening_row(
        proj, seg, sel, original, phase="phase1_preassembly", basis=basis,
        pool_fingerprint=pool_fingerprint, pool_source_count=pool_source_count,
        quote_branch=str((authorization or {}).get("quote_branch", "") or "")
        if isinstance(authorization, dict) else "",
        authorization=authorization, cfg=cfg)
    payload = _merge_softening_payload(
        proj, [row], basis=basis,
        existing=(getattr(proj, "meta", {}) or {}).get(
            "selection_relevance_gap_softening"))
    _persist_softening_payload(proj, payload)
    return row


def _phase1_softening_attempt(proj, seg, sel, used, attempt, *, basis: str,
                              marker_original: dict | None = None,
                              authorization=None, cfg=None) -> bool:
    """Run one authorized phase-1 policy mutation as an all-or-nothing transaction."""
    seg_before = copy.deepcopy(vars(seg))
    sel_before = copy.deepcopy(vars(sel))
    used_before = copy.deepcopy(used)
    original = copy.deepcopy(marker_original) if marker_original is not None \
        else _capture_softening_state(seg, sel)
    pool_fp, pool_n = _gap_pool_fingerprint(proj)

    def _rollback() -> None:
        vars(seg).clear()
        vars(seg).update(copy.deepcopy(seg_before))
        vars(sel).clear()
        vars(sel).update(copy.deepcopy(sel_before))
        if hasattr(used, "clear") and hasattr(used, "update"):
            used.clear()
            used.update(copy.deepcopy(used_before))

    try:
        ok = bool(attempt())
        if not ok:
            _rollback()
            return False
        current_fp, current_n = _gap_pool_fingerprint(proj)
        if (current_fp, current_n) != (pool_fp, pool_n):
            raise RuntimeError(
                "source/index pool changed while semantic specificity was being softened")
        _record_phase1_softening(
            proj, seg, sel, original, basis=basis,
            pool_fingerprint=pool_fp, pool_source_count=pool_n,
            authorization=authorization, cfg=cfg)
        return True
    except Exception:
        _rollback()
        raise


def restore_stale_selection_relevance_softenings(proj, segments, *, cfg=None, log=None) -> dict:
    """Restore active semantic softenings before publication when their pool binding is stale.

    ``evaluate_selection_relevance`` remains read-only.  Its asserting wrapper calls this helper
    first, so newly available footage revives the original exact promise and sends that beat through
    the strict gate/recovery machinery instead of leaving an old abstraction permanently accepted.
    """
    marker = copy.deepcopy((getattr(proj, "meta", {}) or {}).get(
        "selection_relevance_gap_softening") or {})
    if not marker or marker.get("active") is False:
        return {"restored": [], "unchanged": [], "pool_fingerprint": ""}
    if int(marker.get("schema_version", 0) or 0) != _SOFTENING_SCHEMA:
        raise RuntimeError(
            "active semantic-softening marker predates reversible pool-bound schema")
    rows = marker.get("beats")
    if not isinstance(rows, list):
        raise RuntimeError("active semantic-softening marker has no beat rows")
    current_fp, current_n = _gap_pool_fingerprint(proj)
    passed = {int(getattr(s, "index", -1)): s for s in (segments or [])}
    persisted = {
        int(getattr(s, "index", -1)): s
        for s in (getattr(proj, "segments", None) or [])
    }
    selections = {
        int(getattr(s, "segment_index", -1)): s
        for s in (getattr(proj, "selections", None) or [])
    }
    from .match import banned_source_ids
    banned = banned_source_ids(proj, include_auto=True)
    eligible_sources = {
        str(getattr(src, "id", "") or "")
        for src in (getattr(proj, "sources", None) or [])
        if str(getattr(src, "status", "") or "") == SOURCE_OK
    }

    def _softened_owner_stale_reason(row, idx: int) -> str:
        """Reject a rebind when its actually installed fallback owner is no longer eligible."""
        new_state = row.get("new") if isinstance(row.get("new"), dict) else {}
        saved_meta = (new_state.get("image_meta")
                      if isinstance(new_state.get("image_meta"), dict) else {})
        current_sel = selections.get(idx)
        current_meta = (getattr(current_sel, "image_meta", None) or {}) \
            if current_sel is not None else {}
        saved_sid = str(saved_meta.get("src", "") or "")
        current_sid = str(current_meta.get("src", "") or "")
        if not saved_sid or not current_sid:
            return "softened_still_owner_missing"
        if saved_sid != current_sid:
            return "softened_still_owner_changed"
        if saved_sid not in eligible_sources:
            return "softened_still_owner_not_source_ok"
        if saved_sid in banned:
            return "softened_still_owner_banned"
        return ""
    active_rows = [
        row for row in rows if isinstance(row, dict)
        and row.get("status") == "softened" and row.get("active", True)
    ]
    stale = []
    unchanged = []
    rebound = []
    stale_reasons = {}
    for row in active_rows:
        idx = int(row.get("segment_index", -1))
        bound = str(row.get("pool_fingerprint", "") or "")
        if not bound:
            raise RuntimeError(
                f"softened beat {row.get('segment_index')} lacks a pool fingerprint")
        reason = ""
        authorization = row.get("authorization")
        auth_kind = str((authorization or {}).get("authorization_kind", "") or "") \
            if isinstance(authorization, dict) else ""
        original = row.get("original")
        new_state = row.get("new") if isinstance(row.get("new"), dict) else {}
        cleared_quote = bool(
            isinstance(original, dict) and original.get("quote")
            and not new_state.get("quote"))
        native_surrender = bool(
            auth_kind == _NATIVE_HD_GAP_CLASSIFICATION
            or (isinstance(authorization, dict)
                and authorization.get("surrender_verbatim_quote") is True)
            or (cleared_quote and str(row.get("quote_branch", "") or "") == "verbatim"))
        evidence_bound = bool(
            isinstance(authorization, dict)
            and authorization.get("strict_acquisition_exhausted") is True)
        if isinstance(original, dict) and str(original.get("quote", "") or "").strip():
            stored_retrieval = row.get("quote_retrieval_binding")
            strict_base = persisted.get(idx) or passed.get(idx)
            current_retrieval = _quote_retrieval_pool_binding(
                proj, cfg, strict_segment=strict_base, original=original)
            if not isinstance(stored_retrieval, dict) \
                    or stored_retrieval.get("complete") is not True:
                reason = "quote_retrieval_binding_missing_or_incomplete"
            elif current_retrieval.get("complete") is not True:
                reason = "quote_retrieval_pool_no_longer_complete"
            else:
                for field in (
                        "generation_fingerprint", "eligible_source_count",
                        "sidecar_pool_fingerprint"):
                    if stored_retrieval.get(field) != current_retrieval.get(field):
                        reason = f"quote_retrieval_{field}_changed"
                        break
            if not reason and strict_base is None:
                reason = "quote_branch_strict_original_segment_missing"
            if not reason:
                stored_branch_binding = row.get("quote_branch_binding")
                current_branch_binding = _strict_original_quote_branch_binding(
                    proj, strict_base, original, cfg)
                expected_branch = "verbatim" if native_surrender else "paraphrase"
                if not isinstance(stored_branch_binding, dict):
                    reason = "quote_branch_binding_missing"
                elif str(stored_branch_binding.get("branch", "") or "") != expected_branch:
                    reason = f"stored_quote_branch_not_{expected_branch}"
                elif str(current_branch_binding.get("branch", "") or "") != expected_branch:
                    reason = f"current_quote_branch_not_{expected_branch}"
                elif stored_branch_binding != current_branch_binding:
                    reason = "quote_branch_binding_changed"
        if not reason and native_surrender and (
                auth_kind != _NATIVE_HD_GAP_CLASSIFICATION
                or authorization.get("surrender_verbatim_quote") is not True):
            reason = "native_gap_authorization_missing_or_malformed"
        elif not reason and (evidence_bound or native_surrender):
            base = persisted.get(idx) or passed.get(idx)
            if not isinstance(original, dict) or base is None:
                reason = "gap_evidence_original_state_missing"
            else:
                strict_seg = copy.deepcopy(base)
                try:
                    for field in _SOFTENING_SEGMENT_FIELDS:
                        setattr(strict_seg, field, copy.deepcopy(original[field]))
                except Exception:
                    reason = "gap_evidence_original_state_malformed"
                if not reason:
                    current_evidence, evidence_reason = \
                        _reviewed_exhaustion_evidence(proj, strict_seg)
                    if current_evidence is None:
                        reason = f"gap_evidence_stale:{evidence_reason}"
                    else:
                        for field in (
                                "authorization_kind", "evidence_source", "evidence_sha256",
                                "evidence_schema_version", "nonpublishable_exact_sources"):
                            if current_evidence.get(field) != authorization.get(field):
                                reason = f"gap_authorization_{field}_changed"
                                break
                        if (not reason and native_surrender
                                and current_evidence.get(
                                    "surrender_verbatim_quote") is not True):
                            reason = "native_gap_quote_surrender_no_longer_authorized"
        if not reason and bound != current_fp:
            if evidence_bound:
                # A matcher-only eligibility/ban change can alter the publishable pool while the
                # stronger content-hashed all-indexed exhaustion proof remains current.  Rebind the
                # active softening instead of reviving an exact promise that the complete pool is
                # still proven unable to satisfy.  New/index-changed sources already failed the
                # evidence validation above and therefore restore normally.
                reason = _softened_owner_stale_reason(row, idx)
                if not reason:
                    row["rebound_from_pool_fingerprint"] = bound
                    row["rebound_from_pool_source_count"] = row.get("pool_source_count")
                    row["pool_fingerprint"] = current_fp
                    row["pool_source_count"] = current_n
                    row["rebound_reason"] = "current_all_indexed_strict_exhaustion_evidence"
                    rebound.append(row)
                    if callable(log):
                        log(f"semantic-gap: beat {idx} rebound to the current publishable pool; "
                            "its all-indexed strict-exhaustion evidence is still current")
            else:
                reason = "publishable_pool_changed"
        if reason:
            stale.append(row)
            stale_reasons[idx] = reason
        else:
            unchanged.append(row)
    if not stale:
        if rebound:
            marker["pool_fingerprint"] = current_fp
            marker["pool_source_count"] = current_n
            marker["last_checked_pool_fingerprint"] = current_fp
            marker["last_checked_pool_source_count"] = current_n
            marker["rebound_count"] = int(marker.get("rebound_count", 0) or 0) + len(rebound)
            _persist_softening_payload(proj, marker)
        return {
            "restored": [],
            "unchanged": sorted(int(r.get("segment_index", -1)) for r in unchanged),
            "rebound": sorted(int(r.get("segment_index", -1)) for r in rebound),
            "pool_fingerprint": current_fp,
        }

    snapshots = []
    restored = []
    try:
        for row in stale:
            idx = int(row.get("segment_index", -1))
            sel = selections.get(idx)
            seg_targets = []
            for candidate in (persisted.get(idx), passed.get(idx)):
                if candidate is not None and all(candidate is not old for old in seg_targets):
                    seg_targets.append(candidate)
            if not seg_targets or sel is None:
                raise RuntimeError(
                    f"cannot restore softened beat {idx}: segment or selection is absent")
            original = row.get("original")
            if not isinstance(original, dict):
                raise RuntimeError(f"softened beat {idx} lacks its original state")
            for target in seg_targets:
                snapshots.append((target, copy.deepcopy(vars(target))))
            snapshots.append((sel, copy.deepcopy(vars(sel))))
            # Validate and restore the shared selection once, then mirror the original segment
            # fields to both the persisted project row and the caller's row when they differ.
            _restore_softening_state(seg_targets[0], sel, original)
            for target in seg_targets[1:]:
                for field in _SOFTENING_SEGMENT_FIELDS:
                    setattr(target, field, copy.deepcopy(original[field]))
            row["active"] = False
            restore_reason = stale_reasons.get(idx, "publishable_pool_changed")
            row["status"] = ("restored_pool_changed"
                             if restore_reason == "publishable_pool_changed"
                             else "restored_evidence_stale")
            row["restore_reason"] = restore_reason
            row["restored_against_pool_fingerprint"] = current_fp
            row["restored_against_pool_source_count"] = current_n
            restored.append(idx)
            if callable(log):
                log(f"semantic-gap: beat {idx} restored to its original authored promise "
                    f"because its softening authorization is stale ({restore_reason})")
        marker["active"] = any(
            isinstance(r, dict) and r.get("status") == "softened" and r.get("active", True)
            for r in rows)
        marker["softened_count"] = sum(
            1 for r in rows if isinstance(r, dict)
            and r.get("status") == "softened" and r.get("active", True))
        marker["restored_count"] = sum(
            1 for r in rows if isinstance(r, dict)
            and str(r.get("status", "")).startswith("restored_"))
        marker["last_checked_pool_fingerprint"] = current_fp
        marker["last_checked_pool_source_count"] = current_n
        if rebound:
            marker["pool_fingerprint"] = current_fp
            marker["pool_source_count"] = current_n
            marker["rebound_count"] = int(marker.get("rebound_count", 0) or 0) + len(rebound)
        _persist_softening_payload(proj, marker)
    except Exception:
        for obj, state in reversed(snapshots):
            vars(obj).clear()
            vars(obj).update(copy.deepcopy(state))
        raise
    return {
        "restored": sorted(restored),
        "unchanged": sorted(int(r.get("segment_index", -1)) for r in unchanged),
        "rebound": sorted(int(r.get("segment_index", -1)) for r in rebound),
        "pool_fingerprint": current_fp,
    }


def _phase1_softening_authorization(proj, seg, cfg=None, *,
                                    require_eligible_pool: bool = True,
                                    allow_verified_native_gap: bool = False) -> tuple[bool, str]:
    """Fail-closed authorization for phase-1 exact→character→abstract policy mutation.

    The legacy pre-assembly self-heal sees a deliberately incomplete structural blocker set.  It
    may search, acquire, and install strict evidence for any beat, but it may surrender specificity
    only when an actual-frame/pool review names this exact authored promise and is still bound to the
    complete current source/index pool.  Acquisition can change that pool inside the same healing
    pass, so callers check this immediately before every mutation rather than caching the answer.
    The sole ``require_eligible_pool=False`` caller immediately validates a schema-2 or schema-4,
    content-hashed exhaustion artifact over *all* indexed OK sources; it may ignore a matcher-only
    pool shrink, but never a new source or changed media/index artifact.  Schema 4 can additionally
    authorize one proven native-HD absence without changing the ordinary quote contract.
    """
    try:
        review = (getattr(proj, "meta", {}) or {}).get(
            "selection_relevance_gap_review") or {}
        if not review:
            return False, "selection_relevance_gap_review_missing"
        schema = int(review.get("schema_version", 0) or 0)
        if schema != 2:
            return False, f"selection_relevance_gap_review_schema_{schema}_not_2"
        raw = review.get("confirmed_gap_beats")
        if not isinstance(raw, (list, tuple, set)):
            return False, "confirmed_gap_beats_missing_or_malformed"
        confirmed = {int(i) for i in raw}
        idx = int(getattr(seg, "index", -1))
        if idx not in confirmed:
            return False, "beat_not_confirmed_as_footage_gap"
        bound = review.get("beat_fingerprints") or {}
        expected_beat = str(bound.get(str(idx), "") or "")
        current_beat = _gap_beat_fingerprint(seg)
        if not expected_beat:
            return False, "confirmed_beat_fingerprint_missing"
        if expected_beat != current_beat:
            return False, "stale_gap_review_beat_changed"
        quote_binding_reason = _gap_review_quote_binding_reason(
            proj, seg, review, cfg,
            allow_native_verbatim=bool(allow_verified_native_gap))
        if quote_binding_reason:
            return False, quote_binding_reason
        if require_eligible_pool:
            expected_pool = str(review.get("pool_fingerprint", "") or "")
            if not expected_pool:
                return False, "gap_review_pool_fingerprint_missing"
            current_pool, _current_count = _gap_pool_fingerprint(proj)
            if expected_pool != current_pool:
                return False, ("stale_gap_review_source_pool_changed:"
                               f"{expected_pool[:12]}!={current_pool[:12]}")
        evidence_ok, evidence_reason = _phase1_current_gap_evidence(
            proj, seg, cfg, allow_verified_native_gap=allow_verified_native_gap)
        if not evidence_ok:
            return False, evidence_reason
        return True, "authorized_by_bound_gap_review_and_current_semantic_gap"
    except Exception as exc:                              # noqa: BLE001 — mutation must fail closed
        return False, f"gap_review_validation_error:{type(exc).__name__}"


_STRICT_ACQUISITION_EVIDENCE_SCHEMA = 2
_NATIVE_HD_GAP_EVIDENCE_SCHEMA = 4
_NATIVE_HD_GAP_CLASSIFICATION = "publishable_native_hd_absence"
_TOTAL_ABSENCE_GAP_SCOPE = "all_indexed_absence"
_STRICT_ACQUISITION_CLAIMS = (
    ("classification", "footage_gap", "strict_acquisition_classification_not_footage_gap"),
    ("actual_frame_pool_audit", True,
     "strict_acquisition_actual_frame_pool_audit_absent"),
    ("whole_pool_reviewed", True, "strict_acquisition_whole_pool_review_absent"),
    ("correct_footage_present_in_pool", False,
     "strict_acquisition_pool_absence_not_proven"),
    ("pipeline_bug_ruled_out", True, "strict_acquisition_pipeline_bug_not_ruled_out"),
    ("strict_acquisition_status", "exhausted",
     "strict_acquisition_status_not_exhausted"),
    ("technical_status", "complete", "strict_acquisition_technical_status_not_complete"),
)

_NATIVE_HD_GAP_CLAIMS = (
    ("classification", _NATIVE_HD_GAP_CLASSIFICATION,
     "strict_acquisition_classification_not_publishable_native_hd_absence"),
    ("gap_scope", _NATIVE_HD_GAP_CLASSIFICATION,
     "strict_acquisition_gap_scope_not_publishable_native_hd_absence"),
    ("actual_frame_pool_audit", True,
     "strict_acquisition_actual_frame_pool_audit_absent"),
    ("whole_pool_reviewed", True, "strict_acquisition_whole_pool_review_absent"),
    ("correct_footage_present_in_pool", True,
     "strict_acquisition_native_only_footage_presence_not_proven"),
    ("correct_publishable_footage_present_in_pool", False,
     "strict_acquisition_publishable_pool_absence_not_proven"),
    ("pipeline_bug_ruled_out", True, "strict_acquisition_pipeline_bug_not_ruled_out"),
    ("strict_acquisition_status", "exhausted",
     "strict_acquisition_status_not_exhausted"),
    ("technical_status", "complete", "strict_acquisition_technical_status_not_complete"),
)

_SCHEMA3_TOTAL_ABSENCE_CLAIMS = (
    *_STRICT_ACQUISITION_CLAIMS,
    ("gap_scope", _TOTAL_ABSENCE_GAP_SCOPE,
     "strict_acquisition_gap_scope_not_all_indexed_absence"),
)


def _strict_claim_failure(row: dict, claims) -> str:
    for field, expected, failure in claims:
        value = row.get(field)
        valid = value is expected if isinstance(expected, bool) \
            else str(value or "") == expected
        if not valid:
            return failure
    return ""


def _validated_native_hd_gap_row(proj, seg, row: dict) -> tuple[dict | None, str]:
    """Re-prove a schema-4 SD-only exact-scene claim from current bytes and ASR.

    The independent artifact owns the actual-frame/whole-pool classification.  Runtime still owns
    every objective premise that can drift: publication floors, source bytes/dimensions, the
    matcher rejection that keeps those pixels out of every visual pool, and the complete-pool
    quote locations.  A schema-4 row is deliberately useful only for a located real quote; ordinary
    quoted gaps retain the global verbatim denial and schema 2 keeps its total-absence meaning.
    """
    failure = _strict_claim_failure(row, _NATIVE_HD_GAP_CLAIMS)
    if failure:
        return None, failure
    try:
        from .quality_contract import (
            MIN_NATIVE_LONG_EDGE, MIN_NATIVE_SHORT_EDGE,
            native_video_ok, probe_native_video_info,
        )
    except Exception as exc:                             # noqa: BLE001 — quality proof fails closed
        return None, f"strict_acquisition_native_probe_unavailable:{type(exc).__name__}"

    if row.get("publication_min_short_edge") != MIN_NATIVE_SHORT_EDGE \
            or row.get("publication_min_long_edge") != MIN_NATIVE_LONG_EDGE:
        return None, "strict_acquisition_native_publication_floor_mismatch"
    raw_sources = row.get("nonpublishable_exact_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return None, "strict_acquisition_nonpublishable_exact_sources_missing"

    auto_reasons = (getattr(proj, "meta", {}) or {}).get("auto_rejected_reasons") or {}
    auto_rejected = {
        str(source_id) for source_id in
        ((getattr(proj, "meta", {}) or {}).get("auto_rejected_sources") or [])
    }
    normalized_sources = []
    by_source = {}
    for source_row in raw_sources:
        if not isinstance(source_row, dict):
            return None, "strict_acquisition_nonpublishable_exact_source_malformed"
        sid = str(source_row.get("source_id", "") or "").strip()
        if not sid or sid in by_source:
            return None, "strict_acquisition_nonpublishable_exact_source_id_malformed"
        src = proj.source(sid)
        if src is None or str(getattr(src, "status", "") or "") != SOURCE_OK:
            return None, "strict_acquisition_nonpublishable_exact_source_unavailable"
        checksum = str(getattr(src, "checksum", "") or "").strip()
        if not checksum or str(source_row.get("source_checksum", "") or "") != checksum:
            return None, "strict_acquisition_nonpublishable_exact_source_checksum_mismatch"
        # The reason map is audit prose; the rejected-ID set is what actually excludes a source
        # from matcher/self-heal pools.  Require both or a stale reason could authorize quote
        # surrender while the same 640×360 bytes are re-admitted and aired as an image still.
        if sid not in auto_rejected:
            return None, "strict_acquisition_nonpublishable_exact_source_not_currently_rejected"
        if str(auto_reasons.get(sid, "") or "") != "sub_native_hd" \
                or str(source_row.get("auto_reject_reason", "") or "") != "sub_native_hd":
            return None, "strict_acquisition_nonpublishable_exact_source_not_rejected_as_sd"
        dims = dict(probe_native_video_info(getattr(src, "local_path", "") or "") or {})
        try:
            width, height = int(dims.get("width") or 0), int(dims.get("height") or 0)
        except (TypeError, ValueError, OverflowError):
            width = height = 0
        if not width or not height:
            return None, "strict_acquisition_nonpublishable_exact_source_unprobeable"
        if native_video_ok(dims):
            return None, "strict_acquisition_nonpublishable_exact_source_is_native_hd"
        if source_row.get("native_width") != width or source_row.get("native_height") != height:
            return None, "strict_acquisition_nonpublishable_exact_source_dimensions_changed"
        if source_row.get("actual_frame_target_verified") is not True:
            return None, "strict_acquisition_nonpublishable_exact_frame_not_verified"
        span = source_row.get("timed_asr_span")
        ratio = source_row.get("timed_asr_ratio")
        if not isinstance(span, list) or len(span) != 2 \
                or isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            return None, "strict_acquisition_nonpublishable_exact_quote_span_malformed"
        try:
            q0, q1, qratio = float(span[0]), float(span[1]), float(ratio)
        except (TypeError, ValueError, OverflowError):
            return None, "strict_acquisition_nonpublishable_exact_quote_span_malformed"
        if not (q1 > q0 >= 0.0) or not (0.0 <= qratio <= 1.0):
            return None, "strict_acquisition_nonpublishable_exact_quote_span_malformed"
        confirmation_key = str(source_row.get(
            "quote_confirmation_artifact_key", "") or "").strip().lower()
        confirmation_decoder = str(source_row.get(
            "quote_confirmation_decoder_fingerprint", "") or "").strip().lower()
        if len(confirmation_key) != 64 \
                or any(c not in "0123456789abcdef" for c in confirmation_key):
            return None, "strict_acquisition_quote_confirmation_artifact_key_malformed"
        if len(confirmation_decoder) != 64 \
                or any(c not in "0123456789abcdef" for c in confirmation_decoder):
            return None, "strict_acquisition_quote_confirmation_decoder_malformed"
        normalized = {
            "source_id": sid,
            "source_checksum": checksum,
            "auto_reject_reason": "sub_native_hd",
            "native_width": width,
            "native_height": height,
            "actual_frame_target_verified": True,
            "timed_asr_span": [q0, q1],
            "timed_asr_ratio": qratio,
            "quote_confirmation_artifact_key": confirmation_key,
            "quote_confirmation_decoder_fingerprint": confirmation_decoder,
        }
        by_source[sid] = normalized
        normalized_sources.append(normalized)

    quote = str(getattr(seg, "quote", "") or "").strip()
    if not quote:
        return None, "strict_acquisition_native_gap_verbatim_quote_absent"
    try:
        from . import relevance_contract as _rel_native
        contract = dict(_rel_native._quote_pool_branches(proj, [seg]).get(
            int(getattr(seg, "index", -1))) or {})
    except Exception as exc:                             # noqa: BLE001 — ASR proof fails closed
        return None, f"strict_acquisition_native_gap_quote_scan_error:{type(exc).__name__}"
    if str(contract.get("branch", "") or "") != "verbatim":
        return None, "strict_acquisition_native_gap_quote_not_currently_verbatim"
    expected_asr_fingerprint = str(
        contract.get("asr_prompt_fingerprint_expected", "") or "").strip()
    if not expected_asr_fingerprint:
        return None, "strict_acquisition_native_gap_asr_fingerprint_missing"
    expected_confirmation_decoder = str(contract.get(
        "confirmation_decoder_fingerprint_expected", "") or "").strip().lower()
    if len(expected_confirmation_decoder) != 64 \
            or any(c not in "0123456789abcdef" for c in expected_confirmation_decoder):
        return None, "strict_acquisition_native_gap_confirmation_decoder_missing"
    try:
        invalid_asr_count = int(contract.get("asr_provenance_invalid_source_count", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return None, "strict_acquisition_native_gap_asr_provenance_count_malformed"
    if invalid_asr_count or contract.get("asr_provenance_invalid_sources"):
        # A located SD quote proves presence, but quote surrender additionally asserts absence of
        # any publishable-HD match.  One ASR-unknown source makes that absence unknowable, so this
        # exceptional path must stop even though the ordinary quote branch remains verbatim.
        return None, "strict_acquisition_native_gap_asr_pool_incomplete"
    for field, malformed_reason, nonzero_reason in (
            ("retrieval_truncated_stream_count",
             "strict_acquisition_native_gap_retrieval_truncation_count_malformed",
             "strict_acquisition_native_gap_retrieval_scan_truncated"),
            ("unprompted_confirmation_inconclusive_count",
             "strict_acquisition_native_gap_confirmation_inconclusive_count_malformed",
             "strict_acquisition_native_gap_confirmation_inconclusive")):
        value = contract.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None, malformed_reason
        if value != 0:
            # One confirmed native-SD occurrence types the quote as real, but it cannot prove that
            # every remaining candidate failed publication. A truncated candidate stream or an
            # unjudged no-prompt decode leaves a possible HD occurrence unresolved, so specificity
            # surrender must remain blocked.
            return None, nonzero_reason
    matches = list(contract.get("pool_matches") or [])
    if not matches and isinstance(contract.get("pool_match"), dict):
        matches = [contract["pool_match"]]
    if not matches:
        return None, "strict_acquisition_native_gap_quote_match_absent"
    current_ids = set()
    for match in matches:
        if not isinstance(match, dict):
            return None, "strict_acquisition_native_gap_quote_match_malformed"
        sid = str(match.get("source_id", "") or "")
        current_ids.add(sid)
        source_row = by_source.get(sid)
        if source_row is None:
            return None, "strict_acquisition_native_gap_quote_match_unlisted"
        match_span = match.get("timed_asr_span")
        try:
            m0, m1 = float(match_span[0]), float(match_span[1])
            mratio = float(match.get("timed_asr_ratio"))
        except (TypeError, ValueError, IndexError, OverflowError):
            return None, "strict_acquisition_native_gap_quote_match_malformed"
        recorded = source_row["timed_asr_span"]
        if abs(recorded[0] - m0) > 0.01 or abs(recorded[1] - m1) > 0.01 \
                or abs(source_row["timed_asr_ratio"] - mratio) > 0.001:
            return None, "strict_acquisition_native_gap_quote_match_changed"
        confirmation = match.get("unprompted_confirmation")
        if not isinstance(confirmation, dict) \
                or str(confirmation.get("status", "") or "") != "confirmed":
            return None, "strict_acquisition_native_gap_quote_confirmation_absent"
        current_key = str(confirmation.get("artifact_key", "") or "").strip().lower()
        current_decoder = str(
            confirmation.get("decoder_fingerprint", "") or "").strip().lower()
        if current_decoder != expected_confirmation_decoder:
            return None, "strict_acquisition_native_gap_quote_confirmation_decoder_changed"
        if source_row["quote_confirmation_artifact_key"] != current_key:
            return None, "strict_acquisition_native_gap_quote_confirmation_artifact_changed"
        if source_row["quote_confirmation_decoder_fingerprint"] != current_decoder:
            return None, "strict_acquisition_native_gap_quote_confirmation_decoder_changed"
        confirmed_span = confirmation.get("confirmed_span")
        try:
            c0, c1, cratio = (float(confirmed_span[0]), float(confirmed_span[1]),
                              float(confirmed_span[2]))
        except (IndexError, TypeError, ValueError, OverflowError):
            return None, "strict_acquisition_native_gap_quote_confirmation_span_malformed"
        if abs(c0 - m0) > 0.001 or abs(c1 - m1) > 0.001 \
                or abs(cratio - mratio) > 0.001:
            return None, "strict_acquisition_native_gap_quote_confirmation_span_changed"
    if current_ids != set(by_source):
        # Do not let an artifact pad its proof with a visually asserted source that the current
        # complete-pool quote scan cannot reproduce.  Exact equality keeps this exception narrow.
        return None, "strict_acquisition_native_gap_source_set_changed"

    return {
        **{field: row[field] for field, _expected, _failure in _NATIVE_HD_GAP_CLAIMS},
        "publication_min_short_edge": MIN_NATIVE_SHORT_EDGE,
        "publication_min_long_edge": MIN_NATIVE_LONG_EDGE,
        "nonpublishable_exact_sources": sorted(
            normalized_sources, key=lambda item: item["source_id"]),
        "quote_branch": "verbatim",
        "surrender_verbatim_quote": True,
        "authorization_kind": _NATIVE_HD_GAP_CLASSIFICATION,
        "strict_acquisition_exhausted": True,
    }, "strict_acquisition_native_hd_gap_valid"


def _validated_schema4_gap_row(proj, seg, row: dict) -> tuple[dict | None, str]:
    """Validate either row profile allowed inside one mixed schema-4 evidence artifact."""
    classification = str(row.get("classification", "") or "")
    if classification == "footage_gap":
        failure = _strict_claim_failure(row, _SCHEMA3_TOTAL_ABSENCE_CLAIMS)
        if failure:
            return None, failure
        return {
            **{field: row[field] for field, _expected, _failure
               in _SCHEMA3_TOTAL_ABSENCE_CLAIMS},
            "authorization_kind": "total_absence",
            "surrender_verbatim_quote": False,
            "strict_acquisition_exhausted": True,
            "nonpublishable_exact_sources": [],
        }, "strict_acquisition_total_absence_valid"
    if classification == _NATIVE_HD_GAP_CLASSIFICATION:
        return _validated_native_hd_gap_row(proj, seg, row)
    return None, "strict_acquisition_schema4_gap_classification_unknown"


def _strict_acquisition_evidence(proj, segments, beat_indices, *, source: str,
                                 expected_sha256: str = "", stored_records=None) \
        -> tuple[dict | None, str]:
    """Validate the authoritative independent exhaustion artifact against the current world.

    The project review is a cached authorization, not the evidence itself.  Therefore neither the
    maker nor a hand-edited ``project.json`` may manufacture claims from a beat list.  Both creation
    and consumption come through this function, which reads the content-hashed JSON artifact and
    binds every requested beat plus the complete all-``SOURCE_OK`` indexed universe before returning
    normalized rows.  That conservative superset is stable across matcher-only bans while additions
    and artifact rewrites still invalidate it.
    """
    evidence_source = str(source or "").strip()
    if not evidence_source:
        return None, "strict_acquisition_evidence_source_missing"
    evidence_path = Path(evidence_source).expanduser()
    if not evidence_path.is_absolute():
        evidence_path = Path(getattr(proj, "root", "") or ".") / evidence_path
    try:
        evidence_path = evidence_path.resolve()
    except Exception:                                    # noqa: BLE001 — read below fails closed
        pass
    if not evidence_path.is_file():
        return None, "strict_acquisition_evidence_artifact_missing"

    expected_sha = str(expected_sha256 or "").strip().lower()
    if stored_records is not None and not expected_sha:
        return None, "strict_acquisition_evidence_sha256_missing_or_malformed"
    if expected_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return None, "strict_acquisition_evidence_sha256_missing_or_malformed"
    try:
        raw = evidence_path.read_bytes()
    except Exception as exc:                              # noqa: BLE001 — evidence must be readable
        return None, f"strict_acquisition_evidence_artifact_unreadable:{type(exc).__name__}"
    actual_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        return None, "strict_acquisition_evidence_artifact_hash_mismatch"
    try:
        artifact = json.loads(raw.decode("utf-8"))
    except Exception as exc:                              # noqa: BLE001 — malformed is not evidence
        return None, f"strict_acquisition_evidence_json_invalid:{type(exc).__name__}"
    if not isinstance(artifact, dict):
        return None, "strict_acquisition_evidence_json_not_object"
    schema = artifact.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) \
            or schema not in (_STRICT_ACQUISITION_EVIDENCE_SCHEMA,
                              _NATIVE_HD_GAP_EVIDENCE_SCHEMA):
        return None, "strict_acquisition_evidence_schema_not_2_or_4"
    if str(artifact.get("status", "") or "") != "complete":
        return None, "strict_acquisition_evidence_status_not_complete"
    if str(artifact.get("pool_scope", "") or "") != _GAP_ABSENCE_POOL_SCOPE:
        return None, "strict_acquisition_evidence_pool_scope_not_all_source_ok_indexed"

    current_pool, current_count = _gap_absence_pool_fingerprint(proj)
    artifact_pool = str(artifact.get("pool_fingerprint", "") or "")
    if not artifact_pool:
        return None, "strict_acquisition_evidence_pool_fingerprint_missing"
    if artifact_pool != current_pool:
        return None, "strict_acquisition_evidence_source_pool_changed"
    artifact_count = artifact.get("pool_source_count")
    if isinstance(artifact_count, bool) or not isinstance(artifact_count, int):
        return None, "strict_acquisition_evidence_pool_source_count_malformed"
    if artifact_count != current_count:
        return None, "strict_acquisition_evidence_pool_source_count_changed"

    artifact_beats = artifact.get("beats")
    if not isinstance(artifact_beats, dict):
        return None, "strict_acquisition_evidence_beats_missing_or_malformed"
    by_idx = {int(getattr(s, "index", -1)): s for s in (segments or [])}
    normalized = {}
    for raw_idx in (beat_indices or []):
        if isinstance(raw_idx, bool):
            return None, "strict_acquisition_evidence_beat_index_malformed"
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            return None, "strict_acquisition_evidence_beat_index_malformed"
        seg = by_idx.get(idx)
        if seg is None:
            return None, "strict_acquisition_evidence_beat_absent"
        row = artifact_beats.get(str(idx))
        if not isinstance(row, dict):
            return None, "strict_acquisition_evidence_beat_record_missing"
        current_beat = _gap_beat_fingerprint(seg)
        if str(row.get("beat_fingerprint", "") or "") != current_beat:
            return None, "strict_acquisition_evidence_beat_fingerprint_mismatch"
        if schema == _STRICT_ACQUISITION_EVIDENCE_SCHEMA:
            failure = _strict_claim_failure(row, _STRICT_ACQUISITION_CLAIMS)
            if failure:
                return None, failure
            normalized_row = {
                "beat_fingerprint": current_beat,
                **{field: row[field] for field, _expected, _failure
                   in _STRICT_ACQUISITION_CLAIMS},
                "authorization_kind": "total_absence",
                "surrender_verbatim_quote": False,
                "strict_acquisition_exhausted": True,
                "nonpublishable_exact_sources": [],
            }
        else:
            schema4_row, schema4_reason = _validated_schema4_gap_row(proj, seg, row)
            if schema4_row is None:
                return None, schema4_reason
            normalized_row = {"beat_fingerprint": current_beat, **schema4_row}

        # The artifact is authoritative, but retain its normalized claims in project.json so audit
        # output remains self-describing.  A hand edit must agree byte-for-byte with the artifact;
        # it cannot turn an incomplete/pipeline-bug record into an early authorization.
        stored = (stored_records or {}).get(str(idx)) if stored_records is not None else None
        if stored_records is not None and not isinstance(stored, dict):
            return None, "beat_strict_acquisition_exhaustion_missing"
        if stored is not None:
            if str(stored.get("beat_fingerprint", "") or "") != current_beat:
                return None, "strict_acquisition_evidence_beat_fingerprint_mismatch"
            if schema == _STRICT_ACQUISITION_EVIDENCE_SCHEMA:
                claims = _STRICT_ACQUISITION_CLAIMS
            elif str(row.get("classification", "") or "") == "footage_gap":
                claims = _SCHEMA3_TOTAL_ABSENCE_CLAIMS
            else:
                claims = _NATIVE_HD_GAP_CLAIMS
            failure = _strict_claim_failure(stored, claims)
            if failure:
                return None, failure
            for field, value in normalized_row.items():
                if stored.get(field) != value:
                    return None, f"strict_acquisition_evidence_stored_{field}_mismatch"
        normalized[str(idx)] = normalized_row

    return {
        "evidence_source": str(evidence_path),
        "evidence_sha256": actual_sha,
        "pool_scope": _GAP_ABSENCE_POOL_SCOPE,
        "pool_fingerprint": current_pool,
        "pool_source_count": current_count,
        "evidence_schema_version": schema,
        "beats": normalized,
    }, "strict_acquisition_evidence_valid"


def _reviewed_exhaustion_evidence(proj, seg) -> tuple[dict | None, str]:
    """Return one current content-hashed exhaustion record, without semantic mutation authority."""
    try:
        review = (getattr(proj, "meta", {}) or {}).get(
            "selection_relevance_gap_review") or {}
        records = review.get("strict_acquisition_exhaustion")
        if not isinstance(records, dict):
            return None, "strict_acquisition_exhaustion_missing_or_malformed"
        idx = int(getattr(seg, "index", -1))
        record = records.get(str(idx))
        if not isinstance(record, dict):
            return None, "beat_strict_acquisition_exhaustion_missing"
        validated, evidence_reason = _strict_acquisition_evidence(
            proj, [seg], [idx], source=record.get("evidence_source", ""),
            expected_sha256=record.get("evidence_sha256", ""), stored_records=records)
        if validated is None:
            return None, evidence_reason
        if str(review.get("absence_pool_scope", "") or "") != \
                str(validated.get("pool_scope", "") or ""):
            return None, "strict_acquisition_review_pool_scope_mismatch"
        if str(review.get("absence_pool_fingerprint", "") or "") != \
                str(validated.get("pool_fingerprint", "") or ""):
            return None, "strict_acquisition_review_source_pool_changed"
        try:
            review_absence_count = int(review.get("absence_pool_source_count"))
        except (TypeError, ValueError):
            return None, "strict_acquisition_review_pool_source_count_malformed"
        if review_absence_count != int(validated.get("pool_source_count", -1)):
            return None, "strict_acquisition_review_pool_source_count_changed"
        beat = dict((validated.get("beats") or {}).get(str(idx)) or {})
        if not beat:
            return None, "strict_acquisition_evidence_beat_record_missing"
        beat.update({
            "evidence_source": validated["evidence_source"],
            "evidence_sha256": validated["evidence_sha256"],
            "evidence_schema_version": validated["evidence_schema_version"],
        })
        return beat, "strict_acquisition_evidence_valid"
    except Exception as exc:                              # noqa: BLE001 — mutation must fail closed
        return None, f"strict_acquisition_review_validation_error:{type(exc).__name__}"


def _phase1_reviewed_exhaustion_details(proj, seg, cfg=None) \
        -> tuple[dict | None, str]:
    """Authorize the sole pre-acquisition specificity-ladder path and return its proof."""
    evidence, evidence_reason = _reviewed_exhaustion_evidence(proj, seg)
    if evidence is None:
        return None, evidence_reason
    native_gap = evidence.get("authorization_kind") == _NATIVE_HD_GAP_CLASSIFICATION
    # Matcher source gates may shrink the publishable pool after the independent audit.  The
    # content-hashed artifact covers all indexed OK sources, so it owns pool identity here.  The
    # common authorization still binds the reviewed beat and current semantic facts.  Only the
    # schema-4 native-HD proof may admit a currently verbatim quote.
    ok, reason = _phase1_softening_authorization(
        proj, seg, cfg, require_eligible_pool=False,
        allow_verified_native_gap=bool(native_gap))
    if not ok:
        return None, reason
    return evidence, "authorized_by_bound_completed_strict_acquisition_exhaustion"


def _phase1_reviewed_exhaustion_authorization(proj, seg, cfg=None) -> tuple[bool, str]:
    """Compatibility boolean wrapper for the evidence-rich authorization."""
    details, reason = _phase1_reviewed_exhaustion_details(proj, seg, cfg)
    return details is not None, reason


def make_selection_relevance_gap_review(proj, segments, confirmed_gap_beats, *,
                                        method: str, source: str = "",
                                        strict_acquisition_exhausted_beats=None,
                                        cfg=None) -> dict:
    """Create a tamper/staleness-bound authorization for the specificity ladder.

    Bare beat numbers are unsafe: matching can rewrite a beat and strict recovery can add source
    material after a viewer judged the old pool.  Consumers verify both fingerprints immediately
    before softening; any change leaves the strict publication blocker in place.
    """
    wanted = sorted({int(i) for i in (confirmed_gap_beats or [])})
    by_idx = {int(getattr(s, "index", -1)): s for s in (segments or [])}
    missing = [i for i in wanted if i not in by_idx]
    if missing:
        raise ValueError(f"cannot bind unknown gap beat(s): {missing}")
    pool_fp, pool_n = _gap_pool_fingerprint(proj)
    absence_pool_fp, absence_pool_n = _gap_absence_pool_fingerprint(proj)
    payload = {
        "schema_version": 2,
        "method": str(method or "actual_frame_and_pool_audit"),
        "source": str(source or ""),
        "confirmed_gap_beats": wanted,
        "beat_fingerprints": {str(i): _gap_beat_fingerprint(by_idx[i]) for i in wanted},
        "pool_fingerprint": pool_fp,
        "pool_source_count": pool_n,
        "absence_pool_scope": _GAP_ABSENCE_POOL_SCOPE,
        "absence_pool_fingerprint": absence_pool_fp,
        "absence_pool_source_count": absence_pool_n,
    }
    quoted_review_segments = [
        by_idx[i] for i in wanted
        if str(getattr(by_idx[i], "quote", "") or "").strip()
    ]
    if quoted_review_segments:
        payload["quote_retrieval_binding"] = _quote_retrieval_pool_binding(proj, cfg)
        payload["quote_branch_bindings"] = _quote_review_branch_bindings(
            proj, quoted_review_segments, cfg)
    exhausted = sorted({int(i) for i in (strict_acquisition_exhausted_beats or [])})
    if any(i not in wanted for i in exhausted):
        raise ValueError("strict-acquisition exhaustion must be a subset of confirmed gap beats")
    if exhausted:
        evidence_source = str(source or "").strip()
        if not evidence_source:
            raise ValueError("strict-acquisition exhaustion requires a non-empty evidence source")
        validated, evidence_reason = _strict_acquisition_evidence(
            proj, [by_idx[i] for i in exhausted], exhausted, source=evidence_source)
        if validated is None:
            if evidence_reason == "strict_acquisition_evidence_artifact_missing":
                raise ValueError(
                    "strict-acquisition exhaustion evidence source must be an existing file")
            raise ValueError(f"invalid strict-acquisition exhaustion evidence: {evidence_reason}")
        if str(method or "") != "actual_frame_and_pool_audit":
            raise ValueError(
                "strict-acquisition exhaustion requires method=actual_frame_and_pool_audit")
        payload["source"] = validated["evidence_source"]
        payload["strict_acquisition_exhaustion"] = {
            str(i): {
                **validated["beats"][str(i)],
                "evidence_source": validated["evidence_source"],
                "evidence_sha256": validated["evidence_sha256"],
            }
            for i in exhausted
        }
    return payload


_NATIVE_QUOTE_CONTENT_REASONS = frozenset({
    "exact_quote_dialogue_signal_below_floor",
    "exact_quote_timed_asr_span_absent",
    "exact_quote_timed_asr_outside_selected_window",
    "exact_quote_unprompted_confirmation_rejected",
})


def _semantic_gap_authorizations(proj, audit: dict, *, cfg=None) \
        -> tuple[dict[int, dict], str]:
    """Typed, fail-closed candidates for the post-recovery specificity ladder.

    A persisted viewer/pool audit may confirm genuine gaps and is authoritative when present. This
    keeps a known retrieval/ranking bug out of the softening path: a wrong pick is not permission to
    erase specificity. Without that marker nothing is softened. Even confirmed beats are excluded
    for technical/evidence-binding failures and real/indeterminate quote promises.
    """
    review = (getattr(proj, "meta", {}) or {}).get("selection_relevance_gap_review") or {}
    confirmed_raw = review.get("confirmed_gap_beats")
    confirmed = {int(i) for i in confirmed_raw} \
        if isinstance(confirmed_raw, (list, tuple, set)) else None
    # A semantic negative alone cannot distinguish "pool lacks it" from "matcher ignored the
    # correct pool shot"—the exact classification error this fix must not repeat. Without an
    # actual-frame/pool audit marker, fail closed and leave the publication blocker intact.
    if confirmed is None:
        return {}, "no_confirmed_actual_frame_gap_audit"
    if int(review.get("schema_version", 0) or 0) < 2:
        return {}, "unbound_gap_review"

    by_seg = {int(getattr(s, "index", -1)): s for s in (getattr(proj, "segments", None) or [])}
    bound_beats = review.get("beat_fingerprints") or {}
    if any(i not in by_seg or str(bound_beats.get(str(i), "")) != _gap_beat_fingerprint(by_seg[i])
           for i in confirmed):
        return {}, "stale_gap_review_beat_changed"
    pool_fp, _pool_n = _gap_pool_fingerprint(proj)
    review_pool_stale = bool(
        not review.get("pool_fingerprint")
        or str(review.get("pool_fingerprint")) != pool_fp)

    from .relevance_contract import completed_deliberate_exact_downgrade
    out: dict[int, dict] = {}
    quote_binding_denial = ""
    for entry in (audit.get("blockers") or []):
        idx = int(entry.get("segment_index", -1))
        reasons = [str(r) for r in (entry.get("reasons") or [])]
        branch = str((entry.get("quote_evidence") or {}).get("branch", "") or "")
        seg = by_seg.get(idx)
        quote = str(getattr(seg, "quote", "") or "").strip() if seg is not None else ""
        if idx not in confirmed or seg is None:
            continue
        # A real quote remains immutable by default.  The only exception is a current, hashed
        # schema-4 actual-frame/pool proof that every exact quote source is native-SD and no
        # publishable-HD copy exists.  Indeterminate/missing quote typing can never enter either
        # path.  Nonquoted and confirmed-paraphrase beats retain the existing schema-2 behavior.
        evidence, _evidence_reason = _reviewed_exhaustion_evidence(proj, seg)
        quote_binding_reason = _gap_review_quote_binding_reason(
            proj, seg, review, cfg,
            allow_native_verbatim=bool(
                isinstance(evidence, dict)
                and evidence.get("authorization_kind") == _NATIVE_HD_GAP_CLASSIFICATION))
        if quote_binding_reason:
            quote_binding_denial = quote_binding_reason
            continue
        # Ordinary actual-frame gap classifications are bound to the publishable pool and become
        # stale when it changes.  A completed strict-exhaustion row is stronger: it is content-
        # hashed against every indexed OK source, so matcher-only pool drift may still authorize
        # this one beat after the per-beat evidence validator above succeeds.  Any all-indexed
        # source/index change makes ``evidence`` None and continues to fail closed.
        if review_pool_stale and evidence is None:
            continue
        native_quote_gap = bool(
            quote and branch == "verbatim" and isinstance(evidence, dict)
            and evidence.get("authorization_kind") == _NATIVE_HD_GAP_CLASSIFICATION)
        if quote and branch != "paraphrase" and not native_quote_gap:
            continue
        if branch == "indeterminate" or (quote and not branch):
            continue
        if any(any(r.startswith(p) for p in _SEMANTIC_TECHNICAL_REASONS) for r in reasons):
            continue                                    # a code/evidence fault cannot buy a downgrade
        deliberate_downgrade = completed_deliberate_exact_downgrade(entry)
        semantic = [
            reason for reason in reasons
            if reason in _SEMANTIC_NEGATIVE_REASONS
        ]
        native_quote_content = [
            reason for reason in reasons
            if native_quote_gap and reason in _NATIVE_QUOTE_CONTENT_REASONS
        ]
        # Fail closed on every reason outside the narrow semantic-negative vocabulary.  A mixed
        # blocker (for example verdict_replace + stale evidence) is still a technical blocker; the
        # viewer's pool-gap review cannot turn it into permission to mutate the beat contract.
        if not deliberate_downgrade and len(semantic) + len(native_quote_content) != len(reasons):
            continue
        if not (semantic or native_quote_content or deliberate_downgrade):
            continue
        authorization = {
            "authorization_kind": "confirmed_actual_frame_audit",
            "strict_acquisition_exhausted": False,
            "surrender_verbatim_quote": False,
        }
        if isinstance(evidence, dict):
            authorization.update({
                "authorization_kind": str(evidence.get("authorization_kind", "") or
                                           "total_absence"),
                "strict_acquisition_exhausted": True,
                "surrender_verbatim_quote": bool(
                    evidence.get("surrender_verbatim_quote") and native_quote_gap),
                "evidence_source": str(evidence.get("evidence_source", "") or ""),
                "evidence_sha256": str(evidence.get("evidence_sha256", "") or ""),
                "evidence_schema_version": int(
                    evidence.get("evidence_schema_version", 0) or 0),
                "nonpublishable_exact_sources": copy.deepcopy(
                    evidence.get("nonpublishable_exact_sources") or []),
            })
        out[idx] = authorization
    if out and review_pool_stale:
        return out, "current_all_indexed_strict_exhaustion_evidence"
    if review_pool_stale:
        return {}, "stale_gap_review_source_pool_changed"
    if quote_binding_denial:
        return {}, quote_binding_denial
    return out, "confirmed_actual_frame_audit"


def semantic_gap_candidates(proj, audit: dict) -> tuple[list[int], str]:
    """Compatibility surface returning the indices from evidence-rich authorizations."""
    authorizations, basis = _semantic_gap_authorizations(proj, audit)
    return sorted(authorizations), basis


def reviewed_exhausted_gap_authorizations(proj, audit: dict, *, cfg=None) -> dict[int, dict]:
    """Current gap blockers that may bypass a duplicate strict acquisition/image-fallback page."""
    authorizations, _basis = _semantic_gap_authorizations(proj, audit, cfg=cfg)
    return {
        idx: details for idx, details in authorizations.items()
        if details.get("strict_acquisition_exhausted") is True
    }


def heal_selection_relevance_gaps(proj, segments, cfg, audit: dict, *, policy: str,
                                  eng=None, log=print) -> dict:
    """Run exact→character→abstract only after strict-positive semantic recovery is exhausted.

    The unchanged selection-relevance contract is evaluated again by the caller. This helper never
    turns a technical failure into an abstract pass. A located real quote is surrenderable only by
    its current schema-4 native-HD-absence proof, and every surrendered requirement is persisted for
    the next audit.
    """
    from .config import engine_config
    from . import policy as P
    authorizations, basis = _semantic_gap_authorizations(proj, audit, cfg=cfg)
    candidates = sorted(authorizations)
    by_seg = {int(getattr(s, "index", -1)): s for s in (segments or [])}
    by_sel = {int(getattr(s, "segment_index", -1)): s for s in (proj.selections or [])}
    by_entry = {int(e.get("segment_index", -1)): e for e in (audit.get("blockers") or [])}
    eng = eng or engine_config()
    pool = _clean_pool(proj) if candidates else []
    softening_pool_fp, softening_pool_n = _gap_pool_fingerprint(proj)
    used = {str(getattr(s, "image_path", "") or "") for s in (proj.selections or [])
            if getattr(s, "image_path", "")}
    # This is a transaction with respect to the publication contract.  A verifier, cache, disk, or
    # audit failure must not strand a beat in the abstract policy (which the strict contract skips).
    # Snapshot every potentially-mutated object and the prior audit/meta before the first rung.
    snapshots = {
        idx: (copy.deepcopy(vars(by_seg[idx])), copy.deepcopy(vars(by_sel[idx])))
        for idx in candidates if idx in by_seg and idx in by_sel
    }
    old_meta_present = "selection_relevance_gap_softening" in (getattr(proj, "meta", {}) or {})
    old_meta = copy.deepcopy((getattr(proj, "meta", {}) or {}).get(
        "selection_relevance_gap_softening"))
    dest = Path(proj.output_dir) / "semantic_gap_softening_audit.json"
    old_audit_present = dest.is_file()
    old_audit = dest.read_bytes() if old_audit_present else b""
    rows = []
    try:
        for idx in candidates:
            seg, sel = by_seg.get(idx), by_sel.get(idx)
            entry = by_entry.get(idx) or {}
            if seg is None or sel is None:
                rows.append({"segment_index": idx, "status": "not_attempted",
                             "reason": "segment_or_selection_absent"})
                continue
            old = _capture_softening_state(seg, sel)
            if old["image_path"]:
                try:
                    from .relevance_contract import verified_still_coverage
                    covered, why = verified_still_coverage(sel, seg)
                except Exception as exc:                 # noqa: BLE001 — fail closed
                    covered, why = False, f"still evidence check failed: {type(exc).__name__}"
                if not covered:
                    _discard_invalid_still(sel, why, log)
            log(f"semantic-gap: beat {idx} exhausted strict-positive recovery; running the "
                f"specificity ladder ({basis})")
            authorization = authorizations.get(idx) or {}
            ok = _soften_and_retry(
                proj, seg, sel, eng, pool, used, log,
                surrender_verbatim_quote=bool(
                    authorization.get("surrender_verbatim_quote")))
            if ok:
                row = _softening_row(
                    proj, seg, sel, old, phase="phase2_publication_recovery", basis=basis,
                    pool_fingerprint=softening_pool_fp,
                    pool_source_count=softening_pool_n,
                    trigger_reasons=entry.get("reasons") or [],
                    quote_branch=str((entry.get("quote_evidence") or {}).get(
                        "branch", "") or ""), authorization=authorization, cfg=cfg)
            else:
                row = {
                    "segment_index": idx,
                    "phase": "phase2_publication_recovery",
                    "basis": basis,
                    "status": "still_blocked",
                    "active": False,
                    "pool_fingerprint": softening_pool_fp,
                    "pool_source_count": softening_pool_n,
                    "original": old,
                    "new": _softening_new_state(seg, sel),
                    "trigger_reasons": list(entry.get("reasons") or []),
                    "quote_branch": str((entry.get("quote_evidence") or {}).get(
                        "branch", "") or ""),
                    "authorization": copy.deepcopy(authorization),
                    "surrendered_quote": bool(
                        old.get("quote") and not getattr(seg, "quote", "")),
                    "dropped_requirement": False,
                }
            rows.append(row)
        current_pool_fp, current_pool_n = _gap_pool_fingerprint(proj)
        if any(r.get("status") == "softened" for r in rows) \
                and (current_pool_fp, current_pool_n) != (softening_pool_fp, softening_pool_n):
            raise RuntimeError(
                "source/index pool changed while semantic specificity was being softened")
        payload = _merge_softening_payload(
            proj, rows, basis=basis, existing=old_meta if old_meta_present else None)
        _persist_softening_payload(proj, payload)
        _venue_cache_save(proj)
        return payload
    except Exception:
        # Restore in-memory state first: even if disk itself is failing, the caller's immediate
        # contract evaluation must see the original strict promises and block publication.
        for idx, (seg_state, sel_state) in snapshots.items():
            vars(by_seg[idx]).clear()
            vars(by_seg[idx]).update(copy.deepcopy(seg_state))
            vars(by_sel[idx]).clear()
            vars(by_sel[idx]).update(copy.deepcopy(sel_state))
        if old_meta_present:
            proj.meta["selection_relevance_gap_softening"] = old_meta
        else:
            proj.meta.pop("selection_relevance_gap_softening", None)
        try:
            if old_audit_present:
                restore_tmp = dest.with_suffix(dest.suffix + ".rollback.tmp")
                restore_tmp.write_bytes(old_audit)
                restore_tmp.replace(dest)
            else:
                dest.unlink(missing_ok=True)
            proj.save()
        except Exception:                                # noqa: BLE001 — gate uses restored memory
            pass
        raise


def run(proj, segments, cfg, analysis, *, policy: str, faceid_obj=None, refs=None,
        roster=None, log=print) -> str | None:
    """The bounded self-healing loop around the pre-assembly gate. Returns the final gate
    reason (None = clear). Never weakens the gate: it only adds verified footage/stills."""
    from .build import preassemble_release_block_reason
    if not _env_on("VIDLORE_CLIPSTUDIO_SELFHEAL", "1"):
        return preassemble_release_block_reason(proj, segments, analysis)
    rounds = max(1, _env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_ROUNDS", 3))
    pre = preassemble_release_block_reason(proj, segments, analysis)
    r = 0
    while pre and r < rounds:
        r += 1
        idxs = parse_blocked(pre) or blocked_indexes(proj)
        log(f"self-heal: round {r}/{rounds} — {len(idxs)} blocked beat(s): {idxs}")
        t0 = time.time()
        n = heal_blocked_beats(
            proj, segments, cfg, blocked=idxs, policy=policy,
            allow_acquire=(r >= 1), faceid_obj=faceid_obj, refs=refs,
            roster=roster, log=log)
        proj.save()
        log(f"self-heal: round {r} resolved {n} beat(s) in {(time.time() - t0) / 60:.1f} min")
        pre = preassemble_release_block_reason(proj, segments, analysis)
        if pre and n == 0:
            log("self-heal: no progress this round — stopping (gate stays authoritative)")
            break
    if not pre:
        log("self-heal: pre-assembly gate CLEAR")
    return pre
