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

Bounded and fail-open everywhere: max rounds, max sources per beat, a no-progress round stops
the loop, and every gate decision stays with the existing gate code. Kill switch:
VIDLORE_CLIPSTUDIO_SELFHEAL=0.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from .models import ClipProject, SOURCE_OK

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


def _soften_to_abstract(seg, log) -> None:
    log(f"self-heal: beat {seg.index} asks for gate-forbidden content "
        f"({(seg.expected_visual or seg.text or '')[:60]!r}) — softened to an abstract beat "
        f"(a meta line must not fail the video)")
    seg.visual_policy = "abstract_effect"
    seg.required_entity = ""
    seg.required_kind = ""
    try:
        seg.is_specific_claim = False
    except Exception:                                    # noqa: BLE001
        pass
    seg.expected_visual = ("Neutral, era-appropriate imagery from the show — a wide or "
                           "atmospheric shot as visual rest under a meta line.")


def _soften_to_character(seg, log) -> bool:
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
    log(f"self-heal: beat {seg.index} — the exact moment is not in reach after acquisition; "
        f"settling for the right subject instead of failing the video "
        f"(exact_scene -> character_specific"
        + (f", dropping the {_kind or 'unnamed'} requirement "
           f"{(getattr(seg, 'required_entity', '') or '')[:40]!r}" if _drop else "") + ")")
    seg.visual_policy = _P.CHARACTER
    if _drop:
        seg.required_entity = ""
        seg.required_kind = ""
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
    pool = []
    for f in sorted(Path(proj.index_dir).glob("*.shots.json")):
        sid = f.name.replace(".shots.json", "")
        if sid in banned:
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
            model=(model_id or _vision_model(eng_cfg)), venue_fallback=True, must_see="")
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
        if _hit is not None and _verdict_schema_ok(_hit) \
                and _hit_provider_ok(_hit, _vision_model(eng_cfg)):
            _pm_vv.incr("selfheal.venue.cache_hit")
            return dict(_hit)                            # copy: callers mutate their verdict
    _pm_vv.incr("selfheal.venue.call")
    v = verify_frame(str(kf_path), seg.text, getattr(seg, "required_entity", "") or "",
                     getattr(seg, "required_kind", "") or "", list(faces or []), eng_cfg,
                     is_specific=False,
                     expected_visual=getattr(seg, "expected_visual", "") or "",
                     scene_query=getattr(seg, "scene_query", "") or "",
                     venue_fallback=True)
    if _fp and v is not None and _verdict_schema_ok({**v, "status": "ok"}):
        cache[_fp] = {k: val for k, val in v.items() if k != "reused"}
    return v


def _install_still(sel, kf_path: str, sid: str, shot_idx: int, rel: float) -> None:
    sel.image_path = str(kf_path)
    sel.image_meta = {"source": "source-frame-recovery", "score": round(float(rel or 0), 3),
                      "src": sid, "shot": int(shot_idx),
                      "relevance_class": "contextual_fallback", "still_verified": True,
                      "lowres_still": False, "exact_scene_missing": True}


def still_recover(proj, seg, sel, eng_cfg, *, pool=None, cand_n: int = 8,
                  used_paths=None, log=print) -> bool:
    """Pool-wide venue-bar still recovery for one beat. True when a verified still installed."""
    from . import image_fallback as IF
    if getattr(sel, "image_path", ""):
        return True
    pool = pool if pool is not None else _clean_pool(proj)
    used_paths = used_paths if used_paths is not None else set()
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
        return any(isinstance(verdicts.get(sh.keyframe_path), dict)
                   and verdicts[sh.keyframe_path].get("verdict") == "keep"
                   for _r, _s, sh in batch)

    for _i in range(0, len(cands), _wave):
        if _warm_wave(cands[_i:_i + _wave]):
            break
    for rel, sid, sh in cands:
        v = verdicts.get(sh.keyframe_path)
        if v is None:
            v = _venue_verify(sh.keyframe_path, seg, getattr(sh, "face_ids", []), eng_cfg,
                              proj=proj, cache=_vv_cache)
        if isinstance(v, dict) and v.get("verdict") == "keep":
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
                cands.append((float(arr.mean()), str(fp)))
            except Exception:                            # noqa: BLE001
                continue
    cands.sort(reverse=True)
    for luma, fp in cands[:4]:
        if _frame_text_dirty(fp):
            continue                                     # burned subs/meme text/watermark card
        v = _venue_verify(fp, seg, [], eng_cfg, proj=proj, cache=_venue_cache(proj))
        if isinstance(v, dict) and v.get("verdict") == "keep":
            _install_still(sel, fp, src.id, -1, 0.8)
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


def acquire_for_beat(proj, seg, cfg, *, policy: str, log=print) -> list:
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
        cands = cands + (D.discover_sources(ana, cfg_r, segments=[seg], progress=None) or [])
    except Exception as e:                               # noqa: BLE001
        log(f"self-heal: discovery failed ({str(e)[:60]})")
        if not cands:
            return []
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
    try:
        download_candidates(proj, new, cfg, policy=policy, progress=None)
    except Exception as e:                               # noqa: BLE001
        log(f"self-heal: download failed ({str(e)[:60]})")
        return []
    fresh = []
    for sv in proj.sources:
        if sv.url in {c.url for c in new} and sv.status == SOURCE_OK and sv.local_path:
            try:
                I.index_source(proj, sv, cfg, progress=None)
                fresh.append(sv)
            except Exception as e:                       # noqa: BLE001
                log(f"self-heal: index failed for {sv.id} ({str(e)[:60]})")
    return fresh


# ── the loop ──────────────────────────────────────────────────────────────────────────────────

def _soften_and_retry(proj, seg, sel, eng, pool, used, log) -> bool:
    """Soften an unreachable exact beat and search the SAME pool once more under the looser bar.

    Deliberately no re-acquisition and no new pool: the point is not to try harder, it is to stop
    asking for something retrieval cannot return. The candidate depth is raised because a
    character-policy still only has to show the right subject, so it is worth looking past the
    handful of frames that already failed the exact test."""
    if not _soften_to_character(seg, log):
        return False
    return bool(still_recover(
        proj, seg, sel, eng, pool=pool, used_paths=used,
        cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_SOFT_CANDS", 16), log=log))


def heal_blocked_beats(proj, segments, cfg, *, blocked: list[int], policy: str,
                       allow_acquire: bool = True, log=print) -> int:
    """One healing pass over `blocked` beat indexes. Returns beats resolved this pass."""
    from .config import engine_config
    eng = engine_config()
    segs = {s.index: s for s in segments}
    sels = {s.segment_index: s for s in proj.selections}
    pool = None
    used = {getattr(s, "image_path", "") for s in proj.selections
            if getattr(s, "image_path", "")}
    resolved = 0
    for bidx in blocked:
        seg = segs.get(bidx)
        sel = sels.get(bidx)
        if seg is None or sel is None:
            continue
        if getattr(sel, "image_path", ""):
            resolved += 1
            continue
        if beat_unfillable(seg):
            _soften_to_abstract(seg, log)
        if pool is None:
            pool = _clean_pool(proj)
        if still_recover(proj, seg, sel, eng, pool=pool, used_paths=used,
                         cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", 8),
                         log=log):
            resolved += 1
            continue
        if allow_acquire:
            fresh = acquire_for_beat(proj, seg, cfg, policy=policy, log=log)
            if fresh:
                pool = _clean_pool(proj)                 # refresh with the new shots
                if still_recover(proj, seg, sel, eng, pool=pool, used_paths=used,
                                 cand_n=_env_int("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", 8),
                                 log=log):
                    resolved += 1
                    continue
                for sv in fresh:
                    if _region_frames_recover(proj, seg, sel, sv, eng, log=log):
                        resolved += 1
                        break
                else:
                    if _soften_and_retry(proj, seg, sel, eng, pool, used, log):
                        resolved += 1
                        continue
                    log(f"self-heal: beat {bidx} unresolved this pass")
                    continue
                continue
        if _soften_and_retry(proj, seg, sel, eng, pool, used, log):
            resolved += 1
            continue
        log(f"self-heal: beat {bidx} unresolved this pass")
    # persist this pass's venue verdicts so the NEXT round (and the review-draft retry, which
    # re-runs the whole heal) replays them instead of re-buying identical answers
    _venue_cache_save(proj)
    return resolved


def run(proj, segments, cfg, analysis, *, policy: str, log=print) -> str | None:
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
        n = heal_blocked_beats(proj, segments, cfg, blocked=idxs, policy=policy,
                               allow_acquire=(r >= 1), log=log)
        proj.save()
        log(f"self-heal: round {r} resolved {n} beat(s) in {(time.time() - t0) / 60:.1f} min")
        pre = preassemble_release_block_reason(proj, segments, analysis)
        if pre and n == 0:
            log("self-heal: no progress this round — stopping (gate stays authoritative)")
            break
    if not pre:
        log("self-heal: pre-assembly gate CLEAR")
    return pre
