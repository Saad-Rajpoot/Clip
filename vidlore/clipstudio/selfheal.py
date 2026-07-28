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


def _venue_verify(kf_path: str, seg, faces, eng_cfg):
    """One venue-bar vision verdict (right scene/venue, no contradiction) — the still layer's
    real question. Returns the verdict dict or None."""
    from .verify import verify_frame
    return verify_frame(str(kf_path), seg.text, getattr(seg, "required_entity", "") or "",
                        getattr(seg, "required_kind", "") or "", list(faces or []), eng_cfg,
                        is_specific=False,
                        expected_visual=getattr(seg, "expected_visual", "") or "",
                        scene_query=getattr(seg, "scene_query", "") or "",
                        venue_fallback=True)


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
    ranked = []
    for sid, sh in pool:
        rel = None
        try:
            rel = IF._clip_relevance(Path(sh.keyframe_path), q)
        except Exception:                                # noqa: BLE001
            rel = None
        ranked.append((rel if rel is not None and rel >= 0 else 0.0, sid, sh))
    ranked.sort(key=lambda c: -c[0])
    tried = 0
    for rel, sid, sh in ranked:
        if tried >= cand_n:
            break
        if sh.keyframe_path in used_paths:
            continue
        tried += 1
        v = _venue_verify(sh.keyframe_path, seg, getattr(sh, "face_ids", []), eng_cfg)
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
        v = _venue_verify(fp, seg, [], eng_cfg)
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


def acquire_for_beat(proj, seg, cfg, *, policy: str, log=print) -> list:
    """Discover→download→index up to SELFHEAL_MAX_SRC new sources for this beat's scene.
    Returns the newly indexed SourceVideo objects."""
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
    try:
        cands = D.discover_sources(ana, cfg_r, segments=[seg], progress=None) or []
    except Exception as e:                               # noqa: BLE001
        log(f"self-heal: discovery failed ({str(e)[:60]})")
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
                    log(f"self-heal: beat {bidx} unresolved this pass")
                    continue
                continue
        log(f"self-heal: beat {bidx} unresolved this pass")
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
