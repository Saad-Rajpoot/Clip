"""Phase 6 — mandatory AI visual verification.

A second pass that shows each selected clip's representative frame to Claude (vision) alongside
the narration line + the entity that line demands + the automatic Face-ID result, and asks:
does this clip actually match? is the correct actor/character visible? is it specific enough?
is the quality acceptable? On a "replace" verdict the clip is swapped for the next-best
alternate and re-verified — so weak/wrong/blurry picks are repaired automatically.

Uses the engine's Claude key. If no key, this pass is skipped (the pipeline still produces a
video; the QC report notes verification was unavailable). The verifier never claims certainty —
its verdict is recorded per clip for the QC report and to drive replacement.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from .models import ClipProject, ScriptSegment, ClipSelection, FLAG_EXACT_MISSING
from .config import ClipConfig
from . import index as _index
from . import cut as _cut
from . import policy as _policy

_VSYS = (
    "You are a strict film-footage QC editor. You judge whether ONE clip's representative frame "
    "correctly illustrates a narration line. Be skeptical: if the specific person/character/object "
    "the line is about is not clearly visible, or the frame is blurry, a title card, a watermark, or "
    "only loosely related, you must fail it. Reply with ONLY a JSON object."
)


def _img_block(path: Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def verify_frame(keyframe_path: str, narration: str, required_entity: str, required_kind: str,
                 faceid_names: list[str], eng_cfg, model: str = "", is_specific: bool = True) -> dict | None:
    """One vision verdict for a single frame (Gemini brain → Claude fallback). None on error.

    `is_specific` carries the beat's is_specific_claim: a SPECIFIC line ("Tyrion shoots Tywin with a
    crossbow") demands the EXACT scene; a GENERIC line ("and everything changed") only needs a
    thematically-relevant filler — so the verifier is told to grade leniently there."""
    if not keyframe_path or not Path(keyframe_path).exists():
        return None
    from . import llm as _llm
    _rule = (
        "This line refers to a SPECIFIC scene/moment — the footage must show THAT exact scene/"
        "subject. Be STRICT.\n" if is_specific else
        "This is a GENERIC narration line (no specific scene claim) — a thematically RELEVANT filler "
        "clip is acceptable. Mark 'replace' ONLY if the footage is off-topic, jarring, or shows the "
        "WRONG character/era — NOT merely because it isn't a specific/exact scene.\n")
    txt = (
        f'Narration line: "{narration}"\n'
        f"This clip should show: {required_entity or '(a general scene fitting the line)'} "
        f"(kind: {required_kind or 'any'}).\n"
        + _rule +
        f"Automatic Face-ID on this frame detected: {', '.join(faceid_names) if faceid_names else 'none'}.\n\n"
        "Answer ONLY this JSON:\n"
        '{"matches_narration": true/false, "correct_subject_visible": true/false, '
        '"specific_enough": true/false, "quality_ok": true/false, '
        '"confidence": 0.0-1.0, "verdict": "keep" or "replace", "reason": "one short sentence"}'
    )
    import time
    content = [_img_block(Path(keyframe_path)), {"type": "text", "text": txt}]
    for attempt in range(1, 5):                       # retry transient overload / rate limits
        try:
            out = _llm.complete(system=_VSYS, max_tokens=400,
                                messages=[{"role": "user", "content": content}],
                                eng_cfg=eng_cfg, model=model)
            m = re.search(r"\{.*\}", out, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception:                             # transient overload / rate limit → back off
            if attempt == 4:
                return None
            time.sleep(min(1.5 * (2 ** attempt), 16))
    return None


def _shot_lookup(proj: ClipProject):
    cache: dict[str, dict] = {}

    def get(source_id, shot_index):
        if not source_id:
            return None
        if source_id not in cache:
            cache[source_id] = {s.index: s for s in _index.load_shots(proj, source_id)}
        return cache[source_id].get(shot_index)

    def all_shots(source_id):
        if not source_id:
            return []
        if source_id not in cache:
            cache[source_id] = {s.index: s for s in _index.load_shots(proj, source_id)}
        return list(cache[source_id].values())

    get.all_shots = all_shots
    return get


def verify_and_repair(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig,
                      eng_cfg, *, max_replacements: int = 3, progress=None) -> dict:
    """Verify every selection; replace failures with the best passing alternate; re-cut swaps.
    Returns a summary. No-op (records 'unavailable') if there's no LLM key."""
    def log(m):
        if progress:
            progress(m)

    from . import llm as _llm
    if not _llm.has_llm(eng_cfg):
        for sel in proj.selections:
            sel.verifier = {"status": "unavailable", "reason": "no LLM key"}
        log("verify: skipped (no LLM key)")
        return {"verified": 0, "replaced": 0, "failed": 0, "available": False}

    get_shot = _shot_lookup(proj)
    by_idx = {s.index: s for s in segments}
    model = eng_cfg.anthropic_model
    verified = replaced = failed = 0

    for sel in proj.selections:
        if not sel.source_id:
            continue
        seg = by_idx.get(sel.segment_index)
        if seg is None:
            continue
        shot = get_shot(sel.source_id, sel.shot_index)
        kf = shot.keyframe_path if shot else ""
        faceid_names = (shot.face_ids if shot else []) or ([sel.identity] if sel.identity else [])
        _exact = _policy.verify_strict(seg)               # exact_scene → strict; else lenient (filler ok)
        v = verify_frame(kf, seg.text, seg.required_entity, seg.required_kind, faceid_names,
                         eng_cfg, model, is_specific=_exact)
        verified += 1
        if v is None:
            sel.verifier = {"status": "error"}
            continue
        v["status"] = "ok"
        v["visual_policy"] = _policy.policy_of(seg)
        # NON-EXACT LENIENCY (user rule: exact clip only for a SPECIFIC scene; a relevant FILLER is
        # fine for generic/character/abstract beats). Don't replace an on-topic, right-subject clip on
        # a non-exact beat just because it isn't the exact scene — only off-topic / wrong-character.
        if not _exact and v.get("verdict") == "replace" and v.get("matches_narration") \
                and v.get("correct_subject_visible") is not False:
            v["verdict"] = "keep"
            v["relaxed"] = "non-exact beat: relevant filler accepted"
        sel.verifier = v

        if v.get("verdict") == "replace":
            swapped = False
            tried = 0
            failed_wins: list = []      # alternates the verifier explicitly REJECTED on the way
            for alt in sel.alternates:
                if tried >= max_replacements:
                    break
                tried += 1
                ashot = get_shot(alt.source_id, alt.shot_index)
                if ashot is None:
                    continue
                anames = ashot.face_ids or []
                av = verify_frame(ashot.keyframe_path, seg.text, seg.required_entity,
                                  seg.required_kind, anames, eng_cfg, model, is_specific=_exact)
                if av is not None and av.get("verdict") != "keep":
                    # an explicit non-keep judgment (av None = transport error, NOT a judgment)
                    failed_wins.append((alt.source_id, float(alt.in_point)))
                if av and av.get("verdict") == "keep":
                    # CUT-WINDOW FLAG VALIDATION on the promotion — the repair must not swap a
                    # rejected clip for one whose PADDED render window airs an adjacent shot's
                    # burned subs / logo / murk. Same PRODUCTION validator as match selections:
                    # moment-locked beats (exact/quote/character) may only shorten around the
                    # alternate's own selected moment — never slide to a different moment —
                    # else this alternate is skipped for the next relevance-ranked one.
                    import os as _os_w
                    if _os_w.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() \
                            not in ("0", "false", "no"):
                        from .match import validate_candidate_window, _wqc_log_line
                        # stub-tolerant: tests monkeypatch _shot_lookup with a bare function —
                        # no shot list then means nothing to validate (fail-open)
                        _wshots = getattr(get_shot, "all_shots", lambda _s: [])(alt.source_id)
                        _wact, _wwhy, _wmeta = validate_candidate_window(
                            alt, ashot, _wshots, cfg, seg)
                        if _wact == "rejected":
                            log(f"window-qc: rejected verify-promotion seg{sel.segment_index} "
                                f"alt={alt.source_id[:28]} {_wqc_log_line(_wact, _wmeta, _wwhy)}")
                            failed_wins.append((alt.source_id, float(alt.in_point)))
                            continue
                        if _wact == "shortened":
                            log(f"window-qc: shortened verify-promotion seg{sel.segment_index} "
                                f"{_wqc_log_line(_wact, _wmeta, _wwhy)}")
                    # promote the alternate into the selection
                    old_sid, old_in = sel.source_id, sel.in_point
                    sel.source_id = alt.source_id
                    sel.shot_index = alt.shot_index
                    sel.in_point = alt.in_point
                    sel.out_point = alt.out_point
                    sel.signals = alt.signals
                    sel.confidence = alt.score
                    sel.source_url = (proj.source(alt.source_id).url if proj.source(alt.source_id) else "")
                    sel.identity = (anames[0] if anames else "")
                    # build_video plays the scene's beats from beat_windows (rejected pick is
                    # FIRST there) — drop it AND every alternate the verifier explicitly failed
                    # on the way here, then lead with the promoted window; otherwise rejected
                    # footage still airs on the scene's later beats.
                    new_win = [alt.source_id, round(alt.in_point, 3), round(alt.out_point, 3)]
                    kept = [w for w in (sel.beat_windows or [])
                            if not (w and w[0] == old_sid and abs(float(w[1]) - float(old_in)) < 0.05)
                            and not (w and w[0] == new_win[0] and abs(float(w[1]) - new_win[1]) < 0.05)
                            and not any(w and w[0] == fs and abs(float(w[1]) - fi) < 0.05
                                        for fs, fi in failed_wins)]
                    sel.beat_windows = [new_win] + kept
                    av["status"] = "ok"
                    av["replaced_from"] = {"shot": shot.index if shot else -1}
                    sel.verifier = av
                    _cut.cut_selection(proj, sel, cfg)     # re-cut the new in/out
                    replaced += 1
                    swapped = True
                    log(f"verify: seg{sel.segment_index} replaced → {alt.source_id}#{alt.shot_index}")
                    break
            if not swapped:
                failed += 1
                if "verifier_failed" not in sel.flag_reasons:
                    sel.flag_reasons.append("verifier_failed")
                # EXACT-SCENE MISSING (req. 9): an exact_scene beat with no passing real footage must
                # be marked for MANUAL REVIEW — the image-fallback will NOT silently cover it with a
                # web/AI image or loose filler (only a real source-frame of the exact scene may).
                if _exact and FLAG_EXACT_MISSING not in sel.flag_reasons:
                    sel.flag_reasons.append(FLAG_EXACT_MISSING)
                    log(f"verify: seg{sel.segment_index} EXACT-SCENE MISSING → manual review "
                        f"(no real footage of the exact scene)")
                sel.flagged = True
                log(f"verify: seg{sel.segment_index} FAILED, no passing alternate")
        if progress and sel.segment_index % 10 == 0:
            log(f"verify: {verified} checked, {replaced} replaced, {failed} unresolved")

    proj.save()
    log(f"verify: done — {verified} checked, {replaced} replaced, {failed} unresolved")
    return {"verified": verified, "replaced": replaced, "failed": failed, "available": True}
