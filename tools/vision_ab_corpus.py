#!/usr/bin/env python3
"""Build a corpus of REAL verifier questions from a finished job, for the live vision A/B.

A "question" is exactly what verify_frame would be asked for one (beat, candidate frame) pair:
the beat's storyboard fields plus the frame on disk. Reconstructing them from a real job (rather
than inventing prompts) is what makes the A/B measure production behaviour.

Stratified so the corpus mirrors the population each cost item would actually touch:
  * LENIENT questions (is_specific=False) — the only class item 1 would re-route to flash-lite:
    venue-bar screens and contextual/generic re-asks.
  * STRICT questions (is_specific=True) — kept on flash; included as a control so a model
    difference that only shows up on strict asks cannot hide.
Frames come from each beat's own selection and its ranked alternates, so both accepted and
rejected-looking footage is represented.

    python3 tools/vision_ab_corpus.py [job_dir] [n] > corpus.json
"""
import json
import os
import random
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE))

DEFAULT_JOB = "/Users/hussnain/Desktop/clipstudio_output/portal/canary_spd_b"


def build(job_dir: str, n: int = 120, seed: int = 17) -> list:
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio import index as IX
    from vidlore.clipstudio import policy as P
    from vidlore.clipstudio import verify as V

    proj = ClipProject.load(job_dir)
    segs = {s.index: s for s in proj.segments}
    analysis = (proj.meta or {}).get("analysis") or {}
    global_era = str(analysis.get("episode_hint", "") or "")
    single = analysis.get("video_type") == "single_scene"
    shots_by_src = {}

    def shots_of(sid):
        if sid not in shots_by_src:
            try:
                shots_by_src[sid] = IX.load_shots(proj, sid)
            except Exception:                            # noqa: BLE001
                shots_by_src[sid] = []
        return shots_by_src[sid]

    def shot(sid, ix):
        for sh in shots_of(sid):
            if getattr(sh, "index", None) == ix:
                return sh
        return None

    items = []
    for sel in proj.selections:
        seg = segs.get(sel.segment_index)
        if seg is None:
            continue
        cands = [(sel.source_id, sel.shot_index)]
        for alt in (getattr(sel, "alternates", None) or [])[:2]:
            cands.append((alt.source_id, alt.shot_index))
        for sid, six in cands:
            sh = shot(sid, six)
            kf = getattr(sh, "keyframe_path", "") if sh is not None else ""
            if not kf or not os.path.exists(kf):
                continue
            faces = list(getattr(sh, "face_ids", None) or [])
            era = V._beat_era(seg, global_era, single, anchor_eras=None) or ""
            try:
                must_see = P.deictic_target(seg) or ""
            except Exception:                            # noqa: BLE001
                must_see = ""
            base = dict(keyframe=kf, narration=seg.text,
                        required_entity=getattr(seg, "required_entity", "") or "",
                        required_kind=getattr(seg, "required_kind", "") or "",
                        faceid_names=faces,
                        expected_visual=getattr(seg, "expected_visual", "") or "",
                        scene_query=getattr(seg, "scene_query", "") or "",
                        seg_index=seg.index, source_id=sid, shot_index=six)
            # STRICT ask, exactly as the primary/promotion path sends it
            items.append(dict(base, cls="strict", is_specific=True, era_hint=era,
                              must_see=must_see, venue_fallback=False))
            # LENIENT asks: the venue-bar screen (no era, no must_see — see selfheal/_still) and
            # the contextual re-ask (era kept). These are item 1's whole population.
            items.append(dict(base, cls="venue", is_specific=False, era_hint="",
                              must_see="", venue_fallback=True))
            items.append(dict(base, cls="contextual", is_specific=False, era_hint=era,
                              must_see="", venue_fallback=False))

    rnd = random.Random(seed)
    out = []
    # stratify: half the budget to the lenient classes item 1 targets, the rest strict control
    for cls, share in (("venue", 0.34), ("contextual", 0.33), ("strict", 0.33)):
        pool = [i for i in items if i["cls"] == cls]
        rnd.shuffle(pool)
        out.extend(pool[:max(1, int(n * share))])
    rnd.shuffle(out)
    return out


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JOB
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    corpus = build(job, n)
    from collections import Counter
    print(json.dumps(corpus, indent=1))
    print(f"# {len(corpus)} questions: {dict(Counter(c['cls'] for c in corpus))}",
          file=sys.stderr)
