#!/usr/bin/env python3
"""Batch still-recovery for every release-blocked beat: pool-wide CLIP-ranked candidate
frames, vision-verified at the pipeline's own venue bar, installed with the honest
contextual_fallback labeling. Exactly the in-render still pass, with a wider net."""
import json
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
J = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))
for _line in (MAIN / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from vidlore.clipstudio.models import ClipProject          # noqa: E402
from vidlore.clipstudio.config import ClipConfig, engine_config  # noqa: E402
from vidlore.clipstudio import index as I                  # noqa: E402
from vidlore.clipstudio.verify import verify_frame         # noqa: E402
from vidlore.clipstudio.match import (_shot_unreadable,    # noqa: E402
                                      _ocr_text_heavy, _ocr_is_junk,
                                      _shot_static_collage, _shot_numeral_overlay,
                                      banned_source_ids)
from vidlore.clipstudio import image_fallback as IF        # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


proj = ClipProject.load(str(J))
cfg = ClipConfig()
eng = engine_config()
segs = {s.index: s for s in proj.segments}
sels = {s.segment_index: s for s in proj.selections}

blocked = [e["seg_index"] for e in (json.load(open(J / "output" / "rejected_footage_audit.json"))
                                    .get("unresolved_release_block") or [])]
log(f"blocked beats: {blocked}")

banned = banned_source_ids(proj, include_auto=True)
pool = []                                   # (sid, Shot) all clean shots, loaded once
for f in sorted((J / "index").glob("*.shots.json")):
    sid = f.name.replace(".shots.json", "")
    if sid in banned:
        continue
    try:
        shots = I.load_shots(proj, sid)
    except Exception:
        continue
    for sh in shots:
        kf = getattr(sh, "keyframe_path", "") or ""
        if not kf or not Path(kf).exists():
            continue
        if (_shot_unreadable(sh) or _ocr_is_junk(sh) or _ocr_text_heavy(sh)
                or _shot_static_collage(sh) or _shot_numeral_overlay(sh)):
            continue
        if float(getattr(sh, "luma_avg", -1) or -1) >= 0 and float(sh.luma_avg) < 10:
            continue
        pool.append((sid, sh))
log(f"clean candidate pool: {len(pool)} shots")

used_paths = {getattr(s, "image_path", "") for s in proj.selections if getattr(s, "image_path", "")}
installed = failed = 0
for bidx in blocked:
    seg = segs[bidx]
    sel = sels[bidx]
    if getattr(sel, "image_path", ""):
        continue
    q = " ".join(x for x in (getattr(seg, "scene_query", ""), seg.expected_visual) if x) \
        or seg.text
    ranked = []
    for sid, sh in pool:
        rel = IF._shot_relevance(sh, sh.keyframe_path, q,
                                 embeds_of=None, rel_memo=None)
        if rel is None or rel < 0:
            rel = IF._clip_relevance(Path(sh.keyframe_path), q)
        ranked.append((rel, sid, sh))
    ranked.sort(key=lambda c: -(c[0] or 0))
    got = False
    tried = 0
    for rel, sid, sh in ranked:
        if tried >= 8:
            break
        if sh.keyframe_path in used_paths:
            continue
        tried += 1
        v = verify_frame(sh.keyframe_path, seg.text, seg.required_entity or "",
                         seg.required_kind or "", list(getattr(sh, "face_ids", []) or []),
                         eng, is_specific=False,
                         expected_visual=seg.expected_visual or "",
                         scene_query=getattr(seg, "scene_query", "") or "",
                         venue_fallback=True)
        if isinstance(v, dict) and v.get("verdict") == "keep":
            sel.image_path = str(sh.keyframe_path)
            sel.image_meta = {"source": "source-frame-recovery",
                              "score": round(float(rel or 0), 3), "src": sid,
                              "shot": int(sh.index), "relevance_class": "contextual_fallback",
                              "still_verified": True, "lowres_still": False,
                              "exact_scene_missing": True}
            used_paths.add(sh.keyframe_path)
            log(f"beat {bidx}: INSTALLED {sid[:36]} shot {sh.index} rel={rel:.2f} "
                f"| {(v.get('reason') or '')[:60]}")
            installed += 1
            got = True
            break
    if not got:
        log(f"beat {bidx}: NO candidate passed (tried {tried})")
        failed += 1

proj.save()
log(f"done — installed {installed}, unresolved {failed}")
