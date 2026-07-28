#!/usr/bin/env python3
"""Surgical, gate-honest still installation for the two residual beats (39, 127).

Same mechanism _fill_image_fallbacks uses — a REAL source keyframe, vision-verified at the
lenient bar, installed as sel.image_path with the honest contextual_fallback labeling — but
with a wider candidate net than the in-render pass (which starved: beat 127's scene is a dark
S2E3 chamber; the fine re-index at 1080p now has 60 shots to choose from).
"""
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
                                      _ocr_text_heavy, _ocr_is_junk)
from vidlore.clipstudio import image_fallback as IF        # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


proj = ClipProject.load(str(J))
cfg = ClipConfig()
eng = engine_config()
segs = {s.index: s for s in proj.segments}
sels = {s.segment_index: s for s in proj.selections}

# candidate plans per beat: (source filter, preferred shot range, extra query)
PLANS = {
    39: dict(src_like=("king_j", "s4e2", "purple", "joffre"), rng=None,
             query="Joffrey mocks Tyrion cupbearer Purple Wedding confrontation"),
    127: dict(src_like=("tyrion_confronts_maste",), rng=(20, 60),
              query="Pycelle seized guards Tyrion black cell arrest"),
}

for bidx, plan in PLANS.items():
    seg = segs[bidx]
    sel = sels[bidx]
    if getattr(sel, "image_path", ""):
        log(f"beat {bidx}: already has a still — skip")
        continue
    cands = []
    for sid_dir in (J / "index").iterdir():
        if not sid_dir.name.endswith(".shots.json"):
            continue
        sid = sid_dir.name.replace(".shots.json", "")
        if not any(t in sid for t in plan["src_like"]):
            continue
        try:
            shots = I.load_shots(proj, sid)
        except Exception:
            import json as _json
            shots = None
        if not shots:
            continue
        for sh in shots:
            if plan["rng"] and not (plan["rng"][0] <= sh.index <= plan["rng"][1]):
                continue
            kf = getattr(sh, "keyframe_path", "") or ""
            if not kf or not Path(kf).exists():
                continue
            if _shot_unreadable(sh) or _ocr_is_junk(sh) or _ocr_text_heavy(sh):
                continue
            la = float(getattr(sh, "luma_avg", -1) or -1)
            if 0 <= la < 12:
                continue
            rel = IF._clip_relevance(Path(kf), f"{seg.scene_query} {plan['query']}")
            cands.append((rel, sid, sh))
    cands.sort(key=lambda c: -c[0])
    log(f"beat {bidx}: {len(cands)} candidates, verifying top 10 at the lenient bar")
    installed = False
    for rel, sid, sh in cands[:10]:
        v = verify_frame(sh.keyframe_path, seg.text, seg.required_entity or "",
                         seg.required_kind or "", list(getattr(sh, "face_ids", []) or []),
                         eng, is_specific=False,
                         expected_visual=seg.expected_visual or "",
                         scene_query=seg.scene_query or "")
        ok = bool(isinstance(v, dict) and v.get("verdict") == "keep")
        log(f"  cand {sid[:34]} shot {sh.index} rel={rel:.2f} -> "
            f"{v.get('verdict') if isinstance(v, dict) else 'error'}"
            f" ({(v.get('reason') or '')[:70] if isinstance(v, dict) else ''})")
        if ok:
            sel.image_path = str(sh.keyframe_path)
            sel.image_meta = {"source": "source-frame-recovery", "score": round(float(rel), 3),
                              "src": sid, "shot": int(sh.index),
                              "relevance_class": "contextual_fallback",
                              "still_verified": True, "lowres_still": False,
                              "exact_scene_missing": True}
            log(f"beat {bidx}: INSTALLED verified still — {sid} shot {sh.index}")
            installed = True
            break
    if not installed:
        log(f"beat {bidx}: NO candidate passed the lenient vision bar")

proj.save()
log("done")
