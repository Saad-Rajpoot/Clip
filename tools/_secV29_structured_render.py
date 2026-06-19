#!/usr/bin/env python3
"""V2.9 STRUCTURED-ASSET PROOF RENDER — "How a Carrier Group Crosses an Ocean".

A short (~70s) portal-equivalent doc through the SHARED Vidlore pipeline whose
reviewed scenes carry EXPLICIT structured graphic_body so the two previously-
unreachable Section-C primitives fire naturally:
  • silhouette_scale_compare  (graphic_kind=scale_compare, items=...)
  • footage_route_trace       (graphic_kind=route_trace,  points=...)
Footage-first. AI VIDEO OFF; FAL stills budget 0.
"""
import hashlib, json, os, re as _re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "0")
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ["VIDLORE_FAL_MAX_IMAGES"] = "0"
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")

TITLE = "How a Carrier Group Crosses an Ocean"
NICHE = "history"
PROMPT = ("How a navy moves an aircraft carrier and its escorts across an entire "
          "ocean as one coordinated group")

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SCENES = [
    ("To move a single aircraft carrier across an ocean, a navy moves an entire "
     "city of steel around it.", 4, "hook", "", "", ""),
    # silhouette_scale_compare — the real size gap, true-scale
    ("The carrier itself dwarfs the ships that guard it — over twice the length "
     "of the destroyers at its side.", 3, "scale", "scale_compare",
     "SIZE COMPARISON",
     "items=Aircraft Carrier:330:length|Destroyer:155:length ; title=SIZE COMPARISON"),
    ("Around that one ship sails a ring of escorts: destroyers, a cruiser, and a "
     "supply vessel, each with a job.", 3, "group", "", "", ""),
    ("Together they are called a carrier strike group, and they almost never "
     "travel alone.", 3, "name", "", "", ""),
    ("The crossing is planned like a single journey, leg by leg, from home port "
     "to the far station.", 4, "plan", "", "", ""),
    # footage_route_trace — the ocean crossing traced over a sea shot
    ("The group threads a careful route: out from the home port, through a "
     "mid-ocean waypoint, and on to the distant station.", 3, "route",
     "route_trace", "THE CROSSING",
     "points=0.16:0.72:Home Port|0.48:0.50|0.82:0.30:Station ; title=THE CROSSING"),
    ("For days the formation holds, the smaller ships shadowing the giant at its "
     "centre.", 3, "hold", "", "", ""),
    ("It is less a ship crossing an ocean than a small nation on the move — "
     "carrying its own power across the water.", 4, "thesis", "statement",
     "A NATION ON THE MOVE", ""),
]


def _kw(nar, gt):
    kws = []
    if gt and not gt.isupper():
        kws.append(gt.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    _stop = {"through", "around", "across", "before", "every", "their", "would",
             "together", "almost", "single", "around"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:5]


def build_script():
    scenes = []
    for i, (nar, inten, role, gk, gt, gb) in enumerate(SCENES):
        scenes.append({"narration": nar, "keywords": _kw(nar, gt), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": gk, "graphic_text": gt,
                       "graphic_body": gb})
    return {"title": TITLE,
            "source_sha256": hashlib.sha256(
                (TITLE + "".join(s["narration"] for s in scenes)).encode()).hexdigest(),
            "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    from vidlore.motion_graphics import director as mgdir
    cfg = load_config()
    print(f"FAL configured: {'yes' if (os.environ.get('FAL_KEY') or getattr(cfg,'fal_key','')) else 'no'}", flush=True)
    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt=PROMPT, fmt="documentary", duration="1-2",
                  theme="modern", captions=True, background="auto", extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script()
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    mg = [{"index": i, "role": s["role"], "graphic_kind": s["graphic_kind"],
           "intensity": s["intensity"], "narration": s["narration"],
           "emphasis": "", "assets": {}} for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.45)
    print("DIRECTOR_FIRED " + json.dumps([(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
