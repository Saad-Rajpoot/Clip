#!/usr/bin/env python3
"""REAL-PIPELINE QA RENDER — "How a Simple Chemical Reaction Prevents Corrosion".

Section-C SCIENCE sample through the SHARED Vidlore pipeline (mirrors
tools/_secC_tech_render.py). Exercises NEW V2.8 primitives that suit science —
measurement_callout, footage_fact_overlay, footage_object_callout — alongside
the engine's existing explainer cards, while staying FOOTAGE-FIRST and avoiding
the classroom-infographic / clip-art look. AI VIDEO OFF; FAL stills capped 0.
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

TITLE = "How a Simple Chemical Reaction Prevents Corrosion"
NICHE = "science"
PROMPT = ("How a simple, self-healing chemical reaction quietly protects metal "
          "from rust and keeps bridges, ships and pipelines alive for decades")

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SCENES = [
    ("Every piece of steel in the world is locked in a slow, silent war with the "
     "air around it.", 4, "hook", "", "", ""),
    ("Left unprotected, iron reacts with oxygen and water, and within years a "
     "solid beam can crumble into flakes of rust.", 4, "problem", "", "", ""),
    # footage_fact_overlay — the scale of the problem
    ("Corrosion quietly destroys more than 2 trillion dollars of infrastructure "
     "every single year.", 3, "scale", "fact_overlay", "", ""),
    ("But one elegant trick lets engineers stop that reaction almost completely.",
     3, "promise", "", "", ""),
    # measurement_callout — the protective layer is astonishingly thin
    ("The secret is a protective film just 5 nanometres thick that forms on the "
     "metal's surface.", 3, "detail", "measurement", "5 nanometres", ""),
    ("When the right metal meets oxygen, it doesn't keep rusting — it seals "
     "itself.", 3, "mechanism", "", "", ""),
    ("Aluminium is the quiet hero here: the instant it's exposed, it grows its "
     "own glassy shield.", 3, "hero", "", "", ""),
    # footage_object_callout — point at the oxide layer on the surface
    ("Notice the oxide layer on this dull grey aluminium — that microscopic film "
     "is what stands between the metal and total decay.", 3, "callout",
     "object_callout", "", ""),
    ("Engineers learned to force this reaction on purpose, thickening the shield "
     "in a bath of acid and current.", 4, "process", "", "", ""),
    ("The same principle protects ships' hulls, where a block of cheaper metal is "
     "sacrificed so the steel survives.", 4, "sacrifice", "", "", ""),
    ("Drop by drop, the sacrificial metal dissolves instead of the hull.", 3,
     "detail2", "", "", ""),
    ("It is chemistry working as a bodyguard, taking the damage so the structure "
     "never has to.", 3, "metaphor", "", "", ""),
    ("From the bridge you drove over this morning to the can in your fridge, this "
     "reaction is everywhere and almost invisible.", 4, "ubiquity", "", "", ""),
    ("A simple reaction, understood and tamed, quietly buys our built world "
     "decades of extra life.", 4, "thesis", "statement", "BORROWED TIME", ""),
]


def _kw(nar, gt):
    kws = []
    if gt and not gt.isupper():
        kws.append(gt.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    _stop = {"through", "around", "across", "before", "every", "their", "would",
             "within", "almost", "between", "instead", "really", "another"}
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
    brief = Brief(title=TITLE, prompt=PROMPT, fmt="documentary", duration="6-8",
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
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.40)
    print("DIRECTOR_FIRED " + json.dumps([(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
