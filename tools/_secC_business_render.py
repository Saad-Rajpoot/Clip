#!/usr/bin/env python3
"""REAL-PIPELINE QA RENDER — "How an Oil Monopoly Controlled Railroads and Refineries".

Section-C BUSINESS sample through the SHARED Vidlore pipeline (mirrors
tools/_secC_tech_render.py). Exercises NEW V2.8 business/systems primitives —
acquisition_timeline, supply_chain_network, system_planview_flow,
footage_fact_overlay — staying FOOTAGE-FIRST and AVOIDING the all-at-once
spreadsheet / pitch-deck dashboard look. AI VIDEO OFF; FAL stills capped 0.
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

TITLE = "How an Oil Monopoly Controlled Railroads and Refineries"
NICHE = "business"
PROMPT = ("How a single oil empire used secret railroad deals and relentless "
          "acquisitions to control refining and crush every competitor")

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SCENES = [
    ("In the years after the Civil War, oil was chaos — hundreds of tiny "
     "refiners, all fighting to survive.", 4, "hook", "", "", ""),
    ("One company looked at that chaos and saw a machine it could build.", 3,
     "setup", "", "", ""),
    # system_planview_flow — the value chain it intended to own
    ("Its plan was to control every step: from the wells, to the railroads, to "
     "the refineries, to the barrels on the dock.", 3, "system", "architecture",
     "THE VALUE CHAIN", ""),
    # supply_chain_network — crude flowing through the controlled network
    ("Crude flowed from the oil fields, onto the rail lines it secretly "
     "controlled, and into the refineries it already owned.", 4, "supply",
     "supply_chain", "THE FLOW OF CRUDE", ""),
    # footage_fact_overlay — the scale of control
    ("Within a decade, the company controlled more than 90 percent of American "
     "refining.", 3, "scale", "fact_overlay", "", ""),
    ("The weapon behind it all was the railroads.", 4, "weapon", "", "", ""),
    ("It cut secret deals: lower shipping rates for itself, punishing rates for "
     "everyone else.", 4, "deal", "", "", ""),
    ("A rival shipping the same oil simply could not match the price — and slowly "
     "went broke.", 4, "squeeze", "", "", ""),
    # acquisition_timeline — the wave of takeovers (named acquirer + targets)
    ("Then came the offers. Standard moved methodically — Bostwick, Pratt and "
     "Warden were bought out, absorbed, or quietly crushed.", 4, "acquire",
     "acquisition", "THE TAKEOVERS", ""),
    ("Some owners sold and grew rich; others held out and lost everything.", 4,
     "split", "", "", ""),
    ("Each refinery that fell made the next one easier to take.", 3, "snowball",
     "", "", ""),
    ("By the 1880s, the chaos was gone — replaced by one quiet, total empire.",
     4, "empire", "", "", ""),
    ("From the lamp oil in a farmhouse to the price at every depot, one company "
     "set the terms.", 4, "reach", "", "", ""),
    ("It was not built on a single invention, but on control of the road every "
     "barrel had to travel.", 4, "thesis", "statement", "CONTROL THE ROAD", ""),
]


def _kw(nar, gt):
    kws = []
    if gt and not gt.isupper():
        kws.append(gt.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    _stop = {"through", "around", "across", "before", "every", "their", "would",
             "within", "almost", "between", "instead", "simply", "slowly",
             "company", "others"}
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
