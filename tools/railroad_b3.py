#!/usr/bin/env python3
"""Transcontinental Railroad — Batch-3 validation sample exercising the THREE
new V1.4 motion-graphics primitives (growth_curve_chart, map_route_spread,
annotated_detail_callout) alongside stable ones, in a fresh history topic.

Original script (not copied from any reference). ~13 scenes (~2.5 min); graphic
beats spaced so no two are adjacent.

  python3 tools/railroad_b3.py            # author script.json + FREE dry-run
  python3 tools/railroad_b3.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Transcontinental Railroad: Binding a Continent"
NICHE = "history"
OUTDIR = ROOT / "research/motion_graphics_qa/railroad_batch3"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("A journey across America could once take six brutal months — by wagon, "
     "by ship, by sheer endurance.",
     4, "hook", "", "", "",
     ["covered wagon prairie", "1800s pioneers trail", "vintage map america west"],
     "A lone covered wagon crossing an endless dusty prairie at golden hour, period photograph.",
     "six months"),
    ("One engineer believed he could change that. Theodore Judah saw a railroad "
     "where everyone else saw an impassable wall of granite.",
     3, "intro", "", "", "",
     ["victorian engineer portrait", "1800s railroad surveyor", "vintage man portrait sepia"],
     "Formal seated portrait of a determined Victorian engineer, sepia photograph.",
     "Theodore Judah"),
    ("His plan was audacious: a single iron line stretching across two thousand "
     "miles of plains, desert, and mountain.",
     3, "plan", "route", "THE PACIFIC RAILROAD · 1869",
     "stops=Sacramento:0.16,0.54|Promontory:0.40,0.50|Cheyenne:0.58,0.46|Omaha:0.74,0.44",
     ["antique map american west", "old railroad route map", "vintage cartography"],
     "An antique map of the American West with a rail line inked across it.",
     "two thousand miles"),
    ("Two companies took up the challenge — the Central Pacific pushing east, the "
     "Union Pacific driving west, racing toward each other.",
     3, "build", "", "", "",
     ["1800s railroad construction", "transcontinental railroad workers", "steam locomotive vintage"],
     "Crews of labourers laying heavy iron rails across open country, archival photograph.",
     "racing"),
    ("And the pace grew relentless. Year after year the miles of track climbed, "
     "then surged, then exploded across the map.",
     3, "data", "growth", "MILES OF TRACK LAID",
     "points=1863:30|1865:120|1867:300|1869:560",
     ["railroad tracks vanishing point", "iron rails construction", "vintage railway"],
     "Freshly laid iron rails stretching to the horizon, low sun glinting on steel.",
     "surged"),
    ("Through the High Sierra, workers blasted tunnels through solid granite — "
     "inch by inch, by hand and black powder.",
     4, "obstacle", "", "", "",
     ["mountain tunnel granite", "1800s mining dynamite", "sierra nevada cliffs"],
     "Workers chiselling a tunnel mouth in a sheer granite cliff, dramatic shadow.",
     "granite"),
    ("On May tenth, 1869, the two lines finally met — a moment captured in one of "
     "the most famous photographs of the century.",
     4, "climax", "detail", "East and West Shaking Hands",
     "focus=0.5,0.43 ; tag=MAY 10, 1869",
     ["golden spike ceremony 1869", "promontory summit photograph", "vintage railroad crowd"],
     "The famous Promontory Summit gathering, two locomotives nose to nose, crowd of workers.",
     "finally met"),
    ("Two locomotives touched cowcatchers as a golden spike was driven into the "
     "final tie.",
     3, "moment", "", "", "",
     ["golden spike railroad", "vintage steam locomotive front", "1869 ceremony"],
     "A ceremonial golden spike held above the final railroad tie, hands reaching in.",
     "golden spike"),
    ("It happened here — a windswept patch of Utah desert that, for one day, "
     "became the centre of the nation.",
     3, "place", "location", "Promontory Summit, Utah",
     "The Golden Spike · 1869",
     ["utah desert landscape", "promontory summit", "vintage american west"],
     "A windswept high-desert basin under a vast sky, distant mountains.",
     "Promontory Summit"),
    ("The news flashed across telegraph wires to a waiting country, and church "
     "bells rang from coast to coast.",
     3, "react", "", "", "",
     ["telegraph wires vintage", "1800s celebration crowd", "old church bell"],
     "Telegraph key clattering under lamplight, sparks of connection.",
     "coast to coast"),
    ("A journey that once swallowed six months could now be made in a single "
     "week. A continent had been bound together in iron.",
     4, "payoff", "", "", "",
     ["transcontinental railroad sunset", "steam train prairie", "vintage railway journey"],
     "A steam train racing across golden plains into the sunset, smoke trailing.",
     "a single week"),
    ("The frontier that had defined America was closing. The modern nation had "
     "arrived — on rails of iron and ambition.",
     3, "outro", "", "", "",
     ["steam train vintage americana", "railroad horizon", "1800s industrial america"],
     "A lone locomotive silhouetted against a burning horizon, end-of-era mood.",
     "modern nation"),
]


def build_script() -> dict:
    scenes = []
    for i, (nar, inten, role, gk, gt, gb, kws, vis, emph) in enumerate(SCENES):
        scenes.append({
            "narration": nar, "keywords": kws, "visual": vis,
            "intensity": inten, "emphasis": emph,
            "shot_type": "wide" if i % 3 == 0 else "medium",
            "role": role, "graphic_kind": gk, "graphic_text": gt,
            "graphic_body": gb,
        })
    body = f"{TITLE}\n\n" + "\n\n".join(s["narration"] for s in scenes)
    return {"title": TITLE,
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for

    cfg = load_config()
    brief = Brief(title=TITLE, prompt="The building of the transcontinental railroad",
                  fmt="documentary", duration="6-8", theme="history",
                  captions=False, background="auto", extra={"niche": NICHE})
    out = ROOT / "output"
    run_dir = run_dir_for(brief, out)
    run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script()
    body = f"{TITLE}\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (OUTDIR / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    print(f"RUN_DIR {run_dir}", flush=True)

    # FREE dry-run via the real pipeline-equivalent director plan
    from vidlore.motion_graphics import director as mgdir, registry as mgreg
    import re as _re
    mg = []
    for i, sc in enumerate(script["scenes"]):
        a = {}
        gk = (sc["graphic_kind"] or "").lower()
        gt, gb = sc["graphic_text"] or "", sc["graphic_body"] or ""
        if gk in ("name_reveal",) and gt:
            a["portrait_path"] = "/tmp/_d.png"; a["name"] = gt
        if gk in ("growth", "trend", "line_chart") and (gt or gb):
            m = _re.search(r"points=([^;]+)", gb)
            if m:
                a["points"] = [[p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                               for p in m.group(1).split("|") if ":" in p]
            if gt:
                a["title"] = gt
        if gk in ("route", "journey", "expansion", "map_route") and (gt or gb):
            m = _re.search(r"stops=([^;]+)", gb)
            if m:
                a["stops"] = [[p.split(":", 1)[0].strip(),
                               (p.split(":", 1)[1].strip() if ":" in p else None)]
                              for p in m.group(1).split("|") if p.strip()]
            if gt:
                a["title"] = gt
        if gk in ("detail", "annotate") and gt:
            a["label"] = gt
            mf = _re.search(r"focus=([\d.]+\s*,\s*[\d.]+)", gb)
            if mf:
                a["focus"] = mf.group(1).replace(" ", "")
            a["image_path"] = "/tmp/_d.png"
        if gk in ("location", "establish") and gt:
            a["place"] = gt
            a["sub"] = gb if "=" not in gb else ""
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"growth_curve_chart", "map_route_spread", "annotated_detail_callout"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-3 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
          flush=True)
    print(f"total primitives: {len(mgreg.all_ids())}", flush=True)

    if "--render" in sys.argv:
        from vidlore.pipeline import render_from_script
        t0 = time.time()
        render_from_script(brief, cfg, out, keep_work=True)
        vid = run_dir / f"{run_dir.name}.mp4"
        print(f"\nRENDER_DONE wall={time.time()-t0:.1f}s video={vid.exists()} "
              f"path={vid}", flush=True)


if __name__ == "__main__":
    main()
