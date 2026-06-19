#!/usr/bin/env python3
"""The Robber Barons — Batch-8 validation sample exercising the THREE new V1.9
motion-graphics primitives (ranked_list_countdown, sankey_flow, era_band_timeline)
alongside stable ones (name_reveal portrait, location).

Original script (not copied from any reference). ~12 scenes (~2.5 min). The Gilded
Age fits all three: a ranking of the great fortunes (leaderboard), how a trust's
revenue split (sankey money flow), and the ages of American industry (era bands).
Graphic beats are spaced so no two are adjacent AND the two charts-family cards
(ranked + sankey) sit 4 scenes apart (the same-family guard wants >=3).

  python3 tools/robber_barons_b8.py            # author script.json + FREE dry-run
  python3 tools/robber_barons_b8.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Robber Barons: The Men Who Built and Broke the Gilded Age"
NICHE = "business"
OUTDIR = ROOT / "research/motion_graphics_qa/robber_barons_batch8"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the half-century after the Civil War, a handful of men amassed fortunes "
     "so vast they could bend the whole American economy to their will.",
     4, "hook", "", "", "",
     ["gilded age mansion vintage", "1900 wall street", "industrial america 1880s"],
     "A marble robber-baron mansion looming over a smoky industrial skyline.",
     "bend the whole economy"),
    ("None loomed larger than the man who turned oil into an empire — John D. "
     "Rockefeller.",
     3, "intro", "name_reveal", "John D. Rockefeller",
     "1839 – 1937 · Founder of Standard Oil",
     ["john rockefeller portrait", "1900s industrialist portrait", "gilded age tycoon"],
     "A severe sepia portrait of a thin, watchful oil magnate.",
     "John D. Rockefeller"),
    ("But he was one of several titans, and their wealth was staggering even "
     "ranked side by side.",
     3, "wealth", "", "", "",
     ["stacks of gold coins vintage", "1900 bank vault", "gilded age wealth"],
     "Ledgers and gold stacked on a banker's desk under a green lamp.",
     "staggering"),
    ("Ranked from largest to smallest, the great fortunes of 1900 dwarfed anything "
     "the country had ever seen.",
     4, "ranking", "leaderboard", "THE GREAT FORTUNES · 1900",
     "items=Standard Oil:900|Carnegie Steel:480|Vanderbilt Rail:230|Astor Estates:110|Gould Rail:77 ; prefix=$ ; suffix=M",
     ["gilded age tycoons", "1900 wealth comparison", "industrial fortunes vintage"],
     "Engraved portraits of rival magnates arranged like a hall of power.",
     "dwarfed anything"),
    ("This wealth was no accident. It rode a wave of industry that had remade the "
     "nation in distinct ages.",
     3, "transition", "", "", "",
     ["industrial revolution montage", "steam factory 1800s", "american industry vintage"],
     "Factory wheels and steam engines turning in a dim mill.",
     "distinct ages"),
    ("American industry moved through four great ages of power, each one building "
     "on the last.",
     4, "eras", "eras", "THE AGES OF AMERICAN INDUSTRY",
     "eras=Canals:1790-1840|Railroads:1840-1890|Steel & Oil:1890-1920|Autos:1920-1950",
     ["canal barge 1800s", "steam locomotive vintage", "1920s assembly line"],
     "A sweep from canal barges to locomotives to an auto assembly line.",
     "four great ages"),
    ("At the center of the oil age sat one machine for making money — and every "
     "dollar that flowed in was ruthlessly divided.",
     4, "transition2", "", "", "",
     ["oil refinery 1900 vintage", "standard oil refinery", "industrial pipelines 1900"],
     "Pipes and valves hissing in a vast refinery at dusk.",
     "ruthlessly divided"),
    ("For every dollar of revenue, the Standard Oil trust funnelled it into three "
     "tightly controlled arms.",
     4, "money_split", "money_split", "REVENUE",
     "branches=Refining:62|Pipelines:23|Distribution:15 ; title=WHERE EVERY DOLLAR WENT ; prefix=$ ; suffix=M",
     ["standard oil refinery vintage", "1900 oil barrels", "industrial money flow"],
     "Barrels rolling down a line as clerks tally figures in ledgers.",
     "every dollar"),
    ("That iron grip began in one unremarkable city, where a young bookkeeper "
     "first saw the future in oil.",
     3, "place_intro", "", "", "",
     ["cleveland ohio 1870 vintage", "1870s american city", "industrial cleveland"],
     "Cobbled streets and brick warehouses beside a smoky river.",
     "the future in oil"),
    ("Cleveland, Ohio. The quiet birthplace of the largest monopoly the world had "
     "ever known.",
     3, "place", "location", "Cleveland, Ohio",
     "Birthplace of Standard Oil · 1870",
     ["cleveland ohio map vintage", "1870 ohio city", "industrial river city"],
     "A period map vignette of Cleveland ringed by refineries.",
     "Cleveland"),
    ("By the time the courts finally broke the trust apart, the age of the robber "
     "barons had already remade America for good.",
     4, "outro", "", "", "",
     ["1911 supreme court vintage", "standard oil breakup", "gilded age end"],
     "Newspaper presses printing the headline of the great breakup.",
     "remade America"),
    ("Their mansions still stand — monuments to an era when a few men held the "
     "wealth of a nation in their hands.",
     3, "close", "", "", "",
     ["gilded age mansion interior", "newport mansion vintage", "industrial legacy"],
     "An empty gilded ballroom, dust drifting in a shaft of light.",
     "the wealth of a nation"),
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
    brief = Brief(title=TITLE, prompt="How the robber barons built and broke the Gilded Age",
                  fmt="documentary", duration="6-8", theme="standard",
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

    from vidlore.motion_graphics import director as mgdir, registry as mgreg
    import re as _re
    mg = []
    for i, sc in enumerate(script["scenes"]):
        a = {}
        gk = (sc["graphic_kind"] or "").lower()
        gt, gb = sc["graphic_text"] or "", sc["graphic_body"] or ""
        if gk == "name_reveal" and gt:
            a["portrait_path"] = "/tmp/_d.png"; a["name"] = gt
        if gk in ("location", "establish") and gt:
            a["place"] = gt
            a["sub"] = "" if "=" in gb else gb
        # V1.9 — Batch 8 dry-run adapter (mirror of pipeline.py branches)
        if gk in ("ranking", "leaderboard", "top_list", "ranked",
                  "top_five", "top_ten", "countdown_list") and (gt or gb):
            m = _re.search(r"items=([^;]+)", gb)
            if m:
                items = []
                for p in m.group(1).split("|"):
                    if ":" in p:
                        nm, vv = p.rsplit(":", 1)
                        try:
                            items.append([nm.strip(), float(_re.sub(r"[^\d.\-]", "", vv))])
                        except ValueError:
                            pass
                if items:
                    a["items"] = items
            if gt:
                a["title"] = gt
            m = _re.search(r"prefix=([^;|]+)", gb)
            if m:
                a["prefix"] = m.group(1).strip()
            m = _re.search(r"suffix=([^;|]+)", gb)
            if m:
                a["suffix"] = m.group(1).strip()
        if gk in ("sankey", "money_split", "allocation", "flow_breakdown",
                  "where_it_goes") and (gt or gb):
            m = _re.search(r"branches=([^;]+)", gb)
            if m:
                brs = []
                for p in m.group(1).split("|"):
                    if ":" in p:
                        nm, vv = p.rsplit(":", 1)
                        try:
                            brs.append([nm.strip(), float(_re.sub(r"[^\d.\-]", "", vv))])
                        except ValueError:
                            pass
                if brs:
                    a["branches"] = brs
            m = _re.search(r"source=([^;|]+)", gb)
            if m:
                a["source"] = m.group(1).strip()
            elif gt:
                a["source"] = gt
            m = _re.search(r"title=([^;|]+)", gb)
            if m:
                a["title"] = m.group(1).strip()
            m = _re.search(r"prefix=([^;|]+)", gb)
            if m:
                a["prefix"] = m.group(1).strip()
            m = _re.search(r"suffix=([^;|]+)", gb)
            if m:
                a["suffix"] = m.group(1).strip()
        if gk in ("eras", "ages", "era_bands", "periods", "epochs") and (gt or gb):
            m = _re.search(r"eras=([^;]+)", gb)
            if m:
                ers = []
                for p in m.group(1).split("|"):
                    if ":" in p:
                        nm, rg = p.split(":", 1)
                        rg = rg.replace("—", "-").replace("–", "-")
                        pp = [x for x in rg.split("-") if x.strip()]
                        if len(pp) >= 2:
                            ers.append([nm.strip(), pp[0].strip(), pp[1].strip()])
                        elif nm.strip():
                            ers.append([nm.strip(), rg.strip()])
                if ers:
                    a["eras"] = ers
            if gt:
                a["title"] = gt
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"ranked_list_countdown", "sankey_flow", "era_band_timeline"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-8 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
