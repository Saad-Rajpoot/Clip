#!/usr/bin/env python3
"""The Industrial Revolution — Batch-6 validation sample exercising the THREE new
V1.7 motion-graphics primitives (definition_card, vs_balance_scale,
before_after_slider) alongside stable ones.

Original script (not copied from any reference). ~12 scenes (~2.5 min); graphic
beats spaced so no two are adjacent and the families never clash.

  python3 tools/industrial_rev_b6.py            # author script.json + FREE dry-run
  python3 tools/industrial_rev_b6.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Industrial Revolution: The Machine That Remade the World"
NICHE = "history"
OUTDIR = ROOT / "research/motion_graphics_qa/industrial_rev_batch6"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("For ten thousand years, human life had moved at the speed of muscle, wind, "
     "and water. Then, in a single century, everything changed.",
     4, "hook", "", "", "",
     ["misty english countryside", "water wheel mill vintage", "preindustrial village"],
     "A quiet pre-industrial valley at dawn, a single watermill turning.",
     "everything changed"),
    ("At its heart stood an inventor whose engine would set the world in motion — "
     "James Watt.",
     3, "intro", "name_reveal", "James Watt",
     "1736 – 1819 · Father of the Steam Age",
     ["james watt portrait", "1700s inventor engraving", "georgian gentleman portrait"],
     "An engraved portrait of a stern Georgian engineer, candlelit.",
     "James Watt"),
    ("His steam engine unleashed a force the world had never organised before — a "
     "force we now call by a single word.",
     3, "concept", "", "", "",
     ["steam engine vintage", "industrial machinery 1800s", "factory steam power"],
     "Pistons and brass valves hissing with steam, lit by furnace glow.",
     "a single word"),
    ("Industrialisation: the wholesale shift from hand production in homes to "
     "powered machines in factories.",
     4, "define", "definition", "Industrialisation",
     "pos=noun ; definition=the shift from hand production in the home to powered machinery in the factory.",
     ["textile factory machines", "industrial loom vintage", "1800s factory floor"],
     "Rows of mechanical looms thundering in a vast brick mill.",
     "Industrialisation"),
    ("It built unimaginable wealth. But that wealth was split with brutal "
     "unfairness between those who owned the machines and those who ran them.",
     4, "wealth", "", "", "",
     ["industrial wealth contrast", "factory owner mansion", "wealth inequality 1800s"],
     "A grand mill-owner's house looming over rows of cramped worker cottages.",
     "brutal unfairness"),
    ("A new balance of power had been struck — and it tipped hard toward the "
     "owners of capital.",
     4, "power", "balance", "A NEW BALANCE OF POWER",
     "pair=Factory Owners|Workers ; values=78|22",
     ["factory owner vs worker", "industrial class divide", "1800s labour"],
     "Two worlds on a scale: a top hat on one side, a worker's cap on the other.",
     "tipped"),
    ("Cities exploded. Within a lifetime, quiet market towns became roaring "
     "engines of smoke and iron.",
     4, "growth", "", "", "",
     ["industrial city smoke", "victorian factory chimneys", "1800s manchester"],
     "A skyline of smokestacks belching black smoke over crowded streets.",
     "exploded"),
    ("The transformation was total. The same valley, fifty years apart, was "
     "almost unrecognisable.",
     4, "transform", "before_after", "The same valley, fifty years apart",
     "before=1750 ; after=1800",
     ["before after landscape", "rural to industrial", "valley transformation"],
     "A green valley dissolving into a forest of factory chimneys.",
     "unrecognisable"),
    ("Nowhere felt this more than the world's first true industrial city — a "
     "place that became a symbol of the age.",
     3, "place_intro", "", "", "",
     ["manchester industrial vintage", "victorian city streets", "industrial england"],
     "Cobbled streets slick with rain beneath towering mill walls.",
     "symbol"),
    ("Manchester. Cottonopolis. The shock city of the industrial world.",
     3, "place", "location", "Manchester, England",
     "Cottonopolis · 1800s",
     ["manchester england vintage", "1800s industrial city", "cotton mills"],
     "A period map vignette of Manchester ringed by mills and canals.",
     "Manchester"),
    ("It was filthy, brutal, and astonishing — the blueprint for every modern "
     "city that followed.",
     3, "react", "", "", "",
     ["industrial pollution vintage", "victorian slum", "1800s city life"],
     "Crowded smoke-darkened streets, washing strung between tenements.",
     "blueprint"),
    ("The machine had remade the world. And there was no going back to the quiet "
     "valley ever again.",
     4, "outro", "", "", "",
     ["industrial sunset chimneys", "steam train vintage", "modern world dawn"],
     "A steam train racing into a smoky dusk, the old world behind it.",
     "no going back"),
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
    brief = Brief(title=TITLE, prompt="How the industrial revolution remade the world",
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

    from vidlore.motion_graphics import director as mgdir, registry as mgreg
    import re as _re
    mg = []
    for i, sc in enumerate(script["scenes"]):
        a = {}
        gk = (sc["graphic_kind"] or "").lower()
        gt, gb = sc["graphic_text"] or "", sc["graphic_body"] or ""
        if gk == "name_reveal" and gt:
            a["portrait_path"] = "/tmp/_d.png"; a["name"] = gt
        if gk in ("definition", "define", "term") and gt:
            a["term"] = gt
            m = _re.search(r"definition=([^;|]+)", gb)
            if m:
                a["definition"] = m.group(1).strip()
            m = _re.search(r"pos=([^;|]+)", gb)
            if m:
                a["pos"] = m.group(1).strip()
        if gk in ("balance", "tradeoff", "tension", "scales") and (gt or gb):
            m = _re.search(r"pair=([^;|]+\|[^;]+)", gb)
            if m:
                lr = m.group(1).split("|"); a["left"], a["right"] = lr[0].strip(), lr[1].strip()
            m = _re.search(r"values=([\d.]+)\|([\d.]+)", gb)
            if m:
                a["leftval"], a["rightval"] = float(m.group(1)), float(m.group(2))
            if gt:
                a["title"] = gt
        if gk in ("before_after", "transformation", "then_now") and (gt or gb):
            m = _re.search(r"before=([^;|]+)", gb)
            a["before_label"] = m.group(1).strip() if m else "BEFORE"
            m = _re.search(r"after=([^;|]+)", gb)
            a["after_label"] = m.group(1).strip() if m else "AFTER"
            if gt:
                a["caption"] = gt
            a["image_path"] = "/tmp/_d.png"
        if gk in ("location", "establish") and gt:
            a["place"] = gt
            a["sub"] = "" if "=" in gb else gb
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"definition_card", "vs_balance_scale", "before_after_slider"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-6 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
