#!/usr/bin/env python3
"""Standard Oil — Batch-4 validation sample exercising the THREE new V1.5
motion-graphics primitives (proportion_ring, process_flow_steps,
org_hierarchy_tree) alongside stable ones, in a business topic.

Original script (not copied from any reference). ~13 scenes (~2.5 min); graphic
beats spaced so the two node-diagram primitives are >2 scenes apart.

  python3 tools/standard_oil_b4.py            # author script.json + FREE dry-run
  python3 tools/standard_oil_b4.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "Standard Oil: Anatomy of a Monopoly"
NICHE = "business"
OUTDIR = ROOT / "research/motion_graphics_qa/standard_oil_batch4"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("By the turn of the century, one company had come to control the lifeblood "
     "of the modern industrial world: oil.",
     4, "hook", "", "", "",
     ["vintage oil refinery 1900", "industrial smokestacks archival", "oil barrels warehouse"],
     "A vast 1900s oil refinery at dusk, smokestacks and iron tanks silhouetted.",
     "oil"),
    ("It was built by one relentless, secretive man — John D. Rockefeller.",
     3, "intro", "name_reveal", "John D. Rockefeller",
     "1839 – 1937 · The Titan of Oil",
     ["John D Rockefeller portrait", "1800s industrialist photograph", "victorian businessman portrait"],
     "Formal seated portrait of a stern Gilded-Age industrialist, sepia photograph.",
     "John D. Rockefeller"),
    ("He began with a single refinery in Cleveland — and a conviction that "
     "chaos in the oil trade could be tamed by total control.",
     3, "origin", "", "", "",
     ["1870s oil refinery interior", "vintage cleveland industry", "old industrial machinery"],
     "A cramped early refinery lit by furnace glow, workers among copper stills.",
     "total control"),
    ("Within twenty years, Standard Oil controlled an astonishing ninety percent "
     "of all oil refining in America.",
     4, "scale", "share", "90",
     "label=of all U.S. oil refining ; sub=Standard Oil · 1890",
     ["oil refinery aerial vintage", "industrial empire archival", "1890s oil tanks"],
     "An endless field of oil storage tanks stretching to the horizon.",
     "ninety percent"),
    ("How did one company swallow an entire industry? It followed a ruthless, "
     "repeatable playbook.",
     3, "method", "", "", "",
     ["vintage business ledger", "1800s boardroom", "old contract documents"],
     "A dim boardroom, a ledger open under a banker's lamp.",
     "playbook"),
    ("First, buy or build the refinery. Then undercut on price until the rival "
     "bled. Then starve them of railroads and barrels. Then absorb what remained.",
     4, "scheme", "process", "THE STANDARD OIL PLAYBOOK",
     "steps=Buy the refinery|Undercut on price|Starve the rival|Absorb it",
     ["vintage railroad oil tankers", "1800s price war newspaper", "industrial takeover"],
     "Freight cars of oil rolling past a shuttered competitor's gate.",
     "absorb"),
    ("Rivals who refused to sell found their costs mysteriously rising and their "
     "customers quietly vanishing.",
     3, "pressure", "", "", "",
     ["abandoned factory vintage", "1800s bankruptcy", "empty industrial yard"],
     "A padlocked refinery gate, rust on the chains, weeds in the yard.",
     "vanishing"),
    ("It all radiated outward from where it began — Cleveland, Ohio.",
     3, "place", "location", "Cleveland, Ohio",
     "The First Refinery · 1870",
     ["cleveland ohio vintage", "1800s industrial city", "old american city skyline"],
     "A period map vignette of Cleveland on Lake Erie, smoke on the lakefront.",
     "Cleveland"),
    ("To hide the scale of its control, Standard Oil hid behind a new legal "
     "machine — the Trust.",
     4, "structure", "", "", "",
     ["vintage legal documents", "1800s certificate archival", "old corporate seal"],
     "An ornate 1882 trust certificate under glass, wax seal gleaming.",
     "the Trust"),
    ("On paper, dozens of separate companies. In reality, one hand at the top "
     "moved them all.",
     3, "reveal", "hierarchy", "Standard Oil Trust",
     "children=Standard of Ohio|Standard of New Jersey|Standard of New York ; title=THE TRUST · 1882",
     ["puppet strings vintage", "1800s many hands", "industrial control room"],
     "Many small flames bending toward a single unseen draft.",
     "one hand"),
    ("The backlash was inevitable. In 1911, the Supreme Court ordered the giant "
     "shattered into thirty-four pieces.",
     4, "fall", "", "", "",
     ["supreme court vintage", "1911 newspaper headline", "broken chain symbolic"],
     "A newspaper front page splitting under a hard light, 1911.",
     "shattered"),
    ("Yet from those fragments grew the oil giants we still know today — the "
     "monopoly broken, but never truly undone.",
     3, "legacy", "", "", "",
     ["modern oil company vintage", "gas station americana", "oil industry legacy"],
     "Dawn over a modern refinery built on century-old foundations.",
     "legacy"),
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
    brief = Brief(title=TITLE, prompt="The rise and breakup of the Standard Oil monopoly",
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
        plain = "" if any(x in gb for x in ("=",)) else gb
        if gk == "name_reveal" and gt:
            a["portrait_path"] = "/tmp/_d.png"; a["name"] = gt
        if gk in ("share", "proportion", "percent") and (gt or gb):
            m = _re.search(r"share=([\d.]+)", gb) or _re.search(r"([\d.]+)", gt)
            if m:
                a["share"] = float(m.group(1))
            ml = _re.search(r"label=([^;|]+)", gb)
            if ml:
                a["label"] = ml.group(1).strip()
            ms = _re.search(r"sub=([^;|]+)", gb)
            if ms:
                a["center_sub"] = ms.group(1).strip()
        if gk in ("process", "method", "flow", "steps") and (gt or gb):
            m = _re.search(r"steps=([^;]+)", gb)
            if m:
                a["steps"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
            if gt:
                a["title"] = gt
        if gk in ("hierarchy", "org_chart", "structure") and gt:
            a["root"] = gt
            m = _re.search(r"children=([^;]+)", gb)
            if m:
                a["children"] = [c.strip() for c in m.group(1).split("|") if c.strip()]
            mt = _re.search(r"title=([^;|]+)", gb)
            if mt:
                a["title"] = mt.group(1).strip()
        if gk in ("location", "establish") and gt:
            a["place"] = gt
            a["sub"] = plain
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"proportion_ring", "process_flow_steps", "org_hierarchy_tree"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-4 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
