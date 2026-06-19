#!/usr/bin/env python3
"""The Road to War — Batch-10 validation sample exercising the THREE new V2.1
motion-graphics primitives (map_region_highlight, cause_effect_chain,
spectrum_meter) alongside stable ones (name_reveal portrait, chronology_timeline).

Original script (not copied from any reference). ~11 scenes (~2.5 min). 1930s
appeasement fits all three: a highlighted territory (the Rhineland), a causal chain
(how appeasement failed step by step), and a qualitative threat gauge. All five
graphic scenes are DIFFERENT families (portraits / maps / diagrams / meters /
timelines) so the same-family guard never engages; beats spaced >=2 apart.
Collision-free graphic_kinds (territory / causation / threat_level) keep the legacy
footage.py map_region/cause_effect card builders out of the way.

  python3 tools/appeasement_b10.py            # author script.json + FREE dry-run
  python3 tools/appeasement_b10.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Road to War: How Appeasement Failed to Stop Hitler"
NICHE = "history"
OUTDIR = ROOT / "research/motion_graphics_qa/appeasement_batch10"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the 1930s, the leaders of Europe believed that giving a dictator what he "
     "wanted would keep the peace. They were catastrophically wrong.",
     4, "hook", "", "", "",
     ["1930s europe newsreel", "1930s diplomacy vintage", "storm clouds over europe"],
     "Grey diplomats descending marble steps under a darkening sky.",
     "catastrophically wrong"),
    ("No one embodied that hope more than the British prime minister who promised "
     "peace for our time — Neville Chamberlain.",
     3, "intro", "name_reveal", "Neville Chamberlain",
     "1869 – 1940 · British Prime Minister",
     ["neville chamberlain portrait", "1930s british politician", "1938 prime minister"],
     "A sepia portrait of a gaunt, formal statesman with an umbrella.",
     "Neville Chamberlain"),
    ("Chamberlain's strategy was simple — meet each demand, and trust that a "
     "dictator's appetite could be satisfied with enough concessions.",
     3, "setup", "", "", "",
     ["1930s diplomacy handshake", "1938 negotiation vintage", "1930s europe meeting"],
     "Two delegations facing each other across a long polished table.",
     "satisfied with enough concessions"),
    ("The testing began at the edges. In 1936, German troops reoccupied the "
     "territory of the Rhineland — and the Allies did nothing.",
     4, "region", "territory", "The Rhineland",
     "pos=0.52,0.42 ; sub=Demilitarized Zone · Reoccupied 1936 ; title=HITLER'S FIRST GAMBLE",
     ["1936 rhineland troops", "1930s german army", "rhineland bridge vintage"],
     "Columns of soldiers crossing a Rhine bridge into open country.",
     "did nothing"),
    ("Two years later at Munich, Chamberlain handed over a slice of Czechoslovakia "
     "and flew home waving a promise of peace.",
     4, "munich", "", "", "",
     ["1938 munich agreement", "chamberlain peace paper 1938", "1938 crowd cheering"],
     "A statesman on an airfield holding a single sheet of paper aloft.",
     "a promise of peace"),
    ("Each concession only fed the next. One broken promise led to another, which "
     "in turn caused the whole structure of peace to collapse.",
     4, "chain", "causation", "HOW APPEASEMENT FAILED",
     "steps=A broken treaty|Each demand is met|The aggressor emboldened|The slide to war",
     ["1938 europe map", "1930s appeasement vintage", "1939 crisis montage"],
     "Hands signing a document as clocks tick on a panelled wall.",
     "led to another"),
    ("By the time the Allies understood what they were facing, the danger had "
     "grown beyond anything diplomacy could contain.",
     4, "realize", "", "", "",
     ["1939 european crisis", "1930s war preparation", "1939 newspaper headlines"],
     "War-room maps bristling with markers under a single lamp.",
     "beyond diplomacy"),
    ("By 1939, every intelligence service in Europe rated the threat level the "
     "same way — severe.",
     5, "threat", "threat_level", "THREAT LEVEL",
     "value=90 ; bands=LOW|GUARDED|ELEVATED|HIGH|SEVERE ; readout=SEVERE ; title=THE 1939 ASSESSMENT",
     ["1939 intelligence vintage", "1930s war room", "1939 military buildup"],
     "An analyst sliding a marker to the far end of a wall chart.",
     "severe"),
    ("The gamble that began at a single river had become a fire no concession "
     "could put out.",
     4, "stakes", "", "", "",
     ["1939 europe at war", "1930s military mobilization", "1939 troops marching"],
     "Endless ranks of soldiers marching past a reviewing stand.",
     "a fire"),
    ("From the first quiet violation to the outbreak of total war took barely "
     "three years.",
     3, "timeline", "timeline", "THE SLIDE TO WAR",
     "events=1936:Rhineland|1938:Munich Pact|Mar 1939:Prague|Sep 1939:War",
     ["1930s timeline montage", "1939 war declaration", "1930s europe archive"],
     "A spread of dated front pages sliding toward a black headline.",
     "barely three years"),
    ("Appeasement did not prevent the war. It taught the aggressor that the world "
     "would not stop him — until it was far too late.",
     4, "outro", "", "", "",
     ["1939 war dusk", "1930s europe ruins", "second world war opening"],
     "A single newspaper blowing across an empty, darkening square.",
     "far too late"),
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
    brief = Brief(title=TITLE, prompt="How appeasement failed to stop Hitler",
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
        if gk in ("timeline", "era_banner", "chronology", "chapter") and (gt or gb):
            m = _re.search(r"events=([^;]+)", gb)
            if m:
                a["events"] = [[p.split(":", 1)[0].strip(),
                                (p.split(":", 1)[1].strip() if ":" in p else "")]
                               for p in m.group(1).split("|") if p.strip()]
        # V2.1 — Batch 10 dry-run adapter (mirror of pipeline.py branches)
        if gk in ("region", "map_region", "territory", "region_highlight",
                  "region_map") and gt:
            a["region"] = gt
            m = _re.search(r"pos=([^;|]+)", gb)
            if m:
                a["pos"] = m.group(1).strip()
            m = _re.search(r"sub=([^;|]+)", gb)
            if m:
                a["sub"] = m.group(1).strip()
            m = _re.search(r"title=([^;|]+)", gb)
            if m:
                a["title"] = m.group(1).strip()
        if gk in ("cause_effect", "causation", "cause_chain", "domino",
                  "chain_reaction") and (gt or gb):
            m = _re.search(r"steps=([^;]+)", gb)
            if m:
                a["steps"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
            if gt:
                a["title"] = gt
        if gk in ("gauge", "meter", "threat_level", "severity", "rating") and (gt or gb):
            m = _re.search(r"value=([\d.,]+)", gb)
            if m:
                a["value"] = float(m.group(1).replace(",", ""))
            m = _re.search(r"bands=([^;]+)", gb)
            if m:
                a["bands"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
            m = _re.search(r"readout=([^;|]+)", gb)
            if m:
                a["readout"] = m.group(1).strip()
            m = _re.search(r"title=([^;|]+)", gb)
            if m:
                a["title"] = m.group(1).strip()
            if gt:
                a["label"] = gt
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"map_region_highlight", "cause_effect_chain", "spectrum_meter"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-10 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
