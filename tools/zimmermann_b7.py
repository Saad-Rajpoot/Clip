#!/usr/bin/env python3
"""The Zimmermann Telegram (1917) — Batch-7 validation sample exercising the THREE
new V1.8 motion-graphics primitives (redacted_document, headline_montage,
map_heat_spread) alongside stable ones (name_reveal, chronology_timeline).

Original script (not copied from any reference). ~12 scenes (~2.5 min). The story
is a natural fit: a secret decoded cable (redacted_document), an eruption across
the American press (headline_montage), and outrage spreading city-to-city across
the U.S. (map_heat_spread). Graphic beats are spaced so no two are adjacent and
no two same-family cards clash.

  python3 tools/zimmermann_b7.py            # author script.json + FREE dry-run
  python3 tools/zimmermann_b7.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Zimmermann Telegram: The Secret Message That Pulled America Into War"
NICHE = "history"
OUTDIR = ROOT / "research/motion_graphics_qa/zimmermann_batch7"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the winter of 1917, a single coded message crossed the Atlantic — and "
     "quietly set the most powerful neutral nation on Earth on the road to war.",
     4, "hook", "", "", "",
     ["1917 telegraph office vintage", "world war one atlantic cable", "wartime europe map"],
     "A dim telegraph office, a clerk hunched over a chattering key at night.",
     "road to war"),
    ("It was sent by Germany's foreign secretary, a career diplomat who believed "
     "he could change the course of the war with a few hundred words — Arthur Zimmermann.",
     3, "intro", "name_reveal", "Arthur Zimmermann",
     "1864 – 1940 · German Foreign Secretary",
     ["arthur zimmermann portrait", "german diplomat 1910s", "edwardian statesman portrait"],
     "A formal sepia portrait of a stern moustached German official.",
     "Arthur Zimmermann"),
    ("His plan was audacious. If the United States entered the war, Germany would "
     "offer Mexico an alliance — and a chance to reclaim lost territory.",
     4, "plan", "", "", "",
     ["1917 germany war room", "old mexico map vintage", "wartime strategy table"],
     "Gloved hands moving markers across a candlelit war-room map.",
     "audacious"),
    ("The cable was coded, secret, and explosive. Decoded, one line laid the whole "
     "scheme bare.",
     5, "reveal", "classified", "We propose an alliance with Mexico against the United States.",
     "title=GERMAN FOREIGN OFFICE — CABLE No. 0158 ; stamp=DECODED",
     ["coded telegram vintage", "cipher decryption 1917", "secret document old paper"],
     "A typed cipher sheet, most of it blacked out, one line burning through.",
     "explosive"),
    ("But Germany had made one fatal mistake. British codebreakers in a secret room "
     "called Room 40 were already listening.",
     4, "twist", "", "", "",
     ["room 40 codebreakers", "british naval intelligence 1917", "cipher room vintage"],
     "A hushed room of analysts bent over intercepts under green lamps.",
     "fatal mistake"),
    ("When the decoded message reached American newspapers, the reaction was "
     "instant. The presses erupted.",
     5, "press", "press", "AMERICA REELS · MARCH 1917",
     "headlines=GERMAN PLOT TO ATTACK U.S. EXPOSED|MEXICO URGED TO JOIN WAR ON AMERICA|THE TELEGRAM THAT MEANS WAR",
     ["1917 newspaper front page", "newsboy headlines vintage", "printing press 1917"],
     "Newsboys waving papers as a crowd surges around a kiosk.",
     "erupted"),
    ("Overnight, a distant European war became a direct threat to American soil. "
     "The mood of a whole country began to turn.",
     4, "shift", "", "", "",
     ["1917 american crowd", "wartime patriotism vintage", "us flag rally 1917"],
     "A street rally, flags raised, faces hard with new resolve.",
     "direct threat"),
    ("From coast to coast, outrage spread through American cities — and with it, the "
     "last resistance to war collapsed.",
     5, "spread", "contagion", "OUTRAGE SPREADS · SPRING 1917",
     "hotspots=New York:0.76,0.36|Washington:0.74,0.44|Chicago:0.60,0.36|San Francisco:0.40,0.42|New Orleans:0.58,0.56",
     ["1917 us cities montage", "wartime america map", "patriotic rally vintage"],
     "A national map glowing as anger catches from city to city.",
     "spread"),
    ("What no German U-boat campaign had managed in two years, a single telegram "
     "achieved in a matter of weeks.",
     4, "weigh", "", "", "",
     ["world war one submarine", "1917 atlantic convoy", "wartime propaganda poster"],
     "A periscope wake cutting across a grey, hostile sea.",
     "a single telegram"),
    ("The path from secret cable to open war was breathtakingly short.",
     4, "timeline", "timeline", "THE ROAD TO WAR",
     "events=Jan 1917:Cable sent|Feb 1917:British decode it|Mar 1917:Press publish it|Apr 1917:U.S. declares war",
     ["1917 timeline montage", "wartime calendar vintage", "us congress 1917"],
     "A spread of dated front pages fanning toward a declaration.",
     "breathtakingly short"),
    ("On the sixth of April, 1917, the United States declared war on Germany. The "
     "neutral giant had been pulled in at last.",
     4, "stakes", "", "", "",
     ["us troops 1917 embark", "world war one american soldiers", "1917 war declaration"],
     "Columns of fresh troops marching toward waiting transport ships.",
     "pulled in"),
    ("A few hundred coded words had rewritten the map of the twentieth century — "
     "and no cipher could ever take them back.",
     4, "outro", "", "", "",
     ["world war one battlefield dusk", "1918 victory parade", "20th century history montage"],
     "A telegraph wire silhouetted against a blood-orange wartime sky.",
     "rewritten the map"),
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
    brief = Brief(title=TITLE, prompt="How the Zimmermann Telegram pulled America into WWI",
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
                a["events"] = [
                    [p.split(":", 1)[0].strip(),
                     (p.split(":", 1)[1].strip() if ":" in p else "")]
                    for p in m.group(1).split("|") if p.strip()]
        # V1.8 — Batch 7 dry-run adapter (mirror of pipeline.py branches)
        if gk in ("headlines", "press", "press_frenzy", "media_frenzy",
                  "montage", "scandal", "coverage") and (gt or gb):
            m = _re.search(r"headlines=([^;]+)", gb)
            if m:
                a["headlines"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
                if gt:
                    a["title"] = gt
            elif gt:
                a["headlines"] = [s.strip() for s in gt.split("|") if s.strip()]
            m = _re.search(r"title=([^;|]+)", gb)
            if m:
                a["title"] = m.group(1).strip()
        if gk in ("heat_spread", "heatmap", "contagion", "outbreak",
                  "epidemic", "wildfire") and (gt or gb):
            m = _re.search(r"hotspots=([^;]+)", gb)
            if m:
                hl = []
                for p in m.group(1).split("|"):
                    if ":" in p:
                        nm, pos = p.split(":", 1)
                        hl.append([nm.strip(), pos.strip()])
                    elif p.strip():
                        hl.append([p.strip(), ""])
                if hl:
                    a["hotspots"] = hl
            if gt:
                a["title"] = gt
        if gk in ("classified", "redacted", "secret", "top_secret",
                  "declassified", "dossier", "leak") and gt:
            a["reveal"] = gt
            m = _re.search(r"title=([^;|]+)", gb)
            if m:
                a["title"] = m.group(1).strip()
            m = _re.search(r"stamp=([^;|]+)", gb)
            if m:
                a["stamp"] = m.group(1).strip()
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"redacted_document", "headline_montage", "map_heat_spread"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-7 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
