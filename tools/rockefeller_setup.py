#!/usr/bin/env python3
"""Author the Rockefeller / Standard Oil business-documentary validation sample
and run a FREE director dry-run to confirm all six motion-graphics primitives
get a genuine contextual opportunity BEFORE any paid render.

Original script (not copied from any reference). 16 scenes (~2.5-3 min), 6
graphic beats spaced so no two are adjacent and the two portrait-family
primitives are >2 scenes apart (incompatible-adjacent rule).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "John D. Rockefeller: The Man Who Built Standard Oil"

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    # 0 — HOOK (footage)
    ("By the time he died, one quiet bookkeeper had become the richest human "
     "being who ever lived — and had reshaped the entire modern world.",
     4, "hook", "", "", "",
     ["vintage oil refinery", "industrial smokestacks 1900s", "gilded age city archival", "black and white factory"],
     "Sweeping archival shot of a vast 1900s oil refinery at dusk, smokestacks and storage tanks, sepia tone, cinematic.",
     "richest"),
    # 1 — PORTRAIT → cinematic_portrait_hold
    ("His name was John Davison Rockefeller, a disciplined, soft-spoken clerk "
     "from upstate New York who would come to control the lifeblood of the "
     "industrial age.",
     3, "intro", "name_reveal", "John D. Rockefeller",
     "1839 – 1937 · Founder of Standard Oil",
     ["John D Rockefeller portrait", "gilded age businessman portrait", "1800s man formal portrait"],
     "Formal seated portrait of a stern 19th-century businessman in a dark suit, sepia photograph.",
     "John Davison Rockefeller"),
    # 2 — footage
    ("He had no fortune, no famous family, and no inheritance. What he had was "
     "patience, precision, and an almost religious devotion to order.",
     2, "context", "", "", "",
     ["antique ledger accounting", "old fountain pen writing", "vintage office desk 1800s"],
     "Close-up of ink-stained hands writing neat columns of figures in a leather ledger by candlelight.",
     "order"),
    # 3 — footage
    ("Oil was chaos. Wildcat drillers struck it rich one month and went broke "
     "the next, as prices swung violently with every new discovery.",
     3, "context", "", "", "",
     ["1800s oil derrick", "gushing oil well vintage", "oil boom town archival"],
     "A wooden oil derrick erupting with crude oil, workers scrambling, muddy boomtown, period photograph.",
     "chaos"),
    # 4 — NUMBER → gold_number_callout
    ("Rockefeller saw opportunity in the disorder. He bought, merged, and "
     "undercut relentlessly — until Standard Oil was earning more than "
     "$30 million in profit every single year.",
     4, "turn", "stat_insight", "$30 MILLION",
     "label=Standard Oil annual profit · 1880",
     ["stacks of gold coins vintage", "1800s bank vault", "money counting archival"],
     "Towering stacks of gold coins and banknotes on a dark wooden table, dramatic side light.",
     "$30 million"),
    # 5 — footage
    ("To win, he did not simply sell oil. He controlled how it moved — the "
     "barrels, the tank cars, the freight rates that decided who survived.",
     3, "context", "", "", "",
     ["vintage railroad freight", "oil barrels warehouse", "steam train 1800s"],
     "A long line of oil tank rail cars stretching into the distance through an industrial rail yard, sepia.",
     "controlled"),
    # 6 — MAP → portrait_name_over_map
    ("It had all begun in a single refinery on the banks of the Cuyahoga "
     "River, in the smoke and clatter of Cleveland, Ohio.",
     3, "origin", "map_reveal", "John D. Rockefeller",
     "place=Cleveland, Ohio · 1863",
     ["Cleveland Ohio 1800s", "vintage river refinery", "antique Ohio map", "old industrial riverbank"],
     "An antique sepia map of Ohio with Cleveland marked on a river bend, aged paper texture.",
     "Cleveland"),
    # 7 — footage
    ("Rivals were given a simple choice: sell their company to Rockefeller, or "
     "be crushed by a man who could always afford to charge less.",
     4, "rising", "", "", "",
     ["abandoned factory 1800s", "empty warehouse vintage", "industrial ruins archival"],
     "A shuttered, empty competitor refinery, broken windows, weeds, melancholy period photograph.",
     "crushed"),
    # 8 — footage
    ("One by one, the independents fell. Within a decade, a single company "
     "refined nearly all of the oil in the United States.",
     3, "context", "", "", "",
     ["1800s factory workers", "oil refinery interior vintage", "industrial machinery archival"],
     "Rows of workers tending enormous refining stills inside a cavernous 19th-century plant.",
     "all"),
    # 9 — KEYWORD → kinetic_keyword
    ("What Rockefeller had built had a name that would soon frighten an entire "
     "nation. America had never seen anything like it. A monopoly.",
     5, "thesis", "define_the_term", "MONOPOLY", "",
     ["dark storm clouds dramatic", "ominous industrial skyline", "shadow over city vintage"],
     "Brooding dark sky over an endless industrial skyline, single shaft of light, ominous mood.",
     "monopoly"),
    # 10 — footage
    ("For thirty years his grip seemed unbreakable. Politicians feared him, "
     "journalists hunted him, and the public slowly began to turn.",
     3, "context", "", "", "",
     ["1900s newspaper printing press", "vintage protest crowd", "old newspaper headlines"],
     "A thundering newspaper printing press spitting out papers, ink and motion, period scene.",
     "turn"),
    # 11 — EMPIRE → money_flow_empire
    ("By then Standard Oil was no longer a company. It was an empire of "
     "subsidiaries — pipelines, railroads, refineries, and barrel works — all "
     "funneling wealth to one secret holding company.",
     4, "empire", "network_graph", "STANDARD OIL",
     "branches=Pipelines:$40M|Railroads:$25M|Refineries:$90M|Barrel Works:$15M",
     ["vintage pipeline construction", "oil empire industrial network", "1900s corporate archival"],
     "An aerial period view of sprawling refineries, pipelines and rail lines radiating outward, sepia.",
     "empire"),
    # 12 — footage
    ("The bigger it grew, the more dangerous it looked to a country built on "
     "competition and the promise of a fair chance.",
     3, "context", "", "", "",
     ["1900s city street crowd", "vintage americana flag", "old courthouse building"],
     "A bustling early-1900s American main street, horse carts and pedestrians, weathered photograph.",
     "dangerous"),
    # 13 — footage
    ("The reckoning came not from a rival, but from the highest court in the "
     "land.",
     4, "rising", "", "", "",
     ["supreme court building vintage", "old courthouse columns", "1900s government archival"],
     "The marble columns of an early-1900s supreme court building, low dramatic angle, sepia.",
     "reckoning"),
    # 14 — DOCUMENT → headline_document_reveal
    ("In 1911, the Supreme Court of the United States ruled against him, "
     "ordering the empire dismantled into thirty-four separate companies.",
     5, "climax", "news_article", "SUPREME COURT ORDERS STANDARD OIL DISSOLVED",
     "The Supreme Court of the United States · May 1911",
     ["vintage newspaper front page", "1911 newspaper archival", "old court document"],
     "A yellowed 1911 newspaper front page with a bold black headline, aged paper, dramatic light.",
     "dissolved"),
    # 15 — ENDING (footage, restrained)
    ("But the breakup only multiplied his fortune. Rockefeller died in 1937, "
     "his name carved into universities, hospitals, and the foundations of the "
     "modern age.",
     2, "resolution", "", "", "",
     ["grand university building", "vintage hospital archival", "old foundation stone engraving"],
     "Golden-hour shot of a grand stone university facade, ivy and carved lettering, serene, cinematic.",
     "foundations"),
]


def build_script() -> dict:
    scenes = []
    for i, (nar, inten, role, gk, gt, gb, kws, vis, emph) in enumerate(SCENES):
        scenes.append({
            "narration": nar,
            "keywords": kws,
            "visual": vis,
            "intensity": inten,
            "emphasis": emph,
            "shot_type": "wide" if i % 3 == 0 else "medium",
            "role": role,
            "graphic_kind": gk,
            "graphic_text": gt,
            "graphic_body": gb,
        })
    body = f"{TITLE}\n\n" + "\n\n".join(s["narration"] for s in scenes)
    import hashlib
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return {"title": TITLE, "source_sha256": sha, "scenes": scenes}


def dry_run(script: dict) -> None:
    """FREE — no render, no API. Mirror the pipeline adapter + run director."""
    import re as _re2
    from vidlore.motion_graphics import director as mgdir
    mg_scenes = []
    for i, sc in enumerate(script["scenes"]):
        a = {}
        gk = (sc["graphic_kind"] or "").lower()
        gt = sc["graphic_text"] or ""
        gb = sc["graphic_body"] or ""
        mpl = _re2.search(r"place=([^;|]+)", gb)
        mbr = _re2.search(r"branches=([^;]+)", gb)
        mlbl = _re2.search(r"label=([^;|]+)", gb)
        plain = "" if any(h in gb for h in ("place=", "branches=", "label=")) else gb
        if mlbl:
            a["label"] = mlbl.group(1).strip()
        # simulate a resolved portrait photo for portrait/map kinds
        if gk in ("name_reveal", "mugshot", "bio", "mini_bio", "map_reveal", "map_route") and gt:
            a["portrait_path"] = "/tmp/_dryrun_portrait.png"
        if gk in ("name_reveal", "mugshot", "bio", "mini_bio") and gt:
            a["name"] = gt
            if plain:
                a["sub"] = plain
            if mpl:
                a["place"] = mpl.group(1).strip()
        if gk in ("map_reveal", "map_route") and gt:
            a["name"] = gt
            if mpl:
                a["place"] = mpl.group(1).strip()
        if gk in ("news_article", "document", "headline", "redacted", "statement", "press_release") and gt:
            a["headline"] = gt
            if plain:
                a["source"] = plain
        if gk in ("define_the_term", "quote_highlight") and gt:
            a["keyword"] = gt.split()[0]
        if gk in ("network_graph", "conspiracy_board", "org_tree") and gt and mbr:
            a["center"] = gt
            a["branches"] = [[p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                             for p in mbr.group(1).split("|") if ":" in p]
        mg_scenes.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                          "intensity": int(sc["intensity"]),
                          "narration": sc["narration"], "assets": a})
    seed = abs(hash(TITLE)) % 100000
    dec = mgdir.plan(mg_scenes, niche="business", seed=seed, density=0.45)
    summ = mgdir.plan_summary(dec)
    print("=== DIRECTOR DRY-RUN (free, niche=business, density=0.45) ===")
    print(json.dumps(summ, indent=2))
    fired = {d.primitive for d in dec if d.primitive}
    want = {"gold_number_callout", "cinematic_portrait_hold", "headline_document_reveal",
            "portrait_name_over_map", "kinetic_keyword", "money_flow_empire"}
    print("\nPer-scene:")
    for d in dec:
        if d.primitive:
            print(f"  sc{d.scene_index:<2} {d.primitive:26s} score={d.score:.2f} "
                  f"inputs={sorted(d.inputs.keys())}")
    missing = want - fired
    print(f"\nFIRED {len(fired)}/6: {sorted(fired)}")
    print(f"MISSING: {sorted(missing) if missing else 'NONE — all six fire ✅'}")


if __name__ == "__main__":
    script = build_script()
    if "--write" in sys.argv:
        from vidlore.brief import Brief
        from vidlore.pipeline import run_dir_for
        brief = Brief(title=TITLE, prompt="Rockefeller/Standard Oil business documentary validation sample",
                      fmt="documentary", duration="6-8", theme="history",
                      captions=False, background="auto",
                      extra={"niche": "business"})
        run_dir = run_dir_for(brief, ROOT / "output")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
        # script.txt = the reviewed body (title + blank-line-separated narration);
        # its SHA matches script.json.source_sha256 → pipeline reuses the rich
        # JSON (keeps every graphic_kind tag) and SKIPS the paid LLM editor.
        body = f"{TITLE}\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
        (run_dir / "script.txt").write_text(body, encoding="utf-8")
        bj = {"title": TITLE, "fmt": "documentary", "duration": "6-8",
              "theme": "history", "captions": False, "music": None, "voice": None,
              "background": "auto", "voiceover": None, "extra": {"niche": "business"}}
        (run_dir / "brief.json").write_text(json.dumps(bj, indent=2), encoding="utf-8")
        print(f"WROTE {run_dir}/script.json + brief.json ({len(script['scenes'])} scenes)")
    dry_run(script)
