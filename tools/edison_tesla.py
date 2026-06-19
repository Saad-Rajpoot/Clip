#!/usr/bin/env python3
"""Edison vs Tesla — War of the Currents. Batch-1 validation sample that
exercises the THREE new V1.2 motion-graphics primitives (chronology_timeline,
pull_quote_portrait, comparison_split) alongside the stable six, in a fresh
topic + palette (geopolitics → cold-steel) for variety vs the amber Rockefeller.

Original script (not copied from any reference). ~18 scenes (~3 min); graphic
beats spaced so no two are adjacent and the three portrait-family primitives are
>2 scenes apart.

Usage:
  python3 tools/edison_tesla.py            # author script.json + FREE dry-run
  python3 tools/edison_tesla.py --render   # + real cost-tracked render
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "Edison vs Tesla: The War of the Currents"
NICHE = "geopolitics"          # cold_steel/amber palette → variety vs Rockefeller
OUTDIR = ROOT / "research/motion_graphics_qa/edison_tesla_batch1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the 1880s, two of the greatest minds who ever lived went to war — not "
     "with armies, but with electricity itself.",
     4, "hook", "", "", "",
     ["vintage electricity laboratory", "edison light bulb archival", "tesla coil sparks", "1880s science lab"],
     "A darkened 1880s laboratory crackling with electrical arcs, brass instruments, dramatic blue light.",
     "war"),
    ("On one side stood Thomas Edison — already the most famous inventor on "
     "earth, the man who had lit up the modern world.",
     3, "intro", "name_reveal", "Thomas Edison",
     "1847 – 1931 · The Wizard of Menlo Park",
     ["Thomas Edison portrait", "1800s inventor portrait", "victorian scientist photograph"],
     "Formal seated portrait of Thomas Edison in a dark suit, stern gaze, sepia photograph.",
     "Thomas Edison"),
    ("Edison had given the world the phonograph, the motion picture camera, and "
     "the glowing electric light bulb.",
     2, "context", "", "", "",
     ["antique phonograph", "early light bulb glowing", "vintage invention workshop"],
     "Close-up of an early carbon-filament bulb glowing warm against black, dust motes drifting.",
     "light bulb"),
    ("Between 1879 and 1893, their rivalry would play out in laboratories, "
     "courtrooms, and on the world's greatest stage.",
     3, "context", "timeline", "The War of the Currents",
     "events=1879:Edison's Bulb|1882:Pearl Street|1888:Tesla's Motor|1893:Chicago Fair",
     ["vintage timeline archival", "1800s calendar pages", "historical montage sepia"],
     "A sweeping montage of 1880s industrial scenes dissolving one into the next, aged film texture.",
     "rivalry"),
    ("Edison's system was direct current — D.C. — steady, simple, and the only "
     "kind of electricity he had ever sold.",
     2, "context", "", "", "",
     ["vintage power station", "edison dynamo machine", "1800s electrical generator"],
     "A massive belt-driven Edison dynamo humming inside a brick power house, warm gaslight.",
     "direct current"),
    ("It all ran from his laboratory in Menlo Park, New Jersey — the world's "
     "first true invention factory.",
     3, "origin", "location", "Menlo Park, New Jersey",
     "The Invention Factory · 1876",
     ["Menlo Park New Jersey vintage", "antique New Jersey map", "1800s laboratory building"],
     "An antique sepia map of New Jersey with Menlo Park marked, aged paper, compass rose.",
     "Menlo Park"),
    ("But direct current carried a fatal flaw. It could not travel — and beyond "
     "a mile or two it faded to nothing. Meanwhile a strange, brilliant "
     "immigrant named Nikola Tesla had built something radical: alternating "
     "current, which could leap across hundreds of miles.",
     3, "rising", "", "", "",
     ["power lines fading distance", "tesla coil experiment", "1800s electrical sparks"],
     "A line of early electric poles marching into fog, the lamps dimming to darkness down the road.",
     "faded"),
    ("Now the lines were drawn. Edison versus Tesla. Direct current against "
     "alternating current. One mile of reach, compared to two hundred.",
     4, "thesis", "comparison", "Edison vs Tesla",
     "pair=Edison · DC|Tesla · AC;values=1|200",
     ["split screen vintage", "two inventors archival", "electrical diagram 1800s"],
     "A symmetrical composition of two glowing electrical towers facing each other across darkness.",
     "versus"),
    ("George Westinghouse saw which way the future pointed. He sided with Tesla "
     "and bet his entire company on the alternating-current design.",
     3, "turn", "", "", "",
     ["1800s industrialist portrait", "vintage factory westinghouse", "electrical works archival"],
     "A wide shot of a sprawling 1880s electrical works, workers dwarfed by humming machinery.",
     "future"),
    ("To own that future, Westinghouse paid Tesla over one million dollars for "
     "his patents — a staggering fortune for an idea.",
     4, "turn", "evidence", "Westinghouse Buys Tesla's AC Patents · 1888",
     "tag=EXHIBIT B",
     ["stacks of money vintage", "1800s bank notes", "gold coins dramatic"],
     "A leather satchel spilling 1880s banknotes across a desk, single hard light, deep shadow.",
     "one million dollars"),
    ("Edison fought back with fear. He staged gruesome public demonstrations, "
     "determined to brand alternating current a silent killer.",
     4, "rising", "", "", "",
     ["1800s newspaper scare", "vintage warning poster", "ominous electrical sparks"],
     "A crowd gathered around a roped-off demonstration, sparks and smoke, faces lit with alarm.",
     "fear"),
    ("Tesla, calm and certain, refused to flinch. He simply said: \"The present "
     "is theirs; the future, for which I really worked, is mine.\"",
     3, "reflection", "pull_quote",
     "The present is theirs; the future, for which I really worked, is mine.",
     "Nikola Tesla",
     ["Nikola Tesla portrait", "tesla laboratory vintage", "1800s scientist photograph"],
     "A contemplative portrait of Nikola Tesla, sharp eyes, dark background, dramatic side light.",
     "the future"),
    ("The decision would come down to a single, spectacular test: the World's "
     "Fair of 1893, in the city of Chicago.",
     4, "rising", "", "", "",
     ["1893 chicago worlds fair", "white city exposition vintage", "grand fairground archival"],
     "The gleaming 'White City' of the 1893 Chicago Exposition at dusk, reflecting pools, sepia grandeur.",
     "Chicago"),
    ("Whoever could light the fair would light America. And in the end, only "
     "one current could power a continent.",
     5, "climax_setup", "define_the_term", "ALTERNATING CURRENT", "",
     ["electric lights blazing", "worlds fair illumination night", "dramatic power grid"],
     "Thousands of bulbs igniting at once across a vast fairground, the night turning to day.",
     "current"),
    ("Westinghouse and Tesla won the contract — and on opening night, alternating "
     "current lit two hundred thousand lamps at once.",
     4, "climax", "", "", "",
     ["worlds fair lights night vintage", "thousands of bulbs glowing", "1893 illumination archival"],
     "An aerial period view of the fair ablaze with electric light, crowds gazing upward in wonder.",
     "won"),
    ("From that victory grew an empire of power — generators, transmission "
     "lines, and the great hydro-plant that would harness Niagara Falls.",
     4, "empire", "bar_chart", "AC POWER PLANTS",
     "bars=1893:1|1896:12|1900:200",
     ["niagara falls power plant vintage", "transmission towers archival", "1900s electrical grid"],
     "Towering transmission lines marching from a thundering Niagara powerhouse into the distance.",
     "empire"),
    ("Edison had lost the war of the currents. But history would remember them "
     "not as rivals — but as the two men who electrified the world.",
     2, "resolution", "", "", "",
     ["city skyline lights night", "modern power grid", "electric city glowing vintage"],
     "A golden-hour view of an early electrified city skyline, countless windows aglow, serene.",
     "electrified"),
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
    import hashlib
    return {"title": TITLE, "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for

    cfg = load_config()
    brief = Brief(title=TITLE, prompt="Edison vs Tesla war of the currents",
                  fmt="documentary", duration="6-8", theme="history",
                  captions=False, background="auto", extra={"niche": NICHE})
    out = ROOT / "output"
    run_dir = run_dir_for(brief, out)
    run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script()
    # script.txt MUST be the exact body source_sha256 was computed from, so the
    # reuse path (_load_reviewed_script) SHA-matches and reuses the rich JSON.
    body = f"{TITLE}\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (OUTDIR / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    print(f"RUN_DIR {run_dir}", flush=True)
    print(f"wrote script.json ({len(script['scenes'])} scenes)", flush=True)

    # FREE dry-run: which primitives does the director pick?
    from vidlore.motion_graphics import director as mgdir, registry as mgreg
    import re as _re2
    mg_scenes = []
    for i, sc in enumerate(script["scenes"]):
        a = {}
        gk = (sc["graphic_kind"] or "").lower()
        gt = sc["graphic_text"] or ""
        gb = sc["graphic_body"] or ""
        plain = "" if any(h in gb for h in ("place=", "branches=", "label=",
                                            "events=", "pair=", "values=",
                                            "tag=", "bars=", "coords=",
                                            "suffix=", "prefix=")) else gb
        if gk in ("name_reveal", "mugshot", "bio", "mini_bio", "map_reveal", "map_route") and gt:
            a["portrait_path"] = "/tmp/_d.png"
        if gk in ("pull_quote", "long_quote", "pull_quote_portrait") and gt:
            a["portrait_path"] = "/tmp/_d.png"; a["quote"] = gt
            if plain:
                a["name"] = plain
        if gk in ("name_reveal", "mugshot", "bio", "mini_bio") and gt:
            a["name"] = gt
        if gk in ("map_reveal", "map_route") and gt:
            a["name"] = gt
            m = _re2.search(r"place=([^;|]+)", gb)
            if m:
                a["place"] = m.group(1).strip()
        if gk in ("news_article", "document", "statement") and gt:
            a["headline"] = gt
        if gk in ("define_the_term", "quote_highlight") and gt:
            a["keyword"] = gt.split()[0]
        if gk in ("network_graph", "conspiracy_board", "org_tree") and gt:
            m = _re2.search(r"branches=([^;]+)", gb)
            if m:
                a["center"] = gt
                a["branches"] = [[p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                                 for p in m.group(1).split("|") if ":" in p]
        if gk == "stat_insight":
            mm = _re2.search(r"\$\s?([\d.]+)", gt)
            if mm:
                a["value"] = float(mm.group(1)); a["prefix"] = "$"
                if "million" in gt.lower():
                    a["suffix"] = " MILLION"
            ml = _re2.search(r"label=([^;|]+)", gb)
            if ml:
                a["label"] = ml.group(1).strip()
        if gk in ("timeline", "era_banner", "chronology", "chapter"):
            m = _re2.search(r"events=([^;]+)", gb)
            if m:
                a["events"] = [[p.split(":", 1)[0].strip(),
                                (p.split(":", 1)[1].strip() if ":" in p else "")]
                               for p in m.group(1).split("|") if p.strip()]
        if gk in ("comparison", "versus", "vs", "ratio"):
            m = _re2.search(r"pair=([^;|]+\|[^;]+)", gb)
            if m:
                lr = m.group(1).split("|")
                a["left"], a["right"] = lr[0].strip(), lr[1].strip()
            mv = _re2.search(r"values=([\d.]+)\|([\d.]+)", gb)
            if mv:
                a["leftval"], a["rightval"] = float(mv.group(1)), float(mv.group(2))
        # V1.3 Batch-2 dry-run adapters (mirror pipeline.py)
        if gk in ("evidence", "exhibit", "artifact", "framed_photo") and gt:
            a["caption"] = gt
            mt = _re2.search(r"tag=([^;|]+)", gb)
            a["tag"] = mt.group(1).strip() if mt else "EVIDENCE"
            a["image_path"] = "/tmp/_d.png"
        if gk in ("bar_chart", "stat_bars", "breakdown", "data_viz") and (gt or gb):
            mb = _re2.search(r"bars=([^;]+)", gb)
            if mb:
                a["bars"] = [[p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                             for p in mb.group(1).split("|") if ":" in p]
            if gt:
                a["title"] = gt
        if gk in ("location", "establish", "setting", "place_card") and gt:
            mp = _re2.search(r"place=([^;|]+)", gb)
            a["place"] = mp.group(1).strip() if mp else gt
            if plain:
                a["sub"] = plain
        mg_scenes.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                          "intensity": sc["intensity"], "narration": sc["narration"],
                          "assets": a})
    import hashlib as _hl
    seed = int(_hl.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(os.environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg_scenes, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN (free) ===", flush=True)
    print(json.dumps(summ, indent=1), flush=True)
    new3 = {"framed_evidence_spotlight", "statistic_bar_reveal",
            "location_establish_card"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-2 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}", flush=True)
    print(f"total primitives available: {len(mgreg.all_ids())}", flush=True)

    if "--render" in sys.argv:
        _render(brief, cfg, out, run_dir)


def _render(brief, cfg, out, run_dir):
    from vidlore.pipeline import render_from_script
    t0 = time.time()
    res = render_from_script(brief, cfg, out, keep_work=True)
    wall = time.time() - t0
    vid = run_dir / f"{run_dir.name}.mp4"
    print(f"\nRENDER_DONE wall={wall:.1f}s video={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
