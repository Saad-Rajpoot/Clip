#!/usr/bin/env python3
"""The Teapot Dome Scandal — Batch-9 validation sample exercising the THREE new
V2.0 motion-graphics primitives (countdown_clock, connection_web, quote_stream)
alongside stable ones (name_reveal portrait, chronology_timeline).

Original script (not copied from any reference). ~12 scenes (~2.5 min). The 1920s
corruption scandal fits all three: a ticking deadline before the grand jury, a web
of officials and oilmen, and a chorus of press condemnation. Graphic beats are
spaced so no two are adjacent; all five graphic scenes are different families so
no same-family clash. Collision-free graphic_kinds (deadline / conspiracy_web /
reactions) are used so the legacy footage.py countdown/quote_stream card builders
stay out of the way and the premium MG primitives render.

  python3 tools/teapot_dome_b9.py            # author script.json + FREE dry-run
  python3 tools/teapot_dome_b9.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "Teapot Dome: The Scandal That Put a Cabinet Officer in Prison"
NICHE = "crime"
OUTDIR = ROOT / "research/motion_graphics_qa/teapot_dome_batch9"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the 1920s, a secret deal over a patch of Wyoming oil land became the "
     "biggest corruption scandal the United States had ever seen.",
     4, "hook", "", "", "",
     ["1920s oil derrick vintage", "wyoming oil field 1920", "teapot dome rock"],
     "A lone oil derrick silhouetted against a dusty Wyoming dawn.",
     "biggest corruption scandal"),
    ("At its centre was the man who held the keys to the nation's oil reserves — "
     "Interior Secretary Albert B. Fall.",
     3, "intro", "name_reveal", "Albert B. Fall",
     "1861 – 1944 · U.S. Secretary of the Interior",
     ["albert fall portrait", "1920s cabinet official", "1920s politician portrait"],
     "A stern sepia portrait of a moustached cabinet official.",
     "Albert B. Fall"),
    ("Fall had quietly leased the reserves to private oilmen for bribes. But "
     "investigators were closing in.",
     4, "setup", "", "", "",
     ["1920s senate hearing", "1920s investigation vintage", "oil lease documents"],
     "Stacks of leases and ledgers on a Senate committee table.",
     "closing in"),
    ("The clock was ticking. The grand jury would convene within seventy-two "
     "hours, and the deadline was closing fast.",
     5, "deadline", "deadline", "Until the Grand Jury",
     "value=72 ; to=0 ; unit=HOURS",
     ["courthouse clock vintage", "1920s grand jury", "ticking clock dramatic"],
     "A courthouse clock face ticking toward the hour.",
     "the deadline"),
    ("As the evidence mounted, the bribes traced straight back to the men at the "
     "very top of the oil business.",
     4, "evidence", "", "", "",
     ["1920s money bribery vintage", "oil barons 1920s", "cash payoff vintage"],
     "A black satchel of cash sliding across a polished desk.",
     "straight back to the top"),
    ("What investigators uncovered was a tangle of connections — a web of "
     "officials and oilmen who all knew each other well.",
     4, "web", "conspiracy_web", "THE TEAPOT DOME WEB",
     "nodes=Albert Fall|Harry Sinclair|Edward Doheny|Sen. Walsh|Pres. Harding ; "
     "links=Albert Fall-Harry Sinclair|Albert Fall-Edward Doheny|Sen. Walsh-Albert Fall|Pres. Harding-Albert Fall|Harry Sinclair-Edward Doheny",
     ["1920s conspiracy chart", "oil tycoons 1920s", "1920s corruption network"],
     "Portraits pinned to a wall, strung together with red thread.",
     "a web of officials"),
    ("For the first time in American history, a sitting cabinet officer would face "
     "a criminal trial.",
     4, "stakes", "", "", "",
     ["1920s courtroom vintage", "1920s trial gavel", "1920s justice department"],
     "An empty courtroom dock beneath tall arched windows.",
     "criminal trial"),
    ("When the story broke, the press erupted. The reaction was a chorus of "
     "condemnation from every major paper in the country.",
     4, "reactions", "reactions", "THE VERDICT OF THE PRESS",
     "quotes=The greatest scandal in our history.:The New York Times|Corruption "
     "laid bare.:The Tribune|A government up for sale.:The Nation",
     ["1920s newspaper headlines", "1920s printing press", "newsboys 1920s"],
     "Front pages spinning off the presses with damning headlines.",
     "a chorus of condemnation"),
    ("In 1929, Albert Fall became the first former cabinet member ever sent to "
     "prison for crimes committed in office.",
     4, "verdict", "", "", "",
     ["1929 prison gate vintage", "1920s convict vintage", "1920s prison cell"],
     "A heavy iron prison gate swinging shut at dusk.",
     "first ever sent to prison"),
    ("The path from secret lease to prison cell took most of the decade to walk.",
     3, "timeline", "timeline", "THE ROAD TO JUSTICE",
     "events=1921:Leases signed|1922:Senate probe opens|1924:Scandal breaks|1929:Fall convicted",
     ["1920s timeline montage", "1920s newspaper archive", "1929 prison vintage"],
     "A row of dated front pages leading toward a prison gate.",
     "most of the decade"),
    ("Teapot Dome became a byword for corruption — proof that no office sat above "
     "the law.",
     3, "close", "", "", "",
     ["1920s capitol building", "american flag vintage", "1920s washington dc"],
     "The Capitol dome standing cold against a grey winter sky.",
     "no office above the law"),
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
    brief = Brief(title=TITLE, prompt="How the Teapot Dome scandal jailed a cabinet officer",
                  fmt="documentary", duration="6-8", theme="crime",
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
        # V2.0 — Batch 9 dry-run adapter (mirror of pipeline.py branches)
        if gk in ("countdown", "deadline", "timer", "clock", "ticking_clock",
                  "urgency_clock") and (gt or gb):
            m = _re.search(r"value=([\d.,]+)", gb)
            if m:
                a["value"] = float(m.group(1).replace(",", ""))
            m = _re.search(r"to=([\d.,]+)", gb)
            if m:
                a["to"] = float(m.group(1).replace(",", ""))
            m = _re.search(r"unit=([^;|]+)", gb)
            if m:
                a["unit"] = m.group(1).strip()
            if gt:
                a["label"] = gt
        if gk in ("connection_web", "network_web", "relationship_map",
                  "conspiracy_web", "web_of_connections", "connections") and (gt or gb):
            m = _re.search(r"nodes=([^;]+)", gb)
            if m:
                a["nodes"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
            m = _re.search(r"links=([^;]+)", gb)
            if m:
                a["links"] = [s.strip() for s in m.group(1).split("|") if s.strip()]
            if gt:
                a["title"] = gt
        if gk in ("quote_stream", "quotes", "chorus", "reactions",
                  "verdict_quotes") and (gt or gb):
            m = _re.search(r"quotes=([^;]+)", gb)
            if m:
                qs = []
                for p in m.group(1).split("|"):
                    if ":" in p:
                        tx, by = p.rsplit(":", 1)
                        qs.append([tx.strip(), by.strip()])
                    elif p.strip():
                        qs.append([p.strip(), ""])
                if qs:
                    a["quotes"] = qs
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
    new3 = {"countdown_clock", "connection_web", "quote_stream"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-9 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
