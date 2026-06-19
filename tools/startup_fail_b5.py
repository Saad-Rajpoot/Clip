#!/usr/bin/env python3
"""Why Startups Fail — Batch-5 validation sample exercising the THREE new V1.6
motion-graphics primitives (statement_card, pictograph_scale, composition_stack).

Original script (not copied from any reference). ~12 scenes (~2.5 min); the two
chart-family primitives (pictograph / composition) are kept >2 scenes apart.

  python3 tools/startup_fail_b5.py            # author script.json + FREE dry-run
  python3 tools/startup_fail_b5.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "Why Startups Fail: The Brutal Math"
NICHE = "business"
OUTDIR = ROOT / "research/motion_graphics_qa/startup_fail_batch5"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("Every year, a million new companies are born in a blaze of optimism. Most "
     "will not live to see their tenth birthday.",
     4, "hook", "", "", "",
     ["startup office modern", "entrepreneur working late", "city skyline ambition"],
     "A lone founder at a laptop in a dark office, city lights beyond the glass.",
     "optimism"),
    ("Startups rarely die because a competitor beats them. They die from "
     "something far closer to home.",
     4, "thesis", "statement", "Most startups aren't killed — they kill themselves.",
     "emphasis=kill themselves ; source=The Brutal Math",
     ["empty office chairs", "abandoned workspace", "closed sign window"],
     "Rows of empty desks, monitors dark, a single light still on.",
     "themselves"),
    ("It begins with a dream and a pitch deck — a product the world supposedly "
     "can't live without.",
     3, "dream", "", "", "",
     ["pitch deck presentation", "startup whiteboard ideas", "team brainstorm"],
     "A whiteboard crowded with arrows and dollar signs, fresh coffee steaming.",
     "dream"),
    ("But the numbers are merciless. Of every ten startups that launch with "
     "fanfare, nine are gone within a decade.",
     4, "ratio", "pictograph", "startups that fail within ten years",
     "count=9 ; total=10",
     ["startup failure graph", "declining business chart", "empty startup office"],
     "Ten small office windows, nine of them gone dark.",
     "nine"),
    ("The first enemy is cash. A startup doesn't run on ideas — it runs on "
     "runway, the months of money it has left.",
     3, "cash", "", "", "",
     ["burning money concept", "cash counting vintage", "empty wallet desk"],
     "A bank balance ticking down under a desk lamp, red ink spreading.",
     "runway"),
    ("And that runway burns shockingly fast, consumed long before the product "
     "ever finds its market.",
     4, "burn", "", "", "",
     ["fuel gauge empty", "candle burning down", "hourglass running out"],
     "An hourglass nearly empty, the last grains falling.",
     "burns"),
    ("Look at where the money actually goes — and the problem becomes obvious.",
     3, "spend", "composition", "WHERE THE MONEY GOES",
     "segments=Salaries:46|Marketing:24|Product:20|Overhead:10 ; suffix=%",
     ["office expenses receipts", "payroll documents", "marketing budget chart"],
     "A stack of invoices fanned across a desk under hard light.",
     "where the money goes"),
    ("Payroll devours nearly half before a single customer is won. Marketing "
     "eats the rest chasing growth that never compounds.",
     3, "drain", "", "", "",
     ["office workers many", "marketing billboard city", "spending money fast"],
     "A busy open-plan office, the hum of a company spending faster than it earns.",
     "devours"),
    ("So founders pivot. They chase the next idea, then the next, burning trust "
     "and cash with every turn.",
     4, "pivot", "", "", "",
     ["business pivot strategy", "changing direction arrows", "stressed founder"],
     "A founder erasing a plan and redrawing it, again and again.",
     "pivot"),
    ("And in the end, almost all of them run out of the one resource no "
     "investor can ever refill — time.",
     5, "end", "statement", "Startups don't run out of ideas. They run out of time.",
     "emphasis=run out of time ; source=The Brutal Math",
     ["clock midnight dramatic", "empty office final", "last light switching off"],
     "A wall clock at midnight, the office finally dark.",
     "time"),
    ("The doors close quietly. The website goes blank. The dream is filed away "
     "as a lesson.",
     3, "close", "", "", "",
     ["closed business sign", "blank website screen", "empty office moving out"],
     "A laptop lid closing on a blank screen, boxes by the door.",
     "close"),
    ("Yet from every ten that fall, one survives — and occasionally, that one "
     "changes the world.",
     3, "survive", "", "", "",
     ["successful startup celebration", "sunrise city skyline", "one light glowing"],
     "Dawn over a city, a single office window still glowing warm.",
     "one survives"),
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
    brief = Brief(title=TITLE, prompt="Why most startups fail — the brutal economics",
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
        if gk in ("statement", "thesis", "claim") and gt:
            a["text"] = gt
            m = _re.search(r"emphasis=([^;|]+)", gb)
            if m:
                a["emphasis"] = m.group(1).strip()
            m = _re.search(r"source=([^;|]+)", gb)
            if m:
                a["source"] = m.group(1).strip()
        if gk in ("pictograph", "ratio", "figures") and (gt or gb):
            m = _re.search(r"count=(\d+)", gb)
            if m:
                a["count"] = int(m.group(1))
            m = _re.search(r"total=(\d+)", gb)
            if m:
                a["total"] = int(m.group(1))
            if gt:
                a["label"] = gt
        if gk in ("composition", "breakdown", "stacked", "split") and (gt or gb):
            m = _re.search(r"segments=([^;]+)", gb)
            if m:
                a["segments"] = [[p.split(":", 1)[0].strip(), p.split(":", 1)[1].strip()]
                                 for p in m.group(1).split("|") if ":" in p]
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
    new3 = {"statement_card", "pictograph_scale", "composition_stack"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-5 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
