#!/usr/bin/env python3
"""The Man Who Said No — Batch-11 validation sample exercising the THREE new V2.2
motion-graphics primitives (spotlight_object_hold, flowchart_decision,
world_map_arc) alongside stable ones (chronology_timeline, spectrum_meter).

Original script (not copied from any reference). ~11 scenes (~2.5 min). The 1983
Stanislav Petrov nuclear false-alarm naturally fits all three: a dramatic reveal
of the man on duty (spotlight), a literal yes/no decision fork (report the launch
or not), and a great-circle missile arc across the globe. All five graphic scenes
are DIFFERENT families (reveals / maps / diagrams / timelines / meters) so the
same-family guard never engages; beats spaced >=2 apart at indices 1/3/5/7/9.
Collision-free graphic_kinds (reveal / world_arc / decision) keep the legacy
footage.py spotlight/etc. card builders out of the way.

  python3 tools/petrov_b11.py            # author script.json + FREE dry-run
  python3 tools/petrov_b11.py --render   # + real cost-tracked render
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "The Man Who Said No: How One Officer's Decision Stopped a Nuclear War"
NICHE = "history"
OUTDIR = ROOT / "research/motion_graphics_qa/batch11_render"
OUTDIR.mkdir(parents=True, exist_ok=True)

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body, keywords, visual, emphasis)
SCENES = [
    ("In the dead of night, near the most dangerous moment of the entire Cold "
     "War, the world came far closer to catastrophe than almost anyone would "
     "ever know.",
     4, "hook", "", "", "",
     ["cold war bunker night", "soviet command center dark", "nuclear war room vintage"],
     "A darkened Soviet command bunker lit only by warning panels.",
     "far closer to catastrophe"),
    ("Meet the man who was on duty that night — the one officer whose single "
     "decision would decide whether the missiles flew.",
     4, "reveal", "reveal", "STANISLAV PETROV",
     "kicker=THE MAN ON DUTY ; sub=Serpukhov-15 early-warning bunker · 00:14 ; title=ONE DECISION",
     ["soviet officer 1983 portrait", "lieutenant colonel uniform", "cold war officer"],
     "A lone uniformed officer at a console, face underlit by red light.",
     "the man who"),
    ("Petrov's job was simple to describe and impossible to bear — watch the "
     "satellites, and if America ever launched, sound the alarm.",
     3, "setup", "", "", "",
     ["spy satellite orbit", "1980s radar screen", "soviet missile early warning"],
     "An Oko early-warning satellite drifting over a night-side Earth.",
     "sound the alarm"),
    ("The system was built to catch a strike the instant it left the ground — "
     "warheads that would cross thousands of miles in barely half an hour.",
     3, "arc", "world_arc", "U.S. SILOS",
     "to=MOSCOW ; from_pos=0.22,0.40 ; to_pos=0.78,0.45 ; title=THE 28-MINUTE FLIGHT",
     ["intercontinental missile arc", "global map cold war", "missile trajectory map"],
     "A great-circle line arcing from the American plains toward Moscow.",
     "thousands of miles"),
    ("Then, at fourteen minutes past midnight, the panel screamed a single, "
     "impossible word: LAUNCH. Then another. Then three more.",
     5, "alarm", "", "", "",
     ["red alert klaxon", "warning panel flashing", "missile launch detected screen"],
     "A console blazing red as five launch indicators light in sequence.",
     "LAUNCH"),
    ("Every protocol said report it up the chain at once. But Petrov faced a "
     "choice no training could settle — call it real, or call it a glitch.",
     4, "decision", "decision", "Report the launch as real?",
     "yes=Soviet missiles fly in minutes ; no=Trust the system is wrong ; "
     "chosen=no ; title=THE CHOICE",
     ["soviet officer thinking", "hand over red phone", "decision under pressure"],
     "A hand hovering over a red reporting phone, not yet lifting it.",
     "faced a choice"),
    ("Something didn't fit. A genuine first strike would come in the hundreds, "
     "not a tidy five — and the ground radar saw nothing at all.",
     4, "reason", "", "", "",
     ["empty radar screen", "soviet ground radar", "1983 military analysts"],
     "A ground-radar scope sweeping clean, empty arcs through the dark.",
     "saw nothing at all"),
    ("For tense minutes he held the line, reporting a system malfunction while "
     "the clock ran down — until the sequence of that night told its own story.",
     3, "timeline", "timeline", "THE LONGEST NIGHT",
     "events=00:14:First alarm|00:17:Radar silent|00:23:Called a glitch|"
     "Dawn:Sun-glint confirmed",
     ["1983 timeline montage", "clock ticking night", "cold war archive"],
     "A spread of timestamps sliding past as the bunker clock advances.",
     "told its own story"),
    ("The cause was almost poetic: high-altitude sunlight glinting off cloud "
     "tops had fooled the satellite into seeing fire that was never there.",
     4, "cause", "", "", "",
     ["sunlight through clouds", "satellite sensor glare", "high altitude clouds dawn"],
     "Dawn light flaring off a bank of cloud, seen from orbit.",
     "fire that was never there"),
    ("Analysts later judged how near the brink that night had brought the "
     "world — and the assessment was sobering.",
     5, "gauge", "threat_level", "PROXIMITY TO WAR",
     "value=95 ; bands=CALM|TENSE|ALERT|CRISIS|THE BRINK ; readout=THE BRINK ; "
     "title=HOW CLOSE IT CAME",
     ["1983 nuclear tension", "cold war crisis map", "doomsday clock"],
     "An analyst sliding a marker to the far end of a wall gauge.",
     "the brink"),
    ("Stanislav Petrov was never rewarded and never punished. He simply went "
     "home. But because one man paused, the world woke up the next morning.",
     4, "outro", "", "", "",
     ["sunrise over city", "quiet apartment 1983", "dawn after the storm"],
     "A grey Moscow dawn over silent rooftops, the crisis unknown to all.",
     "one man paused"),
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


def _dryrun_assets(gk: str, gt: str, gb: str) -> dict:
    """Mirror of the pipeline.py MG adapter branches for the kinds this sample
    uses — so the FREE dry-run sees the same assets the real render will."""
    import re as _re
    a = {}
    # NEW V2.2 — spotlight reveal
    if gk in ("spotlight", "reveal", "subject_reveal", "behold", "unveil") and gt:
        a["subject"] = gt
        m = _re.search(r"kicker=([^;|]+)", gb)
        if m:
            a["kicker"] = m.group(1).strip()
        m = _re.search(r"sub=([^;|]+)", gb)
        if m:
            a["sub"] = m.group(1).strip()
        m = _re.search(r"title=([^;|]+)", gb)
        if m:
            a["title"] = m.group(1).strip()
    # NEW V2.2 — decision fork
    if gk in ("decision", "flowchart", "decision_tree", "branch", "choice",
              "fork", "yes_no") and gt:
        a["question"] = gt
        for key in ("yes", "no", "chosen", "yes_label", "no_label"):
            m = _re.search(rf"{key}=([^;|]+)", gb)
            if m:
                a[key] = m.group(1).strip()
        m = _re.search(r"title=([^;|]+)", gb)
        if m:
            a["title"] = m.group(1).strip()
    # NEW V2.2 — world arc
    if gk in ("world_arc", "world_map_arc", "arc", "global_link",
              "transatlantic", "great_circle", "globe_arc") and gt:
        a["from_place"] = gt
        m = _re.search(r"to=([^;|]+)", gb)
        if m:
            a["to_place"] = m.group(1).strip()
        m = _re.search(r"from_pos=([^;|]+)", gb)
        if m:
            a["from_pos"] = m.group(1).strip()
        m = _re.search(r"to_pos=([^;|]+)", gb)
        if m:
            a["to_pos"] = m.group(1).strip()
        m = _re.search(r"title=([^;|]+)", gb)
        if m:
            a["title"] = m.group(1).strip()
    # STABLE — timeline
    if gk in ("timeline", "era_banner", "chronology", "chapter") and (gt or gb):
        m = _re.search(r"events=([^;]+)", gb)
        if m:
            a["events"] = [[p.split(":", 1)[0].strip(),
                            (p.split(":", 1)[1].strip() if ":" in p else "")]
                           for p in m.group(1).split("|") if p.strip()]
        if gt and not _re.search(r"\d", gt):
            a["title"] = gt
    # STABLE — spectrum meter
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
    return a


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for

    cfg = load_config()
    brief = Brief(title=TITLE, prompt="How one Soviet officer's decision stopped a nuclear war",
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
    mg = []
    for i, sc in enumerate(script["scenes"]):
        gk = (sc["graphic_kind"] or "").lower()
        a = _dryrun_assets(gk, sc["graphic_text"] or "", sc["graphic_body"] or "")
        mg.append({"index": i, "role": sc["role"], "graphic_kind": gk,
                   "intensity": sc["intensity"], "narration": sc["narration"],
                   "assets": a})
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dens = float(__import__("os").environ.get("VIDLORE_MG_DENSITY", "0.5"))
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=dens)
    summ = mgdir.plan_summary(dec)
    print("\n=== DIRECTOR DRY-RUN ===\n" + json.dumps(summ, indent=1), flush=True)
    new3 = {"spotlight_object_hold", "flowchart_decision", "world_map_arc"}
    fired = set(summ["by_primitive"])
    print(f"\nBATCH-11 fired: {sorted(new3 & fired)} · missing: {sorted(new3 - fired)}",
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
