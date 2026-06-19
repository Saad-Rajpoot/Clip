#!/usr/bin/env python3
"""REAL-PIPELINE QA RENDER — "How a Hidden Vulnerability Exposed Millions of Devices".

Section-C technology sample through the SHARED Vidlore pipeline (mirrors
tools/_secB_spy_render.py / _iran_iraq_render.py). Exercises the NEW V2.8
systems/mechanism/hybrid primitives naturally via the real director —
system_planview_flow, packet_path_trace, exploit_chain, measurement_callout,
footage_fact_overlay, footage_object_callout — while staying FOOTAGE-FIRST and
anti-dashboard. AI VIDEO OFF; FAL stills budget-capped 0.
"""
import hashlib, json, os, re as _re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "0")
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ["VIDLORE_FAL_MAX_IMAGES"] = "0"
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")

TITLE = "How a Hidden Vulnerability Exposed Millions of Devices"
NICHE = "tech"
PROMPT = ("How a single hidden software vulnerability let attackers compromise "
          "millions of connected devices worldwide")

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SCENES = [
    ("Somewhere in the code that runs millions of everyday devices, a single "
     "flaw sat hidden for years.", 4, "hook", "", "", ""),
    # system_planview_flow — how a request moves through the system
    ("To understand the breach, you first have to see the system: a request "
     "travels from the client, through the gateway, into the auth service, and "
     "down to the database.", 3, "system", "architecture", "THE REQUEST PATH", ""),
    ("Each of those layers was built by a different team, in a different decade, "
     "and nobody owned the whole picture.", 3, "setup", "", "", ""),
    # footage_fact_overlay — a hard stat over footage
    ("By the time anyone noticed, the flawed component was running on more than "
     "2 billion devices around the world.", 4, "scale", "fact_overlay", "", ""),
    ("It shipped inside routers, cameras, thermostats — the quiet machines that "
     "fill our homes and offices.", 3, "spread", "", "", ""),
    # measurement_callout — a real dimension
    ("The vulnerable buffer was tiny: just 64 bytes of memory, but it was wide "
     "enough to let an attacker slip past every defence.", 4, "detail",
     "measurement", "64 bytes", ""),
    ("A single oversized message was all it took to overwhelm it.", 4, "trigger",
     "", "", ""),
    # packet_path_trace — the malicious packet's journey
    ("The malicious packet travelled from the attacker's machine, through the "
     "ISP router, across the exchange, and straight into the exposed server.",
     4, "packet", "packet_route", "THE PAYLOAD", ""),
    ("From there, it didn't stop.", 3, "pivot", "", "", ""),
    # exploit_chain — the kill chain
    ("The attack unfolded in stages: first reconnaissance, then the breach, then "
     "privilege escalation, and finally the quiet exfiltration of data.", 5,
     "chain", "kill_chain", "THE KILL CHAIN", ""),
    ("Millions of devices became part of a botnet their owners never knew "
     "existed.", 4, "botnet", "", "", ""),
    ("Investigators raced to trace the damage before it spread further.", 4,
     "race", "", "", ""),
    # footage_object_callout — point at the component
    ("Notice the small wireless chip on the board — that unassuming component "
     "was the doorway the attackers walked through.", 3, "callout",
     "object_callout", "", ""),
    ("Within weeks, an emergency patch was pushed to every device that could "
     "still receive one.", 4, "patch", "", "", ""),
    ("But millions of older devices, long abandoned by their makers, stayed "
     "vulnerable — and many still are today.", 4, "legacy", "", "", ""),
    ("The lesson was simple and uncomfortable: in a connected world, the weakest "
     "line of code anywhere can become everyone's problem.", 4, "thesis",
     "statement", "THE WEAKEST LINE", ""),
]


def _kw(nar, gt):
    kws = []
    if gt and not gt.isupper():
        kws.append(gt.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    for w in _re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in {"through", "around", "across", "before", "every", "their", "would"} \
                and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:5]


def build_script():
    scenes = []
    for i, (nar, inten, role, gk, gt, gb) in enumerate(SCENES):
        scenes.append({"narration": nar, "keywords": _kw(nar, gt), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": gk, "graphic_text": gt,
                       "graphic_body": gb})
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in scenes)
    return {"title": TITLE, "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    from vidlore.motion_graphics import director as mgdir
    cfg = load_config()
    print(f"FAL configured: {'yes' if (os.environ.get('FAL_KEY') or getattr(cfg,'fal_key','')) else 'no'}", flush=True)
    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt=PROMPT, fmt="documentary", duration="6-8",
                  theme="modern", captions=True, background="auto", extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script()
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    mg = [{"index": i, "role": s["role"], "graphic_kind": s["graphic_kind"],
           "intensity": s["intensity"], "narration": s["narration"],
           "emphasis": "", "assets": {}} for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.40)
    print("DIRECTOR_FIRED " + json.dumps([(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
