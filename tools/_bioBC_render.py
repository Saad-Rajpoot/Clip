#!/usr/bin/env python3
"""V3.0 BIOGRAPHY Samples B + C — real portal-equivalent renders through the
SHARED pipeline. Usage: `_bioBC_render.py B`  (modern founder, Steve Jobs)
or `_bioBC_render.py C`  (historical figure, Napoleon). Footage-first, AI VIDEO
OFF, fal stills allowed. Avoids unsupported accusations — narration sticks to
well-established public history.
"""
import hashlib, json, os, re as _re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"VIDLORE_AIMG": "1", "VIDLORE_SUBJECT_FLOOR": "1",
             "VIDLORE_MOTION_GRAPHICS": "1", "VIDLORE_STOCK_FILMABLE": "1",
             "VIDLORE_OVERLAY_RESTRAINT": "1", "VIDLORE_AI_VIDEO": "0",
             "VIDLORE_VISUAL_RELEVANCE": "1", "VIDLORE_VISUAL_RELEVANCE_CACHE": "0",
             "VIDLORE_FAL_IMAGE_BUDGET_MODE": "quality_first",
             "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

SAMPLES = {
    "B": {
        "title": "How Steve Jobs Turned a Small Computer Company Into an Empire",
        "niche": "biography",
        "prompt": ("The life of Steve Jobs — from a college dropout in a garage "
                   "to the co-founder of Apple and one of tech's great showmen"),
        "scenes": [
            ("Steve Jobs was born in 1955 and adopted by a working-class family "
             "in California.", 4, "hook"),
            ("He was restless, intense, and convinced from an early age that he "
             "would change the world.", 3, "context"),
            ("By 1976, in his parents' garage, the dream began to take shape.",
             3, "context"),
            ("Alongside his brilliant partner Steve Wozniak, and later against "
             "his rival Bill Gates, Jobs set out to build.", 3, "context"),
            ("Their first machine was crude, but it carried a radical idea: a "
             "computer for ordinary people.", 4, "turn"),
            ("Across his career, from 1976 to his exile in 1985 to his return in "
             "1997, the story kept turning.", 3, "context"),
            ("Pushed out of his own company, he spent a decade in the "
             "wilderness, learning patience.", 4, "escalation"),
            ("When he returned, Apple was near bankruptcy — and he rebuilt it "
             "into the most valuable company on Earth.", 4, "proof"),
            ("By 2011 Apple was worth more than 300 billion dollars, a figure "
             "few could have imagined.", 4, "proof"),
            ("To his admirers he was a visionary; to those who worked for him he "
             "could be impossible.", 3, "reaction"),
            ("He is remembered as both a demanding perfectionist and a genuine "
             "artist — the man who made technology feel human.", 4, "thesis"),
        ],
    },
    "C": {
        "title": "How Napoleon Rose From Soldier to Emperor",
        "niche": "history",
        "prompt": ("The rise of Napoleon Bonaparte — from a minor officer on "
                   "Corsica to Emperor of the French"),
        "scenes": [
            ("Napoleon Bonaparte was born in 1769 on the island of Corsica, far "
             "from the centres of power.", 4, "hook"),
            ("He was a quiet, bookish boy who devoured histories of the great "
             "generals of the past.", 3, "context"),
            ("By 1789, as revolution swept through France, the young officer saw "
             "his chance.", 3, "context"),
            ("Alongside his loyal marshals and against his great rival the Duke "
             "of Wellington, Napoleon began to climb.", 3, "context"),
            ("His genius was speed: he moved armies faster than anyone believed "
             "possible.", 4, "turn"),
            ("Across his life, from the siege of 1793 to his coup in 1799 to his "
             "fall in 1815, France was never still.", 3, "context"),
            ("One by one, the old kingdoms of Europe were beaten, bargained "
             "with, or swept aside.", 4, "escalation"),
            ("In 1804 he crowned himself Emperor, master of a continent at the "
             "height of his power.", 4, "proof"),
            ("At its peak his empire stretched across much of Europe, ruling "
             "tens of millions of people.", 4, "proof"),
            ("To some he was a liberator who spread new laws; to others a tyrant "
             "who drowned a generation in war.", 3, "reaction"),
            ("History remembers him as both a military genius and a ruinous "
             "tyrant — for better or worse, a man who remade Europe.", 4, "thesis"),
        ],
    },
}


def _kw(nar):
    kws = _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    _stop = {"through", "around", "across", "before", "every", "their", "would",
             "alongside", "against", "throughout"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:5]


def build_script(spec):
    scenes = []
    for i, (nar, inten, role) in enumerate(spec["scenes"]):
        scenes.append({"narration": nar, "keywords": _kw(nar), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": "", "graphic_text": "",
                       "graphic_body": ""})
    return {"title": spec["title"],
            "source_sha256": hashlib.sha256(
                (spec["title"] + "".join(s["narration"] for s in scenes)).encode()).hexdigest(),
            "scenes": scenes}


def main():
    which = (sys.argv[1] if len(sys.argv) > 1 else "B").upper()
    spec = SAMPLES[which]
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    from vidlore.motion_graphics import director as mgdir
    cfg = load_config()
    print(f"SAMPLE {which} | FAL configured: {'yes' if (os.environ.get('FAL_KEY') or getattr(cfg,'fal_key','')) else 'no'}", flush=True)
    out = ROOT / "output"
    brief = Brief(title=spec["title"], prompt=spec["prompt"], fmt="documentary",
                  duration="6-8", theme="modern", captions=True, background="auto",
                  extra={"niche": spec["niche"]})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script(spec)
    body = spec["title"] + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    mg = [{"index": i, "role": s["role"], "graphic_kind": "", "intensity": s["intensity"],
           "narration": s["narration"], "emphasis": "", "assets": {}}
          for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(spec["title"].encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=spec["niche"], seed=seed, density=0.42)
    print("DIRECTOR_FIRED " + json.dumps([(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(spec['title'])}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
