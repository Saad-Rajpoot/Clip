#!/usr/bin/env python3
"""V3.0 BIOGRAPHY Sample A — "How Rockefeller Built the Most Powerful Oil Empire
in America". Real portal-equivalent render through the SHARED pipeline. Exercises
the biography family (portrait_legend_reveal / wealth_arc_counter / relationship_
roster / life_milestone_spine / verdict_duality_card / era_stamp_overlay) via the
real director on a biography-niche doc. Footage-first. AI VIDEO OFF; fal stills
allowed (real archival-portrait fallback for concrete person beats).
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
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")

TITLE = "How Rockefeller Built the Most Powerful Oil Empire in America"
NICHE = "biography"
PROMPT = ("The life of John D. Rockefeller — from a poor boy in upstate New York "
          "to the founder of Standard Oil and the world's first billionaire")

# (narration, intensity, role, graphic_kind, graphic_text, graphic_body)
SCENES = [
    # portrait_legend_reveal — the legend cold-open (named subject, born + year)
    ("John D. Rockefeller was born in 1839, the son of a travelling con-man in "
     "upstate New York.", 4, "hook", "", "", ""),
    ("As a boy he was quiet, careful, and obsessed with one thing above all "
     "else: money.", 3, "context", "", "", ""),
    # era_stamp_overlay — orient the era/place
    ("By 1859, in the booming oil town of Cleveland, the young man saw his "
     "chance.", 3, "context", "", "", ""),
    ("Oil was chaos — hundreds of small refiners, all undercutting each other "
     "into the ground.", 4, "problem", "", "", ""),
    # relationship_roster — the cast (rival + partner)
    ("Alongside his loyal partner Henry Flagler and against his great rival Tom "
     "Scott, Rockefeller began to build.", 3, "context", "", "", ""),
    ("His method was simple and merciless: control the railroads, and you "
     "control the oil.", 4, "turn", "", "", ""),
    # life_milestone_spine — the life arc across years
    ("Throughout his life, from 1859 to 1870 to the breakup of 1911, his empire "
     "only ever grew.", 3, "context", "", "", ""),
    ("One by one, the exhausted competitors were bought, absorbed, or quietly "
     "crushed.", 4, "escalation", "", "", ""),
    # wealth_arc_counter — the fortune
    ("By 1916 his personal fortune reached nearly one billion dollars, making "
     "him the richest man who had ever lived.", 4, "proof", "", "", ""),
    ("To the public he was a robber baron; to his church he was a humble, "
     "generous man.", 3, "reaction", "", "", ""),
    ("In his final decades he gave away more money than almost anyone in "
     "history.", 3, "resolution", "", "", ""),
    # verdict_duality_card — the finale judgment
    ("History remembers him as both a ruthless monster and a generous genius — "
     "for better or worse, the man who built modern business.", 4, "thesis",
     "", "", ""),
]


def _kw(nar, gt):
    kws = []
    if gt and not gt.isupper():
        kws.append(gt.strip())
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    _stop = {"through", "around", "across", "before", "every", "their", "would",
             "alongside", "against", "throughout", "himself"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
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
    return {"title": TITLE,
            "source_sha256": hashlib.sha256(
                (TITLE + "".join(s["narration"] for s in scenes)).encode()).hexdigest(),
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
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.42)
    print("DIRECTOR_FIRED " + json.dumps([(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
