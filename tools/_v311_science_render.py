#!/usr/bin/env python3
"""V3.1.1 SCIENCE — real portal-equivalent render validating labeled_cross_section.
"How an Electron Microscope Reveals Objects Smaller Than Light Can See" through the
SHARED pipeline. The cross-section beat is justified NATURALLY by a component-list
sentence ("consists of an electron gun, magnetic lenses, a specimen stage, and a
detector"). AI VIDEO OFF; fal stills allowed; footage-first.
"""
import hashlib, json, os, re as _re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"VIDLORE_AIMG": "1", "VIDLORE_SUBJECT_FLOOR": "1",
             "VIDLORE_MOTION_GRAPHICS": "1", "VIDLORE_STOCK_FILMABLE": "1",
             "VIDLORE_OVERLAY_RESTRAINT": "1", "VIDLORE_AI_VIDEO": "0",
             "VIDLORE_REAL_PERSON": "1", "VIDLORE_WIKIMEDIA": "1",
             "VIDLORE_VISUAL_RELEVANCE": "1", "VIDLORE_VISUAL_RELEVANCE_CACHE": "0",
             "VIDLORE_FAL_IMAGE_BUDGET_MODE": "quality_first",
             # modest density so the cross-section earns a slot without a slideshow
             "VIDLORE_MG_DENSITY": "0.4",
             "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

TITLE = "How an Electron Microscope Reveals Objects Smaller Than Light Can See"
NICHE = "science"
PROMPT = ("How an electron microscope works — trading light for electrons to see "
          "structures far smaller than any light microscope ever could.")

# (narration, intensity, role)
SCENES = [
    ("Every light microscope hits the same wall: it can never see something "
     "smaller than a single wavelength of light.", 4, "hook"),
    ("To go smaller, scientists reached for electrons — particles whose "
     "wavelength is thousands of times shorter than light.", 3, "context"),
    # the cross-section beat — a natural component list justifies labeled_cross_section
    ("Inside, an electron microscope consists of an electron gun, magnetic "
     "lenses, a specimen stage, and a detector.", 3, "explain"),
    ("The electron gun fires a tightly focused beam straight down the column.",
     4, "turn"),
    ("Magnetic lenses bend and squeeze that beam, exactly as glass lenses bend "
     "ordinary light.", 3, "context"),
    ("When the beam strikes the specimen, the detector records how each electron "
     "scatters away.", 4, "proof"),
    ("The result is an image at magnifications light could never reach — down to "
     "the scale of individual atoms.", 4, "proof"),
    ("By trading light for electrons, we learned to see a world that was always "
     "there, simply too small for light to show us.", 4, "thesis"),
]


def _kw(nar):
    kws = _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", nar or "")
    for w in _re.findall(r"\b[a-z]{5,}\b", (nar or "").lower()):
        if w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        if k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out[:5]


def build_script():
    scenes = []
    for i, (nar, inten, role) in enumerate(SCENES):
        scenes.append({"narration": nar, "keywords": _kw(nar), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": "", "graphic_text": "",
                       "graphic_body": ""})
    return {"title": TITLE, "source_sha256": hashlib.sha256(
        (TITLE + "".join(s["narration"] for s in scenes)).encode()).hexdigest(),
        "scenes": scenes}


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    from vidlore.motion_graphics import director as mgdir
    cfg = load_config()
    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt=PROMPT, fmt="documentary", duration="6-8",
                  theme="modern", captions=True, background="auto",
                  extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script()
    body = TITLE + "\n\n" + "\n\n".join(s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    mg = [{"index": i, "role": s["role"], "graphic_kind": "", "intensity": s["intensity"],
           "narration": s["narration"], "emphasis": "", "assets": {}}
          for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.4)
    print("DIRECTOR_FIRED " + json.dumps(
        [(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s "
          f"video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
