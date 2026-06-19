#!/usr/bin/env python3
"""V3.0.1 — act_chapter_card REAL-PIPELINE proof (+ live portrait ladder).

A lean ~60-90s Rockefeller biography rendered through the SHARED pipeline
(`render_from_script`). Three explicit ACT breaks (graphic_kind="chapter",
non-numeric titles) so `act_chapter_card` fires as short cinematic punctuation;
the cold open exercises the new portrait ladder live (real archival portrait).

AI VIDEO OFF; fal stills allowed; editorial-QA ON. Footage-first.
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
             # deliberate, documented density bump for a CHAPTER-STRUCTURED piece
             # so the act breaks survive the restraint cap (≈3-4 MG over 8 scenes,
             # still restrained — not a slideshow). Default biography density caps
             # at 2 and squeezed act_chapter out behind the portrait + verdict.
             "VIDLORE_MG_DENSITY": "0.45",
             "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

TITLE = "How Rockefeller Built His Oil Empire — A Life in Three Acts"
NICHE = "biography"
PROMPT = ("The life of John D. Rockefeller in three acts — early years, the rise "
          "of Standard Oil, and his legacy.")

# (narration, intensity, role, graphic_kind, graphic_text)
# Act-break scenes are NARRATED (the act title is the spoken line) so they
# survive TTS and carry the chapter title → act_chapter_card fires as the
# scene's graphic. Silent scenes get dropped by the pipeline (no narration).
SCENES = [
    # cold open — portrait_legend_reveal (real portrait via the new ladder)
    ("John D. Rockefeller was born in 1839, a quiet, careful boy obsessed with "
     "one thing above all else.", 4, "hook", "", ""),
    # ACT I chapter card (narrated act break)
    ("Act one. The early years.", 3, "context", "chapter", "ACT I — EARLY YEARS"),
    ("As a young clerk in Cleveland, he counted every cent and dreamed of "
     "something far larger.", 3, "context", "", ""),
    # ACT II chapter card (narrated act break)
    ("Act two. The rise.", 3, "turn", "chapter", "ACT II — THE RISE"),
    ("His method was simple and merciless: control the railroads, and you "
     "control the oil.", 4, "escalation", "", ""),
    ("One by one, his rivals were bought, absorbed, or quietly crushed beneath "
     "Standard Oil.", 4, "proof", "", ""),
    # ACT III chapter card (narrated act break)
    ("Act three. The legacy.", 3, "context", "chapter", "ACT III — LEGACY"),
    ("History remembers him as both a ruthless monster and a generous genius — "
     "the man who built modern business.", 4, "thesis", "", ""),
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
    for i, (nar, inten, role, gk, gt) in enumerate(SCENES):
        scenes.append({"narration": nar, "keywords": _kw(nar), "visual": "",
                       "intensity": inten, "emphasis": "",
                       "shot_type": "wide" if i % 3 == 0 else "medium",
                       "role": role, "graphic_kind": gk, "graphic_text": gt,
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
    body = TITLE + "\n\n" + "\n\n".join(
        s["narration"] for s in script["scenes"] if s["narration"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    mg = [{"index": i, "role": s["role"], "graphic_kind": s["graphic_kind"],
           "intensity": s["intensity"], "narration": s["narration"],
           "emphasis": "", "assets": ({"title": s["graphic_text"]}
                                       if s["graphic_kind"] == "chapter" else {})}
          for i, s in enumerate(script["scenes"])]
    seed = int(hashlib.sha1(TITLE.encode()).hexdigest()[:8], 16) % 100000
    dec = mgdir.plan(mg, niche=NICHE, seed=seed, density=0.5)
    print("DIRECTOR_FIRED " + json.dumps(
        [(d.scene_index, d.primitive) for d in dec if d.primitive]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s "
          f"video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
