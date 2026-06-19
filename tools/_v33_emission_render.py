#!/usr/bin/env python3
"""V3.3 STEP 8 — short business render whose MG cards are emitted by the REAL
editor LLM (natural emission), then assembled through the shared pipeline. Proves
the newly-unlocked vocabulary appears in a real MP4, fabrication-free, footage-first.
AI-video OFF; CPU relevance (stable). ~10 scenes (~2-3 min)."""
import hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"VIDLORE_AIMG": "1", "VIDLORE_MOTION_GRAPHICS": "1",
             "VIDLORE_STOCK_FILMABLE": "1", "VIDLORE_OVERLAY_RESTRAINT": "1",
             "VIDLORE_AI_VIDEO": "0", "VIDLORE_VISUAL_RELEVANCE": "1",
             "VIDLORE_VR_ACCELERATOR": "cpu", "VIDLORE_FAL_IMAGE_BUDGET_MODE": "quality_first",
             "VIDLORE_MG_DENSITY": "0.45", "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

TITLE = "The Rise of Standard Oil"
NICHE = "business"
# real-data documentary narration (every figure is the REAL one the card will show)
SCENES = [
    ("In the years after the Civil War, one company set out to control American oil.", 4, "hook"),
    ("By 1880, Standard Oil controlled ninety percent of the nation's refining; the independents held just ten percent.", 4, "proof"),
    ("Its method never varied: buy the refinery, undercut the price, starve the rival, then absorb it.", 4, "turn"),
    ("The empire's revenue split three ways: half was reinvested, thirty percent paid dividends, twenty percent went to reserves.", 3, "context"),
    ("Above it all sat a pyramid — one holding company above the regional firms above the refineries.", 3, "context"),
    ("Rivals who refused to sell were broken within a single year.", 4, "escalation"),
    ("By 1882, the Standard Oil Trust was the most powerful company on earth.", 4, "climax"),
    ("In 1911, the Supreme Court ordered the trust dissolved into thirty-four separate companies.", 4, "turn"),
    ("History remembers it as both the model of efficiency and the warning against monopoly.", 4, "thesis"),
]


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.script_gen import Scene, _apply_editor_decisions
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    cfg = load_config()
    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt="How Standard Oil built and lost a monopoly.",
                  fmt="documentary", duration="6-8", theme="modern", captions=True,
                  background="auto", extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out); run_dir.mkdir(parents=True, exist_ok=True)
    # REAL editor-LLM emission of graphic kinds from the narration
    scenes = [Scene(index=i, narration=n, keywords=[], visual="", intensity=inten,
                    emphasis="", shot_type="", role=r, graphic_kind="",
                    graphic_text="", graphic_body="") for i, (n, inten, r) in enumerate(SCENES)]
    print("editor LLM emitting graphic kinds ...", flush=True)
    _apply_editor_decisions(TITLE, scenes, cfg)
    emitted = [(s.index, s.graphic_kind) for s in scenes if s.graphic_kind]
    print("EMITTED " + json.dumps(emitted), flush=True)
    script = {"title": TITLE, "source_sha256": hashlib.sha256(
        (TITLE + "".join(s.narration for s in scenes)).encode()).hexdigest(),
        "scenes": [{"narration": s.narration, "keywords": s.keywords, "visual": s.visual,
                    "intensity": s.intensity, "emphasis": s.emphasis,
                    "shot_type": s.shot_type or "medium", "role": s.role,
                    "graphic_kind": s.graphic_kind, "graphic_text": s.graphic_text,
                    "graphic_body": s.graphic_body} for s in scenes]}
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "script.txt").write_text(TITLE + "\n\n" + "\n\n".join(s.narration for s in scenes), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time()-t0,1)}s video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
