#!/usr/bin/env python3
"""V3.3.2 STEP 8 — short (~60-90s) business render proving the centralized
factual guard end-to-end:

  * EXPLICIT-DATA scenes (real market-share / revenue / ranking / dated event)
    keep their naturally-emitted cards, values preserved exactly.
  * VAGUE / hallucination-bait scenes carry INJECTED fabricated legacy cards
    (donut 67%, vertical_bar_chart, timeline, comparison) — exactly the
    pre-V3.3.2 fabrication tendency — and the pipeline factual guard must DROP
    them to footage-only and log each rejection in the manifest.

AI-generated video OFF; CPU relevance (stable). Heavy MP4 deleted by the
caller after proof extraction.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {"VIDLORE_AIMG": "1", "VIDLORE_MOTION_GRAPHICS": "1",
             "VIDLORE_STOCK_FILMABLE": "1", "VIDLORE_OVERLAY_RESTRAINT": "1",
             "VIDLORE_AI_VIDEO": "0", "VIDLORE_VISUAL_RELEVANCE": "1",
             "VIDLORE_VR_ACCELERATOR": "cpu",
             "VIDLORE_FAL_IMAGE_BUDGET_MODE": "quality_first",
             "VIDLORE_MG_DENSITY": "0.5", "VIDLORE_EDITORIAL_QA": "1"}.items():
    os.environ.setdefault(k, v)

TITLE = "Standard Oil — Facts and Fabrications"
NICHE = "business"

# (narration, intensity, role, kind:  "explicit" keeps natural emission,
#  "bait:<forced_kind>|<text>|<body>" injects a fabricated legacy card.)
SCENES = [
    ("After the Civil War, one company set out to control American oil.",
     3, "hook", "explicit"),
    ("By 1880, Standard Oil controlled 90 percent of the nation's refining; "
     "the independents held just 10 percent.", 4, "proof", "explicit"),
    ("Its revenue climbed from $12 million in 1870 to $800 million by 1900.",
     4, "context", "explicit"),
    ("In 1911, the Supreme Court split the trust into 34 separate companies.",
     4, "turn", "explicit"),
    # ---- vague / bait scenes carry INJECTED fabricated legacy cards ----
    ("By then it controlled most of the market.",
     3, "context", "bait:donut_chart|67%|67|MARKET SHARE"),
    ("Production expanded dramatically over the following decades.",
     3, "escalation", "bait:vertical_bar_chart|OUTPUT|1870|10|;1900|90|"),
    ("Many years later, the empire finally fell.",
     4, "reveal", "bait:timeline|KEY EVENTS|events=1865:Rise|1911:Fall"),
    ("It faced several major competitors who soon disappeared.",
     4, "climax", "bait:comparison|RIVALS|EMPIRE|90%;;RIVALS|10%"),
    ("History remembers it as both a model of efficiency and a warning "
     "against monopoly.", 3, "thesis", "explicit"),
]


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.script_gen import Scene, _apply_editor_decisions
    from vidlore.pipeline import run_dir_for, render_from_script, _slug
    cfg = load_config()
    out = ROOT / "output"
    brief = Brief(title=TITLE, prompt="Standard Oil's monopoly — facts vs fabrication.",
                  fmt="documentary", duration="1-2", theme="modern", captions=True,
                  background="auto", extra={"niche": NICHE})
    run_dir = run_dir_for(brief, out)
    run_dir.mkdir(parents=True, exist_ok=True)

    scenes = [Scene(index=i, narration=n, keywords=[], visual="", intensity=inten,
                    emphasis="", shot_type="", role=r, graphic_kind="",
                    graphic_text="", graphic_body="")
              for i, (n, inten, r, _) in enumerate(SCENES)]
    print("editor LLM emitting graphic kinds ...", flush=True)
    _apply_editor_decisions(TITLE, scenes, cfg)

    # Inject the fabricated legacy cards onto the vague/bait scenes.
    injected = []
    for s, (_, _, _, tag) in zip(scenes, SCENES):
        if tag.startswith("bait:"):
            body = tag[len("bait:"):]
            kind, text, gbody = (body.split("|", 2) + ["", ""])[:3]
            s.graphic_kind, s.graphic_text, s.graphic_body = kind, text, gbody
            injected.append((s.index, kind))
    emitted = [(s.index, s.graphic_kind) for s in scenes if s.graphic_kind]
    print("EMITTED " + json.dumps(emitted), flush=True)
    print("INJECTED_FABRICATIONS " + json.dumps(injected), flush=True)

    script = {"title": TITLE, "source_sha256": hashlib.sha256(
        (TITLE + "".join(s.narration for s in scenes)).encode()).hexdigest(),
        "scenes": [{"narration": s.narration, "keywords": s.keywords,
                    "visual": s.visual, "intensity": s.intensity,
                    "emphasis": s.emphasis, "shot_type": s.shot_type or "medium",
                    "role": s.role, "graphic_kind": s.graphic_kind,
                    "graphic_text": s.graphic_text, "graphic_body": s.graphic_body}
                   for s in scenes]}
    (run_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (run_dir / "script.txt").write_text(
        TITLE + "\n\n" + "\n\n".join(s.narration for s in scenes), encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"RUN_DIR {run_dir}", flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, out, keep_work=True, run_dir=run_dir)
    vid = run_dir / f"{_slug(TITLE)}.mp4"
    print(f"RENDER_DONE wall={round(time.time() - t0, 1)}s "
          f"video_exists={vid.exists()} path={vid}", flush=True)


if __name__ == "__main__":
    main()
