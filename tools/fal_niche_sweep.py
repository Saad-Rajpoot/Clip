"""SEVEN-NICHE PRODUCTION VALIDATION SWEEP for the fal-primary provider policy.

Renders one original short documentary per niche through the REAL pipeline with
the new provider policy (fal.ai PRIMARY, no fixed ceiling, Pollinations
emergency-only, every fal still validated + retried) and audits the per-niche
asset mix: real stock / archive-web / fal stills / same-scene fal reuse /
Pollinations emergency / premium cards / emergency text slides / AI video. Fails
loud if any niche ships excess text slides or irrelevant footage.

    python3 tools/fal_niche_sweep.py spy            # one niche
    python3 tools/fal_niche_sweep.py all            # all 7 niches, sequential

Niches: spy, crime (true-crime), business, history, geopolitics, science,
technology. Each is an ORIGINAL script (tools/multiniche_validate.py SAMPLES).
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── PROVIDER POLICY (user, 2026-06-02) ─ same flags as the proof render ──────
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")                 # hard rule: OFF
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
# Rerender freshness: VR score cache OFF so every asset is re-scored with the
# NEW designed-graphic gate (a cached score predates graphic_dom and would
# bypass the gate). MG cards re-render fresh because OUT is a new run dir.
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "0")
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ.setdefault("VIDLORE_FAL_MAX_IMAGES", "0")          # 0 ⇒ no ceiling
os.environ.setdefault("VIDLORE_FAL_MAX_ATTEMPTS", "4")
os.environ.setdefault("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES", "0")
os.environ.setdefault("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10")
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")

from tools.multiniche_validate import SAMPLES, build_script        # noqa: E402
from vidlore.config import load_config                             # noqa: E402
from vidlore.brief import Brief                                    # noqa: E402
from vidlore.pipeline import run_dir_for, render_from_script       # noqa: E402

OUT = ROOT / os.environ.get("VIDLORE_SWEEP_OUT", "output")
REPORT = ROOT / os.environ.get("VIDLORE_SWEEP_REPORT",
                               "research/editorial_qa/fal_niche_sweep")
REPORT.mkdir(parents=True, exist_ok=True)
NICHES = ["spy", "crime", "business", "history", "geopolitics",
          "science", "technology"]


def _cat(b, is_video):
    b = (b or "").lower()
    if b.startswith("fal_"):
        return "fal_still"
    if b.startswith("aimg_"):
        return "pollinations_still"
    if b.startswith(("vrslide_", "slide_")):
        return "text_slide"
    if b.startswith(("wiki_", "loc_", "ia_", "ov_")):
        return "archive_web_image"
    if b.startswith(("card_", "mg_", "tmpl_")):
        return "premium_card"
    if is_video or b.endswith(".mp4"):
        return "stock_video"
    return "other"


def audit_run(run_dir: Path) -> dict:
    rd = Path(run_dir)
    # manifest + AI_STILL_LOG live under workdir.parent (a work_* subdir)
    man = next(rd.rglob("ASSET_DECISION_MANIFEST.json"), None)
    ail = next(rd.rglob("AI_STILL_LOG.json"), None)
    mix, outcomes = Counter(), Counter()
    if man:
        d = json.loads(man.read_text())
        for be in d.get("beats", []):
            oc = be.get("outcome") or ""
            outcomes[oc] += 1
            fin = be.get("final") or be.get("source") or ""
            if oc.startswith("replaced:subject-slide"):
                mix["text_slide"] += 1
            elif oc == "replaced:ai-still":
                mix["fal_still"] += 1
            elif oc == "replaced:ai-still-reuse":
                mix["fal_still_scene_reuse"] += 1
            elif oc.startswith("replaced:better-stock"):
                mix["stock_video"] += 1
            else:
                mix[_cat(fin, be.get("is_video", False))] += 1
    ai = json.loads(ail.read_text()).get("summary", {}) if ail else {}
    total = sum(mix.values()) or 1
    slides = mix.get("text_slide", 0)
    return {"asset_mix": dict(mix), "outcomes": dict(outcomes),
            "ai_still": ai, "slide_ratio_pct": round(100.0 * slides / total, 1),
            "concrete_beats": total}


import re as _re


def _derive_keywords(narration: str, graphic_text: str, title: str) -> list:
    """Portal-equivalent keyword derivation: production scripts carry LLM
    keywords. Derive concrete subject keywords from the scene's narration +
    explicit graphic subject + the video title's proper nouns, so the rerender
    exercises the keyword path (footage relevance + title-anchor) the way a real
    portal render would — instead of the artificial empty-keyword test."""
    kws = []
    if graphic_text:
        kws.append(graphic_text.strip())
    # multi-word proper nouns (Eli Cohen, Standard Oil, Broad Street)
    kws += _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", narration or "")
    # concrete long lowercase nouns as fallback subject words
    _stop = {"which", "their", "there", "where", "would", "could", "should",
             "about", "these", "those", "after", "before", "every", "while",
             "first", "single", "nation", "people", "years", "world"}
    for w in _re.findall(r"\b[a-z]{5,}\b", (narration or "").lower()):
        if w not in _stop and w not in [k.lower() for k in kws]:
            kws.append(w)
    seen, out = set(), []
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
    return out[:5]


def run_niche(sid: str) -> dict:
    sample = SAMPLES[sid]
    cfg = load_config()
    brief = Brief(title=sample["title"], prompt=sample["prompt"],
                  fmt="documentary", duration="6-8", theme=sample["theme"],
                  captions=True, background="auto",
                  extra={"niche": sample["niche"]})
    run_dir = run_dir_for(brief, OUT)
    run_dir.mkdir(parents=True, exist_ok=True)
    script = build_script(sample)
    # portal-equivalent: populate per-scene keywords (production always has them)
    # unless VIDLORE_SWEEP_WEAK_KW=1 → the weak-keyword fallback test (empty kw).
    if os.environ.get("VIDLORE_SWEEP_WEAK_KW") != "1":
        for _s in script["scenes"]:
            _s["keywords"] = _derive_keywords(
                _s.get("narration", ""), _s.get("graphic_text", ""),
                sample["title"])
    body = sample["title"] + "\n\n" + "\n\n".join(
        s["narration"] for s in script["scenes"])
    (run_dir / "script.txt").write_text(body, encoding="utf-8")
    (run_dir / "script.json").write_text(json.dumps(script, indent=2),
                                         encoding="utf-8")
    (run_dir / "variants.json").unlink(missing_ok=True)
    print(f"\n===== NICHE {sid} -> {run_dir.name} =====", flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, OUT, keep_work=True, run_dir=run_dir)
    wall = round(time.time() - t0, 1)
    rep = audit_run(run_dir)
    rep.update({"niche": sid, "title": sample["title"],
                "run_dir": str(run_dir), "wall_s": wall})
    ai = rep["ai_still"]
    print(f"[{sid}] DONE wall={wall}s slides={rep['slide_ratio_pct']}% "
          f"mix={rep['asset_mix']} fal_gens={ai.get('fal_generations')} "
          f"fal_cost=${ai.get('fal_total_cost_usd_est')} "
          f"polli_emergency={ai.get('pollinations_emergency_used')}", flush=True)
    (REPORT / f"{sid}.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = NICHES if sid == "all" else [sid]
    results = []
    for n in targets:
        try:
            results.append(run_niche(n))
        except Exception as e:                                     # noqa: BLE001
            import traceback
            print(f"[{n}] FAILED: {e}\n{traceback.format_exc()}", flush=True)
            results.append({"niche": n, "error": str(e)})
    (REPORT / "SWEEP_REPORT.json").write_text(json.dumps(results, indent=2))
    print("\n===== SWEEP SUMMARY =====", flush=True)
    for r in results:
        if r.get("error"):
            print(f"  {r['niche']:12s} ERROR {r['error'][:60]}", flush=True)
        else:
            ai = r.get("ai_still", {})
            print(f"  {r['niche']:12s} slides={r['slide_ratio_pct']}% "
                  f"fal_gens={ai.get('fal_generations')} "
                  f"polli={ai.get('pollinations_emergency_used')} "
                  f"mix={r['asset_mix']}", flush=True)


if __name__ == "__main__":
    main()
