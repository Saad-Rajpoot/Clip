#!/usr/bin/env python3
"""Cost-instrumented REAL render of the Rockefeller / Standard Oil business
sample through the full Vidlore pipeline with the Motion-Graphics engine ON.

Tracks every paid API call (fal.ai image gen) and counts free-provider usage
(Pexels stock, pollinations AI, edge-tts). Writes api_cost_breakdown.json.

Env (set by the caller):
  VIDLORE_MOTION_GRAPHICS=1   VIDLORE_MG_DENSITY=0.45
  VIDLORE_REUSE_SCRIPT_JSON=force   VIDLORE_AIMG=1   VIDLORE_TTS_BACKEND=legacy
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TITLE = "John D. Rockefeller: The Man Who Built Standard Oil"
OUTDIR = ROOT / "research/motion_graphics_qa/rockefeller_business_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── cost / call instrumentation ──────────────────────────────────────────
_LOCK = threading.Lock()
COST = {
    "fal_ai_image": {"provider": "fal.ai", "model": "fal-ai/flux/schnell",
                     "calls": 0, "ok": 0, "fail": 0, "seconds": 0.0,
                     "unit_usd_est": 0.003, "paid": True, "notes": []},
    "pexels": {"provider": "Pexels", "calls": 0, "ok": 0, "fail": 0,
               "paid": False, "notes": ["free tier"]},
    "pollinations_ai": {"provider": "pollinations.ai", "calls": 0,
                        "paid": False, "notes": ["free AI image fallback"]},
    "edge_tts": {"provider": "Microsoft edge-tts", "paid": False,
                 "notes": ["free neural TTS"]},
    "anthropic_llm": {"provider": "Anthropic", "calls": 0, "paid": True,
                      "notes": ["SKIPPED — script force-reused, no LLM"]},
}


def _instrument():
    import vidlore.footage as F

    # paid: fal.ai image generation
    if hasattr(F, "_fal_image"):
        _orig_fal = F._fal_image

        def _wrapped_fal(*a, **kw):
            t0 = time.time()
            ok = False
            try:
                ok = _orig_fal(*a, **kw)
                return ok
            finally:
                with _LOCK:
                    COST["fal_ai_image"]["calls"] += 1
                    COST["fal_ai_image"]["seconds"] += time.time() - t0
                    if ok:
                        COST["fal_ai_image"]["ok"] += 1
                    else:
                        COST["fal_ai_image"]["fail"] += 1
        F._fal_image = _wrapped_fal

    # free: Pexels HTTP (count search/download requests if a helper exists)
    for fn_name in ("_pexels_search", "_pexels_get", "pexels_search"):
        if hasattr(F, fn_name):
            _orig = getattr(F, fn_name)

            def _mk(o):
                def _w(*a, **kw):
                    with _LOCK:
                        COST["pexels"]["calls"] += 1
                    try:
                        r = o(*a, **kw)
                        with _LOCK:
                            COST["pexels"]["ok"] += 1
                        return r
                    except Exception:
                        with _LOCK:
                            COST["pexels"]["fail"] += 1
                        raise
                return _w
            setattr(F, fn_name, _mk(_orig))
            break


def main():
    from vidlore.config import load_config
    from vidlore.brief import Brief
    from vidlore.pipeline import render_from_script, run_dir_for

    cfg = load_config()
    _instrument()

    brief = Brief(title=TITLE,
                  prompt="Rockefeller/Standard Oil premium business documentary",
                  fmt="documentary", duration="6-8", theme="history",
                  captions=False, background="auto", extra={"niche": "business"})
    out = ROOT / "output"
    run_dir = run_dir_for(brief, out)
    print(f"RUN_DIR {run_dir}", flush=True)

    t0 = time.time()
    res = render_from_script(brief, cfg, out, keep_work=True)
    wall = time.time() - t0

    # post-render: count footage by provider from the produced files
    vid = run_dir / f"{run_dir.name}.mp4"
    work_glob = sorted(run_dir.glob("work_*"))
    fal_files = aimg_files = pex_files = 0
    for w in work_glob:
        for p in w.rglob("*"):
            n = p.name
            if n.startswith("fal_"):
                fal_files += 1
            elif n.startswith(("aimg_", "ov_")):
                aimg_files += 1
            elif n.startswith(("pexels_", "px_")) or (p.suffix == ".mp4" and "clip" not in n):
                pex_files += 1
    COST["pollinations_ai"]["files_present"] = aimg_files
    COST["fal_ai_image"]["files_present"] = fal_files
    COST["fal_ai_image"]["usd_total_est"] = round(
        COST["fal_ai_image"]["ok"] * COST["fal_ai_image"]["unit_usd_est"], 4)

    total_paid = COST["fal_ai_image"]["usd_total_est"]
    summary = {
        "title": TITLE,
        "video": str(vid),
        "video_exists": vid.exists(),
        "video_seconds": getattr(res, "video_seconds", None),
        "render_wall_seconds": round(wall, 1),
        "providers": COST,
        "total_paid_usd_est": round(total_paid, 4),
        "free_providers": ["Pexels stock", "pollinations.ai", "edge-tts",
                           "Wikimedia/web-image"],
        "notes": [
            "LLM skipped (script force-reused) → $0 Anthropic.",
            "edge-tts free → $0 narration.",
            "Pexels free tier → $0 footage.",
            "Only paid line = fal.ai flux/schnell images actually generated "
            "(ok count × ~$0.003/img est).",
        ],
    }
    (OUTDIR / "api_cost_breakdown.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== COST BREAKDOWN ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nRENDER_DONE wall={wall:.1f}s video={vid.exists()} "
          f"fal_ok={COST['fal_ai_image']['ok']} "
          f"paid_usd_est={total_paid:.4f}", flush=True)


if __name__ == "__main__":
    main()
