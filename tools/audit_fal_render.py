"""Audit a fal-primary render: asset mix + AI-still cost + QA gate.

Reads a render's run_dir and reports the user's full final-audit checklist:
asset mix (stock / archive-web / fal stills / same-scene reuse / Pollinations /
cards / text slides / AI video), fal totals (generations, est cost, avg
latency, rejected, retried, cache hits), text-slide ratio, and the editorial-QA
gate. Pure reporting — never mutates the render.

    python3 tools/audit_fal_render.py <run_dir>
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _cat(basename: str, is_video: bool) -> str:
    b = (basename or "").lower()
    if b.startswith("fal_"):
        return "fal_still"
    if b.startswith("aimg_"):
        return "pollinations_still"
    if b.startswith("vrslide_") or b.startswith("slide_"):
        return "text_slide"
    if b.startswith(("wiki_", "loc_", "ia_", "ov_")):
        return "archive_web_image"
    if b.startswith(("card_", "mg_", "tmpl_")):
        return "premium_card"
    if is_video or b.endswith(".mp4"):
        return "stock_video"
    return "other"


def main(run_dir: str):
    rd = Path(run_dir)
    print(f"=== FAL-PRIMARY RENDER AUDIT: {rd.name} ===\n")

    # ---- asset mix from ASSET_DECISION_MANIFEST (concrete VR-scored beats) --
    man = _load(rd / "ASSET_DECISION_MANIFEST.json")
    mix = Counter()
    outcomes = Counter()
    if man and isinstance(man.get("beats"), list):
        for be in man["beats"]:
            oc = (be.get("outcome") or "")
            outcomes[oc] += 1
            final = be.get("final") or be.get("source") or ""
            if oc.startswith("replaced:subject-slide"):
                mix["text_slide"] += 1
            elif oc in ("replaced:ai-still",):
                mix["fal_still"] += 1
            elif oc in ("replaced:ai-still-reuse",):
                mix["fal_still_scene_reuse"] += 1
            elif oc.startswith("replaced:better-stock"):
                mix["stock_video"] += 1
            else:
                mix[_cat(final, be.get("is_video", False))] += 1
        print(f"VR-scored concrete beats: {sum(outcomes.values())}")
        print("  outcomes:", dict(outcomes))
    else:
        print("(no ASSET_DECISION_MANIFEST.json)")

    # text slides actually emitted as files (catch any not in manifest)
    work = list(rd.rglob("vrslide_*.png")) + list(rd.rglob("slide_*.png"))
    print(f"\nasset mix (concrete beats): {dict(mix)}")
    print(f"text-slide PNG files on disk: {len(work)}")

    # ---- AI-still telemetry (authoritative for fal cost/latency) -----------
    ail = _load(rd / "AI_STILL_LOG.json")
    if ail:
        s = ail.get("summary", {})
        print("\n--- AI-STILL LOG (fal primary) ---")
        for k in ("fal_generations", "fal_accepted", "fal_rejected_images",
                  "fal_retried_beats", "fal_cache_hits", "fal_avg_latency_s",
                  "fal_total_cost_usd_est", "cost_per_image_usd_est",
                  "pollinations_emergency_used", "total_attempts"):
            print(f"  {k}: {s.get(k)}")
    else:
        print("\n(no AI_STILL_LOG.json)")

    # ---- text-slide ratio --------------------------------------------------
    rm = _load(rd / "render_meta.json")
    total_beats = None
    if rm:
        total_beats = (len(rm.get("scene_starts", []))
                       if rm.get("scene_starts") else None)
    slides = mix.get("text_slide", 0)
    denom = sum(mix.values()) or 1
    print(f"\ntext-slide ratio (concrete beats): {slides}/{denom} = "
          f"{100.0*slides/denom:.1f}%  (cap 10%)")

    # ---- editorial-QA gate -------------------------------------------------
    qa = _load(rd / "EDITORIAL_QA_REPORT.json") or _load(
        rd / "editorial_qa_report.json")
    if qa:
        print("\n--- EDITORIAL QA ---")
        print("  gate:", qa.get("gate"))
        print("  by_severity:", (qa.get("summary") or {}).get("by_severity"))
    autolog = _load(rd / "EDITORIAL_QA_AUTOFIX_LOG.json")
    if autolog:
        print("  autofix final_gate:", autolog.get("final_gate"),
              "passes:", len(autolog.get("passes", [])))

    print("\n=== END AUDIT ===")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "output/_v15_falprimary/the-1860s-secret--how-to-end-garden-pests-permanen")
