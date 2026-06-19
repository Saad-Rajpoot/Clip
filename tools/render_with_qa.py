"""Render the Two Cheap Metals video through the autonomous editorial-QA loop.

This is the portal-style entrypoint that proves `qa_autofix.run_with_qa`
end-to-end: render → editorial QA scan → re-route blocking scenes (bump their
footage variant) → re-render only those scenes → re-scan → repeat until the
strict gate passes or the pass cap is hit. Output goes to a FRESH folder; prior
renders and snapshots are never overwritten.

    VIDLORE_EDITORIAL_QA_MAX_PASSES=<n> python3 tools/render_with_qa.py
"""
import os
import sys
import json
import shutil
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# same production flags as the validated _after9 render …
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")              # hard rule: OFF
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "1")
# PROVIDER POLICY (user, 2026-06-02): fal.ai PRIMARY, NO fixed image ceiling;
# Pollinations emergency-only; every fal still validated + retried.
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ.setdefault("VIDLORE_FAL_MAX_IMAGES", "0")        # 0 ⇒ no ceiling
os.environ.setdefault("VIDLORE_FAL_MAX_ATTEMPTS", "4")      # validate+retry
# emergency text slides stay a last resort, capped + non-silent
os.environ.setdefault("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES", "0")
os.environ.setdefault("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10")
# … plus the autonomous editorial-QA loop
os.environ.setdefault("VIDLORE_EDITORIAL_QA", "1")
os.environ.setdefault("VIDLORE_EDITORIAL_QA_AUTOFIX", "1")
os.environ.setdefault("VIDLORE_EDITORIAL_QA_MAX_PASSES", "2")
os.environ.setdefault("VIDLORE_ALLOW_EDITORIAL_CALLBACKS", "0")   # appears once

from vidlore.brief import Brief                                    # noqa: E402
from vidlore.config import load_config                             # noqa: E402
from vidlore import pipeline as P                                  # noqa: E402
from vidlore import qa_autofix as QA                               # noqa: E402

SRC = Path("dist/Vidlore-Mac/output/the-1860s-secret--how-to-end-garden-pests-permane")
# FRESH folder per run so prior renders/snapshots are never overwritten.
OUTP = Path(os.environ.get("VIDLORE_QA_OUTP", "output/_v14qa"))
OUTP.mkdir(parents=True, exist_ok=True)

bj = json.loads((SRC / "brief.json").read_text())
brief = Brief(
    title=(bj.get("title", "") or "").strip().strip('"'),
    prompt="reviewed", fmt=bj.get("fmt", "documentary"),
    duration=bj.get("duration", "1-2"), theme=bj.get("theme", "standard"),
    voice=bj.get("voice") or None, captions=bool(bj.get("captions", False)),
    background=bj.get("background", "auto"))

run_dir = P.run_dir_for(brief, OUTP)
run_dir.mkdir(parents=True, exist_ok=True)
for f in ("script.txt", "script.json", "brief.json"):
    shutil.copy2(SRC / f, run_dir / f)
# start clean: no carried-over variant re-routes
(run_dir / "variants.json").unlink(missing_ok=True)

cfg = load_config()
# Secure secret handling: report ONLY configured yes/no — never the key value.
print(f"[v14qa] run_dir={run_dir}  max_passes={QA.max_passes()}  "
      f"callbacks={QA.allow_callbacks()}", flush=True)
print(f"[v14qa] provider policy: fal PRIMARY "
      f"(configured={'yes' if getattr(cfg, 'fal_key', '') else 'no'}, "
      f"ceiling=NONE/quality_first, attempts="
      f"{os.environ.get('VIDLORE_FAL_MAX_ATTEMPTS')}); "
      f"Pollinations emergency-only; AI video OFF", flush=True)

# STEP 4 — AI-still provider PREFLIGHT. Emergency text slides are a last
# resort, not the normal output: if NO image provider can produce a still and
# degraded mode is off, stop BEFORE the expensive render instead of shipping a
# text-slide-heavy video.
_pf = QA.preflight_ai_providers(cfg)
print(f"[v14qa] AI-provider preflight: fal={_pf['fal']} "
      f"pollinations={_pf['pollinations']} any_ok={_pf['any_ok']}", flush=True)
if _pf["block"]:
    print("[v14qa] ABORT — no AI-still provider is producing images and "
          "VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES=0. Fix provider connectivity or "
          "set the flag to allow degraded text-slide output.", flush=True)
    raise SystemExit(2)


def render_once():
    # stable run_dir across passes so the bumped variants.json is re-read
    return P.render_from_script(brief, cfg, OUTP, keep_work=True, run_dir=run_dir)


try:
    result, report, passes, history = QA.run_with_qa(render_once, run_dir=run_dir)
    print(f"\n[v14qa] DONE after {passes} pass(es). final gate="
          f"{getattr(report, 'gate', '?')}", flush=True)
    print("[v14qa] autofix history:", json.dumps(history, indent=2), flush=True)
    print(f"[v14qa] video -> {getattr(result, 'video', None)}", flush=True)
except Exception:
    print("[v14qa] FAILED:\n" + traceback.format_exc(), flush=True)
    raise
