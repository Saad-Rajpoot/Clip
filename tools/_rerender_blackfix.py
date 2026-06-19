"""Re-render the Two Cheap Metals / 1860s-Secret project to validate the
dark-stat-card black-dip fix (gold_number_callout charcoal floor + foreground
fade). Output → output/_v16_blackfix (a FRESH dir, _v15_falprimary preserved).

Caches (footage + AI stills + narration) are COPIED from the validated _v15
run so this re-render reuses them — NO new fal calls (the AI-provider path is
already validated and must not be touched). Only the MG cards regenerate
(render_dispatch.VERSION was bumped to mg-0.2.1), so the dip fix is exercised.
"""
import os
import sys
import json
import shutil
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same production flags as the validated _v15_falprimary render.
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_SUBJECT_FLOOR", "1")
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_STOCK_FILMABLE", "1")
os.environ.setdefault("VIDLORE_OVERLAY_RESTRAINT", "1")
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")               # hard rule: OFF
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE_CACHE", "1")
os.environ.setdefault("VIDLORE_FAL_IMAGE_BUDGET_MODE", "quality_first")
os.environ.setdefault("VIDLORE_FAL_MAX_IMAGES", "0")
os.environ.setdefault("VIDLORE_FAL_MAX_ATTEMPTS", "4")
os.environ.setdefault("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES", "0")
os.environ.setdefault("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10")

from vidlore.brief import Brief                                  # noqa: E402
from vidlore.config import load_config                           # noqa: E402
from vidlore import pipeline as P                                # noqa: E402

SRCPROJ = Path("output/_v15_falprimary/"
               "the-1860s-secret--how-to-end-garden-pests-permanen")
OUTP = Path("output/_v16_blackfix")
DSTPROJ = OUTP / SRCPROJ.name
OUTP.mkdir(parents=True, exist_ok=True)

# Seed the fresh run_dir from _v15: copy cache + scripts (reuse footage/AI/TTS,
# no fal). Skip the heavy scratch (work_*), the prior final mp4, and the old MG
# clips (those MUST regenerate under the bumped VERSION = the whole point).
if not DSTPROJ.exists():
    DSTPROJ.mkdir(parents=True, exist_ok=True)
    # Copy every child EXCEPT the heavy scratch (work_*), the old MG clips
    # (must regenerate), and the prior FINAL videos (top-level *.mp4). cache/
    # is copied IN FULL — it contains cached footage .mp4s + AI .jpgs that we
    # rely on to avoid any re-fetch / fal call.
    for child in SRCPROJ.iterdir():
        if child.name.startswith("work_") or child.name == "motion_graphics":
            continue
        if child.is_file() and child.suffix == ".mp4":      # top-level final video only
            continue
        dst = DSTPROJ / child.name
        if child.is_dir():
            shutil.copytree(child, dst)                     # full copy (cache mp4s kept)
        else:
            shutil.copy2(child, dst)
    print(f"[blackfix] seeded {DSTPROJ} from _v15 "
          f"(cache entries: {len(list((DSTPROJ / 'cache').glob('*'))) if (DSTPROJ / 'cache').exists() else 0})",
          flush=True)

bj = json.loads((DSTPROJ / "brief.json").read_text())
brief = Brief(
    title=(bj.get("title", "") or "").strip().strip('"'),
    prompt="reviewed", fmt=bj.get("fmt", "documentary"),
    duration=bj.get("duration", "1-2"), theme=bj.get("theme", "standard"),
    voice=bj.get("voice") or None, captions=bool(bj.get("captions", False)),
    background=bj.get("background", "auto"))

# Pass run_dir EXPLICITLY (P4 fix) so the slug-quote gotcha can't redirect us.
print(f"[blackfix] run_dir={DSTPROJ}  MG VERSION bump → cards regenerate", flush=True)
try:
    res = P.render_from_script(brief, load_config(), OUTP, keep_work=True,
                               run_dir=DSTPROJ)
    print(f"[blackfix] RENDER OK -> {getattr(res, 'video', None)}  "
          f"secs={getattr(res, 'video_seconds', None)}", flush=True)
except Exception:
    print("[blackfix] RENDER FAILED:\n" + traceback.format_exc(), flush=True)
    raise
