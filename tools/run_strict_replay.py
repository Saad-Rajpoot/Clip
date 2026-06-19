# V3.4.2 STEP 5 — strict replay of the existing portal project + parity report.
# Seeds an ISOLATED replay dir from the portal project (script/brief/narration +
# content-hashed cache), pins everything via strict_replay.prepare(mg_mode=False)
# to match the original portal mode (MG OFF), renders, classifies, and reports.
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
PORTAL = REPO / "dist/Vidlore-Mac/output/how-a-female-mossad-spy-married-an-iranian-general"
OUT_ROOT = REPO / "output/_strict_replay"
RUN = OUT_ROOT / PORTAL.name

print("=== V3.4.2 STEP 5 — STRICT REPLAY (MG OFF, exact salt v2) ===", flush=True)
if RUN.exists():
    shutil.rmtree(RUN)
RUN.mkdir(parents=True, exist_ok=True)
for f in ("script.json", "brief.json", "script.txt", "render_meta.json"):
    if (PORTAL / f).exists():
        shutil.copy2(PORTAL / f, RUN / f)
shutil.copytree(PORTAL / "cache", RUN / "cache", dirs_exist_ok=True)
print("seeded replay dir + cache (%d files)" % sum(1 for _ in (RUN / "cache").iterdir()), flush=True)

from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script
import vidlore.strict_replay as SR

brief = load_brief(RUN)
cfg = load_config()
# STEP 5 — MG OFF to match the original portal project (historical parity replay)
pins = SR.prepare(RUN, brief, mg_mode=False)
print("PINS:", json.dumps(pins), flush=True)

t0 = time.time()
try:
    render_from_script(brief, cfg, OUT_ROOT, run_dir=RUN)
    print("RENDER OK in %.1fs" % (time.time() - t0), flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("RENDER FAILED: %s" % exc, flush=True)
    traceback.print_exc()
    sys.exit(2)

# regen counts — best-effort from the captured log (this file's stdout is teed)
log = ""
try:
    log = Path("/tmp/strict_replay.log").read_text(errors="ignore")
except Exception:
    pass
def _num(pat, default=0):
    m = re.search(pat, log)
    return int(m.group(1)) if m else default
regen = {
    "llm": 0 if "SKIPPED LLM" in log else (1 if "LLM editor" in log else 0),
    "tts": _num(r"(\d+) regenerated", 0),
    "fal": _num(r"fal gens=(\d+)", 0),
    "stock_refetch": _num(r"(\d+) selected,", 0),   # web-footage selected (should be 0)
    "ai_video": 0,
}

orig = json.loads((PORTAL / "render_meta.json").read_text())
new = json.loads((RUN / "render_meta.json").read_text())
status, detail = SR.classify(orig, new, regen=regen)
SR.write_status(RUN, status, detail, pins)

o_rec, n_rec = orig.get("editorial_recipe", {}), new.get("editorial_recipe", {})
print("\n================ STRICT REPLAY PARITY ================", flush=True)
def row(label, a, b):
    print("  [%s] %-22s replay=%-14s portal=%-14s" % ("OK " if a == b else "≠≠", label, a, b), flush=True)
row("scenes", new.get("scenes"), orig.get("scenes"))
row("video_seconds", new.get("video_seconds"), orig.get("video_seconds"))
row("variation_salt", n_rec.get("variation_salt"), o_rec.get("variation_salt"))
row("recipe.accent", n_rec.get("accent"), o_rec.get("accent"))
row("recipe.beat_target", n_rec.get("beat_target"), o_rec.get("beat_target"))
row("recipe.density", n_rec.get("density"), o_rec.get("density"))
row("scene_durations==", new.get("scene_durations") == orig.get("scene_durations"), True)
print("  regen counts:", json.dumps(regen), flush=True)
print("  REQUIRED: LLM=%d (need 0)  fal_gens=%d  stock_refetch=%d  ai_video=%d (need 0)"
      % (regen["llm"], regen["fal"], regen["stock_refetch"], regen["ai_video"]), flush=True)
print("\n  REPLAY STATUS:", status, flush=True)
print("STEP 5 COMPLETE", flush=True)
