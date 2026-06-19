# PHASE B — real ENGINE render of the EXACT portal script.json via the backend
# pipeline (render_from_script), reusing the portal's content-hashed cache (TTS
# wavs + footage) so it's near-zero-cost and isolates PIPELINE determinism from
# asset-fetch noise. MG kept OFF to match the portal render. Compares the new
# render_meta/metrics against the portal's.
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
PORTAL = REPO / "dist/Vidlore-Mac/output/how-a-female-mossad-spy-married-an-iranian-general"
OUT_ROOT = REPO / "output/_parity_engine"
RUN = OUT_ROOT / PORTAL.name

print("=== PHASE B: engine re-render (reuse exact script.json + cache, MG OFF) ===", flush=True)

# 1. Seed a fresh run_dir with the EXACT script.json + brief.json + cache (reuse assets)
if RUN.exists():
    shutil.rmtree(RUN)
RUN.mkdir(parents=True, exist_ok=True)
for f in ("script.json", "brief.json", "script.txt"):
    shutil.copy2(PORTAL / f, RUN / f)
shutil.copytree(PORTAL / "cache", RUN / "cache", dirs_exist_ok=True)
print("seeded run_dir:", RUN, flush=True)
print("  copied script.json + brief.json + cache (%d files)"
      % sum(1 for _ in (RUN / "cache").iterdir()), flush=True)

# 2. Match the portal config: MG OFF (no manifest, legacy graphic_kind path)
os.environ.pop("VIDLORE_MOTION_GRAPHICS", None)
os.environ.setdefault("VIDLORE_REUSE_SCRIPT_JSON", "1")  # reuse script.json, skip LLM

from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script

brief = load_brief(RUN)
cfg = load_config()
print("brief.title=%r  captions=%s" % (getattr(brief, "title", None), getattr(brief, "captions", None)), flush=True)

# 3. Render
t0 = time.time()
try:
    res = render_from_script(brief, cfg, OUT_ROOT, run_dir=RUN)
    print("RENDER OK in %.1fs -> %s" % (time.time() - t0, getattr(res, "video", res)), flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("RENDER FAILED after %.1fs: %s" % (time.time() - t0, exc), flush=True)
    traceback.print_exc()
    sys.exit(2)

# 4. Compare render_meta + metrics vs the portal
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

pm, em = load(PORTAL / "render_meta.json"), load(RUN / "render_meta.json")
pmet, emet = load(PORTAL / "render_metrics.json"), load(RUN / "render_metrics.json")
pr, er = pm.get("editorial_recipe", {}), em.get("editorial_recipe", {})

print("\n================ PARITY: ENGINE(CLI) vs PORTAL ================", flush=True)
def row(label, a, b):
    same = "OK " if a == b else "DIFF"
    print("  [%s] %-22s engine=%-22s portal=%-22s" % (same, label, str(a), str(b)), flush=True)

row("scenes", em.get("scenes"), pm.get("scenes"))
row("beats", em.get("beats"), pm.get("beats"))
row("cuts", em.get("cuts"), pm.get("cuts"))
row("video_seconds", em.get("video_seconds"), pm.get("video_seconds"))
row("scene_durations==", em.get("scene_durations") == pm.get("scene_durations"), True)
row("recipe.variation_salt", er.get("variation_salt"), pr.get("variation_salt"))
row("recipe._attempt", er.get("_attempt"), pr.get("_attempt"))
row("recipe.niche", er.get("niche"), pr.get("niche"))
row("recipe.accent", er.get("accent"), pr.get("accent"))
row("recipe.beat_target", er.get("beat_target"), pr.get("beat_target"))
row("recipe.density", er.get("density"), pr.get("density"))
row("recipe.transition_palette", er.get("transition_palette"), pr.get("transition_palette"))
row("metrics.lufs", (emet.get("audio") or {}).get("lufs"), (pmet.get("audio") or {}).get("lufs"))
row("metrics.black_frames", emet.get("black_frames"), pmet.get("black_frames"))
row("metrics.qa_verdict", (emet.get("qa") or {}).get("verdict"), (pmet.get("qa") or {}).get("verdict"))

# recipe full diff (which axes differ if salt differs)
diffs = {k: (er.get(k), pr.get(k)) for k in pr if k != "_attempt" and er.get(k) != pr.get(k)}
print("\nrecipe axes that differ: %d" % len(diffs), flush=True)
for k, (a, b) in diffs.items():
    print("    %-22s engine=%r  portal=%r" % (k, a, b), flush=True)

print("\nPHASE B COMPLETE", flush=True)
