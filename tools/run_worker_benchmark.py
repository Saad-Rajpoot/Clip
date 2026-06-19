# V1 worker benchmark — Mode A (4/5/4) vs Mode B (6/6/4), SAME 60-90s sample,
# SAME cached assets (portal cache → fal=0), SEQUENTIAL (no CPU contention),
# equivalence-verified. No code change. Cleans up after itself.
import json, os, shutil, time
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
PORTAL = REPO / "dist/Vidlore-Mac/output/how-a-female-mossad-spy-married-an-iranian-general"
OUT = REPO / "output/_wbench"
N = 11  # ~65s

import vidlore.pipeline as P
from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script

# ---- build the shared sample template once (11 scenes, MG OFF) -------------
src = json.loads((PORTAL / "script.json").read_text())
scenes = src["scenes"][:N]
title = src.get("title", "Worker Bench")
body = title + "\n\n" + "\n\n".join(s.get("narration", "") for s in scenes)
TPL = OUT / "_tpl"
if OUT.exists():
    shutil.rmtree(OUT)
TPL.mkdir(parents=True)
(TPL / "script.json").write_text(json.dumps({"title": title, "source_sha256": P._sha(body), "scenes": scenes}, indent=1))
(TPL / "script.txt").write_text(body, encoding="utf-8")
br = json.loads((PORTAL / "brief.json").read_text()); br["captions"] = True
(TPL / "brief.json").write_text(json.dumps(br, indent=1))
shutil.copytree(PORTAL / "cache", TPL / "cache", dirs_exist_ok=True)
print("sample: %d scenes (~65s), MG OFF, portal cache copied" % len(scenes), flush=True)

# generate ONE recipe and lock it for EVERY render → identical salt (no rotation,
# honours the "same seed/salt" requirement + removes salt as a timing confound)
from vidlore import editorial_recipe as _er
_b0 = load_brief(TPL); _b0.extra = {"niche": "spy_intel"}
try:
    LOCK = _er.generate_recipe(_b0, "spy_intel", channel=None, variation=0)
except Exception as e:
    LOCK = None; print("recipe-gen failed:", e, flush=True)
print("locked recipe salt:", (LOCK or {}).get("variation_salt"), flush=True)

_MODES = {"A": {"VIDLORE_FFMPEG_WORKERS": "4", "VIDLORE_AI_IMAGE_WORKERS": "5", "VIDLORE_AUDIO_WORKERS": "4"},
          "B": {"VIDLORE_FFMPEG_WORKERS": "6", "VIDLORE_AI_IMAGE_WORKERS": "6", "VIDLORE_AUDIO_WORKERS": "4"}}
# order controllable: BENCH_ORDER="B,A" runs B first (reverse-order confound control)
_order = (os.environ.get("BENCH_ORDER") or "A,B").split(",")
MODES = [(n, _MODES[n]) for n in _order]
print("run order:", _order, flush=True)

results = {}
for name, env in MODES:
    rd = OUT / ("mode_" + name)
    shutil.rmtree(rd, ignore_errors=True)
    rd.mkdir(parents=True)
    for f in ("script.json", "script.txt", "brief.json"):
        shutil.copy2(TPL / f, rd / f)
    shutil.copytree(TPL / "cache", rd / "cache", dirs_exist_ok=True)
    # workers for THIS mode; MG OFF; AI-video OFF; no new web fetch
    for k, v in env.items():
        os.environ[k] = v
    os.environ.pop("VIDLORE_MOTION_GRAPHICS", None)
    os.environ.pop("VIDLORE_AI_VIDEO", None); os.environ.pop("VIDLORE_AI_VIDEO", None)
    os.environ["WEB_FOOTAGE_ENGINE"] = "0"; os.environ["WEB_IMAGE_ENGINE"] = "0"
    os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"
    brief = load_brief(rd); brief.extra = {"niche": "spy_intel"}
    if LOCK:
        brief.extra["editorial_recipe_lock"] = dict(LOCK)  # pin salt for both modes
    cfg = load_config()
    print("\n========== MODE %s  (FFMPEG=%s AI_IMAGE=%s AUDIO=%s) =========="
          % (name, env["VIDLORE_FFMPEG_WORKERS"], env["VIDLORE_AI_IMAGE_WORKERS"], env["VIDLORE_AUDIO_WORKERS"]), flush=True)
    t0 = time.time()
    render_from_script(brief, cfg, OUT, run_dir=rd)
    wall = time.time() - t0
    meta = json.loads((rd / "render_meta.json").read_text())
    mets = json.loads((rd / "render_metrics.json").read_text()) if (rd / "render_metrics.json").exists() else {}
    results[name] = {"wall": wall, "meta": meta, "mets": mets,
                     "mp4": any(rd.glob("*.mp4"))}
    print("MODE %s wall=%.1fs" % (name, wall), flush=True)

# ---- compare ---------------------------------------------------------------
a, b = results["A"], results["B"]
am, bm = a["meta"], b["meta"]
print("\n================ WORKER BENCHMARK: A vs B ================", flush=True)
print("  total time   A=%.1fs   B=%.1fs   gain=%.1f%%"
      % (a["wall"], b["wall"], (a["wall"] - b["wall"]) / a["wall"] * 100.0), flush=True)
def eq(label, va, vb):
    print("  [%s] %-20s A=%s  B=%s" % ("OK " if va == vb else "DIFF", label, va, vb), flush=True)
eq("scenes", am.get("scenes"), bm.get("scenes"))
eq("video_seconds", am.get("video_seconds"), bm.get("video_seconds"))
eq("fps", am.get("fps"), bm.get("fps"))
eq("scene_durations==", am.get("scene_durations") == bm.get("scene_durations"), True)
eq("recipe.salt", (am.get("editorial_recipe") or {}).get("variation_salt"), (bm.get("editorial_recipe") or {}).get("variation_salt"))
eq("mp4_produced", a["mp4"], b["mp4"])
print("  metrics A:", json.dumps(a["mets"]), flush=True)
print("  metrics B:", json.dumps(b["mets"]), flush=True)
print("WORKER BENCHMARK COMPLETE", flush=True)
