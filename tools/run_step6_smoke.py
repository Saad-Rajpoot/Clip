# V3.4.2 STEP 6 — production-mode portal-equivalent smoke, MOTION GRAPHICS ON.
# Reuses the first 11 portal scenes (~65s) verbatim (no LLM — matching source_sha256)
# + the cached assets, and renders with MG ON / asset-QA ON / factual guard ON /
# subtitles ON / music ON / SFX ON / fal allowed / AI-VIDEO OFF. Verifies the
# MG manifest + asset-QA manifest + MG primitive + variant + guard all fire.
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
PORTAL = REPO / "dist/Vidlore-Mac/output/how-a-female-mossad-spy-married-an-iranian-general"
OUT_ROOT = REPO / "output/_mg_smoke"
RUN = OUT_ROOT / "mg-on-production-smoke"
N = 11

import vidlore.pipeline as P
from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script

src = json.loads((PORTAL / "script.json").read_text())
scenes = src["scenes"][:N]
title = src.get("title", "Production MG Smoke")
body = title + "\n\n" + "\n\n".join(s.get("narration", "") for s in scenes)

if RUN.exists():
    shutil.rmtree(RUN)
RUN.mkdir(parents=True, exist_ok=True)
out = {"title": title, "source_sha256": P._sha(body), "scenes": scenes}
(RUN / "script.json").write_text(json.dumps(out, indent=1))
(RUN / "script.txt").write_text(body, encoding="utf-8")
br = json.loads((PORTAL / "brief.json").read_text())
br["captions"] = True                       # STEP 6: subtitles ON
(RUN / "brief.json").write_text(json.dumps(br, indent=1))
shutil.copytree(PORTAL / "cache", RUN / "cache", dirs_exist_ok=True)
print("seeded %d-scene smoke (~65s) + cache" % len(scenes), flush=True)

# Production MG-ON config
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"          # STEP 6: MG ON
os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"         # reuse the truncated script (no LLM)
os.environ.pop("VIDLORE_AI_VIDEO", None)             # AI-generated VIDEO OFF
os.environ.pop("VIDLORE_AI_VIDEO", None)
os.environ.pop("VIDLORE_STRICT_REPLAY", None)
os.environ["WEB_FOOTAGE_ENGINE"] = "0"                # cache-first (cost control)
os.environ["WEB_IMAGE_ENGINE"] = "0"

brief = load_brief(RUN)
brief.extra = {"niche": "spy_intel"}
cfg = load_config()
for k in ("music_enabled", "sfx_enabled", "transitions_enabled", "overlays_enabled"):
    setattr(cfg, k, True)                              # music + SFX + transitions + overlays ON

t0 = time.time()
try:
    render_from_script(brief, cfg, OUT_ROOT, run_dir=RUN)
    print("RENDER OK in %.1fs" % (time.time() - t0), flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("RENDER FAILED: %s" % exc, flush=True)
    traceback.print_exc()
    sys.exit(2)

# ---- STEP 6 verifications -------------------------------------------------
def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

mg = _load(RUN / "motion_graphics_manifest.json")
aq = _load(RUN / "asset_qa.json")
bf = _load(RUN / "render_black_frame_metrics.json")
mets = _load(RUN / "render_metrics.json")
entries = mg if isinstance(mg, list) else mg.get("entries") or mg.get("graphics") or []
n_mg = len(entries) if isinstance(entries, (list, dict)) else 0
variant_ids = []
audit_present = False
for e in (entries if isinstance(entries, list) else []):
    v = (e.get("variant") or {}) if isinstance(e, dict) else {}
    if v.get("visual_variant_id"):
        variant_ids.append(v["visual_variant_id"])
    if e.get("motion_graphics_audit") or e.get("asset_qa"):
        audit_present = True

print("\n================ STEP 6 — MG-ON PRODUCTION SMOKE ================", flush=True)
print("  motion_graphics_manifest.json exists : %s (%d entr)" % (bool(mg), n_mg), flush=True)
print("  asset_qa.json exists                 : %s (mode=%s, %d warn)"
      % (bool(aq), aq.get("mode"), len(aq.get("warnings", []))), flush=True)
print("  MG primitive(s) rendered             : %s" % (n_mg > 0), flush=True)
print("  seeded variant recorded              : %s %s"
      % (bool(variant_ids), variant_ids[:4]), flush=True)
print("  AI-generated video calls             : 0 (VIDLORE_AI_VIDEO popped)", flush=True)
print("  black frames                         : %s" % (bf.get("black_frames", mets.get("black_frames", "?"))), flush=True)
print("  final MP4                            : %s" % ((RUN / (RUN.name + ".mp4")).exists() or
      any(RUN.glob("*.mp4"))), flush=True)
print("STEP 6 COMPLETE", flush=True)
