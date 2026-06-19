# RC4 targeted rerender of the SAME Iran-Iraq project, reusing cache, with the
# acceptance fixes active. Isolated output dir (keeps the failed fixture intact).
# Settings: MG ON, AI stills ON, AI-video OFF, captions burned-in ON, SFX ON,
# strict relevance ON, niche empty(->history), default workers 4/5/4.
import json, os, shutil, time
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
SRC = REPO / "dist/Vidlore-Mac/output/how-the-iran-iraq-war-reshaped-the-middle-east"
RD = REPO / "output/_rc4_rerender"

if RD.exists():
    shutil.rmtree(RD)
RD.mkdir(parents=True)
for f in ("script.json", "script.txt", "brief.json"):
    if (SRC / f).exists():
        shutil.copy2(SRC / f, RD / f)
print("copying cache (reuse footage/TTS/fal stills)…", flush=True)
shutil.copytree(SRC / "cache", RD / "cache", dirs_exist_ok=True)

# clear the WRONG-FACE portrait cache so the patched identity gate re-fetches
import vidlore.footage as F
cleared = 0
for nm in ("ayatollah khomeini", "khomeini", "saddam hussein", "saddam",
           "ruhollah khomeini"):
    k = F.scene_key("realperson", nm)
    for base in (RD / "cache", Path.home() / ".vidlore", Path.home() / ".vidlore/cache"):
        if base.exists():
            for p in base.glob(f"{k}*"):
                try:
                    p.unlink(); cleared += 1
                except Exception:
                    pass
print(f"cleared {cleared} stale portrait-cache entr(ies) (Khomeini/Saddam)", flush=True)

# production env for the rerender
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"     # MG ON
os.environ.pop("VIDLORE_AI_VIDEO", None); os.environ.pop("VIDLORE_AI_VIDEO", None)  # AI-video OFF
os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"    # exact same script
os.environ["WEB_FOOTAGE_ENGINE"] = "0"; os.environ["WEB_IMAGE_ENGINE"] = "0"  # cache-only footage
os.environ.setdefault("VIDLORE_REAL_PERSON", "1")  # portrait sourcing (wikipedia) ON

from vidlore.pipeline import load_brief, render_from_script
from vidlore.config import load_config

brief = load_brief(RD)
brief.captions = True                            # burn captions IN
if not isinstance(getattr(brief, "extra", None), dict):
    brief.extra = {}
brief.extra["sfx"] = True                        # SFX ON
brief.extra.pop("niche", None)                   # empty -> classify -> history (fix)
cfg = load_config()
try:
    cfg.sfx_enabled = True
except Exception:
    pass
print("rendering (MG ON, captions ON, SFX ON, AI-video OFF, niche=auto)…", flush=True)
t0 = time.time()
render_from_script(brief, cfg, RD.parent, run_dir=RD)
print("RC4 RERENDER DONE wall=%.1fs" % (time.time() - t0), flush=True)
mp4 = sorted(RD.glob("*.mp4"))
print("mp4:", [p.name for p in mp4], flush=True)
