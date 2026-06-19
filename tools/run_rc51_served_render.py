# RC5.1 STEP 7 — render the SAME Iran-Iraq project with current (RC5.1) code
# INTO THE SERVED dist location the dashboard reads, so the file the user would
# watch is the file we verify. Keeps the original (stale, game-UI) project folder
# intact as evidence; renders into a NEW "-rc51-verified" sibling.
#
# Settings: MG ON, AI stills ON, AI-video OFF, captions burned-in ON, SFX ON,
# strict visual-relevance gate ON, post-render QA ON, niche empty(->history),
# web engines OFF (so the cached-asset re-validation path must REJECT the known
# junk — game-UI sc0 / sign sc6 / anime sc22 — and route to fallback).
import os, shutil, time, hashlib
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
SRC = REPO / "dist/Vidlore-Mac/output/how-the-iran-iraq-war-reshaped-the-middle-east"
RD = REPO / "dist/Vidlore-Mac/output/how-the-iran-iraq-war-reshaped-the-middle-east-rc51-verified"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# Back up a prior -rc51-verified (don't overwrite a previous verified render);
# NEVER touch the original served folder (kept as stale-output evidence).
if RD.exists():
    bak = RD.with_name(RD.name + ".bak")
    if bak.exists():
        shutil.rmtree(bak)
    shutil.move(str(RD), str(bak))
    print(f"backed up existing {RD.name} -> {bak.name}", flush=True)
RD.mkdir(parents=True)

for f in ("script.json", "script.txt", "brief.json"):
    if (SRC / f).exists():
        shutil.copy2(SRC / f, RD / f)
print("copying cache (reuse footage/TTS/fal stills + the known junk for re-validation)…", flush=True)
shutil.copytree(SRC / "cache", RD / "cache", dirs_exist_ok=True)

# clear the WRONG-FACE portrait cache so the patched identity gate re-fetches real
import vidlore.footage as F
cleared = 0
for nm in ("ayatollah khomeini", "khomeini", "saddam hussein", "saddam", "ruhollah khomeini"):
    k = F.scene_key("realperson", nm)
    for base in (RD / "cache", Path.home() / ".vidlore", Path.home() / ".vidlore/cache"):
        if base.exists():
            for p in base.glob(f"{k}*"):
                try:
                    p.unlink(); cleared += 1
                except Exception:
                    pass
print(f"cleared {cleared} stale portrait-cache entr(ies) (Khomeini/Saddam)", flush=True)

# production env for the served render
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"            # MG ON
os.environ["VIDLORE_VISUAL_RELEVANCE"] = "1"           # CLIP relevance gate ON (now default, explicit)
os.environ["VIDLORE_POSTRENDER_QA"] = "1"              # post-render QA ON
os.environ.pop("VIDLORE_AI_VIDEO", None)
os.environ.pop("VIDLORE_AI_VIDEO", None)                # AI-video OFF
os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"           # exact same frozen script
os.environ["WEB_FOOTAGE_ENGINE"] = "0"
os.environ["WEB_IMAGE_ENGINE"] = "0"                    # cached re-validation path (junk must be rejected from cache)
os.environ.setdefault("VIDLORE_REAL_PERSON", "1")       # portrait sourcing (wikipedia) ON

from vidlore.pipeline import load_brief, render_from_script
from vidlore.config import load_config

brief = load_brief(RD)
brief.captions = True                                   # burn captions IN
if not isinstance(getattr(brief, "extra", None), dict):
    brief.extra = {}
brief.extra["sfx"] = True                               # SFX ON
brief.extra.pop("niche", None)                          # empty -> classify -> history
cfg = load_config()
try:
    cfg.sfx_enabled = True
except Exception:
    pass

print("rendering into SERVED location (MG ON, captions ON, SFX ON, AI-video OFF, niche=auto, gate ON)…", flush=True)
t0 = time.time()
render_from_script(brief, cfg, RD.parent, run_dir=RD)
dt = time.time() - t0
print("RC5.1 SERVED RENDER DONE wall=%.1fs" % dt, flush=True)

mp4s = sorted(RD.glob("*.mp4"))
print("mp4:", [p.name for p in mp4s], flush=True)
for p in mp4s:
    print(f"  served_sha256[{p.name}] = {sha256(p)}", flush=True)

qa = RD / "render_relevance_qa.json"
if qa.exists():
    print("---- render_relevance_qa.json ----", flush=True)
    print(qa.read_text()[:2400], flush=True)
else:
    print("WARNING: render_relevance_qa.json NOT written (post-render QA did not run)", flush=True)
print("RC51_SERVED_RENDER_COMPLETE", flush=True)
