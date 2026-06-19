import os, json, shutil, traceback
from pathlib import Path
os.environ["VIDLORE_AIMG"] = "1"             # AI images ON
os.environ["VIDLORE_SUBJECT_FLOOR"] = "1"   # new subject-presence escalation ON
os.environ["VIDLORE_REUSE_SCRIPT_JSON"] = "1"  # identical narration (skip LLM)
from vidlore.brief import Brief
from vidlore.config import load_config
from vidlore.pipeline import produce

SRC = Path("dist/Vidlore-Mac/output/the-1860s-secret--how-to-end-garden-pests-permane")
OUT = Path("output/_after_two_cheap_metals/the-1860s-secret--how-to-end-garden-pests-permane")
OUT.mkdir(parents=True, exist_ok=True)
for f in ("script.json", "brief.json"):       # carry identical script
    shutil.copy2(SRC / f, OUT / f)
bj = json.loads((SRC / "brief.json").read_text())
brief = Brief(
    title=(bj.get("title", "") or "").strip().strip('"'),
    prompt="reviewed",
    fmt=bj.get("fmt", "documentary"),
    duration=bj.get("duration", "1-2"),
    theme=bj.get("theme", "standard"),
    voice=bj.get("voice") or None,
    captions=bool(bj.get("captions", False)),
    background=bj.get("background", "auto"),
)
print(f"[after] brief: {brief.title!r} dur={brief.duration} theme={brief.theme} voice={brief.voice}", flush=True)
try:
    res = produce(brief, load_config(), OUT, keep_work=True)
    print(f"[after] RENDER OK -> {getattr(res,'video',None)}  secs={getattr(res,'video_seconds',None)}", flush=True)
except Exception:
    print("[after] RENDER FAILED:\n" + traceback.format_exc(), flush=True)
    raise
