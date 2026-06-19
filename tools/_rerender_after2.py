import os, sys, json, shutil, traceback
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VIDLORE_AIMG"] = "1"             # AI images ON
os.environ["VIDLORE_SUBJECT_FLOOR"] = "1"   # new subject-presence escalation ON
from vidlore.brief import Brief
from vidlore.config import load_config
from vidlore import pipeline as P

SRC = Path("dist/Vidlore-Mac/output/the-1860s-secret--how-to-end-garden-pests-permane")
OUTP = Path("output/_after2")                # parent; run_dir_for appends the slug
OUTP.mkdir(parents=True, exist_ok=True)

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

run_dir = P.run_dir_for(brief, OUTP)
run_dir.mkdir(parents=True, exist_ok=True)
# carry the EXACT reviewed script (txt + json) so render_from_script reuses it
for f in ("script.txt", "script.json", "brief.json"):
    shutil.copy2(SRC / f, run_dir / f)

# confirm the reuse path will trigger (sha match OR narrations line up)
try:
    body = (run_dir / "script.txt").read_text(encoding="utf-8")
    data = json.loads((run_dir / "script.json").read_text())
    sha_ok = data.get("source_sha256") == P._sha(body)
    print(f"[after2] run_dir={run_dir}  scenes={len(data.get('scenes', []))}  sha_match={sha_ok}", flush=True)
except Exception as _e:
    print(f"[after2] preflight note: {_e}", flush=True)

try:
    res = P.render_from_script(brief, load_config(), OUTP, keep_work=True)
    print(f"[after2] RENDER OK -> {getattr(res, 'video', None)}  secs={getattr(res, 'video_seconds', None)}", flush=True)
except Exception:
    print("[after2] RENDER FAILED:\n" + traceback.format_exc(), flush=True)
    raise
