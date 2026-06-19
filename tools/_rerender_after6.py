import os, sys, json, shutil, traceback
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── FINAL VISUAL-POLISH re-render (V1.1 candidate) ──────────────────────────
# Exact same 33-scene script as the portal/_after renders. All Priority-1..5
# fixes live in source:
#   P1 black-frame splice  → assemble._repair_black_frames (re-encode + iterate)
#   P2 dark-empty guard    → footage._is_blank_bright_clip (symmetric arm)
#   P3 readability         → footage._slide alpha-28 headline fix
#   P4 clutter guard       → pipeline restraint (niche-aware)
#   P5 injected-stat SFX   → pipeline _mg_primitives → assemble SFX forwarding
os.environ["VIDLORE_AIMG"] = "1"               # AI stills ON
os.environ["VIDLORE_SUBJECT_FLOOR"] = "1"     # subject-presence escalation ON
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"   # MG director ON (gold_number_callout → P5 SFX)
os.environ["VIDLORE_STOCK_FILMABLE"] = "1"    # filmable stock ON (real footage)
os.environ["VIDLORE_OVERLAY_RESTRAINT"] = "1" # restrained, footage-first overlays
os.environ["VIDLORE_AI_VIDEO"] = "0"          # AI VIDEO OFF (stills only — hard rule)

from vidlore.brief import Brief
from vidlore.config import load_config
from vidlore import pipeline as P

SRC = Path("dist/Vidlore-Mac/output/the-1860s-secret--how-to-end-garden-pests-permane")
OUTP = Path("output/_after6")                  # FRESH folder — never overwrite prior AFTERs
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
    print(f"[after6] run_dir={run_dir}  scenes={len(data.get('scenes', []))}  sha_match={sha_ok}", flush=True)
except Exception as _e:
    print(f"[after6] preflight note: {_e}", flush=True)

try:
    res = P.render_from_script(brief, load_config(), OUTP, keep_work=True)
    print(f"[after6] RENDER OK -> {getattr(res, 'video', None)}  secs={getattr(res, 'video_seconds', None)}", flush=True)
except Exception:
    print("[after6] RENDER FAILED:\n" + traceback.format_exc(), flush=True)
    raise
