#!/usr/bin/env python3
"""Resume the olenna_v2_allfixes render after the gap fixes (warm caches on disk)."""
import os, sys, time
from pathlib import Path
WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))
for _line in (MAIN / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")
os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)
from vidlore.clipstudio.orchestrate import produce_auto
script = (P / "script.txt").read_text()
t0 = time.time()
res = produce_auto(str(P), topic="Olenna Tyrell Didn't Poison The Wine", script_text=script,
                   movie_hint="Game of Thrones", policy="approved_testing", max_sources=96,
                   theme="history", captions=True, verify=True, do_build=True,
                   voiceover=str(P / "voiceover.mp3"), resume=True)
print(f"[resume] done in {(time.time()-t0)/3600:.2f} h -> {res.get('output')}", flush=True)
s = res.get("summary") or {}
print("[resume] flags:", s.get("flag_breakdown"), "| sources:", s.get("sources_used"),
      "| mean_conf:", s.get("mean_confidence"), flush=True)
