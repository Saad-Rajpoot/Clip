#!/usr/bin/env python3
"""Run the NEW selfheal machinery on the current blocked beats, then rebuild via the
finish script. Live dogfood of commit 89f1069."""
import os, sys, time
from pathlib import Path
WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
J = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))
for _line in (MAIN / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio import selfheal as SH

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

proj = ClipProject.load(str(J))
cfg = ClipConfig()
blocked = SH.blocked_indexes(proj)
log(f"blocked: {blocked}")
n = SH.heal_blocked_beats(proj, list(proj.segments), cfg, blocked=blocked,
                          policy="approved_testing", log=log)
proj.save()
log(f"selfheal resolved {n}/{len(blocked)}")
