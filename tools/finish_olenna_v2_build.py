#!/usr/bin/env python3
"""Direct BUILD of olenna_v2_allfixes from the CURRENT (frozen) selections — no re-match, so
the two surgically installed, vision-verified stills (beats 39, 127) survive. The pre-assembly
gate is re-checked first and the authoritative in-build release gate still applies unchanged."""
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
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")
os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.build import build_video, preassemble_release_block_reason
from vidlore import musiclib

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

proj = ClipProject.load(str(J))
cfg = ClipConfig()
segs = list(proj.segments)
cats = musiclib.scan()
assert len(cats) == 11 and sum(len(v) for v in cats.values()) == 118, "music library incomplete"
pre = preassemble_release_block_reason(proj, segs, (proj.meta or {}).get("analysis"))
log(f"pre-assembly gate: {pre or 'CLEAR'}")
if pre:
    sys.exit(3)
t0 = time.time()
out = build_video(proj, segs, cfg, captions=True, title="Olenna Tyrell Didn't Poison The Wine",
                  theme_name="history", voiceover=str(J / "voiceover.mp3"),
                  use_tts=True, progress=log)
log(f"BUILD DONE in {(time.time()-t0)/60:.1f} min -> {out}")
