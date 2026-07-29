#!/usr/bin/env python3
"""Finish jaime_brienne_v2: targeted fetch for the 2 gate-blocked scenes, heal, direct build."""
import os, sys, time
from pathlib import Path
WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
J = Path("/Users/hussnain/Desktop/clipstudio_output/portal/jaime_brienne_v2")
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
os.environ.setdefault("VIDLORE_CLIPSTUDIO_SELFHEAL_STILL_CANDS", "20")
os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)
from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio import selfheal as SH
from vidlore.clipstudio.build import build_video, preassemble_release_block_reason
from vidlore import musiclib

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

proj = ClipProject.load(str(J))
cfg = ClipConfig()
segs = {s.index: s for s in proj.segments}
segs[71].scene_query = "Game of Thrones Jaime kills Alton Lannister cell scene S02E07"
segs[73].scene_query = "Game of Thrones Riverrun surrenders Edmure opens gates scene S06E08"
n = SH.heal_blocked_beats(proj, list(proj.segments), cfg, blocked=[71, 73],
                          policy="approved_testing", log=log)
proj.save()
log(f"heal resolved {n}/2")
pre = preassemble_release_block_reason(proj, list(proj.segments), (proj.meta or {}).get("analysis"))
log(f"pre-assembly gate: {pre or 'CLEAR'}")
if pre:
    sys.exit(3)
cats = musiclib.scan()
assert len(cats) == 11 and sum(len(v) for v in cats.values()) == 118
t0 = time.time()
out = build_video(proj, list(proj.segments), cfg, captions=True,
                  title="The Real Reason Jaime Lied to Brienne That Night",
                  theme_name="history", voiceover=str(J / "voiceover.mp3"),
                  use_tts=True, progress=log)
log(f"BUILD DONE in {(time.time()-t0)/60:.1f} min -> {out}")
