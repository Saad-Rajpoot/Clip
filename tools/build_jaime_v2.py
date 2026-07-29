#!/usr/bin/env python3
"""THE one rerender of the Jaime/Brienne essay (original portal job 9e3a5b9bfd) — fresh job so
the permanent-HD layer re-downloads at 720-1080p and anchor-coverage guarantees the S8E4
courtyard scene's own uploads before match. Original uploaded voiceover + exact script."""
import json, os, shutil, sys, time
from pathlib import Path
WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
OLD = Path("/Users/hussnain/Desktop/clipstudio_output/portal/9e3a5b9bfd")
P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/jaime_brienne_v2")
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
from vidlore import musiclib

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

old = json.load(open(OLD / "project.json"))
segs = sorted(old["segments"], key=lambda s: s["index"])
script = "\n".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
topic = ((old.get("meta") or {}).get("analysis") or {}).get("topic") \
    or "The Real Reason Jaime Lied to Brienne That Night"
P.mkdir(parents=True, exist_ok=True)
(P / "script.txt").write_text(script)
if not (P / "voiceover.mp3").exists():
    shutil.copy2(OLD / "voiceover.mp3", P / "voiceover.mp3")
cats = musiclib.scan()
assert len(cats) == 11 and sum(len(v) for v in cats.values()) == 118, "music library incomplete"
t0 = time.time()
res = produce_auto(str(P), topic=topic, script_text=script, movie_hint="Game of Thrones",
                   policy="approved_testing", max_sources=96, theme="history", captions=True,
                   verify=True, do_build=True, voiceover=str(P / "voiceover.mp3"))
log(f"done in {(time.time()-t0)/3600:.2f} h -> {res.get('output')}")
s = res.get("summary") or {}
log(f"flags: {s.get('flag_breakdown')} | sources: {s.get('sources_used')} | conf: {s.get('mean_confidence')}")
