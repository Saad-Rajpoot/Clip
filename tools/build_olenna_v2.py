#!/usr/bin/env python3
"""FULL re-render of the Olenna long video (original job 5462677f95) with ALL fixes — a FRESH
job so the fixed HD downloader re-fetches sources at 720p-1080p and the new index carries the
static/pair-diff fields the recalibrated gates need (the old index predates them).

Reuses the original UPLOADED VOICEOVER and the exact script (reconstructed from the original
job's segments — the portal's in-memory script registry did not survive its restart).

Run FROM the worktree:  python3 tools/build_olenna_v2.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
OLD = Path("/Users/hussnain/Desktop/clipstudio_output/portal/5462677f95")
P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/olenna_v2_allfixes")
sys.path.insert(0, str(WORKTREE))

for _line in (MAIN / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")
# FINAL render: never weaken the release gate (handover rule — warn is for review drafts only)
os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)

from vidlore.clipstudio.orchestrate import produce_auto            # noqa: E402
from vidlore import musiclib                                       # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


old = json.load(open(OLD / "project.json"))
segs = sorted(old["segments"], key=lambda s: s["index"])
script = "\n".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
topic = ((old.get("meta") or {}).get("analysis") or {}).get("topic") \
    or "Olenna Tyrell Didn't Poison The Wine"

P.mkdir(parents=True, exist_ok=True)
(P / "script.txt").write_text(script, encoding="utf-8")
if not (P / "voiceover.mp3").exists():
    shutil.copy2(OLD / "voiceover.mp3", P / "voiceover.mp3")

cats = musiclib.scan()
n_tracks = sum(len(v) for v in cats.values())
log(f"musiclib: {len(cats)} categories / {n_tracks} tracks (need 11/118)")
assert len(cats) == 11 and n_tracks == 118, "music library incomplete — aborting before render"

t0 = time.time()
res = produce_auto(
    str(P),
    topic=topic,
    script_text=script,
    movie_hint="Game of Thrones",
    policy="approved_testing",
    max_sources=96,
    theme="history",
    captions=True,
    verify=True,
    do_build=True,
    voiceover=str(P / "voiceover.mp3"),
)
log(f"produce_auto done in {(time.time() - t0) / 3600:.2f} h")
log(f"output: {res.get('output')}")
summary = res.get("summary") or {}
log(f"flags: {summary.get('flag_breakdown')} · sources: {summary.get('sources_used')} "
    f"· mean_conf: {summary.get('mean_confidence')}")
