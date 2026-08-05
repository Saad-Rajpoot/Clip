#!/usr/bin/env python3
"""The real ~100-beat end-to-end validation.

Runs against `e2e_101_val` — an APFS clone of the 101-beat job `e2e_b79_101_c0f5443`, so the
downloaded sources and the built index are reused and nothing is re-fetched from YouTube. Only the
stages the current code actually changed are re-run; everything upstream of them keeps its
checkpoint.

Which stages get invalidated is passed in, because it depends on what changed:

    python3 tools/validate_101_beats.py cut verify recover backfill

The final assemble always re-runs. A gate that blocks is a RESULT, not a failure of this script —
it reports the block and exits non-zero rather than manufacturing a file.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
sys.path.insert(0, str(REPO))

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

from vidlore.clipstudio.orchestrate import produce_auto           # noqa: E402
from vidlore import musiclib                                      # noqa: E402

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/e2e_101_val")
TOPIC = "Littlefinger Was Never A Genius — One Man Proved It"
STAGES = sys.argv[1:] or ["cut", "verify", "recover", "backfill"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Invalidate exactly the named checkpoints, so `resume=True` re-runs them under current code and
# keeps analyze/discover/download/index/match as they are.
pj = P / "project.json"
doc = json.loads(pj.read_text(encoding="utf-8"))
stages = doc.get("meta", {}).get("pipeline", {}).get("stages", {})
dropped = [s for s in STAGES if s in stages]
for s in dropped:
    stages.pop(s, None)
# Selections belong to the segmentation that produced them. Re-running `match` means they are
# about to be rebuilt, and leaving the old ones behind makes the pool audit see selections
# pointing at sources the current pool no longer holds — measured: 11 of 13
# `selected_source_not_in_usable_pool` rows in a run whose fresh analyze changed 101 beats to 97.
n_sel = len(doc.get("selections") or [])
if "match" in dropped and n_sel:
    doc["selections"] = []
tmp = pj.with_suffix(".json.tmp")
tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
os.replace(tmp, pj)
log(f"invalidated checkpoints: {dropped or '(none present)'}"
    + (f"; dropped {n_sel} stale selection(s) belonging to the previous segmentation"
       if "match" in dropped and n_sel else ""))
log(f"kept: {sorted(stages)}")

cats = musiclib.scan()
log(f"musiclib: {len(cats)} categories / {sum(len(v) for v in cats.values())} tracks (need 11/118)")

t0 = time.time()
try:
    res = produce_auto(
        str(P),
        topic=TOPIC,
        script_path=str(P / "script.txt"),
        movie_hint="Game of Thrones",
        policy="block",
        max_sources=6,
        theme="history",
        captions=True,
        caption_style="professional",
        voiceover=str(P / "voiceover.wav"),
        use_tts=False,
        verify=True,
        do_build=True,
        resume=True,
        progress=log,
    )
except Exception as e:                                            # noqa: BLE001
    log(f"BLOCKED/FAILED after {(time.time() - t0) / 60:.1f} min: {type(e).__name__}: {e}")
    raise
log(f"produce_auto done in {(time.time() - t0) / 60:.1f} min → {res}")
