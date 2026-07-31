#!/usr/bin/env python3
"""Resume the Arya rerender from its checkpoints and finish it as a review draft.

The first pass ran 2h17m and stopped at the pre-assembly feasibility gate: beat 263 asks for a
MONTAGE ("a quick succession of shots of her targets: Meryn Trant, Walder Frey, Littlefinger"), so
no single shot can satisfy the same-scene test and self-heal resolved nothing in two rounds. That is
1 unresolved beat out of 288 — against 29 in the render this replaces — so the right outcome is a
watchable draft with that one beat reported, not a discarded render.

Stages analyze → recover are checkpointed `done`, so resume skips ~2h and goes straight to assembly.
RELEASE_BLOCK_MODE=warn is what makes the gate report instead of abort; the build now renames the
result to *.REVIEW_DRAFT.mp4 so the file says what it is.
"""
import json
import os
import sys
import time
from pathlib import Path

MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
JOB = Path("/Users/hussnain/Desktop/clipstudio_output/portal/arya_v2")
sys.path.insert(0, str(MAIN))

for _line in (MAIN / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")
os.environ["VIDLORE_CLIPSTUDIO_OCR_POOL_OK"] = "1"
os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore import musiclib

    cats = musiclib.scan()
    n = sum(len(v) for v in cats.values())
    log(f"music: {len(cats)} categories / {n} tracks")
    if len(cats) != 11 or n != 118:
        log("ABORT: music library incomplete — the build would raise at the end")
        return 2

    old = json.loads((JOB / "project.json").read_text())
    done = [k for k, v in (((old.get("meta") or {}).get("pipeline") or {}).get("stages")
                           or {}).items() if v.get("status") == "done"]
    log(f"checkpoints already done: {sorted(done)}")
    segs = sorted(old["segments"], key=lambda s: s["index"])
    script = "\n".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
    topic = ((old.get("meta") or {}).get("analysis") or {}).get("topic") \
        or "Season 1 Already Showed Arya Killing The Night King"

    t0 = time.time()
    res = produce_auto(str(JOB), topic=topic, script_text=script,
                       movie_hint="Game of Thrones", policy="approved_testing",
                       max_sources=110, theme="history", captions=True,
                       verify=True, do_build=True, resume=True,
                       voiceover=str(JOB / "voiceover.mp3"), progress=log)
    log(f"DONE in {(time.time() - t0) / 60:.1f} min -> {res.get('output')}")
    s = res.get("summary") or {}
    log(f"flags: {s.get('flag_breakdown')} | sources: {s.get('sources_used')}")
    c = res.get("cost") or {}
    if c.get("calls"):
        log(f"cost: ${c['usd']:.2f} over {c['calls']} call(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
