#!/usr/bin/env python3
"""THE rerender of the High Sparrow essay, with every fix from the frame audit of fc41397ea5.

A FRESH job on purpose. The fixes that matter most change what gets DISCOVERED and DOWNLOADED
(source screening) and what gets SELECTED (window substitution, legibility), so resuming the old
job's checkpoints would preserve exactly the decisions the audit found fault with.

The one thing a rerender alone could not fix was a pool gap: the audit proved by ASR scan that no
source in the 78-source pool carried Olenna at Dragonstone (S07E02) or Highgarden (S07E03), which is
why beat 102 release-blocked and why the 92-111 arc averaged 5.26. `max_sources` is a FLOOR, not a
cap — the previous run passed 8, which on a 181-beat script resolves to a budget of ~46 — so this
asks for a materially wider pool and lets discovery reach that material.

Same script, same uploaded voiceover as the original.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
OLD = Path("/Users/hussnain/Desktop/clipstudio_output/portal/fc41397ea5")
NEW = Path("/Users/hussnain/Desktop/clipstudio_output/portal/high_sparrow_v2")
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
# Deliver the video even if a beat still cannot be resolved: the build now RENAMES a blocked render
# to *.REVIEW_DRAFT.mp4, so an unpublishable result announces itself instead of masquerading.
os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore.clipstudio import hd_download as HD
    from vidlore import musiclib

    # PRE-FLIGHT — an earlier render burned four hours producing SD because nothing checked this,
    # and the HD path has since broken in three distinct ways (PO-token, Windows cookies, the
    # JS-challenge solver script). Probe a real URL from the pool this render will reuse.
    if not HD.available():
        log("ABORT: HD path unavailable — the render would be built from 360p")
        return 2
    old = json.loads((OLD / "project.json").read_text())
    _yt = [s["url"] for s in old["sources"]
           if (s.get("url") or "").find("youtube") >= 0][:1]
    if _yt:
        h = HD.probe_max_height(_yt[0], max_height=1080)
        log(f"HD pre-flight: probe returns {h}p")
        if h < 720:
            log("ABORT: HD probe sub-720p — fix the HD path before spending a render")
            return 2
    cats = musiclib.scan()
    n = sum(len(v) for v in cats.values())
    log(f"music: {len(cats)} categories / {n} tracks")
    if len(cats) != 11 or n != 118:
        log("ABORT: music library incomplete — build would raise at the end")
        return 2

    segs = sorted(old["segments"], key=lambda s: s["index"])
    script = "\n".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
    ana = (old.get("meta") or {}).get("analysis") or {}
    topic = ana.get("topic") or "The High Sparrow's One Mistake Doomed King's Landing"

    NEW.mkdir(parents=True, exist_ok=True)
    (NEW / "script.txt").write_text(script, encoding="utf-8")
    if not (NEW / "voiceover.mp3").exists():
        shutil.copy2(OLD / "voiceover.mp3", NEW / "voiceover.mp3")
    log(f"script {len(script.split())} words / {len(segs)} beats -> {NEW}")

    t0 = time.time()
    res = produce_auto(str(NEW), topic=topic, script_text=script,
                       movie_hint=ana.get("movie_title") or "Game of Thrones",
                       policy="approved_testing", max_sources=70, theme="history",
                       captions=True, verify=True, do_build=True,
                       voiceover=str(NEW / "voiceover.mp3"), progress=log)
    log(f"DONE in {(time.time() - t0) / 3600:.2f} h -> {res.get('output')}")
    s = res.get("summary") or {}
    log(f"flags: {s.get('flag_breakdown')} | sources: {s.get('sources_used')}")
    c = res.get("cost") or {}
    if c.get("calls"):
        log(f"cost: ${c['usd']:.2f} over {c['calls']} call(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
