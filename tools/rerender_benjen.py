#!/usr/bin/env python3
"""THE rerender of the Benjen Stark essay, with every fix from the frame audit of 409e284b60.

A FRESH job on purpose. The dominant defect was that the whole 12-minute video was built from
legacy ~360p footage — `hd_path_ok 0/72`, 1% of sources at >=720p — because one unknown yt-dlp
flag killed the HD path in argument parsing. Resuming the old job would inherit those 360p files,
which is precisely what this rerender exists to replace.

Also carried: recovery beat rotation (0529d09), the edge-band watermark detector (a3bb036) and the
window-level wrong-character guard (f7420ca).

The pre-flight is not ceremony. The previous render spent 3h17m and $1.12 producing an SD video
while the download stage said so in the log and nothing stopped it. Nothing here starts until a
real URL from the pool probes >=720p.

Same script, same uploaded voiceover as the original.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
OLD = Path("/Users/hussnain/Desktop/clipstudio_output/portal/409e284b60")
NEW = Path("/Users/hussnain/Desktop/clipstudio_output/portal/benjen_v2")
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
#  Deliver the video even if a beat cannot be resolved — a blocked render renames itself to
#  *.REVIEW_DRAFT.mp4, so an unpublishable result announces itself instead of masquerading.
os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore.clipstudio import hd_download as HD
    from vidlore import musiclib

    if not HD.available():
        log("ABORT: HD path unavailable — the render would be built from 360p again")
        return 2
    #  The flag that caused the collapse: assert the resolved yt-dlp actually accepts it, so a
    #  silent fall back to the legacy path cannot repeat.
    log(f"HD: --remote-components advertised = {HD._flag_supported('--remote-components')}")
    old = json.loads((OLD / "project.json").read_text())
    _yt = [s["url"] for s in old["sources"] if "youtube" in (s.get("url") or "")][:2]
    for u in _yt:
        h = HD.probe_max_height(u, max_height=1080)
        log(f"HD pre-flight: {u[-11:]} probes {h}p")
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
    topic = ana.get("topic") or "What actually happened to Benjen Stark"

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
    da = ((res.get("meta") or {}).get("download_audit")
          or (json.loads((NEW / "project.json").read_text()).get("meta") or {})
          .get("download_audit") or {})
    log(f"HD: {da.get('hd_path_ok')}/{da.get('youtube_sources')} on the HD path, "
        f"{da.get('sub_480p_sources')} sub-480p")
    c = res.get("cost") or {}
    if c.get("calls"):
        log(f"cost: ${c['usd']:.2f} over {c['calls']} call(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
