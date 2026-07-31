#!/usr/bin/env python3
"""THE rerender of the Arya / Night-King essay (original portal job 5cab63d801).

A FRESH job on purpose: the original shipped 110/110 sources at 360p because every HD download
was rejected with YouTube "Error code: 152" (a PO-token/SABR failure the classifier read as
"other", so the recovery sweep never fired). That is now classified as transient, and the same
pool re-probes at ~79% >=720p — but only a fresh job re-downloads and re-indexes at HD. Reusing
the old index would keep the 360p embeds, the 15%-named Face-ID, and therefore the blind gates
that produced the 5.43/10 audit.

Same script, same uploaded voiceover as the original.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
OLD = Path("/Users/hussnain/Desktop/clipstudio_output/portal/5cab63d801")
NEW = Path("/Users/hussnain/Desktop/clipstudio_output/portal/arya_v2")
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
os.environ["VIDLORE_CLIPSTUDIO_OCR_POOL_OK"] = "1"          # __main__-guarded, spawn-safe
os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore.clipstudio import hd_download as HD
    from vidlore import musiclib

    # PRE-FLIGHT — the previous render burned 4 hours producing SD because nothing checked this
    if not HD.available():
        log("ABORT: HD path unavailable — a rerender would repeat the 360p render")
        return 2
    h = HD.probe_max_height("https://www.youtube.com/watch?v=-rwPa3H1MFU", max_height=1080)
    log(f"HD pre-flight: probe returns {h}p")
    if h < 720:
        log("ABORT: HD probe still sub-720p — fix the PO-token path first")
        return 2
    cats = musiclib.scan()
    n = sum(len(v) for v in cats.values())
    log(f"music: {len(cats)} categories / {n} tracks")
    if len(cats) != 11 or n != 118:
        log("ABORT: music library incomplete — build would raise at the end")
        return 2

    old = json.load(open(OLD / "project.json"))
    segs = sorted(old["segments"], key=lambda s: s["index"])
    script = "\n".join((s.get("text") or "").strip() for s in segs if (s.get("text") or "").strip())
    topic = ((old.get("meta") or {}).get("analysis") or {}).get("topic") \
        or "Season 1 Already Showed Arya Killing The Night King"

    NEW.mkdir(parents=True, exist_ok=True)
    (NEW / "script.txt").write_text(script, encoding="utf-8")
    if not (NEW / "voiceover.mp3").exists():
        shutil.copy2(OLD / "voiceover.mp3", NEW / "voiceover.mp3")
    log(f"script {len(script.split())} words / {len(segs)} beats -> {NEW}")

    t0 = time.time()
    res = produce_auto(str(NEW), topic=topic, script_text=script,
                       movie_hint="Game of Thrones", policy="approved_testing",
                       max_sources=110, theme="history", captions=True,
                       verify=True, do_build=True,
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
