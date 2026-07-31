#!/usr/bin/env python3
"""Render ONLY the intro of an existing job — a real MP4, not a frame-level proxy.

The scene-identity fixes (f62f225) were measured with research/eval/intro_ab.py, which judges
frames pulled straight from the source files at the selected window. That is fast and it is what
the viewer will see, but it is NOT a render: no cuts, no captions, no music, no encode. This
produces the actual short video so the intro can be watched.

Beat durations come from the job's ledger and are NARRATION spans (the final timeline is longer —
breakouts splice extra audio in), so summing them gives the point to cut the uploaded voiceover.

    python3 tools/render_intro_test.py <job dir> <out job dir> [--beats 20]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
sys.path.insert(0, str(MAIN))

for _line in (MAIN / ".env").read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, v = _line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))

FF = ("/Users/hussnain/Library/Python/3.9/lib/python/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--beats", type=int, default=20)
    a = ap.parse_args()
    src, dst = Path(a.job), Path(a.out)

    from vidlore import musiclib
    cats = musiclib.scan()
    n = sum(len(v) for v in cats.values())
    log(f"music: {len(cats)} categories / {n} tracks")
    if len(cats) != 11 or n != 118:
        log("ABORT: music library incomplete — the build would ship silent")
        return 2

    # ---- narration span of the intro, from the ledger's per-beat durations
    rows = [json.loads(l) for l in (src / "ledger.jsonl").read_text().splitlines() if l.strip()]
    span = sum(float(r.get("duration") or 0) for r in rows[:a.beats])
    log(f"intro = beats 0-{a.beats-1}, {span:.2f}s of narration")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("sources", "index"):
        os.symlink(src / name, dst / name)

    d = json.loads((src / "project.json").read_text())
    d["segments"] = [s for s in d["segments"] if s["index"] < a.beats]
    d["selections"] = []                      # re-match from scratch with the fixed ranking
    (dst / "project.json").write_text(json.dumps(d))
    log(f"job: {len(d['segments'])} beats / {len(d['sources'])} sources -> {dst}")

    # ---- voiceover slice (+1.5s so the last word is not clipped mid-alignment)
    vo = dst / "voiceover.mp3"
    r = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i",
                        str(src / "voiceover.mp3"), "-t", f"{span + 1.5:.2f}",
                        "-c", "copy", str(vo)], capture_output=True, text=True)
    if r.returncode or not vo.exists():
        log(f"ABORT: voiceover slice failed — {r.stderr[-200:]}")
        return 2
    log(f"voiceover sliced -> {vo.stat().st_size/1e6:.1f} MB")

    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio.build import build_video
    from vidlore.clipstudio import llm

    llm.reset_usage()
    proj = ClipProject.load(dst)
    cfg, eng = ClipConfig(), engine_config()
    segs = list(proj.segments)

    log("match…")
    proj.selections = match_segments(proj, segs, cfg, progress=None)
    log("verify…")
    V.verify_and_repair(proj, segs, cfg, eng, progress=None)
    proj.save()
    log("build…")
    out = build_video(proj, segs, cfg, voice="", captions=True, theme_name="history",
                      voiceover=str(vo), use_tts=False, progress=log)
    u = llm.usage_summary()
    log(f"DONE -> {out}   (${u['usd']:.2f} / {u['calls']} call(s))")

    # ---- NEVER call a render final without probing it (a 20-min cut once shipped with no VO)
    pr = subprocess.run(["/usr/bin/env", "ffprobe", "-v", "error", "-show_entries",
                         "format=duration:stream=codec_type,codec_name", "-of", "json", str(out)],
                        capture_output=True, text=True)
    log("probe: " + " ".join(pr.stdout.split()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
