"""Direct, recipe-independent test of assemble._repair_black_frames.

Runs the ACTUAL repair function on a given MP4 (a real rendered video,
ideally one that has black spans) in an isolated work dir, then reports
blackdetect spans + duration BEFORE and AFTER. This isolates the
black-frame repair from the (non-deterministic) editorial-recipe render
so the fix can be proven on the exact same input.

Usage:
    .venv/bin/python tools/test_repair.py <video.mp4> [fps]
"""
import shutil
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vidlore.ffmpeg_tool import ffmpeg_exe                 # noqa: E402
from vidlore import assemble as A                          # noqa: E402

FF = ffmpeg_exe()


def scan(video, d=0.15, pic_th=0.98, pix_th=0.10):
    r = subprocess.run(
        [FF, "-hide_banner", "-nostats", "-i", str(video),
         "-vf", f"blackdetect=d={d}:pic_th={pic_th}:pix_th={pix_th}",
         "-an", "-f", "null", "-"], capture_output=True, text=True)
    out = []
    for ln in r.stderr.splitlines():
        ms = re.search(r"black_start:([\d.]+)", ln)
        me = re.search(r"black_end:([\d.]+)", ln)
        if ms and me:
            out.append((float(ms.group(1)), float(me.group(1))))
    return out


def dur(video):
    r = subprocess.run([FF, "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
            + float(m.group(3))) if m else 0.0


def main():
    src = Path(sys.argv[1]).resolve()
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    work = ROOT / "output" / "_repair_test_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    vin = work / "video_in.mp4"
    shutil.copyfile(src, vin)

    b_spans = scan(vin)
    b_dur = dur(vin)
    print(f"INPUT  {src.name}")
    print(f"  duration {b_dur:.3f}s   spans(d=0.15 pic_th=0.98): {len(b_spans)}")
    for s, e in b_spans:
        print(f"    [{s:.3f}..{e:.3f}] {e-s:.3f}s")

    print("running assemble._repair_black_frames ...", flush=True)
    out = A._repair_black_frames(vin, work, fps)

    a_spans = scan(out)
    a_dur = dur(out)
    print(f"OUTPUT {out.name}")
    print(f"  duration {a_dur:.3f}s   spans: {len(a_spans)}")
    for s, e in a_spans:
        print(f"    [{s:.3f}..{e:.3f}] {e-s:.3f}s")
    print(f"DURATION DELTA: {a_dur - b_dur:+.3f}s "
          f"({'OK' if abs(a_dur - b_dur) < 0.20 else 'DRIFT!'})")
    print("RESULT:",
          "CLEAN ✅" if not a_spans else f"{len(a_spans)} SPAN(S) REMAIN ❌")
    return 0 if not a_spans else 1


if __name__ == "__main__":
    sys.exit(main())
