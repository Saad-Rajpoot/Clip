"""Standalone black-span + luma analyzer for Vidlore QA.

Usage:
    python tools/blackscan.py <video.mp4> [d] [pic_th]

Prints every blackdetect span (default d=0.15 pic_th=0.98 — the Editor
Release QA params) plus the mean luma of each span's anchor frames
(the frame just before/after the span), so we can see whether the
black-frame repair's freeze anchor would itself be dark.
"""
import subprocess
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vidlore.ffmpeg_tool import ffmpeg_exe  # noqa: E402

FF = ffmpeg_exe()


def spans(video, d, pic_th, pix_th=0.10):
    r = subprocess.run(
        [FF, "-hide_banner", "-nostats", "-i", str(video),
         "-vf", f"blackdetect=d={d}:pic_th={pic_th}:pix_th={pix_th}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    out = []
    for ln in r.stderr.splitlines():
        ms = re.search(r"black_start:([\d.]+)", ln)
        me = re.search(r"black_end:([\d.]+)", ln)
        md = re.search(r"black_duration:([\d.]+)", ln)
        if ms and me:
            out.append((float(ms.group(1)), float(me.group(1)),
                        float(md.group(1)) if md else 0.0))
    return out


def luma(video, ts):
    r = subprocess.run(
        [FF, "-hide_banner", "-loglevel", "error",
         "-ss", f"{max(0.0, ts):.4f}", "-i", str(video),
         "-frames:v", "1", "-vf", "format=gray,scale=1:1:flags=area",
         "-f", "rawvideo", "-"],
        capture_output=True)
    if r.returncode == 0 and r.stdout:
        return r.stdout[0]
    return -1


def duration(video):
    r = subprocess.run([FF, "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0


def main():
    video = sys.argv[1]
    d = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    pic_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.98
    dur = duration(video)
    sp = spans(video, d, pic_th)
    print(f"VIDEO  {video}")
    print(f"DURATION {dur:.2f}s  ·  blackdetect d={d} pic_th={pic_th} pix_th=0.10")
    print(f"SPANS  {len(sp)}")
    for s, e, dd in sp:
        lb = luma(video, max(0.0, s - 0.05))   # just before span
        la = luma(video, min(dur, e + 0.05))   # just after span
        lm = luma(video, (s + e) / 2.0)        # mid span
        print(f"  [{s:8.3f} .. {e:8.3f}]  dur={dd:.3f}s  "
              f"luma(before/mid/after)={lb}/{lm}/{la}")
    print("RESULT", "CLEAN ✅" if not sp else f"{len(sp)} SPAN(S) ❌")
    return 0 if not sp else 1


if __name__ == "__main__":
    sys.exit(main())
