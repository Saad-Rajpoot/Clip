"""Forensic black-dip verifier for a rendered video.

Reports the two acceptance gates for the dark-stat-card fix:
  1) ffmpeg blackdetect span count (d=0.05 pic_th=0.98)  -> must be 0
  2) per-frame YAVG (signalstats) global min + any frames < 25  -> none

Usage: python tools/verify_blackfix.py <video.mp4> [focus_start focus_end]
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vidlore.ffmpeg_tool import ffmpeg_exe   # noqa: E402

FF = ffmpeg_exe()


def blackdetect(video, d=0.05, pic_th=0.98):
    cmd = [FF, "-hide_banner", "-i", str(video),
           "-vf", f"blackdetect=d={d}:pic_th={pic_th}", "-an", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    spans = re.findall(r"black_start:(\S+) black_end:(\S+) black_duration:(\S+)",
                       r.stderr)
    return spans


def yavg_scan(video, ss=None, t=None):
    cmd = [FF, "-hide_banner"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    if t is not None:
        cmd += ["-t", str(t)]
    cmd += ["-i", str(video),
            "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
            "-an", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    vals, times = [], []
    cur_t = None
    for line in r.stderr.splitlines():
        m = re.search(r"pts_time:(\S+)", line)
        if m:
            cur_t = float(m.group(1))
        m = re.search(r"YAVG=(\S+)", line)
        if m:
            vals.append(float(m.group(1)))
            times.append(cur_t)
    return times, vals


def main():
    video = sys.argv[1]
    focus = None
    if len(sys.argv) >= 4:
        focus = (float(sys.argv[2]), float(sys.argv[3]))
    print(f"VIDEO: {video}")
    spans = blackdetect(video)
    print(f"\n[GATE 1] blackdetect (d=0.05 pic_th=0.98): {len(spans)} span(s)")
    for s, e, d in spans:
        print(f"   black {s}-{e} ({d}s)")

    times, vals = yavg_scan(video)
    if vals:
        gmin = min(vals)
        gmin_t = times[vals.index(gmin)]
        under = [(times[i], v) for i, v in enumerate(vals) if v < 25.0]
        print(f"\n[GATE 2] YAVG over {len(vals)} frames: "
              f"global min = {gmin:.2f} @ t={gmin_t}s")
        print(f"   frames with YAVG < 25: {len(under)}")
        for tt, v in under[:12]:
            print(f"      t={tt}s  YAVG={v:.2f}")
    verdict = (len(spans) == 0 and vals and min(vals) >= 25.0)
    print(f"\n==> {'PASS' if verdict else 'FAIL'} "
          f"(0 black spans AND no YAVG dip < 25)")

    if focus:
        ft, fv = yavg_scan(video, ss=focus[0], t=focus[1] - focus[0])
        if fv:
            print(f"\n[FOCUS {focus[0]}-{focus[1]}s] "
                  f"min YAVG = {min(fv):.2f}, max = {max(fv):.2f}")


if __name__ == "__main__":
    main()
