"""Deterministic test of assemble._repair_black_frames against the two
concrete flaws behind the deleted-scene black-span bug.

Builds tiny synthetic videos with known black spans, runs the REAL
_repair_black_frames on each, and reports blackdetect (QA params
d=0.15 pic_th=0.98) + duration before/after. Run it once on the
current code, then again after the fix, to see the difference.

  Case A  d-blind-spot : [bright 2s][BLACK 0.20s][bright 2s]
          A 0.20s span is < the old repair's min_d=0.30 -> never
          repaired, yet QA (d=0.15) flags it.
  Case B  dark-anchor  : [BLACK 0.60s][bright 2s]
          Span starts at t=0, so the "previous frame" (s-1/fps) is
          itself black -> the old freeze-hold repeats black.
  Case C  no-regression: [bright 2s][BLACK 1.0s][bright 2s]
          The classic corrupt-scene case: a bright frame precedes the
          gap, so both old and new repair must clean it.

Usage:  .venv/bin/python tools/synth_black_test.py
"""
import shutil
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vidlore.ffmpeg_tool import ffmpeg_exe          # noqa: E402
from vidlore import assemble as A                   # noqa: E402

FF = ffmpeg_exe()
W = ROOT / "output" / "_synth_black_test"


def _seg(color, dur, out):
    subprocess.run(
        [FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c={color}:s=1280x720:r=30:d={dur}",
         "-vf", "format=yuv420p", "-pix_fmt", "yuv420p", str(out)],
        check=True)


def build(name, parts):
    d = W / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    segs = []
    for i, (c, dur) in enumerate(parts):
        s = d / f"s{i}.mp4"
        _seg(c, dur, s)
        segs.append(s)
    lst = d / "list.txt"
    lst.write_text("\n".join(f"file '{s.name}'" for s in segs) + "\n")
    out = d / "in.mp4"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    return out, d


def scan(video, d=0.15, pic_th=0.98, pix_th=0.10):
    r = subprocess.run(
        [FF, "-hide_banner", "-nostats", "-i", str(video),
         "-vf", f"blackdetect=d={d}:pic_th={pic_th}:pix_th={pix_th}",
         "-an", "-f", "null", "-"], capture_output=True, text=True)
    return [(float(a), float(b)) for a, b in re.findall(
        r"black_start:([\d.]+).*?black_end:([\d.]+)", r.stderr)]


def dur(video):
    r = subprocess.run([FF, "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
            + float(m.group(3))) if m else 0.0


CASES = [
    ("A_d_blind_spot", [("0x808080", 2.0), ("black", 0.20), ("0x808080", 2.0)]),
    ("B_dark_anchor",  [("black", 0.60), ("0x808080", 2.0)]),
    ("C_no_regression", [("0x808080", 2.0), ("black", 1.0), ("0x808080", 2.0)]),
]


def main():
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    allpass = True
    for name, parts in CASES:
        vin, d = build(name, parts)
        b = scan(vin)
        bd = dur(vin)
        out = A._repair_black_frames(vin, d, 30)
        a = scan(out)
        ad = dur(out)
        ok = (not a) and abs(ad - bd) < 0.20
        allpass = allpass and ok
        print(f"[{name}]")
        print(f"   before: {len(b)} span(s) {[f'{s:.2f}-{e:.2f}' for s,e in b]}"
              f"  dur={bd:.2f}s")
        print(f"   after : {len(a)} span(s) {[f'{s:.2f}-{e:.2f}' for s,e in a]}"
              f"  dur={ad:.2f}s  (Δ{ad-bd:+.2f})")
        print(f"   => {'PASS ✅' if ok else 'FAIL ❌'}")
    print("OVERALL:", "ALL PASS ✅" if allpass else "FAILURES ❌")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
