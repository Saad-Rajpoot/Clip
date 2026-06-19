#!/usr/bin/env python3
"""Deep QA for a finished clipstudio job.

Usage: qa_video.py <job_dir> [final.mp4 name] [--frames]

Reports:
  • caption sync  — SRT cue freeze scan (cues holding >= FREEZE_S)
  • breakouts     — count + windows from the SRT gap structure + project meta
  • footage relevance — review_queue.json flagged beats / reasons / score spread
  • repetition    — same (source,in-point) reused across beats
  • length        — video vs voiceover
With --frames: extracts a keyframe at each scene midpoint into <job>/qa_frames/
so the frames can be eyeballed for scene-exactness.
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FREEZE_S = 6.0

job = Path(sys.argv[1])
mp4_name = next((a for a in sys.argv[2:] if a.endswith(".mp4")), None)
want_frames = "--frames" in sys.argv
out = job / "output"


def srt_cues(p):
    cues = []
    for blk in Path(p).read_text(encoding="utf-8").strip().split("\n\n"):
        ln = blk.splitlines()
        if len(ln) >= 3 and "-->" in ln[1]:
            def s(t):
                t = t.replace(",", ":").split(":")
                return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2]) + int(t[3]) / 1000
            a, b = ln[1].split(" --> ")
            cues.append((s(a), s(b), " ".join(ln[2:])))
    return cues


print(f"\n{'='*70}\nDEEP QA · {job.name}\n{'='*70}")

# ---- caption sync ----
srt = out / (Path(mp4_name).with_suffix(".srt").name if mp4_name else "final.srt")
if not srt.exists():
    srt = out / "final.srt"
if srt.exists():
    c = srt_cues(srt)
    freezes = [(a, b, t) for a, b, t in c if (b - a) >= FREEZE_S]
    mx = max((b - a for a, b, t in c), default=0)
    print(f"\n[CAPTIONS] {srt.name}: {len(c)} cues · longest={mx:.1f}s · freezes(>={FREEZE_S}s)={len(freezes)}")
    for a, b, t in freezes[:10]:
        print(f"    FREEZE [{a:6.1f}->{b:6.1f}] {b-a:4.1f}s : {t[:60]}")
    # gaps in the caption stream (breakout windows = silence with no cue)
    gaps = []
    for i in range(1, len(c)):
        g = c[i][0] - c[i - 1][1]
        if g >= 2.0:
            gaps.append((c[i - 1][1], c[i][0], g))
    print(f"[GAPS] {len(gaps)} caption gaps >=2s (breakout/pause windows):")
    for a, b, g in gaps[:12]:
        print(f"    gap [{a:6.1f}->{b:6.1f}] {g:4.1f}s")
else:
    print("\n[CAPTIONS] no SRT found")

# ---- footage relevance (review_queue) ----
rq = job / "review_queue.json"
if rq.exists():
    d = json.load(open(rq))
    items = d.get("items", [])
    total = None
    pj = job / "project.json"
    if pj.exists():
        total = len(json.load(open(pj)).get("segments", []))
    reasons = Counter()
    for it in items:
        for r in it.get("reasons", []):
            reasons[r] += 1
    print(f"\n[RELEVANCE] flagged {len(items)}/{total} beats" + (f" ({100*len(items)/total:.0f}%)" if total else ""))
    for r, n in reasons.most_common():
        print(f"    {n:3d} × {r}")
else:
    print("\n[RELEVANCE] no review_queue.json")

# ---- repetition + breakouts (project.json selections) ----
pj = job / "project.json"
if pj.exists():
    proj = json.load(open(pj))
    sels = proj.get("selections", [])
    src_use = Counter(s.get("source_id", "?") for s in sels)
    win_use = Counter((s.get("source_id", "?"), round(float(s.get("in_point", 0)) / 3.0)) for s in sels)
    print(f"\n[REPETITION] {len(sels)} selections · {len(src_use)} distinct sources")
    print(f"    most-reused source: {src_use.most_common(1)}")
    rep = [(k, n) for k, n in win_use.most_common() if n >= 3]
    print(f"    (source,~3s window) reused >=3x: {len(rep)}")
    for k, n in rep[:6]:
        print(f"        {n}× {k}")

# ---- length ----
def dur(path):
    try:
        import imageio_ffmpeg, re
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return None


mp4 = out / mp4_name if mp4_name else (out / "final.mp4")
vo = job / "voiceover.mp3"
dv, da = dur(mp4), dur(vo)
print(f"\n[LENGTH] video={dv}s · voiceover={da}s" + (f" · diff={dv-da:+.1f}s" if (dv and da) else ""))

# ---- frames ----
if want_frames and mp4.exists() and srt.exists():
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    fdir = job / "qa_frames"
    fdir.mkdir(exist_ok=True)
    c = srt_cues(srt)
    # one frame at each ~30s, plus midpoints of caption gaps (breakouts)
    times = sorted(set([round(t, 1) for t in [i * 30 for i in range(int((dv or 0) // 30) + 1)]]))
    for t in times:
        op = fdir / f"t{int(t):04d}.jpg"
        subprocess.run([ff, "-ss", str(t), "-i", str(mp4), "-frames:v", "1", "-q:v", "3",
                        "-y", str(op)], capture_output=True)
    print(f"\n[FRAMES] {len(times)} keyframes → {fdir}")

print(f"\n{'='*70}\nQA DONE\n{'='*70}")
