#!/usr/bin/env python3
"""A/B the aired-window legibility check on a SHORT slice of a finished job.

The full-render version of this measurement costs two 22-minute builds. This cuts a contiguous beat
range out of an existing job — same sources, same index, same match/verify decisions — and builds
that twice, flag off then on. The slice is chosen for DENSITY of the thing under test, not at
random: probing every beat's first-choice window found 21 of 272 dark, and beats 126-149 hold 5 of
them (including beat 131, which the paired A/B named as a 9 -> 3 regression).

What it cannot show, and the full render can: the headline "31 freeze-replacements -> N" number.
What it does show: whether a dark first choice is replaced by a legible alternate from the beat's own
list, whether anything else moves (it should not), and whether rejection grows (it must not).

    python3 tools/ab_window_legibility_slice.py <job> <out> [--lo 126] [--hi 150]
"""
import argparse
import hashlib
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
os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"

FF = ("/Users/hussnain/Library/Python/3.9/lib/python/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def make_slice(src: Path, dst: Path, lo: int, hi: int) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in ("sources", "index"):
        os.symlink(src / name, dst / name)

    # THE SLICE MUST START AT BEAT 0. A first attempt cut mid-video using cumulative ledger
    # `duration`, and both arms died with a 76s TimelineSyncError: that field is the CLIP length,
    # not the narration span — it sums to 843s against a 1296s voiceover — so the cut landed
    # hundreds of seconds from the right words and the aligner matched almost nothing. There is no
    # cheap, reliable beat->voiceover offset (the final timeline carries breakout inserts too), and
    # an approximate one silently produces a garbage measurement, which is worse than none.
    if lo != 0:
        raise SystemExit(
            f"--lo must be 0: a mid-video slice needs a voiceover offset this harness cannot "
            f"compute reliably (got lo={lo}). Widen --hi instead to reach the beats you want.")
    rows = [json.loads(l) for l in (src / "ledger.jsonl").read_text().splitlines() if l.strip()]
    off = 0.0
    span = sum(float(r.get("duration") or 0.0) for r in rows[lo:hi]) * 1.9 + 20.0
    log(f"slice beats {lo}-{hi-1}: voiceover from 0.0s, taking {span:.1f}s "
        f"(generous — the aligner stops at the last beat's words)")

    d = json.loads((src / "project.json").read_text())
    d["segments"] = [s for s in d["segments"] if lo <= s["index"] < hi]
    d["selections"] = [s for s in d.get("selections", []) if lo <= s["segment_index"] < hi]
    (dst / "project.json").write_text(json.dumps(d))

    vo = dst / "voiceover.mp3"
    r = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{off:.3f}", "-i", str(src / "voiceover.mp3"),
                        "-t", f"{span + 1.0:.3f}", "-c", "copy", str(vo)],
                       capture_output=True, text=True)
    if r.returncode or not vo.exists():
        raise SystemExit(f"voiceover slice failed: {r.stderr[-200:]}")
    log(f"slice job ready: {len(d['segments'])} beats -> {dst}")


def run_arm(job: Path, out: Path, tag: str, on: bool) -> dict:
    os.environ["VIDLORE_CLIPSTUDIO_WINDOW_LEGIBILITY"] = "1" if on else "0"
    for m in [m for m in list(sys.modules) if m.startswith("vidlore")]:
        del sys.modules[m]
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.build import build_video

    lines = []
    proj = ClipProject.load(job)
    segs = list(proj.segments)
    t = time.time()
    vid = None
    try:
        vid = build_video(proj, segs, ClipConfig(), voice="", captions=True,
                          theme_name="history", voiceover=str(job / "voiceover.mp3"),
                          use_tts=False, progress=lambda m: lines.append(str(m)))
    except Exception as e:
        log(f"{tag}: build raised {type(e).__name__}: {str(e)[:200]}")
    wall = time.time() - t

    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "build_lines.txt").write_text("\n".join(lines), encoding="utf-8")
    for name in ("rejected_footage_audit.json", "final_black_failures.json"):
        s = job / "output" / name
        if s.exists():
            shutil.copy2(s, d / name)
    if vid and Path(vid).exists():
        shutil.copy2(vid, d / "final.mp4")
    clips = {p.name: sha(p) for p in sorted((job / "clips").glob("beat_*.mp4"))}
    (d / "clip_hashes.json").write_text(json.dumps(clips, indent=1))

    frz = [l for l in lines if "unreadable-clip removal" in l]
    res = {"tag": tag, "on": on, "wall_s": round(wall, 1), "clips": len(clips),
           "freeze_replaced": len(frz),
           "cross_scene_donor": sum(1 for l in frz if "PREVIOUS-scene" in l),
           "all_windows_dark": sum(1 for l in lines if "keeping the ranked first choice" in l),
           "video": str(vid) if vid else ""}
    log(f"{tag}: {json.dumps(res)}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--lo", type=int, default=126)
    ap.add_argument("--hi", type=int, default=150)
    a = ap.parse_args()
    src, out = Path(a.job), Path(a.out)
    sl = out.parent / (out.name + "_job")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    make_slice(src, sl, a.lo, a.hi)
    saved = out / "project_slice.json"
    shutil.copy2(sl / "project.json", saved)

    off = run_arm(sl, out, "off", on=False)
    shutil.copy2(saved, sl / "project.json")          # each arm starts from the same state
    shutil.rmtree(sl / "clips", ignore_errors=True)
    on = run_arm(sl, out, "on", on=True)

    h_off = json.loads((out / "off" / "clip_hashes.json").read_text())
    h_on = json.loads((out / "on" / "clip_hashes.json").read_text())
    common = set(h_off) & set(h_on)
    same = [k for k in common if h_off[k] == h_on[k]]
    diff = sorted(k for k in common if h_off[k] != h_on[k])

    print("\n" + "=" * 74)
    print(f"{'':22}{'OFF':>10}{'ON':>10}")
    for k in ("freeze_replaced", "cross_scene_donor", "all_windows_dark", "clips", "wall_s"):
        print(f"{k:<22}{str(off[k]):>10}{str(on[k]):>10}")
    print(f"\nclips byte-identical: {len(same)}/{len(common)} · changed: {len(diff)}")
    print(f"changed: {diff}")
    for name in ("rejected_footage_audit.json",):
        x, y = out / "off" / name, out / "on" / name
        if x.exists() and y.exists():
            print(f"{name}: {'IDENTICAL' if sha(x) == sha(y) else 'DIFFERS — investigate'}")
        else:
            print(f"{name}: absent in one/both arms ({x.exists()}/{y.exists()})")
    (out / "summary.json").write_text(json.dumps(
        {"off": off, "on": on, "identical": len(same), "changed": diff}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
