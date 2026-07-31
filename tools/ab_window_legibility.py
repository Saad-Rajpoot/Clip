#!/usr/bin/env python3
"""A/B the aired-window legibility check on a real render — build stage only.

match/verify are untouched by the change, so re-running them would only add noise and cost. This
re-runs build_video twice from the same finished checkpoints, flag off then on, and reports the
measurements the design named as decisive:

  PRIMARY   how many clips the post-cut sweep had to freeze-replace, and how many of those took a
            donor from a DIFFERENT SCENE. That is the damage the change exists to prevent.
  NO-OP     how many of the delivered per-beat clips are byte-identical between the arms. The design
            predicts ~244 of 272 unchanged; anything else means it is reaching beats it should not.
  STARVATION rejected_footage_audit.json and the window-qc counts must not move. The option set is
            supposed to be a strict superset, so rejection cannot grow.

    python3 tools/ab_window_legibility.py <job dir> <out dir>
"""
import hashlib
import json
import os
import re
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

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def run_arm(job: Path, out: Path, tag: str, on: bool) -> dict:
    os.environ["VIDLORE_CLIPSTUDIO_WINDOW_LEGIBILITY"] = "1" if on else "0"
    for m in [m for m in list(sys.modules) if m.startswith("vidlore")]:
        del sys.modules[m]
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.build import build_video

    lines = []
    proj = ClipProject.load(job)
    cfg = ClipConfig()
    segs = list(proj.segments)
    t = time.time()
    try:
        vid = build_video(proj, segs, cfg, voice="", captions=True, theme_name="history",
                          voiceover=str(job / "voiceover.mp3"), use_tts=False,
                          progress=lambda m: lines.append(str(m)))
    except Exception as e:
        log(f"{tag}: build raised {type(e).__name__}: {str(e)[:160]}")
        vid = None
    wall = time.time() - t

    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "build_lines.txt").write_text("\n".join(lines), encoding="utf-8")
    for name in ("rejected_footage_audit.json", "final_black_failures.json"):
        src = job / "output" / name
        if src.exists():
            shutil.copy2(src, d / name)
    clips = {p.name: sha(p) for p in sorted((job / "clips").glob("beat_*.mp4"))}
    (d / "clip_hashes.json").write_text(json.dumps(clips, indent=1))

    freeze = [l for l in lines if "unreadable-clip removal" in l]
    cross = [l for l in freeze if "PREVIOUS-scene" in l or "cross-scene" in l]
    kept = [l for l in lines if "keeping the ranked first choice" in l]
    wqc = [l for l in lines if "window-qc" in l]
    res = {"tag": tag, "on": on, "video": str(vid) if vid else "", "wall_s": round(wall, 1),
           "clips": len(clips), "freeze_replaced": len(freeze), "cross_scene_donor": len(cross),
           "all_windows_dark": len(kept), "window_qc_lines": len(wqc)}
    log(f"{tag}: {json.dumps(res)}")
    return res


def main():
    job, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    saved = out / "project_before.json"
    shutil.copy2(job / "project.json", saved)

    try:
        off = run_arm(job, out, "off", on=False)
        on = run_arm(job, out, "on", on=True)
    finally:
        shutil.copy2(saved, job / "project.json")

    h_off = json.loads((out / "off" / "clip_hashes.json").read_text())
    h_on = json.loads((out / "on" / "clip_hashes.json").read_text())
    common = set(h_off) & set(h_on)
    same = [k for k in common if h_off[k] == h_on[k]]
    diff = sorted(k for k in common if h_off[k] != h_on[k])

    print("\n" + "=" * 78)
    print(f"{'':22}{'OFF':>10}{'ON':>10}")
    for k in ("freeze_replaced", "cross_scene_donor", "all_windows_dark", "clips", "wall_s"):
        print(f"{k:<22}{str(off[k]):>10}{str(on[k]):>10}")
    print(f"\nclips byte-identical: {len(same)}/{len(common)}  changed: {len(diff)}")
    print(f"changed clips: {diff[:20]}")

    for name in ("rejected_footage_audit.json",):
        a, b = out / "off" / name, out / "on" / name
        if a.exists() and b.exists():
            print(f"{name}: {'IDENTICAL' if sha(a) == sha(b) else 'DIFFERS — investigate'}")
    (out / "summary.json").write_text(json.dumps(
        {"off": off, "on": on, "identical": len(same), "changed": diff}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
