#!/usr/bin/env python3
"""Render the SAME window with and without the legibility grade, for a vision A/B.

A brightness change is a taste decision and taste decisions have to be looked at, not asserted.
This takes the darkest beats of a real job, cuts each window twice — as the pipeline does today and
as it would with the grade — and writes the pairs plus a measured luma table.

    python3 tools/legibility_ab.py <job_dir> <out_dir> [--n 14]

Emits <out>/before/beat_<i>.jpg, <out>/after/beat_<i>.jpg and <out>/ab.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidlore.clipstudio.config import ffmpeg_exe  # noqa: E402
from vidlore.clipstudio.cut import _luma_stats, legibility_gamma  # noqa: E402


def frame(src: str, t: float, vf: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-nostats", "-ss", f"{t:.3f}", "-i", src,
           "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-q:v", "2", str(dest)]
    p = subprocess.run(cmd, capture_output=True, timeout=90)
    return p.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=14)
    a = ap.parse_args()

    job, out = Path(a.job), Path(a.out)
    proj = json.loads((job / "project.json").read_text(encoding="utf-8"))
    srcs = {s["id"]: s for s in proj.get("sources") or []}

    rows = []
    for sel in proj.get("selections") or []:
        src = srcs.get(sel.get("source_id") or "")
        if not src or not src.get("local_path"):
            continue
        i0, i1 = float(sel["in_point"]), float(sel["out_point"])
        st = _luma_stats(src["local_path"], i0, max(0.5, i1 - i0))
        if st is None:
            continue
        yavg, spread = st
        rows.append(dict(beat=sel["segment_index"], src=src["local_path"],
                         title=(src.get("title") or "")[:60],
                         t=i0 + (i1 - i0) * 0.5, yavg=round(yavg, 1),
                         spread=round(spread, 1), gamma=legibility_gamma(yavg, spread)))

    graded = [r for r in rows if r["gamma"] > 1.0]
    graded.sort(key=lambda r: r["yavg"])
    pick = graded[: a.n]
    print(f"{len(rows)} beats probed · {len(graded)} would be graded "
          f"({100 * len(graded) / max(1, len(rows)):.0f}%) · rendering {len(pick)} pairs")

    done = 0
    for r in pick:
        ok_b = frame(r["src"], r["t"], "", out / "before" / f"beat_{r['beat']:03d}.jpg")
        ok_a = frame(r["src"], r["t"], f"eq=gamma={r['gamma']}:contrast=1.06",
                     out / "after" / f"beat_{r['beat']:03d}.jpg")
        r["rendered"] = bool(ok_b and ok_a)
        done += r["rendered"]

    (out / "ab.json").write_text(json.dumps(
        {"probed": len(rows), "would_grade": len(graded), "pairs": pick}, indent=1),
        encoding="utf-8")
    print(f"rendered {done}/{len(pick)} pairs → {out}")
    print(f"\n{'beat':>5} {'YAVG':>6} {'spread':>7} {'gamma':>6}  source")
    for r in pick:
        print(f"{r['beat']:>5} {r['yavg']:>6} {r['spread']:>7} {r['gamma']:>6}  {r['title']}")
    if done < len(pick):
        raise SystemExit(f"only {done}/{len(pick)} pairs rendered — do not judge on a partial set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
