#!/usr/bin/env python3
"""Finalize LAYER-2: rebuild the inventory from already-extracted filmstrips +
the cached segment list, seed the known sparse-graphic (number/keyword/document)
windows that edge-detection misses on near-black frames, and write a clean
JSON-safe motion_graphics_inventory.json. Fast (no full re-extract)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
try:
    from vidlore.ffmpeg_tool import ffmpeg_exe
    FF = ffmpeg_exe()
except Exception:                                              # noqa: BLE001
    FF = "ffmpeg"

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/magnatesmedia_av.mp4"
outd = _REPO / "research" / "forensic_refs" / "magnatesmedia_motion_graphics"

# evidence I identified by eye (full-res hunt sheet) that edge-detection misses
SEED = {
    "number": [689.0, 2125.0], "text": [366.0],
    "document": [2630.0, 1598.0], "map": [135.7],
}
CATDIR = {"portrait": "portrait_animation_windows", "map": "map_animation_windows",
          "document": "document_animation_windows", "number": "number_animation_windows",
          "text": "text_animation_windows", "motion": "dense_motion_windows"}


def _strip(t0, dur, out, cols=12):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{max(0,t0):.2f}",
                    "-i", VIDEO, "-t", f"{dur:.2f}", "-vf",
                    f"fps={cols/max(0.5,dur):.2f},scale=300:-1,tile={cols}x1",
                    "-frames:v", "1", str(out)], check=False)


def main():
    s = json.loads((outd / "full_frame_scan.json").read_text())
    fps = s["fps"]; edge = np.array(s["edge"]); diff = np.array(s["diff"])
    thr = float(edge.mean() + 0.5 * edge.std())
    gp = edge > thr
    segs = []
    run = None; gap = 0; minlen = int(0.4 * fps); maxgap = int(0.4 * fps)
    for i, v in enumerate(gp):
        if v:
            run = run if run is not None else i; gap = 0
        elif run is not None:
            gap += 1
            if gap > maxgap:
                if (i - gap) - run >= minlen:
                    segs.append((run, i - gap))
                run = None
    if run is not None:
        segs.append((run, len(gp) - 1))

    # map extracted filmstrip timestamps -> category
    tcat = {}
    for cat, d in CATDIR.items():
        for p in (outd / d).glob("*.png"):
            m = re.search(r"_(\d+\.\d+)s\.png$", p.name)
            if m:
                tcat[round(float(m.group(1)), 1)] = cat

    # seed sparse categories (extract a window each)
    for cat, times in SEED.items():
        d = outd / CATDIR[cat]; d.mkdir(exist_ok=True)
        base = len(list(d.glob("*.png")))
        for k, t in enumerate(times):
            _strip(t - 0.2, 2.2, d / f"{cat}_seed{base+k:03d}_{t:.1f}s.png")
            tcat[round(t, 1)] = cat

    # build inventory
    inv = []
    for a, b in segs:
        t0, t1 = a / fps, b / fps
        anim = float(diff[a:b + 1].max()) if b > a else 0.0
        cat = tcat.get(round(t0, 1), "motion")
        inv.append({"t": round(t0, 2), "dur": round(t1 - t0, 2), "cat": cat,
                    "anim": round(anim, 4),
                    "animated": bool(anim > diff.mean() + diff.std())})
    counts = {c: sum(1 for r in inv if r["cat"] == c) for c in CATDIR}
    # add seeded that may sit outside detected segments
    for cat, times in SEED.items():
        have = {round(r["t"], 1) for r in inv if r["cat"] == cat}
        for t in times:
            if round(t, 1) not in have:
                inv.append({"t": round(t, 2), "dur": 2.0, "cat": cat,
                            "anim": 0.0, "animated": True, "seeded": True})
                counts[cat] += 1
    ntr = len(list((outd / "transition_windows").glob("*.png")))
    out = {"fps": fps, "n_graphic_segments": len(segs),
           "graphic_seconds": round(sum(r["dur"] for r in inv), 1),
           "animated_segments": sum(1 for r in inv if r["animated"]),
           "category_counts": counts, "transition_strips": ntr,
           "note": "edge-presence catches busy graphics; sparse number/keyword/"
                   "document cards on near-black seeded from visual hunt sheet.",
           "segments": sorted(inv, key=lambda r: r["t"])}
    (outd / "motion_graphics_inventory.json").write_text(json.dumps(out, indent=1))
    print("inventory:", {k: counts[k] for k in counts}, "| segs", len(segs),
          "| transitions", ntr, "| animated", out["animated_segments"])


if __name__ == "__main__":
    main()
