#!/usr/bin/env python3
"""LAYER-2 (v2): graphic-presence segments + categorised dense windows.

Reads the cached LAYER-1 full_frame_scan.json (30 fps arrays) — no video
re-stream — and detects GRAPHIC-PRESENCE segments (sustained high edge), which
captures graphics that reveal-then-HOLD (the ones the motion-only pass missed).
Each segment is categorised (portrait via cv2 face / map / document / number /
text / motion) using 3-frame sampling and a dense filmstrip is saved.

Usage: python3 tools/mg_layer2.py <video> <slug>
"""
import json
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
try:
    from vidlore.ffmpeg_tool import ffmpeg_exe
    FF = ffmpeg_exe()
except Exception:                                              # noqa: BLE001
    FF = "ffmpeg"


def _frame_rgb(video, t, w=480, h=270):
    fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd)
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i",
                    video, "-frames:v", "1", "-vf", f"scale={w}:{h}", p], check=False)
    try:
        from PIL import Image
        a = np.asarray(Image.open(p).convert("RGB"))
    except Exception:                                          # noqa: BLE001
        a = None
    finally:
        try: os.unlink(p)
        except Exception: pass                                 # noqa: E722
    return a


def _filmstrip(video, t0, dur, out, cols=12):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{max(0,t0):.2f}",
                    "-i", video, "-t", f"{dur:.2f}", "-vf",
                    f"fps={cols/max(0.5,dur):.2f},scale=300:-1,tile={cols}x1",
                    "-frames:v", "1", str(out)], check=False)


def _cat(frames, face):
    """Vote a category across sampled frames."""
    votes = []
    for rgb in frames:
        if rgb is None:
            continue
        h, w, _ = rgb.shape
        r, g, b = (rgb[..., i].astype(float) for i in range(3))
        gray = rgb.mean(2)
        if face is not None:
            try:
                import cv2
                gg = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if len(face.detectMultiScale(gg, 1.2, 5, minSize=(54, 54))):
                    votes.append("portrait"); continue
            except Exception:                                  # noqa: BLE001
                pass
        mapness = float(((g > r + 4) & (g > b + 4) & (g > 55) & (g < 205)).mean())
        paper = float(((r > 115) & (g > 95) & (b > 60) & (r >= g - 5) & (g >= b - 5)).mean())
        hedge = np.abs(np.diff(gray, axis=1)).mean() / 255.0
        cy0, cy1, cx0, cx1 = int(h*.28), int(h*.72), int(w*.30), int(w*.70)
        ctr = gray[cy0:cy1, cx0:cx1]
        if mapness > 0.32:
            votes.append("map")
        elif paper > 0.16 and hedge > 0.045:
            votes.append("document")
        elif gray.mean() < 80 and ctr.mean() > gray.mean() + 35 and hedge > 0.05:
            # bright central cluster on dark -> number vs text by spread
            bright_cols = (gray > 150).mean(0)
            spread = float((bright_cols > 0.04).mean())
            votes.append("number" if spread < 0.42 else "text")
        elif gray.mean() < 95 and hedge > 0.06:
            votes.append("text")
        else:
            votes.append("motion")
    if not votes:
        return "motion"
    # priority: portrait/document/map beat text/number/motion on ties
    from collections import Counter
    c = Counter(votes)
    for pri in ("portrait", "document", "map", "number", "text", "motion"):
        if c.get(pri) and c[pri] == max(c.values()):
            return pri
    return c.most_common(1)[0][0]


def main():
    video, slug = sys.argv[1], sys.argv[2]
    outd = _REPO / "research" / "forensic_refs" / slug
    s = json.loads((outd / "full_frame_scan.json").read_text())
    fps = s["fps"]
    edge = np.array(s["edge"]); diff = np.array(s["diff"]); bright = np.array(s["bright"])
    thr = float(edge.mean() + 0.5 * edge.std())
    gp = edge > thr
    # segments: runs >= 0.4s, merge gaps < 0.4s
    segs = []
    run = None; gap = 0
    minlen = int(0.4 * fps); maxgap = int(0.4 * fps)
    for i, v in enumerate(gp):
        if v:
            if run is None:
                run = i
            gap = 0
        elif run is not None:
            gap += 1
            if gap > maxgap:
                if (i - gap) - run >= minlen:
                    segs.append((run, i - gap))
                run = None
    if run is not None:
        segs.append((run, len(gp) - 1))
    print(f"[l2] graphic-present segments: {len(segs)} (thr edge>{thr:.4f})", flush=True)

    cats = {"portrait": "portrait_animation_windows", "map": "map_animation_windows",
            "document": "document_animation_windows", "number": "number_animation_windows",
            "text": "text_animation_windows", "motion": "dense_motion_windows"}
    for d in cats.values():
        dd = outd / d
        dd.mkdir(exist_ok=True)
        for f in dd.glob("*.png"):       # clear the weak first-pass strips
            f.unlink()
    face = None
    try:
        import cv2
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:                                          # noqa: BLE001
        pass

    CAP = 70
    counts = {k: 0 for k in cats}
    inv = []
    for k, (a, b) in enumerate(segs):
        t0, t1 = a / fps, b / fps
        dur = t1 - t0
        # animation strength inside the segment (reveal / count / move)
        seg_diff = diff[a:b + 1]
        anim = float(seg_diff.max()) if len(seg_diff) else 0.0
        # categorise via 3 samples
        fr = [_frame_rgb(video, t0 + 0.30), _frame_rgb(video, (t0 + t1) / 2),
              _frame_rgb(video, max(t0, t1 - 0.25))]
        cat = _cat(fr, face)
        rec = {"t": round(t0, 2), "dur": round(dur, 2), "cat": cat,
               "anim": round(anim, 4), "animated": anim > diff.mean() + diff.std()}
        inv.append(rec)
        if counts[cat] < CAP:
            _filmstrip(video, t0 - 0.15, min(2.4, dur + 0.3),
                       outd / cats[cat] / f"{cat}_{counts[cat]:03d}_{t0:.1f}s.png")
            counts[cat] += 1
    # transitions already in transition_windows/ from layer-1 cuts
    n_tr = len(list((outd / "transition_windows").glob("*.png"))) if (outd / "transition_windows").exists() else 0
    inv_out = {
        "fps": fps, "n_graphic_segments": len(segs),
        "graphic_seconds": round(sum(r["dur"] for r in inv), 1),
        "animated_segments": sum(1 for r in inv if r["animated"]),
        "category_counts": {k: sum(1 for r in inv if r["cat"] == k) for k in cats},
        "transition_strips": n_tr,
        "segments": inv,
    }
    (outd / "motion_graphics_inventory.json").write_text(json.dumps(inv_out, indent=1), encoding="utf-8")
    print("[l2] categories: " + ", ".join(
        f"{k}={inv_out['category_counts'][k]}" for k in cats) +
        f" | animated={inv_out['animated_segments']}/{len(segs)}", flush=True)
    print(f"[l2] DONE -> {outd}", flush=True)


if __name__ == "__main__":
    main()
