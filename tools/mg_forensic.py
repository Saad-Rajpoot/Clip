#!/usr/bin/env python3
"""Two-layer motion-graphics forensic (LAYER 1 every-frame + LAYER 2 dense windows).

LAYER 1 — FULL FRAME STREAM @ source 30 fps via a single ffmpeg gray pipe
(240x135), no 89k PNGs. Per-frame: brightness, edge density, frame-diff (motion),
text-band edge. Detect: cuts, animation windows, graphic-present, text-change,
black frames — at full 30 fps precision (NOT 3 fps).

LAYER 2 — for every detected event, extract a dense filmstrip (12 frames) at
higher res and CATEGORISE it (heuristic, to be visually verified):
  face -> portrait | map-colour -> map | paper+text -> document |
  small bright central change -> number | else -> motion/text/transition.

Usage:  python3 tools/mg_forensic.py <video_with_audio> <slug>
Local only (ffmpeg + numpy + optional cv2 for faces). No paid API.
"""
import csv
import json
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

SW, SH = 240, 135            # layer-1 proxy resolution


def _probe(video):
    r = subprocess.run([FF, "-hide_banner", "-i", video], capture_output=True, text=True)
    import re
    t = r.stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", t)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    mv = re.search(r"Video:.* (\d{2,5})x(\d{2,5})", t)
    w, h = (int(mv.group(1)), int(mv.group(2))) if mv else (1920, 1080)
    mf = re.search(r"([\d.]+) fps", t)
    fps = float(mf.group(1)) if mf else 30.0
    return {"duration_s": round(dur, 2), "width": w, "height": h, "fps": fps}


def layer1(video, fps):
    """Stream EVERY frame as 240x135 gray via ffmpeg; compute per-frame metrics."""
    cmd = [FF, "-hide_banner", "-loglevel", "error", "-i", video,
           "-vf", f"fps={fps},scale={SW}:{SH}", "-f", "rawvideo",
           "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=SW * SH * 64)
    fsz = SW * SH
    bright, edge, diff, tband = [], [], [], []
    prev = None
    while True:
        buf = proc.stdout.read(fsz)
        if len(buf) < fsz:
            break
        g = np.frombuffer(buf, np.uint8).reshape(SH, SW).astype(np.float32)
        bright.append(g.mean() / 255.0)
        gx = np.abs(np.diff(g, axis=1)).mean()
        gy = np.abs(np.diff(g, axis=0)).mean()
        edge.append((gx + gy) / 2.0 / 255.0)
        diff.append(0.0 if prev is None else float(np.abs(g - prev).mean() / 255.0))
        tb = g[SH * 2 // 3:, :]
        tbx = np.abs(np.diff(tb, axis=1)).mean()
        tband.append(tbx / 255.0)
        prev = g
    proc.stdout.close()
    proc.wait()
    return (np.array(bright), np.array(edge), np.array(diff), np.array(tband))


def detect_events(bright, edge, diff, tband, fps):
    n = len(diff)
    cut_thr = float(diff.mean() + 4.0 * diff.std())
    e_hi = float(edge.mean() + edge.std())
    m_lo = float(diff.mean() + 0.8 * diff.std())
    # cuts = local-max diff spikes above cut threshold
    cuts = []
    for i in range(1, n - 1):
        if diff[i] > cut_thr and diff[i] >= diff[i - 1] and diff[i] >= diff[i + 1]:
            if not cuts or i - cuts[-1] > int(0.2 * fps):
                cuts.append(i)
    # graphic-present mask + text-change mask
    gpres = edge > e_hi
    # animation windows = runs of sustained moderate motion w/ graphic present,
    # that are NOT cuts, lasting >= 0.30 s
    anim = []
    run = None
    for i in range(n):
        active = (diff[i] > m_lo) and (diff[i] <= cut_thr) and gpres[i]
        if active:
            run = run if run is not None else i
        elif run is not None:
            if (i - run) >= int(0.30 * fps):
                anim.append((run, i))
            run = None
    black = [i for i in range(n) if bright[i] < 0.04]
    return {"cut_thr": cut_thr, "e_hi": e_hi, "m_lo": m_lo,
            "cuts": cuts, "anim": anim, "black_n": len(black)}


def _filmstrip(video, t0, dur, outpng, fps_w=12):
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i",
                    video, "-t", f"{dur:.2f}", "-vf",
                    f"fps={fps_w},scale=300:-1,tile={fps_w}x1", "-frames:v", "1",
                    str(outpng)], check=False)


def _midframe_bgr(video, t):
    import tempfile
    import os
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i",
                    video, "-frames:v", "1", "-vf", "scale=480:270", p], check=False)
    try:
        from PIL import Image
        arr = np.asarray(Image.open(p).convert("RGB"))
    except Exception:                                          # noqa: BLE001
        arr = None
    finally:
        try:
            os.unlink(p)
        except Exception:                                      # noqa: BLE001
            pass
    return arr


def _categorise(rgb, face_cascade):
    """Heuristic category from the mid frame (verified visually later)."""
    if rgb is None:
        return "motion"
    h, w, _ = rgb.shape
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    # face -> portrait
    if face_cascade is not None:
        try:
            import cv2
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces):
                return "portrait"
        except Exception:                                      # noqa: BLE001
            pass
    # map -> greenish/tan large mid-tone region
    mapness = float(((g > r) & (g > b) & (g > 60) & (g < 200)).mean())
    if mapness > 0.30:
        return "map"
    # document -> warm paper tone + high horizontal edge (text lines)
    paper = float(((r > 120) & (g > 100) & (b > 70) & (r >= g) & (g >= b)).mean())
    edge = np.abs(np.diff(g, axis=1)).mean() / 255.0
    if paper > 0.18 and edge > 0.05:
        return "document"
    # number -> small bright cluster near centre on dark field
    cy0, cy1, cx0, cx1 = int(h*0.30), int(h*0.70), int(w*0.30), int(w*0.70)
    centre = g[cy0:cy1, cx0:cx1]
    if g.mean() < 70 and centre.mean() > g.mean() + 40 and float((centre > 160).mean()) < 0.25:
        return "number"
    # text -> bright text-like edges on dark, wider spread
    if g.mean() < 90 and edge > 0.06:
        return "text"
    return "motion"


def main():
    video, slug = sys.argv[1], sys.argv[2]
    outd = _REPO / "research" / "forensic_refs" / slug
    outd.mkdir(parents=True, exist_ok=True)
    meta = _probe(video)
    fps = round(meta["fps"]) or 30
    print(f"[mg] {slug} {meta}", flush=True)

    print("[mg] LAYER 1: every-frame stream @ %d fps ..." % fps, flush=True)
    bright, edge, diff, tband = layer1(video, fps)
    n = len(diff)
    print(f"[mg] layer1: {n} frames ({n/fps:.1f}s)", flush=True)
    ev = detect_events(bright, edge, diff, tband, fps)
    print(f"[mg] events: cuts={len(ev['cuts'])} anim_windows={len(ev['anim'])} "
          f"black={ev['black_n']}", flush=True)

    # full_frame_scan.json — compact arrays @ full fps
    (outd / "full_frame_scan.json").write_text(json.dumps({
        "fps": fps, "n_frames": n, "proxy": f"{SW}x{SH}",
        "thresholds": {k: round(ev[k], 5) for k in ("cut_thr", "e_hi", "m_lo")},
        "summary": {"mean_bright": round(float(bright.mean()), 4),
                    "mean_edge": round(float(edge.mean()), 4),
                    "mean_diff": round(float(diff.mean()), 4),
                    "n_cuts": len(ev["cuts"]), "n_anim_windows": len(ev["anim"]),
                    "n_black": ev["black_n"]},
        "bright": [round(float(x), 4) for x in bright],
        "edge": [round(float(x), 4) for x in edge],
        "diff": [round(float(x), 4) for x in diff],
    }), encoding="utf-8")

    # LAYER 2 — categorised dense windows
    cats = {"portrait": "portrait_animation_windows", "map": "map_animation_windows",
            "document": "document_animation_windows", "number": "number_animation_windows",
            "text": "text_animation_windows", "motion": "dense_motion_windows",
            "transition": "transition_windows"}
    for d in cats.values():
        (outd / d).mkdir(exist_ok=True)
    face_cascade = None
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:                                          # noqa: BLE001
        print("[mg] cv2 face cascade unavailable; portraits via colour only", flush=True)

    # transitions from cuts (the boundary windows); cap per category to keep disk sane
    CAP = 80
    inv = {k: [] for k in cats}
    # transitions
    for k, ci in enumerate(ev["cuts"][:CAP * 3:max(1, len(ev["cuts"]) // (CAP) or 1)][:CAP]):
        t = ci / fps
        _filmstrip(video, max(0, t - 0.5), 1.2, outd / cats["transition"] /
                   f"trans_{k:03d}_{t:.1f}s.png")
        inv["transition"].append(round(t, 2))
    # animation windows -> categorise + filmstrip
    counts = {k: 0 for k in cats}
    anim = ev["anim"]
    # spread sampling if too many
    step = max(1, len(anim) // (CAP * 4) or 1)
    for k, (a, b) in enumerate(anim[::step]):
        t0, t1 = a / fps, b / fps
        dur = max(0.5, min(2.2, t1 - t0 + 0.4))
        mid = (t0 + t1) / 2
        rgb = _midframe_bgr(video, mid)
        cat = _categorise(rgb, face_cascade)
        if counts[cat] >= CAP:
            continue
        idx = counts[cat]
        _filmstrip(video, max(0, t0 - 0.2), dur, outd / cats[cat] /
                   f"{cat}_{idx:03d}_{t0:.1f}s.png")
        inv[cat].append({"t": round(t0, 2), "dur": round(t1 - t0, 2)})
        counts[cat] += 1

    (outd / "motion_graphics_inventory.json").write_text(json.dumps({
        "fps": fps, "total_cuts": len(ev["cuts"]),
        "total_anim_windows": len(ev["anim"]),
        "category_counts": {k: len(v) for k, v in inv.items()},
        "windows": inv,
    }, indent=1), encoding="utf-8")

    # full_contact_sheet.png alias (keep the 3 fps one too)
    src = outd / "contact_sheet_full.png"
    if src.exists():
        import shutil
        shutil.copyfile(src, outd / "full_contact_sheet.png")

    print("[mg] category counts: " +
          ", ".join(f"{k}={len(v)}" for k, v in inv.items()), flush=True)
    print(f"[mg] DONE -> {outd}", flush=True)


if __name__ == "__main__":
    main()
