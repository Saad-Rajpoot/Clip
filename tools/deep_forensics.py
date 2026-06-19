"""Deep frame-level forensic pass on a LOCAL reference documentary.

Scans the ENTIRE video (not a few keyframes) and writes a structured
database under research/forensic_refs/<slug>/ for editorial-primitive
extraction.  Local tools only (ffmpeg + OpenCV + numpy + PIL +
PySceneDetect) — NO paid APIs.

Strategy for speed on long (30-60 min) videos: ffmpeg extracts a
low-res frame stream ONCE (single decode pass), then numpy/cv2 analyse
the small PNGs.  Dense windows around detected events are pulled with
fast input-seek ffmpeg.

Usage:  python tools/deep_forensics.py "<video path>" <slug> [--fps 2]

Outputs (per the reference-database spec):
  metadata.json full_frame_scan.json shot_list.json shot_timing.csv
  contact_sheet_full.png dense_motion_windows/ transition_windows/
  graphics_windows/  audio_analysis.json silence_pockets.json
  sfx_events.json music_curve.json text_overlay_inventory.json
  motion_graphics_inventory.json
(the interpretive files — extracted_editing_primitives.json /
 useful_patterns.md / anti_patterns.md / implementation_backlog.json —
 are written by the analysis agent that reads this database.)
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = "/Users/hussnain/Desktop/vidrush-clone"
FF = str(next(Path(_REPO).glob(
    ".venv/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))


# ───────────────────────── helpers ─────────────────────────
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _probe(video):
    r = _run([FF, "-hide_banner", "-i", video])
    txt = r.stderr
    dur = 0.0
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    w = h = fps = 0
    mv = re.search(r"Video:.* (\d{2,5})x(\d{2,5})", txt)
    if mv:
        w, h = int(mv.group(1)), int(mv.group(2))
    mf = re.search(r"([\d.]+) fps", txt)
    if mf:
        fps = float(mf.group(1))
    mb = re.search(r"bitrate: (\d+) kb/s", txt)
    return {"duration_s": round(dur, 2), "width": w, "height": h,
            "fps": fps, "bitrate_kbps": int(mb.group(1)) if mb else 0}


def _frame_stats(arr):
    """arr = HxWx3 uint8 RGB. Returns brightness, saturation, edge_density."""
    f = arr.astype(np.float32) / 255.0
    bright = float(f.mean())
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = float(np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0).mean())
    g = f.mean(axis=2)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    edge = float((gx + gy) / 2.0)
    return round(bright, 4), round(sat, 4), round(edge, 4)


def _palette(arr, k=4):
    small = arr[::8, ::8].reshape(-1, 3)
    q = (small // 48 * 48).astype(int)
    vals, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [[int(c) for c in vals[i]] for i in order]


# ───────────────────────── stages ─────────────────────────
def shots(video, outd, dur):
    """PySceneDetect content shots across the WHOLE video."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        v = open_video(video)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=8))
        sm.detect_scenes(v, show_progress=False)
        sl = sm.get_scene_list()
        shots_ = [{"i": i, "start": round(s.get_seconds(), 2),
                   "end": round(e.get_seconds(), 2),
                   "len": round(e.get_seconds() - s.get_seconds(), 2)}
                  for i, (s, e) in enumerate(sl)]
    except Exception as e:                                          # noqa: BLE001
        print(f"[deep] scenedetect failed ({e}); empty shot list", flush=True)
        shots_ = []
    (outd / "shot_list.json").write_text(json.dumps(
        {"n_shots": len(shots_), "shots": shots_}, indent=1))
    with open(outd / "shot_timing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shot", "start_s", "end_s", "len_s"])
        for s in shots_:
            w.writerow([s["i"], s["start"], s["end"], s["len"]])
    lens = [s["len"] for s in shots_ if s["len"] > 0]
    print(f"[deep] shots: {len(shots_)} | "
          f"median {np.median(lens):.2f}s | <1s {sum(l<1 for l in lens)} "
          f"| >6s {sum(l>=6 for l in lens)}" if lens else "[deep] shots: 0",
          flush=True)
    return shots_


def full_scan(video, outd, dur, fps):
    """Single-decode frame stream → per-frame stats + motion + event flags."""
    td = tempfile.mkdtemp()
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", video,
                    "-vf", f"fps={fps},scale=320:-1", f"{td}/f%06d.png"],
                   check=False)
    files = sorted(Path(td).glob("*.png"))
    frames = []
    prev = None
    for idx, fp in enumerate(files):
        arr = np.asarray(Image.open(fp).convert("RGB"))
        b, s, e = _frame_stats(arr)
        g = arr.mean(axis=2)
        motion = 0.0 if prev is None else float(
            np.abs(g - prev).mean() / 255.0)
        prev = g
        frames.append({"t": round(idx / fps, 2), "bright": b, "sat": s,
                       "edge": e, "motion": round(motion, 4),
                       "pal": _palette(arr)})
    # event detection (relative to running stats)
    edges = np.array([f["edge"] for f in frames]) if frames else np.array([0.])
    mots = np.array([f["motion"] for f in frames]) if frames else np.array([0.])
    brts = np.array([f["bright"] for f in frames]) if frames else np.array([0.])
    e_hi = edges.mean() + edges.std()
    m_hi = mots.mean() + 1.5 * mots.std()
    for i, f in enumerate(frames):
        f["graphic"] = bool(f["edge"] > e_hi)            # text/overlay present
        f["motion_spike"] = bool(f["motion"] > m_hi)     # cut / fast move
        f["black"] = bool(f["bright"] < 0.04)
    (outd / "full_frame_scan.json").write_text(json.dumps(
        {"fps": fps, "n_frames": len(frames),
         "summary": {"mean_bright": round(float(brts.mean()), 4),
                     "mean_sat": round(float(np.mean([f["sat"] for f in frames] or [0])), 4),
                     "mean_edge": round(float(edges.mean()), 4),
                     "edge_hi_thresh": round(float(e_hi), 4),
                     "motion_hi_thresh": round(float(m_hi), 4)},
         "frames": frames}, indent=0))
    print(f"[deep] full scan: {len(frames)} frames @ {fps}fps "
          f"| graphic-present {sum(f['graphic'] for f in frames)} "
          f"| motion-spikes {sum(f['motion_spike'] for f in frames)} "
          f"| black {sum(f['black'] for f in frames)}", flush=True)
    # contact sheet (whole video)
    n = len(files)
    if n:
        step = max(1, n // 120)
        sel = files[::step][:120]
        thumbs = [np.asarray(Image.open(p).convert("RGB").resize((192, 108)))
                  for p in sel]
        cols, rows = 12, (len(thumbs) + 11) // 12
        sheet = np.zeros((rows * 108, cols * 192, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            sheet[r*108:r*108+108, c*192:c*192+192] = t
        Image.fromarray(sheet).save(outd / "contact_sheet_full.png")
    return frames


def _windows(video, outd, sub, events, dur, label):
    d = outd / sub
    d.mkdir(exist_ok=True)
    for k, t in enumerate(events[:40]):
        ss = max(0, t - 0.6)
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", f"{ss:.2f}",
                        "-i", video, "-t", "1.4", "-vf",
                        "fps=8,scale=240:-1,tile=7x1", "-frames:v", "1",
                        str(d / f"{label}_{k:03d}_{t:.1f}s.png")], check=False)
    print(f"[deep] {sub}: {min(len(events),40)} strips", flush=True)


def audio(video, outd, dur):
    # RMS / loudness curve via astats per 0.5s window (ametadata print)
    r = _run([FF, "-i", video, "-af",
              "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
              "-f", "null", "-"])
    rms = []
    for m in re.finditer(r"RMS_level=(-?[\d.]+|-inf)", r.stderr):
        v = m.group(1)
        rms.append(-90.0 if v == "-inf" else float(v))
    # silence pockets
    rs = _run([FF, "-i", video, "-af", "silencedetect=n=-32dB:d=0.5",
               "-f", "null", "-"])
    sil = []
    for m in re.finditer(r"silence_start: ([\d.]+)", rs.stderr):
        sil.append({"start": float(m.group(1))})
    ends = re.findall(r"silence_end: ([\d.]+)", rs.stderr)
    for i, e in enumerate(ends):
        if i < len(sil):
            sil[i]["end"] = float(e)
            sil[i]["dur"] = round(float(e) - sil[i]["start"], 2)
    arr = np.array(rms) if rms else np.array([-60.0])
    rate = (len(arr) / dur) if dur else 1.0          # true RMS sample rate (Hz)
    floor = np.percentile(arr, 15)
    music_pct = round(float((arr > floor + 6).mean()) * 100, 1)
    jumps = [i for i in range(1, len(arr)) if arr[i] - arr[i-1] > 8]
    sfx_t = [round(i / rate, 2) for i in jumps]      # FIX: index → seconds
    # downsample the loudness curve to ~2 Hz → compact, usable music curve
    bin_ = max(1, int(round(rate / 2.0)))
    curve = [round(float(arr[i:i+bin_].mean()), 1)
             for i in range(0, len(arr), bin_)]
    (outd / "audio_analysis.json").write_text(json.dumps(
        {"n_rms_samples": len(arr), "rms_rate_hz": round(rate, 2),
         "mean_db": round(float(arr.mean()), 2),
         "floor_db": round(float(floor), 2), "music_presence_pct": music_pct,
         "n_silence_pockets": len(sil), "n_sfx_events": len(jumps)}, indent=1))
    (outd / "music_curve.json").write_text(json.dumps(
        {"hz": 2.0, "rms_db": curve}, indent=0))
    (outd / "silence_pockets.json").write_text(json.dumps(sil, indent=1))
    (outd / "sfx_events.json").write_text(json.dumps(
        {"n": len(jumps), "times_s": sfx_t[:300]}, indent=1))
    print(f"[deep] audio: music≈{music_pct}% | silence pockets {len(sil)} "
          f"| sfx events {len(jumps)} @ {rate:.1f}Hz", flush=True)


def inventories(frames, outd, fps):
    """Text-overlay + motion-graphics segments from the edge/motion series."""
    def segments(flagkey, min_len=3):
        segs, run = [], None
        for i, f in enumerate(frames):
            if f.get(flagkey):
                run = run or i
            elif run is not None:
                if i - run >= min_len:
                    segs.append({"start": round(run/fps, 2),
                                 "end": round(i/fps, 2),
                                 "dur": round((i-run)/fps, 2)})
                run = None
        return segs
    text_segs = segments("graphic")
    mg = [{"start": round(i/fps, 2)} for i, f in enumerate(frames)
          if f.get("graphic") and f.get("motion_spike")]
    (outd / "text_overlay_inventory.json").write_text(json.dumps(
        {"n_segments": len(text_segs), "total_s": round(sum(s["dur"] for s in text_segs), 1),
         "segments": text_segs}, indent=1))
    (outd / "motion_graphics_inventory.json").write_text(json.dumps(
        {"n_animating": len(mg), "events": mg[:300]}, indent=1))
    print(f"[deep] inventories: {len(text_segs)} text/overlay segments "
          f"| {len(mg)} animating-graphic frames", flush=True)
    return text_segs, mg


def main():
    video = sys.argv[1]
    slug = sys.argv[2]
    fps = 2.0
    if "--fps" in sys.argv:
        fps = float(sys.argv[sys.argv.index("--fps") + 1])
    outd = Path(_REPO) / "research" / "forensic_refs" / slug
    outd.mkdir(parents=True, exist_ok=True)
    print(f"[deep] === {slug} === {video}", flush=True)
    meta = _probe(video)
    (outd / "metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"[deep] meta: {meta}", flush=True)
    dur = meta["duration_s"]
    sl = shots(video, outd, dur)
    frames = full_scan(video, outd, dur, fps)
    audio(video, outd, dur)
    inventories(frames, outd, fps)
    # dense windows around events
    cuts = [s["start"] for s in sl][1:]
    motion_evt = [f["t"] for f in frames if f.get("motion_spike")]
    gfx_evt = [f["t"] for f in frames if f.get("graphic")]
    # thin gfx events so windows are spread, not clustered
    gfx_evt = gfx_evt[::max(1, len(gfx_evt)//40)] if gfx_evt else []
    _windows(video, outd, "transition_windows", cuts, dur, "cut")
    _windows(video, outd, "dense_motion_windows", motion_evt, dur, "mot")
    _windows(video, outd, "graphics_windows", gfx_evt, dur, "gfx")
    print(f"[deep] DONE -> {outd}", flush=True)


if __name__ == "__main__":
    main()
