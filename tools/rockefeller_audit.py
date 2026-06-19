#!/usr/bin/env python3
"""Deep frame-level audit of the Rockefeller business render.

Per primitive: standalone first/mid/last frames + an in-context proof clip from
the FINAL mp4 at the scene's timestamp. Plus black-frame, audio (mean/peak +
LUFS), density, spacing, contact sheet, and a render_metrics.json roll-up.

Runs entirely on the bundled imageio-ffmpeg binary (no ffprobe).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "output/john-d--rockefeller--the-man-who-built-standard-oi"
PKG = ROOT / "research/motion_graphics_qa/rockefeller_business_validation"
LOG = Path("/tmp/rockefeller/render.log")

import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
WANT = ["gold_number_callout", "cinematic_portrait_hold", "headline_document_reveal",
        "portrait_name_over_map", "kinetic_keyword", "money_flow_empire"]


def sh(args):
    return subprocess.run([FF, *args], capture_output=True, text=True)


def dur_of(path):
    r = sh(["-i", str(path), "-f", "null", "-"])
    m = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", r.stderr)
    if m:
        h, mm, s = m[-1]
        return int(h) * 3600 + int(mm) * 60 + float(s)
    return 0.0


def black_count(path, d):
    r = sh(["-i", str(path), "-vf", f"blackdetect=d={d}:pix_th=0.10",
            "-an", "-f", "null", "-"])
    return len(re.findall(r"black_start", r.stderr))


def audio_metrics(path):
    r = sh(["-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)", r.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+)", r.stderr)
    # integrated LUFS via ebur128
    r2 = sh(["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    lufs = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r2.stderr)
    lra = re.findall(r"LRA:\s*(-?[\d.]+)\s*LU", r2.stderr)
    return {
        "mean_db": float(mean.group(1)) if mean else None,
        "peak_db": float(peak.group(1)) if peak else None,
        "integrated_lufs": float(lufs[-1]) if lufs else None,
        "lra_lu": float(lra[-1]) if lra else None,
    }


def grab(path, t, out):
    sh(["-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-q:v", "2",
        "-y", str(out)])
    return out.exists()


def clip(path, t, length, out):
    sh(["-ss", f"{max(0,t):.3f}", "-i", str(path), "-t", f"{length:.2f}",
        "-c", "copy", "-y", str(out)])
    if not out.exists() or out.stat().st_size < 2000:
        sh(["-ss", f"{max(0,t):.3f}", "-i", str(path), "-t", f"{length:.2f}",
            "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-y", str(out)])
    return out.exists()


def find_manifest():
    for p in RUN.rglob("motion_graphics_manifest.json"):
        return p
    return None


def scene_starts():
    """Approx per-scene start time in the final cut from the per-scene VO wavs
    in the work dir (+ a small constant pad per scene)."""
    works = sorted(RUN.glob("work_*"))
    if not works:
        return {}
    wavs = {}
    for w in works:
        for p in w.rglob("*.wav"):
            m = re.search(r"(?:vo|narr|scene)[_-]?(\d+)", p.name)
            if m:
                wavs.setdefault(int(m.group(1)), p)
    if not wavs:
        return {}
    starts, t = {}, 0.0
    for i in sorted(wavs):
        starts[i] = t
        t += dur_of(wavs[i]) + 0.35      # ~pad between scenes
    return starts


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG / "first_mid_last_frames").mkdir(exist_ok=True)
    (PKG / "proof_clips").mkdir(exist_ok=True)
    vid = RUN / f"{RUN.name}.mp4"
    out = {"video": str(vid), "exists": vid.exists()}
    if not vid.exists():
        print("NO VIDEO", vid); print(json.dumps(out)); return
    out["duration_s"] = round(dur_of(vid), 2)

    # black + audio
    bf = {"d0.30": black_count(vid, 0.30), "d0.15": black_count(vid, 0.15)}
    am = audio_metrics(vid)
    (PKG / "black_frame_metrics.json").write_text(json.dumps(bf, indent=2))
    (PKG / "audio_metrics.json").write_text(json.dumps(am, indent=2))

    # manifest → fired primitives
    man_p = find_manifest()
    manifest = json.loads(man_p.read_text()) if man_p else {}
    # manifest = {"summary": {...}, "scenes": [ per-scene entries ]}
    entries = (manifest.get("scenes") or manifest.get("entries") or []) \
        if isinstance(manifest, dict) else manifest
    if man_p:
        (PKG / "motion_graphics_manifest.json").write_text(json.dumps(manifest, indent=2))
    starts = scene_starts()

    fired = []
    for e in (entries or []):
        if not isinstance(e, dict):
            continue
        pid = e.get("primitive")
        if not pid or e.get("skipped") or not e.get("ok", True):
            continue
        si = e.get("scene_index")
        cpath = e.get("path") or ""
        rec = {"primitive": pid, "scene_index": si,
               "cache": "hit" if e.get("cache_hit") else "miss",
               "fallback": bool(e.get("fallback")),
               "render_s": e.get("render_s") or e.get("seconds"),
               "reason": e.get("reason", ""),
               "inputs": sorted((e.get("inputs") or {}).keys()),
               "clip": cpath}
        # standalone first/mid/last from the MG clip
        if cpath and Path(cpath).exists():
            cd = dur_of(cpath)
            fr = PKG / "first_mid_last_frames"
            for tag, tt in (("first", 0.25), ("mid", cd / 2), ("last", max(0.3, cd - 0.4))):
                grab(cpath, tt, fr / f"{si:02d}_{pid}_{tag}.png")
            # in-context proof clip from the FINAL mp4 at scene start
            st = starts.get(si)
            if st is not None:
                clip(vid, st, min(6.0, cd), PKG / "proof_clips" / f"sc{si:02d}_{pid}.mp4")
                grab(vid, st + min(2.5, cd / 2), PKG / "first_mid_last_frames" / f"{si:02d}_{pid}_INCONTEXT.png")
        fired.append(rec)

    # contact sheet of the final video (5x3 grid)
    cs = PKG / "contact_sheet.png"
    sh(["-i", str(vid), "-vf", "fps=1/8,scale=320:-1,tile=5x3", "-frames:v", "1",
        "-y", str(cs)])

    # parse log
    logtxt = LOG.read_text(errors="ignore") if LOG.exists() else ""
    mg_line = ""
    for ln in logtxt.splitlines():
        if "motion graphics:" in ln and "scene(s)" in ln:
            mg_line = ln.strip()
    ai_line = next((l.strip() for l in logtxt.splitlines() if "ai usage:" in l), "")
    bf_line = next((l.strip() for l in logtxt.splitlines() if "black-frame repair: clean" in l), "")

    fired_pids = sorted({f["primitive"] for f in fired})
    metrics = {
        "title": "John D. Rockefeller: The Man Who Built Standard Oil",
        "duration_s": out["duration_s"],
        "scenes_total": 16,
        "graphics_total": len(fired),
        "density_pct": round(100 * len(fired) / 16, 1),
        "primitives_fired": fired_pids,
        "primitives_missing": [p for p in WANT if p not in fired_pids],
        "fired_detail": fired,
        "black_frames": bf,
        "audio": am,
        "log_mg": mg_line, "log_ai": ai_line, "log_blackrepair": bf_line,
    }
    (PKG / "render_metrics.json").write_text(json.dumps(metrics, indent=2))

    # copy reference artifacts
    import shutil
    for src, dst in ((vid, "final_sample.mp4"),
                     (RUN / "script.txt", "sample_script.txt")):
        if Path(src).exists():
            shutil.copy(Path(src), PKG / dst)
    for extra in ("director_decisions.json", "motion_graphics_decisions.json"):
        sp = next(RUN.rglob(extra), None)
        if sp:
            shutil.copy(sp, PKG / "director_decisions.json")

    print("=== RENDER METRICS ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
