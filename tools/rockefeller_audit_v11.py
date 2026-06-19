#!/usr/bin/env python3
"""V1.1 audit — real portrait + duration windows + 18-point check, → v1_1 package."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "output/john-d--rockefeller--the-man-who-built-standard-oi"
PKG = ROOT / "research/motion_graphics_qa/rockefeller_business_validation_v1_1"
LOG = Path("/tmp/rockefeller/render_v11.log")
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
WANT = ["gold_number_callout", "cinematic_portrait_hold", "headline_document_reveal",
        "portrait_name_over_map", "kinetic_keyword", "money_flow_empire"]


def sh(a): return subprocess.run([FF, *a], capture_output=True, text=True)


def dur_of(p):
    m = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", sh(["-i", str(p), "-f", "null", "-"]).stderr)
    if m:
        h, mm, s = m[-1]; return int(h)*3600+int(mm)*60+float(s)
    return 0.0


def black(p, d):
    return len(re.findall(r"black_start", sh(["-i", str(p), "-vf",
        f"blackdetect=d={d}:pix_th=0.10", "-an", "-f", "null", "-"]).stderr))


def audio(p):
    r = sh(["-i", str(p), "-af", "volumedetect", "-f", "null", "-"])
    r2 = sh(["-i", str(p), "-af", "ebur128=peak=true", "-f", "null", "-"])
    g = lambda rx, s: (re.findall(rx, s) or [None])[-1]
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)", r.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+)", r.stderr)
    return {"mean_db": float(mean.group(1)) if mean else None,
            "peak_db": float(peak.group(1)) if peak else None,
            "integrated_lufs": float(g(r"I:\s*(-?[\d.]+)\s*LUFS", r2.stderr) or 0) or None,
            "lra_lu": float(g(r"LRA:\s*(-?[\d.]+)\s*LU", r2.stderr) or 0) or None}


def grab(p, t, o): sh(["-ss", f"{t:.3f}", "-i", str(p), "-frames:v", "1", "-q:v", "2", "-y", str(o)])
def clip(p, t, ln, o):
    sh(["-ss", f"{max(0,t):.3f}", "-i", str(p), "-t", f"{ln:.2f}", "-c", "copy", "-y", str(o)])
    if not o.exists() or o.stat().st_size < 2000:
        sh(["-ss", f"{max(0,t):.3f}", "-i", str(p), "-t", f"{ln:.2f}", "-c:v", "libx264",
            "-crf", "20", "-c:a", "aac", "-y", str(o)])


def scene_starts():
    works = sorted(RUN.glob("work_*"))
    if not works:
        return {}
    wavs = {}
    for w in works:
        for p in w.rglob("*.wav"):
            mt = re.search(r"(?:vo|narr|scene)[_-]?(\d+)", p.name)
            if mt:
                wavs.setdefault(int(mt.group(1)), p)
    s, t = {}, 0.0
    for i in sorted(wavs):
        s[i] = t; t += dur_of(wavs[i]) + 0.35
    return s


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    (PKG/"first_mid_last_frames").mkdir(exist_ok=True)
    (PKG/"proof_clips").mkdir(exist_ok=True)
    vid = RUN/f"{RUN.name}.mp4"
    out = {"exists": vid.exists(), "duration_s": round(dur_of(vid), 2) if vid.exists() else 0}
    if not vid.exists():
        print("NO VIDEO"); return
    bf = {"d0.30": black(vid, 0.30), "d0.15": black(vid, 0.15), "d0.10": black(vid, 0.10)}
    am = audio(vid)

    man = json.loads((RUN/"motion_graphics_manifest.json").read_text())
    entries = man.get("scenes", [])
    starts = scene_starts()
    # portrait provenance from the realperson cache
    prov = {}
    for pj in (RUN/"cache").glob("*.provenance.json"):
        try:
            d = json.loads(pj.read_text())
            if "rockefeller" in (d.get("person", "").lower()):
                prov = d
        except Exception: pass
    # copy the real portrait into the package
    real_port = None
    for pj in (RUN/"cache").glob("*.jpg"):
        # heuristic: realperson cache jpgs are portraits
        pass
    if prov.get("cached_path") and Path(prov["cached_path"]).exists():
        import shutil
        real_port = PKG/"real_rockefeller_portrait.jpg"
        shutil.copy(prov["cached_path"], real_port)

    fired = []
    for e in entries:
        if not e.get("primitive") or e.get("skipped") or not e.get("ok", True):
            continue
        si, pid, cpath = e.get("scene_index"), e["primitive"], e.get("path") or ""
        rec = {"primitive": pid, "scene_index": si, "clip_dur": e.get("duration"),
               "cache": "hit" if e.get("cache_hit") else "miss",
               "fallback": bool(e.get("fallback")), "reason": e.get("reason", "")}
        if cpath and Path(cpath).exists():
            cd = dur_of(cpath); rec["clip_real_dur"] = round(cd, 2)
            for tag, tt in (("first", .3), ("mid", cd/2), ("last", max(.3, cd-.5))):
                grab(cpath, tt, PKG/"first_mid_last_frames"/f"{si:02d}_{pid}_{tag}.png")
            st = starts.get(si)
            if st is not None:
                clip(vid, st, min(8, cd+2), PKG/"proof_clips"/f"sc{si:02d}_{pid}.mp4")
                grab(vid, st+min(2, cd/2), PKG/"first_mid_last_frames"/f"{si:02d}_{pid}_INCONTEXT.png")
        fired.append(rec)

    sh(["-i", str(vid), "-vf", "fps=1/8,scale=320:-1,tile=5x3", "-frames:v", "1", "-y", str(PKG/"contact_sheet.png")])
    logtxt = LOG.read_text(errors="ignore") if LOG.exists() else ""
    def _last(sub):
        return next((l.strip() for l in reversed(logtxt.splitlines()) if sub in l), "")
    fired_pids = sorted({f["primitive"] for f in fired})
    metrics = {
        "version": "v1.1", "duration_s": out["duration_s"], "scenes_total": 16,
        "graphics_total": len(fired), "density_pct": round(100*len(fired)/16, 1),
        "primitives_fired": fired_pids, "primitives_missing": [p for p in WANT if p not in fired_pids],
        "fired_detail": fired, "black_frames": bf, "audio": am,
        "portrait_provenance": prov, "real_portrait_used": bool(prov.get("source")),
        "log_mg": _last("motion graphics:"), "log_blackrepair": _last("black-frame repair: clean"),
        "log_realperson": _last("[real-person] John"),
    }
    (PKG/"render_metrics.json").write_text(json.dumps(metrics, indent=2))
    (PKG/"audio_metrics.json").write_text(json.dumps(am, indent=2))
    (PKG/"black_frame_metrics.json").write_text(json.dumps(bf, indent=2))
    (PKG/"portrait_provenance.json").write_text(json.dumps(prov, indent=2))
    import shutil
    for s, d in ((vid, "final_sample.mp4"), (RUN/"script.txt", "sample_script.txt"),
                 (RUN/"motion_graphics_manifest.json", "motion_graphics_manifest.json")):
        if Path(s).exists(): shutil.copy(Path(s), PKG/d)
    print(json.dumps({k: metrics[k] for k in ("duration_s", "density_pct", "primitives_fired",
          "primitives_missing", "black_frames", "audio", "real_portrait_used")}, indent=2))
    print("portrait:", prov.get("source"), "| nm:", prov.get("name_match"), "| val:", prov.get("validator"))
    print("windows (clip_dur per fired):", [(f["scene_index"], f["primitive"], f.get("clip_dur")) for f in fired])


if __name__ == "__main__":
    main()
