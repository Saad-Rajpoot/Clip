#!/usr/bin/env python3
"""Per-sample audit for the multi-niche validation. Reads a finished render's
run_dir + manifests, runs blackdetect / EBU R128 (LUFS + true peak), checks temp
cleanup, parses the render log for cost / cache / footage stats, extracts every
MG card frame + a few footage frames, and writes audit.json. Visual judgements
(portrait likeness, footage quality, premium feel) are made by viewing frames.

  python3 tools/multiniche_audit.py <id>      # audit one sample
  python3 tools/multiniche_audit.py all       # audit all 5 + cross-niche table
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTROOT = ROOT / "research/motion_graphics_qa/multiniche"
TITLES = {
    "spy": "our-man-in-damascus", "crime": "the-untouchable",
    "business": "the-richest-man", "history": "the-road-to-moscow",
    "geopolitics": "thirteen-days",
}


def ffexe():
    try:
        from vidlore.ffmpeg_tool import ffmpeg_exe
        return ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def find_run_dir(sid):
    pref = TITLES[sid]
    cands = sorted((ROOT / "output").glob(f"{pref}*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def blackdetect(ff, vid):
    r = subprocess.run([ff, "-hide_banner", "-i", str(vid), "-vf",
                        "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    spans = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)
    return [[float(a), float(b)] for a, b in spans]


def loudness(ff, vid):
    r = subprocess.run([ff, "-hide_banner", "-i", str(vid), "-af",
                        "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    out = r.stderr
    def g(pat):
        m = re.findall(pat, out)
        return float(m[-1]) if m else None
    return {"lufs_integrated": g(r"I:\s*(-?[\d.]+)\s*LUFS"),
            "lra": g(r"LRA:\s*(-?[\d.]+)\s*LU"),
            "true_peak_dbfs": g(r"Peak:\s*(-?[\d.]+)\s*dBFS")}


def duration(ff, vid):
    r = subprocess.run([ff, "-hide_banner", "-i", str(vid)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def audit(sid):
    ff = ffexe()
    outdir = OUTROOT / sid
    outdir.mkdir(parents=True, exist_ok=True)
    rd = find_run_dir(sid)
    res = {"sample": sid, "run_dir": str(rd) if rd else None}
    if not rd:
        res["error"] = "no run_dir"
        return res
    vid = rd / f"{rd.name}.mp4"
    res["video"] = str(vid)
    res["video_exists"] = vid.exists()

    # manifest
    mpath = rd / "motion_graphics_manifest.json"
    if mpath.exists():
        man = json.loads(mpath.read_text())
        s = man["summary"]
        sel = [{"scene": e["scene_index"], "primitive": e["primitive"],
                "ok": e["ok"], "fallback": e["fallback"], "cache_hit": e["cache_hit"],
                "duration": e.get("duration"), "render_s": e.get("render_s")}
               for e in man["scenes"] if e.get("primitive") and not e.get("skipped")]
        res["mg"] = {"rendered": s["graphics_rendered"], "fallbacks": s["fallbacks"],
                     "cache_hits": s["cache_hits"], "by_primitive": s["by_primitive"],
                     "selected": sel, "total_render_s": s.get("total_render_s")}
        res["mg_scenes_total"] = s["scenes"]
        res["density"] = round(s["graphics_rendered"] / max(1, s["scenes"]), 3)

    # dryrun opportunities (skipped = opportunities - selected scenes)
    dpath = outdir / "dryrun.json"
    if dpath.exists():
        dj = json.loads(dpath.read_text())
        res["opportunities"] = dj.get("opportunities")
        res["palette"] = dj.get("palette")

    # render_meta
    rmeta = rd / "render_meta.json"
    if rmeta.exists():
        rm = json.loads(rmeta.read_text())
        res["meta"] = {k: rm.get(k) for k in ("beats", "scenes", "seconds", "avg_shot_s")}

    # log parse: cost (fal images), cache, footage, black-repair, sfx
    log = outdir / "render.log"
    if log.exists():
        t = log.read_text(errors="ignore")
        res["render_wall_s"] = (lambda m: float(m.group(1)) if m else None)(
            re.search(r"RENDER_DONE wall=([\d.]+)s", t))
        res["ai_images"] = len(re.findall(r"fal|flux|schnell", t, re.I)) and \
            (lambda m: int(m.group(1)) if m else None)(
                re.search(r"target=(\d+) from", t))
        res["black_repair"] = (lambda m: m.group(0) if m else "")(
            re.search(r"black-frame repair: clean.*", t))
        res["sfx_line"] = (lambda m: m.group(0).strip() if m else "")(
            re.search(r"transitions:.*", t))
        res["webfootage"] = (lambda m: m.group(0).strip() if m else "")(
            re.search(r"\[web-footage\] pool:.*on-topic.*", t))

    # audio + black + duration on the final video
    if vid.exists():
        res["duration_s"] = duration(ff, vid)
        res["black_spans_ge_0.5s"] = blackdetect(ff, vid)
        res["loudness"] = loudness(ff, vid)

    # temp leak
    leak = subprocess.run(["find", "/tmp", "/private/tmp", "/var/folders",
                           "-maxdepth", "3", "-name", "f00000.png"],
                          capture_output=True, text=True)
    res["temp_leak"] = len([x for x in leak.stdout.split("\n") if x.strip()])

    # extract MG card frames (mid clip) + 6 evenly-spaced footage frames
    if rd and (rd / "motion_graphics").exists():
        for clip in sorted((rd / "motion_graphics").glob("mg_*.mp4")):
            pid = clip.name.split("mg_", 1)[1].rsplit("_", 1)[0]
            subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-ss",
                            "4.5", "-i", str(clip), "-frames:v", "1",
                            str(outdir / f"card_{pid}.jpg")], capture_output=True)
    if vid.exists() and res.get("duration_s"):
        dur = res["duration_s"]
        for k in range(6):
            t = dur * (k + 0.5) / 6
            subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-ss",
                            f"{t:.1f}", "-i", str(vid), "-frames:v", "1",
                            str(outdir / f"frame_{k}_{int(t)}s.jpg")], capture_output=True)

    (outdir / "audit.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "all"
    ids = list(TITLES) if sid == "all" else [sid]
    rows = []
    for s in ids:
        r = audit(s)
        rows.append(r)
        mg = r.get("mg", {})
        ld = r.get("loudness", {})
        print(f"\n=== {s} ===")
        print(f"  video={r.get('video_exists')} dur={r.get('duration_s')}s "
              f"wall={r.get('render_wall_s')}s palette={r.get('palette')}")
        print(f"  graphics={mg.get('rendered')} fallbacks={mg.get('fallbacks')} "
              f"density={r.get('density')} by={mg.get('by_primitive')}")
        print(f"  LUFS={ld.get('lufs_integrated')} peak={ld.get('true_peak_dbfs')}dBFS "
              f"LRA={ld.get('lra')} black={len(r.get('black_spans_ge_0.5s', []))} "
              f"temp_leak={r.get('temp_leak')}")
    if sid == "all":
        (OUTROOT / "audit_all.json").write_text(json.dumps(rows, indent=2))
        print("\nwrote", OUTROOT / "audit_all.json")


if __name__ == "__main__":
    main()
