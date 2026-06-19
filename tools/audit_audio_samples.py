#!/usr/bin/env python3
"""Run the audio QA gate on the 5 niche samples + emit a cross-niche report.

For each sample it locates the final MP4 + the cue sheets (music_cue_sheet.json in
work_*/, sfx_cue_sheet.json beside the MP4), runs tools/audio_quality_audit.py, and
aggregates the verdicts into research/audio_engine/CROSS_NICHE_AUDIO_REPORT.md +
cross_niche_audio.json.

Usage:  python3 tools/audit_audio_samples.py
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "audio_engine"

SAMPLES = {
    "spy":         "our-man-in-damascus*",
    "crime":       "the-untouchable*",
    "business":    "john-d--rockefeller*",
    "history":     "the-road-to-moscow*",
    "geopolitics": "thirteen-days*",
}


def _find(pattern: str):
    dirs = glob.glob(str(ROOT / "output" / pattern))
    if not dirs:
        return None
    d = Path(dirs[0])
    mp4s = sorted(d.glob("*.mp4"), key=lambda p: -p.stat().st_mtime)
    if not mp4s:
        return None
    music_cue = next(iter(sorted(d.glob("work_*/music_cue_sheet.json"))), None)
    sfx_cue = d / "sfx_cue_sheet.json"
    return {"dir": d, "mp4": mp4s[0],
            "music_cue": music_cue, "sfx_cue": sfx_cue if sfx_cue.exists() else None}


def audit_one(niche: str, info: dict) -> dict:
    cmd = [sys.executable, str(ROOT / "tools" / "audio_quality_audit.py"),
           str(info["mp4"]), "--scope", "audio_validation"]
    if info.get("music_cue"):
        cmd += ["--music-cues", str(info["music_cue"])]
    if info.get("sfx_cue"):
        cmd += ["--sfx-cues", str(info["sfx_cue"])]
    subprocess.run(cmd, capture_output=True, text=True)
    rep_path = info["mp4"].parent / "audio_quality_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    # pull intro lift from the music cue sheet
    intro = {}
    if info.get("music_cue"):
        ms = json.loads(Path(info["music_cue"]).read_text())
        intro = ms.get("intro_profile", {})
        rep["_lead_category"] = ms.get("lead_category")
        rep["_categories"] = ms.get("categories_used", [])
        rep["_chapters"] = ms.get("chapters")
    if info.get("sfx_cue"):
        sc = json.loads(Path(info["sfx_cue"]).read_text())
        rep["_sfx_per_min"] = sc.get("sfx_per_min")
        rep["_whoosh_per_min"] = sc.get("whoosh_per_min")
        rep["_sfx_families"] = sc.get("families_used", [])
    rep["_intro"] = intro
    return rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for niche, pat in SAMPLES.items():
        info = _find(pat)
        if not info:
            print(f"  {niche:11} NO RENDER FOUND ({pat})")
            results[niche] = {"verdict": "MISSING"}
            continue
        rep = audit_one(niche, info)
        results[niche] = rep
        print(f"  {niche:11} {rep.get('verdict','?'):5} "
              f"LUFS={rep.get('integrated_lufs')} peak={rep.get('true_peak_dbtp')} "
              f"lead={rep.get('_lead_category')} sfx/min={rep.get('_sfx_per_min')}")

    (OUT / "cross_niche_audio.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # markdown report
    lines = ["# Cross-Niche Audio Validation — AudioEngine V1.0", "",
             "| niche | verdict | LUFS | true-peak | lead category | intro lift | sfx/min | whoosh/min | families |",
             "|---|---|---|---|---|---|---|---|---|"]
    for niche in SAMPLES:
        r = results.get(niche, {})
        intro = r.get("_intro", {})
        lift = f"{intro.get('start_mult','—')}× ({intro.get('texture','')[:18]})" if intro else "—"
        fams = ",".join(r.get("_sfx_families", [])[:4])
        lines.append(
            f"| {niche} | {r.get('verdict','?')} | {r.get('integrated_lufs','—')} | "
            f"{r.get('true_peak_dbtp','—')} | {r.get('_lead_category','—')} | {lift} | "
            f"{r.get('_sfx_per_min','—')} | {r.get('_whoosh_per_min','—')} | {fams} |")
    # cross-video repetition: are the lead categories distinct?
    leads = [results.get(n, {}).get("_lead_category") for n in SAMPLES]
    leads = [x for x in leads if x]
    distinct = len(set(leads))
    lines += ["", f"**Cross-video anti-repetition:** {distinct}/{len(leads)} distinct lead "
              f"categories across the 5 niches ({', '.join(leads)}).", "",
              "**Verdicts:** " + ", ".join(
                  f"{n}={results.get(n,{}).get('verdict','?')}" for n in SAMPLES), ""]
    (OUT / "CROSS_NICHE_AUDIO_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT/'CROSS_NICHE_AUDIO_REPORT.md'}")
    fails = [n for n in SAMPLES if results.get(n, {}).get("verdict") == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
