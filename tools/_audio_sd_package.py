"""Validate the 5 niche renders + build the manual-listening package.

For each niche (spy/crime/business/history/geopolitics):
  • run the perceptual probe (audio_probe, LIVE gains),
  • measure the final-mix transient at every SFX cue, rank the strongest moments,
  • extract clips for listening: 45-60s intro, 3+ reveal moments, a body-music clip,
  • copy music_cue_sheet / sfx_cue_sheet / MUSIC_CREDITS.txt / audio_quality_report,
  • run the engineering checklist (LUFS ~-16, true-peak headroom, no clipping,
    no harsh spikes, whoosh-spam, black-frame regression, temp leak),
and writes everything + a listening checklist into
research/audio_engine/post_fix_manual_review/.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audio_probe as AP  # noqa: E402  (live-gain probe)

FF = imageio_ffmpeg.get_ffmpeg_exe()
NICHES = ["spy", "crime", "business", "history", "geopolitics"]
PKG = ROOT / "research" / "audio_engine" / "post_fix_manual_review"


def vol(path, t0, t1):
    return AP.vol_window(Path(path), t0, t1)        # (mean, max) dBFS


def clip(mp4, t0, dur, dest):
    subprocess.run([FF, "-y", "-hide_banner", "-nostats", "-ss", f"{max(0,t0):.2f}",
                    "-t", f"{dur:.2f}", "-i", str(mp4), "-c", "copy",
                    "-avoid_negative_ts", "make_zero", str(dest)],
                   capture_output=True)
    if not dest.exists() or dest.stat().st_size < 1000:   # copy failed at cut -> re-encode
        subprocess.run([FF, "-y", "-hide_banner", "-nostats", "-ss", f"{max(0,t0):.2f}",
                        "-t", f"{dur:.2f}", "-i", str(mp4), "-c:v", "libx264",
                        "-c:a", "aac", "-b:a", "192k", str(dest)], capture_output=True)


def dur_of(mp4):
    out = subprocess.run([FF, "-hide_banner", "-i", str(mp4)], capture_output=True,
                         text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", out)
    return (int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))) if m else 0.0


def process(niche: str) -> dict:
    base = ROOT / "output" / f"_sd5_{niche}"
    subs = [d for d in base.glob("*") if d.is_dir() and (list(d.glob("*.mp4")))]
    if not subs:
        return {"niche": niche, "ok": False, "error": "no render dir/mp4"}
    rdir = subs[0]
    mp4 = sorted(rdir.glob("*.mp4"))[0]
    work = next(iter(rdir.glob("work_*")), None)
    total = dur_of(mp4)
    outd = PKG / niche
    outd.mkdir(parents=True, exist_ok=True)

    # copy the QA artifacts
    copied = []
    for name in ("sfx_cue_sheet.json", "audio_quality_report.json",
                 "MUSIC_CREDITS.txt", "render_black_frame_metrics.json",
                 "render_meta.json"):
        s = rdir / name
        if s.exists():
            shutil.copy2(s, outd / name)
            copied.append(name)
    if work and (work / "music_cue_sheet.json").exists():
        shutil.copy2(work / "music_cue_sheet.json", outd / "music_cue_sheet.json")
        copied.append("music_cue_sheet.json")

    # SFX cues -> measure final-mix transient, rank strongest
    cues = []
    scs = rdir / "sfx_cue_sheet.json"
    if scs.exists():
        cues = json.loads(scs.read_text()).get("events", [])
    strongest = []
    for e in cues:
        t = float(e.get("time_s", 0))
        bmean, _ = vol(mp4, t - 1.3, t - 0.3)
        amean, amax = vol(mp4, t - 0.05, t + 0.45)
        strongest.append({"t": round(t, 2), "kind": e.get("kind"),
                          "family": e.get("family"),
                          "jump_db": round(amean - bmean, 1), "at_peak": round(amax, 1)})
    strongest.sort(key=lambda x: -x["jump_db"])
    top = [s for s in strongest if s["jump_db"] >= 2.0][:5] or strongest[:3]

    # clips
    clips = []
    clip(mp4, 0, min(58, total), outd / f"{niche}_intro_0-58s.mp4")
    clips.append(f"{niche}_intro_0-58s.mp4")
    # body-music clip: a mid stretch away from reveals
    bstart = min(max(total * 0.45, 30), max(0, total - 18))
    clip(mp4, bstart, 16, outd / f"{niche}_body_music_{int(bstart)}s.mp4")
    clips.append(f"{niche}_body_music_{int(bstart)}s.mp4")
    for i, s in enumerate(top[:4]):
        c = f"{niche}_reveal{i+1}_{s['t']:.0f}s.mp4"
        clip(mp4, max(0, s["t"] - 2.5), 8, outd / c)
        clips.append(c)

    # checklist measurements
    ln = AP.loudnorm_summary(mp4)
    i_mean, _ = vol(mp4, 0, 3)
    b_mean, _ = vol(mp4, bstart, 3)
    # black-frame + temp
    bf = {}
    bfp = rdir / "render_black_frame_metrics.json"
    if bfp.exists():
        try:
            bf = json.loads(bfp.read_text())
        except Exception:                                          # noqa: BLE001
            bf = {}
    sfx_sheet = json.loads(scs.read_text()) if scs.exists() else {}
    tp = ln.get("TP")
    checklist = {
        "LUFS": ln.get("I"), "LUFS_ok": (ln.get("I") is not None and -17.5 <= ln["I"] <= -14.5),
        "true_peak_dbtp": tp, "peak_headroom_ok": (tp is not None and tp <= -0.5),
        "no_clipping": (tp is not None and tp < 0.0),
        "intro_vs_body_finalmix_db": round(i_mean - b_mean, 1),
        "sfx_events": sfx_sheet.get("total_events"),
        "whoosh_per_min": sfx_sheet.get("whoosh_per_min"),
        "no_whoosh_spam": (float(sfx_sheet.get("whoosh_per_min", 0) or 0) <= 3.0),
        "strong_sfx_moments": len([s for s in strongest if s["jump_db"] >= 3.0]),
        "black_frame_verdict": bf.get("verdict") or bf.get("status") or "n/a",
        "temp_leak": _temp_leak(rdir, work),
    }
    return {"niche": niche, "ok": True, "mp4": str(mp4), "duration_s": round(total, 1),
            "work": str(work) if work else None, "copied": copied, "clips": clips,
            "strongest_sfx": top, "checklist": checklist,
            "music_cue_sheet": (outd / "music_cue_sheet.json").exists(),
            "credits": (outd / "MUSIC_CREDITS.txt").exists()}


def _temp_leak(rdir, work):
    # stray temp files OUTSIDE the intentionally-kept work dir
    strays = []
    for pat in ("*.part", "*.tmp", "*~"):
        strays += list(rdir.glob(pat))
    sys_tmp = list(Path("/tmp").glob("sfxbed_*")) + list(Path("/tmp").glob("vidlore_*"))
    return {"stray_in_outdir": len(strays), "stray_in_tmp": len(sys_tmp),
            "ok": (len(strays) == 0 and len(sys_tmp) == 0)}


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    rows = [process(n) for n in NICHES]
    (PKG / "package_index.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    # listening checklist + index markdown
    md = ["# Post-fix audio — manual listening package (v14)", "",
          "Listen to each clip and tick the box. Engine fixes: stronger hold-then-",
          "recede intro, intro accent, tiered SFX gains, audible reveals.", "",
          "## Listening checklist (per niche)",
          "- [ ] intro music clearly STRONGER than body music",
          "- [ ] intro stays BALANCED with the voiceover (voice clear)",
          "- [ ] meaningful visual actions have an AUDIBLE sfx",
          "- [ ] sfx are noticeable but NOT excessive / no harsh spikes",
          "- [ ] no repeated whoosh spam",
          "- [ ] reveal timing feels correct; rhythm matches narration",
          "- [ ] body music supports without distracting", ""]
    for r in rows:
        md.append(f"## {r['niche'].upper()}")
        if not r.get("ok"):
            md.append(f"  ❌ {r.get('error')}"); md.append(""); continue
        c = r["checklist"]
        md.append(f"- video: `{r['mp4']}`  ({r['duration_s']}s)")
        md.append(f"- LUFS {c['LUFS']} ({'OK' if c['LUFS_ok'] else 'CHECK'}) · "
                  f"true-peak {c['true_peak_dbtp']} dBTP "
                  f"({'headroom OK' if c['peak_headroom_ok'] else 'CHECK'}, "
                  f"{'no clip' if c['no_clipping'] else 'CLIP!'})")
        md.append(f"- intro vs body (final mix): {c['intro_vs_body_finalmix_db']:+} dB · "
                  f"sfx events {c['sfx_events']} · whoosh/min {c['whoosh_per_min']} "
                  f"({'no spam' if c['no_whoosh_spam'] else 'SPAM?'})")
        md.append(f"- black-frame: {c['black_frame_verdict']} · "
                  f"temp-leak: {'none' if c['temp_leak']['ok'] else c['temp_leak']}")
        md.append(f"- strongest SFX moments: " +
                  ", ".join(f"{s['t']}s {s['kind']}(+{s['jump_db']}dB)"
                            for s in r["strongest_sfx"]))
        md.append("- clips: " + ", ".join(f"`{r['niche']}/{cn}`" for cn in r["clips"]))
        md.append("")
    (PKG / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[package] -> {PKG}")


if __name__ == "__main__":
    main()
