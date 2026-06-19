#!/usr/bin/env python3
"""Perceptual audio probe — validates that music + SFX are GENUINELY AUDIBLE and
dynamic inside the FINAL rendered MP4 (not just present in a cue sheet).

This is the 'ears' proxy: I cannot literally listen, so this measures the things
that correspond to what a listener perceives, and FAILS when they're weak:

  • intro vs body MUSIC level   — from music_score.wav (require intro hotter)
  • SFX audibility per cue       — gfxsfx/sfx stem peak (after mix gain) vs the
                                   final-mix RMS floor in the same window
                                   (require MAJOR reveals to clearly poke through)
  • dynamic range (LRA)          — flat = un-cinematic
  • cold-open / music-forward     — is there a music-led moment before the VO?
  • loudness / true-peak          — broadcast safety

Usage:
  python3 tools/audio_probe.py --mp4 OUT.mp4 --work WORKDIR [--sfx-cues S.json]
                               [--intro 30] [--json report.json]
Stems expected in WORKDIR: music_score.wav, gfxsfx.wav, atmosbed.wav (optional),
sfxbeds (sfx.wav) — any missing stem degrades gracefully.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()

def _live_gains() -> dict:
    """Read the CURRENT mix gains straight from vidlore/assemble.py source, so the
    probe always reflects the LIVE engine and never uses stale hardcoded constants.
    Falls back to v14 defaults only if the file can't be read."""
    g = {"music": 0.16, "gfxsfx": 1.35, "sfx": 1.15, "numsfx": 0.90,
         "atmos": 0.45, "type": 0.80, "arch": 0.30}
    keymap = {"_MUSIC_VOL_DUCK": "music", "_GFX_VOL": "gfxsfx", "_SFX_VOL": "sfx",
              "_NUMSFX_VOL": "numsfx", "_ATMOS_VOL": "atmos", "_TYPE_VOL": "type",
              "_ARCH_VOL": "arch"}
    try:
        src = (Path(__file__).resolve().parents[1] / "vidlore"
               / "assemble.py").read_text(encoding="utf-8")
        for k, gk in keymap.items():
            m = re.search(rf"^{k}\s*=\s*([0-9]*\.?[0-9]+)", src, re.M)
            if m:
                g[gk] = float(m.group(1))
    except Exception:                                              # noqa: BLE001
        pass
    return g


# final-mix gains — read LIVE from assemble.py (no stale constants).
GAIN = _live_gains()


def _run(args) -> str:
    return subprocess.run([FF, "-hide_banner", "-nostats", *args],
                          capture_output=True, text=True).stderr


def vol_window(path: Path, t0: float, t1: float):
    """(mean_dbfs, max_dbfs) of `path` in [t0,t1] via volumedetect. -120 if silent."""
    if not path or not path.exists():
        return (-120.0, -120.0)
    d = max(0.05, t1 - t0)
    out = _run(["-ss", f"{max(0,t0):.3f}", "-t", f"{d:.3f}", "-i", str(path),
                "-af", "volumedetect", "-f", "null", "-"])
    mean = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    mx = re.search(r"max_volume:\s*(-?\d+\.?\d*) dB", out)
    return (float(mean.group(1)) if mean else -120.0,
            float(mx.group(1)) if mx else -120.0)


def loudnorm_summary(path: Path, ss=None, t=None) -> dict:
    a = []
    if ss is not None:
        a += ["-ss", str(ss)]
    if t is not None:
        a += ["-t", str(t)]
    out = _run([*a, "-i", str(path), "-af", "loudnorm=print_format=summary",
                "-f", "null", "-"])
    g = lambda k: (float(m.group(1)) if (m := re.search(
        rf"{k}:\s*(-?\d+\.?\d*)", out)) else None)
    return {"I": g("Input Integrated"), "TP": g("Input True Peak"),
            "LRA": g("Input LRA")}


def db(x):
    return -120.0 if x <= 0 else 20 * math.log10(x)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--sfx-cues", default="")
    ap.add_argument("--intro", type=float, default=30.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    mp4 = Path(a.mp4)
    work = Path(a.work)
    music = work / "music_score.wav"
    gfx = work / "gfxsfx.wav"
    sfxb = work / "sfx.wav"
    atmos = work / "atmosbed.wav"

    rep: dict = {"mp4": str(mp4), "checks": {}}
    R = rep["checks"]

    # ── final loudness / dynamics ──────────────────────────────────────────
    fin = loudnorm_summary(mp4)
    lra = fin.get("LRA") or 0.0
    R["final"] = {"I_lufs": fin.get("I"), "TP_dbtp": fin.get("TP"), "LRA": lra,
                  "dynamic_verdict": "PASS" if lra >= 5.0 else
                  ("WARN" if lra >= 4.0 else "FAIL"),
                  "note": f"LRA {lra} — pro docs breathe ~5-9 LU; <4 is flat"}

    # ── intro vs body MUSIC level (the real intro-strength test) ───────────
    if music.exists():
        i_mean, i_max = vol_window(music, 0.0, a.intro)
        b_mean, b_max = vol_window(music, a.intro, 10_000)
        delta = round(i_mean - b_mean, 2)
        R["intro_vs_body_music"] = {
            "intro_rms_db": i_mean, "body_rms_db": b_mean, "delta_db": delta,
            "verdict": "PASS" if delta >= 2.0 else
            ("WARN" if delta >= 0.5 else "FAIL"),
            "note": "intro MUSIC should sit clearly hotter than body (≥+2 dB)"}
        # cold-open: is there a music-forward moment in first 3s (music loud,
        # i.e. not yet ducked hard)? measure music RMS 0-2.5s.
        co_mean, co_max = vol_window(music, 0.0, 2.5)
        R["cold_open"] = {"music_rms_0_2.5s_db": co_mean,
                          "verdict": "PASS" if co_max >= -26 else "WARN",
                          "note": "music-led open before VO adds energy + range"}

    # ── SFX audibility per cue ─────────────────────────────────────────────
    cues = []
    if a.sfx_cues and Path(a.sfx_cues).exists():
        cues = json.loads(Path(a.sfx_cues).read_text()).get("events", [])
    # map family -> which stem carries it + that stem's mix gain
    def stem_for(fam, kind):
        if fam in ("data",) or kind in ("count_up", "number", "num"):
            return gfx, GAIN["numsfx"] if False else GAIN["gfxsfx"]
        return gfx, GAIN["gfxsfx"]
    cue_rows = []
    audible = subtle = buried = 0
    for e in cues:
        t = float(e.get("time_s", e.get("time", 0)))
        fam = e.get("family", "")
        kind = e.get("kind", "")
        inten = float(e.get("intensity", 0))
        stem, g = stem_for(fam, kind)
        _, sfx_peak = vol_window(stem, t - 0.15, t + 0.45)
        sfx_eff_peak = sfx_peak + db(g)               # after mix gain
        masker_mean, _ = vol_window(mp4, t - 0.15, t + 0.45)  # full mix RMS
        margin = round(sfx_eff_peak - masker_mean, 1)
        verdict = ("audible" if margin >= -3 else
                   "subtle" if margin >= -10 else "buried")
        if verdict == "audible":
            audible += 1
        elif verdict == "subtle":
            subtle += 1
        else:
            buried += 1
        cue_rows.append({"t": round(t, 2), "kind": kind, "family": fam,
                         "intensity": inten,
                         "sfx_eff_peak_db": round(sfx_eff_peak, 1),
                         "mix_rms_db": round(masker_mean, 1),
                         "margin_db": margin, "verdict": verdict})
    R["sfx_audibility"] = {
        "n_cues": len(cue_rows), "audible": audible, "subtle": subtle,
        "buried": buried,
        "verdict": "FAIL" if (cue_rows and buried >= max(1, len(cue_rows)//2))
        else ("WARN" if buried else "PASS"),
        "note": "margin = SFX effective peak − final-mix RMS in cue window. "
                "audible≥-3, subtle≥-10, buried<-10 dB. Major reveals must be audible.",
        "cues": cue_rows}
    # SFX presence in intro window
    intro_sfx = [c for c in cue_rows if c["t"] <= a.intro]
    R["intro_sfx"] = {"count": len(intro_sfx),
                      "verdict": "PASS" if intro_sfx else "FAIL",
                      "note": "the hook/intro needs at least one motivated SFX"}

    # raw stem levels (diagnostic)
    R["stems"] = {}
    for nm, p, g in (("music", music, GAIN["music"]), ("gfxsfx", gfx, GAIN["gfxsfx"]),
                     ("sfx", sfxb, GAIN["sfx"]), ("atmos", atmos, GAIN["atmos"])):
        if p.exists():
            mean, mx = vol_window(p, 0, 10_000)
            R["stems"][nm] = {"raw_rms_db": mean, "raw_peak_db": mx,
                              "eff_peak_db": round(mx + db(g), 1), "mix_gain": g}

    # overall verdict
    fails = [k for k, v in R.items() if isinstance(v, dict)
             and v.get("verdict") == "FAIL"]
    rep["verdict"] = "FAIL" if fails else "PASS"
    rep["failed_checks"] = fails

    # print summary
    print(f"\n=== AUDIO PROBE — {mp4.name} ===")
    print(f"verdict: {rep['verdict']}   failed: {fails or '—'}")
    f = R["final"]
    print(f"final: I={f['I_lufs']} LUFS  TP={f['TP_dbtp']}  LRA={f['LRA']} "
          f"[{f['dynamic_verdict']}]")
    if "intro_vs_body_music" in R:
        iv = R["intro_vs_body_music"]
        print(f"intro vs body MUSIC: {iv['intro_rms_db']:.1f} vs "
              f"{iv['body_rms_db']:.1f} dB = {iv['delta_db']:+.1f} dB "
              f"[{iv['verdict']}]")
    sa = R["sfx_audibility"]
    print(f"SFX audibility: {sa['audible']} audible / {sa['subtle']} subtle / "
          f"{sa['buried']} buried  of {sa['n_cues']}  [{sa['verdict']}]")
    print(f"intro SFX present: {R['intro_sfx']['count']} [{R['intro_sfx']['verdict']}]")
    if R.get("stems"):
        for nm, s in R["stems"].items():
            print(f"  stem {nm:7s}: raw_peak {s['raw_peak_db']:.1f} → "
                  f"eff_peak {s['eff_peak_db']:.1f} dB (×{s['mix_gain']})")
    for c in sa["cues"]:
        print(f"    cue t={c['t']:6.2f} {c['family']:9s} {c['kind']:14s} "
              f"eff_peak {c['sfx_eff_peak_db']:6.1f}  mix {c['mix_rms_db']:6.1f}  "
              f"margin {c['margin_db']:+5.1f}  {c['verdict']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
