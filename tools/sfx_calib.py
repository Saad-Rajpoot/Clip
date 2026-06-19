#!/usr/bin/env python3
"""Fast SFX-audibility calibration harness (no full render).

Synthesises a representative SFX bed with the REAL vidlore.sfx engine, mixes it
against a -16 LUFS broadband voice proxy + the real music bed using the SAME
formula as assemble.py, and reports per-event audibility (effective peak vs
final-mix RMS). Lets us tune synth levels + _GFX_VOL in seconds and SEE majors
become audible / minors stay subtle, before committing to a full render.

Usage: python3 tools/sfx_calib.py [--gfxvol 0.5] [--music WAV]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore import sfx as _sfx  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
DUR = 26.0

# representative beats: (t, kind, intensity, tier) — one per family + tier
EVENTS = [
    (1.6, "impact", 0.95, "MAJOR"),
    (3.4, "stamp", 0.70, "MAJOR"),
    (5.2, "percent_hit", 0.85, "MAJOR"),
    (7.0, "doc_slide", 0.60, "MID"),
    (8.6, "map_pin", 0.60, "MID"),
    (10.2, "map_route", 0.60, "MID"),
    (11.8, "timeline_draw", 0.60, "MID"),
    (13.0, "bar_grow", 0.60, "MID"),
    (14.4, "reveal", 0.70, "MID"),
    (16.0, "timeline_tick", 0.45, "MINOR"),
    (16.5, "timeline_tick", 0.45, "MINOR"),
    (17.0, "stat_tick", 0.45, "MINOR"),
    (18.4, "node_connect", 0.50, "MINOR"),
    (19.8, "map_pulse", 0.50, "MINOR"),
    (21.2, "countdown_clock", 0.55, "MID"),
    (22.6, "whoosh", 0.60, "MID"),
]


def vol_window(path, t0, t1):
    d = max(0.05, t1 - t0)
    out = subprocess.run([FF, "-hide_banner", "-nostats", "-ss", f"{max(0,t0):.3f}",
                          "-t", f"{d:.3f}", "-i", str(path), "-af", "volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    mx = re.search(r"max_volume:\s*(-?\d+\.?\d*) dB", out)
    mean = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    return ((float(mean.group(1)) if mean else -120.0),
            (float(mx.group(1)) if mx else -120.0))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfxvol", type=float, default=0.50)
    ap.add_argument("--music", default="output/our-man-in-damascus--the-spy-who-reached-the-heart/work_22374/music_score.wav")
    a = ap.parse_args(argv)
    tmp = Path("/tmp/sfx_calib")
    tmp.mkdir(exist_ok=True)
    bed = tmp / "bed.wav"
    _sfx.build_event_bed([(t, k, q) for t, k, q, _ in EVENTS], DUR, bed)

    # voice proxy: pink noise, -16 LUFS (broadband masker like speech energy)
    voice = tmp / "voice.wav"
    subprocess.run([FF, "-y", "-hide_banner", "-nostats", "-f", "lavfi",
                    "-i", f"anoisesrc=color=pink:r={SR}:amplitude=0.5:d={DUR}",
                    "-af", "loudnorm=I=-16:TP=-2", "-ac", "2", str(voice)],
                   capture_output=True)
    music = Path(a.music)
    # mix exactly like assemble: voice + music(0.16,ducked) + sfx(gfxvol)
    mix = tmp / f"mix_{a.gfxvol:.2f}.wav"
    inputs = ["-i", str(voice), "-i", str(bed)]
    legs = [f"[0:a]asplit=2[n][nkey]"]
    mixlbl = "[n]"
    if music.exists():
        inputs += ["-i", str(music)]
        legs.append(f"[2:a]atrim=0:{DUR},volume=0.16[mraw];"
                    f"[mraw][nkey]sidechaincompress=threshold=0.03:ratio=12:"
                    f"attack=8:release=300[m]")
        mixlbl += "[m]"
    legs.append(f"[1:a]volume={a.gfxvol}[s]")
    mixlbl += "[s]"
    k = mixlbl.count("[")
    fc = ";".join(legs) + f";{mixlbl}amix=inputs={k}:normalize=0,alimiter=limit=0.94[a]"
    subprocess.run([FF, "-y", "-hide_banner", "-nostats", *inputs,
                    "-filter_complex", fc, "-map", "[a]", "-t", f"{DUR}", str(mix)],
                   capture_output=True)

    print(f"\n=== SFX CALIB  gfxvol={a.gfxvol} ===")
    print(f"{'t':>5} {'kind':14} {'tier':6} {'sfx_eff_pk':>10} {'mix_rms':>8} "
          f"{'margin':>7}  verdict")
    import math
    gdb = 20 * math.log10(a.gfxvol) if a.gfxvol > 0 else -120
    n_ok = {"MAJOR": 0, "MID": 0, "MINOR": 0}
    n_tot = {"MAJOR": 0, "MID": 0, "MINOR": 0}
    for t, k_, q, tier in EVENTS:
        _, pk = vol_window(bed, t - 0.15, t + 0.45)
        eff = pk + gdb
        mrms, _ = vol_window(mix, t - 0.15, t + 0.45)
        margin = eff - mrms
        # majors must be audible (>=-3), mids subtle-to-audible (>=-9),
        # minors present (>=-13) but not dominating
        want = {"MAJOR": -3, "MID": -8, "MINOR": -13}[tier]
        ok = margin >= want
        n_tot[tier] += 1
        n_ok[tier] += int(ok)
        verdict = ("audible" if margin >= -3 else "subtle" if margin >= -10
                   else "buried")
        flag = "OK " if ok else "LOW"
        print(f"{t:5.1f} {k_:14} {tier:6} {eff:10.1f} {mrms:8.1f} {margin:+7.1f}  "
              f"{verdict:7} {flag}")
    print(f"\npass-by-tier: MAJOR {n_ok['MAJOR']}/{n_tot['MAJOR']}  "
          f"MID {n_ok['MID']}/{n_tot['MID']}  MINOR {n_ok['MINOR']}/{n_tot['MINOR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
