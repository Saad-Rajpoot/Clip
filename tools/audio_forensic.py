#!/usr/bin/env python3
"""Deep audio forensic analysis (numpy + scipy + ffmpeg).

Analyses the FULL audio timeline of a documentary MP4 — for forensic comparison
ONLY (no asset extraction/reuse). Emits a timeline JSON + metrics:

  • integrated LUFS / true-peak / LRA (ffmpeg loudnorm)
  • short-term loudness timeline (RMS dBFS @ 0.2 s hop)
  • SFX onsets (spectral-flux peak-pick) + rough category (impact/whoosh/click)
    + density per minute + whoosh density per minute
  • silence pockets (count, total, durations)
  • intro (0-30 s) vs body loudness
  • music/chapter-switch candidates (spectral self-similarity novelty)
  • outro fade behaviour (last 15 s loudness slope)
  • reveal transient strength (top onset jumps)

Usage: python3 tools/audio_forensic.py VIDEO.mp4 [--json OUT.json] [--label NAME]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 22050


def decode_mono(path: Path) -> np.ndarray:
    p = subprocess.run([FF, "-v", "quiet", "-i", str(path), "-ac", "1",
                        "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64) / 32768.0


def loudnorm(path: Path) -> dict:
    out = subprocess.run([FF, "-hide_banner", "-nostats", "-i", str(path),
                          "-af", "loudnorm=print_format=summary", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    g = lambda k: (float(m.group(1)) if (m := re.search(rf"{k}:\s*(-?\d+\.?\d*)", out)) else None)
    return {"I": g("Input Integrated"), "TP": g("Input True Peak"), "LRA": g("Input LRA")}


def _db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-9))


def analyze(path: Path, label: str = "") -> dict:
    a = decode_mono(path)
    n = len(a)
    dur = n / SR
    # ── loudness envelope (RMS dBFS @ 0.2 s hop, 0.4 s window) ───────────────
    win = int(0.4 * SR); hop = int(0.2 * SR)
    p2 = uniform_filter1d(a * a, size=win)
    env = np.sqrt(p2[::hop])
    te = np.arange(len(env)) * hop / SR
    edb = _db(env)
    floor = np.percentile(edb[edb > -90], 5)

    # ── STFT for onsets + bands ─────────────────────────────────────────────
    nper = 1024; nov = 512
    f, tt, Z = signal.stft(a, SR, nperseg=nper, noverlap=nov, boundary=None)
    mag = np.abs(Z)
    frame_dt = (nper - nov) / SR
    # spectral flux onset strength
    flux = np.sum(np.maximum(0.0, np.diff(mag, axis=1)), axis=0)
    flux = np.concatenate([[0.0], flux])
    fn = flux / (np.median(flux) + 1e-9)
    pk, props = signal.find_peaks(fn, height=float(np.percentile(fn, 93)),
                                  distance=max(1, int(0.20 / frame_dt)))
    onset_t = tt[pk]

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return mag[m].sum(axis=0)
    low, mid, high = band(20, 250), band(250, 3000), band(3000, 11025)
    tot = low + mid + high + 1e-9

    # classify each onset by local band balance + how long the rise lasts
    cats = {"impact": 0, "whoosh": 0, "click": 0, "sfx": 0}
    onset_rows = []
    for ot, oi in zip(onset_t, pk):
        lo, mi, hi = low[oi], mid[oi], high[oi]
        s = lo + mi + hi + 1e-9
        # rise length: frames around the peak above 60% of peak flux
        l = oi
        while l > 0 and fn[l] > 0.4 * fn[oi]:
            l -= 1
        r = oi
        while r < len(fn) - 1 and fn[r] > 0.4 * fn[oi]:
            r += 1
        rise = (r - l) * frame_dt
        if lo / s > 0.45 and rise < 0.5:
            cat = "impact"
        elif hi / s > 0.5 and rise < 0.12:
            cat = "click"
        elif rise > 0.28:
            cat = "whoosh"
        else:
            cat = "sfx"
        cats[cat] += 1
        # transient strength: env jump at the onset (dB)
        k = int(ot / (hop / SR))
        before = edb[max(0, k - 6):max(1, k - 1)].mean() if k > 1 else floor
        at = edb[k:k + 3].max() if k < len(edb) else floor
        onset_rows.append({"t": round(float(ot), 2), "cat": cat,
                           "rise_s": round(float(rise), 2),
                           "jump_db": round(float(at - before), 1)})

    # ── silence pockets (env below floor+8 dB for >= 0.25 s) ────────────────
    sil_th = floor + 8
    is_sil = edb < sil_th
    pockets = []
    i = 0
    while i < len(is_sil):
        if is_sil[i]:
            j = i
            while j < len(is_sil) and is_sil[j]:
                j += 1
            d = (j - i) * hop / SR
            if d >= 0.25:
                pockets.append(round(d, 2))
            i = j
        else:
            i += 1

    # ── novelty (chapter / music switch candidates) ─────────────────────────
    # feature per 1 s: [low,mid,high fractions, centroid] -> cosine self-sim novelty
    fps = max(1, int(1.0 / frame_dt))
    feat = np.stack([low / tot, mid / tot, high / tot,
                     (f[:, None] * mag).sum(0) / (mag.sum(0) + 1e-9) / SR], axis=1)
    fa = np.array([feat[i:i + fps].mean(0) for i in range(0, len(feat) - fps, fps)])
    fa = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-9)
    nov = np.array([1 - float(fa[i] @ fa[i + 1]) for i in range(len(fa) - 1)])
    nv_pk, _ = signal.find_peaks(nov, height=float(np.percentile(nov, 90)),
                                 distance=8)
    switches = [round(float(p), 1) for p in nv_pk]   # seconds

    # ── intro vs body, outro fade ───────────────────────────────────────────
    intro = edb[te <= 30]
    body = edb[(te > 30) & (te < dur - 20)]
    outro = edb[te > dur - 15]
    # outro slope (dB per s) over last 15 s
    if len(outro) > 3:
        xo = te[te > dur - 15]
        slope = float(np.polyfit(xo - xo[0], outro, 1)[0])
    else:
        slope = 0.0

    ln = loudnorm(path)
    strong = sorted(onset_rows, key=lambda x: -x["jump_db"])[:12]
    return {
        "label": label or path.name, "path": str(path), "duration_s": round(dur, 1),
        "integrated_lufs": ln["I"], "true_peak_dbtp": ln["TP"], "lra": ln["LRA"],
        "intro_rms_db": round(float(np.median(intro)), 1) if len(intro) else None,
        "body_rms_db": round(float(np.median(body)), 1) if len(body) else None,
        "intro_minus_body_db": round(float(np.median(intro) - np.median(body)), 1)
        if len(intro) and len(body) else None,
        "outro_fade_db_per_s": round(slope, 2),
        "onsets_total": len(onset_rows),
        "sfx_per_min": round(len(onset_rows) / (dur / 60), 1),
        "whoosh_per_min": round(cats["whoosh"] / (dur / 60), 2),
        "onset_categories": cats,
        "silence_pockets": len(pockets),
        "silence_total_s": round(float(sum(pockets)), 1),
        "silence_median_s": round(float(np.median(pockets)), 2) if pockets else 0,
        "music_switch_candidates": len(switches),
        "switches_per_min": round(len(switches) / (dur / 60), 2),
        "avg_segment_s": round(dur / max(1, len(switches) + 1), 1),
        "strongest_reveals": strong,
        "switch_times_s": switches[:60],
        "_timeline": {"t": [round(float(x), 2) for x in te[::5]],
                      "rms_db": [round(float(x), 1) for x in edb[::5]]},
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    r = analyze(Path(a.video), a.label)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1), encoding="utf-8")
    show = {k: v for k, v in r.items() if not k.startswith("_") and k != "strongest_reveals"
            and k != "switch_times_s"}
    print(json.dumps(show, indent=1))
    print("strongest reveals:", [f"{s['t']}s {s['cat']}+{s['jump_db']}dB"
                                 for s in r["strongest_reveals"][:8]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
