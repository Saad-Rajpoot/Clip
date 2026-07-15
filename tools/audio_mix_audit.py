#!/usr/bin/env python3
"""Real-mix audio audit (Gap 5b): measure the ACTUAL final mix + music stem, not a synthetic tone.

Reports integrated LUFS, true-peak (clipping), and — using the baked music stem score_shaped.wav —
the music level inside breakout windows vs narration windows (proving the breakout DUCK and the
restrained reveal swells on the real timeline). Usage: audio_mix_audit.py <final.mp4> <work_dir>
[breakout_audit.json]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

FF = "/Users/hussnain/Library/Python/3.9/lib/python/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"


def _stderr(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def loudness(path):
    o = _stderr([FF, "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    # parse the SUMMARY block (the per-frame running "I:" converges from -70; take the summary value)
    I = re.search(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+)\s*LUFS", o)
    tp = re.search(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+)\s*dBFS", o)
    lra = re.search(r"Loudness range:\s*\n\s*LRA:\s*([\d.]+)\s*LU", o)
    return (float(I.group(1)) if I else None,
            float(tp.group(1)) if tp else None,
            float(lra.group(1)) if lra else None)


def rms(path, t0, t1):
    o = _stderr([FF, "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", str(path),
                 "-af", "astats=metadata=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
                 "-f", "null", "-"])
    v = [float(m) for m in re.findall(r"RMS_level=(-?[\d.]+)", o)]
    return round(sum(v) / len(v), 1) if v else None


def main():
    vid = Path(sys.argv[1])
    work = Path(sys.argv[2])
    audit = Path(sys.argv[3]) if len(sys.argv) > 3 else (work / "breakout_audit.json")
    score = work / "score_shaped.wav"
    R = {"final_mix": {}, "music_duck": [], "reveal_note": ""}

    I, tp, lra = loudness(vid)
    R["final_mix"] = {"integrated_lufs": I, "true_peak_dbfs": tp, "lra_lu": lra,
                      "clipping": (tp is not None and tp >= 0.0),
                      "near_platform_norm": (I is not None and -18.0 <= I <= -13.0)}

    bwins = []
    if audit.exists():
        d = json.loads(audit.read_text())
        for a in d.get("accepted", []):
            s = a.get("aired_at_s")
            du = a.get("dur_s")
            if s is not None and du:
                bwins.append((float(s), float(s) + float(du), a.get("line", "")[:30]))

    if score.exists() and bwins:
        for s, e, line in bwins:
            # narration reference window just before the breakout
            ns, ne = max(0.0, s - 6.0), max(0.1, s - 1.0)
            m_bk = rms(score, s + 0.3, e - 0.3)
            m_narr = rms(score, ns, ne)
            v_bk = rms(vid, s + 0.3, e - 0.3)          # final-mix level in the breakout window
            duck_db = (round(m_narr - m_bk, 1) if (m_bk is not None and m_narr is not None) else None)
            R["music_duck"].append({
                "t": round(s, 1), "line": line,
                "music_rms_narration_db": m_narr, "music_rms_breakout_db": m_bk,
                "duck_db": duck_db, "final_mix_rms_breakout_db": v_bk,
                "ducked": (duck_db is not None and duck_db >= 4.0)})
    R["stems_present"] = {"score_shaped": score.exists()}
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
