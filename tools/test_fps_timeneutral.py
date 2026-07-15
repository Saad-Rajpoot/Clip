#!/usr/bin/env python3
"""FPS time-neutrality proof for the Ken-Burns cut path (build._recut_to_duration / _ken_burns_filter).

A zoomed beat must consume EXACTLY the requested source-time interval [start, start+need]
regardless of the source's frame rate — otherwise the window-QC-cleared range is not what airs
(the root cause of the Max/WarnerMedia end-slate overrun: a 23.976fps source consumed ~25% more
source than the cleared window and ran into the outro card).

Method (deterministic, no OCR): each test source is a spatially-UNIFORM frame whose luma ramps
linearly 0->255 across the whole clip. A frame at source time t therefore has luma ~= 255*t/T.
The Ken-Burns chain (scale/crop/zoom/denoise/unsharp/cas) preserves a uniform field's mean, so
measuring the output's first/last-frame luma reveals exactly which SOURCE interval aired. We test
23.976 / 25 / 29.97 / 30 / 50 / 59.94 / 60 fps and a variable-frame-rate source.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.build import _recut_to_duration          # noqa: E402
from vidlore.clipstudio.config import ffmpeg_exe                 # noqa: E402

FF = ffmpeg_exe()
PASS = FAIL = 0


def _say(ok, msg):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def _make_ramp(path: Path, fps: str, T: float, vfr: bool = False) -> None:
    """Uniform-luma ramp 0->255 over T seconds at the given fps (geq uses the frame timestamp T)."""
    # cap luma just under 255 so the top never clips flat across the last frames
    lum = f"clip(254*T/{T:.3f},0,254)"
    if vfr:
        # variable frame rate: alternate long/short frame durations via a timestamp jitter, then
        # let the source keep its irregular cadence (no -r on output)
        src = (f"color=c=gray:s=320x180:r=60:d={T:.3f}")
        vf = f"geq=lum='{lum}':cb=128:cr=128,setpts='PTS+0.004*sin(40*T)/TB',fps=fps=60"
        cmd = [FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-vf", vf,
               "-vsync", "vfr", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               str(path)]
    else:
        src = f"color=c=gray:s=320x180:r={fps}:d={T:.3f}"
        vf = f"geq=lum='{lum}':cb=128:cr=128"
        cmd = [FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-vf", vf,
               "-r", fps, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _luma_at(path: Path, when: str, dur: float) -> float:
    """Mean luma of the frame at `when` ('first'|'last') of an output clip."""
    ss = "0" if when == "first" else f"{max(0.0, dur - 0.05):.3f}"
    tmp = path.with_suffix(f".{when}.png")
    subprocess.run([FF, "-y", "-loglevel", "error", "-ss", ss, "-i", str(path),
                    "-frames:v", "1", str(tmp)], check=True, capture_output=True)
    out = subprocess.run([FF, "-hide_banner", "-i", str(tmp), "-vf", "signalstats,metadata=print",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    tmp.unlink(missing_ok=True)
    import re
    m = re.search(r"lavfi\.signalstats\.YAVG=([\d.]+)", out)
    return float(m.group(1)) if m else -1.0


def _out_dur(path: Path) -> float:
    out = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    import re
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else -1.0


def main():
    import tempfile
    T = 16.0
    start, need = 4.0, 6.0                          # aired source interval must be [4.0, 10.0]
    exp_first = 254 * start / T                      # ~63.5
    exp_last = 254 * (start + need) / T              # ~158.75
    cases = [("23.976", "24000/1001"), ("25", "25"), ("29.97", "30000/1001"),
             ("30", "30"), ("50", "50"), ("59.94", "60000/1001"), ("60", "60")]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for label, fps in cases:
            src = td / f"ramp_{label}.mp4"
            _make_ramp(src, fps, T)
            dest = td / f"cut_{label}.mp4"
            got = _recut_to_duration(str(src), start, need, T, dest, zoom=1.10)
            if not got or not dest.exists():
                _say(False, f"{label}fps: recut produced no clip")
                continue
            od = _out_dur(dest)
            fl = _luma_at(dest, "first", need)
            ll = _luma_at(dest, "last", need)
            dur_ok = abs(od - need) <= 0.15
            first_ok = abs(fl - exp_first) <= 12
            last_ok = abs(ll - exp_last) <= 12          # THE overrun test: last frame = source 10.0s
            _say(dur_ok, f"{label}fps: output duration {od:.2f}s == requested {need:.2f}s")
            _say(first_ok, f"{label}fps: aired source START correct "
                           f"(first-frame luma {fl:.0f} ~= {exp_first:.0f} @ src {start:.1f}s)")
            _say(last_ok, f"{label}fps: aired source END correct — NO overrun "
                          f"(last-frame luma {ll:.0f} ~= {exp_last:.0f} @ src {start + need:.1f}s; "
                          f"a 24fps overrun would read ~{254 * (start + need * 30 / 23.976) / T:.0f})")
        # VFR source
        vsrc = td / "ramp_vfr.mp4"
        _make_ramp(vsrc, "60", T, vfr=True)
        vdest = td / "cut_vfr.mp4"
        if _recut_to_duration(str(vsrc), start, need, T, vdest, zoom=1.10) and vdest.exists():
            od = _out_dur(vdest)
            ll = _luma_at(vdest, "last", need)
            _say(abs(od - need) <= 0.2, f"VFR: output duration {od:.2f}s == requested {need:.2f}s")
            _say(abs(ll - exp_last) <= 14, f"VFR: aired source END correct (last-frame luma {ll:.0f} "
                                           f"~= {exp_last:.0f})")
        else:
            _say(False, "VFR: recut produced no clip")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
