#!/usr/bin/env python3
"""Music dynamics envelope — measured proof (the closest to 'listening' we can automate).

Shape a constant tone with a breakout window and a reveal window, then measure the tone's RMS
inside each region: the breakout window must be strongly DUCKED, the reveal gently SWELLED, and
normal narration regions sit at unity — with smooth ramps (no jumps).
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.build import _music_envelope_expr, _shape_music_envelope   # noqa: E402
from vidlore.clipstudio.config import ffmpeg_exe                                    # noqa: E402

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


def _rms_db(path, t0, t1):
    out = subprocess.run(
        [FF, "-hide_banner", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", str(path),
         "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    vals = [float(m) for m in re.findall(r"RMS_level=(-?[\d.]+)", out)]
    return sum(vals) / len(vals) if vals else -120.0


def main():
    # (1) pure expr: contains a breakout dip term and a reveal boost term; no-op when empty
    e = _music_envelope_expr([(4.0, 7.0)], [(9.0, 11.0)])
    _say("1-0.850" in e or "1-0.85" in e, "expr contains a breakout dip factor")
    _say("1+0.150" in e or "1+0.15" in e, "expr contains a reveal boost factor")
    _say(_music_envelope_expr([], []) == "1.0", "empty windows → constant 1.0 (no-op)")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tone = td / "tone.wav"
        subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=frequency=220:duration=13", "-ar", "44100", "-ac", "2",
                        str(tone)], check=True, capture_output=True)
        shaped = _shape_music_envelope(str(tone), 12.0, [(4.0, 7.0)], [(9.0, 11.0)], td, log=None)
        _say(shaped != str(tone) and Path(shaped).exists(), "envelope produced a shaped track")

        normal = _rms_db(shaped, 1.0, 3.0)
        breakout = _rms_db(shaped, 4.8, 6.5)
        reveal = _rms_db(shaped, 9.4, 10.6)
        edge = _rms_db(shaped, 3.6, 3.9)         # mid-ramp into the breakout — between the two levels
        print(f"       normal={normal:.1f}dB breakout={breakout:.1f}dB reveal={reveal:.1f}dB "
              f"ramp={edge:.1f}dB")
        _say(breakout < normal - 8, f"breakout window strongly DUCKED ({breakout:.1f} << {normal:.1f} dB)")
        _say(reveal > normal + 0.5, f"reveal window SWELLED ({reveal:.1f} > {normal:.1f} dB)")
        _say(breakout < edge < normal + 0.5,
             f"ramp is gradual, not a jump (edge {edge:.1f} sits between {breakout:.1f} and normal)")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
