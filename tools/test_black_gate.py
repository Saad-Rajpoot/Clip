#!/usr/bin/env python3
"""Final-video sustained-black / legibility release gate (Gap 3) — end-to-end proof.

A sustained near-black region must block publication; a short fade must not; a bright video passes;
and a gate that cannot extract frames fails closed.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore.clipstudio.build import _final_video_black_gate, _frame_luma_hi   # noqa: E402
from vidlore.clipstudio.config import ffmpeg_exe                                # noqa: E402

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


def _concat(parts, dest, td):
    lst = td / "l.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(dest)], check=True, capture_output=True)


def _seg(color, dur, dest):
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"color=c={color}:s=640x360:rate=30", "-t", str(dur),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)], check=True, capture_output=True)


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        work = td / "output" / "work"
        work.mkdir(parents=True)
        bright = td / "bright.mp4"; dark = td / "dark.mp4"; fade = td / "fade.mp4"
        _seg("gray", 3, bright)
        _seg("0x0A0A0A", 3, dark)          # near-black, sustained (luma_hi ~10)
        _seg("0x000000", 0.4, fade)        # a 0.4s black dip (a fade, allowed)

        # (1) a video with a SUSTAINED (3s) near-black region → blocked
        v1 = work / "v1.mp4"
        _concat([bright, dark, bright], v1, td)
        blocked = False
        try:
            _final_video_black_gate(v1, work, log=lambda m: None)
        except RuntimeError:
            blocked = True
        _say(blocked, "sustained near-black region BLOCKS publication")
        _say((work.parent / "final_black_failures.json").exists(), "writes final_black_failures.json")

        # (2) a video with only a SHORT 0.4s dark dip (fade) → passes
        v2 = work / "v2.mp4"
        _concat([bright, fade, bright], v2, td)
        out = _final_video_black_gate(v2, work, log=lambda m: None)
        _say(out == v2 and v2.exists(), "short fade is allowed (does NOT block)")

        # (3) a fully bright video → passes
        v3 = work / "v3.mp4"
        _seg("gray", 5, v3)
        out3 = _final_video_black_gate(v3, work, log=lambda m: None)
        _say(out3 == v3 and v3.exists(), "bright video passes")

        # (3b) R4-2 TAIL fixture: a sustained-black interval ONLY in the FINAL 1.0s must block
        bright5 = td / "_b5.mp4"; blk1 = td / "_blk1.mp4"; tail = work / "tail.mp4"
        _seg("gray", 5, bright5); _seg("0x080808", 1, blk1)
        _concat([bright5, blk1], tail, td)
        blocked_tail = False
        try:
            _final_video_black_gate(tail, work, log=lambda m: None)
        except RuntimeError:
            blocked_tail = True
        _say(blocked_tail, "sustained-black interval in the FINAL 1.0s is reached + BLOCKED")

        # (4) luma_hi measurement: dark low, bright high
        import os as _os
        b_img = td / "b.jpg"; d_img = td / "d.jpg"
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(bright), "-frames:v", "1", str(b_img)],
                       check=True, capture_output=True)
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(dark), "-frames:v", "1", str(d_img)],
                       check=True, capture_output=True)
        _say(_frame_luma_hi(b_img) > 100 and _frame_luma_hi(d_img) < 40,
             f"luma_hi separates bright ({_frame_luma_hi(b_img):.0f}) from near-black "
             f"({_frame_luma_hi(d_img):.0f})")

    print(f"\n{PASS} passed · {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
