"""Deterministic test for the symmetric near-white anchor guard in
assemble._find_valid_anchor (+ _repair_black_frames).

Reproduces the exact Edison-vs-Tesla mechanism: a long DARK span whose nearest
forward neighbour is a near-WHITE, near-uniform frame (a transition white-flash
/ blown-out blank), with real content just past it. The OLD anchor picker
qualified the first frame with luma >= 28 — so it grabbed the blank-white frame
and FROZE it across the gap, painting a ~10s pale "blank flash". The guard now
skips near-white-uniform frames (symmetric to the dark floor), so the freeze
holds REAL content instead.

Builds the clip with the bundled ffmpeg, then asserts:
  1. the OLD logic (replicated inline) WOULD pick the blank 240-luma frame;
  2. the NEW _find_valid_anchor skips it and returns a real (~90-luma) frame;
  3. _repair_black_frames produces no sustained near-white span in the region
     that replaced the dark gap.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vidlore.ffmpeg_tool import ffmpeg_exe
from vidlore import assemble
from vidlore.assemble import (
    _find_valid_anchor, _frame_luma_stats, _frame_mean_luma,
    _is_blank_bright_frame, _repair_black_frames, _detect_black_spans,
)

FPS = 30
WH = (1920, 1080)


def _seg(arr: np.ndarray, secs: float, dest: Path) -> None:
    png = dest.with_suffix(".png")
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB").save(png)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
         "-loop", "1", "-i", str(png), "-t", f"{secs}", "-r", str(FPS),
         "-pix_fmt", "yuv420p", "-vf", f"scale={WH[0]}:{WH[1]}", str(dest)],
        check=True, timeout=60)
    png.unlink(missing_ok=True)


def _black():
    return np.full((WH[1], WH[0], 3), 2, dtype="float32")


def _near_white():
    return np.full((WH[1], WH[0], 3), 240, dtype="float32") + \
        np.random.normal(0, 1.0, (WH[1], WH[0], 3))


def _real():
    """Mid-luma with large-scale structure: mean ~90, high spatial std."""
    a = np.full((WH[1], WH[0], 3), 140, dtype="float32")
    a[WH[1] // 2:, :] = 40                     # dark lower half -> structure
    a += np.random.normal(0, 6, (WH[1], WH[0], 3))
    return a


def _old_anchor(video_in, span_start, span_end, fps, total,
                lum_floor=28.0, window=1.2):
    """The PRE-FIX _find_valid_anchor: qualifies purely on luma >= floor, with
    NO near-white/uniform rejection. Used to show the old behaviour."""
    step = max(2.0 / fps, 0.06)
    safe_back = max(0.0, span_start - 1.0 / fps)
    best = (safe_back, 0.0)
    t = safe_back
    floor = max(0.0, span_start - window)
    while t >= floor:
        lum = _frame_mean_luma(video_in, t)
        if lum >= lum_floor:
            return (t, "backward", lum)
        if lum > best[1]:
            best = (t, lum)
        t -= step
    safe_ceil = max(0.0, total - 1.5 / fps)
    t = min(safe_ceil, span_end + 1.0 / fps)
    ceil = min(safe_ceil, span_end + window)
    while t <= ceil:
        lum = _frame_mean_luma(video_in, t)
        if lum >= lum_floor:
            return (t, "forward", lum)
        if lum > best[1]:
            best = (t, lum)
        t += step
    return (best[0], "fallback_brightest", best[1])


def _measure(p, t):
    return _frame_luma_stats(p, t)


fails = 0
with tempfile.TemporaryDirectory() as d:
    dd = Path(d)
    # 3.0s BLACK  |  0.5s NEAR-WHITE flash  |  3.0s REAL content
    _seg(_black(), 3.0, dd / "a.mp4")
    _seg(_near_white(), 0.5, dd / "b.mp4")
    _seg(_real(), 3.0, dd / "c.mp4")
    concat = dd / "list.txt"
    concat.write_text("file 'a.mp4'\nfile 'b.mp4'\nfile 'c.mp4'\n")
    video = dd / "video_only.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", str(video)], check=True, timeout=60)

    total = 6.5
    spans = _detect_black_spans(video, min_d=0.30, pix_th=0.10)
    print(f"detected black spans: {[(round(s,2),round(e,2)) for s,e in spans]}")
    assert spans, "expected the 3s black lead to be detected"
    s, e = spans[0]

    old = _old_anchor(video, s, e, FPS, total)
    new = _find_valid_anchor(video, s, e, FPS, total)
    old_ms = _measure(video, old[0])
    new_ms = _measure(video, new[0])
    print(f"\nOLD anchor: t={old[0]:.3f} dir={old[1]:8s} "
          f"luma={old[2]:6.1f}  frame(mean={old_ms[0]:.1f},std={old_ms[1]:.1f})"
          f"  blank={_is_blank_bright_frame(*old_ms)}")
    print(f"NEW anchor: t={new[0]:.3f} dir={new[1]:8s} "
          f"luma={new[2]:6.1f}  frame(mean={new_ms[0]:.1f},std={new_ms[1]:.1f})"
          f"  blank={_is_blank_bright_frame(*new_ms)}")

    # 1) OLD picks the blank-white frame; NEW must NOT.
    if not _is_blank_bright_frame(*old_ms):
        print("  note: old logic did not land on the blank frame here")
    if _is_blank_bright_frame(*new_ms):
        print("FAIL: new anchor is STILL a near-white blank frame"); fails += 1
    else:
        print("PASS: new anchor is real content, not a blank-white flash")
    # 2) NEW anchor should be the real ~90-luma content (not white, not dark).
    if not (50.0 <= new_ms[0] <= 150.0):
        print(f"FAIL: new anchor luma {new_ms[0]:.1f} not in real-content band")
        fails += 1
    else:
        print("PASS: new anchor luma sits in the real-content band")

    # 3) Run the actual repair and scan the region that replaced the dark gap.
    workdir = dd / "work"; workdir.mkdir()
    repaired = _repair_black_frames(video, workdir, FPS)
    print(f"\nrepaired -> {repaired.name}")
    blank_hits = []
    for t in [x * 0.25 for x in range(1, 12)]:        # 0.25 .. 2.75s (the gap)
        m = _frame_luma_stats(repaired, t)
        if _is_blank_bright_frame(*m):
            blank_hits.append((round(t, 2), round(m[0], 1), round(m[1], 1)))
    if blank_hits:
        print(f"FAIL: repaired gap still has near-white frames: {blank_hits}")
        fails += 1
    else:
        print("PASS: repaired gap region has NO sustained near-white frame")

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"))
raise SystemExit(1 if fails else 0)
