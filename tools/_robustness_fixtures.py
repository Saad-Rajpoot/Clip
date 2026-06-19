#!/usr/bin/env python3
"""V3.2.2 STEP 1 — durable corrupt-clip regression fixtures.

Generates 8 local MP4/codec fixtures that reproduce the real-world failure modes
(a clip that downloads + passes a lightweight check but later fails on FFmpeg seek
or assembly). NOT dependent on any live CDN. Idempotent: re-run to regenerate.

  python tools/_robustness_fixtures.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore.ffmpeg_tool import ffmpeg_exe                      # noqa: E402

FF = ffmpeg_exe()
OUT = ROOT / "research/motion_graphics_expansion/render_robustness/fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def _mk_base(path: Path, secs=5, with_audio=True, codec="libx264", container_args=None):
    """A valid, normal stock-like clip (moov at end by default)."""
    cmd = [FF, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={secs}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}"]
    cmd += ["-c:v", codec, "-pix_fmt", "yuv420p", "-g", "50"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += (container_args or []) + [str(path)]
    subprocess.run(cmd, capture_output=True)
    return path.exists()


def _corrupt_bytes(src: Path, dst: Path, start_frac: float, span: int = 4096):
    """Copy src→dst, then overwrite `span` bytes starting at start_frac of the
    file with 0xFF (simulates a corrupt chunk that a from-start decode may pass
    but a seek into that region fails)."""
    data = bytearray(src.read_bytes())
    n = len(data)
    s = max(0, min(n - span, int(n * start_frac)))
    for i in range(s, min(n, s + span)):
        data[i] = 0xFF
    dst.write_bytes(bytes(data))


def main():
    res = {}
    base = OUT / "valid_normal.mp4"
    res["valid_normal"] = _mk_base(base)                                  # (8) valid
    # (1) corrupt header — clobber the first 2KB (ftyp/moov head if faststart)
    faststart = OUT / "_faststart.mp4"
    _mk_base(faststart, container_args=["-movflags", "+faststart"])
    hdr = OUT / "corrupt_header.mp4"
    d = bytearray(faststart.read_bytes())
    for i in range(0, min(2048, len(d))):
        d[i] = 0x00
    hdr.write_bytes(bytes(d)); res["corrupt_header"] = hdr.exists()
    # (2) truncated MP4 — keep first 55% of a +faststart clip (moov present at
    #     front, mdat cut short → end seek fails)
    trunc = OUT / "truncated.mp4"
    fd = faststart.read_bytes()
    trunc.write_bytes(fd[:int(len(fd) * 0.55)]); res["truncated"] = trunc.exists()
    # (3) mid-seek-fail — valid moov, corrupt a chunk at ~50%
    mid = OUT / "midseek_fail.mp4"
    _corrupt_bytes(faststart, mid, 0.50, span=8192); res["midseek_fail"] = mid.exists()
    # (4) end-seek-fail — valid moov, corrupt a chunk at ~88%
    end = OUT / "endseek_fail.mp4"
    _corrupt_bytes(faststart, end, 0.88, span=8192); res["endseek_fail"] = end.exists()
    # (4b) STSC/STCO seek-table corruption — the EXACT archive.org failure:
    # decodes from the start but a seek can't resolve a chunk offset. Corrupt the
    # `stco` (chunk-offset) entries inside the moov.
    stco = OUT / "stco_corrupt.mp4"
    sd = bytearray(faststart.read_bytes())
    si = sd.find(b"stco")
    if si >= 0:
        s = si + 12                                            # past atom hdr+version+count
        for j in range(s, min(len(sd), s + 64)):
            sd[j] = 0xFF
        stco.write_bytes(bytes(sd))
    res["stco_corrupt"] = stco.exists()
    # (5) zero-duration — a single still frame muxed to ~0s
    zero = OUT / "zero_duration.mp4"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=320x240:d=0.001:r=25", "-frames:v", "1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(zero)], capture_output=True)
    res["zero_duration"] = zero.exists()
    # (6) no-audio (valid video) — should be ALLOWED for footage (video-only)
    res["no_audio"] = _mk_base(OUT / "no_audio.mp4", with_audio=False)
    # (7) unusual-but-valid codec/container — mpeg4 in .mkv
    res["unusual_valid"] = _mk_base(OUT / "unusual_valid.mkv", codec="mpeg4",
                                    with_audio=False)
    faststart.unlink(missing_ok=True)
    print("fixtures written ->", OUT)
    for k, v in res.items():
        print(f"  {k:16s} {'ok' if v else 'FAILED'}")
    return res


if __name__ == "__main__":
    main()
