"""Unit test for footage._is_blank_bright_clip — the near-white / near-blank
stock-clip reject (symmetric twin of the black-frame guard).

Synthesizes short clips with controlled mean luma + spatial variance, encodes
each to a real mp4 with the bundled ffmpeg (yuv420p, same path the pipeline
uses), then asserts the helper rejects ONLY the bright + near-uniform ones and
keeps every bright-but-detailed / mid / dark clip.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vidlore.ffmpeg_tool import ffmpeg_exe
from vidlore.footage import _is_blank_bright_clip

W, H = 1280, 720


def _encode(arr: np.ndarray, dest: Path) -> None:
    """Encode a single RGB frame as a 5s, 25fps yuv420p mp4 (looped still),
    exactly the kind of trimmed stock beat the selector hands downstream."""
    png = dest.with_suffix(".png")
    Image.fromarray(arr.astype("uint8"), "RGB").save(png)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
         "-loop", "1", "-i", str(png), "-t", "5", "-r", "25",
         "-pix_fmt", "yuv420p", str(dest)],
        check=True, timeout=60)
    png.unlink(missing_ok=True)


def _measure(path: Path, t: float = 2.5):
    """Re-read one frame the way the helper does (128x72, BT.601 luma) so we
    can print the actual post-encode mean/std the gate sees."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        fp = tf.name
    try:
        subprocess.run(
            [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
             "-q:v", "3", fp], check=True, timeout=15)
        a = np.asarray(Image.open(fp).convert("RGB").resize((128, 72),
                       Image.BILINEAR), dtype="float32")
        L = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return float(L.mean()), float(L.std())
    finally:
        Path(fp).unlink(missing_ok=True)


def _gradient(lo: int, hi: int) -> np.ndarray:
    """Smooth vertical gray gradient lo->hi — a blown/empty 'pale gradient'."""
    col = np.linspace(lo, hi, H, dtype="float32")[:, None]
    g = np.repeat(col, W, axis=1)
    return np.stack([g, g, g], axis=-1)


def _flat(v: int, noise: float = 2.0) -> np.ndarray:
    base = np.full((H, W, 3), v, dtype="float32")
    return base + np.random.normal(0, noise, (H, W, 3))


def _bright_detailed() -> np.ndarray:
    """Bright field (snow / bright lab) WITH a real dark subject + texture:
    high mean luma but high spatial variance -> must be KEPT."""
    a = np.full((H, W, 3), 205, dtype="float32")
    a += np.random.normal(0, 18, (H, W, 3))          # texture / grain
    a[180:560, 430:850] = 45                          # dark subject blob
    a[300:360, 200:1080] = 70                         # a dark horizon band
    return a


def _mid_documentary() -> np.ndarray:
    a = np.full((H, W, 3), 95, dtype="float32")
    a += np.random.normal(0, 40, (H, W, 3))
    a[:, :, 0] += 20                                   # warm cast
    return a


def _bright_banded() -> np.ndarray:
    """Bright overall (~176 mean) but with LARGE-SCALE structure — a darker
    horizon band. Structural contrast (unlike fine pixel noise) survives the
    128x72 downscale, so std stays above the floor and the clip is KEPT.
    This is the realistic 'bright but detailed' boundary the gate must respect.
    """
    a = np.full((H, W, 3), 200, dtype="float32")
    a[300:520, :] = 120                                # wide darker band ~30%
    a += np.random.normal(0, 6, (H, W, 3))
    return a


CASES = [
    # (name, frame, expect_reject)
    ("near_white_gradient_165_210", _gradient(165, 210), True),
    ("near_white_gradient_178_205", _gradient(178, 205), True),
    ("flat_white_200",              _flat(200, 2.0),     True),
    ("flat_white_185",              _flat(185, 3.0),     True),
    ("bright_detailed_snow/lab",    _bright_detailed(),  False),
    ("mid_documentary_95",          _mid_documentary(),  False),
    ("dark_clip_6",                 _flat(6, 2.0),       False),
    ("uniform_midgray_120",         _flat(120, 4.0),     False),  # flat but NOT bright
    ("bright_lowvar_borderline",    _flat(176, 6.0),     True),   # both gates trip
    ("bright_structured_banded",    _bright_banded(),    False),  # structure saves it
]

np.random.seed(7)
failures = 0
with tempfile.TemporaryDirectory() as d:
    dd = Path(d)
    print(f"{'case':32s} {'mean':>7s} {'std':>7s} {'reject':>7s} "
          f"{'expect':>7s}  result")
    print("-" * 78)
    for name, frame, expect in CASES:
        clip = dd / (name.replace("/", "_") + ".mp4")
        _encode(np.clip(frame, 0, 255), clip)
        mean, std = _measure(clip)
        got = _is_blank_bright_clip(clip)
        ok = (got == expect)
        failures += (not ok)
        print(f"{name:32s} {mean:7.1f} {std:7.1f} {str(got):>7s} "
              f"{str(expect):>7s}  {'PASS' if ok else 'FAIL  <<<'}")

print("-" * 78)
print("ALL PASS" if failures == 0 else f"{failures} FAILURE(S)")
raise SystemExit(1 if failures else 0)
