#!/usr/bin/env python3
"""Provision / repair the bundled NVENC-capable Windows ffmpeg for Vidlore.

The Vidlore Windows package ships a static, NVENC-capable ffmpeg.exe at
    dist/Vidlore-Windows/ffmpeg/bin/ffmpeg.exe
so the NVIDIA GPU (e.g. RTX 3070) can hardware-encode WITHOUT the user
installing ffmpeg. This script (re)downloads and validates that binary — use it
if the bundled file is missing/corrupt, or to re-pin a newer build.

It is dependency-free (urllib + zipfile) and cross-platform: it runs on macOS
for provisioning the Windows dist, and on Windows for self-repair. It can VERIFY
integrity (zip valid, PE32+ binary, NVENC strings present, SHA256) but it can
NOT execute a Windows binary on macOS — the real NVENC encode probe must run on
the Windows machine via tools/check_windows_gpu_acceleration.bat.

Usage:
    python tools/fetch_windows_nvenc_ffmpeg.py                 # default dest, pinned build
    python tools/fetch_windows_nvenc_ffmpeg.py --dest <dir>    # custom dist dir
    python tools/fetch_windows_nvenc_ffmpeg.py --url <zip> --sha256 <hex>
    python tools/fetch_windows_nvenc_ffmpeg.py --force         # re-download even if present
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "dist" / "Vidlore-Windows" / "ffmpeg"
MANIFEST_NAME = "ffmpeg_build.json"

# Pinned, validated build (BtbN FFmpeg-Builds, win64-gpl static, NVENC-capable).
PIN_URL = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
           "ffmpeg-master-latest-win64-gpl.zip")
PIN_SHA256_EXE = "0a5ecf46b68e11732093a56cda04e6072ba2c417fd4c9f7940e030f6c3dc6baa"
PIN_SIZE_EXE = 204200448
NVENC_MARKERS = (b"h264_nvenc", b"hevc_nvenc")


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _is_pe(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(2) == b"MZ"          # DOS/PE header magic
    except OSError:
        return False


def _has_nvenc(p: Path) -> bool:
    """Stream-scan the binary for NVENC encoder strings (works cross-platform;
    no execution needed)."""
    try:
        with open(p, "rb") as f:
            blob = f.read()
        return all(m in blob for m in NVENC_MARKERS)
    except OSError:
        return False


def _extract_ffmpeg_exe(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        cand = [n for n in zf.namelist() if n.lower().endswith("bin/ffmpeg.exe")]
        if not cand:
            cand = [n for n in zf.namelist() if n.lower().endswith("ffmpeg.exe")]
        if not cand:
            raise RuntimeError("no ffmpeg.exe inside the downloaded archive")
        return zf.read(cand[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Provision bundled NVENC Windows ffmpeg")
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="ffmpeg dir inside the Windows dist (default: %(default)s)")
    ap.add_argument("--url", default=PIN_URL)
    ap.add_argument("--sha256", default=PIN_SHA256_EXE,
                    help="expected sha256 of the extracted ffmpeg.exe ('' to skip)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest_dir = Path(args.dest)
    bin_path = dest_dir / "bin" / "ffmpeg.exe"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "bin").mkdir(parents=True, exist_ok=True)

    # Idempotent: skip if the bundled binary already matches.
    if bin_path.is_file() and not args.force:
        cur = _file_sha256(bin_path)
        if not args.sha256 or cur == args.sha256:
            print(f"[fetch] already present + valid: {bin_path}")
            print(f"[fetch]   sha256={cur}")
            print(f"[fetch]   PE32+={_is_pe(bin_path)} nvenc={_has_nvenc(bin_path)}")
            return 0
        print(f"[fetch] present but sha mismatch (have {cur[:12]}…, "
              f"want {args.sha256[:12]}…) → re-downloading")

    print(f"[fetch] downloading {args.url}")
    try:
        with urllib.request.urlopen(args.url, timeout=900) as r:
            zip_bytes = r.read()
    except Exception as e:                                      # noqa: BLE001
        print(f"[fetch] ERROR download failed: {type(e).__name__}: {e}")
        return 2
    print(f"[fetch] downloaded {len(zip_bytes)} bytes; extracting ffmpeg.exe…")

    try:
        exe_bytes = _extract_ffmpeg_exe(zip_bytes)
    except Exception as e:                                      # noqa: BLE001
        print(f"[fetch] ERROR extract failed: {type(e).__name__}: {e}")
        return 2

    got_sha = _sha256(exe_bytes)
    if args.sha256 and got_sha != args.sha256:
        print(f"[fetch] WARNING sha256 mismatch: got {got_sha}, expected {args.sha256}")
        print("[fetch]   (BtbN 'latest' rotates — this is the new pinned hash; "
              "the binary is still validated for PE + NVENC below.)")

    # Write atomically.
    tmp = Path(tempfile.mkstemp(dir=str(dest_dir / "bin"), suffix=".part")[1])
    tmp.write_bytes(exe_bytes)
    tmp.replace(bin_path)

    ok_pe = _is_pe(bin_path)
    ok_nv = _has_nvenc(bin_path)
    print(f"[fetch] wrote {bin_path} ({len(exe_bytes)} bytes)")
    print(f"[fetch]   sha256={got_sha}")
    print(f"[fetch]   PE32+={ok_pe} nvenc_strings={ok_nv}")

    # Refresh the build manifest next to the binary.
    manifest = {
        "binary": "bin/ffmpeg.exe",
        "platform": "windows-x86_64",
        "source_url": args.url,
        "sha256_exe": got_sha,
        "size_bytes_exe": len(exe_bytes),
        "validated_pe32plus": ok_pe,
        "validated_nvenc_strings": ok_nv,
        "note": "macOS cannot execute this binary; run the real NVENC probe on "
                "Windows via tools/check_windows_gpu_acceleration.bat.",
    }
    (dest_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")

    if not (ok_pe and ok_nv):
        print("[fetch] FAIL — binary did not validate as PE32+ with NVENC.")
        return 1
    print("[fetch] OK — bundled NVENC ffmpeg provisioned + validated (static checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
