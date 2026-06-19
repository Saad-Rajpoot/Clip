"""P0 — full-video differentiation comparison.

After all three channel renders finish, this:
  1. Runs the forensic analyzer on each final MP4.
  2. Extracts 9 evenly-spaced frame stills from each video.
  3. Builds two contact sheets:
     a. Same-timecode 3-up rows so you can see what each channel
        chose to show at each moment of the same script.
     b. Side-by-side metric table for the brutal-honesty verdict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent

CHANNELS = [
    ("midnight_pacific", "output/lf_midnight"),
    ("atlas_explained",  "output/lf_atlas"),
    ("amber_chronicles", "output/lf_amber"),
]

OUT = ROOT / "output" / "lf_compare"
OUT.mkdir(parents=True, exist_ok=True)


def _find_mp4(rundir: Path) -> Optional[Path]:
    if not rundir.exists():
        return None
    # Skip work_*/ intermediates and cache/ — pick the final TOP-LEVEL
    # MP4 with audio/music baked in (sits at <run>/<slug>/<slug>.mp4).
    def _bad(p):
        return any(part.startswith("work_") or part == "cache"
                   for part in p.parts)
    mp4s = [p for p in rundir.rglob("*.mp4") if not _bad(p)]
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: p.stat().st_size)


def _ffprobe_duration(mp4: Path) -> float:
    out = subprocess.run(
        ["/Users/hussnain/pinokio/bin/miniconda/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", str(mp4)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _extract_frames(mp4: Path, dest_dir: Path, n: int = 9) -> list[Path]:
    """Extract n evenly-spaced frames from the video, skipping the
    first 1.0s and the last 1.0s (avoid title cards / fade-outs)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dur = max(2.5, _ffprobe_duration(mp4))
    paths: list[Path] = []
    for i in range(n):
        t = 1.0 + (i + 0.5) * (dur - 2.0) / n
        out = dest_dir / f"frame_{i:02d}_t{int(t * 100):05d}.jpg"
        subprocess.run(
            ["/Users/hussnain/pinokio/bin/miniconda/bin/ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(mp4),
             "-frames:v", "1", "-q:v", "3", str(out)],
            capture_output=True)
        if out.exists():
            paths.append(out)
    return paths


def _run_forensics(mp4: Path, channel: str) -> dict:
    """Use the project's built-in forensic tool to dump metrics."""
    out_json = OUT / f"{channel}_forensic.json"
    try:
        subprocess.run(
            [sys.executable, "-m", "vidlore.forensics", str(mp4),
             "--out", str(out_json)],
            cwd=str(ROOT), check=False, capture_output=True, timeout=240)
        if out_json.exists():
            return json.loads(out_json.read_text())
    except Exception as exc:                              # noqa: BLE001
        return {"_error": str(exc)}
    return {}


def _build_contact_sheet(frames_by_channel: dict[str, list[Path]],
                         out: Path) -> None:
    """3 columns (channels) × 9 rows (timecodes).  Same row index =
    same timecode in the source — direct A/B/C comparison."""
    from PIL import Image, ImageDraw, ImageFont
    THUMB_W = 480
    THUMB_H = int(THUMB_W * 9 / 16)
    GAP = 10
    LABEL_H = 36
    rows = 9
    cols = len(frames_by_channel)
    sheet_w = LABEL_H + GAP + cols * (THUMB_W + GAP) + GAP
    sheet_h = LABEL_H + GAP + rows * (THUMB_H + GAP) + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    # column headers
    for i, name in enumerate(frames_by_channel.keys()):
        x = LABEL_H + GAP + i * (THUMB_W + GAP)
        d.text((x + 6, 8), name, fill=(230, 230, 230), font=font)
    # row labels + thumbnails
    chans = list(frames_by_channel.keys())
    for r in range(rows):
        y = LABEL_H + GAP + r * (THUMB_H + GAP)
        d.text((8, y + THUMB_H // 2 - 8),
               f"t{r+1}", fill=(180, 180, 180), font=font)
        for i, ch in enumerate(chans):
            x = LABEL_H + GAP + i * (THUMB_W + GAP)
            paths = frames_by_channel[ch]
            if r < len(paths) and paths[r].exists():
                im = Image.open(paths[r]).convert("RGB").resize(
                    (THUMB_W, THUMB_H), Image.LANCZOS)
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + THUMB_W, y + THUMB_H],
                             fill=(50, 30, 30))
    sheet.save(out)


def main():
    frames_by = {}
    forensic_by = {}
    for ch, rundir in CHANNELS:
        rp = ROOT / rundir
        mp4 = _find_mp4(rp)
        if not mp4:
            print(f"[{ch}] no mp4 found in {rp}")
            continue
        print(f"[{ch}] mp4 = {mp4.relative_to(ROOT)}  "
              f"({mp4.stat().st_size / 1e6:.1f} MB, "
              f"{_ffprobe_duration(mp4):.1f}s)")
        # frames
        sub = OUT / ch
        frames = _extract_frames(mp4, sub, n=9)
        frames_by[ch] = frames
        # forensic
        forensic_by[ch] = _run_forensics(mp4, ch)

    sheet = OUT / "contact_sheet_full_video.png"
    _build_contact_sheet(frames_by, sheet)
    print(f"\nCONTACT SHEET -> {sheet}")

    # ---- print metric comparison table -------------------------------
    print("\n" + "=" * 90)
    print("FORENSIC COMPARISON")
    print("=" * 90)
    if forensic_by:
        keys = sorted({k for v in forensic_by.values() for k in v
                       if not k.startswith("_") and not isinstance(v[k], dict)})
        # narrow to the metrics that matter for editorial DNA
        focus = [
            "cuts_per_min", "median_shot_seconds", "p25_shot_seconds",
            "median_loudness_db", "music_coverage_pct",
            "silence_count_over_0_8s", "mean_brightness",
            "mean_saturation", "text_density_pct",
            "average_edge_count", "dominant_hue_h",
        ]
        keys = [k for k in focus if any(k in v for v in forensic_by.values())]
        if not keys:
            keys = sorted({k for v in forensic_by.values() for k in v})[:14]
        hdr = "metric".ljust(36)
        for ch, _ in CHANNELS:
            hdr += ch[:14].rjust(15)
        print(hdr)
        print("-" * 90)
        for k in keys:
            row = k.ljust(36)
            for ch, _ in CHANNELS:
                v = forensic_by.get(ch, {}).get(k, "—")
                if isinstance(v, float):
                    row += f"{v:>15.2f}"
                else:
                    row += str(v).rjust(15)
            print(row)
    # dump aggregate JSON
    (OUT / "all_forensics.json").write_text(
        json.dumps(forensic_by, indent=2))


if __name__ == "__main__":
    main()
