#!/usr/bin/env python3
"""compare_looks.py — side-by-side editorial-recipe + look diff for 2+ renders.

Given N run directories (each containing a final .mp4 + render_meta.json),
print a comparison table of the per-video editorial RECIPE axes (accent /
map / beat / grade / hold / music) + measured shot-length + editor
signature, and tile a few evenly-spaced frames from each video into one
stacked proof PNG so the look difference is visible at a glance.

Local-only (ffmpeg + PIL). No renders, no network. Usage:
    python tools/compare_looks.py <run_dir_a> <run_dir_b> [more...] [--out proof.png]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
except Exception:                                                  # noqa: BLE001
    FFMPEG = "ffmpeg"


def _find(run: Path) -> tuple[Path | None, dict]:
    mp4 = next((p for p in sorted(run.glob("*.mp4"))), None)
    meta = {}
    mp = run / "render_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            meta = {}
    return mp4, meta


def _frames(mp4: Path, n: int, w: int = 480) -> list[Path]:
    """Extract n evenly-spaced frames; return their paths."""
    out: list[Path] = []
    # probe duration
    try:
        r = subprocess.run([FFMPEG, "-i", str(mp4)], capture_output=True,
                           text=True)
        import re
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
        dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
               + float(m.group(3))) if m else 60.0
    except Exception:                                              # noqa: BLE001
        dur = 60.0
    tmp = mp4.parent / "_cmp_frames"
    tmp.mkdir(exist_ok=True)
    for i in range(n):
        t = dur * (i + 0.5) / n
        fp = tmp / f"{mp4.stem}_{i}.png"
        subprocess.run([FFMPEG, "-y", "-ss", f"{t:.2f}", "-i", str(mp4),
                        "-frames:v", "1", "-vf", f"scale={w}:-1", str(fp)],
                       capture_output=True)
        if fp.exists():
            out.append(fp)
    return out


def _row(label: str, rec: dict, meta: dict) -> str:
    sig = meta.get("editor_signature", {})
    sl = meta.get("shot_len_s", {})
    acc = rec.get("accent")
    return (f"{label:<26} niche={str(rec.get('niche')):<11} "
            f"preset={str(sig.get('look')):<16} "
            f"accent={str(acc):<17} map={str(rec.get('map_style')):<10} "
            f"beat={str(rec.get('beat_target')):<6} "
            f"grade={str(rec.get('grade_sat')):<6} "
            f"hold={str(rec.get('hold_mult')):<6} "
            f"music={str(rec.get('music_bed')):<6} "
            f"med_shot={sl.get('median')}")


def main(argv: list[str]) -> int:
    out_png = "look_compare.png"
    dirs = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_png = argv[i + 1]; i += 2; continue
        dirs.append(Path(argv[i])); i += 1
    if len(dirs) < 2:
        print("need >=2 run dirs"); return 2

    print("\n=== EDITORIAL RECIPE / LOOK COMPARISON ===")
    strips = []
    for d in dirs:
        mp4, meta = _find(d)
        rec = meta.get("editorial_recipe", {}) or {}
        print(_row(d.name[:26], rec, meta))
        if meta.get("editorial_recipe_summary"):
            print(f"    summary: {meta['editorial_recipe_summary']}")
        if mp4:
            strips.append((d.name, _frames(mp4, 4)))
        else:
            print(f"    (no .mp4 in {d})")

    # tile into one stacked proof PNG (one row per render)
    try:
        from PIL import Image, ImageDraw
        rows = [s for s in strips if s[1]]
        if rows:
            cell_w = 480
            imgs = [[Image.open(p) for p in fr] for _, fr in rows]
            cell_h = max(im.height for row in imgs for im in row)
            grid_w = cell_w * max(len(r) for r in imgs)
            grid_h = (cell_h + 22) * len(rows)
            canvas = Image.new("RGB", (grid_w, grid_h), (12, 14, 20))
            dr = ImageDraw.Draw(canvas)
            y = 0
            for (name, _), row in zip(rows, imgs):
                dr.text((6, y + 4), name[:60], fill=(220, 220, 230))
                x = 0
                for im in row:
                    canvas.paste(im, (x, y + 22))
                    x += cell_w
                y += cell_h + 22
            canvas.save(out_png)
            print(f"\nproof PNG: {out_png}  ({grid_w}x{grid_h})")
    except Exception as e:                                         # noqa: BLE001
        print(f"(proof PNG skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
