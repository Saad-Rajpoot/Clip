#!/usr/bin/env python3
"""RC5.1 STEP 8 — beat-audit contact sheet.

Extracts frames from the served MP4 (auto-sampled evenly + any explicit
timestamps) and tiles them into a single labelled montage PNG so the whole
video can be visually inspected in one image-read. Each thumb is labelled with
its timestamp and (when a manifest is supplied) the beat decision/verdict.

Usage:
  python tools/rc51_contact_sheet.py <mp4> <out.png> [--n 20] [--at 28,43,205]
                                      [--manifest ASSET_DECISION_MANIFEST.json]
"""
import argparse, json, subprocess, sys, tempfile, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FFMPEG = "/Users/hussnain/Desktop/vidrush-clone/.venv/lib/python3.9/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"


def _duration(mp4):
    # ffmpeg writes duration to stderr; parse it (no ffprobe in this env).
    out = subprocess.run([FFMPEG, "-i", str(mp4)], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def _grab(mp4, t, dst):
    subprocess.run([FFMPEG, "-y", "-ss", f"{t:.2f}", "-i", str(mp4),
                    "-frames:v", "1", "-q:v", "3", str(dst)],
                   capture_output=True)
    return dst.exists() and dst.stat().st_size > 0


def _label_for(t, manifest):
    if not manifest:
        return ""
    best = None
    for b in manifest:
        bt = b.get("t") or b.get("time_s") or b.get("start") or 0
        try:
            bt = float(bt)
        except Exception:
            continue
        if bt <= t + 0.01:
            if best is None or bt >= best[0]:
                best = (bt, b)
    if not best:
        return ""
    b = best[1]
    kind = b.get("kind") or b.get("source") or b.get("asset_kind") or ""
    verdict = b.get("verdict") or b.get("decision") or b.get("relevance") or ""
    txt = " ".join(x for x in (str(kind), str(verdict)) if x and x != "None")
    return txt[:42]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--at", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    mp4 = Path(args.mp4)
    dur = _duration(mp4)
    if dur <= 0:
        print("ERROR: could not read duration", file=sys.stderr)
        sys.exit(2)

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        try:
            data = json.loads(Path(args.manifest).read_text())
            manifest = data if isinstance(data, list) else (
                data.get("beats") or data.get("decisions") or data.get("assets"))
        except Exception:
            manifest = None

    # evenly-sampled timestamps (skip the very first/last 0.5s)
    times = []
    if args.n > 0:
        step = (dur - 1.0) / max(1, args.n - 1)
        times = [0.5 + i * step for i in range(args.n)]
    # explicit timestamps appended + sorted-unique
    for tok in args.at.split(","):
        tok = tok.strip()
        if tok:
            try:
                times.append(float(tok))
            except ValueError:
                pass
    times = sorted({round(min(max(t, 0.2), dur - 0.2), 2) for t in times})

    tw, th = 480, 270  # thumb size
    pad, labelh = 6, 26
    cols = args.cols
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad,
                              rows * (th + labelh + pad) + pad), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
    except Exception:
        font = ImageFont.load_default()

    tmp = Path(tempfile.mkdtemp())
    ok = 0
    for i, t in enumerate(times):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + labelh + pad)
        frame = tmp / f"f{i}.jpg"
        if _grab(mp4, t, frame):
            try:
                im = Image.open(frame).convert("RGB").resize((tw, th))
                sheet.paste(im, (x, y + labelh))
                ok += 1
            except Exception:
                draw.rectangle([x, y + labelh, x + tw, y + th + labelh], fill=(60, 30, 30))
        else:
            draw.rectangle([x, y + labelh, x + tw, y + th + labelh], fill=(60, 30, 30))
        mm, ss = divmod(int(t), 60)
        lab = f"{mm:01d}:{ss:02d}.{int((t%1)*10)}"
        extra = _label_for(t, manifest)
        if extra:
            lab += f"  [{extra}]"
        draw.text((x + 3, y + 4), lab, fill=(255, 214, 120), font=font)

    sheet.save(args.out, quality=88)
    print(f"contact sheet: {args.out}  ({ok}/{len(times)} frames, dur={dur:.1f}s, {cols}x{rows})")


if __name__ == "__main__":
    main()
