#!/usr/bin/env python3
"""Prepare a finished render for a frame-level audit: beat -> aired timecode -> real frames.

Two traps this exists to avoid, both burned earlier in this project:

  DO NOT QA FROM THE LEDGER. For one beat, ledger.jsonl named a source and window whose content is
  a different scene entirely from what is on screen at that timecode. Any audit run off ledger rows
  or clips/seg_NNN.mp4 grades images the viewer never sees.

  MAP BY CAPTION, AND LET THE CAPTION PROVE IT. A beat's narration appears burned into the frame,
  so the extracted frame carries its own evidence that the mapping is right. If the caption you see
  is not the narration you were handed, the mapping is wrong.

Emits <out>/frames/b<NNN>_<k>.jpg plus manifest.json, ready for a vision pass.

    python3 render_audit.py <job dir> <out dir> [--every 1] [--frames 3]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FF = ("/Users/hussnain/Library/Python/3.9/lib/python/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")


def srt_cues(p: Path):
    out = []
    for blk in re.split(r"\n\s*\n", p.read_text(encoding="utf-8", errors="ignore")):
        m = re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", blk)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = " ".join(l.strip() for l in blk.strip().splitlines()[2:] if l.strip())
        if body:
            out.append((a, b, body))
    return out


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1, help="take every Nth mapped beat")
    ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)

    vids = sorted(job.glob("output/final*.mp4"))
    vids = [v for v in vids if "FAILED" not in v.name] or vids
    if not vids:
        raise SystemExit(f"no rendered video in {job}/output")
    vid = vids[0]
    srt = vid.with_suffix(".srt")
    if not srt.exists():
        cands = sorted(job.glob("output/final*.srt"))
        srt = cands[0] if cands else None
    if not srt:
        raise SystemExit("no SRT beside the video — cannot map beats by caption")

    proj = json.loads((job / "project.json").read_text())
    segs = {s["index"]: s for s in proj["segments"]}
    srcs = {s["id"]: s for s in proj["sources"]}
    sel = {s["segment_index"]: s for s in proj.get("selections", [])}
    cues = srt_cues(srt)

    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows, used = [], set()
    for i in sorted(segs):
        body = norm(segs[i].get("text") or "")
        if len(body) < 12:
            continue
        for ci, (t0, t1, ctext) in enumerate(cues):
            if ci in used:
                continue
            c = norm(ctext)
            if len(c) > 12 and c[:40] in body:
                s = sel.get(i) or {}
                src = srcs.get(s.get("source_id") or "") or {}
                rows.append({
                    "beat": i, "t0": round(t0, 2), "t1": round(t1, 2),
                    "narration": (segs[i].get("text") or "").strip(),
                    "caption_seen": ctext.strip(),
                    "policy": segs[i].get("visual_policy") or "generic_filler",
                    "expected_visual": (segs[i].get("expected_visual") or "").strip()[:160],
                    "ledger_source_title": (src.get("title") or "")[:90],
                    "ledger_source_height": src.get("height"),
                    "is_image_still": bool(s.get("image_path")),
                })
                used.add(ci)
                break

    rows = rows[::max(1, a.every)]
    plan = []
    for r in rows:
        r["frames"] = []
        span = max(0.4, r["t1"] - r["t0"])
        for k in range(a.frames):
            fr = (k + 1) / (a.frames + 1)
            fp = out / "frames" / f"b{r['beat']:03d}_{k}.jpg"
            plan.append((f"{r['t0'] + span * fr:.3f}", str(fp)))
            r["frames"].append(str(fp))
    for t, fp in plan:
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", t,
                        "-i", str(vid), "-frames:v", "1", "-q:v", "3",
                        "-vf", "scale=1280:-1", fp], check=False)

    (out / "manifest.json").write_text(json.dumps(
        {"video": str(vid), "srt": str(srt), "beats_total": len(segs),
         "beats_mapped": len(rows), "rows": rows}, indent=1))
    print(f"{vid.name}: {len(segs)} beats, {len(rows)} mapped by caption, "
          f"{len(plan)} frames -> {out}")


if __name__ == "__main__":
    main()
