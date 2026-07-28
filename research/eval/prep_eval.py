#!/usr/bin/env python3
"""Ground-truth relevance eval WITHOUT a render.

Three renders shipped on proxy metrics (did we pick a plausible source? did moment_bonus fire? did
repeats drop?) and the frame-level audit then measured 5.18/10 — the proxies never measured the
thing that matters, which is whether the moment the voiceover describes is actually on screen.

This builds the same evidence the audit used, straight from each beat's SELECTED WINDOW in the
source file, so a fix can be scored in minutes instead of a 1.7-hour render:

    python3 prep_eval.py --job <job dir> --out <dir> [--beats 0-40]

Emits <out>/frames/beat_<i>_<n>.jpg + <out>/eval_slices/slice_NN.json, ready for the vision pass.
The window is sampled at 15%/50%/85% of its span, which is what the renderer actually shows after
its trim — the extremes are often transition frames.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

FF = ("/Users/hussnain/Library/Python/3.9/lib/python/site-packages/"
      "imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
FRACS = (0.15, 0.50, 0.85)


def crop_filters(job: Path) -> dict:
    """{source_id: ffmpeg crop filter} — mirrors the punch-in crop build applies to watermarked
    sources, so the eval judges what the VIEWER sees.

    Without this the eval reads raw source frames and reports a channel logo on every beat from a
    source build already crops away. Measured: 12 of 84 sources were cropped in job 69d80e9dd4_v4,
    and the source-window eval reported 18 watermark beats against the rendered audit's 8."""
    try:
        sys.path.insert(0, "/Users/hussnain/Desktop/vidlore-clipstudio/.clipstudio_libs")
        sys.path.insert(0, "/Users/hussnain/Desktop/vidlore-clipstudio/.claude/worktrees/"
                            "clipstudio-handover-review-113723")
        from vidlore.clipstudio.models import ClipProject
        from vidlore.clipstudio.build import _watermark_crop_filter
        from vidlore.clipstudio import index as _I
        from vidlore.clipstudio.match import _source_corner_logo
        proj = ClipProject.load(job)
        out = {}
        for s in proj.sources:
            if getattr(s, "status", "") != "ok":
                continue
            try:
                corner = _source_corner_logo(_I.load_shots(proj, s.id))
            except Exception:
                corner = ""
            if corner:
                out[s.id] = _watermark_crop_filter(corner)
        return out
    except Exception as e:                                     # noqa: BLE001
        print(f"warn: crop mirror unavailable ({str(e)[:60]}) — frames will be UNCROPPED",
              file=sys.stderr)
        return {}


def parse_range(s):
    if not s:
        return None
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beats", default="")
    ap.add_argument("--per-slice", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    job, out = Path(a.job), Path(a.out)
    frames = out / "frames"
    slices = out / "eval_slices"
    for d in (frames, slices):
        d.mkdir(parents=True, exist_ok=True)

    proj = json.loads((job / "project.json").read_text())
    segs = {s["index"]: s for s in proj["segments"]}
    srcs = {s["id"]: s for s in proj["sources"]}
    rng = parse_range(a.beats)

    crops = crop_filters(job)
    if crops:
        print(f"mirroring build's watermark crop on {len(crops)} source(s)")

    rows, plan = [], []
    for sel in proj.get("selections", []):
        i = sel.get("segment_index")
        if i is None or (rng and not (rng[0] <= i <= rng[1])):
            continue
        seg = segs.get(i) or {}
        sid = sel.get("source_id") or ""
        src = srcs.get(sid) or {}
        path = src.get("local_path") or ""
        i0, i1 = float(sel.get("in_point", 0)), float(sel.get("out_point", 0))
        shots = []
        if path and os.path.exists(path) and i1 > i0:
            for n, f in enumerate(FRACS):
                fp = frames / f"beat_{i:03d}_{n}.jpg"
                plan.append((f"{i0 + (i1 - i0) * f:.3f}", path, str(fp),
                             crops.get(sid) or "-"))
                shots.append(str(fp))
        rows.append({
            "beat": i,
            "narration": (seg.get("text") or "").strip(),
            "quote": (seg.get("quote") or "").strip(),
            "expected_visual": (seg.get("expected_visual") or "").strip(),
            "visual_policy": seg.get("visual_policy") or "",
            "required_entity": seg.get("required_entity") or "",
            "source_id": sid,
            "source_title": src.get("title") or "",
            "window": [round(i0, 2), round(i1, 2)],
            "confidence": sel.get("confidence"),
            "frames": shots,
            "no_footage": not shots,
        })

    # extract (parallel; ffmpeg -ss before -i is the fast seek)
    script = out / "_extract.sh"
    script.write_text(
        "#!/bin/sh\n"
        'VF="scale=1280:-1"\n'
        '[ "$4" != "-" ] && VF="$4,scale=1280:-1"\n'
        f'"{FF}" -y -hide_banner -loglevel error -ss "$1" -i "$2" '
        '-frames:v 1 -q:v 3 -vf "$VF" "$3"\n')
    script.chmod(0o755)
    if plan:
        p = subprocess.run(["xargs", "-P", str(a.jobs), "-n", "4", str(script)],
                           input="\n".join("\t".join(x) for x in plan),
                           text=True, capture_output=True)
        if p.returncode:
            print(p.stderr[-800:], file=sys.stderr)

    got = sum(1 for r in rows for f in r["frames"] if os.path.exists(f))
    rows.sort(key=lambda r: r["beat"])
    (out / "eval_manifest.json").write_text(json.dumps(rows, indent=1))
    for n in range(0, len(rows), a.per_slice):
        (slices / f"slice_{n // a.per_slice:02d}.json").write_text(
            json.dumps(rows[n:n + a.per_slice], indent=1))

    if plan and got < len(plan) * 0.95:
        raise SystemExit(f"FRAME EXTRACTION FAILED: {got}/{len(plan)} — refusing to emit a "
                         f"manifest that would score mostly-missing frames")
    print(f"beats={len(rows)}  frames planned={len(plan)}  extracted={got}  "
          f"slices={(len(rows) + a.per_slice - 1) // a.per_slice}")
    print(f"no-footage beats: {sum(1 for r in rows if r['no_footage'])}")
    print(f"manifest: {out / 'eval_manifest.json'}")


if __name__ == "__main__":
    main()
