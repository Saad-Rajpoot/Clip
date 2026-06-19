#!/usr/bin/env python3
"""Reusable Vidlore render auditor (MNT_2). One tool, any render dir.

Usage:
    python3 tools/audit_render.py [RENDER_DIR_OR_MP4] [--baseline 7.512]

Emits the full audit: MP4 path, contact sheet, exact duration, word count,
scene count, benchmark (15-dim + overall), black-frame count, audio loudness,
footage source breakdown, card/graphic breakdown, honest PASS/FAIL. Reads the
render_meta.json sidecar (MNT_1) for ground-truth pacing. Pure local — no API,
no re-render. If no dir is given, audits the newest render under output/.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import importlib
import tools.benchmark_engine as B
importlib.reload(B)
from vidlore.ffmpeg_tool import ffmpeg_exe  # noqa: E402

FF = ffmpeg_exe()
_INTERMEDIATE = re.compile(r"video_only|video_baked|video_repaired|_seg|_stage|thumb")


def _find_mp4(target: Path) -> Path:
    if target.is_file() and target.suffix == ".mp4":
        return target
    base = target if target.is_dir() else (ROOT / "output")
    cands = [Path(p) for p in glob.glob(str(base / "**/*.mp4"), recursive=True)
             if not _INTERMEDIATE.search(p)]
    if not cands:
        raise SystemExit(f"no final MP4 found under {base}")
    return sorted(cands, key=os.path.getmtime)[-1]


def _duration(mp4: Path) -> float:
    r = subprocess.run([FF, "-i", str(mp4)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            ) if m else 0.0


def _find_log(rd: Path) -> str:
    # render log may sit next to the dir (e.g. output/_imp026_render_log.txt)
    for cand in [rd.parent / f"{rd.name}_log.txt",
                 ROOT / "output" / f"{rd.parent.name}_log.txt",
                 *rd.glob("*.log"), *rd.glob("*.txt")]:
        try:
            if cand and cand.is_file():
                return cand.read_text(errors="ignore")
        except Exception:                                      # noqa: BLE001
            pass
    return ""


def audit(target: Path, baseline: float = 7.512) -> bool:
    mp4 = _find_mp4(target)
    rd = mp4.parent
    sj = rd / "script.json"
    log_text = _find_log(rd)
    dur = _duration(mp4)
    meta = {}
    mp = rd / "render_meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())

    scenes = json.loads(sj.read_text()).get("scenes", []) if sj.exists() else []
    words = sum(len((s.get("narration", "") or "").split()) for s in scenes)
    cards = {}
    for s in scenes:
        gk = (s.get("graphic_kind") or "").strip()
        if gk:
            cards[gk] = cards.get(gk, 0) + 1

    score = B.score_render(mp4, sj, work_dir=rd) if sj.exists() else None
    sc = score["scores"] if score else {}

    rb = subprocess.run([FF, "-i", str(mp4), "-vf",
                         "blackdetect=d=0.10:pic_th=0.98", "-an", "-f", "null", "-"],
                        capture_output=True, text=True)
    blacks = re.findall(r"black_start:([\d.]+)", rb.stderr)

    src = {
        "pexels/stock": len(re.findall(r"\bpexels\b", log_text, re.I)),
        "ai/fal":       len(re.findall(r"\bfal\b|flux|symbolic_concept", log_text, re.I)),
        "web_footage":  len(re.findall(r"web-footage", log_text, re.I)),
        "web_image":    len(re.findall(r"web-image", log_text, re.I)),
        "archive.org":  len(re.findall(r"archive\.org", log_text, re.I)),
        "NIM":          len(re.findall(r"\[NIM\]", log_text)),
    }

    cs = ROOT / "research" / f"audit_{rd.name[:40]}.png"
    try:
        B.generate_contact_sheet(mp4, cs, grid="6x6")
    except Exception as e:                                     # noqa: BLE001
        print("contact sheet failed:", e)

    ov = sc.get("overall_quality", 0.0)
    dur_ok = dur > 1.0
    black_ok = len(blacks) == 0
    bench_ok = ov >= baseline - 0.5
    passed = dur_ok and black_ok and bench_ok and mp4.stat().st_size > 1e6

    print("\n" + "=" * 62)
    print(f"RENDER AUDIT · {rd.name}")
    print("=" * 62)
    print(f"final MP4      : {mp4}  ({mp4.stat().st_size/1e6:.0f} MB)")
    print(f"contact sheet  : {cs}")
    print(f"exact duration : {dur:.1f}s = {dur/60:.2f} min")
    print(f"word count     : {words}")
    print(f"scene count    : {len(scenes)}"
          + (f"  | beats {meta.get('beats')} (avg shot "
             f"{meta.get('shot_len_s', {}).get('avg')}s)" if meta else
             "  | no render_meta (pacing via scene-detection)"))
    print(f"benchmark      : OVERALL {ov}  (baseline {baseline})")
    print(f"black frames   : {len(blacks)}")
    print(f"audio loudness : {score.get('lufs') if score else '?'} LUFS / "
          f"{score.get('true_peak_db') if score else '?'} dBFS TP")
    print(f"sources        : { {k: v for k, v in src.items() if v} }")
    print(f"cards/graphics : {sum(cards.values())} cards / {len(scenes)} scenes "
          f"({100*sum(cards.values())/max(1,len(scenes)):.0f}%) "
          f"· {len(cards)} kinds -> {cards}")
    if sc:
        print("dims           :")
        for k, v in sc.items():
            if k != "overall_quality":
                print(f"    {k:22s} {v}")
    print("=" * 62)
    print(f"VERDICT: {'PASS' if passed else 'REVIEW'}  "
          f"(dur {'ok' if dur_ok else 'X'}, blacks {'ok' if black_ok else 'X'}, "
          f"bench {'ok' if bench_ok else 'X'})")
    print("=" * 62)
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="output",
                    help="render dir or mp4 (default: newest under output/)")
    ap.add_argument("--baseline", type=float, default=7.512)
    a = ap.parse_args()
    audit(Path(a.target), a.baseline)
