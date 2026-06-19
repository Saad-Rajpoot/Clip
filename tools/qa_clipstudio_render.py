"""Deep QA for a finished ClipStudio render.

    python3 tools/qa_clipstudio_render.py <project_dir>

Checks the FINAL MP4 (streams, black frames, loudness, true peak) and the pipeline
artifacts (sources integrity, selections/confidence, beat_windows contract, alternates
ordering, verifier outcomes, ledger/review surfaces), then dumps per-scene midpoint
frames to <project>/qa_frames/ for visual relevance inspection.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidlore.clipstudio.config import ffmpeg_exe  # noqa: E402

PASS, WARN, FAIL = [], [], []


def res(level, name, detail=""):
    {"PASS": PASS, "WARN": WARN, "FAIL": FAIL}[level].append((name, detail))
    print(f"  {level:4} {name}" + (f" — {detail}" if detail else ""))


def ff(args, timeout=600):
    return subprocess.run([ffmpeg_exe()] + args, capture_output=True, text=True, timeout=timeout)


def main(proj_dir):
    proj_dir = Path(proj_dir)
    print(f"[QA] project: {proj_dir}")
    manifest = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    out = proj_dir / "output" / "final.mp4"
    if not out.exists():
        cand = sorted((proj_dir / "output").glob("*.mp4")) if (proj_dir / "output").exists() else []
        out = cand[0] if cand else None
    if not out or not out.exists():
        res("FAIL", "final mp4 exists", "no output mp4 found")
        return finish()
    print(f"[QA] output: {out} ({out.stat().st_size/1e6:.1f} MB)")

    # --- 1) container / streams ---
    from vidlore.clipstudio.ingest import probe
    info = probe(out)
    res("PASS" if info.get("width") == 1920 and info.get("height") == 1080 else "FAIL",
        "1920x1080 canvas", f"{info.get('width')}x{info.get('height')}")
    res("PASS" if abs(info.get("fps", 0) - 30.0) < 0.6 else "WARN", "30 fps", f"{info.get('fps')}")
    dur = info.get("duration", 0.0)
    res("PASS" if dur > 30 else "FAIL", "duration sane", f"{dur:.1f}s")
    p = ff(["-i", str(out), "-hide_banner"])
    has_audio = "Audio:" in (p.stderr or "")
    res("PASS" if has_audio else "FAIL", "audio stream present")

    # --- 2) black frames ---
    p = ff(["-i", str(out), "-vf", "blackdetect=d=0.4:pix_th=0.10", "-an", "-f", "null", "-"])
    blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", p.stderr or "")
    # ignore the first/last second (fade in/out)
    bad = [(a, b) for a, b in blacks if float(a) > 1.2 and float(b) < dur - 1.2]
    res("PASS" if not bad else "FAIL", "no sustained black frames",
        f"{len(bad)} window(s): {bad[:4]}")

    # --- 3) loudness / peak (parse the trailing Summary block — running I: lines start at -70) ---
    p = ff(["-i", str(out), "-af", "ebur128=peak=true", "-f", "null", "-"])
    err = p.stderr or ""
    summ = err[err.rfind("Summary:"):] if "Summary:" in err else err
    m = re.search(r"I:\s*(-?[\d.]+) LUFS", summ)
    pk = re.findall(r"Peak:\s*(-?[\d.]+) dBFS", summ)
    lufs = float(m.group(1)) if m else None
    res("PASS" if lufs is not None and -19.5 <= lufs <= -13.0 else "WARN",
        "integrated loudness ~-16 LUFS", f"{lufs} LUFS")
    if pk:
        tp = max(float(x) for x in pk)
        res("PASS" if tp <= -0.8 else "WARN", "true peak <= -0.8 dB", f"{tp} dBTP")

    # --- 4) sources integrity ---
    srcs = manifest.get("sources", [])
    ok_srcs = [s for s in srcs if s.get("status") == "ok"]
    res("PASS" if len(ok_srcs) >= 3 else "WARN", "downloaded sources",
        f"{len(ok_srcs)} ok / {len(srcs)} total")
    frags = [f.name for f in (proj_dir / "sources").glob("*.f[0-9]*.*")]
    res("PASS" if not frags else "FAIL", "no DASH fragments left in sources/", str(frags[:4]))
    parts = [f.name for f in (proj_dir / "sources").glob("*.part")] + \
            [f.name for f in (proj_dir / "sources").glob("*.ytdl")]
    res("PASS" if not parts else "WARN", "no partial downloads left", str(parts[:4]))
    hd = sum(1 for s in ok_srcs if (s.get("height") or 0) >= 720)
    res("PASS" if hd >= max(1, len(ok_srcs) // 2) else "WARN", "HD share",
        f"{hd}/{len(ok_srcs)} sources >=720p")

    # --- 5) selections / matching contract ---
    sels = manifest.get("selections", [])
    segs = manifest.get("segments", [])
    res("PASS" if sels and len(sels) == len(segs) else "FAIL",
        "one selection per segment", f"{len(sels)} sel / {len(segs)} seg")
    nosrc = [s["segment_index"] for s in sels if not s.get("source_id")]
    res("PASS" if not nosrc else "WARN", "every beat has footage", f"missing: {nosrc}")
    confs = [s.get("confidence", 0.0) for s in sels if s.get("source_id")]
    if confs:
        res("PASS" if all(0.0 <= c <= 1.0 for c in confs) else "FAIL",
            "confidence within [0,1]", f"min={min(confs):.2f} max={max(confs):.2f} "
            f"mean={sum(confs)/len(confs):.2f}")
    bw_bad, alt_bad, lead_bad = [], [], []
    for s in sels:
        if not s.get("source_id"):
            continue
        bw = s.get("beat_windows") or []
        if bw and not (bw[0][0] == s["source_id"] and abs(float(bw[0][1]) - s["in_point"]) < 0.06):
            lead_bad.append(s["segment_index"])
        if any(len(w) != 3 or w[2] <= w[1] for w in bw):
            bw_bad.append(s["segment_index"])
        alts = s.get("alternates") or []
        scores = [a.get("score", 0) for a in alts]
        if any(scores[i] < scores[i + 1] - 1e-6 for i in range(len(scores) - 1)):
            # ordering is by score+anchor_bonus; allow inversions only when bonus explains them
            b = [a.get("signals", {}).get("anchor_bonus", 0.0) for a in alts]
            eff = [scores[i] + b[i] for i in range(len(scores))]
            if any(eff[i] < eff[i + 1] - 1e-6 for i in range(len(eff) - 1)):
                alt_bad.append(s["segment_index"])
    res("PASS" if not lead_bad else "FAIL", "beat_windows lead = chosen pick", str(lead_bad[:6]))
    res("PASS" if not bw_bad else "FAIL", "beat_windows well-formed", str(bw_bad[:6]))
    res("PASS" if not alt_bad else "FAIL", "alternates ranked best-first", str(alt_bad[:6]))
    flagged = [s["segment_index"] for s in sels if s.get("flagged")]
    res("PASS", "flagged for human review", f"{len(flagged)}/{len(sels)}: {flagged[:8]}")

    # --- 6) verifier outcomes ---
    vstat = {}
    for s in sels:
        v = (s.get("verifier") or {}).get("status", "none")
        vstat[v] = vstat.get(v, 0) + 1
    res("PASS" if vstat.get("ok") else "WARN", "AI verifier ran", str(vstat))
    rejected_airs = []
    for s in sels:
        v = s.get("verifier") or {}
        if v.get("verdict") == "replace":            # unrepaired reject must be flagged
            if not s.get("flagged"):
                rejected_airs.append(s["segment_index"])
    res("PASS" if not rejected_airs else "FAIL",
        "unrepaired verifier rejects are flagged", str(rejected_airs))

    # --- 7) ledger / review surfaces ---
    res("PASS" if (proj_dir / "ledger.jsonl").exists() else "FAIL", "ledger.jsonl written")
    res("PASS" if (proj_dir / "review_queue.json").exists() else "FAIL", "review_queue.json written")
    rh = list(proj_dir.glob("review*.html"))
    res("PASS" if rh else "WARN", "review HTML written", rh[0].name if rh else "")

    # --- 8) clips on disk ---
    clips = list((proj_dir / "clips").glob("*.mp4"))
    res("PASS" if len(clips) >= len(segs) else "WARN", "cut clips on disk",
        f"{len(clips)} clips for {len(segs)} scenes")

    # --- 9) per-scene midpoint frames for visual inspection ---
    qa_dir = proj_dir / "qa_frames"
    qa_dir.mkdir(exist_ok=True)
    n = max(1, len(segs))
    starts = [dur * i / n for i in range(n)]
    for i, t in enumerate(starts):
        mid = t + (dur / n) / 2.0
        ff(["-y", "-ss", f"{mid:.2f}", "-i", str(out), "-frames:v", "1",
            str(qa_dir / f"scene_{i:02d}.jpg")], timeout=60)
    got = len(list(qa_dir.glob("scene_*.jpg")))
    res("PASS" if got == n else "WARN", "midpoint frames extracted", f"{got}/{n} → {qa_dir}")

    return finish()


def finish():
    print(f"\n[QA] {len(PASS)} pass · {len(WARN)} warn · {len(FAIL)} fail")
    for nm, d in FAIL:
        print(f"  FAIL: {nm} — {d}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output/clipstudio_test_joker"))
