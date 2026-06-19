#!/usr/bin/env python3
"""V3.0.1 — verify a finished biography render against the polish gates.
Usage: _v301_verify.py <run_dir> [log_path]
Checks: MG manifest (act_chapter + portrait_legend + 5 portrait fields),
black frames (blackdetect), loudness (LUFS) + true-peak (dBTP ≤ -1.0),
subtitles present, AI-video calls = 0. Deletes nothing.
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore.ffmpeg_tool import ffmpeg_exe  # noqa: E402

run = Path(sys.argv[1])
log = Path(sys.argv[2]) if len(sys.argv) > 2 else None
ff = ffmpeg_exe()
mp4 = next((p for p in run.glob("*.mp4") if "editor_cache" not in str(p)), None)
print("RUN_DIR:", run)
print("MP4:", mp4)

# ── 1. MG manifest ───────────────────────────────────────────────
m = run / "motion_graphics_manifest.json"
if m.exists():
    d = json.loads(m.read_text())
    print("\n[MANIFEST] by_primitive:", d.get("summary", {}).get("by_primitive"))
    for e in d.get("scenes", []):
        if e.get("primitive") in ("act_chapter_card", "portrait_legend_reveal"):
            pf = {k: e.get(k) for k in ("portrait_subject", "portrait_source_type",
                  "portrait_validation_score", "portrait_fallback_reason") if k in e}
            print(f"  scene {e['scene_index']}: {e['primitive']} ok={e.get('ok')} {pf}")
else:
    print("\n[MANIFEST] MISSING")

# ── 2. black frames (blackdetect) ────────────────────────────────
if mp4:
    r = subprocess.run([ff, "-hide_banner", "-i", str(mp4), "-vf",
                        "blackdetect=d=0.05:pic_th=0.98", "-f", "null", "-"],
                       capture_output=True, text=True)
    spans = re.findall(r"black_start:(\S+) black_end:(\S+)", r.stderr)
    print(f"\n[BLACK] blackdetect spans (d>=0.05s): {len(spans)}")
    for s, e in spans[:8]:
        print(f"   {s}..{e}")

# ── 3. loudness + true-peak (loudnorm analysis) ──────────────────
if mp4:
    r = subprocess.run([ff, "-hide_banner", "-i", str(mp4), "-af",
                        "loudnorm=print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    mj = re.search(r"\{[^{}]*input_i[^{}]*\}", r.stderr, re.S)
    if mj:
        j = json.loads(mj.group(0))
        print(f"\n[AUDIO] integrated={j.get('input_i')} LUFS | "
              f"true_peak={j.get('input_tp')} dBTP | LRA={j.get('input_lra')}")
        try:
            tp = float(j.get("input_tp"))
            print(f"   true-peak <= -1.0 dBTP: {'PASS' if tp <= -1.0 else 'FAIL (' + str(tp) + ')'}")
        except Exception:
            pass

# ── 4. subtitles present ─────────────────────────────────────────
subs = list(run.glob("*.srt")) + list(run.glob("**/*.srt"))
print(f"\n[SUBS] .srt files: {len(subs)}", [s.name for s in subs[:3]])
if mp4:
    r = subprocess.run([ff, "-hide_banner", "-i", str(mp4)], capture_output=True, text=True)
    print("   subtitle stream in MP4:", "Subtitle" in r.stderr)

# ── 5. AI-video calls (from log) ─────────────────────────────────
if log and log.exists():
    txt = log.read_text(errors="ignore")
    aiv = len(re.findall(r"_fal_video|ai[- ]video (?:call|gen)|VIDEO GEN", txt, re.I))
    print(f"\n[AI-VIDEO] fal_video / ai-video markers in log: {aiv} (must be 0)")
    legend = re.findall(r"\[legend-portrait\] (.+)", txt)
    for l in legend[:4]:
        print("   legend-portrait:", l.strip())
