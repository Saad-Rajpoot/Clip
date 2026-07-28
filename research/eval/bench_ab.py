#!/usr/bin/env python3
"""A/B the deep bench the ONLY honest way: verify with it off vs on, everything else identical.

The first read compared v5 match-only (no verify at all) against verify+bench and reported two
beats "regressed" 9 -> 4/5. That was a stage confound — those beats fail _contextual_subject_ok, so
verify replaces them with or without the bench. Same class of mistake as comparing v5 against v4.

    python3 bench_ab.py <job dir> <out dir>
Emits <out>/off/ and <out>/on/ eval trees over the SAME affected beats.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, f"{MAIN}/.claude/worktrees/clipstudio-handover-review-113723")

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(Path(MAIN) / ".env")
os.environ.setdefault("VIDLORE_CLIPSTUDIO_MOMENT_LOCK", "1")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def run(job: Path, bench: str) -> dict:
    """match + verify at a given bench setting; returns {beat: (source_id, in, out)}."""
    os.environ["VIDLORE_CLIPSTUDIO_DEEP_BENCH"] = bench
    for m in [m for m in list(sys.modules) if m.startswith("vidlore")]:
        del sys.modules[m]
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio import verify as V

    proj = ClipProject.load(job)
    cfg, eng = ClipConfig(), engine_config()
    segs = list(proj.segments)
    proj.selections = match_segments(proj, segs, cfg, progress=None)
    V.verify_and_repair(proj, segs, cfg, eng, progress=None)
    proj.save()
    return {s.segment_index: (s.source_id, round(s.in_point, 2), round(s.out_point, 2))
            for s in proj.selections}


def main():
    job, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    log("run A — deep bench OFF")
    off = run(job, "0")
    shutil.copy2(job / "project.json", out / "project_off.json")
    log("run B — deep bench ON")
    on = run(job, "1")
    shutil.copy2(job / "project.json", out / "project_on.json")

    changed = sorted(i for i in off if off[i] != on.get(i))
    log(f"beats the bench actually CHANGED: {len(changed)} → {changed}")
    (out / "changed.json").write_text(json.dumps(changed))
    if not changed:
        return

    for tag, src in (("off", out / "project_off.json"), ("on", out / "project_on.json")):
        d = out / tag
        d.mkdir(exist_ok=True)
        shutil.copy2(src, job / "project.json")
        subprocess.run([sys.executable, str(HERE / "prep_eval.py"),
                        "--job", str(job), "--out", str(d), "--per-slice", "6"],
                       check=True, capture_output=True)
        man = json.loads((d / "eval_manifest.json").read_text())
        keep = [r for r in man if r["beat"] in changed]
        sl = d / "eval_slices"
        for f in sl.iterdir():
            f.unlink()
        for i in range(0, len(keep), 6):
            (sl / f"slice_{i // 6:02d}.json").write_text(json.dumps(keep[i:i + 6], indent=1))
        log(f"{tag}: {len(keep)} beats in {(len(keep) + 5) // 6} slice(s) → {d}")
    shutil.copy2(out / "project_on.json", job / "project.json")     # leave the job in the ON state


if __name__ == "__main__":
    main()
