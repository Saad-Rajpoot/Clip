#!/usr/bin/env python3
"""A/B any set of pipeline gates: match+verify with them OFF, then ON, everything else identical.

Generalised from bench_ab.py after the deep-bench measurement, because three earlier comparisons in
this project were invalidated by comparing DIFFERENT PIPELINE STAGES (match-only vs verified) rather
than the change under test. One flag set, one job, one code path — only the gates move.

    python3 gate_ab.py <job> <out> VAR1=0,VAR2=0 VAR1=1,VAR2=1

Emits <out>/{off,on}/ eval trees over exactly the beats whose pick differs, plus changed.json.
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


def run(job: Path, env_spec: str) -> dict:
    for kv in env_spec.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k.strip()] = v.strip()
    for m in [m for m in list(sys.modules) if m.startswith("vidlore")]:
        del sys.modules[m]
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import llm

    llm.reset_usage()
    proj = ClipProject.load(job)
    cfg, eng = ClipConfig(), engine_config()
    segs = list(proj.segments)
    proj.selections = match_segments(proj, segs, cfg, progress=None)
    V.verify_and_repair(proj, segs, cfg, eng, progress=None)
    proj.save()
    u = llm.usage_summary()
    log(f"  {env_spec} → ${u['usd']:.2f} / {u['calls']} calls")
    return {s.segment_index: (s.source_id, round(s.in_point, 2), round(s.out_point, 2),
                              bool(getattr(s, "image_path", "")))
            for s in proj.selections}


def main():
    job, out = Path(sys.argv[1]), Path(sys.argv[2])
    spec_off = sys.argv[3] if len(sys.argv) > 3 else ""
    spec_on = sys.argv[4] if len(sys.argv) > 4 else ""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    log(f"run A — OFF ({spec_off})")
    off = run(job, spec_off)
    shutil.copy2(job / "project.json", out / "project_off.json")
    log(f"run B — ON ({spec_on})")
    on = run(job, spec_on)
    shutil.copy2(job / "project.json", out / "project_on.json")

    changed = sorted(i for i in off if off[i] != on.get(i))
    log(f"beats CHANGED by the gates: {len(changed)} → {changed}")
    (out / "changed.json").write_text(json.dumps(changed))
    if not changed:
        log("nothing changed — the gates are inert on this job")
        return

    for tag, src in (("off", out / "project_off.json"), ("on", out / "project_on.json")):
        d = out / tag
        d.mkdir(exist_ok=True)
        shutil.copy2(src, job / "project.json")
        r = subprocess.run([sys.executable, str(HERE / "prep_eval.py"), "--job", str(job),
                            "--out", str(d), "--per-slice", "6"], capture_output=True, text=True)
        if r.returncode:
            log(f"{tag}: prep FAILED — {r.stdout[-200:]}{r.stderr[-200:]}")
            return
        keep = [x for x in json.loads((d / "eval_manifest.json").read_text())
                if x["beat"] in changed]
        sl = d / "eval_slices"
        for f in sl.iterdir():
            f.unlink()
        for i in range(0, len(keep), 6):
            (sl / f"slice_{i // 6:02d}.json").write_text(json.dumps(keep[i:i + 6], indent=1))
        log(f"{tag}: {len(keep)} beats in {(len(keep) + 5) // 6} slice(s) → {d}")
    shutil.copy2(out / "project_on.json", job / "project.json")


if __name__ == "__main__":
    main()
