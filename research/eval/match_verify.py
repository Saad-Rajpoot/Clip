#!/usr/bin/env python3
"""Run match + verify on a job and report what the deep bench rescued, with the cost.

    python3 match_verify.py <job dir>
"""
import json, os, shutil, sys, time
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, f"{MAIN}/.claude/worktrees/clipstudio-handover-review-113723")

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(Path(MAIN) / ".env")
os.environ.setdefault("VIDLORE_CLIPSTUDIO_MOMENT_LOCK", "1")

t0 = time.time()
COUNT = {"deep_rescue": 0, "downgrade": 0, "replaced": 0, "failed": 0}


def log(m):
    s = str(m)
    if "rescued from the deep bench" in s:
        COUNT["deep_rescue"] += 1
    elif "exact→contextual" in s or "exact→generic" in s:
        COUNT["downgrade"] += 1
    elif " replaced → " in s:
        COUNT["replaced"] += 1
    elif "FAILED" in s:
        COUNT["failed"] += 1
    print(f"[{time.time()-t0:6.1f}s] {s}", flush=True)


def main():
    job = Path(sys.argv[1])
    warm = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if warm and (warm / "verdict_cache.json").exists() and not (job / "verdict_cache.json").exists():
        shutil.copy2(warm / "verdict_cache.json", job / "verdict_cache.json")
        log(f"warm verdict cache copied from {warm.name}")

    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import llm

    llm.reset_usage()
    proj = ClipProject.load(job)
    cfg, eng = ClipConfig(), engine_config()
    segs = list(proj.segments)

    log(f"match: {len(segs)} beats · {sum(1 for s in proj.sources if s.status=='ok')} sources")
    proj.selections = match_segments(proj, segs, cfg, progress=None)
    proj.save()
    bench = [len(getattr(s, "deep_alternates", None) or []) for s in proj.selections]
    log(f"match done — deep bench: mean {sum(bench)/max(1,len(bench)):.1f} extra candidates/beat")

    log("verify: starting")
    res = V.verify_and_repair(proj, segs, cfg, eng, progress=log) or {}
    proj.save()

    u = llm.usage_summary()
    log("")
    log(f"RESULT  {json.dumps({k: res.get(k) for k in ('verified','replaced','failed','unresolved')})}")
    log(f"        deep-bench rescues: {COUNT['deep_rescue']}   downgrades: {COUNT['downgrade']}")
    log(f"COST    ${u['usd']:.2f} over {u['calls']} call(s) "
        f"({u['prompt']/1000:.0f}k in / {u['completion']/1000:.0f}k out)")


if __name__ == "__main__":
    main()
