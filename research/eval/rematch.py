#!/usr/bin/env python3
"""Re-run ONLY match on a job, so a match-affecting fix can be scored on the eval in minutes.

    python3 rematch.py <job dir>
"""
import os, sys, time
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, f"{MAIN}/.claude/worktrees/clipstudio-handover-review-113723")

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(Path(MAIN) / ".env")
os.environ.setdefault("VIDLORE_CLIPSTUDIO_MOMENT_LOCK", "1")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    job = Path(sys.argv[1])
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.match import match_segments

    proj = ClipProject.load(job)
    ok = sum(1 for s in proj.sources if s.status == "ok")
    log(f"{job.name}: {len(proj.segments)} beats · {ok} ok sources")
    sels = match_segments(proj, list(proj.segments), ClipConfig(), progress=log)
    proj.selections = sels
    proj.save()
    picked = sum(1 for s in sels if s.source_id)
    log(f"matched {picked}/{len(sels)} beats · saved")


if __name__ == "__main__":
    main()
