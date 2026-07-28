#!/usr/bin/env python3
"""Re-index a job's sources WITH Face-ID, so a backfilled source is not handicapped by missing
cast data (w_face is 0.30 of the score — a source indexed with faceid=False cannot compete).

    python3 reindex_faceid.py <job dir> [sid ...]
"""
import json, os, sys, time
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, f"{MAIN}/.claude/worktrees/clipstudio-handover-review-113723")

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(Path(MAIN) / ".env")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    job = Path(sys.argv[1])
    only = set(sys.argv[2:])
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.index import index_all
    from vidlore.clipstudio import faceid as _faceid
    import dataclasses

    proj = ClipProject.load(job)
    an = (proj.meta or {}).get("analysis") or {}
    flds = {f.name for f in dataclasses.fields(ScriptAnalysis)}
    analysis = ScriptAnalysis(**{k: v for k, v in an.items() if k in flds})

    if not _faceid.available():
        raise SystemExit("Face-ID unavailable")
    fid = _faceid.FaceID()
    refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir, fid,
                                    progress=log)
    log(f"refs built: {len(refs)} identities")

    # drop the stale index for the targets so index_all recomputes them WITH faces
    if only:
        for sid in only:
            for suf in (".shots.json", ".index.meta.json", ".words.json",
                        ".embeds.npy", ".embeds.manifest.json"):
                p = proj.index_dir / f"{sid}{suf}"
                if p.exists():
                    p.unlink()
            log(f"cleared stale index for {sid}")

    index_all(proj, ClipConfig(), references=refs, faceid=fid, roster=analysis.actors,
              force=False, progress=log)

    for sid in (only or set()):
        m = proj.index_dir / f"{sid}.index.meta.json"
        if m.exists():
            d = json.loads(m.read_text())
            log(f"{sid}: faceid={d.get('faceid')} roster={d.get('roster')}")
    log("DONE")


if __name__ == "__main__":
    main()
