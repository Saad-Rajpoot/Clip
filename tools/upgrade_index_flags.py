#!/usr/bin/env python3
"""Upgrade an EXISTING ClipStudio project's index with multi-frame shot flags
(subs_flag / text_conf / luma_avg / luma_hi / corner_masks) WITHOUT re-running
the expensive index stages (ASR / CLIP / Face-ID / OCR stay untouched).

    python3 tools/upgrade_index_flags.py /path/to/project_dir

Decodes 3 frames per shot (start/mid/end) per source and re-saves shots.json
atomically. Safe to re-run: sources whose shots already carry flags are skipped
(pass --force to recompute).
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pathlib import Path                                    # noqa: E402
from vidlore.clipstudio.models import ClipProject, SOURCE_OK  # noqa: E402
from vidlore.clipstudio import index as I                   # noqa: E402
from vidlore.clipstudio.match import _shot_unreadable       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project_dir")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

proj = ClipProject.load(args.project_dir)
t0 = time.time()
done = skipped = failed = 0
for src in proj.sources:
    if src.status != SOURCE_OK or not src.local_path or not Path(src.local_path).exists():
        continue
    try:
        shots = I.load_shots(proj, src.id)
    except Exception:
        continue
    if not shots:
        continue
    if not args.force and all(int(getattr(sh, "subs_flag", -1) or -1) >= 0 for sh in shots):
        skipped += 1
        continue
    n = I.compute_shot_flags(src.local_path, shots)
    if n:
        shots_file = proj.shots_path(src.id) if hasattr(proj, "shots_path") else None
        if shots_file is None:                         # resolve via the index module's convention
            shots_file = Path(proj.index_dir) / f"{src.id}.shots.json"
        tmp = Path(str(shots_file) + ".tmp")
        tmp.write_text(json.dumps([s.to_dict() for s in shots], indent=1), encoding="utf-8")
        tmp.replace(shots_file)
        subs = sum(1 for s in shots if s.subs_flag == 1)
        dark = sum(1 for s in shots if _shot_unreadable(s))
        print(f"  {src.id[:44]:44} {n:4d} shots · subs={subs:3d} · unreadable-dark={dark:3d}")
        done += 1
    else:
        failed += 1
        print(f"  {src.id[:44]:44} FLAGS FAILED (undecodable?)")
print(f"upgraded {done} source(s), skipped {skipped}, failed {failed} in {time.time()-t0:.0f}s")
