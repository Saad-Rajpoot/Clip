#!/usr/bin/env python3
"""Certify LEGACY per-source embedding matrices with a verified manifest.

For every source with embeds.npy + shots.json but NO manifest: recompute the embedding of
each shot's CURRENT keyframe with the ACTIVE model and compare it to the stored row.
Only when EVERY populated row is bit-identical is the manifest written (schema, model
identity, dim, rows, per-row shot/keyframe/content-hash). Any mismatch -> the source is
left un-certified (its still-pass relevance simply stays on the live path).

    python3 tools/backfill_embed_manifests.py /path/to/project_dir [--limit N]
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                               # noqa: E402
from PIL import Image                                            # noqa: E402

from vidlore.clipstudio.models import ClipProject                # noqa: E402
from vidlore.clipstudio import index as I                        # noqa: E402
from vidlore import visual_relevance as vr                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("project_dir")
ap.add_argument("--limit", type=int, default=0, help="stop after N sources (0 = all)")
args = ap.parse_args()

if not vr.available():
    sys.exit("CLIP model unavailable — cannot certify embeddings")

proj = ClipProject.load(args.project_dir)
done = certified = skipped = mismatched = 0
for src in proj.sources:
    if args.limit and done >= args.limit:
        break
    sid = src.id
    mp = I._manifest_path(proj, sid)
    f = proj.embeds_path(sid)
    if mp.exists() or not f.exists():
        continue
    shots = I.load_shots(proj, sid)
    if not shots:
        continue
    done += 1
    mat = np.load(f)
    ok = True
    n_checked = 0
    for sh in shots:
        r = -1 if sh.embed_row is None else int(sh.embed_row)
        if r < 0:
            continue
        if r >= mat.shape[0] or not sh.keyframe_path or not os.path.exists(sh.keyframe_path):
            ok = False
            break
        fresh = np.asarray(vr._img_embed(Image.open(sh.keyframe_path)), dtype="float32")
        if not np.array_equal(fresh, np.asarray(mat[r], dtype="float32")):
            ok = False
            break
        n_checked += 1
    if ok and n_checked:
        I.write_embed_manifest(proj, sid, shots, int(mat.shape[0]), int(mat.shape[1]))
        certified += 1
        print(f"certified  {sid}  ({n_checked} rows verified bit-identical)")
    else:
        mismatched += ok is False
        skipped += 1
        print(f"SKIPPED    {sid}  (row mismatch or missing keyframe — stays on live path)")
print(f"\n{certified} certified · {skipped} skipped · scanned {done}")
