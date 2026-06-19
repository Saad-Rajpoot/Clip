#!/usr/bin/env python3
"""Exclude one source from the pool (e.g. a clip carrying a channel branding/intro card), then
re-match -> cut -> assemble FOOTAGE-ONLY. Reuses the existing index/downloads. No verify/refs/
image-fallback needed — the pool is already era/purity/wrong-show filtered.

Usage: python3 tools/rerender_drop_source.py <project_dir> <voiceover_mp3> <source_id_prefix>
"""
import sys
from pathlib import Path

from vidlore.clipstudio.models import ClipProject, SOURCE_OK
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.match import match_segments
from vidlore.clipstudio.cut import cut_all
from vidlore.clipstudio.build import build_video
from vidlore.clipstudio import ledger, review as _review


def main():
    proj_dir = Path(sys.argv[1])
    voiceover = sys.argv[2] if len(sys.argv) > 2 else None
    drop_prefix = sys.argv[3] if len(sys.argv) > 3 else ""

    def log(m):
        print(m, flush=True)

    cfg = load_clip_config()
    proj = ClipProject.load(proj_dir)
    proj.ensure_dirs()
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    segs = proj.segments

    dropped = 0
    for s in proj.sources:
        if drop_prefix and s.id.startswith(drop_prefix) and s.status == SOURCE_OK:
            s.status = "excluded_branding"
            dropped += 1
            log(f"excluded source {s.id} ({(s.title or '')[:50]!r})")
    # also strip any previously attached web-image stills → footage only
    for sel in proj.selections:
        if getattr(sel, "image_path", None):
            sel.image_path = ""
            sel.image_meta = {}
    proj.save()
    log(f"dropped {dropped} source(s); footage-only re-match of {len(segs)} beats")

    log("· match")
    match_segments(proj, segs, cfg, analysis=analysis, progress=log)
    log("· cut")
    cut_all(proj, cfg, progress=log)
    ledger.finalize(proj, segs, cfg)
    _review.write_review(proj, segs)
    proj.save()
    log("· assemble")
    out = build_video(proj, segs, cfg, captions=True,
                      title=analysis.movie_title or proj.name,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"DONE → {out}")


if __name__ == "__main__":
    main()
