#!/usr/bin/env python3
"""Re-run ONLY the image-fallback + assemble stages on an existing project whose match/cut/verify
are already done. Used to re-pick web stills after the domain blacklist was fixed (the warm search
cache had resurrected banned AI-art/clipart domains). Footage selections are untouched.

Usage: python3 tools/rerender_images_build.py <project_dir> <voiceover_mp3>
"""
import sys
from pathlib import Path

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video
from vidlore.clipstudio import faceid as _faceid
from vidlore.clipstudio import ledger, review as _review
from vidlore.clipstudio.orchestrate import _fill_image_fallbacks


def main():
    proj_dir = Path(sys.argv[1])
    voiceover = sys.argv[2] if len(sys.argv) > 2 else None

    def log(m):
        print(m, flush=True)

    cfg = load_clip_config()
    proj = ClipProject.load(proj_dir)
    proj.ensure_dirs()
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    segs = proj.segments
    log(f"loaded: {len(segs)} beats · {len(proj.selections)} selections")

    log("· build Face-ID references")
    faceid_obj, refs = None, {}
    if _faceid.available():
        faceid_obj = _faceid.FaceID()
        refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                        faceid_obj, progress=log)

    log("· image fallback (clean domain cache)")
    n = _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log)
    log(f"· image fallback filled {n} beat(s)")

    ledger.finalize(proj, segs, cfg)
    _review.write_review(proj, segs)
    proj.save()

    log("· assemble final video")
    out = build_video(proj, segs, cfg, captions=True,
                      title=analysis.movie_title or proj.name,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"DONE → {out}")


if __name__ == "__main__":
    main()
