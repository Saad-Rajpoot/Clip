#!/usr/bin/env python3
"""Fast re-render of an EXISTING clipstudio project: reuse the downloaded sources + deep index,
re-run only match -> cut -> verify -> image-fallback -> assemble. This applies the new pool
filters (era / single-scene purity / wrong-show) and the hook + dynamic-music build changes
WITHOUT paying for discovery, download, or indexing again.

Usage: python3 tools/rerender_clean_pool.py <project_dir> <voiceover_mp3>
"""
import sys
from pathlib import Path

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.match import match_segments
from vidlore.clipstudio.cut import cut_all
from vidlore.clipstudio.build import build_video
from vidlore.clipstudio import faceid as _faceid
from vidlore.clipstudio import verify as _verify
from vidlore.clipstudio import ledger, review as _review
from vidlore.clipstudio.orchestrate import _fill_image_fallbacks
from vidlore.config import load_config as engine_config


def main():
    proj_dir = Path(sys.argv[1])
    voiceover = sys.argv[2] if len(sys.argv) > 2 else None

    def log(m):
        print(m, flush=True)

    cfg = load_clip_config()
    eng = engine_config()
    proj = ClipProject.load(proj_dir)
    proj.ensure_dirs()
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    segs = proj.segments
    log(f"loaded: movie={analysis.movie_title!r} · {len(analysis.actors)} actors · "
        f"{len(segs)} beats · {len([s for s in proj.sources if s.status=='ok'])} ok sources")

    # 4 — Face-ID references (needed by verify + image fallback)
    log("4/9 · build Face-ID references")
    faceid_obj, refs = None, {}
    if _faceid.available():
        faceid_obj = _faceid.FaceID()
        refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                        faceid_obj, progress=log)

    # 6 — match (NEW: era + single-scene purity + wrong-show pool filters apply here)
    log("6/9 · match")
    match_segments(proj, segs, cfg, analysis=analysis, progress=log)

    # 7 — cut
    log("7/9 · cut")
    cut_all(proj, cfg, progress=log)

    # 8 — AI verify + repair
    log("8/9 · AI verify + repair")
    try:
        _verify.verify_and_repair(proj, segs, cfg, eng, progress=log)
    except Exception as e:
        log(f"verify: skipped ({type(e).__name__}: {e})")

    # 8b — image fallback for weak beats
    import os as _os
    if _os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip() not in ("0", "false", "no"):
        try:
            _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log)
        except Exception as e:
            log(f"image-fallback: skipped ({type(e).__name__}: {e})")

    summ = ledger.finalize(proj, segs, cfg)
    _review.write_review(proj, segs)
    proj.save()
    log(f"  QC · {summ['flagged_for_review']}/{summ['segments']} flagged · mean conf {summ['mean_confidence']}")

    # 9 — assemble (hook + dynamic music arc apply here; user voiceover forced-aligned)
    log("9/9 · assemble final video")
    out = build_video(proj, segs, cfg, captions=True,
                      title=analysis.movie_title or proj.name,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"DONE → {out}")


if __name__ == "__main__":
    main()
