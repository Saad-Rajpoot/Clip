#!/usr/bin/env python3
"""Re-assemble using FOOTAGE ONLY — strip every web-image still and rebuild. For a single-scene
video with a rich, era-filtered footage pool, real on-era footage beats a web image that risks
being AI/replica art, a quote-card overlay, or a wrong-era still of the right actor. Every beat
already has a footage selection, so dropping images leaves no gaps.

Usage: python3 tools/rerender_footage_only.py <project_dir> <voiceover_mp3>
"""
import sys
from pathlib import Path

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video


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

    stripped = 0
    for sel in proj.selections:
        if getattr(sel, "image_path", None):
            sel.image_path = ""
            sel.image_meta = {}
            stripped += 1
    proj.save()
    log(f"stripped {stripped} web-image still(s) — footage-only build of {len(segs)} beats")

    out = build_video(proj, segs, cfg, captions=True,
                      title=analysis.movie_title or proj.name,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"DONE → {out}")


if __name__ == "__main__":
    main()
