#!/usr/bin/env python3
"""Re-build ONLY the final video for the Daenerys project with the fixed (chunked) caption sync.

The relevance-improved footage selections (45 sources) are already matched in the project — this
re-runs just build_video (narration word-sync + assemble), so it's fast (no re-discover/index/match).
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_envf = Path(__file__).resolve().parent.parent / ".env"
for _line in _envf.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _v:
            os.environ[_k] = _v

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/0f51bf02ec")
proj = ClipProject.load(P)
cfg = load_clip_config()
vo = str(P / "voiceover.mp3")

# preserve the relevance-good / caption-drift build before overwriting
out = P / "output"
for f, bak in (("final.mp4", "final_RELEVANCE_oldcap.mp4"), ("final.srt", "final_RELEVANCE_oldcap.srt")):
    if (out / f).exists() and not (out / bak).exists():
        shutil.copy2(out / f, out / bak)


def log(m):
    print(m, flush=True)


log(f"[diag] project={P.name} · segments={len(proj.segments)} · sources={len(proj.sources)}")
res = build_video(
    proj, proj.segments, cfg,
    captions=True,
    title="How Daenerys Became the Villain Nobody Saw Coming",
    theme_name="history",
    use_tts=True,
    voiceover=vo,            # chunked word-synced captions
    progress=log,
)
log(f"OUTPUT → {res}")
