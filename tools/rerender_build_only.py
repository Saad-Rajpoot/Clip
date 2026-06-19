#!/usr/bin/env python3
"""Build-only re-render of the cached Daenerys portal job (0f51bf02ec).

Only the BUILD stage changed (breakouts default ON + captions._group gap-break +
breakout caption-suppress windows), and the sources/index/match/cut are all cached,
so we skip discover/match/cut entirely and call build_video() directly on the loaded
project — fully local, no network. Produces breakouts-ON footage with SYNCED captions.

Output is renamed to final_BREAKOUTS_SYNCED.mp4/.srt so the existing deliverables stay.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load .env (brain keys, music, etc.) so a blank env can't suppress config
_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k and _v:
                os.environ[_k] = _v

# the fix under test: breakouts stay ON, breakout dialogue captioned
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/0f51bf02ec")
proj = ClipProject.load(P)
cfg = load_clip_config()
meta = json.loads((P / "project.json").read_text(encoding="utf-8")).get("meta", {})
title = (meta.get("analysis", {}) or {}).get("topic") or "How Daenerys Became the Villain Nobody Saw Coming"
voiceover = str(P / "voiceover.mp3")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"build-only re-render · {len(proj.segments)} segs · {len(proj.selections)} selections · "
    f"{len(proj.sources)} sources · voiceover={Path(voiceover).exists()}")
log(f"breakouts={os.environ['VIDLORE_CLIPSTUDIO_BREAKOUTS']} · title={title!r}")

t0 = time.time()
out = build_video(
    proj, proj.segments, cfg,
    voice="", captions=True, title=title,
    theme_name="history", use_tts=True,
    voiceover=voiceover, progress=log,
)
log(f"build_video done in {time.time()-t0:.0f}s → {out}")

# preserve the canonical names; deliver under an explicit name
out = Path(out)
outdir = out.parent
final_mp4 = outdir / "final_BREAKOUTS_SYNCED.mp4"
final_srt = outdir / "final_BREAKOUTS_SYNCED.srt"
if out.exists():
    out.replace(final_mp4)
    log(f"renamed → {final_mp4}")
src_srt = out.with_suffix(".srt")
if src_srt.exists():
    src_srt.replace(final_srt)
    log(f"renamed → {final_srt}")
log("DONE")
