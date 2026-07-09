#!/usr/bin/env python3
"""Build-only re-render of portal job 98a781306e to validate the caption-desync fix.

Only the BUILD stage is affected (the _apply_breakouts atomicity fix + _splice_audio t=0),
and sources/index/match/selections are all cached, so we call build_video() directly on the
loaded project — fully local, no network. Self-daemonizes so it survives harness teardown.
"""
import os
import sys
import time
import traceback
from pathlib import Path

LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/98a781306e/output/rerender_captionfix.log"


def _daemonize(logpath):
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(logpath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)


_daemonize(LOG_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k and _v:
                os.environ[_k] = _v

os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

import json
from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/98a781306e")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    proj = ClipProject.load(P)
    cfg = load_clip_config()
    meta = json.loads((P / "project.json").read_text(encoding="utf-8")).get("meta", {})
    title = (meta.get("analysis", {}) or {}).get("topic") or proj.name
    voiceover = str(P / "voiceover.mp3")
    log(f"caption-fix re-render · segs={len(proj.segments)} sel={len(proj.selections)} "
        f"sources={len(proj.sources)} · title={title!r}")
    t0 = time.time()
    out = build_video(proj, proj.segments, cfg, voice="", captions=True, title=title,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"build_video done in {time.time()-t0:.0f}s → {out}")
    out = Path(out)
    final_mp4 = out.parent / "final_CAPTIONFIX.mp4"
    final_srt = out.parent / "final_CAPTIONFIX.srt"
    if out.exists():
        out.replace(final_mp4); log(f"renamed → {final_mp4}")
    if out.with_suffix(".srt").exists():
        out.with_suffix(".srt").replace(final_srt); log(f"renamed → {final_srt}")
    log("DONE")
except Exception:
    log("FATAL:\n" + traceback.format_exc())
    raise
