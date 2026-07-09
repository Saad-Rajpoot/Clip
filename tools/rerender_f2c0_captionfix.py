#!/usr/bin/env python3
"""Build-only re-render of f2c0ab1512 with the transcription-caption fix. The pasted script and the
uploaded voiceover are the same Hound essay but edited (different hook + word changes) so they don't
word-align; the fix now captions from the voiceover transcription (synced). Cached footage reused."""
import os, sys, time, traceback, json
from pathlib import Path
LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/f2c0ab1512/output/rerender_captionfix.log"


def _daemonize(p):
    if os.fork() > 0: os._exit(0)
    os.setsid()
    if os.fork() > 0: os._exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2); os.dup2(os.open(os.devnull, os.O_RDONLY), 0)


_daemonize(LOG_PATH)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, v = _l.split("=", 1); os.environ[k.strip()] = v.strip().strip('"').strip("'")
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.build import build_video

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/f2c0ab1512")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    proj = ClipProject.load(P)
    cfg = load_clip_config()
    meta = json.loads((P / "project.json").read_text()).get("meta", {})
    title = (meta.get("analysis", {}) or {}).get("topic") or proj.name
    t0 = time.time()
    out = build_video(proj, proj.segments, cfg, voice="", captions=True, title=title,
                      theme_name="history", voiceover=str(P / "voiceover.mp3"),
                      use_tts=True, progress=log)
    log(f"build_video done in {time.time()-t0:.0f}s → {out}")
    out = Path(out)
    if out.exists(): out.replace(out.parent / "final_CAPTIONFIX.mp4"); log("renamed → final_CAPTIONFIX.mp4")
    if out.with_suffix(".srt").exists(): out.with_suffix(".srt").replace(out.parent / "final_CAPTIONFIX.srt")
    log("DONE")
except Exception:
    log("FATAL:\n" + traceback.format_exc()); raise
