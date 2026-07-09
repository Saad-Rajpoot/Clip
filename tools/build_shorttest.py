#!/usr/bin/env python3
"""Short end-to-end test render — validates the caption-sync + cold-open fixes on a tiny
uploaded-voiceover job (39s). Self-daemonizes so it survives harness teardown."""
import os
import sys
import time
import traceback
from pathlib import Path

LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/shorttest_lannister/build.log"


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
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

from vidlore.clipstudio.orchestrate import produce_auto

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/shorttest_lannister")
script_text = (P / "script.txt").read_text(encoding="utf-8").strip()
voiceover = str(P / "voiceover.mp3")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    t0 = time.time()
    res = produce_auto(
        str(P),
        topic="A Lannister Always Pays His Debts — the real meaning of Tywin's power",
        script_text=script_text,
        movie_hint="Game of Thrones",
        policy="approved_testing",
        max_sources=6,
        theme="history",
        captions=True,
        voiceover=voiceover,      # TTS-made, used AS an uploaded voiceover → word-aligned VO path
        use_tts=True,
        verify=True,
        do_build=True,
        progress=log,
    )
    log(f"PIPELINE done in {time.time()-t0:.0f}s")
    log(f"OUTPUT  → {res.get('output')}")
    log("DONE")
except Exception:
    log("FATAL:\n" + traceback.format_exc())
    raise
