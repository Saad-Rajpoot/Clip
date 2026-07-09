#!/usr/bin/env python3
"""Short mismatch test — pasted script (Lannister) != uploaded voiceover (Hound). Validates that
captions come from the voiceover transcription (synced), not the drifting engine split. Daemonized."""
import os, sys, time, traceback
from pathlib import Path
LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/mismatch_test/build.log"


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

from vidlore.clipstudio.orchestrate import produce_auto
P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/mismatch_test")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    res = produce_auto(
        str(P),
        topic="A Lannister Always Pays His Debts",          # matches the PASTED script
        script_text=(P / "script.txt").read_text().strip(),  # Lannister
        movie_hint="Game of Thrones",
        policy="approved_testing", max_sources=6, theme="history", captions=True,
        voiceover=str(P / "voiceover.mp3"),                  # HOUND — mismatched on purpose
        use_tts=True, verify=True, do_build=True, progress=log,
    )
    log(f"OUTPUT → {res.get('output')}")
    log("DONE")
except Exception:
    log("FATAL:\n" + traceback.format_exc()); raise
