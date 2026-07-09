#!/usr/bin/env python3
"""Full build: 'Everyone Was Wrong About the Hound' — user's matched script + voiceover. Daemonized."""
import os, sys, time, traceback
from pathlib import Path
LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/hound_proof/build.log"
def _daemonize(p):
    if os.fork() > 0: os._exit(0)
    os.setsid()
    if os.fork() > 0: os._exit(0)
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd,1); os.dup2(fd,2); os.dup2(os.open(os.devnull,os.O_RDONLY),0)
_daemonize(LOG_PATH)
sys.path.insert(0, "/Users/hussnain/Desktop/vidlore-clipstudio")
_envf=Path("/Users/hussnain/Desktop/vidlore-clipstudio/.env")
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        _l=_l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k,v=_l.split("=",1); os.environ[k.strip()]=v.strip().strip('"').strip("'")
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"]="1"
from vidlore.clipstudio.orchestrate import produce_auto
P=Path("/Users/hussnain/Desktop/clipstudio_output/portal/hound_proof")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
try:
    t0=time.time()
    res=produce_auto(str(P), topic="Everyone Was Wrong About the Hound — Here's the Proof",
        script_text=(P/"script.txt").read_text().strip(), movie_hint="Game of Thrones",
        policy="approved_testing", max_sources=8, theme="history", captions=True,
        voiceover=str(P/"voiceover.mp3"), use_tts=True, verify=True, do_build=True, progress=log)
    log(f"PIPELINE done in {time.time()-t0:.0f}s")
    log(f"OUTPUT → {res.get('output')}"); log("DONE")
except Exception:
    log("FATAL:\n"+traceback.format_exc()); raise
