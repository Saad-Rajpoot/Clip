#!/usr/bin/env python3
"""Full-auto render of the Arthur Dayne / Tower-of-Joy essay from script + owner voiceover,
with every current fix (fabricated-footage gates, window-QC, release-gate recovery, sub-HD
last-resort, era-canonical, Gemini timeout). Self-daemonizes so it survives terminal teardown.

Progress → <project>/output/_render.log ; sentinel → <project>/output/_render.done
"""
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PD = "/Users/hussnain/Desktop/clipstudio_output/portal/arthur_dayne"
SCRIPT = os.path.join(PD, "script.txt")
VOICEOVER = ("/Users/hussnain/Downloads/"
             "The Scene Where Ser Arthur Dayne Chose to Die (Everyone Gets This Wrong).mp3")
OUTDIR = os.path.join(PD, "output")
LOG = os.path.join(OUTDIR, "_render.log")
DONE = os.path.join(OUTDIR, "_render.done")
os.makedirs(OUTDIR, exist_ok=True)

# double-fork daemonize
if os.fork() > 0:
    print(f"render daemonized → {LOG}")
    sys.exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
try:
    os.remove(DONE)
except OSError:
    pass
_f = open(LOG, "a", buffering=1)
os.dup2(_f.fileno(), 1)
os.dup2(_f.fileno(), 2)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    # keys live in <repo>/.env — load them regardless of CWD before anything reads them
    from pathlib import Path as _P
    from vidlore.config import _load_dotenv
    _load_dotenv(_P(REPO) / ".env")
    # SAFETY NET: a few backstory beats (Lyanna/Rhaegar/Aerys — barely-filmed characters)
    # may have no exact footage even after recovery + web-still fallback. warn-mode completes a
    # watchable REVIEW draft instead of aborting; the audit then reports the footage-limited
    # beats honestly. Recovery + web-image fallback still run first and prefer REAL footage.
    os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"
    from vidlore.clipstudio.orchestrate import produce_auto
    import os as _oe
    log(f"env: gemini={'set' if _oe.environ.get('GEMINI_API_KEY') else 'MISSING'} "
        f"deepseek={'set' if _oe.environ.get('DEEPSEEK_API_KEY') else 'MISSING'}")
    log("=== Arthur Dayne full render START ===")
    res = produce_auto(
        PD,
        topic="Ser Arthur Dayne the Sword of the Morning, the Tower of Joy duel, "
              "young Ned Stark, Rhaegar Targaryen and Lyanna Stark, Robert's Rebellion, "
              "Game of Thrones season 6 flashback",
        script_path=SCRIPT,
        movie_hint="Game of Thrones",
        voiceover=VOICEOVER,          # owner's own VO, force-aligned to the script
        use_tts=True,                 # aligns the provided VO; TTS only where needed
        max_sources=16,               # footage-constrained single scene → wide real-footage pool
        policy="approved_testing",    # SAME policy the portal runs (web.py PORTAL_POLICY default)
        theme="history",
        title="The Scene Where Ser Arthur Dayne Chose to Die",
        verify=True,
        do_build=True,
        resume=True,                  # reuse the cached analyze (203 beats) — continue at discover
        progress=log,
    )
    out = res.get("output", "")
    log(f"BUILD DONE -> {out}")
    with open(DONE, "w") as fh:
        fh.write(str(out))
except Exception as e:  # noqa: BLE001
    log("ERROR: " + repr(e))
    log(traceback.format_exc())
    with open(DONE, "w") as fh:
        fh.write("ERROR: " + repr(e))
