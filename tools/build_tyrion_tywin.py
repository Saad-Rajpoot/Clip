#!/usr/bin/env python3
"""Full-pipeline build of the Tyrion & Tywin scene-analysis video.

Fresh job: analyze -> discover -> download (GoT sources) -> Face-ID -> deep index ->
match -> cut -> AI verify -> assemble. Uses the user's uploaded 23-min voiceover, with
the caption/breakout fixes (breakouts ON + _group gap-break) live in the engine.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load .env (DeepSeek brain keys, music, etc.) FORCE-set so a blank env can't suppress them
_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k and _v:
                os.environ[_k] = _v

# breakouts ON (the feature is wanted; the gap-break keeps captions synced)
os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

from vidlore.clipstudio.orchestrate import produce_auto
from vidlore.clipstudio import llm as _llm
from vidlore.config import load_config as _cfg

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/tyrion_tywin_scene")
script_text = (P / "script.txt").read_text(encoding="utf-8").strip()
voiceover = str(P / "voiceover.mp3")
TOPIC = "This One Scene Explains the Entire Tyrion & Tywin Relationship"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"brain = {_llm.active_provider(_cfg())}")
log(f"script = {len(script_text.split())} words · voiceover={Path(voiceover).exists()} "
    f"({Path(voiceover).stat().st_size//1024//1024}MB)")
log(f"breakouts={os.environ['VIDLORE_CLIPSTUDIO_BREAKOUTS']}")

t0 = time.time()
res = produce_auto(
    str(P),
    topic=TOPIC,
    script_text=script_text,
    movie_hint="Game of Thrones",
    policy="approved_testing",      # user's standing testing approval (downloads)
    max_sources=8,                  # AUTO-SCALES with beat count for a long multi_scene essay
    theme="history",
    captions=True,
    voiceover=voiceover,            # user's real voice — word-synced captions
    use_tts=True,                   # fallback only (voiceover takes priority)
    verify=True,
    do_build=True,
    progress=log,
)
log(f"PIPELINE done in {time.time()-t0:.0f}s")
log(f"OUTPUT  → {res.get('output')}")
log(f"REVIEW  → {res.get('review_html')}")
log(f"SUMMARY → {res.get('summary')}")
log("DONE")
