#!/usr/bin/env python3
"""Re-render the Daenerys portal video with the two fixes applied:
  1) caption word-sync (per-scene-tolerant alignment — no more drift)
  2) footage-budget scaling (181-beat essay now pulls ~37 specific-scene sources, not 14 generic)

Reuses the existing portal project dir, so the 14 already-downloaded+indexed sources are cached;
only the NEW specific scenes download. Original output is backed up first.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load ALL .env keys (DeepSeek primary brain etc.) FORCE-set so a blank env can't suppress them
_envf = Path(__file__).resolve().parent.parent / ".env"
for _line in _envf.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _v:
            os.environ[_k] = _v

from vidlore.clipstudio.orchestrate import produce_auto
from vidlore.clipstudio import llm as _llm
from vidlore.config import load_config as _cfg

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/0f51bf02ec")
proj = json.load(open(P / "project.json"))
an = proj["meta"]["analysis"]
segs = proj["segments"]
script_text = " ".join((s.get("text") or "") for s in segs).strip()
voiceover = str(P / "voiceover.mp3")

# back up the original deliverable for side-by-side comparison
out = P / "output"
for f, bak in (("final.mp4", "final_ORIGINAL.mp4"), ("final.srt", "final_ORIGINAL.srt")):
    if (out / f).exists() and not (out / bak).exists():
        shutil.copy2(out / f, out / bak)

print(f"[diag] brain = {_llm.active_provider(_cfg())}", flush=True)
print(f"[diag] script = {len(segs)} beats / {len(script_text.split())} words · voiceover={Path(voiceover).exists()}", flush=True)


def log(m):
    print(m, flush=True)


res = produce_auto(
    str(P),
    topic=an.get("topic") or "How Daenerys Became the Villain Nobody Saw Coming",
    script_text=script_text,
    movie_hint=(an.get("movie_title") or "Game of Thrones"),
    policy="approved_testing",          # user's standing testing approval (downloads)
    max_sources=8,                      # AUTO-SCALES to ~37 for a 181-beat multi_scene essay
    theme="history",
    captions=True,
    voiceover=voiceover,                # user's real voice — word-synced captions
    use_tts=True,                       # fallback only (voiceover takes priority)
    verify=True,
    do_build=True,
    progress=log,
)
log(f"OUTPUT → {res.get('output')}")
log(f"SUMMARY → {res.get('summary')}")
