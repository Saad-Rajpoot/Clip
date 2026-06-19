#!/usr/bin/env python3
"""Full end-to-end NEW-VIDEO test: topic + script in → finished video out, via produce_auto with
ALL the new features default-on (angle variants, era filter, purity, wrong-show, source-stills,
hook, music arc). Edge-TTS narration. Used to validate the pipeline on a fresh topic.

Usage: python3 tools/test_new_video.py <project_dir> <script_path>
"""
import os
import sys
from pathlib import Path

# Load the project's .env so the LLM analysis (Anthropic) is used instead of the heuristic
# fallback. ONLY the Anthropic keys are needed for analysis; Pexels/FAL stay unset so the test
# stays on real YouTube footage (no stock/AI-video fallback).
_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        # load ALL keys (Anthropic + DeepSeek + provider selection) so the configured primary LLM
        # is actually used (analyze falls back to a heuristic if no LLM key is visible). FORCE-set
        # (not setdefault) so a pre-existing BLANK env var can't suppress the .env value.
        if _k and _v:
            os.environ[_k] = _v

from vidlore.clipstudio.orchestrate import produce_auto
from vidlore.clipstudio import llm as _diag_llm
from vidlore.config import load_config as _diag_cfg
print(f"[diag] LLM active provider = {_diag_llm.active_provider(_diag_cfg())} · "
      f"DEEPSEEK_API_KEY={'set' if os.environ.get('DEEPSEEK_API_KEY') else 'MISSING'}", flush=True)


def main():
    proj_dir = sys.argv[1]
    script_path = sys.argv[2]
    topic = sys.argv[3] if len(sys.argv) > 3 else ""
    movie = sys.argv[4] if len(sys.argv) > 4 else ""

    def log(m):
        print(m, flush=True)

    res = produce_auto(
        proj_dir,
        topic=topic or "Why Tyrion's trial speech is the best monologue in Game of Thrones",
        script_path=script_path,
        movie_hint=movie or "Game of Thrones",
        policy="approved_testing",      # user's standing testing approval (downloads)
        max_sources=12,
        theme="history",
        captions=True,
        use_tts=True,                   # edge-tts (no user voiceover for this topic)
        verify=True,
        do_build=True,
        progress=log,
    )
    log(f"OUTPUT → {res.get('output')}")
    log(f"SUMMARY → {res.get('summary')}")


if __name__ == "__main__":
    main()
