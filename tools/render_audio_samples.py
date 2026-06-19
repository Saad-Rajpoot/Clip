#!/usr/bin/env python3
"""Render the 5 niche validation samples with the Phase-4 AUDIO ENGINE on.

Re-renders the existing cached samples (reusing each cached script.json -> no LLM
cost) with the niche set, so the music/sfx directors activate. After each render
it runs the audio QA gate. Usage:

  python3 tools/render_audio_samples.py [spy|crime|business|history|geopolitics|all]
"""
import os
import sys
from pathlib import Path

# ── render environment (matches the MG validation env) ───────────────────────
os.environ.setdefault("VIDLORE_MOTION_GRAPHICS", "1")
os.environ.setdefault("VIDLORE_MG_DENSITY", "0.35")
os.environ.setdefault("VIDLORE_REUSE_SCRIPT_JSON", "force")
os.environ.setdefault("VIDLORE_AIMG", "1")
os.environ.setdefault("VIDLORE_REAL_PERSON", "1")
os.environ.setdefault("VIDLORE_TTS_BACKEND", "legacy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.brief import Brief                                    # noqa: E402
from vidlore.config import load_config                            # noqa: E402
from vidlore.pipeline import produce                              # noqa: E402

# niche -> (exact cached title, theme). Title must slug to the existing output
# dir so the cached script.json is reused.
SAMPLES = {
    "spy":         ("Our Man in Damascus: The Spy Who Reached the Heart", "history"),
    "crime":       ("The Untouchable: How They Finally Caught Al Capone", "history"),
    "business":    ("John D. Rockefeller: The Man Who Built Standard Oil", "history"),
    "history":     ("The Road to Moscow: Napoleon's Catastrophe of 1812", "history"),
    "geopolitics": ("Thirteen Days: The World on the Edge of Nuclear War", "history"),
}


def render(niche: str) -> int:
    title, theme = SAMPLES[niche]
    brief = Brief(title=title, prompt=f"{niche} documentary (audio-engine validation)",
                  fmt="documentary", duration="6-8", theme=theme,
                  extra={"niche": niche})
    cfg = load_config()
    print(f"\n{'='*70}\n[{niche.upper()}] rendering: {title!r}\n{'='*70}", flush=True)
    r = produce(brief, cfg, ROOT / "output", keep_work=True)
    print(f"[{niche}] DONE -> {r.video}  ({getattr(r, 'video_seconds', 0):.0f}s)", flush=True)
    return 0


def main(argv):
    which = (argv[0] if argv else "all").lower()
    targets = list(SAMPLES) if which == "all" else [which]
    rc = 0
    for n in targets:
        if n not in SAMPLES:
            print(f"unknown niche {n!r}; choose from {list(SAMPLES)}")
            return 2
        try:
            render(n)
        except Exception as e:                                     # noqa: BLE001
            import traceback
            print(f"[{n}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
