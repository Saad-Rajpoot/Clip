"""IMP_026 generalization — does the duration fix scale to LARGER buckets?
Calls _llm_script directly (gen + expansion + scene-clamp) WITHOUT the
expensive editor batching, so it's a cheap word/scene/duration check."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from vidlore.brief import Brief, WORDS_PER_SECOND, DURATION_BUCKETS  # noqa: E402
from vidlore.config import load_config            # noqa: E402
from vidlore.script_gen import _llm_script        # noqa: E402

cfg = load_config(ROOT)
# word bands the user specified (mapped to existing keys)
BANDS = {"6-8": (1100, 1500), "10-12": (1800, 2400), "18-20": (3100, 3900)}
for dur in ["18-20"]:
    brief = Brief(
        title="The Cocaine Empire: Pablo Escobar and the Medellin Cartel",
        prompt=("A long-form cinematic crime documentary on Pablo Escobar and "
                "the Medellin Cartel — origins, the cocaine pipeline to Miami, "
                "the war with the Colombian state, the violence, the fortune, "
                "and the 1993 manhunt and death. Footage-first, premium."),
        duration=dur, theme="crime", fmt="documentary",
    )
    spec = DURATION_BUCKETS[dur]
    try:
        script = _llm_script(brief, cfg)
        scenes = script.scenes
        words = sum(len((s.narration or "").split()) for s in scenes)
        mins = words / (WORDS_PER_SECOND * 60.0)
        lo_w, hi_w = BANDS[dur]
        sc_lo, sc_hi = brief.target_scenes
        lo_m, hi_m = brief.target_minutes
        ok_w = lo_w * 0.9 <= words <= hi_w * 1.15
        ok_m = (lo_m - 1.5) <= mins <= (hi_m + 2)
        print(f"\n=== {dur} (target ~{spec['words']}w / {sc_lo}-{sc_hi} scenes / "
              f"{lo_m:.0f}-{hi_m:.0f} min) ===")
        print(f"   words={words} [{lo_w}-{hi_w}] {'PASS' if ok_w else 'SHORT/LONG'}"
              f" | scenes={len(scenes)} | est {mins:.1f} min "
              f"{'PASS' if ok_m else 'OFF'}")
    except Exception as e:                                     # noqa: BLE001
        print(f"\n=== {dur} === ERROR: {e}")
