#!/usr/bin/env python3
"""V3.3 STEP 8 (natural-emission proof). Runs the REAL editor LLM
(_apply_editor_decisions) on real-data narration + one plain (no-data) scene, and
verifies the newly-taught vocabulary is requested NATURALLY on real data, with NO
fabrication on the plain scene. Requires ANTHROPIC_API_KEY (the live editor brain).

  python tools/test_v33_natural_emission.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore.config import load_config                        # noqa: E402
from vidlore.script_gen import Scene, _apply_editor_decisions  # noqa: E402

UNLOCK = {"bar_chart", "versus", "balance", "composition", "process", "hierarchy",
          "before_after", "decision", "sankey", "eras", "headlines", "gauge"}
NARR = [
    ("Standard Oil controlled ninety percent of America's refining; the independents held just ten percent.", "proof"),
    ("Inside the empire, revenue split three ways: half was reinvested, thirty percent paid dividends, twenty percent went to reserves.", "context"),
    ("His method never varied: buy the refinery, undercut the price, starve the rival, then absorb it.", "turn"),
    ("The trust was a pyramid — a single holding company above the regional firms above the refineries.", "context"),
    ("He walked the length of the empty refinery floor at dawn, alone with the silence.", "reaction"),  # PLAIN no-data
    ("By 1882, the Standard Oil Trust was the most powerful company on earth.", "climax"),
]


def main():
    cfg = load_config()
    scenes = [Scene(index=i, narration=n, keywords=[], visual="", intensity=3 + (i % 2),
                    emphasis="", shot_type="", role=r, graphic_kind="",
                    graphic_text="", graphic_body="") for i, (n, r) in enumerate(NARR)]
    _apply_editor_decisions("The Rise of Standard Oil", scenes, cfg)
    used = [(s.graphic_kind or "").lower() for s in scenes]
    for s in scenes:
        gk = (s.graphic_kind or "").lower()
        print(f"  scene {s.index} [{s.role}]: {gk or '-':12s} "
              f"{'UNLOCKED' if gk in UNLOCK else ''}  {(s.graphic_body or '')[:46]}")
    unlocked = [k for k in used if k in UNLOCK]
    plain_safe = used[4] not in {"bar_chart", "versus", "balance", "composition", "sankey", "gauge"}
    ok = bool(unlocked) and plain_safe
    print(f"\n  unlocked-kinds requested: {unlocked}")
    print(f"  plain no-data scene fabricated a data card: {'NO (safe)' if plain_safe else 'YES — FAIL'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
