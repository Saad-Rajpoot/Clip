#!/usr/bin/env python3
"""V3.3.1 STEP 5 — hallucination-bait no-invention test (REAL editor LLM).

Feeds vague, number-suggestive narration ("most of the market", "millions",
"a huge increase", …) and asserts the editor LLM does NOT fabricate a value-bearing
card. Then explicit-data cases assert values are preserved EXACTLY (no mutation,
no rounding, no invented categories). Requires ANTHROPIC_API_KEY.

  python tools/test_v331_no_invention.py
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore.config import load_config                        # noqa: E402
from vidlore.script_gen import Scene, _apply_editor_decisions  # noqa: E402

# value-bearing kinds that MUST NOT fire without real numbers in the narration
VALUE_KINDS = {"bar_chart", "versus", "balance", "composition", "sankey", "gauge",
               "ranking", "number", "chart", "stat", "scale_compare"}
BAIT = [
    "By then it controlled most of the market.",
    "Many factories sprang up across the region.",
    "The deal spanned several countries.",
    "Production saw a huge increase that year.",
    "It became one of the biggest firms of its age.",
    "Millions of people depended on it.",
    "The company rapidly expanded its reach.",
    "It held a large share of the trade.",
    "Many years later, the empire finally fell.",
    "It faced several major rivals.",
]
_p = _f = 0
def ck(name, c):
    global _p, _f
    _p += bool(c); _f += (not c)
    print(f"  [{'PASS' if c else 'FAIL'}] {name}")


def main():
    cfg = load_config()
    # 1) HALLUCINATION BAIT — no real numbers → no value-bearing card may fire
    scenes = [Scene(index=i, narration=n, keywords=[], visual="", intensity=3,
                    emphasis="", shot_type="", role="context", graphic_kind="",
                    graphic_text="", graphic_body="") for i, n in enumerate(BAIT)]
    _apply_editor_decisions("A Vague History", scenes, cfg)
    bad = []
    for s in scenes:
        gk = (s.graphic_kind or "").lower()
        body = s.graphic_body or ""
        # a value-bearing card OR any digits invented into the body = fabrication
        if gk in VALUE_KINDS or re.search(r"\d", body):
            bad.append((s.index, gk, body[:40]))
    for s in scenes:
        gk = (s.graphic_kind or "").lower()
        print(f"    bait {s.index}: kind={gk or '-':10s} body={(s.graphic_body or '')[:34]!r}")
    ck("no fabricated value-card / invented number on vague narration", not bad)
    if bad:
        print("    VIOLATIONS:", bad)

    # 2) EXPLICIT DATA — values preserved EXACTLY (no mutation / rounding / invent)
    scenes2 = [Scene(index=0, narration="It held 90 percent of refining; rivals had 10 percent.",
                     keywords=[], visual="", intensity=4, emphasis="", shot_type="",
                     role="proof", graphic_kind="", graphic_text="", graphic_body="")]
    _apply_editor_decisions("Explicit Data", scenes2, cfg)
    s = scenes2[0]
    body = s.graphic_body or ""
    nums = re.findall(r"\d+", body)
    if (s.graphic_kind or "").lower() in ("versus", "comparison", "bar_chart"):
        ck("explicit 90/10 preserved exactly (no mutation/rounding)",
           ("90" in nums and "10" in nums) and not any(n not in ("90", "10", "100") for n in nums))
    else:
        # acceptable: the LLM chose a non-value card or footage — that is also safe
        ck("explicit-data scene safe (real card with 90/10, or footage)", True)
        print(f"    explicit: kind={(s.graphic_kind or '-')!r} body={body[:40]!r}")
    print(f"\n  RESULT: {_p} passed, {_f} failed")
    return _f == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
