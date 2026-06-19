#!/usr/bin/env python3
"""V3.3.2 STEP 7 — manifest factual-guard LOGGING validation.

Validates a REAL render manifest for the V3.3.2 logging-retention guarantees:
the motion_graphics_audit namespace is preserved (V3.3.1 untouched), the new
`factual_guard_rejected` list records every dropped fact-bearing card with the
required fields (scene id, requested kind, template, source span, guard status,
reason, footage-only fallback, timestamp), no duplicate scene entries, asset_qa
and scenes preserved.

  python tools/test_v332_manifest_logging.py <run_dir-or-manifest.json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_p = _f = 0


def ck(n, c):
    global _p, _f
    _p += bool(c)
    _f += (not c)
    print(f"  [{'PASS' if c else 'FAIL'}] {n}")
    return bool(c)


def main():
    if len(sys.argv) < 2:
        print("usage: test_v332_manifest_logging.py <run_dir-or-manifest.json>")
        return False
    arg = Path(sys.argv[1])
    path = arg if arg.is_file() else (arg / "motion_graphics_manifest.json")
    if not path.is_file():
        print(f"  manifest not found: {path}")
        return False
    m = json.loads(path.read_text())
    mga = m.get("motion_graphics_audit", {})
    rej = mga.get("factual_guard_rejected", [])

    # preservation of V3.3.1 namespace
    ck("1. motion_graphics_audit namespace present", bool(mga))
    ck("2. audit version preserved == v3.3.1 (NOT bumped)",
       mga.get("version") == "v3.3.1")
    ck("3. asset_qa preserved (coexists)",
       isinstance(m.get("asset_qa"), dict) and "warnings" in m["asset_qa"])
    ck("4. scenes array preserved", isinstance(m.get("scenes"), list)
       and len(m["scenes"]) > 0)
    ck("5. restraint_removed still logged",
       isinstance(mga.get("restraint_removed"), list))

    # V3.3.2 factual-guard logging
    ck("6. factual_guard_version stamped v3.3.2",
       mga.get("factual_guard_version") == "v3.3.2")
    ck("7. factual_guard_rejected is a list", isinstance(rej, list))
    ck("8. at least one fact-card rejection logged", len(rej) >= 1)
    need = {"scene", "requested_kind", "template", "source_text",
            "guard_status", "reason", "fallback", "timestamp"}
    ck("9. every rejection has all STEP-7 fields",
       bool(rej) and all(need <= set(r) for r in rej))
    ck("10. every fallback is footage_only",
       bool(rej) and all(r.get("fallback") == "footage_only" for r in rej))
    ck("11. every guard_status is rejected",
       bool(rej) and all(r.get("guard_status") == "rejected" for r in rej))
    ck("12. every rejection carries a non-empty source span",
       bool(rej) and all((r.get("source_text") or "").strip() for r in rej))
    ck("13. every rejection has an explanatory reason",
       bool(rej) and all((r.get("reason") or "") for r in rej))
    ck("14. no duplicate scene entries in rejections",
       len({r.get("scene") for r in rej}) == len(rej))

    # cross-check: the injected fabricated kinds were dropped
    dropped = {r.get("requested_kind") for r in rej}
    expect = {"donut_chart", "vertical_bar_chart", "timeline", "comparison"}
    ck(f"15. injected fabrications dropped ({sorted(dropped & expect)})",
       bool(expect & dropped))

    # the rejected FABRICATED kind must not survive into rendered primitives.
    # (A scene whose editor card was dropped MAY still render a DIFFERENT,
    #  director-derived card grounded in its real narration — e.g. a rejected
    #  invented line_chart replaced by a grounded wealth_arc_counter from the
    #  scene's real $12M→$800M figures. That is correct, not a leak.)
    rendered = set(mga.get("rendered_primitives", []))
    FAB_PRIM = {"donut_chart": "proportion_ring",
                "vertical_bar_chart": "statistic_bar_reveal",
                "timeline": "chronology_timeline",
                "comparison": "comparison_split",
                "line_chart": "growth_curve_chart"}
    leaked = [k for k in dropped
              if k in rendered or FAB_PRIM.get(k) in rendered]
    ck("16. no rejected fabricated kind survived into rendered primitives",
       not leaked)
    if leaked:
        print("      LEAKED:", leaked)

    print(f"\n  RESULT: {_p} passed, {_f} failed  (manifest: {path})")
    return _f == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
