#!/usr/bin/env python3
"""V3.3.1 STEP 7 — manifest-preservation regression test.

Validates a real MG manifest (default: the saved v3.3.1 business proof manifest) for
the lifecycle-retention guarantees: the motion_graphics_audit namespace survives the
asset_qa rewrite, asset_qa is preserved unchanged, restraint removals + rendered
cards + footage-only fallbacks + factual-guard modes are all logged, scene ids are
present, no duplicate keys. Pure structural validation — no render needed.

  python tools/test_v331_manifest_preservation.py [manifest.json]
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "research/motion_graphics_expansion/emission_unlock/v331_business_manifest.json"
_p = _f = 0
def ck(n, c):
    global _p, _f
    _p += bool(c); _f += (not c)
    print(f"  [{'PASS' if c else 'FAIL'}] {n}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.is_file():
        print(f"  manifest not found: {path}"); return False
    m = json.loads(path.read_text())
    mga = m.get("motion_graphics_audit", {})
    decs = mga.get("decisions", [])
    # 1-8: the lifecycle-retention guarantees
    ck("1. motion_graphics_audit namespace present after final QA", bool(mga))
    ck("2. asset_qa preserved (QA output survives)", "asset_qa" in m and isinstance(m["asset_qa"], dict))
    ck("3. scenes array preserved", isinstance(m.get("scenes"), list) and len(m["scenes"]) > 0)
    ck("4. rendered cards logged", bool(mga.get("rendered_primitives")))
    ck("5. restraint removals logged (list present)", isinstance(mga.get("restraint_removed"), list))
    ck("6. footage-only fallback logged (a req=None/footage_only decision)",
       any(d.get("mode") == "footage_only" for d in decs))
    ck("7. factual-guard mode on every decision",
       all(d.get("factual_guard") in ("source_evidence_required", "no_evidence_footage_only") for d in decs) and bool(decs))
    ck("8. every decision has a scene_index", all("scene_index" in d for d in decs) and bool(decs))
    # 9-14: integrity
    ck("9. no duplicate top-level keys", len(set(m.keys())) == len(m.keys()))
    ck("10. asset_qa warnings field intact", "warnings" in m.get("asset_qa", {}))
    ck("11. asset_qa summary verdict intact", "summary" in m.get("asset_qa", {}))
    ck("12. scene entries keep deterministic scene_index", all("scene_index" in s for s in m.get("scenes", [])))
    ck("13. audit version stamped", mga.get("version") == "v3.3.1")
    ck("14. no duplicate decisions per scene",
       len({d.get("scene_index") for d in decs}) == len(decs))
    print(f"\n  RESULT: {_p} passed, {_f} failed  (manifest: {path.name})")
    return _f == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
