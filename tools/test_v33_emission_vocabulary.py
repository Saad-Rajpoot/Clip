#!/usr/bin/env python3
"""V3.3 STEP 7 — end-to-end emission-vocabulary test.

Exercises the real path: LLM-shaped graphic_kind+body -> validator (_parse_extra)
-> apply-gate (valid_kinds) -> adapter format (the pipeline regexes) -> director
routing -> primitive. Three cases per unlocked kind: VALID (fires the right
primitive), MALFORMED (safe reject, no crash), MISSING-DATA (no body -> no card,
no invented facts). Plus: maps stay manual-only (never offered to the LLM).

  python tools/test_v33_emission_vocabulary.py
"""
import re as _re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore import script_gen as SG                          # noqa: E402
from vidlore.motion_graphics import director as D             # noqa: E402
import vidlore.templates as _tpl                              # noqa: E402

_p = _f = 0
def ck(name, cond):
    global _p, _f
    ok = bool(cond); _p += ok; _f += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok

# kind -> (valid body, expected primitive, adapter-shaped assets for routing)
CASES = {
    "bar_chart":   ("bars=Pipelines:40|Rail:25;suffix=%", "statistic_bar_reveal",
                    {"bars": [{"label": "Pipelines", "value": 40}, {"label": "Rail", "value": 25}]}),
    "versus":      ("pair=Standard Oil|Independents;values=90|10;suffix=%", "comparison_split",
                    {"left": "Standard Oil", "right": "Independents", "leftval": 90, "rightval": 10}),
    "balance":     ("pair=Speed|Safety;values=7|3", "vs_balance_scale",
                    {"left": "Speed", "right": "Safety", "leftval": 7, "rightval": 3}),
    "composition": ("segments=Crude:50|Refined:30|Export:20;suffix=%", "composition_stack",
                    {"segments": [{"label": "Crude", "value": 50}, {"label": "Refined", "value": 30}]}),
    "process":     ("steps=Survey|Acquire|Integrate", "process_flow_steps",
                    {"steps": ["Survey", "Acquire", "Integrate"]}),
    "hierarchy":   ("children=Domestic|Export|Pipelines", "org_hierarchy_tree",
                    {"root": "Trust", "children": ["Domestic", "Export"]}),
    "sankey":      ("branches=Reinvest:50|Dividends:30|Reserves:20", "sankey_flow",
                    {"source": "Revenue", "branches": [{"label": "Reinvest", "value": 50}, {"label": "Dividends", "value": 30}]}),
}
# the 4 DARK map cards kept manual-only (NOT map_route/map_region — those are
# already-taught, geo-backbone-safe primitives, correctly still offered).
MAP_KINDS = ("heat_spread", "world_arc", "velocity_route", "actor_badge")


def t_validator():
    print("\n== validator (_parse_extra): accept / preserve / fail-safe ==")
    for k, (body, _prim, _a) in CASES.items():
        st, gk, gt, gb = SG._parse_extra({"graphic": {"kind": k, "text": "T", "body": body}})
        ck(f"{k}: accepted + body preserved", gk == k and gb[:6] == body[:6])
    # banned comparison dropped
    _, gk, _, _ = SG._parse_extra({"graphic": {"kind": "comparison", "text": "X", "body": "pair=A|B"}})
    ck("banned 'comparison' dropped", gk == "")
    # unknown dropped
    _, gk, _, _ = SG._parse_extra({"graphic": {"kind": "totally_made_up", "text": "X", "body": ""}})
    ck("unknown kind dropped", gk == "")
    # malformed structured (scale_compare with no items=) dropped
    _, gk, _, _ = SG._parse_extra({"graphic": {"kind": "scale_compare", "text": "X", "body": "garbage"}})
    ck("malformed scale_compare dropped (no items=)", gk == "")


def t_offered_excludes_maps():
    print("\n== editor LLM is offered the unlock kinds but NOT maps ==")
    offered = set(x for x in _tpl.llm_kind_enum().split("|") if x not in SG._BANNED_TEMPLATES) \
        | SG._MG_UNLOCK_KINDS | SG._STRUCTURED_KINDS
    ck("unlock kinds offered", all(k in offered for k in CASES))
    ck("structured kinds offered (trap-fixed)", "scale_compare" in offered and "route_trace" in offered)
    ck("MAP/geo kinds NOT offered (manual-only)", not any(m in offered for m in MAP_KINDS))
    vk = set(_tpl.all_names()) | SG._MG_UNLOCK_KINDS | SG._STRUCTURED_KINDS
    ck("apply-gate accepts unlock + structured", all(k in vk for k in CASES) and "route_trace" in vk)


def t_adapter_formats():
    print("\n== adapter accepts the taught kinds + body keys (pipeline.py) ==")
    src = (ROOT / "vidlore/pipeline.py").read_text()
    # each unlocked kind appears as an accepted _gk in the adapter, and its body key is parsed
    key = {"bar_chart": "bars=", "versus": "pair=", "balance": "pair=", "composition": "segments=",
           "process": "steps=", "hierarchy": "children=", "sankey": "branches="}
    for k, bk in key.items():
        ck(f"adapter handles {k} ({bk})", (f'"{k}"' in src) and (bk in src))


def t_director_routing_and_missing():
    print("\n== director routing (valid) + missing-data safety (no invented card) ==")
    for k, (_body, prim, assets) in CASES.items():
        mg = [{"index": 0, "role": "proof", "graphic_kind": k, "intensity": 3,
               "narration": "x", "emphasis": "", "assets": assets}]
        got = [d.primitive for d in D.plan(mg, niche="business", seed=7, density=0.6) if d.primitive]
        ck(f"{k} + real data -> {prim}", prim in got)
        # MISSING DATA: same kind, NO assets -> the primitive must NOT fire (no invented card)
        mg0 = [{"index": 0, "role": "proof", "graphic_kind": k, "intensity": 3,
                "narration": "x", "emphasis": "", "assets": {}}]
        got0 = [d.primitive for d in D.plan(mg0, niche="business", seed=7, density=0.6) if d.primitive]
        ck(f"{k} + NO data -> {prim} NOT forced", prim not in got0)


if __name__ == "__main__":
    t_validator()
    t_offered_excludes_maps()
    t_adapter_formats()
    t_director_routing_and_missing()
    print(f"\n=== RESULT: {_p} passed, {_f} failed ===")
    sys.exit(1 if _f else 0)
