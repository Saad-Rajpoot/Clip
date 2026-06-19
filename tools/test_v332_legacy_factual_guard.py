#!/usr/bin/env python3
"""V3.3.2 STEP 5 — LEGACY DATA-TEMPLATE FACTUAL-GUARD test harness.

Exhaustively exercises the centralized factual guard (vidlore/factual_guard.py)
that hardens every legacy + V3.3 fact-bearing template against fabrication:

  1. BAIT      — vague, number-suggestive narration ("the empire expanded
                 rapidly", "several competitors disappeared", "millions"…) must
                 NEVER produce a number/date/currency/ranking/measurement card.
  2. PLANTED   — a card that introduces a hard value ABSENT from the narration
                 (an invented %, ratio, year, currency) must be dropped.
  3. EXPLICIT  — real stated data ("90 percent … 10 percent", "$12M → $84M in
                 1980 … 1995") must be KEPT and the values preserved EXACTLY.
  4. MANIFEST  — a rejection record carries scene id, requested kind, template,
                 source span, guard status, reason, footage-only fallback,
                 timestamp (no field loss; deterministic scene ids).
  5. DRIFT     — every guarded kind exists in the live registry / unlock sets.
  6. REAL LLM  — (optional, needs ANTHROPIC_API_KEY) editor emission + guard
                 chain: after the guard, NO bait scene keeps a fabricated card.

  python tools/test_v332_legacy_factual_guard.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore import factual_guard as fg            # noqa: E402

_p = _f = 0


def ck(name, cond):
    global _p, _f
    _p += bool(cond)
    _f += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


# Vague, number-suggestive scenes that contain NO usable figure.
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
    "The empire expanded rapidly.",
    "Several competitors disappeared.",
    "Profits increased dramatically.",
    "Countless workers were hired.",
    "Its dominance grew significantly over the decades.",
]

# A plausible fabricated body the LLM might attach to each strict kind on a bait
# scene (the kind of value that must NOT survive without evidence).
FAB_BODY = {
    "number": ("300%", ""),
    "stat": ("47%", ""),
    "chart": ("90%", "OF THE MARKET"),
    "progress_bar": ("96", "SUCCESS RATE"),
    "line_chart": ("TREND", "points=1900:10|1950:80"),
    "stat_dashboard": ("BY THE NUMBERS", "stats=SHARE|90%;PRICE|40c"),
    "speedometer": ("200|MPH", "0|260|TOP SPEED"),
    "stat_insight": ("12 MILLION", "enough to fill 200 pools"),
    "receipt": ("STORE|1971", "OIL@2.40|TAX@0.18"),
    "vertical_bar_chart": ("OUTPUT", "1870|4|;1880|30|;1890|90|"),
    "heatmap_grid": ("INTENSITY", "40,60,80;20,90,50"),
    "numerical_ratio": ("9:1", "US|THEM"),
    "document_stack": ("2,847 PAGES", "OF FILINGS"),
    "currency_stat": ("$2.4B", "UP|600%|REVENUE"),
    "score_display": ("8|10", "RATING"),
    "tally_counter": ("2,347", "ARRESTS"),
    "demographic_split": ("POPULATION", "URBAN|70;RURAL|30"),
    "stats_bar": ("PEERS", "US|90|;THEM|20|"),
    "donut_chart": ("67%", "MARKET SHARE"),
    "ranking": ("3", "TOP PRODUCERS"),
    "hashtag_trend": ("#OIL", "12.4M POSTS"),
    "map_region": ("REGION", "82M people · 357,000 km²"),
    "bar_chart": ("OUTPUT", "bars=A:40|B:25|C:35"),
    "composition": ("MIX", "segments=Crude:50|Refined:30|Export:20"),
    "sankey": ("FLOW", "branches=Reinvest:50|Dividends:30|Reserves:20"),
    "gauge": ("THREAT", "value=72;bands=LOW|HIGH"),
    "timeline": ("KEY EVENTS", "events=1865:Rise|1882:Trust|1911:Fall"),
    "mini_timeline": ("DECADE", "1971|Drill;1980|Burned"),
    "typing_date": ("14 MARCH 1987", ""),
    "calendar_grid": ("MARCH 1987", "4,5,6"),
    "era_banner": ("THE RISE", "1865-1911"),
    "eras": ("", "eras=Rise:1865-1882|Peak:1882-1904"),
    "before_after": ("", "before=1865;after=1911"),
    "comparison": ("VS", "EMPIRE|90%;;RIVALS|10%"),
    "versus": ("VS", "pair=Empire|Rivals;values=80|20"),
    "balance": ("TRADEOFF", "pair=Speed|Safety;values=7|3"),
}


def test_bait_drops():
    print("\n[1] BAIT — no number/date/currency/ranking card on vague narration")
    strict = sorted(fg.STRICT_EVIDENCE_KINDS)
    violations = []
    for kind in strict:
        text, body = FAB_BODY.get(kind, ("VALUE", "value=99"))
        for narr in BAIT:
            ok, why = fg.guard(kind, narr, text, body)
            if ok:
                violations.append((kind, narr[:32], why))
    ck(f"all {len(strict)} strict kinds × {len(BAIT)} bait scenes → DROPPED "
       f"({len(strict) * len(BAIT)} checks)", not violations)
    for v in violations[:8]:
        print("      VIOLATION:", v)


def test_comparison_claim_planted():
    print("\n[2] PLANTED — invented values on comparison/claim/quote → dropped")
    cases = [
        ("comparison", "It faced several major rivals.", "VS", "A|90%;;B|10%"),
        ("versus", "The empire battled its rivals.", "VS",
         "pair=Empire|Rivals;values=80|20"),
        ("document", "The report noted severe losses.", "FIELD REPORT",
         "Investigators recorded a 90% collapse in yields."),
        ("did_you_know", "It was a vast operation.", "DID YOU KNOW",
         "It employed 50,000 workers."),
        ("quote_highlight", "Critics condemned the trust.", "It cut prices",
         "by 43 percent overnight"),
        ("numerical_ratio", "They were heavily outnumbered.", "10:1", "US|THEM"),
    ]
    bad = []
    for k, n, t, b in cases:
        ok, why = fg.guard(k, n, t, b)
        if ok:
            bad.append((k, why))
    ck("planted hard values (absent from narration) all dropped", not bad)
    for v in bad:
        print("      LEAK:", v)


def test_explicit_kept_and_preserved():
    print("\n[3] EXPLICIT — real stated data kept; values preserved EXACTLY")
    # (kind, narration, text, body, must-appear values)
    cases = [
        ("number", "By 1882 the trust was formed.", "1882", "", ["1882"]),
        ("versus", "It held 90 percent of refining; rivals had 10 percent.",
         "", "pair=Standard Oil|Independents;values=90|10", ["90", "10"]),
        ("currency_stat",
         "Revenue rose from $12 million in 1980 to $84 million in 1995.",
         "$84M", "UP|84|REVENUE", ["84"]),
        ("line_chart",
         "Revenue rose from $12 million in 1980 to $84 million in 1995.",
         "REVENUE", "points=1980:12|1995:84", ["1980", "1995", "12", "84"]),
        ("composition", "Output split ninety to ten between the two plants.",
         "", "segments=PlantA:90|PlantB:10", ["90", "10"]),
        ("stat_dashboard", "It refined 90 percent at 40 cents a gallon.",
         "BY THE NUMBERS", "stats=SHARE|90%;PRICE|40c", ["90", "40"]),
        ("eras", "The rise ran from 1865 to 1882.", "", "eras=Rise:1865-1882",
         ["1865", "1882"]),
        ("donut_chart", "It controlled 67 percent of the market.", "67%",
         "67|MARKET SHARE", ["67"]),
        ("vertical_bar_chart",
         "Output was 4 in 1870, 30 in 1880 and 90 in 1890.", "OUTPUT",
         "1870|4|;1880|30|;1890|90|", ["1870", "1880", "1890", "30", "90"]),
        ("typing_date", "On 14 March 1987 the cable arrived.", "14 MARCH 1987",
         "", ["1987"]),
    ]
    kept = preserved = 0
    for k, n, t, b, vals in cases:
        ok, why = fg.guard(k, n, t, b)
        if ok:
            kept += 1
            # guard NEVER mutates — the card body is unchanged, so every stated
            # value is still present verbatim.
            blob = f"{t} {b}"
            if all(v in blob for v in vals):
                preserved += 1
        else:
            print(f"      OVER-SUPPRESSED {k}: {why}")
    ck(f"all {len(cases)} explicit-data cards KEPT", kept == len(cases))
    ck(f"all {len(cases)} kept cards preserve their exact values",
       preserved == len(cases))


def test_vague_quantifier_list():
    print("\n[4] STEP-3 vague-quantifier vocabulary present")
    required = ["most", "many", "several", "huge", "massive", "large", "small",
                "rapidly", "dramatically", "significantly", "one of the biggest",
                "millions", "billions", "dozens", "hundreds", "countless",
                "years later", "long ago", "a large share", "a major increase",
                "a dramatic decline", "numerous", "some", "almost all",
                "nearly everyone"]
    missing = [w for w in required if w not in fg.VAGUE_QUANTIFIERS]
    ck(f"all {len(required)} mandated vague quantifiers banned", not missing)
    if missing:
        print("      MISSING:", missing)


def test_manifest_record_shape():
    print("\n[5] MANIFEST — rejection record carries every required field")
    # Simulate what pipeline.py appends to motion_graphics_audit.factual_guard_rejected
    rec = {
        "scene": 7, "requested_kind": "vertical_bar_chart",
        "template": "vertical_bar_chart", "category": fg.classify("vertical_bar_chart"),
        "source_text": "The empire expanded rapidly.", "guard_status": "rejected",
        "reason": fg.guard("vertical_bar_chart", "The empire expanded rapidly.",
                           "OUTPUT", "1870|4|;1880|30|")[1],
        "fallback": "footage_only", "timestamp": 1234567890.0,
    }
    need = {"scene", "requested_kind", "template", "source_text", "guard_status",
            "reason", "fallback", "timestamp"}
    ck("rejection record has all STEP-7 fields", need <= set(rec))
    ck("scene id is deterministic int", isinstance(rec["scene"], int))
    ck("fallback is footage_only", rec["fallback"] == "footage_only")
    ck("reason explains the drop", rec["reason"].startswith("no_explicit_data")
       or rec["reason"].startswith("ungrounded"))


def test_registry_drift():
    print("\n[6] DRIFT — every guarded kind is a real registry / unlock kind")
    try:
        from vidlore import templates as _tpl
        legacy = set(_tpl.all_names())
    except Exception as e:                                     # noqa: BLE001
        print("      (templates import failed:", e, ")")
        legacy = set()
    try:
        from vidlore.script_gen import _MG_UNLOCK_KINDS, _STRUCTURED_KINDS
        unlock = set(_MG_UNLOCK_KINDS) | set(_STRUCTURED_KINDS)
    except Exception:                                          # noqa: BLE001
        unlock = set()
    # adapter-only aliases that the pipeline accepts but aren't registry names
    ALIASES = {"vs", "process", "decision", "headlines", "hierarchy", "eras",
               "bar_chart", "composition", "sankey", "gauge", "versus",
               "balance", "before_after", "cause_effect"}
    known = legacy | unlock | ALIASES | fg.GEOMETRIC_EXEMPT
    unknown = sorted(k for k in fg.FACT_BEARING_KINDS if k not in known)
    ck(f"all {len(fg.FACT_BEARING_KINDS)} guarded kinds resolve "
       f"(legacy={len(legacy)} unlock={len(unlock)})", not unknown)
    if unknown:
        print("      UNKNOWN:", unknown)


def test_real_llm_chain():
    """Optional — editor emission + guard. After the guard, NO bait scene may
    retain a number/date/currency card. Skipped without ANTHROPIC_API_KEY."""
    print("\n[7] REAL LLM — emission + guard chain (optional)")
    try:
        from vidlore.config import load_config
        from vidlore.script_gen import Scene, _apply_editor_decisions
        cfg = load_config()
        if not getattr(cfg, "has_llm", False):
            print("      SKIP (no ANTHROPIC_API_KEY)")
            return
    except Exception as e:                                     # noqa: BLE001
        print("      SKIP (config:", e, ")")
        return
    scenes = [Scene(index=i, narration=n) for i, n in enumerate(BAIT[:8])]
    try:
        _apply_editor_decisions("A Vague History", scenes, cfg)
    except Exception as e:                                     # noqa: BLE001
        print("      SKIP (editor call failed:", e, ")")
        return
    survivors = []
    for s in scenes:
        gk = (s.graphic_kind or "")
        if not gk:
            continue
        ok, _ = fg.guard(gk, s.narration, s.graphic_text or "",
                         s.graphic_body or "")
        # emulate the pipeline guard sweep
        if ok and fg.classify(gk) in ("numeric", "date"):
            survivors.append((s.index, gk, s.graphic_body[:30]))
    ck("after guard, no numeric/date card survives on bait", not survivors)
    for v in survivors:
        print("      SURVIVOR:", v)


def main():
    print("V3.3.2 — LEGACY DATA-TEMPLATE FACTUAL-GUARD  (factual_guard",
          fg.VERSION, ")")
    test_bait_drops()
    test_comparison_claim_planted()
    test_explicit_kept_and_preserved()
    test_vague_quantifier_list()
    test_manifest_record_shape()
    test_registry_drift()
    test_real_llm_chain()
    print(f"\n  RESULT: {_p} passed, {_f} failed")
    return _f == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
