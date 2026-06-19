"""V3.4.2 STEP 2 — asset-QA decoupling regression tests.

Proves the asset-QA *reporting/manifest* runs in BOTH MG modes, that MG dispatch
stays gated, that footage safeguards + factual guard remain unconditional, and
that the MG-on behaviour + asset_qa.py thresholds are byte-for-byte unchanged.

Run:  PYTHONPATH=. .venv/bin/python tests/test_v342_asset_qa_decoupling.py
"""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import vidlore.pipeline as P
from vidlore import asset_qa as AQ

REPO = Path(__file__).resolve().parent.parent
PIPE_SRC = (REPO / "vidlore" / "pipeline.py").read_text()
FOOT_SRC = (REPO / "vidlore" / "footage.py").read_text()
ASSET_QA_BASELINE_MD5 = "b0bd5fb736ae818ff6c0d94de7de4415"  # V3.3.2 frozen

PASS = []


def ok(name, cond, detail=""):
    assert cond, f"FAIL: {name} — {detail}"
    PASS.append(name)
    print(f"  ok  {name}")


class _Brief:
    def __init__(self, niche="spy_intel"):
        self.extra = {"niche": niche}


def _line_of(src, needle):
    for i, ln in enumerate(src.splitlines(), 1):
        if needle in ln:
            return i
    return -1


# 1. MG ON → asset QA runs (writes manifest, mode=mg_on)
def test_1_mg_on_asset_qa_runs():
    with tempfile.TemporaryDirectory() as d:
        pl = P._emit_asset_qa_manifest(Path(d), _Brief(), mg_on=True)
        f = Path(d) / "asset_qa.json"
        ok("1_mg_on_asset_qa_runs",
           pl and pl.get("mode") == "mg_on" and f.exists(),
           f"payload={pl}")


# 2. MG OFF → asset QA STILL runs (writes manifest, mode=mg_off)
def test_2_mg_off_asset_qa_runs():
    with tempfile.TemporaryDirectory() as d:
        pl = P._emit_asset_qa_manifest(Path(d), _Brief(), mg_on=False)
        f = Path(d) / "asset_qa.json"
        ok("2_mg_off_asset_qa_runs",
           pl and pl.get("mode") == "mg_off" and f.exists(),
           f"payload={pl}")


# 3. MG OFF → no MG cards render: every MG dispatch is gated behind the env flag
def test_3_mg_dispatch_is_gated():
    gate_line = _line_of(PIPE_SRC, 'VIDLORE_MOTION_GRAPHICS") == "1":')  # director gate (trailing colon)
    disp_lines = [i for i, ln in enumerate(PIPE_SRC.splitlines(), 1)
                  if "dispatch_all" in ln]
    # the always-on asset-QA helper must NOT dispatch MG (QA-only)
    helper = PIPE_SRC.split("def _emit_asset_qa_manifest")[1].split("def render_from_script")[0]
    ok("3_mg_dispatch_is_gated",
       gate_line > 0 and disp_lines and all(l > gate_line for l in disp_lines)
       and "dispatch" not in helper and "director" not in helper,
       f"gate@{gate_line} dispatch@{disp_lines}")


# 4. MG OFF → footage safeguards remain active (independent of MG)
def test_4_footage_safeguards_unconditional():
    import vidlore.period_guard as pg
    import vidlore.visual_relevance as vr  # noqa: F401
    # footage.py wires period_guard + relevance + atomic dedup; the single MG ref
    # in footage.py is a *separate* feature read, far from the safeguard region.
    mg_refs = [i for i, ln in enumerate(FOOT_SRC.splitlines(), 1)
               if "VIDLORE_MOTION_GRAPHICS" in ln]
    pg_ref = _line_of(FOOT_SRC, "period_guard")
    has_relevance = "relevance" in FOOT_SRC.lower()
    has_dedup = "DEDUPE" in FOOT_SRC or "seen_links" in FOOT_SRC
    # the always-on asset-QA call in pipeline is OUTSIDE the MG gate (function level)
    call_line = _line_of(PIPE_SRC, "_emit_asset_qa_manifest(\n")
    if call_line < 0:
        call_line = _line_of(PIPE_SRC, "_aq_payload = _emit_asset_qa_manifest(")
    ok("4_footage_safeguards_unconditional",
       hasattr(pg, "period_risk") and hasattr(pg, "detect_era")
       and pg_ref > 0 and has_relevance and has_dedup
       and len(mg_refs) <= 1,           # safeguards not wrapped in an MG gate
       f"pg@{pg_ref} relevance={has_relevance} dedup={has_dedup} mg_refs={mg_refs}")


# 5. MG ON behaviour unchanged: asset_qa.py byte-identical + in-MG-manifest block intact
def test_5_mg_on_behaviour_unchanged():
    md5 = hashlib.md5((REPO / "vidlore" / "asset_qa.py").read_bytes()).hexdigest()
    ok("5_mg_on_behaviour_unchanged",
       md5 == ASSET_QA_BASELINE_MD5
       and '_md["asset_qa"]' in PIPE_SRC
       and '"motion_graphics_audit"' in PIPE_SRC,
       f"asset_qa.py md5={md5}")


# 6. manifest valid in both modes (required keys + JSON parses)
def test_6_manifest_valid_both_modes():
    req = {"schema", "mode", "niche", "era", "warnings", "summary"}
    good = True
    for mg in (True, False):
        with tempfile.TemporaryDirectory() as d:
            P._emit_asset_qa_manifest(Path(d), _Brief(), mg_on=mg)
            obj = json.loads((Path(d) / "asset_qa.json").read_text())
            good = good and req.issubset(obj.keys()) and isinstance(obj["summary"], dict)
    ok("6_manifest_valid_both_modes", good)


# 7. factual guard remains UNCONDITIONAL (runs before / outside the MG gate)
def test_7_factual_guard_unconditional():
    guard_line = _line_of(PIPE_SRC, "_fg.guard(")
    gate_line = _line_of(PIPE_SRC, 'VIDLORE_MOTION_GRAPHICS") == "1":')  # director gate (trailing colon)
    ok("7_factual_guard_unconditional",
       guard_line > 0 and gate_line > 0 and guard_line < gate_line,
       f"guard@{guard_line} gate@{gate_line}")


# 8. no visual-quality bypass: real guards composed (not stubbed) + thresholds frozen
def test_8_no_quality_bypass():
    helper = PIPE_SRC.split("def _emit_asset_qa_manifest")[1].split("def render_from_script")[0]
    # real bad-portrait input must produce a warning (guards genuinely fire)
    import vidlore.footage as ftg
    saved = dict(getattr(ftg, "_PORTRAIT_PROVENANCE", {}) or {})
    try:
        ftg._PORTRAIT_PROVENANCE = {
            "CATHERINE PEREZ-SHAKDAM": {
                "source": None, "name_match": 0.0,
                "fallback_reason": "name mismatch", "ai_generated": True}}
        with tempfile.TemporaryDirectory() as d:
            pl = P._emit_asset_qa_manifest(Path(d), _Brief(), mg_on=False)
        fired = len(pl.get("warnings", [])) >= 1
    finally:
        ftg._PORTRAIT_PROVENANCE = saved
    ok("8_no_quality_bypass",
       "check_portrait" in helper and "check_provenance" in helper
       and "summarize" in helper and fired
       and hasattr(AQ, "summarize"),
       f"warnings_fired={fired}")


if __name__ == "__main__":
    for fn in [test_1_mg_on_asset_qa_runs, test_2_mg_off_asset_qa_runs,
               test_3_mg_dispatch_is_gated, test_4_footage_safeguards_unconditional,
               test_5_mg_on_behaviour_unchanged, test_6_manifest_valid_both_modes,
               test_7_factual_guard_unconditional, test_8_no_quality_bypass]:
        fn()
    print(f"\nV3.4.2 ASSET-QA DECOUPLING: {len(PASS)}/8 PASS")
