#!/usr/bin/env python3
"""V3.0.1 regression — portrait fallback-ladder DECISION logic.

Deterministic (no network): monkeypatches footage._real_person_image to drive
each tier, asserting resolve_legend_portrait picks the right source_type,
surfaces the validation score, and logs the correct fallback reason. The real
network tiers + face validators are proven separately by the micro-render
(_legend_portrait_ladder_proof.py).
"""
import os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore import footage  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _make_fake_image(p: Path):
    from PIL import Image
    Image.new("RGB", (640, 800), (90, 80, 70)).save(p)
    return p


def with_real(returns_path, prov):
    """Return a patched _real_person_image that records prov + returns a path/None."""
    def _fn(name, role, dest, *, gender="unknown", cache_dir=None):
        footage._PORTRAIT_PROVENANCE[name] = dict(prov)
        if returns_path:
            _make_fake_image(Path(dest))
            return str(dest)
        return None
    return _fn


def run():
    td = Path(tempfile.mkdtemp())
    orig = footage._real_person_image
    try:
        # ── A1: archival hit (Wikipedia lead) → archival_public_domain, score ──
        footage._real_person_image = with_real(True, {
            "source": "Wikipedia lead image", "name_match": 1.0,
            "fallback_reason": None})
        r = footage.resolve_legend_portrait("Test Subject", dest=td / "a1.jpg",
                                            cache_dir=td, allow_ai=False)
        check("A1 archival source_type",
              r["portrait_source_type"] == "archival_public_domain", r)
        check("A1 portrait_path set", bool(r["portrait_path"]))
        check("A1 score surfaced", r["portrait_validation_score"] == 1.0)
        check("A1 no fallback reason", r["portrait_fallback_reason"] is None)
        check("A1 subject recorded", r["portrait_subject"] == "Test Subject")

        # ── A2: Commons (web) hit → validated_web ──
        footage._real_person_image = with_real(True, {
            "source": "Some Other Source", "name_match": 0.7,
            "fallback_reason": None})
        r = footage.resolve_legend_portrait("Web Subject", dest=td / "a2.jpg",
                                            cache_dir=td, allow_ai=False)
        check("A2 validated_web source_type",
              r["portrait_source_type"] == "validated_web", r)

        # ── B: no real portrait, AI off → monogram + reason ──
        footage._real_person_image = with_real(False, {
            "source": None, "name_match": None,
            "fallback_reason": "no wikipedia lead image"})
        r = footage.resolve_legend_portrait("Nobody Atall", dest=td / "b.jpg",
                                            cache_dir=td, allow_ai=False)
        check("B monogram", r["portrait_source_type"] == "monogram_pedestal", r)
        check("B no portrait_path", not r["portrait_path"])
        check("B fallback reason logged",
              "wikipedia" in (r["portrait_fallback_reason"] or ""))

        # ── C: bad candidate rejected → monogram + rejection reason surfaced ──
        footage._real_person_image = with_real(False, {
            "source": None, "name_match": 0.18,
            "fallback_reason": "validator rejected: wrong face (unrelated_subject)"})
        r = footage.resolve_legend_portrait("Wrong Face", dest=td / "c.jpg",
                                            cache_dir=td, allow_ai=False)
        check("C monogram after reject",
              r["portrait_source_type"] == "monogram_pedestal", r)
        check("C rejection reason surfaced",
              "reject" in (r["portrait_fallback_reason"] or "").lower())

        # ── D: empty name → monogram + 'no subject name' ──
        r = footage.resolve_legend_portrait("   ", dest=td / "d.jpg",
                                            cache_dir=td, allow_ai=False)
        check("D empty-name monogram",
              r["portrait_source_type"] == "monogram_pedestal"
              and not r["portrait_path"], r)
        check("D empty-name reason", r["portrait_fallback_reason"] == "no subject name")

        # ── E: never raises even if the resolver internals explode ──
        def _boom(*a, **k):
            raise RuntimeError("simulated source failure")
        footage._real_person_image = _boom
        r = footage.resolve_legend_portrait("Crash Test", dest=td / "e.jpg",
                                            cache_dir=td, allow_ai=False)
        check("E never raises → monogram",
              r["portrait_source_type"] == "monogram_pedestal", r)

        # ── F: AI-off never invents a portrait (no fal even if cfg present) ──
        class _Cfg:
            fal_key = "fake"; fal_model = "fal-ai/flux/schnell"
        footage._real_person_image = with_real(False, {
            "source": None, "name_match": None, "fallback_reason": "none found"})
        r = footage.resolve_legend_portrait("Real Person No Image", dest=td / "f.jpg",
                                            cache_dir=td, cfg=_Cfg(), allow_ai=False)
        check("F AI-off → no fabricated portrait",
              r["portrait_source_type"] == "monogram_pedestal"
              and not r["portrait_path"], r)
    finally:
        footage._real_person_image = orig

    print(f"\nportrait-ladder: {PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
