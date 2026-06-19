"""RC5.1 — prove `process_flow_steps` (the cheap four-box template) is gone from
PRODUCTION and can never be re-introduced through any live path.

The four-box `process_flow_steps` primitive was cut from production. Its renderer
file (diagrams/process_flow_steps.py) stays on disk but INERT — it is no longer
imported into REGISTRY. This test locks in every removal surface so a future edit
can't quietly resurrect it:

  1. it is NOT in the motion-graphics REGISTRY;
  2. the scriptwriter is neither OFFERED nor will ACCEPT a kind that routes to it
     (gone from script_gen._MG_UNLOCK_KINDS and from the _MG_VOCAB prompt, and
     _scene_graphic() drops a stray `process` kind to no-graphic);
  3. registry.eligible() never returns it for ANY niche, even with every input
     present (so the director can never select it);
  4. the dispatcher, handed a `process_flow_steps` decision (e.g. from a legacy /
     cached script), refuses it gracefully — skip → footage, no clip, no crash;
  5. the production REGISTRY length is exactly 70.

Pure + deterministic: no render, no network.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import script_gen                                    # noqa: E402
from vidlore.motion_graphics import registry                      # noqa: E402
from vidlore.motion_graphics import render_dispatch               # noqa: E402
from vidlore.motion_graphics.director import Decision             # noqa: E402

_GONE = "process_flow_steps"


# ── 1) not in the REGISTRY ────────────────────────────────────────────────────
def test_not_in_registry():
    assert _GONE not in registry.REGISTRY
    assert registry.get(_GONE) is None
    assert _GONE not in registry.all_ids()


# ── 2) not in the scriptwriter vocabulary / structured-asset allow-list ───────
def test_not_in_scriptwriter_vocab():
    # (a) the kind that routed to it ('process') is no longer an offered MG kind
    assert "process" not in script_gen._MG_UNLOCK_KINDS
    # (b) and the literal primitive id never appears in any scriptwriter kind set
    for setname in ("_MG_UNLOCK_KINDS", "_STRUCTURED_KINDS",
                    "_GRAPHIC_KINDS", "_BODY_KINDS"):
        assert _GONE not in getattr(script_gen, setname), setname
    # (c) the prompt text no longer teaches the LLM a `process ->` card line
    vocab = script_gen._MG_VOCAB.lower()
    assert _GONE not in vocab
    assert "* process ->" not in vocab
    # (d) a stray legacy 'process' kind from the LLM is dropped to no-graphic by
    #     the scene-graphic validator (so even if a model emits it, nothing
    #     routes to the four-box). _parse_extra returns (shot,kind,text,body).
    _st, gk, gt, gb = script_gen._parse_extra(
        {"shot_type": "wide",
         "graphic": {"kind": "process", "text": "Survey then Acquire",
                     "body": "steps=Survey|Acquire|Integrate"}})
    assert gk == "", f"a 'process' kind must drop to no-graphic, got {gk!r}"
    assert gt == "" and gb == ""


# ── 3) registry.eligible() never offers it for ANY niche ──────────────────────
def test_eligible_never_returns_it():
    # every input key any primitive could ask for, so the only reason it could be
    # excluded is that it is gone from REGISTRY (not a missing-input filter)
    all_inputs = {
        "steps", "nodes", "parts", "items", "value", "name", "people",
        "title", "milestones", "verdict", "year", "place", "region",
        "bars", "pair", "segments", "children", "before", "after",
        "points", "headlines", "value_a", "value_b", "quote",
    }
    niches = [None, "business", "biography", "crime", "true_crime", "spy",
              "science", "tech", "technology", "engineering", "history",
              "geopolitics", "war", "education", "health", "nature",
              "general", "finance", "politics", "sports"]
    for nm in niches:
        for inten in (1, 2, 3, 4, 5):
            ids = {e["id"] for e in registry.eligible(
                niche=nm, intensity=inten, have_inputs=all_inputs)}
            assert _GONE not in ids, (
                f"{_GONE} must be ineligible for niche={nm!r} intensity={inten}")


# ── 4) the dispatcher refuses a process_flow_steps decision gracefully ────────
def test_dispatcher_refuses_gracefully():
    with tempfile.TemporaryDirectory() as run_dir:
        # a legacy/cached scene still carrying the removed primitive, fully
        # populated so the ONLY reason it can't render is the REGISTRY removal
        dec = Decision(
            scene_index=0, primitive=_GONE,
            inputs={"steps": ["Survey", "Acquire", "Integrate"],
                    "title": "How it worked"},
            score=9.9, reason="legacy")
        entry = render_dispatch.dispatch(dec, run_dir=run_dir, duration=6.0)
        # no crash, no clip, explicitly skipped to footage — never rendered
        assert entry["ok"] is False
        assert entry["skipped"] is True
        assert entry["path"] is None
        assert _GONE in entry["reason"] and "footage" in entry["reason"]

        # and it is counted as neither a rendered graphic nor a fallback card
        res = render_dispatch.dispatch_all(
            [dec], run_dir=run_dir, durations={0: 6.0}, write_manifest=False)
        assert res["summary"]["graphics_rendered"] == 0
        assert _GONE not in res["summary"]["by_primitive"]
        assert res["summary"]["fallbacks"] == 0


# ── 5) REGISTRY length is exactly 70 ──────────────────────────────────────────
def test_registry_length_is_70():
    assert len(registry.REGISTRY) == 70


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED — "
          f"REGISTRY={len(registry.REGISTRY)}, {_GONE!r} fully removed")
