"""V2.8 structured-asset adapter — Section-C primitives that need explicit data.

Two Section-C cards carry inherently STRUCTURED inputs the director can never
mine from prose:

  • silhouette_scale_compare  ← items=[{label,size,note}]   (scale-true sizes)
  • footage_route_trace       ← points=[{x,y,label}]        (route coordinates)

Unlike regions/hops/stages/parent/targets/fact (which the director auto-derives
from narration), there is no prose to derive these from — so without the
graphic_body adapter parsing `items=` / `points=` they can NEVER fire. This pins
that a script beat tagged `scale_compare` + `items=…` fires
silhouette_scale_compare, and `route_trace` + `points=…` fires
footage_route_trace, end-to-end through the REAL adapter parse
(`vidlore.pipeline._mg_structured_assets`) and the REAL `director.plan()`.

    python3 tools/test_structured_mg_assets.py

AI-video stays OFF (VIDLORE_AI_VIDEO=0); this is a pure planner test — no render,
no network, no paid API.
"""
import os
import sys

os.environ["VIDLORE_AI_VIDEO"] = "0"          # keep AI-video OFF (explicit)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.pipeline import _mg_structured_assets          # noqa: E402
from vidlore.motion_graphics import director as mgdir        # noqa: E402


def _scene(gk, gb, narration):
    """The per-scene dict exactly as the pipeline hands it to the director —
    graphic_body parsed to assets through the REAL adapter helper."""
    return {"index": 0, "graphic_kind": gk, "intensity": 3,
            "narration": narration, "assets": _mg_structured_assets(gk, gb)}


def _fired(gk, gb, narration, niche):
    sc = _scene(gk, gb, narration)
    decisions = mgdir.plan([sc], niche=niche, seed=0)
    return sc["assets"], (decisions[0].primitive if decisions else None)


# (label, graphic_kind, graphic_body, narration, niche, want_primitive, key, want_n)
CASES = [
    ("scale_compare + items= → silhouette_scale_compare",
     "scale_compare", "items=Bus:12:city bus|Truck:8:road ; title=TRUE SCALE",
     "Side by side, the true scale of the two is hard to believe.",
     "science", "silhouette_scale_compare", "items", 2),
    ("size_compare + items= → silhouette_scale_compare",
     "size_compare", "items=Whale:30:blue whale|Diver:2:human",
     "One dwarfs the other completely.",
     "science", "silhouette_scale_compare", "items", 2),
    ("route_trace + points= → footage_route_trace",
     "route_trace", "points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End",
     "The convoy traced the long route across the open terrain.",
     "geopolitics", "footage_route_trace", "points", 3),
]


def main():
    fails = 0
    total = 0
    print(f"{'case':52} {'parsed':7} {'fired':28} status")
    for label, gk, gb, narr, niche, want, key, want_n in CASES:
        total += 1
        assets, fired = _fired(gk, gb, narr, niche)
        parsed_ok = isinstance(assets.get(key), list) and len(assets[key]) == want_n
        good = parsed_ok and fired == want
        if not good:
            fails += 1
        print(f"{label:52} {('OK' if parsed_ok else 'BAD'):7} "
              f"{str(fired):28} {'OK' if good else '*** FAIL ***'}")

    # Load-bearing proof: the SAME beat WITHOUT the parsed assets must NOT fire
    # the structured primitive (items/points are never director-derived).
    total += 1
    bare = {"index": 0, "graphic_kind": "scale_compare", "intensity": 3,
            "narration": "Side by side, the true scale is hard to believe.",
            "assets": {}}
    bare_fired = mgdir.plan([bare], niche="science", seed=0)[0].primitive
    bare_ok = bare_fired != "silhouette_scale_compare"
    if not bare_ok:
        fails += 1
    print(f"{'scale_compare WITHOUT items= must NOT fire silhouette':52} "
          f"{'—':7} {str(bare_fired):28} {'OK' if bare_ok else '*** FAIL ***'}")

    # Exact-shape spot-checks (the contracts the two primitives normalise).
    it = _mg_structured_assets("scale_compare",
                               "items=Bus:12:city bus|Truck:8:road")["items"]
    pt = _mg_structured_assets("route_trace",
                               "points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End")["points"]
    checks = [
        ("items[0] == {label,size,note}",
         it[0] == {"label": "Bus", "size": 12.0, "note": "city bus"}),
        ("items size coerced to float", isinstance(it[0]["size"], float)),
        ("points[0] == {x,y,label}",
         pt[0] == {"x": 0.2, "y": 0.7, "label": "Start"}),
        ("points[1] label empty when omitted", pt[1]["label"] == ""),
        ("note keeps trailing colons (split maxsplit=2)",
         _mg_structured_assets("scale_compare", "items=A:5:a:b:c")["items"][0]["note"]
         == "a:b:c"),
        ("wrong graphic_kind parses nothing",
         _mg_structured_assets("timeline", "items=Bus:12:city") == {}),
        ("malformed records dropped, valid kept",
         _mg_structured_assets("scale_compare", "items=NoSize|Good:9")["items"]
         == [{"label": "Good", "size": 9.0, "note": ""}]),
    ]
    for name, ok in checks:
        total += 1
        if not ok:
            fails += 1
        print(f"{name:52} {'':7} {'':28} {'OK' if ok else '*** FAIL ***'}")

    print(f"\n{total - fails}/{total} structured-asset checks passed"
          + ("" if fails == 0 else f"  ({fails} FAILURE(S))"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
