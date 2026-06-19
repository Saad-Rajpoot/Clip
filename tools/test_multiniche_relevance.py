"""Multi-niche relevance regression — the GENERALIZED safeguard, not one video.

Scores the known wrong assets (war crowd, soldiers, preacher collage, modern
road) against narration/subjects from many niches and asserts they are REJECTED
on every off-topic niche, while a genuinely war/history scene KEEPS soldiers and
a relevant agriculture clip is KEPT. Run after any scorer change:

    VIDLORE_VISUAL_RELEVANCE=1 python3 tools/test_multiniche_relevance.py

Stable assets live in research/visual_relevance/test_assets/ (copied from a real
render so the test is reproducible without re-fetching footage).

DETERMINISM NOTE: the scorer runs in single-thread sequential-CPU mode here
(VIDLORE_VR_DETERMINISTIC=1), so any one asset/scene scores bit-identically on
repeat calls. The HARD requirement of this gate is that the universally-wrong
assets — the war crowd, the vintage soldiers, the preacher collage — REJECT on
every off-topic niche (they score well above the gates: war-crowd dom~0.056,
ppl~0.063, war~0.047). Those are rock-solid. Three synthetic "modern road in a
pre-1945 scene" / borderline-period cases sit right on the CLIP margin (dom
~0.03 vs a 0.05 gate) and can flip ±1 across processes due to onnxruntime
first-inference warmup; in REAL renders the period guard (era context) plus the
war/people probes are the backstops for those. Treat <13 as a soft warn when
only the modern-road/period rows differ; treat any war/soldier/collage KEEP as a
hard regression.
"""
import os
import sys

# Reproducible scorer for the gate — single-thread sequential CPU removes the
# CoreML/multi-thread floating-point variance that otherwise tips borderline
# assets across the relevance threshold run-to-run. MUST be set before the
# scorer's onnxruntime session is created (i.e. before the first score).
os.environ.setdefault("VIDLORE_VR_DETERMINISTIC", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import footage as F  # noqa: E402

TA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "research", "visual_relevance", "test_assets")
WAR = os.path.join(TA, "wrong_war_crowd.mp4")
SOL = os.path.join(TA, "wrong_soldiers.mp4")
COLLAGE = os.path.join(TA, "wrong_preacher_collage.jpg")
ROAD = os.path.join(TA, "wrong_modern_road.mp4")
GOOD = os.path.join(TA, "good_copper_soil.mp4")


class _Scene:
    def __init__(self, narration, keywords, visual="", role="evidence", index=1):
        self.narration = narration
        self.keywords = keywords
        self.visual = visual
        self.role = role
        self.index = index


class _Item:
    def __init__(self, path, is_video=True):
        self.path = path
        self.is_video = is_video


def judge(asset, scene, is_video=True):
    ok, s, why = F._vr_judge(scene, _Item(asset, is_video))
    return ok, s, why


# (label, asset, scene, want_reject)
CASES = [
    # ── NEGATIVE: universally-wrong assets on off-topic niches must REJECT ──
    ("metals → war crowd", WAR, _Scene(
        "two of the cheapest metals on earth, copper and zinc you can hold",
        ["copper pennies", "zinc nails", "open palm", "workbench"]), True),
    ("agriculture → soldiers", SOL, _Scene(
        "not a single beetle on the cabbage, the copper barrier held",
        ["cabbage", "beetle", "copper barrier", "garden bed"]), True),
    ("Cornell study → preacher collage", COLLAGE, _Scene(
        "Cornell University confirmed a 96% reduction in a controlled study",
        ["university study", "copper slug barrier", "garden research"]), True),
    ("Amish farm → modern road", ROAD, _Scene(
        "the Amish families in Lancaster County farm the old way",
        ["amish farmer", "lancaster county", "farmland", "straw hat"]), True),
    ("spy doc → war crowd", WAR, _Scene(
        "the classified dossier crossed the embassy desk at midnight",
        ["classified document", "embassy", "dossier", "desk"], role="archive"), True),
    ("true crime → preacher collage", COLLAGE, _Scene(
        "the crime family ran the docks through fear and silence",
        ["mafia family", "docks", "police archive"], role="archive"), True),
    ("business → soldiers", SOL, _Scene(
        "the refinery turned crude oil into a financial empire",
        ["oil refinery", "factory", "boardroom", "railway"]), True),
    ("science → war crowd", WAR, _Scene(
        "inside the laboratory the chemical reaction released heat",
        ["laboratory", "chemical reaction", "experiment", "beaker"]), True),
    ("health → preacher collage", COLLAGE, _Scene(
        "the cells in the bloodstream divided and repaired the tissue",
        ["bloodstream", "cells", "tissue", "microscope"]), True),
    ("technology → modern road on a 1850s scene", ROAD, _Scene(
        "in the 1850s the first telegraph wires crossed the prairie",
        ["telegraph", "1850s", "wires", "prairie"], role="archive"), True),

    # ── POSITIVE: must KEEP ──
    ("war/history → soldiers (on topic)", SOL, _Scene(
        "Napoleon's army marched into the battle, thousands of soldiers",
        ["napoleon", "army", "soldiers", "battle", "formation"],
        role="archive"), False),
    ("WWII → soldiers (on topic)", SOL, _Scene(
        "the infantry advanced under fire across the war-torn field",
        ["wwii", "infantry", "war", "soldiers"], role="archive"), False),
    ("agriculture → copper soil (relevant)", GOOD, _Scene(
        "a weathered copper strip pressed into the garden soil for years",
        ["copper strip", "garden soil", "metal barrier", "plants"]), False),
]


def main():
    missing = [p for p in (WAR, SOL, COLLAGE, ROAD, GOOD) if not os.path.exists(p)]
    if missing:
        print("MISSING test assets:", missing)
        return 2
    # Simulate a historical render so the period-risk gate is live (in a real
    # render the period-guard sets this; the modern-road cases are pre-1945).
    F._VIDEO_CTX["era"] = {"year": 1860, "label": "industrial", "confidence": 0.8}
    npass = 0
    for label, asset, scene, want_reject in CASES:
        is_vid = not asset.endswith((".jpg", ".png", ".jpeg"))
        ok, s, why = judge(asset, scene, is_vid)
        rejected = not ok
        good = (rejected == want_reject)
        npass += good
        verdict = "REJECT" if rejected else "keep"
        want = "REJECT" if want_reject else "keep"
        print(f"  [{'PASS' if good else 'FAIL'}] {label:42s} -> {verdict:6s} "
              f"(want {want:6s})  dom={s.get('distractor_dom')} "
              f"ppl={s.get('people_dom')} why={why[:30]}")
    print(f"\n{npass}/{len(CASES)} multi-niche relevance cases passed")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
