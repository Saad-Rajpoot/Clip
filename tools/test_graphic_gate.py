"""Designed-graphic relevance gate — weak/empty-keyword regression.

The seven-niche sweep (research/editorial_qa/fal_niche_sweep/) found that on
scenes with weak/empty `keywords` the visual-relevance gate ACCEPTED off-topic
DESIGNED GRAPHICS — a Democrat/Republican party-logo clip-art, a "POLYSEXUAL /
PANSEXUAL" text image, a modern mortgage-rates infographic — because the
expected subject was too vague to reject them and no gate looked for "is this a
graphic instead of footage?".  visual_relevance now carries a keyword-
INDEPENDENT designed-graphic probe (graphic_dom = how much more the frame looks
like an infographic/chart/logo/clip-art/poster/text image than a real
photograph).  This test pins the behaviour with the EXACT assets from the sweep
renders:

    VIDLORE_VISUAL_RELEVANCE=1 python3 tools/test_graphic_gate.py

HARD requirements (any failure = regression):
  • every wrong graphic REJECTS — even with the weak 2-word subject the sweep
    actually used, AND in guard-only (abstract/stat-card-background) mode, since
    that is where an off-topic infographic lands as a motion-graphics card bg.
  • every good photo / real-footage clip is KEPT (no false-reject).

Assets live in research/visual_relevance/test_assets/ (copied from the sweep so
the test is reproducible without re-fetching). Determinism: single-thread
sequential CPU removes onnxruntime float-accumulation variance.
"""
import os
import sys

os.environ.setdefault("VIDLORE_VR_DETERMINISTIC", "1")
os.environ.setdefault("VIDLORE_VISUAL_RELEVANCE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore import visual_relevance as VR  # noqa: E402

TA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "research", "visual_relevance", "test_assets")


def _p(name):
    return os.path.join(TA, name)


# (label, file, is_video, weak_expected_subject, want_reject, also_guard)
# weak_expected_subject == the vague 2-word fragment the sweep auto-derived for
# that scene — proving the gate does NOT depend on strong keywords.
# also_guard: also assert REJECT in guard-only (abstract / stat-card-background)
# mode. True for the designed-graphics the CLIP graphic_dom probe owns (an
# infographic/logo/text image is wrong even on an abstract beat). The cartoon is
# caught by the CLARITY gate (its graphic_dom is photo-ish), which — by design —
# only applies to concrete scenes (guard-only preserves soft atmospheric b-roll),
# so it is checked concrete-only; in practice its scene IS concrete.
CASES = [
    # --- wrong DESIGNED GRAPHICS (must REJECT, keyword-independent) ---
    ("party-logo clip-art",      "wrong_party_logos.jpg",          False, "returning political",  True,  True),
    ("POLYSEXUAL text image",    "wrong_polysexual_text.jpg",      False, "tightening foreign",   True,  True),
    ("'Politics' text sign",     "wrong_politics_sign.jpg",        False, "returning political",  True,  True),
    ("modern mortgage infographic", "wrong_mortgage_infographic.mp4", True, "war confident decisive", True, True),
    ("cartoon clip-art",         "wrong_cartoon_clipart.jpg",      False, "bonaparte commander",  True,  False),
    # --- good photos / real footage (must KEEP) ---
    ("fal character portrait",   "good_fal_portrait.jpg",          False, "returning political",  False, False),
    ("real archival photo",      "good_archival_photo.jpg",        False, "army city continent",  False, False),
    ("real snowy-soldiers clip", "good_snowy_soldiers.mp4",        True,  "deliberate advancing", False, False),
    ("real field clip",          "good_field.mp4",                 True,  "deliberate advancing", False, False),
]


def _verdict(path, is_video, expected, guard_only):
    # crowd_ok / vehicle_ok True so this test isolates the GRAPHIC gate and never
    # trips the war/people/vehicle probes on the real-footage control clips.
    ok, s, why = VR.accept(path, is_video, expected=expected,
                           concrete=not guard_only, guard_only=guard_only,
                           crowd_ok=True, vehicle_ok=True)
    return ok, s.get("graphic_dom"), why


def main():
    if not VR.available():
        print("visual_relevance unavailable (model missing / flag off) — "
              "cannot run; treating as SKIP.")
        return 0
    fails = 0
    total = 0
    print(f"{'case':30} {'mode':6} {'verdict':7} {'gdom':>7}  reason")
    for label, fn, is_video, exp, want_reject, also_guard in CASES:
        path = _p(fn)
        if not os.path.exists(path):
            print(f"{label:30} MISSING ASSET {path}")
            fails += 1
            total += 1
            continue
        modes = [("concrete", False)]
        if also_guard:
            modes.append(("guard", True))
        for mode_name, guard in modes:
            total += 1
            ok, gdom, why = _verdict(path, is_video, exp, guard)
            rejected = not ok
            good = (rejected == want_reject)
            status = "OK" if good else "*** REGRESSION ***"
            if not good:
                fails += 1
            print(f"{label:30} {mode_name:6} "
                  f"{'REJECT' if rejected else 'keep':7} {str(gdom):>7}  "
                  f"{status} {why[:44]}")
    print(f"\n{total - fails}/{total} graphic-gate checks passed"
          + ("" if fails == 0 else f"  ({fails} REGRESSION(S))"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
