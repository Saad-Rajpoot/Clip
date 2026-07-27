"""P4 source screening: listicle titles, numeral-overlay source gate, static_frac plumbing,
breakout cursor probe wiring.

Calibration evidence (job 5462677f95): listicle source carried lone-digit OCR on 75% of its
shots vs <=2.7% (OCR noise) everywhere else; the screen-recording source burned a white mouse
cursor (frozen 255/std-0 core) into a 12.6s breakout.

    python3 tests/test_source_screening_p4.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidlore.clipstudio.discover import is_unwanted_source_title as unw   # noqa: E402
from vidlore.clipstudio.match import _shot_numeral_overlay                # noqa: E402
from vidlore.clipstudio.models import Shot                                # noqa: E402
from vidlore.clipstudio.index import _flags_from_frames                   # noqa: E402

FAILS = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


def _shot(**kw):
    base = dict(source_id="s", index=0, start=0.0, end=4.0)
    base.update(kw)
    return Shot(**base)


def test_listicle_titles_rejected():
    for t in ("Game Of Thrones Purple Wedding - Top 5 Suspects (Who Poisoned Joffrey?)",
              "Top 10 Game of Thrones Moments RANKED",
              "5 Things You Missed at the Purple Wedding",
              "Game of Thrones Hidden Details in the Red Wedding",
              "The True Story of the Red Wedding | Video Essay"):
        check(f"listicle/essay title rejected: {t[:44]!r}", unw(t))


def test_scene_titles_still_pass():
    for t in ("S4E2 Game of Thrones: Joffrey and Margaery gets married (Purple Wedding Part 1/4)",
              "Game of Thrones - King Joffreys Death (Poisoned at his wedding) + BONUS Scene",
              "The Mountain vs Oberyn Martell Full Fight 1080p",
              "Game of Thrones 4x02 Joffrey death scene",
              "Sansa Stark scene pack | Game of thrones season four",
              "Tyrion Top form at his trial"):
        check(f"legit scene title passes: {t[:44]!r}", not unw(t))


def test_numeral_overlay_shot_rule():
    check("lone digit fires", _shot_numeral_overlay(_shot(ocr_text="3")))
    check("digit with punctuation fires", _shot_numeral_overlay(_shot(ocr_text=" #2. ")))
    check("two-digit fires", _shot_numeral_overlay(_shot(ocr_text="10)")))
    check("digit inside words does not fire",
          not _shot_numeral_overlay(_shot(ocr_text="Season 4 Episode 2")))
    check("empty OCR does not fire", not _shot_numeral_overlay(_shot(ocr_text="")))
    check("3-digit number does not fire (timestamp/year class)",
          not _shot_numeral_overlay(_shot(ocr_text="447")))


def test_static_frac_computed_and_persisted():
    import numpy as np
    a = np.full((360, 640), 100.0, dtype="float32")
    b = a.copy()
    c = a + 30.0                                          # a real cut/motion
    out = _flags_from_frames([a, b, b])
    check("identical samples -> static_frac 1.0", out.get("static_frac") == 1.0)
    out2 = _flags_from_frames([a, c, a])
    check("moving samples -> static_frac 0.0", out2.get("static_frac") == 0.0)
    out3 = _flags_from_frames([a])
    check("single sample -> unknown (-1)", out3.get("static_frac") == -1.0)
    check("Shot model persists static_frac (default -1)",
          _shot().static_frac == -1.0 and "static_frac" in _shot().to_dict())


def test_numeral_source_gate_wired_into_pool_load():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vidlore", "clipstudio", "match.py")).read()
    check("pool-load drops numeral-overlay sources",
          '_reject(src.id, "numeral_overlay")' in src)
    check("gate is source-level (>=3 hits + frac floor)",
          "_n_num >= 3" in src and "VIDLORE_CLIPSTUDIO_NUMERAL_SRC_FRAC" in src)
    osrc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "vidlore", "clipstudio", "orchestrate.py")).read()
    check("backfill treats numeral_overlay as a replaceable QUALITY reject",
          '"numeral_overlay"' in osrc)


def test_static_collage_gate_calibrated():
    from vidlore.clipstudio.match import _shot_static_collage, _slideshow_source_verdict
    # FREEZE tier keys on pair_diff_max: a frozen digital card is zero-diff at ANY resolution
    check("frozen digital card gated (pmax 0.00)",
          _shot_static_collage(_shot(pair_diff_max=0.0, luma_avg=35.3)))
    check("uncorroborated frozen card still gated",
          _shot_static_collage(_shot(pair_diff_max=0.05, luma_avg=40.0, graphics_flag=0,
                                     ocr_text="", faces=1)))
    # the HD calibration-transfer FP: real slow dark scene (Ramsay bedroom pmax 0.61) SPARED
    check("HD slow real footage NOT gated (pmax 0.61 — the Ramsay FP)",
          not _shot_static_collage(_shot(pair_diff_max=0.61, luma_avg=14.9, static_frac=1.0)))
    # near-black protected by the luma guard
    check("near-black shot NOT gated",
          not _shot_static_collage(_shot(pair_diff_max=0.0, luma_avg=3.2)))
    # STILL tier needs graphics/text corroboration AND near-frozen diffs
    check("near-frozen art card + graphics band gated",
          _shot_static_collage(_shot(pair_diff_max=0.3, luma_avg=60.0, graphics_flag=1)))
    check("slow real shot without corroboration NOT gated",
          not _shot_static_collage(_shot(pair_diff_max=1.8, luma_avg=60.0,
                                         graphics_flag=0, ocr_text="", faces=1)))
    check("faces alone never corroborate (HD two-person dialogue FP)",
          not _shot_static_collage(_shot(pair_diff_max=0.3, luma_avg=60.0,
                                         graphics_flag=0, ocr_text="", faces=2)))
    check("slow-but-not-frozen band shot NOT gated (HD slow scene)",
          not _shot_static_collage(_shot(pair_diff_max=1.8, luma_avg=60.0, graphics_flag=1)))
    check("normal footage NOT gated",
          not _shot_static_collage(_shot(pair_diff_max=25.0, luma_avg=60.0)))
    check("old index (-1 sentinels) fails open", not _shot_static_collage(_shot()))
    # SLIDESHOW verdict now needs corroboration: slowness alone does not transfer to HD
    # ('When Roses' 8 hard-gfx vs Cersei-dinner 0 hard at the same slowness)
    essay = [_shot(index=i, pair_diff_mean=2.0, pair_diff_max=25.0,
                   graphics_flag=(2 if i < 4 else 0)) for i in range(18)]
    hd_dialogue = [_shot(index=i, pair_diff_mean=2.0, pair_diff_max=25.0,
                         graphics_flag=0) for i in range(18)]
    check("AI-art essay flagged (slow + >=3 hard gfx)",
          _slideshow_source_verdict(essay))
    check("HD slow dialogue scene SPARED (slow but 0 hard gfx — the Cersei-dinner FP)",
          not _slideshow_source_verdict(hd_dialogue))
    frozen_cards = [_shot(index=i, pair_diff_mean=2.0,
                          pair_diff_max=(0.0 if i < 4 else 25.0), luma_avg=40.0)
                    for i in range(18)]
    check("card-deck slideshow flagged via >=3 frozen cards",
          _slideshow_source_verdict(frozen_cards))
    check("small sources never qualify (<12 measured)",
          not _slideshow_source_verdict(essay[:8]))
    old = [_shot(index=i) for i in range(20)]
    check("old index never qualifies", not _slideshow_source_verdict(old))


def test_cursor_source_gate_wired_into_pool_load():
    m = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "vidlore", "clipstudio", "match.py")).read()
    check("pool-load probes whole sources for a burned cursor",
          '_reject(src.id, "screen_recording")' in m
          and "VIDLORE_CLIPSTUDIO_CURSOR_SRC_GATE" in m)
    check("verdict is cached per source on the project",
          '"cursor_scan"' in m)
    o = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "vidlore", "clipstudio", "orchestrate.py")).read()
    check("backfill treats screen_recording as replaceable", '"screen_recording"' in o)


def test_cursor_probe_wired_into_breakout_qc():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vidlore", "clipstudio", "build.py")).read()
    check("breakout window-QC consults the cursor probe",
          "_breakout_cursor_probe(v, float(real))" in src)
    check("cursor probe has a kill switch",
          "VIDLORE_CLIPSTUDIO_BREAKOUT_CURSOR_GATE" in src)
    check("probe demands a solid-white frozen core (rim-light FP guard)",
          "_core < 3" in src)
    check("probe skips the opening dissolve region", "(0.30, 0.50, 0.70, 0.90)" in src)


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        print(f"[{fn}]")
        globals()[fn]()
    print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
