#!/usr/bin/env python3
"""Measure the REPEAT-pass cost — where the self-heal cache fix actually pays.

Production re-asks the same self-heal questions constantly: up to 3 rounds per pass, and the
whole pass again on a review-draft retry / resume (the cost audit measured ~550-600 re-paid
vision calls on one such cycle). This probe runs the still-recovery heal TWICE over the same
beats and counts the vision calls of the SECOND pass — which should be ~0 once venue verdicts
are cached, and a full re-buy before.

    python3 tools/cost_repeat_probe.py            # prints calls for pass 1 and pass 2
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
SRC_JOB = Path("/Users/hussnain/Desktop/clipstudio_output/portal/canary_spd_b")
sys.path.insert(0, str(WORKTREE))


def _reset_cache(SH):
    """Simulate a fresh process: drop the in-memory venue cache (pre-fix code has none)."""
    if hasattr(SH, "_VV_CACHE"):
        SH._VV_CACHE["root"], SH._VV_CACHE["data"] = "", {}


def main():
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio import selfheal as SH
    from vidlore.clipstudio import verify as V

    dst = Path(tempfile.mkdtemp()) / "repeat"
    (dst / "sources").mkdir(parents=True)
    for f in (SRC_JOB / "sources").iterdir():
        if f.is_file():
            os.link(f, dst / "sources" / f.name)
    shutil.copytree(SRC_JOB / "index", dst / "index")
    p = json.load(open(SRC_JOB / "project.json"))
    p["name"], p["root"] = "repeat", str(dst)
    json.dump(p, open(dst / "project.json", "w"), indent=1)
    (dst / "verdict_cache.json").write_text("{}")

    proj = ClipProject.load(str(dst))
    segs = list(proj.segments)
    # force a handful of beats to look unresolved so the heal pass has real work
    blocked = []
    for s in proj.selections[:8]:
        s.image_path = ""
        s.flag_reasons = list(set((s.flag_reasons or []) + ["verifier_failed"]))
        s.verifier = {"status": "ok", "verdict": "replace"}
        blocked.append(s.segment_index)

    calls = [0]
    o_vf = V.verify_frame
    from vidlore.clipstudio.config import engine_config
    _eng = engine_config()
    # stamp the REAL model id: _hit_provider_ok refuses to serve a verdict whose recorded
    # server differs from the current model, so a fake provider would (correctly) make every
    # cache lookup miss and measure nothing
    _served = SH._vision_model(_eng) if hasattr(SH, "_vision_model") else ""

    def oracle(*a, **k):
        calls[0] += 1
        return {"verdict": "replace", "correct_subject_visible": False,
                "matches_narration": False, "wrong_subject_visible": False,
                "quality_ok": True, "confidence": 0.8, "reason": "o",
                "vision_served_by": _served}          # never keeps → full candidate window

    V.verify_frame = oracle
    try:
        _reset_cache(SH)
        SH.heal_blocked_beats(proj, segs, ClipConfig(), blocked=blocked,
                              policy="approved_testing", allow_acquire=False, log=lambda m: None)
        first = calls[0]
        # pass 2: the SAME questions (another round / the review-draft retry)
        for s in proj.selections[:8]:
            s.image_path = ""
        _reset_cache(SH)                       # a fresh process would reload from disk too
        SH.heal_blocked_beats(proj, segs, ClipConfig(), blocked=blocked,
                              policy="approved_testing", allow_acquire=False, log=lambda m: None)
        second = calls[0] - first
    finally:
        V.verify_frame = o_vf
    print(f"pass 1: {first} vision call(s)")
    print(f"pass 2 (identical questions): {second} vision call(s)")
    print(f"repeat-pass saving: {first - second} call(s) "
          f"({100.0 * (first - second) / max(first, 1):.0f}%)")


if __name__ == "__main__":
    main()
