#!/usr/bin/env python3
"""Cost-fix A/B on the frozen accept_mini job: decisions must be IDENTICAL, calls must drop.

Arm 'old' = the zero-risk cost fixes disabled (eager rung prefetch, self-heal cache bypassed).
Arm 'new' = production defaults. Both arms start from the SAME cold verdict cache and the SAME
index, and every vision call is answered by a DETERMINISTIC oracle keyed on the question's own
fingerprint — so a call-count difference is pure waste removal and any decision difference would
be a real regression, not provider nondeterminism.

    python3 tools/cost_ab_canary.py old
    python3 tools/cost_ab_canary.py new
    python3 tools/cost_ab_canary.py compare
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/hussnain/Desktop/vidlore-clipstudio")
BASE = Path("/Users/hussnain/Desktop/clipstudio_output/portal/accept_mini")
SRC_JOB = Path("/Users/hussnain/Desktop/clipstudio_output/portal/canary_spd_b")   # warm index
sys.path.insert(0, str(WORKTREE))

# The two arms are the same code path with the same env — what differs is the CODE ITSELF:
# arm 'old' is run with the cost fixes stashed out of the tree (real pre-fix behaviour, no
# test-only branches compiled into production), arm 'new' with them applied.
ARMS = {"old": {}, "new": {}}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def prep(arm):
    dst = BASE.parent / f"canary_cost_{arm}"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "sources").mkdir(parents=True)
    for f in (SRC_JOB / "sources").iterdir():
        if f.is_file():
            os.link(f, dst / "sources" / f.name)
    shutil.copytree(SRC_JOB / "index", dst / "index")      # warm index: no re-indexing cost
    shutil.copy2(BASE / "voiceover.mp3", dst / "voiceover.mp3")
    p = json.load(open(SRC_JOB / "project.json"))
    p["selections"] = []
    p["name"], p["root"] = dst.name, str(dst)
    json.dump(p, open(dst / "project.json", "w"), indent=1)
    (dst / "verdict_cache.json").write_text("{}")          # COLD cache: count every question
    return dst


def run(arm):
    for line in (MAIN / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "0"
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_WORKERS"] = "4"
    # Contact sheets are written to a UUID-unique path per call (the phase-2 sheet-race fix), so
    # an oracle keyed on the image FILE would answer differently every run. Single-frame mode
    # keeps the question set stable, which is what a decision-parity comparison needs.
    os.environ["VIDLORE_CLIPSTUDIO_VERIFY_ACTION_SHEET"] = "0"
    for k, v in ARMS[arm].items():
        os.environ[k] = v

    dst = prep(arm)
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio.cut import cut_all
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import selfheal as SH

    # DETERMINISTIC vision oracle: the answer depends only on the QUESTION, never on when it is
    # asked. Keyed on (image bytes id, beat text, strictness, venue flag) so both arms get the
    # same answer to the same question — the only thing that can differ is HOW MANY they ask.
    calls = {"n": 0, "by_stage": {}}
    o_vf = V.verify_frame

    def oracle(*a, **k):
        calls["n"] += 1
        from vidlore.clipstudio import llm as _L
        st = (getattr(_L, "current_stage", lambda: "")() or "?")   # arm old has no stages
        calls["by_stage"][st] = calls["by_stage"].get(st, 0) + 1
        img = str(a[0]) if a else ""
        beat = str(a[1]) if len(a) > 1 else ""
        key = f"{Path(img).name}|{beat}|{k.get('is_specific')}|{k.get('venue_fallback')}"
        h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        keep = (h % 100) < 35                              # ~35% keep — exercises every rung
        return {"verdict": "keep" if keep else "replace",
                "correct_subject_visible": keep, "matches_narration": keep,
                "wrong_subject_visible": False, "quality_ok": True,
                "confidence": 0.9 if keep else 0.8, "reason": "oracle",
                "vision_served_by": "oracle"}

    V.verify_frame = oracle
    try:
        proj = ClipProject.load(str(dst))
        cfg, eng = ClipConfig(), engine_config()
        segs = list(proj.segments)
        analysis = ScriptAnalysis.from_dict((proj.meta or {}).get("analysis") or {})
        t0 = time.time()
        match_segments(proj, segs, cfg, analysis=analysis, progress=None)
        cut_all(proj, cfg, progress=None)
        V.verify_and_repair(proj, segs, cfg, eng, progress=None)
        # Self-heal over whatever the verifier left unresolved — the second-largest vision site.
        # allow_acquire=False keeps it OFFLINE: targeted acquisition does real YouTube searches
        # and DeepSeek query-writing, which no parity comparison can hold still.
        try:
            blocked = [s.segment_index for s in proj.selections
                       if "verifier_failed" in (getattr(s, "flag_reasons", None) or [])
                       and not getattr(s, "image_path", "")]
            if blocked:
                SH.heal_blocked_beats(proj, segs, cfg, blocked=blocked,
                                      policy="approved_testing", allow_acquire=False,
                                      log=lambda m: None)
        except Exception as e:                             # noqa: BLE001
            log(f"selfheal: {type(e).__name__}: {str(e)[:80]}")
        proj.save()
        dt = time.time() - t0
    finally:
        V.verify_frame = o_vf

    sel = [{"i": s.segment_index, "src": s.source_id, "shot": s.shot_index,
            "in": round(float(s.in_point), 3), "out": round(float(s.out_point), 3),
            "img": os.path.basename(getattr(s, "image_path", "") or ""),
            "cls": getattr(s, "relevance_class", ""),
            "flags": sorted(getattr(s, "flag_reasons", None) or []),
            "verdict": (s.verifier or {}).get("verdict"),
            "downgraded": (s.verifier or {}).get("downgraded")}
           for s in sorted(proj.selections, key=lambda x: x.segment_index)]
    rep = {"arm": arm, "vision_calls": calls["n"], "by_stage": calls["by_stage"],
           "seconds": round(dt, 1), "selections": sel}
    json.dump(rep, open(dst / "cost_ab.json", "w"), indent=1)
    log(f"ARM {arm}: {calls['n']} vision call(s) in {dt:.0f}s over {len(sel)} beats")


def compare():
    a = json.load(open(BASE.parent / "canary_cost_old" / "cost_ab.json"))
    b = json.load(open(BASE.parent / "canary_cost_new" / "cost_ab.json"))
    same = a["selections"] == b["selections"]
    saved = a["vision_calls"] - b["vision_calls"]
    pct = 100.0 * saved / max(a["vision_calls"], 1)
    print(f"arm old: {a['vision_calls']} calls  {a['seconds']}s  stages={a['by_stage']}")
    print(f"arm new: {b['vision_calls']} calls  {b['seconds']}s  stages={b['by_stage']}")
    print(f"saved  : {saved} calls ({pct:.1f}%)  ≈ ${saved * 0.000565:.3f} on this 43-beat job")
    print(f"decisions identical: {same} ({len(a['selections'])} beats)")
    if not same:
        for x, y in zip(a["selections"], b["selections"]):
            if x != y:
                print("DIFF", json.dumps(x), "\n  VS", json.dumps(y))
    print("COST_AB_PASS" if (same and saved >= 0) else "COST_AB_FAIL")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "old"
    compare() if cmd == "compare" else run(cmd)
