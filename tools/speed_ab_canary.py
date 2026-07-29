#!/usr/bin/env python3
"""Speed-pass decision-parity A/B on the frozen accept_mini job (43 beats, 33 sources).

Arm 'a' = every speed kill-switch OFF (serial paths). Arm 'b' = production defaults.
Same code, same inputs, same verdict cache — any decision diff is caused by the parallel
paths. Run arm a first; arm b is seeded with arm a's POST-run verdict cache so uncached
vision questions replay a's verdicts instead of re-asking the API (removes API
nondeterminism from the comparison). Breakouts OFF in both arms (their dialogue-classifier
LLM calls are nondeterministic between runs and are not touched by the speed pass).

    python3 tools/speed_ab_canary.py a
    python3 tools/speed_ab_canary.py b
    python3 tools/speed_ab_canary.py compare
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
sys.path.insert(0, str(WORKTREE))

ARM_ENVS = {
    "a": {  # serial: pre-speed-pass behavior
        "VIDLORE_CLIPSTUDIO_FLAGS_FAST": "0",
        "VIDLORE_CLIPSTUDIO_KF_PREEXTRACT": "0",
        "VIDLORE_CLIPSTUDIO_OCR_POOL_OK": "0",
        "VIDLORE_CLIPSTUDIO_INDEX_OVERLAP": "0",
        "VIDLORE_CLIPSTUDIO_ADSCAN_WORKERS": "1",
        "VIDLORE_CLIPSTUDIO_QA_SWEEP_WORKERS": "1",
        "VIDLORE_CLIPSTUDIO_VERIFY_WORKERS": "1",
        "VIDLORE_CLIPSTUDIO_SELFHEAL_VERIFY_WORKERS": "1",
        "VIDLORE_CLIPSTUDIO_EARLY_RF_GATE": "0",
    },
    "b": {  # production defaults (the speed pass)
        "VIDLORE_CLIPSTUDIO_OCR_POOL_OK": "1",
        "VIDLORE_CLIPSTUDIO_VERIFY_WORKERS": "4",
    },
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def prep(arm: str) -> Path:
    dst = BASE.parent / f"canary_spd_{arm}"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "sources").mkdir(parents=True)
    for f in (BASE / "sources").iterdir():
        if f.is_file():
            os.link(f, dst / "sources" / f.name)
    shutil.copy2(BASE / "voiceover.mp3", dst / "voiceover.mp3")
    shutil.copy2(BASE / "verdict_cache.json", dst / "verdict_cache.json")
    (dst / "index").mkdir()
    if (BASE / "index" / "face_refs").is_dir():
        shutil.copytree(BASE / "index" / "face_refs", dst / "index" / "face_refs")
    p = json.load(open(BASE / "project.json"))
    p["selections"] = []
    p["name"] = dst.name
    p["root"] = str(dst)
    json.dump(p, open(dst / "project.json", "w"), indent=1)
    return dst


def run(arm: str):
    for _line in (MAIN / ".env").read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            k, v = _line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault("VIDLORE_MUSIC_DIR", str(MAIN / "vidlore" / "assets" / "music"))
    os.environ.setdefault("VIDLORE_HD_PYTHON", str(MAIN / ".hdvenv" / "bin" / "python"))
    os.environ.setdefault("VIDLORE_HD_POT_DIR", str(MAIN / ".pot" / "server"))
    os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "0"
    os.environ.pop("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", None)
    for k, v in ARM_ENVS[arm].items():
        os.environ[k] = v

    dst = prep(arm)
    if arm == "b":
        seed = BASE.parent / "canary_spd_a" / "verdict_cache.json"
        if seed.exists():
            shutil.copy2(seed, dst / "verdict_cache.json")
            log("arm b: seeded with arm a's post-run verdict cache")

    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.index import index_all
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio.cut import cut_all
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import faceid as F
    from vidlore.clipstudio.build import build_video

    proj = ClipProject.load(str(dst))
    cfg = ClipConfig()
    eng = engine_config()
    segs = list(proj.segments)
    analysis = (proj.meta or {}).get("analysis") or {}
    times = {}

    t0 = time.time()
    faceid_obj, refs = None, {}
    if F.available():
        faceid_obj = F.FaceID()
        idents = [{"name": c.get("name", ""), "kind": "character",
                   "actor": c.get("actor", "")} for c in (analysis.get("characters") or [])]
        refs = F.build_references(idents, proj.index_dir, faceid_obj, progress=None)
    times["refs"] = time.time() - t0

    t0 = time.time()
    index_all(proj, cfg, references=refs, faceid=faceid_obj,
              roster=[c.get("actor", "") for c in (analysis.get("characters") or [])],
              progress=lambda m: None)
    times["index"] = time.time() - t0
    log(f"index done in {times['index']:.0f}s")

    t0 = time.time()
    match_segments(proj, segs, cfg, analysis=analysis, progress=None)
    times["match"] = time.time() - t0
    t0 = time.time()
    cut_all(proj, cfg, progress=None)
    times["cut"] = time.time() - t0
    log(f"match {times['match']:.0f}s cut {times['cut']:.0f}s")

    t0 = time.time()
    V.verify_and_repair(proj, segs, cfg, eng, progress=None)
    times["verify"] = time.time() - t0
    proj.save()
    log(f"verify done in {times['verify']:.0f}s")

    t0 = time.time()
    out = None
    err = ""
    try:
        out = build_video(proj, segs, cfg, captions=True, title="canary",
                          theme_name="history", voiceover=str(dst / "voiceover.mp3"),
                          use_tts=True, progress=lambda m: None)
    except Exception as e:                                # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
    times["build"] = time.time() - t0
    log(f"build done in {times['build']:.0f}s out={out} err={err}")

    sel = []
    for s in sorted(proj.selections, key=lambda x: x.segment_index):
        sel.append({"i": s.segment_index, "src": s.source_id,
                    "in": round(float(s.in_point), 3), "out": round(float(s.out_point), 3),
                    "img": os.path.basename(getattr(s, "image_path", "") or ""),
                    "cls": getattr(s, "relevance_class", ""),
                    "flags": sorted(getattr(s, "flag_reasons", None) or [])})
    report = {"arm": arm, "times": {k: round(v, 1) for k, v in times.items()},
              "total_s": round(sum(times.values()), 1),
              "selections": sel, "build_error": err,
              "srt_md5": (_md5(dst / "output" / "final.srt")
                          if (dst / "output" / "final.srt").exists() else ""),
              "rf_audit": (json.load(open(dst / "output" / "rejected_footage_audit.json"))
                           if (dst / "output" / "rejected_footage_audit.json").exists()
                           else None)}
    json.dump(report, open(dst / "ab_report.json", "w"), indent=1)
    log(f"ARM {arm} COMPLETE total={report['total_s']}s")


def compare():
    a = json.load(open(BASE.parent / "canary_spd_a" / "ab_report.json"))
    b = json.load(open(BASE.parent / "canary_spd_b" / "ab_report.json"))
    same_sel = a["selections"] == b["selections"]
    same_srt = a["srt_md5"] == b["srt_md5"] and a["srt_md5"]
    same_err = a["build_error"] == b["build_error"]
    same_rf = a["rf_audit"] == b["rf_audit"]
    print(f"arm a total={a['total_s']}s  stages={a['times']}")
    print(f"arm b total={b['total_s']}s  stages={b['times']}")
    print(f"speedup={a['total_s'] / max(b['total_s'], 0.1):.2f}x")
    print(f"selections identical: {same_sel} ({len(a['selections'])} beats)")
    print(f"srt identical: {bool(same_srt)}  build_error identical: {same_err}  "
          f"rf_audit identical: {same_rf}")
    if not same_sel:
        for x, y in zip(a["selections"], b["selections"]):
            if x != y:
                print("DIFF", json.dumps(x), "VS", json.dumps(y))
    print("AB_DECISION_PARITY_PASS" if (same_sel and same_err and same_rf)
          else "AB_DECISION_PARITY_FAIL")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "a"
    if cmd == "compare":
        compare()
    else:
        run(cmd)
