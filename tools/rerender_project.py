#!/usr/bin/env python3
"""Reusable surgical re-render QA tool: re-run match → cut → verify → image
fallbacks → ledger → build on an EXISTING ClipStudio project (sources + index
reused; no re-analyze/discover/download). Use after a matcher/build fix to
validate it on a real project without paying for the full pipeline.

    python3 tools/rerender_project.py /path/to/project_dir [--theme history]

Self-daemonizes (double-fork) so the render survives terminal/harness teardown.
Progress → <project>/output/_rerender.log ; sentinel → <project>/output/_rerender.done
"""
import argparse
import dataclasses
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("project_dir")
ap.add_argument("--theme", default="history")
args = ap.parse_args()
PD = os.path.abspath(args.project_dir)
LOG = os.path.join(PD, "output", "_rerender.log")
DONE = os.path.join(PD, "output", "_rerender.done")
# Machine-readable outcome for whatever drives this job. `_rerender.done` is a human sentinel whose
# only contract is "the process stopped"; a driver needs to know WHY before deciding to retry.
STATUS = os.path.join(PD, "output", "_rerender.status.json")
if not os.path.exists(os.path.join(PD, "project.json")):
    sys.exit(f"not a ClipStudio project dir (no project.json): {PD}")
os.makedirs(os.path.join(PD, "output"), exist_ok=True)

# --- daemonize (double fork + setsid) ---
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
try:
    os.remove(DONE)
except OSError:
    pass
_f = open(LOG, "a", buffering=1)
os.dup2(_f.fileno(), 1)
os.dup2(_f.fileno(), 2)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    from vidlore.clipstudio.models import ClipProject, ScriptSegment
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.config import load_clip_config, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio.cut import cut_all
    from vidlore.clipstudio import verify as _verify
    from vidlore.clipstudio import ledger
    from vidlore.clipstudio.orchestrate import _fill_image_fallbacks, _recover_unresolved_beats
    from vidlore.clipstudio import faceid as _faceid
    from vidlore.clipstudio.build import build_video

    proj = ClipProject.load(PD)
    raw = json.load(open(os.path.join(PD, "project.json")))
    F = {f.name for f in dataclasses.fields(ScriptSegment)}
    segs = [ScriptSegment(**{k: v for k, v in s.items() if k in F}) for s in raw["segments"]]
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    cfg = load_clip_config()
    eng = engine_config()
    log(f"loaded {len(segs)} beats · {len(proj.sources)} sources (reusing index/downloads)")

    faceid_obj, refs = None, {}
    if _faceid.available():
        faceid_obj = _faceid.FaceID()
        refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                        faceid_obj, progress=log)

    log("1/5 match")
    match_segments(proj, segs, cfg, analysis=analysis, progress=log)
    log("2/5 cut")
    cut_all(proj, cfg, progress=log)
    log("3/5 AI verify + repair")
    _verify.verify_and_repair(proj, segs, cfg, eng, progress=log)
    # bounded recovery (R4-5) — same stage order as the full pipeline (verify → recovery →
    # stills). Without it a verifier-rejected beat with no valid hold FATALs the build at the
    # very end ("rediscovery needed") with no rediscovery ever attempted.
    log("3b/5 bounded recovery")
    try:
        _recover_unresolved_beats(proj, segs, analysis, cfg, eng, faceid_obj=faceid_obj,
                                  refs=refs, roster=analysis.actors,
                                  policy=str(proj.meta.get("policy", "") or "block"), log=log)
    except Exception as e:  # noqa: BLE001
        log(f"recovery: skipped ({type(e).__name__}: {e})")
    log("4/5 image fallbacks")
    # eng_cfg enables the vision-verified still path (same as the portal pipeline) — without it
    # the stage silently fell to deterministic-only checks ("no vision model")
    _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log, eng_cfg=eng)
    summ = ledger.finalize(proj, segs, cfg)
    proj.save()
    log(f"QC: {summ['flagged_for_review']}/{summ['segments']} flagged · "
        f"mean conf {summ['mean_confidence']}")

    log("5/5 build")
    vo = os.path.join(PD, "voiceover.mp3")
    # respect the project's persisted caption settings so a re-render keeps the same design
    _capset = (proj.meta.get("caption_settings") or {}) if hasattr(proj, "meta") else {}
    out = build_video(proj, segs, cfg,
                      captions=bool(_capset.get("enabled", True)),
                      caption_style=str(_capset.get("style", "") or ""),
                      title=(analysis.movie_title or proj.name),
                      theme_name=args.theme,
                      voiceover=(vo if os.path.exists(vo) else None),
                      use_tts=True, progress=log)
    log(f"BUILD DONE -> {out}")
    with open(DONE, "w") as fh:
        fh.write(str(out))
    with open(STATUS, "w") as fh:
        json.dump({"status": "ok", "retryable": False, "output": str(out), "project": PD}, fh,
                  indent=1)
except Exception as e:  # noqa: BLE001
    # MACHINE-READABLE OUTCOME. The old sentinel was the string "ERROR: <repr>", which gave a
    # driver no way to tell a content verdict from a network blip — so a release-block looked
    # retryable and the pipeline was restarted 8 times over 11 hours. The 8th attempt "passed" only
    # because the vision API had died by then: 0 verdicts, 0 rejections, 0 unresolved, publish.
    # Scene 25 was never fixed; it stopped being checked.
    #
    # A CONTENT failure is a judgment about this render. Re-running it unchanged cannot help, and
    # retrying until a gate goes quiet is not success — it is the absence of measurement.
    from vidlore.clipstudio.verify import NonRetryableBuildError
    _content = isinstance(e, NonRetryableBuildError)
    log("ERROR: " + repr(e))
    log(traceback.format_exc())
    _status = {
        "status": "content_failure" if _content else "error",
        "retryable": not _content,
        "error_type": type(e).__name__,
        "error": str(e)[:2000],
        "project": PD,
    }
    if _content:
        log("BUILD FAILED (content_failure, retryable=false) — re-running this render unchanged "
            "cannot fix it; it needs rediscovery/re-matching or a working verifier.")
    with open(DONE, "w") as fh:
        fh.write("ERROR: " + repr(e))          # keep the legacy line: existing readers grep it
    with open(STATUS, "w") as fh:
        json.dump(_status, fh, indent=1)
    # NOTE: this process double-forks + setsids above, so the exit code reaches nobody — the parent
    # returned 0 long ago. The STATUS FILE is the only channel a driver can actually read, which is
    # precisely why the outcome had to become a persisted fact rather than a return value.
    sys.exit(2 if _content else 1)
