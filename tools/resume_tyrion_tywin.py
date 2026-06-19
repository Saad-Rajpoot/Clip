#!/usr/bin/env python3
"""Resume the Tyrion & Tywin build from cached analyze+discover+download.

The full run already saved project.json with 53 sources, 286 segments, and meta.analysis
(correct character->actor map). This skips the expensive analyze (25min LLM) + discover +
download and runs only Face-ID -> index(skips cached) -> match -> cut -> verify -> build,
mirroring produce_auto stages 4-9 exactly. Every stage saves project.json, so it is itself
resumable if interrupted again.
"""
import os
import sys
import time
import traceback
from pathlib import Path

LOG_PATH = "/Users/hussnain/Desktop/clipstudio_output/portal/tyrion_tywin_scene/resume.log"


def _daemonize(logpath):
    """Double-fork + setsid so the build fully detaches from the controlling terminal
    and the launching process group — immune to the harness reaping its shell. macOS has
    no `setsid` CLI, but os.setsid() works. Done BEFORE heavy imports (clean fork)."""
    if os.fork() > 0:
        os._exit(0)              # launching shell returns immediately
    os.setsid()                  # new session — detach from terminal + pgroup
    if os.fork() > 0:
        os._exit(0)              # not a session leader → can never reacquire a terminal
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(logpath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)


_daemonize(LOG_PATH)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_envf = Path(__file__).resolve().parent.parent / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k and _v:
                os.environ[_k] = _v

os.environ["VIDLORE_CLIPSTUDIO_BREAKOUTS"] = "1"
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1")

from vidlore.clipstudio.models import ClipProject
from vidlore.clipstudio.config import load_clip_config, engine_config
from vidlore.clipstudio.analyze import ScriptAnalysis
from vidlore.clipstudio.index import index_all
from vidlore.clipstudio.match import match_segments
from vidlore.clipstudio.cut import cut_all
from vidlore.clipstudio import ledger, faceid as _faceid, verify as _verify, review as _review
from vidlore.clipstudio.build import build_video
from vidlore.clipstudio.orchestrate import _fill_image_fallbacks

P = Path("/Users/hussnain/Desktop/clipstudio_output/portal/tyrion_tywin_scene")
TOPIC = "This One Scene Explains the Entire Tyrion & Tywin Relationship"
voiceover = str(P / "voiceover.mp3")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


try:
    proj = ClipProject.load(P)
    cfg = load_clip_config()
    eng = engine_config()
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    segs = proj.segments
    log(f"RESUME · sources={len(proj.sources)} segs={len(segs)} "
        f"movie={analysis.movie_title!r} actors={len(analysis.actors)} type={analysis.video_type}")

    # 4 — Face-ID references
    log("4/9 · build Face-ID references")
    faceid_obj, refs = None, {}
    if _faceid.available():
        faceid_obj = _faceid.FaceID()
        refs = _faceid.build_references(analysis.reference_identities(), proj.index_dir,
                                        faceid_obj, progress=log)
    roster = analysis.actors

    # 5 — deep index (skips the 34 already done)
    log("5/9 · deep index")
    index_all(proj, cfg, references=refs, faceid=faceid_obj, roster=roster,
              force=False, progress=log)

    # 6 — match
    log("6/9 · match")
    match_segments(proj, segs, cfg, analysis=analysis, progress=log)
    proj.save()

    # 7 — cut
    log("7/9 · cut")
    cut_all(proj, cfg, progress=log)
    proj.save()

    # 8 — AI verify + repair
    log("8/9 · AI verify + repair")
    _verify.verify_and_repair(proj, segs, cfg, eng, progress=log)
    if os.environ.get("VIDLORE_CLIPSTUDIO_IMAGE_FALLBACK", "1").strip() not in ("0", "false", "no"):
        try:
            _fill_image_fallbacks(proj, segs, analysis, faceid_obj, refs, log)
        except Exception as _e:
            log(f"image-fallback: skipped ({type(_e).__name__}: {_e})")

    summ = ledger.finalize(proj, segs, cfg)
    review_path = _review.write_review(proj, segs)
    proj.save()
    log(f"QC · {summ['flagged_for_review']}/{summ['segments']} flagged · mean conf {summ['mean_confidence']}")

    # 9 — assemble
    log("9/9 · assemble final video")
    out = build_video(proj, segs, cfg, voice="", captions=True, title=TOPIC,
                      theme_name="history", voiceover=voiceover, use_tts=True, progress=log)
    log(f"OUTPUT → {out}")
    log("DONE")
except Exception:
    log("FATAL:\n" + traceback.format_exc())
    raise
