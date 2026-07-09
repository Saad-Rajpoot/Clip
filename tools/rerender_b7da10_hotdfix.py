#!/usr/bin/env python3
"""Re-render b7da10d8 after the House-of-the-Dragon cross-show fix.
Reuses the existing pool/index/selections (beats 85 & 33 already swapped to GoT and
re-cut in project.json); build_video re-selects breakouts under the new wrong-show
gate (drops the Aemond cold-open) and re-assembles. Self-daemonizes so the harness
can't reap it. Progress -> output/_rerender.log ; sentinel -> output/_rerender.done
"""
import os, sys, json, time, dataclasses, traceback

REPO = "/Users/hussnain/Desktop/vidlore-clipstudio"
PD = "/Users/hussnain/Desktop/clipstudio_output/portal/b7da10d81e"
LOG = PD + "/output/_rerender.log"
DONE = PD + "/output/_rerender.done"
sys.path.insert(0, REPO)

# --- daemonize (double-fork + setsid) so the render survives tool teardown ---
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
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_ALLOW_NO_CLIP", "")  # keep CLIP fail-closed default
    from vidlore.clipstudio.models import ClipProject, ScriptSegment
    from vidlore.clipstudio.analyze import ScriptAnalysis
    from vidlore.clipstudio.config import load_clip_config
    from vidlore.clipstudio.build import build_video

    log("loading project (reusing pool/index/selections; beats 85 & 33 already swapped to GoT)")
    proj = ClipProject.load(PD)
    raw = json.load(open(PD + "/project.json"))
    F = {f.name for f in dataclasses.fields(ScriptSegment)}
    segs = [ScriptSegment(**{k: v for k, v in s.items() if k in F}) for s in raw["segments"]]
    analysis = ScriptAnalysis.from_dict(proj.meta["analysis"])
    cfg = load_clip_config()

    log("build_video START (breakout wrong-show gate active; ~15-30 min encode)")
    out = build_video(
        proj, segs, cfg,
        captions=True,
        title=(analysis.movie_title or proj.name),
        theme_name="history",
        voiceover=PD + "/voiceover.mp3",
        use_tts=True,
        progress=log,
    )
    log(f"BUILD DONE -> {out}")
    with open(DONE, "w") as fh:
        fh.write(str(out))
except Exception as e:  # noqa: BLE001
    log("ERROR: " + repr(e))
    log(traceback.format_exc())
    with open(DONE, "w") as fh:
        fh.write("ERROR: " + repr(e))
