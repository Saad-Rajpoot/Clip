#!/usr/bin/env python3
"""Does the backfill pass actually find CLEAN replacements for what the gates threw out?

Runs only stage 5b against a copy of the finished job. Sources and index are symlinked FILE BY FILE
(not as directories) so anything the pass downloads lands in the new job and the original v2/v4
pools are never touched.
"""
import json, os, shutil, sys, time
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
REPO = f"{MAIN}/.claude/worktrees/clipstudio-handover-review-113723"
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, REPO)

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(Path(MAIN) / ".env")
for k in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
    if not os.environ.get(k):
        raise SystemExit(f"{k} missing")
os.environ.setdefault("VIDLORE_HD_PYTHON", f"{MAIN}/.hdvenv/bin/python")
os.environ.setdefault("VIDLORE_HD_POT_DIR", f"{MAIN}/.pot/server")
os.environ.setdefault("VIDLORE_CLIPSTUDIO_BACKFILL_ROUNDS", "1")

SRC = Path("/Users/hussnain/Desktop/clipstudio_output/portal/69d80e9dd4_v4")
POOL = Path("/Users/hussnain/Desktop/clipstudio_output/portal/69d80e9dd4_v2")
DST = Path("/Users/hussnain/Desktop/clipstudio_output/portal/69d80e9dd4_v5")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def link_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        q = dst / p.name
        if q.exists():
            continue
        if p.is_dir():
            link_tree(p, q)
        else:
            os.symlink(p.resolve(), q)


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    link_tree(POOL / "sources", DST / "sources")
    link_tree(POOL / "index", DST / "index")
    shutil.copy2(SRC / "voiceover.mp3", DST / "voiceover.mp3")
    log(f"pool linked: {len(list((DST/'sources').iterdir()))} source files")

    d = json.loads((SRC / "project.json").read_text())
    d["name"] = DST.name
    d["root"] = str(DST)
    d["selections"] = []
    (d.get("meta") or {}).pop("auto_rejected_sources", None)
    (DST / "project.json").write_text(json.dumps(d))

    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio import orchestrate as O
    from vidlore.clipstudio.analyze import ScriptAnalysis
    import dataclasses

    proj = ClipProject.load(DST)
    cfg = ClipConfig()
    an = (proj.meta or {}).get("analysis") or {}
    _fields = {f.name for f in dataclasses.fields(ScriptAnalysis)}
    analysis = ScriptAnalysis(**{k: v for k, v in an.items() if k in _fields})
    log(f"analysis: {analysis.movie_title!r} type={analysis.video_type} "
        f"anchors={len(analysis.anchor_scenes)} key_scenes={len(analysis.key_scenes)}")

    before = {s.id for s in proj.sources if s.status == "ok"}
    log(f"before: {len(before)} ok sources")
    n = O._backfill_rejected_sources(
        proj, list(proj.segments), analysis, cfg,
        refs=None, faceid_obj=None, roster=None,
        policy="approved_testing", max_sources=8, show_title="Game of Thrones", log=log)
    after = [s for s in proj.sources if s.status == "ok" and s.id not in before]
    log(f"ADMITTED {n} new source(s)")
    for s in after:
        log(f"   + {s.id[:34]:36s} {(s.title or '')[:64]}")
    log(json.dumps((proj.meta or {}).get("backfill_audit") or {}, indent=1)[:1800])


if __name__ == "__main__":
    main()
