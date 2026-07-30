#!/usr/bin/env python3
"""Resume a portal job from DISK after a portal restart.

The portal's Resume button replays a job's launch params from its in-memory registry — a
restarted portal has lost them (`/retry` returns 410) even though the ENTIRE project state
(analysis, segments, sources, index, selections, verdicts) is cached on disk. This driver
reconstructs the resumable call from the project dir alone and runs produce_auto(resume=True)
in THIS process — so it always executes the current code, not whatever the portal captured at
its import time.

    python3 tools/resume_job.py <job_dir> [--review]

  <job_dir>   a portal job dir (contains project.json)
  --review    start directly in review mode (RELEASE_BLOCK_MODE=warn). Without it, the driver
              mirrors the portal: strict first, then auto-falls back to a review draft on a
              CONTENT failure (footage gap), exactly like _run_job's auto-review path.

Uses VIDLORE_CLIPSTUDIO_RESUME_TRUST_CACHED=1 (orchestrate) so the cached analyze artifact is
trusted without the original raw script text. Progress is appended to output/build.log.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    review = "--review" in sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    proj_dir = Path(args[0]).expanduser().resolve()
    pj = proj_dir / "project.json"
    if not pj.exists():
        print(f"no project.json under {proj_dir}")
        return 2
    meta = json.loads(pj.read_text(encoding="utf-8")).get("meta", {})
    analysis = meta.get("analysis") or {}
    caps = meta.get("caption_settings") or {}
    topic = (analysis.get("topic") or proj_dir.name).strip()
    movie = (analysis.get("movie_title") or "").strip()
    # placeholder narration text — analyze is skipped via the trusted-cache signature, so this
    # only satisfies the non-empty guard; it is NEVER re-analyzed
    segs_txt = "(trusted-cache resume — cached segments govern)"
    vo = next((str(p) for p in proj_dir.glob("voiceover.*")), None)

    # env parity with web._run_job — a resume must not run under different knobs than the portal
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_LLM_PROVIDER", "deepseek")
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_DEEPSEEK_MODEL", "deepseek-v4-pro")
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_MAX_CPU", "1")
    os.environ.setdefault("VIDLORE_CLIPSTUDIO_VERIFY_WORKERS", "4")
    os.environ["VIDLORE_CLIPSTUDIO_RESUME_TRUST_CACHED"] = "1"
    os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn" if review else "block"

    log_path = proj_dir / "output" / "build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    fh.write(f"\n===== CLI resume start {time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"job={proj_dir.name} review={review} =====\n")

    def log(m):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        try:
            fh.write(line + "\n")
            fh.flush()
        except Exception:
            pass

    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore.clipstudio.verify import NonRetryableBuildError, VisionBackendError

    def _render(**over):
        return produce_auto(
            str(proj_dir), topic=topic, title=topic, script_text=segs_txt, movie_hint=movie,
            policy=os.environ.get("VIDLORE_CLIPSTUDIO_PORTAL_POLICY", "approved_testing").strip(),
            max_sources=8, theme="history",
            captions=bool(caps.get("enabled", True)),
            caption_style=str(caps.get("style") or "professional"),
            voiceover=vo, use_tts=True, verify=True, do_build=True,
            resume=True, progress=log)

    try:
        res = _render()
    except Exception as e:                                   # noqa: BLE001
        # AUTO REVIEW DRAFT on a content failure — mirrors the portal's _run_job fallback
        if not review and isinstance(e, NonRetryableBuildError) \
                and not isinstance(e, VisionBackendError):
            log("↻ FOOTAGE GAP → auto-building a REVIEW DRAFT (missing-footage beats air "
                "flagged, not for publication). Resuming from cached stages…")
            os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"
            res = _render()
        else:
            log(f"FATAL: {e}")
            raise
    log(f"done → {res.get('output') if isinstance(res, dict) else res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
