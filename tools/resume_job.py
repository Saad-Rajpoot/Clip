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
    # SAME environment contract as the portal, checked before any work. A resume that runs on a
    # different native stack than the render it continues is not a resume — and the stack that
    # aborted two renders was exactly the one a bare `python3 tools/resume_job.py` picks up.
    from vidlore.clipstudio import runtime_env as _env
    try:
        env_line = _env.enforce(driver="resume_job")
    except RuntimeError as ee:
        print(f"\n✗ {ee}\n", file=sys.stderr)
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

    # THE SAME on-disk run record the portal writes, and the SAME lock. Without this a CLI resume
    # is invisible: an observer would find a stale `running` state from an earlier portal run, see
    # the lock free because nobody took it, and report "died" while this render is very much alive.
    # A false death is the worse failure — it sends an operator to kill work that is fine.
    from vidlore.clipstudio import run_state as _rs
    _run_lock = _rs.RunLock(proj_dir)
    if not _run_lock.acquire():
        print(f"another driver is already rendering {proj_dir.name} "
              f"(run.lock is held) — refusing to run two writers on one project dir")
        return 2

    log_path = proj_dir / "output" / "build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    fh.write(f"\n===== CLI resume start {time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"job={proj_dir.name} review={review} =====\n")
    fh.write(f"      env {env_line}\n")

    def log(m):
        try:
            if isinstance(m, str) and "/9 · " in m:
                _rs.touch(proj_dir, phase=m.strip())
        except Exception:                                 # noqa: BLE001 — never break a render
            pass
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        try:
            fh.write(line + "\n")
            fh.flush()
        except Exception:
            pass

    _rs.write(proj_dir, proj_dir.name, driver="resume", extra={"review_mode": bool(review)})
    _heart = _rs.Heartbeat(proj_dir).start()

    from vidlore.clipstudio.orchestrate import produce_auto
    from vidlore.clipstudio.verify import is_content_stop

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
        # AUTO REVIEW DRAFT on a content failure — the SAME predicate the portal's _run_job uses,
        # imported rather than restated so the two drivers cannot drift apart again.
        if not review and is_content_stop(e):
            log("↻ FOOTAGE GAP → auto-building a REVIEW DRAFT (missing-footage beats air "
                "flagged, not for publication). Resuming from cached stages…")
            os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn"
            res = _render()
        else:
            log(f"FATAL: {e}")
            _heart.stop(status="error", phase="failed")
            _run_lock.release()
            raise
    log(f"done → {res.get('output') if isinstance(res, dict) else res}")
    _heart.stop(status="ok", phase="finished")
    _run_lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
