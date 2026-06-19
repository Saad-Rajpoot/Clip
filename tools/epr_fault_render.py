"""EPR fault-injection E2E test (encode-pool reliability).

Reproduces the EXACT conditions of the original 4-worker crash and proves
the render now recovers instead of aborting:

  1. Scene-0 footage clips are forced to read 'not ready' (simulating the
     partially-written / unreadable stock clips from the real crash).
  2. The first few graded-slate SPAWNS raise BlockingIOError(EAGAIN) — the
     fork-under-load failure that previously propagated out of the unguarded
     tier-4 slate and killed `pool.map`.

Runs the real Rockefeller render at the DEFAULT 4 workers. Test scaffolding
only — it monkeypatches the already-imported modules; no production code is
changed. Exit 0 + RENDER_DONE in the log == the fix holds.
"""
import runpy
import subprocess as _sp
import sys
import threading
from pathlib import Path

sys.path.insert(0, ".")
import vidlore.assemble as A  # noqa: E402

# ── 1) force scene-0 video clips to report "not ready" (partial-dl sim) ──
_real_ready = A._clip_ready


def _patched_ready(path, **kw):
    nm = Path(str(path)).name
    if nm.startswith(("clip_000_", "fal_000_", "px_000", "pexels_000")):
        return (False, "INJECTED-partial(scene0)")
    return _real_ready(path, **kw)


A._clip_ready = _patched_ready

# ── 2) inject transient fork-pressure into the first 3 slate spawns ──────
_lock = threading.Lock()
_hits = {"n": 0}
_real_spawn = _sp.run


def _patched_spawn(cmd, **kw):
    try:
        is_slate = any("color=" in str(a) for a in (cmd or []))
    except Exception:
        is_slate = False
    if is_slate:
        with _lock:
            _hits["n"] += 1
            if _hits["n"] <= 3:
                raise BlockingIOError(35, "INJECTED EAGAIN (fork pressure)")
    return _real_spawn(cmd, **kw)


_sp.run = _patched_spawn

print("EPR FAULT INJECTION ACTIVE: scene-0 clips not-ready + first 3 slate "
      "spawns raise EAGAIN · workers=default", flush=True)

# ── run the real render (its __main__ calls main()) ──────────────────────
runpy.run_path("tools/rockefeller_render.py", run_name="__main__")

print(f"EPR injected slate-spawn EAGAIN hits consumed: {_hits['n']}",
      flush=True)
