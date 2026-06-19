"""vidlore/qa_autofix.py — autonomous editorial-QA loop for the render pipeline.

This turns `editorial_qa.py` from a manual developer tool into the always-on
quality gate the portal render workflow runs after every render:

    render → editorial QA scan → if critical/high issues remain and autofix is
    on → re-route the offending scenes (bump their footage variant so a fresh
    asset is chosen) → re-render only those scenes → scan again → repeat until
    the strict gate passes, no further progress is made, or the pass cap is hit.

Flags (all default-safe; the loop is opt-in so existing renders are unchanged):
    VIDLORE_EDITORIAL_QA=1          run a QA scan + write reports after a render
    VIDLORE_EDITORIAL_QA_AUTOFIX=1  also run the re-route + re-render loop
    VIDLORE_EDITORIAL_QA_MAX_PASSES=3   hard cap on automatic passes
    VIDLORE_ALLOW_EDITORIAL_CALLBACKS=0 (default) one footage asset appears
                                         once; =1 allows a single intro callback

The re-route lever is the existing per-scene `variants.json` counter that the
footage layer already reads (footage.py: `bv = variants.get(sc.index, 0)`), so
no new render architecture is introduced — bumping a scene's counter makes the
footage selector pick a different asset for that scene on the next render.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

from . import editorial_qa as EQ


# ───────────────────────────── flags ───────────────────────────────────────
def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default) in ("1", "true", "yes", "on")


def qa_enabled() -> bool:
    return _flag("VIDLORE_EDITORIAL_QA")


def autofix_enabled() -> bool:
    return _flag("VIDLORE_EDITORIAL_QA_AUTOFIX")


def allow_callbacks() -> bool:
    return _flag("VIDLORE_ALLOW_EDITORIAL_CALLBACKS")


def max_passes() -> int:
    try:
        return max(1, int(os.environ.get("VIDLORE_EDITORIAL_QA_MAX_PASSES", "3")))
    except Exception:
        return 3


def allow_degraded_slides() -> bool:
    return _flag("VIDLORE_ALLOW_DEGRADED_TEXT_SLIDES")


def max_text_slide_ratio() -> float:
    try:
        return float(os.environ.get("VIDLORE_MAX_TEXT_SLIDE_RATIO", "0.10"))
    except Exception:
        return 0.10


# ───────────────────── AI-provider preflight (Step 4) ───────────────────────
def preflight_ai_providers(cfg=None) -> dict:
    """Health-check the still-image providers BEFORE an expensive render, using
    the SAME code path production uses. Emergency text slides must be a last
    resort, not the normal output — so if NO image provider can produce a still
    and degraded mode is off, the caller should abort before burning a render
    on a text-slide-heavy result.

    Returns {fal: {...}, pollinations: {...}, any_ok: bool, block: bool}.
    No secrets are logged — only booleans / latency / error category.
    """
    import time
    from . import footage as _F
    try:
        from .config import load_config
        cfg = cfg or load_config()
    except Exception:
        cfg = None

    out = {"fal": {}, "pollinations": {}, "any_ok": False, "block": False}

    # fal.ai — configured + endpoint reachable (a real paid gen would spend
    # credits, so we only confirm config+reachability here; the render itself
    # exercises it). Pollinations is free, so we do a real generation test.
    fal_key = bool(getattr(cfg, "fal_key", "")) if cfg else False
    out["fal"] = {"configured": fal_key, "reachable": _tcp_ok("fal.run")}

    polli_ok = False
    polli_latency = None
    try:
        import tempfile
        from pathlib import Path
        t0 = time.time()
        d = Path(tempfile.mkdtemp(prefix="aiprefl_"))
        ok = _F._pollinations_image(
            "documentary still of a garden and soil, warm light",
            d / "t.jpg", getattr(cfg, "pollinations_api_key", "") if cfg else "",
            424242)
        polli_latency = round(time.time() - t0, 2)
        polli_ok = bool(ok and (d / "t.jpg").exists())
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:                                       # noqa: BLE001
        out["pollinations"]["error"] = type(e).__name__
    out["pollinations"].update({"keyless": True, "test_image": polli_ok,
                                "latency_s": polli_latency,
                                "reachable": _tcp_ok("image.pollinations.ai")})

    # "any_ok" = at least one provider can produce a relevant still.
    out["any_ok"] = bool(polli_ok or (fal_key and out["fal"]["reachable"]))
    out["block"] = bool((not out["any_ok"]) and (not allow_degraded_slides()))
    return out


def _tcp_ok(host: str, port: int = 443, timeout: float = 6.0) -> bool:
    import socket
    try:
        socket.create_connection((socket.gethostbyname(host), port),
                                 timeout=timeout).close()
        return True
    except Exception:
        return False


# ───────────────────────── scan + report ───────────────────────────────────
def scan_and_report(video: Path, run_dir: Path, fps: float = EQ.DEFAULT_FPS):
    """Run the editorial-QA sweep and persist the report next to the render.
    Returns the QAReport (or None on failure — QA never breaks a render)."""
    try:
        rep = EQ.run_qa(Path(video), run_dir=Path(run_dir), fps=fps)
        EQ.write_report(rep, Path(run_dir))
        print(f"  [editorial-qa] gate={rep.gate} issues={rep.summary.get('total')} "
              f"{rep.summary.get('by_severity')}", flush=True)
        return rep
    except Exception as e:                                        # noqa: BLE001
        print(f"  [editorial-qa] scan skipped: {e}", flush=True)
        return None


# ───────────────────────── scene mapping ───────────────────────────────────
def _scene_table(run_dir: Path):
    """(scene_starts, scene_durations) from render_meta.json, or ([], [])."""
    try:
        m = json.loads((Path(run_dir) / "render_meta.json").read_text())
        return (list(m.get("scene_starts", [])),
                list(m.get("scene_durations", [])))
    except Exception:
        return [], []


def _scene_of(t: float, starts: list, durs: list) -> Optional[int]:
    if not starts:
        return None
    for i, s in enumerate(starts):
        e = s + (durs[i] if i < len(durs) else 1e9)
        if s <= t < e:
            return i
    # past the end → last scene
    return len(starts) - 1 if t >= starts[-1] else None


# ───────────────────────── override planning ───────────────────────────────
# Issue types whose fix is "give this scene a different/fresher asset".
_REROUTE_TYPES = {"repeated_footage", "unreadable_text", "washed_frame",
                  "black_frame", "blank_flash", "irrelevant_footage",
                  "wrong_subject", "wrong_era", "repeated_card_bg",
                  "dim_visual"}


def plan_overrides(report, run_dir: Path) -> dict:
    """Map each blocking issue to the scene that owns it and bump that scene's
    footage variant counter in variants.json. Returns {scene_index: reason}.

    `repeated_footage` honours VIDLORE_ALLOW_EDITORIAL_CALLBACKS: when callbacks
    are allowed the FIRST reappearance of a shot is permitted (an intentional
    intro callback) and only later repeats are re-routed; by default every
    repeat is re-routed (one asset → once per video).
    """
    starts, durs = _scene_table(run_dir)
    vpath = Path(run_dir) / "variants.json"
    try:
        variants = {int(k): int(v) for k, v in
                    json.loads(vpath.read_text()).items()}
    except Exception:
        variants = {}

    touched: dict = {}
    callback_used = False
    # only act on blocking severities
    blocking = [i for i in report.issues
                if i.get("severity") in ("critical", "high")
                and i.get("issue_type") in _REROUTE_TYPES]
    for it in blocking:
        itype = it.get("issue_type")
        # for a duplicate, re-route the LATER occurrence (keep the first use)
        t = it.get("second_timestamp")
        if t is None:
            t = it.get("timestamp", 0.0)
        if itype == "repeated_footage" and allow_callbacks() and not callback_used:
            callback_used = True            # permit one intentional callback
            continue
        sc = _scene_of(float(t), starts, durs)
        if sc is None:
            continue
        variants[sc] = variants.get(sc, 0) + 1
        touched[sc] = itype

    if touched:
        vpath.write_text(json.dumps({str(k): v for k, v in variants.items()},
                                    indent=2))
    return touched


# ───────────────────────── the autofix loop ────────────────────────────────
def run_with_qa(render_fn: Callable, *, run_dir: Path,
                video_attr: str = "video", fps: float = EQ.DEFAULT_FPS):
    """Render → scan → re-route → re-render until the strict gate passes.

    `render_fn()` performs one render and returns an object whose `video_attr`
    is the final MP4 path (e.g. pipeline.Result). It is called once per pass; on
    re-render passes it must read the same run_dir (so the bumped variants.json
    is picked up) — the standard pipeline does this automatically.

    Returns (result, report, passes, history) where history is a per-pass list
    of {pass, gate, totals, rerouted_scenes}. Honest by design: it never claims
    a pass it didn't make and logs every unresolved issue.
    """
    run_dir = Path(run_dir)
    cap = max_passes()
    history = []
    result = None
    report = None
    last_blocking = None
    for p in range(1, cap + 1):
        result = render_fn()
        video = Path(getattr(result, video_attr))
        report = scan_and_report(video, run_dir, fps=fps)
        if report is None:
            break
        blocking = report.summary.get("by_severity", {})
        n_block = blocking.get("critical", 0) + blocking.get("high", 0)
        if not autofix_enabled():
            history.append({"pass": p, "gate": report.gate,
                            "blocking": n_block, "rerouted_scenes": {}})
            break
        if report.gate != "FAIL":
            history.append({"pass": p, "gate": report.gate,
                            "blocking": n_block, "rerouted_scenes": {}})
            print(f"  [qa-autofix] pass {p}: gate {report.gate} — done", flush=True)
            break
        # FAIL → plan re-routes. Stop if we can't make progress (same count twice).
        touched = plan_overrides(report, run_dir)
        history.append({"pass": p, "gate": report.gate, "blocking": n_block,
                        "rerouted_scenes": touched})
        print(f"  [qa-autofix] pass {p}: gate FAIL, {n_block} blocking → "
              f"re-routing scenes {sorted(touched.keys())}", flush=True)
        if not touched:
            print("  [qa-autofix] no actionable re-routes — stopping (real "
                  "blocker, logged honestly)", flush=True)
            break
        if last_blocking is not None and n_block >= last_blocking and p > 1:
            print("  [qa-autofix] no progress vs previous pass — stopping to "
                  "avoid an endless loop (remaining issues logged)", flush=True)
            break
        last_blocking = n_block
        if p == cap:
            print(f"  [qa-autofix] hit max passes ({cap}); {n_block} issue(s) "
                  f"remain — logged honestly, not silently passed", flush=True)
    # persist the loop history for the audit trail
    try:
        (run_dir / "EDITORIAL_QA_AUTOFIX_LOG.json").write_text(
            json.dumps({"passes": history,
                        "final_gate": report.gate if report else "UNKNOWN",
                        "allow_callbacks": allow_callbacks(),
                        "max_passes": cap}, indent=2))
    except Exception:
        pass
    return result, report, len(history), history
