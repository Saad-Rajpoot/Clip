"""Vidlore Improvement Engine — safe autonomous code improvement loop.

Reads the improvement backlog, picks the highest-priority pending item,
applies the code change, runs a short test render, scores before/after,
and keeps or reverts the change based on whether quality improved.

Usage (CLI):
    python tools/improvement_engine.py list
    python tools/improvement_engine.py run [--dry-run]
    python tools/improvement_engine.py add --title "..." --file vidlore/assemble.py
    python tools/improvement_engine.py status IMP_001

Usage (module):
    from tools.improvement_engine import load_backlog, get_next_improvement
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_BACKLOG_FILE  = _ROOT / "research" / "improvement_backlog.json"
_REPORTS_DIR   = _ROOT / "research" / "reports"
_OUTPUT_DIR    = _ROOT / "output"

_DIST_MAC     = _ROOT / "dist" / "Vidlore-Mac"
_DIST_WIN     = _ROOT / "dist" / "Vidlore-Windows"

_MAX_TEST_SCENES = 8  # hard cap for safety test renders

# ── Timestamp helper ─────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str, report_path: Optional[Path] = None) -> None:
    line = f"[{_now()}] {msg}"
    print(line)
    if report_path:
        try:
            with report_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ── Backlog I/O ───────────────────────────────────────────────────────────────

def load_backlog() -> list[dict]:
    """Load improvement_backlog.json; return empty list if missing."""
    if not _BACKLOG_FILE.exists():
        return []
    try:
        data = json.loads(_BACKLOG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_backlog(items: list[dict]) -> None:
    """Persist the full backlog list to disk."""
    _BACKLOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BACKLOG_FILE.write_text(
        json.dumps(items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_next_improvement() -> Optional[dict]:
    """Return the highest-priority pending item (lowest priority number = highest)."""
    backlog = load_backlog()
    pending = [
        item for item in backlog
        if item.get("status", "pending") == "pending"
    ]
    if not pending:
        return None
    return min(pending, key=lambda x: x.get("priority", 999))


def mark_improvement(item_id: str, status: str, result: dict) -> None:
    """Update an item's status and result in the backlog."""
    backlog = load_backlog()
    for item in backlog:
        if item.get("id") == item_id:
            item["status"] = status
            item["result"] = result
            item["implemented_at"] = _now() if status == "implemented" else None
            break
    save_backlog(backlog)


# ── Code change helpers ───────────────────────────────────────────────────────

def _backup_path(file_path: Path) -> Path:
    """Return a .bak path in /tmp for the given file."""
    safe_name = str(file_path).replace("/", "_").replace("\\", "_")
    return Path("/tmp") / f"vidlore_bak_{safe_name}.bak"


def apply_code_change(change: dict) -> bool:
    """Write file changes from a change dict, backing up first.

    Expected change dict format:
    {
        "file": "vidlore/assemble.py",
        "old": "exact text to replace",
        "new": "replacement text"
    }
    Or for whole-file replacement:
    {
        "file": "vidlore/assemble.py",
        "content": "full new file content"
    }
    Returns True on success, False on failure.
    """
    file_rel = change.get("file", "")
    if not file_rel:
        print("[improvement] ERROR: change dict missing 'file' key", file=sys.stderr)
        return False

    file_path = _ROOT / file_rel
    if not file_path.exists():
        print(f"[improvement] ERROR: target file not found: {file_path}", file=sys.stderr)
        return False

    # Backup first — ALWAYS
    bak = _backup_path(file_path)
    try:
        shutil.copy2(str(file_path), str(bak))
    except Exception as e:
        print(f"[improvement] ERROR: backup failed: {e}", file=sys.stderr)
        return False

    original = file_path.read_text(encoding="utf-8")

    try:
        if "content" in change:
            # Full file replacement
            file_path.write_text(change["content"], encoding="utf-8")
        elif "old" in change and "new" in change:
            old_text = change["old"]
            new_text = change["new"]
            if old_text not in original:
                print(
                    f"[improvement] ERROR: 'old' text not found in {file_rel}",
                    file=sys.stderr,
                )
                # Restore backup
                shutil.copy2(str(bak), str(file_path))
                return False
            updated = original.replace(old_text, new_text, 1)
            file_path.write_text(updated, encoding="utf-8")
        else:
            print("[improvement] ERROR: change dict must have 'content' or 'old'+'new'",
                  file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[improvement] ERROR applying change: {e}", file=sys.stderr)
        # Attempt to restore backup
        try:
            shutil.copy2(str(bak), str(file_path))
        except Exception:
            pass
        return False


def revert_code_change(change: dict) -> bool:
    """Restore backed-up file. Returns True on success."""
    file_rel = change.get("file", "")
    if not file_rel:
        return False
    file_path = _ROOT / file_rel
    bak = _backup_path(file_path)
    if not bak.exists():
        print(f"[improvement] ERROR: no backup found at {bak}", file=sys.stderr)
        return False
    try:
        shutil.copy2(str(bak), str(file_path))
        print(f"[improvement] Reverted {file_rel} from backup")
        return True
    except Exception as e:
        print(f"[improvement] ERROR reverting {file_rel}: {e}", file=sys.stderr)
        return False


# ── Dist sync ─────────────────────────────────────────────────────────────────

def sync_to_dist(files: list[str]) -> None:
    """Copy source files to both Mac and Windows dist trees."""
    dist_trees = [_DIST_MAC, _DIST_WIN]
    for file_rel in files:
        src = _ROOT / file_rel
        if not src.exists():
            print(f"[improvement] sync: source not found: {src}", file=sys.stderr)
            continue
        for dist_root in dist_trees:
            dst = dist_root / file_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(src), str(dst))
                print(f"[improvement] Synced {file_rel} -> {dst.relative_to(_ROOT)}")
            except Exception as e:
                print(f"[improvement] sync ERROR {file_rel} -> {dst}: {e}",
                      file=sys.stderr)


# ── Test render ───────────────────────────────────────────────────────────────

def run_test_render(
    brief_dict: dict,
    scenes: int = 5,
    report_path: Optional[Path] = None,
) -> Optional[Path]:
    """Run a short test render (max _MAX_TEST_SCENES scenes).

    brief_dict must match Brief field names. Returns path to output MP4 or None.
    Safety: scenes is capped at _MAX_TEST_SCENES = 8.
    """
    scenes = min(scenes, _MAX_TEST_SCENES)
    _log(f"Starting test render: '{brief_dict.get('title', 'untitled')}' "
         f"({scenes} scenes max)", report_path)

    # Write a temporary brief YAML to /tmp. The vidlore Brief schema REQUIRES
    # a `prompt` field (the 4-pillars creative brief). Backlog test briefs
    # sometimes carry `topic` instead — map it so the render never fails on a
    # missing prompt. Force the shortest duration band for a fast test render.
    import yaml  # type: ignore[import]
    brief_dict = dict(brief_dict)               # don't mutate caller's dict
    if not brief_dict.get("prompt"):
        brief_dict["prompt"] = (
            brief_dict.get("topic")
            or brief_dict.get("title")
            or "A short documentary about a pivotal moment in history."
        )
    brief_dict.pop("topic", None)               # not a Brief field
    brief_file = Path("/tmp") / f"vidlore_test_brief_{uuid.uuid4().hex[:8]}.yaml"
    brief_dict.setdefault("duration", "1-2")
    brief_dict.setdefault("fmt", "documentary")
    brief_dict.setdefault("theme", "history")

    try:
        brief_file.write_text(yaml.dump(brief_dict, allow_unicode=True), encoding="utf-8")
    except Exception as e:
        _log(f"ERROR writing test brief: {e}", report_path)
        return None

    # Build render command. NOTE: the vidlore CLI has NO --max-scenes flag —
    # scene count is derived from the brief's `duration` band. We keep test
    # renders short by forcing the smallest duration ("1-2") on the brief
    # above; `scenes` only bounds our own logging/expectations.
    python_bin = sys.executable
    cmd = [
        python_bin, "-m", "vidlore",
        "--brief", str(brief_file),
        "--auto",
    ]

    _log(f"Render cmd: {' '.join(cmd[:5])} ...", report_path)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=str(_ROOT),
            timeout=600,  # 10 minute hard cap for test renders
        )
        elapsed = time.time() - t0
        _log(f"Render completed in {elapsed:.1f}s (rc={result.returncode})",
             report_path)
        if result.returncode != 0:
            _log(f"Render stderr (last 500 chars): {result.stderr[-500:]}",
                 report_path)
            return None
    except subprocess.TimeoutExpired:
        _log("ERROR: Test render timed out (>600s)", report_path)
        return None
    except Exception as e:
        _log(f"ERROR running render: {e}", report_path)
        return None
    finally:
        try:
            brief_file.unlink(missing_ok=True)
        except Exception:
            pass

    # Find the output MP4 (most recently modified)
    output_dir = _ROOT / "output"
    mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        _log("No MP4 found in output dir after render", report_path)
        return None

    mp4 = mp4s[-1]
    _log(f"Test render output: {mp4}", report_path)
    return mp4


# ── Score comparison ──────────────────────────────────────────────────────────

def evaluate_improvement(before_score: dict, after_score: dict) -> dict:
    """Compare before/after benchmark score dicts. Return evaluation result."""
    before = before_score.get("scores", {})
    after = after_score.get("scores", {})

    before_overall = before.get("overall_quality", 0.0)
    after_overall = after.get("overall_quality", 0.0)
    delta = round(after_overall - before_overall, 4)

    # Per-dimension deltas
    all_dims = set(before) | set(after)
    dim_deltas = {
        dim: round(after.get(dim, 0.0) - before.get(dim, 0.0), 4)
        for dim in all_dims
    }

    improved = delta > 0
    verdict = "keep" if improved else "revert"

    result = {
        "verdict": verdict,
        "before_overall": before_overall,
        "after_overall": after_overall,
        "delta": delta,
        "dim_deltas": dim_deltas,
        "improved": improved,
        "evaluated_at": _now(),
    }

    return result


# ── Safety guards ─────────────────────────────────────────────────────────────

def _check_render_health(score: dict) -> bool:
    """Return True if render is healthy enough to proceed."""
    reliability = score.get("scores", {}).get("render_reliability", 10.0)
    if reliability < 8.0:
        print(
            f"[improvement] SAFETY: render_reliability={reliability:.1f} < 8.0 — "
            "refusing to apply improvement on broken render.",
            file=sys.stderr,
        )
        return False
    return True


# ── Main improvement loop ─────────────────────────────────────────────────────

def run_improvement_cycle(dry_run: bool = False) -> dict:
    """Pick the next improvement, test it, keep or revert, log everything.

    Returns a summary dict with the outcome.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = _REPORTS_DIR / f"improvement_{time.strftime('%Y%m%d_%H%M%S')}.log"

    _log("=" * 60, report_file)
    _log("Vidlore Improvement Cycle", report_file)
    _log("=" * 60, report_file)

    # 1. Get next improvement
    item = get_next_improvement()
    if item is None:
        _log("No pending improvements found in backlog.", report_file)
        return {"status": "no_pending_improvements"}

    item_id = item["id"]
    title = item.get("title", "untitled")
    _log(f"Selected: [{item_id}] priority={item.get('priority')} — {title}", report_file)

    if dry_run:
        _log("[DRY RUN] Would apply this improvement but skipping.", report_file)
        return {"status": "dry_run", "item": item}

    # Mark in-progress
    mark_improvement(item_id, "in_progress", {})

    # 2. Prepare test brief from item
    test_brief = item.get("test_render_brief", {})
    if not test_brief:
        test_brief = {
            "title": f"Benchmark Test — {title}",
            "prompt": item.get("hypothesis", "A short documentary about history."),
            "duration": "1-2",
        }

    # 3. Score BEFORE render (use existing baseline if available)
    _log("Step 1/5: Running BEFORE test render...", report_file)
    before_mp4 = run_test_render(test_brief, scenes=5, report_path=report_file)
    if before_mp4 is None:
        _log("BEFORE render failed — aborting cycle.", report_file)
        mark_improvement(item_id, "skipped", {"reason": "before render failed"})
        return {"status": "before_render_failed", "item_id": item_id}

    # Score the before render
    _log("Step 2/5: Scoring BEFORE render...", report_file)
    script_json_before = before_mp4.parent / "script.json"
    from tools.benchmark_engine import score_render
    before_score = score_render(before_mp4, script_json_before, before_mp4.parent)

    # Safety gate: don't improve a broken render
    if not _check_render_health(before_score):
        mark_improvement(item_id, "skipped",
                         {"reason": "render_reliability < 8 on baseline"})
        return {"status": "render_unhealthy", "item_id": item_id}

    # 4. Apply code change(s)
    _log("Step 3/5: Applying code changes...", report_file)
    affected_files = item.get("affected_files", [])
    change = {
        "file": affected_files[0] if affected_files else "",
        **item.get("change_description_structured", {}),
    }

    if not change.get("file"):
        _log("ERROR: No affected file specified in improvement item.", report_file)
        mark_improvement(item_id, "skipped", {"reason": "no affected file"})
        return {"status": "no_affected_file", "item_id": item_id}

    applied = apply_code_change(change)
    if not applied:
        _log("ERROR: Failed to apply code change — aborting.", report_file)
        mark_improvement(item_id, "skipped", {"reason": "apply_code_change failed"})
        return {"status": "apply_failed", "item_id": item_id}

    _log(f"Applied change to: {change['file']}", report_file)

    # 5. Run AFTER render
    _log("Step 4/5: Running AFTER test render...", report_file)
    after_mp4 = run_test_render(test_brief, scenes=5, report_path=report_file)
    if after_mp4 is None:
        _log("AFTER render failed — reverting change.", report_file)
        revert_code_change(change)
        mark_improvement(item_id, "reverted",
                         {"reason": "after render failed", "reverted": True})
        return {"status": "after_render_failed", "item_id": item_id}

    # Score the after render
    _log("Step 5/5: Scoring AFTER render and evaluating...", report_file)
    script_json_after = after_mp4.parent / "script.json"
    after_score = score_render(after_mp4, script_json_after, after_mp4.parent)
    evaluation = evaluate_improvement(before_score, after_score)

    _log(
        f"Evaluation: before={evaluation['before_overall']:.3f} "
        f"after={evaluation['after_overall']:.3f} "
        f"delta={evaluation['delta']:+.3f} => {evaluation['verdict'].upper()}",
        report_file,
    )

    # 6. Keep or revert
    if evaluation["verdict"] == "keep":
        _log("Quality improved — keeping change.", report_file)
        # Sync to both dist trees
        sync_to_dist(affected_files)
        mark_improvement(item_id, "implemented", evaluation)
        final_status = "implemented"
    else:
        _log("Quality did not improve — reverting change.", report_file)
        reverted = revert_code_change(change)
        evaluation["reverted"] = reverted
        mark_improvement(item_id, "reverted", evaluation)
        final_status = "reverted"

    _log(f"Cycle complete. Status: {final_status}", report_file)
    _log(f"Report: {report_file}", report_file)

    return {
        "status":        final_status,
        "item_id":       item_id,
        "title":         title,
        "evaluation":    evaluation,
        "report_file":   str(report_file),
        "before_score":  before_score.get("scores", {}),
        "after_score":   after_score.get("scores", {}),
    }


# ── Backlog management helpers ────────────────────────────────────────────────

def _add_item(
    title: str,
    hypothesis: str = "",
    priority: int = 5,
    affected_files: Optional[list] = None,
    test_topic: str = "brief history documentary",
    dimension: str = "overall_quality",
    delta: str = "+0.5",
) -> dict:
    """Create and append a new improvement item to the backlog."""
    backlog = load_backlog()
    item_id = f"IMP_{len(backlog) + 1:03d}"
    item = {
        "id": item_id,
        "priority": priority,
        "status": "pending",
        "title": title,
        "hypothesis": hypothesis or title,
        "source": "manual",
        "affected_files": affected_files or [],
        "change_description": "",
        "change_description_structured": {},
        "expected_improvement": {"dimension": dimension, "delta": delta},
        "test_render_brief": {
            "title": f"Benchmark Test — {title}",
            "prompt": test_topic,
            "duration": "1-2",
        },
        "result": None,
        "implemented_at": None,
    }
    backlog.append(item)
    save_backlog(backlog)
    print(f"Added [{item_id}]: {title}")
    return item


def _print_backlog() -> None:
    backlog = load_backlog()
    if not backlog:
        print("Backlog is empty.")
        return
    print(f"{'ID':<10} {'P':>3}  {'Status':<12}  Title")
    print("-" * 70)
    for item in sorted(backlog, key=lambda x: x.get("priority", 999)):
        print(
            f"{item['id']:<10} "
            f"{item.get('priority', '?'):>3}  "
            f"{item.get('status', 'pending'):<12}  "
            f"{item.get('title', '(no title)')}"
        )


def _print_status(item_id: str) -> None:
    backlog = load_backlog()
    for item in backlog:
        if item.get("id") == item_id:
            print(json.dumps(item, indent=2, ensure_ascii=False))
            return
    print(f"Item '{item_id}' not found in backlog.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Vidlore Improvement Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    # list
    sub.add_parser("list", help="List all backlog items")

    # run
    sc_run = sub.add_parser("run", help="Run one improvement cycle")
    sc_run.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without changing anything")

    # status
    sc_status = sub.add_parser("status", help="Show a specific item")
    sc_status.add_argument("item_id")

    # add
    sc_add = sub.add_parser("add", help="Add a new improvement item")
    sc_add.add_argument("--title", required=True)
    sc_add.add_argument("--hypothesis", default="")
    sc_add.add_argument("--priority", type=int, default=5)
    sc_add.add_argument("--file", dest="affected_file", default="")
    sc_add.add_argument("--dimension", default="overall_quality")
    sc_add.add_argument("--topic", default="brief history documentary")

    # evaluate
    sc_eval = sub.add_parser("evaluate",
                              help="Compare two benchmark score files")
    sc_eval.add_argument("before_json", type=Path)
    sc_eval.add_argument("after_json", type=Path)

    args = p.parse_args()

    if args.cmd == "list":
        _print_backlog()
        return 0

    elif args.cmd == "run":
        result = run_improvement_cycle(dry_run=args.dry_run)
        print("\nResult:")
        print(json.dumps(
            {k: v for k, v in result.items()
             if k not in ("before_score", "after_score")},
            indent=2,
        ))
        if "evaluation" in result:
            ev = result["evaluation"]
            print(f"\nVerdict: {ev.get('verdict', '?').upper()}")
            print(f"  Before: {ev.get('before_overall', 0):.3f}")
            print(f"  After:  {ev.get('after_overall', 0):.3f}")
            print(f"  Delta:  {ev.get('delta', 0):+.3f}")
        return 0 if result.get("status") in ("implemented", "dry_run",
                                               "no_pending_improvements") else 1

    elif args.cmd == "status":
        _print_status(args.item_id)
        return 0

    elif args.cmd == "add":
        _add_item(
            title=args.title,
            hypothesis=args.hypothesis,
            priority=args.priority,
            affected_files=[args.affected_file] if args.affected_file else [],
            test_topic=args.topic,
            dimension=args.dimension,
        )
        return 0

    elif args.cmd == "evaluate":
        try:
            before = json.loads(args.before_json.read_text(encoding="utf-8"))
            after = json.loads(args.after_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"ERROR loading score files: {e}", file=sys.stderr)
            return 1
        ev = evaluate_improvement(before, after)
        print(json.dumps(ev, indent=2))
        return 0

    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
