#!/usr/bin/env python3
"""Build the Vidlore_PostRenderDashboard_V1.0_PremiumResultsUX snapshot.

Redesigns the post-render results page (`/job/<job_id>`, the `_JOB` template) from a raw
developer page (static "Rendering…" heading + bare bar + browser-default links) into a
premium two-state dashboard: a polished RENDERING state (branding, status badge,
humanized stage from the real msg, animated progress) and a premium COMPLETE state
(success header, 16:9 cache-busted video card, strong "Open Review Editor" primary +
grouped secondary buttons, project-details + quality-check panels with an expandable
technical view), plus a polished error card, tooltips and responsive layout. Backend:
one additive `GET /job/<id>/summary` route (QA + meta facts). Render pipeline untouched.

ONLY vidlore/web.py changed. Records sha256 (incl. dist copies). Does NOT overwrite any
earlier snapshot.

    .venv/bin/python tools/build_post_render_dashboard_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_PostRenderDashboard_V1.0_PremiumResultsUX"
DATE = "2026-06-03"
SRCS = ["vidlore/web.py", "tools/test_post_render_dashboard.py"]
DIST_TRACKED = ["vidlore/web.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if SNAP.exists():
        print(f"REFUSING to overwrite existing snapshot: {SNAP}")
        return 1
    keep = [p.name for p in (ROOT / "snapshots").glob("Vidlore_ReviewEditor_*")]
    for p in sorted(keep):
        print(f"prior snapshot present, NOT touched: {p}")
    (SNAP / "source").mkdir(parents=True)
    hashes, drift, drift_ok = [], {}, True
    for rel in SRCS:
        dst = SNAP / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        s = sha(ROOT / rel)
        hashes.append(f"{s}  {rel}")
        if rel in DIST_TRACKED:
            mac = ROOT / "dist/Vidlore-Mac" / rel
            win = ROOT / "dist/Vidlore-Windows" / rel
            ms = sha(mac) if mac.exists() else "MISSING"
            ws = sha(win) if win.exists() else "MISSING"
            synced = (s == ms == ws)
            drift_ok = drift_ok and synced
            drift[rel] = {"src": s[:16], "mac": ms[:16], "win": ws[:16], "synced": synced}
            hashes += [f"{ms}  dist/Vidlore-Mac/{rel}", f"{ws}  dist/Vidlore-Windows/{rel}"]
    (SNAP / "HASHES.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    manifest = {
        "label": "Vidlore Post-Render Dashboard V1.0 — Premium Results UX",
        "date": DATE,
        "kind": "Redesign the /job/<id> post-render page into a premium two-state dashboard. ONLY vidlore/web.py changed (the _JOB template + an additive /job/<id>/summary route); render pipeline / QA / audio untouched; AI video OFF.",
        "changed_files": {
            "vidlore/web.py": "Rewrote the _JOB inline template: self-contained dark theme matching _PAGE "
                              "tokens; RENDERING state (Vidlore Studio branding, pulsing status badge, "
                              "spinner, 'Creating your documentary', humanized stage via stageLabel() from "
                              "the REAL job msg, animated progress + %, helpful hint); COMPLETE state "
                              "(success header + '✓ Quality checked' badge — no stale 'Rendering…', 16:9 "
                              "cache-busted video card with poster, strong 'Open Review Editor' primary + "
                              "grouped Download/Create/My Videos/Thumbnail secondary buttons, Project-details "
                              "panel from /summary, Quality-check panel with an expandable technical view); "
                              "polished ERROR card (Try again / Return / View technical details, no raw "
                              "stack trace); compact tooltip engine (data-tip, 300ms, clamp, focus, Esc); "
                              "responsive (@media stack); polling stops after completion. + additive route "
                              "GET /job/<id>/summary (duration/scenes/resolution/fps/captions/look/size/"
                              "completed + QA verdict/summary/black_frames/lufs) and helpers _fmt_dur, "
                              "_probe_resolution.",
        },
        "preserved": ["/job/<id>/status", "/job/<id>/file/<kind>", "/job/<id>/retry", "render-job polling",
                      "video playback", "Review Editor link", "MP4 download (cache-busted)", "thumbnail",
                      "My Videos / Create-another", "render pipeline / QA / audio / AI-still", "AI video OFF"],
        "real_browser_verification": {
            "tool": "Chrome MCP (real mouse), real re-render job 0541ada16965",
            "checks": ["rendering state premium (badge, humanized stage, animated bar)",
                       "refresh during render -> polling resumes",
                       "auto-transition to complete on done (no stale 'Rendering…')",
                       "complete: '✓ Quality checked', 16:9 video, primary + secondary buttons, project details, QA panel",
                       "technical-details expand/collapse + tooltip",
                       "responsive at ~760px (no overflow)"],
            "console_errors": 0,
            "docs": ["research/post_render_dashboard/POST_RENDER_DASHBOARD_AUDIT.md",
                     "research/post_render_dashboard/POST_RENDER_DASHBOARD_MANUAL_QA.md",
                     "research/post_render_dashboard/POST_RENDER_DASHBOARD_FINAL_REPORT.md"],
        },
        "validation": {
            "regression_dashboard": "tools/test_post_render_dashboard.py -> 35/35",
            "regression_editor": "tools/test_review_editor_repairs.py -> 124/124 (unchanged)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "ai_video": "OFF",
        },
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into both dist trees (byte-identical); sha-verify.",
            "Backend pipeline untouched -> restoring web.py restores the prior _JOB page.",
        ],
        "note_dist_run": "The live app is run from inside dist/Vidlore-Mac (OUT = dist/Vidlore-Mac/output); restart the server with CWD=dist/Vidlore-Mac. Job pages are in-memory per server session (a job_id 404s after restart, by design).",
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
