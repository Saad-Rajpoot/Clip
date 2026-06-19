#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX snapshot.

This pass fixed the 3 user-reported REAL browser bugs (Replace feedback, timeline
playhead drag/click-to-seek, timeline crowding) — all in vidlore/web.py; the render
pipeline + editor_manifest are byte-identical to V1.2. Copies the editor source +
the regression test, records sha256 (incl. dist copies = 0-drift proof), reads the
live render_metrics.json from the footage-scene render proof, and writes the
manifest. Does NOT overwrite V1.0 / V1.1 / V1.2.

    .venv/bin/python tools/build_review_editor_v13_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX"
PROJ = "the-1860s-secret--how-to-end-garden-pests-permane"
DATE = "2026-06-02"
SRCS = ["vidlore/web.py", "vidlore/editor_manifest.py", "tools/test_review_editor_repairs.py"]
PRIOR = ["Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
         "Vidlore_ReviewEditor_V1.1_CapCutCleanUX",
         "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if SNAP.exists():
        print(f"REFUSING to overwrite existing snapshot: {SNAP}")
        return 1
    for p in PRIOR:
        if (ROOT / "snapshots" / p).exists():
            print(f"prior snapshot present, NOT touched: {p}")
    (SNAP / "source").mkdir(parents=True)
    hashes, drift, drift_ok = [], {}, True
    for rel in SRCS:
        dst = SNAP / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        s = sha(ROOT / rel)
        hashes.append(f"{s}  {rel}")
        if rel.startswith("vidlore/"):
            mac = ROOT / "dist/Vidlore-Mac" / rel
            win = ROOT / "dist/Vidlore-Windows" / rel
            ms = sha(mac) if mac.exists() else "MISSING"
            ws = sha(win) if win.exists() else "MISSING"
            synced = (s == ms == ws)
            drift_ok = drift_ok and synced
            drift[rel] = {"src": s[:16], "mac": ms[:16], "win": ws[:16], "synced": synced}
            hashes += [f"{ms}  dist/Vidlore-Mac/{rel}", f"{ws}  dist/Vidlore-Windows/{rel}"]
    (SNAP / "HASHES.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    rm = ROOT / "output" / PROJ / "render_metrics.json"
    qa = json.loads(rm.read_text()) if rm.exists() else {}

    manifest = {
        "label": "Vidlore Review Editor V1.3 — Real-Browser-Verified Clean UX (playhead scrub, replace feedback, timeline tiers)",
        "parent": "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish (untouched)",
        "date": DATE,
        "kind": "Fix the 3 user-reported REAL browser bugs with actual mouse gestures (Chrome MCP). Render pipeline + editor_manifest byte-identical to V1.2; AI video OFF.",
        "changed_files": {
            "vidlore/web.py": "Bug 2 — timeline click-to-seek + playhead drag-scrub via MOUSE events "
                              "(__edTLInit/__edTLSeekX, document-level move/up, bound clamp) + grabbable "
                              ".phknob. Bug 1 — replace feedback: search-pick toast + 're-render' wording + "
                              "instant corner pip (#edrepov) showing the new visual on the selected scene. "
                              "Bug 3 — width-tiered timeline labels (number / number+keyword / number+narration).",
        },
        "unchanged_backend": ["vidlore/pipeline.py", "vidlore/assemble.py", "vidlore/editor_manifest.py (V1.2)"],
        "bugs_fixed": {
            "bug1_replace_feedback": "data path already worked (thumbnail/override/badge/persist/final-MP4); added toast + instant pip so it's no longer a silent/confusing no-op.",
            "bug2_playhead": "was a passive div with NO handler + only pointerdown; now real mouse drag + click-to-seek (verified: click->121s, drag 2:00->1:00 = 62s).",
            "bug3_timeline_crowding": "width-tiered labels + grabbable knob.",
        },
        "real_browser_verification": {
            "tool": "Chrome MCP `computer` (real CDP mouse) at screenshot-pixel coords",
            "gestures": ["mouse-drag timeline scrub 977->560 -> currentTime 62",
                         "mouse-click ruler @2:00 -> currentTime 121",
                         "mouse-click scene 3 -> selected + instant ladybug pip",
                         "type+click search 'ladybug' -> replace applied",
                         "mouse-click Menu open + outside-click close"],
            "console_errors": 0,
        },
        "render_proof": {
            "scene": "scene 3 (FOOTAGE-only, no card) replaced with a ladybug stock photo",
            "qa": qa.get("qa", {}),
            "black_frames": qa.get("black_frames"),
            "audio_lufs": (qa.get("audio") or {}).get("lufs"),
            "note": "footage-scene proof (not a card scene); see REAL_BROWSER_CONTROL_BY_CONTROL_QA.md",
        },
        "validation": {
            "regression": "tools/test_review_editor_repairs.py -> 27/27 (incl. seek/clamp math)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "use_only_audio_in_dist": 0,
            "ai_video": "OFF",
        },
        "reports": [
            "research/review_editor/REAL_BROWSER_CONTROL_BY_CONTROL_QA.md",
            "research/review_editor/FINAL_CLEAN_DASHBOARD_AUDIT.md",
            "research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md",
        ],
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into dist/Vidlore-Mac/vidlore/ + dist/Vidlore-Windows/vidlore/ (byte-identical); sha-verify.",
            "Backend + editor_manifest untouched this pass -> restoring web.py restores V1.2. The V1.2/V1.1/V1.0 snapshots remain deeper rollback points.",
        ],
        "remaining_minor": [
            "Native file picker not driveable via Chrome MCP (upload path proven via prior magenta render + search-pick).",
            "Timeline block drag-reorder from V1.2 not re-stress-tested with synthesized drags.",
            "Volume slider verified by value-set, not synthesized slider drag.",
            "100+ scene timeline not fixture-tested.",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    print(f"  render QA: {qa.get('qa', {}).get('verdict', '(none)')}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
