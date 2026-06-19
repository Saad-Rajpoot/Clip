#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.1_CapCutCleanUX snapshot.

This pass (CapCut-clean UX) changed ONLY vidlore/web.py (editor frontend). The
render pipeline is byte-identical to V1.0. Copies web.py into source/, records
sha256 (incl. dist copies for a 0-drift proof), reads the live render_metrics.json
QA result, and writes SNAPSHOT_MANIFEST.json. Does NOT overwrite the V1.0 snapshot.

    .venv/bin/python tools/build_review_editor_v11_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.1_CapCutCleanUX"
PROJ = "the-1860s-secret--how-to-end-garden-pests-permane"
DATE = "2026-06-02"
CHANGED = ["vidlore/web.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if SNAP.exists():
        print(f"REFUSING to overwrite existing snapshot: {SNAP}")
        return 1
    if (ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX").exists():
        print("V1.0 snapshot present and will NOT be touched.")
    (SNAP / "source").mkdir(parents=True)
    hashes, drift, drift_ok = [], {}, True
    for rel in CHANGED:
        src = ROOT / rel
        dst = SNAP / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        s = sha(src)
        mac = ROOT / "dist/Vidlore-Mac" / rel
        win = ROOT / "dist/Vidlore-Windows" / rel
        ms = sha(mac) if mac.exists() else "MISSING"
        ws = sha(win) if win.exists() else "MISSING"
        synced = (s == ms == ws)
        drift_ok = drift_ok and synced
        drift[rel] = {"src": s[:16], "mac": ms[:16], "win": ws[:16], "synced": synced}
        hashes += [f"{s}  {rel}", f"{ms}  dist/Vidlore-Mac/{rel}", f"{ws}  dist/Vidlore-Windows/{rel}"]
    (SNAP / "HASHES.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    # live QA result from the real re-render
    rm = ROOT / "output" / PROJ / "render_metrics.json"
    qa = json.loads(rm.read_text()) if rm.exists() else {}

    manifest = {
        "label": "Vidlore Review Editor V1.1 — CapCut-Clean UX (Menu dropdown, contextual popovers, smooth drag, clean timeline)",
        "parent": "Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX (untouched)",
        "date": DATE,
        "kind": "UX SIMPLIFICATION ONLY — fewer visible controls, secondary actions behind a Menu / popovers / row ••• menu; backend & render pipeline byte-identical to V1.0; AI video OFF.",
        "changed_files": {
            "vidlore/web.py": "Menu dropdown (Project/Video settings/Advanced; click+outside+Escape close); "
                              "reusable .edpop popup primitive; whole-video controls relocated into a 'Whole video settings' "
                              "popover (compact summary line replaces the always-on audiobar; paintAudio fill unchanged); "
                              "scene-row ••• overflow menu (Re-voice/Reset/Preview-or-Restore-card/Delete); drag-drop guard "
                              "(no reorder while a render runs); timeline scene-number-primary block labels (.blknum, narration "
                              "when wide, full in tooltip); wording (Music level->Music volume, Fit to width->Fit whole video); "
                              "Reset-all moved from toolbar into the Menu.",
        },
        "backend_unchanged": ["vidlore/pipeline.py", "vidlore/assemble.py", "vidlore/editor_manifest.py",
                              "(identical to V1.0 — verified by sha in V1.0 snapshot; no edits this pass)"],
        "ux_results": {
            "toolbar": "3 buttons (Undo / Apply & re-render / ☰ Menu) — was 3 + a 6-control audiobar always visible",
            "menu_dropdown": "grouped, closes on outside-click + Escape, only supported actions (no fakes), no console errors",
            "whole_video_popover": "centered modal + backdrop; all 8 global controls; closes on Escape/✕/backdrop",
            "scene_row_overflow": "••• menu with contextual secondary actions; does not select the row",
            "timeline": "scene-number-primary labels (33 chips) + full-narration tooltips",
            "decluttered": "secondary actions hidden-but-discoverable; primary actions stay visible",
        },
        "validation": {
            "chrome_mcp": "Menu open/close (Escape+outside), popover open/close, ••• menu, scene-select->seek(46s), "
                          "timeline-click->seek(62s), preview play, mute, timeline number chips — all Chrome-verified; 0 console errors.",
            "real_rerender": "edit (Background music ON) via popover -> Apply -> real job; new MP4 produced.",
            "qa_metrics": qa.get("qa", {}),
            "black_frames": qa.get("black_frames"),
            "audio_lufs": (qa.get("audio") or {}).get("lufs"),
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "use_only_audio_in_dist": 0,
            "ai_video": "OFF",
        },
        "reports": [
            "research/review_editor/CAPCUT_UX_AUDIT.md",
            "research/review_editor/CAPCUT_UX_MANUAL_QA.md",
            "research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md",
        ],
        "screenshots": "research/review_editor/manual_review/ (v11_* if captured) + session screenshots "
                       "ss_6063l9pap (clean toolbar), ss_5411fdyrp (Menu), ss_87981zght (popover), "
                       "ss_68886ch8o (row ••• menu), ss_5822i6fsx (timeline numbers)",
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into dist/Vidlore-Mac/vidlore/ + dist/Vidlore-Windows/vidlore/ (byte-identical); sha-verify.",
            "Backend untouched this pass — reverting web.py fully restores V1.0 editor UX. The V1.0 snapshot "
            "(Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX) remains a deeper rollback point.",
        ],
        "remaining_minor": [
            "Inspector full 8->4 section restructure (light touch only; already had collapsible sections; "
            "major declutter achieved via Menu + popover + ••• instead).",
            "Timeline drag-reorder (kept list-only for safety).",
            "First-open manifest+timeline build file-lock; deep upload codec validation (carried from V1.0).",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    print(f"  QA verdict from re-render: {qa.get('qa', {}).get('verdict', '(none)')}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
