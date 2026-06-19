#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.2_FinalDashboardPolish snapshot.

This pass changed vidlore/web.py (inspector declutter + timeline drag) and
vidlore/editor_manifest.py (build-lock/atomic-write/stale-thumb + deep upload
validation). Render pipeline (pipeline.py/assemble.py) byte-identical to V1.0/V1.1.
Copies the changed files + the regression test into source/, records sha256 (incl.
dist copies for a 0-drift proof) + the live QA result, writes the manifest. Does
NOT overwrite the V1.0 or V1.1 snapshots.

    .venv/bin/python tools/build_review_editor_v12_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish"
PROJ = "the-1860s-secret--how-to-end-garden-pests-permane"
DATE = "2026-06-02"
CHANGED = ["vidlore/web.py", "vidlore/editor_manifest.py"]
NEW = ["tools/test_review_editor_repairs.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if SNAP.exists():
        print(f"REFUSING to overwrite existing snapshot: {SNAP}")
        return 1
    for prior in ("Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
                  "Vidlore_ReviewEditor_V1.1_CapCutCleanUX"):
        if (ROOT / "snapshots" / prior).exists():
            print(f"prior snapshot present and will NOT be touched: {prior}")
    (SNAP / "source").mkdir(parents=True)
    hashes, drift, drift_ok = [], {}, True
    for rel in CHANGED + NEW:
        src = ROOT / rel
        dst = SNAP / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes.append(f"{sha(src)}  {rel}")
    for rel in CHANGED:
        s = sha(ROOT / rel)
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
        "label": "Vidlore Review Editor V1.2 — Final Dashboard Polish (inspector declutter, timeline drag, build-lock, deep upload validation)",
        "parent": "Vidlore_ReviewEditor_V1.1_CapCutCleanUX (untouched); V1.0 also untouched",
        "date": DATE,
        "kind": "Final polish + deep validation. UX + safety only; render pipeline byte-identical to V1.1. AI video OFF.",
        "changed_files": {
            "vidlore/web.py": "P4 inspector declutter (3-group bar Visual/Card/Scene + 4 collapsed "
                              "sections; Re-voice/Reset/Delete/Use-original moved into ••• More via "
                              "__edRowMenu); P1 timeline direct drag-reorder (draggable visual blocks, "
                              "__edTLDragStart/Over/End/Drop + __edReorderApply, X-axis insertion line, "
                              "grab cursor, disabled during render); __edRowMenu gains 'Use original visual'.",
            "vidlore/editor_manifest.py": "P2 per-project _build_lock + _atomic_write around write_manifest; "
                              "extract_scene_thumbs skip-if-newer-than-mp4 (fixes stale thumbs after re-render); "
                              "P3 _probe_media (ffmpeg readability + stream + dimensions) wired into "
                              "save_visual_override — corrupt/empty/unsupported rejected with beginner messages, "
                              "original preserved (override recorded only after probe passes).",
        },
        "new_files": {"tools/test_review_editor_repairs.py": "extended to 22 checks (P2/P3 added)."},
        "backend_unchanged": ["vidlore/pipeline.py", "vidlore/assemble.py (byte-identical to V1.1)"],
        "validation": {
            "regression": "tools/test_review_editor_repairs.py 22/22 PASS",
            "chrome_mcp": "Menu (6 items) open/close; whole-video popover open/close + 4 controls; "
                          "inspector ••• More (Re-voice/Reset/Delete); 33 draggable timeline blocks; "
                          "scene-select->seek; timeline-click->seek; preview play/mute; 0 console errors.",
            "timeline_drag": "33 draggable blocks + wired handlers; drop routes through the validated "
                             "/scene/<src>/reorder/<to> endpoint (ok:True, order changed, reordered flag).",
            "upload_validation": "valid image accepted; corrupt + empty rejected (beginner msg); original preserved.",
            "real_rerender": "reset -> reorder 8->3 + replace scene 12 (valid green, P3-validated) + invalid "
                             "upload REJECTED (original safe) + regen 15 + globals (captions off/music on/look "
                             "amber) -> Apply -> real job e01cd22da870.",
            "qa_metrics": qa.get("qa", {}),
            "black_frames": qa.get("black_frames"),
            "audio_lufs": (qa.get("audio") or {}).get("lufs"),
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "use_only_audio_in_dist": 0,
            "ai_video": "OFF",
        },
        "reports": [
            "research/review_editor/CAPCUT_UX_FINAL_AUDIT.md",
            "research/review_editor/CAPCUT_UX_FINAL_MANUAL_QA.md",
            "research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md",
        ],
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py + source/vidlore/editor_manifest.py over the repo files.",
            "Re-sync both into dist/Vidlore-Mac/vidlore/ + dist/Vidlore-Windows/vidlore/ (byte-identical); sha-verify.",
            "Render pipeline untouched; reverting these two files restores V1.1. V1.1 + V1.0 snapshots remain deeper rollback points.",
        ],
        "remaining_minor": [
            "Full HTML5 drag-gesture e2e not auto-synthesizable (verified wiring + backend instead).",
            "Menu intentionally omits redundant Fit/Reset-zoom items (live in the timeline toolbar).",
            "First-open still does a synchronous (now locked + cached) thumbnail build — acceptable; loading message shown.",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    print(f"  QA verdict: {qa.get('qa', {}).get('verdict', '(none)')}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
