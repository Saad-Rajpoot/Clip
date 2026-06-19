#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.4.1_SceneSelectionSyncFix snapshot.

Pass 7 — fixes the user-reported scene-selection ↔ preview off-by-one: clicking a
scene seeked the preview to scene.start+0.05s, landing inside the incoming dissolve
(assemble.py dissolves run up to ~0.85s) so the flattened MP4 still showed the PREVIOUS
scene. Fix (web.py, preview-only): __edSeekAnchor (dissolve-clearing safe anchor),
unify selection on the stable scene_index (reorder-safe), final-scene active clamp, and
inspector-follow on paused scrubs. ONLY vidlore/web.py changed; editor_manifest.py /
pipeline.py / assemble.py are byte-identical to V1.4.

Copies the editor source + the regression test, records sha256 (incl. dist copies =
0-drift proof), and writes the manifest. Does NOT overwrite V1.0 / V1.1 / V1.2 / V1.3 /
V1.4.

    .venv/bin/python tools/build_review_editor_v141_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.4.1_SceneSelectionSyncFix"
DATE = "2026-06-03"
SRCS = ["vidlore/web.py", "tools/test_review_editor_repairs.py"]
DIST_TRACKED = ["vidlore/web.py"]   # the only shipped file changed this pass
PRIOR = ["Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
         "Vidlore_ReviewEditor_V1.1_CapCutCleanUX",
         "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish",
         "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX",
         "Vidlore_ReviewEditor_V1.4_LivePreviewLayers"]


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
        "label": "Vidlore Review Editor V1.4.1 — Scene-Selection ↔ Preview Sync Fix (click a scene → that scene shows, never the previous)",
        "parent": "Vidlore_ReviewEditor_V1.4_LivePreviewLayers (untouched)",
        "date": DATE,
        "kind": "Fix the user-reported scene-selection ↔ preview off-by-one. ONLY vidlore/web.py changed (preview seek + selection mapping); render pipeline + editor_manifest byte-identical to V1.4; AI video OFF.",
        "bug": "Clicking a scene selected it in row/inspector/timeline/playhead, but the center preview showed the PREVIOUS scene — the click seeked to scene.start+0.05s, inside the incoming dissolve (up to ~0.85s) where the flattened MP4 still shows the previous scene.",
        "changed_files": {
            "vidlore/web.py": "(1) __edSeekAnchor(sc) = start+min(max(0.9,dur*0.15),dur*0.5) — a "
                              "dissolve-clearing safe preview anchor, used by __edsel + seekSel (was "
                              "start+0.05 / start+0.03). (2) left scene rows now pass the stable "
                              "sc.scene_index (matching the timeline) so __edsel's findIndex mapping is "
                              "reorder-safe. (3) __edActiveScene clamps the final-scene/end-of-video edge "
                              "(no -1). (4) onTime keeps the inspector on the active scene during a PAUSED "
                              "seek/scrub/playhead-drag (skipped while a #edinsp field is focused).",
        },
        "unchanged": ["vidlore/editor_manifest.py (V1.4)", "vidlore/pipeline.py", "vidlore/assemble.py",
                      "footage ladder / audio engine / visual-relevance / motion-graphics / cache / black-frame repair"],
        "frame_proof": {
            "project": "the-road-to-moscow--napoleon-s-catastrophe-of-1812 (scene 2 starts 7.671s)",
            "before_start_plus_0_05": "research/review_editor/manual_review/nap_sc2_t7.72.png = scene 1's face (the bug)",
            "after_start_plus_0_9": "research/review_editor/manual_review/nap_sc2_t8.57.png = scene 2's Napoleon card (fixed)",
        },
        "real_browser_verification": {
            "tool": "Chrome MCP `computer` (real mouse)",
            "checks": ["scenes 2/3/4/13 reproduced broken on old build, verified fixed (center = selected scene, card+footage, intro→outro)",
                       "seek-bar scrub to 0:53 → inspector follows to 'Scene 9 MASSACRE' (was lagging on Scene 2)",
                       "click-after-scrub jumps cleanly to the clicked scene",
                       "page refresh loads clean (Scene 1, title card)"],
            "console_errors": 0,
            "docs": ["research/review_editor/SCENE_SELECTION_SYNC_AUDIT.md",
                     "research/review_editor/SCENE_SELECTION_SYNC_MANUAL_QA.md"],
        },
        "validation": {
            "regression": "tools/test_review_editor_repairs.py -> 51/51 (17 new: seek-anchor math incl. dissolve-clearing floor + short-scene clamp + old-0.05 regression guard, stable scene_index->pos mapping incl. reordered, half-open active-scene ranges + final-scene clamp)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "ai_video": "OFF",
        },
        "out_of_scope_findings": [
            "Reorder still needs a re-render to preview the new order (flattened-MP4 limit — draft layers + selection are correct, only the base footage can't reshuffle live).",
            "Undo doesn't revert the FIRST edit on a fresh project: editor_manifest._save_overrides skips the undo snapshot when prev is None, so undo_stack stays empty. Pre-existing, unrelated — flagged for a separate fix.",
        ],
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into dist/Vidlore-Mac/vidlore/ + dist/Vidlore-Windows/vidlore/ (byte-identical); sha-verify.",
            "editor_manifest.py / pipeline.py / assemble.py untouched this pass -> restoring web.py restores V1.4. V1.4/V1.3/V1.2/V1.1/V1.0 snapshots remain deeper rollback points.",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
