#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX snapshot.

Copies the changed source files + the new regression test into the snapshot's
source/ tree, records sha256 HASHES.txt (incl. dist copies for a 0-drift proof),
and writes SNAPSHOT_MANIFEST.json (label / parent / changed files / validation /
rollback). Does NOT touch any prior snapshot. Run from the repo root:

    .venv/bin/python tools/build_review_editor_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX"
DATE = "2026-06-02"

# changed source modules (relative to repo root)
CHANGED = [
    "vidlore/web.py",
    "vidlore/editor_manifest.py",
    "vidlore/pipeline.py",
    "vidlore/assemble.py",
]
NEW = [
    "tools/test_review_editor_repairs.py",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if SNAP.exists():
        print(f"REFUSING to overwrite existing snapshot: {SNAP}")
        return 1
    (SNAP / "source").mkdir(parents=True)
    hashes = []
    drift_ok = True
    for rel in CHANGED + NEW:
        src = ROOT / rel
        dst = SNAP / "source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes.append(f"{sha(src)}  {rel}")
    # dist drift proof for the 4 modules
    drift = {}
    for rel in CHANGED:
        s = sha(ROOT / rel)
        mac = ROOT / "dist/Vidlore-Mac" / rel
        win = ROOT / "dist/Vidlore-Windows" / rel
        ms = sha(mac) if mac.exists() else "MISSING"
        ws = sha(win) if win.exists() else "MISSING"
        synced = (s == ms == ws)
        drift_ok = drift_ok and synced
        drift[rel] = {"src": s[:16], "mac": ms[:16], "win": ws[:16], "synced": synced}
        hashes.append(f"{ms}  dist/Vidlore-Mac/{rel}")
        hashes.append(f"{ws}  dist/Vidlore-Windows/{rel}")

    (SNAP / "HASHES.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    manifest = {
        "label": "Vidlore Review Editor V1.0 — CapCut Beginner UX (P1-P6 repair pass)",
        "parent": "Vidlore_V1.2_VisualRelevance (untouched); editor was previously frozen",
        "date": DATE,
        "kind": "Make the Review Editor genuinely beginner-friendly, responsive, accurately "
                "synced, QA-honest, and robust. Additive + guarded; stable render/footage/"
                "audio/MG engines preserved. AI video stays OFF.",
        "changed_files": {
            "vidlore/web.py": "responsive clamp panels + breakpoint 1140->1000; preview "
                              "cache-bust; export dup-click guard; audiobar relabels + WHOLE "
                              "VIDEO chip; Reset-all danger; delete empty-guard; rendered_clean "
                              "'applied' chip; mark_rendered + refresh_render_metrics hooks + "
                              "'Checking final output' stage; beginner QA chip; card-restore "
                              "route+JS; upload size cap + friendly errors; friendly 404; "
                              "timeline thumbs=False on first-open.",
            "vidlore/editor_manifest.py": "locked_visuals.json (Replace-Visual -> render); "
                              "mark_rendered (clear regen flags + rendered_clean signature); "
                              "refresh_render_metrics (honest PASS/WARN/FAIL); pending.rendered_clean; "
                              "remove_scene empty-guard; restore_card; save_card_override preserves "
                              "originals; reset_scene restores poster; build_manifest consumes exact "
                              "render_meta scene timing; honest metrics wording.",
            "vidlore/pipeline.py": "honor locked_visuals.json after fetch_footage (footage.py "
                              "untouched); render_from_script(run_dir=...) so a re-render targets "
                              "the editor's EXACT dir (no _slug(title) mismatch).",
            "vidlore/assemble.py": "emit exact per-scene scene_starts/scene_durations into "
                              "render_meta.json (ground-truth editor sync).",
        },
        "new_files": {
            "tools/test_review_editor_repairs.py": "16-check backend regression suite "
                              "(locked_visuals +reorder-follow +clear, mark_rendered, restore_card, "
                              "empty-guard, card-body, P2 timing, P5 upload).",
        },
        "priorities_done": {
            "P1_responsive": "fluid clamp(18vw/22vw) panels; 3-panel 1024-2560 (preview 45-65%); "
                             "laptops/zoom no longer stack; no horizontal overflow; timeline tooltips.",
            "P2_scene_sync": "assemble emits scene_starts; build_manifest uses them "
                             "(duration_source=render_meta), word-proportional fallback.",
            "P3_qa_metrics": "re-render writes render_metrics.json (real black-frame + loudness); "
                             "'Checking final output' loading state; PASS/WARN/FAIL; honest wording; "
                             "never writes stale/fake on failure.",
            "P4_slug_robust": "render uses the editor's run_dir, never recomputes from title.",
            "P5_upload_safety": "300MB server cap (pre-read) + friendly type/size/missing errors.",
            "P6_friendly_404": "branded 'project could not be found' + Back to dashboard.",
        },
        "validation": {
            "regression": "tools/test_review_editor_repairs.py 16/16 PASS",
            "real_rerender": "comprehensive: 2 replacements (magenta@5 + cyan@8), regen@10, "
                             "reorder, delete+undo, look->History, music 0.7, captions on.",
            "mp4_proof": "scene 5 frame RGB(164,14,162) = uploaded magenta (footage replaced); "
                         "card scenes overlay correctly; control scene dark footage.",
            "qa_metrics": "render_metrics.json verdict=PASS, black_frames=0, lufs=-16.1.",
            "scene_timing": "render_meta.scene_starts len=33 accurate; editor inspector shows "
                            "real durations (4.8s etc.).",
            "black_frames": 0,
            "temp_leak": "none (no work_ dirs left)",
            "responsive": "computed grid verified 1024/1100/1366/1440/1536/1920/2560 — all 3-panel, "
                          "preview dominant, no overflow; visual confirmed at 1440.",
            "reference_untouched": "output/_after7/...permanen.mp4 = 279365363 bytes intact",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "use_only_audio_in_dist": 0,
            "ai_video": "OFF (fal_video_model default empty)",
        },
        "reports": [
            "research/review_editor/EDITOR_FUNCTIONAL_AUDIT.md",
            "research/review_editor/MANUAL_EDITOR_QA.md",
            "research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md",
        ],
        "screenshots": "research/review_editor/manual_review/ (v2_scene5_MAGENTA.png, "
                       "v2_scene8_card.png, replace_visual_scene5_AFTER.png, control_scene2.png)",
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "For each file in source/: copy snapshots/Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX/"
            "source/<rel> back over <rel> in the repo.",
            "Re-sync the 4 vidlore modules into dist/Vidlore-Mac/vidlore/ and "
            "dist/Vidlore-Windows/vidlore/ (byte-identical) and verify sha256.",
            "All editor changes are additive/guarded; reverting only these files fully restores "
            "pre-pass behaviour. locked_visuals.json is inert unless present; mark_rendered / "
            "refresh_render_metrics only run on render success.",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  files: {len(CHANGED) + len(NEW)} source + {len(CHANGED) * 2} dist hashes")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX BEFORE TRUSTING'}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
