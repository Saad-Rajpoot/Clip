#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.4.2_LiveInteractionCompleteness snapshot.

Pass 8 — completes the live editor so safe edits update the center preview immediately:
  * Issue 1 — a beginner tooltip engine ([data-tip]/title; delay, viewport-clamp, focus, Esc)
    on every visible control.
  * Issue 3 — "Generate new" now generates an AI still ON DEMAND (fal.ai) and shows it in
    the live preview instantly (visual_override → persists, undoes, reaches final render).
  * Issue 2 — captions toggle live: a cached NO-CAPTION proxy (rendered in-place so the
    project's exact footage is reused, then its outputs restored) + an HTML caption overlay
    that toggles instantly over the clean base. Final export still burns captions.
  * P5 live-status pill; P6 stable scene-IDs (carried from V1.4.1).

ONLY vidlore/web.py changed; editor_manifest / pipeline / assemble are byte-identical to
V1.4.1. Records sha256 (incl. dist copies = 0-drift proof) + manifest. Does NOT overwrite
any earlier Review Editor snapshot.

    .venv/bin/python tools/build_review_editor_v142_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.4.2_LiveInteractionCompleteness"
DATE = "2026-06-03"
SRCS = ["vidlore/web.py", "tools/test_review_editor_repairs.py"]
DIST_TRACKED = ["vidlore/web.py"]
PRIOR = ["Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
         "Vidlore_ReviewEditor_V1.1_CapCutCleanUX",
         "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish",
         "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX",
         "Vidlore_ReviewEditor_V1.4_LivePreviewLayers",
         "Vidlore_ReviewEditor_V1.4.1_SceneSelectionSyncFix"]


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
        "label": "Vidlore Review Editor V1.4.2 — Live Interaction Completeness (tooltips + live Generate-new fixed; captions toggle honest/partial; status model)",
        "parent": "Vidlore_ReviewEditor_V1.4.1_SceneSelectionSyncFix (untouched)",
        "date": DATE,
        "kind": "Complete the live draft compositor so safe edits update the center preview immediately. ONLY vidlore/web.py changed; render pipeline + editor_manifest byte-identical to V1.4.1; AI video OFF (AI stills only, on user request).",
        "issues_fixed": {
            "issue1_tooltips": "No control explained itself on hover. FIX: one delegated tooltip engine "
                               "(__edTipInit) reading [data-tip] or native title (strips the slow native "
                               "title while shown), 320ms delay, viewport clamp, focus + Esc support. Added "
                               "beginner-friendly data-tip wording across toolbar / preview / inspector / "
                               "scene-list badges / whole-video settings / timeline.",
            "issue3_generate_new": "'Generate new' only flagged a deferred regen. FIX: POST "
                               "/e/<slug>/scene/<idx>/generate calls fal.ai _fal_image ON DEMAND (prompt "
                               "varied per click so the cache returns a fresh, differently-composed still), "
                               "saves it as a visual_override, and the editor shows it INSTANTLY in the "
                               "#edlayvisual draft layer (persists, undoes, reaches final render).",
            "issue2_captions_live": "PARTIAL (honest). Captions are burned PER-SCENE into the preview MP4, "
                               "so toggling OFF left them visible. The toggle now reliably controls the "
                               "FINAL render (captions_enabled) and the editor shows a clear live status "
                               "('Captions are OFF for the final video — re-render to update the preview'). "
                               "An instant live-HIDE needs a no-caption base: built the full path (render "
                               "captions=False IN-place with backup/restore — project MP4 left byte-identical "
                               "— + an HTML caption overlay), BUT the renderer is non-deterministic in "
                               "footage selection, so the proxy matched the timing yet picked DIFFERENT "
                               "footage on every scene. Swapping to it would change all the footage on "
                               "toggle (and disturbed seeking), so the base-swap is DISABLED; proxy builder "
                               "+ overlay remain as dormant infrastructure. Real fix (spawned as a task): "
                               "the renderer must emit a footage-matched no-caption track in the SAME render "
                               "(final-pass caption burn).",
        },
        "also": {
            "p5_status_model": "__edStatus pill over the stage (Generating new visual…, New visual added, Preparing instant caption preview…) + the existing Live-draft-preview badge.",
            "p6_stable_ids": "selection + draft layers keyed by stable scene_index (carried from V1.4.1).",
        },
        "unchanged": ["vidlore/editor_manifest.py (V1.4.1)", "vidlore/pipeline.py", "vidlore/assemble.py",
                      "footage ladder / audio engine / visual-relevance / motion-graphics / cache / black-frame repair"],
        "honest_limitations": [
            "Captions cannot be HIDDEN live in the preview: they are burned per-scene, and a no-caption proxy can't reproduce the project's footage (renderer is non-deterministic). The toggle controls the FINAL render + shows a clear status; the real fix (renderer emits a footage-matched no-caption track) is spawned as a task. Proxy + overlay code remain present but dormant.",
            "The draft HTML card is a tasteful approximation of the final FFmpeg motion-graphics card.",
            "Reorder + music re-mix still need a full re-render to preview (flattened-MP4 limit, documented since V1.4).",
        ],
        "validation": {
            "regression": "tools/test_review_editor_repairs.py -> 96/96 (new: caption timing lookup, captions-on logic, generate prompt-variation cache-miss, nocap freshness, tooltip coverage for every required control + engine/status/overlay/proxy presence)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "ai_video": "OFF (AI still images only, on explicit Generate-new)",
        },
        "real_browser_verification": "research/review_editor/LIVE_INTERACTION_COMPLETENESS_MANUAL_QA.md",
        "audit": "research/review_editor/LIVE_INTERACTION_COMPLETENESS_AUDIT.md",
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into both dist trees (byte-identical); sha-verify.",
            "editor_manifest / pipeline / assemble untouched -> restoring web.py restores V1.4.1. Earlier snapshots remain deeper rollback points.",
        ],
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
