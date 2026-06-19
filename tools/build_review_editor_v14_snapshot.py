#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.4_LivePreviewLayers snapshot.

This pass replaced the V1.3 corner-pip with a TRUE live draft preview layer stack:
the MAIN center preview now updates immediately (full-stage visual replacement,
live card-text, per-scene layer on/off) without a re-render — re-render only bakes
the final MP4. Touches vidlore/web.py (draft layer engine + Layers panel + silent
live-text save) and vidlore/editor_manifest.py (set_layer + layers_off in
edit_status). The production renderer (pipeline.py, assemble.py, footage/audio/
visual-relevance/motion-graphics/cache/black-repair) is byte-identical to V1.3.

Copies the editor + manifest source + the regression test, records sha256 (incl.
dist copies = 0-drift proof), reads the live render_metrics.json from the
footage-scene render proof, and writes the manifest. Does NOT overwrite
V1.0 / V1.1 / V1.2 / V1.3.

    .venv/bin/python tools/build_review_editor_v14_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.4_LivePreviewLayers"
PROJ = "the-1860s-secret--how-to-end-garden-pests-permane"
DATE = "2026-06-02"
SRCS = ["vidlore/web.py", "vidlore/editor_manifest.py", "tools/test_review_editor_repairs.py"]
PRIOR = ["Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
         "Vidlore_ReviewEditor_V1.1_CapCutCleanUX",
         "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish",
         "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX"]


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
        "label": "Vidlore Review Editor V1.4 — TRUE Live Preview Layer System (full-stage replace, live card-text, per-scene layer on/off; re-render only for final MP4)",
        "parent": "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX (untouched)",
        "date": DATE,
        "kind": "Replace the V1.3 corner-pip with an additive draft layer stack ABOVE the rendered MP4 so the MAIN preview updates instantly (CapCut-style). Production renderer byte-identical to V1.3; AI video OFF.",
        "changed_files": {
            "vidlore/web.py": "Draft layer stack in #edpvstage (#edlayvisual full-stage replacement w/ "
                              "Ken-Burns drift + #edlaycard HTML card approximation + #eddraftlbl) replacing "
                              "the corner pip (#edrepov). Engine: __edActiveScene / __edLayerOff / "
                              "__edDraftCardHTML / __edDraftSync (currentTime+selection driven, _draftKey "
                              "cheap-resync guard); onTime()/select() drive it. Context-aware Layers panel "
                              "(_lrow eye toggles -> __edLayer). Live card-text: __edSaveCardSilent (POST "
                              "/card -> refresh manifest/timeline -> __edDraftSync, no inspector re-render = "
                              "keeps field focus) + 600ms debounce. CSS for .edlay/#edlayvisual/.kb/"
                              ".dc*/.eddraftlbl/.edlayers.",
            "vidlore/editor_manifest.py": "set_layer(run_dir, idx, name, visible): card->card_removed "
                              "(reaches final render), captions->global.captions_enabled (whole-video), "
                              "visual/text->per-scene layers_off (draft). edit_status now exposes layers_off. "
                              "_history_append for Undo.",
        },
        "unchanged_backend": ["vidlore/pipeline.py", "vidlore/assemble.py",
                              "footage ladder / audio engine / visual-relevance / motion-graphics / cache / black-frame repair"],
        "live_preview_model": {
            "base": "<video id=edvid> = last rendered MP4 (source of truth, unchanged)",
            "draft_layers": "#edlayvisual (replaced image/clip, full stage), #edlaycard (live card text), #eddraftlbl",
            "driven_by": "__edDraftSync() from onTime() (playback) + select() (seek); shown only during active scene",
            "persistence": "layer state -> scene override -> final renderer (card hidden -> card_removed honored)",
            "honest_limitation": "cannot reveal un-rendered footage behind a card in the DRAFT (flattened MP4); shows honest note + still removes the card from the exported MP4 (verified).",
        },
        "real_browser_verification": {
            "tool": "Chrome MCP `computer` (real CDP mouse) at screenshot-pixel coords",
            "checks": ["P1 select footage scene -> search-replace -> FULL center stage shows new visual + draft label, corner pip gone",
                       "P2 edit card title -> draft card overlay updates live + persists (no focus loss)",
                       "P3 Layers panel context-aware (footage->Captions; card->Card/Graphic+Captions)",
                       "P4 card-hide eye -> overlay gone + footage + honest note; CARD REMOVED badge; cardRemovedInManifest:true",
                       "P8 draft layers swap per active scene via currentTime"],
            "console_errors": 0,
            "doc": "research/review_editor/LIVE_PREVIEW_LAYER_MANUAL_QA.md",
        },
        "render_proof": {
            "footage_replace": "scene idx 4 (FOOTAGE, baked gk='') replaced with an orange-sunset image -> VISUALLY confirmed FULL-FRAME in final MP4 @~30s (narration 'The Amish families…')",
            "card_text": "scene idx 3 (cause_effect CARD) text -> 'POISON BACKFIRE TEST' (BUGS BUILD IMMUNITY -> SPRAYING NEVER ENDS) baked into final MP4 @~25s",
            "card_hidden": "scene idx 1 (was callout CARD, baked gk='') -> node-diagram callout GONE; scene now plays as footage (pest montage + 'AGAIN') @~4.8s",
            "job": "bfc43053d5c4",
            "method": "each change LOCATED by its narration line in the .srt (NOT by summing scene durations) then the exact frame extracted + viewed. Per-scene time drifts heavily in the assembled MP4 (crossfade overlaps + card timing).",
            "test_bugs_caught": ["attempt-1 (job 81b1251611ad): sampled wrong timestamps + hid a card on a footage scene (no-op) — caught by viewing frames",
                                 "attempt-2 (this job): hardcoded times AND a mean-RGB sunset-detector both failed (drift + color grade); fixed by locating scenes via .srt narration then looking"],
            "qa": qa.get("qa", {}),
            "black_frames": qa.get("black_frames"),
            "audio_lufs": (qa.get("audio") or {}).get("lufs"),
            "frames": ["research/review_editor/manual_review/v14p2_idx4_at30.png",
                       "research/review_editor/manual_review/v14p2_idx3_poison_at25.png",
                       "research/review_editor/manual_review/v14p2_idx1_at4.8.png"],
        },
        "validation": {
            "regression": "tools/test_review_editor_repairs.py -> 34/34 (7 new layer tests incl. hidden card removed from rendered script.json)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "use_only_audio_in_dist": 0,
            "ai_video": "OFF",
        },
        "reports": [
            "research/review_editor/LIVE_PREVIEW_LAYER_AUDIT.md",
            "research/review_editor/LIVE_PREVIEW_LAYER_MANUAL_QA.md",
            "research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md",
            "research/review_editor/CURRENT_STATE.md",
        ],
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py + source/vidlore/editor_manifest.py over vidlore/.",
            "Re-sync both into dist/Vidlore-Mac/vidlore/ + dist/Vidlore-Windows/vidlore/ (byte-identical); sha-verify.",
            "pipeline.py + assemble.py untouched this pass -> restoring the two files restores V1.3. V1.3/V1.2/V1.1/V1.0 snapshots remain deeper rollback points.",
        ],
        "remaining_minor": [
            "Native file picker not driveable via Chrome MCP (upload path proven via prior render + search-pick).",
            "Draft card is a tasteful HTML approximation, not pixel-identical to the FFmpeg-rendered card.",
            "Footage-behind-card reveal shown as an honest note in the draft (flattened-MP4 constraint); still removed from final MP4.",
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
