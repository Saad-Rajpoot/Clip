#!/usr/bin/env python3
"""Build the Vidlore_ReviewEditor_V1.4.4_LiveTextOverlayFix snapshot.

Fixes the duplicate / oversized live-text overlay bug: the V1.4 draft engine drew an
HTML card approximation on top of EVERY card scene — but the base MP4 already contains
the baked card, so every card was doubled (and complex kinds fell through to a giant
full-screen overlay). FIX (vidlore/web.py, preview-only): draw a card overlay ONLY when
the card was edited this session (sc.card.edited); for edited simple kinds a clean
compact draft, for edited complex kinds an honest note, for unedited cards nothing
(the base MP4 is correct). + typography hard-limits.

ONLY vidlore/web.py changed; render pipeline + editor_manifest untouched (the card
payload + final render are unaffected — the duplication only existed in the live HTML
preview). Records sha256 (incl. dist copies). Does NOT overwrite any earlier snapshot.

    .venv/bin/python tools/build_review_editor_v144_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "snapshots" / "Vidlore_ReviewEditor_V1.4.4_LiveTextOverlayFix"
DATE = "2026-06-03"
SRCS = ["vidlore/web.py", "tools/test_review_editor_repairs.py"]
DIST_TRACKED = ["vidlore/web.py"]
PRIOR = ["Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX",
         "Vidlore_ReviewEditor_V1.1_CapCutCleanUX",
         "Vidlore_ReviewEditor_V1.2_FinalDashboardPolish",
         "Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX",
         "Vidlore_ReviewEditor_V1.4_LivePreviewLayers",
         "Vidlore_ReviewEditor_V1.4.1_SceneSelectionSyncFix",
         "Vidlore_ReviewEditor_V1.4.2_LiveInteractionCompleteness"]


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
        "label": "Vidlore Review Editor V1.4.4 — Live Text Overlay Duplication Fix",
        "parent": "Vidlore_ReviewEditor_V1.4.2_LiveInteractionCompleteness (+ the V1.4.3 caption-emit task; both untouched)",
        "date": DATE,
        "kind": "Stop the editor drawing duplicate / oversized HTML text on top of already-baked cards. ONLY vidlore/web.py changed (live draft layer); render pipeline + card payload untouched.",
        "bug": "__edDraftSync drew __edDraftCardHTML for EVERY card scene (cardLive = is-a-card), on top of the base MP4 that already contains the baked card → duplicate title + body; complex kinds (statement/classified) fell through to a giant full-screen .dctitle/.dcbody. Reproduced on lower_third 'TEHRAN', statement 'opens now', classified 'EXEC ORDER' — all UNEDITED.",
        "fix": {
            "gate_on_edited": "draw a card overlay ONLY when sc.card.edited (card_text_override exists this session). Unedited card -> no overlay (the base MP4 already shows it correctly). Removes all three duplications.",
            "whitelist": "__edSimpleCardKinds (lower_third/lower/title_card/title/chapter/location/label/name_reveal/name, body<=160) -> clean compact draft via __edDraftCardHTML (which now only emits .dclower / .dctitle). Anything else -> honest note 'Text saved — the final card styling appears after re-render.' (base MP4 kept).",
            "typography_limits": "#edlaycard .dcbody/.dctitle/.dclower line-clamped + overflow:hidden; lower-third max-height; .edcarddraft dashed outline marks it as an editing aid.",
            "cleanup": "__edDraftSync re-runs on scene change / seek / layer toggle / undo / reset / refresh and now hides the card unless edited (no stale overlay).",
        },
        "unchanged": ["vidlore/editor_manifest.py", "vidlore/pipeline.py", "vidlore/assemble.py",
                      "card payload (apply_overrides / card_text_override) -> final MP4 cards unchanged"],
        "real_browser_verification": {
            "tool": "Chrome MCP (real mouse)",
            "checks": ["unedited lower_third / statement / classified -> NO duplicate overlay (3 scenes)",
                       "edit a complex (classified) card -> honest note, not a giant duplicate",
                       "edit a simple (name_reveal) card -> compact lower-third draft",
                       "Undo reverts the edit -> overlay clears",
                       "switch scenes -> no stale overlay"],
            "console_errors": 0,
            "doc": "research/review_editor/LIVE_TEXT_OVERLAY_DUPLICATION_MANUAL_QA.md",
        },
        "validation": {
            "regression": "tools/test_review_editor_repairs.py -> 124/124 (24 new: unedited->none every kind, edited-simple->draft, edited-complex/long-body->honest note, removed->hidden note, source guards)",
            "dist_zero_drift": drift,
            "dist_zero_drift_ok": drift_ok,
            "ai_video": "OFF",
            "final_render": "preview-only fix — card payload + renderer untouched, so the exported MP4 cards are unchanged (the duplication only ever existed in the editor HTML preview).",
        },
        "audit": "research/review_editor/LIVE_TEXT_OVERLAY_DUPLICATION_AUDIT.md",
        "snapshot_path": str(SNAP.relative_to(ROOT)),
        "rollback_steps": [
            "Restore source/vidlore/web.py over vidlore/web.py.",
            "Re-sync web.py into both dist trees (byte-identical); sha-verify.",
            "Backend untouched -> restoring web.py restores the prior pass.",
        ],
        "note_dist_run": "The live app is being run from inside dist/Vidlore-Mac (OUT = dist/Vidlore-Mac/output), a known hygiene quirk — projects render there. Unrelated to this fix.",
    }
    (SNAP / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Snapshot built: {SNAP}")
    print(f"  dist 0-drift: {'YES' if drift_ok else 'NO — FIX FIRST'}")
    return 0 if drift_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
