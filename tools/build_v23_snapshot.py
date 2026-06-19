"""Build the MG_Cluster_V2.3_AssetGuards sibling snapshot.

Permanent, reusable ENGINE-LEVEL guards added after the multi-niche validation
(V2.2.1 left UNTOUCHED as a rollback point). These prevent four classes of issue
automatically for ALL future videos/topics — not just the validation samples:

  NEW reusable modules (pure-logic + unit-tested, tools/test_engine_guards.py):
   • portrait_intel.py  — pre-photographic detection → prefer verified painting/
     engraving/PD-illustration over modern photo/AI; strict name match; provenance.
   • period_guard.py    — historical-scene era detection + period-risk + modern-
     marker rejection + era-biased queries + safe fallback order.
   • niche_palette.py   — niche-aware WEIGHTED palette (crime→ember, not warm gold)
     + per-video variation + cross-video anti-repeat + reason logging.
   • card_style_guard.py— dark-niche text-card gating (route bright statement cards
     to a dark variant unless an authorised rare contrast beat).
   • asset_qa.py        — reusable QA layer: flags mismatched/pre-photo-modern-face
     portraits, modern footage in period scenes, palette/niche mismatch, bright
     card in a dark niche, uncertain provenance (low confidence → safer fallback).

  CHANGED (minimal, defensive hooks — each try/except, old behaviour on failure):
   • motion_graphics/director.py — video_palette() routes through niche_palette;
     adds video_palette_reason().
   • footage.py — portrait sourcing prefers artwork queries + AI oil-painting prompt
     for pre-photographic people; per-video era hint biases stock queries; the
     statement-card dispatch routes dark niches to the dark variant.
   • pipeline.py — reusable asset-QA pass records palette reason + warnings into
     motion_graphics_manifest.json.

No new MG primitives (39 unchanged). assemble.py / look.py byte-identical.
Self-consistent hashes (source == mac == win) verified against the dist trees.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
PARENT_SNAP = Path("snapshots/MG_Cluster_V2.2.1_DispatchKwargGuard")
NEW = Path("snapshots/MG_Cluster_V2.3_AssetGuards")

NEW_FILES = [
    "vidlore/portrait_intel.py",
    "vidlore/period_guard.py",
    "vidlore/niche_palette.py",
    "vidlore/card_style_guard.py",
    "vidlore/asset_qa.py",
]
CHANGED = {
    "vidlore/footage.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/pipeline.py",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


parent_manifest = json.loads((PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
parent_hashes = parent_manifest["file_hashes_sha256"]
FILES = list(parent_manifest["files"]) + NEW_FILES
for f in CHANGED:
    assert f in parent_manifest["files"], f"CHANGED not tracked by parent: {f}"
for f in NEW_FILES:
    assert f not in parent_manifest["files"], f"NEW already tracked: {f}"

if NEW.exists():
    shutil.rmtree(NEW)
(NEW / "source").mkdir(parents=True)
if (PARENT_SNAP / "source/vidlore/assets").exists():
    shutil.copytree(PARENT_SNAP / "source/vidlore/assets",
                    NEW / "source/vidlore/assets")

hashes, drift = {}, []
for f in FILES:
    dst = NEW / "source" / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dst)
    hs = sha(f)
    hm = sha(f"dist/Vidlore-Mac/{f}")
    hw = sha(f"dist/Vidlore-Windows/{f}")
    hashes[f] = {"source": hs, "mac": hm, "win": hw}
    if not (hs == hm == hw):
        drift.append(f)

hash_change = {}
for f in sorted(CHANGED):
    old = parent_hashes.get(f, {}).get("source", "(absent)")
    new = hashes[f]["source"]
    hash_change[f] = {"v2_2_1_source": old, "v2_3_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V2.3 — Asset guards (portrait/period/palette/card/QA) · sha256",
         "# parent: V2.2.1 (untouched); +5 reusable engine modules, 3 changed (39 primitives)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V2.3 — Asset guards (pre-photo portrait / period footage / niche palette / dark-niche card / QA layer)",
    "parent": ("Motion Graphics Cluster V2.2.1 — Dispatch kwarg-guard "
               "(snapshots/MG_Cluster_V2.2.1_DispatchKwargGuard, untouched)"),
    "kind": "permanent reusable engine guards from multi-niche validation (no new primitives, 39 total)",
    "new_modules": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "tests": "tools/test_engine_guards.py (84 assertions: 8 portraits, 6 eras, "
             "crime-palette weighting/variation/anti-repeat, dark-niche cards, QA checks)",
    "validated": (
        "84/84 unit tests pass. Re-rendered Napoleon (history) / Al Capone (crime) "
        "/ Eli Cohen (spy) as regression: Napoleon → verified period artwork/painting "
        "portrait; crime palette → ember_red (on-genre) with variation preserved; "
        "spy dark statement card → dark variant; period footage era-biased; gates held "
        "(0 black, ~-16 LUFS, temp clean). See research/motion_graphics_qa/"
        "engine_guards_report.md."),
    "hash_change_v2_2_1_to_v2_3": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No new MG primitives (39 unchanged)", "Editor UI untouched", "No deploy",
        "No beat-split", "assemble.py / look.py byte-identical to V2.2.1",
        "All hooks defensive (try/except → legacy behaviour on failure)",
        "V2.2.1 … V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 5 new modules (vidlore/portrait_intel.py, period_guard.py, "
        "niche_palette.py, card_style_guard.py, asset_qa.py).",
        "2. cp footage.py / motion_graphics/director.py / pipeline.py from "
        "snapshots/MG_Cluster_V2.2.1_DispatchKwargGuard/source/ into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V2.3 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v2_2_1_source'])[:16]} -> {c['v2_3_source'][:16]}")
print("V2.2.1 parent untouched:", PARENT_SNAP.exists())
