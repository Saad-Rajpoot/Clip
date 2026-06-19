"""Build the MG_Cluster_V2.2.1_DispatchKwargGuard sibling snapshot.

A small ROBUSTNESS fix on top of V2.2 (Spotlight/Decision/World-Arc), found during
the multi-niche real validation. V2.2 is LEFT UNTOUCHED as a rollback point.

What changed since V2.2 — exactly ONE file:
  • render_dispatch.py — defensive kwarg-filter in dispatch(): pass only the kwargs
    a primitive's render() actually declares. A name_reveal scene folds a `place=`
    hint into its assets (for the sibling portrait_name_over_map); when the director
    instead picks cinematic_portrait_hold (which takes portrait_path, not place),
    the stray `place`/`name` kwarg used to crash the render → silent fallback (the
    subject portrait vanished). The filter kills this whole class of bug. Primitives
    with **kwargs still receive everything.

No primitives added/removed (39, unchanged). No other files touched. Verified by
re-rendering the 3 affected multi-niche samples → 0 fallbacks, portraits restored.

Self-consistent hashes (source == mac == win) verified against the dist trees,
which must be synced FIRST.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
PARENT_SNAP = Path("snapshots/MG_Cluster_V2.2_SpotlightDecisionArc")
NEW = Path("snapshots/MG_Cluster_V2.2.1_DispatchKwargGuard")

NEW_FILES = []
CHANGED = {"vidlore/motion_graphics/render_dispatch.py"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


parent_manifest = json.loads((PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
parent_hashes = parent_manifest["file_hashes_sha256"]
FILES = list(parent_manifest["files"])           # 63 files carried forward verbatim
for f in CHANGED:
    assert f in FILES, f"CHANGED file not in parent manifest: {f}"

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
    old = parent_hashes.get(f, {}).get("source", "(absent in V2.2)")
    new = hashes[f]["source"]
    hash_change[f] = {"v2_2_source": old, "v2_2_1_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V2.2.1 — Dispatch kwarg-guard · sha256",
         "# parent: V2.2 Spotlight/Decision/World-Arc (untouched); 1 robustness fix (39 primitives)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = "  [CHANGED]" if f in CHANGED else ""
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V2.2.1 — Dispatch kwarg-guard",
    "parent": ("Motion Graphics Cluster V2.2 — Spotlight-Reveal / Decision-Fork / "
               "World-Arc (snapshots/MG_Cluster_V2.2_SpotlightDecisionArc, untouched)"),
    "kind": "robustness fix found during multi-niche validation (no primitive change, 39 total)",
    "fix": (
        "render_dispatch.dispatch() now filters render kwargs to the primitive's "
        "declared signature (inspect.signature; **kwargs primitives still get all). "
        "Fixes: cinematic_portrait_hold / map_route_spread crashed with 'unexpected "
        "keyword argument place/name' when a name_reveal scene's place= hint was "
        "folded into assets and the director picked the photo (not map) portrait — a "
        "silent fallback that dropped the subject reveal in crime/business/history."),
    "validated": (
        "Unit: dispatch(cinematic_portrait_hold, inputs={portrait_path, name, place, "
        "sub}) → ok=True, fallback=False. Real: re-rendered crime/business/history → "
        "0 fallbacks (was 1/1/2); Capone/Carnegie/Napoleon portraits restored. All 5 "
        "multi-niche samples: 0 black, ~-16 LUFS, peak <= -1.9 dBFS, 0 temp leak. See "
        "research/motion_graphics_qa/multiniche/CROSS_NICHE_REPORT.md."),
    "cost": "fal images on the 3 re-renders only (~$0.045); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v2_2_to_v2_2_1": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No new primitives (39 unchanged)", "No beat-split", "Editor UI untouched",
        "No deploy", "footage.py / assemble.py / look.py / director.py / registry.py "
        "/ pipeline.py byte-identical to V2.2",
        "Only render_dispatch.py changed (one defensive filter)",
        "V2.2 … V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "cp render_dispatch.py from snapshots/MG_Cluster_V2.2_SpotlightDecisionArc/"
        "source/ into vidlore/motion_graphics/, then re-sync to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V2.2.1 snapshot built:", NEW)
print("files:", len(FILES), "· changed:", len(CHANGED), "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v2_2_source'])[:16]} -> {c['v2_2_1_source'][:16]}")
print("V2.2 parent untouched:", PARENT_SNAP.exists())
