"""Build the MG_Cluster_V1.5_ProportionProcessHierarchy sibling snapshot.

Feature update over V1.4 (Growth / Annotation / Route), LEFT UNTOUCHED as a
rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.4 — MagnatesMedia "Batch 4": three NEW premium, reusable
motion-graphic primitives plus the director/registry/dispatch/pipeline wiring:

  • proportion_ring (charts) — a PARTS-OF-A-WHOLE share: a gold arc sweeps from 12
    o'clock around a faint ring, filling to the percentage, while the numeral
    counts up in the centre + a label names what the share is OF (e.g. 90 %).
  • process_flow_steps (diagrams, NEW family) — 2-5 numbered nodes left→right
    joined by arrows that draw in one after another, each with a short label: an
    ordered mechanism / scheme (buy → undercut → starve → absorb).
  • org_hierarchy_tree (diagrams) — a root node over 2-4 child nodes joined by
    clean elbow connectors, revealed top-down: a power / corporate structure.

Each self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 15 prior
primitives are carried forward verbatim.

Self-consistent hashes (source == mac == win) are verified against the dist
trees, which must be synced FIRST.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
ROOT = Path(".")
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.4_GrowthAnnotationRoute")
NEW = Path("snapshots/MG_Cluster_V1.5_ProportionProcessHierarchy")

NEW_FILES = [
    "vidlore/motion_graphics/diagrams/__init__.py",
    "vidlore/motion_graphics/diagrams/process_flow_steps.py",
    "vidlore/motion_graphics/diagrams/org_hierarchy_tree.py",
    "vidlore/motion_graphics/charts/proportion_ring.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/charts/__init__.py",
}

# Full V1.5 cluster file set = V1.4 set + the new diagrams package + the new
# proportion_ring chart module.
FILES = [
    "vidlore/ffmpeg_tool.py",
    "vidlore/footage.py",
    "vidlore/pipeline.py",
    "vidlore/assemble.py",
    "vidlore/motion_graphics/__init__.py",
    "vidlore/motion_graphics/look.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/numbers/gold_number_callout.py",
    "vidlore/motion_graphics/portraits/cinematic_portrait_hold.py",
    "vidlore/motion_graphics/documents/headline_document_reveal.py",
    "vidlore/motion_graphics/maps/__init__.py",
    "vidlore/motion_graphics/maps/portrait_name_over_map.py",
    "vidlore/motion_graphics/maps/location_establish_card.py",
    "vidlore/motion_graphics/maps/map_route_spread.py",
    "vidlore/motion_graphics/typography/kinetic_keyword.py",
    "vidlore/motion_graphics/charts/__init__.py",
    "vidlore/motion_graphics/charts/money_flow_empire.py",
    "vidlore/motion_graphics/charts/statistic_bar_reveal.py",
    "vidlore/motion_graphics/charts/growth_curve_chart.py",
    "vidlore/motion_graphics/charts/proportion_ring.py",
    "vidlore/motion_graphics/timelines/__init__.py",
    "vidlore/motion_graphics/timelines/chronology_timeline.py",
    "vidlore/motion_graphics/quotes/__init__.py",
    "vidlore/motion_graphics/quotes/pull_quote_portrait.py",
    "vidlore/motion_graphics/comparison/__init__.py",
    "vidlore/motion_graphics/comparison/comparison_split.py",
    "vidlore/motion_graphics/evidence/__init__.py",
    "vidlore/motion_graphics/evidence/framed_evidence_spotlight.py",
    "vidlore/motion_graphics/annotations/__init__.py",
    "vidlore/motion_graphics/annotations/annotated_detail_callout.py",
    "vidlore/motion_graphics/diagrams/__init__.py",
    "vidlore/motion_graphics/diagrams/process_flow_steps.py",
    "vidlore/motion_graphics/diagrams/org_hierarchy_tree.py",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


parent_manifest = json.loads((PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
parent_hashes = parent_manifest["file_hashes_sha256"]

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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.4)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_4_source": old, "v1_5_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.5 — Proportion / Process / Hierarchy · sha256",
         "# parent: V1.4 Growth/Annotation/Route (left untouched); +3 premium "
         "primitives (18 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.5 — Proportion / Process / Hierarchy",
    "parent": ("Motion Graphics Cluster V1.4 — Growth / Annotation / Route "
               "(snapshots/MG_Cluster_V1.4_GrowthAnnotationRoute, left untouched)"),
    "kind": "feature update — MagnatesMedia Batch 4 (+3 primitives, 18 total)",
    "new_primitives": {
        "proportion_ring": (
            "family=charts. A parts-of-a-whole share: a gold arc sweeps from 12 "
            "o'clock filling to the %, with a centre count-up numeral + a label "
            "naming what the share is OF (e.g. 90%)."),
        "process_flow_steps": (
            "family=diagrams (NEW). 2-5 numbered nodes left->right joined by "
            "arrows that draw in one after another with short labels — an ordered "
            "mechanism / scheme."),
        "org_hierarchy_tree": (
            "family=diagrams. A root over 2-4 child nodes joined by clean elbow "
            "connectors, revealed top-down — a power / corporate structure."),
    },
    "wiring": [
        "registry.py: import + register the 3 (REGISTRY now 18); REQUIRED_INPUTS "
        "{share} / {steps} / {root,children}; 5 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += proportion 5.5, process 6.5, "
        "hierarchy 6.0.",
        "director.py: _SHARE / _PROCESS / _HIERARCHY lexicons + share/process/"
        "hierarchy cues; _GK_AFFINITY + scoring bonuses; share derivation from a "
        "percent; passthrough keys share/steps/root/children/center_sub.",
        "pipeline.py: 3 additive adapter branches mapping graphic_kind "
        "share|process|hierarchy (+ aliases) → inputs, parsing share= / label= / "
        "sub= / steps= / children= / title= hints; _plain_gb hint guard extended.",
        "charts/__init__.py: import proportion_ring; new diagrams/ package.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run: the 3 kinds "
        "select their primitives (share 4.8, process 4.6, hierarchy 4.0); the two "
        "node-diagram primitives are correctly held >2 scenes apart by the "
        "INCOMPATIBLE_ADJACENT rule. Real Standard-Oil render exercised all 3 live "
        "alongside portrait + location; frame-level QA confirmed each premium. "
        "Each primitive self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_4_to_v1_5": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "15 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.4 / V1.3 / V1.2.x / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new primitive modules + diagrams/ package and revert "
        "charts/__init__.py to the V1.4 copy.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.4_GrowthAnnotationRoute/source/ back into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.5 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_4_source'])[:16]} -> {c['v1_5_source'][:16]}")
print("V1.4 parent untouched:", PARENT_SNAP.exists())
