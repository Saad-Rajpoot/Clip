"""Build the MG_Cluster_V1.6_StatementPictographComposition sibling snapshot.

Feature update over V1.5 (Proportion / Process / Hierarchy), LEFT UNTOUCHED as a
rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.5 — MagnatesMedia "Batch 5": three NEW premium, reusable
motion-graphic primitives plus the director/registry/dispatch/pipeline wiring:

  • statement_card (statements, NEW family) — one bold editorial CLAIM held
    full-screen in large serif, lines rising in, a key phrase gold-underscored,
    an optional source tag. The narrator's thesis, typeset to land.
  • pictograph_scale (charts) — a grid of figure icons where the first N are lit
    in gold and the rest stay muted, making a ratio countable ("3 IN 10").
  • composition_stack (charts) — ONE 100 % horizontal bar split into labelled
    segments on a gold→muted ramp, wiping in left→right ("where every dollar went").

Each self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 18 prior
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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.5_ProportionProcessHierarchy")
NEW = Path("snapshots/MG_Cluster_V1.6_StatementPictographComposition")

NEW_FILES = [
    "vidlore/motion_graphics/statements/__init__.py",
    "vidlore/motion_graphics/statements/statement_card.py",
    "vidlore/motion_graphics/charts/pictograph_scale.py",
    "vidlore/motion_graphics/charts/composition_stack.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/charts/__init__.py",
}

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
    "vidlore/motion_graphics/charts/pictograph_scale.py",
    "vidlore/motion_graphics/charts/composition_stack.py",
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
    "vidlore/motion_graphics/statements/__init__.py",
    "vidlore/motion_graphics/statements/statement_card.py",
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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.5)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_5_source": old, "v1_6_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.6 — Statement / Pictograph / Composition · sha256",
         "# parent: V1.5 Proportion/Process/Hierarchy (left untouched); +3 premium "
         "primitives (21 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.6 — Statement / Pictograph / Composition",
    "parent": ("Motion Graphics Cluster V1.5 — Proportion / Process / Hierarchy "
               "(snapshots/MG_Cluster_V1.5_ProportionProcessHierarchy, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 5 (+3 primitives, 21 total)",
    "new_primitives": {
        "statement_card": (
            "family=statements (NEW). One bold editorial claim in large serif, "
            "lines rising in, a key phrase gold-underscored, optional source tag."),
        "pictograph_scale": (
            "family=charts. A grid of figure icons, first N lit gold + rest muted, "
            "making a ratio countable ('3 IN 10') with a count-up line."),
        "composition_stack": (
            "family=charts. One 100% horizontal bar split into labelled segments "
            "on a gold->muted ramp, wiping in left->right with % + names."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 21); REQUIRED_INPUTS {text} / "
        "{count} / {segments}; 5 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += statement 5.0, pictograph 5.5, "
        "composition 5.5.",
        "director.py: _CLAIM / _PICTO / _COMPOSITION lexicons + _RATIO_RE; "
        "claim/picto/composition cues; affinity + scoring; ratio derivation "
        "(N in M -> count/total, total folded into ins); passthrough text/"
        "emphasis/count/total/segments.",
        "pipeline.py: 3 adapter branches (emphasis=/source=/count=/total=/"
        "segments=/suffix= hints); _plain_gb hint guard extended.",
        "charts/__init__.py: import pictograph_scale + composition_stack; new "
        "statements/ package.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run: the 3 kinds "
        "select their primitives (statement 4.6, composition 4.6, pictograph "
        "2.8); '3 in 10' narration derives pictograph_scale with count/total; the "
        "two chart primitives (pictograph / composition) are held >2 scenes apart "
        "by the same-family guard. Real Startup-Failure render exercised all 3 "
        "(statement x2, pictograph, composition); frame-level QA premium. Each "
        "self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_5_to_v1_6": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "18 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.5 / V1.4 / V1.3 / V1.2.x / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new primitive modules + statements/ package and revert "
        "charts/__init__.py to the V1.5 copy.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.5_ProportionProcessHierarchy/source/ into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.6 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_5_source'])[:16]} -> {c['v1_6_source'][:16]}")
print("V1.5 parent untouched:", PARENT_SNAP.exists())
