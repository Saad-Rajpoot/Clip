"""Build the MG_Cluster_V1.4_GrowthAnnotationRoute sibling snapshot.

Feature update over V1.3 (Evidence / Data / Location), LEFT UNTOUCHED as a
rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.3 — MagnatesMedia "Batch 3": three NEW premium, reusable
motion-graphic primitives plus the director/registry/dispatch/pipeline wiring:

  • growth_curve_chart (charts) — a smooth Catmull-Rom time-series that draws in
    left→right with a glowing plot-head, a serif count-up of the value under the
    head, faint y-gridlines and x-axis labels that light as the head passes
    (e.g. 4 → 55 → 310). A continuous TREND, distinct from discrete bars.
  • annotated_detail_callout (annotations, NEW family) — points INTO a photo:
    a local spotlight crushes everything but a soft disc around a focus point, a
    bright ring + cardinal ticks frame the detail, and a leader line tethers a
    label chip ("look HERE"). Premium aged archival plate when no photo resolves.
  • map_route_spread (maps) — a route draws across a graded antique map: a faint
    ghost path, then a bright traced line with a comet-head, origin/destination
    pins and waypoint names that light as the head passes (an expansion/journey).

Each self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 12 prior
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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.3_EvidenceDataLocation")
NEW = Path("snapshots/MG_Cluster_V1.4_GrowthAnnotationRoute")

NEW_FILES = [
    "vidlore/motion_graphics/annotations/__init__.py",
    "vidlore/motion_graphics/annotations/annotated_detail_callout.py",
    "vidlore/motion_graphics/charts/growth_curve_chart.py",
    "vidlore/motion_graphics/maps/map_route_spread.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/charts/__init__.py",
    "vidlore/motion_graphics/maps/__init__.py",
}

# Full V1.4 cluster file set = V1.3 set + the new annotations package + the
# three new Batch-3 primitive modules.
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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.3)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_3_source": old, "v1_4_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.4 — Growth / Annotation / Route · sha256",
         "# parent: V1.3 Evidence/Data/Location (left untouched); +3 premium "
         "primitives (15 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.4 — Growth / Annotation / Route",
    "parent": ("Motion Graphics Cluster V1.3 — Evidence / Data / Location "
               "(snapshots/MG_Cluster_V1.3_EvidenceDataLocation, left untouched)"),
    "kind": "feature update — MagnatesMedia Batch 3 (+3 primitives, 15 total)",
    "new_primitives": {
        "growth_curve_chart": (
            "family=charts. Smooth Catmull-Rom time-series that draws in with a "
            "glowing plot-head + serif count-up + faint gridlines + x-labels that "
            "light as the head passes. A continuous trend (4->55->310)."),
        "annotated_detail_callout": (
            "family=annotations (NEW). Local spotlight + bright ring + cardinal "
            "ticks + leader line to a label chip — points INTO a photo. Premium "
            "aged archival plate when no photo resolves."),
        "map_route_spread": (
            "family=maps. A route draws across a graded antique map: ghost path "
            "then bright traced line + comet-head + origin/destination pins + "
            "waypoint names lighting as the head passes (expansion / journey)."),
    },
    "wiring": [
        "registry.py: import + register the 3 (REGISTRY now 15); REQUIRED_INPUTS "
        "{points} / {label} / {stops}; 6 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += growth 6.5, annotate 6.0, route 6.5.",
        "director.py: _GROWTH / _ROUTE / _DETAIL lexicons + growth/route/detail "
        "cues; _GK_AFFINITY + scoring bonuses; passthrough keys points/stops/"
        "focus/place/map_image.",
        "pipeline.py: 3 additive adapter branches mapping graphic_kind "
        "growth|detail|route (+ aliases) → inputs, parsing points= / focus= / "
        "tag= / stops= / suffix= / prefix= hints.",
        "charts/__init__.py + maps/__init__.py: import the new modules; new "
        "annotations/ package with __init__.py.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run: the 3 kinds "
        "select their primitives (growth/route/detail all 4.60) and narration "
        "cues derive growth_curve_chart / map_route_spread with no graphic_kind. "
        "Real Transcontinental-Railroad render exercised all 3 live + "
        "location_establish_card; frame-level QA confirmed each premium. Each "
        "primitive self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.02 for the validation render); $0 for code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_3_to_v1_4": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "12 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.3 / V1.2.1 / V1.2 / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new primitive modules + annotations/ package and revert "
        "charts/__init__.py + maps/__init__.py to the V1.3 copies.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.3_EvidenceDataLocation/source/ back into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.4 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_3_source'])[:16]} -> {c['v1_4_source'][:16]}")
print("V1.3 parent untouched:", PARENT_SNAP.exists())
