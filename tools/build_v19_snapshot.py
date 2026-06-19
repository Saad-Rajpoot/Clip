"""Build the MG_Cluster_V1.9_RankedSankeyEra sibling snapshot.

Feature update over V1.8 (Headlines / Heat-Spread / Redacted), LEFT UNTOUCHED as a
rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.8 — MagnatesMedia "Batch 8": three NEW premium, reusable
motion-graphic primitives plus director/registry/dispatch/pipeline wiring:

  • ranked_list_countdown (charts) — a Top-N leaderboard; rows drop in bottom-rank
    UP to #1, each with a rank numeral, label, proportional bar and value; #1 is
    crowned in gold.
  • sankey_flow (charts) — a source column splits into proportional-width gold
    bezier ribbons flowing to labelled branches; thin dark outlines separate them.
  • era_band_timeline (timelines) — a horizontal time axis divided into labelled
    era bands whose WIDTH is their span, wiping in left→right with a tonal ramp.

No stable renderer files were modified (footage.py untouched this batch). Each
self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 27 prior primitives
are carried forward verbatim.

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.8_HeadlinesHeatRedacted")
NEW = Path("snapshots/MG_Cluster_V1.9_RankedSankeyEra")

NEW_FILES = [
    "vidlore/motion_graphics/charts/ranked_list_countdown.py",
    "vidlore/motion_graphics/charts/sankey_flow.py",
    "vidlore/motion_graphics/timelines/era_band_timeline.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/charts/__init__.py",
    "vidlore/motion_graphics/timelines/__init__.py",
}

# every file carried forward as of V1.8 (49 files).
_V18 = [
    "vidlore/ffmpeg_tool.py", "vidlore/footage.py", "vidlore/pipeline.py",
    "vidlore/assemble.py", "vidlore/motion_graphics/__init__.py",
    "vidlore/motion_graphics/look.py", "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/numbers/gold_number_callout.py",
    "vidlore/motion_graphics/portraits/cinematic_portrait_hold.py",
    "vidlore/motion_graphics/documents/__init__.py",
    "vidlore/motion_graphics/documents/headline_document_reveal.py",
    "vidlore/motion_graphics/documents/redacted_document.py",
    "vidlore/motion_graphics/maps/__init__.py",
    "vidlore/motion_graphics/maps/portrait_name_over_map.py",
    "vidlore/motion_graphics/maps/location_establish_card.py",
    "vidlore/motion_graphics/maps/map_route_spread.py",
    "vidlore/motion_graphics/maps/map_heat_spread.py",
    "vidlore/motion_graphics/media/__init__.py",
    "vidlore/motion_graphics/media/headline_montage.py",
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
    "vidlore/motion_graphics/statements/definition_card.py",
    "vidlore/motion_graphics/scales/__init__.py",
    "vidlore/motion_graphics/scales/vs_balance_scale.py",
    "vidlore/motion_graphics/reveals/__init__.py",
    "vidlore/motion_graphics/reveals/before_after_slider.py",
]
FILES = _V18 + NEW_FILES


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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.8)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_8_source": old, "v1_9_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.9 — Ranked-List / Sankey-Flow / Era-Band · sha256",
         "# parent: V1.8 Headlines/Heat/Redacted (left untouched); +3 premium "
         "primitives (30 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.9 — Ranked-List / Sankey-Flow / Era-Band",
    "parent": ("Motion Graphics Cluster V1.8 — Headlines / Heat-Spread / Redacted "
               "(snapshots/MG_Cluster_V1.8_HeadlinesHeatRedacted, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 8 (+3 primitives, 30 total)",
    "new_primitives": {
        "ranked_list_countdown": (
            "family=charts. A Top-N leaderboard; rows drop in bottom-rank UP to #1, "
            "each with a rank numeral, label, proportional bar and value; #1 crowned "
            "in gold."),
        "sankey_flow": (
            "family=charts. A source column splits into proportional-width gold "
            "bezier ribbons flowing to labelled branches; thin dark outlines keep "
            "adjacent ribbons distinct."),
        "era_band_timeline": (
            "family=timelines. A horizontal time axis divided into labelled era "
            "bands whose WIDTH is their span, wiping in left→right with a dim→bright "
            "tonal ramp and year markers."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 30); REQUIRED_INPUTS {items} / "
        "{branches} / {eras}; 7 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += ranked 6.5, sankey 6.0, era 6.5. cache "
        "+ manifest generic (auto-wired via the registry).",
        "director.py: _RANK/_FLOW/_ERA lexicons + cues; affinity (ranking/"
        "leaderboard, sankey/money_split/allocation, eras/ages/periods) + scoring. "
        "ranked_list_countdown requires {items} so the legacy footage.py 'ranking' "
        "carousel (per-scene, footage-backed) is untouched when no item list is "
        "supplied.",
        "pipeline.py: 3 adapter branches (items=/branches=/eras= + source/prefix/"
        "suffix/title hints); _plain_gb guard extended with items=/eras=.",
        "charts/__init__ imports ranked_list_countdown + sankey_flow; "
        "timelines/__init__ imports era_band_timeline. footage.py UNCHANGED.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run on a Robber-Barons "
        "script: leaderboard→ranked_list_countdown (4.60), money_split→sankey_flow "
        "(4.60), eras→era_band_timeline (2.80), each with correct inputs; charts "
        "pair (ranked@3, sankey@7) sit 4 apart so the same-family guard is "
        "satisfied; stable portrait + location still fire (no regression). Real "
        "render exercised all 3 live; frame-level QA premium. Each self-cleans its "
        "PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_8_to_v1_9": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "27 prior primitives carried forward verbatim",
        "footage.py / assemble.py / look.py renderers untouched this batch",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.8 / V1.7 / V1.6 / V1.5 / V1.4 / V1.3 / V1.2.x / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new modules; revert charts/__init__.py and "
        "timelines/__init__.py to drop the new imports.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.8_HeadlinesHeatRedacted/source/ into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.9 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_8_source'])[:16]} -> {c['v1_9_source'][:16]}")
print("V1.8 parent untouched:", PARENT_SNAP.exists())
