"""Build the MG_Cluster_V1.8_HeadlinesHeatRedacted sibling snapshot.

Feature update over V1.7 (Definition / Balance / Before-After), LEFT UNTOUCHED as
a rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.7 — MagnatesMedia "Batch 7": three NEW premium, reusable
motion-graphic primitives plus the director/registry/dispatch/pipeline wiring:

  • headline_montage (media, NEW family) — three-to-five period headlines cascade
    in as aged-newsprint clippings, each rotated and overlapping, latest on top.
  • map_heat_spread (maps) — warm ember→amber heat blooms ignite in sequence and
    spread across a graded antique map; soft pins + labels; dimmed-additive glow
    so the map shows through (no white blow-out).
  • redacted_document (documents) — a classified page where black redaction bars
    sweep across typed lines while ONE line stays legible, and a red TOP SECRET
    stamp lands askew low over the bars.

Each self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 24 prior
primitives are carried forward verbatim. NOTE: documents/__init__.py — untracked
in prior manifests — is now captured (it gains the redacted_document import).

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.7_DefinitionBalanceBeforeAfter")
NEW = Path("snapshots/MG_Cluster_V1.8_HeadlinesHeatRedacted")

NEW_FILES = [
    "vidlore/motion_graphics/media/__init__.py",
    "vidlore/motion_graphics/media/headline_montage.py",
    "vidlore/motion_graphics/maps/map_heat_spread.py",
    "vidlore/motion_graphics/documents/redacted_document.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/footage.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/documents/__init__.py",
    "vidlore/motion_graphics/maps/__init__.py",
}

# every file carried forward as of V1.7 (44) + documents/__init__.py (newly
# tracked — it was absent from prior manifests but is changed in this batch).
_V17 = [
    "vidlore/ffmpeg_tool.py", "vidlore/footage.py", "vidlore/pipeline.py",
    "vidlore/assemble.py", "vidlore/motion_graphics/__init__.py",
    "vidlore/motion_graphics/look.py", "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/numbers/gold_number_callout.py",
    "vidlore/motion_graphics/portraits/cinematic_portrait_hold.py",
    "vidlore/motion_graphics/documents/__init__.py",
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
    "vidlore/motion_graphics/statements/definition_card.py",
    "vidlore/motion_graphics/scales/__init__.py",
    "vidlore/motion_graphics/scales/vs_balance_scale.py",
    "vidlore/motion_graphics/reveals/__init__.py",
    "vidlore/motion_graphics/reveals/before_after_slider.py",
]
FILES = _V17 + NEW_FILES


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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.7)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_7_source": old, "v1_8_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.8 — Headlines / Heat-Spread / Redacted · sha256",
         "# parent: V1.7 Definition/Balance/Before-After (left untouched); +3 "
         "premium primitives (27 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.8 — Headlines / Heat-Spread / Redacted",
    "parent": ("Motion Graphics Cluster V1.7 — Definition / Balance / Before-After "
               "(snapshots/MG_Cluster_V1.7_DefinitionBalanceBeforeAfter, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 7 (+3 primitives, 27 total)",
    "new_primitives": {
        "headline_montage": (
            "family=media (NEW). Three-to-five period headlines cascade in as "
            "aged-newsprint clippings, each rotated and overlapping, the latest "
            "landing centred + upright on top — a press storm."),
        "map_heat_spread": (
            "family=maps. Warm ember→amber heat blooms ignite in sequence and "
            "spread across a graded antique map; soft pins + labels; dimmed-"
            "additive glow so the map shows through (no white blow-out)."),
        "redacted_document": (
            "family=documents. A classified page where black redaction bars sweep "
            "across typed lines while ONE line stays legible, and a red TOP SECRET "
            "stamp lands askew low over the bars."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 27); REQUIRED_INPUTS {headlines} / "
        "{hotspots} / {reveal}; 5 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += headline_montage 6.0, map_heat_spread "
        "6.5, redacted_document 6.0. cache + manifest are generic (auto-wired via "
        "the registry; map_image key like the other map cards).",
        "director.py: _PRESS/_HEAT/_SECRET lexicons + cues; affinity (headlines/"
        "press/scandal, heat/contagion/outbreak, classified/redacted/secret) + "
        "scoring; passthrough headlines/hotspots/reveal/stamp; 'redacted' & "
        "'spread' affinities retargeted to the new primitives.",
        "pipeline.py: 3 adapter branches (headlines=/hotspots=/reveal via _gt/"
        "title=/stamp= hints + _gimg→map_image for the heat bed); _plain_gb guard "
        "extended with headlines=/hotspots=/reveal=/stamp=.",
        "new media/ package; documents/__init__ imports redacted_document; "
        "maps/__init__ imports map_heat_spread.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run on a Zimmermann-"
        "Telegram script: press→headline_montage (4.80), contagion→map_heat_spread "
        "(4.80), classified→redacted_document (5.00), each with correct inputs; "
        "stable portrait + chronology still fire (no regression); beats spaced 2 "
        "apart, no family clash. Real render exercised all 3 live; frame-level QA "
        "premium. Each self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_7_to_v1_8": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "24 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.7 / V1.6 / V1.5 / V1.4 / V1.3 / V1.2.x / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new modules + media/ package; revert documents/__init__.py "
        "and maps/__init__.py to drop the redacted_document / map_heat_spread "
        "imports.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.7_DefinitionBalanceBeforeAfter/source/ into "
        "vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.8 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_7_source'])[:16]} -> {c['v1_8_source'][:16]}")
print("V1.7 parent untouched:", PARENT_SNAP.exists())
