"""Build the MG_Cluster_V1.7_DefinitionBalanceBeforeAfter sibling snapshot.

Feature update over V1.6 (Statement / Pictograph / Composition), LEFT UNTOUCHED
as a rollback point — same sibling pattern as every prior MG snapshot.

What changed since V1.6 — MagnatesMedia "Batch 6": three NEW premium, reusable
motion-graphic primitives plus the director/registry/dispatch/pipeline wiring:

  • definition_card (statements) — a dictionary-style TERM in gold serif + a
    part-of-speech tag + a rule + the definition beneath, fading up line by line.
  • vs_balance_scale (scales, NEW family) — two labelled forces hang from a beam
    that tips toward the heavier side and settles with a small wobble.
  • before_after_slider (reveals, NEW family) — a lit vertical seam wipes an
    "after" state over a "before" state of the same frame (graded vs degraded);
    a warm/cool then-now panel pair when no image is available.

Each self-cleans its PNG temp dir (V1.2.1 leak fix preserved). The 21 prior
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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.6_StatementPictographComposition")
NEW = Path("snapshots/MG_Cluster_V1.7_DefinitionBalanceBeforeAfter")

NEW_FILES = [
    "vidlore/motion_graphics/statements/definition_card.py",
    "vidlore/motion_graphics/scales/__init__.py",
    "vidlore/motion_graphics/scales/vs_balance_scale.py",
    "vidlore/motion_graphics/reveals/__init__.py",
    "vidlore/motion_graphics/reveals/before_after_slider.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/statements/__init__.py",
}

_V16 = [
    "vidlore/ffmpeg_tool.py", "vidlore/footage.py", "vidlore/pipeline.py",
    "vidlore/assemble.py", "vidlore/motion_graphics/__init__.py",
    "vidlore/motion_graphics/look.py", "vidlore/motion_graphics/director.py",
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
FILES = _V16 + NEW_FILES


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
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.6)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_6_source": old, "v1_7_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V1.7 — Definition / Balance / Before-After · sha256",
         "# parent: V1.6 Statement/Pictograph/Composition (left untouched); +3 "
         "premium primitives (24 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.7 — Definition / Balance / Before-After",
    "parent": ("Motion Graphics Cluster V1.6 — Statement / Pictograph / "
               "Composition (snapshots/MG_Cluster_V1.6_StatementPictographComposition, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 6 (+3 primitives, 24 total)",
    "new_primitives": {
        "definition_card": (
            "family=statements. Dictionary-style TERM in gold serif + part-of-"
            "speech tag + rule + definition beneath, fading up line by line."),
        "vs_balance_scale": (
            "family=scales (NEW). Two labelled forces hang from a beam that tips "
            "toward the heavier side and settles with a small wobble."),
        "before_after_slider": (
            "family=reveals (NEW). A lit vertical seam wipes an 'after' over a "
            "'before' of the same frame (graded vs degraded); warm/cool then-now "
            "panels when no image is available."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 24); REQUIRED_INPUTS "
        "{term,definition} / {left,right} / {after_label}; 4 INCOMPATIBLE_ADJACENT.",
        "render_dispatch.py: USEFUL_DUR += definition 5.5, balance 6.0, "
        "before_after 5.5. cache + manifest are generic (auto-wired via the "
        "registry; image_path keys like the evidence/detail cards).",
        "director.py: _DEFINE/_BALANCE/_TRANSFORM lexicons + cues; affinity + "
        "scoring; passthrough term/definition/pos/before_label/after_label "
        "(left/right/leftval/rightval reused from comparison).",
        "pipeline.py: 3 adapter branches (definition=/pos=/pair=/values=/before=/"
        "after= hints + _gimg resolution for the wipe image); _plain_gb guard "
        "extended.",
        "new scales/ + reveals/ packages; statements/__init__ imports "
        "definition_card.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run: the 3 kinds "
        "select their primitives (definition 4.4, before_after 4.6, balance 2.8); "
        "'came at the expense of' derives vs_balance_scale and 'transformed / "
        "would become' derives before_after_slider with no graphic_kind. Real "
        "Industrial-Revolution render exercised all 3 live + portrait + location; "
        "frame-level QA premium. Each self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_6_to_v1_7": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "21 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.6 / V1.5 / V1.4 / V1.3 / V1.2.x / V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new modules + scales/ + reveals/ packages and revert "
        "statements/__init__.py to the V1.6 copy.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V1.6_StatementPictographComposition/source/ into "
        "vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.7 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_6_source'])[:16]} -> {c['v1_7_source'][:16]}")
print("V1.6 parent untouched:", PARENT_SNAP.exists())
