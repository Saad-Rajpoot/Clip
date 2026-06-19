"""Build the MG_Cluster_V2.1_RegionCauseGauge sibling snapshot.

Feature update over V2.0 (Countdown / Connection-Web / Quote-Stream), LEFT
UNTOUCHED as a rollback point — same sibling pattern as every prior MG snapshot.

What changed since V2.0 — MagnatesMedia "Batch 10": three NEW premium, reusable
motion-graphic primitives plus director/registry/dispatch/pipeline wiring:

  • map_region_highlight (maps) — a graded period map with ONE region singled out
    by a soft gold glow + an irregular hand-drawn boundary + a pin + name/sub, the
    rest of the map dimmed.
  • cause_effect_chain (diagrams) — 2-4 cause cards in a domino row linked by bold
    gold chevrons, revealed left→right, the final OUTCOME card crowned in gold.
  • spectrum_meter (meters, NEW family) — a qualitative gauge: a cool→warm gradient
    band track with labelled bands and a gold needle that sweeps to the value and
    settles, the landed band brightening; a big serif readout above.

No stable renderer files were modified (footage.py / assemble.py / look.py
untouched this batch). Each self-cleans its PNG temp dir (V1.2.1 leak fix
preserved). The 33 prior primitives are carried forward verbatim.

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V2.0_CountdownWebQuotes")
NEW = Path("snapshots/MG_Cluster_V2.1_RegionCauseGauge")

NEW_FILES = [
    "vidlore/motion_graphics/maps/map_region_highlight.py",
    "vidlore/motion_graphics/diagrams/cause_effect_chain.py",
    "vidlore/motion_graphics/meters/__init__.py",
    "vidlore/motion_graphics/meters/spectrum_meter.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/maps/__init__.py",
    "vidlore/motion_graphics/diagrams/__init__.py",
}

# every file carried forward as of V2.0 (56 files).
_V20 = [
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
    "vidlore/motion_graphics/charts/ranked_list_countdown.py",
    "vidlore/motion_graphics/charts/sankey_flow.py",
    "vidlore/motion_graphics/timelines/__init__.py",
    "vidlore/motion_graphics/timelines/chronology_timeline.py",
    "vidlore/motion_graphics/timelines/era_band_timeline.py",
    "vidlore/motion_graphics/quotes/__init__.py",
    "vidlore/motion_graphics/quotes/pull_quote_portrait.py",
    "vidlore/motion_graphics/quotes/quote_stream.py",
    "vidlore/motion_graphics/comparison/__init__.py",
    "vidlore/motion_graphics/comparison/comparison_split.py",
    "vidlore/motion_graphics/evidence/__init__.py",
    "vidlore/motion_graphics/evidence/framed_evidence_spotlight.py",
    "vidlore/motion_graphics/annotations/__init__.py",
    "vidlore/motion_graphics/annotations/annotated_detail_callout.py",
    "vidlore/motion_graphics/diagrams/__init__.py",
    "vidlore/motion_graphics/diagrams/process_flow_steps.py",
    "vidlore/motion_graphics/diagrams/org_hierarchy_tree.py",
    "vidlore/motion_graphics/diagrams/connection_web.py",
    "vidlore/motion_graphics/statements/__init__.py",
    "vidlore/motion_graphics/statements/statement_card.py",
    "vidlore/motion_graphics/statements/definition_card.py",
    "vidlore/motion_graphics/scales/__init__.py",
    "vidlore/motion_graphics/scales/vs_balance_scale.py",
    "vidlore/motion_graphics/reveals/__init__.py",
    "vidlore/motion_graphics/reveals/before_after_slider.py",
    "vidlore/motion_graphics/clocks/__init__.py",
    "vidlore/motion_graphics/clocks/countdown_clock.py",
]
FILES = _V20 + NEW_FILES


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
    old = parent_hashes.get(f, {}).get("source", "(absent in V2.0)")
    new = hashes[f]["source"]
    hash_change[f] = {"v2_0_source": old, "v2_1_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V2.1 — Region-Highlight / Cause-Effect / Spectrum-Meter · sha256",
         "# parent: V2.0 Countdown/Web/Quotes (left untouched); +3 premium primitives (36 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V2.1 — Region-Highlight / Cause-Effect / Spectrum-Meter",
    "parent": ("Motion Graphics Cluster V2.0 — Countdown / Connection-Web / "
               "Quote-Stream (snapshots/MG_Cluster_V2.0_CountdownWebQuotes, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 10 (+3 primitives, 36 total)",
    "new_primitives": {
        "map_region_highlight": (
            "family=maps. A graded period map with ONE region singled out by a soft "
            "gold glow + an irregular hand-drawn boundary + a pin + name/sub; the "
            "rest of the map is dimmed so the region is the focus."),
        "cause_effect_chain": (
            "family=diagrams. 2-4 cause cards in a domino row linked by bold gold "
            "chevrons, revealed left→right, the final OUTCOME card crowned in gold "
            "(brighter border, gold rule + diamonds, ~1.12x)."),
        "spectrum_meter": (
            "family=meters (NEW). A qualitative gauge: a cool→warm gradient band "
            "track with labelled bands + a gold needle that sweeps to the value and "
            "settles, the landed band brightening; a big serif readout above."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 36); REQUIRED_INPUTS {region} / "
        "{steps} / {value}; 9 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += region 6.0, cause_effect 6.5, meter 5.5. "
        "cache + manifest generic (auto-wired via the registry).",
        "director.py: _REGION/_CAUSAL/_GAUGE lexicons + cues; affinity (region/"
        "territory, cause_effect/causation/domino, gauge/meter/threat_level) + "
        "scoring; passthrough bands/readout (pos/sub/value/label reused). "
        "spectrum_meter requires {value} (low score without a gauge kind/cue, so it "
        "never steals plain-number scenes from gold_number_callout); the legacy "
        "footage.py map_region/cause_effect card builders only build images (never "
        "mutate the kind) so they stay out of the way.",
        "pipeline.py: 3 adapter branches (region via _gt + pos=/sub=/title= + _gimg→"
        "map_image; steps=; value=/bands=/readout= + label via _gt); _plain_gb guard "
        "extended with bands=/readout=.",
        "new meters/ package; maps/__init__ imports map_region_highlight; "
        "diagrams/__init__ imports cause_effect_chain. footage.py UNCHANGED.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Built + micro-rendered via a 3-agent "
        "parallel workflow, then each frame human-QA'd premium. Director dry-run on "
        "a Road-to-War (appeasement) script: territory→map_region_highlight (2.80), "
        "causation→cause_effect_chain (4.60), threat_level→spectrum_meter (5.00), "
        "each with correct inputs; all five graphic scenes are different families so "
        "the same-family guard never engages, spaced 2 apart; stable portrait + "
        "chronology still fire (no regression). Real render exercised all 3 live; "
        "frame-level QA premium. Each self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v2_0_to_v2_1": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "33 prior primitives carried forward verbatim",
        "footage.py / assemble.py / look.py renderers untouched this batch",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V2.0 … V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new modules + meters/ package; revert maps/__init__.py and "
        "diagrams/__init__.py to drop the new imports.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V2.0_CountdownWebQuotes/source/ into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V2.1 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v2_0_source'])[:16]} -> {c['v2_1_source'][:16]}")
print("V2.0 parent untouched:", PARENT_SNAP.exists())
