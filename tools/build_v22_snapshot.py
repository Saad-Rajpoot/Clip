"""Build the MG_Cluster_V2.2_SpotlightDecisionArc sibling snapshot.

Feature update over V2.1 (Region-Highlight / Cause-Effect / Spectrum-Meter), LEFT
UNTOUCHED as a rollback point — same sibling pattern as every prior MG snapshot.

What changed since V2.1 — MagnatesMedia "Batch 11": three NEW premium, reusable
motion-graphic primitives plus director/registry/dispatch/pipeline wiring:

  • spotlight_object_hold (reveals) — a text-led "behold" reveal: a near-black stage,
    a warm-gold radial spotlight pool sweeps in (easeOutExpo) and settles centre,
    lifting a bold serif SUBJECT out of darkness; KICKER above, sub below, title on
    a hairline. NO frame, NO photo — the signature is the MOVING light.
  • flowchart_decision (diagrams) — a single yes/no DECISION FORK: a gold-outlined
    diamond question node at top, two diverging connectors to outcome cards with
    YES/NO chips; `chosen` ignites the taken path to gold and dims the other.
  • world_map_arc (maps) — ONE great-circle ARC bows from an origin city to a distant
    destination over a purpose-built antique world chart (clean graticule, no blobs);
    a comet head traces it in, pulsing pins + gold city labels at both ends.

No stable renderer files were modified (footage.py / assemble.py / look.py
untouched this batch). world_map_arc carries its OWN _world_bed (decoupled from the
stable location_establish_card, which is byte-identical). Each self-cleans its PNG
temp dir (V1.2.1 leak fix preserved). The 36 prior primitives carry forward verbatim.

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V2.1_RegionCauseGauge")
NEW = Path("snapshots/MG_Cluster_V2.2_SpotlightDecisionArc")

NEW_FILES = [
    "vidlore/motion_graphics/reveals/spotlight_object_hold.py",
    "vidlore/motion_graphics/diagrams/flowchart_decision.py",
    "vidlore/motion_graphics/maps/world_map_arc.py",
]
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/reveals/__init__.py",
    "vidlore/motion_graphics/diagrams/__init__.py",
    "vidlore/motion_graphics/maps/__init__.py",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


parent_manifest = json.loads((PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
parent_hashes = parent_manifest["file_hashes_sha256"]
# carry forward EXACTLY the 60 files the V2.1 snapshot tracked
_V21 = list(parent_manifest["files"])
FILES = _V21 + NEW_FILES

# sanity: every CHANGED file must already be tracked by the parent
for f in CHANGED:
    assert f in _V21, f"CHANGED file not in parent manifest: {f}"
for f in NEW_FILES:
    assert f not in _V21, f"NEW file already in parent manifest: {f}"

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
    old = parent_hashes.get(f, {}).get("source", "(absent in V2.1)")
    new = hashes[f]["source"]
    hash_change[f] = {"v2_1_source": old, "v2_2_source": new, "changed": old != new}

lines = ["# Motion Graphics Cluster V2.2 — Spotlight-Reveal / Decision-Fork / World-Arc · sha256",
         "# parent: V2.1 Region/Cause/Gauge (left untouched); +3 premium primitives (39 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V2.2 — Spotlight-Reveal / Decision-Fork / World-Arc",
    "parent": ("Motion Graphics Cluster V2.1 — Region-Highlight / Cause-Effect / "
               "Spectrum-Meter (snapshots/MG_Cluster_V2.1_RegionCauseGauge, untouched)"),
    "kind": "feature update — MagnatesMedia Batch 11 (+3 primitives, 39 total)",
    "new_primitives": {
        "spotlight_object_hold": (
            "family=reveals. A text-led 'behold' reveal: a near-black stage; a warm-"
            "gold radial spotlight pool sweeps in from an edge (easeOutExpo) and "
            "settles centre, lifting a bold serif SUBJECT (gold_fill) out of the dark "
            "as the beam reaches it; condensed KICKER above, sub below, title on a "
            "hairline. NO frame / NO photo — distinct from framed_evidence_spotlight."),
        "flowchart_decision": (
            "family=diagrams. A single yes/no DECISION FORK: a gold-outlined diamond "
            "question node at top-centre, two diverging connectors down to outcome "
            "cards with YES/NO chips; if `chosen`, that path ignites to accent_hi gold "
            "and the other dims — the verdict. A Y-fork, not a row (vs cause_effect)."),
        "world_map_arc": (
            "family=maps. ONE great-circle ARC bows up from an origin city to a "
            "distant destination over a purpose-built antique world chart (its own "
            "_world_bed: clean lat/long graticule + even aged-paper mottling, no blobs, "
            "faint equator/meridian). A comet head traces it in (easeInOutCubic), "
            "pulsing pins + gold serif city labels at both ends. Two points, one link "
            "— distinct from map_route_spread (3-5 stop polyline)."),
    },
    "wiring": [
        "registry.py: register 3 (REGISTRY now 39); REQUIRED_INPUTS {subject} / "
        "{question} / {from_place,to_place}; 13 INCOMPATIBLE_ADJACENT pairs.",
        "render_dispatch.py: USEFUL_DUR += spotlight 5.5, flowchart 6.5, world_arc "
        "6.5. cache + manifest generic (auto-wired via the registry).",
        "director.py: _REVEAL/_DECISION/_ARC lexicons + cues; affinity (spotlight/"
        "reveal/behold, decision/flowchart/branch/fork, world_arc/arc/transatlantic) + "
        "scoring; passthrough subject/kicker/question/yes/no/yes_label/no_label/"
        "chosen/from_place/to_place/from_pos/to_pos. flowchart_decision + world_map_arc "
        "intensity_range [2,4]; spotlight [3,5]. The legacy footage.py 'spotlight' card "
        "builder only builds an image (never mutates the kind) so it stays out of the "
        "way; validation uses collision-free kinds reveal/decision/world_arc.",
        "pipeline.py: 3 adapter branches (subject via _gt + kicker=/sub=/title=; "
        "question via _gt + yes=/no=/chosen=/yes_label=/no_label=/title=; from_place "
        "via _gt + to=/from_pos=/to_pos=/title= + _gimg→map_image); _plain_gb guard "
        "extended with kicker=/yes=/no=/chosen=/yes_label=/no_label=/from_pos=/to_pos=.",
        "reveals/__init__ imports spotlight_object_hold; diagrams/__init__ imports "
        "flowchart_decision; maps/__init__ imports world_map_arc. footage.py / "
        "assemble.py / look.py UNCHANGED; location_establish_card byte-identical "
        "(world_map_arc carries its own _world_bed).",
    ],
    "validated": (
        "All changed/new files py_compile OK. Built + micro-rendered via a 3-agent "
        "parallel workflow, then each frame human-QA'd premium (world_map_arc polished "
        "after QA: own _world_bed + frame-spanning composition). Director dry-run on a "
        "Petrov-1983 (nuclear false-alarm) script: reveal→spotlight_object_hold, "
        "world_arc→world_map_arc, decision→flowchart_decision each fired with correct "
        "inputs at scenes 1/3/5; stable chronology_timeline (7) + spectrum_meter (9) "
        "still fire (no regression). All five graphic scenes are different families so "
        "the same-family guard never engages, spaced 2 apart. Real render exercised all "
        "3 live; frame-level QA premium. Each self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.01-0.02 for the validation render); $0 code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v2_1_to_v2_2": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "36 prior primitives carried forward verbatim",
        "footage.py / assemble.py / look.py renderers untouched this batch",
        "location_establish_card byte-identical (world_map_arc decoupled via _world_bed)",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V2.1 … V1.1.x snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new modules; revert reveals/__init__.py, diagrams/__init__.py "
        "and maps/__init__.py to drop the new imports.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py from "
        "snapshots/MG_Cluster_V2.1_RegionCauseGauge/source/ into vidlore/.",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V2.2 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES), "· changed:", len(CHANGED),
      "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v2_1_source'])[:16]} -> {c['v2_2_source'][:16]}")
print("V2.1 parent untouched:", PARENT_SNAP.exists())
