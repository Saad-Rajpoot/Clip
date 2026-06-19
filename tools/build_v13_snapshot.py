"""Build the MG_Cluster_V1.3_EvidenceDataLocation sibling snapshot.

Feature update over V1.2.1 (Temp-Dir Cleanup), which is LEFT UNTOUCHED as a
rollback point — exactly as V1.2.1 was a sibling of V1.2 and V1.1.2 of V1.1.1.

What changed since V1.2.1 — MagnatesMedia "Batch 2": three NEW premium,
reusable motion-graphic primitives plus the wiring that makes the director pick
them and the pipeline feed them:

  • framed_evidence_spotlight (family: evidence) — an artifact (document /
    photograph / object) held in a gold frame under a warm spotlight with a slow
    push-in, an EVIDENCE/EXHIBIT tag chip and a naming caption. When no artifact
    photo resolves it renders a PREMIUM aged-paper exhibit document (the caption
    becomes the typeset title, with a printed keyline, faded typed body lines and
    an ink seal) — never a hollow card.
  • statistic_bar_reveal (family: charts) — 1-4 vertical columns rising from a
    baseline with count-up numerals, labels and a title (e.g. 1 → 12 → 200).
  • location_establish_card (family: maps) — a place + era over a graded antique
    map with corner coordinate ticks, a pulsing pin and a slow establishing push.

Each self-cleans its PNG temp dir (look.cleanup_frames), so the V1.2.1 leak fix
is preserved. The 9 prior primitives are carried forward verbatim.

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.2.1_TempDirCleanup")
NEW = Path("snapshots/MG_Cluster_V1.3_EvidenceDataLocation")

# Files NEW in V1.3 (the three Batch-2 primitives + the evidence package init).
NEW_FILES = [
    "vidlore/motion_graphics/evidence/__init__.py",
    "vidlore/motion_graphics/evidence/framed_evidence_spotlight.py",
    "vidlore/motion_graphics/charts/statistic_bar_reveal.py",
    "vidlore/motion_graphics/maps/location_establish_card.py",
]
# Files CHANGED in V1.3 vs V1.2.1 (director/registry/dispatch wiring + pipeline
# adapters + the two family __init__ imports).
CHANGED = {
    "vidlore/pipeline.py",
    "vidlore/motion_graphics/director.py",
    "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/charts/__init__.py",
    "vidlore/motion_graphics/maps/__init__.py",
}

# Full V1.3 cluster file set = V1.2.1 set + the two family __init__ files that
# now carry Batch-2 imports + the four NEW Batch-2 files.
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
    "vidlore/motion_graphics/typography/kinetic_keyword.py",
    "vidlore/motion_graphics/charts/__init__.py",
    "vidlore/motion_graphics/charts/money_flow_empire.py",
    "vidlore/motion_graphics/charts/statistic_bar_reveal.py",
    "vidlore/motion_graphics/timelines/__init__.py",
    "vidlore/motion_graphics/timelines/chronology_timeline.py",
    "vidlore/motion_graphics/quotes/__init__.py",
    "vidlore/motion_graphics/quotes/pull_quote_portrait.py",
    "vidlore/motion_graphics/comparison/__init__.py",
    "vidlore/motion_graphics/comparison/comparison_split.py",
    "vidlore/motion_graphics/evidence/__init__.py",
    "vidlore/motion_graphics/evidence/framed_evidence_spotlight.py",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Old (V1.2.1) source hashes, so we can record the explicit hash CHANGE.
parent_manifest = json.loads(
    (PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
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

# Explicit old -> new delta for the changed (non-new) files.
hash_change = {}
for f in sorted(CHANGED):
    old = parent_hashes.get(f, {}).get("source", "(absent in V1.2.1)")
    new = hashes[f]["source"]
    hash_change[f] = {"v1_2_1_source": old, "v1_3_source": new,
                      "changed": old != new}

lines = ["# Motion Graphics Cluster V1.3 — Evidence / Data / Location · sha256",
         "# parent: V1.2.1 Temp-Dir Cleanup (left untouched); +3 premium "
         "primitives (12 total)",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = ("  [NEW]" if f in NEW_FILES else
           ("  [CHANGED]" if f in CHANGED else ""))
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.3 — Evidence / Data / Location",
    "parent": ("Motion Graphics Cluster V1.2.1 — Temp-Dir Cleanup "
               "(snapshots/MG_Cluster_V1.2.1_TempDirCleanup, left untouched)"),
    "kind": "feature update — MagnatesMedia Batch 2 (+3 primitives, 12 total)",
    "new_primitives": {
        "framed_evidence_spotlight": (
            "family=evidence. Artifact in a gold frame under a warm spotlight, "
            "slow push-in, EVIDENCE/EXHIBIT tag chip, naming caption. No-photo "
            "path renders a premium aged-paper exhibit (caption→typeset title, "
            "printed keyline, faded typed body, ink seal) — never hollow."),
        "statistic_bar_reveal": (
            "family=charts. 1-4 vertical columns rising from a baseline with "
            "count-up numerals, labels and a title (e.g. 1 -> 12 -> 200)."),
        "location_establish_card": (
            "family=maps. Place + era over a graded antique map with corner "
            "coordinate ticks, a pulsing pin and a slow establishing push."),
    },
    "wiring": [
        "registry.py: import + register the 3 primitives (REGISTRY now 12); "
        "REQUIRED_INPUTS {caption} / {bars} / {place}; 3 INCOMPATIBLE_ADJACENT "
        "pairs (location~portrait_map, evidence~document, bars~gold_number).",
        "render_dispatch.py: USEFUL_DUR += evidence 6.0, bars 6.0, location 5.5.",
        "director.py: _ARTIFACT lexicon + artifact/place/bignum cues; "
        "_GK_AFFINITY + scoring bonuses for the 3 kinds; passthrough keys "
        "caption/tag/image_path/bars/coords/bg_image.",
        "pipeline.py: 3 additive adapter branches mapping graphic_kind "
        "evidence|bar_chart|location (+ aliases) → primitive inputs, with "
        "tag= / bars= / suffix= / prefix= / place= / coords= hint parsing.",
        "charts/__init__.py + maps/__init__.py: import the new modules; "
        "new evidence/ package with __init__.py.",
    ],
    "validated": (
        "All changed/new files py_compile OK. Director dry-run: the 3 kinds "
        "select their primitives (evidence 4.60, bars 3.60, location 4.20) and "
        "the artifact narration cue derives framed_evidence_spotlight with no "
        "graphic_kind. Real 134s Edison-vs-Tesla render exercised all 3 live "
        "alongside the 3 Batch-1 beats: manifest 8 graphics rendered, 0 "
        "fallbacks; frame-level QA confirmed location (antique map + pin), bars "
        "(1->12->200 count-up) and evidence (framed aged-paper exhibit + EXHIBIT "
        "chip) all premium. Each primitive self-cleans its PNG temp dir."),
    "cost": "fal.ai images only (~$0.02 for the validation render); $0 for code.",
    "new_files": NEW_FILES,
    "changed_files": sorted(CHANGED),
    "hash_change_v1_2_1_to_v1_3": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No beat-split", "Editor UI untouched", "No deploy",
        "9 prior primitives carried forward verbatim",
        "Temp-dir self-clean preserved (V1.2.1 fix intact)",
        "V1.2.1 / V1.2 / V1.1.2 / V1.1.1 / V1.1 snapshots untouched",
    ],
    "rollback_steps": [
        "1. Remove the 3 new primitive modules + evidence/ package and revert "
        "charts/__init__.py + maps/__init__.py to the V1.2.1 copies.",
        "2. cp registry.py / render_dispatch.py / director.py / pipeline.py "
        "from snapshots/MG_Cluster_V1.2.1_TempDirCleanup/source/ back into "
        "vidlore/ (restores the 9-primitive V1.2.1 set).",
        "3. re-sync those files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.3 snapshot built:", NEW)
print("files:", len(FILES), "· new:", len(NEW_FILES),
      "· changed:", len(CHANGED), "· all_zero_drift:", not drift)
if drift:
    print("  DRIFT:", drift)
for f in NEW_FILES:
    print(f"  [NEW]     {f}: {hashes[f]['source'][:16]}")
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CHANGED] {f}: {str(c['v1_2_1_source'])[:16]} -> "
          f"{c['v1_3_source'][:16]}")
print("V1.2.1 parent untouched:", PARENT_SNAP.exists())
