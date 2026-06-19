"""Build the MG_Cluster_V1.2.1_TempDirCleanup sibling snapshot.

Maintenance update over V1.2 (Storytelling Beats), which is LEFT UNTOUCHED as a
rollback point — exactly as V1.1.2 was built as a sibling of V1.1.1.

What changed since V1.2: the 6 originally-frozen motion-graphics primitives each
called tempfile.mkdtemp() to stage a 1080p PNG frame sequence for ffmpeg but
never removed it, leaking gigabytes of PNGs to the temp dir over long / repeated
renders. Each now calls `shutil.rmtree(td, ignore_errors=True)` right after the
ffmpeg encode (and before return) — the SAME self-clean the 3 new V1.2 primitives
already had. The cleanup runs AFTER ffmpeg has read the PNGs, so the encoded mp4
bytes are output-identical; only the temp-dir leak is fixed.

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
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.2_StorytellingBeats")
NEW = Path("snapshots/MG_Cluster_V1.2.1_TempDirCleanup")

# Full V1.2 cluster file set (carried forward verbatim from the V1.2 manifest).
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
    "vidlore/motion_graphics/maps/portrait_name_over_map.py",
    "vidlore/motion_graphics/typography/kinetic_keyword.py",
    "vidlore/motion_graphics/charts/money_flow_empire.py",
    "vidlore/motion_graphics/timelines/__init__.py",
    "vidlore/motion_graphics/timelines/chronology_timeline.py",
    "vidlore/motion_graphics/quotes/__init__.py",
    "vidlore/motion_graphics/quotes/pull_quote_portrait.py",
    "vidlore/motion_graphics/comparison/__init__.py",
    "vidlore/motion_graphics/comparison/comparison_split.py",
]
CHANGED = {  # the 6 frozen primitives that got the rmtree temp-dir cleanup
    "vidlore/motion_graphics/numbers/gold_number_callout.py",
    "vidlore/motion_graphics/portraits/cinematic_portrait_hold.py",
    "vidlore/motion_graphics/documents/headline_document_reveal.py",
    "vidlore/motion_graphics/maps/portrait_name_over_map.py",
    "vidlore/motion_graphics/typography/kinetic_keyword.py",
    "vidlore/motion_graphics/charts/money_flow_empire.py",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Old (V1.2) source hashes, so we can record the explicit hash CHANGE.
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

# Explicit old -> new delta for the 6 changed primitives.
hash_change = {}
for f in sorted(CHANGED):
    old = parent_hashes[f]["source"]
    new = hashes[f]["source"]
    hash_change[f] = {"v1_2_source": old, "v1_2_1_source": new,
                      "changed": old != new}

lines = ["# Motion Graphics Cluster V1.2.1 — Temp-Dir Cleanup · sha256",
         "# parent: V1.2 Storytelling Beats (left untouched); 6 frozen "
         "primitives now self-clean their PNG temp dir",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = "  [CLEAN]" if f in CHANGED else ""
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.2.1 — Temp-Dir Cleanup",
    "parent": ("Motion Graphics Cluster V1.2 — Storytelling Beats "
               "(snapshots/MG_Cluster_V1.2_StorytellingBeats, left untouched)"),
    "kind": "maintenance update (no new primitives, no feature change)",
    "fix": (
        "The 6 originally-frozen primitives each created a tempfile.mkdtemp() "
        "PNG frame-sequence dir for the ffmpeg encode but never removed it, "
        "leaking gigabytes of 1080p PNGs to the temp dir over long / repeated "
        "renders (one session leaked ~40 GB and nearly hit a disk-full stop). "
        "Each now calls shutil.rmtree(td, ignore_errors=True) right after the "
        "subprocess.run() encode and before return — the same self-clean the 3 "
        "new V1.2 primitives (timeline/quote/comparison) already shipped with."),
    "output_identical": (
        "Cleanup runs AFTER ffmpeg has read the PNG frames, so the encoded mp4 "
        "bytes are unchanged; only the temp-dir leak is fixed."),
    "changes": [
        "numbers/gold_number_callout.py (in its _encode helper): + rmtree(td).",
        "portraits/cinematic_portrait_hold.py: + rmtree(td) after encode.",
        "documents/headline_document_reveal.py: + rmtree(td) after encode.",
        "maps/portrait_name_over_map.py: + rmtree(td) after encode.",
        "typography/kinetic_keyword.py: + rmtree(td) after encode.",
        "charts/money_flow_empire.py: + rmtree(td) after encode.",
    ],
    "validated": (
        "All 6 py_compile OK. Micro-render of each (320x180, 8fps, 3 frames) "
        "→ ok=True and a valid mp4 that DECODES cleanly through ffmpeg "
        "(-f null, rc=0). Each render created exactly 1 tempfile.mkdtemp() dir "
        "and 0 remained on disk afterward; 0 stray f00000.png frame dirs left "
        "under a sandboxed TMPDIR. dist Mac + Win re-synced: all 6 0-drift "
        "(source == mac == win)."),
    "changed_files": sorted(CHANGED),
    "hash_change_v1_2_to_v1_2_1": hash_change,
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "scope_guardrails": [
        "No new primitives", "No beat-split", "Editor UI untouched",
        "No deploy", "No paid APIs ($0.00)",
        "V1.2 / V1.1.2 / V1.1.1 / V1.1 snapshots untouched",
        "Output-identical mp4 bytes (cleanup is post-encode only).",
    ],
    "rollback_steps": [
        "1. cp the 6 primitives from "
        "snapshots/MG_Cluster_V1.2_StorytellingBeats/source/vidlore/"
        "motion_graphics/{numbers/gold_number_callout,"
        "portraits/cinematic_portrait_hold,documents/headline_document_reveal,"
        "maps/portrait_name_over_map,typography/kinetic_keyword,"
        "charts/money_flow_empire}.py back into vidlore/ (restores pre-cleanup "
        "V1.2; reinstates the temp-dir leak).",
        "2. re-sync those 6 files to both dist trees.",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.2.1 snapshot built:", NEW)
print("files:", len(FILES), "· all_zero_drift:", not drift, "· drift:", drift)
for f in sorted(CHANGED):
    c = hash_change[f]
    print(f"  [CLEAN] {f}: {c['v1_2_source'][:16]} -> {c['v1_2_1_source'][:16]}")
print("V1.2 parent untouched:", PARENT_SNAP.exists())
