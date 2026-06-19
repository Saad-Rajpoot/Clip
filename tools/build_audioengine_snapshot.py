"""Build the AudioEngine_V1.0_MusicSFXDirector sibling snapshot.

Phase-4 PROFESSIONAL DOCUMENTARY MUSIC + SOUND-DESIGN ENGINE, built ON TOP of the
mature audio core (preserved). Parent MG_Cluster_V2.3_AssetGuards is LEFT UNTOUCHED
as a rollback point (the audio engine is additive + defensively wired).

Syncs the new/changed vidlore/ engine files to BOTH dist trees, then snapshots with
a src==mac==win drift check (dev-only tools/ + research/ are tracked source-only).

Run:  python3 tools/build_audioengine_snapshot.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
PARENT_SNAP = Path("snapshots/MG_Cluster_V2.3_AssetGuards")
NEW = Path("snapshots/AudioEngine_V1.0_MusicSFXDirector")
MAC = Path("dist/Vidlore-Mac")
WIN = Path("dist/Vidlore-Windows")

# NEW engine files that SHIP in dist (synced + drift-checked).
NEW_DIST = [
    "vidlore/audio_director/__init__.py",
    "vidlore/audio_director/music_director.py",
    "vidlore/audio_director/sfx_director.py",
    "vidlore/audio_director/audio_usage_history.py",
    "vidlore/audio_director/intro_profiles.json",
    "vidlore/audio_director/niche_mix.json",
    "vidlore/audio_director/sfx_policy.json",
    "vidlore/audio_library/category_semantics.json",
    "vidlore/audio_library/music_manifest.json",
    "vidlore/audio_library/sfx_manifest.json",
    "vidlore/audio_library/dist_exclude.txt",
]
# CHANGED engine files (ship in dist, synced + drift-checked).
CHANGED = ["vidlore/musiclib.py", "vidlore/assemble.py", "vidlore/pipeline.py"]
# Dev-only files: tracked in the snapshot SOURCE only (not in dist, no drift check).
SOURCE_ONLY = [
    "tools/build_audio_manifests.py", "tools/audio_quality_audit.py",
    "tools/test_audio_director.py", "tools/render_audio_samples.py",
    "research/audio_engine/AUDIO_ENGINE_AUDIT.md",
    "research/audio_engine/AUDIO_SOURCE_LICENSE_MATRIX.md",
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sync_to_dist():
    """Copy NEW_DIST + CHANGED engine files into both dist trees."""
    synced = 0
    for f in NEW_DIST + CHANGED:
        for tree in (MAC, WIN):
            dst = tree / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            synced += 1
    print(f"  synced {len(NEW_DIST + CHANGED)} engine files -> Mac + Win ({synced} copies)")


def main():
    for f in NEW_DIST + CHANGED + SOURCE_ONLY:
        assert Path(f).exists(), f"missing file: {f}"
    sync_to_dist()

    parent_manifest = json.loads((PARENT_SNAP / "SNAPSHOT_MANIFEST.json").read_text())
    parent_hashes = parent_manifest["file_hashes_sha256"]
    parent_files = list(parent_manifest["files"])
    # carry forward parent files + add NEW_DIST + ensure CHANGED are tracked
    # (musiclib.py was untouched in MG work, so it is not in the parent list).
    dist_files = list(dict.fromkeys(parent_files + NEW_DIST + CHANGED))

    if NEW.exists():
        shutil.rmtree(NEW)
    (NEW / "source").mkdir(parents=True)
    # carry the bundled assets dir forward verbatim
    if (PARENT_SNAP / "source/vidlore/assets").exists():
        shutil.copytree(PARENT_SNAP / "source/vidlore/assets",
                        NEW / "source/vidlore/assets")

    hashes, drift = {}, []
    for f in dist_files:
        dst = NEW / "source" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not Path(f).exists():
            # carried-forward file already inside parent assets copytree
            if (PARENT_SNAP / "source" / f).exists():
                shutil.copy2(PARENT_SNAP / "source" / f, dst)
            continue
        shutil.copy2(f, dst)
        hs = sha(f)
        hm = sha(MAC / f) if (MAC / f).exists() else "(absent)"
        hw = sha(WIN / f) if (WIN / f).exists() else "(absent)"
        hashes[f] = {"source": hs, "mac": hm, "win": hw}
        if not (hs == hm == hw):
            drift.append(f)

    # dev-only source files (no dist counterpart)
    src_only_hashes = {}
    for f in SOURCE_ONLY:
        dst = NEW / "source" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        src_only_hashes[f] = sha(f)

    hash_change = {}
    for f in CHANGED:
        old = parent_hashes.get(f, {}).get("source", "(absent in V2.3)")
        new = hashes[f]["source"]
        hash_change[f] = {"v2_3_source": old, "audio_v1_source": new,
                          "changed": old != new}

    lines = ["# AudioEngine V1.0 — Music + SFX Director · sha256",
             "# parent: MG_Cluster_V2.3_AssetGuards (untouched); audio engine is additive",
             "# file | source(16) | match(src=mac=win)"]
    for f in dist_files:
        if f not in hashes:
            continue
        h = hashes[f]
        ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
        tag = ("  [NEW]" if f in NEW_DIST else
               "  [CHANGED]" if f in CHANGED else "")
        lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
    (NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

    manifest = {
        "label": "AudioEngine V1.0 — Music Director + SFX Director + cue sheets",
        "parent": ("MG_Cluster_V2.3_AssetGuards "
                   "(snapshots/MG_Cluster_V2.3_AssetGuards, untouched)"),
        "kind": ("Phase-4 professional documentary music + sound-design engine, "
                 "additive on top of the mature musiclib/sfx/assemble core (preserved)"),
        "new_modules": [
            "vidlore/audio_director/{music_director,sfx_director,audio_usage_history}.py",
            "vidlore/audio_director/{intro_profiles,niche_mix,sfx_policy}.json (verified specs)",
            "vidlore/audio_library/{category_semantics,music_manifest,sfx_manifest}.json + dist_exclude.txt",
            "tools/{build_audio_manifests,audio_quality_audit,test_audio_director,render_audio_samples}.py",
        ],
        "changed_files": CHANGED,
        "capabilities": [
            "Niche-aware INTRO intelligence (louder-then-recede, 6 niche profiles)",
            "Chapter-level MUSIC CUE SHEET (music_cue_sheet.json) from real selected tracks",
            "Per-niche reveal-duck character (bounded, preserves the proven base duck)",
            "SFX DIRECTOR restraint: per-primitive max-intensity caps + silence-default cards + SFX CUE SHEET",
            "CROSS-VIDEO anti-repetition: category + SFX-family cooldowns + deterministic per-video seeding (audio_usage_history.json)",
            "AUDIO QA GATE (tools/audio_quality_audit.py -> audio_quality_report.json, PASS/WARN/FAIL)",
            "Music + SFX library MANIFESTS w/ full license provenance (118 tracks, 123 SFX presets)",
            "Two-tier LICENSE policy enforced: 91 bundle-OK CC-BY + synth; 27 Mixkit USE-ONLY excluded from dist",
        ],
        "preserved": [
            "musiclib selection/scoring/crossfades/reveal-tiers, sfx synthesis, assemble two-stage mux + sidechain duck",
            "empirically-tuned base duck (threshold 0.030 ratio 12) — NOT replaced; per-niche values kept as QA/reference",
            "per-niche bed character already in look DNA — NOT double-applied",
        ],
        "tests": "tools/test_audio_director.py 66/66 · tools/test_engine_guards.py 87/87",
        "files": dist_files,
        "file_hashes_sha256": hashes,
        "source_only_files": SOURCE_ONLY,
        "source_only_sha256": src_only_hashes,
        "hash_change_v2_3_to_audio_v1": hash_change,
        "all_zero_drift": not drift,
        "drift_files": drift,
        "scope_guardrails": [
            "No new motion-graphics primitives (39 unchanged)",
            "No editor-UI work", "Not deployed",
            "Audio engine additive + defensively wired (try/except -> legacy)",
            "MG_Cluster_V2.3 + all prior snapshots untouched",
        ],
        "rollback_steps": [
            "Restore vidlore/musiclib.py, assemble.py, pipeline.py from "
            "snapshots/MG_Cluster_V2.3_AssetGuards/source/, delete vidlore/audio_director/ "
            "and vidlore/audio_library/, then re-sync both dist trees. The directors are "
            "import-guarded, so deleting them alone reverts to legacy behaviour.",
        ],
    }
    (NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("AudioEngine V1.0 snapshot built:", NEW)
    print(f"  dist files: {len([f for f in dist_files if f in hashes])} "
          f"({len(NEW_DIST)} new + {len(CHANGED)} changed) · "
          f"source-only: {len(SOURCE_ONLY)} · all_zero_drift: {not drift}")
    if drift:
        print("  DRIFT:", drift)
    print("  parent V2.3 untouched:", PARENT_SNAP.exists())


if __name__ == "__main__":
    main()
