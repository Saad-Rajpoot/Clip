"""Build the MG_Cluster_V1.1.2_EncodePoolReliability sibling snapshot.

Leaves V1.1.1 (and V1.1) untouched. V1.1.2 = V1.1.1's file set + ffmpeg_tool.py
(new), with the 3 encode-pool-reliability files changed. Self-consistent hashes
(source == mac == win) verified against the dist trees, which must be synced
FIRST.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
ROOT = Path(".")
PARENT_SNAP = Path("snapshots/MG_Cluster_V1.1.1_PortraitFix")
NEW = Path("snapshots/MG_Cluster_V1.1.2_EncodePoolReliability")

FILES = [
    "vidlore/ffmpeg_tool.py",                # NEW to the snapshot set [EPR]
    "vidlore/footage.py",                    # [EPR atomic dl] + [V1.1.1]
    "vidlore/pipeline.py",
    "vidlore/assemble.py",                   # [EPR readiness/slate/pool]
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
]
CHANGED = {  # files the encode-pool reliability fix touched
    "vidlore/ffmpeg_tool.py",
    "vidlore/assemble.py",
    "vidlore/footage.py",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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

lines = ["# Motion Graphics Cluster V1.1.2 — Encode-Pool Reliability · sha256",
         "# parent: V1.1.1 Portrait Fix · only the reliability files differ",
         "# file | source(16) | match(src=mac=win)"]
for f in FILES:
    h = hashes[f]
    ok = "OK" if (h["source"] == h["mac"] == h["win"]) else "DRIFT"
    tag = "  [EPR]" if f in CHANGED else ""
    lines.append(f"{f} | {h['source'][:16]} | {ok}{tag}")
(NEW / "HASHES.txt").write_text("\n".join(lines) + "\n")

manifest = {
    "label": "Motion Graphics Cluster V1.1.2 — Encode-Pool Reliability",
    "parent": ("Motion Graphics Cluster V1.1.1 — Portrait Fix "
               "(snapshots/MG_Cluster_V1.1.1_PortraitFix, left untouched)"),
    "fix": (
        "Default 4-worker parallel encode pool no longer crashes when a "
        "scene's footage clip is partial / zero-byte / corrupt. Root cause: "
        "the tier-4 'guaranteed' graded slate was the one unguarded run() in "
        "_scene_video, so when its subprocess spawn was rejected under "
        "fd/process pressure (OSError EAGAIN) the exception propagated and "
        "pool.map abandoned the whole render."),
    "changes": [
        "ffmpeg_tool.run(): retry transient spawn failures (EAGAIN/EMFILE/"
        "ENFILE) with backoff — absorbs fork-under-load at the lowest level.",
        "assemble._clip_ready(): validate a clip (exists, stable size, min "
        "size, video stream, real duration, 3-frame decode) BEFORE the encode "
        "tiers; bad clips route straight to the slate with a logged reason.",
        "assemble._safe_slate(): the slate safety net can no longer raise "
        "(graded lavfi -> plain lavfi -> PIL still -> bare black).",
        "assemble encode pool: pool.map -> submit/as_completed; each beat is "
        "wrapped so one failure can't abort the render (full leaf traceback "
        "logged to encode_failures.log, retry once, then emergency slate).",
        "footage._stream_download_atomic(): stock clips download to a .part "
        "file then os.replace into place — the pool never reads a partial file.",
    ],
    "validated": (
        "Fault-injection render at DEFAULT 4 workers (scene-0 clips forced "
        "not-ready + first 3 slate spawns raise EAGAIN) → RENDER_DONE, 0 "
        "tracebacks, 0 black frames, $0. Plus 3x clean 4-worker + 1x single-"
        "worker stability renders + synthetic unit suite (readiness verdicts, "
        "pool-drain proof, spawn-retry, atomic download)."),
    "changed_files": sorted(CHANGED),
    "files": FILES,
    "file_hashes_sha256": hashes,
    "all_zero_drift": not drift,
    "drift_files": drift,
    "env_flags": {
        "VIDLORE_ENCODE_WORKERS": "default 4; any int. Multi-worker is the "
                                  "default and is now crash-safe.",
        "VIDLORE_MOTION_GRAPHICS": "1 to enable MG (default OFF = legacy).",
    },
    "scope_guardrails": [
        "No new primitives", "No beat-split", "Editor UI untouched",
        "No deploy", "No paid APIs ($0.00)",
        "Additive — healthy renders are byte-path-identical (no reliability "
        "events fire); only failure handling changed.",
    ],
    "new_log_artifacts": [
        "<workdir>/encode_failures.log — full leaf traceback + scene index + "
        "footage path + size for any beat that needed retry/slate.",
        "stdout '[5/5] footage not ready (scene N, file, reason); graded "
        "slate' and '[5/5] encode reliability: ok/retried/slated/failed'.",
    ],
    "rollback_steps": [
        "1. cp -r snapshots/MG_Cluster_V1.1.1_PortraitFix/source/vidlore/* "
        "vidlore/  (back to pre-EPR V1.1.1)",
        "2. re-sync ffmpeg_tool.py/assemble.py/footage.py to both dist trees",
        "3. (workaround without rollback: VIDLORE_ENCODE_WORKERS=1)",
    ],
}
(NEW / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("V1.1.2 snapshot built:", NEW)
print("files:", len(FILES), "· all_zero_drift:", not drift, "· drift:", drift)
for f in sorted(CHANGED):
    print(f"  [EPR] {f}: {hashes[f]['source'][:16]}")
print("V1.1.1 untouched:", PARENT_SNAP.exists())
