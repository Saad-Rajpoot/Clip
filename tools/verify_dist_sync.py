#!/usr/bin/env python3
"""STRICT dist-sync verifier — root vidlore/ (source of truth) vs both dist packages.

Hashes (sha256) every file in the vidlore/ package across ROOT / MAC / WIN, buckets
by component type, explicitly checks each named engine tool, verifies the packaging
layer, and SEPARATES intentional dev-only exclusions from real drift.

Usage:  python3 tools/verify_dist_sync.py
Exit 0 = no real drift.  Exit 1 = drift found (missing/mismatched non-excluded files).
"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKG_TREES = {
    "ROOT": os.path.join(REPO, "vidlore"),
    "MAC":  os.path.join(REPO, "dist", "Vidlore-Mac", "vidlore"),
    "WIN":  os.path.join(REPO, "dist", "Vidlore-Windows", "vidlore"),
}

# Files that are EXPECTED to be absent in dist (dev-only) — not drift.
def is_dev_only(rel):
    if rel.startswith("audio_library/ytal_cache/"):  # 3.4G dev cache, re-ingestable
        return True
    if rel.endswith(".pyc") or "__pycache__/" in rel:
        return True
    if rel.endswith(".DS_Store"):
        return True
    return False


# Machine-specific state the ENGINE regenerates at runtime — never an authored asset,
# so a root/dist difference here is EXPECTED, not drift. Evidence (musiclib.py):
#   * scan() rebuilds _index.json from a {relpath:mtime_ns} signature; the music root
#     is resolved at runtime from Path(__file__)/assets/music (or $VIDLORE_MUSIC_DIR),
#     NOT from the absolute paths baked into _index.json. Stale/missing -> rebuilt.
#   * _usage/_history/audio_usage_history are written every render (anti-repetition).
#   * ytal_*_manifest.json index the dev-only ytal_cache (which is itself dist-excluded).
RUNTIME_STATE = {
    "assets/music/_index.json",                # library cache, rebuilt by scan()
    "assets/music/_usage.json",                # anti-repeat play counts (per-render)
    "assets/music/_history.json",              # selection history (per-render)
    "assets/music/audio_usage_history.json",   # usage history (per-render)
    "audio_library/ytal_music_manifest.json",  # indexes dev-only ytal_cache
    "audio_library/ytal_sfx_manifest.json",    # indexes dev-only ytal_cache
}
def is_runtime_state(rel):
    return rel in RUNTIME_STATE


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel.endswith(".pyc") or rel.endswith(".DS_Store"):
                continue
            try:
                out[rel] = (sha256_of(full), os.path.getsize(full))
            except OSError as exc:
                out[rel] = ("ERR:%s" % exc, -1)
    return out


def bucket_of(rel):
    if rel.endswith(".py"):
        return "engine_code_py"
    if rel.endswith(".ttf") or rel.endswith(".otf"):
        return "fonts"
    if rel.startswith("assets/music/"):
        return "music"
    if rel.startswith("assets/geo/"):
        return "geo_map_data"
    if rel.startswith("audio_library/ytal_cache/"):
        return "ytal_cache_devonly"
    if rel.endswith(".json"):
        return "data_manifests_json"
    return "other_assets"


NAMED_COMPONENTS = [
    ("Visual-variation selector + anti-repetition", "motion_graphics/variants.py"),
    ("Within-video anti-repetition (recipe)",       "editorial_recipe.py"),
    ("Factual guard (no-fabrication)",              "factual_guard.py"),
    ("Quality check - asset QA",                    "asset_qa.py"),
    ("Quality check - editorial QA",                "editorial_qa.py"),
    ("Quality check - QA autofix",                  "qa_autofix.py"),
    ("Black-frame / corrupt-clip quarantine",       "render_quarantine.py"),
    ("Black-frame detect+repair (assembly)",        "assemble.py"),
    ("Primitive registry (71)",                     "motion_graphics/registry.py"),
    ("Variant render dispatch",                     "motion_graphics/render_dispatch.py"),
    ("MG director",                                 "motion_graphics/director.py"),
    ("Card-style guard (dark niche)",               "card_style_guard.py"),
    ("Period guard (wrong-era footage)",            "period_guard.py"),
    ("Visual relevance (CLIP) scorer",              "visual_relevance.py"),
    ("Music director / selection",                  "music.py"),
    ("SFX engine",                                  "sfx.py"),
    ("Pipeline (guard sweep wiring)",               "pipeline.py"),
    ("Dashboard backend",                           "web.py"),
]

mans = {name: manifest(root) for name, root in PKG_TREES.items()}
root_man = mans["ROOT"]

report = {"package": {}, "named_components": {}, "buckets": {}, "packaging": {}}

# ---- per-dist drift (vs ROOT, ignoring dev-only) -------------------------------
print("=" * 78)
print("STRICT DIST-SYNC VERIFICATION  (source of truth = root vidlore/)")
print("=" * 78)

overall_ok = True
for dist in ("MAC", "WIN"):
    dm = mans[dist]
    missing, mismatch, extra, runtime = [], [], [], []
    for rel, (h, sz) in root_man.items():
        if is_dev_only(rel):
            continue
        if is_runtime_state(rel):
            runtime.append(rel)  # expected to differ/absent; engine regenerates
            continue
        if rel not in dm:
            missing.append(rel)
        elif dm[rel][0] != h:
            mismatch.append(rel)
    for rel in dm:
        if rel not in root_man and not is_dev_only(rel) and not is_runtime_state(rel):
            extra.append(rel)
    ok = not missing and not mismatch
    overall_ok = overall_ok and ok
    shippable = sum(1 for r in root_man if not is_dev_only(r) and not is_runtime_state(r))
    identical = sum(1 for r in root_man
                    if not is_dev_only(r) and not is_runtime_state(r)
                    and r in dm and dm[r][0] == root_man[r][0])
    report["package"][dist] = {
        "shippable_files_in_root": shippable,
        "byte_identical_in_dist": identical,
        "missing": missing, "mismatch": mismatch, "extra": extra,
        "runtime_state_regenerated": runtime, "synced": ok,
    }
    print()
    print("[%s]  vs ROOT" % dist)
    print("  shippable files in ROOT : %d  (excl. dev-only cache + runtime state)" % shippable)
    print("  byte-identical in dist  : %d" % identical)
    print("  MISSING in dist         : %d" % len(missing))
    print("  HASH MISMATCH           : %d" % len(mismatch))
    print("  extra (dist-only)       : %d" % len(extra))
    print("  runtime-state (expected): %d  (engine rebuilds on target machine)" % len(runtime))
    for r in missing[:25]:
        print("     MISSING  %s" % r)
    for r in mismatch[:25]:
        print("     DIFFERS  %s" % r)
    for r in extra[:25]:
        print("     EXTRA    %s" % r)
    print("  VERDICT: %s" % ("SYNCED (0 drift)" if ok else "DRIFT DETECTED"))

# ---- per-bucket counts (ROOT vs each dist, byte-identical) ---------------------
print()
print("-" * 78)
print("COMPONENT BUCKETS  (count = byte-identical matches ROOT->dist)")
print("-" * 78)
buckets = {}
for rel in root_man:
    buckets.setdefault(bucket_of(rel), []).append(rel)
hdr = "  %-26s %7s %9s %9s" % ("bucket", "ROOT", "MAC=", "WIN=")
print(hdr)
for b in sorted(buckets):
    rels = buckets[b]
    n = len(rels)
    mac = sum(1 for r in rels if r in mans["MAC"] and mans["MAC"][r][0] == root_man[r][0])
    win = sum(1 for r in rels if r in mans["WIN"] and mans["WIN"][r][0] == root_man[r][0])
    tag = "  (dev-only, dist-excluded by design)" if b == "ytal_cache_devonly" else ""
    print("  %-26s %7d %9d %9d%s" % (b, n, mac, win, tag))
    report["buckets"][b] = {"root": n, "mac_identical": mac, "win_identical": win}

# ---- runtime-state transparency (NOT drift; shown for honesty) ----------------
print()
print("-" * 78)
print("RUNTIME-STATE / DEV-ONLY (regenerated on target; difference is EXPECTED)")
print("-" * 78)
_reasons = {
    "assets/music/_index.json": "music library cache - musiclib.scan() rebuilds from relpath:mtime signature",
    "assets/music/_usage.json": "anti-repetition play counts - written every render",
    "assets/music/_history.json": "music selection history - written every render",
    "assets/music/audio_usage_history.json": "usage history - written every render",
    "audio_library/ytal_music_manifest.json": "indexes dev-only ytal_cache (3.4G, dist-excluded)",
    "audio_library/ytal_sfx_manifest.json": "indexes dev-only ytal_cache (3.4G, dist-excluded)",
}
for rel in sorted(RUNTIME_STATE):
    in_mac = "present" if rel in mans["MAC"] else "absent "
    in_win = "present" if rel in mans["WIN"] else "absent "
    print("  %-40s MAC:%s WIN:%s  (%s)" % (rel, in_mac, in_win, _reasons.get(rel, "")))
report["runtime_state_files"] = sorted(RUNTIME_STATE)

# ---- named component explicit check -------------------------------------------
print()
print("-" * 78)
print("NAMED ENGINE TOOLS  (present + byte-identical across ROOT/MAC/WIN)")
print("-" * 78)
all_named_ok = True
for label, rel in NAMED_COMPONENTS:
    hr = root_man.get(rel, (None,))[0]
    hm = mans["MAC"].get(rel, (None,))[0]
    hw = mans["WIN"].get(rel, (None,))[0]
    present = hr is not None and hm is not None and hw is not None
    identical = present and hr == hm == hw
    all_named_ok = all_named_ok and identical
    state = "OK  identical" if identical else ("MISSING" if not present else "DRIFT")
    short = (hr[:12] if hr else "------------")
    print("  [%-13s] %-44s sha=%s" % (state, rel, short))
    report["named_components"][rel] = {
        "label": label, "present_all": present, "identical": identical,
        "sha_root": hr, "sha_mac": hm, "sha_win": hw,
    }

# ---- packaging layer (top-level support files) --------------------------------
print()
print("-" * 78)
print("PACKAGING LAYER  (launcher + support files)")
print("-" * 78)
PKG_LAYER = [
    ("requirements.txt", "requirements.txt", "requirements.txt"),
    ("examples/sample_brief.yaml", "examples/sample_brief.yaml", "examples/sample_brief.yaml"),
]
def h_at(base, rel):
    p = os.path.join(REPO, base, rel)
    return sha256_of(p) if os.path.isfile(p) else None
for root_rel, mac_rel, win_rel in PKG_LAYER:
    hr = h_at(".", root_rel)
    hm = h_at(os.path.join("dist", "Vidlore-Mac"), mac_rel)
    hw = h_at(os.path.join("dist", "Vidlore-Windows"), win_rel)
    ok = hr is not None and hr == hm == hw
    print("  [%s] %s" % ("OK identical" if ok else "CHECK", root_rel))
    report["packaging"][root_rel] = {"root": hr, "mac": hm, "win": hw, "identical": ok}
# launcher presence (platform-specific, NOT expected to cross-match)
for base, fn in [("dist/Vidlore-Mac", "run-mac.command"),
                 ("dist/Vidlore-Windows", "run-windows.bat")]:
    p = os.path.join(REPO, base, fn)
    print("  [%s] %s (platform launcher)" % ("present" if os.path.isfile(p) else "MISSING", os.path.join(base, fn)))

# ---- use-only music licensing status (informational) --------------------------
excl_file = os.path.join(REPO, "vidlore", "audio_library", "dist_exclude.txt")
use_only = []
if os.path.isfile(excl_file):
    with open(excl_file) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                use_only.append(line)
print()
print("-" * 78)
print("USE-ONLY MUSIC (licensing flag, informational)")
print("-" * 78)
def in_dist(base, rel):
    return os.path.isfile(os.path.join(REPO, "dist", base, rel))
present_mac = sum(1 for r in use_only if in_dist("Vidlore-Mac", r))
present_win = sum(1 for r in use_only if in_dist("Vidlore-Windows", r))
print("  %d use-only tracks listed in dist_exclude.txt" % len(use_only))
print("  present in MAC dist: %d   present in WIN dist: %d" % (present_mac, present_win))
print("  (these ARE synced/present; flagged for removal before PUBLIC redistribution)")
report["use_only_music"] = {"listed": len(use_only), "in_mac": present_mac, "in_win": present_win}

# ---- final verdict ------------------------------------------------------------
print()
print("=" * 78)
verdict = "PASS - dist packages SYNCED with root (0 real drift)" if (overall_ok and all_named_ok) \
          else "FAIL - drift detected (see above)"
print("OVERALL: %s" % verdict)
print("=" * 78)
report["overall_synced"] = bool(overall_ok and all_named_ok)

out_json = os.path.join(REPO, "research", "final_release", "DIST_SYNC_STRICT_REPORT.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as fh:
    json.dump(report, fh, indent=2)
print("JSON report: %s" % out_json)
sys.exit(0 if report["overall_synced"] else 1)
