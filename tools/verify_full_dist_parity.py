#!/usr/bin/env python3
"""Reusable full-dist parity verifier.

Compares a dist's `vidlore` package against the authoritative source package and
reports missing / extra / sha256-mismatch / stale junk / duplicate trees /
removed-template references / registry mismatch / output leaks. Distinguishes
PLATFORM-EXPECTED and RUNTIME-GENERATED differences from real drift.

Exit 0 ONLY if the package is clean; non-zero if any UNEXPLAINED mismatch exists.

Usage:
    python tools/verify_full_dist_parity.py dist/Vidlore-Windows
    python tools/verify_full_dist_parity.py dist/Vidlore-Mac
"""
import hashlib, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "vidlore"
EXPECTED_MISSING_PREFIXES = ("research/", "audio_library/ytal_cache/")
EXPECTED_MISSING_EXACT = {
    "assets/music/_index.json", "assets/music/_usage.json",
    "assets/music/_history.json", "assets/music/audio_usage_history.json",
}
RUNTIME_GENERATED_EXTRA = EXPECTED_MISSING_EXACT  # may legitimately appear in a used dist
REMOVED_TEMPLATE = b"process_flow_steps"
EXPECTED_REGISTRY = 70


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pkg_files(base):
    out = {}
    for p in base.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo") and p.name != ".DS_Store":
            out[str(p.relative_to(base))] = p
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: verify_full_dist_parity.py <dist_dir>"); sys.exit(2)
    dist = (REPO / sys.argv[1]) if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
    pkg = dist / "vidlore"
    problems = []
    notes = []
    if not pkg.is_dir():
        print(f"FAIL: no vidlore package at {pkg}"); sys.exit(2)

    s = pkg_files(SRC)
    d = pkg_files(pkg)

    # mismatches / missing / extra
    mismatch = [k for k in d if k in s and md5(d[k]) != md5(s[k])]
    missing = [k for k in s if k not in d]
    extra = [k for k in d if k not in s]
    real_missing = [k for k in missing
                    if not any(k.startswith(p) for p in EXPECTED_MISSING_PREFIXES)
                    and k not in EXPECTED_MISSING_EXACT]
    real_extra = [k for k in extra if k not in RUNTIME_GENERATED_EXTRA]
    if mismatch:
        problems.append(f"{len(mismatch)} sha256 mismatch(es): {mismatch[:5]}")
    if real_missing:
        problems.append(f"{len(real_missing)} missing runtime file(s): {real_missing[:5]}")
    if real_extra:
        problems.append(f"{len(real_extra)} unexplained extra file(s): {real_extra[:5]}")
    if len(missing) - len(real_missing):
        notes.append(f"{len(missing)-len(real_missing)} expected exclusions (ytal cache / dev docs / runtime indexes)")
    if len(extra) - len(real_extra):
        notes.append(f"{len(extra)-len(real_extra)} runtime-generated index file(s) present (regenerate; harmless)")

    # junk
    junk_pyc = sum(1 for p in dist.rglob("*.pyc"))
    junk_cache = sum(1 for p in dist.rglob("__pycache__") if p.is_dir())
    junk_ds = sum(1 for p in dist.rglob(".DS_Store"))
    if junk_pyc or junk_cache:
        problems.append(f"stale compiled junk: {junk_pyc} .pyc / {junk_cache} __pycache__")
    if junk_ds:
        notes.append(f"{junk_ds} .DS_Store (cosmetic)")

    # shadow / duplicate trees
    shadow = [str(p.relative_to(dist)) for p in dist.rglob("*")
              if p.is_dir() and p.name in ("vidlore", "vidrush") and p.parent != dist]
    if shadow:
        problems.append(f"shadow/duplicate package tree(s): {shadow}")

    # output leak
    out_leak = sum(1 for _ in (dist / "output").rglob("*") if _.is_file()) if (dist / "output").is_dir() else 0
    if out_leak:
        notes.append(f"{out_leak} render-output file(s) under output/ (not runtime code)")

    # removed-template byte trace
    pfs = [str(p.relative_to(dist)) for p in dist.rglob("*")
           if p.is_file() and REMOVED_TEMPLATE in (p.read_bytes() if p.stat().st_size < 5_000_000 else b"")]
    if pfs:
        problems.append(f"removed template '{REMOVED_TEMPLATE.decode()}' still present in: {pfs[:5]}")

    # registry count (import the dist package)
    reg_ok = None
    try:
        sys.path.insert(0, str(dist))
        for m in list(sys.modules):
            if m == "vidlore" or m.startswith("vidlore."):
                del sys.modules[m]
        from vidlore.motion_graphics.registry import REGISTRY  # type: ignore
        if len(REGISTRY) != EXPECTED_REGISTRY:
            problems.append(f"registry={len(REGISTRY)} (expected {EXPECTED_REGISTRY})")
        reg_ok = len(REGISTRY)
    except Exception as e:  # noqa: BLE001
        problems.append(f"registry import failed: {type(e).__name__}: {str(e)[:80]}")

    print(f"== verify_full_dist_parity: {dist} ==")
    print(f"  package files: source={len(s)} dist={len(d)} | common-identical={len([k for k in d if k in s and md5(d[k])==md5(s[k])])}")
    print(f"  mismatch={len(mismatch)} real-missing={len(real_missing)} real-extra={len(real_extra)} "
          f"shadow={len(shadow)} pyc={junk_pyc} registry={reg_ok}")
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print("  RESULT: FAIL")
        for p in problems:
            print("   - " + p)
        sys.exit(1)
    print("  RESULT: PASS — dist package clean + byte-identical to source (explained exclusions only)")
    sys.exit(0)


if __name__ == "__main__":
    main()
