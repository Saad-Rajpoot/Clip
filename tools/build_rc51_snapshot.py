"""Build the RC5.1 snapshot, faithful to the RC5 pattern:
  snapshots/Vidlore_V1.0_RC5.1_VisualPolishAndResidualBypassFix/
    source/{vidlore,tests,tools}   (secret-free, no __pycache__, no output, no mp4)
    final_release/{visual_polish,relevance_gate,final_acceptance_fix}
    SHARED_FILE_HASHES.txt
    ROLLBACK.md
Preserves every existing snapshot. Does NOT create a FinalRelease snapshot.
"""
import hashlib, shutil, subprocess
from pathlib import Path

REPO = Path("/Users/hussnain/Desktop/vidrush-clone")
SNAP = REPO / "snapshots/Vidlore_V1.0_RC5.1_VisualPolishAndResidualBypassFix"

EXCLUDES = ["--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.DS_Store",
            "--exclude=.env", "--exclude=.git", "--exclude=output",
            "--exclude=*.mp4", "--exclude=editor_cache", "--exclude=*.part"]

RC51_FILES = [
    "vidlore/relevance_qa.py", "vidlore/pipeline.py", "vidlore/script_gen.py",
    "vidlore/assemble.py", "vidlore/motion_graphics/render_dispatch.py",
    "vidlore/motion_graphics/look.py", "vidlore/motion_graphics/registry.py",
    "vidlore/motion_graphics/director.py", "vidlore/motion_graphics/_shared.py",
]


def main():
    if SNAP.exists():
        shutil.rmtree(SNAP)
    (SNAP / "source").mkdir(parents=True)
    (SNAP / "final_release").mkdir(parents=True)

    # --- source: vidlore + tests (mirror RC5) ---
    for d in ("vidlore", "tests"):
        subprocess.run(["rsync", "-a", *EXCLUDES,
                        f"{REPO/d}/", f"{SNAP/'source'/d}/"], check=True)
    # --- source/tools: the rc4/rc5/rc5.1 test + verification harnesses (restorable) ---
    (SNAP / "source/tools").mkdir(parents=True, exist_ok=True)
    tool_globs = ["test_rc4*.py", "test_rc5*.py", "test_rc51*.py",
                  "run_rc51_served_render.py", "rc51_contact_sheet.py",
                  "build_rc51_snapshot.py"]
    n_tools = 0
    for g in tool_globs:
        for p in (REPO / "tools").glob(g):
            shutil.copy2(p, SNAP / "source/tools" / p.name); n_tools += 1

    # --- final_release reports ---
    for sub in ("visual_polish", "relevance_gate", "final_acceptance_fix"):
        s = REPO / "research/final_release" / sub
        if s.exists():
            subprocess.run(["rsync", "-a", "--exclude=.DS_Store",
                            f"{s}/", f"{SNAP/'final_release'/sub}/"], check=True)

    # --- SHARED_FILE_HASHES.txt (RC5.1 files + the load-bearing RC5 gate files) ---
    hash_files = RC51_FILES + ["vidlore/visual_relevance.py", "vidlore/footage.py",
                               "vidlore/period_guard.py", "vidlore/web.py"]
    lines = []
    for f in hash_files:
        p = REPO / f
        if p.exists():
            lines.append(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {f}")
    (SNAP / "SHARED_FILE_HASHES.txt").write_text("\n".join(lines) + "\n")

    # --- ROLLBACK.md ---
    (SNAP / "ROLLBACK.md").write_text(
        "# RC5.1 ROLLBACK — visual polish + residual game-UI-bypass fix\n"
        "Revert these source files (+ tools/test_rc51_*.py), re-sync both dist trees:\n"
        + "".join(f"  {f}\n" for f in RC51_FILES)
        + "\nRC5.1 deltas vs RC5:\n"
        "  * process_flow_steps removed from REGISTRY (71->70) + scriptwriter vocab + dispatcher\n"
        "  * relevance_qa: post-render stale-output hash guard + per-beat coverage\n"
        "  * relevance_qa: ui_geom game-UI probe exempt inside engine card windows\n"
        "    (redacted_document/dashboard/diagram cards) — VIDLORE_QA_CARD_UI_STRICT=1 reverts\n"
        "  * assemble: _OVR overlay-restraint live in the final export clarity gate\n"
        "  * motion_graphics/look + _shared: central text-readability gate (conservative)\n"
        "\nReverting these restores RC5 behaviour. Additive; no data/settings migration.\n"
        "Restore point: snapshot Vidlore_V1.0_RC5.1_VisualPolishAndResidualBypassFix.\n"
    )

    size = subprocess.run(["du", "-sh", str(SNAP)], capture_output=True, text=True).stdout.split()[0]
    py = subprocess.run(["bash", "-c", f"find {SNAP/'source'} -name '*.py' | wc -l"],
                        capture_output=True, text=True).stdout.strip()
    print(f"snapshot built: {SNAP.name}")
    print(f"  size={size}  source_py_files={py}  tool_harnesses={n_tools}")
    print(f"  has .env: {any(SNAP.rglob('.env'))}  has output/: {any(p.is_dir() and p.name=='output' for p in SNAP.rglob('output'))}")


if __name__ == "__main__":
    main()
