#!/usr/bin/env python3
"""Windows/dist runtime-path probe — prove which PHYSICAL files the app loads.
Prints filenames only (never secrets). Run with PYTHONPATH set to the dist whose
package you want to probe, e.g.:
    PYTHONPATH=dist/Vidlore-Windows python tools/windows_runtime_path_probe.py
    PYTHONPATH=dist/Vidlore-Mac     python tools/windows_runtime_path_probe.py
    PYTHONPATH=.                    python tools/windows_runtime_path_probe.py   # source
"""
import importlib
import os
from pathlib import Path


def fileof(modname):
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__file__", "<no __file__>")
    except Exception as e:  # noqa: BLE001
        return f"<IMPORT FAIL: {type(e).__name__}: {str(e)[:80]}>"


def main():
    print("=" * 72)
    print("VIDLORE RUNTIME-PATH PROBE")
    print("  sys.path[0:3] =", __import__("sys").path[:3])
    print("=" * 72)

    pkg = fileof("vidlore")
    print(f"vidlore.__file__            : {pkg}")
    pkg_dir = Path(pkg).resolve().parent if "<" not in pkg else None
    print(f"vidlore package dir         : {pkg_dir}")

    for label, mod in (
        ("registry", "vidlore.motion_graphics.registry"),
        ("director", "vidlore.motion_graphics.director"),
        ("dispatcher", "vidlore.motion_graphics.render_dispatch"),
        ("web backend", "vidlore.web"),
        ("pipeline", "vidlore.pipeline"),
        ("visual_relevance", "vidlore.visual_relevance"),
        ("relevance_qa", "vidlore.relevance_qa"),
        ("script_gen", "vidlore.script_gen"),
        ("sfx_director", "vidlore.audio_director.sfx_director"),
    ):
        print(f"{label:<27}: {fileof(mod)}")

    # registry count + removed-MG presence
    try:
        from vidlore.motion_graphics.registry import REGISTRY, eligible
        n = len(REGISTRY)
        pfs_reg = "process_flow_steps" in REGISTRY
        pfs_elig = False
        for nm in (None, "history", "business", "science", "spy_intel", "true_crime"):
            for it in (1, 3, 5):
                if "process_flow_steps" in {e["id"] for e in eligible(
                        niche=nm, intensity=it,
                        have_inputs={k: True for k in ("title", "steps", "stat", "quote", "person", "place")})}:
                    pfs_elig = True
        print(f"REGISTRY count             : {n}  (expected 70)")
        print(f"process_flow_steps in REG  : {pfs_reg}")
        print(f"process_flow_steps eligible: {pfs_elig}")
    except Exception as e:  # noqa: BLE001
        print("REGISTRY probe FAILED:", e)

    # removed-MG renderer file physically present in the LOADED package?
    if pkg_dir:
        f = pkg_dir / "motion_graphics" / "diagrams" / "process_flow_steps.py"
        print(f"renderer file on disk      : {'PRESENT (STALE!)' if f.exists() else 'ABSENT'}  [{f}]")
        diagdir = pkg_dir / "motion_graphics" / "diagrams"
        if diagdir.is_dir():
            names = sorted(p.name for p in diagdir.glob("*.py"))
            print(f"diagrams/*.py ({len(names)})        : {', '.join(names)}")

    # Flask static/template dirs (the dashboard is mostly inline strings)
    try:
        from vidlore.web import app
        print(f"Flask static_folder        : {getattr(app, 'static_folder', None)}")
        print(f"Flask template_folder      : {getattr(app, 'template_folder', None)}")
    except Exception as e:  # noqa: BLE001
        print("Flask app probe note       :", str(e)[:80])

    # output / cache resolution (filenames only)
    try:
        from vidlore.web import OUT
        print(f"web OUT dir                : {OUT.resolve() if hasattr(OUT, 'resolve') else OUT}")
    except Exception as e:  # noqa: BLE001
        print("OUT probe note             :", str(e)[:80])
    print(f"HOME caches                : {Path.home()/'.vidlore'} | {Path.home()/'.cache'/'vidlore_clip'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
