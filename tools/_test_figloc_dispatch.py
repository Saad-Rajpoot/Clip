"""Deterministic wiring test for IMP_021 figure_locator dispatch.

No paid API: has_fal=False, portrait supplied via footage_paths fallback
(a local real photo). Exercises build_graphic_images -> figure_locator
branch -> _render_portrait_map_tether_card (real geocode + satellite map).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vidlore import footage  # noqa: E402
from vidlore.script_gen import Scene  # noqa: E402

work = ROOT / "output" / "_imp021_dispatch_test"
work.mkdir(parents=True, exist_ok=True)
cache = work / "cache"
cache.mkdir(parents=True, exist_ok=True)

portrait_src = ROOT / "research" / "imp010_real_photo.png"
assert portrait_src.exists(), "need a local face photo for the fallback"

scenes = [
    Scene(
        index=0,
        narration="He built his empire from the hills of Medellin, and for "
                  "a decade the city answered to one man.",
        keywords=["empire", "city", "power"],
        intensity=4,
        role="reveal",
        graphic_kind="figure_locator",
        graphic_text="Pablo Escobar",
        graphic_body="Medellin, Colombia",
    ),
]

cfg = SimpleNamespace(
    has_fal=False,
    fal_key="",
    fal_model="",
)

theme = {"accent": "#d4af37", "name": "investigation"}

out = footage.build_graphic_images(
    scenes, cfg, work,
    cache_dir=cache,
    footage_paths={0: str(portrait_src)},
    theme=theme,
)
print("RESULT:", out)
if out.get(0):
    asset = work / out[0]
    print("ASSET:", asset, "exists=", asset.exists(),
          "bytes=", asset.stat().st_size if asset.exists() else 0)
else:
    print("NO ASSET PRODUCED")
