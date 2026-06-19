"""IMP_026 confirming render — full pipeline on the '6-8' Escobar brief to
verify the FIXED duration end-to-end (actual minutes + benchmark + 40-scene
scaling). Fresh output dir so the pre-fix validation render is preserved."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("WEB_IMAGE_ENGINE", "1")
os.environ.setdefault("WEB_IMAGE_MIX", "light")

from vidlore.brief import Brief             # noqa: E402
from vidlore.config import load_config      # noqa: E402
from vidlore.pipeline import produce        # noqa: E402

cfg = load_config(ROOT)
brief = Brief(
    title="Pablo Escobar: The Rise and Fall of the King of Cocaine",
    prompt=("A cinematic, factual crime documentary on Pablo Escobar — his "
            "rise from Medellin, the cartel empire, the violence, the money, "
            "and his 1993 downfall. Footage-first, restrained, premium."),
    duration="6-8", theme="crime", fmt="documentary",
)
out_dir = ROOT / "output" / "_imp026_render"
out_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
res = produce(brief, cfg, out_dir)
dt = time.time() - t0
print("\n==== IMP_026 RENDER DONE ====", flush=True)
print(f"  video:     {res.video}", flush=True)
print(f"  wallclock: {dt:.1f}s", flush=True)
print(f"  exists:    {Path(res.video).exists()} "
      f"({Path(res.video).stat().st_size/1e6:.1f} MB)" if Path(res.video).exists()
      else "  MISSING", flush=True)
