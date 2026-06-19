"""Audio sound-design validation render (v14).

Short, reveal-rich crime/investigation doc to exercise the new audio engine:
strong hook (intro music + intro accent), concrete figures (number reveals),
a timeline, a map, a document/evidence reveal, transitions (whoosh/boom). AI
images OFF (cheap/fast — audio is what we validate); SFX + motion graphics ON.
"""
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

os.environ["VIDLORE_SFX"] = "1"               # transition whoosh/boom bed ON
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"  # overlays + reveal SFX bed ON
os.environ["VIDLORE_AIMG"] = "0"              # no AI images (cheap/fast)
os.environ["WEB_IMAGE_ENGINE"] = "0"          # skip web image discovery (faster)

from vidlore.brief import Brief             # noqa: E402
from vidlore.config import load_config      # noqa: E402
from vidlore.pipeline import produce        # noqa: E402

cfg = load_config(ROOT)
brief = Brief(
    title="The Empire of Cheap: How a Retail Giant Took Over",
    prompt=(
        "A cinematic, factual business documentary that USES motion graphics to "
        "explain the numbers. (1) HOOK — open on the staggering scale. (2) GROWTH "
        "CHART — show revenue climbing from 44,000,000 dollars to 611,000,000,000 "
        "across the decades. (3) TIMELINE — the key years: 1962 founding, 1972, "
        "1991, 2005, 2018. (4) MARKET SHARE — a breakdown of its share vs rivals "
        "(percent split). (5) MAP — its expansion across regions. (6) PROCESS — "
        "the supply-chain steps that made it unbeatable. (7) KEY FIGURES — "
        "2,300,000 employees, 10,500 stores, 90 percent of households. Use clear "
        "data cards, a stat reveal, and a chart. Premium and restrained."
    ),
    duration="1-2",
    theme="modern",
    fmt="documentary",
)
out_dir = ROOT / "output" / "_audio_sd_test"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[sd] config: {cfg.describe()}", flush=True)
t0 = time.time()
res = produce(brief, cfg, out_dir, keep_work=True)   # keep stems for the probe
dt = time.time() - t0
print("\n==== AUDIO SD RENDER DONE ====", flush=True)
print(f"  video:   {res.video}", flush=True)
print(f"  seconds: {res.seconds:.1f}", flush=True)
print(f"  wall:    {dt:.1f}s", flush=True)
print(f"  exists:  {Path(res.video).exists()}", flush=True)
