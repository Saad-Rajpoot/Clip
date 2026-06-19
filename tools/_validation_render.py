"""Cumulative validation render — exercises all 24 improvements in one doc.
Crime/biography brief chosen to trigger: NIM (crime niche), figure_locator
(Escobar<->Medellin), floating_stat (comma-grouped figures), maps, document
highlights, photo callouts, archival 4:3, push-in, room tone, silence pockets,
tension cadence, breathing room. AI images + Pexels stock + web images (light).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# load .env
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# exercise the web image discovery engine, lightly (budget-safe)
os.environ.setdefault("WEB_IMAGE_ENGINE", "1")
os.environ.setdefault("WEB_IMAGE_MIX", "light")
# keep production sound defaults (SFX off — user preference); music/atmos on

from vidlore.brief import Brief             # noqa: E402
from vidlore.config import load_config      # noqa: E402
from vidlore.pipeline import produce        # noqa: E402

cfg = load_config(ROOT)
brief = Brief(
    title="Pablo Escobar: The Rise and Fall of the King of Cocaine",
    prompt=(
        "A cinematic, factual crime documentary on Pablo Escobar. PILLARS: "
        "(1) HOOK — open on the peak of his power, the most feared man in the "
        "world. (2) PLACE — establish that he ruled from Medellin, Colombia, "
        "the city and country he controlled (person tied to a place). "
        "(3) SCALE — use concrete comma-grouped figures over b-roll: he earned "
        "an estimated 420,000,000 dollars a week, smuggled 15,000 kilograms of "
        "cocaine, and his violence claimed more than 4,000 lives. (4) EVIDENCE "
        "— cite DEA reports and court records; show the geography of his "
        "smuggling routes from Colombia to Miami; reference 1980s archival "
        "footage and photographs of the cartel. End on his 1993 downfall. "
        "Keep it footage-first, restrained, and premium — no cheap infographic "
        "cards."
    ),
    duration="6-8",            # ~7 min — medium length, exercises many scenes
    theme="crime",
    fmt="documentary",
)

out_dir = ROOT / "output" / "_cumulative_validation"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[validate] config: {cfg.describe()}", flush=True)
print(f"[validate] WEB_IMAGE_ENGINE={os.environ.get('WEB_IMAGE_ENGINE')} "
      f"mix={os.environ.get('WEB_IMAGE_MIX')}", flush=True)
t0 = time.time()
res = produce(brief, cfg, out_dir)
dt = time.time() - t0
print("\n==== VALIDATION RENDER DONE ====", flush=True)
print(f"  video:    {res.video}", flush=True)
print(f"  seconds:  {res.seconds:.1f}", flush=True)
print(f"  wallclock:{dt:.1f}s", flush=True)
print(f"  thumb:    {res.thumbnail}", flush=True)
print(f"  srt:      {res.srt}", flush=True)
print(f"  exists:   {Path(res.video).exists()} "
      f"({Path(res.video).stat().st_size/1e6:.1f} MB)" if Path(res.video).exists()
      else "  MISSING", flush=True)
