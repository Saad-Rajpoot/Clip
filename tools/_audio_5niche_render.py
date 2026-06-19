"""Post-fix multi-niche audio validation — 5 short docs on the v14 audio engine.
spy / true-crime / business / history / geopolitics. SFX + motion graphics ON,
AI images OFF (cheap/fast — audio is what we validate), work dirs kept for the
manual-review package. Each niche isolated so one failure doesn't block the rest.
"""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ["VIDLORE_SFX"] = "1"
os.environ["VIDLORE_MOTION_GRAPHICS"] = "1"
os.environ["VIDLORE_AIMG"] = "0"
os.environ["WEB_IMAGE_ENGINE"] = "0"

from vidlore.brief import Brief             # noqa: E402
from vidlore.config import load_config      # noqa: E402
from vidlore.pipeline import produce        # noqa: E402

cfg = load_config(ROOT)

SAMPLES = [
    ("spy", "crime",
     "The Handler: A Cold War Spy's Last Mission",
     "A cinematic intelligence/espionage documentary. HOOK on the most dangerous "
     "spy of the era. Concrete figures over b-roll: he passed 8,000 classified "
     "documents, was paid 1,200,000 dollars, operated for 9 years. TIMELINE of the "
     "operation's key years. MAP of the dead-drop sites across the city. A redacted "
     "DOCUMENT / case-file reveal. End on the betrayal. Restrained, premium."),
    ("crime", "crime",
     "Blood Money: The Rise of a Crime Empire",
     "A factual true-crime documentary. HOOK on the body count. Figures: 4,000 "
     "killings, 420,000,000 dollars a week, 15,000 kilograms smuggled. TIMELINE of "
     "the empire. MAP of the smuggling routes. A police EVIDENCE / mugshot reveal. "
     "End on the downfall. Footage-first, restrained."),
    ("business", "modern",
     "The Empire of Cheap: How a Retail Giant Took Over",
     "A business documentary that uses motion graphics. HOOK on scale. GROWTH "
     "CHART revenue 44,000,000 to 611,000,000,000. TIMELINE 1962/1972/1991/2005. "
     "MARKET-SHARE split vs rivals. MAP of expansion. KEY FIGURES 2,300,000 "
     "employees, 10,500 stores. Premium, restrained."),
    ("history", "history",
     "The Bridge That Built a Nation",
     "A cinematic history documentary. HOOK on an engineering marvel. Figures: "
     "27,000 tons of steel, 14 years, 600 workers. TIMELINE of the construction "
     "years. MAP of the route it connected. An archival DOCUMENT / blueprint "
     "reveal. End on its legacy. Reverent, premium."),
    ("geopolitics", "history",
     "The Line on the Map: How Two Nations Were Drawn",
     "A geopolitics documentary on a contested border. HOOK on the stakes. "
     "Figures: 1,200 kilometres of frontier, 3 wars, 60,000,000 people. TIMELINE "
     "of the key treaties and conflicts. MAP of the shifting border. A treaty "
     "DOCUMENT reveal. Restrained, premium."),
]

results = []
for niche, theme, title, prompt in SAMPLES:
    out_dir = ROOT / "output" / f"_sd5_{niche}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== RENDER [{niche}] =====", flush=True)
    t0 = time.time()
    try:
        brief = Brief(title=title, prompt=prompt, duration="1-2",
                      theme=theme, fmt="documentary")
        res = produce(brief, cfg, out_dir, keep_work=True)
        ok = Path(res.video).exists()
        print(f"  [{niche}] {'OK' if ok else 'MISSING'} {res.video} "
              f"({res.seconds:.0f}s, wall {time.time()-t0:.0f}s)", flush=True)
        results.append((niche, str(res.video), ok))
    except Exception:                                              # noqa: BLE001
        print(f"  [{niche}] FAILED:\n{traceback.format_exc()}", flush=True)
        results.append((niche, "", False))

print("\n===== 5-NICHE RENDER SUMMARY =====", flush=True)
for niche, video, ok in results:
    print(f"  {niche:12} {'OK ' if ok else 'FAIL'} {video}", flush=True)
