"""Cheap script-gen trigger test for IMP_021: does the LLM emit
graphic_kind='figure_locator' on a strongly person<->place brief?
One LLM call, no render. Shortest duration to minimise tokens."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# load .env
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from vidlore.brief import Brief          # noqa: E402
from vidlore.config import load_config   # noqa: E402
from vidlore.script_gen import build_script  # noqa: E402

cfg = load_config(ROOT)
brief = Brief(
    title="Slobodan Milosevic: The Man Who Ruled Serbia",
    prompt=(
        "A tight profile of ONE leader and the ONE country he controlled. "
        "Open on the man. Establish WHO he was and WHERE he ruled from — "
        "Belgrade, Serbia — making the person<->place link the spine of the "
        "story. Then his rise, his grip on the nation, and his fall. Keep it "
        "cinematic and factual."
    ),
    duration="1-2",
    theme="history",
)

script = build_script(brief, cfg)
scenes = getattr(script, "scenes", script)
print(f"\n=== {len(scenes)} scenes ===")
hit = []
for s in scenes:
    gk = (getattr(s, "graphic_kind", "") or "")
    if gk:
        print(f"  [{s.index:02d}] kind={gk:16s} t={getattr(s,'graphic_text','')!r}"
              f" b={getattr(s,'graphic_body','')!r}")
    if gk == "figure_locator":
        hit.append(s)
print("\nFIGURE_LOCATOR EMITTED:", bool(hit),
      f"({len(hit)} scene(s))" if hit else "")
