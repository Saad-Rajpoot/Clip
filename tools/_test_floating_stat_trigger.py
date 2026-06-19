"""Cheap content-trigger check for IMP_023: does a real economics-stat
script produce footage-only scenes whose narration names a comma-grouped
figure (the floating_stat signal)? One LLM call, no render."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
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
    title="The Numbers Behind the Collapse",
    prompt=("A data-driven economics explainer. Use several concrete, "
            "comma-grouped figures over b-roll: jobs lost, dollars wiped out, "
            "people affected (e.g. 250,000 jobs; 1,200,000 homes). Keep it "
            "footage-first and factual."),
    duration="1-2", theme="modern",
)
script = build_script(brief, cfg)
scenes = getattr(script, "scenes", script)
NUM = re.compile(r"\d{1,3}(?:,\d{3})+")
print(f"\n=== {len(scenes)} scenes ===")
would_fire = 0
for s in scenes:
    gk = (getattr(s, "graphic_kind", "") or "")
    figs = NUM.findall(getattr(s, "narration", "") or "")
    flag = ""
    if not gk and figs:
        flag = "  <-- footage-only + comma figure (floating_stat candidate)"
        would_fire += 1
    print(f"  [{s.index:02d}] kind={gk or '-':14s} figs={figs} {flag}")
print(f"\nfootage-only comma-grouped numeric scenes: {would_fire}")
