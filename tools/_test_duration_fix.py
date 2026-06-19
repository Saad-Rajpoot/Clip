"""IMP_026 validation — script-gen only (no render). Build the same Escobar
'6-8' brief and verify the script now hits the word/scene/duration targets."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for ln in (ROOT / ".env").read_text().splitlines():
    ln = ln.strip()
    if ln and not ln.startswith("#") and "=" in ln:
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from vidlore.brief import Brief, WORDS_PER_SECOND   # noqa: E402
from vidlore.config import load_config              # noqa: E402
from vidlore.script_gen import build_script         # noqa: E402

cfg = load_config(ROOT)
brief = Brief(
    title="Pablo Escobar: The Rise and Fall of the King of Cocaine",
    prompt=("A cinematic, factual crime documentary on Pablo Escobar — his "
            "rise from Medellin, the cartel empire, the violence, the money, "
            "and his 1993 downfall. Footage-first, restrained, premium."),
    duration="6-8", theme="crime", fmt="documentary",
)
script = build_script(brief, cfg)
scenes = getattr(script, "scenes", script)
words = sum(len((s.narration or "").split()) for s in scenes)
mins = words / (WORDS_PER_SECOND * 60.0)
lo_w, hi_w = 950, 1300
lo_m, hi_m = brief.target_minutes
sc_lo, sc_hi = brief.target_scenes
print(f"\n=== IMP_026 RESULT ===")
print(f"  scenes      : {len(scenes)}  (target {sc_lo}-{sc_hi})")
print(f"  total words : {words}  (target ~{brief.target_words}, band {lo_w}-{hi_w})")
print(f"  est minutes : {mins:.1f}  (requested {lo_m:.0f}-{hi_m:.0f})")
# crude repetition check: duplicate narration sentences
narrs = [(s.narration or "").strip().lower() for s in scenes]
dups = len(narrs) - len(set(narrs))
print(f"  dup scenes  : {dups}")
ok_w = lo_w <= words <= hi_w + 200
ok_sc = sc_lo <= len(scenes) <= sc_hi + 5
ok_m = (lo_m - 1) <= mins <= (hi_m + 2)
print(f"\n  word target {'PASS' if ok_w else 'CHECK'} | "
      f"scene target {'PASS' if ok_sc else 'CHECK'} | "
      f"duration {'PASS' if ok_m else 'CHECK'} | dups {'OK' if dups == 0 else 'WARN'}")
print("\n  first 3 + last 2 narrations:")
for s in scenes[:3] + scenes[-2:]:
    print(f"   [{s.index:02d}] ({len((s.narration or '').split())}w) {(s.narration or '')[:90]}")
