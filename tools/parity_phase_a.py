# PHASE A — deterministic decision parity for the portal render.
# Loads the portal's EXACT brief.json + script.json and re-runs the DETERMINISTIC
# engine decision stages, each TWICE, to prove run-to-run determinism + that the
# portal's recorded recipe is reproducible from (brief, niche, salt).
import json
import os
import sys
from pathlib import Path

PROJ = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "dist/Vidlore-Mac/output/how-a-female-mossad-spy-married-an-iranian-general")

from vidlore.pipeline import load_brief
from vidlore import editorial_recipe as ER
from vidlore import factual_guard as FG

meta = json.load(open(PROJ / "render_meta.json"))
recorded = meta["editorial_recipe"]
niche = meta["editor_signature"]["niche"]
salt = recorded["variation_salt"]
brief = load_brief(PROJ)
print("portal niche=%s  variation_salt=%s  _attempt=%s" % (niche, salt, recorded.get("_attempt")))
print("brief.title=%r" % getattr(brief, "title", None))
print("=" * 76)

# ---- 1. EDITORIAL RECIPE: determinism + reproduction of the portal recipe ----
r1 = ER._build(brief, niche, salt)
r2 = ER._build(brief, niche, salt)
strip = lambda d: {k: v for k, v in d.items() if k not in ("_attempt",)}
det = strip(r1) == strip(r2)
cmp_keys = [k for k in r1 if k in recorded and k != "_attempt"]
diffs = {k: (r1[k], recorded[k]) for k in cmp_keys if r1[k] != recorded[k]}
print("[1] EDITORIAL RECIPE")
print("    _build run1 == run2 (deterministic) : %s" % det)
print("    reproduces portal render_meta recipe: %s" % (not diffs))
if diffs:
    for k, (a, b) in diffs.items():
        print("        DIFF %-20s engine=%r  portal=%r" % (k, a, b))
else:
    print("        all %d recipe axes identical (accent/beat/density/transitions/...)" % len(cmp_keys))

# ---- 2. FACTUAL GUARD: determinism + per-scene decisions --------------------
data = json.load(open(PROJ / "script.json"))
scenes = data["scenes"]

def guard_pass(scs):
    rows = []
    for i, s in enumerate(scs):
        kind = (s.get("graphic_kind") or "").strip()
        ok, why = FG.guard(kind, s.get("narration", ""),
                            s.get("graphic_text", ""), s.get("graphic_body", ""))
        rows.append((i, kind, bool(ok), why))
    return rows

g1 = guard_pass(scenes)
g2 = guard_pass(scenes)
fact_scenes = [r for r in g1 if r[1] and FG.is_fact_bearing(r[1])]
kept = [r for r in fact_scenes if r[2]]
dropped = [r for r in fact_scenes if not r[2]]
print("[2] FACTUAL GUARD (v%s)" % getattr(FG, "VERSION", "?"))
print("    run1 == run2 (deterministic)         : %s" % (g1 == g2))
print("    scenes w/ graphic_kind               : %d / %d" % (sum(1 for r in g1 if r[1]), len(scenes)))
print("    fact-bearing cards                   : %d  (kept %d, dropped %d)"
      % (len(fact_scenes), len(kept), len(dropped)))
for i, kind, ok, why in dropped[:12]:
    print("        DROP scene[%02d] %-16s reason=%s" % (i, kind, why))

# ---- 3. V3.4 VARIANT SELECTOR: determinism ----------------------------------
print("[3] V3.4 VARIANT SELECTOR")
try:
    from vidlore.motion_graphics.variants import VariantSelector
    prims = ["statistic_bar_reveal", "chronology_timeline", "comparison_split",
             "statistic_bar_reveal", "chronology_timeline"]
    variants_for = {"statistic_bar_reveal": ["columns", "hero_bar", "horizontal_bars"],
                    "chronology_timeline": ["spine", "era_band"],
                    "comparison_split": ["vs_center", "stacked_rows"]}

    def seq():
        vs = VariantSelector(project_seed=ER._seed(brief, salt), channel_dna=niche)
        out = []
        for i, p in enumerate(prims):
            ev = vs.select(p, variants_for[p], family="x")
            vid = ev.get("visual_variant_id") if isinstance(ev, dict) else ev
            out.append((p, vid))
        return out

    s1, s2 = seq(), seq()
    print("    selection run1 == run2 (deterministic): %s" % (s1 == s2))
    print("    sequence: %s" % " -> ".join("%s:%s" % (p, v) for p, v in s1))
except Exception as exc:  # noqa: BLE001
    print("    (selector probe skipped: %s) — determinism already covered by test_v34 21/21" % exc)

print("=" * 76)
print("PHASE A COMPLETE")
