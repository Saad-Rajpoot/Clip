#!/usr/bin/env python3
"""V1.1 PRIORITY 1 — validate famous-person portrait sourcing.

Calls the real _real_person_image() for historical (B&W-era) + modern figures,
checks: a REAL portrait was fetched (not AI), name-matched, B&W/sepia accepted,
provenance recorded. Saves each portrait + provenance + a tolerant-validator
verdict. Free (Wikipedia/Commons APIs only)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore import footage as F

OUT = ROOT / "research/motion_graphics_qa/portrait_sourcing_v1_1"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

# (name, role, era) — 6 historical (B&W/sepia) + modern regression
PEOPLE = [
    ("John D. Rockefeller", "founder of Standard Oil", "historical"),
    ("Nikola Tesla", "inventor", "historical"),
    ("Napoleon Hill", "author", "historical"),
    ("Winston Churchill", "prime minister", "historical"),
    ("J. P. Morgan", "banker", "historical"),
    ("Henry Ford", "industrialist", "historical"),
    # modern regression — existing lookup must still work
    ("Barack Obama", "president", "modern"),
    ("Elon Musk", "ceo", "modern"),
]


def main():
    results = []
    for name, role, era in PEOPLE:
        dest = OUT / (name.replace(" ", "_").replace(".", "") + ".jpg")
        F._PORTRAIT_PROVENANCE.pop(name, None)
        path = None
        try:
            path = F._real_person_image(name, role, dest, cache_dir=CACHE)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {e}")
        prov = F._PORTRAIT_PROVENANCE.get(name, {})
        rec = {"name": name, "role": role, "era": era,
               "real_portrait": bool(path), "saved": dest.exists(),
               "source": prov.get("source"), "domain": prov.get("domain"),
               "name_match": prov.get("name_match"),
               "validator": prov.get("validator"),
               "license": prov.get("license"),
               "public_domain": prov.get("public_domain"),
               "source_url": prov.get("source_url"),
               "fallback_reason": prov.get("fallback_reason")}
        if dest.exists():
            try:
                from PIL import Image
                im = Image.open(dest)
                rec["dims"] = list(im.size)
                # report grayscale-ness (was this accepted as B&W?)
                import numpy as np
                a = np.asarray(im.convert("RGB").resize((48, 48)), dtype="float32")
                mx, mn = a.max(-1), a.min(-1)
                rec["mean_saturation"] = round(float(((mx - mn) / (mx + 1e-3)).mean()), 3)
            except Exception:  # noqa: BLE001
                pass
        results.append(rec)
        ok = "✅REAL" if path else "⚠️AI-fallback"
        print(f"  {ok:14s} {name:24s} src={rec['source']} "
              f"nm={rec['name_match']} val={rec['validator']} "
              f"sat={rec.get('mean_saturation')} pd={rec['public_domain']}")

    (OUT / "portrait_validation_report.json").write_text(
        json.dumps({"results": results}, indent=2), encoding="utf-8")
    real = sum(1 for r in results if r["real_portrait"])
    hist_real = sum(1 for r in results if r["real_portrait"] and r["era"] == "historical")
    print(f"\nREAL portraits: {real}/{len(results)} "
          f"(historical {hist_real}/6) → portrait_validation_report.json")


if __name__ == "__main__":
    main()
