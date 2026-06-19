#!/usr/bin/env python3
"""Reusable post-render asset-QA pass (engine-level QA layer surface).

Runs the vidlore.asset_qa checks over a finished render's artifacts and writes
`asset_qa.json` next to the manifest, so ANY future render can be flagged for:
mismatched / pre-photographic-modern-face portraits, modern footage in a period
scene, palette-niche mismatch, bright card in a dark niche, uncertain provenance.

  python3 tools/render_asset_qa.py <run_dir> [--niche crime]

It reads script.json (era + title), motion_graphics_manifest.json (palette), and
any *.provenance.json (portraits). Confidence-low items are warnings, never hard
failures — the guards already chose safer fallbacks at render time.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vidlore import asset_qa, niche_palette, period_guard  # noqa: E402


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                              # noqa: BLE001
        return None


def run(run_dir: str, niche_hint: str = "") -> dict:
    rd = Path(run_dir)
    warnings = []
    script = _load(rd / "script.json") or {}
    scenes = script.get("scenes", [])
    title = script.get("title", "")
    blob = " ".join((s.get("narration", "") or "") for s in scenes)
    era = period_guard.detect_era(blob, title=title)

    # infer niche if not supplied: weighted-niche normaliser over title+blob
    niche = niche_hint or _infer_niche(title + " " + blob)

    man = _load(rd / "motion_graphics_manifest.json") or {}
    palettes = sorted({(e.get("palette") or e.get("inputs", {}).get("palette_name"))
                       for e in man.get("scenes", [])
                       if (e.get("palette") or e.get("inputs", {}).get("palette_name"))})
    for pal in palettes:
        warnings += asset_qa.check_palette_niche(niche, pal)

    # portrait provenance (look in run_dir + a sibling cache)
    prov_files = list(rd.glob("**/*.provenance.json"))
    for pf in prov_files:
        prov = _load(pf) or {}
        person = prov.get("person", "")
        if person:
            # use the provenance's RECORDED name_match score (not the source-type
            # string) so we don't false-flag a correctly-sourced portrait.
            warnings += asset_qa.check_portrait(
                person, prov=prov, title="",
                name_match_score=prov.get("name_match"),
                ai_generated=bool(prov.get("ai_generated")))

    # period-footage advisory (text-level): flag that the doc is period-sensitive
    if era.get("period_sensitive"):
        warnings.append({"check": "period_doc", "severity": "info",
                         "message": f"period documentary (~{era.get('year')}, "
                                    f"{era.get('label')}) — footage must stay era-appropriate",
                         "suggestion": "era-biased queries + AI-historical fallback engaged"})

    out = {"run_dir": str(rd), "title": title, "niche": niche, "era": era,
           "palettes": palettes, "warnings": warnings,
           "summary": asset_qa.summarize(warnings)}
    (rd / "asset_qa.json").write_text(json.dumps(out, indent=2))
    return out


def _infer_niche(text: str) -> str:
    t = text.lower()
    table = [("crime", ("murder", "gangster", "mob", "capone", "killer", "heist",
                        "racketeer", "prohibition", "fbi")),
             ("spy", ("spy", "espionage", "mossad", "cia", "kgb", "covert",
                      "intelligence", "agent", "damascus")),
             ("geopolitics", ("missile", "cold war", "nuclear", "treaty",
                              "superpower", "blockade", "crisis")),
             ("history", ("1812", "napoleon", "empire", "ancient", "medieval",
                          "war", "century", "campaign", "revolution")),
             ("business", ("steel", "oil", "fortune", "company", "empire of",
                           "industrialist", "market", "profit", "wealth"))]
    best, score = "_default", 0
    for niche, words in table:
        c = sum(t.count(w) for w in words)
        if c > score:
            best, score = niche, c
    return niche_palette.normalize_niche(best)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    niche = ""
    for i, a in enumerate(sys.argv):
        if a == "--niche" and i + 1 < len(sys.argv):
            niche = sys.argv[i + 1]
    if not args:
        print("usage: render_asset_qa.py <run_dir> [--niche X]")
        sys.exit(1)
    res = run(args[0], niche)
    print(f"niche={res['niche']} era={res['era']['label']} "
          f"palettes={res['palettes']}")
    print(f"warnings: {res['summary']}")
    for w in res["warnings"]:
        print(f"  [{w['severity']}] {w['check']}: {w['message']}")


if __name__ == "__main__":
    main()
