#!/usr/bin/env python3
"""Curate the ingested YTAL SFX into a documentary-useful set (no weak filler).

The official @audiolibrary feed is a broad SFX grab-bag (alarm clocks, animals,
kitchen, body sounds …). This deterministic pass keeps only SFX whose title maps
to a DOCUMENTARY family and drops the rest, then writes a clean
ytal_sfx_manifest.json (USE-ONLY; raw files never bundled). Re-runnable.

Usage:  python3 tools/curate_ytal_sfx.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache" / "sfx"

# documentary SFX families -> title keywords (first match wins)
_FAMILIES = [
    ("impact",       r"\b(impact|hit|boom|thud|slam|punch|crash|smash|bang|stomp|whack|knock)\b"),
    ("transition",   r"\b(whoosh|swoosh|swish|woosh|riser|sweep|transition|swipe|rush|air)\b"),
    ("tech",         r"\b(beep|blip|digital|computer|type|keyboard|radio|static|glitch|scan|electronic|interface|signal|data|alarm beep|sci-?fi|laser|power)\b"),
    ("paper",        r"\b(paper|page|book|write|writing|pen|pencil|stamp|tear|rip|folder|envelope|scribble)\b"),
    ("mechanical",   r"\b(click|switch|button|mechanism|lock|gear|ratchet|machine|metal|chain|wrench|spring|hinge|latch|valve|motor)\b"),
    ("ambience",     r"\b(ambien|drone|atmosphere|atmospher|room tone|hum|wind|rain|thunder|storm|fire|crackle|nature|forest|ocean|cave|underground)\b"),
    ("camera",       r"\b(camera|shutter|photo|flash|film|projector|reel|click photo)\b"),
    ("foley",        r"\b(door|glass|footstep|foot step|walk|gravel|cloth|fabric|coin|money|cash|clock tick|tick|heartbeat|breath)\b"),
]
# hard drops (never documentary-useful)
_REJECT = re.compile(
    r"\b(fart|burp|cough|sneeze|kiss|chew|slurp|gulp|cat|dog|bird|cow|duck|frog|"
    r"animal|purr|meow|bark|quack|moo|toy|cartoon|boing|squeak|fris?bee|"
    r"kitchen|toilet|flush|fridge|microwave|blender|baby|snore|whistle party|"
    r"balloon|bubble|pop cork|zipper|velcro)\b", re.I)


def _family(title: str) -> str | None:
    t = (title or "").lower()
    if _REJECT.search(t):
        return None
    for fam, pat in _FAMILIES:
        if re.search(pat, t):
            return fam
    return None  # unknown → drop (conservative: no weak filler)


def main() -> int:
    sidecars = sorted(CACHE.glob("*.json")) if CACHE.exists() else []
    kept, dropped = [], []
    by_fam: dict = {}
    for sc in sidecars:
        try:
            d = json.loads(sc.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        fam = _family(d.get("title", ""))
        if not fam:
            dropped.append(d.get("title", ""))
            continue
        d["documentary_family"] = fam
        by_fam[fam] = by_fam.get(fam, 0) + 1
        kept.append(d)

    man = {
        "schema_version": 1, "kind": "ytal_sfx",
        "distribution_tier": "USE_ONLY (official YTAL royalty-free; raw files git-ignored, excluded from dist)",
        "license_tier": "ytal_official (no attribution required, commercial-OK)",
        "total_ingested": len(sidecars),
        "documentary_useful": len(kept),
        "dropped_filler": len(dropped),
        "by_family": dict(sorted(by_fam.items(), key=lambda kv: -kv[1])),
        "tracks": sorted(kept, key=lambda t: (t.get("documentary_family", ""), t.get("title", ""))),
        "dropped_sample": dropped[:40],
    }
    (ROOT / "vidlore" / "audio_library" / "ytal_sfx_manifest.json").write_text(
        json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[sfx-curate] {len(kept)}/{len(sidecars)} documentary-useful "
          f"({len(dropped)} filler dropped)")
    print(f"[sfx-curate] families: {man['by_family']}")
    print(f"[sfx-curate] manifest -> vidlore/audio_library/ytal_sfx_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
