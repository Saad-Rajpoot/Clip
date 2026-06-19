#!/usr/bin/env python3
"""Recover the EXACT attribution text for ingested CC-BY YTAL music tracks.

License compliance is non-negotiable: a CC BY 4.0 track may only be used when its
verbatim credit is recorded. The first ingest stored attribution_required=true but
failed to capture the credit text. This pass recovers it:

  1. KNOWN CC-BY COMPOSERS (Scott Buckley, Alexander Nakarada, Savfk, Kevin
     MacLeod, Audionautix, GoSoundtrack …) → their STANDARD published CC-BY
     credit, derived from the artist named in the title (no network needed).
  2. UNKNOWN tracks → re-fetch the description and extract the credit block /
     "Music by …" line.
  3. Tracks where NO exact credit can be recovered → attribution left empty;
     merge_ytal_music then DROPS them (incomplete attribution = excluded).

Updates each sidecar's `attribution` in place. Usage:
  python3 tools/recover_ytal_credits.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache" / "music"
_YT_CLIENTS = ["android", "ios", "tv", "mweb", "web_safari"]

# Known CC-BY composers -> (title-match regex, exact standard CC-BY credit).
_KNOWN = [
    (re.compile(r"scott\s*buckley", re.I),
     "Music by Scott Buckley – www.scottbuckley.com.au – Licensed under Creative Commons: CC BY 4.0"),
    (re.compile(r"alexander\s*nakarada|serpent\s*sound", re.I),
     "Music by Alexander Nakarada (www.serpentsoundstudios.com) – Licensed under Creative Commons: CC BY 4.0"),
    (re.compile(r"\bsavfk\b", re.I),
     "Music by Savfk (www.youtube.com/savfkmusic) – Licensed under Creative Commons: CC BY 4.0"),
    (re.compile(r"audionautix", re.I),
     "Music by Audionautix (audionautix.com) – Licensed under Creative Commons: CC BY 4.0"),
    (re.compile(r"kevin\s*macleod|incompetech", re.I),
     "Music by Kevin MacLeod (incompetech.com) – Licensed under Creative Commons: CC BY 4.0"),
    (re.compile(r"go\s*soundtrack", re.I),
     "Music by GoSoundtrack (www.gosoundtrack.com) – Licensed under Creative Commons: CC BY 4.0"),
]


def _ydl_desc(vid: str) -> str:
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                "noplaylist": True,
                "extractor_args": {"youtube": {"player_client": _YT_CLIENTS}}}) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}",
                                  download=False) or {}
        return info.get("description") or ""
    except Exception:                                              # noqa: BLE001
        return ""


def _extract_from_desc(desc: str) -> str:
    lines = [l.strip() for l in (desc or "").splitlines()]
    # "Music by ... CC BY ..." single line
    for l in lines:
        if re.search(r"\bmusic by\b|\btrack[:\s]", l, re.I) and \
           re.search(r"cc[\s-]?by|creative commons|www\.|http", l, re.I):
            return l[:300]
    # credit block after a "copy/paste credits" marker
    grab, out = False, []
    for l in lines:
        low = l.lower()
        if ("copy" in low and "paste" in low and "credit" in low) or low.startswith("credit"):
            grab = True
            continue
        if grab and l:
            out.append(l)
            if len(out) >= 3 or re.search(r"cc[\s-]?by|creative commons", l, re.I):
                break
        elif grab and out:
            break
    return " / ".join(out)[:300]


def main() -> int:
    sidecars = sorted(glob.glob(str(CACHE / "*" / "*.json")))
    known = refetched = dropped = 0
    for j in sidecars:
        p = Path(j)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if (d.get("attribution") or "").strip():
            continue
        title = d.get("title", "")
        credit = ""
        for pat, std in _KNOWN:
            if pat.search(title):
                credit = std
                known += 1
                break
        if not credit:
            desc = _ydl_desc(d.get("id", ""))
            credit = _extract_from_desc(desc)
            if credit:
                refetched += 1
        if credit:
            d["attribution"] = credit
            d["attribution_text"] = credit
            p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        else:
            dropped += 1
            print(f"  ✗ no credit recoverable: {title[:54]}")
    print(f"\n[credits] recovered {known} (known composer) + {refetched} (re-fetched) "
          f"· {dropped} un-attributable (will be dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
