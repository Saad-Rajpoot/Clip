#!/usr/bin/env python3
"""Ingest OFFICIAL YouTube Audio Library music from a LOCAL FOLDER you downloaded.

This is the safe path for the official YTAL "No Attribution Required" music (your
preference #1), which lives behind the YouTube Studio login. You perform ONE safe
local step — no credentials are ever shared with me:

  1. Open YouTube Studio → Audio Library → Music tab.
  2. Filter by mood / genre (e.g. Dramatic, Dark, Cinematic) and Attribution =
     "Not required" (preferred) — the UI shows each track's license + any credit.
  3. Click download on the tracks you want; save them into one folder.
  4. Run:   python3 tools/ingest_local_music.py --folder /path/to/that/folder
            python3 tools/merge_ytal_music.py            # organise + classify

Every file is analysed (duration, LUFS, peak, BPM, features), checksummed, deduped,
and written to the USE-ONLY cache with full provenance. Default license is the
official YTAL no-attribution terms; pass --attribution "<credit>" for the rare
attribution-required track (the UI tells you which).

Usage:
  python3 tools/ingest_local_music.py --folder PATH
       [--license "..."] [--attribution "exact credit text"] [--artist "..."]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache" / "music"
_AUDIO = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus"}


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--license",
                    default="YouTube Audio Library — royalty-free (no claim)")
    ap.add_argument("--attribution", default="",
                    help="exact credit text if the track requires attribution")
    ap.add_argument("--artist", default="")
    a = ap.parse_args(argv)
    src = Path(a.folder).expanduser()
    if not src.exists():
        print(f"folder not found: {src}")
        return 2

    from vidlore import music_classify as mc
    from tools.ingest_ytal import _measure  # reuse LUFS/peak measurement

    CACHE.mkdir(parents=True, exist_ok=True)
    seen = {}
    for sc in CACHE.rglob("*.json"):
        try:
            d = json.loads(sc.read_text(encoding="utf-8"))
            if d.get("checksum_sha1"):
                seen[d["checksum_sha1"]] = d.get("id")
        except Exception:                                          # noqa: BLE001
            pass

    files = [p for p in src.rglob("*") if p.suffix.lower() in _AUDIO]
    today = date.today().isoformat()
    added = dup = 0
    for p in sorted(files):
        sha = _sha1(p)
        if sha in seen:
            dup += 1
            continue
        vid = "local_" + sha[:12]
        dst = CACHE / f"{vid}.mp3"
        try:
            if p.suffix.lower() == ".mp3":
                shutil.copy2(p, dst)
            else:
                import subprocess
                import imageio_ffmpeg
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                                "-hide_banner", "-loglevel", "error", "-i", str(p),
                                "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                                str(dst)], check=True, timeout=180)
        except Exception as e:                                     # noqa: BLE001
            print(f"  ! skip {p.name}: {e}")
            continue
        rec = {
            "id": vid, "title": p.stem, "artist": a.artist,
            "source": "YouTube Audio Library", "source_url": "studio.youtube.com/audiolibrary",
            "license_tier_src": "cc_by_4_0" if a.attribution else "ytal_official",
            "license_type": a.license,
            "attribution_required": bool(a.attribution),
            "attribution_text": a.attribution,
            "commercial_use": True, "download_date": today,
            "distribution_tier": "USE_ONLY", "kind": "music",
            "checksum_sha1": sha, "path": str(dst.relative_to(ROOT)),
        }
        rec.update(_measure(dst))
        try:
            f = mc.extract_features(dst)
            rec.update({"measured_duration": round(f.duration, 2), "bpm": f.bpm,
                        "tension": round(getattr(f, "tension", 0.0), 3),
                        "darkness": round(getattr(f, "darkness", 0.0), 3),
                        "silence_ratio": round(getattr(f, "silence_ratio", 0.0), 3),
                        "energy_arc": getattr(f, "energy_arc", "steady")})
        except Exception:                                          # noqa: BLE001
            rec["bpm"] = None
        dst.with_suffix(".json").write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        seen[sha] = vid
        added += 1
        print(f"  + {p.name[:50]:52} lufs={rec.get('lufs')}")
    print(f"\n[local] ingested {added} tracks ({dup} dup) into {CACHE}")
    print("[local] next: python3 tools/merge_ytal_music.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
