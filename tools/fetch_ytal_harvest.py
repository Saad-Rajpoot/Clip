#!/usr/bin/env python3
"""Fetch official YouTube Audio Library music from a browser-harvested URL blob.

The YTAL *music* library is OAuth-gated inside studio.youtube.com, so the tracks
can't be enumerated by yt-dlp. Instead we drove the user's own logged-in Chrome
(Chrome MCP), intercepted the per-track `creator_music/get_tracks` API responses
(which carry the signed, public `googlevideo.com/videoplayback` URL), and saved
them — plus the on-screen provenance (title, artist, genre, mood, license,
attribution) — to ~/Downloads/ytal_harvest.json as:

    {"u": ["<base64 of signed url>", ...], "m": ["<base64 of meta json>", ...]}

This script decodes that, pairs each URL with its metadata by "Title - Artist",
then curls every track DIRECTLY from the signed CDN URL (no Chrome download
throttle), writing into the USE-ONLY cache as:

    vidlore/audio_library/ytal_cache/music/<id>.mp3
    vidlore/audio_library/ytal_cache/music/<id>.json   (full provenance schema)

…in the exact sidecar shape tools/merge_ytal_music.py expects, so the existing
classify + 17-niche-tag + quality-gate + organise pipeline absorbs them.

Signed URLs expire ~20-25 min after harvest, so run this PROMPTLY after harvest.
Idempotent: already-fetched ids are skipped, so re-running tops up failures.

Usage: python3 tools/fetch_ytal_harvest.py [--harvest PATH] [--workers 6]
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import date
from pathlib import Path

ROOT = Path("/Users/hussnain/Desktop/vidrush-clone")
CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache" / "music"


def b64d(s: str) -> str:
    return base64.b64decode(s).decode("utf-8", "replace")


def qparam(url: str, key: str) -> str:
    m = re.search(r"[?&]" + re.escape(key) + r"=([^&]*)", url)
    return urllib.parse.unquote(m.group(1)) if m else ""


def dur_to_sec(s: str) -> float:
    s = (s or "").strip()
    if not re.match(r"^\d+:\d{2}$", s):
        return 0.0
    mm, ss = s.split(":")
    return int(mm) * 60 + int(ss)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default=os.path.expanduser("~/Downloads/ytal_harvest.json"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(a.harvest).read_text(encoding="utf-8"))
    urls = [b64d(x) for x in data.get("u", [])]
    metas = []
    for x in data.get("m", []):
        try:
            metas.append(json.loads(b64d(x)))
        except Exception:                                          # noqa: BLE001
            pass

    # index metadata by normalized "title - artist" and by bare title
    midx: dict = {}
    for m in metas:
        t = (m.get("title") or "").strip()
        ar = (m.get("artist") or "").strip()
        if t:
            midx[(t + " - " + ar).lower()] = m
            midx.setdefault(t.lower(), m)

    # build dedup job list keyed by the CDN id
    seen, jobs = set(), []
    for url in urls:
        if "videoplayback" not in url:
            continue
        vid = qparam(url, "id") or hashlib.sha1(url.encode()).hexdigest()[:12]
        if vid in seen:
            continue
        seen.add(vid)
        ta = qparam(url, "title")  # "Title - Artist"
        meta = midx.get(ta.lower())
        if not meta and " - " in ta:
            meta = midx.get(ta.rsplit(" - ", 1)[0].lower())
        jobs.append((vid, url, ta, meta or {}))

    print(f"[fetch] {len(jobs)} unique tracks (from {len(urls)} urls, {len(metas)} meta)")

    def fetch(job):
        vid, url, ta, meta = job
        mp3 = CACHE / f"{vid}.mp3"
        if mp3.exists() and mp3.stat().st_size > 20000:
            return ("skip", ta)
        tmp = str(mp3) + ".part"
        try:
            r = subprocess.run(
                ["curl", "-sS", "-L", "--retry", "8", "--retry-delay", "2",
                 "--retry-all-errors", "--max-time", "180", "-A", "Mozilla/5.0",
                 "-o", tmp, url],
                capture_output=True)
        except Exception:                                          # noqa: BLE001
            return ("fail", ta)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 20000:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return ("fail", ta)
        os.replace(tmp, mp3)
        title = meta.get("title") or (ta.rsplit(" - ", 1)[0] if " - " in ta else ta)
        artist = meta.get("artist") or (ta.rsplit(" - ", 1)[1] if " - " in ta else "")
        attr_req = bool(meta.get("attribution_required"))
        note = (meta.get("attribution_note") or "").strip()
        rec = {
            "id": vid, "name": vid, "title": title, "artist": artist,
            "source": "youtube_al",
            "source_url": "https://studio.youtube.com/channel/UCYn15Dqi2WyULvlX7h05reQ/music",
            "license_type": ("YouTube Audio Library License"
                             + (" — attribution required" if attr_req else "")),
            "attribution_required": attr_req,
            "attribution": note if attr_req else "",
            "attribution_text": note if attr_req else "",
            "commercial_use": True,
            "download_date": date.today().isoformat(),
            "checksum_sha1": hashlib.sha1(mp3.read_bytes()).hexdigest(),
            "duration": dur_to_sec(meta.get("duration", "")),
            "measured_duration": dur_to_sec(meta.get("duration", "")),
            "genre_ytal": meta.get("genre", ""),
            "mood_ytal": meta.get("mood", ""),
            "tags": [t for t in [meta.get("genre", ""), meta.get("mood", "")] if t],
            "ingest_method": "chrome_mcp_network_capture",
        }
        (CACHE / f"{vid}.json").write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        return ("ok", ta)

    ok = skip = fail = 0
    fails = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for status, ta in ex.map(fetch, jobs):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                fails.append(ta)
    print(f"[fetch] DONE  ok={ok}  skip={skip}  fail={fail}  "
          f"total_in_cache={len(list(CACHE.glob('*.mp3')))}")
    if fails:
        print(f"[fetch] {len(fails)} failed (expired/503); re-run to retry. "
              f"sample: {[f[:28] for f in fails[:6]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
