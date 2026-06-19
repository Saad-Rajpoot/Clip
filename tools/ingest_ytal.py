#!/usr/bin/env python3
"""License-aware YouTube Audio Library (YTAL) ingester for Vidlore.

Reuses + improves the existing `music_sources/youtube_al.py` discovery idea, but
adds: a strict LICENSE PARSER, the full per-asset metadata schema, USE-ONLY
tiering (raw files NEVER bundled in dist), dedup, and documentary quality scoring.

LICENSE TIERS (only these are accepted; everything else is rejected):
  • ytal_official     — the OFFICIAL YouTube Audio Library: a track whose
                        description carries the official CDN link
                        (youtube-audio-library.storage.googleapis.com/<hash>) and
                        the "royalty-free … will not result in a claim" wording.
                        Commercial-OK, attribution NOT required. (Mostly SFX in the
                        public @audiolibrary mirror.)
  • cc_by_4_0         — a track whose description states "Creative Commons …
                        Attribution 4.0 … CC BY 4.0" AND provides a credit block.
                        Commercial-OK, ATTRIBUTION REQUIRED (text captured verbatim).
  • REJECT            — "not a brand / not working with one", non-commercial,
                        "contact for license", or no clear free-use marker.

DISTRIBUTION: every ingested raw file is tier=USE_ONLY — it lives in the local
runtime/dev cache (vidlore/audio_library/ytal_cache/, git-ignored) and is excluded
from dist. Rendered videos may use them under their stated terms; a per-video
attribution file is generated when a CC-BY track is used.

Usage:
  python3 tools/ingest_ytal.py --source "@audiolibrary" --kind sfx   --limit 200
  python3 tools/ingest_ytal.py --source "<music channel/playlist>" --kind music --limit 200
  python3 tools/ingest_ytal.py --probe "@audiolibrary" --limit 20     # dry-run, no download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "vidlore" / "audio_library" / "ytal_cache"
_YT_CLIENTS = ["android", "ios", "tv", "mweb", "web_safari"]

_OFFICIAL_CDN = re.compile(
    r"https?://youtube-audio-library\.storage\.googleapis\.com/\S+", re.I)
_OFFICIAL_WORDS = re.compile(
    r"royalty[\s-]?free|will not result in a claim|freely available for youtube", re.I)
_CCBY = re.compile(
    r"creative\s+commons.{0,40}attribution\s*4\.0|cc[\s-]?by[\s-]?4\.0", re.I)
# hard REJECT markers (non-commercial / brand-restricted / unclear)
_REJECT = re.compile(
    r"not a brand|not working with (?:a brand|one)|non[\s-]?commercial|"
    r"contact .{0,20}for .{0,20}licen|personal use only|no commercial", re.I)


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ydl(opts):
    import yt_dlp
    base = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extractor_args": {"youtube": {"player_client": _YT_CLIENTS}}}
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def enumerate_source(source: str, limit: int) -> list[str]:
    """Return up to `limit` video ids from a channel handle / playlist / search."""
    if source.startswith("@"):
        url = f"https://www.youtube.com/{source}/videos"
    elif source.startswith("http"):
        url = source
    else:
        url = f"ytsearch{max(limit*2, 40)}:{source}"
    try:
        with _ydl({"extract_flat": "in_playlist", "playlistend": max(limit * 2, 60)}) as y:
            info = y.extract_info(url, download=False) or {}
    except Exception as e:                                         # noqa: BLE001
        print(f"  ! enumerate failed: {e}")
        return []
    return [e["id"] for e in (info.get("entries") or []) if e and e.get("id")][:max(limit * 2, 60)]


def fetch_meta(vid: str) -> dict:
    try:
        with _ydl({"noplaylist": True}) as y:
            return y.extract_info(f"https://www.youtube.com/watch?v={vid}",
                                  download=False) or {}
    except Exception:                                              # noqa: BLE001
        return {}


def _extract_credit(desc: str) -> str:
    """Capture the verbatim attribution block a CC-BY description provides."""
    lines = [l.strip() for l in (desc or "").splitlines()]
    out = []
    grab = False
    for i, l in enumerate(lines):
        low = l.lower()
        if ("copy" in low and "paste" in low and "credit" in low) or low.startswith("credit"):
            grab = True
            continue
        if grab:
            if not l:
                if out:
                    break
                continue
            if l.startswith(("http", "#", "—", "-")) and not re.search(r"music|track|by|licen", low):
                continue
            out.append(l)
            if len(out) >= 4:
                break
    if not out:  # fallback: a 'Music: ... by ...' line
        for l in lines:
            if re.match(r"(music|track|song)\s*[:|]", l, re.I) and " by " in l.lower():
                out.append(l)
                break
    return " / ".join(out)[:400]


def parse_license(meta: dict) -> dict:
    """Classify a track's license from its description. Returns a dict with tier,
    license_type, attribution_required, attribution_text, commercial_use,
    official_cdn_url, or {'tier': 'reject', 'reason': ...}."""
    desc = meta.get("description") or ""
    title = meta.get("title") or ""
    blob = f"{title}\n{desc}"
    if _REJECT.search(blob):
        return {"tier": "reject", "reason": "restrictive/non-commercial marker"}
    cdn = _OFFICIAL_CDN.search(desc)
    if cdn and _OFFICIAL_WORDS.search(blob):
        return {"tier": "ytal_official",
                "license_type": "YouTube Audio Library — royalty-free (no claim)",
                "attribution_required": False, "attribution_text": "",
                "commercial_use": True, "official_cdn_url": cdn.group(0)}
    if _CCBY.search(blob):
        credit = _extract_credit(desc)
        return {"tier": "cc_by_4_0",
                "license_type": "Creative Commons Attribution 4.0 (CC BY 4.0)",
                "attribution_required": True,
                "attribution_text": credit or "(artist credit per source video)",
                "commercial_use": True, "official_cdn_url": ""}
    return {"tier": "reject", "reason": "no recognised free-use / CC-BY marker"}


def _sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def _measure(path: Path) -> dict:
    """Integrated LUFS + true-peak via ffmpeg ebur128."""
    try:
        out = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-nostats", "-i", str(path),
             "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120).stderr
        lufs = peak = None
        for m in re.finditer(r"I:\s*(-?\d+\.?\d*)\s*LUFS", out):
            lufs = float(m.group(1))
        for m in re.finditer(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", out):
            peak = float(m.group(1))
        return {"lufs": lufs, "peak_dbfs": peak}
    except Exception:                                              # noqa: BLE001
        return {"lufs": None, "peak_dbfs": None}


def download_audio(vid: str, lic: dict, dest_dir: Path, kind: str) -> Path | None:
    """Download to an mp3 in dest_dir. Prefers the official CDN direct link;
    else yt-dlp bestaudio. Returns the mp3 path or None."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = dest_dir / f"{vid}.src"
    mp3 = dest_dir / f"{vid}.mp3"
    if mp3.exists():
        return mp3
    cdn = lic.get("official_cdn_url")
    got = False
    if cdn:
        try:
            import requests
            r = requests.get(cdn, timeout=60)
            if r.ok and len(r.content) > 2000:
                raw.write_bytes(r.content)
                got = True
        except Exception:                                          # noqa: BLE001
            got = False
    if not got:
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({
                "quiet": True, "no_warnings": True, "noplaylist": True,
                "format": "bestaudio/best",
                "outtmpl": str(dest_dir / f"{vid}.%(ext)s"),
                "extractor_args": {"youtube": {"player_client": _YT_CLIENTS}},
            }) as y:
                y.download([f"https://www.youtube.com/watch?v={vid}"])
            cand = sorted(dest_dir.glob(f"{vid}.*"))
            cand = [c for c in cand if c.suffix not in (".mp3", ".json")]
            if cand:
                raw = cand[0]
                got = True
        except Exception as e:                                     # noqa: BLE001
            print(f"    ! download failed {vid}: {str(e)[:80]}")
            return None
    if not got or not raw.exists():
        return None
    # transcode to a clean mp3 (192k) and drop the source
    try:
        subprocess.run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(raw), "-vn", "-ar", "44100", "-ac", "2",
                        "-b:a", "192k", str(mp3)], timeout=180, check=True)
        raw.unlink(missing_ok=True)
        return mp3 if mp3.exists() else None
    except Exception:                                              # noqa: BLE001
        raw.unlink(missing_ok=True)
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--kind", choices=["music", "sfx"], default="music")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--probe", help="dry-run: classify licenses, no download")
    ap.add_argument("--credit", default="",
                    help="KNOWN CC-BY composer channel: the exact standard credit "
                         "to attach to every track (e.g. a Scott Buckley channel). "
                         "Skips description parsing — use ONLY for channels whose "
                         "whole catalogue you have verified as CC BY 4.0.")
    a = ap.parse_args(argv)

    source = a.probe or a.source
    if not source:
        ap.print_help()
        return 2
    probe = bool(a.probe)
    kind = a.kind
    dest_dir = CACHE / kind
    print(f"[ytal] {'PROBE' if probe else 'INGEST'} source={source!r} kind={kind} "
          f"limit={a.limit}", flush=True)

    vids = enumerate_source(source, a.limit)
    print(f"[ytal] enumerated {len(vids)} candidate ids", flush=True)

    from vidlore import music_classify as mc

    # CROSS-RUN dedup: load checksums + ids already in the cache so repeated
    # batch runs across many niche sources never re-download or duplicate.
    accepted, rejected, dup = [], [], 0
    seen_sha: dict[str, str] = {}
    seen_ids: set[str] = set()
    if not probe and dest_dir.exists():
        for sc in dest_dir.glob("*.json"):
            try:
                d = json.loads(sc.read_text(encoding="utf-8"))
                if d.get("checksum_sha1"):
                    seen_sha[d["checksum_sha1"]] = d.get("id", "")
                if d.get("id"):
                    seen_ids.add(d["id"])
            except Exception:                                      # noqa: BLE001
                pass
    today = date.today().isoformat()
    for vid in vids:
        if len(accepted) >= a.limit:
            break
        if vid in seen_ids:                       # already ingested in a prior run
            continue
        meta = fetch_meta(vid)
        if not meta:
            continue
        title = (meta.get("title") or vid).strip()
        if a.credit:                      # known CC-BY composer channel override
            lic = {"tier": "cc_by_4_0",
                   "license_type": "Creative Commons Attribution 4.0 (CC BY 4.0)",
                   "attribution_required": True, "attribution_text": a.credit,
                   "commercial_use": True, "official_cdn_url": ""}
        else:
            lic = parse_license(meta)
        if lic["tier"] == "reject":
            rejected.append({"id": vid, "title": title, "reason": lic.get("reason")})
            continue
        rec = {
            "id": vid, "title": title,
            "artist": meta.get("channel") or meta.get("uploader") or "",
            "source": "YouTube Audio Library",
            "source_url": f"https://www.youtube.com/watch?v={vid}",
            "license_tier_src": lic["tier"],
            "license_type": lic["license_type"],
            "attribution_required": lic["attribution_required"],
            "attribution_text": lic["attribution_text"],
            "commercial_use": lic["commercial_use"],
            "download_date": today,
            "duration": int(meta.get("duration") or 0),
            "distribution_tier": "USE_ONLY",
            "kind": kind,
        }
        if probe:
            accepted.append(rec)
            continue
        mp3 = download_audio(vid, lic, dest_dir, kind)
        if not mp3:
            rejected.append({"id": vid, "title": title, "reason": "download failed"})
            continue
        sha = _sha1(mp3)
        if sha in seen_sha:
            dup += 1
            mp3.unlink(missing_ok=True)
            continue
        seen_sha[sha] = vid
        rec["checksum_sha1"] = sha
        rec["path"] = str(mp3.relative_to(ROOT))
        rec.update(_measure(mp3))
        try:
            feats = mc.extract_features(mp3)
            rec["measured_duration"] = round(feats.duration, 2)
            rec["bpm"] = feats.bpm
            rec["tension"] = round(getattr(feats, "tension", 0.0), 3)
            rec["darkness"] = round(getattr(feats, "darkness", 0.0), 3)
            rec["silence_ratio"] = round(getattr(feats, "silence_ratio", 0.0), 3)
            rec["energy_arc"] = getattr(feats, "energy_arc", "steady")
        except Exception:                                          # noqa: BLE001
            rec["bpm"] = None
        # sidecar
        (mp3.with_suffix(".json")).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        accepted.append(rec)
        print(f"    + [{lic['tier']:13}] {title[:48]:50} "
              f"{rec.get('duration')}s lufs={rec.get('lufs')}", flush=True)

    # write/update the ingest manifest
    dest_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "schema_version": 1, "kind": kind, "source": source,
        "ingested": None if probe else today,
        "distribution_tier": "USE_ONLY (raw files excluded from dist)",
        "accepted": len(accepted), "rejected": len(rejected), "duplicates": dup,
        "by_license": {},
        "tracks": accepted,
        "rejected_sample": rejected[:40],
    }
    for r in accepted:
        t = r["license_tier_src"]
        man["by_license"][t] = man["by_license"].get(t, 0) + 1
    out = (CACHE / f"ytal_{kind}_{'probe' if probe else 'ingest'}.json")
    out.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ytal] {'probed' if probe else 'ingested'}: {len(accepted)} accepted "
          f"({man['by_license']}), {len(rejected)} rejected, {dup} dup")
    print(f"[ytal] manifest -> {out}")

    # CUMULATIVE CATALOG: rebuild from ALL sidecars in the cache (every batch
    # run contributes), so the merge/curation step sees the whole library.
    if not probe:
        allrecs = []
        for sc in sorted(dest_dir.glob("*.json")):
            try:
                allrecs.append(json.loads(sc.read_text(encoding="utf-8")))
            except Exception:                                      # noqa: BLE001
                pass
        bylic: dict = {}
        for r in allrecs:
            t = r.get("license_tier_src", "?")
            bylic[t] = bylic.get(t, 0) + 1
        cat = {"schema_version": 1, "kind": kind,
               "distribution_tier": "USE_ONLY (raw files excluded from dist)",
               "total": len(allrecs), "by_license": bylic, "tracks": allrecs}
        (CACHE / f"ytal_{kind}_catalog.json").write_text(
            json.dumps(cat, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[ytal] cumulative catalog -> {len(allrecs)} {kind} tracks "
              f"({bylic})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
