"""Music library orchestrator -- pulls + normalizes + classifies music.

Three running modes:

    # 1. FULLY AUTOMATIC (Stage 1+2)
    #    Crawl every trusted royalty-free source, classify with audio
    #    analysis, drop into the right category folder, normalize loudness,
    #    write sidecar JSON + LICENSES.md, refresh _index.json.
    python -m vidlore.music_extract --auto

    # 2. MANUAL CURATION  (legacy / power-user)
    #    Process every entry listed in vidlore/assets/music/_sources.yaml.
    python -m vidlore.music_extract --reindex

    # 3. AD-HOC SEARCH
    python -m vidlore.music_extract \
        --search "no copyright dark documentary" \
        --category dark_investigation --max 5

Common flags:
    --category <name>   only ingest tracks classified into this bucket
    --max <int>         cap candidates per source / search (default 8)
    --per-cat-cap <int> cap how many NEW tracks per category per run (default 8)
    --min-sec / --max-sec   duration window for cue suitability (45..420 s)
    --dry-run           list what would be fetched, fetch nothing
    --reindex           refresh musiclib _index.json on exit
    --sources <path>    override _sources.yaml path
    --only <a,b,c>      restrict --auto to these source names

USAGE HISTORY  (Stage 4 foundation)
The orchestrator maintains ``vidlore/assets/music/_history.json`` with a
running fingerprint set, last-fetched date per source, and a per-track
content hash so the same audio file isn't ingested twice even across
sources (a frequent occurrence -- e.g. Pixabay and a YouTube mirror often
host the same track).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .ffmpeg_tool import ffmpeg_exe
from .musiclib import CATEGORIES, library_root, scan as scan_library
from .music_classify import (Classification, classify_title,
                             classify as classify_track)
from .music_quality import doc_quality

# yt_dlp player-client rotation -- mirrors footage.py to dodge PO-token 403s.
_YT_CLIENTS = ("android", "ios", "tv", "mweb", "web_safari")

# How many bytes of the normalized MP3 we hash for cross-source dedup.
# 256 KB is enough to differentiate distinct music while staying cheap.
_FP_BYTES = 256 * 1024


# --------------------------------------------------------------------------- #
# Public re-export so other modules can keep calling auto_classify(title)
# --------------------------------------------------------------------------- #
def auto_classify(title: str, fallback: str = "neutral") -> str:
    return classify_title(title, fallback=fallback)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _slug(name: str, n: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s[:n] or "track"


def _existing_slugs(category_dir: Path) -> set[str]:
    return {p.stem for p in category_dir.glob("*.mp3")}


def _file_fingerprint(path: Path) -> str:
    """Cheap content fingerprint: sha256 of first _FP_BYTES of the file.
    Catches exact re-uploads across sources.  A real chromaprint hash would
    catch re-encodes too -- that lands in Stage 4 when ``fpcalc`` is added."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(_FP_BYTES))
        return h.hexdigest()
    except Exception:                                       # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# History (Stage 4 foundation)
# --------------------------------------------------------------------------- #
class History:
    """``_history.json`` -- persistent state across runs."""
    def __init__(self, root: Path) -> None:
        self.path = root / "_history.json"
        self.data: dict = {
            "fingerprints": {},     # sha256-prefix -> {slug, category, source}
            "urls": {},             # canonical URL -> slug
            "yt_ids": {},           # YouTube id -> slug
            "sources_last_run": {}, # name -> iso date
            "usage": {},            # slug -> play count across renders (Stage 4)
        }
        if self.path.exists():
            try:
                self.data.update(json.loads(
                    self.path.read_text(encoding="utf-8")))
            except Exception:                              # noqa: BLE001
                pass

    def has_url(self, url: str) -> bool:
        return bool(url) and url in self.data.get("urls", {})

    def has_yt_id(self, yt_id: str | None) -> bool:
        return bool(yt_id) and yt_id in self.data.get("yt_ids", {})

    def has_fp(self, fp: str) -> bool:
        return bool(fp) and fp in self.data.get("fingerprints", {})

    def record(self, *, slug: str, category: str, source: str,
               url: str = "", yt_id: str | None = None,
               fp: str = "") -> None:
        if url:
            self.data["urls"][url] = slug
        if yt_id:
            self.data["yt_ids"][yt_id] = slug
        if fp:
            self.data["fingerprints"][fp] = {
                "slug": slug, "category": category, "source": source}

    def mark_source_run(self, name: str) -> None:
        self.data["sources_last_run"][name] = time.strftime("%Y-%m-%d")

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=1),
                                 encoding="utf-8")
        except Exception as e:                              # noqa: BLE001
            print(f"  ! history save failed: {e}")


# --------------------------------------------------------------------------- #
# Download primitives
# --------------------------------------------------------------------------- #
def _download_direct(url: str, dst: Path, timeout: int = 60) -> Path | None:
    """Stream a direct audio URL to disk.  Returns the file path or None."""
    try:
        import requests
    except Exception:                                       # noqa: BLE001
        return None
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/118.0 Safari/537.36"),
        "Referer": "https://www.google.com/",
    }
    try:
        with requests.get(url, headers=headers, stream=True,
                          timeout=timeout) as r:
            r.raise_for_status()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    if chunk:
                        f.write(chunk)
        return dst if (dst.exists() and dst.stat().st_size > 50_000) else None
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! direct fetch failed ({url}): {e}")
        return None


def _yt_list(url: str, max_items: int) -> list[dict]:
    """Flat-extract a YouTube URL / search / playlist into entry dicts."""
    try:
        import yt_dlp
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! yt_dlp import failed: {e}")
        return []
    opts = {
        "quiet": True, "no_warnings": True, "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": max_items if max_items else None,
        "extractor_args": {"youtube": {
            "player_client": list(_YT_CLIENTS)}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! list failed for {url}: {e}")
        return []
    if not info:
        return []
    entries = ([info] if info.get("_type") in (None, "video")
               else list(info.get("entries", []) or []))
    out: list[dict] = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id") or ""
        out.append({
            "id": vid,
            "title": (e.get("title") or vid).strip(),
            "channel": (e.get("channel") or e.get("uploader") or ""),
            "duration": int(e.get("duration") or 0),
            "url": (e.get("webpage_url")
                    or f"https://www.youtube.com/watch?v={vid}"),
        })
    return out[:max_items] if max_items else out


def _yt_download_audio(vid: str, out_template: Path) -> Path | None:
    """Download bestaudio for ``vid`` to a file beside ``out_template``."""
    try:
        import yt_dlp
    except Exception:                                       # noqa: BLE001
        return None
    out_template.parent.mkdir(parents=True, exist_ok=True)
    for client in _YT_CLIENTS:
        for f in out_template.parent.glob(out_template.stem + ".*"):
            f.unlink(missing_ok=True)
        opts = {
            "quiet": True, "no_warnings": True, "noprogress": True,
            "noplaylist": True, "retries": 2, "socket_timeout": 30,
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": str(out_template.with_suffix("")) + ".%(ext)s",
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download(
                    [f"https://www.youtube.com/watch?v={vid}"])
        except Exception:                                   # noqa: BLE001
            continue
        for cand in out_template.parent.glob(
                out_template.stem + ".*"):
            if (cand.suffix.lower() in (".m4a", ".webm", ".opus",
                                        ".mp3", ".aac", ".ogg")
                    and cand.stat().st_size > 50_000):
                return cand
    return None


# --------------------------------------------------------------------------- #
# ffmpeg: transcode + EBU R128 loudnorm
# --------------------------------------------------------------------------- #
def _to_normalized_mp3(src: Path, dst: Path, target_lufs: float = -23.0,
                       fade_ms: int = 800) -> bool:
    if dst.exists():
        return True
    filt = (f"loudnorm=I={target_lufs}:TP=-2.0:LRA=11,"
            f"afade=in:st=0:d={fade_ms / 1000:.2f}")
    args = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "2", "-ar", "44100",
        "-af", filt,
        "-c:a", "libmp3lame", "-q:a", "4",
        str(dst),
    ]
    try:
        r = subprocess.run(args, capture_output=True, text=True,
                           timeout=240)
        return (r.returncode == 0 and dst.exists()
                and dst.stat().st_size > 10_000)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! ffmpeg failed: {e}")
        return False


def _probe_dur(path: Path) -> float:
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
        if m:
            return (int(m.group(1)) * 3600
                    + int(m.group(2)) * 60
                    + float(m.group(3)))
    except Exception:                                       # noqa: BLE001
        pass
    return 0.0


def _within_duration(dur: float, min_s: int, max_s: int) -> bool:
    return min_s <= dur <= max_s


# --------------------------------------------------------------------------- #
# LICENSES.md updater
# --------------------------------------------------------------------------- #
def _add_to_licenses(root: Path, entry_line: str) -> None:
    lic = root / "LICENSES.md"
    header = "# Music Library - Track Sources & Licenses\n\n"
    body = lic.read_text(encoding="utf-8") if lic.exists() else header
    if entry_line not in body:
        lic.write_text(body + entry_line + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Manual _sources.yaml (legacy)
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    url: str
    category: str        # category name or "auto"
    license: str = ""
    attribution: str = ""
    max: int = 8


def _load_sources(path: Path) -> list[Source]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! cannot parse {path}: {e}")
        return []
    out: list[Source] = []
    for entry in data.get("sources", []) or []:
        try:
            cat = (entry.get("category") or "auto").strip().lower()
            out.append(Source(
                url=entry["url"].strip(),
                category=cat,
                license=(entry.get("license") or "").strip(),
                attribution=(entry.get("attribution") or "").strip(),
                max=int(entry.get("max") or 8),
            ))
        except Exception:                                   # noqa: BLE001
            continue
    return out


# --------------------------------------------------------------------------- #
# Track ingest -- the path that downloads, normalizes, classifies, lands
# --------------------------------------------------------------------------- #
def _ingest_track(*, title: str, source: str, page_url: str,
                  download_url: str | None, yt_id: str | None,
                  channel: str, license: str, attribution: str,
                  category_hint: str, tags: list[str],
                  root: Path, work: Path,
                  history: History, per_cat_count: dict[str, int],
                  per_cat_cap: int, only_category: str | None,
                  min_s: int, max_s: int, dry_run: bool,
                  audio_classify: bool) -> tuple[bool, str]:
    """Returns (added?, resolved_category)."""
    if not license:
        return False, ""        # safety: reject unlicensed
    # Dedup by the most specific identifier available.  Some sources
    # (Incompetech catalogue) share a single page_url across many tracks,
    # so the per-track download_url is the safer dedup key.
    dedup_key = download_url or page_url
    if history.has_url(dedup_key):
        return False, ""
    if yt_id and history.has_yt_id(yt_id):
        return False, ""
    # Documentary-quality filter -- reject corporate / EDM / vlog / comedy
    # / overdramatic-trailer noise BEFORE we burn bandwidth.
    quality_blob = " ".join([title, " ".join(tags), category_hint, channel])
    dq = doc_quality(quality_blob)
    if dq.verdict == "reject":
        print(f"  . reject (quality={dq.score:.2f}): {title!r}  "
              f"{','.join(dq.reasons[:3])}")
        return False, ""
    # title-pass routing for early per-category-cap rejection
    tentative = (category_hint if category_hint in CATEGORIES
                 else classify_title(title))
    if tentative in CATEGORIES and per_cat_count.get(
            tentative, 0) >= per_cat_cap:
        return False, tentative
    if only_category and tentative != only_category and tentative in CATEGORIES:
        # tentative says wrong cat -- but final classify may move it; let
        # title-pass make the decision here to avoid downloads we'll throw
        # away
        return False, tentative

    if dry_run:
        print(f"  + DRY-RUN  [{source}] {title!r}  -> tentative={tentative}")
        return True, tentative

    # ---- download ----
    work.mkdir(parents=True, exist_ok=True)
    tmp_stem = work / _slug(f"{source}-{(yt_id or title)[:60]}")
    raw: Path | None = None
    if download_url:
        ext = ".mp3" if download_url.lower().endswith(".mp3") else ".m4a"
        raw = _download_direct(download_url, tmp_stem.with_suffix(ext))
    elif yt_id:
        raw = _yt_download_audio(yt_id, tmp_stem)
    if not raw:
        print(f"  ! download failed: {title}")
        return False, tentative

    raw_dur = _probe_dur(raw)
    if not _within_duration(raw_dur, min_s, max_s):
        print(f"  . {title!r} duration {raw_dur:.1f}s outside "
              f"[{min_s},{max_s}] -- discarded")
        raw.unlink(missing_ok=True)
        return False, tentative

    # ---- audio classify (Stage 2) ----
    cls: Classification
    if audio_classify:
        cls = classify_track(title=title, hint=category_hint,
                             tags=tags, audio=raw)
    else:
        cls = classify_track(title=title, hint=category_hint, tags=tags)
    cat = cls.category if cls.category in CATEGORIES else tentative
    if cat not in CATEGORIES:
        cat = "neutral"
    if only_category and cat != only_category:
        raw.unlink(missing_ok=True)
        return False, cat
    if per_cat_count.get(cat, 0) >= per_cat_cap:
        raw.unlink(missing_ok=True)
        return False, cat

    cat_dir = root / cat
    cat_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(title)
    # collision-safe slug
    if slug in _existing_slugs(cat_dir):
        slug = f"{slug}-{(yt_id or 'x')[:6]}"
        if slug in _existing_slugs(cat_dir):
            raw.unlink(missing_ok=True)
            return False, cat
    dst = cat_dir / f"{slug}.mp3"
    ok = _to_normalized_mp3(raw, dst)
    raw.unlink(missing_ok=True)
    if not ok:
        print(f"  ! transcode failed: {title}")
        return False, cat

    # ---- fingerprint dedup (Stage 4 foundation) ----
    fp = _file_fingerprint(dst)
    if fp and history.has_fp(fp):
        prev = history.data["fingerprints"][fp]
        print(f"  . fingerprint dup of {prev['source']}:{prev['slug']} "
              "-- discarded")
        dst.unlink(missing_ok=True)
        return False, cat

    # ---- sidecar JSON ----
    meta = {
        "name": slug,
        "category": cat,
        "title": title,
        "channel": channel,
        "source": source,
        "source_url": page_url,
        "license": license,
        "attribution": attribution,
        "dur": round(raw_dur, 2),
        "tags": tags,
        "fetched": time.strftime("%Y-%m-%d"),
        "classify": cls.to_meta(),
        "doc_quality": {
            "score": dq.score,
            "verdict": dq.verdict,
            "reasons": dq.reasons[:6],
        },
    }
    dst.with_suffix(".json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    _add_to_licenses(
        root,
        f"- **{cat}/{slug}** -- {title}"
        + (f" -- {channel}" if channel else "")
        + (f" -- {license}" if license else "")
        + f" -- <{page_url}>")
    history.record(slug=slug, category=cat, source=source,
                   url=dedup_key, yt_id=yt_id, fp=fp)
    per_cat_count[cat] = per_cat_count.get(cat, 0) + 1
    badge = ",".join(cls.sources) if cls.sources else "?"
    print(f"  + {cat}/{slug}.mp3  ({raw_dur:.1f}s)  "
          f"voted_by={badge}  conf={cls.confidence:.2f}")
    return True, cat


# --------------------------------------------------------------------------- #
# Manual source iterator (legacy YAML path)
# --------------------------------------------------------------------------- #
def _process_yaml_source(src: Source, root: Path, history: History, *,
                         per_cat_count: dict[str, int], per_cat_cap: int,
                         only_category: str | None,
                         min_s: int, max_s: int, dry_run: bool,
                         audio_classify: bool) -> int:
    print(f"\n-> {src.url}  (yaml: cat={src.category}, max={src.max})")
    entries = _yt_list(src.url, src.max * 2)
    if not entries:
        print("  ! no entries resolved")
        return 0
    added = 0
    work = root / ".tmp"
    for e in entries:
        if added >= src.max:
            break
        ok, _ = _ingest_track(
            title=e["title"], source="manual", page_url=e["url"],
            download_url=None, yt_id=e["id"],
            channel=e.get("channel", ""),
            license=src.license or "User-supplied -- verify license",
            attribution=src.attribution,
            category_hint=src.category, tags=[],
            root=root, work=work, history=history,
            per_cat_count=per_cat_count, per_cat_cap=per_cat_cap,
            only_category=only_category,
            min_s=min_s, max_s=max_s, dry_run=dry_run,
            audio_classify=audio_classify,
        )
        if ok:
            added += 1
    return added


# --------------------------------------------------------------------------- #
# --auto orchestrator (the Stage 1 main path)
# --------------------------------------------------------------------------- #
def _process_auto(root: Path, history: History, *,
                  per_source_limit: int, per_cat_count: dict[str, int],
                  per_cat_cap: int, only_category: str | None,
                  only_sources: list[str] | None,
                  min_s: int, max_s: int, dry_run: bool,
                  audio_classify: bool) -> int:
    from .music_sources import discover_all, available_sources
    available = available_sources()
    enabled = [s for s in available
               if (not only_sources or s in only_sources)]
    print(f"\n=== AUTO MODE -- sources: {', '.join(enabled) or '(none)'} ===")
    candidates = discover_all(per_source_limit=per_source_limit,
                              only=enabled)
    if not candidates:
        print("  ! no candidates from any source")
        return 0
    print(f"  total candidates after dedup-safe filter: {len(candidates)}")
    added = 0
    work = root / ".tmp"
    for c in candidates:
        ok, _ = _ingest_track(
            title=c.title, source=c.source, page_url=c.url,
            download_url=c.download_url, yt_id=c.yt_id,
            channel=c.channel, license=c.license,
            attribution=c.attribution,
            category_hint=c.category_hint, tags=c.tags,
            root=root, work=work, history=history,
            per_cat_count=per_cat_count, per_cat_cap=per_cat_cap,
            only_category=only_category,
            min_s=min_s, max_s=max_s, dry_run=dry_run,
            audio_classify=audio_classify,
        )
        if ok:
            added += 1
        # mark each source as run once we've at least walked it
        history.mark_source_run(c.source)
    return added


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pull, normalize & classify music for the Vidlore "
                    "library.")
    p.add_argument("--auto", action="store_true",
                   help="Crawl every trusted royalty-free source (Stage 1)")
    p.add_argument("--sources", default=None,
                   help="Override path to _sources.yaml (manual mode)")
    p.add_argument("--only", default="",
                   help="Comma-list of source names to enable in --auto "
                        "(default: all). e.g. incompetech,mixkit")
    p.add_argument("--category", default=None,
                   help="Only land tracks classified into this bucket")
    p.add_argument("--search", default=None,
                   help="One-off YouTube search (use with --category)")
    p.add_argument("--max", type=int, default=8,
                   help="Per-source / per-search candidate cap (default 8)")
    p.add_argument("--per-cat-cap", type=int, default=8,
                   help="Cap NEW tracks per category per run (default 8)")
    p.add_argument("--min-sec", type=int, default=45)
    p.add_argument("--max-sec", type=int, default=420)
    p.add_argument("--no-audio-classify", action="store_true",
                   help="Skip the audio-feature pass (title-only routing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be fetched, fetch nothing")
    p.add_argument("--reindex", action="store_true",
                   help="Refresh musiclib _index.json on exit")
    p.add_argument("--refresh-features", action="store_true",
                   help="Re-run audio analysis on every track already in "
                        "the library and rewrite its sidecar JSON with "
                        "the latest features (BPM, tension, darkness, "
                        "orchestral_density, rhythmic_aggression).")
    args = p.parse_args(argv)

    if args.refresh_features:
        from .music_classify import extract_features
        from dataclasses import asdict
        root = library_root()
        mp3s = list(root.rglob("*.mp3"))
        print(f"=== REFRESHING FEATURES on {len(mp3s)} track(s) ===")
        ok = 0
        for i, mp3 in enumerate(mp3s, 1):
            sj = mp3.with_suffix(".json")
            if not sj.exists():
                continue
            try:
                meta = json.loads(sj.read_text(encoding="utf-8"))
            except Exception:                              # noqa: BLE001
                continue
            try:
                feat = extract_features(mp3)
            except Exception as e:                         # noqa: BLE001
                print(f"  ! {mp3.name}: {e}")
                continue
            cls = meta.get("classify") or {}
            cls["features"] = asdict(feat)
            meta["classify"] = cls
            sj.write_text(json.dumps(meta, indent=1), encoding="utf-8")
            ok += 1
            if i % 10 == 0 or i == len(mp3s):
                print(f"  refreshed {i}/{len(mp3s)} "
                      f"({mp3.parent.name}/{mp3.stem})")
        print(f"=== {ok}/{len(mp3s)} sidecars updated ===")
        if args.reindex:
            try:
                from .musiclib import scan as scan_library2
                scan_library2(force=True)
                print("musiclib reindexed")
            except Exception as e:                         # noqa: BLE001
                print(f"reindex failed: {e}")
        return 0

    root = library_root()
    root.mkdir(parents=True, exist_ok=True)
    history = History(root)
    per_cat_count: dict[str, int] = {}
    audio_classify = not args.no_audio_classify
    total = 0

    # ad-hoc search short-circuit
    if args.search:
        if not args.category:
            print("--search needs --category")
            return 2
        src = Source(
            url=f"ytsearch{args.max}:{args.search}",
            category=args.category,
            license="YouTube search -- user must verify license",
            max=args.max,
        )
        total += _process_yaml_source(
            src, root, history,
            per_cat_count=per_cat_count, per_cat_cap=args.per_cat_cap,
            only_category=args.category,
            min_s=args.min_sec, max_s=args.max_sec,
            dry_run=args.dry_run, audio_classify=audio_classify,
        )

    if args.auto:
        only_sources = [s.strip() for s in args.only.split(",") if s.strip()]
        total += _process_auto(
            root, history,
            per_source_limit=args.max,
            per_cat_count=per_cat_count, per_cat_cap=args.per_cat_cap,
            only_category=args.category,
            only_sources=(only_sources or None),
            min_s=args.min_sec, max_s=args.max_sec,
            dry_run=args.dry_run, audio_classify=audio_classify,
        )

    if not args.auto and not args.search:
        src_path = (Path(args.sources) if args.sources
                    else root / "_sources.yaml")
        sources = _load_sources(src_path)
        if not sources:
            print(f"No sources in {src_path}, --auto not set, no --search. "
                  "Try one of:")
            print("  python -m vidlore.music_extract --auto --reindex")
            print("  python -m vidlore.music_extract --auto --max 3 "
                  "--per-cat-cap 2 --dry-run")
            return 1
        for s in sources:
            total += _process_yaml_source(
                s, root, history,
                per_cat_count=per_cat_count, per_cat_cap=args.per_cat_cap,
                only_category=args.category,
                min_s=args.min_sec, max_s=args.max_sec,
                dry_run=args.dry_run, audio_classify=audio_classify,
            )

    if not args.dry_run:
        history.save()

    print(f"\n=== {total} track(s) added ===")
    if per_cat_count:
        breakdown = ", ".join(f"{k}:+{v}" for k, v in
                              sorted(per_cat_count.items()))
        print(f"per-category landed: {breakdown}")

    if args.reindex and not args.dry_run:
        try:
            lib = scan_library(force=True)
            counts = ", ".join(
                f"{k}:{len(v)}" for k, v in sorted(lib.items()))
            print(f"musiclib reindexed -> {counts or '(empty)'}")
        except Exception as e:                              # noqa: BLE001
            print(f"musiclib reindex failed: {e}")

    tmp = root / ".tmp"
    if tmp.exists():
        for f in tmp.glob("*"):
            f.unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
