"""Stage 1 — ingest.

Batch-ingest permitted sources (URLs and/or local files) into a project. The permission gate
runs FIRST: a source whose permission is `unverified` is blocked before any download — the
module never pulls content the user hasn't asserted rights to. Local files are referenced in
place (read-only; we only ever read them and write trimmed copies into clips/), so nothing is
duplicated and no original is modified. URLs download into <project>/sources/ via yt-dlp,
using the engine's bundled ffmpeg for muxing.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import SourceVideo, ClipProject, PERMISSION_BLOCKS, SOURCE_OK, SOURCE_BLOCKED, SOURCE_FAILED
from .config import ClipConfig, ffmpeg_exe, ffprobe_exe
from ..ffmpeg_tool import ytdlp_ffmpeg_dir

_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".ts")

# yt-dlp writes per-stream intermediates as <stem>.fNNN.<ext> before merging — those are
# video-only (silent), so picking one kills ASR/dialogue-lock downstream.
_YTDLP_FRAG_RX = re.compile(r"\.f\d+$")


def find_produced_video(directory: Path, stem: str) -> Optional[Path]:
    """The file a yt-dlp run actually produced for `stem`. Prefers the merge target
    (<stem>.mp4 first) and never returns a video-only .fNNN fragment or a merge temp."""
    for ext in _VIDEO_EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    for p in sorted(directory.glob(f"{stem}.*")):
        if p.suffix.lower() not in _VIDEO_EXTS:
            continue
        inner = Path(p.stem)             # "<stem>.f616.mp4" -> "<stem>.f616"
        if _YTDLP_FRAG_RX.search(inner.name) or inner.suffix in (".part", ".temp", ".ytdl"):
            continue
        if p.stat().st_size > 0:
            return p
    return None


@dataclass
class SourceSpec:
    """User-supplied request for one source. `permission` defaults to the blocking value."""
    ref: str                      # a URL or a local file path
    permission: str = "unverified"
    permission_note: str = ""
    title: str = ""

    @property
    def is_local(self) -> bool:
        if re.match(r"^[a-z][a-z0-9+.-]*://", self.ref, re.IGNORECASE):
            return False         # an uppercase-scheme URL must never be probed as a file path
        return Path(self.ref).expanduser().exists()


def _slug(text: str, n: int = 28) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (s[:n] or "src")


def _source_id(spec: SourceSpec) -> str:
    h = hashlib.sha1(spec.ref.encode("utf-8")).hexdigest()[:8]
    base = spec.title or (Path(spec.ref).stem if spec.is_local else spec.ref)
    return f"{_slug(base, 24)}_{h}"


def _sha1_file(path: Path, full: bool = False) -> str:
    """Content hash for provenance. Default samples head/tail+size (fast, stable for big files)."""
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        if full or size <= 8 << 20:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(2 << 20))            # head
            f.seek(-(2 << 20), 2)
            h.update(f.read(2 << 20))            # tail
    return h.hexdigest()[:16]


def probe(path: Path) -> dict:
    """Return {duration,width,height,fps}. Tries PyAV (in-process, no external binary) first,
    then ffprobe. imageio-ffmpeg ships ffmpeg only (no ffprobe), so PyAV is the reliable path."""
    # --- primary: PyAV (already a dependency; reads container without a subprocess) ---
    try:
        import av
        with av.open(str(path)) as c:
            v = next((s for s in c.streams if s.type == "video"), None)
            dur = (c.duration / 1_000_000.0) if c.duration else 0.0
            if v is None:
                return {"duration": round(dur, 3), "width": 0, "height": 0, "fps": 0.0}
            if not dur and v.duration and v.time_base:
                dur = float(v.duration * v.time_base)
            try:
                fps = float(v.average_rate) if v.average_rate else (
                    float(v.guessed_rate) if getattr(v, "guessed_rate", None) else 0.0)
            except Exception:
                fps = 0.0
            cc = v.codec_context
            return {"duration": round(dur, 3),
                    "width": int(cc.width or 0), "height": int(cc.height or 0),
                    "fps": round(fps, 3)}
    except Exception:
        pass
    # --- fallback: ffprobe (if discoverable on the system) ---
    try:
        out = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return {}
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps = 0.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
    try:
        num, _, den = rate.partition("/")
        fps = round(float(num) / float(den), 3) if float(den or 0) else 0.0
    except Exception:
        fps = 0.0
    dur = 0.0
    try:
        dur = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
    except Exception:
        dur = 0.0
    return {"duration": dur, "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0), "fps": fps}


def _ingest_local(spec: SourceSpec, sid: str) -> SourceVideo:
    path = Path(spec.ref).expanduser().resolve()
    sv = SourceVideo(id=sid, url=f"local:{path}", local_path=str(path),
                     title=spec.title or path.stem,
                     permission=spec.permission, permission_note=spec.permission_note)
    if not path.exists():
        sv.status, sv.error = SOURCE_FAILED, "local file not found"
        return sv
    info = probe(path)
    sv.duration = info.get("duration", 0.0)
    sv.width, sv.height, sv.fps = info.get("width", 0), info.get("height", 0), info.get("fps", 0.0)
    sv.checksum = _sha1_file(path)
    sv.status = SOURCE_OK
    return sv


def _ingest_url(spec: SourceSpec, sid: str, proj: ClipProject, cfg: ClipConfig) -> SourceVideo:
    from . import hd_download as _hd
    sv = SourceVideo(id=sid, url=spec.ref, title=spec.title,
                     permission=spec.permission, permission_note=spec.permission_note)
    dest_stem = proj.sources_dir / sid
    maxh = cfg.max_height

    # --- HD-first path: defeat YouTube SABR/PO-token (legacy yt-dlp below only sees 360p). ---
    # Same SABR-capable recipe as download.py; falls through to the legacy downloader on any failure.
    if _hd.available() and _hd.is_youtube(spec.ref):
        try:
            hi = _hd.download_hd(spec.ref, str(dest_stem), max_height=maxh,
                                 ffmpeg_dir=str(Path(ffmpeg_exe()).parent),
                                 retries=cfg.download_retries)
        except Exception:                                  # noqa: BLE001
            hi = None
        if hi and hi.get("path") and Path(hi["path"]).exists():
            sv.local_path = hi["path"]
            sv.title = spec.title or hi.get("title") or sid
            sv.duration = float(hi.get("duration") or 0.0)
            sv.width = int(hi.get("width") or 0)
            sv.height = int(hi.get("height") or 0)
            sv.fps = float(hi.get("fps") or 0.0)
            sv.extra = {"hd_path": True}
            if hi.get("uploader"):
                sv.extra["uploader"] = hi["uploader"]
            if not sv.duration or not sv.width:
                info2 = probe(Path(hi["path"]))
                sv.duration = sv.duration or info2.get("duration", 0.0)
                sv.width = sv.width or info2.get("width", 0)
                sv.height = sv.height or info2.get("height", 0)
                sv.fps = sv.fps or info2.get("fps", 0.0)
            sv.checksum = _sha1_file(Path(hi["path"]))
            sv.status = SOURCE_OK
            return sv

    import yt_dlp  # local import: heavy, only needed for URLs (legacy path; YouTube ≤360p)
    ydl_opts = {
        "format": f"bestvideo[height<={maxh}]+bestaudio/best[height<={maxh}]/best",
        "outtmpl": str(dest_stem) + ".%(ext)s",
        "merge_output_format": "mp4",
        # see ffmpeg_tool.ytdlp_ffmpeg_dir — the binary's own parent dir holds a VERSIONED name that
        # yt-dlp never matches, which disables the merger and aborts every download on Windows.
        "ffmpeg_location": (ytdlp_ffmpeg_dir() or str(Path(ffmpeg_exe()).parent)),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "retries": cfg.download_retries, "fragment_retries": cfg.download_retries,
        "ignoreerrors": False, "noplaylist": True,
    }
    last_err = ""
    for attempt in range(1, cfg.download_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(spec.ref, download=True)
            sv.title = spec.title or info.get("title", "") or sid
            sv.duration = float(info.get("duration") or 0.0)
            sv.width = int(info.get("width") or 0)
            sv.height = int(info.get("height") or 0)
            sv.fps = float(info.get("fps") or 0.0)
            sv.extra = {k: info.get(k) for k in ("uploader", "license", "channel", "upload_date")
                        if info.get(k)}
            break
        except Exception as e:                       # noqa: BLE001 — record + backoff + retry
            last_err = str(e)[:300]
            time.sleep(min(2 ** attempt, 10))
    # locate the produced file (merge target preferred; .fNNN fragments never accepted)
    produced = find_produced_video(proj.sources_dir, sid)
    if produced is None:
        sv.status, sv.error = SOURCE_FAILED, last_err or "download produced no video file"
        return sv
    info2 = probe(produced)
    # unreadable = NEITHER the probe NOR yt-dlp's own metadata can vouch for it (a valid .ts
    # capture may probe without duration — yt-dlp already reported one). Delete the artifact,
    # or every later run hits yt-dlp's "already downloaded" skip and fails forever.
    if not (info2.get("duration") or sv.duration) or not (info2.get("width") or sv.width):
        try:
            produced.unlink(missing_ok=True)
        except OSError:
            pass
        sv.status, sv.error = SOURCE_FAILED, (last_err or f"unreadable download: {produced.name}")
        return sv
    sv.local_path = str(produced)
    sv.duration = sv.duration or info2.get("duration", 0.0)
    sv.width = sv.width or info2.get("width", 0)
    sv.height = sv.height or info2.get("height", 0)
    sv.fps = sv.fps or info2.get("fps", 0.0)
    sv.checksum = _sha1_file(produced)
    sv.status = SOURCE_OK
    return sv


def ingest_one(spec: SourceSpec, proj: ClipProject, cfg: ClipConfig) -> SourceVideo:
    sid = _source_id(spec)
    # --- permission gate FIRST: block before any network/download. The value must be a KNOWN
    # assertion — a typo ("fair-use", "yes") must block, not silently land in the ledger ---
    from .models import PERMISSIONS
    if spec.permission in PERMISSION_BLOCKS or not spec.permission \
            or spec.permission not in PERMISSIONS:
        return SourceVideo(
            id=sid, url=(f"local:{spec.ref}" if spec.is_local else spec.ref),
            title=spec.title,
            permission=(spec.permission if spec.permission in PERMISSIONS else "unverified"),
            permission_note=spec.permission_note, status=SOURCE_BLOCKED,
            error="permission unverified/unknown — set owner|licensed|public_domain|cc|fair_use_claim to ingest",
        )
    if spec.is_local:
        return _ingest_local(spec, sid)
    # scheme-less ref that's neither on disk nor URL-shaped = a mistyped local path; handing it
    # to yt-dlp produces a baffling extractor error instead of "file not found"
    if not re.match(r"^[a-z][a-z0-9+.-]*://", spec.ref, re.IGNORECASE) and \
            not re.match(r"^(www\.)?[\w-]+(\.[\w-]+)+(:\d+)?(/|$)", spec.ref):
        return SourceVideo(
            id=sid, url=f"local:{spec.ref}", title=spec.title, permission=spec.permission,
            permission_note=spec.permission_note, status=SOURCE_FAILED,
            error=f"local file not found: {spec.ref}")
    return _ingest_url(spec, sid, proj, cfg)


def ingest_sources(proj: ClipProject, specs: list[SourceSpec], cfg: ClipConfig,
                   progress=None) -> list[SourceVideo]:
    """Ingest all specs (concurrent for URL downloads). Updates proj.sources and saves."""
    proj.ensure_dirs()
    # dedup by source-id BEFORE submission (mirrors download.py): the same ref given twice would
    # put two concurrent workers racing on the same <sid>.* files
    locals_, urls = [], []
    _queued: set[str] = set()
    for sp in specs:
        sid = _source_id(sp)
        if sid in _queued:
            continue
        _queued.add(sid)
        (locals_ if sp.is_local else urls).append(sp)

    results: list[SourceVideo] = []

    # local files + blocked sources are cheap → do inline
    for sp in locals_:
        sv = ingest_one(sp, proj, cfg)
        results.append(sv)
        if progress:
            progress(f"ingest: {sv.id} [{sv.status}]")

    # URL downloads → bounded concurrency, politely paced
    if urls:
        with cf.ThreadPoolExecutor(max_workers=max(1, cfg.download_concurrency)) as ex:
            futs = {}
            for sp in urls:
                futs[ex.submit(ingest_one, sp, proj, cfg)] = sp
                time.sleep(0.5)                       # stagger starts (per-host politeness)
            for fut in cf.as_completed(futs):
                sv = fut.result()
                results.append(sv)
                if progress:
                    progress(f"ingest: {sv.id} [{sv.status}]")

    # merge into project: PRESERVE pre-existing sources, refresh/append this batch by id in spec
    # order — replacing the whole list dropped every previously-ingested source (permissions,
    # checksums, provenance) on an incremental --add
    by_id = {sv.id: sv for sv in results}
    order = [_source_id(sp) for sp in specs]
    new_in_order = [by_id[i] for i in order if i in by_id]
    new_ids = {sv.id for sv in new_in_order}
    # the sid embeds the user-supplied title, so a re-ingest after a title edit gets a NEW id —
    # also displace kept entries pointing at the same ref or the same content (checksum), or the
    # project accumulates duplicate copies of one video
    new_urls = {sv.url for sv in new_in_order if sv.url}
    new_sums = {sv.checksum for sv in new_in_order if sv.status == SOURCE_OK and sv.checksum}
    kept = [s for s in proj.sources
            if s.id not in new_ids and s.url not in new_urls
            and not (s.status == SOURCE_OK and s.checksum and s.checksum in new_sums)]
    proj.sources = kept + new_in_order
    proj.save()
    return proj.sources


def load_specs(path: Path) -> list[SourceSpec]:
    """Load source specs from a JSON or YAML list of {ref/url/file, permission, note, title}."""
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    items = data.get("sources", data) if isinstance(data, dict) else data
    specs = []
    for it in items:
        if isinstance(it, str):
            specs.append(SourceSpec(ref=it))
        else:
            specs.append(SourceSpec(
                ref=it.get("ref") or it.get("url") or it.get("file") or "",
                permission=it.get("permission", "unverified"),
                permission_note=it.get("permission_note", it.get("note", "")),
                title=it.get("title", ""),
            ))
    return [s for s in specs if s.ref]
