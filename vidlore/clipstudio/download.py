"""Phase 3 — authorized batch downloader.

Downloads discovered candidates into the project: bounded concurrency, retries, RESUMABLE
(yt-dlp continuedl), best quality ≤ max_height, duplicate detection (by content checksum),
and normalization to a uniform .mp4 container. Never bypasses access controls.

The PERMISSION POLICY governs what may be fetched when the user supplied no per-source rights
(they gave only a topic + script):
  - "block"            : nothing downloads (discovered sources are unverified) — default, safest
  - "open_only"        : only archive.org / public-domain-hinted sources → permission=public_domain
  - "approved_testing" : user explicitly approves these sources for testing → permission=fair_use_claim
The chosen policy + per-source permission are written to the ledger; the tool certifies nothing.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

from .models import SourceVideo, ClipProject, SOURCE_OK, SOURCE_BLOCKED, SOURCE_FAILED
from .config import ClipConfig, ffmpeg_exe
from ..ffmpeg_tool import ytdlp_ffmpeg_dir
from .discover import SourceCandidate
from .ingest import probe, _sha1_file, _slug, _VIDEO_EXTS, find_produced_video
import hashlib

POLICIES = ("block", "open_only", "approved_testing")


def policy_permission(cand: SourceCandidate, policy: str) -> tuple[str, bool, str]:
    """(permission, allowed, note) for a candidate under a policy."""
    if policy == "approved_testing":
        return ("fair_use_claim", True, "user approved-for-testing policy")
    if policy == "open_only":
        if cand.permission_hint == "public_domain":
            return ("public_domain", True, "archive.org public-domain listing (verify before publishing)")
        return ("unverified", False, "open_only policy: non-public-domain source skipped")
    return ("unverified", False, "no permission policy set (default block)")


def _cand_id(cand: SourceCandidate) -> str:
    h = hashlib.sha1((cand.url or cand.id).encode("utf-8")).hexdigest()[:8]
    return f"{_slug(cand.title or cand.id or 'src', 22)}_{h}"


def _download_one(cand: SourceCandidate, sid: str, perm: str, note: str,
                  proj: ClipProject, cfg: ClipConfig, progress=None) -> SourceVideo:
    from . import hd_download as _hd
    sv = SourceVideo(id=sid, url=cand.url, title=cand.title, permission=perm,
                     permission_note=note, extra={"provider": cand.provider,
                                                   "relevance": cand.relevance, "query": cand.query})
    if getattr(cand, "anchor_verified", False):
        sv.extra["anchor_verified"] = True       # dialogue-proven exact scene — match.py trusts it
    dest_stem = proj.sources_dir / sid
    maxh = cfg.max_height

    # --- HD-first path: defeat YouTube SABR/PO-token (engine yt-dlp can only see 360p). ---
    # Uses the user's own logged-in session + YouTube's public player clients (no access bypass).
    # Falls through to the legacy in-process downloader below on any failure — but NEVER silently:
    # the reason is logged and recorded (a whole run once fell back to 360p with zero trace).
    if _hd.available() and _hd.is_youtube(cand.url):
        _hd_msgs: list = []
        try:
            hi = _hd.download_hd(cand.url, str(dest_stem), max_height=maxh,
                                 ffmpeg_dir=str(Path(ffmpeg_exe()).parent),
                                 retries=cfg.download_retries, progress=_hd_msgs.append)
        except Exception as e:                             # noqa: BLE001
            hi = None
            _hd_msgs.append(f"hd: exception {str(e)[:160]}")
        if not hi:
            sv.extra["hd_fallback"] = (_hd_msgs[-1] if _hd_msgs else "hd: no result")[:200]
            if progress:
                progress(f"download: {sid} HD path fell back → legacy "
                         f"({sv.extra['hd_fallback'][:90]})")
        if hi and hi.get("path") and Path(hi["path"]).exists():
            sv.local_path = hi["path"]
            sv.title = cand.title or hi.get("title") or sid
            sv.duration = float(hi.get("duration") or cand.duration or 0.0)
            sv.width = int(hi.get("width") or 0)
            sv.height = int(hi.get("height") or cand.height or 0)
            sv.fps = float(hi.get("fps") or 0.0)
            if hi.get("uploader"):
                sv.extra["uploader"] = hi["uploader"]
            sv.extra["hd_path"] = True
            info2 = probe(Path(hi["path"]))
            sv.duration = sv.duration or info2.get("duration", 0.0)
            sv.width = sv.width or info2.get("width", 0)
            sv.height = sv.height or info2.get("height", 0)
            sv.fps = sv.fps or info2.get("fps", 0.0)
            sv.checksum = _sha1_file(Path(hi["path"]))
            sv.status = SOURCE_OK
            return sv

    import yt_dlp                                            # legacy path only (≤360p for YouTube)
    opts = {
        # prefer >=720p (crisp), then best <=1080, then anything — never silently take 360p if HD exists
        "format": (f"bestvideo[height>=720][height<={maxh}]+bestaudio/"
                   f"bestvideo[height<={maxh}]+bestaudio/best[height<={maxh}]/best"),
        "outtmpl": str(dest_stem) + ".%(ext)s",
        "merge_output_format": "mp4",                 # normalize container
        # NEVER Path(ffmpeg_exe()).parent — the imageio binary is VERSIONED
        # (ffmpeg-win-x86_64-v7.1.exe), and yt-dlp only recognises a file literally named
        # ffmpeg/ffprobe in the dir it is given. Handing it the binary's own dir makes the merger
        # unavailable and aborts every bestvideo+bestaudio download (no footage at all on Windows).
        "ffmpeg_location": (ytdlp_ffmpeg_dir() or str(Path(ffmpeg_exe()).parent)),
        "quiet": True, "no_warnings": True, "noprogress": True,
        "continuedl": True,                           # RESUME partial downloads
        "retries": cfg.download_retries, "fragment_retries": cfg.download_retries,
        "noplaylist": True, "ignoreerrors": False, "socket_timeout": 30,
        # YouTube's default web client is often throttled (403); these are its STANDARD public
        # player clients (not an auth/DRM bypass) and keep automated extraction working.
        "extractor_args": {"youtube": {"player_client": ["android", "web_safari", "web"]}},
    }
    last_err = ""
    for attempt in range(1, cfg.download_retries + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                info = y.extract_info(cand.url, download=True)
            sv.title = cand.title or info.get("title", "") or sid
            sv.duration = float(info.get("duration") or cand.duration or 0.0)
            sv.width = int(info.get("width") or 0)
            sv.height = int(info.get("height") or cand.height or 0)
            sv.fps = float(info.get("fps") or 0.0)
            sv.extra.update({k: info.get(k) for k in ("uploader", "license", "channel", "upload_date")
                             if info.get(k)})
            break
        except Exception as e:                        # noqa: BLE001
            last_err = str(e)[:300]
            time.sleep(min(2 ** attempt, 12))
    produced = find_produced_video(proj.sources_dir, sid)
    if produced is None:
        sv.status, sv.error = SOURCE_FAILED, last_err or "download produced no file"
        return sv
    info2 = probe(produced)
    # unreadable = neither probe nor yt-dlp metadata vouches for it; delete the artifact so the
    # next run actually re-downloads instead of hitting "already downloaded" on a corrupt file
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


def download_candidates(proj: ClipProject, candidates: list[SourceCandidate], cfg: ClipConfig, *,
                        policy: str = "block", limit: int = 0, progress=None) -> list[SourceVideo]:
    """Download permitted candidates with concurrency/retry/resume/dedup. Updates proj.sources."""
    proj.ensure_dirs()

    def log(m):
        if progress:
            progress(m)

    if policy not in POLICIES:
        policy = "block"
    queue = candidates if not limit else candidates[:limit]

    # split into allowed vs policy-blocked (record both); the same video discovered twice must
    # be submitted ONCE — two workers would race writing the same <sid>.* files
    allowed, results = [], []
    _queued: set[str] = set()
    for c in queue:
        perm, ok, note = policy_permission(c, policy)
        sid = _cand_id(c)
        if sid in _queued:
            continue
        _queued.add(sid)
        if not ok:
            results.append(SourceVideo(id=sid, url=c.url, title=c.title, permission=perm,
                                       permission_note=note, status=SOURCE_BLOCKED, error=note,
                                       extra={"provider": c.provider}))
        else:
            allowed.append((c, sid, perm, note))
    log(f"download: policy={policy} · {len(allowed)} allowed / {len(queue)-len(allowed)} blocked")

    # concurrent downloads
    seen_checksums: set[str] = set()
    if allowed:
        with cf.ThreadPoolExecutor(max_workers=max(1, cfg.download_concurrency)) as ex:
            futs = {}
            for (c, sid, perm, note) in allowed:
                futs[ex.submit(_download_one, c, sid, perm, note, proj, cfg, progress)] = sid
                time.sleep(0.5)                       # stagger (per-host politeness)
            for fut in cf.as_completed(futs):
                sv = fut.result()
                # duplicate detection by content checksum
                if sv.status == SOURCE_OK and sv.checksum:
                    if sv.checksum in seen_checksums:
                        try:
                            Path(sv.local_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        sv.status, sv.error = "duplicate", "content checksum matches another source"
                    else:
                        seen_checksums.add(sv.checksum)
                results.append(sv)
                log(f"download: {sv.id} [{sv.status}] {sv.height or '?'}p {sv.duration:.0f}s")
                # QUALITY AUDIT — requested vs ACTUAL height, per source. The format string asks
                # for >=prefer_height first, so landing below it means the upload simply has no HD
                # stream (kept anyway: relevance-first). Recorded on the source + one greppable line
                # so a soft render can be traced to its low-res sources afterwards.
                if sv.status == SOURCE_OK and sv.height:
                    sv.extra["quality_audit"] = {
                        "requested_max": int(cfg.max_height),
                        "preferred_min": int(cfg.discover_prefer_height),
                        "actual": int(sv.height),
                    }
                    if sv.height < cfg.discover_prefer_height:
                        sv.extra["quality_audit"]["reason"] = "source_has_no_hd_stream"
                        log(f"download: {sv.id} LOW-RES {sv.height}p < preferred "
                            f"{cfg.discover_prefer_height}p (source max — kept for relevance)")

    # merge into project: PRESERVE any pre-existing sources, then add/refresh the just-downloaded
    # ones in discovery order. (For a fresh build proj.sources is empty so this is identical to the
    # old replace; for an INCREMENTAL add it APPENDS instead of dropping the originals — the old
    # `proj.sources = [...]` silently discarded every previously-indexed source.)
    by_id = {sv.id: sv for sv in results}
    order = [_cand_id(c) for c in queue]
    new_in_order = [by_id[i] for i in order if i in by_id]
    new_ids = {sv.id for sv in new_in_order}
    kept = [s for s in proj.sources if s.id not in new_ids]
    proj.sources = kept + new_in_order
    proj.save()
    ok = sum(1 for s in proj.sources if s.status == SOURCE_OK)
    log(f"download: {ok} usable sources downloaded")
    return proj.sources
