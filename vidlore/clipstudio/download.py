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
import os
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
        # weekly yt-dlp self-update BEFORE the batch — the maintainers' standing fix for
        # fleet-wide 403 waves is a new release; version rot re-earns the block
        try:
            from . import hd_download as _hd_up
            _hd_up.maybe_update_ytdlp(log)
        except Exception:                                # noqa: BLE001
            pass
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

    # 403 RECOVERY SWEEP. A 403 on the HD data URL is TRANSIENT (burst throttle / stale PO token),
    # not "this upload has no HD" — but the per-source fallback made it permanent: one bad window
    # turned an entire render into a 360p upscale (measured: 83/86 sources on one job). After the
    # parallel pass, re-attempt the HD path serially for every 403-fallen source after a cooldown;
    # only genuine unavailability may leave a source at 360p. Env: VIDLORE_HD_403_SWEEP=0 disables,
    # _WAIT cooldown seconds, _MAX caps the sweep size.
    from . import hd_download as _hd
    if os.environ.get("VIDLORE_HD_403_SWEEP", "1").lower() not in ("0", "false", "no"):
        _fell403 = [s for s in proj.sources
                    if s.status == SOURCE_OK and s.url and _hd.is_youtube(s.url)
                    and "403" in ((s.extra or {}).get("hd_fallback") or "")
                    and 0 < int(s.height or 0) < cfg.discover_prefer_height]
        _cap = max(0, int(os.environ.get("VIDLORE_HD_403_SWEEP_MAX", "24") or 24))
        _fell403 = _fell403[:_cap]
        if _fell403 and _hd.available():
            _wait = max(0, int(os.environ.get("VIDLORE_HD_403_SWEEP_WAIT", "45") or 45))
            log(f"download: 403 sweep — {len(_fell403)} source(s) fell back on HTTP 403; "
                f"cooling down {_wait}s then retrying the HD path serially")
            time.sleep(_wait)
            _rec = 0
            for sv in _fell403:
                stem = proj.sources_dir / f"{sv.id}.hdretry"
                try:
                    hi = _hd.download_hd(sv.url, str(stem), max_height=cfg.max_height,
                                         ffmpeg_dir=str(Path(ffmpeg_exe()).parent),
                                         retries=1, progress=None)
                except Exception:                            # noqa: BLE001
                    hi = None
                got = Path(hi["path"]) if (hi and hi.get("path")
                                           and Path(hi["path"]).exists()) else None
                if got and int(hi.get("height") or 0) > int(sv.height or 0):
                    final = proj.sources_dir / f"{sv.id}{got.suffix.lower()}"
                    try:
                        old = Path(sv.local_path) if sv.local_path else None
                        if old and old.exists() and old.resolve() != got.resolve():
                            old.unlink(missing_ok=True)
                        got.replace(final)
                        sv.local_path = str(final)
                        sv.width = int(hi.get("width") or sv.width or 0)
                        sv.height = int(hi.get("height") or sv.height or 0)
                        sv.fps = float(hi.get("fps") or sv.fps or 0.0)
                        sv.duration = float(hi.get("duration") or sv.duration or 0.0)
                        sv.checksum = _sha1_file(final)
                        sv.extra.pop("hd_fallback", None)
                        sv.extra["hd_path"] = True
                        sv.extra["hd_recovered"] = True
                        if "quality_audit" in sv.extra:
                            sv.extra["quality_audit"]["actual"] = int(sv.height)
                            sv.extra["quality_audit"].pop("reason", None)
                        _rec += 1
                        log(f"download: {sv.id} HD RECOVERED on sweep → {sv.height}p")
                    except OSError as e:
                        log(f"download: {sv.id} sweep replace failed ({e}) — keeping legacy file")
                else:
                    # clean the temp namespace; the legacy file stays authoritative
                    for f in proj.sources_dir.glob(f"{sv.id}.hdretry*"):
                        try:
                            f.unlink()
                        except OSError:
                            pass
            if _rec:
                proj.save()
            log(f"download: 403 sweep done — {_rec}/{len(_fell403)} source(s) recovered to HD")
            # SECOND sweep after a LONG cooldown when the first recovered nothing: a 45s pause
            # cannot outlast a sustained IP block (measured: 0/24 recovered, block held for
            # hours). Update yt-dlp first (the maintainers' standing fix), wait, retry once.
            if _rec == 0 and os.environ.get("VIDLORE_HD_403_SWEEP2", "1").strip().lower() \
                    not in ("0", "false", "no"):
                _wait2 = max(0, int(os.environ.get("VIDLORE_HD_403_SWEEP2_WAIT", "900") or 900))
                _hd.maybe_update_ytdlp(log, force=True)
                log(f"download: 403 sweep-2 — first sweep recovered nothing (sustained block); "
                    f"cooling down {_wait2 // 60} min then retrying once more")
                time.sleep(_wait2)
                _rec2 = 0
                for sv in _fell403:
                    stem = proj.sources_dir / f"{sv.id}.hdretry2"
                    try:
                        hi = _hd.download_hd(sv.url, str(stem), max_height=cfg.max_height,
                                             ffmpeg_dir=str(Path(ffmpeg_exe()).parent),
                                             retries=2, progress=None)
                    except Exception:                    # noqa: BLE001
                        hi = None
                    got = Path(hi["path"]) if (hi and hi.get("path")
                                               and Path(hi["path"]).exists()) else None
                    if got and int(hi.get("height") or 0) > int(sv.height or 0):
                        final = proj.sources_dir / f"{sv.id}{got.suffix.lower()}"
                        try:
                            old = Path(sv.local_path) if sv.local_path else None
                            if old and old.exists() and old.resolve() != got.resolve():
                                old.unlink(missing_ok=True)
                            got.replace(final)
                            sv.local_path = str(final)
                            sv.width = int(hi.get("width") or sv.width or 0)
                            sv.height = int(hi.get("height") or sv.height or 0)
                            sv.checksum = _sha1_file(final)
                            sv.extra.pop("hd_fallback", None)
                            sv.extra["hd_path"] = True
                            sv.extra["hd_recovered"] = True
                            _rec2 += 1
                            log(f"download: {sv.id} HD RECOVERED on sweep-2 → {sv.height}p")
                        except OSError:
                            pass
                    else:
                        for f in proj.sources_dir.glob(f"{sv.id}.hdretry2*"):
                            try:
                                f.unlink()
                            except OSError:
                                pass
                if _rec2:
                    proj.save()
                log(f"download: 403 sweep-2 done — {_rec2}/{len(_fell403)} recovered")
        elif _fell403:
            log("download: 403 sweep skipped — HD path unavailable (check .hdvenv/.pot)")

    # HD-PATH HEALTH. The per-source fallback reason is already recorded in extra['hd_fallback'],
    # but nothing ever SUMMARISED it — so a machine-wide HD failure degraded a whole render to
    # 360p in total silence. Measured on two finished jobs: 42/44 and 34/34 sources fell back with
    # 'HTTP Error 403: Forbidden' (the PO-token server was not up), leaving 95% of the footage
    # sub-480p on a 1080p canvas. Both renders were then audited as murky/low-res/illegible and
    # the cause was invisible in the build log. One loud line, and the numbers on the project.
    _yt = [s for s in proj.sources if s.status == SOURCE_OK
           and ((s.extra or {}).get("hd_path") or (s.extra or {}).get("hd_fallback"))]
    if _yt:
        _fell = [s for s in _yt if not (s.extra or {}).get("hd_path")]
        _sd = [s for s in proj.sources
               if s.status == SOURCE_OK and 0 < int(getattr(s, "height", 0) or 0) < 480]
        _frac = len(_fell) / float(len(_yt))
        proj.meta.setdefault("download_audit", {})
        proj.meta["download_audit"].update({
            "youtube_sources": len(_yt),
            "hd_path_ok": len(_yt) - len(_fell),
            "hd_fallback": len(_fell),
            "hd_recovered": sum(1 for s in _yt if (s.extra or {}).get("hd_recovered")),
            "hd_403_fallbacks": sum(1 for s in _fell
                                    if "403" in ((s.extra or {}).get("hd_fallback") or "")),
            "sub_480p_sources": len(_sd),
            "top_fallback_reason": (sorted(
                ((s.extra or {}).get("hd_fallback", "") for s in _fell))[0][:160] if _fell else ""),
        })
        if _frac >= 0.5:
            log(f"download: ⚠ HD PATH DEGRADED — {len(_fell)}/{len(_yt)} YouTube source(s) fell "
                f"back to the legacy 360p downloader; {len(_sd)} source(s) are sub-480p. "
                f"The whole render will be built from SD footage upscaled to the 1080p canvas. "
                f"First reason: {(proj.meta['download_audit']['top_fallback_reason'] or '?')[:120]}")
            log("download: ⚠ fix the HD path before trusting this render's image quality "
                "(needs .hdvenv + node + deno + the .pot PO-token server reachable)")
    return proj.sources
