"""Native-media publication contracts for ClipStudio."""
from __future__ import annotations

import json
import os
from pathlib import Path


MIN_NATIVE_SHORT_EDGE = 720
MIN_NATIVE_LONG_EDGE = 1280
# Compatibility name for callers/audits that historically described only the
# vertical floor.  Admission itself now checks both decoded dimensions.
MIN_NATIVE_VIDEO_HEIGHT = MIN_NATIVE_SHORT_EDGE

# Backfill and match can rebuild the same pool several times in one process. Cache actual-byte
# probes by immutable file identity so the native invariant does not spawn hundreds of redundant
# ffprobe processes. The final publication assertion below deliberately keeps its own probe.
_NATIVE_PROBE_CACHE: dict[tuple[str, int, int], dict] = {}


def native_video_ok(info, minimum: int = MIN_NATIVE_SHORT_EDGE,
                    minimum_long: int = MIN_NATIVE_LONG_EDGE) -> bool:
    """True only when the decoded bytes contain a real HD raster.

    Height alone is not sufficient: a narrow 640x720 file still needs a 2x
    horizontal enlargement to fill an HD canvas.  Short/long-edge admission is
    orientation agnostic while requiring at least 1280x720 worth of detail.
    """
    if not isinstance(info, dict):
        return False
    try:
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        short, long = sorted((width, height))
        return short >= int(minimum) and long >= int(minimum_long)
    except (TypeError, ValueError, OverflowError):
        return False


def probe_native_video_info(path: Path | str) -> dict:
    """Probe decoded dimensions from local bytes; missing/unreadable media returns ``{}``.

    Discovery metadata is never consulted.  Replacing the file changes its stat identity and
    therefore forces a fresh probe before those new bytes can enter a visual pool.
    """
    try:
        media = Path(path)
        stat = media.stat()
        if not media.is_file() or stat.st_size <= 0:
            return {}
        key = (str(media.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    except (OSError, TypeError, ValueError):
        return {}
    cached = _NATIVE_PROBE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    try:
        from .ingest import probe
        raw = probe(media) or {}
        info = {"width": int(raw.get("width") or 0),
                "height": int(raw.get("height") or 0)}
    except Exception:                                  # noqa: BLE001 — unknown fails closed
        info = {}
    # A zero/unknown result is a technical observation, not stable media identity. Never memoize
    # it: a transient ffprobe failure must be retryable within this same process.
    if not info.get("width") or not info.get("height"):
        return {}
    while len(_NATIVE_PROBE_CACHE) >= 512:
        try:
            _NATIVE_PROBE_CACHE.pop(next(iter(_NATIVE_PROBE_CACHE)))
        except (KeyError, StopIteration):
            break
    _NATIVE_PROBE_CACHE[key] = dict(info)
    return info


def assert_native_hd_selections(proj, selections, audit_path: Path,
                                *, minimum: int = MIN_NATIVE_SHORT_EDGE,
                                minimum_long: int = MIN_NATIVE_LONG_EDGE) -> dict:
    """Fail before render when ordinary moving footage is natively below 720p.

    Image fallbacks are checked after their full-resolution rescue in build;
    this contract covers video sources.  Probe the local bytes, never requested
    download metadata, so a 360p fallback mislabeled 1080p cannot pass.
    """
    from .ingest import probe
    rows, failures, cache = [], [], {}
    for sel in selections or []:
        image_path = str(getattr(sel, "image_path", "") or "")
        if image_path and Path(image_path).exists() and Path(image_path).stat().st_size > 0:
            continue
        sid = str(getattr(sel, "source_id", "") or "")
        src = proj.source(sid) if sid else None
        path = str(getattr(src, "local_path", "") or "") if src else ""
        if path not in cache:
            try:
                cache[path] = probe(Path(path)) if path and Path(path).exists() else {}
            except Exception:
                cache[path] = {}
        info = cache[path]
        row = {
            "original_beat": int(getattr(sel, "segment_index", -1)),
            "source_id": sid,
            "source_title": str(getattr(src, "title", "") or "")[:160] if src else "",
            "path": path,
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "minimum_short_edge": int(minimum),
            "minimum_long_edge": int(minimum_long),
            "passed": native_video_ok(info, minimum, minimum_long),
        }
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
    payload = {"schema": "native_resolution/2",
               "minimum_short_edge": int(minimum),
               "minimum_long_edge": int(minimum_long),
               "passed": not failures, "selections": rows, "failures": failures}
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = audit_path.with_name(audit_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, audit_path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    if failures:
        from .verify import NonRetryableBuildError
        first = failures[0]
        reason = (f"{first['width']}x{first['height']}"
                  if first["width"] and first["height"] else "unprobeable")
        raise NonRetryableBuildError(
            f"native-resolution gate: {len(failures)} selection(s) are below "
            f"{minimum_long}x{minimum} "
            f"or unverifiable (first beat {first['original_beat']}: {reason}); "
            f"upscaling does not create HD detail. See {audit_path.name}",
            kind="native_resolution")
    return payload
