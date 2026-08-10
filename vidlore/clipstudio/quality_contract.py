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
# probes by file identity — (resolved path, st_size, st_mtime_ns), re-stat'ed on EVERY call, so a
# replaced, shrunken or deleted file always forces a fresh probe — and the native invariant stops
# spawning hundreds of redundant ffprobe processes.
#
# The final publication assertion below now SHARES this cache; it used to keep its own probe. That
# is safe for exactly two reasons, and both must hold for any future writer: the key is derived
# from the bytes themselves rather than from a name, and `probe_native_video_info` is the sole
# writer and refuses to memoize a zero/unknown result. Seeding this cache from download metadata
# would break the assertion's own contract ("probe the local bytes, never requested download
# metadata, so a 360p fallback mislabeled 1080p cannot pass") — do not.
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


# The three ways a selection can fail the native-HD floor. They look identical in the audit and
# they are not the same fact at all, which is why they are named here and carried in every row.
MEASURED_SUB_HD = "measured_sub_hd"   # real bytes, real numbers, genuinely below the floor
UNPROBEABLE = "unprobeable"           # a real file is there and ffprobe told us nothing
NO_SOURCE = "no_source"               # nothing to measure: no source bound, or the file is gone


def _native_failure_class(path: str, info: dict) -> str:
    """Why this selection failed — a MEASUREMENT, or the absence of one.

    Only the first is a content fact. "We could not measure it" is never content, even though the
    thing being measured is: a missing ffprobe, a deleted source or an unbound selection would
    otherwise be forgiven as "imperfect footage" and ride out inside a review draft, which is the
    fail-open class that once hid a dead code path for months.
    """
    if int(info.get("width") or 0) > 0 and int(info.get("height") or 0) > 0:
        return MEASURED_SUB_HD
    try:
        media = Path(path)
        if path and media.is_file() and media.stat().st_size > 0:
            return UNPROBEABLE
    except OSError:
        pass
    return NO_SOURCE


def assert_native_hd_selections(proj, selections, audit_path: Path,
                                *, minimum: int = MIN_NATIVE_SHORT_EDGE,
                                minimum_long: int = MIN_NATIVE_LONG_EDGE) -> dict:
    """Fail before render when ordinary moving footage is natively below 720p.

    Image fallbacks are checked after their full-resolution rescue in build;
    this contract covers video sources.  Probe the local bytes, never requested
    download metadata, so a 360p fallback mislabeled 1080p cannot pass.
    """
    rows, failures, by_path = [], [], {}

    def _measure(path: str) -> dict:
        """One reconciled measurement per PATH, per call — with exactly one retry.

        The old code kept a per-path dict but memoized the EMPTY result of a failed probe, so one
        flaky ffprobe became a permanent verdict. Dropping the dict entirely traded that for two
        worse bugs: the same bytes could be probed once per SELECTION and produce CONTRADICTORY
        rows in one audit (a transient miss on beat 12 and a clean 1920x1080 on beat 40 for the
        same file), and the unmeasured row wins the raise — so one flaky probe turns a deliverable
        draft into a hard failure, the exact outcome this whole change exists to remove. It also
        paid N ffprobe attempts for one corrupt source bound to N beats.

        So: memoize per call, retry an empty result ONCE, and let that decision govern every row
        for those bytes. Cross-call retryability is untouched — probe_native_video_info still
        refuses to memoize a zero globally, so the next render re-measures from scratch.
        """
        if path in by_path:
            return by_path[path]
        info = probe_native_video_info(path) if path else {}
        if not info and path:
            info = probe_native_video_info(path)          # one bounded retry, then it is a fact
        by_path[path] = info
        return info

    for sel in selections or []:
        image_path = str(getattr(sel, "image_path", "") or "")
        if image_path and Path(image_path).exists() and Path(image_path).stat().st_size > 0:
            continue
        sid = str(getattr(sel, "source_id", "") or "")
        src = proj.source(sid) if sid else None
        path = str(getattr(src, "local_path", "") or "") if src else ""
        info = _measure(path)
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
        if not row["passed"]:
            row["failure_class"] = _native_failure_class(path, info)
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
        # UNMEASURED FIRST, and fatal in every mode. A review draft exists to show a human footage
        # that is honestly ours and imperfect — it cannot show them footage nobody measured. If any
        # selection is unprobeable or has no source at all, that is a broken ffprobe, a deleted
        # file or an unbound selection, and forgiving it would let a flaky probe become a
        # publication verdict. Raised before the content branch so a single unmeasured row cannot
        # ride out inside a batch of genuine sub-HD ones.
        unmeasured = [f for f in failures if f.get("failure_class") != MEASURED_SUB_HD]
        if unmeasured:
            u = unmeasured[0]
            raise NonRetryableBuildError(
                f"native-resolution gate: {len(unmeasured)} of {len(failures)} selection(s) could "
                f"not be MEASURED (first beat {u['original_beat']}: {u.get('failure_class')}"
                + (f", {u['path']}" if u.get("path") else ", no source bound")
                + f"); this is a broken probe or a missing file, not sub-HD footage — fix it "
                  f"rather than publish an unmeasured frame. See {audit_path.name}",
                kind="native_resolution_probe")
        # Every failure is a real measurement of real bytes that are genuinely below the floor.
        # THAT is a content verdict: block mode still refuses to publish it, and review mode may
        # deliver it marked so a human can see exactly which beats need better footage. Nothing is
        # accepted and no threshold moves — but be exact: a delivered draft AIRS those beats, and
        # build's picture chain enlarges anything under 1280 wide onto the 1080p canvas. Seeing
        # the upscale is the point of the draft; shipping it is what block mode still prevents.
        first = failures[0]
        raise NonRetryableBuildError(
            f"native-resolution gate: {len(failures)} selection(s) are below "
            f"{minimum_long}x{minimum} "
            f"(first beat {first['original_beat']}: {first['width']}x{first['height']}); "
            f"upscaling does not create HD detail. See {audit_path.name}",
            kind="native_resolution")
    return payload
