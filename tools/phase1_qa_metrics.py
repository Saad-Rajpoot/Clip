#!/usr/bin/env python3
"""Phase-1 QA metrics for a rendered documentary MP4.

Self-contained, cross-platform (macOS / Windows). Does NOT import or modify
any engine code under ``vidlore/``. Uses the bundled ffmpeg shipped with
``imageio_ffmpeg`` for frame extraction, and cv2 + numpy + PIL for analysis.

Usage::

    python tools/phase1_qa_metrics.py <path-to.mp4>

Writes ``<mp4_dir>/phase1_qa_metrics.json`` and prints a 2-column table.

It auto-discovers these sibling files in the MP4's folder when present and
uses them to enrich / ground the metrics (each is optional):

    motion_graphics_manifest.json
    render_meta.json
    script.json
    render_dedup_log.json
    asset_qa.json
    render_relevance_qa.json

Design notes
------------
* Every metric is wrapped in try/except so a single failing stage never kills
  the rest. A failed metric reports ``{"error": "..."}`` in the JSON.
* Where a metric cannot be computed exactly from ground truth, a clearly
  labelled best-effort PROXY is used and the method is recorded under a
  ``"method"`` (and usually a ``"proxy": true``) key in the JSON.
* At most ~250 frames are sampled (1 fps, capped) for speed. Frames are
  decoded in a single ffmpeg pass to rawvideo rgb24 on stdout.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is expected in .venv
    cv2 = None

try:
    from PIL import Image  # noqa: F401  (imported for availability / future use)
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()

MAX_FRAMES = 250          # hard cap on sampled frames for speed
SAMPLE_FPS = 1.0          # nominal sampling rate (frames per second of video)
ANALYZE_W = 320           # downscale width for per-frame numpy analysis

# Tunable thresholds (documented in the JSON output where used)
WHITE_FLASH_LUMA = 0.93   # mean luma above this ...
WHITE_FLASH_STD = 0.04    # ... AND std below this  => white flash
BLACK_FRAME_LUMA = 0.06   # mean luma below this    => black frame
NEAR_DUP_HAMMING = 10     # phash Hamming <= this & not adjacent => near-dup
ARCHIVAL_CHROMA = 0.06    # mean chroma (sat proxy) below this => archival-ish
CARD_EDGE_DENSITY = 0.045  # below this edge density + uniform bg => flat card


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _err(exc: Exception) -> dict:
    """Compact error payload for a failed metric."""
    return {"error": f"{type(exc).__name__}: {exc}"}


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def _round(x, n=4):
    try:
        if x is None:
            return None
        return round(float(x), n)
    except Exception:
        return x


# --------------------------------------------------------------------------- #
# ffmpeg: container / stream metadata via `ffmpeg -i` stderr parsing
# (imageio_ffmpeg ships ffmpeg but not ffprobe, so we parse the banner)
# --------------------------------------------------------------------------- #
def probe_meta(video: Path) -> dict:
    out = {
        "duration_s": None,
        "width": None,
        "height": None,
        "resolution": None,
        "fps": None,
        "bitrate_kbps": None,
        "file_size_bytes": None,
        "integrated_bitrate_kbps": None,
        "method": "ffmpeg -i banner parse (ffprobe not bundled)",
    }
    try:
        out["file_size_bytes"] = video.stat().st_size
    except Exception:
        pass

    r = _run([FF, "-hide_banner", "-i", str(video)], text=True)
    txt = r.stderr or ""

    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", txt)
    if m:
        out["duration_s"] = (
            int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        )
    mb = re.search(r"bitrate:\s*(\d+)\s*kb/s", txt)
    if mb:
        out["bitrate_kbps"] = int(mb.group(1))

    # First video stream line.
    mv = re.search(r"Stream #\d+:\d+.*?:\s*Video:.*", txt)
    if mv:
        sline = mv.group(0)
        mres = re.search(r"(\d{2,5})x(\d{2,5})", sline)
        if mres:
            out["width"] = int(mres.group(1))
            out["height"] = int(mres.group(2))
            out["resolution"] = f"{out['width']}x{out['height']}"
        mfps = re.search(r"([\d.]+)\s*fps", sline)
        if mfps:
            out["fps"] = float(mfps.group(1))
        mtbr = None
        if out["fps"] is None:
            mtbr = re.search(r"([\d.]+)\s*tbr", sline)
            if mtbr:
                out["fps"] = float(mtbr.group(1))

    # Integrated bitrate computed from actual file size + duration (honest;
    # independent of the container's declared average).
    try:
        if out["file_size_bytes"] and out["duration_s"]:
            out["integrated_bitrate_kbps"] = _round(
                out["file_size_bytes"] * 8.0 / out["duration_s"] / 1000.0, 1
            )
    except Exception:
        pass
    return out


def extract_frames(video: Path, duration_s, meta_fps):
    """Decode evenly-sampled frames in a single ffmpeg pass.

    Returns (frames_uint8_list, timestamps_list, info_dict). Frames are RGB
    uint8 arrays downscaled to width ANALYZE_W. Aims for <= MAX_FRAMES frames
    at ~SAMPLE_FPS, raising the effective interval if the video is long.
    """
    info = {"requested_fps": SAMPLE_FPS, "max_frames": MAX_FRAMES}

    # Decide the sampling fps so that total sampled frames <= MAX_FRAMES.
    eff_fps = SAMPLE_FPS
    if duration_s and duration_s > 0:
        if duration_s * SAMPLE_FPS > MAX_FRAMES:
            eff_fps = MAX_FRAMES / float(duration_s)
    info["effective_fps"] = eff_fps

    # Determine output frame height preserving aspect (even number for codecs).
    w = ANALYZE_W
    h = -2  # let ffmpeg keep aspect, force even

    vf = f"fps={eff_fps:.6f},scale={w}:{h}:flags=area,format=rgb24"
    cmd = [
        FF, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", vf,
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    r = _run(cmd)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(
            "ffmpeg frame extraction failed: "
            + (r.stderr.decode("utf-8", "replace")[-400:] if r.stderr else "no output")
        )

    raw = r.stdout

    # We told ffmpeg width but height is aspect-derived; recover height from
    # the source resolution if we have it, else infer from byte count.
    src_w = src_h = None
    mres = None
    if meta_fps:  # meta_fps presence is incidental; resolution carried separately
        pass
    # Try to get exact frame height by a tiny probe of the scaled stream:
    fh = None
    # Recover height: total_bytes = N * w * fh * 3. We know w; find fh that
    # divides cleanly for an integer N. Prefer the aspect-derived height.
    nbytes = len(raw)
    # Candidate heights from common aspect ratios at width w:
    candidates = []
    for ar in (16 / 9, 9 / 16, 4 / 3, 3 / 4, 1.0, 1.91, 2.0, 21 / 9):
        ch = int(round(w / ar))
        if ch % 2:
            ch += 1
        candidates.append(ch)
    # Also brute force any plausible height 2..2000.
    fh = None
    frame_bytes = None
    for ch in sorted(set(candidates)) + list(range(2, 2001, 2)):
        if ch <= 0:
            continue
        fb = w * ch * 3
        if fb > 0 and nbytes % fb == 0:
            fh = ch
            frame_bytes = fb
            break
    if fh is None:
        raise RuntimeError(
            f"could not infer frame height from {nbytes} bytes at width {w}"
        )

    n = nbytes // frame_bytes
    arr = np.frombuffer(raw, dtype=np.uint8)[: n * frame_bytes]
    arr = arr.reshape(n, fh, w, 3)
    frames = [arr[i] for i in range(n)]
    ts = [(i / eff_fps) if eff_fps else float(i) for i in range(n)]

    info.update({"frame_count": n, "frame_w": w, "frame_h": fh})
    return frames, ts, info


# --------------------------------------------------------------------------- #
# Per-frame primitive features (computed once, reused by several metrics)
# --------------------------------------------------------------------------- #
def frame_features(frames):
    """Compute reusable per-frame features.

    Returns dict of numpy arrays / lists keyed by feature name.
    """
    feats = {
        "mean_luma": [],     # 0..1
        "std_luma": [],      # 0..1
        "mean_chroma": [],   # 0..1 (HSV saturation mean, archival proxy)
        "edge_density": [],  # fraction of Canny edge pixels
        "phash": [],         # 64-bit int dHash
        "ahash": [],         # 64-bit int aHash
        "bg_uniform": [],    # bool: dominant border background uniform
        "white_cluster": [],  # fraction of bright pixels (text proxy)
        "local_contrast": [],  # std of local contrast (text-on-flat proxy)
    }
    for fr in frames:
        # Luma (Rec.601), 0..1.
        r = fr[:, :, 0].astype(np.float32)
        g = fr[:, :, 1].astype(np.float32)
        b = fr[:, :, 2].astype(np.float32)
        y = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        feats["mean_luma"].append(float(y.mean()))
        feats["std_luma"].append(float(y.std()))

        # Chroma proxy: HSV saturation mean (archival = low chroma).
        try:
            hsv = cv2.cvtColor(fr, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1].astype(np.float32) / 255.0
            feats["mean_chroma"].append(float(sat.mean()))
        except Exception:
            # max-min over channels as a chroma fallback.
            mx = np.max(fr, axis=2).astype(np.float32)
            mn = np.min(fr, axis=2).astype(np.float32)
            denom = np.clip(mx, 1.0, None)
            feats["mean_chroma"].append(float(((mx - mn) / denom).mean()))

        gray = (y * 255.0).astype(np.uint8)

        # Edge density via Canny.
        try:
            edges = cv2.Canny(gray, 60, 160)
            feats["edge_density"].append(float((edges > 0).mean()))
        except Exception:
            gx = np.abs(np.diff(y, axis=1)).mean()
            gy = np.abs(np.diff(y, axis=0)).mean()
            feats["edge_density"].append(float((gx + gy)))

        # Perceptual hashes (dHash + aHash) on an 8x8/9x8 grid.
        try:
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            diff = small[:, 1:] > small[:, :-1]      # 8x8 -> 64 bits
            dh = 0
            for bit in diff.flatten():
                dh = (dh << 1) | int(bool(bit))
            feats["phash"].append(dh)

            small8 = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
            ah_bits = small8 > small8.mean()
            ah = 0
            for bit in ah_bits.flatten():
                ah = (ah << 1) | int(bool(bit))
            feats["ahash"].append(ah)
        except Exception:
            feats["phash"].append(0)
            feats["ahash"].append(0)

        # Border-background uniformity (flat/graphic card cue): sample a
        # border ring and measure its colour spread.
        try:
            hh, ww = gray.shape
            t = max(2, hh // 12)
            ring = np.concatenate([
                fr[:t, :, :].reshape(-1, 3),
                fr[-t:, :, :].reshape(-1, 3),
                fr[:, :t, :].reshape(-1, 3),
                fr[:, -t:, :].reshape(-1, 3),
            ], axis=0).astype(np.float32)
            ring_std = float(ring.std(axis=0).mean())
            feats["bg_uniform"].append(ring_std < 14.0)
        except Exception:
            feats["bg_uniform"].append(False)

        # Text-on-low-contrast proxy features.
        try:
            bright = (y > 0.80)
            feats["white_cluster"].append(float(bright.mean()))
            # Local contrast: std over small blocks; low std under bright text
            # over a flat bed is the readability-risk signature.
            k = 16
            hh, ww = y.shape
            ys = (hh // k) * k
            xs = (ww // k) * k
            if ys >= k and xs >= k:
                blk = y[:ys, :xs].reshape(ys // k, k, xs // k, k)
                block_std = blk.std(axis=(1, 3))
                feats["local_contrast"].append(float(block_std.mean()))
            else:
                feats["local_contrast"].append(float(y.std()))
        except Exception:
            feats["white_cluster"].append(0.0)
            feats["local_contrast"].append(float(y.std()))

    return feats


def hamming64(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


# --------------------------------------------------------------------------- #
# Metric 2: full-screen card runtime %
# --------------------------------------------------------------------------- #
# Primitives that occupy the FULL screen when rendered (flat graphic cards),
# as opposed to lower-thirds / overlays composited over footage.
FULLSCREEN_CARD_PRIMITIVES = {
    "act_chapter_card", "statement_card", "chronology_timeline",
    "silhouette_scale_compare", "gold_number_callout", "quote_card",
    "stat_card", "stat_insight", "title_card", "section_card",
    "definition_card", "map_card", "comparison_card", "timeline",
    "end_card", "intro_card", "number_callout", "fullscreen_stat",
}


def metric_fullscreen_card(manifest, total_dur):
    try:
        if not manifest:
            return {"value": None, "proxy": True,
                    "method": "no manifest; see fullscreen_card_proxy"}
        scenes = manifest.get("scenes") or []
        card_s = 0.0
        counted = []
        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            prim = sc.get("primitive")
            if not prim:
                continue
            if sc.get("skipped") or sc.get("fallback"):
                continue
            layout = (sc.get("layout") or "").lower()
            variant = ""
            try:
                variant = (sc.get("variant") or {}).get("visual_variant_id", "")
            except Exception:
                variant = ""
            is_full = (
                prim in FULLSCREEN_CARD_PRIMITIVES
                or "fullscreen" in layout
                or "fullscreen" in (variant or "").lower()
            )
            if is_full:
                d = sc.get("duration")
                if isinstance(d, (int, float)) and d > 0:
                    card_s += float(d)
                    counted.append({"scene_index": sc.get("scene_index"),
                                    "primitive": prim, "duration": float(d)})
        pct = (card_s / total_dur * 100.0) if total_dur else None
        return {
            "value": _round(pct, 2),
            "full_screen_card_seconds": _round(card_s, 2),
            "total_duration_s": _round(total_dur, 2),
            "cards_counted": counted,
            "proxy": False,
            "method": ("sum durations of full-screen card primitives in "
                       "motion_graphics_manifest.json / total duration"),
        }
    except Exception as e:
        return _err(e)


def metric_fullscreen_card_proxy(feats, total_dur, eff_fps):
    """Frame-based proxy used when the manifest is missing/empty."""
    try:
        ed = np.array(feats["edge_density"], dtype=np.float32)
        bg = np.array(feats["bg_uniform"], dtype=bool)
        looks_card = (ed < CARD_EDGE_DENSITY) & bg
        frac = float(looks_card.mean()) if looks_card.size else 0.0
        return {
            "value": _round(frac * 100.0, 2),
            "proxy": True,
            "method": (f"fraction of sampled frames with edge_density<"
                       f"{CARD_EDGE_DENSITY} AND uniform border background"),
            "frames_flagged": int(looks_card.sum()),
            "frames_sampled": int(looks_card.size),
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 3: repeated / near-duplicate assets via perceptual hash
# --------------------------------------------------------------------------- #
def metric_dup_assets(feats):
    try:
        ph = feats["phash"]
        n = len(ph)
        if n == 0:
            return {"error": "no frames"}

        # EXACT repeats: same phash, excluding temporally-adjacent frames
        # (which are almost always the same continuous shot). We count, for
        # each phash value, the number of "non-adjacent re-appearances".
        from collections import defaultdict
        idx_by_hash = defaultdict(list)
        for i, h in enumerate(ph):
            idx_by_hash[h].append(i)

        exact_repeat_events = 0
        exact_example_hashes = 0
        for h, idxs in idx_by_hash.items():
            if len(idxs) < 2:
                continue
            # Count appearances that are NOT adjacent to the previous kept one.
            kept = [idxs[0]]
            extra = 0
            for j in idxs[1:]:
                if j - kept[-1] > 1:   # not temporally adjacent
                    extra += 1
                    kept.append(j)
                else:
                    kept[-1] = j
            if extra > 0:
                exact_repeat_events += extra
                exact_example_hashes += 1

        # NEAR-duplicate pairs: Hamming <= NEAR_DUP_HAMMING and NOT temporally
        # adjacent (|i-j| > 1). O(n^2) over <=250 frames is fine.
        near_pairs = 0
        seen_examples = []
        for i in range(n):
            for j in range(i + 1, n):
                if j - i <= 1:
                    continue  # adjacent same-shot
                d = hamming64(ph[i], ph[j])
                if d <= NEAR_DUP_HAMMING:
                    near_pairs += 1
                    if len(seen_examples) < 8:
                        seen_examples.append({"i": i, "j": j, "hamming": d})

        return {
            "repeated_asset_count": exact_repeat_events,
            "repeated_asset_distinct_frames": exact_example_hashes,
            "near_duplicate_count": near_pairs,
            "near_duplicate_examples": seen_examples,
            "frames_sampled": n,
            "method": (
                "1 fps (capped) sampling; 64-bit dHash per frame. "
                "repeated_asset_count = exact phash re-appearances excluding "
                "temporally-adjacent frames. near_duplicate_count = non-adjacent "
                f"frame pairs with Hamming<= {NEAR_DUP_HAMMING}."),
            "proxy": True,
            "note": ("frame-level proxy for asset reuse; one stock clip held on "
                     "screen yields adjacent identical frames which are excluded, "
                     "but distinct re-uses of the same asset are counted."),
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 4: repeated motion-graphics family count (from manifest)
# --------------------------------------------------------------------------- #
def metric_repeated_mg_family(manifest):
    try:
        if not manifest:
            return {"value": None, "method": "no manifest",
                    "error": "motion_graphics_manifest.json not found"}
        from collections import Counter
        counts = Counter()
        # Prefer the explicit by_primitive summary when present.
        by_prim = None
        summ = manifest.get("summary") or {}
        if isinstance(summ.get("by_primitive"), dict):
            by_prim = summ["by_primitive"]
        if by_prim:
            for k, v in by_prim.items():
                try:
                    counts[k] += int(v)
                except Exception:
                    pass
        else:
            for sc in manifest.get("scenes") or []:
                if isinstance(sc, dict) and sc.get("primitive") and not sc.get("skipped"):
                    counts[sc["primitive"]] += 1

        repeated = {k: c for k, c in counts.items() if c > 1}
        max_repeat = max(counts.values()) if counts else 0
        max_family = max(counts, key=counts.get) if counts else None
        return {
            "repeated_mg_family_count": len(repeated),
            "max_repeat": int(max_repeat),
            "max_repeat_family": max_family,
            "repeated_families": {k: int(v) for k, v in repeated.items()},
            "family_histogram": {k: int(v) for k, v in counts.items()},
            "method": "count MG families used > once in motion_graphics_manifest.json",
            "proxy": False,
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 5: human-shot ratio (Haar face/person)
# --------------------------------------------------------------------------- #
def metric_human_shot_ratio(frames, feats):
    try:
        if cv2 is None:
            raise RuntimeError("cv2 unavailable")
        data_dir = Path(cv2.data.haarcascades)
        cascade_files = {
            "frontalface": "haarcascade_frontalface_default.xml",
            "profileface": "haarcascade_profileface.xml",
            "fullbody": "haarcascade_fullbody.xml",
            "upperbody": "haarcascade_upperbody.xml",
        }
        cascades = {}
        for name, fn in cascade_files.items():
            p = data_dir / fn
            if p.exists():
                c = cv2.CascadeClassifier(str(p))
                if not c.empty():
                    cascades[name] = c
        if not cascades:
            return _human_skin_proxy(frames)

        hits = 0
        per_frame = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            found = False
            for name, casc in cascades.items():
                try:
                    objs = casc.detectMultiScale(
                        gray, scaleFactor=1.15, minNeighbors=5,
                        minSize=(int(gray.shape[1] * 0.06),
                                 int(gray.shape[0] * 0.06)))
                    if len(objs) > 0:
                        found = True
                        break
                except Exception:
                    continue
            per_frame.append(found)
            if found:
                hits += 1
        n = len(frames)
        return {
            "value": _round(hits / n, 4) if n else None,
            "frames_with_human": hits,
            "frames_sampled": n,
            "cascades_used": sorted(cascades.keys()),
            "method": ("cv2 Haar frontalface_default + profileface "
                       "(+ full/upper body) detection per sampled frame"),
            "proxy": False,
        }
    except Exception as e:
        # Fall back to the skin proxy on any hard failure.
        try:
            return _human_skin_proxy(frames)
        except Exception:
            return _err(e)


def _human_skin_proxy(frames):
    hits = 0
    for fr in frames:
        try:
            ycc = cv2.cvtColor(fr, cv2.COLOR_RGB2YCrCb)
            cr = ycc[:, :, 1].astype(np.int16)
            cb = ycc[:, :, 2].astype(np.int16)
            skin = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
            if skin.mean() > 0.06:
                hits += 1
        except Exception:
            continue
    n = len(frames)
    return {
        "value": _round(hits / n, 4) if n else None,
        "frames_with_human": hits,
        "frames_sampled": n,
        "method": "skin-tone YCrCb heuristic (Haar cascade unavailable)",
        "proxy": True,
    }


# --------------------------------------------------------------------------- #
# Metric 6: archival-specific ratio (near-grayscale / sepia)
# --------------------------------------------------------------------------- #
def metric_archival_ratio(feats):
    try:
        chroma = np.array(feats["mean_chroma"], dtype=np.float32)
        n = chroma.size
        if n == 0:
            return {"error": "no frames"}
        archival = chroma < ARCHIVAL_CHROMA
        return {
            "value": _round(float(archival.mean()), 4),
            "frames_archival": int(archival.sum()),
            "frames_sampled": int(n),
            "mean_chroma_overall": _round(float(chroma.mean()), 4),
            "method": (f"fraction of frames with mean HSV saturation < "
                       f"{ARCHIVAL_CHROMA} (near-grayscale/sepia proxy)"),
            "proxy": True,
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 7: white-flash / black-frame counts (+ ffmpeg blackdetect)
# --------------------------------------------------------------------------- #
def metric_flash_black(feats, ts):
    try:
        luma = np.array(feats["mean_luma"], dtype=np.float32)
        std = np.array(feats["std_luma"], dtype=np.float32)
        n = luma.size
        white = (luma > WHITE_FLASH_LUMA) & (std < WHITE_FLASH_STD)
        black = (luma < BLACK_FRAME_LUMA)
        white_ts = [_round(ts[i], 2) for i in np.where(white)[0]]
        black_ts = [_round(ts[i], 2) for i in np.where(black)[0]]
        return {
            "white_flash_count": int(white.sum()),
            "white_flash_timestamps": white_ts[:50],
            "black_frame_count": int(black.sum()),
            "black_frame_timestamps": black_ts[:50],
            "frames_sampled": int(n),
            "method": (f"white: mean_luma>{WHITE_FLASH_LUMA} AND std<"
                       f"{WHITE_FLASH_STD}; black: mean_luma<{BLACK_FRAME_LUMA} "
                       "(sampled frames, normalized 0..1)"),
            "proxy": True,
        }
    except Exception as e:
        return _err(e)


def metric_blackdetect(video: Path):
    try:
        r = _run([
            FF, "-hide_banner", "-nostats", "-i", str(video),
            "-vf", "blackdetect=d=0.10:pic_th=0.98:pix_th=0.10",
            "-an", "-f", "null", "-",
        ], text=True)
        spans = []
        for ln in (r.stderr or "").splitlines():
            ms = re.search(r"black_start:([\d.]+)", ln)
            me = re.search(r"black_end:([\d.]+)", ln)
            md = re.search(r"black_duration:([\d.]+)", ln)
            if ms and me:
                spans.append({
                    "start": float(ms.group(1)),
                    "end": float(me.group(1)),
                    "duration": float(md.group(1)) if md else None,
                })
        return {
            "span_count": len(spans),
            "spans": spans[:50],
            "method": "ffmpeg blackdetect=d=0.10:pic_th=0.98:pix_th=0.10",
            "proxy": False,
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 8: median luminance
# --------------------------------------------------------------------------- #
def metric_median_luma(feats):
    try:
        luma = np.array(feats["mean_luma"], dtype=np.float32)
        if luma.size == 0:
            return {"error": "no frames"}
        return {
            "value": _round(float(np.median(luma)), 4),
            "mean": _round(float(luma.mean()), 4),
            "p10": _round(float(np.percentile(luma, 10)), 4),
            "p90": _round(float(np.percentile(luma, 90)), 4),
            "frames_sampled": int(luma.size),
            "method": "median of per-frame mean luma (Rec.601, 0..1)",
            "proxy": False,
        }
    except Exception as e:
        return _err(e)


# --------------------------------------------------------------------------- #
# Metric 9: readability failures (large text over low-contrast bg) — proxy
# --------------------------------------------------------------------------- #
def metric_readability(feats):
    try:
        white = np.array(feats["white_cluster"], dtype=np.float32)
        contrast = np.array(feats["local_contrast"], dtype=np.float32)
        luma = np.array(feats["mean_luma"], dtype=np.float32)
        n = white.size
        if n == 0:
            return {"error": "no frames"}
        # Signature: a meaningful cluster of bright (text-like) pixels sitting
        # on a LOW local-contrast bed, and the frame is not itself a bright
        # full-card (which would be legible by construction). These thresholds
        # are deliberately conservative; this is a flagged PROXY.
        text_present = white > 0.04
        low_contrast_bed = contrast < 0.05
        not_full_white = luma < 0.85
        fail = text_present & low_contrast_bed & not_full_white
        cnt = int(fail.sum())
        return {
            "value": cnt,
            "frames_flagged": cnt,
            "frames_sampled": int(n),
            "method": ("PROXY: frames with bright-pixel cluster (>4%) over low "
                       "local-contrast bed (<0.05) and not a full bright card. "
                       "On-screen text legibility is not reliably detectable "
                       "from pixels alone; treat as best-effort."),
            "proxy": True,
        }
    except Exception as e:
        # Per spec: if not reliably detectable, return 0 with a method note.
        return {"value": 0, "method": f"unavailable: {e}", "proxy": True}


# --------------------------------------------------------------------------- #
# Metric 10: rejected / fallback count from sibling logs
# --------------------------------------------------------------------------- #
def metric_rejected_fallback(siblings):
    try:
        total = 0
        breakdown = {}
        sources_present = []

        dedup = siblings.get("render_dedup_log.json")
        if dedup is not None:
            sources_present.append("render_dedup_log.json")
            c = _count_rejections(dedup, ("rejected", "replaced", "fallback",
                                          "duplicate", "dropped", "swap"))
            breakdown["render_dedup_log"] = c
            total += c

        aqa = siblings.get("asset_qa.json")
        if aqa is not None:
            sources_present.append("asset_qa.json")
            c = 0
            try:
                summ = aqa.get("summary") or {}
                by_sev = summ.get("by_severity") or {}
                # warn-level checks are effectively rejections/flags
                c = int(by_sev.get("warn", 0))
                # plus any explicit "rejected" wording in warnings
                for w in aqa.get("warnings") or []:
                    msg = (w.get("message") or "").lower()
                    if "reject" in msg and w.get("severity") != "warn":
                        c += 1
            except Exception:
                c = _count_rejections(aqa, ("rejected", "reject"))
            breakdown["asset_qa_warnings"] = c
            total += c

        rqa = siblings.get("render_relevance_qa.json")
        if rqa is not None:
            sources_present.append("render_relevance_qa.json")
            c = 0
            try:
                flags = rqa.get("flags") or []
                c = len(flags) if isinstance(flags, list) else 0
            except Exception:
                c = 0
            breakdown["relevance_qa_flags"] = c
            total += c

        # The manifest also records factual-guard rejections + fallbacks.
        manifest = siblings.get("motion_graphics_manifest.json")
        if manifest is not None:
            mg_audit = manifest.get("motion_graphics_audit") or {}
            fg = mg_audit.get("factual_guard_rejected") or []
            fb = 0
            try:
                fb = int((manifest.get("summary") or {}).get("fallbacks", 0))
            except Exception:
                fb = 0
            mg_c = (len(fg) if isinstance(fg, list) else 0) + fb
            breakdown["manifest_factual_guard_rejected"] = (
                len(fg) if isinstance(fg, list) else 0)
            breakdown["manifest_fallbacks"] = fb
            total += mg_c
            sources_present.append("motion_graphics_manifest.json")

        if not sources_present:
            return {"value": 0, "method": "no rejection/fallback logs found",
                    "sources_present": []}
        return {
            "value": int(total),
            "breakdown": breakdown,
            "sources_present": sources_present,
            "method": ("sum of rejections/replacements/fallbacks across "
                       "render_dedup_log.json, asset_qa.json (warn-level), "
                       "render_relevance_qa.json (flags), and the manifest's "
                       "factual_guard_rejected + fallbacks"),
            "proxy": False,
        }
    except Exception as e:
        return _err(e)


def _count_rejections(obj, keywords):
    """Recursively count list items / dict entries whose status or keys hint a
    rejection/replacement/fallback."""
    kw = tuple(k.lower() for k in keywords)
    count = 0

    def walk(o):
        nonlocal count
        if isinstance(o, dict):
            # status-like fields
            for sk in ("status", "guard_status", "action", "result", "verdict",
                       "reason", "outcome"):
                v = o.get(sk)
                if isinstance(v, str) and any(k in v.lower() for k in kw):
                    count += 1
                    break
            # boolean flags
            for bk in ("rejected", "fallback", "replaced", "duplicate", "dropped"):
                if o.get(bk) is True:
                    count += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return count


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
SIBLING_FILES = [
    "motion_graphics_manifest.json",
    "render_meta.json",
    "script.json",
    "render_dedup_log.json",
    "asset_qa.json",
    "render_relevance_qa.json",
]


def discover_siblings(folder: Path) -> dict:
    out = {}
    for name in SIBLING_FILES:
        p = folder / name
        if p.exists():
            out[name] = _load_json(p)
    return out


def build_table(results: dict) -> str:
    rows = []

    def fmt(v):
        if isinstance(v, dict):
            if "error" in v and len(v) == 1:
                return f"ERROR: {v['error']}"
            if "value" in v:
                tag = " (proxy)" if v.get("proxy") else ""
                return f"{v['value']}{tag}"
            return json.dumps(v)[:60]
        return str(v)

    m = results["metrics"]
    basic = results["video"]

    rows.append(("duration_s", basic.get("duration_s")))
    rows.append(("resolution", basic.get("resolution")))
    rows.append(("fps", basic.get("fps")))
    rows.append(("bitrate_kbps (declared)", basic.get("bitrate_kbps")))
    rows.append(("integrated_bitrate_kbps", basic.get("integrated_bitrate_kbps")))
    rows.append(("frames_sampled", results.get("sampling", {}).get("frame_count")))

    def mv(key, sub=None):
        d = m.get(key, {})
        if isinstance(d, dict) and sub and sub in d:
            return d[sub]
        return fmt(d)

    rows.append(("full_screen_card_runtime_pct",
                 m.get("full_screen_card_runtime_pct", {}).get("value")
                 if not m.get("full_screen_card_runtime_pct", {}).get("error")
                 else fmt(m.get("full_screen_card_runtime_pct"))))
    rows.append(("repeated_asset_count",
                 m.get("repeated_near_duplicate", {}).get("repeated_asset_count",
                       fmt(m.get("repeated_near_duplicate")))))
    rows.append(("near_duplicate_count",
                 m.get("repeated_near_duplicate", {}).get("near_duplicate_count",
                       "-")))
    rows.append(("repeated_mg_family_count",
                 m.get("repeated_mg_family", {}).get("repeated_mg_family_count",
                       fmt(m.get("repeated_mg_family")))))
    rows.append(("  max_mg_repeat",
                 m.get("repeated_mg_family", {}).get("max_repeat", "-")))
    rows.append(("human_shot_ratio", mv("human_shot_ratio", "value")))
    rows.append(("archival_specific_ratio", mv("archival_specific_ratio", "value")))
    rows.append(("white_flash_count",
                 m.get("flash_black", {}).get("white_flash_count",
                       fmt(m.get("flash_black")))))
    rows.append(("black_frame_count",
                 m.get("flash_black", {}).get("black_frame_count", "-")))
    rows.append(("blackdetect_spans",
                 m.get("blackdetect", {}).get("span_count",
                       fmt(m.get("blackdetect")))))
    rows.append(("median_luminance", mv("median_luminance", "value")))
    rows.append(("readability_failures", mv("readability_failures", "value")))
    rows.append(("rejected_fallback_count", mv("rejected_fallback_count", "value")))

    width = max(len(str(k)) for k, _ in rows) + 2
    lines = []
    title = "  PHASE-1 QA METRICS  "
    bar = "=" * (width + 26)
    lines.append(bar)
    lines.append(title.center(width + 26))
    lines.append(bar)
    for k, v in rows:
        lines.append(f"{str(k).ljust(width)}| {v}")
    lines.append(bar)
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print("usage: python tools/phase1_qa_metrics.py <path-to.mp4>")
        return 2
    video = Path(argv[1]).expanduser().resolve()
    if not video.exists():
        print(f"error: file not found: {video}")
        return 2

    folder = video.parent
    results = {
        "schema": "phase1_qa_metrics/1",
        "video_path": str(video),
        "tool": "tools/phase1_qa_metrics.py",
        "ffmpeg": FF,
        "siblings_found": [],
        "video": {},
        "sampling": {},
        "metrics": {},
    }

    # --- sibling discovery ---
    try:
        siblings = discover_siblings(folder)
        results["siblings_found"] = sorted(siblings.keys())
    except Exception as e:
        siblings = {}
        results["siblings_error"] = f"{type(e).__name__}: {e}"

    manifest = siblings.get("motion_graphics_manifest.json")
    render_meta = siblings.get("render_meta.json")

    # --- metric 1: container metadata ---
    try:
        meta = probe_meta(video)
        results["video"] = meta
    except Exception as e:
        results["video"] = _err(e)
        meta = {"duration_s": None, "fps": None}

    # Prefer render_meta fps/duration to cross-check when banner parse is thin.
    if isinstance(render_meta, dict):
        if not results["video"].get("fps") and render_meta.get("fps"):
            results["video"]["fps"] = render_meta.get("fps")
            results["video"]["fps_source"] = "render_meta.json"
        results["video"]["render_meta_video_seconds"] = render_meta.get("video_seconds")

    duration_s = results["video"].get("duration_s")
    if not duration_s and isinstance(render_meta, dict):
        duration_s = render_meta.get("video_seconds")

    # --- frame extraction (shared) ---
    frames, ts, feats = [], [], None
    try:
        frames, ts, sinfo = extract_frames(
            video, duration_s, results["video"].get("fps"))
        results["sampling"] = sinfo
    except Exception as e:
        results["sampling"] = _err(e)

    if frames:
        try:
            feats = frame_features(frames)
        except Exception as e:
            results["sampling"]["features_error"] = f"{type(e).__name__}: {e}"
            feats = None

    M = results["metrics"]

    # --- metric 2: full-screen card runtime % ---
    M["full_screen_card_runtime_pct"] = metric_fullscreen_card(manifest, duration_s)
    if (manifest is None or M["full_screen_card_runtime_pct"].get("value") is None) \
            and feats is not None:
        eff = results.get("sampling", {}).get("effective_fps")
        M["full_screen_card_runtime_pct"] = metric_fullscreen_card_proxy(
            feats, duration_s, eff)

    # --- metric 3: repeated + near-duplicate assets ---
    if feats is not None:
        M["repeated_near_duplicate"] = metric_dup_assets(feats)
    else:
        M["repeated_near_duplicate"] = {"error": "no frames/features"}

    # --- metric 4: repeated MG family ---
    M["repeated_mg_family"] = metric_repeated_mg_family(manifest)

    # --- metric 5: human shot ratio ---
    if frames:
        M["human_shot_ratio"] = metric_human_shot_ratio(frames, feats)
    else:
        M["human_shot_ratio"] = {"error": "no frames"}

    # --- metric 6: archival ratio ---
    if feats is not None:
        M["archival_specific_ratio"] = metric_archival_ratio(feats)
    else:
        M["archival_specific_ratio"] = {"error": "no frames/features"}

    # --- metric 7: white-flash / black-frame + blackdetect ---
    if feats is not None:
        M["flash_black"] = metric_flash_black(feats, ts)
    else:
        M["flash_black"] = {"error": "no frames/features"}
    M["blackdetect"] = metric_blackdetect(video)

    # --- metric 8: median luminance ---
    if feats is not None:
        M["median_luminance"] = metric_median_luma(feats)
    else:
        M["median_luminance"] = {"error": "no frames/features"}

    # --- metric 9: readability failures ---
    if feats is not None:
        M["readability_failures"] = metric_readability(feats)
    else:
        M["readability_failures"] = {"value": 0, "method": "no frames", "proxy": True}

    # --- metric 10: rejected / fallback count ---
    M["rejected_fallback_count"] = metric_rejected_fallback(siblings)

    # --- write JSON ---
    out_path = folder / "phase1_qa_metrics.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        results["_written_to"] = str(out_path)
    except Exception as e:
        print(f"warning: could not write JSON: {e}")

    # --- print table ---
    try:
        print(build_table(results))
    except Exception as e:
        print(f"(table render failed: {e})")
        print(json.dumps(results["metrics"], indent=2)[:2000])

    print(f"\nJSON written to: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
